"""SAM 2 image segmentation with q/v-only LoRA adaptation.

The public ``SAM2ImagePredictor`` is intentionally not used here because its
image-embedding path is decorated with ``torch.no_grad``.  This module mirrors
that path while keeping autograd enabled for the LoRA branches in Hiera.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


SAM2_LARGE_CONFIG = "configs/sam2/sam2_hiera_l.yaml"


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank

        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(inputs)) * self.scaling


class LoRAQKV(nn.Module):
    """LoRA residuals for q and v slices of Hiera's fused qkv projection."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float | None = None,
    ) -> None:
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError(
                "Hiera qkv projection output must be divisible into q, k, and v"
            )
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.slice_size = base.out_features // 3
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank

        self.lora_a_q = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b_q = nn.Linear(rank, self.slice_size, bias=False)
        self.lora_a_v = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b_v = nn.Linear(rank, self.slice_size, bias=False)
        for layer in (self.lora_a_q, self.lora_b_q, self.lora_a_v, self.lora_b_v):
            layer.to(device=base.weight.device, dtype=base.weight.dtype)
        for layer in (self.lora_a_q, self.lora_a_v):
            nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
        for layer in (self.lora_b_q, self.lora_b_v):
            nn.init.zeros_(layer.weight)
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        q_delta = self.lora_b_q(self.lora_a_q(inputs)) * self.scaling
        v_delta = self.lora_b_v(self.lora_a_v(inputs)) * self.scaling
        zero_key = torch.zeros_like(q_delta)
        return output + torch.cat((q_delta, zero_key, v_delta), dim=-1)


@dataclass(frozen=True)
class LoRAInjectionReport:
    rank: int
    alpha: float
    hiera_qv_projections: tuple[str, ...]
    decoder_q_projections: tuple[str, ...]
    decoder_v_projections: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


def _is_linear(module: Any) -> bool:
    return isinstance(module, nn.Linear)


def inject_lora_qv(
    sam2_model: nn.Module,
    rank: int = 4,
    alpha: float | None = None,
) -> LoRAInjectionReport:
    """Freeze SAM 2 and inject LoRA into Hiera and mask-decoder q/v paths."""

    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    effective_alpha = float(rank if alpha is None else alpha)
    sam2_model.requires_grad_(False)

    try:
        blocks = sam2_model.image_encoder.trunk.blocks
    except AttributeError as exc:
        raise TypeError("model does not expose SAM 2's Hiera blocks") from exc

    hiera_names: list[str] = []
    for index, block in enumerate(blocks):
        projection = getattr(getattr(block, "attn", None), "qkv", None)
        if not _is_linear(projection):
            raise TypeError(f"Hiera block {index} has no unfused nn.Linear qkv")
        block.attn.qkv = LoRAQKV(projection, rank=rank, alpha=effective_alpha)
        hiera_names.append(f"image_encoder.trunk.blocks.{index}.attn.qkv")

    decoder_q_names: list[str] = []
    decoder_v_names: list[str] = []
    decoder = getattr(sam2_model, "sam_mask_decoder", None)
    if decoder is None:
        raise TypeError("model does not expose sam_mask_decoder")

    # Materialize the list because modules are replaced during traversal.
    attention_modules = list(decoder.named_modules())
    for module_name, module in attention_modules:
        q_proj = getattr(module, "q_proj", None)
        v_proj = getattr(module, "v_proj", None)
        if _is_linear(q_proj) and _is_linear(v_proj):
            module.q_proj = LoRALinear(q_proj, rank=rank, alpha=effective_alpha)
            module.v_proj = LoRALinear(v_proj, rank=rank, alpha=effective_alpha)
            prefix = f"sam_mask_decoder.{module_name}" if module_name else "sam_mask_decoder"
            decoder_q_names.append(f"{prefix}.q_proj")
            decoder_v_names.append(f"{prefix}.v_proj")

    if not hiera_names:
        raise RuntimeError("no Hiera qkv projection received LoRA")
    if not decoder_q_names:
        raise RuntimeError("no mask-decoder q/v projection received LoRA")

    trainable = sum(p.numel() for p in sam2_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sam2_model.parameters())
    return LoRAInjectionReport(
        rank=rank,
        alpha=effective_alpha,
        hiera_qv_projections=tuple(hiera_names),
        decoder_q_projections=tuple(decoder_q_names),
        decoder_v_projections=tuple(decoder_v_names),
        trainable_parameters=trainable,
        total_parameters=total,
    )


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only trainable LoRA tensors, detached on CPU."""

    trainable_names = {name for name, value in model.named_parameters() if value.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def load_adapter_state_dict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    strict: bool = True,
) -> None:
    """Load a compact LoRA state dictionary without requiring base weights."""

    expected = set(adapter_state_dict(model))
    received = set(state)
    missing = sorted(expected - received)
    unexpected = sorted(received - expected)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"LoRA state mismatch; missing={missing}, unexpected={unexpected}"
        )
    current = model.state_dict()
    for name in expected & received:
        current[name] = state[name].to(dtype=current[name].dtype)
    model.load_state_dict(current, strict=True)


class CrackSAM2(nn.Module):
    """Prompt-free or Frangi-mask-prompted binary SAM 2 segmenter."""

    def __init__(self, sam2_model: nn.Module) -> None:
        super().__init__()
        self.sam2 = sam2_model
        self.register_buffer(
            "pixel_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    @property
    def image_size(self) -> int:
        return int(self.sam2.image_size)

    def _encode_images(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        images = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        images = (images - self.pixel_mean) / self.pixel_std
        backbone_out = self.sam2.forward_image(images)
        _, vision_feats, _, feature_sizes = self.sam2._prepare_backbone_features(
            backbone_out
        )
        if self.sam2.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed
        batch_size = images.shape[0]
        features = [
            feature.permute(1, 2, 0).reshape(batch_size, -1, *size)
            for feature, size in zip(vision_feats, feature_sizes)
        ]
        return features[-1], features[:-1]

    def forward(
        self,
        images: torch.Tensor,
        mask_input: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape (B,3,H,W), got {images.shape}")
        if not images.is_floating_point():
            raise TypeError("images must be floating-point tensors in [0, 1]")
        if output_size is None:
            output_size = (int(images.shape[-2]), int(images.shape[-1]))
        batch_size = images.shape[0]

        image_embeddings, high_resolution_features = self._encode_images(images)
        if mask_input is not None:
            if mask_input.ndim != 4 or mask_input.shape[:2] != (batch_size, 1):
                raise ValueError(
                    "mask_input must have shape (B,1,H,W), "
                    f"got {tuple(mask_input.shape)}"
                )
            prompt_dtype = next(self.sam2.sam_prompt_encoder.parameters()).dtype
            mask_input = mask_input.detach().to(
                device=images.device, dtype=prompt_dtype
            )
            required_size = tuple(self.sam2.sam_prompt_encoder.mask_input_size)
            if tuple(mask_input.shape[-2:]) != required_size:
                mask_input = F.interpolate(
                    mask_input.float(),
                    size=required_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            # Frangi is a static, non-differentiable geometric prompt.
        sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=mask_input,
        )
        # With no prompt at all, the official prompt encoder returns a batch of
        # one. CrackSAM is prompt-free but trains multiple independent images.
        if sparse_embeddings.shape[0] == 1 and batch_size > 1:
            sparse_embeddings = sparse_embeddings.expand(batch_size, -1, -1)
        if dense_embeddings.shape[0] == 1 and batch_size > 1:
            dense_embeddings = dense_embeddings.expand(batch_size, -1, -1, -1)
        (
            low_resolution_logits,
            iou_predictions,
            _,
            object_score_logits,
        ) = self.sam2.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_resolution_features,
        )
        logits = F.interpolate(
            low_resolution_logits.float(),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return {
            "logits": logits,
            "low_res_logits": low_resolution_logits.float(),
            "iou_predictions": iou_predictions,
            "object_score_logits": object_score_logits,
        }


def build_cracksam2(
    checkpoint: str | Path,
    rank: int = 4,
    alpha: float | None = None,
    config: str = SAM2_LARGE_CONFIG,
    device: str | torch.device = "cuda",
) -> tuple[CrackSAM2, LoRAInjectionReport]:
    """Build the official SAM 2 Hiera model and inject q/v LoRA branches."""

    try:
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise RuntimeError(
            "SAM 2 is not installed; install requirements-sam2.txt first"
        ) from exc

    base = build_sam2(
        config_file=config,
        ckpt_path=str(checkpoint),
        device=str(device),
        mode="train",
        apply_postprocessing=False,
    )
    report = inject_lora_qv(base, rank=rank, alpha=alpha)
    model = CrackSAM2(base).to(device)
    return model, report


def checkpoint_payload(
    model: CrackSAM2,
    report: LoRAInjectionReport,
    **training_state: Any,
) -> dict[str, Any]:
    """Create a serializable training checkpoint with compact adapter weights."""

    reserved = {"format_version", "adapter", "lora"}
    collisions = reserved.intersection(training_state)
    if collisions:
        raise ValueError(f"training state uses reserved keys: {sorted(collisions)}")
    return {
        "format_version": 1,
        "adapter": adapter_state_dict(model),
        "lora": asdict(report),
        **training_state,
    }
