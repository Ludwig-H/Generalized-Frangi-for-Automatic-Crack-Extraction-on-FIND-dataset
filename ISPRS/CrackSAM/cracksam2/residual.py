"""Minimal raster residual correction on top of a frozen CrackSAM 2 baseline.

The raster maps are auxiliary evidence, never SAM mask prompts.  SAM encodes an
image once, its historical prompt-free decoder produces ``z0``, and the small
adapter predicts only ``delta_z``.  The final projection is initialized to zero,
so the untrained candidate is exactly ``z0``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CrackSAM2, SAM2ImageFeatures

__all__ = [
    "FrangiGraphResidual",
    "RasterResidualAdapter",
    "select_residual_logits",
]


def _resize(inputs: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(inputs.shape[-2:]) == size:
        return inputs
    return F.interpolate(inputs, size=size, mode="bilinear", align_corners=False)


def select_residual_logits(
    baseline_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    accept_residual: bool | torch.Tensor,
) -> torch.Tensor:
    """Select the candidate or fall back exactly to the baseline.

    A tensor gate must be boolean.  A shape ``(B,)`` is interpreted as one
    decision per image; other shapes only need to broadcast to the logits.
    Rejected values come directly from ``baseline_logits`` through
    :func:`torch.where`, rather than through arithmetic with a confidence score.
    """

    if baseline_logits.shape != candidate_logits.shape:
        raise ValueError(
            "baseline and candidate logits must have the same shape, got "
            f"{tuple(baseline_logits.shape)} and {tuple(candidate_logits.shape)}"
        )
    if isinstance(accept_residual, bool):
        return candidate_logits if accept_residual else baseline_logits
    if not isinstance(accept_residual, torch.Tensor):
        raise TypeError("accept_residual must be a bool or a boolean tensor")
    if accept_residual.dtype is not torch.bool:
        raise TypeError("accept_residual tensor must have dtype torch.bool")

    gate = accept_residual.to(device=baseline_logits.device)
    if gate.ndim == 1:
        if gate.shape[0] != baseline_logits.shape[0]:
            raise ValueError(
                "a one-dimensional gate must contain one decision per image"
            )
        gate = gate.reshape(-1, 1, 1, 1)
    try:
        torch.broadcast_shapes(gate.shape, baseline_logits.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"gate shape {tuple(gate.shape)} does not broadcast to logits "
            f"shape {tuple(baseline_logits.shape)}"
        ) from exc
    return torch.where(gate, candidate_logits, baseline_logits)


class RasterResidualAdapter(nn.Module):
    """Predict a one-channel logit correction from SAM and Frangi rasters.

    Parameters
    ----------
    raster_channels:
        Number of channels in the cached Frangi raster tensor.
    high_resolution_channels:
        Channel count of each SAM high-resolution feature map, in the order
        returned by :meth:`CrackSAM2.encode_images`.
    hidden_channels:
        Width of this small convolutional adapter.
    """

    def __init__(
        self,
        raster_channels: int,
        high_resolution_channels: Sequence[int],
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        if raster_channels <= 0:
            raise ValueError("raster_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        feature_channels = tuple(int(value) for value in high_resolution_channels)
        if any(value <= 0 for value in feature_channels):
            raise ValueError("all high_resolution_channels must be positive")

        self.raster_channels = int(raster_channels)
        self.high_resolution_channels = feature_channels
        self.hidden_channels = int(hidden_channels)

        self.raster_projection = nn.Sequential(
            nn.Conv2d(self.raster_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.baseline_projection = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.feature_projections = nn.ModuleList(
            nn.Conv2d(channels, hidden_channels, 1) for channels in feature_channels
        )
        branches = 2 + len(feature_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(branches * hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.output_projection = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        raster: torch.Tensor,
        baseline_logits: torch.Tensor,
        features: SAM2ImageFeatures,
    ) -> torch.Tensor:
        if raster.ndim != 4 or raster.shape[1] != self.raster_channels:
            raise ValueError(
                "raster must have shape "
                f"(B,{self.raster_channels},H,W), got {tuple(raster.shape)}"
            )
        if not raster.is_floating_point():
            raise TypeError("raster must be a floating-point tensor")
        if baseline_logits.ndim != 4 or baseline_logits.shape[1] != 1:
            raise ValueError(
                "baseline_logits must have shape (B,1,H,W), got "
                f"{tuple(baseline_logits.shape)}"
            )
        if raster.shape[0] != baseline_logits.shape[0]:
            raise ValueError("raster and baseline logits must have the same batch size")
        if len(features.high_resolution_features) != len(self.feature_projections):
            raise ValueError(
                "SAM feature count differs from high_resolution_channels: "
                f"{len(features.high_resolution_features)} versus "
                f"{len(self.feature_projections)}"
            )
        if features.batch_size != baseline_logits.shape[0]:
            raise ValueError(
                "SAM features and baseline logits must have the same batch size"
            )

        if features.high_resolution_features:
            fusion_size = tuple(features.high_resolution_features[0].shape[-2:])
        else:
            fusion_size = tuple(baseline_logits.shape[-2:])
        parameter = self.output_projection.weight
        device, dtype = parameter.device, parameter.dtype

        branches = [
            self.raster_projection(
                _resize(raster.to(device=device, dtype=dtype), fusion_size)
            ),
            self.baseline_projection(
                _resize(
                    baseline_logits.detach().to(device=device, dtype=dtype), fusion_size
                )
            ),
        ]
        for index, (projection, feature) in enumerate(
            zip(self.feature_projections, features.high_resolution_features)
        ):
            if feature.ndim != 4:
                raise ValueError(
                    f"SAM high-resolution feature {index} is not four-dimensional"
                )
            if feature.shape[:2] != (
                baseline_logits.shape[0],
                self.high_resolution_channels[index],
            ):
                raise ValueError(
                    f"SAM high-resolution feature {index} must start with shape "
                    f"({baseline_logits.shape[0]},"
                    f"{self.high_resolution_channels[index]}), got "
                    f"{tuple(feature.shape[:2])}"
                )
            projected = projection(feature.detach().to(device=device, dtype=dtype))
            branches.append(_resize(projected, fusion_size))

        hidden = self.fusion(torch.cat(branches, dim=1))
        residual = self.output_projection(hidden)
        return _resize(residual, tuple(baseline_logits.shape[-2:]))


class FrangiGraphResidual(nn.Module):
    """Frozen prompt-free CrackSAM 2 baseline plus a raster logit residual."""

    def __init__(
        self,
        baseline: CrackSAM2,
        raster_channels: int,
        high_resolution_channels: Sequence[int],
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        self.baseline.requires_grad_(False)
        self.baseline.eval()
        self.adapter = RasterResidualAdapter(
            raster_channels=raster_channels,
            high_resolution_channels=high_resolution_channels,
            hidden_channels=hidden_channels,
        )

    def train(self, mode: bool = True) -> FrangiGraphResidual:
        """Train only the adapter; the frozen baseline always remains in eval."""

        super().train(mode)
        self.baseline.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        frangi_raster: torch.Tensor,
        output_size: tuple[int, int] | None = None,
        accept_residual: bool | torch.Tensor = True,
    ) -> dict[str, torch.Tensor]:
        """Return baseline, residual candidate, and exactly selected logits.

        ``accept_residual=False`` is the explicit exact fallback.  A boolean
        tensor permits one decision per image (or any broadcastable spatial
        decision) without mixing baseline and candidate probabilities.
        """

        with torch.no_grad():
            features = self.baseline.encode_images(images)
            baseline_output = self.baseline.decode_features(
                features,
                mask_input=None,
                output_size=output_size,
            )
        baseline_logits = baseline_output["logits"]
        residual_logits = self.adapter(frangi_raster, baseline_logits, features)
        candidate_logits = baseline_logits + residual_logits
        logits = select_residual_logits(
            baseline_logits,
            candidate_logits,
            accept_residual,
        )
        return {
            "logits": logits,
            "baseline_logits": baseline_logits,
            "candidate_logits": candidate_logits,
            "residual_logits": residual_logits,
            "baseline_low_res_logits": baseline_output["low_res_logits"],
            "baseline_iou_predictions": baseline_output["iou_predictions"],
            "baseline_object_score_logits": baseline_output["object_score_logits"],
        }
