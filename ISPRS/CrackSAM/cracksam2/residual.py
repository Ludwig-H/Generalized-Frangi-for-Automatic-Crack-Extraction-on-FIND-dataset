"""Local raster residual correction on top of a frozen CrackSAM 2 baseline.

The raster maps are auxiliary evidence, never SAM mask prompts.  SAM encodes an
image once, its historical prompt-free decoder produces ``z0``, and the small
adapter predicts only ``delta_z``.  The final projection is initialized to zero,
so the untrained candidate is exactly ``z0``.

The historical ``legacy_raster_v1`` adapter is retained for reproducibility.
New runs use ``verified_local_v1``: a learned selector checks each Frangi support
against SAM context and oriented valley-vs-shadow-step luminance profiles.  Only
selected support neighborhoods can change the baseline logits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .evidence_selection import PROFILE_CHANNELS, oriented_dark_ridge_profiles
from .graph_types import FRANGI_RASTER_CHANNEL_INDEX, FRANGI_RASTER_CHANNELS
from .model import CrackSAM2, SAM2ImageFeatures

LEGACY_ADAPTER_MODE = "legacy_raster_v1"
VERIFIED_ADAPTER_MODE = "verified_local_v1"
RESIDUAL_ADAPTER_MODES = (LEGACY_ADAPTER_MODE, VERIFIED_ADAPTER_MODE)

__all__ = [
    "FrangiGraphResidual",
    "LEGACY_ADAPTER_MODE",
    "RasterResidualAdapter",
    "RESIDUAL_ADAPTER_MODES",
    "VERIFIED_ADAPTER_MODE",
    "VerifiedLocalResidualAdapter",
    "select_residual_logits",
]


def _resize(inputs: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(inputs.shape[-2:]) == size:
        return inputs
    return F.interpolate(inputs, size=size, mode="bilinear", align_corners=False)


def _resize_nearest(inputs: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(inputs.shape[-2:]) == size:
        return inputs
    return F.interpolate(inputs, size=size, mode="nearest")


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


class VerifiedLocalResidualAdapter(nn.Module):
    """Propose corrections only near locally verified Frangi supports.

    The verifier observes the seven Frangi maps, frozen SAM context, the
        baseline logits, three luminance profiles and orientation coherence.
    During training its soft score lets the proposal learn; in evaluation it
    becomes a hard local decision.  Multiplication by the binary Frangi support
        gives an exact zero correction when evidence is absent, and the final
        envelope prevents any correction outside the accepted neighborhood.
    """

    def __init__(
        self,
        raster_channels: int,
        high_resolution_channels: Sequence[int],
        hidden_channels: int = 32,
        *,
        profile_radii: Sequence[float] = (1.5, 3.0),
        evidence_dilation: int = 2,
        evidence_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if raster_channels != len(FRANGI_RASTER_CHANNELS):
            raise ValueError(
                "verified_local_v1 requires the canonical seven Frangi channels"
            )
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        feature_channels = tuple(int(value) for value in high_resolution_channels)
        if any(value <= 0 for value in feature_channels):
            raise ValueError("all high_resolution_channels must be positive")
        checked_radii = tuple(float(value) for value in profile_radii)
        if not checked_radii or any(
            not math.isfinite(value) or value <= 0 for value in checked_radii
        ):
            raise ValueError("profile_radii must contain positive values")
        if evidence_dilation < 0:
            raise ValueError("evidence_dilation cannot be negative")
        if not 0.0 < evidence_threshold < 1.0:
            raise ValueError("evidence_threshold must lie in (0, 1)")

        self.raster_channels = int(raster_channels)
        self.high_resolution_channels = feature_channels
        self.hidden_channels = int(hidden_channels)
        self.profile_radii = checked_radii
        self.evidence_dilation = int(evidence_dilation)
        self.evidence_threshold = float(evidence_threshold)
        self.adapter_mode = VERIFIED_ADAPTER_MODE

        self.raster_projection = nn.Sequential(
            nn.Conv2d(self.raster_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.profile_projection = nn.Sequential(
            nn.Conv2d(len(PROFILE_CHANNELS), hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.baseline_projection = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.feature_projections = nn.ModuleList(
            nn.Conv2d(channels, hidden_channels, 1) for channels in feature_channels
        )
        context_branches = 1 + len(feature_channels)
        self.context_fusion = nn.Sequential(
            nn.Conv2d(
                context_branches * hidden_channels,
                hidden_channels,
                3,
                padding=1,
            ),
            nn.GELU(),
        )
        verifier_channels = 3 * hidden_channels
        self.verifier = nn.Sequential(
            nn.Conv2d(verifier_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        # Start conservatively: before auxiliary supervision, most evidence is
        # rejected.  The residual itself remains exactly neutral regardless.
        nn.init.constant_(self.verifier[-1].bias, -2.0)
        self.proposal_fusion = nn.Sequential(
            nn.Conv2d(verifier_channels + 1, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.output_projection = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _validate_inputs(
        self,
        images: torch.Tensor,
        raster: torch.Tensor,
        baseline_logits: torch.Tensor,
        features: SAM2ImageFeatures,
    ) -> None:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape (B,3,H,W), got {images.shape}")
        if raster.ndim != 4 or raster.shape[1] != self.raster_channels:
            raise ValueError(
                "raster must have shape "
                f"(B,{self.raster_channels},H,W), got {tuple(raster.shape)}"
            )
        if not images.is_floating_point() or not raster.is_floating_point():
            raise TypeError("images and raster must be floating-point tensors")
        if baseline_logits.ndim != 4 or baseline_logits.shape[1] != 1:
            raise ValueError(
                "baseline_logits must have shape (B,1,H,W), got "
                f"{tuple(baseline_logits.shape)}"
            )
        if (
            images.shape[0] != raster.shape[0]
            or raster.shape[0] != baseline_logits.shape[0]
        ):
            raise ValueError(
                "images, raster and baseline logits need equal batch sizes"
            )
        if len(features.high_resolution_features) != len(self.feature_projections):
            raise ValueError(
                "SAM feature count differs from high_resolution_channels: "
                f"{len(features.high_resolution_features)} versus "
                f"{len(self.feature_projections)}"
            )
        if features.batch_size != baseline_logits.shape[0]:
            raise ValueError("SAM features and logits must have the same batch size")

    def forward(
        self,
        images: torch.Tensor,
        raster: torch.Tensor,
        baseline_logits: torch.Tensor,
        features: SAM2ImageFeatures,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(images, raster, baseline_logits, features)
        if features.high_resolution_features:
            fusion_size = tuple(features.high_resolution_features[0].shape[-2:])
        else:
            fusion_size = tuple(baseline_logits.shape[-2:])
        parameter = self.output_projection.weight
        device, dtype = parameter.device, parameter.dtype
        raster_fusion = _resize(raster.to(device=device, dtype=dtype), fusion_size)
        images_fusion = images.to(device=device, dtype=dtype)
        baseline_fusion = _resize(
            baseline_logits.detach().to(device=device, dtype=dtype), fusion_size
        )

        support_index = FRANGI_RASTER_CHANNEL_INDEX["support"]
        sin_index = FRANGI_RASTER_CHANNEL_INDEX["orientation_sin2"]
        cos_index = FRANGI_RASTER_CHANNEL_INDEX["orientation_cos2"]
        source = raster.to(device=device, dtype=dtype)
        support_source = (
            source[:, support_index : support_index + 1] > 0.5
        ).to(dtype)
        downsampling = (
            support_source.shape[-2] >= fusion_size[0]
            and support_source.shape[-1] >= fusion_size[1]
        )
        if downsampling:
            # A one-pixel skeleton must not disappear when 448 px rasters are
            # reduced to the SAM high-resolution feature grid.
            support_fusion = F.adaptive_max_pool2d(support_source, fusion_size)
            orientation_weight = F.adaptive_avg_pool2d(
                support_source, fusion_size
            )
            orientation_denominator = orientation_weight.clamp_min(1e-6)
            orientation_sin = F.adaptive_avg_pool2d(
                source[:, sin_index : sin_index + 1] * support_source,
                fusion_size,
            ) / orientation_denominator
            orientation_cos = F.adaptive_avg_pool2d(
                source[:, cos_index : cos_index + 1] * support_source,
                fusion_size,
            ) / orientation_denominator
            orientation_sin = orientation_sin * (orientation_weight > 0).to(dtype)
            orientation_cos = orientation_cos * (orientation_weight > 0).to(dtype)
        else:
            support_fusion = _resize_nearest(support_source, fusion_size)
            orientation_sin = _resize_nearest(
                source[:, sin_index : sin_index + 1] * support_source,
                fusion_size,
            )
            orientation_cos = _resize_nearest(
                source[:, cos_index : cos_index + 1] * support_source,
                fusion_size,
            )
        support = (support_fusion > 0.5).to(dtype)
        # Raster, orientation profile and decision now share the same feature
        # cells and pixel-center convention.
        raster_fusion = raster_fusion.clone()
        raster_fusion[:, support_index : support_index + 1] = support
        raster_fusion[:, sin_index : sin_index + 1] = orientation_sin
        raster_fusion[:, cos_index : cos_index + 1] = orientation_cos

        context_branches = [self.baseline_projection(baseline_fusion)]
        for index, (projection, feature) in enumerate(
            zip(self.feature_projections, features.high_resolution_features)
        ):
            if feature.ndim != 4 or feature.shape[:2] != (
                baseline_logits.shape[0],
                self.high_resolution_channels[index],
            ):
                raise ValueError(
                    f"SAM high-resolution feature {index} has incompatible shape "
                    f"{tuple(feature.shape)}"
                )
            projected = projection(feature.detach().to(device=device, dtype=dtype))
            context_branches.append(_resize(projected, fusion_size))
        context = self.context_fusion(torch.cat(context_branches, dim=1))

        # atan2/grid_sample are deliberately kept in float32 even under AMP;
        # their low-precision behavior is backend-dependent and the descriptor
        # is tiny compared with the frozen SAM encoder.
        with torch.autocast(device_type=device.type, enabled=False):
            profiles = oriented_dark_ridge_profiles(
                images_fusion.float(),
                raster_fusion[:, sin_index : sin_index + 1].float(),
                raster_fusion[:, cos_index : cos_index + 1].float(),
                output_size=fusion_size,
                radii=self.profile_radii,
            )
        profiles = profiles.to(dtype=dtype)
        raster_hidden = self.raster_projection(raster_fusion)
        profile_hidden = self.profile_projection(profiles)
        verifier_input = torch.cat((context, raster_hidden, profile_hidden), dim=1)
        evidence_logits_fusion = self.verifier(verifier_input)

        evidence_soft = torch.sigmoid(evidence_logits_fusion) * support
        if self.training:
            selected_evidence = evidence_soft
        else:
            selected_evidence = (
                evidence_soft >= self.evidence_threshold
            ).to(dtype) * support
        if self.evidence_dilation:
            kernel = 2 * self.evidence_dilation + 1
            correction_envelope = F.max_pool2d(
                selected_evidence,
                kernel_size=kernel,
                stride=1,
                padding=self.evidence_dilation,
            )
        else:
            correction_envelope = selected_evidence

        proposal_input = torch.cat(
            (
                context,
                raster_hidden * selected_evidence,
                profile_hidden * selected_evidence,
                selected_evidence,
            ),
            dim=1,
        )
        proposal = self.output_projection(self.proposal_fusion(proposal_input))
        output_size = tuple(baseline_logits.shape[-2:])
        correction_envelope_output = _resize_nearest(
            correction_envelope, output_size
        )
        residual_output = _resize(proposal, output_size) * correction_envelope_output
        return {
            "residual_logits": residual_output,
            "residual_proposal_logits": _resize(proposal, output_size),
            # Training supervision stays on the exact grid that is thresholded
            # at inference; output-grid maps are diagnostics only.
            "evidence_logits": evidence_logits_fusion,
            "evidence_support": support,
            "evidence_support_fusion": support,
            "evidence_score_fusion": evidence_soft,
            "evidence_selected_fusion": selected_evidence,
            "correction_envelope_fusion": correction_envelope,
            "photometric_profiles_fusion": profiles,
            "evidence_logits_output": _resize(evidence_logits_fusion, output_size),
            # Balanced auxiliary BCE makes this a compatibility score, not a
            # calibrated posterior probability.
            "evidence_score": _resize(evidence_soft, output_size),
            "evidence_selected": _resize_nearest(selected_evidence, output_size),
            "evidence_support_output": _resize_nearest(support, output_size),
            "correction_envelope": correction_envelope_output,
            "photometric_profiles": _resize(profiles, output_size),
        }


class FrangiGraphResidual(nn.Module):
    """Frozen prompt-free CrackSAM 2 baseline plus a raster logit residual."""

    def __init__(
        self,
        baseline: CrackSAM2,
        raster_channels: int,
        high_resolution_channels: Sequence[int],
        hidden_channels: int = 32,
        *,
        adapter_mode: str = LEGACY_ADAPTER_MODE,
        profile_radii: Sequence[float] = (1.5, 3.0),
        evidence_dilation: int = 2,
        evidence_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if adapter_mode not in RESIDUAL_ADAPTER_MODES:
            raise ValueError(f"unknown residual adapter_mode: {adapter_mode!r}")
        self.adapter_mode = adapter_mode
        self.baseline = baseline
        self.baseline.requires_grad_(False)
        self.baseline.eval()
        if adapter_mode == LEGACY_ADAPTER_MODE:
            self.adapter: nn.Module = RasterResidualAdapter(
                raster_channels=raster_channels,
                high_resolution_channels=high_resolution_channels,
                hidden_channels=hidden_channels,
            )
        else:
            self.adapter = VerifiedLocalResidualAdapter(
                raster_channels=raster_channels,
                high_resolution_channels=high_resolution_channels,
                hidden_channels=hidden_channels,
                profile_radii=profile_radii,
                evidence_dilation=evidence_dilation,
                evidence_threshold=evidence_threshold,
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
        diagnostics: dict[str, torch.Tensor] = {}
        if self.adapter_mode == LEGACY_ADAPTER_MODE:
            residual_logits = self.adapter(frangi_raster, baseline_logits, features)
        else:
            verified = self.adapter(images, frangi_raster, baseline_logits, features)
            residual_logits = verified["residual_logits"]
            diagnostics = {
                name: value
                for name, value in verified.items()
                if name != "residual_logits"
            }
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
            **diagnostics,
        }
