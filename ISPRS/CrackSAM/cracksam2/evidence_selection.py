"""Local photometric evidence used to verify Frangi ridge candidates.

Frangi responds to second-order line structure but does not know whether a dark
response is a crack, a cast-shadow boundary, or another elongated texture.  A
dark crack should usually form a bilateral valley across the Hessian normal;
an illumination boundary is closer to a one-sided step.  This module measures
that distinction without changing the cached seven-channel Frangi schema.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

PROFILE_CHANNELS = (
    "dark_valley_symmetry",
    "one_sided_step",
    "bilateral_dark_contrast",
    "orientation_coherence",
)


def _validate_profile_inputs(
    images: torch.Tensor,
    orientation_sin2: torch.Tensor,
    orientation_cos2: torch.Tensor,
    output_size: tuple[int, int],
    radii: Sequence[float],
) -> tuple[float, ...]:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must have shape (B,3,H,W), got {images.shape}")
    if not images.is_floating_point():
        raise TypeError("images must be floating point")
    expected = (images.shape[0], 1)
    for name, value in (
        ("orientation_sin2", orientation_sin2),
        ("orientation_cos2", orientation_cos2),
    ):
        if value.ndim != 4 or value.shape[:2] != expected:
            raise ValueError(
                f"{name} must start with shape {expected}, got {value.shape}"
            )
        if value.shape != orientation_sin2.shape:
            raise ValueError("orientation tensors must have the same shape")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
    height, width = (int(value) for value in output_size)
    if height < 2 or width < 2:
        raise ValueError("output_size must contain at least two pixels per axis")
    checked = tuple(float(radius) for radius in radii)
    if not checked or any(
        not math.isfinite(radius) or radius <= 0 for radius in checked
    ):
        raise ValueError("radii must be a non-empty sequence of positive finite values")
    return checked


def oriented_dark_ridge_profiles(
    images: torch.Tensor,
    orientation_sin2: torch.Tensor,
    orientation_cos2: torch.Tensor,
    *,
    output_size: tuple[int, int],
    radii: Sequence[float] = (1.5, 3.0),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Measure valley-vs-step evidence along the unoriented Hessian normal.

    ``sin(2 theta)`` and ``cos(2 theta)`` reconstruct an unoriented normal, so
    swapping its sign leaves every output unchanged.  For each radius, the
    luminance at the center is compared with samples on both sides.  The four
    returned channels lie in ``[0, 1]`` for input images in ``[0, 1]``:

    * bilateral dark-valley symmetry (high only when both flanks are brighter),
    * one-sided step strength (high for an illumination edge),
    * bilateral dark contrast (the weaker of the two bright flanks),
    * double-angle orientation coherence after feature-cell aggregation.

    Each photometric channel takes its own maximum over the requested radii.
    This makes the descriptor tolerant to crack-width and Frangi-scale mismatch,
    but does not assert multi-radius consistency.  It is a feature for a learned
    verifier, not a hard hand-written crack classifier.
    """

    checked_radii = _validate_profile_inputs(
        images,
        orientation_sin2,
        orientation_cos2,
        output_size,
        radii,
    )
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be positive and finite")

    height, width = output_size
    dtype = images.dtype
    device = images.device
    luminance_weights = images.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    luminance = (images * luminance_weights).sum(dim=1, keepdim=True)
    luminance = F.interpolate(
        luminance, size=output_size, mode="bilinear", align_corners=False
    )

    sin2 = F.interpolate(
        orientation_sin2.to(device=device, dtype=dtype),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    cos2 = F.interpolate(
        orientation_cos2.to(device=device, dtype=dtype),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    angular_magnitude = torch.sqrt(sin2.square() + cos2.square())
    orientation_coherence = angular_magnitude.clamp(0.0, 1.0)
    angular_norm = angular_magnitude.clamp_min(eps)
    theta = 0.5 * torch.atan2(sin2 / angular_norm, cos2 / angular_norm)
    normal_x = torch.cos(theta)[:, 0]
    normal_y = torch.sin(theta)[:, 0]

    base_y, base_x = torch.meshgrid(
        (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0,
        (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0,
        indexing="ij",
    )
    base_x = base_x.unsqueeze(0).expand(images.shape[0], -1, -1)
    base_y = base_y.unsqueeze(0).expand(images.shape[0], -1, -1)

    valley_responses: list[torch.Tensor] = []
    step_responses: list[torch.Tensor] = []
    contrast_responses: list[torch.Tensor] = []
    center = luminance
    for radius in checked_radii:
        offset_x = normal_x * (2.0 * radius / width)
        offset_y = normal_y * (2.0 * radius / height)
        negative_grid = torch.stack((base_x - offset_x, base_y - offset_y), dim=-1)
        positive_grid = torch.stack((base_x + offset_x, base_y + offset_y), dim=-1)
        negative = F.grid_sample(
            luminance,
            negative_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        positive = F.grid_sample(
            luminance,
            positive_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        negative_contrast = negative - center
        positive_contrast = positive - center
        contrast_sum = negative_contrast.abs() + positive_contrast.abs()
        informative = (contrast_sum > 16.0 * eps).to(dtype=dtype)
        denominator = contrast_sum.clamp_min(eps)
        bilateral = torch.minimum(F.relu(negative_contrast), F.relu(positive_contrast))
        valley_responses.append(
            (2.0 * bilateral / denominator).clamp(0.0, 1.0) * informative
        )
        step_responses.append(
            ((negative - positive).abs() / denominator).clamp(0.0, 1.0) * informative
        )
        contrast_responses.append(bilateral.clamp(0.0, 1.0) * informative)

    # When incompatible directions cancel inside a feature cell, atan2(0, 0)
    # would otherwise choose an arbitrary horizontal normal.  Damping every
    # oriented descriptor by the double-angle resultant makes that cell
    # explicitly uninformative while still exposing coherence as its own input.
    return torch.cat(
        (
            torch.stack(valley_responses).amax(dim=0) * orientation_coherence,
            torch.stack(step_responses).amax(dim=0) * orientation_coherence,
            torch.stack(contrast_responses).amax(dim=0) * orientation_coherence,
            orientation_coherence,
        ),
        dim=1,
    )


__all__ = ["PROFILE_CHANNELS", "oriented_dark_ridge_profiles"]
