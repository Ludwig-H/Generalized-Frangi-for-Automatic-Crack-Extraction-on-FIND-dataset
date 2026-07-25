from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from cracksam2.evidence_selection import (  # noqa: E402
    PROFILE_CHANNELS,
    oriented_dark_ridge_profiles,
)
from cracksam2.frangi import generate_frangi_raster  # noqa: E402
from cracksam2.graph_types import FRANGI_RASTER_CHANNEL_INDEX  # noqa: E402


def _rgb(luminance: torch.Tensor) -> torch.Tensor:
    return luminance.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)


def test_dark_ridge_profile_separates_a_bilateral_valley_from_a_shadow_step() -> None:
    size = 33
    center = size // 2
    ridge = torch.full((size, size), 0.8)
    ridge[:, center] = 0.1
    step = torch.full((size, size), 0.2)
    step[:, center + 1 :] = 0.8
    # theta=0 is the horizontal normal of a vertical line.
    sin2 = torch.zeros(1, 1, size, size)
    cos2 = torch.ones_like(sin2)

    ridge_profiles = oriented_dark_ridge_profiles(
        _rgb(ridge), sin2, cos2, output_size=(size, size), radii=(2.0,)
    )
    step_profiles = oriented_dark_ridge_profiles(
        _rgb(step), sin2, cos2, output_size=(size, size), radii=(2.0,)
    )

    valley_index = PROFILE_CHANNELS.index("dark_valley_symmetry")
    step_index = PROFILE_CHANNELS.index("one_sided_step")
    assert ridge_profiles[0, valley_index, center, center] > 0.95
    assert ridge_profiles[0, step_index, center, center] < 0.05
    assert step_profiles[0, valley_index, center, center] < 0.05
    assert step_profiles[0, step_index, center, center] > 0.95


def test_profile_uses_the_hessian_normal_for_a_horizontal_ridge() -> None:
    size = 25
    center = size // 2
    ridge = torch.full((size, size), 0.9)
    ridge[center] = 0.1
    # theta=pi/2 -> sin(2 theta)=0, cos(2 theta)=-1.
    sin2 = torch.zeros(1, 1, size, size)
    cos2 = -torch.ones_like(sin2)

    profiles = oriented_dark_ridge_profiles(
        _rgb(ridge), sin2, cos2, output_size=(size, size), radii=(2.0,)
    )

    assert profiles[0, 0, center, center] > 0.95
    assert profiles[0, 1, center, center] < 0.05


def test_profile_identifies_a_soft_colored_shadow_as_a_one_sided_step() -> None:
    size = 41
    center = size // 2
    horizontal = torch.arange(size, dtype=torch.float32)
    penumbra = 0.2 + 0.6 * torch.sigmoid((horizontal - center) / 2.0)
    # Unequal RGB gains mimic a colored cast shadow while retaining a soft
    # one-sided luminance transition.
    image = torch.stack(
        (
            0.80 * penumbra,
            0.95 * penumbra,
            penumbra,
        )
    )[:, None, :].expand(-1, size, -1)[None]
    sin2 = torch.zeros(1, 1, size, size)
    cos2 = torch.ones_like(sin2)

    profiles = oriented_dark_ridge_profiles(
        image,
        sin2,
        cos2,
        output_size=(size, size),
        radii=(4.0,),
    )

    assert (
        profiles[0, PROFILE_CHANNELS.index("dark_valley_symmetry"), center, center]
        < 0.05
    )
    assert profiles[0, PROFILE_CHANNELS.index("one_sided_step"), center, center] > 0.95


def test_diagonal_profile_stays_aligned_on_the_shared_pixel_center_convention() -> None:
    size = 31
    center = size // 2
    ridge = torch.full((size, size), 0.85)
    coordinates = torch.arange(size)
    ridge[coordinates, coordinates] = 0.1
    # The y=x tangent has a -45-degree normal: sin(2 theta)=-1, cos(2 theta)=0.
    sin2 = -torch.ones(1, 1, size, size)
    cos2 = torch.zeros_like(sin2)

    profiles = oriented_dark_ridge_profiles(
        _rgb(ridge), sin2, cos2, output_size=(size, size), radii=(2.0,)
    )

    assert profiles[0, 0, center, center] > 0.9
    assert profiles[0, 1, center, center] < 0.1


def test_zero_orientation_and_uniform_image_are_finite_and_uninformative() -> None:
    image = torch.full((2, 3, 9, 11), 0.4)
    orientation = torch.zeros(2, 1, 9, 11)

    profiles = oriented_dark_ridge_profiles(
        image,
        orientation,
        orientation,
        output_size=(7, 8),
        radii=(1.0, 2.0),
    )

    assert profiles.shape == (2, len(PROFILE_CHANNELS), 7, 8)
    assert torch.isfinite(profiles).all()
    assert torch.count_nonzero(profiles) == 0


def test_profile_rejects_invalid_geometry() -> None:
    image = torch.zeros(1, 3, 5, 5)
    orientation = torch.zeros(1, 1, 5, 5)
    with pytest.raises(ValueError, match="at least two pixels"):
        oriented_dark_ridge_profiles(
            image, orientation, orientation, output_size=(1, 5)
        )
    with pytest.raises(ValueError, match="positive finite"):
        oriented_dark_ridge_profiles(
            image, orientation, orientation, output_size=(5, 5), radii=(0.0,)
        )


def test_frangi_extractor_orientation_drives_the_expected_cross_ridge_profile() -> None:
    """Freeze the extractor -> cache semantics -> local-profile convention."""

    image = np.ones((48, 48, 3), dtype=np.float32)
    image[:, 23:25] = 0.0
    raster = generate_frangi_raster(
        image,
        scales=(1.0, 2.0),
        R=1,
        tau=0.3,
        min_rel_size=1000.0,
        K=1,
        device="cpu",
    )
    support = raster[FRANGI_RASTER_CHANNEL_INDEX["support"]] > 0.5
    assert support.any()

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    sin2 = torch.from_numpy(
        raster[FRANGI_RASTER_CHANNEL_INDEX["orientation_sin2"]]
    )[None, None]
    cos2 = torch.from_numpy(
        raster[FRANGI_RASTER_CHANNEL_INDEX["orientation_cos2"]]
    )[None, None]
    profiles = oriented_dark_ridge_profiles(
        image_tensor,
        sin2,
        cos2,
        output_size=(48, 48),
        radii=(2.0,),
    )[0]
    valley = profiles[PROFILE_CHANNELS.index("dark_valley_symmetry")].numpy()
    step = profiles[PROFILE_CHANNELS.index("one_sided_step")].numpy()

    assert float(valley[support].mean()) > 0.95
    assert float(step[support].mean()) < 0.05
