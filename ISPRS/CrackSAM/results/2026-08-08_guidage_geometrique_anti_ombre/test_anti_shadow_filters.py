"""Contrôles synthétiques des invariants anti-ombre des filtres géométriques."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from anti_shadow_filters import (  # noqa: E402
    compute_filter_bank,
    dark_phase_symmetry,
    line_step_bic,
    log_luminance,
    oriented_flux_symmetry,
    paired_profile,
)


def _rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(gray, 0.0, 1.0)[..., None], 3, axis=2)


def _vertical_line(size: int = 128, width: int = 3) -> np.ndarray:
    image = np.full((size, size), 0.72, dtype=np.float32)
    center = size // 2
    image[:, center - width // 2 : center + (width + 1) // 2] = 0.16
    return _rgb(image)


def _vertical_step(size: int = 128) -> np.ndarray:
    image = np.full((size, size), 0.72, dtype=np.float32)
    image[:, size // 2 :] *= 0.28
    return _rgb(image)


def _center_band_mean(score: np.ndarray, half_width: int = 3) -> float:
    center = score.shape[1] // 2
    return float(np.mean(score[12:-12, center - half_width : center + half_width + 1]))


def test_uniform_image_abstains() -> None:
    image = _rgb(np.full((96, 96), 0.5, dtype=np.float32))
    result = compute_filter_bank(image)
    for name, score in result.maps.items():
        assert np.all(np.isfinite(score)), name
        assert score.shape == image.shape[:2], name
        assert float(np.max(score)) <= 1e-5, (name, float(np.max(score)))


def test_line_is_preferred_to_step_by_symmetry_filters() -> None:
    line_signal = log_luminance(_vertical_line())
    step_signal = log_luminance(_vertical_step())

    line_ofs, _ = oriented_flux_symmetry(line_signal)
    step_ofs, _ = oriented_flux_symmetry(step_signal)
    line_profile, _ = paired_profile(line_signal)
    step_profile, _ = paired_profile(step_signal)
    line_bic, _ = line_step_bic(line_signal)
    step_bic, _ = line_step_bic(step_signal)
    line_phase, _ = dark_phase_symmetry(line_signal)
    step_phase, _ = dark_phase_symmetry(step_signal)

    assert _center_band_mean(line_ofs) > _center_band_mean(step_ofs) + 0.08
    assert _center_band_mean(line_profile) > _center_band_mean(step_profile) + 0.08
    assert _center_band_mean(line_bic) > _center_band_mean(step_bic) + 0.08
    assert _center_band_mean(line_phase) > _center_band_mean(step_phase) + 0.04


def test_finite_bounded_maps_on_textured_shadow_crossing() -> None:
    rng = np.random.default_rng(20260808)
    size = 128
    image = 0.64 + 0.035 * rng.standard_normal((size, size))
    image[63:66, :] -= 0.36
    image[:, 70:] *= 0.38
    result = compute_filter_bank(_rgb(image.astype(np.float32)))
    for name, score in result.maps.items():
        assert score.shape == (size, size), name
        assert np.all(np.isfinite(score)), name
        assert 0.0 <= float(np.min(score)) <= float(np.max(score)) <= 1.0, name

    fused = result.maps["fusion_precision"]
    crack_response = float(np.mean(fused[60:69, 15:55]))
    shadow_edge_response = float(np.mean(fused[15:55, 67:74]))
    assert crack_response > shadow_edge_response


def test_quarter_turn_rotation_equivariance() -> None:
    vertical = _vertical_line(size=112, width=3)
    horizontal = np.rot90(vertical)
    vertical_map = compute_filter_bank(vertical).maps["fusion_ofs_profile"]
    horizontal_map = compute_filter_bank(horizontal).maps["fusion_ofs_profile"]
    restored = np.rot90(horizontal_map, k=-1)
    assert np.allclose(vertical_map[12:-12, 12:-12], restored[12:-12, 12:-12], atol=3e-3)
