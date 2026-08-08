#!/usr/bin/env python3
"""Benchmark synthétique déterministe des filtres géométriques anti-ombre.

Ce script reste volontairement indépendant de SAM 2 et de tout entraînement.
Il compare notamment une ligne sombre à deux flancs avec une marche d'ombre
ayant exactement le même contraste et le même profil de bord. Il ajoute des
contrôles de largeur, de polarité et d'abstention, puis écrit uniquement :

* ``tables/generated/phantom_metrics.csv`` ;
* ``figures/generated/phantom_benchmark.png``.

Le ratio principal est le quantile 95 % de la réponse près de la marche divisé
par le quantile 95 % de la réponse au centre de la ligne appariée. Une valeur
inférieure à 1 est souhaitable. L'abstention est la fraction des pixels dont la
réponse ne dépasse pas ``ABSTENTION_THRESHOLD``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import erf


os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from anti_shadow_filters import compute_filter_bank  # noqa: E402


SEED = 20_260_808
IMAGE_SIZE = 128
BACKGROUND_LUMINANCE = 0.55
FEATURE_CONTRAST = 0.30
SHADOW_ATTENUATION = 0.55
METRIC_MARGIN = 16
LINE_CORE_HALF_WIDTH = 0.75
STEP_EVALUATION_HALF_WIDTH = 8.0
EDGE_EVALUATION_HALF_WIDTH = 2.5
ABSTENTION_THRESHOLD = 0.10
MIN_REFERENCE_RESPONSE = 1e-5

# Le plan primaire est un demi-factoriel équilibré : toutes les modalités sont
# présentes, sans payer les 96 calculs supplémentaires d'un factoriel complet.
PRIMARY_ANGLES_DEG = (0.0, 30.0, 60.0, 90.0)
PRIMARY_WIDTHS_PX = (2.0, 4.0, 6.0)
PRIMARY_BLURS_PX = (0.7, 1.7)
PRIMARY_NOISE_SIGMAS = (0.0, 0.015)

WIDE_WIDTHS_PX = (12.0, 20.0)
STRESS_ANGLES_DEG = (0.0, 45.0, 90.0)
BRIGHT_WIDTHS_PX = (2.0, 5.0)
RAMP_NOISE_SIGMAS = (0.0, 0.015)
REPRESENTATIVE_MAP = "fusion_precision"

MAP_ORDER = (
    "frangi_similarity_historical",
    "frangi_relative",
    "black_tophat",
    "derivative_pair",
    "paired_profile",
    "line_step_bic",
    "phase_symmetry",
    "ofs",
    "ofs_reflectance",
    "fusion_ofs_profile",
    "fusion_precision",
    "fusion_reflectance",
    "verified_frangi_ofs",
    "verified_frangi_bic",
    "verified_frangi_v2",
    "verified_frangi_consensus",
)
MAP_LABELS = {
    "frangi_similarity_historical": "Frangi-sim. hist.",
    "frangi_relative": "Frangi classique",
    "black_tophat": "Black top-hat",
    "derivative_pair": "Paire/impair",
    "paired_profile": "Profil bilatéral",
    "line_step_bic": "H0/H1 (ΔBIC)",
    "phase_symmetry": "Phase-symétrie",
    "ofs": "OFS",
    "ofs_reflectance": "OFS + réflectance",
    "fusion_ofs_profile": "OFS × profil",
    "fusion_precision": "Fusion précision",
    "fusion_reflectance": "Fusion réflectance",
    "verified_frangi_ofs": "Frangi vérifié OFS",
    "verified_frangi_bic": "Frangi vérifié BIC",
    "verified_frangi_v2": "Frangi vérifié v2",
    "verified_frangi_consensus": "Frangi vérifié consensus",
}


@dataclass(frozen=True)
class Phantom:
    """Image sRGB et régions de mesure connues analytiquement."""

    image: np.ndarray
    regions: Mapping[str, np.ndarray]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE,
        help="Racine recevant tables/generated et figures/generated (défaut: dossier du script).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Exécute des invariants synthétiques courts, sans écrire les sorties du benchmark.",
    )
    return parser.parse_args()


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in (SEED, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _coordinates(size: int, angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.indices((size, size), dtype=np.float32)
    center = (size - 1.0) / 2.0
    x = cols - center
    y = rows - center
    angle = np.deg2rad(float(angle_deg))
    normal = x * np.cos(angle) + y * np.sin(angle)
    tangent = -x * np.sin(angle) + y * np.cos(angle)
    margin = int(METRIC_MARGIN)
    interior = (
        (rows >= margin)
        & (rows < size - margin)
        & (cols >= margin)
        & (cols < size - margin)
    )
    return normal.astype(np.float32), tangent.astype(np.float32), interior


def _band_profile(normal: np.ndarray, width_px: float, blur_px: float) -> np.ndarray:
    """Indicateur de bande convolué analytiquement par une gaussienne 1D."""

    half_width = 0.5 * float(width_px)
    denominator = math.sqrt(2.0) * float(blur_px)
    profile = 0.5 * (
        erf((normal + half_width) / denominator)
        - erf((normal - half_width) / denominator)
    )
    return np.clip(profile, 0.0, 1.0).astype(np.float32)


def _step_profile(normal: np.ndarray, blur_px: float) -> np.ndarray:
    denominator = math.sqrt(2.0) * float(blur_px)
    return (0.5 * (1.0 + erf(normal / denominator))).astype(np.float32)


def _noise_field(size: int, sigma: float, *key: object) -> np.ndarray:
    if float(sigma) <= 0.0:
        return np.zeros((size, size), dtype=np.float32)
    rng = np.random.default_rng(_stable_seed(*key))
    return (float(sigma) * rng.standard_normal((size, size))).astype(np.float32)


def _linear_to_srgb_rgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
    )
    return np.repeat(np.clip(srgb, 0.0, 1.0)[..., None], 3, axis=2).astype(np.float32)


def _line_phantom(
    *,
    size: int,
    angle_deg: float,
    width_px: float,
    blur_px: float,
    noise: np.ndarray,
    bright: bool = False,
) -> Phantom:
    normal, _, interior = _coordinates(size, angle_deg)
    profile = _band_profile(normal, width_px, blur_px)
    polarity = 1.0 if bright else -1.0
    linear = BACKGROUND_LUMINANCE + polarity * FEATURE_CONTRAST * profile + noise
    core = (np.abs(normal) <= LINE_CORE_HALF_WIDTH) & interior
    edges = (
        np.abs(np.abs(normal) - 0.5 * float(width_px)) <= EDGE_EVALUATION_HALF_WIDTH
    ) & interior
    background = (np.abs(normal) >= 0.5 * float(width_px) + 10.0) & interior
    return Phantom(
        image=_linear_to_srgb_rgb(linear),
        regions={"interior": interior, "core": core, "edges": edges, "background": background},
    )


def _step_phantom(
    *,
    size: int,
    angle_deg: float,
    blur_px: float,
    noise: np.ndarray,
) -> Phantom:
    normal, _, interior = _coordinates(size, angle_deg)
    profile = _step_profile(normal, blur_px)
    linear = BACKGROUND_LUMINANCE - FEATURE_CONTRAST * profile + noise
    boundary = (np.abs(normal) <= STEP_EVALUATION_HALF_WIDTH) & interior
    far_field = (np.abs(normal) >= STEP_EVALUATION_HALF_WIDTH + 5.0) & interior
    return Phantom(
        image=_linear_to_srgb_rgb(linear),
        regions={"interior": interior, "boundary": boundary, "background": far_field},
    )


def _ramp_phantom(
    *,
    size: int,
    angle_deg: float,
    noise: np.ndarray,
) -> Phantom:
    normal, _, interior = _coordinates(size, angle_deg)
    angle = np.deg2rad(float(angle_deg))
    extent = 0.5 * (size - 1.0) * (abs(np.cos(angle)) + abs(np.sin(angle)))
    ramp = np.clip(normal / max(float(extent), 1.0), -1.0, 1.0)
    linear = BACKGROUND_LUMINANCE + 0.5 * FEATURE_CONTRAST * ramp + noise
    return Phantom(
        image=_linear_to_srgb_rgb(linear),
        regions={"interior": interior, "nuisance": interior},
    )


def _uniform_phantom(*, size: int, noise: np.ndarray) -> Phantom:
    _, _, interior = _coordinates(size, 0.0)
    linear = np.full((size, size), BACKGROUND_LUMINANCE, dtype=np.float32) + noise
    return Phantom(
        image=_linear_to_srgb_rgb(linear),
        regions={"interior": interior, "nuisance": interior},
    )


def _line_plus_shadow_phantom(
    *,
    size: int,
    line_angle_deg: float,
    shadow_angle_deg: float,
    width_px: float,
    line_blur_px: float,
    shadow_blur_px: float,
    noise: np.ndarray,
    with_shadow: bool,
) -> Phantom:
    line_normal, _, interior = _coordinates(size, line_angle_deg)
    shadow_normal, _, _ = _coordinates(size, shadow_angle_deg)
    line_profile = _band_profile(line_normal, width_px, line_blur_px)
    reflectance = BACKGROUND_LUMINANCE - FEATURE_CONTRAST * line_profile
    shadow_profile = _step_profile(shadow_normal, shadow_blur_px)
    if with_shadow:
        attenuation = 1.0 - SHADOW_ATTENUATION * shadow_profile
    else:
        attenuation = np.ones_like(shadow_profile)
    linear = reflectance * attenuation + noise

    core = (np.abs(line_normal) <= LINE_CORE_HALF_WIDTH) & interior
    transition_guard = max(3.0, 2.5 * float(shadow_blur_px))
    lit_core = core & (shadow_normal <= -transition_guard)
    shadowed_core = core & (shadow_normal >= transition_guard)
    boundary = (
        (np.abs(shadow_normal) <= STEP_EVALUATION_HALF_WIDTH)
        & (np.abs(line_normal) >= 0.5 * float(width_px) + 5.0)
        & interior
    )
    return Phantom(
        image=_linear_to_srgb_rgb(linear),
        regions={
            "interior": interior,
            "core": core,
            "lit_core": lit_core,
            "shadowed_core": shadowed_core,
            "shadow_boundary": boundary,
        },
    )


def _primary_configs() -> list[tuple[float, float, float, float]]:
    configs: list[tuple[float, float, float, float]] = []
    for angle_index, angle in enumerate(PRIMARY_ANGLES_DEG):
        for width_index, width in enumerate(PRIMARY_WIDTHS_PX):
            for blur_index, blur in enumerate(PRIMARY_BLURS_PX):
                for noise_index, noise in enumerate(PRIMARY_NOISE_SIGMAS):
                    # Fraction régulière 1/2 du plan 4 × 3 × 2 × 2.
                    if (angle_index + width_index + blur_index + noise_index) % 2 == 0:
                        configs.append((angle, width, blur, noise))
    return configs


def _validated_maps(image: np.ndarray) -> Mapping[str, np.ndarray]:
    maps = compute_filter_bank(image).maps
    expected_shape = image.shape[:2]
    for name, score in maps.items():
        array = np.asarray(score)
        if array.shape != expected_shape:
            raise RuntimeError(f"Carte {name}: forme {array.shape}, attendu {expected_shape}")
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"Carte {name}: valeurs non finies")
        if float(np.min(array)) < -1e-6 or float(np.max(array)) > 1.0 + 1e-6:
            raise RuntimeError(f"Carte {name}: valeurs hors de [0, 1]")
    return maps


def _common_map_names(*collections: Mapping[str, np.ndarray]) -> list[str]:
    if not collections:
        return []
    names = set(collections[0])
    for collection in collections[1:]:
        names &= set(collection)
    ordered = [name for name in MAP_ORDER if name in names]
    ordered.extend(sorted(names - set(ordered)))
    return ordered


def _region_stats(score: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(score, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return {"mean": math.nan, "q95": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(values)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _global_stats(score: np.ndarray, interior: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(score, dtype=np.float64)[np.asarray(interior, dtype=bool)]
    active_fraction = float(np.mean(values > ABSTENTION_THRESHOLD))
    q99 = float(np.quantile(values, 0.99))
    return {
        "mean": float(np.mean(values)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": q99,
        "max": float(np.max(values)),
        "active_fraction_010": active_fraction,
        "abstention_fraction_010": 1.0 - active_fraction,
        "abstains_q99_010": int(q99 <= ABSTENTION_THRESHOLD),
    }


def _add_prefixed(row: dict[str, object], prefix: str, values: Mapping[str, object]) -> None:
    row.update({f"{prefix}_{name}": value for name, value in values.items()})


def _safe_ratio(numerator: float, denominator: float) -> tuple[float, int]:
    if not math.isfinite(denominator) or denominator <= MIN_REFERENCE_RESPONSE:
        return math.nan, 0
    if not math.isfinite(numerator):
        return math.nan, 0
    return float(numerator / denominator), 1


def _base_row(
    *,
    family: str,
    case_id: str,
    map_name: str,
    angle_deg: float | None = None,
    width_px: float | None = None,
    blur_px: float | None = None,
    noise_sigma: float | None = None,
) -> dict[str, object]:
    return {
        "family": family,
        "case_id": case_id,
        "map": map_name,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "angle_deg": angle_deg,
        "width_px": width_px,
        "blur_px": blur_px,
        "noise_sigma": noise_sigma,
        "abstention_threshold": ABSTENTION_THRESHOLD,
    }


def _capture_representative(
    representatives: dict[str, tuple[np.ndarray, np.ndarray]],
    name: str,
    phantom: Phantom,
    maps: Mapping[str, np.ndarray],
) -> None:
    if name not in representatives:
        representatives[name] = (
            np.asarray(phantom.image, dtype=np.float32),
            np.asarray(maps[REPRESENTATIVE_MAP], dtype=np.float32),
        )


def _benchmark_primary_pairs(
    rows: list[dict[str, object]],
    representatives: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    configs = _primary_configs()
    for index, (angle, width, blur, noise_sigma) in enumerate(configs, start=1):
        case_id = (
            f"pair_a{_token(angle)}_w{_token(width)}_b{_token(blur)}"
            f"_n{_token(noise_sigma)}"
        )
        noise = _noise_field(
            IMAGE_SIZE, noise_sigma, "primary", angle, width, blur, noise_sigma
        )
        line = _line_phantom(
            size=IMAGE_SIZE,
            angle_deg=angle,
            width_px=width,
            blur_px=blur,
            noise=noise,
        )
        step = _step_phantom(
            size=IMAGE_SIZE,
            angle_deg=angle,
            blur_px=blur,
            noise=noise,
        )
        print(f"[phantom {index:02d}/{len(configs):02d}] ligne/marche {case_id}", flush=True)
        line_maps = _validated_maps(line.image)
        step_maps = _validated_maps(step.image)
        _capture_representative(representatives, "dark_line", line, line_maps)
        _capture_representative(representatives, "shadow_step", step, step_maps)

        for map_name in _common_map_names(line_maps, step_maps):
            line_score = line_maps[map_name]
            step_score = step_maps[map_name]
            line_center = _region_stats(line_score, line.regions["core"])
            step_boundary = _region_stats(step_score, step.regions["boundary"])
            mean_ratio, mean_defined = _safe_ratio(
                step_boundary["mean"], line_center["mean"]
            )
            q95_ratio, q95_defined = _safe_ratio(
                step_boundary["q95"], line_center["q95"]
            )
            row = _base_row(
                family="thin_line_vs_shadow_step",
                case_id=case_id,
                map_name=map_name,
                angle_deg=angle,
                width_px=width,
                blur_px=blur,
                noise_sigma=noise_sigma,
            )
            _add_prefixed(row, "line_center", line_center)
            _add_prefixed(row, "step_boundary", step_boundary)
            _add_prefixed(row, "line_global", _global_stats(line_score, line.regions["interior"]))
            _add_prefixed(row, "step_global", _global_stats(step_score, step.regions["interior"]))
            row.update(
                {
                    "step_to_line_ratio_mean": mean_ratio,
                    "step_to_line_ratio_mean_defined": mean_defined,
                    "step_to_line_ratio_q95": q95_ratio,
                    "step_to_line_ratio_q95_defined": q95_defined,
                }
            )
            rows.append(row)


def _benchmark_wide_lines(
    rows: list[dict[str, object]],
    representatives: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    for angle in STRESS_ANGLES_DEG:
        for width in WIDE_WIDTHS_PX:
            blur = 1.7
            noise_sigma = 0.015
            case_id = f"wide_a{_token(angle)}_w{_token(width)}"
            noise = _noise_field(IMAGE_SIZE, noise_sigma, "wide", angle, width)
            phantom = _line_phantom(
                size=IMAGE_SIZE,
                angle_deg=angle,
                width_px=width,
                blur_px=blur,
                noise=noise,
            )
            print(f"[phantom] ligne large {case_id}", flush=True)
            maps = _validated_maps(phantom.image)
            _capture_representative(representatives, "wide_line", phantom, maps)
            for map_name in _common_map_names(maps):
                score = maps[map_name]
                row = _base_row(
                    family="wide_dark_line",
                    case_id=case_id,
                    map_name=map_name,
                    angle_deg=angle,
                    width_px=width,
                    blur_px=blur,
                    noise_sigma=noise_sigma,
                )
                _add_prefixed(row, "wide_center", _region_stats(score, phantom.regions["core"]))
                _add_prefixed(row, "wide_edges", _region_stats(score, phantom.regions["edges"]))
                _add_prefixed(row, "global", _global_stats(score, phantom.regions["interior"]))
                rows.append(row)


def _benchmark_bright_polarity(
    rows: list[dict[str, object]],
    representatives: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    blur = 1.0
    noise_sigma = 0.010
    for angle in STRESS_ANGLES_DEG:
        for width in BRIGHT_WIDTHS_PX:
            case_id = f"polarity_a{_token(angle)}_w{_token(width)}"
            noise = _noise_field(IMAGE_SIZE, noise_sigma, "polarity", angle, width)
            dark = _line_phantom(
                size=IMAGE_SIZE,
                angle_deg=angle,
                width_px=width,
                blur_px=blur,
                noise=noise,
            )
            bright = _line_phantom(
                size=IMAGE_SIZE,
                angle_deg=angle,
                width_px=width,
                blur_px=blur,
                noise=noise,
                bright=True,
            )
            print(f"[phantom] polarité {case_id}", flush=True)
            dark_maps = _validated_maps(dark.image)
            bright_maps = _validated_maps(bright.image)
            _capture_representative(representatives, "bright_line", bright, bright_maps)
            for map_name in _common_map_names(dark_maps, bright_maps):
                dark_center = _region_stats(dark_maps[map_name], dark.regions["core"])
                bright_center = _region_stats(bright_maps[map_name], bright.regions["core"])
                ratio, defined = _safe_ratio(bright_center["q95"], dark_center["q95"])
                row = _base_row(
                    family="dark_vs_bright_line",
                    case_id=case_id,
                    map_name=map_name,
                    angle_deg=angle,
                    width_px=width,
                    blur_px=blur,
                    noise_sigma=noise_sigma,
                )
                _add_prefixed(row, "dark_center", dark_center)
                _add_prefixed(row, "bright_center", bright_center)
                _add_prefixed(
                    row, "bright_global", _global_stats(bright_maps[map_name], bright.regions["interior"])
                )
                row["bright_to_dark_ratio_q95"] = ratio
                row["bright_to_dark_ratio_q95_defined"] = defined
                rows.append(row)


def _benchmark_abstention(
    rows: list[dict[str, object]],
    representatives: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    for angle in STRESS_ANGLES_DEG:
        for noise_sigma in RAMP_NOISE_SIGMAS:
            case_id = f"ramp_a{_token(angle)}_n{_token(noise_sigma)}"
            noise = _noise_field(IMAGE_SIZE, noise_sigma, "ramp", angle, noise_sigma)
            phantom = _ramp_phantom(
                size=IMAGE_SIZE,
                angle_deg=angle,
                noise=noise,
            )
            print(f"[phantom] rampe {case_id}", flush=True)
            maps = _validated_maps(phantom.image)
            _capture_representative(representatives, "ramp", phantom, maps)
            for map_name in _common_map_names(maps):
                score = maps[map_name]
                row = _base_row(
                    family="ramp",
                    case_id=case_id,
                    map_name=map_name,
                    angle_deg=angle,
                    noise_sigma=noise_sigma,
                )
                _add_prefixed(row, "nuisance", _region_stats(score, phantom.regions["nuisance"]))
                _add_prefixed(row, "global", _global_stats(score, phantom.regions["interior"]))
                rows.append(row)

    for noise_sigma in RAMP_NOISE_SIGMAS:
        case_id = f"uniform_n{_token(noise_sigma)}"
        noise = _noise_field(IMAGE_SIZE, noise_sigma, "uniform", noise_sigma)
        phantom = _uniform_phantom(size=IMAGE_SIZE, noise=noise)
        print(f"[phantom] uniforme {case_id}", flush=True)
        maps = _validated_maps(phantom.image)
        for map_name in _common_map_names(maps):
            score = maps[map_name]
            row = _base_row(
                family="uniform",
                case_id=case_id,
                map_name=map_name,
                noise_sigma=noise_sigma,
            )
            _add_prefixed(row, "nuisance", _region_stats(score, phantom.regions["nuisance"]))
            _add_prefixed(row, "global", _global_stats(score, phantom.regions["interior"]))
            rows.append(row)


def _benchmark_line_plus_shadow(
    rows: list[dict[str, object]],
    representatives: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    width = 4.0
    line_blur = 0.8
    shadow_blur = 1.7
    noise_sigma = 0.010
    for line_angle in STRESS_ANGLES_DEG:
        shadow_angle = (line_angle + 60.0) % 180.0
        case_id = f"cross_l{_token(line_angle)}_s{_token(shadow_angle)}"
        noise = _noise_field(IMAGE_SIZE, noise_sigma, "cross", line_angle, shadow_angle)
        clean = _line_plus_shadow_phantom(
            size=IMAGE_SIZE,
            line_angle_deg=line_angle,
            shadow_angle_deg=shadow_angle,
            width_px=width,
            line_blur_px=line_blur,
            shadow_blur_px=shadow_blur,
            noise=noise,
            with_shadow=False,
        )
        shadowed = _line_plus_shadow_phantom(
            size=IMAGE_SIZE,
            line_angle_deg=line_angle,
            shadow_angle_deg=shadow_angle,
            width_px=width,
            line_blur_px=line_blur,
            shadow_blur_px=shadow_blur,
            noise=noise,
            with_shadow=True,
        )
        print(f"[phantom] ligne + ombre {case_id}", flush=True)
        clean_maps = _validated_maps(clean.image)
        shadowed_maps = _validated_maps(shadowed.image)
        _capture_representative(representatives, "line_plus_shadow", shadowed, shadowed_maps)
        for map_name in _common_map_names(clean_maps, shadowed_maps):
            clean_center = _region_stats(clean_maps[map_name], clean.regions["core"])
            lit_center = _region_stats(shadowed_maps[map_name], shadowed.regions["lit_core"])
            shadowed_center = _region_stats(
                shadowed_maps[map_name], shadowed.regions["shadowed_core"]
            )
            boundary = _region_stats(
                shadowed_maps[map_name], shadowed.regions["shadow_boundary"]
            )
            retention, retention_defined = _safe_ratio(
                shadowed_center["mean"], clean_center["mean"]
            )
            boundary_ratio, boundary_defined = _safe_ratio(
                boundary["q95"], clean_center["q95"]
            )
            row = _base_row(
                family="line_plus_shadow",
                case_id=case_id,
                map_name=map_name,
                angle_deg=line_angle,
                width_px=width,
                blur_px=line_blur,
                noise_sigma=noise_sigma,
            )
            row["shadow_angle_deg"] = shadow_angle
            row["shadow_blur_px"] = shadow_blur
            _add_prefixed(row, "clean_center", clean_center)
            _add_prefixed(row, "lit_center", lit_center)
            _add_prefixed(row, "shadowed_center", shadowed_center)
            _add_prefixed(row, "shadow_boundary", boundary)
            _add_prefixed(
                row,
                "shadowed_global",
                _global_stats(shadowed_maps[map_name], shadowed.regions["interior"]),
            )
            row.update(
                {
                    "shadowed_crack_retention_mean": retention,
                    "shadowed_crack_retention_mean_defined": retention_defined,
                    "shadow_boundary_to_clean_ratio_q95": boundary_ratio,
                    "shadow_boundary_to_clean_ratio_q95_defined": boundary_defined,
                }
            )
            rows.append(row)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Aucune métrique à écrire")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                    for key, value in row.items()
                }
            )
    os.replace(temporary, path)


def _median_interval(values: Iterable[object]) -> tuple[float, float, float]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return math.nan, math.nan, math.nan
    low, median, high = np.quantile(finite, (0.25, 0.50, 0.75))
    return float(median), float(low), float(high)


def _ordered_maps(rows: Sequence[Mapping[str, object]]) -> list[str]:
    present = {str(row["map"]) for row in rows}
    ordered = [name for name in MAP_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _summary_figure(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    representatives: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    representative_order = (
        ("dark_line", "Ligne sombre"),
        ("shadow_step", "Marche d'ombre"),
        ("wide_line", "Ligne large"),
        ("ramp", "Rampe"),
        ("bright_line", "Ligne claire"),
        ("line_plus_shadow", "Ligne + ombre"),
    )
    missing = [name for name, _ in representative_order if name not in representatives]
    if missing:
        raise RuntimeError(f"Phantoms représentatifs manquants : {missing}")

    figure = plt.figure(figsize=(21, 16), constrained_layout=True)
    grid = figure.add_gridspec(3, 6, height_ratios=(1.0, 1.0, 2.4))
    for column, (name, title) in enumerate(representative_order):
        image, score = representatives[name]
        image_axis = figure.add_subplot(grid[0, column])
        image_axis.imshow(image)
        image_axis.set_title(title, fontsize=11)
        image_axis.axis("off")

        score_axis = figure.add_subplot(grid[1, column])
        score_axis.imshow(score, cmap="magma", vmin=0.0, vmax=1.0)
        score_axis.set_title(MAP_LABELS.get(REPRESENTATIVE_MAP, REPRESENTATIVE_MAP), fontsize=10)
        score_axis.axis("off")

    map_names = _ordered_maps(rows)
    pair_rows = [row for row in rows if row["family"] == "thin_line_vs_shadow_step"]
    ratio_axis = figure.add_subplot(grid[2, :3])
    medians: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for map_name in map_names:
        values = [
            row.get("step_to_line_ratio_q95")
            for row in pair_rows
            if row["map"] == map_name
            and int(row.get("step_to_line_ratio_q95_defined", 0)) == 1
        ]
        median, low, high = _median_interval(values)
        display_median = max(median, 1e-3) if math.isfinite(median) else 1e-3
        medians.append(display_median)
        lower_errors.append(max(display_median - max(low, 1e-3), 0.0) if math.isfinite(low) else 0.0)
        upper_errors.append(max(high - display_median, 0.0) if math.isfinite(high) else 0.0)
    positions = np.arange(len(map_names))
    ratio_axis.errorbar(
        medians,
        positions,
        xerr=np.vstack((lower_errors, upper_errors)),
        fmt="o",
        color="#375a7f",
        ecolor="#8aa4bf",
        capsize=3,
    )
    ratio_axis.axvline(1.0, color="#b22222", linestyle="--", linewidth=1.4)
    ratio_axis.set_xscale("log")
    ratio_axis.set_yticks(positions, [MAP_LABELS.get(name, name) for name in map_names])
    ratio_axis.invert_yaxis()
    ratio_axis.grid(axis="x", which="both", alpha=0.25)
    ratio_axis.set_xlabel("fuite marche q95 / réponse ligne q95 (< 1 préférable)")
    ratio_axis.set_title("Rejet d'une marche appariée — médiane et IQR")

    abstention_axis = figure.add_subplot(grid[2, 3:])
    x_positions = np.arange(len(map_names))
    bar_width = 0.38
    for offset, family, label, color in (
        (-bar_width / 2.0, "ramp", "Rampe", "#4c956c"),
        (bar_width / 2.0, "uniform", "Uniforme", "#f2cc8f"),
    ):
        values: list[float] = []
        for map_name in map_names:
            family_values = [
                row.get("global_abstention_fraction_010")
                for row in rows
                if row["family"] == family and row["map"] == map_name
            ]
            median, _, _ = _median_interval(family_values)
            values.append(median)
        abstention_axis.bar(x_positions + offset, values, bar_width, label=label, color=color)
    abstention_axis.set_ylim(0.0, 1.02)
    abstention_axis.set_ylabel(f"fraction ≤ {ABSTENTION_THRESHOLD:.2f} (1 préférable)")
    abstention_axis.set_title("Abstention en absence de ligne")
    abstention_axis.set_xticks(
        x_positions,
        [MAP_LABELS.get(name, name) for name in map_names],
        rotation=55,
        ha="right",
    )
    abstention_axis.grid(axis="y", alpha=0.25)
    abstention_axis.legend(loc="lower right")

    figure.suptitle(
        "Benchmark synthétique du guidage géométrique anti-ombre",
        fontsize=17,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=180, facecolor="white")
    plt.close(figure)
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _self_test() -> None:
    size = 96
    zero_noise = np.zeros((size, size), dtype=np.float32)
    uniform = _uniform_phantom(size=size, noise=zero_noise)
    uniform_maps = _validated_maps(uniform.image)
    for name, score in uniform_maps.items():
        _require(float(np.max(score)) <= 1e-4, f"{name} ne s'abstient pas sur l'uniforme")

    line = _line_phantom(
        size=size,
        angle_deg=0.0,
        width_px=4.0,
        blur_px=0.8,
        noise=zero_noise,
    )
    step = _step_phantom(
        size=size,
        angle_deg=0.0,
        blur_px=0.8,
        noise=zero_noise,
    )
    line_repeat = _line_phantom(
        size=size,
        angle_deg=0.0,
        width_px=4.0,
        blur_px=0.8,
        noise=zero_noise,
    )
    _require(np.array_equal(line.image, line_repeat.image), "Génération non déterministe")
    line_maps = _validated_maps(line.image)
    step_maps = _validated_maps(step.image)
    narrow_step_region = step.regions["boundary"] & (
        np.abs(_coordinates(size, 0.0)[0]) <= 3.0
    )
    for name in ("ofs", "paired_profile", "fusion_ofs_profile"):
        line_mean = _region_stats(line_maps[name], line.regions["core"])["mean"]
        step_mean = _region_stats(step_maps[name], narrow_step_region)["mean"]
        _require(
            line_mean > step_mean + 1e-3,
            f"{name}: la ligne sombre n'est pas préférée à la marche ({line_mean}, {step_mean})",
        )
    ratio, defined = _safe_ratio(0.1, 0.2)
    _require(defined == 1 and abs(ratio - 0.5) <= 1e-12, "Ratio interne incorrect")
    undefined_ratio, defined = _safe_ratio(0.0, 0.0)
    _require(defined == 0 and math.isnan(undefined_ratio), "Ratio nul mal signalé")
    print("Self-test phantom: OK", flush=True)


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return

    rows: list[dict[str, object]] = []
    representatives: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    _benchmark_primary_pairs(rows, representatives)
    _benchmark_wide_lines(rows, representatives)
    _benchmark_bright_polarity(rows, representatives)
    _benchmark_abstention(rows, representatives)
    _benchmark_line_plus_shadow(rows, representatives)

    output = args.output.resolve()
    csv_path = output / "tables/generated/phantom_metrics.csv"
    figure_path = output / "figures/generated/phantom_benchmark.png"
    _atomic_csv(csv_path, rows)
    _summary_figure(figure_path, rows, representatives)
    print(f"Métriques : {csv_path}", flush=True)
    print(f"Figure    : {figure_path}", flush=True)
    print(f"Lignes CSV: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
