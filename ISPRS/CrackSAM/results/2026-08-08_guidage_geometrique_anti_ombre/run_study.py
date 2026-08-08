#!/usr/bin/env python3
"""Exécute l'étude filtre-seul anti-ombre sur Khánh Hà et les cas archivés.

Le protocole est exploratoire mais entièrement déterministe :

* les 16 scènes Khánh Hà uniques du rapport 2026-07 sont toujours incluses ;
* chacune est rejouée dans les trois conditions CrackSAM ;
* un complément est tiré par SHA-256 depuis la liste officielle ;
* trois ombres synthétiques connues sont ajoutées aux scènes difficiles ;
* les six scènes Road420/Facade390 de la présentation sont ajoutées lorsque
  ces jeux sont présents sous ``--data-root``.

Le script ne charge ni SAM 2, ni LoRA, et ne lance aucun entraînement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import textwrap
import time
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import pandas as pd  # noqa: E402
from skimage.morphology import dilation, disk, skeletonize  # noqa: E402


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from anti_shadow_filters import compute_filter_bank  # noqa: E402
from ISPRS.CrackSAM.cracksam2.data import apply_noise_perturbation  # noqa: E402


CONDITIONS = ("original", "noisy1", "noisy2")
MANDATORY_PRESENTATION_CASES = {
    "Sylvie_Chambon_319.jpg": "original",
    "CRACK500_20160329_093924_1921_721.jpg": "noisy1",
    "Volker_DSC01646_226_19_1273_1645.jpg": "noisy2",
    "CRACK500_20160308_073532_1_361.jpg": "noisy2",
}
EXTERNAL_PRESENTATION_CASES = (
    ("road420", "2023_11_01_20_33_IMG_6353.jpg", "gain_frangi_ombre"),
    ("road420", "2023_10_30_16_44_IMG_6033.jpg", "perte_frangi"),
    ("road420", "2023_11_05_21_38_IMG_6516.jpg", "deux_bons"),
    ("facade390", "DJ_Wall_66.JPG", "gain_frangi"),
    ("facade390", "DJ_Wall_231.JPG", "perte_frangi"),
    ("facade390", "DJ_Wall_343.JPG", "deux_faibles"),
)
DISPLAY_MAPS = (
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
class CaseSpec:
    dataset: str
    case_name: str
    condition: str
    role: str
    qualitative: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Racine préparée par prepare_cracksam2_data.py.",
    )
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--random-cases",
        type=int,
        default=8,
        help="Complément déterministe aux 16 scènes difficiles (défaut: 8).",
    )
    parser.add_argument(
        "--shadow-cases",
        type=int,
        default=16,
        help="Nombre d'ancres recevant les trois ombres synthétiques (défaut: 16).",
    )
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_260_808)
    parser.add_argument(
        "--skip-case-figures",
        action="store_true",
        help="Ne génère pas les panneaux individuels (tables et atlas conservés).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprend les cas déjà validés dans les CSV générés sans les recalculer.",
    )
    return parser.parse_args()


def _stable_hash(*parts: str) -> bytes:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).digest()


def _slug(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return safe[:180]


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for source in rows:
            writer.writerow(
                {
                    key: ""
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                    for key, value in source.items()
                }
            )
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows: list[dict[str, object]] = []
        for source_row in csv.DictReader(source):
            row: dict[str, object] = {}
            for key, value in source_row.items():
                if value == "":
                    row[key] = math.nan
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
        return rows


def _prepare_output(output: Path, resume: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        return
    # Nettoyage strictement borné aux sorties possédées par ce script.
    for relative in ("figures/generated", "tables/generated"):
        path = output / relative
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    (output / "study_manifest.json").unlink(missing_ok=True)
    (output / "study_progress.json").unlink(missing_ok=True)


def _selected_khanhha_cases(random_cases: int) -> tuple[list[str], set[str]]:
    selected_csv = (
        REPOSITORY_ROOT
        / "ISPRS/CrackSAM/results/frangi_milestone_report/tables/selected_cases.csv"
    )
    with selected_csv.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    anchors = sorted(
        {
            row["case_name"]
            for row in rows
            if row["dataset"].startswith("khanhha_")
        }
    )
    split = (
        REPOSITORY_ROOT
        / "ISPRS/CrackSAM/protocol/cracksam_paper/lists/lists_khanhha/test_vol.txt"
    )
    pool = [
        line.strip()
        for line in split.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    missing = set(anchors) - set(pool)
    if missing:
        raise RuntimeError(f"Ancres absentes du split officiel: {sorted(missing)}")
    remainder = [name for name in pool if name not in anchors]
    remainder.sort(key=lambda name: _stable_hash("anti-shadow-v1", name))
    return anchors + remainder[: max(0, int(random_cases))], set(anchors)


def _build_cases(data_root: Path, random_cases: int) -> tuple[list[CaseSpec], list[str]]:
    names, anchors = _selected_khanhha_cases(random_cases)
    cases = [
        CaseSpec(
            dataset="khanhha",
            case_name=name,
            condition=condition,
            role="anchor" if name in anchors else "hash_sample",
            qualitative=name in anchors,
        )
        for name in names
        for condition in CONDITIONS
    ]
    missing_external: list[str] = []
    for dataset, case_name, role in EXTERNAL_PRESENTATION_CASES:
        image_path = data_root / dataset / "images" / case_name
        if image_path.is_file():
            cases.append(CaseSpec(dataset, case_name, "original", role, True))
        else:
            missing_external.append(f"{dataset}/{case_name}")
    return cases, missing_external


def _dataset_root(data_root: Path, dataset: str) -> Path:
    return data_root / "khanhha" / "test" if dataset == "khanhha" else data_root / dataset


def _case_paths(data_root: Path, dataset: str, case_name: str) -> tuple[Path, Path]:
    root = _dataset_root(data_root, dataset)
    image = root / "images" / case_name
    mask = root / "masks" / case_name
    if not image.is_file() or not mask.is_file():
        raise FileNotFoundError(f"Paire image/masque absente: {image}, {mask}")
    return image, mask


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        raw = np.asarray(source)
        if raw.ndim == 3 and raw.shape[2] == 4 and raw[..., 3].min() != raw[..., 3].max():
            gray = raw[..., 3]
        else:
            gray = np.asarray(source.convert("L"))
    return np.asarray(gray, dtype=np.float32) > 127.5


def _load_case(
    data_root: Path,
    spec: CaseSpec,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    image_path, mask_path = _case_paths(data_root, spec.dataset, spec.case_name)
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.uint8)
    if spec.dataset == "khanhha":
        image = apply_noise_perturbation(image, spec.condition, output_size=(image_size, image_size))
    if image.shape[:2] != (image_size, image_size):
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    mask = _read_mask(mask_path)
    if mask.shape != (image_size, image_size):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image_size, image_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return image, mask


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    target_flat = np.asarray(target, dtype=bool).ravel()
    score_flat = np.asarray(score, dtype=np.float64).ravel()
    positives = int(np.count_nonzero(target_flat))
    if positives == 0:
        return math.nan
    if float(np.ptp(score_flat)) <= 1e-12:
        return positives / target_flat.size
    order = np.argsort(-score_flat, kind="mergesort")
    sorted_target = target_flat[order]
    precision = np.cumsum(sorted_target) / np.arange(1, sorted_target.size + 1)
    return float(np.sum(precision[sorted_target]) / positives)


def _topk_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    flat = np.asarray(score, dtype=np.float32).ravel()
    count = max(1, min(flat.size, int(round(flat.size * fraction))))
    indices = np.argpartition(flat, -count)[-count:]
    selected = np.zeros(flat.shape, dtype=bool)
    selected[indices] = True
    return selected.reshape(score.shape)


def _map_metrics(mask: np.ndarray, score: np.ndarray, tolerance: int = 2) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    score = np.nan_to_num(np.asarray(score, dtype=np.float32), nan=0.0).clip(0.0, 1.0)
    tolerant = dilation(mask, footprint=disk(tolerance)) if np.any(mask) else mask
    skeleton = skeletonize(mask) if np.any(mask) else mask
    total_mass = float(np.sum(score))
    result = {
        "target_fraction": float(np.mean(mask)),
        "score_mean": float(np.mean(score)),
        "score_q99": float(np.quantile(score, 0.99)),
        "score_max": float(np.max(score)),
        "active_fraction_010": float(np.mean(score > 0.10)),
        "response_mass_near_gt": (
            float(np.sum(score[tolerant]) / total_mass) if total_mass > 0.0 else 0.0
        ),
        "ap_exact": _average_precision(mask, score),
        "ap_tol2": _average_precision(tolerant, score),
    }
    for fraction in (0.005, 0.01, 0.02, 0.05):
        selected = _topk_mask(score, fraction)
        suffix = str(fraction).replace("0.", "p")
        result[f"precision_top_{suffix}"] = float(np.mean(tolerant[selected]))
        if np.any(skeleton):
            reached = dilation(selected, footprint=disk(tolerance))
            result[f"skeleton_recall_top_{suffix}"] = float(np.mean(reached[skeleton]))
        else:
            result[f"skeleton_recall_top_{suffix}"] = math.nan
    return result


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _synthetic_shadow(
    image: np.ndarray,
    kind: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    y, x = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0.25, 1.25) * np.pi)
    offset = float(rng.uniform(-0.2, 0.2))
    coordinate = np.cos(angle) * x + np.sin(angle) * y - offset
    if kind == "hard":
        attenuation = np.where(coordinate < 0.0, 0.28, 1.0)
        tint = np.array((0.92, 0.98, 1.08), dtype=np.float32)
    elif kind == "soft":
        transition = 1.0 / (1.0 + np.exp(-coordinate / 0.055))
        attenuation = 0.34 + 0.66 * transition
        tint = np.array((0.96, 1.0, 1.05), dtype=np.float32)
    elif kind == "curved":
        cx, cy = rng.uniform(-0.25, 0.25, size=2)
        ax, ay = rng.uniform(0.48, 0.72), rng.uniform(0.35, 0.58)
        ellipse = ((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2
        inside = (ellipse <= 1.0).astype(np.float32)
        softness = ndi.gaussian_filter(inside, sigma=max(2.0, width / 75.0), mode="nearest")
        attenuation = 1.0 - 0.68 * softness
        tint = np.array((0.88, 0.98, 1.12), dtype=np.float32)
    else:
        raise ValueError(kind)

    linear = _srgb_to_linear(image)
    shadowed = linear * attenuation[..., None] * tint[None, None, :]
    shadowed = _linear_to_srgb(shadowed)
    shadow_region = attenuation < 0.8
    attenuation_float = attenuation.astype(np.float32)
    grad_y, grad_x = np.gradient(attenuation_float)
    edge = np.hypot(grad_x, grad_y) > 0.005
    boundary = dilation(edge, footprint=disk(3))
    return shadowed, shadow_region, boundary


def _shadow_metrics(
    clean_score: np.ndarray,
    shadow_score: np.ndarray,
    target: np.ndarray,
    boundary: np.ndarray,
) -> dict[str, float]:
    target_band = dilation(target, footprint=disk(2)) if np.any(target) else target
    background_boundary = boundary & ~target_band
    skeleton = skeletonize(target) if np.any(target) else target
    clean_crack = float(np.mean(clean_score[skeleton])) if np.any(skeleton) else math.nan
    shadow_crack = float(np.mean(shadow_score[skeleton])) if np.any(skeleton) else math.nan
    clean_edge = (
        float(np.mean(clean_score[background_boundary]))
        if np.any(background_boundary)
        else math.nan
    )
    shadow_edge = (
        float(np.mean(shadow_score[background_boundary]))
        if np.any(background_boundary)
        else math.nan
    )
    return {
        "clean_crack_response": clean_crack,
        "shadow_crack_response": shadow_crack,
        "crack_retention": shadow_crack / (clean_crack + 1e-8) if math.isfinite(clean_crack) else math.nan,
        "clean_boundary_response": clean_edge,
        "shadow_boundary_response": shadow_edge,
        "boundary_response_increase": shadow_edge - clean_edge
        if math.isfinite(clean_edge) and math.isfinite(shadow_edge)
        else math.nan,
    }


def _case_figure(
    output: Path,
    spec: CaseSpec,
    image: np.ndarray,
    mask: np.ndarray,
    maps: Mapping[str, np.ndarray],
) -> None:
    path = _case_figure_path(output, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = 5
    panels: list[tuple[str, np.ndarray, str | None]] = [
        ("Entrée", image, None),
        ("Vérité terrain", mask, "gray"),
    ] + [(MAP_LABELS[name], maps[name], "magma") for name in DISPLAY_MAPS]
    rows = int(math.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 2.8 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for axis, (title, panel, cmap) in zip(axes_array, panels):
        if cmap is None:
            axis.imshow(panel)
        else:
            axis.imshow(panel, cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    for axis in axes_array[len(panels) :]:
        axis.axis("off")
    figure.suptitle(f"{spec.dataset} · {spec.condition} · {spec.case_name}", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _case_figure_path(output: Path, spec: CaseSpec) -> Path:
    return (
        output
        / "figures/generated/cases"
        / spec.dataset
        / spec.condition
        / f"{_slug(spec.case_name)}.png"
    )


def _atlas(
    path: Path,
    records: Sequence[tuple[str, np.ndarray, np.ndarray, Mapping[str, np.ndarray]]],
    map_names: Sequence[str],
) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = 2 + len(map_names)
    figure, axes = plt.subplots(
        len(records),
        columns,
        figsize=(2.35 * columns, 2.25 * len(records)),
        squeeze=False,
    )
    for row, (label, image, mask, maps) in enumerate(records):
        axes[row, 0].imshow(image)
        wrapped_label = "\n".join(textwrap.wrap(label.replace("\n", " · "), width=25))
        axes[row, 0].set_ylabel(
            wrapped_label,
            fontsize=7,
            rotation=0,
            ha="right",
            va="center",
            labelpad=6,
        )
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        for column, name in enumerate(map_names, start=2):
            axes[row, column].imshow(maps[name], cmap="magma", vmin=0.0, vmax=1.0)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    titles = ("Entrée", "GT", *(MAP_LABELS[name] for name in map_names))
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def _bootstrap_interval(values: np.ndarray, repetitions: int, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    draws = rng.integers(0, values.size, size=(repetitions, values.size))
    means = np.mean(values[draws], axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _summaries(rows: Sequence[Mapping[str, object]], bootstrap: int, seed: int) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    result: list[dict[str, object]] = []
    metrics = (
        "ap_exact",
        "ap_tol2",
        "precision_top_p01",
        "skeleton_recall_top_p01",
        "response_mass_near_gt",
        "active_fraction_010",
    )
    rng = np.random.default_rng(seed)
    for (condition, map_name), group in frame.groupby(["condition", "map"]):
        row: dict[str, object] = {
            "condition": condition,
            "map": map_name,
            "n": int(len(group)),
            "nonempty_targets": int(np.count_nonzero(group["target_fraction"].to_numpy() > 0)),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else math.nan
            row[f"{metric}_median"] = float(np.median(finite)) if finite.size else math.nan
            low, high = _bootstrap_interval(values, bootstrap, rng)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        result.append(row)
    return result


def _summary_figure(path: Path, summary_rows: Sequence[Mapping[str, object]]) -> None:
    frame = pd.DataFrame(summary_rows)
    original = frame[frame["condition"] == "original"].copy()
    if original.empty:
        return
    order = original.sort_values("ap_tol2_mean", ascending=False)["map"].tolist()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, metric, title in (
        (axes[0], "ap_tol2_mean", "AP tolérante (2 px)"),
        (axes[1], "precision_top_p01_mean", "Précision du top 1 %"),
    ):
        values = [float(original.loc[original["map"] == name, metric].iloc[0]) for name in order]
        colors = ["#b31b34" if name.startswith("frangi") else "#3478a8" for name in order]
        axis.barh(range(len(order)), values, color=colors)
        axis.set_yticks(range(len(order)), [MAP_LABELS.get(name, name) for name in order], fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, max(0.05, max(values) * 1.08))
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Cartes géométriques seules — scènes originales")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _shadow_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    result: list[dict[str, object]] = []
    for (kind, map_name), group in frame.groupby(["shadow_kind", "map"]):
        result.append(
            {
                "shadow_kind": kind,
                "map": map_name,
                "n": int(len(group)),
                "crack_retention_median": float(np.nanmedian(group["crack_retention"])),
                "boundary_response_increase_mean": float(
                    np.nanmean(group["boundary_response_increase"])
                ),
                "shadow_boundary_response_mean": float(
                    np.nanmean(group["shadow_boundary_response"])
                ),
            }
        )
    return result


def main() -> None:
    args = _parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.image_size < 96:
        raise ValueError("--image-size doit être >= 96")
    _prepare_output(output, args.resume)
    cases, missing_external = _build_cases(data_root, args.random_cases)
    manifest_rows = [spec.__dict__ for spec in cases]
    _atomic_csv(output / "tables/generated/cases_manifest.csv", manifest_rows)

    metric_path = output / "tables/generated/per_case_metrics.csv"
    shadow_path = output / "tables/generated/shadow_stress_per_case.csv"
    progress_path = output / "study_progress.json"
    metric_rows = _read_csv_rows(metric_path) if args.resume else []
    shadow_rows = _read_csv_rows(shadow_path) if args.resume else []
    previous_elapsed = 0.0
    if args.resume and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        previous_elapsed = float(progress.get("elapsed_seconds", 0.0))
    elif args.resume and (output / "study_manifest.json").is_file():
        previous_manifest = json.loads(
            (output / "study_manifest.json").read_text(encoding="utf-8")
        )
        previous_elapsed = float(previous_manifest.get("elapsed_seconds", 0.0))
    started = time.perf_counter()

    def checkpoint() -> None:
        _atomic_json(
            progress_path,
            {
                "schema_version": 1,
                "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                "metric_rows": len(metric_rows),
                "shadow_rows": len(shadow_rows),
            },
        )

    metric_keys = {
        (str(row["dataset"]), str(row["case_name"]), str(row["condition"]), str(row["map"]))
        for row in metric_rows
    }
    if len(metric_keys) != len(metric_rows):
        raise RuntimeError("Le CSV de reprise contient des lignes métriques dupliquées")
    expected_maps = set(DISPLAY_MAPS)
    for index, spec in enumerate(cases, start=1):
        prefix = (spec.dataset, spec.case_name, spec.condition)
        completed_maps = {
            map_name
            for dataset, case_name, condition, map_name in metric_keys
            if (dataset, case_name, condition) == prefix
        }
        figure_ready = (
            not spec.qualitative
            or args.skip_case_figures
            or _case_figure_path(output, spec).is_file()
        )
        if completed_maps == expected_maps and figure_ready:
            print(
                f"[{index:03d}/{len(cases):03d}] repris {spec.dataset}/{spec.condition}/"
                f"{spec.case_name}",
                flush=True,
            )
            continue
        metrics_complete = completed_maps == expected_maps
        if completed_maps and not metrics_complete:
            metric_rows = [
                row
                for row in metric_rows
                if (str(row["dataset"]), str(row["case_name"]), str(row["condition"]))
                != prefix
            ]
            metric_keys = {
                key for key in metric_keys if key[:3] != prefix
            }
        image, mask = _load_case(data_root, spec, args.image_size)
        bank = compute_filter_bank(image)
        if set(bank.maps) != expected_maps:
            raise RuntimeError(
                f"Cartes inattendues pour {spec.case_name}: {sorted(bank.maps)}"
            )
        print(
            f"[{index:03d}/{len(cases):03d}] {spec.dataset}/{spec.condition}/{spec.case_name}",
            flush=True,
        )
        if not metrics_complete:
            for map_name, score in bank.maps.items():
                metric_rows.append(
                    {
                        **spec.__dict__,
                        "map": map_name,
                        **_map_metrics(mask, score),
                    }
                )
                metric_keys.add((*prefix, map_name))
        if spec.qualitative and not args.skip_case_figures:
            _case_figure(output, spec, image, mask, bank.maps)
        _atomic_csv(metric_path, metric_rows)
        checkpoint()

    expected_metric_rows = len(cases) * len(DISPLAY_MAPS)
    if len(metric_rows) != expected_metric_rows:
        raise RuntimeError(
            f"Campagne réelle incomplète: {len(metric_rows)}/{expected_metric_rows} lignes"
        )
    summary_rows = _summaries(metric_rows, args.bootstrap, args.seed)
    _atomic_csv(output / "tables/generated/summary_metrics.csv", summary_rows)

    anchor_names = {
        spec.case_name
        for spec in cases
        if spec.dataset == "khanhha" and spec.role == "anchor"
    }
    khanhha_names = {
        spec.case_name for spec in cases if spec.dataset == "khanhha"
    }
    mandatory_shadow_names = sorted(set(MANDATORY_PRESENTATION_CASES) & khanhha_names)
    other_anchor_names = sorted(
        anchor_names - set(mandatory_shadow_names),
        key=lambda name: _stable_hash("shadow-anchor", name),
    )
    complementary_names = sorted(
        khanhha_names - anchor_names,
        key=lambda name: _stable_hash("shadow-complement", name),
    )
    shadow_names = (mandatory_shadow_names + other_anchor_names + complementary_names)[
        : max(0, int(args.shadow_cases))
    ]
    shadow_keys = {
        (str(row["case_name"]), str(row["shadow_kind"]), str(row["map"]))
        for row in shadow_rows
    }
    if len(shadow_keys) != len(shadow_rows):
        raise RuntimeError("Le CSV de reprise contient des lignes d'ombre dupliquées")
    for case_name in shadow_names:
        complete_case = all(
            (case_name, shadow_kind, map_name) in shadow_keys
            for shadow_kind in ("hard", "soft", "curved")
            for map_name in DISPLAY_MAPS
        )
        if complete_case:
            print(f"[ombre] repris {case_name}", flush=True)
            continue
        clean_spec = CaseSpec("khanhha", case_name, "original", "anchor", False)
        clean_image, mask = _load_case(data_root, clean_spec, args.image_size)
        clean_maps = compute_filter_bank(clean_image).maps
        for shadow_kind in ("hard", "soft", "curved"):
            completed_shadow_maps = {
                map_name
                for stored_case, stored_kind, map_name in shadow_keys
                if stored_case == case_name and stored_kind == shadow_kind
            }
            if completed_shadow_maps == expected_maps:
                print(f"[ombre] repris {shadow_kind}/{case_name}", flush=True)
                continue
            if completed_shadow_maps:
                shadow_rows = [
                    row
                    for row in shadow_rows
                    if not (
                        str(row["case_name"]) == case_name
                        and str(row["shadow_kind"]) == shadow_kind
                    )
                ]
                shadow_keys = {
                    key
                    for key in shadow_keys
                    if not (key[0] == case_name and key[1] == shadow_kind)
                }
            seed = int.from_bytes(_stable_hash(str(args.seed), case_name, shadow_kind)[:8], "little")
            shadow_image, shadow_region, boundary = _synthetic_shadow(
                clean_image, shadow_kind, seed
            )
            shadow_maps = compute_filter_bank(shadow_image).maps
            print(f"[ombre] {shadow_kind}/{case_name}", flush=True)
            for map_name, score in shadow_maps.items():
                shadow_rows.append(
                    {
                        "case_name": case_name,
                        "shadow_kind": shadow_kind,
                        "map": map_name,
                        "shadow_fraction": float(np.mean(shadow_region)),
                        "boundary_fraction": float(np.mean(boundary)),
                        **_shadow_metrics(clean_maps[map_name], score, mask, boundary),
                    }
                )
                shadow_keys.add((case_name, shadow_kind, map_name))
            _atomic_csv(shadow_path, shadow_rows)
            checkpoint()

    expected_shadow_rows = len(shadow_names) * 3 * len(DISPLAY_MAPS)
    if len(shadow_rows) != expected_shadow_rows:
        raise RuntimeError(
            f"Stress-test incomplet: {len(shadow_rows)}/{expected_shadow_rows} lignes"
        )
    shadow_summary = _shadow_summary(shadow_rows)
    _atomic_csv(output / "tables/generated/shadow_stress_summary.csv", shadow_summary)

    mandatory_atlas: list[tuple[str, np.ndarray, np.ndarray, Mapping[str, np.ndarray]]] = []
    for case_name, condition in MANDATORY_PRESENTATION_CASES.items():
        spec = CaseSpec("khanhha", case_name, condition, "presentation", True)
        image, mask = _load_case(data_root, spec, args.image_size)
        mandatory_atlas.append(
            (f"{condition}\n{case_name}", image, mask, compute_filter_bank(image).maps)
        )
    external_atlas: list[tuple[str, np.ndarray, np.ndarray, Mapping[str, np.ndarray]]] = []
    for dataset, case_name, role in EXTERNAL_PRESENTATION_CASES:
        spec = CaseSpec(dataset, case_name, "original", role, True)
        image, mask = _load_case(data_root, spec, args.image_size)
        external_atlas.append(
            (f"{dataset}\n{case_name}", image, mask, compute_filter_bank(image).maps)
        )
    shadow_atlas: list[tuple[str, np.ndarray, np.ndarray, Mapping[str, np.ndarray]]] = []
    for case_name in sorted(MANDATORY_PRESENTATION_CASES):
        spec = CaseSpec("khanhha", case_name, "original", "presentation", True)
        clean_image, mask = _load_case(data_root, spec, args.image_size)
        seed = int.from_bytes(_stable_hash(str(args.seed), case_name, "hard")[:8], "little")
        shadow_image, _, _ = _synthetic_shadow(clean_image, "hard", seed)
        shadow_atlas.append(
            (case_name, shadow_image, mask, compute_filter_bank(shadow_image).maps)
        )
    mandatory_atlas.sort(key=lambda item: item[0])
    _atlas(
        output / "figures/generated/atlas_presentation_khanhha.png",
        mandatory_atlas,
        (
            "frangi_similarity_historical",
            "ofs",
            "line_step_bic",
            "verified_frangi_v2",
            "verified_frangi_consensus",
        ),
    )
    _atlas(
        output / "figures/generated/atlas_presentation_external.png",
        external_atlas,
        (
            "frangi_similarity_historical",
            "ofs",
            "line_step_bic",
            "verified_frangi_v2",
            "verified_frangi_consensus",
        ),
    )
    _atlas(
        output / "figures/generated/atlas_ombres_synthetiques.png",
        shadow_atlas,
        (
            "frangi_similarity_historical",
            "ofs",
            "line_step_bic",
            "verified_frangi_v2",
            "verified_frangi_consensus",
        ),
    )
    _summary_figure(output / "figures/generated/metric_overview.png", summary_rows)

    manifest = {
        "schema_version": 1,
        "data_root_runtime": str(data_root),
        "image_size": args.image_size,
        "random_cases": args.random_cases,
        "shadow_cases": len(shadow_names),
        "conditions": list(CONDITIONS),
        "maps": list(DISPLAY_MAPS),
        "cases": len(cases),
        "metric_rows": len(metric_rows),
        "shadow_rows": len(shadow_rows),
        "missing_external_cases": missing_external,
        "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
        "seed": args.seed,
        "bootstrap": args.bootstrap,
    }
    _atomic_json(output / "study_manifest.json", manifest)
    progress_path.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
