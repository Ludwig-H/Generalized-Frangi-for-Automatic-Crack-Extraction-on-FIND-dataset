#!/usr/bin/env python3
"""Exploratory CPU-only audit of luminance vs chrominance Frangi prompts.

The original RGB datasets are not versioned in this checkout.  This script
therefore extracts the aligned 216 x 216 RGB and target tiles embedded in the
qualitative JPEG panels produced by ``generate_frangi_milestone_report.py``.
Every candidate channel is evaluated from that exact same reconstructed RGB
tile.  The archived prompts are shown only as context; they came from the
original 448 x 448 images and are not a strict numerical reference.

The script deliberately forces the maintained Frangi implementation onto CPU.
It does not contain cloud or VM operations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

# This must be set before importing torch through the CrackSAM modules.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage, stats
from sklearn.metrics import average_precision_score

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
if str(CRACKSAM_ROOT) not in sys.path:
    sys.path.insert(0, str(CRACKSAM_ROOT))

try:
    from .cracksam2.frangi import extract_frangi_graph_gpu
except ImportError:
    from cracksam2.frangi import extract_frangi_graph_gpu


PANEL_RGB_BOX = (41, 101, 257, 317)
PANEL_TARGET_BOX = (309, 101, 525, 317)
PANEL_SIZE = (216, 216)
ORIGINAL_SIZE = (448, 448)
SCALED_FRANGI_SCALES = (0.5, 1.5, 2.5, 4.5, 7.25)
SCALED_GRAPH_RADIUS = 2

# The six most negative and six most positive archived examples form a
# deliberately balanced visual probe, not a representative dataset sample.
CASE_ORDER = (
    ("khanhha_original", "gain_baseline"),
    ("khanhha_noisy1", "gain_baseline"),
    ("khanhha_noisy2", "gain_baseline"),
    ("road420", "gain_baseline"),
    ("facade390", "gain_baseline"),
    ("concrete3k", "gain_baseline"),
    ("khanhha_original", "gain_frangi"),
    ("khanhha_noisy1", "gain_frangi"),
    ("khanhha_noisy2", "gain_frangi"),
    ("road420", "gain_frangi"),
    ("facade390", "gain_frangi"),
    ("concrete3k", "gain_frangi"),
)

FULL_RESOLUTION_SENSITIVITY_KEYS = {
    ("khanhha_original", "gain_baseline"),
    ("khanhha_noisy2", "gain_baseline"),
    ("road420", "gain_baseline"),
    ("road420", "gain_frangi"),
}

VARIANT_ORDER = (
    "luminance",
    "lab_cstar_dark",
    "lab_cstar_bipolar",
    "lab_astar_bipolar",
    "lab_bstar_bipolar",
    "lab_ab_opponent",
    "hsv_saturation_bipolar",
    "log_chroma_pc1_bipolar",
)

VARIANT_LABELS = {
    "luminance": "Luminance Y",
    "lab_cstar_dark": "Lab C*",
    "lab_cstar_bipolar": "Lab C* +/-",
    "lab_astar_bipolar": "Lab a* +/-",
    "lab_bstar_bipolar": "Lab b* +/-",
    "lab_ab_opponent": "Lab max(a*,b*) +/-",
    "hsv_saturation_bipolar": "Saturation HSV +/-",
    "log_chroma_pc1_bipolar": "Log-chroma PC1 +/-",
}


@dataclass(frozen=True)
class Case:
    dataset: str
    category: str
    case_name: str
    delta_iou: float
    target_pixels: int
    panel_path: Path
    prompt_path: Path

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.category}__{self.case_name}"

    @property
    def slug(self) -> str:
        safe = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in self.case_name
        )
        return f"{self.dataset}__{self.category}__{safe}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=CRACKSAM_ROOT / "results" / "frangi_milestone_report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CRACKSAM_ROOT / "results" / "frangi_chrominance_cpu_probe",
    )
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(report_root: Path) -> list[Case]:
    selected = read_csv(report_root / "tables" / "selected_cases.csv")
    prompts = read_csv(report_root / "tables" / "prompt_manifest.csv")
    selected_by_key = {
        (row["dataset"], row["category"], row["case_name"]): row
        for row in selected
    }
    prompt_by_key = {
        (row["dataset"], row["category"], row["case_name"]): row
        for row in prompts
    }
    cases: list[Case] = []
    for dataset, category in CASE_ORDER:
        matches = [
            key
            for key in selected_by_key
            if key[0] == dataset and key[1] == category
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one selected case for {(dataset, category)}, got {matches}"
            )
        key = matches[0]
        row = selected_by_key[key]
        prompt_row = prompt_by_key[key]
        target_pixels_text = row.get("target_pixels") or row["target_pixels_at_selection"]
        cases.append(
            Case(
                dataset=dataset,
                category=category,
                case_name=key[2],
                delta_iou=float(row["delta_iou"]),
                target_pixels=int(target_pixels_text),
                panel_path=report_root / row["panel"],
                prompt_path=report_root / prompt_row["copied_path"],
            )
        )
    return cases


def extract_panel_tiles(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Image.open(case.panel_path) as panel_image:
        panel = panel_image.convert("RGB")
        if panel.size != (2160, 360):
            raise ValueError(f"Unexpected panel size {panel.size}: {case.panel_path}")
        rgb = np.asarray(panel.crop(PANEL_RGB_BOX), dtype=np.uint8)
        target_rgb = np.asarray(panel.crop(PANEL_TARGET_BOX), dtype=np.uint8)

    if rgb.shape[:2] != PANEL_SIZE or target_rgb.shape[:2] != PANEL_SIZE:
        raise AssertionError((rgb.shape, target_rgb.shape))

    # JPEG antialiasing makes a direct 0.5 threshold erase thin cracks.  The
    # report records the exact 448^2 target support, so retain the corresponding
    # number of brightest pixels at 216^2 as an explicitly approximate proxy.
    target_soft = target_rgb.astype(np.float32).mean(axis=2) / 255.0
    expected = int(
        round(case.target_pixels * PANEL_SIZE[0] * PANEL_SIZE[1]
              / (ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1]))
    )
    expected = min(max(expected, 0), target_soft.size)
    target = np.zeros(target_soft.size, dtype=bool)
    if expected:
        indices = np.argpartition(target_soft.ravel(), -expected)[-expected:]
        target[indices] = True
    return rgb, target.reshape(PANEL_SIZE), target_soft


def load_archived_prompt(path: Path) -> np.ndarray:
    logits = np.load(path, allow_pickle=False)
    logits = np.asarray(logits, dtype=np.float32).squeeze()
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return cv2.resize(probability, PANEL_SIZE[::-1], interpolation=cv2.INTER_LINEAR)


def normalize_fixed(value: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((value.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def chrominance_channels(rgb: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    rgb_float = rgb.astype(np.float32) / 255.0
    luminance = np.tensordot(
        rgb_float, np.asarray((0.2989, 0.5870, 0.1140), dtype=np.float32), axes=1
    )

    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    astar = lab[..., 1]
    bstar = lab[..., 2]
    cstar = np.hypot(astar, bstar)
    astar_01 = normalize_fixed(astar, -128.0, 127.0)
    bstar_01 = normalize_fixed(bstar, -128.0, 127.0)
    cstar_01 = np.clip(cstar / math.hypot(128.0, 128.0), 0.0, 1.0)

    hsv = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2HSV)
    saturation = np.clip(hsv[..., 1], 0.0, 1.0)

    # An illumination-ratio representation.  Very dark pixels are excluded
    # from the PCA and filled with its neutral median to avoid division/log
    # artefacts becoming artificial line boundaries.
    epsilon = 1.0 / 255.0
    red, green, blue = (rgb_float[..., index] for index in range(3))
    log_u = np.log(red + epsilon) - np.log(green + epsilon)
    log_v = np.log(blue + epsilon) - np.log(green + epsilon)
    brightness = rgb_float.mean(axis=2)
    valid = brightness > 0.04
    opponent = np.stack((log_u, log_v), axis=-1)
    valid_values = opponent[valid]
    if valid_values.shape[0] < 16:
        pc1_01 = np.full(PANEL_SIZE, 0.5, dtype=np.float32)
        pc1_span = 0.0
    else:
        center = np.median(valid_values, axis=0)
        centered = valid_values - center
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        projection = (opponent - center) @ direction
        valid_projection = projection[valid]
        low, high = np.percentile(valid_projection, (1.0, 99.0))
        pc1_span = float(high - low)
        if pc1_span <= 1e-6:
            pc1_01 = np.full(PANEL_SIZE, 0.5, dtype=np.float32)
        else:
            pc1_01 = np.clip((projection - low) / pc1_span, 0.0, 1.0)
            neutral = float(np.median(pc1_01[valid]))
            pc1_01[~valid] = neutral

    channels = {
        "luminance": luminance.astype(np.float32),
        "lab_cstar": cstar_01.astype(np.float32),
        "lab_astar": astar_01.astype(np.float32),
        "lab_bstar": bstar_01.astype(np.float32),
        "hsv_saturation": saturation.astype(np.float32),
        "log_chroma_pc1": pc1_01.astype(np.float32),
    }
    dynamics = {
        "lab_cstar_p99_minus_p1_delta_e": float(
            np.percentile(cstar, 99.0) - np.percentile(cstar, 1.0)
        ),
        "lab_astar_p99_minus_p1": float(
            np.percentile(astar, 99.0) - np.percentile(astar, 1.0)
        ),
        "lab_bstar_p99_minus_p1": float(
            np.percentile(bstar, 99.0) - np.percentile(bstar, 1.0)
        ),
        "hsv_saturation_p99_minus_p1": float(
            np.percentile(saturation, 99.0) - np.percentile(saturation, 1.0)
        ),
        "log_chroma_pc1_p99_minus_p1": pc1_span,
        "dark_pixel_fraction": float(np.mean(~valid)),
    }
    return channels, dynamics


def frangi_similarity(channel: np.ndarray) -> np.ndarray:
    if channel.shape != PANEL_SIZE:
        raise ValueError(channel.shape)
    _, similarity, _, _, _ = extract_frangi_graph_gpu(
        {"visible": np.asarray(channel, dtype=np.float32)},
        {"visible": 1.0},
        scales=SCALED_FRANGI_SCALES,
        R=SCALED_GRAPH_RADIUS,
        K=1,
        device="cpu",
        compute_centrality=False,
    )
    return np.asarray(similarity, dtype=np.float32)


def calculate_maps(channels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    primitive: dict[str, np.ndarray] = {}
    primitive["luminance"] = frangi_similarity(channels["luminance"])
    for name in (
        "lab_cstar",
        "lab_astar",
        "lab_bstar",
        "hsv_saturation",
        "log_chroma_pc1",
    ):
        primitive[name] = frangi_similarity(channels[name])
        primitive[f"{name}_inverse"] = frangi_similarity(1.0 - channels[name])

    maps = {
        "luminance": primitive["luminance"],
        "lab_cstar_dark": primitive["lab_cstar"],
        "lab_cstar_bipolar": np.maximum(
            primitive["lab_cstar"], primitive["lab_cstar_inverse"]
        ),
        "lab_astar_bipolar": np.maximum(
            primitive["lab_astar"], primitive["lab_astar_inverse"]
        ),
        "lab_bstar_bipolar": np.maximum(
            primitive["lab_bstar"], primitive["lab_bstar_inverse"]
        ),
        "hsv_saturation_bipolar": np.maximum(
            primitive["hsv_saturation"], primitive["hsv_saturation_inverse"]
        ),
        "log_chroma_pc1_bipolar": np.maximum(
            primitive["log_chroma_pc1"], primitive["log_chroma_pc1_inverse"]
        ),
    }
    maps["lab_ab_opponent"] = np.maximum(
        maps["lab_astar_bipolar"], maps["lab_bstar_bipolar"]
    )
    return maps


def top_positive_mask(score: np.ndarray, requested: int) -> tuple[np.ndarray, int]:
    flat = score.ravel()
    positive = np.flatnonzero(flat > 1e-8)
    effective = min(int(requested), int(positive.size))
    result = np.zeros(flat.size, dtype=bool)
    if effective:
        selected_relative = np.argpartition(flat[positive], -effective)[-effective:]
        result[positive[selected_relative]] = True
    return result.reshape(score.shape), effective


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    positive = weights > 0
    if not np.any(positive):
        return float("nan")
    values = values[positive]
    weights = weights[positive]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    threshold = quantile * cumulative[-1]
    return float(values[min(np.searchsorted(cumulative, threshold), len(values) - 1)])


def map_metrics(
    score: np.ndarray,
    target: np.ndarray,
    *,
    target_soft: np.ndarray,
) -> dict[str, float | int]:
    score = np.asarray(score, dtype=np.float32)
    target = np.asarray(target, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    target_tolerant = cv2.dilate(target.astype(np.uint8), kernel).astype(bool)
    target_count = int(np.count_nonzero(target))

    if target_count:
        ap = float(average_precision_score(target.ravel(), score.ravel()))
        ap_tolerant = float(
            average_precision_score(target_tolerant.ravel(), score.ravel())
        )
    else:
        ap = float("nan")
        ap_tolerant = float("nan")

    response_sum = float(score.sum())
    response_mean = float(score.mean())
    active_fraction = float(np.mean(score > 1e-8))
    mass_near_target = (
        float(score[target_tolerant].sum()) / response_sum if response_sum > 0 else 0.0
    )

    top_mask, top_effective = top_positive_mask(score, target_count)
    top_precision = (
        float(np.count_nonzero(top_mask & target_tolerant)) / top_effective
        if top_effective
        else 0.0
    )
    top_tolerant = cv2.dilate(top_mask.astype(np.uint8), kernel).astype(bool)
    target_coverage = (
        float(np.count_nonzero(target & top_tolerant)) / target_count
        if target_count
        else float("nan")
    )

    if target_count:
        distance = ndimage.distance_transform_edt(~target).astype(np.float32)
        weighted_distance_mean = (
            float(np.sum(distance * score) / response_sum)
            if response_sum > 0
            else float("nan")
        )
        weighted_distance_p90 = weighted_quantile(
            distance.ravel(), score.ravel(), 0.90
        )
    else:
        weighted_distance_mean = float("nan")
        weighted_distance_p90 = float("nan")

    component_count = 0
    component_largest = 0
    if top_effective:
        count, labels = cv2.connectedComponents(top_mask.astype(np.uint8), connectivity=8)
        for label in range(1, count):
            component = labels == label
            if not np.any(component & target_tolerant):
                component_count += 1
                component_largest = max(component_largest, int(component.sum()))

    # A soft-target rank correlation is useful because the binary proxy is
    # necessarily approximate after Matplotlib rendering and JPEG compression.
    if float(np.ptp(score)) <= 1e-12 or float(np.ptp(target_soft)) <= 1e-12:
        rank_correlation = float("nan")
    else:
        rank_correlation = stats.spearmanr(
            score.ravel(), target_soft.ravel()
        ).statistic
    return {
        "target_proxy_pixels": target_count,
        "average_precision": ap,
        "average_precision_tolerant_r2": ap_tolerant,
        "mass_near_target_r2": mass_near_target,
        "top_target_count_effective": top_effective,
        "top_target_count_precision_r2": top_precision,
        "target_coverage_r2": target_coverage,
        "weighted_distance_mean_px": weighted_distance_mean,
        "weighted_distance_p90_px": weighted_distance_p90,
        "false_component_count_top_target": component_count,
        "false_component_largest_top_target_px": component_largest,
        "response_mean": response_mean,
        "response_sum": response_sum,
        "active_fraction": active_fraction,
        "response_p99_minus_p50": float(
            np.percentile(score, 99.0) - np.percentile(score, 50.0)
        ),
        "spearman_with_soft_target": float(rank_correlation),
    }


def add_target_contour(axis: plt.Axes, target: np.ndarray) -> None:
    if np.any(target) and not np.all(target):
        axis.contour(target.astype(np.uint8), levels=(0.5,), colors=("cyan",), linewidths=0.45)


def metric_title(name: str, metrics: Mapping[str, float | int]) -> str:
    ap = float(metrics["average_precision_tolerant_r2"])
    mass = float(metrics["mass_near_target_r2"])
    return f"{name}\nAP2={ap:.3f}  masse2={mass:.3f}"


def save_case_figure(
    case: Case,
    rgb: np.ndarray,
    target: np.ndarray,
    channels: Mapping[str, np.ndarray],
    archived: np.ndarray,
    maps: Mapping[str, np.ndarray],
    metrics_by_variant: Mapping[str, Mapping[str, float | int]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 8, figsize=(20, 6), constrained_layout=True)
    top = (
        ("RGB extrait", rgb, None),
        ("GT proxy", target, "gray"),
        ("Y", channels["luminance"], "gray"),
        ("Lab C*", channels["lab_cstar"], "gray"),
        ("Lab a*", channels["lab_astar"], "coolwarm"),
        ("Lab b*", channels["lab_bstar"], "coolwarm"),
        ("Saturation HSV", channels["hsv_saturation"], "gray"),
        ("Log-chroma PC1", channels["log_chroma_pc1"], "coolwarm"),
    )
    for axis, (title, image, cmap) in zip(axes[0], top):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        axis.set_title(title, fontsize=8)
        axis.axis("off")

    archived_metrics = map_metrics(archived, target, target_soft=target.astype(np.float32))
    bottom = (
        ("Prompt archivé*", archived, archived_metrics),
        (VARIANT_LABELS["luminance"], maps["luminance"], metrics_by_variant["luminance"]),
        (VARIANT_LABELS["lab_cstar_dark"], maps["lab_cstar_dark"], metrics_by_variant["lab_cstar_dark"]),
        (VARIANT_LABELS["lab_cstar_bipolar"], maps["lab_cstar_bipolar"], metrics_by_variant["lab_cstar_bipolar"]),
        (VARIANT_LABELS["lab_astar_bipolar"], maps["lab_astar_bipolar"], metrics_by_variant["lab_astar_bipolar"]),
        (VARIANT_LABELS["lab_bstar_bipolar"], maps["lab_bstar_bipolar"], metrics_by_variant["lab_bstar_bipolar"]),
        (VARIANT_LABELS["lab_ab_opponent"], maps["lab_ab_opponent"], metrics_by_variant["lab_ab_opponent"]),
        (VARIANT_LABELS["log_chroma_pc1_bipolar"], maps["log_chroma_pc1_bipolar"], metrics_by_variant["log_chroma_pc1_bipolar"]),
    )
    for axis, (title, image, metrics) in zip(axes[1], bottom):
        axis.imshow(image, cmap="magma", vmin=0, vmax=1)
        add_target_contour(axis, target)
        axis.set_title(metric_title(title, metrics), fontsize=7)
        axis.axis("off")
    figure.suptitle(
        f"{case.dataset} / {case.case_name} — {case.category} — "
        f"Delta IoU Frangi-baseline={case.delta_iou:+.4f}\n"
        "Contours cyan: GT proxy; *prompt archivé non strictement comparable",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=130)
    plt.close(figure)


def save_overview(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    columns = (
        "rgb",
        "target",
        "archived",
        "luminance",
        "lab_cstar_dark",
        "lab_cstar_bipolar",
        "lab_ab_opponent",
        "log_chroma_pc1_bipolar",
    )
    labels = (
        "RGB",
        "GT proxy",
        "Prompt archivé*",
        "Y",
        "Lab C*",
        "Lab C* +/-",
        "Lab a/b +/-",
        "Log-chroma +/-",
    )
    figure, axes = plt.subplots(
        len(results), len(columns), figsize=(18, 2.35 * len(results)), constrained_layout=True
    )
    for row_index, result in enumerate(results):
        case = result["case"]
        assert isinstance(case, Case)
        maps = result["maps"]
        metrics = result["metrics"]
        target = result["target"]
        for column_index, (column, label) in enumerate(zip(columns, labels)):
            axis = axes[row_index, column_index]
            image = result[column] if column in {"rgb", "target", "archived"} else maps[column]
            if column == "rgb":
                axis.imshow(image)
            elif column == "target":
                axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            else:
                axis.imshow(image, cmap="magma", vmin=0, vmax=1)
                add_target_contour(axis, target)
            if row_index == 0:
                axis.set_title(label, fontsize=8)
            if column_index == 0:
                axis.set_ylabel(
                    f"{case.dataset}\n{case.case_name[:28]}\nD={case.delta_iou:+.3f}",
                    fontsize=7,
                )
            if column in metrics:
                metric = metrics[column]
                axis.text(
                    0.02,
                    0.02,
                    f"AP2 {float(metric['average_precision_tolerant_r2']):.2f}\n"
                    f"M2 {float(metric['mass_near_target_r2']):.2f}",
                    transform=axis.transAxes,
                    fontsize=5.5,
                    color="white",
                    va="bottom",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.0},
                )
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Sonde CPU Frangi: luminance vs chrominance — 6 reculs puis 6 gains\n"
        "Contours cyan: GT proxy; *archivé depuis la source 448², seulement contextuel",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def aggregate_metrics(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["group"]), str(row["variant"]))].append(row)
    summary: list[dict[str, object]] = []
    for group in ("failures", "gains", "all"):
        for variant in VARIANT_ORDER:
            rows = grouped[(group, variant)]
            if not rows:
                continue
            summary.append(
                {
                    "group": group,
                    "variant": variant,
                    "cases": len(rows),
                    "mean_average_precision_tolerant_r2": finite_mean(
                        float(row["average_precision_tolerant_r2"]) for row in rows
                    ),
                    "mean_mass_near_target_r2": finite_mean(
                        float(row["mass_near_target_r2"]) for row in rows
                    ),
                    "mean_top_target_count_precision_r2": finite_mean(
                        float(row["top_target_count_precision_r2"]) for row in rows
                    ),
                    "mean_target_coverage_r2": finite_mean(
                        float(row["target_coverage_r2"]) for row in rows
                    ),
                    "mean_weighted_distance_px": finite_mean(
                        float(row["weighted_distance_mean_px"]) for row in rows
                    ),
                    "mean_response_sum": finite_mean(
                        float(row["response_sum"]) for row in rows
                    ),
                    "mean_active_fraction": finite_mean(
                        float(row["active_fraction"]) for row in rows
                    ),
                }
            )
    return summary


def full_resolution_sensitivity_rows(
    case: Case,
    rgb: np.ndarray,
    target: np.ndarray,
    target_soft: np.ndarray,
) -> list[dict[str, object]]:
    """Repeat Y/C* on an upsampled 448² crop with the historical scales.

    Upsampling cannot restore lost chromatic detail.  This is only a check that
    the primary conclusion is not an artefact of scaling sigma and R to 216².
    """
    rgb_448 = cv2.resize(rgb, ORIGINAL_SIZE[::-1], interpolation=cv2.INTER_CUBIC)
    rgb_float = rgb_448.astype(np.float32) / 255.0
    luminance = np.tensordot(
        rgb_float, np.asarray((0.2989, 0.5870, 0.1140), dtype=np.float32), axes=1
    )
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    cstar = np.clip(
        np.hypot(lab[..., 1], lab[..., 2]) / math.hypot(128.0, 128.0),
        0.0,
        1.0,
    )
    rows: list[dict[str, object]] = []
    for variant, channel in (
        ("luminance", luminance),
        ("lab_cstar_dark", cstar),
        ("lab_cstar_light", 1.0 - cstar),
    ):
        _, similarity, _, _, _ = extract_frangi_graph_gpu(
            {"visible": channel},
            {"visible": 1.0},
            scales=(1.0, 3.0, 5.0, 9.0, 15.0),
            R=3,
            K=1,
            device="cpu",
            compute_centrality=False,
        )
        similarity_216 = cv2.resize(
            np.asarray(similarity, dtype=np.float32),
            PANEL_SIZE[::-1],
            interpolation=cv2.INTER_LINEAR,
        )
        rows.append(
            {
                "dataset": case.dataset,
                "category": case.category,
                "case_name": case.case_name,
                "delta_iou_frangi_minus_baseline": case.delta_iou,
                "variant": variant,
                **map_metrics(similarity_216, target, target_soft=target_soft),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(args.threads)

    report_root = args.report_root.resolve()
    output = args.output.resolve()
    figures = output / "figures"
    maps_directory = output / "maps"
    figures.mkdir(parents=True, exist_ok=True)
    maps_directory.mkdir(parents=True, exist_ok=True)

    cases = load_cases(report_root)
    metric_rows: list[dict[str, object]] = []
    dynamics_rows: list[dict[str, object]] = []
    overview_results: list[dict[str, object]] = []
    resolution_sensitivity: list[dict[str, object]] = []
    provenance: list[dict[str, str]] = []
    started = time.time()

    for case_index, case in enumerate(cases, start=1):
        case_started = time.time()
        print(
            f"[{case_index:02d}/{len(cases):02d}] {case.dataset}/{case.case_name}",
            flush=True,
        )
        rgb, target, target_soft = extract_panel_tiles(case)
        archived = load_archived_prompt(case.prompt_path)
        channels, dynamics = chrominance_channels(rgb)
        maps = calculate_maps(channels)
        metrics_by_variant: dict[str, dict[str, float | int]] = {}
        group = "failures" if case.category == "gain_baseline" else "gains"
        for variant in VARIANT_ORDER:
            metrics = map_metrics(maps[variant], target, target_soft=target_soft)
            metrics_by_variant[variant] = metrics
            base = {
                "dataset": case.dataset,
                "category": case.category,
                "group": group,
                "case_name": case.case_name,
                "delta_iou_frangi_minus_baseline": case.delta_iou,
                "variant": variant,
            }
            metric_rows.append({**base, **metrics})
            metric_rows.append({**base, "group": "all", **metrics})
        dynamics_rows.append(
            {
                "dataset": case.dataset,
                "category": case.category,
                "case_name": case.case_name,
                "delta_iou_frangi_minus_baseline": case.delta_iou,
                **dynamics,
            }
        )
        np.savez_compressed(
            maps_directory / f"{case.slug}.npz",
            rgb=rgb,
            target_proxy=target,
            target_soft=target_soft,
            archived_prompt=archived,
            **{f"channel__{name}": value for name, value in channels.items()},
            **{f"similarity__{name}": value for name, value in maps.items()},
        )
        save_case_figure(
            case,
            rgb,
            target,
            channels,
            archived,
            maps,
            metrics_by_variant,
            figures / "cases" / f"{case.slug}.png",
        )
        overview_results.append(
            {
                "case": case,
                "rgb": rgb,
                "target": target,
                "target_soft": target_soft,
                "archived": archived,
                "maps": maps,
                "metrics": metrics_by_variant,
            }
        )
        if (case.dataset, case.category) in FULL_RESOLUTION_SENSITIVITY_KEYS:
            resolution_sensitivity.extend(
                full_resolution_sensitivity_rows(case, rgb, target, target_soft)
            )
        provenance.append(
            {
                "dataset": case.dataset,
                "category": case.category,
                "case_name": case.case_name,
                "panel": str(case.panel_path),
                "panel_sha256": sha256(case.panel_path),
                "archived_prompt": str(case.prompt_path),
                "archived_prompt_sha256": sha256(case.prompt_path),
            }
        )
        print(f"    completed in {time.time() - case_started:.2f}s", flush=True)

    save_overview(overview_results, figures / "overview.png")
    write_csv(output / "metrics_per_case.csv", metric_rows)
    write_csv(output / "channel_dynamics.csv", dynamics_rows)
    summary_rows = aggregate_metrics(metric_rows)
    write_csv(output / "metrics_summary.csv", summary_rows)
    write_csv(output / "resolution_sensitivity_448.csv", resolution_sensitivity)

    manifest = {
        "format": 1,
        "purpose": "exploratory luminance versus chrominance Frangi audit",
        "execution_device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vm_or_cloud_operations_in_script": False,
        "source_limitation": (
            "RGB and GT are reconstructed from 216x216 JPEG panel tiles; original "
            "448x448 data are absent locally. Chroma is JPEG 4:2:0 subsampled."
        ),
        "panel_rgb_box": PANEL_RGB_BOX,
        "panel_target_box": PANEL_TARGET_BOX,
        "panel_size": PANEL_SIZE,
        "original_size": ORIGINAL_SIZE,
        "frangi_scales_at_panel_resolution": SCALED_FRANGI_SCALES,
        "graph_radius_at_panel_resolution": SCALED_GRAPH_RADIUS,
        "full_resolution_sensitivity": {
            "note": "216² JPEG crop bicubic-upsampled to 448²; no detail restored",
            "cases": sorted(FULL_RESOLUTION_SENSITIVITY_KEYS),
            "scales": (1.0, 3.0, 5.0, 9.0, 15.0),
            "graph_radius": 3,
            "output": "resolution_sensitivity_448.csv",
        },
        "frangi_parameters": {
            "ss": 1.0,
            "si": 0.25,
            "sa": 0.3,
            "tau": 0.18,
            "K": 1,
            "compute_centrality": False,
        },
        "threads": args.threads,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "elapsed_seconds": time.time() - started,
        "cases": provenance,
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, ensure_ascii=False)
        destination.write("\n")
    print(f"Wrote CPU probe to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
