#!/usr/bin/env python3
"""Build the CPU-only HSV-polarity gallery used by the SafeFrangi report.

The chrominance probe stores the reconstructed RGB crop, its proxy target and
the scalar HSV saturation channel for each of twelve diagnostic cases.  This
script deliberately recomputes the two Frangi polarities separately:

* ``F(S)`` selects local saturation minima (low-saturation ridges);
* ``F(1-S)`` selects local saturation maxima (high-saturation ridges).

It never contacts a cloud service and forces the maintained Frangi
implementation to run on CPU through ``analyze_frangi_chrominance_cpu``.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_frangi_chrominance_cpu import (
    CASE_ORDER,
    add_target_contour,
    frangi_similarity,
    map_metrics,
)


VARIANTS = (
    ("luminance", "F(Y) : faible luminance"),
    ("hsv_low_saturation", "F(S) : faible saturation"),
    ("hsv_high_saturation", "F(1-S) : forte saturation"),
    ("hsv_bipolar_max", "max des deux polarites"),
)

SELECTED_DETAIL = {
    "concrete3k__gain_baseline__128_23.jpg",
    "facade390__gain_baseline__DJ_Wall_231.JPG",
    "road420__gain_baseline__2023_10_30_16_44_IMG_6033.jpg",
    "road420__gain_frangi__2023_11_01_20_33_IMG_6353.jpg",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=script_dir / "results" / "frangi_chrominance_cpu_probe",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "results" / "frangi_safe_recommendation",
    )
    return parser.parse_args()


def read_metadata(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return {
        (row["dataset"], row["category"], row["case_name"]): row
        for row in rows
        if row["group"] == "all" and row["variant"] == "luminance"
    }


def case_key_from_path(path: Path) -> tuple[str, str, str]:
    dataset, category, case_name = path.stem.split("__", maxsplit=2)
    return dataset, category, case_name


def calculate_case(
    path: Path,
    metadata: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> dict[str, object]:
    key = case_key_from_path(path)
    if key not in metadata:
        raise KeyError(f"Missing metadata for {key}")

    with np.load(path, allow_pickle=False) as archive:
        rgb = np.asarray(archive["rgb"], dtype=np.uint8)
        target = np.asarray(archive["target_proxy"], dtype=bool)
        target_soft = np.asarray(archive["target_soft"], dtype=np.float32)
        saturation = np.asarray(
            archive["channel__hsv_saturation"], dtype=np.float32
        )
        luminance = np.asarray(archive["similarity__luminance"], dtype=np.float32)
        bipolar_archived = np.asarray(
            archive["similarity__hsv_saturation_bipolar"], dtype=np.float32
        )

    low_saturation = frangi_similarity(saturation)
    high_saturation = frangi_similarity(1.0 - saturation)
    bipolar_recomputed = np.maximum(low_saturation, high_saturation)
    if not np.allclose(bipolar_recomputed, bipolar_archived, rtol=0.0, atol=1e-6):
        error = float(np.max(np.abs(bipolar_recomputed - bipolar_archived)))
        raise AssertionError(f"Archived bipolar map mismatch for {path}: {error}")

    maps = {
        "luminance": luminance,
        "hsv_low_saturation": low_saturation,
        "hsv_high_saturation": high_saturation,
        "hsv_bipolar_max": bipolar_recomputed,
    }
    metrics = {
        name: map_metrics(score, target, target_soft=target_soft)
        for name, score in maps.items()
    }
    return {
        "key": key,
        "slug": path.stem,
        "rgb": rgb,
        "target": target,
        "saturation": saturation,
        "maps": maps,
        "metrics": metrics,
        "delta_iou": float(metadata[key]["delta_iou_frangi_minus_baseline"]),
    }


def save_metrics(results: list[dict[str, object]], output: Path) -> None:
    fieldnames = (
        "dataset",
        "category",
        "case_name",
        "delta_iou_frangi_minus_baseline",
        "variant",
        "average_precision_tolerant_r2",
        "mass_near_target_r2",
        "top_target_count_precision_r2",
        "target_coverage_r2",
        "weighted_distance_mean_px",
        "active_fraction",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            dataset, category, case_name = result["key"]
            for name, _ in VARIANTS:
                metric = result["metrics"][name]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "category": category,
                        "case_name": case_name,
                        "delta_iou_frangi_minus_baseline": result["delta_iou"],
                        "variant": name,
                        **{field: metric[field] for field in fieldnames[5:]},
                    }
                )


def save_summary(results: list[dict[str, object]], output: Path) -> None:
    fieldnames = (
        "group",
        "variant",
        "cases",
        "mean_average_precision_tolerant_r2",
        "mean_mass_near_target_r2",
        "mean_active_fraction",
    )
    groups = (
        ("frangi_failures", [r for r in results if r["key"][1] == "gain_baseline"]),
        ("frangi_gains", [r for r in results if r["key"][1] == "gain_frangi"]),
        ("all", results),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for group_name, group_results in groups:
            for variant, _ in VARIANTS:
                metrics = [result["metrics"][variant] for result in group_results]
                writer.writerow(
                    {
                        "group": group_name,
                        "variant": variant,
                        "cases": len(metrics),
                        "mean_average_precision_tolerant_r2": np.mean(
                            [metric["average_precision_tolerant_r2"] for metric in metrics]
                        ),
                        "mean_mass_near_target_r2": np.mean(
                            [metric["mass_near_target_r2"] for metric in metrics]
                        ),
                        "mean_active_fraction": np.mean(
                            [metric["active_fraction"] for metric in metrics]
                        ),
                    }
                )


def metric_annotation(metric: Mapping[str, float | int]) -> str:
    return (
        f"AP2={float(metric['average_precision_tolerant_r2']):.3f}\n"
        f"masse2={float(metric['mass_near_target_r2']):.3f}"
    )


def draw_gallery(results: list[dict[str, object]], output: Path, title: str) -> None:
    columns = (
        ("rgb", "RGB", None),
        ("target", "GT proxy", "gray"),
        ("saturation", "S", "gray"),
        ("saturation_inverse", "1-S", "gray"),
        ("luminance", "F(Y)\nfaible Y", "magma"),
        ("hsv_low_saturation", "F(S)\nfaible S", "magma"),
        ("hsv_high_saturation", "F(1-S)\nforte S", "magma"),
        ("hsv_bipolar_max", "max\n(non recommande)", "magma"),
    )
    figure, axes = plt.subplots(
        len(results),
        len(columns),
        figsize=(19, 2.35 * len(results)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, result in enumerate(results):
        target = result["target"]
        maps = result["maps"]
        metrics = result["metrics"]
        for column_index, (name, label, cmap) in enumerate(columns):
            axis = axes[row_index, column_index]
            if name == "saturation_inverse":
                image = 1.0 - result["saturation"]
            elif name in maps:
                image = maps[name]
            else:
                image = result[name]
            axis.imshow(
                image,
                cmap=cmap,
                vmin=0 if cmap is not None else None,
                vmax=1 if cmap is not None else None,
            )
            if name not in {"rgb", "target"}:
                add_target_contour(axis, target)
            if row_index == 0:
                axis.set_title(label, fontsize=8)
            if name in metrics:
                axis.text(
                    0.02,
                    0.02,
                    metric_annotation(metrics[name]),
                    transform=axis.transAxes,
                    color="white",
                    fontsize=6,
                    va="bottom",
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 1.0},
                )
            if column_index == 0:
                dataset, _, case_name = result["key"]
                axis.set_ylabel(
                    f"{dataset}\n{case_name[:31]}\n"
                    f"DeltaIoU={result['delta_iou']:+.3f}",
                    fontsize=7,
                )
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        title
        + "\nLe Frangi actuel garde lambda2>0 : F(X) detecte les minima de X. "
        "Contours cyan : GT proxy JPEG 216 px.",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    metadata = read_metadata(args.probe_root / "metrics_per_case.csv")
    order = {key: index for index, key in enumerate(CASE_ORDER)}
    results = [
        calculate_case(path, metadata)
        for path in (args.probe_root / "maps").glob("*.npz")
    ]
    results.sort(key=lambda result: order[result["key"][:2]])
    if len(results) != len(CASE_ORDER):
        raise RuntimeError(f"Expected {len(CASE_ORDER)} cases, got {len(results)}")

    save_metrics(results, args.output / "hsv_polarity_per_case.csv")
    save_summary(results, args.output / "hsv_polarity_summary.csv")
    draw_gallery(
        results,
        args.output / "figures" / "hsv_polarity_atlas.png",
        "Atlas CPU des polarites Frangi sur la saturation HSV — 12 cas",
    )
    detail = [result for result in results if result["slug"] in SELECTED_DETAIL]
    draw_gallery(
        detail,
        args.output / "figures" / "hsv_polarity_selected.png",
        "Cas contrastes : la polarite HSV utile depend de l'image",
    )


if __name__ == "__main__":
    main()
