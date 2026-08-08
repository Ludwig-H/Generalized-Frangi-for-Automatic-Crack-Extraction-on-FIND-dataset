#!/usr/bin/env python3
"""Produit les agrégats interprétables de l'étude anti-ombre.

Le runner principal conserve les mesures par observation. Ce script ajoute des
résumés par cohorte physique, des deltas appariés au contrôle historique, les
compromis du stress-test d'ombre et une synthèse des phantoms. Il ne relit pas
les images et ne modifie jamais les mesures sources.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASELINE = "frangi_similarity_historical"
SEED = 20_260_808
METRICS = (
    "ap_tol2",
    "precision_top_p01",
    "skeleton_recall_top_p01",
    "response_mass_near_gt",
    "active_fraction_010",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Aucune ligne à écrire dans {path}")
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


def _interval(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _cohorts(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    return [
        ("pooled_original", frame["condition"].eq("original")),
        (
            "khanhha_original",
            frame["dataset"].eq("khanhha") & frame["condition"].eq("original"),
        ),
        (
            "khanhha_noisy1",
            frame["dataset"].eq("khanhha") & frame["condition"].eq("noisy1"),
        ),
        (
            "khanhha_noisy2",
            frame["dataset"].eq("khanhha") & frame["condition"].eq("noisy2"),
        ),
        (
            "khanhha_anchors_all_conditions",
            frame["dataset"].eq("khanhha") & frame["role"].eq("anchor"),
        ),
        (
            "khanhha_hash_all_conditions",
            frame["dataset"].eq("khanhha") & frame["role"].eq("hash_sample"),
        ),
        ("external_original", ~frame["dataset"].eq("khanhha")),
    ]


def _physical_values(group: pd.DataFrame, metric: str) -> np.ndarray:
    return (
        group.groupby(["dataset", "case_name"], sort=True)[metric]
        .mean()
        .to_numpy(dtype=np.float64)
    )


def _cohort_tables(
    frame: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    key = ["dataset", "case_name", "condition"]
    baseline = frame[frame["map"] == BASELINE][key + list(METRICS)].rename(
        columns={metric: f"{metric}_baseline" for metric in METRICS}
    )
    paired = frame.merge(baseline, on=key, how="inner", validate="many_to_one")
    paired_masks = dict(_cohorts(paired))
    for cohort_name, mask in _cohorts(frame):
        cohort = frame[mask]
        paired_cohort = paired[paired_masks[cohort_name]]
        for map_name, group in cohort.groupby("map", sort=True):
            row: dict[str, object] = {
                "cohort": cohort_name,
                "map": map_name,
                "physical_scenes": int(group[["dataset", "case_name"]].drop_duplicates().shape[0]),
                "observations": int(group[key].drop_duplicates().shape[0]),
            }
            for metric in METRICS:
                values = _physical_values(group, metric)
                finite = values[np.isfinite(values)]
                low, high = _interval(values, repetitions, rng)
                row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else math.nan
                row[f"{metric}_median"] = float(np.median(finite)) if finite.size else math.nan
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
            summary_rows.append(row)

        for map_name, group in paired_cohort.groupby("map", sort=True):
            row = {
                "cohort": cohort_name,
                "map": map_name,
                "baseline": BASELINE,
                "physical_scenes": int(group[["dataset", "case_name"]].drop_duplicates().shape[0]),
            }
            for metric in METRICS:
                difference = group[metric] - group[f"{metric}_baseline"]
                grouped = (
                    group.assign(_difference=difference)
                    .groupby(["dataset", "case_name"], sort=True)["_difference"]
                    .mean()
                    .to_numpy(dtype=np.float64)
                )
                finite = grouped[np.isfinite(grouped)]
                low, high = _interval(grouped, repetitions, rng)
                row[f"delta_{metric}_mean"] = (
                    float(np.mean(finite)) if finite.size else math.nan
                )
                row[f"delta_{metric}_median"] = (
                    float(np.median(finite)) if finite.size else math.nan
                )
                row[f"delta_{metric}_ci95_low"] = low
                row[f"delta_{metric}_ci95_high"] = high
                row[f"delta_{metric}_share_positive"] = (
                    float(np.mean(finite > 0.0)) if finite.size else math.nan
                )
            delta_rows.append(row)
    return summary_rows, delta_rows


def _shadow_table(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for shadow_kind, kind_group in frame.groupby("shadow_kind", sort=True):
        baseline_boundary = float(
            kind_group[kind_group["map"] == BASELINE]["shadow_boundary_response"].mean()
        )
        for map_name, group in kind_group.groupby("map", sort=True):
            boundary = group["shadow_boundary_response"].to_numpy(dtype=float)
            retention = group["crack_retention"].to_numpy(dtype=float)
            rows.append(
                {
                    "shadow_kind": shadow_kind,
                    "map": map_name,
                    "n": int(len(group)),
                    "crack_retention_mean": float(np.nanmean(retention)),
                    "crack_retention_median": float(np.nanmedian(retention)),
                    "shadow_boundary_response_mean": float(np.nanmean(boundary)),
                    "shadow_boundary_response_median": float(np.nanmedian(boundary)),
                    "boundary_response_increase_mean": float(
                        np.nanmean(group["boundary_response_increase"])
                    ),
                    "boundary_reduction_vs_historical": (
                        1.0 - float(np.nanmean(boundary)) / baseline_boundary
                        if baseline_boundary > 0.0
                        else math.nan
                    ),
                }
            )
    return rows


def _empty_table(frame: pd.DataFrame) -> list[dict[str, object]]:
    empty = frame[frame["target_fraction"] == 0.0]
    rows: list[dict[str, object]] = []
    for map_name, group in empty.groupby("map", sort=True):
        rows.append(
            {
                "map": map_name,
                "physical_scenes": int(group["case_name"].nunique()),
                "observations": int(len(group)),
                "score_mean": float(group["score_mean"].mean()),
                "score_q99_mean": float(group["score_q99"].mean()),
                "score_max_mean": float(group["score_max"].mean()),
                "active_fraction_010_mean": float(group["active_fraction_010"].mean()),
            }
        )
    return rows


def _phantom_table(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for map_name, group in frame.groupby("map", sort=True):
        paired = group[group["family"] == "thin_line_vs_shadow_step"]
        crossing = group[group["family"] == "line_plus_shadow"]
        polarity = group[group["family"] == "dark_vs_bright_line"]
        wide = group[group["family"] == "wide_dark_line"]
        ramp = group[group["family"] == "ramp"]
        uniform = group[group["family"] == "uniform"]
        ratio = paired["step_to_line_ratio_q95"].to_numpy(dtype=float)
        rows.append(
            {
                "map": map_name,
                "paired_step_line_ratio_median": float(np.nanmedian(ratio)),
                "paired_step_line_ratio_worst": float(np.nanmax(ratio)),
                "paired_share_ratio_below_1": float(np.mean(ratio < 1.0)),
                "bright_dark_ratio_median": float(
                    np.nanmedian(polarity["bright_to_dark_ratio_q95"])
                ),
                "wide_line_center_q95_median": float(np.nanmedian(wide["wide_center_q95"])),
                "shadowed_crack_retention_median": float(
                    np.nanmedian(crossing["shadowed_crack_retention_mean"])
                ),
                "shadow_boundary_clean_ratio_median": float(
                    np.nanmedian(crossing["shadow_boundary_to_clean_ratio_q95"])
                ),
                "ramp_abstention_fraction_010_median": float(
                    np.nanmedian(ramp["global_abstention_fraction_010"])
                ),
                "uniform_abstention_fraction_010_median": float(
                    np.nanmedian(uniform["global_abstention_fraction_010"])
                ),
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    tables = output / "tables/generated"
    per_case = pd.read_csv(tables / "per_case_metrics.csv")
    shadow = pd.read_csv(tables / "shadow_stress_per_case.csv")
    phantom = pd.read_csv(tables / "phantom_metrics.csv")
    rng = np.random.default_rng(args.seed)
    cohort, paired = _cohort_tables(per_case, args.bootstrap, rng)
    _atomic_csv(tables / "cohort_summary.csv", cohort)
    _atomic_csv(tables / "paired_deltas_vs_historical.csv", paired)
    _atomic_csv(tables / "shadow_tradeoffs.csv", _shadow_table(shadow))
    _atomic_csv(tables / "empty_mask_summary.csv", _empty_table(per_case))
    _atomic_csv(tables / "phantom_summary.csv", _phantom_table(phantom))
    print(
        "Agrégats écrits : cohort_summary, paired_deltas_vs_historical, "
        "shadow_tradeoffs, empty_mask_summary, phantom_summary",
        flush=True,
    )


if __name__ == "__main__":
    main()
