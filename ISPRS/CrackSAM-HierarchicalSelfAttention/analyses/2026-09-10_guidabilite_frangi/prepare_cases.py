"""Classify the archived Frangi-similarity comparison without recomputing scores.

Run this file from any directory. It only reads the historical experiment and
writes this analysis's tables and illustrative copies of its existing panels.
The image-wise oracle below uses test labels and is not a deployable selector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
import shutil
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[4]
HISTORICAL = REPOSITORY / "ISPRS/CrackSAM/results/frangi_milestone_report"
SOURCE = HISTORICAL / "tables/per_image_all_milestones.csv"
MILESTONE = "epoch25_best"
TOLERANCE = 0.01
SENSITIVITY = (0.005, 0.01, 0.02)
CATEGORIES = ("improved", "neutral", "degraded")
DATASETS = (
    "khanhha_original", "khanhha_noisy1", "khanhha_noisy2",
    "road420", "facade390", "concrete3k",
)
LABELS = {"improved": "Améliore", "neutral": "Neutre", "degraded": "Détériore"}
COLORS = {"improved": "#128477", "neutral": "#87929b", "degraded": "#b55d27"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def category(delta: float, tolerance: float = TOLERANCE) -> str:
    return "improved" if delta > tolerance else "degraded" if delta < -tolerance else "neutral"


def load_protocol() -> Any:
    path = REPOSITORY / "ISPRS/CrackSAM/protocol/build_next_protocol.py"
    spec = importlib.util.spec_from_file_location("historical_source_groups", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = load_protocol()
    lists = REPOSITORY / "ISPRS/CrackSAM/protocol/cracksam_paper/lists"
    training_groups = {
        protocol.physical_source_group(name)
        for name in protocol.read_names(lists / "lists_khanhha/train.txt")
    }
    validation_groups = {
        protocol.physical_source_group(name)
        for name in protocol.read_names(lists / "lists_khanhha/val_vol.txt")
    }
    rows = []
    for source in read_csv(SOURCE):
        if source["milestone"] != MILESTONE:
            continue
        dataset, name = source["dataset"], source["case_name"]
        if dataset not in DATASETS:
            raise ValueError(f"Unexpected dataset: {dataset}")
        baseline, frangi = float(source["baseline_iou"]), float(source["frangi_iou"])
        delta = float(source["delta_iou_frangi_minus_baseline"])
        if not all(math.isfinite(value) for value in (baseline, frangi, delta)):
            raise ValueError(f"Non-finite IoU: {dataset}/{name}")
        if not (0 <= baseline <= 1 and 0 <= frangi <= 1):
            raise ValueError(f"IoU outside [0, 1]: {dataset}/{name}")
        if not math.isclose(delta, frangi - baseline, abs_tol=1e-12):
            raise ValueError(f"Inconsistent paired delta: {dataset}/{name}")
        domain = "khanhha" if dataset.startswith("khanhha_") else dataset
        physical_source = protocol.physical_source_group(name)
        rows.append({
            "image_id": f"{dataset}::{name}",
            "dataset": dataset,
            "case_name": name,
            "domain": domain,
            "source_family": protocol.source_family(name) if domain == "khanhha" else domain,
            "physical_group": f"{domain}::{physical_source}",
            "category": category(delta),
            "delta_iou": delta,
            "baseline_iou": baseline,
            "frangi_iou": frangi,
            "historical_train_group": domain == "khanhha" and physical_source in training_groups,
            "historical_validation_group": domain == "khanhha" and physical_source in validation_groups,
        })
    rows.sort(key=lambda row: (DATASETS.index(row["dataset"]), row["case_name"]))
    ids = [row["image_id"] for row in rows]
    if len(rows) != 8895 or len(set(ids)) != len(ids):
        raise ValueError(f"Expected 8895 unique paired observations; got {len(rows)} rows, {len(set(ids))} IDs")
    # Check each archived evaluation against its immutable published test list.
    for dataset in DATASETS:
        domain = "khanhha" if dataset.startswith("khanhha_") else dataset
        expected = set(protocol.read_names(lists / f"lists_{domain}/test_vol.txt"))
        observed = {row["case_name"] for row in rows if row["dataset"] == dataset}
        if expected != observed:
            raise ValueError(f"Published test-list mismatch: {dataset}")
    list_hashes = {
        path.relative_to(REPOSITORY).as_posix(): sha256(path)
        for path in sorted(lists.glob("*/*.txt"))
    }
    return rows, {"historical_lists_sha256": list_hashes}


def subsets(rows: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    values = [("all", "all", rows)]
    values.extend(("dataset", dataset, [row for row in rows if row["dataset"] == dataset]) for dataset in DATASETS)
    values.extend(("domain", domain, [row for row in rows if row["domain"] == domain]) for domain in ("khanhha", "road420", "facade390", "concrete3k"))
    return values


def summarize(scope: str, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["category"] for row in rows)
    baseline = np.array([row["baseline_iou"] for row in rows])
    frangi = np.array([row["frangi_iou"] for row in rows])
    delta = frangi - baseline
    return {
        "scope": scope,
        "name": name,
        "n_images": len(rows),
        "n_physical_groups": len({row["physical_group"] for row in rows}),
        **{key: counts[key] for key in CATEGORIES},
        "baseline_iou": baseline.mean(),
        "frangi_iou": frangi.mean(),
        "mean_delta_iou": delta.mean(),
        "median_delta_iou": np.median(delta),
        "oracle_iou": np.maximum(baseline, frangi).mean(),
        "oracle_gain_iou": np.maximum(delta, 0).mean(),
        "oracle_gain_over_best_constant_iou": np.maximum(baseline, frangi).mean() - max(baseline.mean(), frangi.mean()),
        "oracle_gain_if_delta_above_tolerance_iou": np.where(delta > TOLERANCE, delta, 0).mean(),
        "historical_train_overlap_groups": len({row["physical_group"] for row in rows if row["historical_train_group"]}),
        "historical_validation_overlap_groups": len({row["physical_group"] for row in rows if row["historical_validation_group"]}),
    }


def copy_representatives(rows: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    """Reuse two verified archived panels per category, independently of H.

    Prefer the clean Khanhha and Road420 panels, then other datasets if needed.
    For gains/losses use the archive's explicitly selected extrema; for neutral
    images prefer its median cases and exclude pairs of near-empty predictions.
    These are illustrations of the classes, not a random evaluation subset.
    """
    from PIL import Image

    indexed = {row["image_id"]: row for row in rows}
    candidates = []
    for archived in read_csv(HISTORICAL / "tables/selected_cases.csv"):
        image_id = f"{archived['dataset']}::{archived['case_name']}"
        row = indexed[image_id]
        for old_key, new_key in (("baseline_iou_csv", "baseline_iou"), ("frangi_iou_csv", "frangi_iou"), ("delta_iou", "delta_iou")):
            if not math.isclose(float(archived[old_key]), row[new_key], abs_tol=1e-12):
                raise ValueError(f"Archived panel score mismatch: {image_id}/{old_key}")
        panel = HISTORICAL / archived["panel"]
        if archived["generated"] != "True" or not panel.is_file():
            continue
        if row["category"] == "neutral" and min(row["baseline_iou"], row["frangi_iou"]) < 0.1:
            continue
        with Image.open(panel) as image:
            image.verify()
        candidates.append((row, archived, panel))
    dataset_order = ("khanhha_original", "road420", "facade390", "concrete3k", "khanhha_noisy1", "khanhha_noisy2")
    archive_role = {"improved": "gain_frangi", "neutral": "median", "degraded": "gain_baseline"}
    chosen = []
    for label in CATEGORIES:
        current = [item for item in candidates if item[0]["category"] == label]
        current.sort(key=lambda item: (
            dataset_order.index(item[0]["dataset"]),
            item[1]["category"] != archive_role[label],
            abs(item[0]["delta_iou"]) if label == "neutral" else -abs(item[0]["delta_iou"]),
            item[0]["image_id"],
        ))
        selected_domains = set()
        for row, archived, panel in current:
            if row["domain"] in selected_domains:
                continue
            selected_domains.add(row["domain"])
            destination = output / "figures/cases" / f"{label}__{row['dataset']}__{panel.name}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(panel, destination)
            digest = sha256(panel)
            if sha256(destination) != digest:
                raise IOError(f"Panel copy changed: {destination}")
            chosen.append({
                **row,
                "panel": destination.relative_to(output).as_posix(),
                "source_panel": panel.relative_to(REPOSITORY).as_posix(),
                "panel_sha256": digest,
                "archival_selection": archived["category"],
                "selection_rule": "fixed_dataset_priority_then_archived_role; no_feature_information",
            })
            if len(selected_domains) == 2:
                break
        if not selected_domains:
            raise ValueError(f"No verified archived representative for {label}")
    return chosen


def plots(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    destination = output / "figures/cases"
    destination.mkdir(parents=True, exist_ok=True)
    delta = np.array([row["delta_iou"] for row in rows])
    # A symmetric logarithmic horizontal axis retains both the narrow neutral
    # region and the rare large failures without clipping any observations.
    edges = np.unique(np.r_[-1, -np.geomspace(1e-3, 1, 61)[::-1], np.linspace(-1e-3, 1e-3, 9), np.geomspace(1e-3, 1, 61), 1])
    figure, axis = plt.subplots(figsize=(9, 4.6), layout="constrained")
    for label in CATEGORIES:
        values = [row["delta_iou"] for row in rows if row["category"] == label]
        axis.hist(values, bins=edges, color=COLORS[label], label=f"{LABELS[label]} (n = {len(values):,})", alpha=0.85)
    axis.axvspan(-TOLERANCE, TOLERANCE, color=COLORS["neutral"], alpha=0.10)
    axis.axvline(-TOLERANCE, color="#444444", linestyle=":", linewidth=1)
    axis.axvline(TOLERANCE, color="#444444", linestyle=":", linewidth=1)
    axis.set_xscale("symlog", linthresh=0.01)
    axis.set_xlim(-1, 1)
    axis.set_xticks([-.5, -.1, -.01, 0, .01, .1, .5], ["−0,5", "−0,1", "−0,01", "0", "+0,01", "+0,1", "+0,5"])
    axis.set_xlabel("ΔIoU = Frangi-similarité − baseline (axe symétrique logarithmique)")
    axis.set_ylabel("Nombre d’observations par intervalle")
    axis.set_title("Deux modèles historiques : effet image par image")
    axis.legend(frameon=False, fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(destination / "delta_iou_histogram.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.6), layout="constrained")
    bottoms = np.zeros(len(DATASETS))
    for label in CATEGORIES:
        counts = np.array([sum(row["category"] == label and row["dataset"] == dataset for row in rows) for dataset in DATASETS])
        totals = np.array([sum(row["dataset"] == dataset for row in rows) for dataset in DATASETS])
        proportions = counts / totals * 100
        axis.barh(range(len(DATASETS)), proportions, left=bottoms, color=COLORS[label], label=LABELS[label])
        for index, (count, width) in enumerate(zip(counts, proportions)):
            if width > 7:
                axis.text(bottoms[index] + width / 2, index, f"{count}\n{width:.0f} %", ha="center", va="center", color="white", fontsize=9)
        bottoms += proportions
    axis.set_yticks(range(len(DATASETS)), ["Khanhha propre (1 695)", "Khanhha bruit 1 (1 695)", "Khanhha bruit 2 (1 695)", "Road420 (420)", "Façade390 (390)", "Concrete3k (3 000)"])
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Part des observations (%) — neutralité : |ΔIoU| ≤ 0,01")
    axis.set_title("Les catégories varient fortement selon le jeu de données")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(destination / "category_counts.png", dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    original_hash = sha256(SOURCE)
    rows, provenance = prepare_rows()
    write_csv(output / "tables/categories.csv", rows)
    summary = [summarize(scope, name, selected) for scope, name, selected in subsets(rows)]
    write_csv(output / "tables/summary.csv", summary)
    sensitivity = []
    for tolerance in SENSITIVITY:
        for scope, name, selected in subsets(rows):
            counts = Counter(category(row["delta_iou"], tolerance) for row in selected)
            sensitivity.append({"tolerance_iou": tolerance, "scope": scope, "name": name, "n_images": len(selected), **{key: counts[key] for key in CATEGORIES}})
    write_csv(output / "tables/sensitivity.csv", sensitivity)
    representatives = copy_representatives(rows, output)
    write_csv(output / "tables/representatives.csv", representatives)
    plots(rows, output)
    if sha256(SOURCE) != original_hash:
        raise IOError("Historical source CSV changed during preparation")
    manifest = {
        "source": SOURCE.relative_to(REPOSITORY).as_posix(),
        "source_sha256": original_hash,
        "selected_milestone": MILESTONE,
        "checkpoint_selection": "best validation Dice, independently for baseline (epoch 20) and Frangi (epoch 25)",
        "comparison": "joint difference between two separately trained LoRA checkpoints, with/without a Frangi-similarity dense mask prompt",
        "tolerance_iou": TOLERANCE,
        "category_rule": "improved if delta > tolerance; degraded if delta < -tolerance; neutral otherwise",
        "threshold_status": "practical analysis tolerance, not a significance threshold",
        "sensitivity_tolerances": list(SENSITIVITY),
        "observations": len(rows),
        "physical_groups": len({row["physical_group"] for row in rows}),
        "historical_source_group_parser": "ISPRS/CrackSAM/protocol/build_next_protocol.py:physical_source_group",
        "oracle_definition": "mean(max(baseline_iou, frangi_iou)); upper bound for selecting these two fixed outputs using ground truth, not deployable performance",
        "oracle_tolerance_definition": "baseline plus observed delta only when delta > 0.01",
        "summary_weighting": "all means weight observations equally; each domain row pools its observations, including the three Khanhha conditions",
        "representative_selection": "two domains per category; prefer clean Khanhha then Road420; archived maximum gains/losses or neutral median; no H or UMAP information used",
        "representatives": len(representatives),
        "representative_copies_sha256_verified": True,
        "source_unchanged": True,
        **provenance,
    }
    (output / "tables/preparation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary[0], "representatives": len(representatives), "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
