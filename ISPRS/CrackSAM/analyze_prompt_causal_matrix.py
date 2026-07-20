#!/usr/bin/env python3
"""Analyze the paired causal prompt matrix produced by the fixed SAM 2 weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = (
    "khanhha_original",
    "khanhha_noisy1",
    "khanhha_noisy2",
    "road420",
    "facade390",
    "concrete3k",
)
FAMILY_DATASETS = {
    "khanhha": ("khanhha_original", "khanhha_noisy1", "khanhha_noisy2"),
    "road420": ("road420",),
    "facade390": ("facade390",),
    "concrete3k": ("concrete3k",),
}
BASELINE_EXPECTED_IOU = {
    "khanhha_original": 0.623825,
    "khanhha_noisy1": 0.567816,
    "khanhha_noisy2": 0.513364,
    "road420": 0.483509,
    "facade390": 0.516411,
    "concrete3k": 0.699844,
}


@dataclass(frozen=True)
class ConditionRef:
    checkpoint: str
    prompt: str


@dataclass(frozen=True)
class Comparison:
    name: str
    reference: ConditionRef
    candidate: ConditionRef
    question: str


COMPARISONS = (
    Comparison(
        "baseline_frangi_vs_none",
        ConditionRef("baseline_epoch20", "none"),
        ConditionRef("baseline_epoch20", "frangi"),
        "Effet direct du bon prompt Frangi sur les poids baseline.",
    ),
    Comparison(
        "baseline_zero_logit_vs_none",
        ConditionRef("baseline_epoch20", "none"),
        ConditionRef("baseline_epoch20", "zero_logit"),
        "Effet du passage par l'encodeur de masque avec des logits nuls.",
    ),
    Comparison(
        "baseline_permuted_vs_none",
        ConditionRef("baseline_epoch20", "none"),
        ConditionRef("baseline_epoch20", "permuted"),
        "Effet d'un prompt Frangi provenant d'une autre image.",
    ),
    Comparison(
        "baseline_shifted_vs_none",
        ConditionRef("baseline_epoch20", "none"),
        ConditionRef("baseline_epoch20", "shifted"),
        "Effet d'un prompt Frangi spatialement décalé.",
    ),
    Comparison(
        "matching_vs_permuted",
        ConditionRef("baseline_epoch20", "permuted"),
        ConditionRef("baseline_epoch20", "frangi"),
        "Valeur du contenu Frangi correspondant plutôt qu'un prior quelconque.",
    ),
    Comparison(
        "matching_vs_shifted",
        ConditionRef("baseline_epoch20", "shifted"),
        ConditionRef("baseline_epoch20", "frangi"),
        "Valeur de l'alignement spatial du prompt Frangi.",
    ),
    Comparison(
        "frangi_epoch20_prompt_vs_none",
        ConditionRef("frangi_epoch20", "none"),
        ConditionRef("frangi_epoch20", "frangi"),
        "Effet direct du prompt sur les poids Frangi à durée égale.",
    ),
    Comparison(
        "frangi_training_effect_without_prompt",
        ConditionRef("baseline_epoch20", "none"),
        ConditionRef("frangi_epoch20", "none"),
        "Effet des poids appris avec Frangi lorsque le prompt est retiré.",
    ),
    Comparison(
        "historical_joint_epoch20_vs_baseline",
        ConditionRef("baseline_epoch20", "none"),
        ConditionRef("frangi_epoch20", "frangi"),
        "Effet conjoint des poids Frangi et du prompt à l'époque 20.",
    ),
    Comparison(
        "frangi_best_prompt_vs_none",
        ConditionRef("frangi_best", "none"),
        ConditionRef("frangi_best", "frangi"),
        "Effet direct du prompt sur le meilleur checkpoint Frangi historique.",
    ),
    Comparison(
        "historical_joint_best_vs_baseline",
        ConditionRef("baseline_epoch20", "none"),
        ConditionRef("frangi_best", "frangi"),
        "Comparaison historique du meilleur Frangi à la baseline retenue.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--restoration-tolerance", type=float, default=5e-4)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty analysis table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def read_metric_rows(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    order: list[str] = []
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as source:
        for raw in csv.DictReader(source):
            case_name = raw.get("case_name", "")
            if not case_name or case_name in rows:
                raise ValueError(f"Invalid or duplicate case in {path}: {case_name!r}")
            order.append(case_name)
            rows[case_name] = {
                metric: float(raw[metric]) for metric in ("iou", "dice")
            }
    if not rows:
        raise ValueError(f"Empty result table: {path}")
    return order, rows


def paired_metric(
    root: Path,
    reference: ConditionRef,
    candidate: ConditionRef,
    dataset: str,
    metric: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    reference_order, reference_rows = read_metric_rows(
        root / reference.checkpoint / reference.prompt / dataset / "per_image.csv"
    )
    candidate_order, candidate_rows = read_metric_rows(
        root / candidate.checkpoint / candidate.prompt / dataset / "per_image.csv"
    )
    if reference_order != candidate_order:
        raise ValueError(
            f"Paired order mismatch for {dataset}: {reference} versus {candidate}"
        )
    reference_values = np.asarray(
        [reference_rows[name][metric] for name in reference_order], dtype=np.float64
    )
    candidate_values = np.asarray(
        [candidate_rows[name][metric] for name in reference_order], dtype=np.float64
    )
    if not np.isfinite(reference_values).all() or not np.isfinite(candidate_values).all():
        raise ValueError(f"Non-finite {metric} value for {dataset}")
    return reference_order, reference_values, candidate_values, candidate_values - reference_values


def _seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def bootstrap_mean_ci(
    values: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = max(1, min(512, 8_000_000 // values.size))
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def summarize_delta(
    delta: np.ndarray, *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    ci_low, ci_high = bootstrap_mean_ci(
        delta, samples=bootstrap_samples, seed=seed
    )
    tolerance = 1e-12
    return {
        "samples": int(delta.size),
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std()),
        "median_delta": float(np.median(delta)),
        "p05_delta": float(np.quantile(delta, 0.05)),
        "p95_delta": float(np.quantile(delta, 0.95)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "gains": int(np.count_nonzero(delta > tolerance)),
        "ties": int(np.count_nonzero(np.abs(delta) <= tolerance)),
        "losses": int(np.count_nonzero(delta < -tolerance)),
        "losses_below_minus_005": int(np.count_nonzero(delta < -0.05)),
        "losses_below_minus_010": int(np.count_nonzero(delta < -0.10)),
    }


def family_delta_arrays(
    deltas_by_dataset: dict[str, tuple[list[str], np.ndarray]]
) -> dict[str, np.ndarray]:
    families: dict[str, np.ndarray] = {}
    for family, datasets in FAMILY_DATASETS.items():
        if len(datasets) == 1:
            families[family] = deltas_by_dataset[datasets[0]][1]
            continue
        orders = [deltas_by_dataset[dataset][0] for dataset in datasets]
        if any(order != orders[0] for order in orders[1:]):
            raise ValueError("The three Khanhha conditions do not share the same cases")
        families[family] = np.stack(
            [deltas_by_dataset[dataset][1] for dataset in datasets], axis=0
        ).mean(axis=0)
    return families


def macro_family_bootstrap_ci(
    families: dict[str, np.ndarray], *, samples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    macro = np.zeros(samples, dtype=np.float64)
    for values in families.values():
        means = np.empty(samples, dtype=np.float64)
        chunk = max(1, min(512, 8_000_000 // values.size))
        for start in range(0, samples, chunk):
            stop = min(samples, start + chunk)
            indices = rng.integers(0, values.size, size=(stop - start, values.size))
            means[start:stop] = values[indices].mean(axis=1)
        macro += means / len(families)
    low, high = np.quantile(macro, (0.025, 0.975))
    return float(low), float(high)


def analyze(root: Path, *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    comparison_results: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for comparison in COMPARISONS:
        per_dataset: list[dict[str, Any]] = []
        iou_deltas: dict[str, tuple[list[str], np.ndarray]] = {}
        for dataset in DATASETS:
            for metric in ("iou", "dice"):
                order, reference, candidate, delta = paired_metric(
                    root,
                    comparison.reference,
                    comparison.candidate,
                    dataset,
                    metric,
                )
                stats = summarize_delta(
                    delta,
                    bootstrap_samples=bootstrap_samples,
                    seed=_seed(seed, f"{comparison.name}/{dataset}/{metric}"),
                )
                row = {
                    "comparison": comparison.name,
                    "dataset": dataset,
                    "metric": metric,
                    "reference_mean": float(reference.mean()),
                    "candidate_mean": float(candidate.mean()),
                    **stats,
                }
                table_rows.append(row)
                per_dataset.append(row)
                if metric == "iou":
                    iou_deltas[dataset] = (order, delta)

        families = family_delta_arrays(iou_deltas)
        family_means = {name: float(values.mean()) for name, values in families.items()}
        macro_delta = float(np.mean(list(family_means.values())))
        macro_ci = macro_family_bootstrap_ci(
            families,
            samples=bootstrap_samples,
            seed=_seed(seed, f"{comparison.name}/macro"),
        )
        comparison_results.append(
            {
                "name": comparison.name,
                "question": comparison.question,
                "reference": comparison.reference.__dict__,
                "candidate": comparison.candidate.__dict__,
                "family_iou_delta": family_means,
                "macro_four_families_iou_delta": macro_delta,
                "macro_four_families_iou_delta_ci95": list(macro_ci),
                "datasets": per_dataset,
            }
        )
    return {"comparisons": comparison_results, "table_rows": table_rows}


def restoration_check(root: Path, tolerance: float) -> dict[str, Any]:
    observed: dict[str, float] = {}
    failures: list[str] = []
    for dataset, expected in BASELINE_EXPECTED_IOU.items():
        _, rows = read_metric_rows(
            root / "baseline_epoch20" / "none" / dataset / "per_image.csv"
        )
        value = float(np.mean([row["iou"] for row in rows.values()]))
        observed[dataset] = value
        if abs(value - expected) > tolerance:
            failures.append(dataset)
    return {
        "passed": not failures,
        "tolerance": tolerance,
        "expected": BASELINE_EXPECTED_IOU,
        "observed": observed,
        "failed_datasets": failures,
    }


def _format_markdown(analysis: dict[str, Any], restoration: dict[str, Any]) -> str:
    by_name = {
        comparison["name"]: comparison for comparison in analysis["comparisons"]
    }
    lines = [
        "# Matrice causale du prompt Frangi",
        "",
        "## Contrôle de restauration",
        "",
        (
            "La baseline sans prompt reproduit les résultats historiques dans la "
            "tolérance fixée."
            if restoration["passed"]
            else "**ÉCHEC : la baseline ne reproduit pas les résultats historiques.**"
        ),
        "",
        "| Jeu | IoU attendu | IoU observé | Écart |",
        "|---|---:|---:|---:|",
    ]
    for dataset, expected in restoration["expected"].items():
        observed = restoration["observed"][dataset]
        lines.append(
            f"| {dataset} | {expected:.6f} | {observed:.6f} | {observed - expected:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Effets principaux",
            "",
            "Les écarts sont des différences d'IoU candidat moins référence. Le macro "
            "donne le même poids aux quatre familles Khanhha, Road420, Facade390 et Concrete3k.",
            "",
            "| Comparaison | Macro ΔIoU | IC 95 % | Khanhha | Road | Façade | Béton |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in analysis["comparisons"]:
        family = comparison["family_iou_delta"]
        low, high = comparison["macro_four_families_iou_delta_ci95"]
        lines.append(
            f"| {comparison['name']} | {comparison['macro_four_families_iou_delta']:+.4f} "
            f"| [{low:+.4f}, {high:+.4f}] | {family['khanhha']:+.4f} "
            f"| {family['road420']:+.4f} | {family['facade390']:+.4f} "
            f"| {family['concrete3k']:+.4f} |"
        )
    direct = by_name["baseline_frangi_vs_none"]
    zero = by_name["baseline_zero_logit_vs_none"]
    matching_permuted = by_name["matching_vs_permuted"]
    matching_shifted = by_name["matching_vs_shifted"]
    trained_prompt = by_name["frangi_best_prompt_vs_none"]
    weights_only = by_name["frangi_training_effect_without_prompt"]
    joint = by_name["historical_joint_best_vs_baseline"]
    direct_iou_rows = [
        row for row in direct["datasets"] if row["metric"] == "iou"
    ]
    severe_direct = sum(row["losses_below_minus_005"] for row in direct_iou_rows)
    direct_samples = sum(row["samples"] for row in direct_iou_rows)
    trained_low, trained_high = trained_prompt[
        "macro_four_families_iou_delta_ci95"
    ]
    lines.extend(
        [
            "",
            "## Conclusion causale",
            "",
            f"1. Sur les poids baseline fixes, le prompt Frangi correct retire "
            f"`{abs(direct['macro_four_families_iou_delta']):.4f}` d'IoU macro. "
            f"Un tenseur de logits nuls retire `{abs(zero['macro_four_families_iou_delta']):.4f}` : "
            "`None` et un masque numériquement nul ne sont donc pas équivalents dans SAM 2.",
            f"2. Le bon alignement reste informatif : il bat le prompt d'une autre image "
            f"de `{matching_permuted['macro_four_families_iou_delta']:+.4f}` et le prompt "
            f"décalé de `{matching_shifted['macro_four_families_iou_delta']:+.4f}`. "
            "La géométrie est bien lue, mais l'interface dense la présente comme une hypothèse de masque trop contraignante.",
            f"3. Après l'entraînement historique Frangi, remettre le prompt apporte seulement "
            f"`{trained_prompt['macro_four_families_iou_delta']:+.4f}` "
            f"(IC 95 % `[{trained_low:+.4f}, {trained_high:+.4f}]`) et dégrade encore Façade. "
            f"Les poids appris sans prompt perdent `{abs(weights_only['macro_four_families_iou_delta']):.4f}`.",
            f"4. Au total, le meilleur système historique reste à "
            f"`{joint['macro_four_families_iou_delta']:+.4f}` sous la baseline. Avec le prompt "
            f"appliqué directement à la baseline, `{severe_direct}` images sur `{direct_samples}` "
            "perdent plus de 0,05 IoU.",
            "",
            "**Décision :** abandonner `mask_input` pour Frangi. La suite doit garder la "
            "baseline gelée, traiter Frangi comme des cartes auxiliaires et n'appliquer qu'une "
            "correction résiduelle révocable par une porte de confiance simple.",
        ]
    )
    lines.extend(
        [
            "",
            "## Règle de lecture",
            "",
            "Si le contrôle de restauration échoue, les autres différences ne doivent pas "
            "être interprétées. Un prompt utile doit battre `None`, mais aussi le prompt "
            "permuté et le prompt décalé ; sinon le gain ne démontre pas l'usage de la bonne géométrie.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    if not math.isfinite(args.restoration_tolerance) or args.restoration_tolerance < 0:
        raise ValueError("restoration-tolerance must be finite and non-negative")
    root = args.root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "analysis"
    )
    restoration = restoration_check(root, args.restoration_tolerance)
    analysis = analyze(
        root, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    payload = {
        "format_version": 1,
        "root": str(root),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "restoration": restoration,
        "comparisons": analysis["comparisons"],
    }
    _write_json(output / "causal_analysis.json", payload)
    _write_csv(output / "paired_deltas.csv", analysis["table_rows"])
    markdown = _format_markdown(analysis, restoration)
    report = output / "RAPPORT_MATRICE_CAUSALE.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if restoration["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
