#!/usr/bin/env python3
"""Agrège la matrice d'ablations : deltas appariés, IC95, tableaux Markdown.

Les comparaisons sont celles pré-enregistrées dans ``ablation_matrix.yaml``, dans
l'ordre où elles doivent être lues. ``A2−A1`` et ``A2−A3`` sont les deux qui
décident ; ``A2−A0`` est rapporté en dernier, parce qu'un gain contre A0 seul ne
prouve rien de multimodal — il peut n'être qu'une adaptation de domaine.

Les deltas sont appariés **par image et par graine**, puis moyennés sur les
graines pour l'estimation ponctuelle ; la dispersion inter-graines est rapportée
séparément.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_residual.config import load_ablation_matrix  # noqa: E402
from thermal_residual.constants import ARM_IDENTIFIERS, PRIMARY_TOLERANCE  # noqa: E402
from thermal_residual.evaluation import read_per_image_csv  # noqa: E402
from thermal_residual.provenance import atomic_write_json  # noqa: E402
from thermal_residual.stats import compare_arms, detection_floor, paired_bootstrap, seed_variance  # noqa: E402

PRIMARY = f"iou_buffered_tol{PRIMARY_TOLERANCE}"
REPORTED_METRICS = ("iou", PRIMARY, "dice", "precision", "recall", "cldice", "components_pred")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, type=Path, help="dossier de 06_run_ablations.py")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def load_runs(results: Path, protocol: dict) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    """``{identifiant: {graine: {sample_id: {métrique: valeur}}}}``."""

    index_path = results / "ablation_index.json"
    if not index_path.is_file():
        raise SystemExit(f"index introuvable : {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    known = {entry["identifier"] for entry in protocol["arms"]}

    runs: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for record in index["completed"]:
        identifier = record["identifier"]
        if identifier not in known:
            continue
        csv_path = Path(record["run_dir"]) / "per_image.csv"
        if not csv_path.is_file():
            continue
        runs.setdefault(identifier, {})[int(record["seed"])] = read_per_image_csv(csv_path)
    return runs


def markdown_table(rows: list[list[str]], header: list[str], align: list[str] | None = None) -> str:
    align = align or [":--"] + ["--:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    protocol = load_ablation_matrix(args.protocol)
    runs = load_runs(args.results, protocol)
    if not runs:
        raise SystemExit("aucun résultat exploitable")
    output = args.output or args.results

    # ------------------------------------------------------------------ #
    # Tableau de segmentation, par bras et par graine
    # ------------------------------------------------------------------ #
    segmentation_rows: list[list[str]] = []
    per_arm_means: dict[str, dict[str, list[float]]] = {}
    for identifier in sorted(runs, key=lambda name: list(ARM_IDENTIFIERS).index(name)):
        for seed in sorted(runs[identifier]):
            table = runs[identifier][seed]
            means = {
                metric: float(np.mean([row[metric] for row in table.values() if metric in row]))
                for metric in REPORTED_METRICS
            }
            for metric, value in means.items():
                per_arm_means.setdefault(identifier, {}).setdefault(metric, []).append(value)
            segmentation_rows.append(
                [f"{identifier} {ARM_IDENTIFIERS[identifier]}", str(seed)]
                + [f"{means[metric]:.4f}" for metric in REPORTED_METRICS]
            )

    # ------------------------------------------------------------------ #
    # Deltas appariés
    # ------------------------------------------------------------------ #
    comparisons: list[dict] = []
    for name in protocol["comparisons"]:
        left_id, right_id = name.split("-")
        if left_id not in runs or right_id not in runs:
            continue
        for metric in ("iou", PRIMARY, "dice"):
            deltas: list[float] = []
            per_seed: list[float] = []
            for seed in sorted(runs[left_id]):
                right_seed = seed if seed in runs[right_id] else sorted(runs[right_id])[0]
                left_table = runs[left_id][seed]
                right_table = runs[right_id][right_seed]
                shared = sorted(set(left_table) & set(right_table))
                values = [left_table[s][metric] - right_table[s][metric] for s in shared]
                deltas.extend(values)
                per_seed.append(float(np.mean(values)))
            result = paired_bootstrap(
                deltas,
                comparison=name,
                metric=metric,
                replicates=args.bootstrap,
            )
            payload = result.to_json()
            payload["seed_variance"] = seed_variance(per_seed)
            payload["detection_floor"] = detection_floor(deltas)
            comparisons.append(payload)

    delta_rows = [
        [
            entry["comparison"],
            entry["metric"],
            f"{entry['mean']:+.4f}",
            f"[{entry['ci95_low']:+.4f} ; {entry['ci95_high']:+.4f}]",
            f"{entry['wins']}/{entry['losses']}/{entry['ties']}",
            entry["verdict"],
        ]
        for entry in comparisons
    ]

    # ------------------------------------------------------------------ #
    # Verdict pré-enregistré
    # ------------------------------------------------------------------ #
    def find(comparison: str, metric: str) -> dict | None:
        return next(
            (e for e in comparisons if e["comparison"] == comparison and e["metric"] == metric),
            None,
        )

    verdict: dict[str, object] = {}
    for metric in ("iou", PRIMARY):
        against_recalibration = find("A2-A1", metric)
        against_permuted = find("A2-A3", metric)
        against_raw = find("A2-A4", metric)
        helps = bool(
            against_recalibration
            and against_recalibration["mean"] > 0
            and against_recalibration["excludes_zero"]
            and against_permuted
            and against_permuted["mean"] > 0
        )
        verdict[metric] = {
            "la_frangi_thermique_aide": helps,
            "au_dela_de_la_modalite": bool(
                helps and against_raw and against_raw["mean"] > 0 and against_raw["excludes_zero"]
            ),
            "A2-A1": against_recalibration,
            "A2-A3": against_permuted,
            "A2-A4": against_raw,
        }

    payload = {
        "results": str(args.results),
        "arms": sorted(runs),
        "seeds": {identifier: sorted(seeds) for identifier, seeds in runs.items()},
        "primary_metric": PRIMARY,
        "segmentation": per_arm_means,
        "comparisons": comparisons,
        "verdict": verdict,
    }
    atomic_write_json(output / "report.json", payload)

    markdown = "\n\n".join(
        (
            "## Segmentation, par bras et par graine",
            markdown_table(
                segmentation_rows,
                ["Variante", "Graine", "IoU", f"IoU@{PRIMARY_TOLERANCE}px", "Dice", "Précision", "Rappel", "clDice", "Composantes"],
            ),
            f"## Deltas appariés (bootstrap {args.bootstrap:,}, IC95 par percentiles)".replace(",", " "),
            markdown_table(
                delta_rows,
                ["Comparaison", "Métrique", "Delta moyen", "IC95", "gains/pertes/nuls", "Verdict"],
                [":--", ":--", "--:", ":--:", ":--:", ":--"],
            ),
        )
    )
    (output / "report.md").write_text(markdown + "\n", encoding="utf-8")

    print(markdown)
    print("\n## Verdict pré-enregistré")
    for metric, entry in verdict.items():
        assert isinstance(entry, dict)
        print(
            f"  [{metric}] « la Frangi thermique aide » : {entry['la_frangi_thermique_aide']} — "
            f"« au-delà de la modalité » : {entry['au_dela_de_la_modalite']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
