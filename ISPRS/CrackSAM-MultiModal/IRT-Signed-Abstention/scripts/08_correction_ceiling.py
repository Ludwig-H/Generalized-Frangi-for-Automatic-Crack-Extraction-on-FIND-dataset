#!/usr/bin/env python3
"""Porte de plafond : ce qu'une correction bornée par ``delta_max`` peut atteindre.

À lancer **après** le cache de logits et **avant** la campagne. Il ne coûte rien
— ni GPU, ni entraînement — et il peut arrêter le projet proprement.

La correction du protocole vaut ``Δz = δ_max(π⁺ − π⁻)``, donc ``|Δz| ≤ δ_max``.
Un pixel dont le logit baseline est loin du seuil est **hors d'atteinte par
construction**, quelle que soit l'évidence thermique. Ce script mesure :

* les quantiles de ``|z₀|`` — la spécification clippe à ±10 parce qu'elle sait
  que ces valeurs dépassent 10, puis borne la correction à 4 ;
* la fraction des erreurs de la baseline situées hors de la fenêtre ``±δ_max`` ;
* l'**oracle borné** : ``+δ_max`` sur la vérité, ``−δ_max`` ailleurs. C'est la
  meilleure correction bornée possible, donc le plafond de toute la méthode.

Critère de lecture, à fixer avant de regarder le résultat : si la marge
``oracle − baseline`` en IoU tolérante 3 px n'excède pas nettement le plancher de
détection au N du test, la campagne ne pourra pas conclure — elle ne saura pas
distinguer « la thermique n'aide pas » de « ``δ_max`` est trop petit ».
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_residual.cache import open_cache, validate_baseline_cache  # noqa: E402
from thermal_residual.ceiling import recommend_delta_max, sweep  # noqa: E402
from thermal_residual.constants import PRIMARY_TOLERANCE, SPLITS  # noqa: E402
from thermal_residual.data import load_mask  # noqa: E402
from thermal_residual.manifest import read_manifest  # noqa: E402
from thermal_residual.provenance import atomic_write_json  # noqa: E402
from thermal_residual.splits import read_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--baseline-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--split",
        default="validation",
        choices=SPLITS,
        help="la validation par défaut : le plafond ne doit pas être lu sur le test",
    )
    parser.add_argument(
        "--delta-max",
        type=float,
        nargs="*",
        default=[1.0, 2.0, 4.0, 6.0, 8.0, 12.0],
        help="valeurs balayées ; 4,0 est celle recommandée par la spécification",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = read_manifest(args.manifest)
    split = read_split(args.split_file)
    selected = [s for s in samples if split.assignment.get(s.sample_id) == args.split]
    if not selected:
        raise SystemExit(f"le split « {args.split} » est vide")

    cache = open_cache(args.baseline_cache)
    validate_baseline_cache(cache, selected)

    pairs = [
        (cache.entry(sample.sample_id)["baseline_logits"], load_mask(sample.mask_path))
        for sample in selected
    ]
    reports = sweep(pairs, args.delta_max)

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output / "correction_ceiling.json",
        {
            "split": args.split,
            "count": len(selected),
            "primary_tolerance": PRIMARY_TOLERANCE,
            "reports": [report.to_json() for report in reports],
            "recommended_delta_max_p99": recommend_delta_max(reports[0], coverage=0.99),
            "recommended_delta_max_p95": recommend_delta_max(reports[0], coverage=0.95),
        },
    )

    quantiles = reports[0].logit_abs_quantiles
    print(f"split « {args.split} » — {len(selected)} images")
    print(
        "quantiles de |z0| : "
        + " · ".join(f"{name} {value:.2f}" for name, value in quantiles.items())
    )
    print()
    header = (
        f"{'delta_max':>10} {'baseline':>9} {'oracle':>9} {'marge':>8} "
        f"{'err. hors portée':>17} {'FN hors portée':>15}"
    )
    print(header)
    print("-" * len(header))
    for report in reports:
        print(
            f"{report.delta_max:>10.1f} {report.baseline_iou_tolerant:>9.4f} "
            f"{report.oracle_iou_tolerant:>9.4f} {report.headroom:>+8.4f} "
            f"{report.unreachable_error_fraction:>17.3f} "
            f"{report.unreachable_false_negative_fraction:>15.3f}"
        )
    print()
    print(f"(IoU tolérante {PRIMARY_TOLERANCE} px ; « marge » = oracle − baseline)")
    print(
        f"delta_max suggéré au q99 de |z0| : {recommend_delta_max(reports[0], coverage=0.99):.2f} "
        "— à inscrire dans les sept configurations AVANT le premier entraînement, "
        "jamais après avoir vu les résultats."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
