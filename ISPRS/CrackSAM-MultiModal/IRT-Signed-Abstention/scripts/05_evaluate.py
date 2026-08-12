#!/usr/bin/env python3
"""Évalue un bras sur un split et écrit les métriques par image.

Le bras A0 n'a pas de checkpoint : il s'évalue sur les logits cachés eux-mêmes.
Tous les autres chargent ``best.pt`` — jamais ``latest.pt``, qui n'est là que
pour la reprise après préemption.

Le seuil de décision est gelé à ``0,5`` pour tous les bras. La tolérance primaire
est ``3 px``, et l'IoU stricte reste rapportée à côté.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_residual.cache import (  # noqa: E402
    open_cache,
    validate_baseline_cache,
    validate_thermal_cache,
)
from thermal_residual.config import load_arm_config  # noqa: E402
from thermal_residual.constants import PRIMARY_TOLERANCE, SPLITS  # noqa: E402
from thermal_residual.evaluation import (  # noqa: E402
    evaluate_arm,
    write_per_image_csv,
    write_summary,
)
from thermal_residual.manifest import read_manifest  # noqa: E402
from thermal_residual.model import build_adapter  # noqa: E402
from thermal_residual.splits import read_split  # noqa: E402
from thermal_residual.training import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--baseline-cache", required=True, type=Path)
    parser.add_argument("--thermal-cache", required=True, type=Path)
    parser.add_argument("--run", type=Path, default=None, help="dossier du run contenant best.pt")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-extractor-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_arm_config(args.config)
    samples = read_manifest(args.manifest)
    split = read_split(args.split_file)
    selected = [s for s in samples if split.assignment.get(s.sample_id) == args.split]
    if not selected:
        raise SystemExit(f"le split « {args.split} » est vide")

    baseline = open_cache(args.baseline_cache)
    thermal = open_cache(args.thermal_cache)
    validate_baseline_cache(baseline, selected)
    validate_thermal_cache(
        thermal,
        selected,
        extractor_config={k: v for k, v in config.thermal.items()},
        check_extractor_digest=not args.skip_extractor_check,
    )

    model = None
    checkpoint_info: dict[str, object] = {}
    if config.arm.trained:
        if args.run is None:
            raise SystemExit(f"le bras {config.identifier} est entraîné : --run est obligatoire")
        payload = load_checkpoint(Path(args.run) / "best.pt")
        model = build_adapter(config.arm.model)
        model.load_state_dict(payload["model_state_dict"])
        checkpoint_info = {
            "best_epoch": payload.get("best_epoch"),
            "best_validation_metrics": payload.get("best_validation_metrics"),
            "seed": payload.get("seed"),
            "config_digest": payload.get("config_digest"),
            "baseline_checkpoint_sha256": payload.get("baseline_checkpoint_sha256"),
        }

    result = evaluate_arm(
        samples=selected,
        baseline_cache=baseline,
        thermal_cache=thermal,
        model=model,
        evidence_source=config.arm.evidence_source,
        permuted=config.arm.permuted,
        assignment=split.assignment,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_per_image_csv(args.output / "per_image.csv", result["rows"])
    write_summary(
        args.output / "summary.json",
        {
            "identifier": config.identifier,
            "arm": config.arm.name,
            "split": args.split,
            "seed": args.seed,
            "count": result["count"],
            "summary": result["summary"],
            "checkpoint": checkpoint_info,
            "permutation": result["permutation"] if config.arm.permuted else {},
            "config": config.to_json(),
        },
    )

    summary = result["summary"]
    primary = f"iou_buffered_tol{PRIMARY_TOLERANCE}"
    print(f"{config.identifier} — {config.arm.name} sur {args.split} ({result['count']} images)")
    print(f"  IoU stricte        : {summary.get('iou', 0.0):.4f}")
    print(f"  IoU tolérante 3 px : {summary.get(primary, 0.0):.4f}")
    print(f"  Dice               : {summary.get('dice', 0.0):.4f}")
    print(f"  baseline (rappel)  : {summary.get('baseline_iou', 0.0):.4f} / "
          f"{summary.get('baseline_' + primary, 0.0):.4f}")
    if model is not None:
        print(
            "  actions            : renforcer {:.3f} / supprimer {:.3f} / abstention {:.3f}".format(
                summary.get("fraction_reinforce", 0.0),
                summary.get("fraction_suppress", 0.0),
                summary.get("fraction_abstain", 0.0),
            )
        )
        print(f"  |Δz| moyen         : {summary.get('residual_abs_mean', 0.0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
