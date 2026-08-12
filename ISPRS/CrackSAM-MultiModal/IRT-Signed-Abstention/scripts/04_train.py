#!/usr/bin/env python3
"""Entraîne un bras du correcteur à partir des deux caches.

CrackSAM n'est pas chargé. Le split de test n'est jamais ouvert : le dataset
d'entraînement est filtré sur ``train`` et la sélection de checkpoint se fait sur
``validation``, non augmentée.

Exemple ::

    python scripts/04_train.py \\
      --config configs/irt_signed_abstention_v1.yaml \\
      --manifest data/manifest.csv --split data/split.json \\
      --baseline-cache cache/baseline/manifest.json \\
      --thermal-cache cache/thermal_frangi/manifest.json \\
      --output results/frangi_signed_abstention/seed_13 --seed 13
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_residual.cache import (  # noqa: E402
    open_cache,
    validate_baseline_cache,
    validate_thermal_cache,
)
from thermal_residual.config import load_arm_config  # noqa: E402
from thermal_residual.constants import SPLIT_TEST  # noqa: E402
from thermal_residual.manifest import manifest_digest, read_manifest  # noqa: E402
from thermal_residual.provenance import sha256_file, sha256_json  # noqa: E402
from thermal_residual.splits import assert_disjoint, read_split  # noqa: E402
from thermal_residual.training import train_arm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--baseline-cache", required=True, type=Path)
    parser.add_argument("--thermal-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=0, help="surcharge (0 = valeur du YAML)")
    parser.add_argument(
        "--skip-extractor-check",
        action="store_true",
        help="accepte un cache thermique produit par une autre version du code (à éviter)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_arm_config(args.config)
    if not config.arm.trained:
        print(
            f"le bras {config.identifier} ({config.arm.name}) n'entraîne rien : "
            "il s'évalue directement sur les logits cachés (05_evaluate.py)."
        )
        return 0

    samples = read_manifest(args.manifest)
    split = read_split(args.split)
    assert_disjoint(split)
    trainable = [s for s in samples if split.assignment.get(s.sample_id) != SPLIT_TEST]
    if len(trainable) == len(samples):
        raise SystemExit("le split ne réserve aucune image de test : refus de continuer")

    baseline = open_cache(args.baseline_cache)
    thermal = open_cache(args.thermal_cache)
    validate_baseline_cache(baseline, trainable)
    validate_thermal_cache(
        thermal,
        trainable,
        extractor_config={k: v for k, v in config.thermal.items()},
        check_extractor_digest=not args.skip_extractor_check,
    )

    training = config.training
    if args.max_epochs:
        # Surcharge de mise au point uniquement : elle est tracée dans training.json.
        training.max_epochs = int(args.max_epochs)

    provenance = {
        "baseline_checkpoint_sha256": baseline.manifest["parameters"].get("checkpoint_sha256", "unknown"),
        "baseline_cache_manifest_sha256": sha256_file(Path(baseline.root) / "manifest.json"),
        "thermal_cache_manifest_sha256": sha256_file(Path(thermal.root) / "manifest.json"),
        "dataset_manifest_sha256": manifest_digest(samples),
        "split_sha256": sha256_json(split.to_json()),
        "config_sha256": config.digest(),
        "config_path": str(args.config),
    }

    print(f"bras {config.identifier} — {config.arm.name}, graine {args.seed}")
    print(f"  évidence : {config.arm.evidence_source}, permutée : {config.arm.permuted}")
    print(f"  tête     : {config.arm.model.get('head', 'signed_abstention')}")
    print(f"  sélection: {training.selection_metric}")

    summary = train_arm(
        arm=config.arm,
        samples=trainable,
        assignment=split.assignment,
        baseline_cache=baseline,
        thermal_cache=thermal,
        training=training,
        weights=config.loss,
        output_dir=args.output,
        seed=args.seed,
        device=args.device,
        provenance=provenance,
    )
    print(
        f"terminé : meilleure époque {summary['best_epoch']} "
        f"({summary['best_validation']}) sur {summary['epochs_run']} époques"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
