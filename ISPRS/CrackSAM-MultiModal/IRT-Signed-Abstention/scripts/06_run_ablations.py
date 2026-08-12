#!/usr/bin/env python3
"""Exécute la matrice d'ablations A0–A6 × graines, de façon reprenable.

Chaque couple (bras, graine) écrit un jalon. Un jalon présent fait sauter
l'étape : une préemption Spot ne coûte au pire qu'un bras. Les caches sont
ouverts une seule fois et partagés par tous les bras, ce qui garantit que tous
corrigent **exactement** les mêmes logits.

A0 ne dépend d'aucune graine : il est évalué une fois et réutilisé.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_residual.cache import (  # noqa: E402
    open_cache,
    validate_baseline_cache,
    validate_thermal_cache,
)
from thermal_residual.config import load_ablation_matrix, load_arm_config  # noqa: E402
from thermal_residual.constants import PRIMARY_TOLERANCE, SPLIT_TEST  # noqa: E402
from thermal_residual.evaluation import (  # noqa: E402
    evaluate_arm,
    write_per_image_csv,
    write_summary,
)
from thermal_residual.manifest import manifest_digest, read_manifest  # noqa: E402
from thermal_residual.model import build_adapter  # noqa: E402
from thermal_residual.provenance import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from thermal_residual.splits import assert_disjoint, read_split  # noqa: E402
from thermal_residual.training import load_checkpoint, train_arm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--baseline-cache", required=True, type=Path)
    parser.add_argument("--thermal-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--only", nargs="*", default=None, help="restreindre à certains identifiants (A0 A2 …)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-device", default=None, help="par défaut : identique à --device")
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--skip-extractor-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_ablation_matrix(args.protocol)
    seeds = args.seeds if args.seeds else protocol["seeds"]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "state").mkdir(exist_ok=True)

    samples = read_manifest(args.manifest)
    split = read_split(args.split_file)
    assert_disjoint(split)
    trainable = [s for s in samples if split.assignment.get(s.sample_id) != SPLIT_TEST]
    test_samples = [s for s in samples if split.assignment.get(s.sample_id) == SPLIT_TEST]
    if not test_samples:
        raise SystemExit("le split ne contient aucune image de test")

    baseline = open_cache(args.baseline_cache)
    thermal = open_cache(args.thermal_cache)
    validate_baseline_cache(baseline, samples)

    entries = [entry for entry in protocol["arms"] if not args.only or entry["identifier"] in args.only]
    configs = {entry["identifier"]: load_arm_config(entry["config"]) for entry in entries}
    for identifier, config in configs.items():
        validate_thermal_cache(
            thermal,
            samples,
            extractor_config={k: v for k, v in config.thermal.items()},
            check_extractor_digest=not args.skip_extractor_check,
        )

    provenance = {
        "baseline_checkpoint_sha256": baseline.manifest["parameters"].get("checkpoint_sha256", "unknown"),
        "baseline_cache_manifest_sha256": sha256_file(Path(baseline.root) / "manifest.json"),
        "thermal_cache_manifest_sha256": sha256_file(Path(thermal.root) / "manifest.json"),
        "dataset_manifest_sha256": manifest_digest(samples),
        "split_sha256": sha256_json(split.to_json()),
    }
    eval_device = args.eval_device or args.device
    completed: list[dict] = []
    started = time.time()

    for identifier, config in configs.items():
        arm_seeds = seeds if config.arm.trained else seeds[:1]
        for seed in arm_seeds:
            tag = f"{identifier}_{config.arm.name}_seed{seed}"
            run_dir = output / config.arm.name / f"seed_{seed}"
            milestone = output / "state" / f"{tag}.json"
            if milestone.is_file():
                print(f"[jalon] {tag} déjà terminé")
                continue

            print(f"\n=== {tag} ===")
            if config.arm.trained:
                training = config.training
                if args.max_epochs:
                    training.max_epochs = int(args.max_epochs)
                if not (run_dir / "best.pt").is_file():
                    train_arm(
                        arm=config.arm,
                        samples=trainable,
                        assignment=split.assignment,
                        baseline_cache=baseline,
                        thermal_cache=thermal,
                        training=training,
                        weights=config.loss,
                        output_dir=run_dir,
                        seed=seed,
                        device=args.device,
                        provenance={**provenance, "config_sha256": config.digest()},
                    )
                payload = load_checkpoint(run_dir / "best.pt")
                model = build_adapter(config.arm.model)
                model.load_state_dict(payload["model_state_dict"])
            else:
                model = None
                run_dir.mkdir(parents=True, exist_ok=True)

            result = evaluate_arm(
                samples=test_samples,
                baseline_cache=baseline,
                thermal_cache=thermal,
                model=model,
                evidence_source=config.arm.evidence_source,
                permuted=config.arm.permuted,
                assignment=split.assignment,
                seed=seed,
                device=eval_device,
            )
            write_per_image_csv(run_dir / "per_image.csv", result["rows"])
            write_summary(
                run_dir / "summary.json",
                {
                    "identifier": identifier,
                    "arm": config.arm.name,
                    "split": SPLIT_TEST,
                    "seed": seed,
                    "count": result["count"],
                    "summary": result["summary"],
                    "permutation": result["permutation"] if config.arm.permuted else {},
                    "config": config.to_json(),
                    "provenance": provenance,
                },
            )
            primary = f"iou_buffered_tol{PRIMARY_TOLERANCE}"
            record = {
                "identifier": identifier,
                "arm": config.arm.name,
                "seed": seed,
                "run_dir": str(run_dir),
                "iou": result["summary"].get("iou"),
                primary: result["summary"].get(primary),
                "elapsed_seconds": time.time() - started,
            }
            atomic_write_json(milestone, record)
            completed.append(record)
            print(
                f"  IoU {record['iou']:.4f} | IoU@3px {record[primary]:.4f}"
            )

    atomic_write_json(
        output / "ablation_index.json",
        {
            "protocol": str(args.protocol),
            "seeds": list(seeds),
            "arms": [entry["identifier"] for entry in entries],
            "completed": completed,
            "provenance": provenance,
        },
    )
    print(f"\n{len(completed)} exécution(s) nouvelles. Index : {output / 'ablation_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
