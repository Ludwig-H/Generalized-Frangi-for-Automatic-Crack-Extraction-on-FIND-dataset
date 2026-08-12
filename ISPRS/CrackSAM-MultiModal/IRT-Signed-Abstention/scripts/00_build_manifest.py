#!/usr/bin/env python3
"""Construit le manifeste IRT-Crack et fige le split train / validation / test.

Le manifeste est la seule source d'appariement de toute l'étude : aucun autre
script ne regarde l'ordre des fichiers. Le split est écrit à côté, déterministe,
stratifié, et partagé par tous les bras.

Exemple ::

    python scripts/00_build_manifest.py \\
      --dataset-root /data/IRT-Crack \\
      --official-split /data/IRT-Crack/00_List \\
      --output data/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_residual.manifest import (  # noqa: E402
    ManifestError,
    build_manifest,
    discover_layout,
    manifest_digest,
    write_manifest,
)
from thermal_residual.provenance import atomic_write_json  # noqa: E402
from thermal_residual.splits import (  # noqa: E402
    DEFAULT_TEST_SIZE,
    DEFAULT_VALIDATION_FRACTION,
    assert_disjoint,
    build_split,
    write_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path, help="racine d'IRT-Crack")
    parser.add_argument(
        "--official-split",
        type=Path,
        default=None,
        help="dossier 00_List (train_val.txt / test.txt) ou JSON {train, test}. "
        "Absent, un split dérivé de même effectif est construit et signalé comme tel.",
    )
    parser.add_argument("--output", required=True, type=Path, help="chemin du manifeste CSV")
    parser.add_argument(
        "--split-output",
        type=Path,
        default=None,
        help="chemin du split JSON (par défaut : à côté du manifeste, split.json)",
    )
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument(
        "--allow-empty-masks",
        action="store_true",
        help="accepte les masques sans aucun pixel fissure (refusés par défaut)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        layout = discover_layout(args.dataset_root)
        print(f"visible     : {layout.visible}")
        print(f"infrarouge  : {layout.infrared}")
        print(f"vérité      : {layout.ground_truth}")
        if layout.fusion:
            print(f"fusion      : {layout.fusion} (non utilisée : SAM ne doit voir que le RGB)")

        samples = build_manifest(
            args.dataset_root,
            official_split=args.official_split,
            allow_empty_masks=args.allow_empty_masks,
        )
    except ManifestError as error:
        print(f"ERREUR : {error}", file=sys.stderr)
        return 1

    write_manifest(args.output, samples)
    split = build_split(
        samples,
        validation_fraction=args.validation_fraction,
        test_size=args.test_size,
    )
    assert_disjoint(split)
    split_path = args.split_output or args.output.parent / "split.json"
    write_split(split_path, split)

    sizes = {tuple((sample.height, sample.width)) for sample in samples}
    summary = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "manifest": str(args.output),
        "manifest_sha256": manifest_digest(samples),
        "split": str(split_path),
        "count": len(samples),
        "resolutions": sorted(f"{h}x{w}" for h, w in sizes),
        "split_origin": split.origin,
        "counts": split.counts(),
        "official_split_provided": args.official_split is not None,
    }
    atomic_write_json(args.output.parent / "manifest_summary.json", summary)

    print(f"\n{len(samples)} échantillons appariés, résolutions {summary['resolutions']}")
    print(f"split « {split.origin} » : {split.counts()}")
    if split.origin == "derived":
        print(
            "ATTENTION : le split officiel 358/90 n'a pas été fourni. Le split utilisé\n"
            "           est dérivé, déterministe et stratifié, mais ce n'est PAS celui\n"
            "           des chiffres publiés — le rapport doit le dire."
        )
    print(f"manifeste  : {args.output}")
    print(f"split      : {split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
