#!/usr/bin/env python3
"""Cache l'évidence Frangi double polarité calculée sur la thermique.

Deux appels à l'extracteur par image — ``Φ(Ĩ_T)`` et ``Φ(1−Ĩ_T)`` — avec
``compute_centrality=False`` et ``return_raster_features=False``. Ni MST ni
centralité : c'est une contrainte du protocole, pas une économie.

Le support n'est pas lu dans ``diagnostics["tau_mask"]`` : sur cette branche
l'extracteur y renvoie un plan de zéros. Il est reconstruit par
``thermal_residual.thermal_frangi.support_from_similarity``, dont l'équivalence
au ``tau_mask`` réel est vérifiée par test.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_residual.cache import CacheWriter, extractor_digest  # noqa: E402
from thermal_residual.constants import EVIDENCE_CHANNELS, THERMAL_CACHE_VERSION  # noqa: E402
from thermal_residual.manifest import manifest_digest, read_manifest  # noqa: E402
from thermal_residual.thermal_frangi import (  # noqa: E402
    ThermalEvidenceConfig,
    generate_dual_polarity_thermal_evidence,
)

#: Canaux réellement écrits dans chaque ``.npz``.
CACHED_CHANNELS = ("thermal_decoded", *EVIDENCE_CHANNELS, "thermal_raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None, help="YAML de bras (section « thermal »)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = read_manifest(args.manifest)
    if args.limit:
        samples = samples[: args.limit]

    payload = {}
    if args.config is not None:
        payload = (yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}).get("thermal", {})
    config = ThermalEvidenceConfig.from_mapping(payload)
    print(f"configuration Frangi : {config.to_json()}")

    writer = CacheWriter(
        args.output,
        schema_version=THERMAL_CACHE_VERSION,
        kind="thermal_evidence",
        parameters={**config.to_json(), "extractor_sha256": extractor_digest()},
        extra={
            "channels": list(EVIDENCE_CHANNELS),
            "cached_channels": list(CACHED_CHANNELS),
            "dataset_manifest": str(args.manifest),
            "dataset_manifest_sha256": manifest_digest(samples),
        },
    )

    pending = [sample for sample in samples if not writer.has(sample.sample_id)]
    print(f"{len(samples) - len(pending)} déjà en cache, {len(pending)} à calculer")
    started = time.time()
    encodings: dict[str, int] = {}

    for index, sample in enumerate(pending, start=1):
        try:
            evidence = generate_dual_polarity_thermal_evidence(
                sample.thermal_path, device=args.device, config=config
            )
        except Exception as error:  # noqa: BLE001
            writer.record_error(sample.sample_id, str(error))
            print(f"\nERREUR sur {sample.sample_id} : {error}", file=sys.stderr)
            continue
        decoding = evidence.pop("decoding")
        encodings[decoding.encoding] = encodings.get(decoding.encoding, 0) + 1
        writer.write(
            sample.sample_id,
            {name: np.asarray(evidence[name], dtype=np.float32) for name in CACHED_CHANNELS},
            {
                "source_thermal_sha256": sample.thermal_sha256,
                "height": sample.height,
                "width": sample.width,
                "decoding": decoding.to_json(),
            },
        )
        rate = index / max(1e-6, time.time() - started)
        print(f"  {index}/{len(pending)} — {rate:.2f} img/s", end="\r", flush=True)

    manifest_path = writer.finalize()
    print(f"\ncache écrit : {manifest_path}")
    print(f"{writer.manifest['count']} entrées, {len(writer.manifest['errors'])} erreur(s)")
    if encodings:
        print(f"décodages appliqués : {encodings}")
    return 0 if not writer.manifest["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
