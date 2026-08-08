#!/usr/bin/env python3
"""Calcule l'évidence, propose les fragments et met leurs descripteurs en cache.

Entièrement CPU : ce script n'ouvre ni SAM 2 ni GPU, et peut donc tourner en
parallèle du cache de ``z0``. Il ne stocke pas le tenseur d'évidence complet
(4,4 Mo par image en float16) mais uniquement les polylignes et les features,
soit environ 25 ko par image.

Deux modes :

``--calibrate``
    Estime les seuils **absolus** par source comme le quantile ``q`` de la
    réponse sur un échantillon d'images d'entraînement, puis les écrit une fois
    pour toutes. Ces seuils sont ensuite gelés : ne jamais les recalculer par
    image, c'est précisément le défaut relatif que le rapport reproche à la
    chaîne historique (§2.2).

mode normal
    Charge les seuils gelés et produit les fragments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gfa.evidence import SUPPORT_SOURCES, compute_evidence  # noqa: E402
from gfa.features import evidence_features  # noqa: E402
from gfa.fragments import propose_fragments  # noqa: E402

IMAGE_SIZE = 448
CALIBRATION_QUANTILE = 0.995


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
        )
    return np.asarray(image, dtype=np.uint8)


def _resolve(data_root: Path, split: str, name: str) -> tuple[Path, Path]:
    stem = Path(name.strip()).name
    images = data_root / split / "images"
    masks = data_root / split / "masks"
    image_path = images / stem
    if not image_path.exists():
        candidates = list(images.glob(Path(stem).stem + ".*"))
        if not candidates:
            raise FileNotFoundError(f"image introuvable pour {name!r}")
        image_path = candidates[0]
    mask_path = masks / image_path.name
    if not mask_path.exists():
        candidates = list(masks.glob(image_path.stem + ".*"))
        if not candidates:
            raise FileNotFoundError(f"masque introuvable pour {name!r}")
        mask_path = candidates[0]
    return image_path, mask_path


def _calibration_quantiles(path: Path, quantile: float) -> dict[str, float]:
    evidence = compute_evidence(_load_image(path))
    return {
        name: float(np.quantile(evidence.plane(name), quantile))
        for name in SUPPORT_SOURCES
    }


def calibrate(paths: list[Path], quantile: float, jobs: int = -1) -> dict[str, float]:
    """Seuils absolus, un par source, estimés une seule fois sur le train."""

    from joblib import Parallel, delayed

    per_image = Parallel(n_jobs=jobs)(
        delayed(_calibration_quantiles)(path, quantile) for path in paths
    )
    collected: dict[str, list[float]] = {name: [] for name in SUPPORT_SOURCES}
    for record in per_image:
        for name in SUPPORT_SOURCES:
            collected[name].append(record[name])
    # La médiane des quantiles par image est plus robuste qu'un quantile global
    # calculé sur un échantillon d'images concaténées, où une seule scène très
    # contrastée dominerait.
    return {name: float(np.median(values)) for name, values in collected.items()}


def process_one(
    data_root: Path, split: str, name: str, thresholds: dict[str, float]
) -> dict[str, object]:
    image_path, _ = _resolve(data_root, split, name)
    evidence = compute_evidence(_load_image(image_path))
    fragments = propose_fragments(evidence, thresholds)
    if not fragments:
        return {
            "name": name,
            "n_fragments": 0,
            "polylines": np.zeros((0, 2), dtype=np.int16),
            "offsets": np.zeros(1, dtype=np.int32),
            "radii": np.zeros(0, dtype=np.float32),
            "source_flags": np.zeros((0, len(SUPPORT_SOURCES)), dtype=np.uint8),
            "features": np.zeros((0, 0), dtype=np.float32),
            "noise_sigma": evidence.noise_sigma,
        }
    polylines = np.concatenate([f.pixels for f in fragments]).astype(np.int16)
    offsets = np.cumsum([0] + [f.length for f in fragments]).astype(np.int32)
    radii = np.asarray([f.radius for f in fragments], dtype=np.float32)
    flags = np.asarray(
        [[1 if s in f.sources else 0 for s in SUPPORT_SOURCES] for f in fragments],
        dtype=np.uint8,
    )
    features = np.stack([evidence_features(evidence, f) for f in fragments])
    return {
        "name": name,
        "n_fragments": len(fragments),
        "polylines": polylines,
        "offsets": offsets,
        "radii": radii,
        "source_flags": flags,
        "features": features.astype(np.float32),
        "noise_sigma": evidence.noise_sigma,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-samples", type=int, default=120)
    parser.add_argument("--quantile", type=float, default=CALIBRATION_QUANTILE)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = [
        line.strip()
        for line in args.list_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_samples:
        names = names[: args.max_samples]

    if args.calibrate:
        rng = np.random.default_rng(20260808)
        chosen = rng.choice(
            len(names), size=min(args.calibration_samples, len(names)), replace=False
        )
        paths = [_resolve(args.data_root, args.split, names[i])[0] for i in chosen]
        thresholds = calibrate(paths, args.quantile, args.jobs)
        payload = {
            "thresholds": thresholds,
            "quantile": args.quantile,
            "samples": len(paths),
            "split": args.split,
            "frozen": True,
        }
        destination = args.thresholds or (args.output / "thresholds.json")
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.thresholds is None:
        raise SystemExit("--thresholds est obligatoire hors mode --calibrate")
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))["thresholds"]

    from joblib import Parallel, delayed
    from tqdm import tqdm

    results = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(process_one)(args.data_root, args.split, name, thresholds)
        for name in tqdm(names, desc=f"évidence {args.split}")
    )

    store = args.output / f"fragments_{args.split}.npz"
    payload: dict[str, np.ndarray] = {}
    index: list[dict[str, object]] = []
    for record in results:
        key = Path(str(record["name"])).stem
        payload[f"{key}/polylines"] = record["polylines"]
        payload[f"{key}/offsets"] = record["offsets"]
        payload[f"{key}/radii"] = record["radii"]
        payload[f"{key}/flags"] = record["source_flags"]
        payload[f"{key}/features"] = record["features"]
        index.append(
            {
                "name": record["name"],
                "key": key,
                "n_fragments": int(record["n_fragments"]),
                "noise_sigma": float(record["noise_sigma"]),
            }
        )
    np.savez_compressed(store, **payload)

    counts = np.asarray([row["n_fragments"] for row in index])
    summary = {
        "split": args.split,
        "images": len(index),
        "fragments_total": int(counts.sum()),
        "fragments_mean": float(counts.mean()) if counts.size else 0.0,
        "fragments_median": float(np.median(counts)) if counts.size else 0.0,
        "images_without_fragment": int((counts == 0).sum()),
        "thresholds": thresholds,
    }
    (args.output / f"evidence_summary_{args.split}.json").write_text(
        json.dumps({"summary": summary, "index": index}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
