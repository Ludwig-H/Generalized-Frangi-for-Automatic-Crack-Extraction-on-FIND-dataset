#!/usr/bin/env python3
"""Précalcule et met en cache l'évidence géométrique réaccordée.

Indispensable : `19 s` par image interdit de la recalculer à chaque époque.
Le cache est écrit en float16, un fichier par image, avec écriture atomique
pour être reprenable après une préemption Spot.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geolora.evidence import (  # noqa: E402
    EVIDENCE_SIZE,
    FRANGI_SCALES,
    N_CHANNELS,
    OFS_RADII,
    PHASE_WAVELENGTHS,
    PROFILE_SIGMAS,
    TARGET_WIDTH,
    compute_evidence,
)


def _resolve(data_root: Path, split: str, name: str) -> Path:
    stem = Path(name.strip()).name
    images = data_root / split / "images"
    path = images / stem
    if path.exists():
        return path
    candidates = list(images.glob(Path(stem).stem + ".*"))
    if not candidates:
        raise FileNotFoundError(f"image introuvable pour {name!r}")
    return candidates[0]


def process(data_root: Path, split: str, name: str, destination: Path) -> str:
    key = Path(name.strip()).stem
    target = destination / f"{key}.npy"
    if target.exists():
        return key
    path = _resolve(data_root, split, name)
    with Image.open(path) as handle:
        image = np.asarray(
            handle.convert("RGB").resize((EVIDENCE_SIZE,) * 2, Image.BILINEAR),
            dtype=np.uint8,
        )
    planes = compute_evidence(image).planes.astype(np.float16)
    # np.save ajoute ".npy" si le nom n'en a pas : le temporaire doit donc
    # déjà se terminer par ".npy", faute de quoi os.replace échoue.
    temporary = target.with_name(f"{key}.{os.getpid()}.tmp.npy")
    np.save(temporary, planes)
    os.replace(temporary, target)
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.output / args.split
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        line.strip()
        for line in args.list_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_samples:
        names = names[: args.max_samples]

    from joblib import Parallel, delayed
    from tqdm import tqdm

    Parallel(n_jobs=args.jobs)(
        delayed(process)(args.data_root, args.split, name, destination)
        for name in tqdm(names, desc=f"évidence {args.split}")
    )

    manifest = {
        "split": args.split,
        "images": len(names),
        "cached": len(list(destination.glob("*.npy"))),
        "evidence_size": EVIDENCE_SIZE,
        "channels": N_CHANNELS,
        "target_width_px": round(TARGET_WIDTH, 2),
        "frangi_scales": list(FRANGI_SCALES),
        "ofs_radii": list(OFS_RADII),
        "phase_wavelengths": list(PHASE_WAVELENGTHS),
        "profile_sigmas": list(PROFILE_SIGMAS),
    }
    (args.output / f"manifest_{args.split}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
