#!/usr/bin/env python3
"""Porte n° 1 : l'oracle de source, seuil pré-enregistré ``>= +0,01`` d'IoU.

Ce script ne charge pas SAM 2 : il consomme le ``z0`` mis en cache par
``01_cache_baseline.py`` et les fragments produits par ``02_cache_evidence.py``.

Deux quantités sont rapportées, et la distinction rend un résultat négatif
concluant :

``achievable``
    borne **inférieure** du plafond, par montée de coordonnées sur les actions ;
``upper_bound``
    borne **supérieure** stricte, en choisissant librement le label de chaque
    pixel à l'intérieur de l'union des corridors. Aucune assignation d'actions
    par fragment ne peut la dépasser.

Si ``upper_bound < +0,01``, la famille de candidats est réfutée sans réserve et
l'arbitre ne doit pas être entraîné.
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

from gfa.fragments import Fragment  # noqa: E402
from gfa.oracles import DELTA, marginal_utility, source_oracle  # noqa: E402

GATE_THRESHOLD = 0.01
IMAGE_SIZE = 448
BOOTSTRAP = 2000
SEED = 20260808


def load_fragments(store: np.lib.npyio.NpzFile, key: str) -> list[Fragment]:
    if f"{key}/offsets" not in store:
        return []
    offsets = store[f"{key}/offsets"]
    if offsets.size < 2:
        return []
    polylines = store[f"{key}/polylines"]
    radii = store[f"{key}/radii"]
    flags = store[f"{key}/flags"]
    from gfa.evidence import SUPPORT_SOURCES

    fragments: list[Fragment] = []
    for index in range(len(offsets) - 1):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        sources = tuple(
            name for name, flag in zip(SUPPORT_SOURCES, flags[index]) if flag
        )
        fragments.append(
            Fragment(
                pixels=polylines[start:stop].astype(np.int32),
                radius=float(radii[index]),
                sources=sources,
            )
        )
    return fragments


def load_mask(data_root: Path, split: str, name: str) -> np.ndarray:
    stem = Path(name.strip()).name
    masks = data_root / split / "masks"
    path = masks / stem
    if not path.exists():
        candidates = list(masks.glob(Path(stem).stem + ".*"))
        if not candidates:
            raise FileNotFoundError(f"masque introuvable pour {name!r}")
        path = candidates[0]
    with Image.open(path) as handle:
        mask = handle.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    return np.asarray(mask) > 127


def grouped_bootstrap(values: np.ndarray, rounds: int, seed: int) -> tuple[float, float]:
    """IC95 par rééchantillonnage des scènes physiques, graine fixe."""

    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(rounds, dtype=np.float64)
    for index in range(rounds):
        sample = rng.integers(0, values.size, size=values.size)
        means[index] = float(values[sample].mean())
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--z0-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-utilities", action="store_true")
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = json.loads(args.index.read_text(encoding="utf-8"))["index"]
    if args.max_samples:
        index = index[: args.max_samples]
    store = np.load(args.fragments, allow_pickle=False)

    def _one(entry: dict) -> dict | None:
        key = str(entry["key"])
        logits_path = args.z0_dir / f"{key}.npy"
        if not logits_path.exists():
            return None
        z0 = np.load(logits_path).astype(np.float32)
        truth = load_mask(args.data_root, args.split, str(entry["name"]))
        fragments = load_fragments(store, key)
        result = source_oracle(z0, truth, fragments, delta=args.delta)
        payload = {
            "row": {
                "name": str(entry["name"]),
                "key": key,
                "baseline_iou": result.baseline_iou,
                "achievable_iou": result.achievable_iou,
                "upper_bound_iou": result.upper_bound_iou,
                "achievable_gain": result.achievable_gain,
                "upper_bound_gain": result.upper_bound_gain,
                "n_fragments": result.n_fragments,
                "n_add": result.n_add,
                "n_remove": result.n_remove,
                "corridor_pixels": result.corridor_pixels,
                "gt_empty": int(not truth.any()),
            }
        }
        if args.save_utilities and fragments:
            payload["utilities"] = marginal_utility(
                z0, truth, fragments, delta=args.delta
            )
            payload["best_action"] = np.asarray(
                [{"abstain": 0, "add": 1, "remove": 2}[a] for a in result.actions],
                dtype=np.int8,
            )
        return payload

    from joblib import Parallel, delayed
    from tqdm import tqdm

    # Backend threads : ``_one`` est une fermeture non picklable, et le calcul
    # est presque entièrement NumPy, qui libère le GIL.
    computed = Parallel(n_jobs=args.jobs, prefer="threads", verbose=0)(
        delayed(_one)(entry) for entry in tqdm(index, desc=f"oracle {args.split}")
    )

    rows: list[dict[str, float | str | int]] = []
    utilities_payload: dict[str, np.ndarray] = {}
    missing = 0
    for payload in computed:
        if payload is None:
            missing += 1
            continue
        rows.append(payload["row"])
        if "utilities" in payload:
            key = str(payload["row"]["key"])
            utilities_payload[f"{key}/utilities"] = payload["utilities"]
            utilities_payload[f"{key}/best_action"] = payload["best_action"]

    if not rows:
        raise SystemExit("aucune image évaluée : vérifier --z0-dir et --index")

    achievable = np.asarray([row["achievable_gain"] for row in rows], dtype=np.float64)
    upper = np.asarray([row["upper_bound_gain"] for row in rows], dtype=np.float64)
    baseline = np.asarray([row["baseline_iou"] for row in rows], dtype=np.float64)
    fragments_per_image = np.asarray([row["n_fragments"] for row in rows])

    achievable_ci = grouped_bootstrap(achievable, BOOTSTRAP, SEED)
    upper_ci = grouped_bootstrap(upper, BOOTSTRAP, SEED)

    verdict = "GO" if float(achievable.mean()) >= GATE_THRESHOLD else "NO-GO"
    conclusive = bool(float(upper.mean()) < GATE_THRESHOLD)

    summary = {
        "split": args.split,
        "images": len(rows),
        "images_without_z0": missing,
        "delta": args.delta,
        "baseline_iou_mean": float(baseline.mean()),
        "achievable_gain_mean": float(achievable.mean()),
        "achievable_gain_ci95": achievable_ci,
        "upper_bound_gain_mean": float(upper.mean()),
        "upper_bound_gain_ci95": upper_ci,
        "fragments_mean": float(fragments_per_image.mean()),
        "images_without_fragment": int((fragments_per_image == 0).sum()),
        "gate_threshold": GATE_THRESHOLD,
        "gate1_verdict": verdict,
        "negative_is_conclusive": conclusive,
    }
    (args.output / f"source_oracle_{args.split}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    columns = list(rows[0].keys())
    with (args.output / f"source_oracle_{args.split}.csv").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row[column]) for column in columns) + "\n")
    if utilities_payload:
        np.savez_compressed(
            args.output / f"utilities_{args.split}.npz", **utilities_payload
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
