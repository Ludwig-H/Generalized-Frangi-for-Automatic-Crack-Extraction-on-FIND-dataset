#!/usr/bin/env python3
"""Évaluation face à la baseline gelée, avec les cinq contrôles causaux.

Aucun gain n'est revendiquable sans ces contrôles (§9.1 du rapport du 8 août) :

``none``
    ``z0`` exact, la baseline. C'est la voie SAM 2 officielle **sans prompt**,
    et non un tenseur nul.
``null``
    Les features d'évidence sont mises à zéro, les corridors restent. Teste si
    le gain vient de l'évidence ou de la simple existence d'une enveloppe.
``permuted``
    Les fragments d'une image sont appliqués à une **autre** image. Couverture
    identique, alignement détruit.
``shifted``
    Les fragments sont translatés d'un décalage fixe. Couverture identique,
    alignement local détruit.
``random_support``
    Fragments synthétiques de même nombre, longueur et rayon, placés au hasard.

Le delta face à ``z0`` est apparié par image ; l'IC95 est obtenu par bootstrap
groupé par scène physique, graine fixe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRACKSAM = ROOT.parent / "CrackSAM"
if str(CRACKSAM) not in sys.path:
    sys.path.insert(0, str(CRACKSAM))

from gfa.arbiter import ABSTAIN, ADD, REMOVE, FragmentArbiter  # noqa: E402
from gfa.evidence import SUPPORT_SOURCES  # noqa: E402
from gfa.features import sam_features  # noqa: E402
from gfa.fragments import Fragment  # noqa: E402
from gfa.oracles import DELTA, apply_actions, binary_iou  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402

_oracles_cli = import_module("03_run_oracles")
load_mask = _oracles_cli.load_mask
grouped_bootstrap = _oracles_cli.grouped_bootstrap

CONTROLS = ("none", "gfa", "null", "permuted", "shifted", "random_support")
SHIFT = (17, 23)
IMAGE_SIZE = 448
SEED = 20260808


def read_fragments(store, key: str) -> list[Fragment]:
    return _oracles_cli.load_fragments(store, key)


def shift_fragments(fragments: list[Fragment], offset: tuple[int, int]) -> list[Fragment]:
    moved = []
    for fragment in fragments:
        pixels = fragment.pixels.copy()
        pixels[:, 0] = np.clip(pixels[:, 0] + offset[0], 0, IMAGE_SIZE - 1)
        pixels[:, 1] = np.clip(pixels[:, 1] + offset[1], 0, IMAGE_SIZE - 1)
        moved.append(
            Fragment(pixels=pixels, radius=fragment.radius, sources=fragment.sources)
        )
    return moved


def random_fragments(
    fragments: list[Fragment], rng: np.random.Generator
) -> list[Fragment]:
    """Même nombre, même longueur, même rayon, position et angle au hasard."""

    synthetic = []
    for fragment in fragments:
        length = fragment.length
        angle = float(rng.uniform(0.0, np.pi))
        start_row = float(rng.integers(0, IMAGE_SIZE))
        start_col = float(rng.integers(0, IMAGE_SIZE))
        steps = np.arange(length, dtype=np.float64)
        rows = np.clip(start_row + steps * np.sin(angle), 0, IMAGE_SIZE - 1)
        cols = np.clip(start_col + steps * np.cos(angle), 0, IMAGE_SIZE - 1)
        pixels = np.stack([rows, cols], axis=1).round().astype(np.int32)
        synthetic.append(
            Fragment(pixels=pixels, radius=fragment.radius, sources=fragment.sources)
        )
    return synthetic


def decide(
    model: FragmentArbiter,
    evidence_features: np.ndarray,
    fragments: list[Fragment],
    z0: np.ndarray,
    device: str,
    zero_evidence: bool = False,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not fragments:
        return [], np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    evidence_block = (
        np.zeros_like(evidence_features) if zero_evidence else evidence_features
    )
    sam_block = np.stack([sam_features(z0, f) for f in fragments]).astype(np.float32)
    features = np.concatenate([evidence_block, sam_block], axis=1)
    tensor = torch.from_numpy(features[None, ...]).float().to(device)
    with torch.no_grad():
        output = model(tensor)
        chosen = output.hard_actions()[0].cpu().numpy()
        add = output.add_amplitude[0].cpu().numpy()
        remove = output.remove_amplitude[0].cpu().numpy()
    names = {ABSTAIN: "abstain", ADD: "add", REMOVE: "remove"}
    return [names[int(value)] for value in chosen], add, remove


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--z0-dir", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--arbiter-suffix", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--controls", nargs="+", default=list(CONTROLS), choices=list(CONTROLS)
    )
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_root = args.run_root
    z0_dir = args.z0_dir or (run_root / "baseline" / "z0")

    index = json.loads(
        (run_root / "evidence" / f"evidence_summary_{args.split}.json").read_text(
            encoding="utf-8"
        )
    )["index"]
    if args.max_samples:
        index = index[: args.max_samples]
    store = np.load(run_root / "evidence" / f"fragments_{args.split}.npz")

    checkpoint = torch.load(
        run_root / "arbiter" / f"fold_{args.fold}{args.arbiter_suffix}.pt",
        map_location=device,
    )
    model = FragmentArbiter(input_dim=int(checkpoint["input_dim"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    rng = np.random.default_rng(SEED)
    keys = [str(entry["key"]) for entry in index]
    permutation = rng.permutation(len(keys))

    records: dict[str, list[float]] = {control: [] for control in args.controls}
    names: list[str] = []

    from tqdm import tqdm

    for position, entry in enumerate(tqdm(index, desc=f"évaluation {args.split}")):
        key = str(entry["key"])
        logits_path = z0_dir / f"{key}.npy"
        if not logits_path.exists():
            continue
        z0 = np.load(logits_path).astype(np.float32)
        truth = load_mask(args.data_root, args.split, str(entry["name"]))
        base_fragments = read_fragments(store, key)
        base_features = store[f"{key}/features"]
        names.append(str(entry["name"]))

        for control in args.controls:
            if control == "none":
                records[control].append(binary_iou(z0 > 0.0, truth))
                continue

            fragments, features, zero_evidence = base_fragments, base_features, False
            if control == "null":
                zero_evidence = True
            elif control == "permuted":
                other = keys[int(permutation[position])]
                fragments = read_fragments(store, other)
                features = store[f"{other}/features"]
            elif control == "shifted":
                fragments = shift_fragments(base_fragments, SHIFT)
            elif control == "random_support":
                fragments = random_fragments(base_fragments, rng)

            if not fragments or features.size == 0:
                records[control].append(binary_iou(z0 > 0.0, truth))
                continue
            actions, _, _ = decide(
                model, features, fragments, z0, device, zero_evidence
            )
            corrected = apply_actions(z0, fragments, actions, args.delta)
            records[control].append(binary_iou(corrected > 0.0, truth))

    baseline = np.asarray(records["none"], dtype=np.float64)
    summary: dict[str, object] = {
        "split": args.split,
        "images": len(baseline),
        "fold": args.fold,
        "baseline_iou": float(baseline.mean()),
    }
    for control in args.controls:
        if control == "none":
            continue
        values = np.asarray(records[control], dtype=np.float64)
        delta = values - baseline
        low, high = grouped_bootstrap(delta, 2000, SEED)
        summary[control] = {
            "iou": float(values.mean()),
            "delta_vs_baseline": float(delta.mean()),
            "ci95": [low, high],
            "significant": bool(low > 0.0),
        }

    gfa = summary.get("gfa")
    if isinstance(gfa, dict):
        beats_controls = all(
            gfa["delta_vs_baseline"] > summary[c]["delta_vs_baseline"]
            for c in ("permuted", "shifted", "random_support")
            if isinstance(summary.get(c), dict)
        )
        summary["gate3_real_gain"] = bool(gfa["significant"])
        summary["gate4_causal"] = bool(beats_controls)

    output_dir = run_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"evaluation_{args.split}{args.tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / f"per_image_{args.split}{args.tag}.csv").open("w", encoding="utf-8") as fh:
        fh.write("name," + ",".join(args.controls) + "\n")
        for position, name in enumerate(names):
            row = ",".join(f"{records[c][position]:.6f}" for c in args.controls)
            fh.write(f"{name},{row}\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
