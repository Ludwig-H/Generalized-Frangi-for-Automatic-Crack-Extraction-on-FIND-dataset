#!/usr/bin/env python3
"""Figures du rapport : panneaux par cas et distributions agrégées."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gfa.evidence import CHANNELS, compute_evidence  # noqa: E402
from gfa.oracles import apply_actions, binary_iou, source_oracle  # noqa: E402
from importlib import import_module  # noqa: E402

_cli = import_module("03_run_oracles")

IMAGE_SIZE = 448
ACTION_COLOURS = {"abstain": "#9aa0a6", "add": "#1a73e8", "remove": "#d93025"}


def load_image(data_root: Path, split: str, name: str) -> np.ndarray:
    stem = Path(name.strip()).name
    images = data_root / split / "images"
    path = images / stem
    if not path.exists():
        path = next(iter(images.glob(Path(stem).stem + ".*")))
    with Image.open(path) as handle:
        return np.asarray(
            handle.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        )


def overlay(base: np.ndarray, mask: np.ndarray, colour, alpha: float = 0.55):
    out = base.astype(np.float32) / 255.0
    tint = np.asarray(colour, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * tint
    return np.clip(out, 0, 1)


def case_panel(
    image, truth, z0, evidence, fragments, result, destination: Path, title: str
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(19, 8))
    axes = axes.ravel()

    axes[0].imshow(image); axes[0].set_title("image")
    axes[1].imshow(overlay(image, truth, (0.15, 0.85, 0.25)))
    axes[1].set_title(f"vérité terrain ({truth.mean()*100:.2f} %)")
    prediction = z0 > 0
    axes[2].imshow(overlay(image, prediction, (0.95, 0.6, 0.1)))
    axes[2].set_title(f"baseline $z_0$ — IoU {result.baseline_iou:.3f}")

    for position, channel in enumerate(("frangi_sim", "ofs", "ofa", "dbic")):
        axis = axes[3 + position]
        axis.imshow(evidence.plane(channel), cmap="magma", vmin=0, vmax=1)
        axis.set_title(channel)

    support = np.zeros(truth.shape, dtype=bool)
    for fragment in fragments:
        support[fragment.pixels[:, 0], fragment.pixels[:, 1]] = True
    axes[7].imshow(overlay(image, support, (0.1, 0.4, 0.95), alpha=0.9))
    axes[7].set_title(f"{len(fragments)} fragments proposés")

    action_map = np.zeros((*truth.shape, 3), dtype=np.float32)
    action_map[:] = image.astype(np.float32) / 255.0
    for fragment, action in zip(fragments, result.actions):
        colour = np.asarray(
            [int(ACTION_COLOURS[action][i : i + 2], 16) / 255.0 for i in (1, 3, 5)],
            dtype=np.float32,
        )
        rows, cols = fragment.pixels[:, 0], fragment.pixels[:, 1]
        action_map[rows, cols] = colour
    axes[8].imshow(np.clip(action_map, 0, 1))
    axes[8].set_title(
        f"oracle : +{result.n_add} / −{result.n_remove} / "
        f"={result.n_fragments - result.n_add - result.n_remove}"
    )

    corrected = apply_actions(z0, fragments, list(result.actions))
    axes[9].imshow(overlay(image, corrected > 0, (0.2, 0.75, 0.95)))
    axes[9].set_title(
        f"oracle appliqué — IoU {result.achievable_iou:.3f} "
        f"({result.achievable_gain:+.3f})"
    )

    for axis in axes:
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    figure.savefig(destination, dpi=110, bbox_inches="tight")
    plt.close(figure)


def evidence_panel(image, evidence, destination: Path, title: str) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    axes[0].imshow(image); axes[0].set_title("image");
    for position, channel in enumerate(CHANNELS):
        axis = axes[position + 1]
        plane = evidence.plane(channel)
        if channel in {"cos2theta", "sin2theta"}:
            axis.imshow(plane, cmap="twilight", vmin=-1, vmax=1)
        elif channel == "scale":
            axis.imshow(plane, cmap="viridis")
        else:
            axis.imshow(plane, cmap="magma", vmin=0, vmax=1)
        axis.set_title(channel)
    for axis in axes:
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    figure.savefig(destination, dpi=110, bbox_inches="tight")
    plt.close(figure)


def summary_figures(oracle_csv: Path, utilities: Path, destination: Path) -> None:
    import csv

    with oracle_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    achievable = np.asarray([float(r["achievable_gain"]) for r in rows])
    upper = np.asarray([float(r["upper_bound_gain"]) for r in rows])
    counts = np.asarray([int(r["n_fragments"]) for r in rows])

    store = np.load(utilities)
    best = np.concatenate(
        [np.abs(store[k]).max(axis=1) for k in store.files if k.endswith("/utilities")]
    )

    figure, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    axes[0].hist(achievable, bins=60, color="#1a73e8", alpha=0.85, label="atteignable")
    axes[0].hist(upper, bins=60, color="#d93025", alpha=0.5, label="borne supérieure")
    axes[0].axvline(0.01, color="black", linestyle="--", label="seuil pré-enregistré")
    axes[0].set_xlabel("gain d'IoU par image"); axes[0].set_ylabel("scènes")
    axes[0].set_title("Plafond de l'oracle de source (1 500 scènes)")
    axes[0].legend(fontsize=8); axes[0].set_xlim(-0.01, 0.2)

    order = np.sort(best)[::-1]
    share = np.cumsum(order) / max(order.sum(), 1e-9)
    axes[1].plot(np.arange(1, order.size + 1) / order.size * 100, share * 100,
                 color="#1a73e8")
    axes[1].axhline(90, color="black", linestyle=":", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("% des fragments, triés par utilité décroissante")
    axes[1].set_ylabel("% de l'utilité totale")
    axes[1].set_title("L'utilité est concentrée dans une queue minuscule")

    axes[2].hist(counts, bins=50, color="#188038", alpha=0.85)
    axes[2].set_xlabel("fragments proposés par image"); axes[2].set_ylabel("scènes")
    axes[2].set_title(f"Médiane {np.median(counts):.0f} fragments/image")

    figure.tight_layout()
    figure.savefig(destination, dpi=120, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    oracle_csv = args.run_root / "oracles" / f"source_oracle_{args.split}.csv"
    utilities = args.run_root / "oracles" / f"utilities_{args.split}.npz"
    summary_figures(oracle_csv, utilities, args.output / "oracle_overview.png")
    print("écrit oracle_overview.png")

    import csv

    with oracle_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: float(r["achievable_gain"]), reverse=True)
    empty = [r for r in rows if int(r["gt_empty"]) == 1]
    selection = rows[: args.cases // 2] + rows[-(args.cases // 2) :] + empty[:1]

    store = np.load(args.run_root / "evidence" / f"fragments_{args.split}.npz")
    z0_dir = args.run_root / (
        "baseline_analysis" if args.split == "train" else "baseline"
    ) / "z0"

    for position, row in enumerate(selection):
        name, key = row["name"], row["key"]
        image = load_image(args.data_root, args.split, name)
        truth = _cli.load_mask(args.data_root, args.split, name)
        z0 = np.load(z0_dir / f"{key}.npy").astype(np.float32)
        evidence = compute_evidence(image)
        fragments = _cli.load_fragments(store, key)
        result = source_oracle(z0, truth, fragments)
        label = "gain" if float(row["achievable_gain"]) > 0 else "neutre"
        case_panel(
            image, truth, z0, evidence, fragments, result,
            args.output / f"case_{position:02d}_{label}_{key}.png",
            f"{name} — oracle {result.achievable_gain:+.4f} IoU",
        )
        if position < 2:
            evidence_panel(
                image, evidence,
                args.output / f"channels_{position:02d}_{key}.png",
                f"Les 11 canaux d'évidence — {name}",
            )
        print(f"écrit case {position} ({key})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
