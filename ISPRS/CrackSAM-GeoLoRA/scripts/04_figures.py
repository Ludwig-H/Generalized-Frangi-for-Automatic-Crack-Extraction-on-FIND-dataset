#!/usr/bin/env python3
"""Figures du rapport : réussites, échecs, et vues agrégées.

Le rapport doit montrer les deux : les cas où la géométrie aide et ceux où elle
nuit. Une galerie qui ne montrerait que des réussites serait une illustration,
pas une preuve.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geolora.evidence import EVIDENCE_SIZE  # noqa: E402

SIZE = 448


def load_rgb(data_root: Path, split: str, name: str) -> np.ndarray:
    stem = Path(name.strip()).name
    directory = data_root / split / "images"
    path = directory / stem
    if not path.exists():
        path = next(iter(directory.glob(Path(stem).stem + ".*")))
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))


def load_mask(data_root: Path, split: str, name: str) -> np.ndarray:
    stem = Path(name.strip()).name
    directory = data_root / split / "masks"
    path = directory / stem
    if not path.exists():
        path = next(iter(directory.glob(Path(stem).stem + ".*")))
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 127


def tint(base: np.ndarray, mask: np.ndarray, colour, alpha=0.5) -> np.ndarray:
    out = base.astype(np.float32) / 255.0
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(colour, dtype=np.float32)
    return np.clip(out, 0, 1)


def agreement(base: np.ndarray, prediction: np.ndarray, truth: np.ndarray):
    """Vert = vrai positif, rouge = faux positif, bleu = faux négatif."""

    out = base.astype(np.float32) / 255.0 * 0.55
    out[prediction & truth] = (0.15, 0.85, 0.25)
    out[prediction & ~truth] = (0.90, 0.15, 0.15)
    out[~prediction & truth] = (0.15, 0.45, 0.95)
    return np.clip(out, 0, 1)


def case_panel(row, data_root, split, evidence_root, masks, destination) -> None:
    name = row["name"]
    image = load_rgb(data_root, split, name)
    truth = load_mask(data_root, split, name)
    stem = Path(name).stem
    baseline = np.load(masks["baseline"] / f"{stem}.npy").astype(bool)
    geo = np.load(masks["geo"] / f"{stem}.npy").astype(bool)
    evidence = np.load(evidence_root / "test" / f"{stem}.npy").astype(np.float32)

    figure, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    axes = axes.ravel()
    axes[0].imshow(image); axes[0].set_title("image")
    axes[1].imshow(tint(image, truth, (0.15, 0.85, 0.25)))
    axes[1].set_title(f"vérité terrain ({truth.mean()*100:.1f} %)")
    axes[2].imshow(agreement(image, baseline, truth))
    axes[2].set_title(f"référence — IoU {float(row['baseline_iou']):.3f}")
    axes[3].imshow(agreement(image, geo, truth))
    delta = float(row["geo_iou"]) - float(row["baseline_iou"])
    axes[3].set_title(f"variante — IoU {float(row['geo_iou']):.3f} ({delta:+.3f})")

    for position, (index, label) in enumerate(
        ((0, "frangi_sim"), (1, "ofs"), (2, "ofa"), (5, "phase_sym"))
    ):
        axis = axes[4 + position]
        axis.imshow(evidence[index], cmap="magma", vmin=0, vmax=1)
        axis.set_title(label)

    for axis in axes:
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(
        f"{name} — vert : vrai positif · rouge : faux positif · bleu : manqué",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=100, bbox_inches="tight")
    plt.close(figure)


def aggregate_figure(rows, destination) -> None:
    baseline = np.array([float(r["baseline_iou"]) for r in rows])
    geo = np.array([float(r["geo_iou"]) for r in rows])
    delta = geo - baseline

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].scatter(baseline, geo, s=5, alpha=0.25, color="#1a73e8")
    limits = [0, 1]
    axes[0].plot(limits, limits, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("IoU référence"); axes[0].set_ylabel("IoU variante")
    axes[0].set_title("Par image : au-dessus de la diagonale = gain")
    axes[0].set_xlim(limits); axes[0].set_ylim(limits)

    axes[1].hist(delta, bins=80, color="#188038", alpha=0.85)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].axvline(delta.mean(), color="#d93025", linewidth=2,
                    label=f"moyenne {delta.mean():+.4f}")
    axes[1].set_xlabel("Δ IoU (variante − référence)")
    axes[1].set_ylabel("images"); axes[1].legend(fontsize=9)
    axes[1].set_title("Distribution des deltas appariés")

    order = np.argsort(delta)
    axes[2].plot(np.arange(delta.size), delta[order], color="#1a73e8", linewidth=1)
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[2].fill_between(np.arange(delta.size), delta[order], 0,
                         where=delta[order] > 0, color="#188038", alpha=0.4)
    axes[2].fill_between(np.arange(delta.size), delta[order], 0,
                         where=delta[order] < 0, color="#d93025", alpha=0.4)
    axes[2].set_xlabel("images triées par Δ IoU"); axes[2].set_ylabel("Δ IoU")
    gains = int((delta > 1e-6).sum()); losses = int((delta < -1e-6).sum())
    axes[2].set_title(f"{gains} gains · {losses} pertes · {delta.size-gains-losses} nuls")

    figure.tight_layout()
    figure.savefig(destination, dpi=120, bbox_inches="tight")
    plt.close(figure)


def ladder_figure(summaries: dict, destination) -> None:
    names = list(summaries)
    values = [summaries[n]["with_evidence"]["iou"] for n in names]
    figure, axis = plt.subplots(figsize=(9, 4.6))
    order = ["baseline", "cldice", "geo", "tol3", "geo_tol3", "geo_tol3_permuted", "tol5"]
    names = [n for n in order if n in summaries] + [n for n in names if n not in order]
    values = [summaries[n]["with_evidence"]["iou"] for n in names]
    palette = {"baseline": "#5f6368", "cldice": "#1a73e8", "geo": "#4285f4",
               "tol3": "#188038", "geo_tol3": "#34a853",
               "geo_tol3_permuted": "#d93025", "tol5": "#0b8043"}
    colours = [palette.get(n, "#9aa0a6") for n in names]
    bars = axis.bar(names, values, color=colours)
    reference = summaries.get("baseline", {}).get("with_evidence", {}).get("iou")
    if reference:
        axis.axhline(reference, color="black", linestyle="--", linewidth=1,
                     label=f"baseline {reference:.4f}")
        axis.legend(fontsize=9)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.002,
                  f"{value:.4f}", ha="center", fontsize=9)
    axis.set_ylabel("IoU sur le test Khánh Hà")
    axis.set_ylim(min(values) - 0.03, max(values) + 0.02)
    axis.set_title("Échelle d'ablations, budget strictement égal")
    figure.tight_layout()
    figure.savefig(destination, dpi=120, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument(
        "--pair", nargs=2, default=("baseline", "geo"),
        metavar=("REFERENCE", "VARIANTE"),
        help="paire comparée dans les panneaux par cas",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation = args.run_root / "eval"

    summaries = {}
    for path in sorted(evaluation.glob("eval_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries[payload["variant"]] = payload
    if summaries:
        ladder_figure(summaries, args.output / "ablation_ladder.png")
        print("écrit ablation_ladder.png")

    reference, variant = args.pair
    baseline_csv = evaluation / f"per_image_{reference}.csv"
    geo_csv = evaluation / f"per_image_{variant}.csv"
    if not (baseline_csv.exists() and geo_csv.exists()):
        print("per-image manquant : figures par cas ignorées")
        return 0

    def read(path):
        with path.open(encoding="utf-8") as handle:
            return {r["name"]: r for r in csv.DictReader(handle)}

    base_rows, geo_rows = read(baseline_csv), read(geo_csv)
    rows = []
    for name, entry in base_rows.items():
        if name not in geo_rows:
            continue
        rows.append(
            {
                "name": name,
                "baseline_iou": entry["with_evidence_iou"],
                "geo_iou": geo_rows[name]["with_evidence_iou"],
            }
        )
    aggregate_figure(rows, args.output / f"per_image_{reference}_vs_{variant}.png")
    print(f"écrit per_image_{reference}_vs_{variant}.png ({len(rows)} images)")

    masks = {
        "baseline": evaluation / f"masks_{reference}",
        "geo": evaluation / f"masks_{variant}",
    }
    if not all(path.exists() for path in masks.values()):
        print("masques manquants : panneaux par cas ignorés")
        return 0

    rows.sort(key=lambda r: float(r["geo_iou"]) - float(r["baseline_iou"]))
    selection = (
        [("echec", r) for r in rows[: args.cases]]
        + [("reussite", r) for r in rows[-args.cases :]]
    )
    for position, (label, row) in enumerate(selection):
        stem = Path(row["name"]).stem
        case_panel(
            row, args.data_root, args.split, args.run_root / "evidence", masks,
            args.output / f"case_{reference}_vs_{variant}_{label}_{position:02d}_{stem}.png",
        )
        print(f"écrit case {label} {stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
