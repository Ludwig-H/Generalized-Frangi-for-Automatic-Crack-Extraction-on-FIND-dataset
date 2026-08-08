#!/usr/bin/env python3
"""IoU tolérante : mesurer sans exiger l'exactitude au pixel près.

Sur des structures aussi fines que des fissures, l'IoU stricte est dominée par
le placement exact de la frontière. Mesuré sur ce corpus : dilater la vérité
terrain d'**un seul pixel** la fait chuter à `0,881` contre elle-même, et de deux
pixels à `0,799`. Comparer deux méthodes à `±0,02` d'IoU stricte revient donc
largement à comparer leur biais de largeur.

Deux définitions sont calculées, et elles ne disent pas la même chose.

``dilate_both``
    ``IoU(dilate(P, k), dilate(G, k))``. C'est la convention déjà employée dans
    le dépôt — ``thicken(sk, pixels=6)`` avant Jaccard, dans le code EUVIP.
    Simple, mais elle sature vers 1 quand ``k`` grandit, puisqu'on épaissit tout.

``buffered``
    Précision et rappel tolérants au sens « à moins de ``k`` pixels » :

    * ``P_k = |P ∩ dilate(G,k)| / |P|`` — un pixel prédit compte s'il tombe à
      moins de ``k`` de la vérité ;
    * ``R_k = |G ∩ dilate(P,k)| / |G|`` — un pixel vrai est couvert s'il a une
      prédiction à moins de ``k``.

    Puis ``F1_k`` et l'IoU correspondante. Cette forme ne dilate jamais les deux
    masques simultanément : elle mesure une distance d'appariement, pas un
    recouvrement de versions épaissies. C'est la définition à privilégier pour
    conclure.

La dilatation est euclidienne (transformée de distance), donc isotrope, et non
un élément structurant carré.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

#: Tolérances en pixels, à 448. Zéro reproduit l'IoU stricte.
TOLERANCES = (0, 1, 2, 3, 5, 8)
IMAGE_SIZE = 448


def dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    """Dilatation euclidienne : tout pixel à distance <= radius du support."""

    if radius <= 0:
        return mask.astype(bool)
    if not mask.any():
        return mask.astype(bool)
    return ndi.distance_transform_edt(~mask.astype(bool)) <= float(radius)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    if not a.any() and not b.any():
        return 1.0
    if not a.any() or not b.any():
        return 0.0
    return float(np.count_nonzero(a & b)) / float(np.count_nonzero(a | b))


def tolerant_scores(
    prediction: np.ndarray, truth: np.ndarray, radius: float
) -> dict[str, float]:
    """Les deux familles de scores pour une tolérance donnée."""

    prediction = prediction.astype(bool)
    truth = truth.astype(bool)

    both = _iou(dilate(prediction, radius), dilate(truth, radius))

    if not prediction.any() and not truth.any():
        return {"iou_dilate_both": both, "precision": 1.0, "recall": 1.0,
                "f1": 1.0, "iou_buffered": 1.0}
    if not prediction.any() or not truth.any():
        return {"iou_dilate_both": both, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "iou_buffered": 0.0}

    precision = float(
        np.count_nonzero(prediction & dilate(truth, radius))
    ) / float(np.count_nonzero(prediction))
    recall = float(
        np.count_nonzero(truth & dilate(prediction, radius))
    ) / float(np.count_nonzero(truth))
    denominator = precision + recall
    f1 = 0.0 if denominator <= 0 else 2.0 * precision * recall / denominator
    # IoU équivalente au F1 tolérant : IoU = F1 / (2 - F1).
    buffered = 0.0 if f1 <= 0 else f1 / (2.0 - f1)
    return {
        "iou_dilate_both": both,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_buffered": buffered,
    }


def load_truth(data_root: Path, split: str, name: str) -> np.ndarray:
    stem = Path(name.strip()).name
    directory = data_root / split / "masks"
    path = directory / stem
    if not path.exists():
        path = next(iter(directory.glob(Path(stem).stem + ".*")))
    with Image.open(path) as handle:
        return (
            np.asarray(
                handle.convert("L").resize((IMAGE_SIZE,) * 2, Image.NEAREST)
            )
            > 127
        )


def process(mask_dir: Path, data_root: Path, split: str, stem: str):
    prediction = np.load(mask_dir / f"{stem}.npy").astype(bool)
    truth = load_truth(data_root, split, stem)
    return stem, {r: tolerant_scores(prediction, truth, r) for r in TOLERANCES}


def _plot(summary: dict, destination: Path) -> None:
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    colours = {"baseline": "#5f6368", "cldice": "#1a73e8", "geo": "#d93025"}
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    for variant, payload in summary.items():
        colour = colours.get(variant)
        per = payload["per_tolerance"]
        xs = list(TOLERANCES)
        axes[0].plot(xs, [per[str(k)]["iou_buffered"] for k in xs], "o-",
                     label=variant, color=colour)
        axes[1].plot(xs, [per[str(k)]["iou_dilate_both"] for k in xs], "o-",
                     label=variant, color=colour)
        axes[2].plot([per[str(k)]["recall"] for k in xs],
                     [per[str(k)]["precision"] for k in xs], "o-",
                     label=variant, color=colour)
        for k in xs:
            axes[2].annotate(f"{k}", (per[str(k)]["recall"],
                                      per[str(k)]["precision"]),
                             fontsize=7, xytext=(3, -8),
                             textcoords="offset points")

    axes[0].set_xlabel("tolérance k (px)"); axes[0].set_ylabel("IoU tampon")
    axes[0].set_title("IoU tolérante — appariement à moins de k px")
    axes[1].set_xlabel("tolérance k (px)"); axes[1].set_ylabel("IoU")
    axes[1].set_title("IoU des masques dilatés — convention EUVIP")
    axes[2].set_xlabel("rappel tolérant"); axes[2].set_ylabel("précision tolérante")
    axes[2].set_title("Compromis, annoté par k")
    for axis in axes:
        axis.grid(alpha=0.3); axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(destination, dpi=120, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", type=Path, nargs="+", required=True,
                        help="un répertoire de masques par variante")
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.masks) != len(args.names):
        raise SystemExit("--masks et --names doivent avoir la même longueur")
    args.output.mkdir(parents=True, exist_ok=True)

    from joblib import Parallel, delayed
    from tqdm import tqdm

    summary: dict[str, dict] = {}
    for variant, mask_dir in zip(args.names, args.masks):
        stems = sorted(p.stem for p in mask_dir.glob("*.npy"))
        rows = Parallel(n_jobs=args.jobs)(
            delayed(process)(mask_dir, args.data_root, args.split, stem)
            for stem in tqdm(stems, desc=f"tolérance {variant}")
        )
        per_tolerance: dict[str, dict[str, float]] = {}
        for radius in TOLERANCES:
            keys = rows[0][1][radius].keys()
            per_tolerance[str(radius)] = {
                key: float(np.mean([r[1][radius][key] for r in rows]))
                for key in keys
            }
        summary[variant] = {"images": len(rows), "per_tolerance": per_tolerance}

        with (args.output / f"tolerant_{variant}.csv").open(
            "w", encoding="utf-8"
        ) as handle:
            columns = ["name"] + [
                f"k{r}_{k}" for r in TOLERANCES for k in
                ("iou_dilate_both", "iou_buffered", "precision", "recall", "f1")
            ]
            handle.write(",".join(columns) + "\n")
            for stem, scores in rows:
                values = [stem]
                for radius in TOLERANCES:
                    entry = scores[radius]
                    values += [
                        f"{entry[k]:.6f}" for k in
                        ("iou_dilate_both", "iou_buffered", "precision",
                         "recall", "f1")
                    ]
                handle.write(",".join(values) + "\n")

    _plot(summary, args.output / "tolerant_curves.png")

    (args.output / "tolerant_summary.json").write_text(
        json.dumps({"tolerances": list(TOLERANCES), "variants": summary},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for variant, payload in summary.items():
        print(f"\n=== {variant} ({payload['images']} images) ===")
        print(f"{'k':>3} {'IoU dilat.':>11} {'IoU tampon':>11} "
              f"{'P':>8} {'R':>8} {'F1':>8}")
        for radius in TOLERANCES:
            entry = payload["per_tolerance"][str(radius)]
            print(f"{radius:>3} {entry['iou_dilate_both']:>11.4f} "
                  f"{entry['iou_buffered']:>11.4f} {entry['precision']:>8.4f} "
                  f"{entry['recall']:>8.4f} {entry['f1']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
