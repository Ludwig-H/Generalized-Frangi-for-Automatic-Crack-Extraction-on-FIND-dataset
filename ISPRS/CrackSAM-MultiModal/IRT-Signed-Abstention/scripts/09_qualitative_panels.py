#!/usr/bin/env python3
"""Planches qualitatives : ce que la thermique change, image par image.

Une planche par échantillon, six vignettes :

``RGB`` · ``thermique décodée`` · ``similarité Frangi`` · puis les prédictions de
la baseline gelée, du bras sans thermique (A1) et du bras thermique (A2 ou A7),
en code couleur **vert** vrai positif · **rouge** faux positif · **bleu** manqué.

Les échantillons sont passés en argument : la sélection se fait sur les CSV par
image, pas ici, pour que le choix des exemples soit traçable et non arbitraire.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_residual.cache import open_cache  # noqa: E402
from thermal_residual.config import load_arm_config  # noqa: E402
from thermal_residual.constants import DECISION_THRESHOLD, PRIMARY_TOLERANCE  # noqa: E402
from thermal_residual.data import load_mask  # noqa: E402
from thermal_residual.manifest import read_manifest  # noqa: E402
from thermal_residual.metrics import tolerant_scores  # noqa: E402
from thermal_residual.model import build_adapter  # noqa: E402
from thermal_residual.thermal_decode import load_rgb  # noqa: E402
from thermal_residual.training import load_checkpoint  # noqa: E402


def overlay(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Vert = vrai positif, rouge = faux positif, bleu = manqué."""

    canvas = np.ones((*prediction.shape, 3), dtype=np.float32)
    canvas[truth & prediction] = (0.13, 0.62, 0.31)
    canvas[~truth & prediction] = (0.86, 0.21, 0.21)
    canvas[truth & ~prediction] = (0.16, 0.42, 0.83)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline-cache", required=True, type=Path)
    parser.add_argument("--thermal-cache", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path, help="dossier de 06_run_ablations.py")
    parser.add_argument("--configs", required=True, type=Path, help="dossier configs/")
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--left", default="rgb_recalibration", help="bras de gauche (sans thermique)")
    parser.add_argument("--right", default="frangi_signed_abstention", help="bras de droite (thermique)")
    parser.add_argument("--left-config", default="irt_rgb_recalibration.yaml")
    parser.add_argument("--right-config", default="irt_signed_abstention_v1.yaml")
    # Les titres des vignettes doivent nommer ce qui est comparé. Sans cela une
    # planche « permuté contre aligné » s'affiche « sans thermique contre avec
    # thermique », ce qui est faux et se lit sans qu'on s'en aperçoive.
    parser.add_argument("--left-title", default="sans thermique")
    parser.add_argument("--right-title", default="avec thermique")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def predict(run_dir: Path, config_path: Path, logits: np.ndarray, evidence: np.ndarray):
    import torch

    config = load_arm_config(config_path)
    payload = load_checkpoint(run_dir / "best.pt")
    model = build_adapter(config.arm.model).eval()
    model.load_state_dict(payload["model_state_dict"])
    with torch.no_grad():
        out = model(
            torch.from_numpy(logits[None].astype(np.float32)),
            torch.from_numpy(evidence[None].astype(np.float32)),
            torch.ones(1, dtype=torch.bool),
        )
    return torch.sigmoid(out["logits"])[0, 0].numpy(), out["residual_logits"][0, 0].numpy()


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    samples = {s.sample_id: s for s in read_manifest(args.manifest)}
    baseline = open_cache(args.baseline_cache)
    thermal = open_cache(args.thermal_cache)
    args.output.mkdir(parents=True, exist_ok=True)

    for sample_id in args.samples:
        sample = samples[sample_id]
        logits = np.asarray(baseline.entry(sample_id)["baseline_logits"], dtype=np.float32)
        entry = thermal.entry(sample_id)
        evidence = np.stack(
            [entry[c] for c in ("similarity_dark", "similarity_bright", "similarity_max", "support_union")]
        ).astype(np.float32)
        truth = load_mask(sample.mask_path) > 0.5
        rgb = load_rgb(sample.rgb_path)

        base_pred = 1.0 / (1.0 + np.exp(-logits[0])) > DECISION_THRESHOLD
        left, _ = predict(args.runs / args.left / f"seed_{args.seed}", args.configs / args.left_config, logits, evidence)
        right, residual = predict(args.runs / args.right / f"seed_{args.seed}", args.configs / args.right_config, logits, evidence)
        left_pred = left > DECISION_THRESHOLD
        right_pred = right > DECISION_THRESHOLD

        def score(mask):
            return tolerant_scores(mask, truth, PRIMARY_TOLERANCE)["iou_buffered"]

        panels = [
            (rgb, None, "visible (ce que SAM voit)"),
            (entry["thermal_decoded"], "inferno", "thermique décodée"),
            (entry["similarity_max"], "magma", "similarité Frangi thermique"),
            (overlay(base_pred, truth), None, f"baseline gelée — {score(base_pred):.3f}"),
            (overlay(left_pred, truth), None, f"{args.left_title} — {score(left_pred):.3f}"),
            (overlay(right_pred, truth), None, f"{args.right_title} — {score(right_pred):.3f}"),
        ]
        figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
        for axis, (image, cmap, title) in zip(axes.ravel(), panels):
            axis.imshow(image, cmap=cmap)
            axis.set_title(title, fontsize=11)
            axis.set_xticks([]); axis.set_yticks([])
        figure.suptitle(
            f"{sample_id} — vert : trouvé · rouge : faux positif · bleu : manqué "
            f"(IoU tolérante {PRIMARY_TOLERANCE} px, |Δz| max {np.abs(residual).max():.2f})",
            fontsize=12,
        )
        figure.tight_layout()
        destination = args.output / f"panel_{sample_id}.jpg"
        figure.savefig(destination, dpi=120, pil_kwargs={"quality": 90})
        plt.close(figure)
        print(f"  écrit {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
