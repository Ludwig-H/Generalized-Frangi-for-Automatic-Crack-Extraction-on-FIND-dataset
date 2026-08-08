#!/usr/bin/env python3
"""Évalue une variante sur le test officiel Khánh Hà et compare à la baseline.

Produit, par image : précision, rappel, Dice, IoU, plus deux mesures
topologiques que l'IoU ne capture pas — le nombre de composantes connexes
prédites et la longueur de squelette couverte. C'est par la topologie que se
juge une perte de continuité, pas par le recouvrement seul.

Pour les variantes géométriques, la condition ``no_evidence`` réévalue **le même
checkpoint** sans évidence. C'est le test de nécessité d'entrée : avec un
adapter correctement conçu, il doit restituer la voie sans géométrie.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
CRACKSAM = ROOT.parent / "CrackSAM"
for path in (ROOT, CRACKSAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
sys.path.insert(0, str(ROOT / "scripts"))

from cracksam2.data import CrackSegmentationDataset  # noqa: E402
from cracksam2.metrics import segmentation_metrics  # noqa: E402
from cracksam2.model import build_cracksam2, load_adapter_state_dict  # noqa: E402
from geolora.adapter import GeometryAdapter, GeoGuidedCrackSAM2  # noqa: E402

from importlib import import_module  # noqa: E402

_train_cli = import_module("02_train")
EvidenceDataset = _train_cli.EvidenceDataset

BOOTSTRAP = 2000
SEED = 20260808


def topology(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    predicted = prediction > 0.5
    reference = truth > 0.5
    _, predicted_components = ndi.label(predicted, structure=np.ones((3, 3), int))
    _, reference_components = ndi.label(reference, structure=np.ones((3, 3), int))
    if reference.any():
        reference_skeleton = skeletonize(reference)
        covered = float((reference_skeleton & predicted).sum()) / float(
            max(reference_skeleton.sum(), 1)
        )
    else:
        covered = float("nan")
    return {
        "components_pred": float(predicted_components),
        "components_true": float(reference_components),
        "skeleton_covered": covered,
    }


def grouped_bootstrap(values: np.ndarray, rounds: int, seed: int):
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(rounds)
    for index in range(rounds):
        means[index] = values[rng.integers(0, values.size, values.size)].mean()
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--evidence-split", default="test")
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test_vol")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-masks", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = str(checkpoint.get("variant", args.checkpoint.stem))
    lora = checkpoint.get("lora", {})
    rank, alpha = int(lora.get("rank", 4)), float(lora.get("alpha", 4.0))

    config = checkpoint.get("model_config", "configs/sam2/sam2_hiera_l.yaml")
    backbone, _ = build_cracksam2(
        args.sam2_checkpoint, rank=rank, alpha=alpha, config=config, device=device
    )
    load_adapter_state_dict(backbone, checkpoint["adapter"], strict=True)
    adapter = GeometryAdapter(width=int(checkpoint.get("adapter_width", 48)))
    if "geometry" in checkpoint:
        adapter.load_state_dict(checkpoint["geometry"])
    model = GeoGuidedCrackSAM2(backbone, adapter.to(device)).to(device)
    model.eval()

    base = CrackSegmentationDataset(
        args.data_root,
        list_file=args.list_file,
        split=args.split,
        image_size=448,
        prompt_size=256,
        noise_mode="original",
        augment=False,
        prompt_cache_dir=None,
    )
    if args.max_samples:
        base.sample_names = base.sample_names[: args.max_samples]
    dataset = EvidenceDataset(
        base, args.evidence_root / args.evidence_split, mode="aligned"
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    conditions = ["with_evidence", "no_evidence"]
    records: dict[str, list[dict]] = {c: [] for c in conditions}
    names: list[str] = []
    if args.save_masks:
        args.save_masks.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"éval {variant}"):
            images = batch["image"].to(device, non_blocking=True)
            evidence = batch["evidence"].to(device, non_blocking=True)
            masks = batch["mask"].numpy()
            names.extend(str(n) for n in batch["case_name"])
            for condition in conditions:
                supplied = evidence if condition == "with_evidence" else None
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = model(images, evidence=supplied)["logits"].float()
                probabilities = torch.sigmoid(logits).cpu().numpy()
                for index in range(probabilities.shape[0]):
                    prediction = probabilities[index, 0]
                    truth = masks[index, 0]
                    entry = segmentation_metrics(prediction, truth)
                    entry.update(topology(prediction, truth))
                    records[condition].append(entry)
                    if args.save_masks and condition == "with_evidence":
                        stem = Path(str(batch["case_name"][index])).stem
                        np.save(
                            args.save_masks / f"{stem}.npy",
                            (prediction > 0.5).astype(np.uint8),
                        )

    summary: dict[str, object] = {
        "variant": variant,
        "checkpoint": str(args.checkpoint),
        "images": len(records["with_evidence"]),
        "gamma": float(model.adapter.activation()),
    }
    for condition in conditions:
        rows = records[condition]
        summary[condition] = {
            key: float(np.nanmean([r[key] for r in rows]))
            for key in (
                "precision",
                "recall",
                "dice",
                "iou",
                "components_pred",
                "components_true",
                "skeleton_covered",
            )
        }
    identical = np.allclose(
        [r["iou"] for r in records["with_evidence"]],
        [r["iou"] for r in records["no_evidence"]],
    )
    summary["evidence_is_necessary"] = not bool(identical)

    (args.output / f"eval_{variant}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    columns = ["name"] + [
        f"{c}_{k}" for c in conditions for k in ("iou", "dice", "precision", "recall",
                                                 "components_pred", "skeleton_covered")
    ]
    with (args.output / f"per_image_{variant}.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(columns) + "\n")
        for position, name in enumerate(names):
            values = [name]
            for condition in conditions:
                row = records[condition][position]
                values += [
                    f"{row[k]:.6f}"
                    for k in ("iou", "dice", "precision", "recall",
                              "components_pred", "skeleton_covered")
                ]
            fh.write(",".join(values) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
