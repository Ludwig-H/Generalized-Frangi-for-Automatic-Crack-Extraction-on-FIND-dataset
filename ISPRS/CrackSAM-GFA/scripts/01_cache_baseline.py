#!/usr/bin/env python3
"""Porte n° 0 : reproduire la baseline gelée et mettre ``z0`` en cache.

La baseline est ``baseline_r4/best.pt`` — SAM 2 Hiera-L + LoRA q/v rang 4,
alpha 4, 70 époques sur Khánh Hà, seed 3407, 448x448.

Le prétraitement reprend **exactement** ``CrackSegmentationDataset`` avec les
mêmes paramètres que l'évaluation archivée, sans quoi l'IoU ne serait pas
comparable. Tant que la reproduction n'est pas certifiée, rien d'autre ne doit
être exécuté : c'est la porte n° 0 du pré-enregistrement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

CRACKSAM = Path(__file__).resolve().parents[2] / "CrackSAM"
if str(CRACKSAM) not in sys.path:
    sys.path.insert(0, str(CRACKSAM))

from cracksam2.data import CrackSegmentationDataset  # noqa: E402
from cracksam2.metrics import segmentation_metrics  # noqa: E402
from cracksam2.model import build_cracksam2, load_adapter_state_dict  # noqa: E402

#: Valeurs archivées de ``baseline_r4/best.pt`` sur ``khanhha_original``.
REFERENCE = {
    "khanhha_original": {
        "iou": 0.623804,
        "dice": 0.745320,
        "precision": 0.749322,
        "recall": 0.771146,
    },
    "khanhha_noisy1": {"iou": 0.567812},
    "khanhha_noisy2": {"iou": 0.513344},
}
TOLERANCE = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test_vol")
    parser.add_argument(
        "--noise", default="original", choices=("original", "noisy1", "noisy2")
    )
    parser.add_argument("--condition-name", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="écrit z0 en float16 (400 ko par image)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    condition = args.condition_name or f"khanhha_{args.noise}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    logits_dir = args.output / "z0"
    if args.save_logits:
        logits_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lora = checkpoint.get("lora", {})
    rank = int(lora.get("rank", 4))
    alpha = float(lora.get("alpha", rank))
    model, report = build_cracksam2(
        args.sam2_checkpoint,
        rank=rank,
        alpha=alpha,
        config=checkpoint["model_config"],
        device=device,
    )
    load_adapter_state_dict(model, checkpoint["adapter"], strict=True)
    model.requires_grad_(False)
    model.eval()
    print(
        f"[{condition}] baseline chargée : variante={checkpoint.get('variant')} "
        f"rang={rank} alpha={alpha} params LoRA={report.trainable_parameters:,}"
    )

    dataset = CrackSegmentationDataset(
        args.data_root,
        list_file=args.list_file,
        split=args.split,
        image_size=448,
        prompt_size=256,
        noise_mode=args.noise,
        augment=False,
        prompt_cache_dir=None,
    )
    if args.max_samples:
        dataset.sample_names = dataset.sample_names[: args.max_samples]
    print(f"[{condition}] {len(dataset)} échantillons")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    per_image: list[dict[str, float | str]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].numpy()
            names = batch["case_name"]
            features = model.encode_images(images)
            # mask_input=None : voie SAM 2 officielle sans prompt. C'est la
            # définition exacte de z0, et `None` n'est PAS un tenseur nul.
            decoded = model.decode_features(features, mask_input=None)
            logits = decoded["logits"].float().cpu().numpy()
            if logits.ndim == 4:
                logits = logits[:, 0]
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            for index, name in enumerate(names):
                truth = masks[index]
                if truth.ndim == 3:
                    truth = truth[0]
                scores = segmentation_metrics(probabilities[index], truth)
                per_image.append({"name": str(name), **scores})
                if args.save_logits:
                    destination = logits_dir / f"{Path(str(name)).stem}.npy"
                    np.save(destination, logits[index].astype(np.float16))

    summary = {
        key: float(np.mean([row[key] for row in per_image]))
        for key in ("precision", "recall", "dice", "iou")
    }
    summary["samples"] = len(per_image)
    summary["condition"] = condition

    reference = REFERENCE.get(condition)
    verdict = "NON VÉRIFIÉ"
    if reference is not None:
        delta = abs(summary["iou"] - reference["iou"])
        summary["reference_iou"] = reference["iou"]
        summary["delta_iou"] = summary["iou"] - reference["iou"]
        verdict = "REPRODUITE" if delta <= TOLERANCE else "ÉCART"
    summary["gate0"] = verdict

    (args.output / f"baseline_{condition}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output / f"per_image_{condition}.csv").open("w", encoding="utf-8") as fh:
        fh.write("name,precision,recall,dice,iou\n")
        for row in per_image:
            fh.write(
                f"{row['name']},{row['precision']:.6f},{row['recall']:.6f},"
                f"{row['dice']:.6f},{row['iou']:.6f}\n"
            )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if verdict == "ÉCART":
        print("PORTE 0 ÉCHOUÉE : la chaîne ne reproduit pas la baseline archivée.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
