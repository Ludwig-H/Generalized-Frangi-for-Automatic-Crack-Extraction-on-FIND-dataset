#!/usr/bin/env python3
"""Entraîne une variante de l'échelle d'ablations, à budget strictement égal.

Variantes, dans l'ordre où elles doivent être exécutées :

``baseline``
    LoRA q/v seule, perte CrackSAM d'origine. C'est l'ancre à budget égal :
    la baseline archivée valant `0,6238` a coûté 70 époques, on ne peut pas la
    comparer directement à des variantes entraînées plus court.
``cldice``
    ``baseline`` + ``soft-clDice``. **Isole la perte, sans aucune géométrie.**
    Barreau le plus important et le plus souvent omis : si la continuité seule
    apporte l'essentiel du gain, la géométrie est superflue.
``geo``
    ``cldice`` + adapter géométrique. Isole l'apport propre de la géométrie.
``geo_permuted``
    identique à ``geo``, mais l'évidence d'une image est appariée à une **autre**
    image. Capacité et budget identiques, alignement détruit : c'est le contrôle
    sans lequel ``geo`` n'est pas interprétable.
``geo_noise``
    identique à ``geo``, évidence remplacée par du bruit de même statistique.
    Contrôle de **capacité** : sépare l'apport de la géométrie de celui des
    paramètres ajoutés.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
CRACKSAM = ROOT.parent / "CrackSAM"
for path in (ROOT, CRACKSAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cracksam2.data import CrackSegmentationDataset  # noqa: E402
from cracksam2.losses import cracksam_loss, warmup_poly_lr  # noqa: E402
from cracksam2.metrics import segmentation_metrics  # noqa: E402
from cracksam2.model import (  # noqa: E402
    adapter_state_dict,
    build_cracksam2,
    load_adapter_state_dict,
)
from geolora.adapter import GeometryAdapter, GeoGuidedCrackSAM2  # noqa: E402
from geolora.losses import soft_cldice_loss, tolerant_loss  # noqa: E402

VARIANTS = (
    "baseline", "cldice", "geo", "geo_permuted", "geo_noise",
    # Variantes tolérantes : la perte cesse de pénaliser une erreur de
    # placement inférieure au rayon. Motivées par la mesure du §9 du rapport.
    "tol3", "tol5", "geo_tol3",
    # Contrôle apparié de geo_tol3 : capacité identique (744 049 params),
    # évidence d'une AUTRE image. Sans lui, un écart geo_tol3 - tol3
    # pourrait venir des paramètres ajoutés et non de la géométrie.
    "geo_tol3_permuted",
)
SEED = 3407
CLDICE_WEIGHT = 0.5
TOLERANT_WEIGHT = 0.5


class EvidenceDataset(Dataset):
    """Enveloppe le dataset CrackSAM et y adjoint l'évidence en cache."""

    def __init__(
        self,
        base: CrackSegmentationDataset,
        evidence_dir: Path | None,
        mode: str = "aligned",
        seed: int = SEED,
    ) -> None:
        self.base = base
        self.evidence_dir = evidence_dir
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        if mode == "permuted":
            count = len(base)
            self.permutation = self.rng.permutation(count)
            # Un point fixe rendrait le contrôle partiellement aligné.
            for index in range(count):
                if self.permutation[index] == index:
                    swap = (index + 1) % count
                    self.permutation[[index, swap]] = self.permutation[[swap, index]]

    def __len__(self) -> int:
        return len(self.base)

    def _load(self, index: int) -> np.ndarray:
        name = self.base.sample_names[index]
        path = self.evidence_dir / f"{Path(name).stem}.npy"
        return np.load(path).astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        if self.evidence_dir is None:
            return sample
        if self.mode == "permuted":
            planes = self._load(int(self.permutation[index]))
        elif self.mode == "noise":
            reference = self._load(index)
            # Bruit de même moyenne et même écart-type, canal par canal.
            mean = reference.mean(axis=(1, 2), keepdims=True)
            std = reference.std(axis=(1, 2), keepdims=True)
            planes = (
                mean
                + std
                * self.rng.standard_normal(reference.shape).astype(np.float32)
            ).astype(np.float32)
        else:
            planes = self._load(index)
        sample["evidence"] = torch.from_numpy(np.ascontiguousarray(planes))
        return sample


def build_model(args, device: str):
    checkpoint = torch.load(args.sam2_lora, map_location="cpu", weights_only=False)
    lora = checkpoint.get("lora", {})
    rank = int(lora.get("rank", 4))
    alpha = float(lora.get("alpha", rank))
    backbone, report = build_cracksam2(
        args.sam2_checkpoint,
        rank=rank,
        alpha=alpha,
        config=checkpoint["model_config"],
        device=device,
    )
    if args.init_from_baseline:
        load_adapter_state_dict(backbone, checkpoint["adapter"], strict=True)
    adapter = GeometryAdapter(width=args.adapter_width).to(device)
    model = GeoGuidedCrackSAM2(backbone, adapter).to(device)
    return model, report, rank, alpha


def evaluate(model, loader, device, use_evidence: bool, amp_dtype) -> dict[str, float]:
    model.eval()
    scores: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            evidence = (
                batch["evidence"].to(device, non_blocking=True)
                if use_evidence and "evidence" in batch
                else None
            )
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                logits = model(images, evidence=evidence)["logits"].float()
            probabilities = torch.sigmoid(logits).cpu().numpy()
            masks = batch["mask"].numpy()
            for index in range(probabilities.shape[0]):
                scores.append(
                    segmentation_metrics(probabilities[index, 0], masks[index, 0])
                )
    return {
        key: float(np.mean([s[key] for s in scores]))
        for key in ("precision", "recall", "dice", "iou")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--base-lr", type=float, default=4e-4)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--adapter-width", type=int, default=48)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument(
        "--init-from-baseline",
        action="store_true",
        help="repart de la LoRA archivée au lieu d'une LoRA neuve",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16
    args.output.mkdir(parents=True, exist_ok=True)

    uses_geometry = args.variant in {"geo", "geo_permuted", "geo_noise", "geo_tol3", "geo_tol3_permuted"}
    mode = {
        "geo": "aligned",
        "geo_permuted": "permuted",
        "geo_noise": "noise",
        "geo_tol3": "aligned",
        "geo_tol3_permuted": "permuted",
    }.get(args.variant, "aligned")
    # Les variantes tolérantes isolent la tolérance : elles n'ajoutent PAS
    # clDice, faute de quoi on mesurerait la somme des deux effets.
    tolerance = {
        "tol3": 3, "tol5": 5, "geo_tol3": 3, "geo_tol3_permuted": 3
    }.get(args.variant, 0)
    uses_cldice = args.variant in {"cldice", "geo", "geo_permuted", "geo_noise"}

    def make(split_list: Path, split: str, limit: int | None, evidence_split: str):
        base = CrackSegmentationDataset(
            args.data_root,
            list_file=split_list,
            split=split,
            image_size=448,
            prompt_size=256,
            noise_mode="original",
            augment=(split == "train"),
            prompt_cache_dir=None,
        )
        if limit:
            base.sample_names = base.sample_names[:limit]
        directory = (args.evidence_root / evidence_split) if uses_geometry else None
        return EvidenceDataset(base, directory, mode=mode)

    train_set = make(args.train_list, "train", args.max_train, "train")
    val_set = make(args.val_list, "train", args.max_val, "train")
    print(f"[{args.variant}] train={len(train_set)} val={len(val_set)} device={device}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(2, args.num_workers // 2),
        pin_memory=True,
    )

    model, report, rank, alpha = build_model(args, device)
    checkpoint_config = torch.load(
        args.sam2_lora, map_location="cpu", weights_only=False
    )["model_config"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not uses_geometry:
        # Sans géométrie, l'adapter ne doit pas exister dans le graphe : sinon
        # la comparaison de capacité serait faussée.
        for parameter in model.adapter.parameters():
            parameter.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]

    optimiser = torch.optim.AdamW(trainable, lr=args.base_lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    print(
        f"[{args.variant}] paramètres entraînables : {sum(p.numel() for p in trainable):,}"
        f" (LoRA {report.trainable_parameters:,})"
    )

    history: list[dict] = []
    best = {"iou": -1.0, "epoch": -1}
    step = 0
    start_epoch = 0
    started = time.time()

    # Reprise après préemption Spot : l'état complet est relu, optimiseur
    # compris, faute de quoi une reprise repartirait avec des moments nuls.
    latest_path = args.output / f"{args.variant}_latest.pt"
    if latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        load_adapter_state_dict(model.backbone, state["adapter"], strict=True)
        model.adapter.load_state_dict(state["geometry"])
        optimiser.load_state_dict(state["optimiser"])
        start_epoch = int(state["epoch"]) + 1
        step = int(state["step"])
        history = list(state.get("history", []))
        best = dict(state.get("best", best))
        print(f"[{args.variant}] reprise à l'époque {start_epoch} (step {step})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_set.base.set_epoch(epoch)
        running = 0.0
        for batch in train_loader:
            learning_rate = warmup_poly_lr(
                step,
                total_steps,
                base_lr=args.base_lr,
                warmup_steps=args.warmup_steps,
                power=6.0,
            )
            for group in optimiser.param_groups:
                group["lr"] = learning_rate

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            evidence = (
                batch["evidence"].to(device, non_blocking=True)
                if uses_geometry
                else None
            )
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                logits = model(images, evidence=evidence)["logits"]
            logits = logits.float()
            loss, _, _ = cracksam_loss(logits, masks, ce_weight=0.2)
            if uses_cldice:
                loss = loss + CLDICE_WEIGHT * soft_cldice_loss(
                    torch.sigmoid(logits), masks
                )
            if tolerance:
                loss = loss + TOLERANT_WEIGHT * tolerant_loss(
                    torch.sigmoid(logits), masks, tolerance
                )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimiser.step()
            running += float(loss.detach())
            step += 1

        metrics = evaluate(model, val_loader, device, uses_geometry, amp_dtype)
        record = {
            "epoch": epoch,
            "loss": running / steps_per_epoch,
            "lr": learning_rate,
            "gamma": model.adapter.activation(),
            "elapsed_s": round(time.time() - started, 1),
            **metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        # Sauvegarde inconditionnelle à chaque époque : en Spot, une
        # préemption peut survenir à tout instant et une époque perdue coûte
        # plus cher que l'écriture de 25 Mo.
        torch.save(
            {
                "variant": args.variant,
                "adapter": adapter_state_dict(model.backbone),
                "geometry": model.adapter.state_dict(),
                "optimiser": optimiser.state_dict(),
                "lora": {"rank": rank, "alpha": alpha},
                "model_config": checkpoint_config,
                "epoch": epoch,
                "step": step,
                "history": history,
                "best": best,
                "metrics": metrics,
                "seed": SEED,
                "adapter_width": args.adapter_width,
            },
            latest_path,
        )

        if metrics["iou"] > best["iou"]:
            best = {"epoch": epoch, **metrics}
            torch.save(
                {
                    "variant": args.variant,
                    "adapter": adapter_state_dict(model.backbone),
                    "geometry": model.adapter.state_dict(),
                    "lora": {"rank": rank, "alpha": alpha},
                    "model_config": checkpoint_config,
                    "epoch": epoch,
                    "metrics": metrics,
                    "seed": SEED,
                    "adapter_width": args.adapter_width,
                },
                args.output / f"{args.variant}_best.pt",
            )

    summary = {
        "variant": args.variant,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "base_lr": args.base_lr,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "uses_geometry": uses_geometry,
        "uses_cldice": uses_cldice,
        "tolerance_radius": tolerance,
        "evidence_mode": mode if uses_geometry else None,
        "best": best,
        "history": history,
        "seed": SEED,
        "wall_clock_s": round(time.time() - started, 1),
    }
    (args.output / f"{args.variant}_training.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
