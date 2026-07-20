#!/usr/bin/env python3
"""Train baseline or Frangi-guided CrackSAM 2 with q/v-only LoRA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import signal
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cracksam2.data import (
    PROMPT_CACHE_MANIFEST,
    CrackSegmentationDataset,
    sample_names_sha256,
)
from cracksam2.losses import cracksam_loss, set_optimizer_lr, warmup_poly_lr
from cracksam2.metrics import segmentation_metrics
from cracksam2.model import (
    SAM2_LARGE_CONFIG,
    build_cracksam2,
    checkpoint_payload,
    load_adapter_state_dict,
)


DEFAULT_LIST_ROOT = Path(__file__).parent / "protocol" / "cracksam_paper" / "lists"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument(
        "--train-list",
        type=Path,
        default=DEFAULT_LIST_ROOT / "lists_khanhha" / "train.txt",
    )
    parser.add_argument(
        "--val-list",
        type=Path,
        default=DEFAULT_LIST_ROOT / "lists_khanhha" / "val_vol.txt",
    )
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", default=SAM2_LARGE_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "frangi"), required=True)
    parser.add_argument("--train-prompt-cache", type=Path)
    parser.add_argument("--val-prompt-cache", type=Path)
    parser.add_argument("--rank", type=int, choices=(4, 8), default=4)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=4e-4)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--poly-power", type=float, default=6.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--checkpoint-every-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16", "none"), default="bfloat16"
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume from a checkpoint path, or latest.pt when passed without a path.",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.variant == "frangi" and (
        args.train_prompt_cache is None or args.val_prompt_cache is None
    ):
        raise ValueError("Frangi training requires both prompt cache directories")
    if args.epochs <= 0 or args.batch_size <= 0 or args.val_every <= 0:
        raise ValueError("epochs, batch-size, and val-every must be positive")
    if args.checkpoint_every_steps < 0:
        raise ValueError("checkpoint-every-steps cannot be negative")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _limit_dataset(dataset: CrackSegmentationDataset, limit: int | None) -> None:
    if limit is None:
        return
    if limit <= 0:
        raise ValueError("sample limits must be positive")
    dataset.sample_names = dataset.sample_names[:limit]


def _make_loader(
    dataset: CrackSegmentationDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_worker_init,
        generator=generator,
        persistent_workers=False,
    )


def _autocast_context(device: torch.device, amp_dtype: str):
    if amp_dtype == "none" or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _dataset_contract(dataset: CrackSegmentationDataset) -> dict[str, Any]:
    cache: dict[str, Any] | None = None
    if dataset.prompt_cache_dir is not None:
        manifest_path = dataset.prompt_cache_dir / PROMPT_CACHE_MANIFEST
        cache = {
            "path": str(dataset.prompt_cache_dir.expanduser().resolve()),
            "manifest": _file_identity(manifest_path),
        }
    return {
        "root": str(dataset.root_dir.expanduser().resolve()),
        "image_dir": str(dataset.image_dir.resolve()),
        "mask_dir": str(dataset.mask_dir.resolve()),
        "list": _file_identity(dataset.list_file),
        "selected_samples": len(dataset),
        "selected_sample_names_sha256": sample_names_sha256(dataset.sample_names),
        "image_size": list(dataset.image_size),
        "prompt_size": list(dataset.prompt_size),
        "noise": dataset.noise_mode,
        "prompt_cache": cache,
    }


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contract_mismatches(
    observed: object, expected: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if not isinstance(observed, dict):
        return {"run_contract": {"observed": observed, "expected": expected}}
    return {
        key: {"observed": observed.get(key), "expected": value}
        for key, value in expected.items()
        if observed.get(key) != value
    }


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        output.flush()
        os.fsync(output.fileno())


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
    ce_weight: float,
    threshold: float,
    use_frangi: bool,
) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in ("loss", "ce", "dice_loss", "precision", "recall", "dice", "iou")}
    samples = 0
    for batch in tqdm(loader, desc="validation", unit="batch", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        prompts = (
            batch["prompt"].to(device, non_blocking=True) if use_frangi else None
        )
        with _autocast_context(device, amp_dtype):
            output = model(images, mask_input=prompts, output_size=targets.shape[-2:])
            loss, ce, dice_loss = cracksam_loss(
                output["logits"], targets, ce_weight=ce_weight
            )
        probabilities = torch.sigmoid(output["logits"].float()).cpu()
        targets_cpu = targets.cpu()
        for prediction, target in zip(probabilities, targets_cpu):
            values = segmentation_metrics(prediction, target, threshold=threshold)
            for name, value in values.items():
                totals[name] += value
            samples += 1
        batch_samples = images.shape[0]
        totals["loss"] += float(loss) * batch_samples
        totals["ce"] += float(ce) * batch_samples
        totals["dice_loss"] += float(dice_loss) * batch_samples
    if samples == 0:
        raise RuntimeError("validation dataset is empty")
    return {name: value / samples for name, value in totals.items()}


def main() -> int:
    args = parse_args()
    _validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no CUDA device is available")
    args.output.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_frangi = args.variant == "frangi"
    train_dataset = CrackSegmentationDataset(
        args.train_root,
        list_file=args.train_list,
        split="train",
        image_size=448,
        prompt_size=256,
        augment=True,
        prompt_cache_dir=args.train_prompt_cache if use_frangi else None,
        augmentation_seed=args.seed,
    )
    val_dataset = CrackSegmentationDataset(
        args.val_root,
        list_file=args.val_list,
        split="val_vol",
        image_size=448,
        prompt_size=256,
        augment=False,
        prompt_cache_dir=args.val_prompt_cache if use_frangi else None,
    )
    _limit_dataset(train_dataset, args.max_train_samples)
    _limit_dataset(val_dataset, args.max_val_samples)

    # A loader is recreated per epoch with an epoch-specific seed, making
    # shuffled order reproducible after Spot-VM interruption and resume.
    length_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
    )
    steps_per_epoch = len(length_loader)
    total_steps = args.epochs * steps_per_epoch
    if args.warmup_steps >= total_steps:
        raise ValueError("warmup-steps must be smaller than total training steps")

    model, report = build_cracksam2(
        checkpoint=args.sam2_checkpoint,
        config=args.model_config,
        rank=args.rank,
        alpha=args.lora_alpha,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.base_lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.amp_dtype == "float16"
    )

    start_epoch = 0
    start_batch = 0
    global_step = 0
    best_dice = float("-inf")
    base_checkpoint = {
        "name": args.sam2_checkpoint.name,
        "size": args.sam2_checkpoint.stat().st_size,
        "sha256": _sha256(args.sam2_checkpoint),
    }
    run_contract = {
        "contract_version": 1,
        "variant": args.variant,
        "model_config": args.model_config,
        "base_checkpoint": base_checkpoint,
        "lora": {"rank": report.rank, "alpha": report.alpha},
        "optimizer": {
            "name": "AdamW",
            "base_lr": args.base_lr,
            "betas": [0.9, 0.999],
            "weight_decay": args.weight_decay,
        },
        "schedule": {
            "name": "linear_warmup_poly",
            "warmup_steps": args.warmup_steps,
            "poly_power": args.poly_power,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "amp_dtype": args.amp_dtype,
            "ce_weight": args.ce_weight,
            "dice_definition": "mean_background_foreground_global_v1",
            "threshold": args.threshold,
            "val_every": args.val_every,
            "augmentation": "cracksam_original_stateless_prompt_padding_v2",
        },
        "data": {
            "train": _dataset_contract(train_dataset),
            "validation": _dataset_contract(val_dataset),
        },
        "software": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "code": {
            path.name: _file_identity(path)
            for path in (
                Path(__file__),
                Path(__file__).parent / "cracksam2" / "data.py",
                Path(__file__).parent / "cracksam2" / "losses.py",
                Path(__file__).parent / "cracksam2" / "metrics.py",
                Path(__file__).parent / "cracksam2" / "model.py",
                Path(__file__).parent / "requirements-sam2.txt",
            )
        },
    }
    run_contract_sha256 = _json_sha256(run_contract)
    resume_path: Path | None = None
    resume_running = {name: 0.0 for name in ("loss", "ce", "dice_loss")}
    resume_seen_batches = 0
    if args.resume:
        resume_path = args.output / "latest.pt" if args.resume == "auto" else Path(args.resume)
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or state.get("format_version") != 1:
            raise RuntimeError(
                f"unsupported resume checkpoint format_version: "
                f"{state.get('format_version') if isinstance(state, dict) else None}"
            )
        mismatches = _contract_mismatches(state.get("run_contract"), run_contract)
        if state.get("run_contract_sha256") != run_contract_sha256:
            mismatches["run_contract_sha256"] = {
                "observed": state.get("run_contract_sha256"),
                "expected": run_contract_sha256,
            }
        if mismatches:
            raise RuntimeError(f"resume checkpoint run contract mismatch: {mismatches}")
        load_adapter_state_dict(model, state["adapter"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0))
        start_batch = int(state.get("next_batch", 0))
        global_step = int(state.get("global_step", 0))
        best_dice = float(state.get("best_dice", best_dice))
        observed_running = state.get("epoch_running", {})
        resume_running = {
            name: float(observed_running.get(name, 0.0))
            for name in resume_running
        }
        resume_seen_batches = int(state.get("epoch_seen_batches", 0))
        print(f"Resumed {resume_path} at epoch={start_epoch}, batch={start_batch}")

    config = vars(args).copy()
    config.update(
        {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "lora_report": report.__dict__,
            "run_contract": run_contract,
            "run_contract_sha256": run_contract_sha256,
        }
    )
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, default=str, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"LoRA trainable parameters: {report.trainable_parameters:,} / "
        f"{report.total_parameters:,}"
    )

    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Received signal {signum}; checkpointing after the current step.")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save_training_state(
        epoch: int,
        next_batch: int,
        name: str = "latest.pt",
        *,
        epoch_running: dict[str, float] | None = None,
        epoch_seen_batches: int = 0,
    ) -> None:
        payload = checkpoint_payload(
            model,
            report,
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            epoch=epoch,
            next_batch=next_batch,
            global_step=global_step,
            best_dice=best_dice,
            variant=args.variant,
            model_config=args.model_config,
            base_checkpoint=base_checkpoint,
            run_contract=run_contract,
            run_contract_sha256=run_contract_sha256,
            epoch_running=(
                {key: float(value) for key, value in epoch_running.items()}
                if epoch_running is not None
                else {key: 0.0 for key in ("loss", "ce", "dice_loss")}
            ),
            epoch_seen_batches=int(epoch_seen_batches),
        )
        _atomic_torch_save(payload, args.output / name)

    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        train_loader = _make_loader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            seed=args.seed + epoch,
        )
        model.train()
        if epoch == start_epoch:
            running = resume_running.copy()
            seen_batches = resume_seen_batches
        else:
            running = {"loss": 0.0, "ce": 0.0, "dice_loss": 0.0}
            seen_batches = 0
        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"epoch {epoch + 1}/{args.epochs}",
            unit="batch",
        )
        for batch_index, batch in progress:
            if epoch == start_epoch and batch_index < start_batch:
                continue
            learning_rate = warmup_poly_lr(
                global_step,
                total_steps,
                base_lr=args.base_lr,
                warmup_steps=args.warmup_steps,
                power=args.poly_power,
            )
            set_optimizer_lr(optimizer, learning_rate)
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            prompts = (
                batch["prompt"].to(device, non_blocking=True) if use_frangi else None
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, args.amp_dtype):
                output = model(images, mask_input=prompts, output_size=targets.shape[-2:])
                loss, ce, dice_loss = cracksam_loss(
                    output["logits"], targets, ce_weight=args.ce_weight
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at global step {global_step}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            global_step += 1
            seen_batches += 1
            running["loss"] += float(loss.detach())
            running["ce"] += float(ce.detach())
            running["dice_loss"] += float(dice_loss.detach())
            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}", lr=f"{learning_rate:.2e}"
            )

            next_epoch = epoch
            next_batch = batch_index + 1
            if next_batch >= len(train_loader):
                next_epoch, next_batch = epoch + 1, 0
            periodic = (
                args.checkpoint_every_steps > 0
                and global_step % args.checkpoint_every_steps == 0
            )
            if periodic or stop_requested:
                if next_epoch == epoch:
                    save_training_state(
                        next_epoch,
                        next_batch,
                        epoch_running=running,
                        epoch_seen_batches=seen_batches,
                    )
                else:
                    save_training_state(next_epoch, next_batch)
            if stop_requested:
                return 130

        start_batch = 0
        if seen_batches == 0:
            raise RuntimeError("no training batches were processed")
        train_row = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "loss": running["loss"] / seen_batches,
            "ce": running["ce"] / seen_batches,
            "dice_loss": running["dice_loss"] / seen_batches,
            "lr": optimizer.param_groups[0]["lr"],
        }
        _append_csv(args.output / "train.csv", train_row)

        should_validate = (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        if should_validate:
            val_loader = _make_loader(
                val_dataset,
                batch_size=1,
                num_workers=args.num_workers,
                shuffle=False,
                seed=args.seed,
            )
            values = validate(
                model,
                val_loader,
                device,
                args.amp_dtype,
                args.ce_weight,
                args.threshold,
                use_frangi,
            )
            val_row = {"epoch": epoch + 1, "global_step": global_step, **values}
            _append_csv(args.output / "validation.csv", val_row)
            print("validation", json.dumps(val_row, sort_keys=True))
            if values["dice"] > best_dice:
                best_dice = values["dice"]
                save_training_state(epoch + 1, 0, name="best.pt")

        save_training_state(epoch + 1, 0)

    print(f"Training complete; best validation Dice={best_dice:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
