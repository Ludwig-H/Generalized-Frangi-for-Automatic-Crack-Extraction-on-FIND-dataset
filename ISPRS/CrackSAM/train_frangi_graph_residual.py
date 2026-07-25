#!/usr/bin/env python3
"""Train the small FrangiGraph residual above a frozen SAM 2 baseline.

This command only uses the Khanhha training list and its group-safe five-fold
assignment.  The held-out fold is validation/OOF data; historical test sets are
not accepted by this training interface.
"""

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
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cracksam2.data import read_sample_list, sample_names_sha256
from cracksam2.graph_cache import sha256_file, write_json_atomic
from cracksam2.graph_types import FRANGI_RASTER_CHANNELS
from cracksam2.losses import (
    cracksam_loss_per_image,
    residual_training_loss,
    set_optimizer_lr,
    warmup_poly_lr,
)
from cracksam2.metrics import segmentation_metrics
from cracksam2.model import (
    SAM2_LARGE_CONFIG,
    build_cracksam2,
    load_adapter_state_dict,
)
from cracksam2.oof import (
    RESIDUAL_RUN_CONTRACT_VERSION,
    strict_oof_training_contract,
    validate_oof_run_contract,
)
from cracksam2.residual import (
    LEGACY_ADAPTER_MODE,
    RESIDUAL_ADAPTER_MODES,
    VERIFIED_ADAPTER_MODE,
    FrangiGraphResidual,
)
from cracksam2.residual_data import (
    FrangiGraphRasterDataset,
    FrangiRasterNormalization,
    GraphCacheIndex,
    load_graph_cache_index,
    load_group_safe_fold,
    load_or_fit_frangi_raster_normalization,
)

CRACKSAM_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_LIST = (
    CRACKSAM_ROOT
    / "protocol"
    / "cracksam_paper"
    / "lists"
    / "lists_khanhha"
    / "train.txt"
)
DEFAULT_FOLD_CSV = (
    CRACKSAM_ROOT / "protocol" / "frangigraph_v1" / "train_group_folds.csv"
)
DEFAULT_PROTOCOL_MANIFEST = (
    CRACKSAM_ROOT / "protocol" / "frangigraph_v1" / "manifest.json"
)
CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_TRAINING_STATES = frozenset(("incomplete", "complete"))
SELECTOR_FUSION_GRID_SOURCE = "SAM2ImageFeatures.high_resolution_features[0]"
SELECTOR_FEATURE_CELL_UNIT = "hiera_high_resolution_feature_cells"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--fold-csv", type=Path, default=DEFAULT_FOLD_CSV)
    parser.add_argument(
        "--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST
    )
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument(
        "--exclude-training-fold",
        dest="exclude_training_folds",
        type=int,
        choices=range(5),
        action="append",
        default=None,
        help=(
            "Optional assertion of the automatic strict OOF exclusion. For "
            "folds 0-3 the only accepted value is 4; for fold 4 this option "
            "must be omitted."
        ),
    )
    parser.add_argument("--graph-cache", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument(
        "--adapter-mode",
        choices=RESIDUAL_ADAPTER_MODES,
        default=VERIFIED_ADAPTER_MODE,
        help="Versioned residual architecture; legacy mode reproduces old checkpoints.",
    )
    parser.add_argument(
        "--profile-radii-feature-cells",
        "--profile-radii",
        dest="profile_radii",
        type=float,
        nargs="+",
        default=[1.5, 3.0],
        metavar="RADIUS",
        help=(
            "Sampling radii in cells of the first SAM 2/Hiera high-resolution "
            "feature grid. --profile-radii is a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--evidence-dilation-feature-cells",
        "--evidence-dilation",
        dest="evidence_dilation",
        type=int,
        default=2,
        metavar="CELLS",
        help=(
            "Correction-envelope dilation radius in cells of the first SAM "
            "2/Hiera high-resolution feature grid. --evidence-dilation is a "
            "backward-compatible alias."
        ),
    )
    parser.add_argument("--evidence-threshold", type=float, default=0.5)
    parser.add_argument("--evidence-loss-weight", type=float, default=0.25)
    parser.add_argument("--evidence-target-tolerance", type=int, default=3)
    parser.add_argument(
        "--raster-condition",
        choices=("correct", "no_evidence"),
        default="correct",
        help=(
            "Use the matching Frangi raster, or its canonical absent-evidence "
            "encoding for a legacy equal-capacity control. The verified local "
            "adapter instead uses a same-checkpoint evaluation override."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=4e-4)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--poly-power", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--topology-weight", type=float, default=0.1)
    parser.add_argument("--safety-weight", type=float, default=1.0)
    parser.add_argument("--safety-margin", type=float, default=0.0)
    parser.add_argument("--skeleton-iterations", type=int, default=10)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val-every", type=int, default=1)
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
        help="Resume from a path, or output/latest.pt when passed without a path.",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument(
        "--skip-cache-file-verification",
        action="store_true",
        help="Skip the expensive initial SHA/load audit (per-sample loads stay validated).",
    )
    parser.add_argument(
        "--skip-source-file-verification",
        action="store_true",
        help="Skip checking current image/mask SHA against cache provenance.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "hidden_channels", "val_every"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.checkpoint_every_steps < 0:
        raise ValueError("worker and checkpoint intervals cannot be negative")
    if args.evidence_dilation < 0 or args.evidence_target_tolerance < 0:
        raise ValueError("evidence dilation and target tolerance cannot be negative")
    if not args.profile_radii or any(
        not np.isfinite(value) or value <= 0 for value in args.profile_radii
    ):
        raise ValueError(
            "--profile-radii-feature-cells/--profile-radii must contain "
            "positive values"
        )
    if not 0.0 < args.evidence_threshold < 1.0:
        raise ValueError("--evidence-threshold must lie in (0, 1)")
    for name in (
        "base_lr",
        "poly_power",
        "gradient_clip",
        "safety_weight",
        "topology_weight",
        "evidence_loss_weight",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if args.base_lr == 0.0 or args.poly_power == 0.0 or args.gradient_clip == 0.0:
        raise ValueError(
            "learning rate, polynomial power and gradient clip must be > 0"
        )
    if args.safety_margin < 0.0:
        raise ValueError("--safety-margin cannot be negative")
    if args.adapter_mode == LEGACY_ADAPTER_MODE and args.evidence_loss_weight != 0.0:
        raise ValueError(
            "legacy_raster_v1 has no evidence verifier; pass --evidence-loss-weight 0"
        )
    if (
        args.adapter_mode == VERIFIED_ADAPTER_MODE
        and args.raster_condition == "no_evidence"
    ):
        raise ValueError(
            "verified_local_v1 cannot be trained with no_evidence because its hard "
            "support constraint makes the correction identically zero; evaluate a "
            "correct checkpoint with the no_evidence input-ablation override instead"
        )
    if args.skeleton_iterations <= 0:
        raise ValueError("--skeleton-iterations must be positive")
    if not 0.0 <= args.ce_weight <= 1.0:
        raise ValueError("--ce-weight must lie in [0, 1]")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must lie in (0, 1)")
    for name in ("max_train_samples", "max_validation_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    expected_oof = strict_oof_training_contract(args.fold)
    expected_exclusions = list(
        expected_oof["additional_excluded_training_folds"]
    )
    excluded_folds = tuple(args.exclude_training_folds or ())
    duplicate_folds = sorted(
        {fold for fold in excluded_folds if excluded_folds.count(fold) > 1}
    )
    if duplicate_folds:
        raise ValueError(
            "--exclude-training-fold cannot be repeated for the same fold: "
            f"{duplicate_folds}"
        )
    if excluded_folds and sorted(excluded_folds) != expected_exclusions:
        raise ValueError(
            "--exclude-training-fold cannot override the strict OOF policy; "
            f"fold {args.fold} requires {expected_exclusions}"
        )
    args.exclude_training_folds = expected_exclusions


def _prepare_output_directory(output: Path, *, resume: bool) -> None:
    """Refuse ambiguous output reuse before any training-side work starts."""
    destination = output.expanduser()
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"--output is not a directory: {destination}")
    if destination.is_dir():
        non_empty = next(destination.iterdir(), None) is not None
        if non_empty and not resume:
            raise FileExistsError(
                f"output directory is not empty: {destination}; pass --resume "
                "only to continue its exact run contract"
            )
        if resume and not (destination / "config.json").is_file():
            raise FileNotFoundError(
                f"--resume requires the existing run config: {destination / 'config.json'}"
            )
        if resume and not non_empty:
            raise FileNotFoundError(
                f"--resume requires a non-empty output directory: {destination}"
            )
    elif resume:
        raise FileNotFoundError(
            f"--resume requires an existing output directory: {destination}"
        )
    else:
        destination.mkdir(parents=True, exist_ok=False)


def _publish_or_validate_config(
    path: Path,
    config: Mapping[str, object],
    *,
    resume: bool,
) -> None:
    """Publish a new config, or compare a resume config without overwriting it."""
    if not resume:
        write_json_atomic(path, config)
        return
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid existing resume config: {path}") from exc
    if not isinstance(observed, dict):
        raise ValueError(f"existing resume config must be a JSON object: {path}")
    expected = dict(config)
    if observed != expected:
        differing_keys = sorted(
            key
            for key in set(observed).union(expected)
            if observed.get(key) != expected.get(key)
        )
        raise RuntimeError(
            "resume config is incompatible with the requested run; "
            f"differing top-level keys={differing_keys}"
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _make_loader(
    dataset: FrangiGraphRasterDataset,
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


def _autocast(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path, *, include_path: bool = True) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    identity: dict[str, object] = {
        "name": resolved.name,
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if include_path:
        identity["path"] = str(resolved)
    return identity


def _append_csv(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        output.flush()
        os.fsync(output.fileno())


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as output:
            torch.save(value, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _local_adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.adapter.state_dict().items()  # type: ignore[attr-defined]
    }


def _load_local_adapter_state(
    model: torch.nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    expected = set(model.adapter.state_dict())  # type: ignore[attr-defined]
    observed = set(state)
    if observed != expected:
        raise RuntimeError(
            "residual adapter checkpoint mismatch; "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )
    model.adapter.load_state_dict(dict(state), strict=True)  # type: ignore[attr-defined]


def residual_checkpoint_payload(
    model: torch.nn.Module,
    *,
    high_resolution_channels: tuple[int, ...],
    hidden_channels: int,
    baseline_checkpoint_sha256: str,
    graph_cache: GraphCacheIndex,
    normalization: FrangiRasterNormalization,
    run_contract: Mapping[str, object],
    run_contract_sha256: str,
    optimizer_state: Mapping[str, object],
    scaler_state: Mapping[str, object],
    epoch: int,
    next_batch: int,
    global_step: int,
    best_validation_iou: float,
    epoch_running: Mapping[str, float] | None = None,
    epoch_seen_batches: int = 0,
    training_state: str = "incomplete",
) -> dict[str, object]:
    """Build the stable evaluator/trainer checkpoint contract."""
    validate_oof_run_contract(run_contract)
    if _json_sha256(run_contract) != run_contract_sha256:
        raise ValueError("run_contract_sha256 does not match run_contract")
    channels = list(FRANGI_RASTER_CHANNELS)
    if len(channels) != 7:
        raise AssertionError("FrangiGraph-Residual requires exactly seven channels")
    if training_state not in CHECKPOINT_TRAINING_STATES:
        raise ValueError(
            "training_state must be either 'incomplete' or 'complete'"
        )
    adapter_mode = str(getattr(model, "adapter_mode", LEGACY_ADAPTER_MODE))
    architecture: dict[str, object] = {
        "raster_channels": 7,
        "high_resolution_channels": list(high_resolution_channels),
        "hidden_channels": int(hidden_channels),
        "adapter_mode": adapter_mode,
    }
    if adapter_mode == VERIFIED_ADAPTER_MODE:
        profile_radii = list(model.adapter.profile_radii)  # type: ignore[attr-defined]
        evidence_dilation = int(model.adapter.evidence_dilation)  # type: ignore[attr-defined]
        architecture.update(
            {
                # Historical keys remain authoritative for old evaluators.
                "profile_radii": profile_radii,
                "evidence_dilation": evidence_dilation,
                "evidence_threshold": float(model.adapter.evidence_threshold),  # type: ignore[attr-defined]
                # Explicit-unit aliases make the coordinate system unambiguous.
                "profile_radii_feature_cells": profile_radii,
                "evidence_dilation_feature_cells": evidence_dilation,
            }
        )
        contract_architecture = run_contract.get("residual")
        if isinstance(contract_architecture, Mapping):
            for key, expected in (
                ("profile_radii_feature_cells", profile_radii),
                ("evidence_dilation_feature_cells", evidence_dilation),
            ):
                observed = contract_architecture.get(key)
                if observed is not None and observed != expected:
                    raise ValueError(
                        f"run-contract {key} differs from the verified adapter"
                    )
            selector_grid = contract_architecture.get("selector_grid")
            if selector_grid is not None:
                if not isinstance(selector_grid, Mapping):
                    raise ValueError("run-contract selector_grid must be a mapping")
                architecture["selector_grid"] = dict(selector_grid)
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "training_state": training_state,
        # Keys are local to model.adapter (no leading ``adapter.`` prefix).
        "residual_adapter": _local_adapter_state(model),
        "residual": architecture,
        "baseline_adapter": {"sha256": baseline_checkpoint_sha256},
        "graph_cache": {
            "extractor_sha256": graph_cache.manifest["extractor_sha256"],
            "frangi": graph_cache.manifest["frangi"],
            "channels": channels,
            "manifest_sha256": graph_cache.manifest_sha256,
        },
        "raster_preprocessing": normalization.preprocessing_contract(),
        "run_contract": dict(run_contract),
        "run_contract_sha256": run_contract_sha256,
        "optimizer": dict(optimizer_state),
        "scaler": dict(scaler_state),
        "epoch": int(epoch),
        "next_batch": int(next_batch),
        "global_step": int(global_step),
        "best_validation_iou": float(best_validation_iou),
        "epoch_running": dict(epoch_running or {}),
        "epoch_seen_batches": int(epoch_seen_batches),
    }


def restore_residual_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    expected_run_contract: Mapping[str, object],
    expected_run_contract_sha256: str,
) -> dict[str, object]:
    """Strictly restore a resumable checkpoint after contract comparison."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise RuntimeError("unsupported residual checkpoint format")
    if state.get("training_state") not in CHECKPOINT_TRAINING_STATES:
        raise RuntimeError("resume checkpoint has no valid training_state")
    if state.get("run_contract") != dict(expected_run_contract):
        raise RuntimeError("resume checkpoint run contract mismatch")
    if state.get("run_contract_sha256") != expected_run_contract_sha256:
        raise RuntimeError("resume checkpoint run-contract SHA mismatch")
    adapter = state.get("residual_adapter")
    if not isinstance(adapter, dict):
        raise RuntimeError("resume checkpoint has no residual_adapter")
    _load_local_adapter_state(model, adapter)
    optimizer.load_state_dict(state["optimizer"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state


def _load_frozen_baseline(
    foundation_checkpoint: Path,
    baseline_checkpoint: Path,
    *,
    requested_config: str | None,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object], str]:
    checkpoint = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("baseline adapter checkpoint has an unsupported format")
    if checkpoint.get("variant") != "baseline":
        raise ValueError("--baseline-checkpoint must be the prompt-free baseline")
    lora = checkpoint.get("lora")
    if not isinstance(lora, dict):
        raise ValueError("baseline checkpoint has no LoRA metadata")
    rank = int(lora.get("rank", 0))
    alpha = float(lora.get("alpha", rank))
    if rank not in (4, 8):
        raise ValueError("baseline checkpoint has an unsupported LoRA rank")
    checkpoint_config = str(checkpoint.get("model_config") or SAM2_LARGE_CONFIG)
    config = requested_config or checkpoint_config
    if config != checkpoint_config:
        raise ValueError(
            f"requested SAM 2 config {config!r} differs from baseline {checkpoint_config!r}"
        )
    expected_base = checkpoint.get("base_checkpoint")
    observed_base = _file_identity(foundation_checkpoint, include_path=False)
    if expected_base is not None and expected_base != observed_base:
        raise RuntimeError(
            f"SAM 2 foundation checkpoint differs from baseline: {observed_base}"
        )
    baseline, report = build_cracksam2(
        foundation_checkpoint,
        rank=rank,
        alpha=alpha,
        config=config,
        device=device,
    )
    adapter = checkpoint.get("adapter")
    if not isinstance(adapter, dict):
        raise ValueError("baseline checkpoint has no adapter weights")
    load_adapter_state_dict(baseline, adapter, strict=True)
    baseline.requires_grad_(False)
    baseline.eval()
    metadata = {
        "variant": "baseline",
        "model_config": config,
        "lora": {"rank": report.rank, "alpha": report.alpha},
        "foundation_checkpoint": observed_base,
    }
    return baseline, metadata, sha256_file(baseline_checkpoint)


def _infer_high_resolution_layout(
    baseline: torch.nn.Module,
    example_image: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
) -> tuple[tuple[int, ...], tuple[int, int]]:
    with torch.inference_mode(), _autocast(device, amp_dtype):
        features = baseline.encode_images(example_image.unsqueeze(0).to(device))
    high_resolution_features = features.high_resolution_features
    channels = tuple(int(value.shape[1]) for value in high_resolution_features)
    if not channels or any(value <= 0 for value in channels):
        raise RuntimeError("SAM 2 did not return usable high-resolution features")
    fusion_grid_size = (
        int(high_resolution_features[0].shape[-2]),
        int(high_resolution_features[0].shape[-1]),
    )
    if any(value <= 0 for value in fusion_grid_size):
        raise RuntimeError("SAM 2 returned an invalid selector fusion grid")
    return channels, fusion_grid_size


def _infer_high_resolution_channels(
    baseline: torch.nn.Module,
    example_image: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
) -> tuple[int, ...]:
    """Backward-compatible channel-only wrapper for external smoke utilities."""

    channels, _ = _infer_high_resolution_layout(
        baseline, example_image, device, amp_dtype
    )
    return channels


def _selector_grid_contract(
    *,
    input_image_size: tuple[int, int],
    fusion_grid_size: tuple[int, int],
) -> dict[str, object]:
    """Describe the feature-cell coordinate system used by the local selector."""

    input_height, input_width = (int(value) for value in input_image_size)
    fusion_height, fusion_width = (int(value) for value in fusion_grid_size)
    if min(input_height, input_width, fusion_height, fusion_width) <= 0:
        raise ValueError("selector input and fusion-grid sizes must be positive")
    return {
        "source": SELECTOR_FUSION_GRID_SOURCE,
        "parameter_unit": SELECTOR_FEATURE_CELL_UNIT,
        "input_image_size_pixels": [input_height, input_width],
        "fusion_grid_size_feature_cells": [fusion_height, fusion_width],
        "effective_stride_input_pixels_per_feature_cell": [
            input_height / fusion_height,
            input_width / fusion_width,
        ],
    }


def train_batch(
    model: torch.nn.Module,
    batch: Mapping[str, object],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    amp_dtype: str,
    loss_parameters: Mapping[str, object],
    gradient_clip: float,
) -> dict[str, float]:
    """Run one adapter-only update; exposed for the CPU smoke test."""
    images = batch["image"].to(device, non_blocking=True)  # type: ignore[union-attr]
    targets = batch["mask"].to(device, non_blocking=True)  # type: ignore[union-attr]
    rasters = batch["frangi_raster"].to(device, non_blocking=True)  # type: ignore[union-attr]
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, amp_dtype):
        output = model(images, rasters, output_size=tuple(targets.shape[-2:]))
        losses = residual_training_loss(
            output["candidate_logits"],
            output["baseline_logits"],
            targets,
            evidence_logits=output.get("evidence_logits"),
            evidence_support=output.get("evidence_support"),
            **loss_parameters,
        )
    loss = losses["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite residual loss")
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), gradient_clip)
        optimizer.step()
    if any(parameter.grad is not None for parameter in model.baseline.parameters()):
        raise RuntimeError("frozen baseline unexpectedly received a gradient")
    return {name: float(value.detach()) for name, value in losses.items()}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: str,
    ce_weight: float,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    candidate_losses: list[float] = []
    baseline_ious: list[float] = []
    candidate_ious: list[float] = []
    baseline_dices: list[float] = []
    candidate_dices: list[float] = []
    for batch in tqdm(loader, desc="validation OOF", unit="batch", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        rasters = batch["frangi_raster"].to(device, non_blocking=True)
        with _autocast(device, amp_dtype):
            output = model(images, rasters, output_size=tuple(targets.shape[-2:]))
            loss, _, _ = cracksam_loss_per_image(
                output["candidate_logits"], targets, ce_weight=ce_weight
            )
        candidate_losses.extend(float(value) for value in loss.cpu())
        targets_cpu = targets.cpu()
        baseline_probabilities = torch.sigmoid(output["baseline_logits"].float()).cpu()
        candidate_probabilities = torch.sigmoid(
            output["candidate_logits"].float()
        ).cpu()
        for baseline_probability, candidate_probability, target in zip(
            baseline_probabilities, candidate_probabilities, targets_cpu
        ):
            baseline_values = segmentation_metrics(
                baseline_probability, target, threshold=threshold
            )
            candidate_values = segmentation_metrics(
                candidate_probability, target, threshold=threshold
            )
            baseline_ious.append(baseline_values["iou"])
            candidate_ious.append(candidate_values["iou"])
            baseline_dices.append(baseline_values["dice"])
            candidate_dices.append(candidate_values["dice"])
    if not candidate_ious:
        raise RuntimeError("held-out validation fold is empty")
    delta_iou = np.asarray(candidate_ious) - np.asarray(baseline_ious)
    return {
        "candidate_loss": float(np.mean(candidate_losses)),
        "baseline_iou": float(np.mean(baseline_ious)),
        "candidate_iou": float(np.mean(candidate_ious)),
        "delta_iou": float(np.mean(delta_iou)),
        "delta_iou_p05": float(np.quantile(delta_iou, 0.05)),
        "severe_loss_fraction": float(np.mean(delta_iou < -0.05)),
        "baseline_dice": float(np.mean(baseline_dices)),
        "candidate_dice": float(np.mean(candidate_dices)),
    }


def _limit(names: tuple[str, ...], maximum: int | None) -> tuple[str, ...]:
    return names if maximum is None else names[:maximum]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    _prepare_output_directory(args.output, resume=bool(args.resume))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    _seed_everything(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    full_names = read_sample_list(args.train_list)
    oof_training = strict_oof_training_contract(args.fold)
    fold = load_group_safe_fold(
        args.fold_csv,
        full_names,
        args.fold,
        exclude_training_folds=args.exclude_training_folds,
    )
    if list(fold.excluded_training_folds) != oof_training[
        "additional_excluded_training_folds"
    ]:
        raise AssertionError("runtime fold split violates the strict OOF contract")
    train_names = _limit(fold.train_names, args.max_train_samples)
    validation_names = _limit(fold.validation_names, args.max_validation_samples)
    if set(train_names).intersection(validation_names):
        raise AssertionError("training and held-out sample names overlap")

    cache = load_graph_cache_index(
        args.graph_cache,
        full_names,
        image_size=(448, 448),
        noise_mode="original",
        verify_files=not args.skip_cache_file_verification,
    )
    if list(cache.manifest.get("channels", [])) != list(FRANGI_RASTER_CHANNELS):
        raise ValueError("residual training requires the exact seven-channel cache")
    normalization = load_or_fit_frangi_raster_normalization(
        args.output / "raster_normalization.json",
        cache,
        train_names,
    )
    train_dataset = FrangiGraphRasterDataset(
        args.data_root,
        args.train_list,
        train_names,
        cache,
        normalization,
        split="train",
        verify_source_files=not args.skip_source_file_verification,
        raster_condition=args.raster_condition,
    )
    validation_dataset = FrangiGraphRasterDataset(
        args.data_root,
        args.train_list,
        validation_names,
        cache,
        normalization,
        split="train",
        verify_source_files=not args.skip_source_file_verification,
        raster_condition=args.raster_condition,
    )
    steps_per_epoch = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    if args.warmup_steps < 0 or args.warmup_steps >= total_steps:
        raise ValueError(f"--warmup-steps must be below total steps ({total_steps})")

    baseline, baseline_metadata, baseline_sha256 = _load_frozen_baseline(
        args.sam2_checkpoint,
        args.baseline_checkpoint,
        requested_config=args.model_config,
        device=device,
    )
    example = train_dataset[0]["image"]
    high_resolution_channels, selector_fusion_grid_size = _infer_high_resolution_layout(
        baseline, example, device, args.amp_dtype
    )
    selector_grid = _selector_grid_contract(
        input_image_size=(int(example.shape[-2]), int(example.shape[-1])),
        fusion_grid_size=selector_fusion_grid_size,
    )
    model = FrangiGraphResidual(
        baseline,
        raster_channels=len(FRANGI_RASTER_CHANNELS),
        high_resolution_channels=high_resolution_channels,
        hidden_channels=args.hidden_channels,
        adapter_mode=args.adapter_mode,
        profile_radii=args.profile_radii,
        evidence_dilation=args.evidence_dilation,
        evidence_threshold=args.evidence_threshold,
    ).to(device)
    model.train()
    if any(parameter.requires_grad for parameter in model.baseline.parameters()):
        raise RuntimeError("baseline must be completely frozen")
    adapter_parameters = [
        parameter for parameter in model.adapter.parameters() if parameter.requires_grad
    ]
    if not adapter_parameters:
        raise RuntimeError("residual adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(
        adapter_parameters,
        lr=args.base_lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.amp_dtype == "float16"
    )

    graph_checkpoint_contract = {
        "extractor_sha256": cache.manifest["extractor_sha256"],
        "frangi": cache.manifest["frangi"],
        "channels": list(FRANGI_RASTER_CHANNELS),
        "manifest_sha256": cache.manifest_sha256,
    }
    residual_contract: dict[str, object] = {
        "raster_channels": len(FRANGI_RASTER_CHANNELS),
        "high_resolution_channels": list(high_resolution_channels),
        "hidden_channels": args.hidden_channels,
        "adapter_mode": args.adapter_mode,
        "profile_radii": list(args.profile_radii),
        "evidence_dilation": args.evidence_dilation,
        "evidence_threshold": args.evidence_threshold,
    }
    if args.adapter_mode == VERIFIED_ADAPTER_MODE:
        residual_contract.update(
            {
                "profile_radii_feature_cells": list(args.profile_radii),
                "evidence_dilation_feature_cells": args.evidence_dilation,
                "selector_grid": selector_grid,
            }
        )
    run_contract: dict[str, object] = {
        "contract_version": RESIDUAL_RUN_CONTRACT_VERSION,
        "method": (
            "FrangiGraph-SelectiveResidual-local-v1"
            if args.adapter_mode == VERIFIED_ADAPTER_MODE
            else "FrangiGraph-Residual-raster-v1"
        ),
        "raster_condition": args.raster_condition,
        "held_out_fold": args.fold,
        "oof_training": oof_training,
        "excluded_training_folds": list(fold.excluded_training_folds),
        "all_folds_excluded_from_training": sorted(
            {args.fold, *fold.excluded_training_folds}
        ),
        "baseline": {
            **baseline_metadata,
            "adapter_checkpoint": _file_identity(args.baseline_checkpoint),
        },
        "residual": residual_contract,
        "graph_cache": graph_checkpoint_contract,
        "raster_preprocessing": normalization.preprocessing_contract(),
        "data": {
            "root": str(args.data_root.expanduser().resolve()),
            "train_list": _file_identity(args.train_list),
            "fold_csv": _file_identity(args.fold_csv),
            "protocol_manifest": _file_identity(args.protocol_manifest),
            "train_samples": len(train_names),
            "train_sample_names_sha256": sample_names_sha256(train_names),
            "validation_samples": len(validation_names),
            "validation_sample_names_sha256": sample_names_sha256(validation_names),
            "excluded_training_samples": len(fold.excluded_training_names),
            "excluded_training_sample_names": list(fold.excluded_training_names),
            "excluded_training_sample_names_sha256": sample_names_sha256(
                fold.excluded_training_names
            ),
            "excluded_training_groups": list(fold.excluded_training_groups),
            "excluded_training_group_count": len(fold.excluded_training_groups),
            "excluded_training_group_names_sha256": sample_names_sha256(
                fold.excluded_training_groups
            ),
            "historical_test_sets_used": False,
        },
        "optimizer": {
            "name": "AdamW",
            "base_lr": args.base_lr,
            "betas": [0.9, 0.999],
            "weight_decay": args.weight_decay,
        },
        "schedule": {
            "name": "linear-warmup-polynomial",
            "warmup_steps": args.warmup_steps,
            "poly_power": args.poly_power,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "amp_dtype": args.amp_dtype,
            "ce_weight": args.ce_weight,
            "topology_weight": args.topology_weight,
            "safety_weight": args.safety_weight,
            "safety_margin": args.safety_margin,
            "evidence_loss_weight": args.evidence_loss_weight,
            "evidence_target_tolerance": args.evidence_target_tolerance,
            "skeleton_iterations": args.skeleton_iterations,
            "gradient_clip": args.gradient_clip,
            "threshold": args.threshold,
            "augmentation": "none-raster-v1",
            "held_out_metrics_used_for_checkpoint_selection": False,
            "checkpoint_for_oof_prediction": "latest.pt after the fixed final epoch",
        },
        "software": {"torch": torch.__version__, "numpy": np.__version__},
        "code": {
            path.name: _file_identity(path)
            for path in (
                Path(__file__),
                CRACKSAM_ROOT / "cracksam2" / "oof.py",
                CRACKSAM_ROOT / "cracksam2" / "residual_data.py",
                CRACKSAM_ROOT / "cracksam2" / "residual.py",
                CRACKSAM_ROOT / "cracksam2" / "evidence_selection.py",
                CRACKSAM_ROOT / "cracksam2" / "losses.py",
                CRACKSAM_ROOT / "cracksam2" / "model.py",
            )
        },
    }
    run_contract_sha256 = _json_sha256(run_contract)
    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha256,
        "trainable_parameters": sum(p.numel() for p in adapter_parameters),
    }
    if args.adapter_mode == VERIFIED_ADAPTER_MODE:
        config.update(
            {
                "profile_radii_feature_cells": list(args.profile_radii),
                "evidence_dilation_feature_cells": args.evidence_dilation,
                "selector_grid": selector_grid,
            }
        )
    # ``--resume`` is an invocation detail rather than part of the immutable
    # experiment.  Keeping it null makes the original and resumed config
    # exactly comparable.
    config["resume"] = None
    _publish_or_validate_config(
        args.output / "config.json", config, resume=bool(args.resume)
    )

    start_epoch = 0
    start_batch = 0
    global_step = 0
    best_validation_iou = float("-inf")
    resume_running: dict[str, float] = {}
    resume_seen_batches = 0
    if args.resume:
        resume_path = (
            args.output / "latest.pt" if args.resume == "auto" else Path(args.resume)
        )
        state = restore_residual_checkpoint(
            resume_path,
            model,
            optimizer,
            scaler,
            expected_run_contract=run_contract,
            expected_run_contract_sha256=run_contract_sha256,
        )
        start_epoch = int(state.get("epoch", 0))
        start_batch = int(state.get("next_batch", 0))
        global_step = int(state.get("global_step", 0))
        best_validation_iou = float(
            state.get("best_validation_iou", best_validation_iou)
        )
        running_value = state.get("epoch_running", {})
        if isinstance(running_value, dict):
            resume_running = {
                str(name): float(value) for name, value in running_value.items()
            }
        resume_seen_batches = int(state.get("epoch_seen_batches", 0))
        print(f"Resumed {resume_path} at epoch={start_epoch}, batch={start_batch}")

    loss_parameters = {
        "ce_weight": args.ce_weight,
        "topology_weight": args.topology_weight,
        "safety_weight": args.safety_weight,
        "safety_margin": args.safety_margin,
        "skeleton_iterations": args.skeleton_iterations,
        "evidence_weight": args.evidence_loss_weight,
        "evidence_target_tolerance": args.evidence_target_tolerance,
    }
    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"Received signal {signum}; saving after the current update.")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save_state(
        epoch: int,
        next_batch: int,
        *,
        name: str = "latest.pt",
        epoch_running: Mapping[str, float] | None = None,
        epoch_seen_batches: int = 0,
        training_state: str = "incomplete",
    ) -> None:
        payload = residual_checkpoint_payload(
            model,
            high_resolution_channels=high_resolution_channels,
            hidden_channels=args.hidden_channels,
            baseline_checkpoint_sha256=baseline_sha256,
            graph_cache=cache,
            normalization=normalization,
            run_contract=run_contract,
            run_contract_sha256=run_contract_sha256,
            optimizer_state=optimizer.state_dict(),
            scaler_state=scaler.state_dict(),
            epoch=epoch,
            next_batch=next_batch,
            global_step=global_step,
            best_validation_iou=best_validation_iou,
            epoch_running=epoch_running,
            epoch_seen_batches=epoch_seen_batches,
            training_state=training_state,
        )
        _atomic_torch_save(payload, args.output / name)

    print(
        f"Fold {args.fold}: {len(train_dataset)} training / "
        f"{len(validation_dataset)} held-out; adapter parameters="
        f"{sum(p.numel() for p in adapter_parameters):,}"
    )
    for epoch in range(start_epoch, args.epochs):
        train_loader = _make_loader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            seed=args.seed + epoch,
        )
        model.train()
        running = dict(resume_running) if epoch == start_epoch else {}
        seen_batches = resume_seen_batches if epoch == start_epoch else 0
        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"fold {args.fold} epoch {epoch + 1}/{args.epochs}",
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
            values = train_batch(
                model,
                batch,
                optimizer,
                scaler,
                device=device,
                amp_dtype=args.amp_dtype,
                loss_parameters=loss_parameters,
                gradient_clip=args.gradient_clip,
            )
            global_step += 1
            seen_batches += 1
            for key, value in values.items():
                running[key] = running.get(key, 0.0) + value
            progress.set_postfix(
                loss=f"{values['loss']:.4f}", lr=f"{learning_rate:.2e}"
            )

            next_epoch, next_batch = epoch, batch_index + 1
            if next_batch >= len(train_loader):
                next_epoch, next_batch = epoch + 1, 0
            periodic = (
                args.checkpoint_every_steps > 0
                and global_step % args.checkpoint_every_steps == 0
            )
            if periodic or stop_requested:
                save_state(
                    next_epoch,
                    next_batch,
                    epoch_running=running if next_epoch == epoch else None,
                    epoch_seen_batches=seen_batches if next_epoch == epoch else 0,
                )
            if stop_requested:
                return 130

        start_batch = 0
        resume_running = {}
        resume_seen_batches = 0
        if seen_batches == 0:
            raise RuntimeError("no training batches were processed")
        _append_csv(
            args.output / "train.csv",
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                **{key: value / seen_batches for key, value in running.items()},
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

        should_validate = (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        validation_completed = False
        if should_validate:
            validation_loader = _make_loader(
                validation_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                seed=args.seed,
            )
            values = validate(
                model,
                validation_loader,
                device=device,
                amp_dtype=args.amp_dtype,
                ce_weight=args.ce_weight,
                threshold=args.threshold,
            )
            row = {"epoch": epoch + 1, "global_step": global_step, **values}
            _append_csv(args.output / "validation.csv", row)
            print("validation", json.dumps(row, sort_keys=True))
            # This fold must remain genuinely out of training.  Its metrics are
            # descriptive only: they never choose an epoch or a checkpoint.
            # OOF prediction always uses latest.pt after the fixed final epoch.
            best_validation_iou = max(
                best_validation_iou, values["candidate_iou"]
            )
            validation_completed = True
        save_state(
            epoch + 1,
            0,
            training_state=(
                "complete"
                if epoch + 1 == args.epochs and validation_completed
                else "incomplete"
            ),
        )

    print(
        "Training complete; best observed held-out IoU (descriptive only)="
        f"{best_validation_iou:.6f}; OOF checkpoint=latest.pt"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
