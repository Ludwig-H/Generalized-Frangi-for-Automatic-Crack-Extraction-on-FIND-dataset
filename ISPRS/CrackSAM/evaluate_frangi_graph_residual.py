#!/usr/bin/env python3
"""Evaluate a FrangiGraph residual and emit rows for the logistic gate.

This command never loads a gate and never selects a gate threshold.  In
particular, ``--role historical_test`` is analysis-only and is explicitly
marked as forbidden for later threshold calibration in the run contract.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from cracksam2.data import CrackSegmentationDataset, sample_names_sha256
from cracksam2.model import build_cracksam2, load_adapter_state_dict
from cracksam2.residual import VERIFIED_ADAPTER_MODE, FrangiGraphResidual
from cracksam2.residual_evaluation import (
    EVALUATION_ROLES,
    EVALUATION_SCHEMA,
    EVALUATION_SCHEMA_VERSION,
    GraphRasterEvaluationDataset,
    ensure_evaluation_contract,
    evaluate_residual_loader,
    file_identity,
    load_group_assignments,
    load_group_assignment_records,
    load_safe_torch_checkpoint,
    read_progress_rows,
    read_selector_diagnostics,
    resolve_raster_condition,
    summarize_rows,
    validate_baseline_checkpoint,
    validate_residual_checkpoint,
    write_json_atomic,
    write_rows_csv_atomic,
    write_selector_diagnostics_atomic,
)

SELECTOR_FUSION_GRID_SOURCE = "SAM2ImageFeatures.high_resolution_features[0]"
SELECTOR_FEATURE_CELL_UNIT = "hiera_high_resolution_feature_cells"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "For verified_local_v1, profile radii and evidence dilation are "
            "checkpoint-bound values measured in cells of the first SAM 2/Hiera "
            "high-resolution feature grid; evaluation never overrides them."
        ),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument(
        "--split",
        required=True,
        help="Dataset layout split name, for example train or test_vol.",
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--noise", choices=("original", "noisy1", "noisy2"), default="original"
    )
    parser.add_argument("--graph-cache", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--raster-condition",
        choices=("correct", "no_evidence"),
        help=(
            "Evaluation raster condition. Defaults to the condition recorded "
            "by the residual checkpoint."
        ),
    )
    parser.add_argument(
        "--allow-input-ablation-raster-override",
        "--allow-causal-raster-override",
        dest="allow_causal_raster_override",
        action="store_true",
        help=(
            "Allow the intentional same-checkpoint correct/no_evidence input "
            "ablation (necessity test). The inverse direction remains forbidden. "
            "The older --allow-causal-raster-override spelling is retained as an "
            "alias."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=EVALUATION_ROLES, required=True)
    parser.add_argument(
        "--fold",
        default="",
        help="Held-out physical fold identifier; required for gate rows.",
    )
    parser.add_argument(
        "--group-folds-csv",
        type=Path,
        help=(
            "CSV with sample_name, physical_group and oof_fold. Required for "
            "gate_fit and gate_calibration."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--segmentation-threshold", type=float, default=0.5)
    parser.add_argument(
        "--label-minimum-gain",
        type=float,
        default=0.0,
        help="Strict IoU gain required for candidate_better; default: 0.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16", "none"), default="bfloat16"
    )
    parser.add_argument(
        "--skip-cache-hash-verification",
        action="store_true",
        help="Skip compressed cache-file SHA checks (not recommended).",
    )
    parser.add_argument(
        "--skip-data-hash-verification",
        action="store_true",
        help="Skip source image/mask SHA checks (not recommended).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    if not 0.0 < args.segmentation_threshold < 1.0:
        raise ValueError("segmentation-threshold must lie in (0, 1)")
    if not math.isfinite(args.label_minimum_gain):
        raise ValueError("label-minimum-gain must be finite")
    gate_role = args.role in ("gate_fit", "gate_calibration")
    if gate_role and (args.group_folds_csv is None or not args.fold):
        raise ValueError(
            "gate_fit/gate_calibration require --group-folds-csv and --fold"
        )
    if args.role == "gate_fit" and args.fold not in ("0", "1", "2", "3"):
        raise ValueError("gate_fit is reserved for out-of-fold predictions 0-3")
    if args.role == "gate_calibration" and args.fold != "4":
        raise ValueError("gate_calibration is reserved for held-out fold 4")


def _positive_integer_pair(value: object, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two positive integers")
    pair: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{name} must contain two positive integers")
        try:
            integer = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain two positive integers") from exc
        if integer <= 0 or integer != item:
            raise ValueError(f"{name} must contain two positive integers")
        pair.append(integer)
    return pair


def _selector_metadata_from_checkpoint(
    residual_payload: Mapping[str, object],
    *,
    adapter_mode: str,
    profile_radii: tuple[float, ...],
    evidence_dilation: int,
) -> dict[str, object]:
    """Expose explicit feature-cell aliases and optional recorded grid geometry."""

    if adapter_mode != VERIFIED_ADAPTER_MODE:
        return {}
    architecture = residual_payload.get("residual")
    if not isinstance(architecture, Mapping):
        raise ValueError("Residual checkpoint has no residual architecture")
    radii = list(profile_radii)
    for key, expected in (
        ("profile_radii_feature_cells", radii),
        ("evidence_dilation_feature_cells", evidence_dilation),
    ):
        observed = architecture.get(key)
        if observed is not None and observed != expected:
            raise ValueError(f"Residual checkpoint has inconsistent {key}")
    metadata: dict[str, object] = {
        "profile_radii_feature_cells": radii,
        "evidence_dilation_feature_cells": evidence_dilation,
    }
    raw_grid = architecture.get("selector_grid")
    if raw_grid is None:
        # Checkpoints created before selector-grid provenance was added remain valid.
        return metadata
    if not isinstance(raw_grid, Mapping):
        raise ValueError("Residual selector_grid must be a mapping")
    if raw_grid.get("source") != SELECTOR_FUSION_GRID_SOURCE:
        raise ValueError("Residual selector_grid has an unknown feature source")
    if raw_grid.get("parameter_unit") != SELECTOR_FEATURE_CELL_UNIT:
        raise ValueError("Residual selector_grid has an unknown parameter unit")
    input_size = _positive_integer_pair(
        raw_grid.get("input_image_size_pixels"), "selector input image size"
    )
    grid_size = _positive_integer_pair(
        raw_grid.get("fusion_grid_size_feature_cells"), "selector fusion-grid size"
    )
    raw_stride = raw_grid.get("effective_stride_input_pixels_per_feature_cell")
    if not isinstance(raw_stride, (list, tuple)) or len(raw_stride) != 2:
        raise ValueError("selector effective stride must contain two positive values")
    try:
        stride = [float(value) for value in raw_stride]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "selector effective stride must contain two positive values"
        ) from exc
    expected_stride = [
        input_size[0] / grid_size[0],
        input_size[1] / grid_size[1],
    ]
    if any(
        not math.isfinite(observed)
        or observed <= 0.0
        or not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-12)
        for observed, expected in zip(stride, expected_stride)
    ):
        raise ValueError("selector effective stride is inconsistent with its grid sizes")
    metadata["selector_grid"] = {
        "source": SELECTOR_FUSION_GRID_SOURCE,
        "parameter_unit": SELECTOR_FEATURE_CELL_UNIT,
        "input_image_size_pixels": input_size,
        "fusion_grid_size_feature_cells": grid_size,
        "effective_stride_input_pixels_per_feature_cell": stride,
    }
    return metadata


def _build_model(
    *,
    args: argparse.Namespace,
    device: torch.device,
    baseline_payload: dict[str, Any],
    residual_payload: dict[str, Any],
    cache_manifest: dict[str, object],
    baseline_identity: dict[str, object],
) -> tuple[FrangiGraphResidual, Any]:
    rank, alpha = validate_baseline_checkpoint(baseline_payload)
    foundation_identity = file_identity(args.sam2_checkpoint)
    recorded_foundation = baseline_payload.get("base_checkpoint")
    if not isinstance(recorded_foundation, dict):
        raise ValueError("Baseline checkpoint has no foundation checkpoint identity")
    if recorded_foundation.get("sha256") != foundation_identity["sha256"]:
        raise ValueError("Baseline adapter and supplied SAM 2 checkpoint do not match")
    model_config = baseline_payload.get("model_config")
    if not isinstance(model_config, str) or not model_config:
        raise ValueError("Baseline checkpoint has no valid SAM 2 model_config")
    baseline, _ = build_cracksam2(
        checkpoint=args.sam2_checkpoint,
        config=model_config,
        rank=rank,
        alpha=alpha,
        device=device,
    )
    load_adapter_state_dict(baseline, baseline_payload["adapter"], strict=True)
    residual_spec = validate_residual_checkpoint(
        residual_payload,
        baseline_checkpoint_sha256=str(baseline_identity["sha256"]),
        graph_cache_manifest=cache_manifest,
        require_complete=args.role in ("gate_fit", "gate_calibration"),
        expected_oof_fold=(
            int(args.fold)
            if args.role in ("gate_fit", "gate_calibration")
            else None
        ),
        expected_oof_role=(
            args.role if args.role in ("gate_fit", "gate_calibration") else None
        ),
    )
    model = FrangiGraphResidual(
        baseline,
        raster_channels=residual_spec.raster_channels,
        high_resolution_channels=residual_spec.high_resolution_channels,
        hidden_channels=residual_spec.hidden_channels,
        adapter_mode=residual_spec.adapter_mode,
        profile_radii=residual_spec.profile_radii or (1.5, 3.0),
        evidence_dilation=residual_spec.evidence_dilation,
        evidence_threshold=residual_spec.evidence_threshold,
    ).to(device)
    incompatible = model.adapter.load_state_dict(
        residual_spec.adapter_state, strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Residual adapter state mismatch: {incompatible}")
    model.eval()
    return model, residual_spec


def _evaluation_usage_policy(
    *, role: str, causal_raster_override: bool
) -> dict[str, bool]:
    """Describe whether evaluation rows may enter either gate stage."""
    analytical_only = bool(causal_raster_override)
    return {
        "analytical_only": analytical_only,
        "eligible_for_later_gate_fit": role == "gate_fit" and not analytical_only,
        "eligible_for_later_gate_threshold_calibration": (
            role == "gate_calibration" and not analytical_only
        ),
    }


def _ordered_finalize(
    *,
    output: Path,
    dataset_name: str,
    selected_names: list[str],
    role: str,
    causal_raster_override: bool,
) -> dict[str, object]:
    rows_by_case = read_progress_rows(
        output / "progress.jsonl", expected_dataset=dataset_name
    )
    missing = [name for name in selected_names if name not in rows_by_case]
    extra = sorted(set(rows_by_case) - set(selected_names))
    if missing or extra:
        raise RuntimeError(
            f"Residual evaluation is incomplete; missing={missing[:5]}, extra={extra[:5]}"
        )
    rows = [rows_by_case[name] for name in selected_names]
    write_rows_csv_atomic(output / "per_image.csv", rows)
    selector_diagnostics = read_selector_diagnostics(
        output / "progress.jsonl", expected_dataset=dataset_name
    )
    if selector_diagnostics:
        write_selector_diagnostics_atomic(
            output / "selector_diagnostics.json",
            selector_diagnostics,
            dataset=dataset_name,
            selected_names=selected_names,
        )
    summary = summarize_rows(rows)
    usage_policy = _evaluation_usage_policy(
        role=role, causal_raster_override=causal_raster_override
    )
    summary.update(
        {
            "schema": EVALUATION_SCHEMA,
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "dataset": dataset_name,
            "role": role,
            "gate_threshold_selected": False,
            **usage_policy,
            "historical_test_threshold_calibration_forbidden": role
            == "historical_test",
        }
    )
    write_json_atomic(output / "summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    base = CrackSegmentationDataset(
        args.data_root,
        list_file=args.list_file,
        split=args.split,
        image_size=448,
        augment=False,
        noise_mode=args.noise,
    )
    if args.group_folds_csv is not None:
        group_records = load_group_assignment_records(args.group_folds_csv)
        missing_groups = [
            name for name in base.sample_names if name not in group_records
        ]
        if missing_groups:
            raise ValueError(
                f"Group CSV is missing dataset cases: {missing_groups[:5]}"
            )
        if args.role in ("gate_fit", "gate_calibration"):
            base.sample_names = [
                name
                for name in base.sample_names
                if group_records[name][1] == str(args.fold)
            ]
            if not base.sample_names:
                raise ValueError(f"No samples belong to held-out fold {args.fold}")
        group_identity: dict[str, object] | None = file_identity(args.group_folds_csv)
    else:
        group_records = None
        group_identity = None
    if args.max_samples is not None:
        base.sample_names = base.sample_names[: args.max_samples]
    selected_names = list(base.sample_names)
    if group_records is not None:
        groups = load_group_assignments(
            args.group_folds_csv,
            selected_names=selected_names,
            expected_fold=(
                args.fold if args.role in ("gate_fit", "gate_calibration") else None
            ),
        )
    else:
        # Historical test rows are never used to fit/calibrate the gate.  Their
        # unique identifiers remain explicit, but make no physical-group claim.
        groups = {name: f"{args.dataset_name}::{name}" for name in selected_names}
    graph_dataset = GraphRasterEvaluationDataset(
        base,
        args.graph_cache,
        verify_cache_hashes=not args.skip_cache_hash_verification,
        verify_data_hashes=not args.skip_data_hash_verification,
    )

    baseline_payload = load_safe_torch_checkpoint(args.baseline_checkpoint)
    residual_payload = load_safe_torch_checkpoint(args.residual_checkpoint)
    validate_baseline_checkpoint(baseline_payload)
    baseline_identity = file_identity(args.baseline_checkpoint)
    residual_spec = validate_residual_checkpoint(
        residual_payload,
        baseline_checkpoint_sha256=str(baseline_identity["sha256"]),
        graph_cache_manifest=graph_dataset.manifest,
        require_complete=args.role in ("gate_fit", "gate_calibration"),
        expected_oof_fold=(
            int(args.fold)
            if args.role in ("gate_fit", "gate_calibration")
            else None
        ),
        expected_oof_role=(
            args.role if args.role in ("gate_fit", "gate_calibration") else None
        ),
    )
    evaluation_raster_condition, causal_raster_override = resolve_raster_condition(
        residual_spec.training_raster_condition,
        args.raster_condition,
        allow_causal_override=args.allow_causal_raster_override,
    )
    usage_policy = _evaluation_usage_policy(
        role=args.role, causal_raster_override=causal_raster_override
    )
    residual_contract: dict[str, object] = {
        "raster_channels": residual_spec.raster_channels,
        "high_resolution_channels": list(residual_spec.high_resolution_channels),
        "hidden_channels": residual_spec.hidden_channels,
        # Historical names remain present for consumers of schema version 1.
        "adapter_mode": residual_spec.adapter_mode,
        "profile_radii": list(residual_spec.profile_radii),
        "evidence_dilation": residual_spec.evidence_dilation,
        "evidence_threshold": residual_spec.evidence_threshold,
        "raster_preprocessing": residual_spec.preprocessing.as_dict(),
        "training_raster_condition": residual_spec.training_raster_condition,
        "evaluation_raster_condition": evaluation_raster_condition,
        "causal_raster_override": causal_raster_override,
        "checkpoint_held_out_fold": residual_spec.held_out_fold,
        "checkpoint_oof_training": dict(residual_spec.oof_training),
        "checkpoint_training_state": residual_spec.training_state,
    }
    residual_contract.update(
        _selector_metadata_from_checkpoint(
            residual_payload,
            adapter_mode=residual_spec.adapter_mode,
            profile_radii=residual_spec.profile_radii,
            evidence_dilation=residual_spec.evidence_dilation,
        )
    )
    contract = {
        "schema": EVALUATION_SCHEMA,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "analytical_only": usage_policy["analytical_only"],
        "dataset": {
            "name": args.dataset_name,
            "role": args.role,
            "fold": str(args.fold),
            "root": str(args.data_root.expanduser().resolve()),
            "list": file_identity(args.list_file),
            "split": args.split,
            "noise": args.noise,
            "image_size": [448, 448],
            "selected_samples": len(selected_names),
            "selected_sample_names_sha256": sample_names_sha256(selected_names),
            "group_assignments": group_identity,
        },
        "checkpoints": {
            "sam2": file_identity(args.sam2_checkpoint),
            "baseline": baseline_identity,
            "residual": file_identity(args.residual_checkpoint),
        },
        "graph_cache": {
            "root": str(args.graph_cache.expanduser().resolve()),
            "manifest": file_identity(
                args.graph_cache / ".cracksam2-frangi-graph-v2.json"
            ),
            "extractor_sha256": graph_dataset.manifest.get("extractor_sha256"),
            "frangi": graph_dataset.manifest.get("frangi"),
            "channels": graph_dataset.manifest.get("channels"),
            "verify_cache_hashes": not args.skip_cache_hash_verification,
            "verify_data_hashes": not args.skip_data_hash_verification,
        },
        "residual": residual_contract,
        "segmentation_threshold": args.segmentation_threshold,
        "label_minimum_gain": args.label_minimum_gain,
        "gate_policy": {
            "feature_rows_only": True,
            "spatial_support_source": (
                "accepted_local_selector_output"
                if residual_spec.adapter_mode == VERIFIED_ADAPTER_MODE
                else "raw_frangi_cache_support"
            ),
            "frangi_cache_statistics_source": "raw_frangi_cache",
            "threshold_selected_by_this_command": False,
            "eligible_for_later_gate_fit": usage_policy[
                "eligible_for_later_gate_fit"
            ],
            "threshold_may_later_be_calibrated_from_this_role": usage_policy[
                "eligible_for_later_gate_threshold_calibration"
            ],
            "historical_tests_forbidden_for_threshold_selection": True,
        },
    }
    ensure_evaluation_contract(args.output, contract)

    rows_by_case = read_progress_rows(
        args.output / "progress.jsonl", expected_dataset=args.dataset_name
    )
    unexpected = sorted(set(rows_by_case) - set(selected_names))
    if unexpected:
        raise RuntimeError(
            f"Progress contains cases outside this split: {unexpected[:5]}"
        )
    remaining_indices = [
        index for index, name in enumerate(selected_names) if name not in rows_by_case
    ]
    if remaining_indices:
        model, residual_spec = _build_model(
            args=args,
            device=device,
            baseline_payload=baseline_payload,
            residual_payload=residual_payload,
            cache_manifest=graph_dataset.manifest,
            baseline_identity=baseline_identity,
        )
        loader = DataLoader(
            Subset(graph_dataset, remaining_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        evaluate_residual_loader(
            model,
            loader,
            preprocessing=residual_spec.preprocessing,
            source_groups=groups,
            dataset=args.dataset_name,
            role=args.role,
            fold=str(args.fold),
            progress_path=args.output / "progress.jsonl",
            device=device,
            amp_dtype=args.amp_dtype,
            segmentation_threshold=args.segmentation_threshold,
            label_minimum_gain=args.label_minimum_gain,
            raster_condition=evaluation_raster_condition,
        )
    return _ordered_finalize(
        output=args.output,
        dataset_name=args.dataset_name,
        selected_names=selected_names,
        role=args.role,
        causal_raster_override=causal_raster_override,
    )


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
