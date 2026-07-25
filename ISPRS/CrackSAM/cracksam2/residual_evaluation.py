"""Safe evaluation records for the FrangiGraph residual and its simple gate.

This module deliberately separates two kinds of values:

* the seven gate inputs are computed only from model outputs and the Frangi
  cache, before a target mask is inspected;
* segmentation scores and training labels are computed from the target and are
  never eligible as gate inputs.

The progress journal is append-only and fsynced after every completed batch.
Final CSV and JSON files are rebuilt atomically from that journal, so a Spot VM
can resume without silently mixing two evaluation protocols.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .data import CrackSegmentationDataset, normalize_noise_mode, sample_names_sha256
from .gating import DEFAULT_GATE_FEATURES, compute_gate_features
from .graph_cache import (
    GRAPH_CACHE_MANIFEST,
    graph_cache_relative_path,
    load_graph_raster,
    sha256_file,
)
from .graph_types import (
    FRANGI_RASTER_CHANNELS,
    FRANGI_RASTER_SCHEMA_VERSION,
    no_frangi_evidence_raster,
)
from .metrics import segmentation_metrics
from .oof import validate_oof_run_contract, validate_strict_oof_training_contract
from .residual import (
    LEGACY_ADAPTER_MODE,
    RESIDUAL_ADAPTER_MODES,
    VERIFIED_ADAPTER_MODE,
)

EVALUATION_SCHEMA = "cracksam2.frangigraph-residual-evaluation"
EVALUATION_SCHEMA_VERSION = 1
PROGRESS_SCHEMA_VERSION = 1
SELECTOR_DIAGNOSTICS_SCHEMA = "cracksam2.frangigraph-selector-diagnostics"
SELECTOR_DIAGNOSTICS_SCHEMA_VERSION = 1
EVALUATION_ROLES: tuple[str, ...] = (
    "gate_fit",
    "gate_calibration",
    "development",
    "historical_test",
)

METRIC_FIELDS: tuple[str, ...] = (
    "baseline_iou",
    "candidate_iou",
    "candidate_iou_gain",
    "delta_iou",
    "baseline_dice",
    "candidate_dice",
    "candidate_dice_gain",
)
LABEL_FIELDS: tuple[str, ...] = (
    "candidate_better",
    "candidate_practical_gain",
    "candidate_harmful",
    "candidate_severe_harm",
)
ROW_FIELDS: tuple[str, ...] = (
    "case_name",
    "source_group",
    "dataset",
    "role",
    "fold",
    *METRIC_FIELDS,
    *LABEL_FIELDS,
    *DEFAULT_GATE_FEATURES,
)

# Selector diagnostics deliberately live outside ``ROW_FIELDS``.  Gate-ready
# CSVs and historical progress entries therefore retain their exact schema,
# while new progress entries may atomically carry this optional side record.
SELECTOR_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "case_name",
    "evidence_score_source",
    "gate_spatial_support_source",
    "fusion_height",
    "fusion_width",
    "spatial_cells",
    "selector_support_cells",
    "selector_selected_cells",
    "annotation_overlap_target_support_cells",
    "annotation_overlap_true_positive_cells",
    "selector_support_fraction",
    "evidence_score_support_mean",
    "evidence_score_support_p05",
    "evidence_score_support_p50",
    "evidence_score_support_p95",
    "selector_accepted_fraction_on_support",
    "correction_envelope_fraction_on_fusion_grid",
    "residual_absolute_mean_inside_output_envelope",
    "residual_absolute_mean_outside_output_envelope",
    "annotation_overlap_precision_on_selected_support",
    "annotation_overlap_recall_on_supported_target",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("Only scalar tensors can be serialized as JSON values")
        value = value.detach().cpu().item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON output cannot contain NaN or infinity")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json_atomic(path: str | os.PathLike[str], value: object) -> Path:
    """Write durable JSON through a temporary file in the destination folder."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _json_safe(value),
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def write_rows_csv_atomic(
    path: str | os.PathLike[str], rows: Sequence[Mapping[str, object]]
) -> Path:
    """Publish a gate-ready CSV with one fixed and auditable column order."""
    if not rows:
        raise ValueError("Cannot write an empty residual evaluation CSV")
    normalized = [normalize_evaluation_row(row) for row in rows]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ROW_FIELDS))
            writer.writeheader()
            writer.writerows(normalized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def file_identity(path: str | os.PathLike[str]) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "name": source.name,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def load_group_assignments(
    path: str | os.PathLike[str],
    *,
    selected_names: Sequence[str],
    expected_fold: str | None = None,
) -> dict[str, str]:
    """Load physical groups and optionally verify a single held-out fold."""
    records = load_group_assignment_records(path)
    names = list(selected_names)
    missing = [name for name in names if name not in records]
    if missing:
        raise ValueError(f"Group CSV is missing selected cases: {missing[:5]}")
    if expected_fold is not None:
        wrong = [name for name in names if records[name][1] != str(expected_fold)]
        if wrong:
            raise ValueError(
                f"Selected cases are not all in held-out fold {expected_fold}: "
                f"{wrong[:5]}"
            )
    return {name: records[name][0] for name in names}


def load_group_assignment_records(
    path: str | os.PathLike[str],
) -> dict[str, tuple[str, str]]:
    """Read every ``sample -> (physical group, fold)`` record strictly."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_name", "physical_group", "oof_fold"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Group CSV must contain {sorted(required)}: {source}")
        records: dict[str, tuple[str, str]] = {}
        group_folds: dict[str, str] = {}
        for line_number, row in enumerate(reader, start=2):
            name = (row.get("sample_name") or "").strip()
            group = (row.get("physical_group") or "").strip()
            fold = (row.get("oof_fold") or "").strip()
            if not name or not group or not fold:
                raise ValueError(f"Empty group assignment at {source}:{line_number}")
            if name in records:
                raise ValueError(f"Duplicate group assignment for {name!r}")
            previous_fold = group_folds.setdefault(group, fold)
            if previous_fold != fold:
                raise ValueError(f"Physical group crosses folds: {group!r}")
            records[name] = (group, fold)
    if not records:
        raise ValueError(f"Group CSV is empty: {source}")
    return records


def _as_finite_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def _single_spatial_array(value: Any, name: str) -> np.ndarray:
    array = _as_finite_numpy(value, name)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must describe one 2D image, got {array.shape}")
    return array


def _sigmoid(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value, dtype=np.float64)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _binary_spatial_mask(value: Any, name: str) -> np.ndarray:
    array = _single_spatial_array(value, name)
    tolerance = 1e-6
    if np.any(array < -tolerance) or np.any(array > 1.0 + tolerance):
        raise ValueError(f"{name} must lie in [0, 1]")
    rounded = array > 0.5
    if not np.allclose(array, rounded, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} must be binary")
    return rounded


def build_selector_diagnostic(
    *,
    case_name: str,
    evidence_score_source: str,
    gate_spatial_support_source: str,
    evidence_support_fusion: Any,
    evidence_score_fusion: Any,
    evidence_selected_fusion: Any,
    correction_envelope_fusion: Any,
    correction_envelope_output: Any,
    residual_logits: Any,
    target: Any | None = None,
) -> dict[str, object]:
    """Build one target-isolated local-selector diagnostic record.

    Scores are summarized only on the selector's effective support. Annotation
    overlap is descriptive and never a gate feature or the verifier's dilated
    auxiliary training target: recall is conditioned on exact annotation cells
    that also have Frangi support.
    """

    if not case_name:
        raise ValueError("selector diagnostic case_name cannot be empty")
    if evidence_score_source not in (
        "evidence_score_fusion",
        "evidence_score",
        "evidence_probability",
    ):
        raise ValueError("unknown evidence score output name")
    if gate_spatial_support_source != "accepted_local_selector_output":
        raise ValueError("selector diagnostics require accepted local gate support")
    support = _binary_spatial_mask(
        evidence_support_fusion, "evidence_support_fusion"
    )
    selected = _binary_spatial_mask(
        evidence_selected_fusion, "evidence_selected_fusion"
    )
    envelope_fusion = _binary_spatial_mask(
        correction_envelope_fusion, "correction_envelope_fusion"
    )
    score = _single_spatial_array(evidence_score_fusion, "evidence_score_fusion")
    selector_shapes = {
        support.shape,
        selected.shape,
        envelope_fusion.shape,
        score.shape,
    }
    if len(selector_shapes) != 1:
        raise ValueError(
            f"selector fusion maps have different shapes: {selector_shapes}"
        )
    envelope_output = _binary_spatial_mask(
        correction_envelope_output, "correction_envelope_output"
    )
    residual = _single_spatial_array(residual_logits, "residual_logits")
    if envelope_output.shape != residual.shape:
        raise ValueError(
            "output correction envelope and residual logits must have equal shape"
        )
    tolerance = 1e-6
    if np.any(score < -tolerance) or np.any(score > 1.0 + tolerance):
        raise ValueError("evidence_score must lie in [0, 1]")
    score = np.clip(score, 0.0, 1.0)
    if np.any(selected & ~support):
        raise ValueError("evidence_selected must be a subset of evidence_support")
    if np.any(selected & ~envelope_fusion):
        raise ValueError(
            "correction_envelope_fusion must contain evidence_selected_fusion"
        )

    fusion_height, fusion_width = (int(value) for value in support.shape)
    spatial_cells = int(support.size)
    support_cells = int(np.count_nonzero(support))
    selected_cells = int(np.count_nonzero(selected))
    if support_cells:
        support_scores = score[support]
        score_mean = float(np.mean(support_scores))
        score_p05, score_p50, score_p95 = (
            float(value) for value in np.quantile(support_scores, (0.05, 0.5, 0.95))
        )
        accepted_fraction = selected_cells / support_cells
    else:
        score_mean = score_p05 = score_p50 = score_p95 = 0.0
        accepted_fraction = 0.0

    absolute_residual = np.abs(residual)
    residual_inside = (
        float(np.mean(absolute_residual[envelope_output]))
        if envelope_output.any()
        else 0.0
    )
    outside = ~envelope_output
    residual_outside = (
        float(np.mean(absolute_residual[outside])) if outside.any() else 0.0
    )

    if target is None:
        target_support_cells: int | None = None
        true_positive_cells: int | None = None
        alignment_precision: float | None = None
        alignment_recall: float | None = None
    else:
        truth_output = _single_spatial_array(
            target, "selector_alignment_target"
        ) > 0.5
        if truth_output.shape == support.shape:
            truth_fusion = truth_output
        else:
            truth_tensor = torch.from_numpy(truth_output.astype(np.float32))[None, None]
            if (
                truth_output.shape[0] >= fusion_height
                and truth_output.shape[1] >= fusion_width
            ):
                truth_tensor = torch.nn.functional.adaptive_max_pool2d(
                    truth_tensor, (fusion_height, fusion_width)
                )
            else:
                truth_tensor = torch.nn.functional.interpolate(
                    truth_tensor,
                    size=(fusion_height, fusion_width),
                    mode="nearest",
                )
            truth_fusion = truth_tensor[0, 0].numpy() > 0.5
        supported_target = support & truth_fusion
        true_positive = selected & truth_fusion
        target_support_cells = int(np.count_nonzero(supported_target))
        true_positive_cells = int(np.count_nonzero(true_positive))
        alignment_precision = (
            true_positive_cells / selected_cells if selected_cells else 0.0
        )
        alignment_recall = (
            true_positive_cells / target_support_cells
            if target_support_cells
            else 0.0
        )

    return normalize_selector_diagnostic(
        {
            "case_name": case_name,
            "evidence_score_source": evidence_score_source,
            "gate_spatial_support_source": gate_spatial_support_source,
            "fusion_height": fusion_height,
            "fusion_width": fusion_width,
            "spatial_cells": spatial_cells,
            "selector_support_cells": support_cells,
            "selector_selected_cells": selected_cells,
            "annotation_overlap_target_support_cells": (
                target_support_cells
            ),
            "annotation_overlap_true_positive_cells": (
                true_positive_cells
            ),
            "selector_support_fraction": support_cells / spatial_cells,
            "evidence_score_support_mean": score_mean,
            "evidence_score_support_p05": score_p05,
            "evidence_score_support_p50": score_p50,
            "evidence_score_support_p95": score_p95,
            "selector_accepted_fraction_on_support": accepted_fraction,
            "correction_envelope_fraction_on_fusion_grid": float(
                np.mean(envelope_fusion)
            ),
            "residual_absolute_mean_inside_output_envelope": residual_inside,
            "residual_absolute_mean_outside_output_envelope": residual_outside,
            "annotation_overlap_precision_on_selected_support": (
                alignment_precision
            ),
            "annotation_overlap_recall_on_supported_target": (
                alignment_recall
            ),
        }
    )


def normalize_selector_diagnostic(row: Mapping[str, object]) -> dict[str, object]:
    """Validate the fixed side-record schema without touching ``ROW_FIELDS``."""

    missing = [name for name in SELECTOR_DIAGNOSTIC_FIELDS if name not in row]
    extra = sorted(set(row) - set(SELECTOR_DIAGNOSTIC_FIELDS))
    if missing or extra:
        raise ValueError(
            f"Selector diagnostic fields differ; missing={missing}, extra={extra}"
        )
    case_name = row["case_name"]
    if not isinstance(case_name, str) or not case_name:
        raise ValueError("selector diagnostic case_name must be non-empty text")
    score_source = row["evidence_score_source"]
    if score_source not in (
        "evidence_score_fusion",
        "evidence_score",
        "evidence_probability",
    ):
        raise ValueError("selector diagnostic has an unknown score source")
    gate_support_source = row["gate_spatial_support_source"]
    if gate_support_source != "accepted_local_selector_output":
        raise ValueError("selector diagnostic has an unknown gate support source")

    integer_fields = (
        "fusion_height",
        "fusion_width",
        "spatial_cells",
        "selector_support_cells",
        "selector_selected_cells",
    )
    normalized: dict[str, object] = {
        "case_name": case_name,
        "evidence_score_source": score_source,
        "gate_spatial_support_source": gate_support_source,
    }
    for field in integer_fields:
        value = row[field]
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer count")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer count") from exc
        if integer < 0 or float(value) != integer:
            raise ValueError(f"{field} must be a non-negative integer")
        normalized[field] = integer
    if int(normalized["fusion_height"]) <= 0 or int(normalized["fusion_width"]) <= 0:
        raise ValueError("fusion dimensions must be positive")
    if int(normalized["spatial_cells"]) != int(normalized["fusion_height"]) * int(
        normalized["fusion_width"]
    ):
        raise ValueError("spatial_cells differs from fusion dimensions")
    if int(normalized["selector_support_cells"]) > int(
        normalized["spatial_cells"]
    ):
        raise ValueError("selector_support_cells exceeds spatial_cells")
    if int(normalized["selector_selected_cells"]) > int(
        normalized["selector_support_cells"]
    ):
        raise ValueError("selector_selected_cells exceeds selector_support_cells")

    optional_count_fields = (
        "annotation_overlap_target_support_cells",
        "annotation_overlap_true_positive_cells",
    )
    optional_counts: dict[str, int | None] = {}
    for field in optional_count_fields:
        value = row[field]
        if value is None:
            optional_counts[field] = None
            normalized[field] = None
            continue
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer count or null")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer count or null") from exc
        if integer < 0 or float(value) != integer:
            raise ValueError(f"{field} must be a non-negative integer or null")
        optional_counts[field] = integer
        normalized[field] = integer
    target_count = optional_counts[optional_count_fields[0]]
    true_positive_count = optional_counts[optional_count_fields[1]]
    if (target_count is None) != (true_positive_count is None):
        raise ValueError("selector alignment counts must be both present or both null")
    if target_count is not None:
        if target_count > int(normalized["selector_support_cells"]):
            raise ValueError("alignment target support exceeds selector support")
        if true_positive_count is None or true_positive_count > min(
            target_count, int(normalized["selector_selected_cells"])
        ):
            raise ValueError("alignment true positives exceed their denominators")

    bounded_fields = (
        "selector_support_fraction",
        "evidence_score_support_mean",
        "evidence_score_support_p05",
        "evidence_score_support_p50",
        "evidence_score_support_p95",
        "selector_accepted_fraction_on_support",
        "correction_envelope_fraction_on_fusion_grid",
    )
    for field in bounded_fields:
        try:
            value = float(row[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must lie in [0, 1]")
        normalized[field] = value
    if not (
        float(normalized["evidence_score_support_p05"])
        <= float(normalized["evidence_score_support_p50"])
        <= float(normalized["evidence_score_support_p95"])
    ):
        raise ValueError("evidence score quantiles are not ordered")

    expected_support_fraction = int(normalized["selector_support_cells"]) / int(
        normalized["spatial_cells"]
    )
    support_cells = int(normalized["selector_support_cells"])
    expected_accepted_fraction = (
        int(normalized["selector_selected_cells"]) / support_cells
        if support_cells
        else 0.0
    )
    if not math.isclose(
        float(normalized["selector_support_fraction"]),
        expected_support_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("selector_support_fraction differs from its counts")
    if not math.isclose(
        float(normalized["selector_accepted_fraction_on_support"]),
        expected_accepted_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("accepted support fraction differs from its counts")

    for field in (
        "residual_absolute_mean_inside_output_envelope",
        "residual_absolute_mean_outside_output_envelope",
    ):
        try:
            value = float(row[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
        normalized[field] = value

    alignment_fields = (
        "annotation_overlap_precision_on_selected_support",
        "annotation_overlap_recall_on_supported_target",
    )
    for field in alignment_fields:
        value = row[field]
        if target_count is None:
            if value is not None:
                raise ValueError("alignment metrics need an alignment target")
            normalized[field] = None
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field} must be numeric when target is available"
            ) from exc
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{field} must lie in [0, 1]")
        normalized[field] = numeric
    if target_count is not None and true_positive_count is not None:
        selected_cells = int(normalized["selector_selected_cells"])
        expected_precision = (
            true_positive_count / selected_cells if selected_cells else 0.0
        )
        expected_recall = true_positive_count / target_count if target_count else 0.0
        if not math.isclose(
            float(normalized[alignment_fields[0]]),
            expected_precision,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(normalized[alignment_fields[1]]),
            expected_recall,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("alignment metrics differ from their counts")
    return {field: normalized[field] for field in SELECTOR_DIAGNOSTIC_FIELDS}


def build_evaluation_row(
    *,
    case_name: str,
    source_group: str,
    dataset: str,
    role: str,
    fold: str,
    baseline_logits: Any,
    candidate_logits: Any,
    target: Any,
    frangi_stats: Mapping[str, float],
    frangi_support: Any,
    segmentation_threshold: float = 0.5,
    label_minimum_gain: float = 0.0,
) -> dict[str, object]:
    """Build one row while keeping target-derived values out of gate features."""
    if not case_name or not source_group or not dataset:
        raise ValueError("case_name, source_group and dataset must be non-empty")
    if role not in EVALUATION_ROLES:
        raise ValueError(f"Unknown evaluation role: {role!r}")
    if not 0.0 < segmentation_threshold < 1.0:
        raise ValueError("segmentation_threshold must lie strictly between 0 and 1")
    if not math.isfinite(float(label_minimum_gain)):
        raise ValueError("label_minimum_gain must be finite")

    # This is the complete feature-producing call.  It intentionally receives
    # neither ``target`` nor any metric computed from it.
    features = compute_gate_features(
        baseline_logits,
        candidate_logits,
        frangi_stats,
        frangi_support=frangi_support,
    )
    if tuple(features) != DEFAULT_GATE_FEATURES:
        raise RuntimeError(
            "Gate feature implementation drifted from DEFAULT_GATE_FEATURES: "
            f"{tuple(features)} != {DEFAULT_GATE_FEATURES}"
        )

    baseline = _single_spatial_array(baseline_logits, "baseline_logits")
    candidate = _single_spatial_array(candidate_logits, "candidate_logits")
    truth = _single_spatial_array(target, "target")
    if baseline.shape != candidate.shape or baseline.shape != truth.shape:
        raise ValueError(
            "baseline, candidate and target shapes must match: "
            f"{baseline.shape}, {candidate.shape}, {truth.shape}"
        )
    baseline_metrics = segmentation_metrics(
        _sigmoid(baseline), truth, threshold=segmentation_threshold
    )
    candidate_metrics = segmentation_metrics(
        _sigmoid(candidate), truth, threshold=segmentation_threshold
    )
    iou_gain = candidate_metrics["iou"] - baseline_metrics["iou"]
    dice_gain = candidate_metrics["dice"] - baseline_metrics["dice"]

    row: dict[str, object] = {
        "case_name": case_name,
        "source_group": source_group,
        "dataset": dataset,
        "role": role,
        "fold": str(fold),
        "baseline_iou": baseline_metrics["iou"],
        "candidate_iou": candidate_metrics["iou"],
        "candidate_iou_gain": iou_gain,
        "delta_iou": iou_gain,
        "baseline_dice": baseline_metrics["dice"],
        "candidate_dice": candidate_metrics["dice"],
        "candidate_dice_gain": dice_gain,
        "candidate_better": int(iou_gain > float(label_minimum_gain)),
        "candidate_practical_gain": int(iou_gain >= 0.005),
        "candidate_harmful": int(iou_gain <= -0.005),
        "candidate_severe_harm": int(iou_gain <= -0.05),
        **features,
    }
    return normalize_evaluation_row(row)


def normalize_evaluation_row(row: Mapping[str, object]) -> dict[str, object]:
    """Reject missing, extra, non-finite, or malformed progress values."""
    missing = [name for name in ROW_FIELDS if name not in row]
    extra = sorted(set(row) - set(ROW_FIELDS))
    if missing or extra:
        raise ValueError(
            f"Evaluation row fields differ; missing={missing}, extra={extra}"
        )
    normalized: dict[str, object] = {}
    for field in ("case_name", "source_group", "dataset", "role", "fold"):
        value = row[field]
        if not isinstance(value, str) or (field != "fold" and not value):
            raise ValueError(f"Invalid text field {field!r}")
        normalized[field] = value
    if normalized["role"] not in EVALUATION_ROLES:
        raise ValueError(f"Unknown evaluation role: {normalized['role']!r}")
    for field in METRIC_FIELDS + DEFAULT_GATE_FEATURES:
        try:
            value = float(row[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric field {field!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite numeric field {field!r}")
        normalized[field] = value
    for field in LABEL_FIELDS:
        value = row[field]
        if isinstance(value, bool):
            value = int(value)
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid binary label {field!r}") from exc
        if integer not in (0, 1) or float(value) != integer:
            raise ValueError(f"Binary label {field!r} must be 0 or 1")
        normalized[field] = integer
    return {field: normalized[field] for field in ROW_FIELDS}


def append_progress_batch(
    path: str | os.PathLike[str],
    *,
    dataset: str,
    rows: Sequence[Mapping[str, object]],
    selector_diagnostics: Sequence[Mapping[str, object]] | None = None,
) -> None:
    """Durably append gate rows and optional selector records in one journal line."""
    if not rows:
        raise ValueError("Cannot append an empty progress batch")
    normalized = [normalize_evaluation_row(row) for row in rows]
    if any(row["dataset"] != dataset for row in normalized):
        raise ValueError("Progress rows do not belong to the declared dataset")
    entry = {
        "format_version": PROGRESS_SCHEMA_VERSION,
        "dataset": dataset,
        "rows": normalized,
    }
    if selector_diagnostics is not None:
        if not selector_diagnostics:
            raise ValueError("selector_diagnostics cannot be an empty explicit batch")
        normalized_diagnostics = [
            normalize_selector_diagnostic(row) for row in selector_diagnostics
        ]
        row_names = [str(row["case_name"]) for row in normalized]
        diagnostic_names = [
            str(row["case_name"]) for row in normalized_diagnostics
        ]
        if diagnostic_names != row_names:
            raise ValueError(
                "Selector diagnostics must match progress rows in exact batch order"
            )
        entry["selector_diagnostics"] = normalized_diagnostics
    payload = json.dumps(
        _json_safe(entry), sort_keys=True, ensure_ascii=True, allow_nan=False
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_progress_records(
    path: str | os.PathLike[str], *, expected_dataset: str
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Read both record types and repair only a partially written final line."""
    source_path = Path(path)
    if not source_path.is_file():
        return {}, {}
    size = source_path.stat().st_size
    valid_end = 0
    append_newline = False
    rows: dict[str, dict[str, object]] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    with source_path.open("rb") as stream:
        line_number = 0
        while True:
            line_start = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            line_number += 1
            line_end = stream.tell()
            if not raw.strip():
                valid_end = line_end
                continue
            try:
                entry = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if line_end == size and not raw.endswith(b"\n"):
                    valid_end = line_start
                    break
                raise ValueError(
                    f"Invalid progress journal {source_path} at line {line_number}"
                ) from exc
            if not isinstance(entry, dict) or entry.get("format_version") != 1:
                raise ValueError(f"Unsupported progress entry at line {line_number}")
            if entry.get("dataset") != expected_dataset:
                raise ValueError(f"Progress dataset mismatch at line {line_number}")
            batch_rows = entry.get("rows")
            if not isinstance(batch_rows, list) or not batch_rows:
                raise ValueError(f"Empty progress batch at line {line_number}")
            batch_names: list[str] = []
            for raw_row in batch_rows:
                if not isinstance(raw_row, dict):
                    raise ValueError(f"Invalid progress row at line {line_number}")
                row = normalize_evaluation_row(raw_row)
                name = str(row["case_name"])
                if name in rows:
                    raise ValueError(f"Duplicate progress case {name!r}")
                rows[name] = row
                batch_names.append(name)
            raw_diagnostics = entry.get("selector_diagnostics")
            if raw_diagnostics is not None:
                if not isinstance(raw_diagnostics, list) or not raw_diagnostics:
                    raise ValueError(
                        f"Invalid selector diagnostic batch at line {line_number}"
                    )
                diagnostic_names: list[str] = []
                for raw_diagnostic in raw_diagnostics:
                    if not isinstance(raw_diagnostic, dict):
                        raise ValueError(
                            f"Invalid selector diagnostic at line {line_number}"
                        )
                    diagnostic = normalize_selector_diagnostic(raw_diagnostic)
                    name = str(diagnostic["case_name"])
                    if name in diagnostics:
                        raise ValueError(f"Duplicate selector diagnostic case {name!r}")
                    diagnostics[name] = diagnostic
                    diagnostic_names.append(name)
                if diagnostic_names != batch_names:
                    raise ValueError(
                        "Selector diagnostics differ from their progress batch at "
                        f"line {line_number}"
                    )
            valid_end = line_end
            append_newline = not raw.endswith(b"\n")
    if valid_end < size:
        with source_path.open("r+b") as stream:
            stream.truncate(valid_end)
            stream.flush()
            os.fsync(stream.fileno())
    elif append_newline:
        with source_path.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    return rows, diagnostics


def read_progress_rows(
    path: str | os.PathLike[str], *, expected_dataset: str
) -> dict[str, dict[str, object]]:
    """Read gate rows from old or selector-extended progress journals."""

    rows, _ = _read_progress_records(path, expected_dataset=expected_dataset)
    return rows


def read_selector_diagnostics(
    path: str | os.PathLike[str], *, expected_dataset: str
) -> dict[str, dict[str, object]]:
    """Read optional selector records; schema-v1 historical entries yield none."""

    _, diagnostics = _read_progress_records(path, expected_dataset=expected_dataset)
    return diagnostics


def write_selector_diagnostics_atomic(
    path: str | os.PathLike[str],
    diagnostics: Mapping[str, Mapping[str, object]],
    *,
    dataset: str,
    selected_names: Sequence[str],
) -> Path:
    """Publish ordered selector diagnostics without changing historical CSVs."""

    names = list(selected_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("selected_names must be unique and non-empty")
    extra = sorted(set(diagnostics) - set(names))
    if extra:
        raise ValueError(f"Selector diagnostics contain unexpected cases: {extra[:5]}")
    rows = [
        normalize_selector_diagnostic(diagnostics[name])
        for name in names
        if name in diagnostics
    ]
    if not rows:
        raise ValueError("Cannot publish an empty selector diagnostic artifact")
    missing = [name for name in names if name not in diagnostics]
    payload = {
        "schema": SELECTOR_DIAGNOSTICS_SCHEMA,
        "schema_version": SELECTOR_DIAGNOSTICS_SCHEMA_VERSION,
        "dataset": dataset,
        "selected_cases": len(names),
        "diagnostic_cases": len(rows),
        "complete": not missing,
        "missing_case_names": missing,
        "row_fields": list(SELECTOR_DIAGNOSTIC_FIELDS),
        "semantics": {
            "selector_grid": (
                "fusion-grid decision cells; dimensions are recorded per case"
            ),
            "gate_spatial_support": (
                "verified-local gate spatial summaries use the accepted binary "
                "selector output; cache-level Frangi similarity and density remain raw"
            ),
            "score": (
                "evidence_score on Frangi-supported fusion cells; "
                "evidence_probability is a backward-compatible output alias"
            ),
            "accepted_fraction": "selected support cells / all support cells",
            "envelope_fraction": "accepted correction envelope on the fusion grid",
            "residual_amplitude": (
                "mean absolute residual logit inside/outside the output-resolution "
                "correction envelope"
            ),
            "annotation_overlap": (
                "descriptive only and never a gate feature or the dilated auxiliary "
                "training target; the exact binary annotation is max-pooled to the "
                "fusion grid when downsampling; precision uses selected support cells "
                "and recall uses annotated support cells; an empty denominator maps "
                "to 0.0"
            ),
        },
        "rows": rows,
    }
    return write_json_atomic(path, payload)


def ensure_evaluation_contract(
    output_dir: str | os.PathLike[str], contract: Mapping[str, object]
) -> Path:
    """Create or exactly verify the immutable run contract before resuming."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "evaluation_contract.json"
    expected = _json_safe(dict(contract))
    if path.is_file():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid evaluation contract: {path}") from exc
        if observed != expected:
            differing = sorted(
                key
                for key in set(observed) | set(expected)
                if observed.get(key) != expected.get(key)
            )
            raise RuntimeError(
                "Evaluation output belongs to another immutable contract; "
                f"differing fields: {differing}"
            )
        return path
    stale = [
        root / name
        for name in (
            "progress.jsonl",
            "per_image.csv",
            "summary.json",
            "selector_diagnostics.json",
        )
        if (root / name).exists()
    ]
    if stale:
        raise RuntimeError(
            "Evaluation artifacts exist without their contract: "
            + ", ".join(str(path) for path in stale)
        )
    write_json_atomic(path, expected)
    return path


@dataclass(frozen=True)
class RasterPreprocessing:
    """Explicit channel transformation stored in the residual checkpoint."""

    channel_names: tuple[str, ...]
    log1p_channels: tuple[str, ...]
    offset: tuple[float, ...]
    scale: tuple[float, ...]
    format_version: int | None = None
    fit_sample_count: int | None = None
    fit_sample_names_sha256: str | None = None
    graph_manifest_sha256: str | None = None
    winning_scale_divisor: float | None = None

    def __post_init__(self) -> None:
        if self.channel_names != FRANGI_RASTER_CHANNELS:
            raise ValueError(
                "Raster preprocessing channel order differs from the cache schema"
            )
        if len(set(self.log1p_channels)) != len(self.log1p_channels):
            raise ValueError("log1p_channels must be unique")
        invalid = sorted(set(self.log1p_channels) - set(self.channel_names))
        if invalid:
            raise ValueError(f"Unknown log1p channels: {invalid}")
        if len(self.offset) != len(self.channel_names) or len(self.scale) != len(
            self.channel_names
        ):
            raise ValueError("Raster offset and scale must have one value per channel")
        if not all(math.isfinite(value) for value in self.offset + self.scale):
            raise ValueError("Raster preprocessing values must be finite")
        if any(value <= 0.0 for value in self.scale):
            raise ValueError("Raster preprocessing scales must be positive")
        if self.format_version is not None and self.format_version != 1:
            raise ValueError("Unsupported raster preprocessing format_version")
        if self.fit_sample_count is not None and self.fit_sample_count <= 0:
            raise ValueError("Raster preprocessing fit_sample_count must be positive")
        for name, digest in (
            ("fit_sample_names_sha256", self.fit_sample_names_sha256),
            ("graph_manifest_sha256", self.graph_manifest_sha256),
        ):
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.winning_scale_divisor is not None:
            if (
                not math.isfinite(self.winning_scale_divisor)
                or self.winning_scale_divisor <= 0.0
            ):
                raise ValueError("winning_scale_divisor must be finite and positive")
            winning_index = self.channel_names.index("winning_scale")
            if self.scale[winning_index] != self.winning_scale_divisor:
                raise ValueError(
                    "winning_scale_divisor differs from its affine channel scale"
                )

    @classmethod
    def from_mapping(cls, value: object) -> "RasterPreprocessing":
        if not isinstance(value, Mapping):
            raise ValueError("Residual checkpoint has no raster_preprocessing mapping")
        required = {"channel_names", "log1p_channels", "offset", "scale"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"raster_preprocessing is missing: {missing}")
        try:
            return cls(
                channel_names=tuple(str(item) for item in value["channel_names"]),
                log1p_channels=tuple(str(item) for item in value["log1p_channels"]),
                offset=tuple(float(item) for item in value["offset"]),
                scale=tuple(float(item) for item in value["scale"]),
                format_version=(
                    None
                    if value.get("format_version") is None
                    else int(value["format_version"])
                ),
                fit_sample_count=(
                    None
                    if value.get("fit_sample_count") is None
                    else int(value["fit_sample_count"])
                ),
                fit_sample_names_sha256=(
                    None
                    if value.get("fit_sample_names_sha256") is None
                    else str(value["fit_sample_names_sha256"])
                ),
                graph_manifest_sha256=(
                    None
                    if value.get("graph_manifest_sha256") is None
                    else str(value["graph_manifest_sha256"])
                ),
                winning_scale_divisor=(
                    None
                    if value.get("winning_scale_divisor") is None
                    else float(value["winning_scale_divisor"])
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid raster_preprocessing values") from exc

    def as_dict(self) -> dict[str, object]:
        output: dict[str, object] = {
            "channel_names": list(self.channel_names),
            "log1p_channels": list(self.log1p_channels),
            "offset": list(self.offset),
            "scale": list(self.scale),
        }
        optional = {
            "format_version": self.format_version,
            "fit_sample_count": self.fit_sample_count,
            "fit_sample_names_sha256": self.fit_sample_names_sha256,
            "graph_manifest_sha256": self.graph_manifest_sha256,
            "winning_scale_divisor": self.winning_scale_divisor,
        }
        output.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return output

    def apply(self, raster: torch.Tensor) -> torch.Tensor:
        if raster.ndim != 4 or raster.shape[1] != len(self.channel_names):
            raise ValueError(
                "Frangi raster batch must have shape "
                f"(B,{len(self.channel_names)},H,W), got {tuple(raster.shape)}"
            )
        if not raster.is_floating_point() or not torch.isfinite(raster).all():
            raise ValueError("Frangi raster must be a finite floating-point tensor")
        output = raster.clone()
        bounded = {
            "similarity": (0.0, 1.0),
            "support": (0.0, 1.0),
            "orientation_sin2": (-1.0, 1.0),
            "orientation_cos2": (-1.0, 1.0),
            "distance_to_skeleton": (0.0, 1.0),
        }
        for name, (minimum, maximum) in bounded.items():
            index = self.channel_names.index(name)
            output[:, index] = output[:, index].clamp(minimum, maximum)
        if self.winning_scale_divisor is not None:
            winning_index = self.channel_names.index("winning_scale")
            if torch.any(output[:, winning_index] > self.winning_scale_divisor + 1e-5):
                raise ValueError(
                    "winning_scale exceeds the largest published extraction scale"
                )
        for name in self.log1p_channels:
            index = self.channel_names.index(name)
            if torch.any(output[:, index] < 0):
                raise ValueError(f"Cannot apply log1p to negative channel {name!r}")
            output[:, index] = torch.log1p(output[:, index])
        offset = output.new_tensor(self.offset).view(1, -1, 1, 1)
        scale = output.new_tensor(self.scale).view(1, -1, 1, 1)
        return (output - offset) / scale


def load_safe_torch_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load tensors/primitives only, allowing the historical TorchVersion tag."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            value = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Cannot safely load checkpoint {source}: {exc}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Checkpoint root must be a string-keyed mapping: {source}")
    return value


def validate_baseline_checkpoint(payload: Mapping[str, object]) -> tuple[int, float]:
    """Validate the historical LoRA payload and return its architecture values."""
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported baseline checkpoint format_version")
    if payload.get("variant") != "baseline":
        raise ValueError("Residual evaluation requires a baseline LoRA checkpoint")
    state = payload.get("adapter")
    lora = payload.get("lora")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Baseline checkpoint has no adapter state")
    if not isinstance(lora, Mapping):
        raise ValueError("Baseline checkpoint has no LoRA description")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("Baseline adapter must contain named tensors only")
    for name, tensor in state.items():
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(
                f"Baseline adapter tensor {name!r} is not finite floating point"
            )
    try:
        rank = int(lora["rank"])
        alpha = float(lora["alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid baseline LoRA rank/alpha") from exc
    if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Baseline LoRA rank and alpha must be positive")
    return rank, alpha


@dataclass(frozen=True)
class ResidualCheckpointSpec:
    adapter_state: Mapping[str, torch.Tensor]
    raster_channels: int
    high_resolution_channels: tuple[int, ...]
    hidden_channels: int
    adapter_mode: str
    profile_radii: tuple[float, ...]
    evidence_dilation: int
    evidence_threshold: float
    preprocessing: RasterPreprocessing
    training_raster_condition: str
    held_out_fold: int
    oof_training: Mapping[str, object]
    training_state: str


def validate_residual_checkpoint(
    payload: Mapping[str, object],
    *,
    baseline_checkpoint_sha256: str,
    graph_cache_manifest: Mapping[str, object],
    require_complete: bool = False,
    expected_oof_fold: int | None = None,
    expected_oof_role: str | None = None,
) -> ResidualCheckpointSpec:
    """Strictly bind residual weights to their baseline and Frangi semantics."""
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported residual checkpoint format_version")
    training_state = payload.get("training_state")
    if training_state not in ("incomplete", "complete"):
        raise ValueError("Residual checkpoint has no valid training_state")
    if require_complete and training_state != "complete":
        raise ValueError(
            "Residual checkpoint is incomplete and cannot be used for gate rows"
        )
    state = payload.get("residual_adapter")
    architecture = payload.get("residual")
    baseline = payload.get("baseline_adapter")
    cache = payload.get("graph_cache")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Residual checkpoint has no residual_adapter state")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("residual_adapter must contain named tensors only")
    if not isinstance(architecture, Mapping):
        raise ValueError("Residual checkpoint has no residual architecture")
    if not isinstance(baseline, Mapping):
        raise ValueError("Residual checkpoint is not bound to a baseline adapter")
    if baseline.get("sha256") != baseline_checkpoint_sha256:
        raise ValueError("Residual checkpoint was trained with another baseline")
    if not isinstance(cache, Mapping):
        raise ValueError("Residual checkpoint is not bound to a Frangi cache contract")
    for key in ("extractor_sha256", "frangi", "channels"):
        if cache.get(key) != graph_cache_manifest.get(key):
            raise ValueError(f"Residual and evaluation Frangi cache differ for {key}")
    try:
        raster_channels = int(architecture["raster_channels"])
        high_resolution_channels = tuple(
            int(value) for value in architecture["high_resolution_channels"]
        )
        hidden_channels = int(architecture["hidden_channels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid residual architecture") from exc
    if raster_channels != len(FRANGI_RASTER_CHANNELS):
        raise ValueError("Residual checkpoint does not consume the seven v2 channels")
    if not high_resolution_channels or any(
        value <= 0 for value in high_resolution_channels
    ):
        raise ValueError("Residual high-resolution feature channels are invalid")
    if hidden_channels <= 0:
        raise ValueError("Residual hidden_channels must be positive")
    adapter_mode = str(architecture.get("adapter_mode", LEGACY_ADAPTER_MODE))
    if adapter_mode not in RESIDUAL_ADAPTER_MODES:
        raise ValueError(f"Unknown residual adapter_mode: {adapter_mode!r}")
    if adapter_mode == VERIFIED_ADAPTER_MODE:
        try:
            profile_radii = tuple(float(value) for value in architecture["profile_radii"])
            evidence_dilation = int(architecture["evidence_dilation"])
            evidence_threshold = float(architecture["evidence_threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid verified-local residual architecture") from exc
        if not profile_radii or any(
            not math.isfinite(value) or value <= 0 for value in profile_radii
        ):
            raise ValueError("Residual profile_radii must be positive and finite")
        if evidence_dilation < 0:
            raise ValueError("Residual evidence_dilation cannot be negative")
        if not math.isfinite(evidence_threshold) or not 0 < evidence_threshold < 1:
            raise ValueError("Residual evidence_threshold must lie in (0, 1)")
    else:
        profile_radii = ()
        evidence_dilation = 0
        evidence_threshold = 0.5
    preprocessing = RasterPreprocessing.from_mapping(
        payload.get("raster_preprocessing")
    )
    run_contract = payload.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise ValueError("Residual checkpoint has no run_contract")
    try:
        contract_digest = hashlib.sha256(
            json.dumps(
                run_contract,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError("Residual run_contract is not canonical JSON") from exc
    if payload.get("run_contract_sha256") != contract_digest:
        raise ValueError("Residual run_contract SHA-256 mismatch")
    training_raster_condition = run_contract.get("raster_condition")
    if training_raster_condition not in ("correct", "no_evidence"):
        raise ValueError("Residual checkpoint has no valid raster_condition")
    if (
        adapter_mode == VERIFIED_ADAPTER_MODE
        and training_raster_condition != "correct"
    ):
        raise ValueError(
            "verified_local_v1 checkpoints must be trained with correct evidence"
        )
    if adapter_mode == VERIFIED_ADAPTER_MODE:
        contract_architecture = run_contract.get("residual")
        if not isinstance(contract_architecture, Mapping):
            raise ValueError("Verified residual run contract lacks its architecture")
        expected_verified = {
            "adapter_mode": adapter_mode,
            "profile_radii": list(profile_radii),
            "evidence_dilation": evidence_dilation,
            "evidence_threshold": evidence_threshold,
        }
        mismatches = {
            key: {
                "checkpoint": expected,
                "run_contract": contract_architecture.get(key),
            }
            for key, expected in expected_verified.items()
            if contract_architecture.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "Verified residual architecture differs from its run contract: "
                f"{mismatches}"
            )
    oof_training = validate_oof_run_contract(run_contract)
    held_out_fold = int(oof_training["held_out_fold"])
    validate_strict_oof_training_contract(
        oof_training,
        held_out_fold=expected_oof_fold,
        evaluation_role=expected_oof_role,
    )
    if preprocessing.log1p_channels != ("hessian_magnitude",):
        raise ValueError("Residual v1 preprocessing must log1p only hessian_magnitude")
    if (
        preprocessing.format_version != 1
        or preprocessing.fit_sample_count is None
        or preprocessing.fit_sample_names_sha256 is None
        or preprocessing.graph_manifest_sha256 is None
        or preprocessing.winning_scale_divisor is None
    ):
        raise ValueError("Residual checkpoint lacks fitted preprocessing provenance")
    training_manifest_sha = cache.get("manifest_sha256")
    if training_manifest_sha != preprocessing.graph_manifest_sha256:
        raise ValueError(
            "Residual graph-cache and normalization provenance do not match"
        )
    frangi = cache.get("frangi")
    if not isinstance(frangi, Mapping) or not isinstance(frangi.get("scales"), list):
        raise ValueError("Residual graph-cache contract has no Frangi scales")
    maximum_scale = max(float(value) for value in frangi["scales"])
    if preprocessing.winning_scale_divisor != maximum_scale:
        raise ValueError(
            "Residual winning-scale preprocessing differs from Frangi extraction"
        )
    for name, tensor in state.items():
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(
                f"Residual adapter tensor {name!r} is not finite floating point"
            )
    return ResidualCheckpointSpec(
        adapter_state=state,
        raster_channels=raster_channels,
        high_resolution_channels=high_resolution_channels,
        hidden_channels=hidden_channels,
        adapter_mode=adapter_mode,
        profile_radii=profile_radii,
        evidence_dilation=evidence_dilation,
        evidence_threshold=evidence_threshold,
        preprocessing=preprocessing,
        training_raster_condition=str(training_raster_condition),
        held_out_fold=held_out_fold,
        oof_training=oof_training,
        training_state=str(training_state),
    )


def resolve_raster_condition(
    training_condition: str,
    requested_condition: str | None,
    *,
    allow_causal_override: bool,
) -> tuple[str, bool]:
    """Resolve a safe evaluation condition from the checkpoint by default.

    The only permitted mismatch is the same-checkpoint input ablation
    ``correct`` to ``no_evidence``.  It is a necessity test, not an estimate of
    Frangi's total causal contribution. Feeding real evidence to a model trained
    without it is refused even when the override switch is present.
    """
    allowed = ("correct", "no_evidence")
    if training_condition not in allowed:
        raise ValueError(f"Unknown training raster condition: {training_condition!r}")
    if requested_condition is not None and requested_condition not in allowed:
        raise ValueError(f"Unknown requested raster condition: {requested_condition!r}")
    condition = (
        training_condition if requested_condition is None else requested_condition
    )
    changed = condition != training_condition
    if changed and not allow_causal_override:
        raise ValueError(
            "Raster condition differs from the residual checkpoint; pass "
            "--allow-input-ablation-raster-override only for an intentional "
            "same-checkpoint input ablation"
        )
    if training_condition == "no_evidence" and condition == "correct":
        raise ValueError(
            "A no_evidence control checkpoint must never receive correct Frangi maps"
        )
    return condition, changed


def load_graph_cache_manifest(
    cache_dir: str | os.PathLike[str],
    *,
    selected_names: Sequence[str],
    image_size: Sequence[int],
    noise_mode: str,
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    """Validate manifest structure, its ordered split hash, and selected entries."""
    root = Path(cache_dir)
    path = root / GRAPH_CACHE_MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Frangi graph cache manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Frangi graph cache manifest must be a JSON object")
    expected_scalars = {
        "schema_version": FRANGI_RASTER_SCHEMA_VERSION,
        "status": "complete",
        "image_size": [int(value) for value in image_size],
        "noise": normalize_noise_mode(noise_mode),
        "channels": list(FRANGI_RASTER_CHANNELS),
    }
    mismatches = {
        key: {"observed": manifest.get(key), "expected": value}
        for key, value in expected_scalars.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frangi graph cache contract differs: {mismatches}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Frangi graph cache has no entries")
    if manifest.get("sample_count") != len(entries):
        raise ValueError("Frangi graph cache sample_count differs from its entries")
    by_name: dict[str, Mapping[str, object]] = {}
    ordered_names: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Frangi graph cache entry must be an object")
        name = entry.get("case_name")
        if not isinstance(name, str) or not name or name in by_name:
            raise ValueError(f"Invalid or duplicate Frangi cache case: {name!r}")
        expected_relative = graph_cache_relative_path(name).as_posix()
        if entry.get("cache_file") != expected_relative:
            raise ValueError(f"Unsafe or inconsistent cache path for {name!r}")
        digest = entry.get("cache_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid cache digest for {name!r}")
        ordered_names.append(name)
        by_name[name] = entry
    if manifest.get("sample_names_sha256") != sample_names_sha256(ordered_names):
        raise ValueError("Frangi cache ordered sample hash is invalid")
    names = list(selected_names)
    if len(names) != len(set(names)):
        raise ValueError("Evaluation sample names contain duplicates")
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Frangi graph cache is missing selected cases: {missing[:5]}")
    return manifest, by_name


class GraphRasterEvaluationDataset(Dataset):
    """Attach verified raw Frangi rasters and gate statistics to image samples."""

    def __init__(
        self,
        base: CrackSegmentationDataset,
        cache_dir: str | os.PathLike[str],
        *,
        verify_cache_hashes: bool = True,
        verify_data_hashes: bool = True,
    ) -> None:
        self.base = base
        self.cache_dir = Path(cache_dir)
        self.sample_names = list(base.sample_names)
        self.manifest, self.entries = load_graph_cache_manifest(
            self.cache_dir,
            selected_names=self.sample_names,
            image_size=base.image_size,
            noise_mode=base.noise_mode,
        )
        self.verify_cache_hashes = bool(verify_cache_hashes)
        self.verify_data_hashes = bool(verify_data_hashes)

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name = self.sample_names[index]
        sample = dict(self.base[index])
        if sample.get("case_name") != name:
            raise RuntimeError("Base dataset order changed after cache validation")
        entry = self.entries[name]
        path = self.cache_dir / graph_cache_relative_path(name)
        if self.verify_cache_hashes and sha256_file(path) != entry["cache_sha256"]:
            raise ValueError(f"Frangi cache file SHA-256 mismatch: {path}")
        graph = load_graph_raster(
            path,
            expected_case_name=name,
            expected_image_sha256=str(entry["image_sha256"]),
            expected_mask_sha256=str(entry["mask_sha256"]),
            expected_image_size=self.base.image_size,
        )
        if self.verify_data_hashes:
            image_path, mask_path = self.base._paths(name)
            if sha256_file(image_path) != graph.image_sha256:
                raise ValueError(
                    f"Source image changed since Frangi extraction: {name}"
                )
            if sha256_file(mask_path) != graph.mask_sha256:
                raise ValueError(f"Source mask changed since Frangi extraction: {name}")
        statistics = graph.statistics()["gate_features"]
        if not isinstance(statistics, Mapping):
            raise RuntimeError("Frangi cache produced invalid gate statistics")
        sample["frangi_raster"] = torch.from_numpy(graph.raster.copy())
        sample["frangi_similarity"] = torch.tensor(
            float(statistics["similarity"]), dtype=torch.float64
        )
        sample["frangi_density"] = torch.tensor(
            float(statistics["density"]), dtype=torch.float64
        )
        return sample


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize candidate behavior without fitting or choosing any gate."""
    if not rows:
        raise ValueError("Cannot summarize empty residual evaluation rows")
    normalized = [normalize_evaluation_row(row) for row in rows]
    summary: dict[str, object] = {
        "samples": len(normalized),
        "gate_feature_names": list(DEFAULT_GATE_FEATURES),
        "gate_feature_count": len(DEFAULT_GATE_FEATURES),
        "gate_threshold_selected": False,
    }
    for field in METRIC_FIELDS:
        values = np.asarray([float(row[field]) for row in normalized])
        summary[f"{field}_mean"] = float(values.mean())
        summary[f"{field}_p05"] = float(np.quantile(values, 0.05))
        summary[f"{field}_p95"] = float(np.quantile(values, 0.95))
    for field in LABEL_FIELDS:
        summary[f"{field}_count"] = int(sum(int(row[field]) for row in normalized))
        summary[f"{field}_rate"] = float(
            np.mean([int(row[field]) for row in normalized])
        )
    return summary


def evaluate_residual_loader(
    model: torch.nn.Module,
    loader: Any,
    *,
    preprocessing: RasterPreprocessing,
    source_groups: Mapping[str, str],
    dataset: str,
    role: str,
    fold: str,
    progress_path: str | os.PathLike[str],
    device: str | torch.device,
    amp_dtype: str = "bfloat16",
    segmentation_threshold: float = 0.5,
    label_minimum_gain: float = 0.0,
    raster_condition: str = "correct",
    show_progress: bool = True,
) -> int:
    """Run residual inference and journal complete batches.

    The function accepts any model implementing the small residual output
    contract, which keeps its behavior testable without importing or building
    SAM 2.
    """
    torch_device = torch.device(device)
    if amp_dtype not in ("bfloat16", "float16", "none"):
        raise ValueError(f"Unknown amp_dtype: {amp_dtype!r}")
    if role not in EVALUATION_ROLES:
        raise ValueError(f"Unknown evaluation role: {role!r}")
    if raster_condition not in ("correct", "no_evidence"):
        raise ValueError("raster_condition must be 'correct' or 'no_evidence'")
    support_index = FRANGI_RASTER_CHANNELS.index("support")
    completed = 0
    model.eval()
    progress = tqdm(
        loader,
        desc=dataset,
        unit="batch",
        disable=not show_progress,
    )
    with torch.inference_mode():
        for batch in progress:
            images = batch["image"].to(torch_device, non_blocking=True)
            raw_rasters = batch["frangi_raster"]
            if raster_condition == "no_evidence":
                height, width = (int(value) for value in raw_rasters.shape[-2:])
                absent = torch.from_numpy(no_frangi_evidence_raster(height, width)).to(
                    dtype=raw_rasters.dtype
                )
                raw_rasters = absent.unsqueeze(0).expand(
                    raw_rasters.shape[0], -1, -1, -1
                )
            rasters = preprocessing.apply(raw_rasters).to(
                torch_device, non_blocking=True
            )
            targets = batch["mask"]
            if torch_device.type == "cuda" and amp_dtype != "none":
                dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
                cast = torch.autocast(device_type="cuda", dtype=dtype)
            else:
                cast = nullcontext()
            with cast:
                output = model(
                    images,
                    rasters,
                    output_size=tuple(int(value) for value in targets.shape[-2:]),
                    accept_residual=True,
                )
            if not isinstance(output, Mapping):
                raise ValueError("Residual model output must be a mapping")
            try:
                baseline_logits = output["baseline_logits"].float().cpu()
                candidate_logits = output["candidate_logits"].float().cpu()
            except (KeyError, AttributeError) as exc:
                raise ValueError(
                    "Residual output must contain baseline_logits and candidate_logits"
                ) from exc
            if baseline_logits.shape != candidate_logits.shape:
                raise ValueError("Residual baseline/candidate output shapes differ")
            names = list(batch["case_name"])
            if baseline_logits.shape[0] != len(names):
                raise ValueError("Residual output batch size differs from case names")
            score_source = next(
                (
                    name
                    for name in (
                        "evidence_score_fusion",
                        "evidence_score",
                        "evidence_probability",
                    )
                    if name in output
                ),
                None,
            )
            selector_keys = {
                "evidence_score_fusion",
                "evidence_score",
                "evidence_probability",
                "evidence_support_fusion",
                "evidence_support",
                "evidence_selected_fusion",
                "evidence_selected",
                "correction_envelope_fusion",
                "correction_envelope",
            }
            present_selector_keys = selector_keys.intersection(output)
            if present_selector_keys and score_source is None:
                raise ValueError(
                    "Residual output exposes selector maps without an evidence score"
                )
            selector_diagnostics: list[dict[str, object]] = []
            if score_source is not None:
                support_key = (
                    "evidence_support_fusion"
                    if "evidence_support_fusion" in output
                    else "evidence_support"
                )
                selected_key = (
                    "evidence_selected_fusion"
                    if "evidence_selected_fusion" in output
                    else "evidence_selected"
                )
                envelope_fusion_key = (
                    "correction_envelope_fusion"
                    if "correction_envelope_fusion" in output
                    else "correction_envelope"
                )
                missing_selector_keys = [
                    key
                    for key in (
                        support_key,
                        selected_key,
                        envelope_fusion_key,
                        "evidence_selected",
                        "correction_envelope",
                    )
                    if key not in output
                ]
                if missing_selector_keys:
                    raise ValueError(
                        "Residual selector output is incomplete; missing "
                        + ", ".join(missing_selector_keys)
                    )
            rows: list[dict[str, object]] = []
            for index, name in enumerate(names):
                if name not in source_groups:
                    raise ValueError(f"Missing physical source group for {name!r}")
                raw_support = raw_rasters[index, support_index].unsqueeze(0)
                gate_spatial_support = (
                    output["evidence_selected"][index]
                    if score_source is not None
                    else raw_support
                )
                rows.append(
                    build_evaluation_row(
                        case_name=name,
                        source_group=source_groups[name],
                        dataset=dataset,
                        role=role,
                        fold=str(fold),
                        baseline_logits=baseline_logits[index],
                        candidate_logits=candidate_logits[index],
                        target=targets[index],
                        frangi_stats={
                            "similarity": (
                                float(batch["frangi_similarity"][index])
                                if raster_condition == "correct"
                                else 0.0
                            ),
                            "density": (
                                float(batch["frangi_density"][index])
                                if raster_condition == "correct"
                                else 0.0
                            ),
                        },
                        frangi_support=gate_spatial_support,
                        segmentation_threshold=segmentation_threshold,
                        label_minimum_gain=label_minimum_gain,
                    )
                )
                if score_source is not None:
                    residual_logits = output.get("residual_logits")
                    if residual_logits is None:
                        residual_value = (
                            candidate_logits[index] - baseline_logits[index]
                        )
                    else:
                        residual_value = residual_logits[index]
                    selector_diagnostics.append(
                        build_selector_diagnostic(
                            case_name=name,
                            evidence_score_source=score_source,
                            gate_spatial_support_source=(
                                "accepted_local_selector_output"
                            ),
                            evidence_support_fusion=output[support_key][index],
                            evidence_score_fusion=output[score_source][index],
                            evidence_selected_fusion=output[selected_key][index],
                            correction_envelope_fusion=output[
                                envelope_fusion_key
                            ][index],
                            correction_envelope_output=output[
                                "correction_envelope"
                            ][index],
                            residual_logits=residual_value,
                            target=targets[index],
                        )
                    )
            append_progress_batch(
                progress_path,
                dataset=dataset,
                rows=rows,
                selector_diagnostics=(selector_diagnostics or None),
            )
            completed += len(rows)
    return completed


__all__ = [
    "EVALUATION_ROLES",
    "EVALUATION_SCHEMA",
    "EVALUATION_SCHEMA_VERSION",
    "GraphRasterEvaluationDataset",
    "LABEL_FIELDS",
    "METRIC_FIELDS",
    "ROW_FIELDS",
    "SELECTOR_DIAGNOSTIC_FIELDS",
    "SELECTOR_DIAGNOSTICS_SCHEMA",
    "SELECTOR_DIAGNOSTICS_SCHEMA_VERSION",
    "RasterPreprocessing",
    "ResidualCheckpointSpec",
    "append_progress_batch",
    "build_evaluation_row",
    "build_selector_diagnostic",
    "ensure_evaluation_contract",
    "evaluate_residual_loader",
    "file_identity",
    "load_graph_cache_manifest",
    "load_group_assignment_records",
    "load_group_assignments",
    "load_safe_torch_checkpoint",
    "normalize_evaluation_row",
    "normalize_selector_diagnostic",
    "read_progress_rows",
    "read_selector_diagnostics",
    "resolve_raster_condition",
    "summarize_rows",
    "validate_baseline_checkpoint",
    "validate_residual_checkpoint",
    "write_json_atomic",
    "write_rows_csv_atomic",
    "write_selector_diagnostics_atomic",
]
