#!/usr/bin/env python3
"""Apply one frozen logistic gate to completed residual-evaluation tables.

This command is intentionally analytical.  It never fits a model, changes the
serialized threshold, or inspects target masks.  The target-derived IoU/Dice
columns already present in ``per_image.csv`` are used only after the frozen gate
has made its decision from :data:`DEFAULT_GATE_FEATURES`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from cracksam2.gating import (
    DEFAULT_GATE_FEATURES,
    LABEL_DEFINITION,
    LogisticConfidenceGate,
    inverse_group_frequency_weights,
    probability_reliability_metrics,
    vectorize_feature_rows,
)
from cracksam2.oof import validate_strict_oof_training_contract


ANALYSIS_SCHEMA = "cracksam2.logistic-gate-evaluation"
ANALYSIS_SCHEMA_VERSION = 1
OOF_SCHEMA = "cracksam2.logistic-gate-oof-data"
OOF_SCHEMA_VERSION = 2
RISK_CURVE_GRID_POINTS = 101
EVALUATION_SCHEMA = "cracksam2.frangigraph-residual-evaluation"
EVALUATION_SCHEMA_VERSION = 1
ALLOWED_EVALUATION_ROLES = frozenset(
    ("gate_fit", "gate_calibration", "development", "historical_test")
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
INPUT_ROW_FIELDS: tuple[str, ...] = (
    "case_name",
    "source_group",
    "dataset",
    "role",
    "fold",
    *METRIC_FIELDS,
    *LABEL_FIELDS,
    *DEFAULT_GATE_FEATURES,
)
GATED_FIELDS: tuple[str, ...] = (
    *INPUT_ROW_FIELDS,
    "gate_probability",
    "gate_threshold",
    "gate_open",
    "gate_target_positive",
    "selected_output",
    "gated_iou",
    "gated_iou_gain",
    "gated_dice",
    "gated_dice_gain",
)
RISK_FIELDS: tuple[str, ...] = (
    "scope",
    "dataset",
    "threshold",
    "is_serialized_gate_threshold",
    "samples",
    "selected",
    "coverage",
    "group_balanced_coverage",
    "precision",
    "recall",
    "selected_mean_delta_iou",
    "selected_harm_rate",
    "selected_risk",
    "system_mean_gain_iou",
    "system_p05_gain_iou",
    "system_severe_loss_rate_005",
    "system_severe_loss_rate_010",
)


@dataclass(frozen=True)
class OofCompatibility:
    """Small schema-independent compatibility view of the OOF manifest.

    OOF manifest authoring is maintained by the gate-training pipeline.  This
    adapter deliberately extracts only the immutable quantities consumed here,
    so a future envelope change is localized to :func:`oof_compatibility`.
    """

    baseline_checkpoint_sha256: str
    residual_checkpoint_sha256_by_fold: Mapping[str, str]
    frangi_extractor_sha256: str
    frangi_cache_manifest_sha256: str
    protocol_sha256: str
    frangi_parameters: Mapping[str, object]
    channels: tuple[str, ...]
    feature_names: tuple[str, ...]
    label_minimum_gain: float
    segmentation_threshold: float


@dataclass(frozen=True)
class EvaluationInput:
    root: Path
    contract: Mapping[str, object]
    contract_sha256: str
    csv_sha256: str
    rows: tuple[dict[str, object], ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--oof-manifest", type=Path, required=True)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Completed residual-evaluation directory. Repeat for several "
            "datasets; each directory needs evaluation_contract.json and "
            "per_image.csv."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_names_sha256(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_sha256(value: object, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Contract value is not canonical finite JSON") from exc


def oof_compatibility(manifest: Mapping[str, object]) -> OofCompatibility:
    """Normalize the strict OOF manifest into evaluator-facing invariants.

    Keeping this parsing in one helper makes a future envelope migration local;
    downstream analytical code consumes only :class:`OofCompatibility`.
    """
    required = {
        "schema",
        "schema_version",
        "feature_names",
        "label_minimum_gain",
        "segmentation_threshold",
        "artifacts",
        "folds",
        "outputs",
    }
    if set(manifest) != required:
        raise ValueError("OOF manifest has missing or unknown top-level fields")
    if manifest.get("schema") != OOF_SCHEMA:
        raise ValueError("Unknown OOF manifest schema")
    if manifest.get("schema_version") != OOF_SCHEMA_VERSION:
        raise ValueError("Unsupported OOF manifest schema version")
    artifacts = _mapping(manifest.get("artifacts"), "OOF artifacts")
    if set(artifacts) != {
        "sam2_checkpoint",
        "baseline_checkpoint",
        "graph_cache",
        "protocol",
    }:
        raise ValueError("OOF artifacts have missing or unknown fields")
    baseline = _mapping(artifacts.get("baseline_checkpoint"), "OOF baseline")
    graph_cache = _mapping(artifacts.get("graph_cache"), "OOF graph cache")
    protocol = _mapping(artifacts.get("protocol"), "OOF protocol")
    cache_manifest = _mapping(graph_cache.get("manifest"), "OOF cache manifest")
    parameters = _mapping(graph_cache.get("frangi"), "OOF Frangi parameters")
    if "scales" not in parameters:
        raise ValueError("OOF Frangi parameters have no scale list")
    features = manifest.get("feature_names")
    if not isinstance(features, list) or not all(
        isinstance(item, str) for item in features
    ):
        raise ValueError("OOF feature_names must be a string list")
    feature_names = tuple(features)
    if feature_names != DEFAULT_GATE_FEATURES:
        raise ValueError(
            "OOF manifest gate features differ from DEFAULT_GATE_FEATURES: "
            f"{feature_names}"
        )
    channels = graph_cache.get("channels")
    if not isinstance(channels, list) or not channels or not all(
        isinstance(channel, str) for channel in channels
    ):
        raise ValueError("OOF graph-cache channels must be a non-empty string list")
    label_minimum_gain = float(manifest["label_minimum_gain"])
    segmentation_threshold = float(manifest["segmentation_threshold"])
    if not math.isfinite(label_minimum_gain):
        raise ValueError("OOF label_minimum_gain must be finite")
    if not 0.0 < segmentation_threshold < 1.0:
        raise ValueError("OOF segmentation_threshold must lie in (0, 1)")

    folds = _mapping(manifest.get("folds"), "OOF folds")
    if set(folds) != {"0", "1", "2", "3", "4"}:
        raise ValueError("OOF manifest must contain exactly folds 0 through 4")
    residual_checkpoint_sha256_by_fold: dict[str, str] = {}
    for fold, value in folds.items():
        fold_contract = _mapping(value, f"OOF fold {fold}")
        expected_role = "gate_calibration" if fold == "4" else "gate_fit"
        if fold_contract.get("role") != expected_role:
            raise ValueError(f"OOF fold {fold} has the wrong role")
        residual_checkpoint = _mapping(
            fold_contract.get("residual_checkpoint"),
            f"OOF fold {fold} residual checkpoint",
        )
        residual_checkpoint_sha256_by_fold[fold] = _validate_sha256(
            residual_checkpoint.get("sha256"),
            f"OOF fold {fold} residual checkpoint",
        )
        try:
            validate_strict_oof_training_contract(
                fold_contract.get("oof_training"),
                held_out_fold=int(fold),
                evaluation_role=expected_role,
            )
        except ValueError as exc:
            raise ValueError(
                f"OOF fold {fold} violates the strict OOF training contract: {exc}"
            ) from exc
    if len(set(residual_checkpoint_sha256_by_fold.values())) != len(folds):
        raise ValueError("OOF manifest must bind five distinct residual checkpoints")

    outputs = _mapping(manifest.get("outputs"), "OOF outputs")
    if set(outputs) != {"gate_fit_csv", "gate_calibration_csv"}:
        raise ValueError("OOF outputs must identify fit and calibration CSVs")
    return OofCompatibility(
        baseline_checkpoint_sha256=_validate_sha256(
            baseline.get("sha256"), "OOF baseline"
        ),
        residual_checkpoint_sha256_by_fold=residual_checkpoint_sha256_by_fold,
        frangi_extractor_sha256=_validate_sha256(
            graph_cache.get("extractor_sha256"), "OOF extractor"
        ),
        frangi_cache_manifest_sha256=_validate_sha256(
            cache_manifest.get("sha256"), "OOF graph-cache manifest"
        ),
        protocol_sha256=_validate_sha256(
            protocol.get("composite_sha256"), "OOF protocol"
        ),
        frangi_parameters=dict(parameters),
        channels=tuple(channels),
        feature_names=feature_names,
        label_minimum_gain=label_minimum_gain,
        segmentation_threshold=segmentation_threshold,
    )


def _provenance_oof_sha256(gate: LogisticConfidenceGate) -> tuple[str, str]:
    if gate.provenance is None:
        raise ValueError("Gate has no strict provenance")
    values = gate.provenance.to_dict()
    for key in (
        "oof_manifest_sha256",
        "oof_evaluation_manifest_sha256",
        "oof_protocol_manifest_sha256",
    ):
        if key in values:
            return _validate_sha256(values[key], f"gate provenance {key}"), key
    # Schema-v2 gates used protocol_sha256 for the only protocol artifact.  The
    # fallback is intentionally strict: an older gate whose hash points to the
    # static split protocol, rather than this OOF manifest, is rejected by the
    # byte comparison in ``run`` and must be retrained under the new schema.
    if "protocol_sha256" in values:
        return (
            _validate_sha256(values["protocol_sha256"], "gate protocol_sha256"),
            "protocol_sha256 (legacy OOF binding)",
        )
    raise ValueError("Gate provenance does not bind an OOF manifest SHA-256")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Evaluation contract has no valid {label}")
    return value


def validate_evaluation_contract(
    contract: Mapping[str, object],
    compatibility: OofCompatibility,
) -> Mapping[str, object]:
    """Validate one residual table before any gate probabilities are computed."""
    if contract.get("schema") != EVALUATION_SCHEMA:
        raise ValueError("Unknown residual evaluation schema")
    if contract.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("Unsupported residual evaluation schema version")
    dataset = _mapping(contract.get("dataset"), "dataset contract")
    checkpoints = _mapping(contract.get("checkpoints"), "checkpoint contract")
    baseline = _mapping(checkpoints.get("baseline"), "baseline identity")
    residual = _mapping(checkpoints.get("residual"), "residual identity")
    graph_cache = _mapping(contract.get("graph_cache"), "graph-cache contract")
    gate_policy = _mapping(contract.get("gate_policy"), "gate policy")

    role = dataset.get("role")
    fold = dataset.get("fold")
    if role not in ALLOWED_EVALUATION_ROLES:
        raise ValueError(f"Unknown evaluation role in contract: {role!r}")
    if not isinstance(fold, str):
        raise ValueError("Evaluation role/fold must use a string fold")
    if role == "gate_fit":
        if fold not in ("0", "1", "2", "3"):
            raise ValueError("Evaluation role/fold is incoherent for gate_fit")
        checkpoint_fold = fold
    elif role == "gate_calibration":
        if fold != "4":
            raise ValueError(
                "Evaluation role/fold is incoherent for gate_calibration"
            )
        checkpoint_fold = "4"
    else:
        if fold != "":
            raise ValueError(
                f"Evaluation role/fold is incoherent for analytical role {role}"
            )
        # Fold 4 is the calibrated, deployable residual producer.  Analytical
        # development and historical-test rows must evaluate that exact model.
        checkpoint_fold = "4"

    residual_sha256 = _validate_sha256(
        residual.get("sha256"), "evaluation residual checkpoint"
    )

    mismatches: dict[str, dict[str, object]] = {}
    expected = {
        "baseline_checkpoint_sha256": compatibility.baseline_checkpoint_sha256,
        "residual_checkpoint_sha256": (
            compatibility.residual_checkpoint_sha256_by_fold[checkpoint_fold]
        ),
        "frangi_extractor_sha256": compatibility.frangi_extractor_sha256,
        "frangi_parameters": dict(compatibility.frangi_parameters),
        "graph_channels": list(compatibility.channels),
        "segmentation_threshold": compatibility.segmentation_threshold,
    }
    observed = {
        "baseline_checkpoint_sha256": baseline.get("sha256"),
        "residual_checkpoint_sha256": residual_sha256,
        "frangi_extractor_sha256": graph_cache.get("extractor_sha256"),
        "frangi_parameters": graph_cache.get("frangi"),
        "graph_channels": graph_cache.get("channels"),
        "segmentation_threshold": contract.get("segmentation_threshold"),
    }
    for key, expected_value in expected.items():
        if _canonical(observed[key]) != _canonical(expected_value):
            mismatches[key] = {
                "observed": observed[key],
                "expected": expected_value,
            }
    if mismatches:
        raise ValueError(f"Evaluation contract differs from OOF manifest: {mismatches}")
    if gate_policy.get("feature_rows_only") is not True:
        raise ValueError("Evaluation contract does not certify feature-only gate rows")
    if gate_policy.get("threshold_selected_by_this_command") is not False:
        raise ValueError("Evaluation command selected or recalibrated a gate threshold")
    if not isinstance(dataset.get("name"), str) or not dataset.get("name"):
        raise ValueError("Evaluation contract has no dataset name")
    return dataset


def _finite_float(value: object, field: str, path: Path, line: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} at {path}:{line}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field} at {path}:{line}")
    return number


def _normalize_input_row(
    row: Mapping[str, object], path: Path, line: int
) -> dict[str, object]:
    missing = [field for field in INPUT_ROW_FIELDS if field not in row]
    extra = sorted(set(row) - set(INPUT_ROW_FIELDS))
    if missing or extra:
        raise ValueError(
            f"Residual CSV fields differ at {path}:{line}; "
            f"missing={missing}, extra={extra}"
        )
    output: dict[str, object] = {}
    for field in ("case_name", "source_group", "dataset", "role", "fold"):
        value = row[field]
        if not isinstance(value, str) or (field != "fold" and not value):
            raise ValueError(f"Invalid text field {field!r} at {path}:{line}")
        output[field] = value
    if output["role"] not in ALLOWED_EVALUATION_ROLES:
        raise ValueError(f"Unknown evaluation role at {path}:{line}")
    for field in METRIC_FIELDS + DEFAULT_GATE_FEATURES:
        output[field] = _finite_float(row[field], field, path, line)
    for field in LABEL_FIELDS:
        number = _finite_float(row[field], field, path, line)
        integer = int(number)
        if number != integer or integer not in (0, 1):
            raise ValueError(f"Label {field!r} must be binary at {path}:{line}")
        output[field] = integer

    for field in ("baseline_iou", "candidate_iou", "baseline_dice", "candidate_dice"):
        if not 0.0 <= float(output[field]) <= 1.0:
            raise ValueError(f"Metric {field!r} is outside [0, 1] at {path}:{line}")
    bounded_features = {
        "relevant_baseline_entropy_mean": (0.0, 1.0),
        "baseline_foreground_fraction": (0.0, 1.0),
        "relevant_prediction_disagreement_rate": (0.0, 1.0),
        "support_correction_probability_mean": (0.0, 1.0),
        "foreground_probability_change_mean": (-1.0, 1.0),
        "frangi_similarity_support_mean": (0.0, 1.0),
        "frangi_density": (0.0, 1.0),
    }
    for field, (minimum, maximum) in bounded_features.items():
        if not minimum <= float(output[field]) <= maximum:
            raise ValueError(
                f"Gate feature {field!r} is outside [{minimum}, {maximum}] "
                f"at {path}:{line}"
            )
    tolerance = 2e-12
    if not math.isclose(
        float(output["candidate_iou_gain"]),
        float(output["candidate_iou"]) - float(output["baseline_iou"]),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(f"candidate_iou_gain is inconsistent at {path}:{line}")
    if not math.isclose(
        float(output["delta_iou"]),
        float(output["candidate_iou_gain"]),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(f"delta_iou is inconsistent at {path}:{line}")
    if not math.isclose(
        float(output["candidate_dice_gain"]),
        float(output["candidate_dice"]) - float(output["baseline_dice"]),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(f"candidate_dice_gain is inconsistent at {path}:{line}")
    return {field: output[field] for field in INPUT_ROW_FIELDS}


def load_evaluation_input(
    root: Path,
    compatibility: OofCompatibility,
) -> EvaluationInput:
    resolved = root.expanduser().resolve()
    contract_path = resolved / "evaluation_contract.json"
    csv_path = resolved / "per_image.csv"
    contract = _read_json_object(contract_path, "evaluation contract")
    dataset_contract = validate_evaluation_contract(contract, compatibility)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != INPUT_ROW_FIELDS:
                raise ValueError(
                    f"Residual CSV header differs from the gate contract: {csv_path}"
                )
            rows = tuple(
                _normalize_input_row(dict(row), csv_path, line)
                for line, row in enumerate(reader, start=2)
            )
    except OSError as exc:
        raise ValueError(f"Cannot read residual evaluation CSV: {csv_path}") from exc
    if not rows:
        raise ValueError(f"Residual evaluation CSV is empty: {csv_path}")

    dataset_name = str(dataset_contract["name"])
    role = str(dataset_contract["role"])
    fold = str(dataset_contract.get("fold", ""))
    if any(row["dataset"] != dataset_name for row in rows):
        raise ValueError(f"CSV dataset differs from evaluation contract: {csv_path}")
    if any(row["role"] != role or row["fold"] != fold for row in rows):
        raise ValueError(f"CSV role/fold differs from evaluation contract: {csv_path}")
    names = [str(row["case_name"]) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate case_name in residual CSV: {csv_path}")
    if dataset_contract.get("selected_samples") != len(rows):
        raise ValueError(f"Residual CSV row count differs from contract: {csv_path}")
    if dataset_contract.get("selected_sample_names_sha256") != sample_names_sha256(names):
        raise ValueError(f"Residual CSV sample order differs from contract: {csv_path}")
    return EvaluationInput(
        root=resolved,
        contract=contract,
        contract_sha256=sha256_file(contract_path),
        csv_sha256=sha256_file(csv_path),
        rows=rows,
    )


def _qualified_groups(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(f"{row['dataset']}::{row['source_group']}" for row in rows)


def _weighted_rate(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values.astype(np.float64), weights=weights))


def _decision_statistics(
    rows: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    decisions: np.ndarray,
    *,
    label_minimum_gain: float,
) -> dict[str, object]:
    if not rows or probabilities.shape != decisions.shape or len(rows) != probabilities.size:
        raise ValueError("Decision statistics received inconsistent non-empty arrays")
    delta_iou = np.asarray([float(row["delta_iou"]) for row in rows])
    delta_dice = np.asarray([float(row["candidate_dice_gain"]) for row in rows])
    baseline_iou = np.asarray([float(row["baseline_iou"]) for row in rows])
    candidate_iou = np.asarray([float(row["candidate_iou"]) for row in rows])
    baseline_dice = np.asarray([float(row["baseline_dice"]) for row in rows])
    candidate_dice = np.asarray([float(row["candidate_dice"]) for row in rows])
    labels = delta_iou > label_minimum_gain
    weights = inverse_group_frequency_weights(_qualified_groups(rows))
    gated_delta_iou = np.where(decisions, delta_iou, 0.0)
    gated_delta_dice = np.where(decisions, delta_dice, 0.0)
    gated_iou = np.where(decisions, candidate_iou, baseline_iou)
    gated_dice = np.where(decisions, candidate_dice, baseline_dice)
    selected_weights = weights[decisions]
    selected_labels = labels[decisions]
    positive_weight = float(weights[labels].sum())
    true_positive_weight = float(weights[decisions & labels].sum())
    precision = (
        float(np.average(selected_labels, weights=selected_weights))
        if selected_weights.size
        else 0.0
    )
    recall = true_positive_weight / positive_weight if positive_weight > 0.0 else 0.0
    reliability = probability_reliability_metrics(
        labels.astype(np.int64), probabilities, sample_weight=weights
    )
    return {
        "samples": len(rows),
        "source_groups": len(set(_qualified_groups(rows))),
        "selected": int(np.count_nonzero(decisions)),
        "coverage": float(np.mean(decisions)),
        "group_balanced_coverage": _weighted_rate(decisions, weights),
        "positives": int(np.count_nonzero(labels)),
        "true_positive": int(np.count_nonzero(decisions & labels)),
        "false_positive": int(np.count_nonzero(decisions & ~labels)),
        "false_negative": int(np.count_nonzero(~decisions & labels)),
        "precision": precision,
        "recall": recall,
        "precision_unweighted": (
            float(np.mean(selected_labels)) if selected_labels.size else 0.0
        ),
        "recall_unweighted": (
            float(np.count_nonzero(decisions & labels) / np.count_nonzero(labels))
            if np.any(labels)
            else 0.0
        ),
        **reliability,
        "baseline_iou_mean": float(np.mean(baseline_iou)),
        "candidate_iou_mean": float(np.mean(candidate_iou)),
        "gated_iou_mean": float(np.mean(gated_iou)),
        "gated_iou_gain_mean": float(np.mean(gated_delta_iou)),
        "gated_iou_gain_group_balanced_mean": float(
            np.average(gated_delta_iou, weights=weights)
        ),
        "gated_iou_gain_p05": float(np.quantile(gated_delta_iou, 0.05)),
        "baseline_dice_mean": float(np.mean(baseline_dice)),
        "candidate_dice_mean": float(np.mean(candidate_dice)),
        "gated_dice_mean": float(np.mean(gated_dice)),
        "gated_dice_gain_mean": float(np.mean(gated_delta_dice)),
        "gated_dice_gain_p05": float(np.quantile(gated_delta_dice, 0.05)),
        "loss_below_minus_0_05_count": int(np.count_nonzero(gated_delta_iou < -0.05)),
        "loss_below_minus_0_05_rate": float(np.mean(gated_delta_iou < -0.05)),
        "loss_below_minus_0_10_count": int(np.count_nonzero(gated_delta_iou < -0.10)),
        "loss_below_minus_0_10_rate": float(np.mean(gated_delta_iou < -0.10)),
    }


def _risk_row(
    rows: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    *,
    threshold: float,
    serialized_threshold: float,
    label_minimum_gain: float,
    scope: str,
    dataset: str,
) -> dict[str, object]:
    delta = np.asarray([float(row["delta_iou"]) for row in rows])
    labels = delta > label_minimum_gain
    weights = inverse_group_frequency_weights(_qualified_groups(rows))
    selected = probabilities >= threshold
    gated_delta = np.where(selected, delta, 0.0)
    selected_weights = weights[selected]
    if selected_weights.size:
        precision = float(np.average(labels[selected], weights=selected_weights))
        selected_mean_delta = float(np.average(delta[selected], weights=selected_weights))
        selected_harm_rate = float(
            np.average(delta[selected] < 0.0, weights=selected_weights)
        )
        selected_risk = float(
            np.average(np.maximum(-delta[selected], 0.0), weights=selected_weights)
        )
    else:
        precision = selected_mean_delta = selected_harm_rate = selected_risk = 0.0
    positive_weight = float(weights[labels].sum())
    recall = (
        float(weights[selected & labels].sum()) / positive_weight
        if positive_weight > 0.0
        else 0.0
    )
    return {
        "scope": scope,
        "dataset": dataset,
        "threshold": float(threshold),
        "is_serialized_gate_threshold": int(threshold == serialized_threshold),
        "samples": len(rows),
        "selected": int(np.count_nonzero(selected)),
        "coverage": float(np.mean(selected)),
        "group_balanced_coverage": _weighted_rate(selected, weights),
        "precision": precision,
        "recall": recall,
        "selected_mean_delta_iou": selected_mean_delta,
        "selected_harm_rate": selected_harm_rate,
        "selected_risk": selected_risk,
        "system_mean_gain_iou": float(np.average(gated_delta, weights=weights)),
        "system_p05_gain_iou": float(np.quantile(gated_delta, 0.05)),
        "system_severe_loss_rate_005": float(np.mean(gated_delta < -0.05)),
        "system_severe_loss_rate_010": float(np.mean(gated_delta < -0.10)),
    }


def risk_coverage_rows(
    rows: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    *,
    gate_threshold: float,
    label_minimum_gain: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    scopes: list[tuple[str, str, np.ndarray]] = [
        ("overall", "", np.arange(len(rows), dtype=np.int64))
    ]
    scopes.extend(
        (
            "dataset",
            dataset,
            np.asarray(
                [index for index, row in enumerate(rows) if row["dataset"] == dataset],
                dtype=np.int64,
            ),
        )
        for dataset in datasets
    )
    for scope, dataset, indices in scopes:
        scope_rows = [rows[int(index)] for index in indices]
        scope_probabilities = probabilities[indices]
        # A fixed probability grid bounds analytical cost independently of the
        # number of images.  The exact serialized threshold is always added.
        thresholds = np.unique(
            np.concatenate(
                (
                    np.linspace(0.0, 1.0, RISK_CURVE_GRID_POINTS),
                    np.asarray([gate_threshold], dtype=np.float64),
                )
            )
        )[::-1]
        output.extend(
            _risk_row(
                scope_rows,
                scope_probabilities,
                threshold=float(threshold),
                serialized_threshold=gate_threshold,
                label_minimum_gain=label_minimum_gain,
                scope=scope,
                dataset=dataset,
            )
            for threshold in thresholds
        )
    return output


def apply_gate_to_rows(
    gate: LogisticConfidenceGate,
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    if tuple(gate.feature_names) != DEFAULT_GATE_FEATURES:
        raise ValueError(
            "Serialized gate does not use exactly DEFAULT_GATE_FEATURES: "
            f"{gate.feature_names}"
        )
    feature_rows = [
        {name: float(row[name]) for name in DEFAULT_GATE_FEATURES} for row in rows
    ]
    matrix = vectorize_feature_rows(feature_rows, feature_names=DEFAULT_GATE_FEATURES)
    probabilities = gate.predict_proba(matrix)
    decisions = gate.predict_open(matrix)
    if probabilities.shape != (len(rows),) or decisions.shape != (len(rows),):
        raise RuntimeError("Gate prediction shape differs from evaluation rows")
    if gate.provenance is None:
        raise ValueError("Gate has no strict provenance")
    minimum_gain = gate.provenance.label_minimum_gain
    output: list[dict[str, object]] = []
    for row, probability, decision in zip(
        rows, probabilities, decisions, strict=True
    ):
        baseline_iou = float(row["baseline_iou"])
        candidate_iou = float(row["candidate_iou"])
        baseline_dice = float(row["baseline_dice"])
        candidate_dice = float(row["candidate_dice"])
        # The closed branch is a direct choice of the published baseline value;
        # no score interpolation or probability arithmetic is permitted.
        gated_iou = candidate_iou if bool(decision) else baseline_iou
        gated_dice = candidate_dice if bool(decision) else baseline_dice
        if not decision and (gated_iou != baseline_iou or gated_dice != baseline_dice):
            raise AssertionError("Closed gate did not preserve the baseline row")
        enriched = dict(row)
        enriched.update(
            {
                "gate_probability": float(probability),
                "gate_threshold": gate.threshold,
                "gate_open": int(decision),
                "gate_target_positive": int(float(row["delta_iou"]) > minimum_gain),
                "selected_output": "candidate" if decision else "baseline",
                "gated_iou": gated_iou,
                "gated_iou_gain": (
                    float(row["candidate_iou_gain"]) if decision else 0.0
                ),
                "gated_dice": gated_dice,
                "gated_dice_gain": (
                    float(row["candidate_dice_gain"]) if decision else 0.0
                ),
            }
        )
        output.append(enriched)
    return output, probabilities, decisions


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    destination = path
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(fields), extrasaction="raise"
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON output cannot contain NaN or infinity")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    destination = path
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
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
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> dict[str, object]:
    evaluation_roots = [path.expanduser().resolve() for path in args.evaluation_dir]
    if len(evaluation_roots) != len(set(evaluation_roots)):
        raise ValueError("--evaluation-dir values must be unique")
    output = args.output.expanduser().resolve()
    if output in evaluation_roots:
        raise ValueError("--output must differ from every --evaluation-dir")

    gate_path = args.gate_json.expanduser().resolve()
    oof_path = args.oof_manifest.expanduser().resolve()
    gate = LogisticConfidenceGate.load_json(gate_path)
    if gate.provenance is None:
        raise ValueError("Gate JSON has no provenance")
    if tuple(gate.feature_names) != DEFAULT_GATE_FEATURES:
        raise ValueError("Analytical evaluator accepts only DEFAULT_GATE_FEATURES")
    oof_manifest = _read_json_object(oof_path, "OOF manifest")
    oof_sha256 = sha256_file(oof_path)
    expected_oof_sha256, provenance_field = _provenance_oof_sha256(gate)
    if oof_sha256 != expected_oof_sha256:
        raise ValueError(
            "OOF manifest SHA-256 differs from gate provenance: "
            f"{oof_sha256} != {expected_oof_sha256}"
        )
    compatibility = oof_compatibility(oof_manifest)
    if gate.provenance.baseline_checkpoint_sha256 != (
        compatibility.baseline_checkpoint_sha256
    ):
        raise ValueError("Gate provenance baseline differs from OOF manifest")
    if gate.provenance.frangi_extractor_sha256 != (
        compatibility.frangi_extractor_sha256
    ):
        raise ValueError("Gate provenance extractor differs from OOF manifest")
    if gate.provenance.frangi_cache_manifest_sha256 != (
        compatibility.frangi_cache_manifest_sha256
    ):
        raise ValueError("Gate provenance cache manifest differs from OOF manifest")
    if gate.provenance.protocol_sha256 != compatibility.protocol_sha256:
        raise ValueError("Gate provenance protocol differs from OOF manifest")
    if gate.provenance.label_minimum_gain != compatibility.label_minimum_gain:
        raise ValueError("Gate label minimum gain differs from OOF manifest")

    evaluations = [
        load_evaluation_input(root, compatibility) for root in evaluation_roots
    ]
    rows = [row for evaluation in evaluations for row in evaluation.rows]
    identities = [(str(row["dataset"]), str(row["case_name"])) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate dataset/case_name across evaluation directories")

    gated_rows, probabilities, decisions = apply_gate_to_rows(gate, rows)
    label_minimum_gain = gate.provenance.label_minimum_gain
    overall = _decision_statistics(
        rows,
        probabilities,
        decisions,
        label_minimum_gain=label_minimum_gain,
    )
    by_dataset: dict[str, object] = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["dataset"] == dataset],
            dtype=np.int64,
        )
        by_dataset[dataset] = _decision_statistics(
            [rows[int(index)] for index in indices],
            probabilities[indices],
            decisions[indices],
            label_minimum_gain=label_minimum_gain,
        )
    risk_rows = risk_coverage_rows(
        rows,
        probabilities,
        gate_threshold=gate.threshold,
        label_minimum_gain=label_minimum_gain,
    )
    overall_curve = [row for row in risk_rows if row["scope"] == "overall"]
    summary: dict[str, object] = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "policy": {
            "analytical_only": True,
            "threshold_source": "serialized_gate_json",
            "threshold_selected_or_recalibrated": False,
            "exact_row_level_baseline_fallback": True,
        },
        "gate": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
            "threshold": gate.threshold,
            "feature_names": list(gate.feature_names),
            "label_definition": LABEL_DEFINITION,
            "label_minimum_gain": label_minimum_gain,
            "provenance": gate.provenance.to_dict(),
        },
        "oof_manifest": {
            "path": str(oof_path),
            "sha256": oof_sha256,
            "gate_provenance_field": provenance_field,
            "compatibility": {
                "baseline_checkpoint_sha256": (
                    compatibility.baseline_checkpoint_sha256
                ),
                "frangi_extractor_sha256": compatibility.frangi_extractor_sha256,
                "frangi_cache_manifest_sha256": (
                    compatibility.frangi_cache_manifest_sha256
                ),
                "protocol_sha256": compatibility.protocol_sha256,
                "frangi_parameters": dict(compatibility.frangi_parameters),
                "channels": list(compatibility.channels),
                "feature_names": list(compatibility.feature_names),
                "label_minimum_gain": compatibility.label_minimum_gain,
                "segmentation_threshold": compatibility.segmentation_threshold,
            },
        },
        "evaluations": [
            {
                "path": str(evaluation.root),
                "evaluation_contract_sha256": evaluation.contract_sha256,
                "per_image_csv_sha256": evaluation.csv_sha256,
                "dataset": evaluation.contract["dataset"],
                "rows": len(evaluation.rows),
            }
            for evaluation in evaluations
        ],
        "overall": overall,
        "by_dataset": by_dataset,
        "risk_coverage_curve": overall_curve,
        "risk_coverage_rows": len(risk_rows),
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(output / "per_image_gated.csv", gated_rows, GATED_FIELDS)
    _write_csv_atomic(output / "risk_coverage.csv", risk_rows, RISK_FIELDS)
    _write_json_atomic(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary["overall"], sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
