"""Simple, auditable confidence gate for the Frangi residual candidate.

The gate is deliberately global (one decision per image) and deliberately
small: standardized scalar features followed by an L2-regularized logistic
regression.  It never sees the ground truth at inference time.  Ground truth
is used only to create the training label "the candidate improves the
baseline" and to choose a conservative threshold on a separate calibration
set.

This module depends only on NumPy.  In particular, loading a saved gate never
executes Python objects as pickle-based model formats can do.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


GATE_SCHEMA = "cracksam2.logistic-confidence-gate"
GATE_SCHEMA_VERSION = 3
MAX_GATE_JSON_BYTES = 1_000_000
LABEL_DEFINITION = "candidate_better := delta_iou > label_minimum_gain"
GROUP_WEIGHTING = "inverse_source_group_crop_count"

DEFAULT_MINIMUM_PRECISION = 0.80
DEFAULT_MINIMUM_SELECTED = 10
DEFAULT_MINIMUM_SELECTED_GROUPS = 10
DEFAULT_MINIMUM_MEAN_DELTA_IOU = 0.0
DEFAULT_SEVERE_LOSS_THRESHOLD = -0.05
DEFAULT_MAXIMUM_SEVERE_LOSS_RATE = 0.05

# The order is part of the serialized model contract.  Keep names explicit so
# a CSV column cannot silently be interpreted as a different quantity.
DEFAULT_GATE_FEATURES: tuple[str, ...] = (
    "relevant_baseline_entropy_mean",
    "baseline_foreground_fraction",
    "relevant_prediction_disagreement_rate",
    "support_correction_probability_mean",
    "foreground_probability_change_mean",
    "frangi_similarity_support_mean",
    "frangi_density",
)

REQUIRED_FRANGI_STATS: tuple[str, ...] = (
    "similarity",
    "density",
)


def _validate_sha256(value: str, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a full 64-character SHA-256")
    return normalized


@dataclass(frozen=True)
class GateProvenance:
    """Immutable identities of every artifact used to build a gate."""

    baseline_checkpoint_sha256: str
    oof_manifest_sha256: str
    frangi_extractor_sha256: str
    frangi_cache_manifest_sha256: str
    protocol_sha256: str
    train_csv_sha256: str
    calibration_csv_sha256: str
    label_definition: str
    label_minimum_gain: float
    git_commit: str

    def __post_init__(self) -> None:
        for field_name in (
            "baseline_checkpoint_sha256",
            "oof_manifest_sha256",
            "frangi_extractor_sha256",
            "frangi_cache_manifest_sha256",
            "protocol_sha256",
            "train_csv_sha256",
            "calibration_csv_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_sha256(getattr(self, field_name), field_name),
            )
        if self.label_definition != LABEL_DEFINITION:
            raise ValueError(
                f"label_definition must be exactly {LABEL_DEFINITION!r}"
            )
        minimum_gain = float(self.label_minimum_gain)
        if not math.isfinite(minimum_gain):
            raise ValueError("label_minimum_gain must be finite")
        object.__setattr__(self, "label_minimum_gain", minimum_gain)
        commit = str(self.git_commit).lower()
        if len(commit) not in (40, 64) or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise ValueError("git_commit must be a full 40- or 64-character hash")
        object.__setattr__(self, "git_commit", commit)

    def to_dict(self) -> dict[str, str | float]:
        return {
            "baseline_checkpoint_sha256": self.baseline_checkpoint_sha256,
            "oof_manifest_sha256": self.oof_manifest_sha256,
            "frangi_extractor_sha256": self.frangi_extractor_sha256,
            "frangi_cache_manifest_sha256": self.frangi_cache_manifest_sha256,
            "protocol_sha256": self.protocol_sha256,
            "train_csv_sha256": self.train_csv_sha256,
            "calibration_csv_sha256": self.calibration_csv_sha256,
            "label_definition": self.label_definition,
            "label_minimum_gain": self.label_minimum_gain,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GateProvenance":
        required = {
            "baseline_checkpoint_sha256",
            "oof_manifest_sha256",
            "frangi_extractor_sha256",
            "frangi_cache_manifest_sha256",
            "protocol_sha256",
            "train_csv_sha256",
            "calibration_csv_sha256",
            "label_definition",
            "label_minimum_gain",
            "git_commit",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("Invalid gate provenance payload")
        return cls(**{name: payload[name] for name in required})


def _as_finite_array(value: Any, name: str) -> np.ndarray:
    """Convert NumPy- or Torch-like values without importing Torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid whose output is strictly between 0 and 1."""
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = np.empty_like(logits)
    positive = logits >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    epsilon = np.finfo(np.float64).eps
    return np.clip(probabilities, epsilon, 1.0 - epsilon)


def compute_gate_features(
    baseline_logits: Any,
    candidate_logits: Any,
    frangi_stats: Mapping[str, float],
    *,
    frangi_support: Any,
) -> dict[str, float]:
    """Compute the first global gate features for one image.

    ``similarity`` is the mean historical ``node_sim_max`` on retained Frangi
    support and ``density`` its occupied image fraction.  Both are supplied by
    the cache and lie in ``[0, 1]``.  This function intentionally does not
    invent a multi-scale stability score that the current extractor does not
    yet measure.

    The spatial summaries concentrate on the union of the predicted baseline
    foreground and Frangi support, so thin cracks are not drowned by easy
    background pixels.  A signed whole-image probability change separately
    exposes candidates that expand the foreground aggressively (a common shadow
    failure).  All correction amplitudes are measured in probability space.
    """
    baseline = _as_finite_array(baseline_logits, "baseline_logits")
    candidate = _as_finite_array(candidate_logits, "candidate_logits")
    if baseline.shape != candidate.shape:
        raise ValueError(
            "baseline_logits and candidate_logits must have the same shape: "
            f"{baseline.shape} != {candidate.shape}"
        )
    valid_single_image_shape = (
        baseline.ndim == 2
        or (baseline.ndim == 3 and baseline.shape[0] == 1)
        or (baseline.ndim == 4 and baseline.shape[:2] == (1, 1))
    )
    if not valid_single_image_shape:
        raise ValueError(
            "compute_gate_features expects one image shaped (H,W), (1,H,W), "
            f"or (1,1,H,W), got {baseline.shape}"
        )
    support_array = _as_finite_array(frangi_support, "frangi_support")
    if support_array.shape != baseline.shape:
        raise ValueError(
            "frangi_support must have the same shape as the logits: "
            f"{support_array.shape} != {baseline.shape}"
        )
    if not np.isin(support_array, (0.0, 1.0)).all():
        raise ValueError("frangi_support must be binary")
    support = support_array.astype(bool, copy=False)

    missing = [name for name in REQUIRED_FRANGI_STATS if name not in frangi_stats]
    if missing:
        raise ValueError(f"Missing Frangi statistics: {', '.join(missing)}")
    normalized_stats: dict[str, float] = {}
    for name in REQUIRED_FRANGI_STATS:
        value = float(frangi_stats[name])
        if not math.isfinite(value):
            raise ValueError(f"Frangi statistic {name!r} must be finite")
        if name in ("similarity", "density") and not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Frangi statistic {name!r} must be normalized to [0, 1]"
            )
        normalized_stats[name] = value

    baseline_probability = _sigmoid(baseline)
    candidate_probability = _sigmoid(candidate)
    entropy = -(
        baseline_probability * np.log2(baseline_probability)
        + (1.0 - baseline_probability)
        * np.log2(1.0 - baseline_probability)
    )
    correction = np.abs(candidate_probability - baseline_probability)
    disagreement = (baseline_probability > 0.5) != (
        candidate_probability > 0.5
    )
    baseline_foreground = baseline_probability > 0.5
    relevant = support | baseline_foreground
    if not relevant.any():
        # A fully empty support/foreground case still receives a finite,
        # deterministic description of the baseline rather than a NaN.
        relevant = np.ones_like(support, dtype=bool)
    support_correction = float(np.mean(correction[support])) if support.any() else 0.0

    return {
        "relevant_baseline_entropy_mean": float(np.mean(entropy[relevant])),
        "baseline_foreground_fraction": float(np.mean(baseline_foreground)),
        "relevant_prediction_disagreement_rate": float(
            np.mean(disagreement[relevant])
        ),
        "support_correction_probability_mean": support_correction,
        "foreground_probability_change_mean": float(
            np.mean(candidate_probability - baseline_probability)
        ),
        "frangi_similarity_support_mean": normalized_stats["similarity"],
        "frangi_density": normalized_stats["density"],
    }


def vectorize_feature_rows(
    rows: Sequence[Mapping[str, float]],
    *,
    feature_names: Sequence[str] = DEFAULT_GATE_FEATURES,
) -> np.ndarray:
    """Convert named feature rows to a finite matrix in an explicit order."""
    names = tuple(feature_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("feature_names must be non-empty and unique")
    if not rows:
        raise ValueError("At least one feature row is required")
    matrix = np.empty((len(rows), len(names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        missing = [name for name in names if name not in row]
        if missing:
            raise ValueError(
                f"Feature row {row_index} is missing: {', '.join(missing)}"
            )
        for column_index, name in enumerate(names):
            matrix[row_index, column_index] = float(row[name])
    if not np.isfinite(matrix).all():
        raise ValueError("Feature rows must contain only finite numeric values")
    return matrix


def candidate_improvement_labels(
    baseline_scores: Any,
    candidate_scores: Any,
    *,
    minimum_gain: float = 0.0,
) -> np.ndarray:
    """Create training labels from per-image scores, never inference inputs."""
    baseline = _as_finite_array(baseline_scores, "baseline_scores").reshape(-1)
    candidate = _as_finite_array(candidate_scores, "candidate_scores").reshape(-1)
    if baseline.shape != candidate.shape:
        raise ValueError("baseline_scores and candidate_scores must have equal size")
    minimum_gain = float(minimum_gain)
    if not math.isfinite(minimum_gain):
        raise ValueError("minimum_gain must be finite")
    return (candidate - baseline > minimum_gain).astype(np.int64)


def inverse_group_frequency_weights(
    source_groups: Sequence[str],
    *,
    expected_rows: int | None = None,
) -> np.ndarray:
    """Give every source image total weight one, irrespective of crop count."""
    groups = tuple(str(group) for group in source_groups)
    if expected_rows is not None and len(groups) != expected_rows:
        raise ValueError(f"Expected {expected_rows} source groups, got {len(groups)}")
    if not groups:
        raise ValueError("At least one source_group is required")
    if any(not group.strip() for group in groups):
        raise ValueError("source_group values must be non-empty")
    counts: dict[str, int] = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    return np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)


def _validate_sample_weights(weights: Any, expected_rows: int) -> np.ndarray:
    values = _as_finite_array(weights, "sample_weight").reshape(-1)
    if values.size != expected_rows:
        raise ValueError(f"Expected {expected_rows} sample weights, got {values.size}")
    if np.any(values <= 0.0):
        raise ValueError("sample weights must be strictly positive")
    return values


@dataclass(frozen=True)
class Standardizer:
    """Training-set mean and standard deviation used by the gate."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64).reshape(-1).copy()
        scale = np.asarray(self.scale, dtype=np.float64).reshape(-1).copy()
        if mean.size == 0 or mean.shape != scale.shape:
            raise ValueError(
                "Standardizer mean and scale must have equal non-zero size"
            )
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Standardizer values must be finite")
        if np.any(scale <= 0):
            raise ValueError("Standardizer scales must be positive")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(
        cls, features: Any, *, sample_weight: Any | None = None
    ) -> "Standardizer":
        matrix = _validate_feature_matrix(features)
        if sample_weight is None:
            weights = np.ones(matrix.shape[0], dtype=np.float64)
        else:
            weights = _validate_sample_weights(sample_weight, matrix.shape[0])
        normalized_weights = weights / weights.sum()
        mean = np.sum(matrix * normalized_weights[:, None], axis=0)
        variance = np.sum(
            ((matrix - mean) ** 2) * normalized_weights[:, None], axis=0
        )
        scale = np.sqrt(variance)
        # Constant columns carry no information but remain in the explicit
        # feature contract.  A scale of one maps them exactly to zero.
        scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, features: Any) -> np.ndarray:
        matrix = _validate_feature_matrix(features, expected_columns=self.mean.size)
        return (matrix - self.mean) / self.scale


def _validate_feature_matrix(
    features: Any, *, expected_columns: int | None = None
) -> np.ndarray:
    matrix = _as_finite_array(features, "features")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"features must be a 2D matrix, got shape {matrix.shape}")
    if expected_columns is not None and matrix.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} features, got {matrix.shape[1]}"
        )
    return matrix


def _validate_binary_labels(labels: Any, expected_rows: int) -> np.ndarray:
    values = _as_finite_array(labels, "labels").reshape(-1)
    if values.size != expected_rows:
        raise ValueError(f"Expected {expected_rows} labels, got {values.size}")
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("labels must contain only 0 and 1")
    return values


def _logistic_objective(
    design: np.ndarray,
    labels: np.ndarray,
    parameters: np.ndarray,
    l2: float,
    sample_weight: np.ndarray,
) -> float:
    logits = design @ parameters
    losses = np.logaddexp(0.0, logits) - labels * logits
    data_loss = float(sample_weight @ losses / sample_weight.sum())
    regularization = 0.5 * l2 * float(parameters[:-1] @ parameters[:-1])
    return float(data_loss + regularization)


def _fit_logistic_newton(
    standardized_features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    max_iterations: int,
    tolerance: float,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, float, int, float]:
    """Fit logistic regression with deterministic damped Newton updates."""
    rows, columns = standardized_features.shape
    design = np.column_stack((standardized_features, np.ones(rows)))
    parameters = np.zeros(columns + 1, dtype=np.float64)
    weights = _validate_sample_weights(sample_weight, rows)
    weight_sum = float(weights.sum())

    for iteration in range(1, max_iterations + 1):
        logits = design @ parameters
        probabilities = _sigmoid(logits)
        weighted_residual = weights * (probabilities - labels)
        gradient = design.T @ weighted_residual / weight_sum
        gradient[:-1] += l2 * parameters[:-1]
        if float(np.max(np.abs(gradient))) <= tolerance:
            return (
                parameters[:-1],
                float(parameters[-1]),
                iteration,
                _logistic_objective(design, labels, parameters, l2, weights),
            )

        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design / weight_sum
        hessian[:-1, :-1] += l2 * np.eye(columns)
        # This tiny deterministic ridge only guards the unregularized intercept
        # against numerical singularity; it has no practical training effect.
        hessian[-1, -1] += 1e-12
        try:
            newton_step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            newton_step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        previous_objective = _logistic_objective(
            design, labels, parameters, l2, weights
        )
        step_size = 1.0
        accepted = False
        for _ in range(50):
            candidate = parameters - step_size * newton_step
            candidate_objective = _logistic_objective(
                design, labels, candidate, l2, weights
            )
            if candidate_objective <= previous_objective:
                parameters = candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            raise RuntimeError("Logistic regression line search failed")
        if float(np.max(np.abs(step_size * newton_step))) <= tolerance * (
            1.0 + float(np.max(np.abs(parameters)))
        ):
            return (
                parameters[:-1],
                float(parameters[-1]),
                iteration,
                _logistic_objective(design, labels, parameters, l2, weights),
            )

    raise RuntimeError(
        f"Logistic regression did not converge in {max_iterations} iterations"
    )


def select_conservative_threshold(
    probabilities: Any,
    delta_iou: Any,
    source_groups: Sequence[str],
    *,
    label_minimum_gain: float = 0.0,
    minimum_precision: float = DEFAULT_MINIMUM_PRECISION,
    minimum_selected: int = DEFAULT_MINIMUM_SELECTED,
    minimum_selected_groups: int = DEFAULT_MINIMUM_SELECTED_GROUPS,
    minimum_mean_delta_iou: float = DEFAULT_MINIMUM_MEAN_DELTA_IOU,
    severe_loss_threshold: float = DEFAULT_SEVERE_LOSS_THRESHOLD,
    maximum_severe_loss_rate: float = DEFAULT_MAXIMUM_SEVERE_LOSS_RATE,
) -> tuple[float, dict[str, float | int | str]]:
    """Choose a safe threshold from signed held-out IoU differences.

    Crop-level observations are balanced by ``source_group`` for precision,
    mean gain and severe-loss rate.  Thus twenty crops from one source image do
    not masquerade as twenty independent confirmations.  If no threshold meets
    every constraint, 1.0 closes the gate for all model-generated probabilities.
    """
    label_minimum_gain = float(label_minimum_gain)
    minimum_precision = float(minimum_precision)
    minimum_selected = int(minimum_selected)
    minimum_selected_groups = int(minimum_selected_groups)
    minimum_mean_delta_iou = float(minimum_mean_delta_iou)
    severe_loss_threshold = float(severe_loss_threshold)
    maximum_severe_loss_rate = float(maximum_severe_loss_rate)
    if not math.isfinite(label_minimum_gain):
        raise ValueError("label_minimum_gain must be finite")
    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum_precision must lie in (0, 1]")
    if minimum_selected <= 0:
        raise ValueError("minimum_selected must be positive")
    if minimum_selected_groups <= 0:
        raise ValueError("minimum_selected_groups must be positive")
    if not math.isfinite(minimum_mean_delta_iou):
        raise ValueError("minimum_mean_delta_iou must be finite")
    if not math.isfinite(severe_loss_threshold):
        raise ValueError("severe_loss_threshold must be finite")
    if not 0.0 <= maximum_severe_loss_rate <= 1.0:
        raise ValueError("maximum_severe_loss_rate must lie in [0, 1]")

    probability_array = _as_finite_array(
        probabilities, "probabilities"
    ).reshape(-1)
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    epsilon = np.finfo(np.float64).eps
    probability_array = np.clip(probability_array, epsilon, 1.0 - epsilon)
    delta_array = _as_finite_array(delta_iou, "delta_iou").reshape(-1)
    if delta_array.size != probability_array.size:
        raise ValueError(
            f"Expected {probability_array.size} delta_iou values, "
            f"got {delta_array.size}"
        )
    groups = tuple(str(group) for group in source_groups)
    full_group_weights = inverse_group_frequency_weights(
        groups, expected_rows=probability_array.size
    )
    label_array = (delta_array > label_minimum_gain).astype(np.float64)
    positives = int(np.count_nonzero(label_array == 1.0))
    reliability = probability_reliability_metrics(
        label_array,
        probability_array,
        sample_weight=full_group_weights,
    )
    calibration_groups = len(set(groups))

    best: tuple[
        tuple[int, int, float, float, float],
        float,
        dict[str, float | int | str],
    ] | None
    best = None
    for threshold in np.unique(probability_array)[::-1]:
        selected = probability_array >= threshold
        selected_count = int(np.count_nonzero(selected))
        if selected_count < minimum_selected:
            continue
        selected_group_values = tuple(
            group
            for group, is_selected in zip(groups, selected, strict=True)
            if is_selected
        )
        selected_group_count = len(set(selected_group_values))
        if selected_group_count < minimum_selected_groups:
            continue
        selected_weights = inverse_group_frequency_weights(selected_group_values)
        selected_labels = label_array[selected]
        selected_delta = delta_array[selected]
        true_positive = int(np.count_nonzero(selected & (label_array == 1.0)))
        false_positive = selected_count - true_positive
        precision = float(np.average(selected_labels, weights=selected_weights))
        if precision < minimum_precision:
            continue
        mean_delta = float(np.average(selected_delta, weights=selected_weights))
        if mean_delta < minimum_mean_delta_iou:
            continue
        severe_loss = selected_delta < severe_loss_threshold
        severe_loss_rate = float(np.average(severe_loss, weights=selected_weights))
        if severe_loss_rate > maximum_severe_loss_rate:
            continue
        recall = true_positive / positives if positives else 0.0
        # Prefer coverage across independent sources, then crops.  Remaining
        # ties favor observed gain, precision and finally the higher threshold.
        rank = (
            selected_group_count,
            selected_count,
            mean_delta,
            precision,
            float(threshold),
        )
        metadata: dict[str, float | int | str] = {
            "status": "calibrated",
            "label_definition": LABEL_DEFINITION,
            "label_minimum_gain": label_minimum_gain,
            "minimum_precision": minimum_precision,
            "minimum_selected": minimum_selected,
            "minimum_selected_groups": minimum_selected_groups,
            "minimum_mean_delta_iou": minimum_mean_delta_iou,
            "severe_loss_threshold": severe_loss_threshold,
            "maximum_severe_loss_rate": maximum_severe_loss_rate,
            "selected": selected_count,
            "selected_groups": selected_group_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "precision": precision,
            "recall": recall,
            "selected_mean_delta_iou": mean_delta,
            "selected_severe_loss_rate": severe_loss_rate,
            "calibration_size": int(label_array.size),
            "calibration_groups": calibration_groups,
            "calibration_positives": positives,
            **reliability,
        }
        if best is None or rank > best[0]:
            best = (rank, float(threshold), metadata)

    if best is not None:
        return best[1], best[2]
    return 1.0, {
        "status": "closed_no_reliable_threshold",
        "label_definition": LABEL_DEFINITION,
        "label_minimum_gain": label_minimum_gain,
        "minimum_precision": minimum_precision,
        "minimum_selected": minimum_selected,
        "minimum_selected_groups": minimum_selected_groups,
        "minimum_mean_delta_iou": minimum_mean_delta_iou,
        "severe_loss_threshold": severe_loss_threshold,
        "maximum_severe_loss_rate": maximum_severe_loss_rate,
        "selected": 0,
        "selected_groups": 0,
        "true_positive": 0,
        "false_positive": 0,
        "precision": 0.0,
        "recall": 0.0,
        "selected_mean_delta_iou": 0.0,
        "selected_severe_loss_rate": 0.0,
        "calibration_size": int(label_array.size),
        "calibration_groups": calibration_groups,
        "calibration_positives": positives,
        **reliability,
    }


def probability_reliability_metrics(
    labels: Any,
    probabilities: Any,
    *,
    bins: int = 10,
    sample_weight: Any | None = None,
) -> dict[str, float]:
    """Return Brier score and equal-width expected calibration error (ECE)."""
    bins = int(bins)
    if bins <= 0:
        raise ValueError("bins must be positive")
    probability_array = _as_finite_array(
        probabilities, "probabilities"
    ).reshape(-1)
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    label_array = _validate_binary_labels(labels, probability_array.size)
    if sample_weight is None:
        weights = np.ones(probability_array.size, dtype=np.float64)
    else:
        weights = _validate_sample_weights(sample_weight, probability_array.size)
    normalized_weights = weights / weights.sum()
    brier = float(
        np.sum(normalized_weights * (probability_array - label_array) ** 2)
    )
    expected_calibration_error = 0.0
    # A probability of exactly one belongs to the final bin.
    bin_indices = np.minimum((probability_array * bins).astype(np.int64), bins - 1)
    for bin_index in range(bins):
        in_bin = bin_indices == bin_index
        if not np.any(in_bin):
            continue
        bin_weight = float(np.sum(normalized_weights[in_bin]))
        confidence = float(
            np.average(probability_array[in_bin], weights=weights[in_bin])
        )
        observed_frequency = float(
            np.average(label_array[in_bin], weights=weights[in_bin])
        )
        expected_calibration_error += bin_weight * abs(
            confidence - observed_frequency
        )
    return {
        "brier_score": brier,
        "expected_calibration_error": float(expected_calibration_error),
        "mean_predicted_probability": float(
            np.average(probability_array, weights=weights)
        ),
        "observed_positive_rate": float(np.average(label_array, weights=weights)),
    }


@dataclass(frozen=True)
class LogisticConfidenceGate:
    """Standardized logistic regression with a held-out decision threshold."""

    feature_names: tuple[str, ...]
    standardizer: Standardizer
    coefficients: np.ndarray
    intercept: float
    l2: float
    threshold: float = 0.5
    training_iterations: int = 0
    training_objective: float = math.nan
    training_sample_count: int = 0
    training_source_group_count: int = 0
    calibration: Mapping[str, float | int | str] | None = None
    provenance: GateProvenance | None = None

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.feature_names)
        coefficients = (
            np.asarray(self.coefficients, dtype=np.float64).reshape(-1).copy()
        )
        if not names or len(names) != len(set(names)):
            raise ValueError("feature_names must be non-empty and unique")
        if coefficients.size != len(names):
            raise ValueError("One coefficient is required for every feature")
        if self.standardizer.mean.size != len(names):
            raise ValueError("Standardizer size does not match feature_names")
        numeric_values = (
            *coefficients.tolist(),
            float(self.intercept),
            float(self.l2),
            float(self.threshold),
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Gate model parameters must be finite")
        if float(self.l2) <= 0:
            raise ValueError("l2 must be positive")
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        if int(self.training_iterations) < 0:
            raise ValueError("training_iterations cannot be negative")
        if int(self.training_sample_count) < 0:
            raise ValueError("training_sample_count cannot be negative")
        if int(self.training_source_group_count) < 0:
            raise ValueError("training_source_group_count cannot be negative")
        if int(self.training_source_group_count) > int(self.training_sample_count):
            raise ValueError("source group count cannot exceed sample count")
        objective = float(self.training_objective)
        if not (math.isnan(objective) or math.isfinite(objective)):
            raise ValueError("training_objective must be finite or NaN before fitting")
        coefficients.setflags(write=False)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "l2", float(self.l2))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "training_iterations", int(self.training_iterations))
        object.__setattr__(self, "training_objective", objective)
        object.__setattr__(
            self, "training_sample_count", int(self.training_sample_count)
        )
        object.__setattr__(
            self,
            "training_source_group_count",
            int(self.training_source_group_count),
        )
        object.__setattr__(self, "calibration", dict(self.calibration or {}))
        if self.provenance is not None and not isinstance(
            self.provenance, GateProvenance
        ):
            raise TypeError("provenance must be a GateProvenance")

    @classmethod
    def fit(
        cls,
        features: Any,
        labels: Any,
        *,
        source_groups: Sequence[str],
        feature_names: Sequence[str] = DEFAULT_GATE_FEATURES,
        l2: float = 1.0,
        max_iterations: int = 200,
        tolerance: float = 1e-9,
    ) -> "LogisticConfidenceGate":
        matrix = _validate_feature_matrix(features)
        names = tuple(feature_names)
        if matrix.shape[1] != len(names):
            raise ValueError(
                f"Expected {len(names)} named features, got {matrix.shape[1]}"
            )
        label_array = _validate_binary_labels(labels, matrix.shape[0])
        if np.unique(label_array).size != 2:
            raise ValueError("Training labels must contain both classes")
        l2 = float(l2)
        if not math.isfinite(l2) or l2 <= 0:
            raise ValueError("l2 must be a finite positive number")
        if int(max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(float(tolerance)) or float(tolerance) <= 0:
            raise ValueError("tolerance must be a finite positive number")

        groups = tuple(str(group) for group in source_groups)
        sample_weight = inverse_group_frequency_weights(
            groups, expected_rows=matrix.shape[0]
        )
        standardizer = Standardizer.fit(matrix, sample_weight=sample_weight)
        coefficients, intercept, iterations, objective = _fit_logistic_newton(
            standardizer.transform(matrix),
            label_array,
            l2=l2,
            max_iterations=int(max_iterations),
            tolerance=float(tolerance),
            sample_weight=sample_weight,
        )
        return cls(
            feature_names=names,
            standardizer=standardizer,
            coefficients=coefficients,
            intercept=intercept,
            l2=l2,
            training_iterations=iterations,
            training_objective=objective,
            training_sample_count=matrix.shape[0],
            training_source_group_count=len(set(groups)),
        )

    def predict_proba(self, features: Any) -> np.ndarray:
        standardized = self.standardizer.transform(features)
        return _sigmoid(standardized @ self.coefficients + self.intercept)

    def predict_open(self, features: Any) -> np.ndarray:
        """Return one Boolean "use candidate" decision per feature row."""
        return self.predict_proba(features) >= self.threshold

    def calibrate(
        self,
        features: Any,
        delta_iou: Any,
        *,
        source_groups: Sequence[str],
        label_minimum_gain: float = 0.0,
        minimum_precision: float = DEFAULT_MINIMUM_PRECISION,
        minimum_selected: int = DEFAULT_MINIMUM_SELECTED,
        minimum_selected_groups: int = DEFAULT_MINIMUM_SELECTED_GROUPS,
        minimum_mean_delta_iou: float = DEFAULT_MINIMUM_MEAN_DELTA_IOU,
        severe_loss_threshold: float = DEFAULT_SEVERE_LOSS_THRESHOLD,
        maximum_severe_loss_rate: float = DEFAULT_MAXIMUM_SEVERE_LOSS_RATE,
    ) -> "LogisticConfidenceGate":
        """Return a copy with a threshold chosen on a reserved set."""
        probabilities = self.predict_proba(features)
        threshold, metadata = select_conservative_threshold(
            probabilities,
            delta_iou,
            source_groups,
            label_minimum_gain=label_minimum_gain,
            minimum_precision=minimum_precision,
            minimum_selected=minimum_selected,
            minimum_selected_groups=minimum_selected_groups,
            minimum_mean_delta_iou=minimum_mean_delta_iou,
            severe_loss_threshold=severe_loss_threshold,
            maximum_severe_loss_rate=maximum_severe_loss_rate,
        )
        return replace(self, threshold=threshold, calibration=metadata)

    def with_provenance(
        self, provenance: GateProvenance
    ) -> "LogisticConfidenceGate":
        """Attach the exact artifact identities required for serialization."""
        return replace(self, provenance=provenance)

    def apply(
        self,
        baseline_logits: Any,
        candidate_logits: Any,
        features: Any,
    ) -> np.ndarray:
        """Select candidate logits or return the baseline exactly."""
        return select_logits_with_fallback(
            baseline_logits,
            candidate_logits,
            self.predict_open(features),
        )

    def to_dict(self) -> dict[str, Any]:
        if math.isnan(self.training_objective):
            raise ValueError("Cannot serialize an unfitted gate")
        if not self.calibration:
            raise ValueError("Cannot serialize a gate before held-out calibration")
        if self.provenance is None:
            raise ValueError("Cannot serialize a gate without strict provenance")
        if self.calibration.get("label_definition") != self.provenance.label_definition:
            raise ValueError("Calibration and provenance label definitions differ")
        if float(self.calibration.get("label_minimum_gain", math.nan)) != (
            self.provenance.label_minimum_gain
        ):
            raise ValueError("Calibration and provenance minimum gains differ")
        return {
            "schema": GATE_SCHEMA,
            "schema_version": GATE_SCHEMA_VERSION,
            "model_type": "logistic_regression_l2",
            "feature_names": list(self.feature_names),
            "standardizer": {
                "mean": self.standardizer.mean.tolist(),
                "scale": self.standardizer.scale.tolist(),
            },
            "model": {
                "coefficients": self.coefficients.tolist(),
                "intercept": self.intercept,
                "l2": self.l2,
                "threshold": self.threshold,
            },
            "training": {
                "iterations": self.training_iterations,
                "objective": self.training_objective,
                "sample_count": self.training_sample_count,
                "source_group_count": self.training_source_group_count,
                "weighting": GROUP_WEIGHTING,
            },
            "calibration": dict(self.calibration or {}),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogisticConfidenceGate":
        required = {
            "schema",
            "schema_version",
            "model_type",
            "feature_names",
            "standardizer",
            "model",
            "training",
            "calibration",
            "provenance",
        }
        if set(payload) != required:
            raise ValueError("Gate JSON has missing or unknown top-level fields")
        if payload["schema"] != GATE_SCHEMA:
            raise ValueError("Unknown gate schema")
        if payload["schema_version"] != GATE_SCHEMA_VERSION:
            raise ValueError("Unsupported gate schema version")
        if payload["model_type"] != "logistic_regression_l2":
            raise ValueError("Unsupported gate model type")
        standardizer_payload = payload["standardizer"]
        model_payload = payload["model"]
        training_payload = payload["training"]
        calibration_payload = payload["calibration"]
        provenance_payload = payload["provenance"]
        if not isinstance(standardizer_payload, Mapping) or set(
            standardizer_payload
        ) != {"mean", "scale"}:
            raise ValueError("Invalid standardizer payload")
        if not isinstance(model_payload, Mapping) or set(model_payload) != {
            "coefficients",
            "intercept",
            "l2",
            "threshold",
        }:
            raise ValueError("Invalid model payload")
        if not isinstance(training_payload, Mapping) or set(training_payload) != {
            "iterations",
            "objective",
            "sample_count",
            "source_group_count",
            "weighting",
        }:
            raise ValueError("Invalid training payload")
        if training_payload["weighting"] != GROUP_WEIGHTING:
            raise ValueError("Unsupported gate training weighting")
        if not isinstance(calibration_payload, Mapping):
            raise ValueError("Invalid calibration payload")
        gate = cls(
            feature_names=tuple(payload["feature_names"]),
            standardizer=Standardizer(
                mean=standardizer_payload["mean"],
                scale=standardizer_payload["scale"],
            ),
            coefficients=model_payload["coefficients"],
            intercept=model_payload["intercept"],
            l2=model_payload["l2"],
            threshold=model_payload["threshold"],
            training_iterations=training_payload["iterations"],
            training_objective=training_payload["objective"],
            training_sample_count=training_payload["sample_count"],
            training_source_group_count=training_payload["source_group_count"],
            calibration=dict(calibration_payload),
            provenance=GateProvenance.from_dict(provenance_payload),
        )
        # Reuse serialization invariants so a hand-edited JSON cannot bind a
        # calibration threshold to a different label/gain provenance.
        gate.to_dict()
        return gate

    def save_json(self, path: str | Path) -> None:
        """Atomically save validated plain JSON (never pickle)."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(serialized)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, destination)
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @classmethod
    def load_json(cls, path: str | Path) -> "LogisticConfidenceGate":
        """Load a size-bounded, strictly validated JSON model."""
        source = Path(path)
        if source.stat().st_size > MAX_GATE_JSON_BYTES:
            raise ValueError("Gate JSON is unexpectedly large")

        def reject_constant(value: str) -> None:
            raise ValueError(f"Non-finite JSON number is forbidden: {value}")

        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream, parse_constant=reject_constant)
        if not isinstance(payload, Mapping):
            raise ValueError("Gate JSON root must be an object")
        return cls.from_dict(payload)


def select_logits_with_fallback(
    baseline_logits: Any,
    candidate_logits: Any,
    open_gate: bool | Sequence[bool] | np.ndarray,
) -> np.ndarray:
    """Apply image-level decisions with a bit-exact baseline fallback.

    For one decision, a closed gate returns the original baseline NumPy object,
    not a recomputed value.  For a batch, closed rows are copied byte-for-byte
    from the baseline through ``np.where``.
    """
    baseline = np.asarray(baseline_logits)
    candidate = np.asarray(candidate_logits)
    if baseline.shape != candidate.shape:
        raise ValueError("baseline_logits and candidate_logits must have equal shape")
    decisions = np.asarray(open_gate)
    if decisions.dtype != np.bool_:
        raise TypeError("open_gate must contain Boolean decisions, not scores")
    if decisions.ndim == 0 or decisions.size == 1:
        return candidate if bool(decisions.reshape(-1)[0]) else baseline
    decisions = decisions.reshape(-1)
    if baseline.ndim == 0 or baseline.shape[0] != decisions.size:
        raise ValueError(
            "For multiple decisions, the first logit dimension must be the batch"
        )
    selector_shape = (decisions.size,) + (1,) * (baseline.ndim - 1)
    return np.where(decisions.reshape(selector_shape), candidate, baseline)


__all__ = [
    "DEFAULT_MAXIMUM_SEVERE_LOSS_RATE",
    "DEFAULT_GATE_FEATURES",
    "DEFAULT_MINIMUM_MEAN_DELTA_IOU",
    "DEFAULT_MINIMUM_PRECISION",
    "DEFAULT_MINIMUM_SELECTED",
    "DEFAULT_MINIMUM_SELECTED_GROUPS",
    "DEFAULT_SEVERE_LOSS_THRESHOLD",
    "GATE_SCHEMA",
    "GATE_SCHEMA_VERSION",
    "GROUP_WEIGHTING",
    "GateProvenance",
    "LABEL_DEFINITION",
    "LogisticConfidenceGate",
    "REQUIRED_FRANGI_STATS",
    "Standardizer",
    "candidate_improvement_labels",
    "compute_gate_features",
    "inverse_group_frequency_weights",
    "probability_reliability_metrics",
    "select_conservative_threshold",
    "select_logits_with_fallback",
    "vectorize_feature_rows",
]
