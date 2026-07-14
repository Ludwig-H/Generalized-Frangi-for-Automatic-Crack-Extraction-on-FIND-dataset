"""Segmentation metrics for CrackSAM 2 evaluation."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import ot
import torch


ArrayLike = np.ndarray | torch.Tensor


def _as_numpy(value: ArrayLike) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _binary_pair(
    prediction: ArrayLike, target: ArrayLike, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    prediction_array = _as_numpy(prediction)
    target_array = _as_numpy(target)
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction_array.shape} != "
            f"{target_array.shape}"
        )
    if not np.isfinite(prediction_array).all() or not np.isfinite(target_array).all():
        raise ValueError("Prediction and target must contain only finite values")
    return prediction_array > threshold, target_array > threshold


def segmentation_metrics(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute precision, recall, Dice/F1 and IoU with explicit empty cases.

    Two empty masks are a perfect match (all metrics equal one). If exactly
    one mask is empty, all four metrics equal zero.
    """
    pred, truth = _binary_pair(prediction, target, threshold)
    pred_count = int(np.count_nonzero(pred))
    truth_count = int(np.count_nonzero(truth))
    if pred_count == 0 and truth_count == 0:
        return {"precision": 1.0, "recall": 1.0, "dice": 1.0, "iou": 1.0}
    if pred_count == 0 or truth_count == 0:
        return {"precision": 0.0, "recall": 0.0, "dice": 0.0, "iou": 0.0}

    intersection = int(np.count_nonzero(pred & truth))
    union = pred_count + truth_count - intersection
    return {
        "precision": intersection / pred_count,
        "recall": intersection / truth_count,
        "dice": (2.0 * intersection) / (pred_count + truth_count),
        "iou": intersection / union,
    }


def precision_recall_dice_iou(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    threshold: float = 0.5,
) -> tuple[float, float, float, float]:
    """Tuple-returning compatibility wrapper around :func:`segmentation_metrics`."""
    values = segmentation_metrics(prediction, target, threshold=threshold)
    return values["precision"], values["recall"], values["dice"], values["iou"]


def _as_spatial_mask(mask: ArrayLike, name: str) -> np.ndarray:
    array = _as_numpy(mask).astype(np.float64, copy=False)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional mask, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0):
        raise ValueError(f"{name} cannot contain negative mass")
    return array


def _support_and_weights(
    mask: np.ndarray, max_points: int | None
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask > 0).astype(np.float64)
    if coordinates.size == 0:
        return coordinates.reshape(0, 2), np.empty(0, dtype=np.float64)
    weights = mask[mask > 0].astype(np.float64, copy=False)
    if max_points is not None and coordinates.shape[0] > max_points:
        # Even row-major subsampling is deterministic across processes and devices.
        indices = np.linspace(0, coordinates.shape[0] - 1, max_points, dtype=np.int64)
        coordinates = coordinates[indices]
        weights = weights[indices]
    weights = weights / weights.sum()
    return coordinates, weights


def wasserstein_mask_distance(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    max_points: int | None = 2_000,
) -> float:
    """Compute direct-mask spatial EMD using normalized positive pixel values.

    Coordinates are ``(y, x)`` pixels and the ground cost is Euclidean distance.
    The default deterministic row-major subsampling bounds the transport matrix
    to 2,000 points per mask. Set ``max_points=None`` for the exact support.

    The distance is zero for two empty masks. If exactly one mask is empty, the
    finite penalty is the image diagonal, i.e. the largest possible distance
    between two pixel coordinates in the shared spatial domain.
    """
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be positive or None")
    pred = _as_spatial_mask(prediction, "prediction")
    truth = _as_spatial_mask(target, "target")
    if pred.shape != truth.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {pred.shape} != {truth.shape}"
        )
    pred_coordinates, pred_weights = _support_and_weights(pred, max_points)
    truth_coordinates, truth_weights = _support_and_weights(truth, max_points)

    pred_empty = pred_weights.size == 0
    truth_empty = truth_weights.size == 0
    if pred_empty and truth_empty:
        return 0.0
    if pred_empty or truth_empty:
        height, width = pred.shape
        return float(np.hypot(height - 1, width - 1))

    cost = ot.dist(pred_coordinates, truth_coordinates, metric="euclidean")
    return float(ot.emd2(pred_weights, truth_weights, cost))


def evaluate_masks(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    threshold: float = 0.5,
    max_points: int | None = 2_000,
) -> Mapping[str, float]:
    """Return all roadmap metrics for one pair of masks."""
    values = segmentation_metrics(prediction, target, threshold=threshold)
    values["wasserstein"] = wasserstein_mask_distance(
        prediction, target, max_points=max_points
    )
    return values


# Compatibility names used by the original CrackSAM utilities.
calculate_metric_percase = precision_recall_dice_iou
compute_metrics = segmentation_metrics
wasserstein_distance_masks = wasserstein_mask_distance


__all__ = [
    "calculate_metric_percase",
    "compute_metrics",
    "evaluate_masks",
    "precision_recall_dice_iou",
    "segmentation_metrics",
    "wasserstein_distance_masks",
    "wasserstein_mask_distance",
]
