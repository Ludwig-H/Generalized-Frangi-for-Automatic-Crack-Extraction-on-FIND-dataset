from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from cracksam2.metrics import (  # noqa: E402
    evaluate_masks,
    segmentation_metrics,
    wasserstein_mask_distance,
)


def test_binary_metrics_for_perfect_and_disjoint_masks():
    target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert segmentation_metrics(target, target) == {
        "precision": 1.0,
        "recall": 1.0,
        "dice": 1.0,
        "iou": 1.0,
    }
    disjoint = 1.0 - target
    assert segmentation_metrics(disjoint, target) == {
        "precision": 0.0,
        "recall": 0.0,
        "dice": 0.0,
        "iou": 0.0,
    }


def test_binary_metrics_have_explicit_empty_mask_semantics():
    empty = np.zeros((3, 3), dtype=np.float32)
    nonempty = empty.copy()
    nonempty[1, 1] = 1.0
    assert segmentation_metrics(empty, empty) == {
        "precision": 1.0,
        "recall": 1.0,
        "dice": 1.0,
        "iou": 1.0,
    }
    assert all(value == 0.0 for value in segmentation_metrics(empty, nonempty).values())
    assert all(value == 0.0 for value in segmentation_metrics(nonempty, empty).values())
    assert wasserstein_mask_distance(empty, empty) == 0.0
    expected_diagonal = np.hypot(empty.shape[0] - 1, empty.shape[1] - 1)
    assert wasserstein_mask_distance(empty, nonempty) == pytest.approx(expected_diagonal)
    assert wasserstein_mask_distance(nonempty, empty) == pytest.approx(expected_diagonal)


def test_wasserstein_uses_direct_normalized_mask_values():
    prediction = np.zeros((1, 3), dtype=np.float64)
    prediction[0, 0] = 3.0
    prediction[0, 2] = 1.0
    target = np.zeros((1, 3), dtype=np.float64)
    target[0, 0] = 1.0

    # One quarter of the source mass travels two pixels: 0.25 * 2 = 0.5.
    assert wasserstein_mask_distance(prediction, target, max_points=None) == pytest.approx(0.5)
    assert wasserstein_mask_distance(target, target, max_points=None) == pytest.approx(0.0)


def test_wasserstein_subsampling_is_deterministic_and_evaluate_masks_combines_results():
    prediction = np.arange(1, 101, dtype=np.float64).reshape(10, 10)
    target = np.flipud(prediction).copy()
    first = wasserstein_mask_distance(prediction, target, max_points=11)
    second = wasserstein_mask_distance(prediction, target, max_points=11)
    assert first == second

    values = evaluate_masks(prediction / 100.0, target / 100.0, max_points=11)
    assert set(values) == {"precision", "recall", "dice", "iou", "wasserstein"}


def test_wasserstein_default_support_cap_matches_explicit_2000_points():
    prediction = np.ones((1, 2_001), dtype=np.float64)
    target = np.zeros_like(prediction)
    target[0, -1] = 1.0

    assert wasserstein_mask_distance(prediction, target) == pytest.approx(
        wasserstein_mask_distance(prediction, target, max_points=2_000)
    )


def test_metrics_reject_invalid_shapes_and_negative_transport_mass():
    with pytest.raises(ValueError, match="shapes differ"):
        segmentation_metrics(np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="negative mass"):
        wasserstein_mask_distance(-np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="shapes differ"):
        wasserstein_mask_distance(np.zeros((2, 2)), np.zeros((3, 3)))
