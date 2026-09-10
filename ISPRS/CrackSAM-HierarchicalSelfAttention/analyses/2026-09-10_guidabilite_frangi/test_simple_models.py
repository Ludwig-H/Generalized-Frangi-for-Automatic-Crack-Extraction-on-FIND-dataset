"""Guard against scene/label leakage and incorrect paired bootstrap units."""

import json
import numpy as np
import pandas as pd
import pytest
import torch
from threadpoolctl import threadpool_limits

from compare_simple_classifiers import (
    bootstrap_predictions, class_weights, fit_one, make_splits, scaled_data, split_inner,
)


def synthetic_cases():
    y = np.repeat(np.arange(60) % 3, 2)
    cases = pd.DataFrame({"image_id": [f"image{i}" for i in range(120)],
                          "physical_group": np.repeat([f"scene{i}" for i in range(60)], 2),
                          "domain": np.repeat(np.where(np.arange(60) < 30, "a", "b"), 2)})
    X = np.random.default_rng(42).normal(size=(120, 5)) + y[:, None]
    return cases, X, y


def test_class_weights_equalize_total_training_contribution():
    y = np.array([0] * 20 + [1] * 5 + [2] * 2)
    np.testing.assert_allclose(class_weights(y) * np.bincount(y), np.full(3, len(y) / 3), rtol=1e-6)
    with pytest.raises(ValueError):
        class_weights(np.array([0, 1]))


def test_inner_validation_has_disjoint_scenes_and_no_external_rows():
    cases, _, y = synthetic_cases()
    train = np.arange(96)
    inner, valid = split_inner(cases, y, train, 42)
    assert not set(cases.physical_group.iloc[inner]) & set(cases.physical_group.iloc[valid])
    assert set(inner) | set(valid) == set(train)


def test_scaling_is_independent_of_held_out_extremes():
    _, X, _ = synthetic_cases()
    train, test = np.arange(96), np.arange(96, 120)
    before, _ = scaled_data(X, train, test)
    X[test] += 1e6
    after, _ = scaled_data(X, train, test)
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("model", ["tuned_logistic", "mlp8"])
def test_external_labels_cannot_change_fitting_or_epoch_selection(model):
    cases, X, y = synthetic_cases()
    train, test = np.arange(96), np.arange(96, 120)
    inner, valid = split_inner(cases, y, train, 42)
    split = {"train": train, "test": test, "inner": inner, "valid": valid, "seed": 42}
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        before, info = fit_one(X, y, split, model, epochs=10)
        changed = y.copy()
        changed[test] = (changed[test] + 1) % 3
        after, changed_info = fit_one(X, changed, split, model, epochs=10)
    np.testing.assert_array_equal(before, after)
    assert info["chosen_C"] == changed_info["chosen_C"]
    assert info["best_epoch"] == changed_info["best_epoch"]
    json.dumps(info)  # Fit metadata must remain serializable for safe resume.


def test_paired_bootstrap_resamples_whole_scene_not_its_observations():
    cases = pd.DataFrame({"physical_group": ["scene"] * 3, "domain": ["a"] * 3,
                          "delta_iou": [-.03, 0, .03]})
    y = np.arange(3)
    predictions = {"mean/mlp32": np.eye(3), "mean/tuned_logistic": np.tile([1., 0., 0.], (3, 1))}
    _, result = bootstrap_predictions(cases, y, predictions, repeats=100, seed=42)
    np.testing.assert_allclose(result["difference_balanced_accuracy"]["ci95"], [2/3, 2/3])
    np.testing.assert_allclose(result["difference_gate_delta_iou"]["ci95"], [.01, .01])


def test_split_seeds_are_json_serializable(tmp_path):
    cases, _, y = synthetic_cases()
    cases.assign(fold=np.repeat(np.arange(60) % 5, 2)).to_csv(tmp_path / "folds.csv", index=False)
    splits = make_splits(cases, y, tmp_path / "folds.csv", 42)
    json.dumps([split["seed"] for split in splits])
