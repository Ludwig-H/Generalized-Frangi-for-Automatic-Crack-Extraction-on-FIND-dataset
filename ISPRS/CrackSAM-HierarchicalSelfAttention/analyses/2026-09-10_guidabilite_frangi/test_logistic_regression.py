"""Audit grouped logistic selection, held-out isolation and scene uncertainty."""

import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from compare_logistic_regression import (
    fit_split, inner_splits, make_outer_splits, paired_bootstrap, prepare_design,
)


def synthetic_cases():
    groups = np.arange(60)
    y = np.repeat(groups % 3, 2)
    cases = pd.DataFrame({
        "image_id": [f"image{i}" for i in range(len(y))],
        "physical_group": np.repeat([f"scene{i}" for i in groups], 2),
        "domain": np.repeat(np.where(groups < 30, "a", "b"), 2),
    })
    X = np.random.default_rng(42).normal(size=(len(y), 6))
    X[:, 0] += y
    X[:, 1] += (y == 1) * 1.5
    return cases, X, y


def test_inner_folds_keep_scenes_together_and_cover_outer_training():
    cases, _, y = synthetic_cases()
    train = np.arange(96)
    seen = []
    for fit, valid in inner_splits(cases, y, train, 42):
        assert set(fit) | set(valid) == set(train)
        assert not set(cases.physical_group.iloc[fit]) & set(cases.physical_group.iloc[valid])
        seen.extend(valid.tolist())
    np.testing.assert_array_equal(np.sort(seen), train)


def test_outer_folds_reject_scenes_crossing_the_boundary(tmp_path):
    cases, _, _ = synthetic_cases()
    folds = cases.assign(fold=np.repeat(np.arange(60) % 5, 2))
    path = tmp_path / "folds.csv"
    folds.to_csv(path, index=False)
    splits = make_outer_splits(cases, path)
    assert len(splits) == 7
    for split in splits:
        assert not set(cases.physical_group.iloc[split["train"]]) & set(cases.physical_group.iloc[split["test"]])
    folds.loc[1, "fold"] = 1
    folds.to_csv(path, index=False)
    with pytest.raises(ValueError, match="External scene leakage"):
        make_outer_splits(cases, path)


def test_feature_ranking_and_scaler_ignore_held_out_extremes():
    _, X, y = synthetic_cases()
    train, test = np.arange(96), np.arange(96, 120)
    original, _, channels, scaler = prepare_design(X, y, train, test, 2)
    X[test] += np.arange(1, X.shape[1] + 1) * 1e6
    y[test] = (y[test] + 1) % 3
    altered, _, altered_channels, altered_scaler = prepare_design(X, y, train, test, 2)
    np.testing.assert_array_equal(channels, altered_channels)
    np.testing.assert_array_equal(original, altered)
    np.testing.assert_array_equal(scaler.mean_, altered_scaler.mean_)
    np.testing.assert_array_equal(scaler.scale_, altered_scaler.scale_)


@pytest.mark.parametrize("change", ["labels", "features"])
def test_external_data_cannot_change_selected_or_fitted_logistic_model(change):
    cases, X, y = synthetic_cases()
    split = {"train": np.arange(96), "test": np.arange(96, 120), "seed": 42}
    with threadpool_limits(limits=1):
        original, original_search = fit_split(X, y, cases, split, c_grid=(.01, .1), k_grid=(2, "all"))
        altered_X, altered_y = X.copy(), y.copy()
        if change == "labels":
            altered_y[split["test"]] = (altered_y[split["test"]] + 1) % 3
        else:
            altered_X[split["test"]] += np.arange(1, X.shape[1] + 1) * 1e6
        altered, altered_search = fit_split(altered_X, altered_y, cases, split,
                                            c_grid=(.01, .1), k_grid=(2, "all"))
    assert original_search["candidates"] == altered_search["candidates"]
    for strategy, (proba, info) in original.items():
        changed_proba, changed_info = altered[strategy]
        assert info == changed_info
        if change == "labels":
            np.testing.assert_array_equal(proba, changed_proba)
        np.testing.assert_allclose(proba.sum(axis=1), 1)
        assert np.asarray(info["coefficients"]).shape == (3, len(info["selected_channels"]))
        json.dumps(info)


def test_bootstrap_resamples_whole_scene_instead_of_individual_images():
    cases = pd.DataFrame({"physical_group": ["scene"] * 3, "domain": ["a"] * 3,
                          "delta_iou": [-.03, 0, .03]})
    y = np.arange(3)
    predictions = {"mean/uniform_accuracy": np.eye(3),
                   "always_neutral": np.tile([0., 1., 0.], (3, 1))}
    result = paired_bootstrap(cases, y, predictions, repeats=100, seed=42)
    comparison = result["comparisons"][0]["statistics"]
    for metric in ("accuracy", "balanced_accuracy"):
        np.testing.assert_allclose(comparison[metric]["ci95"], [2 / 3, 2 / 3])
    np.testing.assert_allclose(comparison["hard_gate_mean_delta_iou"]["ci95"], [.01, .01])


def test_bootstrap_uses_identical_scene_draws_for_paired_models():
    y = np.tile(np.arange(3), 4)
    cases = pd.DataFrame({"physical_group": np.repeat(["a", "b", "c", "d"], 3),
                          "domain": ["one"] * 12, "delta_iou": np.tile([-.03, 0, .03], 4)})
    proba = np.eye(3)[np.concatenate([np.arange(3), np.ones(3, dtype=int),
                                     np.zeros(3, dtype=int), np.arange(3)])]
    result = paired_bootstrap(cases, y,
                              {"mean/balanced_ba": proba, "previous_mean_tuned": proba.copy()},
                              repeats=100, seed=42)
    accuracy_interval = result["intervals"]["mean/balanced_ba"]["accuracy"]
    assert accuracy_interval[1] > accuracy_interval[0]
    for stats in result["comparisons"][0]["statistics"].values():
        assert stats["estimate"] == 0
        np.testing.assert_array_equal(stats["ci95"], [0, 0])
