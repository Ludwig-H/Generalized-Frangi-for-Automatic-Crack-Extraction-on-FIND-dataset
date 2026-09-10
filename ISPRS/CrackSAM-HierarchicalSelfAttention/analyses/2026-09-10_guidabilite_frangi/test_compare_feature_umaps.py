"""Regression checks for held-out preprocessing and independent-neighbor scores."""

import numpy as np
import pandas as pd
import pytest

from compare_feature_umaps import macro_neighbor_agreement, saved_splits, transform_values


@pytest.mark.parametrize("transform", ["standard", "l2", "balanced", "pca32"])
def test_held_out_outliers_do_not_change_training_transform(transform):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(60, 40))
    spec = {"values": values, "transform": transform, "blocks": (10, 30)}
    train, test = np.arange(45), np.arange(45, 60)
    before, _ = transform_values(spec, train, test)
    perturbed = values.copy()
    perturbed[test] = rng.normal(10000, 1000, (len(test), 40))
    after, _ = transform_values({**spec, "values": perturbed}, train, test)
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)


def test_balanced_scales_have_equal_total_variance():
    values = np.random.default_rng(7).normal(size=(100, 40))
    result, _ = transform_values({"values": values, "transform": "balanced", "blocks": (10, 30)}, np.arange(100))
    np.testing.assert_allclose(result[:, :10].var(0).sum(), 1, atol=1e-6)
    np.testing.assert_allclose(result[:, 10:].var(0).sum(), 1, atol=1e-6)


def test_neighbors_remove_self_and_respect_domains():
    # The nearest global neighbor has the same class; within each domain it does not.
    X = np.array([[0.0], [.1], [10.0], [10.1]])
    y = np.array([0, 0, 1, 1])
    domains = np.array(["a", "b", "a", "b"])
    assert macro_neighbor_agreement(X, y, k=1) == 1
    assert macro_neighbor_agreement(X, y, domains, k=1) == 0


def test_saved_folds_reject_split_physical_scene(tmp_path):
    cases = pd.DataFrame({"image_id": ["a", "b", "c"], "physical_group": ["scene1", "scene1", "scene2"]})
    recorded = cases.assign(fold=[0, 1, 2])
    path = tmp_path / "folds.csv"
    recorded.to_csv(path, index=False)
    with pytest.raises(ValueError, match="scene leakage"):
        saved_splits(cases, path)
