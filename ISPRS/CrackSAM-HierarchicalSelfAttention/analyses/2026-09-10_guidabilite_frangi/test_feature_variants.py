"""Regression checks for descriptor provenance and variance decomposition."""
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("feature_variants", Path(__file__).with_name("feature_variants.py"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture_features():
    rng = np.random.default_rng(31)
    h = rng.normal(size=(5, 256, 4, 4)).astype(np.float32)
    mean = h.mean(axis=(-2, -1))
    grid = h.reshape(5, 256, 2, 2, 2, 2).mean(axis=(3, 5)).reshape(5, 1024)
    return {
        "mean": mean,
        "std": h.std(axis=(-2, -1)),
        "grid2": grid,
        "multiscale_mean": np.concatenate((mean, rng.normal(size=(5, 96)).astype(np.float32)), axis=1),
        "pre_global_mean": rng.normal(size=(5, 576)).astype(np.float32),
    }, h


def test_variance_components_match_actual_within_quadrant_variance():
    features, h = fixture_features()
    variants = module.build_variants(features)
    between = variants["between_std"]["values"]
    within = variants["within_std"]["values"]
    blocks = h.reshape(5, 256, 2, 2, 2, 2)
    actual_within_variance = blocks.var(axis=(3, 5)).mean(axis=(2, 3))
    np.testing.assert_allclose(within ** 2, actual_within_variance, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(between ** 2 + within ** 2, features["std"] ** 2, rtol=1e-6, atol=1e-7)


def test_high_resolution_partition_and_reused_arrays_are_exact():
    features, _ = fixture_features()
    variants = module.build_variants(features)
    np.testing.assert_array_equal(variants["highres32"]["values"], features["multiscale_mean"][:, 256:288])
    np.testing.assert_array_equal(variants["highres64"]["values"], features["multiscale_mean"][:, 288:352])
    np.testing.assert_array_equal(np.concatenate((variants["highres32"]["values"], variants["highres64"]["values"]), axis=1), variants["highres96"]["values"])
    assert np.shares_memory(variants["highres96"]["values"], features["multiscale_mean"])
    assert variants["mean"]["values"] is variants["mean_cosine"]["values"] is variants["mean_pca32"]["values"]
    assert variants["multiscale_mean"]["values"] is variants["multiscale_balanced"]["values"]


def test_inconsistent_negative_variance_is_rejected():
    features, _ = fixture_features()
    features["std"] = np.zeros_like(features["std"])
    with pytest.raises(ValueError, match="Negative within-quadrant variance"):
        module.build_variants(features)


def test_roundoff_negative_variance_is_clipped_but_not_hidden_in_general():
    mean = np.zeros((1, 1), dtype=np.float32)
    grid = np.array([[-1, 1, -1, 1]], dtype=np.float32)
    std = np.array([[1 - 1e-7]], dtype=np.float32)
    between, within = module.spatial_variance_components(mean, std, grid)
    np.testing.assert_array_equal(between, [[1]])
    np.testing.assert_array_equal(within, [[0]])


def test_factory_has_no_label_argument_and_does_not_fit_preprocessing():
    features, _ = fixture_features()
    originals = {name: value.copy() for name, value in features.items()}
    variants = module.build_variants(features)
    assert list(inspect.signature(module.build_variants).parameters) == ["features"]
    assert list(variants) == ["mean", "std", "mean_std", "grid2", "between_std", "within_std", "highres32", "highres64", "highres96", "multiscale_mean", "pre_global_mean", "mean_cosine", "multiscale_balanced", "mean_pca32"]
    assert variants["mean_cosine"]["transform"] == "l2"
    assert variants["mean_cosine"]["metric"] == "cosine"
    assert variants["multiscale_balanced"]["blocks"] == (256, 32, 64)
    assert variants["mean_pca32"]["values"].shape[1] == 256
    for name, value in originals.items():
        np.testing.assert_array_equal(value, features[name])


def test_misaligned_multiscale_archive_is_rejected():
    features, _ = fixture_features()
    features["multiscale_mean"][0, 0] += 1
    with pytest.raises(ValueError, match="Multiscale prefix"):
        module.build_variants(features)
