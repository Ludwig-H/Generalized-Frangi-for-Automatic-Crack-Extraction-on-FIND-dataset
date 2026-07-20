from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

import analyze_prompt_causal_matrix as analysis  # noqa: E402


def test_bootstrap_mean_is_deterministic_and_contains_observed_mean() -> None:
    values = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.4], dtype=np.float64)
    first = analysis.bootstrap_mean_ci(values, samples=2_000, seed=12)
    repeated = analysis.bootstrap_mean_ci(values, samples=2_000, seed=12)
    assert first == repeated
    assert first[0] < values.mean() < first[1]


def test_delta_summary_counts_catastrophic_losses() -> None:
    delta = np.asarray([-0.2, -0.06, -0.01, 0.0, 0.2])
    result = analysis.summarize_delta(delta, bootstrap_samples=500, seed=1)
    assert result["gains"] == 1
    assert result["ties"] == 1
    assert result["losses"] == 3
    assert result["losses_below_minus_005"] == 2
    assert result["losses_below_minus_010"] == 1


def test_khanhha_family_averages_the_three_conditions_per_case() -> None:
    order = ["a", "b"]
    arrays = {
        "khanhha_original": (order, np.asarray([0.3, 0.0])),
        "khanhha_noisy1": (order, np.asarray([0.0, 0.3])),
        "khanhha_noisy2": (order, np.asarray([0.0, 0.0])),
        "road420": (order, np.asarray([0.1, 0.1])),
        "facade390": (order, np.asarray([0.2, 0.2])),
        "concrete3k": (order, np.asarray([0.4, 0.4])),
    }
    families = analysis.family_delta_arrays(arrays)
    np.testing.assert_allclose(families["khanhha"], [0.1, 0.1])
    assert set(families) == {"khanhha", "road420", "facade390", "concrete3k"}


def test_family_average_refuses_different_khanhha_case_order() -> None:
    arrays = {
        "khanhha_original": (["a", "b"], np.zeros(2)),
        "khanhha_noisy1": (["b", "a"], np.zeros(2)),
        "khanhha_noisy2": (["a", "b"], np.zeros(2)),
        "road420": (["a"], np.zeros(1)),
        "facade390": (["a"], np.zeros(1)),
        "concrete3k": (["a"], np.zeros(1)),
    }
    with pytest.raises(ValueError, match="same cases"):
        analysis.family_delta_arrays(arrays)
