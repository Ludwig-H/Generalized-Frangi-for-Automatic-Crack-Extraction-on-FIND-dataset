"""Small regression tests for alignment, grouped splits and leakage prevention."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SPEC = importlib.util.spec_from_file_location("analyze_features", Path(__file__).with_name("analyze_features.py"))
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def cases_fixture():
    return pd.DataFrame({
        "image_id": [f"original::image{i}" for i in range(12)],
        "dataset": ["original"] * 12,
        "domain": ["A"] * 6 + ["B"] * 6,
        "source_family": ["family"] * 12,
        "physical_group": [f"group{i // 2}" for i in range(12)],
        "category": list(analysis.CATEGORIES) * 4,
        "delta_iou": [-.1, 0, .1] * 4,
    })


class FeatureProbeTests(unittest.TestCase):
    def test_archive_alignment_and_missing_or_duplicate_ids(self):
        cases = cases_fixture()
        values = np.arange(48).reshape(12, 4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases.to_csv(root / "cases.csv", index=False)
            np.savez(root / "features.npz", ids=cases.image_id.to_numpy(str)[::-1], mean=values[::-1])
            loaded_cases, features = analysis.load_inputs(root / "cases.csv", root / "features.npz")
            np.testing.assert_array_equal(features["mean"], values)
            self.assertEqual(loaded_cases.image_id.tolist(), cases.image_id.tolist())
            np.savez(root / "features.npz", ids=cases.image_id.to_numpy(str)[:-1], mean=values[:-1])
            with self.assertRaisesRegex(ValueError, "Missing features"):
                analysis.load_inputs(root / "cases.csv", root / "features.npz")
            ids = cases.image_id.to_numpy(str)
            ids[1] = ids[0]
            np.savez(root / "features.npz", ids=ids, mean=values)
            with self.assertRaisesRegex(ValueError, "unique"):
                analysis.load_inputs(root / "cases.csv", root / "features.npz")

    def test_related_images_stay_in_one_fold(self):
        cases = cases_fixture()
        y = analysis.encode_labels(cases)
        splits = analysis.group_splits(cases, y, n_splits=3)
        tested = []
        for train, test in splits:
            self.assertFalse(set(cases.physical_group.iloc[train]) & set(cases.physical_group.iloc[test]))
            tested.extend(test)
        self.assertEqual(sorted(tested), list(range(len(cases))))

    def test_channel_selection_and_scaling_use_training_only(self):
        y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        X = np.array([[-1, 0], [-1, 0], [0, 0], [0, 0], [1, 0], [1, 0], [100, -100], [100, 0], [100, 100]], dtype=float)
        train, test = np.arange(6), np.arange(6, 9)
        probabilities, selected, fitted = analysis.fit_predict(X, y, train, test, top_k=1)
        self.assertEqual(selected.tolist(), [0])
        np.testing.assert_allclose(fitted.steps[0][1].mean_, [0])
        self.assertEqual(probabilities.shape, (3, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)

    def test_single_training_class_preserves_global_probability_columns(self):
        y = np.array([2, 2, 0, 1])
        probabilities, _, _ = analysis.fit_predict(np.arange(8).reshape(4, 2), y, [0, 1], [2, 3])
        np.testing.assert_array_equal(probabilities, [[0, 0, 1], [0, 0, 1]])
        result = analysis.scores(np.array([1, 1]), probabilities)
        self.assertEqual(result["missing_classes"], ["degraded", "improved"])
        self.assertIsNone(result["macro_auroc_ovr"])

    def test_permutation_preserves_domain_counts(self):
        y = np.array([0, 1, 2, 2, 1, 0, 2])
        strata = np.array(["A", "A", "A", "B", "B", "B", "C"])
        permuted = analysis.permute_within_strata(y, strata, np.random.default_rng(42))
        for domain in np.unique(strata):
            np.testing.assert_array_equal(np.sort(y[strata == domain]), np.sort(permuted[strata == domain]))
        self.assertEqual(permuted[-1], y[-1])

    def test_representative_choice_is_label_independent_and_prefers_clean(self):
        cases = cases_fixture()
        cases.loc[0, "dataset"] = "khanhha_noisy1"
        cases.loc[1, "dataset"] = "khanhha_original"
        before = analysis.representative_indices(cases)
        cases["category"] = list(reversed(cases.category))
        cases["delta_iou"] *= -100
        np.testing.assert_array_equal(before, analysis.representative_indices(cases))
        self.assertIn(1, before)
        self.assertNotIn(0, before)

    def test_permutation_analysis_conditions_on_domain_and_source_family(self):
        cases = cases_fixture()
        cases["source_family"] = ["F0", "F0", "F1", "F1", "F0", "F0"] * 2
        y = analysis.encode_labels(cases)
        X = np.random.default_rng(42).normal(size=(len(cases), 3))
        result, distribution = analysis.permutation_analysis(cases, X, y, repeats=1, seed=42)
        self.assertEqual(result["stratification"], "domain x source_family")
        self.assertEqual(result["n_strata"], 4)
        self.assertEqual(result["n_label_variable_strata"], 2)
        self.assertEqual(result["n_permutable_representatives"], 4)
        self.assertEqual(set(result["stratum_counts"]), {"A::F0", "A::F1", "B::F0", "B::F1"})
        self.assertEqual(len(distribution), 1)

    def test_bootstrap_resamples_scenes_and_has_valid_intervals(self):
        cases = cases_fixture()
        y = analysis.encode_labels(cases)
        probabilities = np.eye(3)[y]
        intervals = analysis.bootstrap_intervals(cases, y, probabilities, repeats=8, seed=1)
        np.testing.assert_allclose(intervals["balanced_accuracy"], [1, 1])
        self.assertEqual(analysis.bootstrap_intervals(cases, y, probabilities, 0, 1)["balanced_accuracy"], None)


if __name__ == "__main__":
    unittest.main()
