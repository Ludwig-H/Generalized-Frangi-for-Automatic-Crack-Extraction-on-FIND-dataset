from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest


CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from cracksam2.gating import (  # noqa: E402
    DEFAULT_GATE_FEATURES,
    LABEL_DEFINITION,
    GateProvenance,
    LogisticConfidenceGate,
    Standardizer,
    candidate_improvement_labels,
    compute_gate_features,
    inverse_group_frequency_weights,
    probability_reliability_metrics,
    select_conservative_threshold,
    select_logits_with_fallback,
    vectorize_feature_rows,
)
from train_logistic_gate import (  # noqa: E402
    assert_exact_oof_partitions,
    assert_train_calibration_disjoint,
    load_gate_csv,
    main as train_gate_main,
    sha256_file,
)


def _separable_data() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    generator = np.random.default_rng(3407)
    features = generator.normal(size=(600, len(DEFAULT_GATE_FEATURES)))
    latent_score = (
        2.2 * features[:, 0]
        - 1.6 * features[:, 2]
        + 1.3 * features[:, 5]
        + 0.15 * generator.normal(size=features.shape[0])
    )
    delta_iou = 0.03 * (latent_score - 0.35)
    labels = (delta_iou > 0.0).astype(np.int64)
    groups = tuple(f"source-{index // 2:03d}" for index in range(features.shape[0]))
    return (
        features[:400],
        labels[:400],
        groups[:400],
        features[400:],
        delta_iou[400:],
        groups[400:],
    )


def _provenance(*, minimum_gain: float = 0.0) -> GateProvenance:
    return GateProvenance(
        baseline_checkpoint_sha256="1" * 64,
        oof_manifest_sha256="2" * 64,
        frangi_extractor_sha256="3" * 64,
        frangi_cache_manifest_sha256="4" * 64,
        protocol_sha256="5" * 64,
        train_csv_sha256="6" * 64,
        calibration_csv_sha256="7" * 64,
        label_definition=LABEL_DEFINITION,
        label_minimum_gain=minimum_gain,
        git_commit="8" * 40,
    )


def _write_gate_csv(
    path: Path,
    rows: list[tuple[str, str, str, str, float]],
) -> None:
    fieldnames = [
        "case_name",
        "source_group",
        "fold",
        "dataset",
        "role",
        "delta_iou",
        *DEFAULT_GATE_FEATURES,
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (case, group, fold, dataset, delta) in enumerate(rows):
            record: dict[str, object] = {
                "case_name": case,
                "source_group": group,
                "fold": fold,
                "dataset": dataset,
                "role": "gate_calibration" if fold == "4" else "gate_fit",
                "delta_iou": delta,
            }
            record.update(
                {
                    name: 0.01 * (row_index + feature_index + 1)
                    for feature_index, name in enumerate(DEFAULT_GATE_FEATURES)
                }
            )
            writer.writerow(record)


def test_features_are_named_bounded_and_vectorized_in_contract_order() -> None:
    assert DEFAULT_GATE_FEATURES == (
        "relevant_baseline_entropy_mean",
        "baseline_foreground_fraction",
        "relevant_prediction_disagreement_rate",
        "support_correction_probability_mean",
        "foreground_probability_change_mean",
        "frangi_similarity_support_mean",
        "frangi_density",
    )
    baseline = np.zeros((4, 4), dtype=np.float64)
    candidate = np.full((4, 4), 2.0, dtype=np.float64)
    features = compute_gate_features(
        baseline,
        candidate,
        {"similarity": 0.7, "density": 0.2},
        frangi_support=np.ones((4, 4), dtype=np.uint8),
    )

    assert tuple(features) == DEFAULT_GATE_FEATURES
    assert features["relevant_baseline_entropy_mean"] == pytest.approx(1.0)
    assert features["baseline_foreground_fraction"] == pytest.approx(0.0)
    assert features["relevant_prediction_disagreement_rate"] == pytest.approx(1.0)
    assert features["support_correction_probability_mean"] > 0.0
    assert features["foreground_probability_change_mean"] > 0.0
    np.testing.assert_allclose(
        vectorize_feature_rows([features])[0],
        [features[name] for name in DEFAULT_GATE_FEATURES],
    )


def test_feature_validation_rejects_shape_and_unnormalized_frangi_values() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_gate_features(
            np.zeros((2, 2)),
            np.zeros((2, 3)),
            {"similarity": 0.1, "density": 0.2},
            frangi_support=np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match=r"normalized to \[0, 1\]"):
        compute_gate_features(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            {"similarity": 1.1, "density": 0.2},
            frangi_support=np.zeros((2, 2)),
        )


def test_gate_features_focus_on_frangi_support_and_keep_signed_area_change() -> None:
    baseline = np.full((4, 4), -8.0)
    candidate = baseline.copy()
    candidate[1, 1] = 8.0
    support = np.zeros((4, 4), dtype=np.uint8)
    support[1, 1] = 1

    features = compute_gate_features(
        baseline,
        candidate,
        {"similarity": 0.2, "density": 1.0 / 16.0},
        frangi_support=support,
    )

    assert features["relevant_prediction_disagreement_rate"] == pytest.approx(1.0)
    assert features["support_correction_probability_mean"] > 0.99
    assert features["foreground_probability_change_mean"] > 0.0


def test_standardizer_maps_constant_columns_to_zero() -> None:
    features = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    standardizer = Standardizer.fit(features)
    transformed = standardizer.transform(features)

    np.testing.assert_array_equal(transformed[:, 0], np.zeros(3))
    assert standardizer.scale[0] == 1.0
    assert transformed[:, 1].mean() == pytest.approx(0.0)


def test_inverse_crop_count_weights_standardization_by_source_group() -> None:
    features = np.array([[0.0], [0.0], [0.0], [10.0]])
    groups = ("many-crops", "many-crops", "many-crops", "single-crop")
    weights = inverse_group_frequency_weights(groups)
    standardizer = Standardizer.fit(features, sample_weight=weights)

    # Each source contributes total weight one: mean([0, 10]) == 5.
    assert weights[:3].sum() == pytest.approx(1.0)
    assert weights[3] == pytest.approx(1.0)
    assert standardizer.mean[0] == pytest.approx(5.0)


def test_logistic_fit_and_calibration_are_deterministic_and_separate() -> None:
    (
        train_x,
        train_y,
        train_groups,
        calibration_x,
        calibration_delta,
        calibration_groups,
    ) = _separable_data()
    first = LogisticConfidenceGate.fit(
        train_x, train_y, source_groups=train_groups, l2=0.05
    )
    second = LogisticConfidenceGate.fit(
        train_x, train_y, source_groups=train_groups, l2=0.05
    )

    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    assert first.intercept == second.intercept
    assert first.threshold == 0.5

    calibrated = first.calibrate(
        calibration_x,
        calibration_delta,
        source_groups=calibration_groups,
        minimum_precision=0.90,
        minimum_selected=20,
        minimum_selected_groups=10,
    )
    decisions = calibrated.predict_open(calibration_x)
    selected_labels = calibration_delta[decisions] > 0.0

    assert calibrated.calibration["status"] == "calibrated"
    assert selected_labels.size >= 20
    assert selected_labels.mean() >= 0.90
    # Calibration returns a new model and cannot alter the training fit.
    assert first.threshold == 0.5
    np.testing.assert_array_equal(first.coefficients, calibrated.coefficients)


def test_logistic_fit_is_invariant_to_duplicate_crops_from_one_source() -> None:
    base_x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    base_y = np.array([0, 0, 1, 1])
    base_groups = ("a", "b", "c", "d")
    duplicated_x = np.vstack((base_x, np.repeat(base_x[[0]], 7, axis=0)))
    duplicated_y = np.concatenate((base_y, np.zeros(7, dtype=np.int64)))
    duplicated_groups = (*base_groups, *("a" for _ in range(7)))

    base_gate = LogisticConfidenceGate.fit(
        base_x,
        base_y,
        source_groups=base_groups,
        feature_names=("signal",),
        l2=0.1,
    )
    duplicated_gate = LogisticConfidenceGate.fit(
        duplicated_x,
        duplicated_y,
        source_groups=duplicated_groups,
        feature_names=("signal",),
        l2=0.1,
    )

    np.testing.assert_allclose(
        duplicated_gate.standardizer.mean, base_gate.standardizer.mean, atol=1e-14
    )
    np.testing.assert_allclose(
        duplicated_gate.coefficients, base_gate.coefficients, atol=1e-12
    )
    assert duplicated_gate.intercept == pytest.approx(
        base_gate.intercept, abs=1e-12
    )


def test_threshold_requires_independent_groups_and_safe_signed_gain() -> None:
    probabilities = np.array([0.99, 0.98, 0.97, 0.96, 0.95])
    # Precision is 0.8, but the single severe loss makes both the mean signed
    # gain and severe-loss rate unsafe.
    deltas = np.array([-0.08, 0.01, 0.01, 0.01, 0.01])
    groups = ("a", "b", "c", "d", "e")
    threshold, metadata = select_conservative_threshold(
        probabilities,
        deltas,
        groups,
        minimum_precision=0.8,
        minimum_selected=5,
        minimum_selected_groups=5,
        minimum_mean_delta_iou=0.0,
        severe_loss_threshold=-0.05,
        maximum_severe_loss_rate=0.10,
    )
    assert threshold == 1.0
    assert metadata["status"] == "closed_no_reliable_threshold"

    safe_threshold, safe_metadata = select_conservative_threshold(
        probabilities[1:],
        deltas[1:],
        groups[1:],
        minimum_precision=1.0,
        minimum_selected=4,
        minimum_selected_groups=4,
        minimum_mean_delta_iou=0.0,
        severe_loss_threshold=-0.05,
        maximum_severe_loss_rate=0.0,
    )
    assert safe_threshold < 1.0
    assert safe_metadata["selected_groups"] == 4
    assert safe_metadata["selected_mean_delta_iou"] > 0.0
    assert safe_metadata["selected_severe_loss_rate"] == 0.0


def test_many_crops_from_one_group_cannot_satisfy_group_minimum() -> None:
    threshold, metadata = select_conservative_threshold(
        np.linspace(0.99, 0.8, 20),
        np.full(20, 0.02),
        ("same-source",) * 20,
        minimum_precision=1.0,
        minimum_selected=10,
        minimum_selected_groups=2,
    )

    assert threshold == 1.0
    assert metadata["status"] == "closed_no_reliable_threshold"


def test_threshold_closes_gate_when_reserved_data_cannot_support_it() -> None:
    threshold, metadata = select_conservative_threshold(
        np.array([0.9, 0.8, 0.7, 0.6]),
        np.array([-0.1, -0.2, -0.3, -0.4]),
        ("a", "b", "c", "d"),
        minimum_precision=0.8,
        minimum_selected=2,
        minimum_selected_groups=2,
    )

    assert threshold == 1.0
    assert metadata["status"] == "closed_no_reliable_threshold"
    assert 0.0 <= metadata["brier_score"] <= 1.0


def test_probability_reliability_metrics_are_zero_for_perfect_predictions() -> None:
    metrics = probability_reliability_metrics(
        np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0])
    )

    assert metrics["brier_score"] == pytest.approx(0.0)
    assert metrics["expected_calibration_error"] == pytest.approx(0.0)


def test_json_round_trip_is_strict_and_prediction_preserving(tmp_path: Path) -> None:
    (
        train_x,
        train_y,
        train_groups,
        calibration_x,
        calibration_delta,
        calibration_groups,
    ) = _separable_data()
    gate = (
        LogisticConfidenceGate.fit(
            train_x,
            train_y,
            source_groups=train_groups,
            l2=0.1,
        )
        .calibrate(
            calibration_x,
            calibration_delta,
            source_groups=calibration_groups,
            minimum_precision=0.85,
            minimum_selected=10,
            minimum_selected_groups=5,
        )
        .with_provenance(_provenance())
    )
    path = tmp_path / "gate.json"
    gate.save_json(path)
    restored = LogisticConfidenceGate.load_json(path)

    np.testing.assert_array_equal(
        restored.predict_proba(calibration_x), gate.predict_proba(calibration_x)
    )
    np.testing.assert_array_equal(
        restored.predict_open(calibration_x), gate.predict_open(calibration_x)
    )
    assert restored.feature_names == gate.feature_names
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["training"]["weighting"] == (
        "inverse_source_group_crop_count"
    )
    assert payload["provenance"]["baseline_checkpoint_sha256"] == "1" * 64
    assert payload["provenance"]["oof_manifest_sha256"] == "2" * 64

    malformed = json.loads(path.read_text(encoding="utf-8"))
    malformed["unexpected"] = "field"
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unknown"):
        LogisticConfidenceGate.load_json(path)


def test_json_refuses_missing_or_mismatched_provenance() -> None:
    train_x, train_y, train_groups, calibration_x, delta, calibration_groups = (
        _separable_data()
    )
    calibrated = LogisticConfidenceGate.fit(
        train_x,
        train_y,
        source_groups=train_groups,
        l2=0.1,
    ).calibrate(
        calibration_x,
        delta,
        source_groups=calibration_groups,
        minimum_selected=10,
        minimum_selected_groups=5,
    )
    with pytest.raises(ValueError, match="without strict provenance"):
        calibrated.to_dict()
    with pytest.raises(ValueError, match="minimum gains differ"):
        calibrated.with_provenance(_provenance(minimum_gain=0.01)).to_dict()


def test_csv_contract_and_train_calibration_leakage_checks(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    calibration_path = tmp_path / "calibration.csv"
    _write_gate_csv(
        train_path,
        [
            ("crop-a", "source-a", "0", "Road420", 0.02),
            ("crop-b", "source-a", "0", "Road420", -0.01),
        ],
    )
    _write_gate_csv(
        calibration_path,
        [("crop-c", "source-c", "1", "Road420", 0.03)],
    )
    train = load_gate_csv(train_path, feature_names=DEFAULT_GATE_FEATURES)
    calibration = load_gate_csv(
        calibration_path, feature_names=DEFAULT_GATE_FEATURES
    )

    assert train.case_names == ("crop-a", "crop-b")
    assert train.source_groups == ("source-a", "source-a")
    assert train.delta_iou.tolist() == pytest.approx([0.02, -0.01])
    assert_train_calibration_disjoint(train, calibration)
    assert len(sha256_file(train_path)) == 64

    _write_gate_csv(
        calibration_path,
        [("crop-c", "source-a", "1", "Road420", 0.03)],
    )
    leaking = load_gate_csv(
        calibration_path, feature_names=DEFAULT_GATE_FEATURES
    )
    with pytest.raises(ValueError, match="source_group overlap"):
        assert_train_calibration_disjoint(train, leaking)


def test_csv_partitions_require_all_fit_folds_and_reserved_fold_four(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.csv"
    calibration_path = tmp_path / "calibration.csv"
    _write_gate_csv(
        train_path,
        [
            (f"fit-{fold}", f"fit-source-{fold}", fold, "Road420", 0.02)
            for fold in ("0", "1", "2", "3")
        ],
    )
    _write_gate_csv(
        calibration_path,
        [("cal-4", "cal-source-4", "4", "Road420", 0.03)],
    )
    train = load_gate_csv(train_path, feature_names=DEFAULT_GATE_FEATURES)
    calibration = load_gate_csv(
        calibration_path, feature_names=DEFAULT_GATE_FEATURES
    )
    assert_exact_oof_partitions(train, calibration)

    _write_gate_csv(
        train_path,
        [("fit-0", "fit-source-0", "0", "Road420", 0.02)],
    )
    incomplete = load_gate_csv(train_path, feature_names=DEFAULT_GATE_FEATURES)
    with pytest.raises(ValueError, match="every OOF fold"):
        assert_exact_oof_partitions(incomplete, calibration)


def test_csv_requires_all_audit_columns(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    path.write_text(
        "case_name,source_group,fold,dataset,role\n"
        "case,group,0,Road420,gate_fit\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="delta_iou"):
        load_gate_csv(path, feature_names=DEFAULT_GATE_FEATURES)


def test_cli_hashes_every_artifact_and_writes_a_loadable_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    train_path = tmp_path / "train.csv"
    calibration_path = tmp_path / "calibration.csv"
    _write_gate_csv(
        train_path,
        [
            ("train-a", "train-source-a", "0", "Road420", -0.02),
            ("train-b", "train-source-b", "1", "Road420", -0.01),
            ("train-c", "train-source-c", "2", "Road420", 0.02),
            ("train-d", "train-source-d", "3", "Road420", 0.03),
        ],
    )
    _write_gate_csv(
        calibration_path,
        [
            ("cal-a", "cal-source-a", "4", "Road420", 0.01),
            ("cal-b", "cal-source-b", "4", "Road420", 0.02),
            ("cal-c", "cal-source-c", "4", "Road420", -0.01),
            ("cal-d", "cal-source-d", "4", "Road420", 0.03),
        ],
    )
    manifest_path = tmp_path / "oof_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest_sha = sha256_file(manifest_path)
    manifest = {
        "label_minimum_gain": 0.0,
        "outputs": {
            "gate_fit_csv": {"rows": 4},
            "gate_calibration_csv": {"rows": 4},
        },
        "artifacts": {
            "baseline_checkpoint": {"sha256": "1" * 64},
            "graph_cache": {
                "extractor_sha256": "2" * 64,
                "manifest": {"sha256": "3" * 64},
            },
            "protocol": {"composite_sha256": "4" * 64},
        },
    }
    monkeypatch.setattr(
        "train_logistic_gate.load_and_validate_oof_manifest",
        lambda *_args, **_kwargs: (manifest, manifest_sha),
    )
    monkeypatch.setattr(
        "train_logistic_gate.validate_manifest_output_csv",
        lambda _manifest, _name, path: {"sha256": sha256_file(Path(path))},
    )
    output = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_logistic_gate.py",
            "--train-csv",
            str(train_path),
            "--calibration-csv",
            str(calibration_path),
            "--output",
            str(output),
            "--oof-manifest",
            str(manifest_path),
            "--git-commit",
            "a" * 40,
            "--label-minimum-gain",
            "0",
            "--minimum-precision",
            "0.5",
            "--minimum-selected",
            "2",
            "--minimum-selected-groups",
            "2",
            "--minimum-mean-delta-iou",
            "-1",
            "--maximum-severe-loss-rate",
            "1",
        ],
    )

    train_gate_main()

    gate = LogisticConfidenceGate.load_json(output)
    summary = json.loads(capsys.readouterr().out)
    assert summary["train_source_groups"] == 4
    assert gate.provenance is not None
    assert gate.provenance.baseline_checkpoint_sha256 == "1" * 64
    assert gate.provenance.oof_manifest_sha256 == manifest_sha
    assert gate.provenance.train_csv_sha256 == sha256_file(train_path)


def test_closed_gate_is_an_exact_baseline_fallback() -> None:
    baseline = np.array([[[-3.0, 0.125], [9.0, -0.0]]], dtype=np.float32)
    candidate = baseline + np.float32(1.75)

    closed = select_logits_with_fallback(baseline, candidate, False)
    assert closed is baseline
    np.testing.assert_array_equal(closed, baseline)

    batch_baseline = np.concatenate((baseline, baseline + 2.0), axis=0)
    batch_candidate = batch_baseline + 7.0
    mixed = select_logits_with_fallback(
        batch_baseline, batch_candidate, np.array([False, True])
    )
    np.testing.assert_array_equal(mixed[0], batch_baseline[0])
    np.testing.assert_array_equal(mixed[1], batch_candidate[1])

    with pytest.raises(TypeError, match="Boolean"):
        select_logits_with_fallback(batch_baseline, batch_candidate, [0.1, 0.9])


def test_improvement_labels_use_scores_only_during_training() -> None:
    labels = candidate_improvement_labels(
        [0.4, 0.7, 0.5, 0.2],
        [0.6, 0.65, 0.52, 0.2],
        minimum_gain=0.01,
    )
    np.testing.assert_array_equal(labels, np.array([1, 0, 1, 0]))
