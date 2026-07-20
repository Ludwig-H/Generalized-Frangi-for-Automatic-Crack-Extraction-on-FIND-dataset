from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest


CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
if str(CRACKSAM_ROOT) not in sys.path:
    sys.path.insert(0, str(CRACKSAM_ROOT))

import evaluate_logistic_gate as evaluation  # noqa: E402
from cracksam2.gating import (  # noqa: E402
    DEFAULT_GATE_FEATURES,
    LABEL_DEFINITION,
    GateProvenance,
    LogisticConfidenceGate,
    Standardizer,
)
from cracksam2.oof import strict_oof_training_contract  # noqa: E402


BASELINE_SHA = "1" * 64
EXTRACTOR_SHA = "2" * 64
CACHE_SHA = "3" * 64
PROTOCOL_SHA = "4" * 64
RESIDUAL_SHA_BY_FOLD = {
    str(fold): character * 64
    for fold, character in enumerate(("a", "b", "c", "d", "e"))
}
CHANNELS = (
    "similarity",
    "support",
    "hessian_magnitude",
    "winning_scale",
    "orientation_sin2",
    "orientation_cos2",
    "distance_to_skeleton",
)
FRANGI = {"scales": [1.0, 3.0], "R": 3, "K": 1, "tau": 0.18}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _oof_manifest() -> dict[str, object]:
    return {
        "schema": evaluation.OOF_SCHEMA,
        "schema_version": evaluation.OOF_SCHEMA_VERSION,
        "feature_names": list(DEFAULT_GATE_FEATURES),
        "label_minimum_gain": 0.0,
        "segmentation_threshold": 0.5,
        "artifacts": {
            "sam2_checkpoint": {"name": "sam2.pt", "bytes": 1, "sha256": "5" * 64},
            "baseline_checkpoint": {
                "name": "baseline.pt",
                "bytes": 1,
                "sha256": BASELINE_SHA,
            },
            "graph_cache": {
                "root": "/cache",
                "manifest": {
                    "name": ".cracksam2-frangi-graph-v2.json",
                    "bytes": 1,
                    "sha256": CACHE_SHA,
                },
                "extractor_sha256": EXTRACTOR_SHA,
                "frangi": FRANGI,
                "channels": list(CHANNELS),
                "contract_sha256": "6" * 64,
            },
            "protocol": {
                "dataset_list": {"name": "train.txt", "bytes": 1, "sha256": "7" * 64},
                "group_assignments": {
                    "name": "folds.csv",
                    "bytes": 1,
                    "sha256": "8" * 64,
                },
                "composite_sha256": PROTOCOL_SHA,
            },
        },
        "folds": {
            str(fold): {
                "role": "gate_calibration" if fold == 4 else "gate_fit",
                "evaluation_directory": f"fold-{fold}",
                "evaluation_contract": {"name": "evaluation_contract.json"},
                "per_image_csv": {"name": "per_image.csv"},
                "residual_checkpoint": {
                    "name": f"fold-{fold}.pt",
                    "bytes": 1,
                    "sha256": RESIDUAL_SHA_BY_FOLD[str(fold)],
                },
                "oof_training": strict_oof_training_contract(fold),
                "rows": 1,
                "source_groups": 1,
            }
            for fold in range(5)
        },
        "outputs": {
            "gate_fit_csv": {
                "path": "gate-fit.csv",
                "name": "gate-fit.csv",
                "bytes": 1,
                "sha256": "9" * 64,
                "folds": [0, 1, 2, 3],
                "rows": 4,
            },
            "gate_calibration_csv": {
                "path": "gate-calibration.csv",
                "name": "gate-calibration.csv",
                "bytes": 1,
                "sha256": "a" * 64,
                "folds": [4],
                "rows": 1,
            },
        },
    }


def _gate(oof_sha256: str) -> LogisticConfidenceGate:
    coefficients = np.zeros(len(DEFAULT_GATE_FEATURES), dtype=np.float64)
    coefficients[0] = 5.0
    return LogisticConfidenceGate(
        feature_names=DEFAULT_GATE_FEATURES,
        standardizer=Standardizer(
            mean=np.zeros(len(DEFAULT_GATE_FEATURES)),
            scale=np.ones(len(DEFAULT_GATE_FEATURES)),
        ),
        coefficients=coefficients,
        intercept=-2.5,
        l2=1.0,
        threshold=0.5,
        training_iterations=2,
        training_objective=0.25,
        training_sample_count=8,
        training_source_group_count=8,
        calibration={
            "status": "calibrated",
            "label_definition": LABEL_DEFINITION,
            "label_minimum_gain": 0.0,
        },
        provenance=GateProvenance(
            baseline_checkpoint_sha256=BASELINE_SHA,
            oof_manifest_sha256=oof_sha256,
            frangi_extractor_sha256=EXTRACTOR_SHA,
            frangi_cache_manifest_sha256=CACHE_SHA,
            protocol_sha256=PROTOCOL_SHA,
            train_csv_sha256="b" * 64,
            calibration_csv_sha256="c" * 64,
            label_definition=LABEL_DEFINITION,
            label_minimum_gain=0.0,
            git_commit="d" * 40,
        ),
    )


def _row(
    *,
    case: str,
    dataset: str,
    signal: float,
    baseline_iou: float,
    candidate_iou: float,
    baseline_dice: float,
    candidate_dice: float,
) -> dict[str, object]:
    delta_iou = candidate_iou - baseline_iou
    delta_dice = candidate_dice - baseline_dice
    values: dict[str, object] = {
        "case_name": case,
        "source_group": f"source-{case}",
        "dataset": dataset,
        "role": "historical_test",
        "fold": "",
        "baseline_iou": baseline_iou,
        "candidate_iou": candidate_iou,
        "candidate_iou_gain": delta_iou,
        "delta_iou": delta_iou,
        "baseline_dice": baseline_dice,
        "candidate_dice": candidate_dice,
        "candidate_dice_gain": delta_dice,
        "candidate_better": int(delta_iou > 0.0),
        "candidate_practical_gain": int(delta_iou >= 0.005),
        "candidate_harmful": int(delta_iou <= -0.005),
        "candidate_severe_harm": int(delta_iou <= -0.05),
        "relevant_baseline_entropy_mean": signal,
        "baseline_foreground_fraction": 0.1,
        "relevant_prediction_disagreement_rate": 0.2,
        "support_correction_probability_mean": 0.3,
        "foreground_probability_change_mean": 0.01,
        "frangi_similarity_support_mean": 0.7,
        "frangi_density": 0.25,
    }
    return {field: values[field] for field in evaluation.INPUT_ROW_FIELDS}


def _evaluation_dir(root: Path, dataset: str, rows: list[dict[str, object]]) -> Path:
    directory = root / dataset
    directory.mkdir(parents=True)
    names = [str(row["case_name"]) for row in rows]
    contract = {
        "schema": evaluation.EVALUATION_SCHEMA,
        "schema_version": evaluation.EVALUATION_SCHEMA_VERSION,
        "dataset": {
            "name": dataset,
            "role": "historical_test",
            "fold": "",
            "selected_samples": len(rows),
            "selected_sample_names_sha256": evaluation.sample_names_sha256(names),
        },
        "checkpoints": {
            "baseline": {"sha256": BASELINE_SHA},
            "residual": {"sha256": RESIDUAL_SHA_BY_FOLD["4"]},
        },
        "graph_cache": {
            "extractor_sha256": EXTRACTOR_SHA,
            "frangi": FRANGI,
            "channels": list(CHANNELS),
        },
        "segmentation_threshold": 0.5,
        "label_minimum_gain": 0.0,
        "gate_policy": {
            "feature_rows_only": True,
            "threshold_selected_by_this_command": False,
        },
    }
    _write_json(directory / "evaluation_contract.json", contract)
    with (directory / "per_image.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluation.INPUT_ROW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return directory


def _fixture(tmp_path: Path):
    oof_path = tmp_path / "oof.json"
    _write_json(oof_path, _oof_manifest())
    gate_path = tmp_path / "gate.json"
    gate = _gate(evaluation.sha256_file(oof_path))
    gate.save_json(gate_path)
    first = _evaluation_dir(
        tmp_path,
        "DatasetA",
        [
            _row(
                case="closed-harm.png",
                dataset="DatasetA",
                signal=0.1,
                baseline_iou=0.8,
                candidate_iou=0.6,
                baseline_dice=0.85,
                candidate_dice=0.7,
            ),
            _row(
                case="open-gain.png",
                dataset="DatasetA",
                signal=0.9,
                baseline_iou=0.4,
                candidate_iou=0.6,
                baseline_dice=0.5,
                candidate_dice=0.7,
            ),
        ],
    )
    return gate_path, oof_path, first


def test_frozen_gate_writes_exact_fallback_and_analytical_outputs(
    tmp_path: Path,
) -> None:
    gate_path, oof_path, source = _fixture(tmp_path)
    gate_before = gate_path.read_bytes()
    output = tmp_path / "analysis"
    args = argparse.Namespace(
        gate_json=gate_path,
        oof_manifest=oof_path,
        evaluation_dir=[source],
        output=output,
    )

    summary = evaluation.run(args)

    assert gate_path.read_bytes() == gate_before
    assert summary["policy"]["threshold_selected_or_recalibrated"] is False
    assert summary["overall"]["coverage"] == pytest.approx(0.5)
    assert summary["overall"]["precision"] == pytest.approx(1.0)
    assert summary["overall"]["recall"] == pytest.approx(1.0)
    assert 0.0 <= summary["overall"]["brier_score"] <= 1.0
    assert 0.0 <= summary["overall"]["expected_calibration_error"] <= 1.0
    assert summary["overall"]["loss_below_minus_0_05_count"] == 0
    assert set(summary["by_dataset"]) == {"DatasetA"}

    with (output / "per_image_gated.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = {row["case_name"]: row for row in csv.DictReader(stream)}
    closed = rows["closed-harm.png"]
    assert closed["selected_output"] == "baseline"
    assert closed["gated_iou"] == closed["baseline_iou"]
    assert closed["gated_dice"] == closed["baseline_dice"]
    assert float(closed["gated_iou_gain"]) == 0.0
    opened = rows["open-gain.png"]
    assert opened["selected_output"] == "candidate"
    assert opened["gated_iou"] == opened["candidate_iou"]

    with (output / "risk_coverage.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        risk = list(csv.DictReader(stream))
    serialized = [
        row
        for row in risk
        if row["scope"] == "overall"
        and row["is_serialized_gate_threshold"] == "1"
    ]
    assert len(serialized) == 1
    assert float(serialized[0]["threshold"]) == pytest.approx(0.5)
    assert (output / "summary.json").is_file()
    assert not list(output.glob("*.tmp"))


def test_gate_rejects_oof_manifest_changed_after_training(tmp_path: Path) -> None:
    gate_path, oof_path, source = _fixture(tmp_path)
    manifest = json.loads(oof_path.read_text(encoding="utf-8"))
    manifest["segmentation_threshold"] = 0.51
    _write_json(oof_path, manifest)

    with pytest.raises(ValueError, match="differs from gate provenance"):
        evaluation.run(
            argparse.Namespace(
                gate_json=gate_path,
                oof_manifest=oof_path,
                evaluation_dir=[source],
                output=tmp_path / "analysis",
            )
        )


def test_gate_rejects_evaluation_contract_from_another_baseline(
    tmp_path: Path,
) -> None:
    gate_path, oof_path, source = _fixture(tmp_path)
    contract_path = source / "evaluation_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["checkpoints"]["baseline"]["sha256"] = "e" * 64
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="differs from OOF manifest"):
        evaluation.run(
            argparse.Namespace(
                gate_json=gate_path,
                oof_manifest=oof_path,
                evaluation_dir=[source],
                output=tmp_path / "analysis",
            )
        )


def test_oof_compatibility_extracts_residual_checkpoint_sha_for_every_fold() -> None:
    compatibility = evaluation.oof_compatibility(_oof_manifest())

    assert dict(compatibility.residual_checkpoint_sha256_by_fold) == (
        RESIDUAL_SHA_BY_FOLD
    )


@pytest.mark.parametrize(
    ("role", "fold", "checkpoint_fold"),
    [
        ("gate_fit", "0", "0"),
        ("gate_fit", "1", "1"),
        ("gate_fit", "2", "2"),
        ("gate_fit", "3", "3"),
        ("gate_calibration", "4", "4"),
        ("development", "", "4"),
        ("historical_test", "", "4"),
    ],
)
def test_evaluation_contract_accepts_only_the_role_checkpoint_producer(
    tmp_path: Path,
    role: str,
    fold: str,
    checkpoint_fold: str,
) -> None:
    _, oof_path, source = _fixture(tmp_path)
    compatibility = evaluation.oof_compatibility(
        json.loads(oof_path.read_text(encoding="utf-8"))
    )
    contract = json.loads(
        (source / "evaluation_contract.json").read_text(encoding="utf-8")
    )
    contract["dataset"]["role"] = role
    contract["dataset"]["fold"] = fold
    contract["checkpoints"]["residual"]["sha256"] = RESIDUAL_SHA_BY_FOLD[
        checkpoint_fold
    ]

    dataset = evaluation.validate_evaluation_contract(contract, compatibility)

    assert dataset["role"] == role
    assert dataset["fold"] == fold


@pytest.mark.parametrize(
    ("role", "fold", "wrong_checkpoint_fold"),
    [
        ("gate_fit", "0", "1"),
        ("gate_calibration", "4", "3"),
        ("development", "", "3"),
        ("historical_test", "", "0"),
    ],
)
def test_evaluation_contract_rejects_checkpoint_from_another_fold(
    tmp_path: Path,
    role: str,
    fold: str,
    wrong_checkpoint_fold: str,
) -> None:
    _, oof_path, source = _fixture(tmp_path)
    compatibility = evaluation.oof_compatibility(
        json.loads(oof_path.read_text(encoding="utf-8"))
    )
    contract = json.loads(
        (source / "evaluation_contract.json").read_text(encoding="utf-8")
    )
    contract["dataset"]["role"] = role
    contract["dataset"]["fold"] = fold
    contract["checkpoints"]["residual"]["sha256"] = RESIDUAL_SHA_BY_FOLD[
        wrong_checkpoint_fold
    ]

    with pytest.raises(ValueError, match="residual_checkpoint_sha256"):
        evaluation.validate_evaluation_contract(contract, compatibility)


@pytest.mark.parametrize(
    ("role", "fold"),
    [
        ("gate_fit", "4"),
        ("gate_calibration", "3"),
        ("development", "4"),
        ("historical_test", "0"),
    ],
)
def test_evaluation_contract_rejects_incoherent_role_and_fold(
    tmp_path: Path,
    role: str,
    fold: str,
) -> None:
    _, oof_path, source = _fixture(tmp_path)
    compatibility = evaluation.oof_compatibility(
        json.loads(oof_path.read_text(encoding="utf-8"))
    )
    contract = json.loads(
        (source / "evaluation_contract.json").read_text(encoding="utf-8")
    )
    contract["dataset"]["role"] = role
    contract["dataset"]["fold"] = fold

    with pytest.raises(ValueError, match="role/fold is incoherent"):
        evaluation.validate_evaluation_contract(contract, compatibility)


def test_oof_manifest_refuses_non_inference_feature(tmp_path: Path) -> None:
    manifest = _oof_manifest()
    manifest["feature_names"] = [*DEFAULT_GATE_FEATURES, "delta_iou"]

    with pytest.raises(ValueError, match="DEFAULT_GATE_FEATURES"):
        evaluation.oof_compatibility(manifest)
