from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest


CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
if str(CRACKSAM_ROOT) not in sys.path:
    sys.path.insert(0, str(CRACKSAM_ROOT))

import analyze_frangigraph_condition_pair as analysis  # noqa: E402
from cracksam2.data import sample_names_sha256  # noqa: E402
from cracksam2.gating import DEFAULT_GATE_FEATURES  # noqa: E402
from cracksam2.residual_evaluation import ROW_FIELDS  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _recorded_identity(path: Path) -> dict[str, object]:
    identity = analysis.file_identity(path)
    return {name: identity[name] for name in ("name", "bytes", "sha256")}


def _row(
    *,
    case: str,
    group: str,
    fold: str,
    candidate_iou: float,
) -> dict[str, object]:
    baseline_iou = 0.5
    baseline_dice = 0.6
    candidate_dice = max(0.0, min(1.0, baseline_dice + candidate_iou - baseline_iou))
    delta_iou = candidate_iou - baseline_iou
    values: dict[str, object] = {
        "case_name": case,
        "source_group": group,
        "dataset": "synthetic_oof",
        "role": "gate_calibration" if fold == "4" else "gate_fit",
        "fold": fold,
        "baseline_iou": baseline_iou,
        "candidate_iou": candidate_iou,
        "candidate_iou_gain": delta_iou,
        "delta_iou": delta_iou,
        "baseline_dice": baseline_dice,
        "candidate_dice": candidate_dice,
        "candidate_dice_gain": candidate_dice - baseline_dice,
        "candidate_better": int(delta_iou > 0.0),
        "candidate_practical_gain": int(delta_iou >= 0.005),
        "candidate_harmful": int(delta_iou <= -0.005),
        "candidate_severe_harm": int(delta_iou <= -0.05),
    }
    values.update(
        {
            name: 0.1 + index / 100.0
            for index, name in enumerate(DEFAULT_GATE_FEATURES)
        }
    )
    return {field: values[field] for field in ROW_FIELDS}


def _write_evaluation(
    run: Path,
    *,
    condition: str,
    fold: str,
    cases: list[tuple[str, str, float]],
    assignment_identity: dict[str, object],
    checkpoint_sha: str,
    legacy_safety_fields: bool = False,
) -> None:
    root = run / "oof_evaluations" / f"fold_{fold}"
    root.mkdir(parents=True)
    rows = [
        _row(case=case, group=group, fold=fold, candidate_iou=candidate_iou)
        for case, group, candidate_iou in cases
    ]
    with (root / "per_image.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    names = [case for case, _, _ in cases]
    contract = {
        "schema": "cracksam2.frangigraph-residual-evaluation",
        "schema_version": 1,
        "analytical_only": False,
        "dataset": {
            "name": "synthetic_oof",
            "role": "gate_calibration" if fold == "4" else "gate_fit",
            "fold": fold,
            "root": "/data",
            "list": {"name": "train.txt", "bytes": 10, "sha256": _sha("list")},
            "split": "train",
            "noise": "original",
            "image_size": [448, 448],
            "selected_samples": len(names),
            "selected_sample_names_sha256": sample_names_sha256(names),
            "group_assignments": assignment_identity,
        },
        "checkpoints": {
            "sam2": {"name": "sam2.pt", "bytes": 1, "sha256": _sha("sam2")},
            "baseline": {
                "name": "baseline.pt",
                "bytes": 2,
                "sha256": _sha("baseline"),
            },
            "residual": {
                "name": "latest.pt",
                "bytes": 3,
                "sha256": checkpoint_sha,
            },
        },
        "graph_cache": {
            "root": "/cache",
            "manifest": {
                "name": ".cracksam2-frangi-graph-v2.json",
                "bytes": 4,
                "sha256": _sha("cache"),
            },
            "extractor_sha256": _sha("extractor"),
            "frangi": {"scales": [1.0, 3.0]},
            "channels": [f"channel-{index}" for index in range(7)],
            "verify_cache_hashes": True,
            "verify_data_hashes": True,
        },
        "residual": {
            "raster_channels": 7,
            "high_resolution_channels": [32, 64],
            "hidden_channels": 32,
            "raster_preprocessing": {"fit": "same"},
            "training_raster_condition": condition,
            "evaluation_raster_condition": condition,
            "causal_raster_override": False,
            "checkpoint_held_out_fold": int(fold),
            "checkpoint_oof_training": {"held_out_fold": int(fold)},
            "checkpoint_training_state": "complete",
        },
        "segmentation_threshold": 0.5,
        "label_minimum_gain": 0.005,
        "gate_policy": {
            "feature_rows_only": True,
            "threshold_selected_by_this_command": False,
            "eligible_for_later_gate_fit": fold != "4",
            "threshold_may_later_be_calibrated_from_this_role": fold == "4",
            "historical_tests_forbidden_for_threshold_selection": True,
        },
    }
    if legacy_safety_fields:
        contract.pop("analytical_only")
        contract["gate_policy"].pop("eligible_for_later_gate_fit")
    _write_json(root / "evaluation_contract.json", contract)


def _write_workflow(
    run: Path,
    condition: str,
    *,
    schema_version: int,
) -> None:
    parameters = {
        "mode": "FULL",
        "fold_csv": f"/relocated/{run.name}/train_group_folds.csv",
        "protocol_manifest": f"/relocated/{run.name}/manifest.json",
        "train_list": f"/relocated/{run.name}/train.txt",
        "git_commit": _sha(f"commit-{run.name}"),
        "seed": "3407",
        "epochs": "3",
        "hidden_channels": "32",
        "graph_cache_root": "/cache",
        "sam2_checkpoint": "/checkpoints/sam2.pt",
        "baseline_checkpoint": "/checkpoints/baseline.pt",
    }
    if schema_version in (2, 3):
        parameters["raster_condition"] = condition
    contract = {
        "schema": "cracksam2.frangigraph-logistic-gate-workflow",
        "schema_version": schema_version,
        "parameters": parameters,
        "historical_test_inputs_used": False,
        "gate_fit_folds": [0, 1, 2, 3],
        "gate_calibration_fold": 4,
    }
    if schema_version == 3:
        contract["selector_parameter_units"] = {
            "profile_radii_feature_cells": "hiera_high_resolution_feature_cells",
            "evidence_dilation_feature_cells": (
                "hiera_high_resolution_feature_cells"
            ),
            "fusion_grid_source": (
                "SAM2ImageFeatures.high_resolution_features[0]"
            ),
        }
    _write_json(run / "workflow_contract.json", contract)


def test_load_workflow_contract_accepts_schema_three_selector_units(
    tmp_path: Path,
) -> None:
    run = tmp_path / "schema-three"
    _write_workflow(run, "correct", schema_version=3)

    payload, _ = analysis.load_workflow_contract(run)

    assert payload["schema_version"] == 3
    assert payload["selector_parameter_units"]["fusion_grid_source"] == (
        "SAM2ImageFeatures.high_resolution_features[0]"
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    correct = tmp_path / "correct"
    control = tmp_path / "no_evidence"
    group_csv = tmp_path / "train_group_folds.csv"
    assignment_rows: list[dict[str, str]] = []
    cases_by_fold: dict[str, list[tuple[str, str, str, float]]] = {}
    for fold in analysis.EXPECTED_FOLDS:
        cases_by_fold[fold] = [
            (f"case-{fold}-a.png", f"group-{fold}-a", "A", 0.1),
            (f"case-{fold}-b.png", f"group-{fold}-b", "B", -0.1),
        ]
    # Unequal crop counts make the crop mean differ from the group-first mean.
    cases_by_fold["0"].append(("case-0-a2.png", "group-0-a", "A", 0.1))
    with group_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sample_name",
                "source_family",
                "physical_group",
                "oof_fold",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for fold in analysis.EXPECTED_FOLDS:
            for case, group, family, _ in cases_by_fold[fold]:
                row = {
                    "sample_name": case,
                    "source_family": family,
                    "physical_group": group,
                    "oof_fold": fold,
                }
                assignment_rows.append(row)
                writer.writerow(row)
    assignment_identity = _recorded_identity(group_csv)
    _write_workflow(
        correct, "correct", schema_version=1
    )
    _write_workflow(
        control, "no_evidence", schema_version=2
    )
    for fold in analysis.EXPECTED_FOLDS:
        correct_cases = [
            (case, group, 0.5 + effect)
            for case, group, _, effect in cases_by_fold[fold]
        ]
        control_cases = [
            (case, group, 0.5) for case, group, _, _ in cases_by_fold[fold]
        ]
        _write_evaluation(
            correct,
            condition="correct",
            fold=fold,
            cases=correct_cases,
            assignment_identity=assignment_identity,
            checkpoint_sha=_sha(f"correct-{fold}"),
            legacy_safety_fields=True,
        )
        _write_evaluation(
            control,
            condition="no_evidence",
            fold=fold,
            cases=control_cases,
            assignment_identity=assignment_identity,
            checkpoint_sha=_sha(f"control-{fold}"),
        )
    return correct, control, group_csv


def _args(
    correct: Path, control: Path, groups: Path, output: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        correct_run=correct,
        no_evidence_run=control,
        group_assignments=groups,
        design="auto",
        output=output,
        bootstrap_repetitions=500,
        seed=3407,
    )


def test_paired_cluster_bootstrap_resamples_groups_within_strata() -> None:
    groups = [
        analysis.GroupEffect(
            group_id=f"dataset::group-{index}",
            dataset="dataset",
            source_group=f"group-{index}",
            source_family="A" if index < 3 else "B",
            fold="0",
            crops=1,
            mean_effect_iou=value,
            minimum_effect_iou=value,
            maximum_effect_iou=value,
        )
        for index, value in enumerate((-0.3, 0.0, 0.6, 0.1))
    ]
    kwargs = {
        "stratum": lambda group: f"{group.fold}::{group.source_family}",
        "repetitions": 1_000,
        "seed": 3407,
    }

    first = analysis.paired_clustered_bootstrap(groups, **kwargs)
    repeated = analysis.paired_clustered_bootstrap(groups, **kwargs)

    assert first == repeated
    assert first["estimate"] == pytest.approx(0.1)
    assert first["ci95_low"] < first["estimate"] < first["ci95_high"]
    assert first["bootstrap_valid_fraction"] == 1.0


def test_end_to_end_retrained_contrast_is_paired_group_first_and_reproducible(
    tmp_path: Path,
) -> None:
    correct, control, groups = _fixture(tmp_path)
    first = analysis.run(_args(correct, control, groups, tmp_path / "analysis-1"))
    repeated = analysis.run(_args(correct, control, groups, tmp_path / "analysis-2"))

    assert first["estimand"]["name"] == (
        "equal_capacity_retrained_condition_contrast"
    )
    assert first["estimand"]["same_checkpoint_ablation"] is False
    workflow_audit = first["provenance"]["workflow_pair_audit"]
    assert workflow_audit["correct_schema_version"] == 1
    assert workflow_audit["no_evidence_schema_version"] == 2
    assert workflow_audit["correct_condition_source"] == (
        "implicit_correct_from_schema_v1"
    )
    assert workflow_audit["git_commit_different"] is True
    assert all(
        item["different"]
        for item in workflow_audit["relocatable_source_paths"].values()
    )
    assert workflow_audit["scientific_parameters_equal"] is True
    assert first["join_audit"]["paired_crops"] == 11
    assert first["join_audit"]["physical_groups"] == 10
    # Each family A group has +0.1 and each B group -0.1, hence equal-group 0.
    assert first["group_balanced_overall"]["estimate"] == pytest.approx(0.0)
    assert first["crop_level_descriptive"]["mean_effect_iou"] != pytest.approx(0.0)
    assert first["group_balanced_overall"] == repeated["group_balanced_overall"]
    assert first["by_source_family"] == repeated["by_source_family"]
    assert (tmp_path / "analysis-1" / "summary.json").is_file()
    assert (tmp_path / "analysis-1" / "per_family.csv").is_file()
    assert (tmp_path / "analysis-1" / "per_fold.csv").is_file()
    assert (tmp_path / "analysis-1" / "per_group.csv").is_file()
    assert (tmp_path / "analysis-1" / "per_crop.csv").is_file()


def test_pair_rejects_hyperparameter_or_contract_drift(tmp_path: Path) -> None:
    correct, control, groups = _fixture(tmp_path)
    workflow_path = control / "workflow_contract.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["parameters"]["seed"] = "99"
    _write_json(workflow_path, workflow)
    with pytest.raises(ValueError, match="beyond allowed provenance fields.*seed"):
        analysis.run(_args(correct, control, groups, tmp_path / "bad-seed"))

    workflow["parameters"]["seed"] = "3407"
    _write_json(workflow_path, workflow)
    contract_path = control / "oof_evaluations" / "fold_2" / "evaluation_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["segmentation_threshold"] = 0.4
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="segmentation_threshold"):
        analysis.run(_args(correct, control, groups, tmp_path / "bad-contract"))


def test_schema_v1_is_implicit_correct_only_when_all_oof_contracts_confirm(
    tmp_path: Path,
) -> None:
    correct, control, groups = _fixture(tmp_path)
    path = correct / "oof_evaluations" / "fold_3" / "evaluation_contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["residual"]["training_raster_condition"] = "no_evidence"
    contract["residual"]["evaluation_raster_condition"] = "no_evidence"
    _write_json(path, contract)

    with pytest.raises(ValueError, match="correct run has wrong condition contract"):
        analysis.run(_args(correct, control, groups, tmp_path / "bad-v1-condition"))


def test_pair_rejects_physical_or_baseline_mismatch(tmp_path: Path) -> None:
    correct, control, groups = _fixture(tmp_path)
    csv_path = control / "oof_evaluations" / "fold_0" / "per_image.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["source_group"] = "wrong-physical-group"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="source_group differs"):
        analysis.run(_args(correct, control, groups, tmp_path / "bad-group"))

    rows[0]["source_group"] = "group-0-a"
    rows[0]["baseline_iou"] = "0.4"
    rows[0]["candidate_iou_gain"] = "0.1"
    rows[0]["delta_iou"] = "0.1"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="baseline metric baseline_iou differs"):
        analysis.run(_args(correct, control, groups, tmp_path / "bad-baseline"))


def test_same_checkpoint_input_ablation_is_detected_and_qualified(
    tmp_path: Path,
) -> None:
    correct, control, groups = _fixture(tmp_path)
    workflow_path = control / "workflow_contract.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["parameters"]["raster_condition"] = "correct"
    _write_json(workflow_path, workflow)
    for fold in analysis.EXPECTED_FOLDS:
        correct_contract = json.loads(
            (
                correct
                / "oof_evaluations"
                / f"fold_{fold}"
                / "evaluation_contract.json"
            ).read_text(encoding="utf-8")
        )
        # Exercise backwards compatibility with primary contracts emitted
        # before the two analytical-safety fields became explicit.
        correct_contract.pop("analytical_only", None)
        correct_contract["gate_policy"].pop("eligible_for_later_gate_fit", None)
        _write_json(
            correct
            / "oof_evaluations"
            / f"fold_{fold}"
            / "evaluation_contract.json",
            correct_contract,
        )
        path = control / "oof_evaluations" / f"fold_{fold}" / "evaluation_contract.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        contract["checkpoints"]["residual"] = correct_contract["checkpoints"][
            "residual"
        ]
        contract["residual"]["training_raster_condition"] = "correct"
        contract["residual"]["evaluation_raster_condition"] = "no_evidence"
        contract["residual"]["causal_raster_override"] = True
        contract["analytical_only"] = True
        contract["gate_policy"][
            "threshold_may_later_be_calibrated_from_this_role"
        ] = False
        contract["gate_policy"]["eligible_for_later_gate_fit"] = False
        _write_json(path, contract)
    # An evaluator-only counterfactual tree is sufficient: its five immutable
    # contracts bind the same checkpoints back to the correct training run.
    workflow_path.unlink()

    summary = analysis.run(
        _args(correct, control, groups, tmp_path / "same-checkpoint")
    )

    assert summary["estimand"]["name"] == "same_checkpoint_input_ablation"
    assert summary["estimand"]["same_checkpoint_ablation"] is True
    assert summary["provenance"]["no_evidence_provenance_mode"].startswith(
        "eval_only"
    )
