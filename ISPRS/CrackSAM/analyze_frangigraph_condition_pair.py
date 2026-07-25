#!/usr/bin/env python3
"""Paired condition contrast between Frangi ``correct`` and ``no_evidence`` runs.

The primary design compares two residuals retrained with identical capacity,
folds, seed and hyperparameters.  The intervention is the availability of the
Frangi raster during both training and inference.  It is therefore a learned-
system condition contrast, not a same-checkpoint input ablation.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cracksam2.data import sample_names_sha256
from cracksam2.graph_cache import sha256_file
from cracksam2.residual_evaluation import (
    EVALUATION_SCHEMA,
    EVALUATION_SCHEMA_VERSION,
    ROW_FIELDS,
    normalize_evaluation_row,
)


ANALYSIS_SCHEMA = "cracksam2.frangigraph-paired-condition-contrast"
ANALYSIS_SCHEMA_VERSION = 1
WORKFLOW_SCHEMA = "cracksam2.frangigraph-logistic-gate-workflow"
WORKFLOW_RELOCATABLE_PATH_KEYS: tuple[str, ...] = (
    "fold_csv",
    "protocol_manifest",
    "train_list",
)
EXPECTED_FOLDS: tuple[str, ...] = ("0", "1", "2", "3", "4")
DEFAULT_BOOTSTRAP_REPETITIONS = 20_000
DEFAULT_SEED = 3407
BOOTSTRAP_CHUNK_SIZE = 512
CI_QUANTILES = (0.025, 0.975)
DESIGNS: tuple[str, ...] = (
    "auto",
    "equal_capacity_retrained_condition_contrast",
    "same_checkpoint_input_ablation",
)


@dataclass(frozen=True)
class Assignment:
    sample_name: str
    source_family: str
    physical_group: str
    fold: str


@dataclass(frozen=True)
class EvaluationTable:
    fold: str
    root: Path
    contract: Mapping[str, Any]
    contract_identity: Mapping[str, object]
    csv_identity: Mapping[str, object]
    rows: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class PairedCrop:
    case_name: str
    dataset: str
    source_group: str
    source_family: str
    fold: str
    correct_candidate_iou: float
    no_evidence_candidate_iou: float
    attributable_effect_iou: float

    @property
    def group_id(self) -> str:
        return f"{self.dataset}::{self.source_group}"


@dataclass(frozen=True)
class GroupEffect:
    group_id: str
    dataset: str
    source_group: str
    source_family: str
    fold: str
    crops: int
    mean_effect_iou: float
    minimum_effect_iou: float
    maximum_effect_iou: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correct-run",
        type=Path,
        required=True,
        help="Run root containing workflow_contract.json and oof_evaluations/.",
    )
    parser.add_argument(
        "--no-evidence-run",
        type=Path,
        required=True,
        help="Matched no_evidence run root.",
    )
    parser.add_argument(
        "--group-assignments",
        type=Path,
        help=(
            "Optional relocation of train_group_folds.csv. By default the path "
            "recorded in the workflow contract is used."
        ),
    )
    parser.add_argument(
        "--design",
        choices=DESIGNS,
        default="auto",
        help=(
            "auto detects retrained versus same-checkpoint pairing from residual "
            "checkpoint SHA-256 values."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def file_identity(path: Path) -> dict[str, object]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "name": source.name,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _read_json_stable(
    path: Path, label: str
) -> tuple[dict[str, Any], dict[str, object]]:
    source = path.expanduser().resolve()
    before = file_identity(source)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label}: {source}") from exc
    after = file_identity(source)
    if before != after:
        raise RuntimeError(f"{label} changed while being read: {source}")
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {source}")
    return value, after


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not Boolean")
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_sha256(value: object, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a full SHA-256")
    return digest


def load_workflow_contract(run_root: Path) -> tuple[dict[str, Any], dict[str, object]]:
    root = run_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Run root does not exist: {root}")
    payload, identity = _read_json_stable(
        root / "workflow_contract.json", "workflow contract"
    )
    required = {
        "schema",
        "schema_version",
        "parameters",
        "historical_test_inputs_used",
        "gate_fit_folds",
        "gate_calibration_fold",
    }
    if payload.get("schema") != WORKFLOW_SCHEMA:
        raise ValueError("Unknown workflow contract schema")
    schema_version = payload.get("schema_version")
    if schema_version not in (1, 2, 3):
        raise ValueError("Paired condition analysis supports workflow schema 1, 2 or 3")
    if schema_version == 3:
        required.add("selector_parameter_units")
    if set(payload) != required:
        raise ValueError("Workflow contract has missing or unknown top-level fields")
    if schema_version == 3:
        selector_units = _mapping(
            payload.get("selector_parameter_units"), "selector parameter units"
        )
        expected_selector_units = {
            "profile_radii_feature_cells": "hiera_high_resolution_feature_cells",
            "evidence_dilation_feature_cells": (
                "hiera_high_resolution_feature_cells"
            ),
            "fusion_grid_source": (
                "SAM2ImageFeatures.high_resolution_features[0]"
            ),
        }
        if selector_units != expected_selector_units:
            raise ValueError("Workflow schema version 3 has invalid selector units")
    parameters = _mapping(payload.get("parameters"), "workflow parameters")
    required_parameters = (
        "fold_csv",
        "protocol_manifest",
        "train_list",
        "git_commit",
        "seed",
    )
    for key in required_parameters:
        if not isinstance(parameters.get(key), str) or not parameters.get(key):
            raise ValueError(f"Workflow contract has no valid {key}")
    if schema_version == 1:
        if "raster_condition" in parameters:
            raise ValueError(
                "Workflow schema version 1 must not declare raster_condition"
            )
    elif parameters.get("raster_condition") not in ("correct", "no_evidence"):
        raise ValueError(
            f"Workflow schema version {schema_version} has no valid raster_condition"
        )
    commit = str(parameters["git_commit"]).lower()
    if len(commit) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("Workflow git_commit must be a full commit hash")
    if payload.get("historical_test_inputs_used") is not False:
        raise ValueError("Workflow contract used historical-test inputs")
    if payload.get("gate_fit_folds") != [0, 1, 2, 3]:
        raise ValueError("Workflow gate-fit folds differ from 0-3")
    if payload.get("gate_calibration_fold") != 4:
        raise ValueError("Workflow calibration fold differs from 4")
    return payload, identity


def _expected_role(fold: str) -> str:
    return "gate_calibration" if fold == "4" else "gate_fit"


def _validate_row_metrics(row: Mapping[str, object], fold: str, line: int) -> None:
    bounded = ("baseline_iou", "candidate_iou", "baseline_dice", "candidate_dice")
    values = {name: _finite_float(row[name], name) for name in bounded}
    for name, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Fold {fold} {name} is outside [0, 1] at line {line}")
    tolerance = 2e-12
    consistency = {
        "candidate_iou_gain": values["candidate_iou"] - values["baseline_iou"],
        "delta_iou": values["candidate_iou"] - values["baseline_iou"],
        "candidate_dice_gain": (
            values["candidate_dice"] - values["baseline_dice"]
        ),
    }
    for name, expected in consistency.items():
        observed = _finite_float(row[name], name)
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"Fold {fold} {name} is internally inconsistent at line {line}"
            )


def load_evaluation_table(root: Path, fold: str) -> EvaluationTable:
    directory = root.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Missing evaluation directory for fold {fold}: {directory}")
    contract, contract_identity = _read_json_stable(
        directory / "evaluation_contract.json", f"fold {fold} evaluation contract"
    )
    required_contract_fields = {
        "schema",
        "schema_version",
        "dataset",
        "checkpoints",
        "graph_cache",
        "residual",
        "segmentation_threshold",
        "label_minimum_gain",
        "gate_policy",
    }
    allowed_contract_fields = required_contract_fields | {"analytical_only"}
    if not required_contract_fields.issubset(contract) or not set(
        contract
    ).issubset(allowed_contract_fields):
        raise ValueError(f"Fold {fold} contract has missing or unknown fields")
    analytical_only = contract.get("analytical_only", False)
    if not isinstance(analytical_only, bool):
        raise ValueError(f"Fold {fold} analytical_only must be Boolean")
    if contract.get("schema") != EVALUATION_SCHEMA:
        raise ValueError(f"Fold {fold} has an unknown evaluation schema")
    if contract.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"Fold {fold} has an unsupported evaluation schema version")
    dataset = _mapping(contract.get("dataset"), f"fold {fold} dataset")
    if dataset.get("fold") != fold or dataset.get("role") != _expected_role(fold):
        raise ValueError(f"Fold {fold} dataset role/fold is incoherent")
    dataset_name = dataset.get("name")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(f"Fold {fold} has no dataset name")
    graph_cache = _mapping(contract.get("graph_cache"), f"fold {fold} graph cache")
    if graph_cache.get("verify_cache_hashes") is not True:
        raise ValueError(f"Fold {fold} did not verify graph-cache file hashes")
    if graph_cache.get("verify_data_hashes") is not True:
        raise ValueError(f"Fold {fold} did not verify source image/mask hashes")

    csv_path = directory / "per_image.csv"
    before = file_identity(csv_path)
    rows: list[Mapping[str, object]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != ROW_FIELDS:
                raise ValueError(f"Fold {fold} CSV header differs from ROW_FIELDS")
            for line, raw in enumerate(reader, start=2):
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError(f"Malformed fold {fold} CSV row at line {line}")
                try:
                    row = normalize_evaluation_row(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid fold {fold} CSV row at line {line}: {exc}"
                    ) from exc
                _validate_row_metrics(row, fold, line)
                rows.append(row)
    except OSError as exc:
        raise ValueError(f"Cannot read fold {fold} CSV: {csv_path}") from exc
    after = file_identity(csv_path)
    if before != after:
        raise RuntimeError(f"Fold {fold} CSV changed while being read")
    if not rows:
        raise ValueError(f"Fold {fold} CSV is empty")
    names = [str(row["case_name"]) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Fold {fold} contains duplicate case_name values")
    if any(
        row["dataset"] != dataset_name
        or row["fold"] != fold
        or row["role"] != _expected_role(fold)
        for row in rows
    ):
        raise ValueError(f"Fold {fold} CSV identity differs from its contract")
    if dataset.get("selected_samples") != len(rows):
        raise ValueError(f"Fold {fold} row count differs from its contract")
    if dataset.get("selected_sample_names_sha256") != sample_names_sha256(names):
        raise ValueError(f"Fold {fold} sample order differs from its contract")
    return EvaluationTable(
        fold=fold,
        root=directory,
        contract=contract,
        contract_identity=contract_identity,
        csv_identity=after,
        rows=tuple(rows),
    )


def load_condition_run(root: Path) -> dict[str, EvaluationTable]:
    run_root = root.expanduser().resolve()
    return {
        fold: load_evaluation_table(
            run_root / "oof_evaluations" / f"fold_{fold}", fold
        )
        for fold in EXPECTED_FOLDS
    }


def _residual_checkpoint_sha(table: EvaluationTable) -> str:
    checkpoints = _mapping(table.contract.get("checkpoints"), "checkpoints")
    residual = _mapping(checkpoints.get("residual"), "residual checkpoint")
    return _validate_sha256(residual.get("sha256"), "residual checkpoint sha256")


def detect_design(
    correct: Mapping[str, EvaluationTable],
    no_evidence: Mapping[str, EvaluationTable],
    requested: str,
) -> str:
    matches = [
        _residual_checkpoint_sha(correct[fold])
        == _residual_checkpoint_sha(no_evidence[fold])
        for fold in EXPECTED_FOLDS
    ]
    if all(matches):
        detected = "same_checkpoint_input_ablation"
    elif not any(matches):
        detected = "equal_capacity_retrained_condition_contrast"
    else:
        raise ValueError("Residual checkpoint pairing mixes same and retrained folds")
    if requested != "auto" and requested != detected:
        raise ValueError(
            f"Requested design {requested!r} differs from detected {detected!r}"
        )
    return detected


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Contract is not canonical finite JSON") from exc


def _diff_paths(left: object, right: object, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        output: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                output.append(path)
            else:
                output.extend(_diff_paths(left[key], right[key], path))
        return output
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        output = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            output.extend(_diff_paths(left_item, right_item, f"{prefix}[{index}]"))
        return output
    return [] if left == right else [prefix]


def _workflow_condition(
    workflow: Mapping[str, Any], label: str
) -> tuple[str, str]:
    version = workflow.get("schema_version")
    parameters = _mapping(workflow.get("parameters"), f"{label} parameters")
    if version == 1:
        if "raster_condition" in parameters:
            raise ValueError(
                f"{label} schema-v1 workflow unexpectedly declares raster_condition"
            )
        return "correct", "implicit_correct_from_schema_v1"
    if version in (2, 3):
        condition = parameters.get("raster_condition")
        if condition not in ("correct", "no_evidence"):
            raise ValueError(f"{label} schema-v{version} workflow has invalid condition")
        return str(condition), f"explicit_from_schema_v{version}"
    raise ValueError(f"{label} workflow has unsupported schema version")


def validate_workflow_pair(
    correct: Mapping[str, Any],
    no_evidence: Mapping[str, Any],
    design: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    correct_parameters = dict(_mapping(correct.get("parameters"), "correct parameters"))
    control_parameters = dict(
        _mapping(no_evidence.get("parameters"), "no_evidence parameters")
    )
    correct_condition, correct_condition_source = _workflow_condition(
        correct, "correct"
    )
    control_condition, control_condition_source = _workflow_condition(
        no_evidence, "no_evidence"
    )
    if design == "equal_capacity_retrained_condition_contrast":
        if (correct_condition, control_condition) != ("correct", "no_evidence"):
            raise ValueError(
                "Retrained contrast requires workflow conditions correct/no_evidence"
            )
    else:
        if (correct_condition, control_condition) != ("correct", "correct"):
            raise ValueError(
                "Same-checkpoint ablation requires two correct-training workflows"
            )

    allowed_parameter_differences = {
        "raster_condition",
        "git_commit",
        *WORKFLOW_RELOCATABLE_PATH_KEYS,
    }
    path_audit = {
        key: {
            "correct": correct_parameters[key],
            "no_evidence": control_parameters[key],
            "different": correct_parameters[key] != control_parameters[key],
        }
        for key in WORKFLOW_RELOCATABLE_PATH_KEYS
    }
    workflow_audit: dict[str, object] = {
        "correct_schema_version": correct["schema_version"],
        "no_evidence_schema_version": no_evidence["schema_version"],
        "correct_condition": correct_condition,
        "no_evidence_condition": control_condition,
        "correct_condition_source": correct_condition_source,
        "no_evidence_condition_source": control_condition_source,
        "legacy_v1_correct_confirmed_by_oof_contracts": (
            correct["schema_version"] == 1
        ),
        "correct_git_commit": correct_parameters["git_commit"],
        "no_evidence_git_commit": control_parameters["git_commit"],
        "git_commit_different": (
            correct_parameters["git_commit"] != control_parameters["git_commit"]
        ),
        "relocatable_source_paths": path_audit,
        "relocatable_path_identities_bound_by_evaluation_contracts": True,
        "allowed_difference_fields": [
            "schema_version",
            "parameters.raster_condition",
            "parameters.git_commit",
            *(f"parameters.{key}" for key in WORKFLOW_RELOCATABLE_PATH_KEYS),
        ],
    }
    for key in allowed_parameter_differences:
        correct_parameters[key] = f"<allowed:{key}>"
        control_parameters[key] = f"<allowed:{key}>"
    normalized_correct = dict(correct)
    normalized_control = dict(no_evidence)
    normalized_correct["schema_version"] = "<compatible:1-or-2-or-3>"
    normalized_control["schema_version"] = "<compatible:1-or-2-or-3>"
    normalized_correct["parameters"] = correct_parameters
    normalized_control["parameters"] = control_parameters
    differences = _diff_paths(normalized_correct, normalized_control)
    if differences:
        raise ValueError(
            "Workflow contracts differ beyond allowed provenance fields: "
            + ", ".join(differences[:12])
        )
    if _canonical(normalized_correct) != _canonical(normalized_control):
        raise AssertionError("Workflow diff detector missed a contract difference")
    scientific_parameters = {
        key: value
        for key, value in _mapping(correct["parameters"], "correct parameters").items()
        if key not in allowed_parameter_differences
    }
    workflow_audit["scientific_parameter_names"] = sorted(scientific_parameters)
    workflow_audit["scientific_parameters_equal"] = True
    return scientific_parameters, workflow_audit


def _condition_values(table: EvaluationTable) -> tuple[str, str, bool]:
    residual = _mapping(table.contract.get("residual"), "residual contract")
    training = residual.get("training_raster_condition")
    evaluation = residual.get("evaluation_raster_condition")
    override = residual.get("causal_raster_override")
    if not isinstance(training, str) or not isinstance(evaluation, str):
        raise ValueError("Residual contract has invalid raster conditions")
    if not isinstance(override, bool):
        raise ValueError("Residual contract has invalid causal_raster_override")
    return training, evaluation, override


def _analytical_only(table: EvaluationTable) -> bool:
    value = table.contract.get("analytical_only", False)
    if not isinstance(value, bool):
        raise ValueError("Evaluation analytical_only must be Boolean")
    return value


def _threshold_calibration_eligible(table: EvaluationTable) -> bool:
    policy = _mapping(table.contract.get("gate_policy"), "gate policy")
    value = policy.get("threshold_may_later_be_calibrated_from_this_role")
    if not isinstance(value, bool):
        raise ValueError("Gate threshold eligibility must be Boolean")
    return value


def _gate_fit_eligible(table: EvaluationTable) -> bool:
    policy = _mapping(table.contract.get("gate_policy"), "gate policy")
    value = policy.get("eligible_for_later_gate_fit")
    if value is None:
        # Compatibility with primary contracts created before this explicit
        # analytical-safety flag existed. Their role carried the same meaning.
        if _analytical_only(table):
            raise ValueError(
                "Analytical-only contract lacks eligible_for_later_gate_fit"
            )
        return table.fold != "4"
    if not isinstance(value, bool):
        raise ValueError("Gate-fit eligibility must be Boolean")
    return value


def _normalized_evaluation_contract(
    contract: Mapping[str, Any], design: str
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(contract))
    analytical_only = normalized.get("analytical_only", False)
    if not isinstance(analytical_only, bool):
        raise ValueError("Evaluation analytical_only must be Boolean")
    normalized["analytical_only"] = analytical_only
    gate_policy = dict(_mapping(normalized["gate_policy"], "gate policy"))
    if "eligible_for_later_gate_fit" not in gate_policy:
        dataset = _mapping(normalized["dataset"], "dataset contract")
        gate_policy["eligible_for_later_gate_fit"] = (
            dataset.get("role") == "gate_fit" and not analytical_only
        )
    normalized["gate_policy"] = gate_policy
    residual = dict(_mapping(normalized["residual"], "residual contract"))
    residual["training_raster_condition"] = "<condition>"
    residual["evaluation_raster_condition"] = "<condition>"
    if design == "same_checkpoint_input_ablation":
        residual["causal_raster_override"] = "<override>"
        normalized["analytical_only"] = "<analytical-only>"
        gate_policy["threshold_may_later_be_calibrated_from_this_role"] = (
            "<analytical-only>"
        )
        gate_policy["eligible_for_later_gate_fit"] = "<analytical-only>"
        normalized["gate_policy"] = gate_policy
    normalized["residual"] = residual
    if design == "equal_capacity_retrained_condition_contrast":
        checkpoints = dict(_mapping(normalized["checkpoints"], "checkpoints"))
        checkpoint = _mapping(checkpoints["residual"], "residual checkpoint")
        # Name remains a useful structural assertion; bytes and digest are
        # expected treatment descendants and therefore may differ.
        checkpoints["residual"] = {
            "name": checkpoint.get("name"),
            "bytes": "<treatment-output>",
            "sha256": "<treatment-output>",
        }
        normalized["checkpoints"] = checkpoints
    return normalized


def validate_evaluation_pairs(
    correct: Mapping[str, EvaluationTable],
    no_evidence: Mapping[str, EvaluationTable],
    design: str,
) -> None:
    correct_shas: list[str] = []
    control_shas: list[str] = []
    for fold in EXPECTED_FOLDS:
        left = correct[fold]
        right = no_evidence[fold]
        correct_shas.append(_residual_checkpoint_sha(left))
        control_shas.append(_residual_checkpoint_sha(right))
        left_condition = _condition_values(left)
        right_condition = _condition_values(right)
        if design == "equal_capacity_retrained_condition_contrast":
            if left_condition != ("correct", "correct", False):
                raise ValueError(
                    f"Fold {fold} correct run has wrong condition contract"
                )
            if right_condition != ("no_evidence", "no_evidence", False):
                raise ValueError(
                    f"Fold {fold} no_evidence run has wrong condition contract"
                )
            if _analytical_only(left) or _analytical_only(right):
                raise ValueError(
                    f"Fold {fold} retrained conditions must not be analytical-only"
                )
            expected_fit_eligibility = fold != "4"
            expected_threshold_eligibility = fold == "4"
            for table in (left, right):
                if _gate_fit_eligible(table) != expected_fit_eligibility:
                    raise ValueError(f"Fold {fold} gate-fit eligibility is invalid")
                if (
                    _threshold_calibration_eligible(table)
                    != expected_threshold_eligibility
                ):
                    raise ValueError(
                        f"Fold {fold} threshold eligibility is invalid"
                    )
        else:
            if left_condition != ("correct", "correct", False):
                raise ValueError(
                    f"Fold {fold} correct run has wrong condition contract"
                )
            if right_condition != ("correct", "no_evidence", True):
                raise ValueError(
                    f"Fold {fold} no_evidence ablation has wrong override contract"
                )
            if _analytical_only(left) or not _analytical_only(right):
                raise ValueError(
                    f"Fold {fold} same-checkpoint analytical-only flags are invalid"
                )
            expected_primary_eligibility = fold == "4"
            if _threshold_calibration_eligible(left) != expected_primary_eligibility:
                raise ValueError(f"Fold {fold} primary gate eligibility is invalid")
            if _threshold_calibration_eligible(right):
                raise ValueError(
                    f"Fold {fold} input ablation must be ineligible for calibration"
                )
            if _gate_fit_eligible(left) != (fold != "4"):
                raise ValueError(f"Fold {fold} primary gate-fit eligibility is invalid")
            if _gate_fit_eligible(right):
                raise ValueError(
                    f"Fold {fold} input ablation must be ineligible for gate fit"
                )
        normalized_left = _normalized_evaluation_contract(left.contract, design)
        normalized_right = _normalized_evaluation_contract(right.contract, design)
        differences = _diff_paths(normalized_left, normalized_right)
        if differences:
            raise ValueError(
                f"Fold {fold} contracts differ beyond the designed condition: "
                + ", ".join(differences[:12])
            )
    if len(set(correct_shas)) != len(EXPECTED_FOLDS):
        raise ValueError("Correct run must bind five distinct OOF residual checkpoints")
    if len(set(control_shas)) != len(EXPECTED_FOLDS):
        raise ValueError("Control run must bind five distinct OOF residual checkpoints")
    if design == "equal_capacity_retrained_condition_contrast" and set(
        correct_shas
    ).intersection(control_shas):
        raise ValueError(
            "Retrained conditions must use disjoint residual checkpoint identities"
        )


def load_assignments(path: Path) -> tuple[dict[str, Assignment], dict[str, object]]:
    source = path.expanduser().resolve()
    before = file_identity(source)
    assignments: dict[str, Assignment] = {}
    group_contracts: dict[str, tuple[str, str]] = {}
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"sample_name", "source_family", "physical_group", "oof_fold"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Group CSV must contain {sorted(required)}")
            for line, row in enumerate(reader, start=2):
                values = {name: (row.get(name) or "").strip() for name in required}
                if any(not value for value in values.values()):
                    raise ValueError(f"Empty group assignment at {source}:{line}")
                name = values["sample_name"]
                if name in assignments:
                    raise ValueError(f"Duplicate group assignment for {name!r}")
                fold = values["oof_fold"]
                if fold not in EXPECTED_FOLDS:
                    raise ValueError(f"Invalid assignment fold for {name!r}: {fold}")
                assignment = Assignment(
                    sample_name=name,
                    source_family=values["source_family"],
                    physical_group=values["physical_group"],
                    fold=fold,
                )
                previous = group_contracts.setdefault(
                    assignment.physical_group,
                    (assignment.source_family, assignment.fold),
                )
                if previous != (assignment.source_family, assignment.fold):
                    raise ValueError(
                        "Physical group crosses family or fold: "
                        f"{assignment.physical_group}"
                    )
                assignments[name] = assignment
    except OSError as exc:
        raise ValueError(f"Cannot read group assignments: {source}") from exc
    after = file_identity(source)
    if before != after:
        raise RuntimeError("Group assignments changed while being read")
    if not assignments:
        raise ValueError("Group assignments are empty")
    return assignments, after


def _recorded_identity_matches(
    observed: Mapping[str, object], recorded: object
) -> bool:
    expected = _mapping(recorded, "recorded file identity")
    return all(
        observed.get(name) == expected.get(name)
        for name in ("name", "bytes", "sha256")
    )


def validate_assignment_provenance(
    assignment_identity: Mapping[str, object],
    *runs: Mapping[str, EvaluationTable],
) -> None:
    for run in runs:
        for fold in EXPECTED_FOLDS:
            dataset = _mapping(run[fold].contract.get("dataset"), "dataset contract")
            if not _recorded_identity_matches(
                assignment_identity, dataset.get("group_assignments")
            ):
                raise ValueError(
                    f"Fold {fold} group assignments differ from evaluation provenance"
                )


def pair_rows(
    correct: Mapping[str, EvaluationTable],
    no_evidence: Mapping[str, EvaluationTable],
    assignments: Mapping[str, Assignment],
) -> list[PairedCrop]:
    pairs: list[PairedCrop] = []
    observed_names: set[str] = set()
    for fold in EXPECTED_FOLDS:
        left_rows = {
            (str(row["dataset"]), str(row["case_name"])): row
            for row in correct[fold].rows
        }
        right_rows = {
            (str(row["dataset"]), str(row["case_name"])): row
            for row in no_evidence[fold].rows
        }
        if set(left_rows) != set(right_rows):
            missing = sorted(set(left_rows) - set(right_rows))[:5]
            extra = sorted(set(right_rows) - set(left_rows))[:5]
            raise ValueError(
                f"Fold {fold} case identities are not paired; "
                f"missing_control={missing}, extra_control={extra}"
            )
        for identity in left_rows:
            left = left_rows[identity]
            right = right_rows[identity]
            case_name = identity[1]
            if case_name in observed_names:
                raise ValueError(f"case_name is not globally unique: {case_name!r}")
            observed_names.add(case_name)
            if case_name not in assignments:
                raise ValueError(f"Missing physical assignment for {case_name!r}")
            assignment = assignments[case_name]
            for field in ("source_group", "dataset", "role", "fold"):
                if left[field] != right[field]:
                    raise ValueError(
                        f"Paired {field} differs for {identity!r}: "
                        f"{left[field]!r} != {right[field]!r}"
                    )
            if left["source_group"] != assignment.physical_group:
                raise ValueError(f"Physical group differs for {case_name!r}")
            if fold != assignment.fold:
                raise ValueError(f"Physical fold differs for {case_name!r}")
            for metric in ("baseline_iou", "baseline_dice"):
                left_baseline = _finite_float(left[metric], f"correct {metric}")
                right_baseline = _finite_float(right[metric], f"control {metric}")
                if not math.isclose(
                    left_baseline, right_baseline, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"Paired baseline metric {metric} differs for {identity!r}"
                    )
            correct_iou = _finite_float(
                left["candidate_iou"], "correct candidate_iou"
            )
            control_iou = _finite_float(
                right["candidate_iou"], "no_evidence candidate_iou"
            )
            pairs.append(
                PairedCrop(
                    case_name=case_name,
                    dataset=identity[0],
                    source_group=str(left["source_group"]),
                    source_family=assignment.source_family,
                    fold=fold,
                    correct_candidate_iou=correct_iou,
                    no_evidence_candidate_iou=control_iou,
                    attributable_effect_iou=correct_iou - control_iou,
                )
            )
    if set(observed_names) != set(assignments):
        missing = sorted(set(assignments) - observed_names)[:5]
        extra = sorted(observed_names - set(assignments))[:5]
        raise ValueError(
            "Paired evaluations do not exactly cover group assignments; "
            f"missing={missing}, extra={extra}"
        )
    return pairs


def aggregate_groups(pairs: Sequence[PairedCrop]) -> list[GroupEffect]:
    grouped: dict[str, list[PairedCrop]] = {}
    for pair in pairs:
        grouped.setdefault(pair.group_id, []).append(pair)
    if not grouped:
        raise ValueError("Cannot aggregate an empty paired dataset")
    output: list[GroupEffect] = []
    for group_id in sorted(grouped):
        members = grouped[group_id]
        datasets = {member.dataset for member in members}
        source_groups = {member.source_group for member in members}
        families = {member.source_family for member in members}
        folds = {member.fold for member in members}
        identities = (datasets, source_groups, families, folds)
        if any(len(values) != 1 for values in identities):
            raise ValueError(f"Physical group crosses dataset/family/fold: {group_id}")
        values = np.asarray(
            [member.attributable_effect_iou for member in members], dtype=np.float64
        )
        output.append(
            GroupEffect(
                group_id=group_id,
                dataset=next(iter(datasets)),
                source_group=next(iter(source_groups)),
                source_family=next(iter(families)),
                fold=next(iter(folds)),
                crops=len(members),
                mean_effect_iou=float(np.mean(values)),
                minimum_effect_iou=float(np.min(values)),
                maximum_effect_iou=float(np.max(values)),
            )
        )
    return output


def paired_clustered_bootstrap(
    groups: Sequence[GroupEffect],
    *,
    stratum: Callable[[GroupEffect], str],
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if not groups:
        raise ValueError("bootstrap requires at least one physical group")
    values = np.asarray([group.mean_effect_iou for group in groups], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Group effects must be finite")
    strata: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        label = stratum(group)
        if not label:
            raise ValueError("Bootstrap stratum labels must be non-empty")
        strata.setdefault(label, []).append(index)
    indices_by_stratum = [
        np.asarray(indices, dtype=np.int64)
        for _, indices in sorted(strata.items())
    ]
    draws = np.empty(repetitions, dtype=np.float64)
    generator = np.random.default_rng(seed)
    for start in range(0, repetitions, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, repetitions)
        chunk = stop - start
        sums = np.zeros(chunk, dtype=np.float64)
        counts = 0
        for indices in indices_by_stratum:
            positions = generator.integers(
                0, indices.size, size=(chunk, indices.size)
            )
            sums += values[indices[positions]].sum(axis=1)
            counts += indices.size
        draws[start:stop] = sums / counts
    low, high = np.quantile(draws, CI_QUANTILES)
    return {
        "estimate": float(np.mean(values)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "groups": len(groups),
        "bootstrap_repetitions": repetitions,
        "bootstrap_valid_repetitions": repetitions,
        "bootstrap_valid_fraction": 1.0,
    }


def _effect_counts(groups: Sequence[GroupEffect]) -> dict[str, int]:
    values = np.asarray([group.mean_effect_iou for group in groups])
    return {
        "positive_groups": int(np.count_nonzero(values > 0.0)),
        "zero_groups": int(np.count_nonzero(values == 0.0)),
        "negative_groups": int(np.count_nonzero(values < 0.0)),
    }


def _subset_statistics(
    groups: Sequence[GroupEffect],
    *,
    stratum: Callable[[GroupEffect], str],
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    result = paired_clustered_bootstrap(
        groups,
        stratum=stratum,
        repetitions=repetitions,
        seed=seed,
    )
    result.update(
        {
            "crops": sum(group.crops for group in groups),
            **_effect_counts(groups),
        }
    )
    return result


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_csv_atomic(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(fields),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> dict[str, object]:
    repetitions = int(args.bootstrap_repetitions)
    seed = int(args.seed)
    if repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    if seed < 0:
        raise ValueError("--seed must be non-negative")
    correct_root = args.correct_run.expanduser().resolve()
    control_root = args.no_evidence_run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if correct_root == control_root:
        raise ValueError("Correct and no_evidence run roots must be distinct")
    if output in (correct_root, control_root):
        raise ValueError("Output must differ from both input run roots")

    correct_workflow, correct_workflow_identity = load_workflow_contract(correct_root)
    correct_evaluations = load_condition_run(correct_root)
    control_evaluations = load_condition_run(control_root)
    design = detect_design(correct_evaluations, control_evaluations, args.design)
    # In particular, this proves that a schema-v1 primary really is the legacy
    # implicit-correct condition before that workflow omission is accepted.
    validate_evaluation_pairs(correct_evaluations, control_evaluations, design)
    control_workflow_path = control_root / "workflow_contract.json"
    control_workflow: dict[str, Any] | None
    control_workflow_identity: dict[str, object] | None
    if control_workflow_path.is_file():
        control_workflow, control_workflow_identity = load_workflow_contract(
            control_root
        )
        scientific_workflow_parameters, workflow_pair_audit = validate_workflow_pair(
            correct_workflow, control_workflow, design
        )
        control_provenance_mode = "matched_workflow_contract"
    elif design == "equal_capacity_retrained_condition_contrast":
        raise ValueError(
            "Retrained no_evidence run requires its own workflow_contract.json"
        )
    else:
        correct_parameters = dict(
            _mapping(correct_workflow.get("parameters"), "correct parameters")
        )
        correct_condition, correct_condition_source = _workflow_condition(
            correct_workflow, "correct"
        )
        if correct_condition != "correct":
            raise ValueError("Same-checkpoint source workflow must train correct")
        scientific_workflow_parameters = {
            key: value
            for key, value in correct_parameters.items()
            if key
            not in {
                "raster_condition",
                "git_commit",
                *WORKFLOW_RELOCATABLE_PATH_KEYS,
            }
        }
        workflow_pair_audit = {
            "correct_schema_version": correct_workflow["schema_version"],
            "no_evidence_schema_version": None,
            "correct_condition": correct_condition,
            "no_evidence_condition": "no_evidence",
            "correct_condition_source": correct_condition_source,
            "no_evidence_condition_source": (
                "five analytical evaluation contracts"
            ),
            "legacy_v1_correct_confirmed_by_oof_contracts": (
                correct_workflow["schema_version"] == 1
            ),
            "correct_git_commit": correct_parameters["git_commit"],
            "no_evidence_git_commit": None,
            "git_commit_different": None,
            "relocatable_source_paths": None,
            "relocatable_path_identities_bound_by_evaluation_contracts": True,
            "allowed_difference_fields": [],
            "scientific_parameter_names": sorted(scientific_workflow_parameters),
            "scientific_parameters_equal": None,
        }
        control_workflow = None
        control_workflow_identity = None
        control_provenance_mode = (
            "eval_only_contracts_bound_to_correct_workflow_checkpoints"
        )
    recorded_fold_csv = Path(
        str(_mapping(correct_workflow["parameters"], "parameters")["fold_csv"])
    )
    assignment_path = (
        args.group_assignments
        if args.group_assignments is not None
        else recorded_fold_csv
    )
    assignments, assignment_identity = load_assignments(assignment_path)
    validate_assignment_provenance(
        assignment_identity, correct_evaluations, control_evaluations
    )
    pairs = pair_rows(correct_evaluations, control_evaluations, assignments)
    groups = aggregate_groups(pairs)

    overall = _subset_statistics(
        groups,
        stratum=lambda group: (
            f"fold={group.fold}::family={group.source_family}"
        ),
        repetitions=repetitions,
        seed=seed,
    )
    families: dict[str, dict[str, float | int]] = {}
    for family in sorted({group.source_family for group in groups}):
        subset = [group for group in groups if group.source_family == family]
        families[family] = _subset_statistics(
            subset,
            stratum=lambda group: f"fold={group.fold}",
            repetitions=repetitions,
            seed=seed,
        )
    folds: dict[str, dict[str, float | int]] = {}
    for fold in EXPECTED_FOLDS:
        subset = [group for group in groups if group.fold == fold]
        folds[fold] = _subset_statistics(
            subset,
            stratum=lambda group: f"family={group.source_family}",
            repetitions=repetitions,
            seed=seed,
        )

    crop_effects = np.asarray(
        [pair.attributable_effect_iou for pair in pairs], dtype=np.float64
    )
    evaluation_provenance: list[dict[str, object]] = []
    for fold in EXPECTED_FOLDS:
        correct_table = correct_evaluations[fold]
        control_table = control_evaluations[fold]
        evaluation_provenance.append(
            {
                "fold": fold,
                "correct": {
                    "path": str(correct_table.root),
                    "contract": correct_table.contract_identity,
                    "per_image_csv": correct_table.csv_identity,
                    "residual_checkpoint_sha256": _residual_checkpoint_sha(
                        correct_table
                    ),
                },
                "no_evidence": {
                    "path": str(control_table.root),
                    "contract": control_table.contract_identity,
                    "per_image_csv": control_table.csv_identity,
                    "residual_checkpoint_sha256": _residual_checkpoint_sha(
                        control_table
                    ),
                },
            }
        )
    summary: dict[str, object] = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "estimand": {
            "name": design,
            "crop_definition": (
                "candidate_iou_correct - candidate_iou_no_evidence"
            ),
            "primary_aggregation": (
                "mean crop effect within dataset::source_group, then equal group mean"
            ),
            "interpretation": (
                "total contribution of Frangi information to the separately learned "
                "equal-capacity system (training plus inference)"
                if design == "equal_capacity_retrained_condition_contrast"
                else (
                    "acute no-evidence input ablation on identical residual checkpoints"
                )
            ),
            "same_checkpoint_ablation": design == "same_checkpoint_input_ablation",
        },
        "bootstrap": {
            "method": "paired clustered stratified percentile",
            "unit": "dataset::source_group",
            "stratification": "fold x source_family",
            "confidence_level": 0.95,
            "repetitions": repetitions,
            "seed": seed,
        },
        "provenance": {
            "correct_run": str(correct_root),
            "no_evidence_run": str(control_root),
            "correct_workflow_contract": correct_workflow_identity,
            "no_evidence_workflow_contract": control_workflow_identity,
            "no_evidence_provenance_mode": control_provenance_mode,
            "workflow_pair_audit": workflow_pair_audit,
            "scientific_workflow_parameters": scientific_workflow_parameters,
            "group_assignments": assignment_identity,
            "evaluation_pairs": evaluation_provenance,
        },
        "join_audit": {
            "paired_crops": len(pairs),
            "physical_groups": len(groups),
            "source_families": len(families),
            "folds": list(EXPECTED_FOLDS),
            "exact_case_identity_match": True,
            "exact_physical_group_and_fold_match": True,
            "baseline_metrics_match": True,
            "protocol_provenance_match": True,
        },
        "crop_level_descriptive": {
            "crops": len(pairs),
            "mean_effect_iou": float(np.mean(crop_effects)),
            "median_effect_iou": float(np.median(crop_effects)),
            "minimum_effect_iou": float(np.min(crop_effects)),
            "maximum_effect_iou": float(np.max(crop_effects)),
            "positive_crops": int(np.count_nonzero(crop_effects > 0.0)),
            "zero_crops": int(np.count_nonzero(crop_effects == 0.0)),
            "negative_crops": int(np.count_nonzero(crop_effects < 0.0)),
        },
        "group_balanced_overall": overall,
        "by_source_family": families,
        "by_fold": folds,
        "limitations": [
            (
                "For the retrained design, the contrast includes optimization-path "
                "differences caused by Frangi availability; it is not a same-weight "
                "input ablation."
            ),
            (
                "The historical baseline was trained on all training folds, so this "
                "condition contrast remains exploratory rather than confirmatory."
            ),
            (
                "Workflow commits may differ to introduce the no_evidence control; "
                "equivalence is enforced for serialized scientific parameters and "
                "evaluation provenance, not asserted from commit identity."
            ),
            (
                "Percentile intervals describe the observed physical-group population "
                "and do not account for new training seeds or domain shift."
            ),
        ],
    }

    family_rows = [
        {"source_family": family, **statistics}
        for family, statistics in families.items()
    ]
    fold_rows = [{"fold": fold, **statistics} for fold, statistics in folds.items()]
    group_rows = [
        {
            "group_id": group.group_id,
            "dataset": group.dataset,
            "source_group": group.source_group,
            "source_family": group.source_family,
            "fold": group.fold,
            "crops": group.crops,
            "mean_effect_iou": group.mean_effect_iou,
            "minimum_effect_iou": group.minimum_effect_iou,
            "maximum_effect_iou": group.maximum_effect_iou,
        }
        for group in groups
    ]
    crop_rows = [
        {
            "case_name": pair.case_name,
            "dataset": pair.dataset,
            "source_group": pair.source_group,
            "source_family": pair.source_family,
            "fold": pair.fold,
            "correct_candidate_iou": pair.correct_candidate_iou,
            "no_evidence_candidate_iou": pair.no_evidence_candidate_iou,
            "attributable_effect_iou": pair.attributable_effect_iou,
        }
        for pair in pairs
    ]
    statistics_fields = (
        "estimate",
        "ci95_low",
        "ci95_high",
        "groups",
        "crops",
        "positive_groups",
        "zero_groups",
        "negative_groups",
        "bootstrap_repetitions",
        "bootstrap_valid_repetitions",
        "bootstrap_valid_fraction",
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / "summary.json", summary)
    _write_csv_atomic(
        output / "per_family.csv",
        family_rows,
        ("source_family", *statistics_fields),
    )
    _write_csv_atomic(
        output / "per_fold.csv", fold_rows, ("fold", *statistics_fields)
    )
    _write_csv_atomic(
        output / "per_group.csv",
        group_rows,
        (
            "group_id",
            "dataset",
            "source_group",
            "source_family",
            "fold",
            "crops",
            "mean_effect_iou",
            "minimum_effect_iou",
            "maximum_effect_iou",
        ),
    )
    _write_csv_atomic(
        output / "per_crop.csv",
        crop_rows,
        (
            "case_name",
            "dataset",
            "source_group",
            "source_family",
            "fold",
            "correct_candidate_iou",
            "no_evidence_candidate_iou",
            "attributable_effect_iou",
        ),
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(
        json.dumps(
            {
                "estimand": summary["estimand"],
                "group_balanced_overall": summary["group_balanced_overall"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
