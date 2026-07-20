#!/usr/bin/env python3
"""Assemble the five immutable out-of-fold evaluations for the logistic gate.

The output manifest is the commit marker for the two assembled CSV files.  It
binds every row to its evaluation contract and, through those contracts, to
the five residual checkpoints and the shared SAM 2, baseline, Frangi cache,
extractor, and group-safe protocol identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cracksam2.gating import DEFAULT_GATE_FEATURES
from cracksam2.oof import validate_strict_oof_training_contract


OOF_MANIFEST_SCHEMA = "cracksam2.logistic-gate-oof-data"
OOF_MANIFEST_SCHEMA_VERSION = 2
EVALUATION_SCHEMA = "cracksam2.frangigraph-residual-evaluation"
EVALUATION_SCHEMA_VERSION = 1
EXPECTED_FOLDS: tuple[str, ...] = ("0", "1", "2", "3", "4")
FIT_FOLDS: tuple[str, ...] = ("0", "1", "2", "3")
CALIBRATION_FOLDS: tuple[str, ...] = ("4",)
MAX_CONTRACT_BYTES = 10_000_000
MAX_OOF_MANIFEST_BYTES = 5_000_000

REQUIRED_ROW_COLUMNS: tuple[str, ...] = (
    "case_name",
    "source_group",
    "dataset",
    "role",
    "fold",
    "delta_iou",
    *DEFAULT_GATE_FEATURES,
)


@dataclass(frozen=True)
class FoldEvaluation:
    fold: str
    role: str
    directory: Path
    contract: dict[str, Any]
    fieldnames: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    evaluation_contract: dict[str, object]
    per_image_csv: dict[str, object]
    residual_checkpoint: dict[str, object]
    oof_training: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-dir",
        action="append",
        required=True,
        metavar="FOLD=PATH",
        help=(
            "Evaluation directory for one held-out fold. Repeat exactly five "
            "times with folds 0, 1, 2, 3 and 4."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory receiving gate_fit.csv, gate_calibration.csv and the manifest.",
    )
    return parser.parse_args()


def sha256_file(path: str | os.PathLike[str]) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: object, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a full SHA-256 digest")
    return digest


def _canonical_json_sha256(value: object) -> str:
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("OOF provenance must be canonical finite JSON") from exc
    return hashlib.sha256(serialized).hexdigest()


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _recorded_identity(value: object, name: str) -> dict[str, object]:
    payload = _require_mapping(value, name)
    required = {"name", "bytes", "sha256"}
    if not required.issubset(payload):
        raise ValueError(f"{name} must contain name, bytes and sha256")
    artifact_name = payload["name"]
    byte_count = payload["bytes"]
    if not isinstance(artifact_name, str) or not artifact_name:
        raise ValueError(f"{name}.name must be non-empty")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError(f"{name}.bytes must be a non-negative integer")
    return {
        "name": artifact_name,
        "bytes": byte_count,
        "sha256": _validate_sha256(payload["sha256"], f"{name}.sha256"),
    }


def local_file_identity(path: str | os.PathLike[str]) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "name": source.name,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _identity_without_path(identity: Mapping[str, object]) -> dict[str, object]:
    return {name: identity[name] for name in ("name", "bytes", "sha256")}


def _same_artifact(
    observed: Mapping[str, object], expected: Mapping[str, object]
) -> bool:
    return all(observed.get(name) == expected.get(name) for name in ("bytes", "sha256"))


def _load_json_stable(
    path: Path, *, maximum_bytes: int
) -> tuple[dict[str, Any], dict[str, object]]:
    before = local_file_identity(path)
    if int(before["bytes"]) > maximum_bytes:
        raise ValueError(f"JSON file is unexpectedly large: {path}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number is forbidden: {value}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    after = local_file_identity(path)
    if before != after:
        raise RuntimeError(f"File changed while being read: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload, after


def _sample_names_sha256(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_fold_directories(values: Sequence[str]) -> dict[str, Path]:
    if len(values) != len(EXPECTED_FOLDS):
        raise ValueError("--fold-dir must be supplied exactly five times")
    directories: dict[str, Path] = {}
    for raw_value in values:
        fold, separator, raw_path = raw_value.partition("=")
        if separator != "=" or fold not in EXPECTED_FOLDS or not raw_path.strip():
            raise ValueError(
                "Each --fold-dir must use FOLD=PATH with FOLD exactly 0, 1, 2, 3 or 4"
            )
        if fold in directories:
            raise ValueError(f"Duplicate --fold-dir for fold {fold}")
        directory = Path(raw_path).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Fold {fold} evaluation directory does not exist: {directory}")
        directories[fold] = directory
    if set(directories) != set(EXPECTED_FOLDS):
        missing = sorted(set(EXPECTED_FOLDS) - set(directories))
        raise ValueError(f"Missing --fold-dir values for folds: {', '.join(missing)}")
    if len(set(directories.values())) != len(directories):
        raise ValueError("Every fold must use a distinct evaluation directory")
    return directories


def _load_csv_stable(
    path: Path,
    *,
    fold: str,
    role: str,
    dataset_name: str,
) -> tuple[tuple[str, ...], tuple[dict[str, str], ...], dict[str, object]]:
    before = local_file_identity(path)
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            fieldnames = tuple(reader.fieldnames)
            if any(not name for name in fieldnames) or len(fieldnames) != len(
                set(fieldnames)
            ):
                raise ValueError(f"CSV has empty or duplicate columns: {path}")
            missing = sorted(set(REQUIRED_ROW_COLUMNS) - set(fieldnames))
            if missing:
                raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
            rows: list[dict[str, str]] = []
            seen_cases: set[str] = set()
            for line_number, raw_row in enumerate(reader, start=2):
                if None in raw_row or any(value is None for value in raw_row.values()):
                    raise ValueError(f"Malformed CSV row at {path}:{line_number}")
                row = {name: str(raw_row[name]) for name in fieldnames}
                case_name = row["case_name"].strip()
                source_group = row["source_group"].strip()
                dataset = row["dataset"].strip()
                if not case_name or not source_group or not dataset:
                    raise ValueError(f"Empty audit identity at {path}:{line_number}")
                if case_name in seen_cases:
                    raise ValueError(f"Duplicate case_name {case_name!r} in fold {fold}")
                seen_cases.add(case_name)
                if row["role"].strip() != role or row["fold"].strip() != fold:
                    raise ValueError(
                        f"Row role/fold mismatch at {path}:{line_number}; "
                        f"expected {role}/{fold}"
                    )
                if dataset != dataset_name:
                    raise ValueError(
                        f"Row dataset differs from its contract at {path}:{line_number}"
                    )
                numeric_names = ("delta_iou", *DEFAULT_GATE_FEATURES)
                numeric_values: dict[str, float] = {}
                for name in numeric_names:
                    try:
                        value = float(row[name])
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid numeric value {name!r} at {path}:{line_number}"
                        ) from exc
                    if not math.isfinite(value):
                        raise ValueError(
                            f"Non-finite numeric value {name!r} at {path}:{line_number}"
                        )
                    numeric_values[name] = value
                if "candidate_iou_gain" in row:
                    try:
                        candidate_gain = float(row["candidate_iou_gain"])
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid candidate_iou_gain at {path}:{line_number}"
                        ) from exc
                    if (
                        not math.isfinite(candidate_gain)
                        or candidate_gain != numeric_values["delta_iou"]
                    ):
                        raise ValueError(
                            f"delta_iou differs from candidate_iou_gain at "
                            f"{path}:{line_number}"
                        )
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Invalid CSV file: {path}") from exc
    after = local_file_identity(path)
    if before != after:
        raise RuntimeError(f"File changed while being read: {path}")
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return fieldnames, tuple(rows), after


def _expected_role(fold: str) -> str:
    return "gate_calibration" if fold == "4" else "gate_fit"


def _validate_contract(
    contract: Mapping[str, Any], *, fold: str, directory: Path
) -> tuple[str, dict[str, object], dict[str, object]]:
    if contract.get("schema") != EVALUATION_SCHEMA:
        raise ValueError(f"Fold {fold} has an unsupported evaluation schema")
    if contract.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"Fold {fold} has an unsupported evaluation schema version")
    role = _expected_role(fold)
    dataset = _require_mapping(contract.get("dataset"), f"fold {fold} dataset")
    if dataset.get("role") != role or str(dataset.get("fold")) != fold:
        raise ValueError(
            f"Fold {fold} evaluation contract must have role/fold {role}/{fold}"
        )
    dataset_name = dataset.get("name")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(f"Fold {fold} contract has no dataset name")
    selected_samples = dataset.get("selected_samples")
    if (
        isinstance(selected_samples, bool)
        or not isinstance(selected_samples, int)
        or selected_samples <= 0
    ):
        raise ValueError(f"Fold {fold} contract has invalid selected_samples")
    _validate_sha256(
        dataset.get("selected_sample_names_sha256"),
        f"fold {fold} selected_sample_names_sha256",
    )
    _recorded_identity(dataset.get("list"), f"fold {fold} dataset list")
    _recorded_identity(
        dataset.get("group_assignments"), f"fold {fold} group assignments"
    )

    checkpoints = _require_mapping(
        contract.get("checkpoints"), f"fold {fold} checkpoints"
    )
    _recorded_identity(checkpoints.get("sam2"), f"fold {fold} SAM 2 checkpoint")
    _recorded_identity(
        checkpoints.get("baseline"), f"fold {fold} baseline checkpoint"
    )
    residual_checkpoint = _recorded_identity(
        checkpoints.get("residual"), f"fold {fold} residual checkpoint"
    )

    graph_cache = _require_mapping(
        contract.get("graph_cache"), f"fold {fold} graph cache"
    )
    _recorded_identity(
        graph_cache.get("manifest"), f"fold {fold} graph cache manifest"
    )
    _validate_sha256(
        graph_cache.get("extractor_sha256"), f"fold {fold} extractor_sha256"
    )
    if not isinstance(graph_cache.get("frangi"), Mapping):
        raise ValueError(f"Fold {fold} contract has no Frangi cache parameters")
    channels = graph_cache.get("channels")
    if (
        not isinstance(channels, list)
        or not channels
        or not all(isinstance(channel, str) and channel for channel in channels)
        or len(channels) != len(set(channels))
    ):
        raise ValueError(f"Fold {fold} contract has no Frangi cache channels")
    if graph_cache.get("verify_cache_hashes") is not True:
        raise ValueError(f"Fold {fold} did not verify graph-cache file hashes")
    if graph_cache.get("verify_data_hashes") is not True:
        raise ValueError(f"Fold {fold} did not verify graph-cache source hashes")

    residual = _require_mapping(contract.get("residual"), f"fold {fold} residual")
    held_out_fold = residual.get("checkpoint_held_out_fold")
    if isinstance(held_out_fold, bool) or not isinstance(held_out_fold, int):
        raise ValueError(f"Fold {fold} has an invalid checkpoint held-out fold")
    normalized_held_out_fold = str(held_out_fold)
    if normalized_held_out_fold != fold:
        raise ValueError(
            f"Fold {fold} uses a residual checkpoint held out on fold "
            f"{normalized_held_out_fold}"
        )
    try:
        oof_training = validate_strict_oof_training_contract(
            residual.get("checkpoint_oof_training"),
            held_out_fold=int(fold),
            evaluation_role=role,
        )
    except ValueError as exc:
        raise ValueError(
            f"Fold {fold} violates the strict OOF training contract: {exc}"
        ) from exc
    policy = _require_mapping(contract.get("gate_policy"), f"fold {fold} gate policy")
    required_policy = {
        "feature_rows_only": True,
        "threshold_selected_by_this_command": False,
        "threshold_may_later_be_calibrated_from_this_role": fold == "4",
        "historical_tests_forbidden_for_threshold_selection": True,
    }
    for name, expected in required_policy.items():
        if policy.get(name) is not expected:
            raise ValueError(f"Fold {fold} has an unsafe gate policy for {name}")
    for name in ("segmentation_threshold", "label_minimum_gain"):
        try:
            value = float(contract[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Fold {fold} has no valid {name}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Fold {fold} has a non-finite {name}")
        if name == "segmentation_threshold" and not 0.0 < value < 1.0:
            raise ValueError(
                f"Fold {fold} segmentation_threshold must lie strictly in (0, 1)"
            )
    return role, residual_checkpoint, oof_training


def load_fold_evaluation(directory: Path, fold: str) -> FoldEvaluation:
    contract_path = directory / "evaluation_contract.json"
    csv_path = directory / "per_image.csv"
    contract, contract_identity = _load_json_stable(
        contract_path, maximum_bytes=MAX_CONTRACT_BYTES
    )
    role, residual_checkpoint, oof_training = _validate_contract(
        contract, fold=fold, directory=directory
    )
    dataset = _require_mapping(contract["dataset"], f"fold {fold} dataset")
    fieldnames, rows, csv_identity = _load_csv_stable(
        csv_path,
        fold=fold,
        role=role,
        dataset_name=str(dataset["name"]),
    )
    if len(rows) != int(dataset["selected_samples"]):
        raise ValueError(
            f"Fold {fold} CSV row count differs from selected_samples in its contract"
        )
    case_names = [row["case_name"].strip() for row in rows]
    if _sample_names_sha256(case_names) != str(
        dataset["selected_sample_names_sha256"]
    ).lower():
        raise ValueError(
            f"Fold {fold} CSV case order/content differs from its evaluation contract"
        )
    if local_file_identity(contract_path) != contract_identity:
        raise RuntimeError(f"File changed while fold {fold} was being validated: {contract_path}")
    return FoldEvaluation(
        fold=fold,
        role=role,
        directory=directory,
        contract=contract,
        fieldnames=fieldnames,
        rows=rows,
        evaluation_contract=contract_identity,
        per_image_csv=csv_identity,
        residual_checkpoint=residual_checkpoint,
        oof_training=oof_training,
    )


def _common_contract_artifacts(
    evaluations: Sequence[FoldEvaluation],
) -> dict[str, object]:
    reference = evaluations[0]
    reference_dataset = _require_mapping(reference.contract["dataset"], "dataset")
    reference_checkpoints = _require_mapping(
        reference.contract["checkpoints"], "checkpoints"
    )
    reference_cache = _require_mapping(reference.contract["graph_cache"], "graph_cache")
    reference_residual = _require_mapping(reference.contract["residual"], "residual")

    dataset_fields = ("name", "root", "list", "split", "noise", "image_size", "group_assignments")
    cache_fields = (
        "root",
        "manifest",
        "extractor_sha256",
        "frangi",
        "channels",
        "verify_cache_hashes",
        "verify_data_hashes",
    )
    residual_fields = (
        "raster_channels",
        "high_resolution_channels",
        "hidden_channels",
        "training_raster_condition",
        "evaluation_raster_condition",
        "causal_raster_override",
    )
    reference_sam2 = _recorded_identity(reference_checkpoints["sam2"], "SAM 2")
    reference_baseline = _recorded_identity(
        reference_checkpoints["baseline"], "baseline"
    )
    reference_list = _recorded_identity(reference_dataset["list"], "dataset list")
    reference_groups = _recorded_identity(
        reference_dataset["group_assignments"], "group assignments"
    )
    reference_cache_manifest = _recorded_identity(
        reference_cache["manifest"], "graph cache manifest"
    )
    cache_root = reference_cache.get("root")
    if not isinstance(cache_root, str) or not cache_root:
        raise ValueError("Graph cache root must be a non-empty path")

    for evaluation in evaluations[1:]:
        dataset = _require_mapping(evaluation.contract["dataset"], "dataset")
        checkpoints = _require_mapping(
            evaluation.contract["checkpoints"], "checkpoints"
        )
        graph_cache = _require_mapping(evaluation.contract["graph_cache"], "graph cache")
        residual = _require_mapping(evaluation.contract["residual"], "residual")
        differing_dataset = [
            name for name in dataset_fields if dataset.get(name) != reference_dataset.get(name)
        ]
        if differing_dataset:
            raise ValueError(
                "OOF evaluation dataset contracts differ for: "
                + ", ".join(differing_dataset)
            )
        sam2 = _recorded_identity(checkpoints.get("sam2"), "SAM 2")
        baseline = _recorded_identity(checkpoints.get("baseline"), "baseline")
        if not _same_artifact(sam2, reference_sam2):
            raise ValueError("OOF evaluations use different SAM 2 checkpoints")
        if not _same_artifact(baseline, reference_baseline):
            raise ValueError("OOF evaluations use different baseline checkpoints")
        differing_cache = [
            name for name in cache_fields if graph_cache.get(name) != reference_cache.get(name)
        ]
        if differing_cache:
            raise ValueError(
                "OOF evaluation graph-cache contracts differ for: "
                + ", ".join(differing_cache)
            )
        differing_residual = [
            name for name in residual_fields if residual.get(name) != reference_residual.get(name)
        ]
        if differing_residual:
            raise ValueError(
                "OOF residual architectures/conditions differ for: "
                + ", ".join(differing_residual)
            )
        for name in ("segmentation_threshold", "label_minimum_gain"):
            if evaluation.contract.get(name) != reference.contract.get(name):
                raise ValueError(f"OOF evaluation contracts differ for {name}")

    residual_shas = {
        str(evaluation.residual_checkpoint["sha256"]) for evaluation in evaluations
    }
    if len(residual_shas) != len(EXPECTED_FOLDS):
        raise ValueError("Every OOF fold must use a distinct residual checkpoint")

    graph_contract = {
        "manifest": reference_cache_manifest,
        "extractor_sha256": _validate_sha256(
            reference_cache["extractor_sha256"], "extractor_sha256"
        ),
        "frangi": reference_cache["frangi"],
        "channels": reference_cache["channels"],
    }
    protocol_contract = {
        "dataset_list": reference_list,
        "group_assignments": reference_groups,
    }
    return {
        "sam2_checkpoint": reference_sam2,
        "baseline_checkpoint": reference_baseline,
        "graph_cache": {
            "root": cache_root,
            **graph_contract,
            "contract_sha256": _canonical_json_sha256(graph_contract),
        },
        "protocol": {
            **protocol_contract,
            "composite_sha256": _canonical_json_sha256(protocol_contract),
        },
    }


def validate_evaluation_set(
    evaluations: Sequence[FoldEvaluation],
) -> dict[str, object]:
    if tuple(evaluation.fold for evaluation in evaluations) != EXPECTED_FOLDS:
        raise ValueError("Evaluations must be ordered as folds 0, 1, 2, 3, 4")
    reference_header = evaluations[0].fieldnames
    seen_cases: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    for evaluation in evaluations:
        if evaluation.fieldnames != reference_header:
            raise ValueError("All five per_image.csv files must have the same header")
        for row in evaluation.rows:
            case = row["case_name"].strip()
            group = row["source_group"].strip()
            previous_case_fold = seen_cases.setdefault(case, evaluation.fold)
            if previous_case_fold != evaluation.fold:
                raise ValueError(
                    f"case_name collision across folds {previous_case_fold} and "
                    f"{evaluation.fold}: {case!r}"
                )
            previous_group_fold = seen_groups.setdefault(group, evaluation.fold)
            if previous_group_fold != evaluation.fold:
                raise ValueError(
                    f"source_group collision across folds {previous_group_fold} and "
                    f"{evaluation.fold}: {group!r}"
                )
    return _common_contract_artifacts(evaluations)


def _write_csv_atomic(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> Path:
    if not rows:
        raise ValueError(f"Cannot publish an empty OOF CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(fieldnames),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def _output_identity(
    path: Path, *, folds: Sequence[str], rows: int
) -> dict[str, object]:
    identity = local_file_identity(path)
    return {**identity, "folds": list(folds), "rows": rows}


def assemble_oof_gate_data(
    fold_directories: Mapping[str, Path], output: Path
) -> dict[str, Any]:
    if set(fold_directories) != set(EXPECTED_FOLDS):
        raise ValueError("Exactly one evaluation directory is required for folds 0-4")
    evaluations = [
        load_fold_evaluation(Path(fold_directories[fold]), fold)
        for fold in EXPECTED_FOLDS
    ]
    artifacts = validate_evaluation_set(evaluations)

    destination = output.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"--output must be a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    fieldnames = evaluations[0].fieldnames
    fit_rows = [row for evaluation in evaluations[:4] for row in evaluation.rows]
    calibration_rows = list(evaluations[4].rows)
    fit_path = _write_csv_atomic(destination / "gate_fit.csv", fieldnames, fit_rows)
    calibration_path = _write_csv_atomic(
        destination / "gate_calibration.csv", fieldnames, calibration_rows
    )

    fold_payloads: dict[str, object] = {}
    for evaluation in evaluations:
        fold_payloads[evaluation.fold] = {
            "role": evaluation.role,
            "evaluation_directory": str(evaluation.directory),
            "evaluation_contract": evaluation.evaluation_contract,
            "per_image_csv": evaluation.per_image_csv,
            "residual_checkpoint": evaluation.residual_checkpoint,
            "oof_training": evaluation.oof_training,
            "rows": len(evaluation.rows),
            "source_groups": len(
                {row["source_group"].strip() for row in evaluation.rows}
            ),
        }
    manifest: dict[str, Any] = {
        "schema": OOF_MANIFEST_SCHEMA,
        "schema_version": OOF_MANIFEST_SCHEMA_VERSION,
        "feature_names": list(DEFAULT_GATE_FEATURES),
        "label_minimum_gain": float(evaluations[0].contract["label_minimum_gain"]),
        "segmentation_threshold": float(
            evaluations[0].contract["segmentation_threshold"]
        ),
        "artifacts": artifacts,
        "folds": fold_payloads,
        "outputs": {
            "gate_fit_csv": _output_identity(
                fit_path, folds=FIT_FOLDS, rows=len(fit_rows)
            ),
            "gate_calibration_csv": _output_identity(
                calibration_path,
                folds=CALIBRATION_FOLDS,
                rows=len(calibration_rows),
            ),
        },
    }
    _write_json_atomic(destination / "oof_manifest.json", manifest)
    return manifest


def _validate_local_identity(
    value: object, name: str, *, verify_file: bool
) -> dict[str, object]:
    payload = _require_mapping(value, name)
    expected_keys = {"path", "name", "bytes", "sha256"}
    if set(payload) != expected_keys:
        raise ValueError(f"{name} has missing or unknown identity fields")
    recorded = _recorded_identity(payload, name)
    path = payload["path"]
    if not isinstance(path, str) or not path:
        raise ValueError(f"{name}.path must be non-empty")
    normalized = {"path": path, **recorded}
    if verify_file:
        observed = local_file_identity(path)
        if observed != normalized:
            raise ValueError(f"{name} no longer matches its recorded SHA/size/path")
    return normalized


def _validate_output_identity(value: object, name: str) -> dict[str, object]:
    payload = _require_mapping(value, name)
    expected_keys = {"path", "name", "bytes", "sha256", "folds", "rows"}
    if set(payload) != expected_keys:
        raise ValueError(f"{name} has missing or unknown output identity fields")
    identity = _validate_local_identity(
        {key: payload[key] for key in ("path", "name", "bytes", "sha256")},
        name,
        verify_file=False,
    )
    folds = payload["folds"]
    rows = payload["rows"]
    if not isinstance(folds, list) or not all(isinstance(fold, str) for fold in folds):
        raise ValueError(f"{name}.folds must be a string list")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError(f"{name}.rows must be a positive integer")
    return {**identity, "folds": folds, "rows": rows}


def load_and_validate_oof_manifest(
    path: str | os.PathLike[str], *, verify_sources: bool = True
) -> tuple[dict[str, Any], str]:
    """Load a manifest and revalidate its five source contracts by default."""
    source = Path(path).expanduser().resolve()
    payload, manifest_identity = _load_json_stable(
        source, maximum_bytes=MAX_OOF_MANIFEST_BYTES
    )
    required_top_level = {
        "schema",
        "schema_version",
        "feature_names",
        "label_minimum_gain",
        "segmentation_threshold",
        "artifacts",
        "folds",
        "outputs",
    }
    if set(payload) != required_top_level:
        raise ValueError("OOF manifest has missing or unknown top-level fields")
    if payload["schema"] != OOF_MANIFEST_SCHEMA:
        raise ValueError("Unknown OOF manifest schema")
    if payload["schema_version"] != OOF_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported OOF manifest schema version")
    if payload["feature_names"] != list(DEFAULT_GATE_FEATURES):
        raise ValueError("OOF manifest gate features differ from the preregistered order")
    for name in ("label_minimum_gain", "segmentation_threshold"):
        try:
            numeric = float(payload[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"OOF manifest has invalid {name}") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"OOF manifest has non-finite {name}")
        if name == "segmentation_threshold" and not 0.0 < numeric < 1.0:
            raise ValueError("OOF segmentation_threshold must lie strictly in (0, 1)")

    folds = _require_mapping(payload["folds"], "OOF folds")
    if set(folds) != set(EXPECTED_FOLDS):
        raise ValueError("OOF manifest must contain exactly folds 0, 1, 2, 3 and 4")
    source_evaluations: list[FoldEvaluation] = []
    residual_shas: set[str] = set()
    for fold in EXPECTED_FOLDS:
        fold_payload = _require_mapping(folds[fold], f"OOF fold {fold}")
        expected_fold_fields = {
            "role",
            "evaluation_directory",
            "evaluation_contract",
            "per_image_csv",
            "residual_checkpoint",
            "oof_training",
            "rows",
            "source_groups",
        }
        if set(fold_payload) != expected_fold_fields:
            raise ValueError(f"OOF fold {fold} has missing or unknown fields")
        if fold_payload["role"] != _expected_role(fold):
            raise ValueError(f"OOF fold {fold} has the wrong role")
        directory_value = fold_payload["evaluation_directory"]
        if not isinstance(directory_value, str) or not directory_value:
            raise ValueError(f"OOF fold {fold} has no evaluation directory")
        contract_identity = _validate_local_identity(
            fold_payload["evaluation_contract"],
            f"OOF fold {fold} evaluation contract",
            verify_file=verify_sources,
        )
        csv_identity = _validate_local_identity(
            fold_payload["per_image_csv"],
            f"OOF fold {fold} per_image.csv",
            verify_file=verify_sources,
        )
        residual_identity = _recorded_identity(
            fold_payload["residual_checkpoint"],
            f"OOF fold {fold} residual checkpoint",
        )
        residual_shas.add(str(residual_identity["sha256"]))
        try:
            recorded_oof_training = validate_strict_oof_training_contract(
                fold_payload["oof_training"],
                held_out_fold=int(fold),
                evaluation_role=_expected_role(fold),
            )
        except ValueError as exc:
            raise ValueError(
                f"OOF fold {fold} violates the strict OOF training contract: {exc}"
            ) from exc
        rows = fold_payload["rows"]
        source_groups = fold_payload["source_groups"]
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or isinstance(source_groups, bool)
            or not isinstance(source_groups, int)
            or source_groups <= 0
            or source_groups > rows
        ):
            raise ValueError(f"OOF fold {fold} has invalid row/group counts")
        if verify_sources:
            evaluation = load_fold_evaluation(Path(directory_value), fold)
            if evaluation.evaluation_contract != contract_identity:
                raise ValueError(f"OOF fold {fold} evaluation contract identity drifted")
            if evaluation.per_image_csv != csv_identity:
                raise ValueError(f"OOF fold {fold} per_image.csv identity drifted")
            if evaluation.residual_checkpoint != residual_identity:
                raise ValueError(f"OOF fold {fold} residual identity differs from contract")
            if evaluation.oof_training != recorded_oof_training:
                raise ValueError(f"OOF fold {fold} OOF training contract drifted")
            if len(evaluation.rows) != rows or len(
                {row["source_group"].strip() for row in evaluation.rows}
            ) != source_groups:
                raise ValueError(f"OOF fold {fold} row/group counts differ from sources")
            source_evaluations.append(evaluation)
    if len(residual_shas) != len(EXPECTED_FOLDS):
        raise ValueError("OOF manifest must bind five distinct residual checkpoints")

    outputs = _require_mapping(payload["outputs"], "OOF outputs")
    if set(outputs) != {"gate_fit_csv", "gate_calibration_csv"}:
        raise ValueError("OOF manifest must bind exactly the fit and calibration CSVs")
    fit_output = _validate_output_identity(outputs["gate_fit_csv"], "gate_fit_csv")
    calibration_output = _validate_output_identity(
        outputs["gate_calibration_csv"], "gate_calibration_csv"
    )
    if fit_output["folds"] != list(FIT_FOLDS):
        raise ValueError("gate_fit_csv must contain exactly folds 0-3")
    if calibration_output["folds"] != list(CALIBRATION_FOLDS):
        raise ValueError("gate_calibration_csv must contain exactly fold 4")
    if fit_output["rows"] != sum(int(folds[fold]["rows"]) for fold in FIT_FOLDS):
        raise ValueError("gate_fit_csv row count differs from its four fold records")
    if calibration_output["rows"] != int(folds["4"]["rows"]):
        raise ValueError("gate_calibration_csv row count differs from fold 4")

    artifacts = _require_mapping(payload["artifacts"], "OOF artifacts")
    if set(artifacts) != {
        "sam2_checkpoint",
        "baseline_checkpoint",
        "graph_cache",
        "protocol",
    }:
        raise ValueError("OOF manifest has missing or unknown artifact bindings")
    _recorded_identity(artifacts["sam2_checkpoint"], "OOF SAM 2 checkpoint")
    _recorded_identity(artifacts["baseline_checkpoint"], "OOF baseline checkpoint")
    graph_cache = _require_mapping(artifacts["graph_cache"], "OOF graph cache")
    if set(graph_cache) != {
        "root",
        "manifest",
        "extractor_sha256",
        "frangi",
        "channels",
        "contract_sha256",
    }:
        raise ValueError("OOF graph-cache binding has missing or unknown fields")
    graph_manifest = _recorded_identity(
        graph_cache["manifest"], "OOF graph cache manifest"
    )
    extractor_sha = _validate_sha256(
        graph_cache["extractor_sha256"], "OOF extractor_sha256"
    )
    graph_contract = {
        "manifest": graph_manifest,
        "extractor_sha256": extractor_sha,
        "frangi": graph_cache["frangi"],
        "channels": graph_cache["channels"],
    }
    if graph_cache["contract_sha256"] != _canonical_json_sha256(graph_contract):
        raise ValueError("OOF graph-cache composite SHA-256 is inconsistent")
    protocol = _require_mapping(artifacts["protocol"], "OOF protocol")
    if set(protocol) != {"dataset_list", "group_assignments", "composite_sha256"}:
        raise ValueError("OOF protocol binding has missing or unknown fields")
    protocol_contract = {
        "dataset_list": _recorded_identity(
            protocol["dataset_list"], "OOF protocol dataset list"
        ),
        "group_assignments": _recorded_identity(
            protocol["group_assignments"], "OOF protocol group assignments"
        ),
    }
    if protocol["composite_sha256"] != _canonical_json_sha256(protocol_contract):
        raise ValueError("OOF protocol composite SHA-256 is inconsistent")

    if verify_sources:
        expected_artifacts = validate_evaluation_set(source_evaluations)
        if artifacts != expected_artifacts:
            raise ValueError("OOF artifact bindings differ from the five source contracts")
        reference = source_evaluations[0].contract
        if float(payload["label_minimum_gain"]) != float(
            reference["label_minimum_gain"]
        ):
            raise ValueError("OOF label_minimum_gain differs from source contracts")
        if float(payload["segmentation_threshold"]) != float(
            reference["segmentation_threshold"]
        ):
            raise ValueError("OOF segmentation_threshold differs from source contracts")
    if local_file_identity(source) != manifest_identity:
        raise RuntimeError(f"OOF manifest changed while being validated: {source}")
    return payload, str(manifest_identity["sha256"])


def validate_manifest_output_csv(
    manifest: Mapping[str, Any], output_name: str, path: str | os.PathLike[str]
) -> dict[str, object]:
    if output_name not in ("gate_fit_csv", "gate_calibration_csv"):
        raise ValueError(f"Unknown OOF output name: {output_name}")
    outputs = _require_mapping(manifest.get("outputs"), "OOF outputs")
    expected = _validate_output_identity(outputs.get(output_name), output_name)
    observed = local_file_identity(path)
    if not _same_artifact(observed, expected):
        raise ValueError(
            f"Supplied {output_name} SHA/size does not match oof_manifest.json"
        )
    return observed


def main() -> None:
    args = parse_args()
    directories = parse_fold_directories(args.fold_dir)
    manifest = assemble_oof_gate_data(directories, args.output)
    manifest_path = args.output.expanduser().resolve() / "oof_manifest.json"
    summary = {
        "output": str(args.output.expanduser().resolve()),
        "manifest": local_file_identity(manifest_path),
        "fit_rows": manifest["outputs"]["gate_fit_csv"]["rows"],
        "calibration_rows": manifest["outputs"]["gate_calibration_csv"]["rows"],
        "residual_checkpoints": {
            fold: manifest["folds"][fold]["residual_checkpoint"]["sha256"]
            for fold in EXPECTED_FOLDS
        },
    }
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
