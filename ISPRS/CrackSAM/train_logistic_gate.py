#!/usr/bin/env python3
"""Train and calibrate the group-balanced Frangi residual confidence gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from assemble_logistic_gate_data import (
    load_and_validate_oof_manifest,
    validate_manifest_output_csv,
)
from cracksam2.gating import (
    DEFAULT_GATE_FEATURES,
    DEFAULT_MAXIMUM_SEVERE_LOSS_RATE,
    DEFAULT_MINIMUM_MEAN_DELTA_IOU,
    DEFAULT_MINIMUM_PRECISION,
    DEFAULT_MINIMUM_SELECTED,
    DEFAULT_MINIMUM_SELECTED_GROUPS,
    DEFAULT_SEVERE_LOSS_THRESHOLD,
    LABEL_DEFINITION,
    GateProvenance,
    LogisticConfidenceGate,
)


REQUIRED_METADATA_COLUMNS: tuple[str, ...] = (
    "case_name",
    "source_group",
    "fold",
    "dataset",
    "role",
    "delta_iou",
)


@dataclass(frozen=True)
class GateCsvData:
    features: np.ndarray
    delta_iou: np.ndarray
    case_names: tuple[str, ...]
    source_groups: tuple[str, ...]
    folds: tuple[str, ...]
    datasets: tuple[str, ...]
    roles: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit on one CSV and choose a safe opening threshold on a distinct, "
            "group-disjoint calibration CSV."
        )
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--oof-manifest",
        type=Path,
        required=True,
        help=(
            "Composite manifest emitted by assemble_logistic_gate_data.py; "
            "it binds the five residual checkpoints and both CSVs."
        ),
    )
    parser.add_argument(
        "--git-commit",
        required=True,
        help="Full 40- or 64-character commit hash of the implementation.",
    )
    parser.add_argument(
        "--label-minimum-gain",
        type=float,
        required=True,
        help="A crop is positive only when delta_iou is strictly above this gain.",
    )
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument(
        "--minimum-precision", type=float, default=DEFAULT_MINIMUM_PRECISION
    )
    parser.add_argument(
        "--minimum-selected", type=int, default=DEFAULT_MINIMUM_SELECTED
    )
    parser.add_argument(
        "--minimum-selected-groups",
        type=int,
        default=DEFAULT_MINIMUM_SELECTED_GROUPS,
    )
    parser.add_argument(
        "--minimum-mean-delta-iou",
        type=float,
        default=DEFAULT_MINIMUM_MEAN_DELTA_IOU,
    )
    parser.add_argument(
        "--severe-loss-threshold",
        type=float,
        default=DEFAULT_SEVERE_LOSS_THRESHOLD,
    )
    parser.add_argument(
        "--maximum-severe-loss-rate",
        type=float,
        default=DEFAULT_MAXIMUM_SEVERE_LOSS_RATE,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Hash a regular file without loading a checkpoint into memory."""
    if not path.is_file():
        raise ValueError(f"Required provenance file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_gate_csv(
    path: Path,
    *,
    feature_names: tuple[str, ...],
) -> GateCsvData:
    """Load the strict gate table; labels are derived later from signed delta."""
    if feature_names != DEFAULT_GATE_FEATURES:
        raise ValueError(
            "Gate features are preregistered and must be exactly "
            "DEFAULT_GATE_FEATURES in their declared order"
        )
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if any(not name for name in reader.fieldnames) or len(reader.fieldnames) != len(
            set(reader.fieldnames)
        ):
            raise ValueError(f"CSV has empty or duplicate columns: {path}")
        missing = [
            name
            for name in (*REQUIRED_METADATA_COLUMNS, *feature_names)
            if name not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        feature_positions = [reader.fieldnames.index(name) for name in feature_names]
        if feature_positions != sorted(feature_positions):
            raise ValueError(
                f"{path} does not declare gate features in the preregistered order"
            )
        feature_rows: list[list[float]] = []
        delta_values: list[float] = []
        case_names: list[str] = []
        source_groups: list[str] = []
        folds: list[str] = []
        datasets: list[str] = []
        roles: list[str] = []
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row in {path} at line {line_number}")
            try:
                values = [float(row[name]) for name in feature_names]
                delta = float(row["delta_iou"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in {path} at line {line_number}"
                ) from error
            metadata = {
                name: str(row[name]).strip()
                for name in (
                    "case_name",
                    "source_group",
                    "fold",
                    "dataset",
                    "role",
                )
            }
            empty = [name for name, value in metadata.items() if not value]
            if empty:
                raise ValueError(
                    f"Empty metadata in {path} at line {line_number}: "
                    f"{', '.join(empty)}"
                )
            if not all(math.isfinite(value) for value in (*values, delta)):
                raise ValueError(
                    f"Non-finite numeric value in {path} at line {line_number}"
                )
            feature_rows.append(values)
            delta_values.append(delta)
            case_names.append(metadata["case_name"])
            source_groups.append(metadata["source_group"])
            folds.append(metadata["fold"])
            datasets.append(metadata["dataset"])
            roles.append(metadata["role"])
    if not feature_rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    case_identities = list(zip(datasets, case_names, strict=True))
    if len(case_identities) != len(set(case_identities)):
        raise ValueError(f"CSV contains duplicate dataset/case_name rows: {path}")
    return GateCsvData(
        features=np.asarray(feature_rows, dtype=np.float64),
        delta_iou=np.asarray(delta_values, dtype=np.float64),
        case_names=tuple(case_names),
        source_groups=tuple(source_groups),
        folds=tuple(folds),
        datasets=tuple(datasets),
        roles=tuple(roles),
    )


def assert_exact_oof_partitions(
    train: GateCsvData, calibration: GateCsvData
) -> None:
    """Require the preregistered fit/calibration roles and all expected folds."""
    train_roles = set(train.roles)
    train_folds = set(train.folds)
    calibration_roles = set(calibration.roles)
    calibration_folds = set(calibration.folds)
    if train_roles != {"gate_fit"}:
        raise ValueError(
            "Training CSV role must be exactly gate_fit; "
            f"observed {sorted(train_roles)}"
        )
    if train_folds != {"0", "1", "2", "3"}:
        raise ValueError(
            "Training CSV must contain every OOF fold 0, 1, 2 and 3 exactly; "
            f"observed {sorted(train_folds)}"
        )
    if calibration_roles != {"gate_calibration"}:
        raise ValueError(
            "Calibration CSV role must be exactly gate_calibration; "
            f"observed {sorted(calibration_roles)}"
        )
    if calibration_folds != {"4"}:
        raise ValueError(
            "Calibration CSV must contain only OOF fold 4; "
            f"observed {sorted(calibration_folds)}"
        )


def assert_train_calibration_disjoint(
    train: GateCsvData, calibration: GateCsvData
) -> None:
    """Reject leakage by crop identity or original source identity."""
    case_overlap = set(train.case_names) & set(calibration.case_names)
    group_overlap = set(train.source_groups) & set(calibration.source_groups)
    problems: list[str] = []
    if case_overlap:
        problems.append(
            "case_name overlap: " + ", ".join(sorted(case_overlap)[:5])
        )
    if group_overlap:
        problems.append(
            "source_group overlap: " + ", ".join(sorted(group_overlap)[:5])
        )
    if problems:
        raise ValueError(
            "Training/calibration leakage detected (" + "; ".join(problems) + ")"
        )


def _hash_stable_csv(path: Path, data_loader: Any) -> tuple[str, GateCsvData]:
    """Detect a CSV modification occurring while it is being parsed."""
    before = sha256_file(path)
    data = data_loader()
    after = sha256_file(path)
    if before != after:
        raise RuntimeError(f"CSV changed while being read: {path}")
    return after, data


def main() -> None:
    args = parse_args()
    train_path = args.train_csv.resolve()
    calibration_path = args.calibration_csv.resolve()
    if train_path == calibration_path:
        raise ValueError("Training and calibration CSVs must be distinct files")
    oof_manifest, oof_manifest_sha256 = load_and_validate_oof_manifest(
        args.oof_manifest.resolve(), verify_sources=True
    )
    feature_names = DEFAULT_GATE_FEATURES
    if float(args.label_minimum_gain) != float(
        oof_manifest["label_minimum_gain"]
    ):
        raise ValueError(
            "--label-minimum-gain differs from the five immutable OOF contracts"
        )

    expected_train_identity = validate_manifest_output_csv(
        oof_manifest, "gate_fit_csv", train_path
    )
    expected_calibration_identity = validate_manifest_output_csv(
        oof_manifest, "gate_calibration_csv", calibration_path
    )

    train_sha256, train = _hash_stable_csv(
        train_path,
        lambda: load_gate_csv(train_path, feature_names=feature_names),
    )
    calibration_sha256, calibration = _hash_stable_csv(
        calibration_path,
        lambda: load_gate_csv(calibration_path, feature_names=feature_names),
    )
    if train_sha256 != expected_train_identity["sha256"]:
        raise RuntimeError("Training CSV changed after OOF manifest validation")
    if calibration_sha256 != expected_calibration_identity["sha256"]:
        raise RuntimeError("Calibration CSV changed after OOF manifest validation")
    if train.features.shape[0] != int(
        oof_manifest["outputs"]["gate_fit_csv"]["rows"]
    ):
        raise ValueError("Training CSV row count differs from the OOF manifest")
    if calibration.features.shape[0] != int(
        oof_manifest["outputs"]["gate_calibration_csv"]["rows"]
    ):
        raise ValueError("Calibration CSV row count differs from the OOF manifest")
    assert_exact_oof_partitions(train, calibration)
    assert_train_calibration_disjoint(train, calibration)

    train_labels = (
        train.delta_iou > float(args.label_minimum_gain)
    ).astype(np.int64)
    gate = LogisticConfidenceGate.fit(
        train.features,
        train_labels,
        source_groups=train.source_groups,
        feature_names=feature_names,
        l2=args.l2,
    ).calibrate(
        calibration.features,
        calibration.delta_iou,
        source_groups=calibration.source_groups,
        label_minimum_gain=args.label_minimum_gain,
        minimum_precision=args.minimum_precision,
        minimum_selected=args.minimum_selected,
        minimum_selected_groups=args.minimum_selected_groups,
        minimum_mean_delta_iou=args.minimum_mean_delta_iou,
        severe_loss_threshold=args.severe_loss_threshold,
        maximum_severe_loss_rate=args.maximum_severe_loss_rate,
    )
    artifacts = oof_manifest["artifacts"]
    graph_cache = artifacts["graph_cache"]
    protocol = artifacts["protocol"]
    provenance = GateProvenance(
        baseline_checkpoint_sha256=artifacts["baseline_checkpoint"]["sha256"],
        oof_manifest_sha256=oof_manifest_sha256,
        frangi_extractor_sha256=graph_cache["extractor_sha256"],
        frangi_cache_manifest_sha256=graph_cache["manifest"]["sha256"],
        protocol_sha256=protocol["composite_sha256"],
        train_csv_sha256=train_sha256,
        calibration_csv_sha256=calibration_sha256,
        label_definition=LABEL_DEFINITION,
        label_minimum_gain=args.label_minimum_gain,
        git_commit=args.git_commit,
    )
    gate = gate.with_provenance(provenance)
    gate.save_json(args.output)
    summary = {
        "output": str(args.output.resolve()),
        "train_rows": int(train.features.shape[0]),
        "train_source_groups": len(set(train.source_groups)),
        "calibration_rows": int(calibration.features.shape[0]),
        "calibration_source_groups": len(set(calibration.source_groups)),
        "train_datasets": sorted(set(train.datasets)),
        "calibration_datasets": sorted(set(calibration.datasets)),
        "train_folds": sorted(set(train.folds)),
        "calibration_folds": sorted(set(calibration.folds)),
        "features": list(feature_names),
        "label_definition": LABEL_DEFINITION,
        "label_minimum_gain": float(args.label_minimum_gain),
        "threshold": gate.threshold,
        "calibration": dict(gate.calibration),
        "provenance": provenance.to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
