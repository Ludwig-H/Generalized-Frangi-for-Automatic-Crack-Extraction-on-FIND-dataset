#!/usr/bin/env python3
"""Clustered-bootstrap analysis of the five-fold FrangiGraph pilot.

The five residual evaluations are out-of-fold, but the logistic gate is fitted
on folds 0--3 and its threshold is selected on fold 4.  Consequently this
command keeps the residual and gate scopes separate and labels every gate
estimate according to how the same rows were used upstream.

All estimands are group-first: crop-level quantities are reduced within
``dataset::source_group`` before source groups receive equal weight.  Bootstrap
replicates resample those source groups, never individual crops.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from assemble_logistic_gate_data import (
    EXPECTED_FOLDS,
    load_and_validate_oof_manifest,
    parse_fold_directories,
)
from cracksam2.gating import DEFAULT_GATE_FEATURES, LogisticConfidenceGate
from evaluate_logistic_gate import (
    GATED_FIELDS,
    apply_gate_to_rows,
    load_evaluation_input,
    oof_compatibility,
    sha256_file,
)


ANALYSIS_SCHEMA = "cracksam2.frangigraph-clustered-bootstrap"
ANALYSIS_SCHEMA_VERSION = 1
DEFAULT_BOOTSTRAP_REPETITIONS = 20_000
DEFAULT_BOOTSTRAP_SEED = 3407
BOOTSTRAP_CHUNK_SIZE = 256
CI_QUANTILES = (0.025, 0.975)

ASSIGNMENT_FIELDS: tuple[str, ...] = (
    "sample_name",
    "source_family",
    "physical_group",
    "oof_fold",
)
RESIDUAL_ESTIMANDS: tuple[str, ...] = (
    "mean_delta_iou",
    "oracle_gain_iou",
    "practical_gain_rate",
    "harmful_rate",
    "severe_loss_rate",
)
GATE_ESTIMANDS: tuple[str, ...] = (
    "coverage",
    "precision",
    "selected_mean_delta_iou",
    "selected_severe_loss_rate",
    "system_mean_gain_iou",
)


@dataclass(frozen=True)
class GroupAssignment:
    sample_name: str
    source_family: str
    physical_group: str
    fold: str


@dataclass(frozen=True)
class GroupSummary:
    """One equally weighted physical source and its crop-reduced metrics."""

    group_id: str
    dataset: str
    source_group: str
    source_family: str
    fold: str
    rows: int
    selected_rows: int
    residual: Mapping[str, float]
    gate: Mapping[str, float]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--oof-manifest", type=Path, required=True)
    parser.add_argument(
        "--fold-dir",
        action="append",
        required=True,
        metavar="FOLD=PATH",
        help="Residual evaluation directory; repeat exactly for folds 0--4.",
    )
    parser.add_argument(
        "--group-assignments",
        type=Path,
        required=True,
        help="Canonical train_group_folds.csv used by the five evaluations.",
    )
    parser.add_argument(
        "--gated-csv",
        type=Path,
        help=(
            "Optional per_image_gated.csv from evaluate_logistic_gate.py. "
            "When supplied, every stored probability and decision is checked "
            "against the frozen gate before analysis."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or existing directory receiving summary.json and estimands.csv.",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def _file_identity(path: Path) -> dict[str, object]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "name": source.name,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _same_recorded_identity(
    observed: Mapping[str, object], expected: Mapping[str, object]
) -> bool:
    return all(
        observed.get(field) == expected.get(field)
        for field in ("name", "bytes", "sha256")
    )


def _read_assignments_stable(
    path: Path,
) -> tuple[dict[str, GroupAssignment], dict[str, object]]:
    source = path.expanduser().resolve()
    before = _file_identity(source)
    assignments: dict[str, GroupAssignment] = {}
    group_contracts: dict[str, tuple[str, str]] = {}
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != ASSIGNMENT_FIELDS:
                raise ValueError(
                    "Group-assignment CSV header must be exactly "
                    f"{ASSIGNMENT_FIELDS!r}"
                )
            for line, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed assignment row at {source}:{line}")
                values = {name: str(row[name]).strip() for name in ASSIGNMENT_FIELDS}
                if any(not value for value in values.values()):
                    raise ValueError(f"Empty assignment value at {source}:{line}")
                fold = values["oof_fold"]
                if fold not in EXPECTED_FOLDS:
                    raise ValueError(f"Invalid OOF fold at {source}:{line}: {fold!r}")
                name = values["sample_name"]
                if name in assignments:
                    raise ValueError(
                        f"Duplicate sample_name at {source}:{line}: {name!r}"
                    )
                assignment = GroupAssignment(
                    sample_name=name,
                    source_family=values["source_family"],
                    physical_group=values["physical_group"],
                    fold=fold,
                )
                group_contract = (assignment.source_family, assignment.fold)
                previous = group_contracts.setdefault(
                    assignment.physical_group, group_contract
                )
                if previous != group_contract:
                    raise ValueError(
                        "One physical_group spans source families or OOF folds: "
                        f"{assignment.physical_group!r}"
                    )
                assignments[name] = assignment
    except OSError as exc:
        raise ValueError(f"Cannot read group assignments: {source}") from exc
    if not assignments:
        raise ValueError(f"Group-assignment CSV is empty: {source}")
    after = _file_identity(source)
    if before != after:
        raise RuntimeError(f"Group-assignment CSV changed while being read: {source}")
    return assignments, after


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not Boolean")
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_gate_and_manifest(
    gate: LogisticConfidenceGate,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> tuple[float, float]:
    if gate.provenance is None:
        raise ValueError("Gate JSON has no strict provenance")
    if tuple(gate.feature_names) != DEFAULT_GATE_FEATURES:
        raise ValueError("Gate feature order differs from DEFAULT_GATE_FEATURES")
    if gate.provenance.oof_manifest_sha256 != manifest_sha256:
        raise ValueError("Gate provenance does not bind the supplied OOF manifest")
    compatibility = oof_compatibility(manifest)
    provenance_checks = {
        "baseline checkpoint": (
            gate.provenance.baseline_checkpoint_sha256,
            compatibility.baseline_checkpoint_sha256,
        ),
        "Frangi extractor": (
            gate.provenance.frangi_extractor_sha256,
            compatibility.frangi_extractor_sha256,
        ),
        "Frangi cache": (
            gate.provenance.frangi_cache_manifest_sha256,
            compatibility.frangi_cache_manifest_sha256,
        ),
        "protocol": (
            gate.provenance.protocol_sha256,
            compatibility.protocol_sha256,
        ),
    }
    for label, (observed, expected) in provenance_checks.items():
        if observed != expected:
            raise ValueError(f"Gate provenance differs from OOF {label}")
    label_threshold = float(compatibility.label_minimum_gain)
    if gate.provenance.label_minimum_gain != label_threshold:
        raise ValueError("Gate label threshold differs from OOF provenance")
    if gate.provenance.train_csv_sha256 != manifest["outputs"]["gate_fit_csv"][
        "sha256"
    ]:
        raise ValueError("Gate training CSV differs from the OOF manifest")
    if gate.provenance.calibration_csv_sha256 != manifest["outputs"][
        "gate_calibration_csv"
    ]["sha256"]:
        raise ValueError("Gate calibration CSV differs from the OOF manifest")
    severe_threshold = _finite_float(
        gate.calibration.get("severe_loss_threshold"),
        "gate calibration severe_loss_threshold",
    )
    calibration_label_threshold = _finite_float(
        gate.calibration.get("label_minimum_gain"),
        "gate calibration label_minimum_gain",
    )
    if calibration_label_threshold != label_threshold:
        raise ValueError("Gate calibration and OOF label thresholds differ")
    return label_threshold, severe_threshold


def _load_five_evaluations(
    directories: Mapping[str, Path], manifest: Mapping[str, Any]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    compatibility = oof_compatibility(manifest)
    all_rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for fold in EXPECTED_FOLDS:
        evaluation = load_evaluation_input(directories[fold], compatibility)
        dataset_contract = evaluation.contract["dataset"]
        expected_role = "gate_calibration" if fold == "4" else "gate_fit"
        if dataset_contract.get("fold") != fold or dataset_contract.get(
            "role"
        ) != expected_role:
            raise ValueError(f"Supplied evaluation directory is not OOF fold {fold}")
        fold_manifest = manifest["folds"][fold]
        expected_contract = fold_manifest["evaluation_contract"]
        expected_csv = fold_manifest["per_image_csv"]
        if evaluation.contract_sha256 != expected_contract.get("sha256"):
            raise ValueError(f"Fold {fold} contract SHA differs from OOF manifest")
        if evaluation.csv_sha256 != expected_csv.get("sha256"):
            raise ValueError(f"Fold {fold} CSV SHA differs from OOF manifest")
        if len(evaluation.rows) != int(fold_manifest["rows"]):
            raise ValueError(f"Fold {fold} row count differs from OOF manifest")
        groups = {
            f"{row['dataset']}::{row['source_group']}" for row in evaluation.rows
        }
        if len(groups) != int(fold_manifest["source_groups"]):
            raise ValueError(f"Fold {fold} group count differs from OOF manifest")
        for row in evaluation.rows:
            identity = (str(row["dataset"]), str(row["case_name"]))
            if identity in seen:
                raise ValueError(f"Duplicate evaluation row identity: {identity!r}")
            seen.add(identity)
            all_rows.append(dict(row))
        provenance.append(
            {
                "fold": fold,
                "role": expected_role,
                "path": str(evaluation.root),
                "rows": len(evaluation.rows),
                "groups": len(groups),
                "evaluation_contract_sha256": evaluation.contract_sha256,
                "per_image_csv_sha256": evaluation.csv_sha256,
            }
        )
    return all_rows, provenance


def join_group_assignments(
    rows: Sequence[Mapping[str, object]],
    assignments: Mapping[str, GroupAssignment],
) -> list[dict[str, object]]:
    """Join source family and enforce the canonical physical grouping."""
    case_names = [str(row["case_name"]) for row in rows]
    if len(case_names) != len(set(case_names)):
        raise ValueError(
            "Canonical group assignments require globally unique case_name values"
        )
    observed = set(case_names)
    expected = set(assignments)
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        unexpected = sorted(observed - expected)[:5]
        raise ValueError(
            "Five OOF evaluations do not exactly cover group assignments; "
            f"missing={missing}, unexpected={unexpected}"
        )
    output: list[dict[str, object]] = []
    for row in rows:
        name = str(row["case_name"])
        assignment = assignments[name]
        source_group = str(row["source_group"])
        fold = str(row["fold"])
        if source_group != assignment.physical_group:
            raise ValueError(
                f"source_group differs from physical_group for {name!r}: "
                f"{source_group!r} != {assignment.physical_group!r}"
            )
        if fold != assignment.fold:
            raise ValueError(
                f"Evaluation fold differs from group assignment for {name!r}: "
                f"{fold!r} != {assignment.fold!r}"
            )
        enriched = dict(row)
        enriched["source_family"] = assignment.source_family
        enriched["group_id"] = f"{row['dataset']}::{source_group}"
        output.append(enriched)
    return output


def _cross_check_gated_csv(
    path: Path,
    calculated_rows: Sequence[Mapping[str, object]],
    *,
    gate_threshold: float,
) -> dict[str, object]:
    source = path.expanduser().resolve()
    before = _file_identity(source)
    try:
        with source.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != GATED_FIELDS:
                raise ValueError("Stored gated CSV header differs from GATED_FIELDS")
            stored = list(reader)
    except OSError as exc:
        raise ValueError(f"Cannot read gated CSV: {source}") from exc
    after = _file_identity(source)
    if before != after:
        raise RuntimeError(f"Gated CSV changed while being read: {source}")

    expected = {
        (str(row["dataset"]), str(row["case_name"])): row
        for row in calculated_rows
    }
    observed = {
        (str(row["dataset"]), str(row["case_name"])): row for row in stored
    }
    if len(observed) != len(stored) or set(observed) != set(expected):
        raise ValueError("Stored gated CSV rows do not match the five OOF evaluations")
    for identity, row in observed.items():
        calculated = expected[identity]
        for field in ("source_group", "role", "fold", "selected_output"):
            if str(row[field]) != str(calculated[field]):
                raise ValueError(
                    f"Stored gated CSV {field} differs for {identity!r}"
                )
        numeric_pairs = (
            (
                "gate_probability",
                _finite_float(
                    calculated["gate_probability"], "calculated gate_probability"
                ),
            ),
            ("gate_threshold", gate_threshold),
            (
                "gate_open",
                _finite_float(calculated["gate_open"], "calculated gate_open"),
            ),
            (
                "gated_iou_gain",
                _finite_float(
                    calculated["gated_iou_gain"], "calculated gated_iou_gain"
                ),
            ),
        )
        for field, expected_value in numeric_pairs:
            observed_value = _finite_float(row[field], f"stored {field}")
            if not math.isclose(
                observed_value, expected_value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Stored gated CSV {field} differs for {identity!r}"
                )
    return after


def build_group_summaries(
    rows: Sequence[Mapping[str, object]],
    *,
    label_minimum_gain: float,
    severe_loss_threshold: float,
) -> list[GroupSummary]:
    """Reduce crops within each qualified physical group before inference."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        group_id = str(row["group_id"])
        if not group_id or "::" not in group_id:
            raise ValueError("Rows need a qualified dataset::source_group group_id")
        grouped.setdefault(group_id, []).append(row)
    if not grouped:
        raise ValueError("Cannot summarize an empty row collection")

    summaries: list[GroupSummary] = []
    for group_id in sorted(grouped):
        members = grouped[group_id]
        datasets = {str(row["dataset"]) for row in members}
        source_groups = {str(row["source_group"]) for row in members}
        source_families = {str(row["source_family"]) for row in members}
        folds = {str(row["fold"]) for row in members}
        identity_sets = (datasets, source_groups, source_families, folds)
        if any(len(values) != 1 for values in identity_sets):
            raise ValueError(
                f"Qualified group crosses dataset/family/fold: {group_id!r}"
            )
        delta = np.asarray(
            [_finite_float(row["delta_iou"], "delta_iou") for row in members]
        )
        opened_values = [
            _finite_float(row["gate_open"], "gate_open") for row in members
        ]
        if any(value not in (0.0, 1.0) for value in opened_values):
            raise ValueError(f"Non-binary gate_open in group {group_id!r}")
        opened = np.asarray(opened_values, dtype=bool)
        if not np.isfinite(delta).all():
            raise ValueError(f"Non-finite delta in group {group_id!r}")
        practical = delta > label_minimum_gain
        severe = delta < severe_loss_threshold
        selected_rows = int(np.count_nonzero(opened))
        residual = {
            "mean_delta_iou": float(np.mean(delta)),
            "oracle_gain_iou": float(np.mean(np.maximum(delta, 0.0))),
            "practical_gain_rate": float(np.mean(practical)),
            "harmful_rate": float(np.mean(delta < -label_minimum_gain)),
            "severe_loss_rate": float(np.mean(severe)),
        }
        # Conditional quantities are genuinely undefined for a source on which
        # the gate never opens. They stay NaN internally and become JSON null.
        gate = {
            "coverage": float(np.mean(opened)),
            "precision": (
                float(np.mean(practical[opened])) if selected_rows else math.nan
            ),
            "selected_mean_delta_iou": (
                float(np.mean(delta[opened])) if selected_rows else math.nan
            ),
            "selected_severe_loss_rate": (
                float(np.mean(severe[opened])) if selected_rows else math.nan
            ),
            "system_mean_gain_iou": float(np.mean(np.where(opened, delta, 0.0))),
        }
        summaries.append(
            GroupSummary(
                group_id=group_id,
                dataset=next(iter(datasets)),
                source_group=next(iter(source_groups)),
                source_family=next(iter(source_families)),
                fold=next(iter(folds)),
                rows=len(members),
                selected_rows=selected_rows,
                residual=residual,
                gate=gate,
            )
        )
    return summaries


def _point_estimates(
    groups: Sequence[GroupSummary],
    *,
    family: str,
    estimands: Sequence[str],
) -> dict[str, float | None]:
    values = np.asarray(
        [
            [float(getattr(group, family)[name]) for name in estimands]
            for group in groups
        ],
        dtype=np.float64,
    )
    output: dict[str, float | None] = {}
    for column, name in enumerate(estimands):
        valid = values[np.isfinite(values[:, column]), column]
        output[name] = float(np.mean(valid)) if valid.size else None
    return output


def clustered_stratified_bootstrap(
    groups: Sequence[GroupSummary],
    *,
    family: str,
    estimands: Sequence[str],
    stratum: Callable[[GroupSummary], str],
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    """Percentile CIs from equal-weight source groups sampled within strata."""
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if not groups:
        raise ValueError("bootstrap requires at least one source group")
    matrix = np.asarray(
        [
            [float(getattr(group, family)[name]) for name in estimands]
            for group in groups
        ],
        dtype=np.float64,
    )
    strata: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        key = stratum(group)
        if not key:
            raise ValueError("Bootstrap stratum labels must be non-empty")
        strata.setdefault(key, []).append(index)
    stratum_indices = {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in sorted(strata.items())
    }
    draws = np.full((repetitions, len(estimands)), np.nan, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, repetitions, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, repetitions)
        chunk_size = stop - start
        sums = np.zeros((chunk_size, len(estimands)), dtype=np.float64)
        counts = np.zeros((chunk_size, len(estimands)), dtype=np.int64)
        for indices in stratum_indices.values():
            sampled_positions = rng.integers(
                0, indices.size, size=(chunk_size, indices.size)
            )
            sampled = matrix[indices[sampled_positions]]
            finite = np.isfinite(sampled)
            sums += np.where(finite, sampled, 0.0).sum(axis=1)
            counts += finite.sum(axis=1)
        np.divide(sums, counts, out=draws[start:stop], where=counts > 0)

    point = _point_estimates(groups, family=family, estimands=estimands)
    output: dict[str, dict[str, float | int | None]] = {}
    for column, name in enumerate(estimands):
        valid = draws[np.isfinite(draws[:, column]), column]
        if valid.size:
            low, high = np.quantile(valid, CI_QUANTILES)
            ci_low: float | None = float(low)
            ci_high: float | None = float(high)
        else:
            ci_low = ci_high = None
        output[name] = {
            "estimate": point[name],
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "bootstrap_valid_repetitions": int(valid.size),
            "bootstrap_valid_fraction": float(valid.size / repetitions),
        }
    return output


def _strata_counts(
    groups: Sequence[GroupSummary], key: Callable[[GroupSummary], str]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in groups:
        label = key(group)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _scope_summary(
    groups: Sequence[GroupSummary],
    *,
    family: str,
    estimands: Sequence[str],
    interpretation: str,
    stratum_name: str,
    stratum: Callable[[GroupSummary], str],
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    selected_groups = sum(group.selected_rows > 0 for group in groups)
    return {
        "interpretation": interpretation,
        "point_weighting": (
            "equal qualified physical groups; conditional gate metrics equally "
            "weight groups with at least one selected crop"
        ),
        "folds": sorted({group.fold for group in groups}),
        "rows": sum(group.rows for group in groups),
        "groups": len(groups),
        "source_families": sorted({group.source_family for group in groups}),
        "selected_rows": sum(group.selected_rows for group in groups),
        "selected_groups": selected_groups,
        "stratification": stratum_name,
        "strata_group_counts": _strata_counts(groups, stratum),
        "estimands": clustered_stratified_bootstrap(
            groups,
            family=family,
            estimands=estimands,
            stratum=stratum,
            repetitions=repetitions,
            seed=seed,
        ),
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
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


def _write_estimands_csv(
    path: Path, scopes: Mapping[str, Mapping[str, object]]
) -> None:
    fields = (
        "scope",
        "interpretation",
        "estimand",
        "estimate",
        "ci95_low",
        "ci95_high",
        "bootstrap_valid_repetitions",
        "bootstrap_valid_fraction",
        "rows",
        "groups",
        "selected_rows",
        "selected_groups",
    )
    rows: list[dict[str, object]] = []
    for scope_name, scope in scopes.items():
        for estimand, statistics in scope["estimands"].items():
            rows.append(
                {
                    "scope": scope_name,
                    "interpretation": scope["interpretation"],
                    "estimand": estimand,
                    **statistics,
                    "rows": scope["rows"],
                    "groups": scope["groups"],
                    "selected_rows": scope["selected_rows"],
                    "selected_groups": scope["selected_groups"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
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
    if repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    seed = int(args.seed)
    if seed < 0:
        raise ValueError("--seed must be non-negative")
    fold_directories = parse_fold_directories(args.fold_dir)
    gate_path = args.gate_json.expanduser().resolve()
    oof_path = args.oof_manifest.expanduser().resolve()
    assignment_path = args.group_assignments.expanduser().resolve()
    output = args.output.expanduser().resolve()

    gate_identity = _file_identity(gate_path)
    gate = LogisticConfidenceGate.load_json(gate_path)
    if _file_identity(gate_path) != gate_identity:
        raise RuntimeError("Gate JSON changed while being loaded")
    oof_identity = _file_identity(oof_path)
    manifest, manifest_sha256 = load_and_validate_oof_manifest(
        oof_path, verify_sources=False
    )
    if _file_identity(oof_path) != oof_identity:
        raise RuntimeError("OOF manifest changed while being loaded")
    if manifest_sha256 != oof_identity["sha256"]:
        raise RuntimeError("OOF manifest loader returned an inconsistent SHA-256")
    label_threshold, severe_threshold = _validate_gate_and_manifest(
        gate, manifest, manifest_sha256
    )
    assignments, assignment_identity = _read_assignments_stable(assignment_path)
    recorded_assignment = manifest["artifacts"]["protocol"]["group_assignments"]
    if not _same_recorded_identity(assignment_identity, recorded_assignment):
        raise ValueError("Group assignments differ from OOF protocol provenance")

    evaluation_rows, evaluation_provenance = _load_five_evaluations(
        fold_directories, manifest
    )
    joined_rows = join_group_assignments(evaluation_rows, assignments)
    gated_rows, probabilities, decisions = apply_gate_to_rows(gate, joined_rows)
    for enriched, joined, probability, decision in zip(
        gated_rows, joined_rows, probabilities, decisions, strict=True
    ):
        enriched["source_family"] = joined["source_family"]
        enriched["group_id"] = joined["group_id"]
        if float(enriched["gate_probability"]) != float(probability):
            raise AssertionError("Gate probability changed while enriching rows")
        if bool(int(enriched["gate_open"])) != bool(decision):
            raise AssertionError("Gate decision changed while enriching rows")

    gated_csv_identity: dict[str, object] | None = None
    if args.gated_csv is not None:
        gated_csv_identity = _cross_check_gated_csv(
            args.gated_csv,
            gated_rows,
            gate_threshold=gate.threshold,
        )

    groups = build_group_summaries(
        gated_rows,
        label_minimum_gain=label_threshold,
        severe_loss_threshold=severe_threshold,
    )
    all5_stratum = lambda group: f"fold={group.fold}::family={group.source_family}"
    fold4_stratum = lambda group: f"family={group.source_family}"
    fit_groups = [group for group in groups if group.fold in ("0", "1", "2", "3")]
    calibration_groups = [group for group in groups if group.fold == "4"]
    if not fit_groups or not calibration_groups:
        raise ValueError("Expected non-empty gate fit folds and calibration fold")

    # Every scope gets a fresh generator initialized from the recorded seed.
    # Therefore a future scope cannot perturb any existing confidence interval.
    scope_seeds = {
        "residual_oof_all5": seed,
        "gate_folds0_3_apparent": seed,
        "gate_fold4_calibration_descriptive": seed,
        "gate_all5_apparent": seed,
    }
    scopes: dict[str, Mapping[str, object]] = {
        "residual_oof_all5": _scope_summary(
            groups,
            family="residual",
            estimands=RESIDUAL_ESTIMANDS,
            interpretation="OOF residual predictions across folds 0-4",
            stratum_name="fold x source_family",
            stratum=all5_stratum,
            repetitions=repetitions,
            seed=scope_seeds["residual_oof_all5"],
        ),
        "gate_folds0_3_apparent": _scope_summary(
            fit_groups,
            family="gate",
            estimands=GATE_ESTIMANDS,
            interpretation="apparent: logistic coefficients fitted on these rows",
            stratum_name="fold x source_family",
            stratum=all5_stratum,
            repetitions=repetitions,
            seed=scope_seeds["gate_folds0_3_apparent"],
        ),
        "gate_fold4_calibration_descriptive": _scope_summary(
            calibration_groups,
            family="gate",
            estimands=GATE_ESTIMANDS,
            interpretation="descriptive calibration: threshold selected on these rows",
            stratum_name="source_family",
            stratum=fold4_stratum,
            repetitions=repetitions,
            seed=scope_seeds["gate_fold4_calibration_descriptive"],
        ),
        "gate_all5_apparent": _scope_summary(
            groups,
            family="gate",
            estimands=GATE_ESTIMANDS,
            interpretation=(
                "apparent composite: coefficient-fit and calibration rows combined"
            ),
            stratum_name="fold x source_family",
            stratum=all5_stratum,
            repetitions=repetitions,
            seed=scope_seeds["gate_all5_apparent"],
        ),
    }
    summary: dict[str, object] = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "policy": {
            "analytical_only": True,
            "group_unit": "dataset::source_group",
            "group_first": True,
            "group_balanced_point_metrics": True,
            "conditional_gate_group_weighting": (
                "equal groups with at least one selected crop"
            ),
            "bootstrap_method": "clustered stratified percentile",
            "confidence_level": 0.95,
            "threshold_source": "frozen gate JSON",
            "threshold_selected_or_recalibrated": False,
            "gate_open_rule": "gate_probability >= gate_threshold",
            "practical_gain_rule": "delta_iou > label_minimum_gain",
            "harmful_rule": "delta_iou < -label_minimum_gain",
            "severe_loss_rule": "delta_iou < severe_loss_threshold",
        },
        "parameters": {
            "bootstrap_repetitions": repetitions,
            "seed": seed,
            "scope_seeds": scope_seeds,
            "label_minimum_gain": label_threshold,
            "severe_loss_threshold": severe_threshold,
            "gate_threshold": gate.threshold,
        },
        "inputs": {
            "gate_json": gate_identity,
            "oof_manifest": {
                **oof_identity,
                "gate_provenance_sha256_match": True,
            },
            "group_assignments": {
                **assignment_identity,
                "oof_protocol_identity_match": True,
            },
            "gated_csv_cross_check": (
                {**gated_csv_identity, "all_rows_match_frozen_gate": True}
                if gated_csv_identity is not None
                else {"supplied": False}
            ),
            "evaluations": evaluation_provenance,
        },
        "join_audit": {
            "assignment_rows": len(assignments),
            "evaluation_rows": len(joined_rows),
            "exact_sample_set_match": True,
            "source_group_equals_physical_group": True,
            "fold_match": True,
            "qualified_groups": len(groups),
        },
        "scopes": scopes,
        "limitations": [
            (
                "The historical baseline was trained on the full training split; "
                "these intervals are exploratory, not confirmatory."
            ),
            (
                "Gate folds 0-3 are apparent for logistic coefficients; fold 4 "
                "is descriptive because it selected the frozen threshold."
            ),
            (
                "Percentile intervals describe sampling of the observed physical "
                "groups and do not correct domain shift or model-selection uncertainty."
            ),
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / "summary.json", summary)
    _write_estimands_csv(output / "estimands.csv", scopes)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(
        json.dumps(
            {
                "output_scopes": list(summary["scopes"]),
                "bootstrap_repetitions": summary["parameters"][
                    "bootstrap_repetitions"
                ],
                "gate_threshold": summary["parameters"]["gate_threshold"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
