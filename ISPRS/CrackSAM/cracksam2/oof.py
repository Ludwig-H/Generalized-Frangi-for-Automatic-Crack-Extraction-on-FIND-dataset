"""Canonical leak-free fold contract for the FrangiGraph gate pilot.

Fold 4 is reserved exclusively for gate-threshold calibration.  Consequently,
residuals producing gate-fit rows for folds 0--3 must never train on fold 4,
whereas the residual producing fold-4 calibration rows trains on folds 0--3.
"""

from __future__ import annotations

from collections.abc import Mapping


OOF_TRAINING_SCHEMA = "cracksam2.frangigraph-strict-oof-training"
OOF_TRAINING_SCHEMA_VERSION = 1
RESIDUAL_RUN_CONTRACT_VERSION = 2
OOF_FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
GATE_FIT_FOLDS: tuple[int, ...] = (0, 1, 2, 3)
GATE_CALIBRATION_FOLD = 4


def strict_oof_training_contract(held_out_fold: int) -> dict[str, object]:
    """Return the only admissible residual-training fold contract."""
    if isinstance(held_out_fold, bool) or held_out_fold not in OOF_FOLDS:
        raise ValueError("held_out_fold must be exactly one of 0, 1, 2, 3, 4")
    additional_exclusions = (
        (GATE_CALIBRATION_FOLD,)
        if held_out_fold in GATE_FIT_FOLDS
        else ()
    )
    all_exclusions = tuple(sorted({held_out_fold, *additional_exclusions}))
    training_folds = tuple(
        fold for fold in OOF_FOLDS if fold not in all_exclusions
    )
    role = (
        "gate_fit"
        if held_out_fold in GATE_FIT_FOLDS
        else "gate_calibration"
    )
    return {
        "schema": OOF_TRAINING_SCHEMA,
        "schema_version": OOF_TRAINING_SCHEMA_VERSION,
        "fold_universe": list(OOF_FOLDS),
        "gate_fit_folds": list(GATE_FIT_FOLDS),
        "gate_calibration_fold": GATE_CALIBRATION_FOLD,
        "held_out_fold": held_out_fold,
        "evaluation_role": role,
        "training_folds": list(training_folds),
        "additional_excluded_training_folds": list(additional_exclusions),
        "all_excluded_training_folds": list(all_exclusions),
    }


def validate_strict_oof_training_contract(
    value: object,
    *,
    held_out_fold: int | None = None,
    evaluation_role: str | None = None,
) -> dict[str, object]:
    """Reject a missing, permissive, stale, or role-incoherent OOF contract."""
    if not isinstance(value, Mapping):
        raise ValueError("strict OOF training contract must be a JSON object")
    recorded_fold = value.get("held_out_fold")
    if isinstance(recorded_fold, bool) or not isinstance(recorded_fold, int):
        raise ValueError("strict OOF training contract has no integer held_out_fold")
    expected = strict_oof_training_contract(recorded_fold)
    observed = dict(value)
    if observed != expected:
        differing = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise ValueError(
            "strict OOF training contract is inconsistent; differing fields: "
            + ", ".join(differing)
        )
    if held_out_fold is not None and recorded_fold != held_out_fold:
        raise ValueError(
            "strict OOF training contract held-out fold differs from evaluation"
        )
    if evaluation_role is not None and expected["evaluation_role"] != evaluation_role:
        raise ValueError(
            "strict OOF training contract role differs from evaluation role"
        )
    return expected


def validate_oof_run_contract(value: object) -> dict[str, object]:
    """Validate the OOF guard and its redundant top-level checkpoint bindings."""
    if not isinstance(value, Mapping):
        raise ValueError("Residual checkpoint has no run_contract")
    if value.get("contract_version") != RESIDUAL_RUN_CONTRACT_VERSION:
        raise ValueError(
            "Residual run_contract predates the strict OOF leakage guard"
        )
    held_out_fold = value.get("held_out_fold")
    if isinstance(held_out_fold, bool) or not isinstance(held_out_fold, int):
        raise ValueError("Residual checkpoint has no valid held_out_fold")
    expected = validate_strict_oof_training_contract(
        value.get("oof_training"), held_out_fold=held_out_fold
    )
    if value.get("excluded_training_folds") != expected[
        "additional_excluded_training_folds"
    ]:
        raise ValueError(
            "Residual run_contract excluded_training_folds violates strict OOF policy"
        )
    if value.get("all_folds_excluded_from_training") != expected[
        "all_excluded_training_folds"
    ]:
        raise ValueError(
            "Residual run_contract all_folds_excluded_from_training violates "
            "strict OOF policy"
        )
    return expected


__all__ = [
    "GATE_CALIBRATION_FOLD",
    "GATE_FIT_FOLDS",
    "OOF_FOLDS",
    "OOF_TRAINING_SCHEMA",
    "OOF_TRAINING_SCHEMA_VERSION",
    "RESIDUAL_RUN_CONTRACT_VERSION",
    "strict_oof_training_contract",
    "validate_oof_run_contract",
    "validate_strict_oof_training_contract",
]
