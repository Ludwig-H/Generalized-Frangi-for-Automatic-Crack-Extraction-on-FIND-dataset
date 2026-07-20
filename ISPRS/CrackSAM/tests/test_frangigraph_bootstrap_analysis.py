from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
if str(CRACKSAM_ROOT) not in sys.path:
    sys.path.insert(0, str(CRACKSAM_ROOT))

import analyze_frangigraph_pilot_bootstrap as analysis  # noqa: E402


def _row(
    case: str,
    group: str,
    delta: float,
    opened: bool,
    *,
    fold: str = "0",
    family: str = "family-a",
) -> dict[str, object]:
    return {
        "case_name": case,
        "dataset": "dataset-a",
        "source_group": group,
        "source_family": family,
        "fold": fold,
        "group_id": f"dataset-a::{group}",
        "delta_iou": delta,
        "gate_open": int(opened),
    }


def _summary(
    group: str,
    *,
    fold: str,
    family: str,
    value: float,
    selected: bool = True,
) -> analysis.GroupSummary:
    residual = {name: value for name in analysis.RESIDUAL_ESTIMANDS}
    gate = {name: value for name in analysis.GATE_ESTIMANDS}
    if not selected:
        for name in (
            "precision",
            "selected_mean_delta_iou",
            "selected_severe_loss_rate",
        ):
            gate[name] = float("nan")
        gate["coverage"] = 0.0
        gate["system_mean_gain_iou"] = 0.0
    return analysis.GroupSummary(
        group_id=f"dataset-a::{group}",
        dataset="dataset-a",
        source_group=group,
        source_family=family,
        fold=fold,
        rows=2,
        selected_rows=1 if selected else 0,
        residual=residual,
        gate=gate,
    )


def test_group_first_estimands_use_strict_gain_harm_and_severe_boundaries() -> None:
    groups = analysis.build_group_summaries(
        [
            _row("a-1", "source-a", 0.005, True),
            _row("a-2", "source-a", -0.05, True),
            _row("b-1", "source-b", 0.1, False),
        ],
        label_minimum_gain=0.005,
        severe_loss_threshold=-0.05,
    )

    assert len(groups) == 2
    source_a = next(group for group in groups if group.source_group == "source-a")
    assert source_a.residual["practical_gain_rate"] == 0.0
    assert source_a.residual["harmful_rate"] == 0.5
    assert source_a.residual["severe_loss_rate"] == 0.0
    assert source_a.gate["precision"] == 0.0
    assert source_a.gate["selected_severe_loss_rate"] == 0.0

    points = analysis._point_estimates(
        groups,
        family="residual",
        estimands=("mean_delta_iou",),
    )
    # Equal source weights: mean(mean(0.005, -0.05), mean(0.1)), not the
    # crop-weighted mean of all three observations.
    assert points["mean_delta_iou"] == pytest.approx((-0.0225 + 0.1) / 2.0)


def test_clustered_bootstrap_is_deterministic_and_preserves_singleton_strata() -> None:
    groups = [
        _summary("a", fold="0", family="A", value=-0.2),
        _summary("b", fold="1", family="B", value=0.4),
    ]
    kwargs = {
        "family": "residual",
        "estimands": ("mean_delta_iou",),
        "stratum": lambda group: f"{group.fold}::{group.source_family}",
        "repetitions": 500,
        "seed": 3407,
    }

    first = analysis.clustered_stratified_bootstrap(groups, **kwargs)
    repeated = analysis.clustered_stratified_bootstrap(groups, **kwargs)

    assert first == repeated
    statistics = first["mean_delta_iou"]
    assert statistics["estimate"] == pytest.approx(0.1)
    assert statistics["ci95_low"] == pytest.approx(0.1)
    assert statistics["ci95_high"] == pytest.approx(0.1)
    assert statistics["bootstrap_valid_fraction"] == 1.0


def test_closed_gate_reports_conditional_estimands_as_undefined() -> None:
    groups = [
        _summary("a", fold="4", family="A", value=0.2, selected=False),
        _summary("b", fold="4", family="A", value=-0.2, selected=False),
    ]

    result = analysis.clustered_stratified_bootstrap(
        groups,
        family="gate",
        estimands=analysis.GATE_ESTIMANDS,
        stratum=lambda group: group.source_family,
        repetitions=200,
        seed=4,
    )

    for name in (
        "precision",
        "selected_mean_delta_iou",
        "selected_severe_loss_rate",
    ):
        assert result[name]["estimate"] is None
        assert result[name]["ci95_low"] is None
        assert result[name]["ci95_high"] is None
        assert result[name]["bootstrap_valid_repetitions"] == 0
        assert result[name]["bootstrap_valid_fraction"] == 0.0
    assert result["coverage"]["estimate"] == 0.0
    assert result["coverage"]["ci95_low"] == 0.0
    assert result["system_mean_gain_iou"]["estimate"] == 0.0


def test_assignment_join_requires_exact_rows_group_and_fold() -> None:
    assignments = {
        "case-a": analysis.GroupAssignment("case-a", "A", "source-a", "0"),
        "case-b": analysis.GroupAssignment("case-b", "B", "source-b", "4"),
    }
    rows = [
        _row("case-a", "source-a", 0.1, True, fold="0", family="ignored"),
        _row("case-b", "source-b", -0.1, False, fold="4", family="ignored"),
    ]

    joined = analysis.join_group_assignments(rows, assignments)
    assert [row["source_family"] for row in joined] == ["A", "B"]
    assert joined[1]["group_id"] == "dataset-a::source-b"

    wrong_group = [dict(rows[0], source_group="not-source-a"), rows[1]]
    with pytest.raises(ValueError, match="physical_group"):
        analysis.join_group_assignments(wrong_group, assignments)
    wrong_fold = [rows[0], dict(rows[1], fold="3")]
    with pytest.raises(ValueError, match="fold differs"):
        analysis.join_group_assignments(wrong_fold, assignments)
    with pytest.raises(ValueError, match="exactly cover"):
        analysis.join_group_assignments(rows[:1], assignments)


def test_gated_csv_cross_check_detects_stored_decision_drift(tmp_path: Path) -> None:
    calculated = {field: "" for field in analysis.GATED_FIELDS}
    calculated.update(
        {
            "dataset": "dataset-a",
            "case_name": "case-a",
            "source_group": "source-a",
            "role": "gate_fit",
            "fold": "0",
            "selected_output": "candidate",
            "gate_probability": 0.75,
            "gate_threshold": 0.5,
            "gate_open": 1,
            "gated_iou_gain": 0.1,
        }
    )
    path = tmp_path / "per_image_gated.csv"

    def write(opened: int) -> None:
        row = dict(calculated, gate_open=opened)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=analysis.GATED_FIELDS)
            writer.writeheader()
            writer.writerow(row)

    write(1)
    identity = analysis._cross_check_gated_csv(
        path, [calculated], gate_threshold=0.5
    )
    assert identity["sha256"]

    write(0)
    with pytest.raises(ValueError, match="gate_open differs"):
        analysis._cross_check_gated_csv(path, [calculated], gate_threshold=0.5)
