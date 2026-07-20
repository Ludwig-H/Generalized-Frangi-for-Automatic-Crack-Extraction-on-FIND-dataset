from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "run_frangigraph_logistic_gate_pilot.sh"


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_has_safe_shell_and_user_facing_modes() -> None:
    text = workflow_text()
    assert "set -Eeuo pipefail" in text
    assert "umask 027" in text
    assert "flock -n 9" in text
    assert "--mode SMOKE|FULL [--resume]" in text
    assert "workflow_contract.json" in text
    assert "failures.log" in text
    assert "PIPESTATUS" in text
    assert 'CACHE_ROOT="${FRANGIGRAPH_GRAPH_CACHE:-${RUN_ROOT}/graph_cache}"' in text
    assert '"graph_cache_root", "train_list"' in text
    assert 'RASTER_CONDITION="${FRANGIGRAPH_RASTER_CONDITION:-correct}"' in text
    assert '"frangi_graph_order", "raster_condition", "hidden_channels"' in text
    assert '"schema_version": 2' in text
    assert (
        'FRANGI_SCALES_TEXT="${FRANGIGRAPH_SCALES:-1.0 3.0 5.0 9.0 15.0}"'
        in text
    )
    assert 'DEFAULT_LABEL_MINIMUM_GAIN="0.0"' in text
    assert 'DEFAULT_LABEL_MINIMUM_GAIN="0.005"' in text
    assert (
        'LABEL_MINIMUM_GAIN="${FRANGIGRAPH_LABEL_MINIMUM_GAIN:-${DEFAULT_LABEL_MINIMUM_GAIN}}"'
        in text
    )

    result = subprocess.run(
        ["bash", str(WORKFLOW), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "SMOKE|FULL" in result.stdout
    assert "FRANGIGRAPH_RASTER_CONDITION" in result.stdout


def test_workflow_binds_equal_capacity_raster_condition_everywhere() -> None:
    text = workflow_text()
    assert re.search(
        r'case "\$\{RASTER_CONDITION\}" in\s+correct\|no_evidence\) ;;',
        text,
    )
    assert (
        "FRANGIGRAPH_RASTER_CONDITION must be exactly correct or no_evidence"
        in text
    )
    assert text.count('--raster-condition "${RASTER_CONDITION}"') == 2
    assert "--raster-condition correct" not in text

    # The resolved value is passed once to the immutable contract generator;
    # `${VAR:-correct}` makes unset, empty, and explicit `correct` byte-equivalent.
    contract_prefix, _ = text.split("<<'PY'", maxsplit=1)
    assert contract_prefix.count('"${RASTER_CONDITION}"') == 2


def test_workflow_rejects_unknown_raster_condition_before_io(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "CRACKSAM2_DATA_ROOT": str(tmp_path / "missing-data"),
        "SAM2_CHECKPOINT": str(tmp_path / "missing-sam2.pt"),
        "BASELINE_CHECKPOINT": str(tmp_path / "missing-baseline.pt"),
        "FRANGIGRAPH_RUN_ROOT": str(tmp_path / "unused-run"),
        "FRANGIGRAPH_RASTER_CONDITION": "permuted",
    }
    result = subprocess.run(
        ["bash", str(WORKFLOW), "--mode", "SMOKE"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    assert (
        "FRANGIGRAPH_RASTER_CONDITION must be exactly correct or no_evidence"
        in result.stderr
    )
    assert "Required file is missing" not in result.stderr


def test_workflow_runs_exactly_five_strict_oof_folds() -> None:
    text = workflow_text()
    assert text.count("for fold in 0 1 2 3 4; do") == 2
    assert re.search(
        r"if \(\(fold < 4\)\); then\s+.*?TRAIN_COMMAND\+=\(--exclude-training-fold 4\)",
        text,
        flags=re.DOTALL,
    )
    assert '[[ "${fold}" == "4" ]] && role="gate_calibration"' in text
    assert "--residual-checkpoint" in text
    assert '"${TRAIN_ROOT}/fold_${fold}/latest.pt"' in text


def test_gate_threshold_inputs_are_oof_only() -> None:
    text = workflow_text()
    assert "assemble_logistic_gate_data.py" in text
    assert "train_logistic_gate.py" in text
    assert "evaluate_logistic_gate.py" in text
    assert "--train-csv" in text
    assert '"${ASSEMBLY_ROOT}/gate_fit.csv"' in text
    assert "--calibration-csv" in text
    assert '"${ASSEMBLY_ROOT}/gate_calibration.csv"' in text
    assert "--fold-dir \"4=${OOF_ROOT}/fold_4\"" in text
    assert "--evaluation-dir \"${OOF_ROOT}/fold_4\"" in text

    # The only occurrence documents the prohibition; no historical list or
    # evaluation role is wired into a command.
    assert text.count("historical_test") == 1
    assert "--role historical_test" not in text


def test_workflow_is_provider_agnostic_and_resumable() -> None:
    text = workflow_text().lower()
    assert "gcloud " not in text
    assert "compute instances" not in text
    assert "--overwrite" not in text
    assert "train_command+=(--resume)" in text
    assert ".cracksam2-frangi-graph-v2.json" in text
    assert "evaluation_contract.json" in text
    assert "oof_manifest.json" in text
    assert "git -c \"${repo_root}\" status --porcelain --untracked-files=all" in text
