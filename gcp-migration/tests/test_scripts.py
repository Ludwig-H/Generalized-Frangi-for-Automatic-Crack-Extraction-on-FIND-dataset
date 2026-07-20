from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SHELL_SCRIPTS = tuple(sorted(SCRIPT_ROOT.glob("*.sh")))


def test_all_shell_scripts_have_valid_bash_syntax() -> None:
    assert SHELL_SCRIPTS
    for script in SHELL_SCRIPTS:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_operational_entrypoints_are_executable() -> None:
    for script in SHELL_SCRIPTS:
        assert os.access(script, os.X_OK), script


def test_g4_quota_contract_is_exact_and_cpu_is_informative() -> None:
    content = (SCRIPT_ROOT / "check_quotas.sh").read_text(encoding="utf-8")
    assert "GPUS-ALL-REGIONS-per-project" in content
    assert "PREEMPTIBLE-NVIDIA-RTX-PRO-6000-GPUS-per-project-region" in content
    assert "HDB-TOTAL-GB-per-project-region" in content
    assert '"PREEMPTIBLE_CPUS (informatif)"' in content
    assert 'quota_is_sufficient "${CPUS_SPOT_AVAILABLE}"' not in content


def test_no_automatic_capacity_fallback() -> None:
    capacity = (SCRIPT_ROOT / "check_capacity.sh").read_text(encoding="utf-8")
    assert "gcloud compute instances create" not in capacity
    assert "Le script ne choisit ni ne crée automatiquement une autre zone" in capacity
