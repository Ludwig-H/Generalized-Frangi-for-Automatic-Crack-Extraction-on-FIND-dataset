#!/usr/bin/env python3
"""Generate the exhaustive French CrackSAM 2 Frangi milestone report.

The script is deliberately post-processing only: it never performs inference and
never loads a model checkpoint.  It consumes the durable artifacts produced by
``evaluate_sam2.py`` and, when present, the exact direct-mask Wasserstein results
published by ``compute_exact_wasserstein.py``.  Missing optional artifacts are
reported explicitly instead of being silently replaced by approximations.

Expected artifact layout::

    ARTIFACT_ROOT/
      baseline_r4/final_evaluation/
      frangi_r4/milestone_comparison/
        epoch20/ epoch25_best/ epoch30/ epoch55/ epoch70/

Each evaluation directory contains a ``summary.csv``, one ``per_image.csv`` per
dataset and (for qualitative panels) saved binary predictions.  Exact
Wasserstein results, if available, live in ``wasserstein_exact/`` below an
evaluation directory.  The legacy ``milestone_epochXX_evaluation`` names are
also recognized to make archived VM snapshots usable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Reporting must also work on a headless VM or in CI.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402


DATASETS: tuple[str, ...] = (
    "khanhha_original",
    "khanhha_noisy1",
    "khanhha_noisy2",
    "road420",
    "facade390",
    "concrete3k",
)
DATASET_LABELS: dict[str, str] = {
    "khanhha_original": "Khanhha original",
    "khanhha_noisy1": "Khanhha bruité 1",
    "khanhha_noisy2": "Khanhha bruité 2",
    "road420": "Road420",
    "facade390": "Facade390",
    "concrete3k": "Concrete3k",
}
DATASET_SHORT_LABELS: dict[str, str] = {
    "khanhha_original": "Original",
    "khanhha_noisy1": "Bruit 1",
    "khanhha_noisy2": "Bruit 2",
    "road420": "Road420",
    "facade390": "Facade390",
    "concrete3k": "Concrete3k",
}
DATASET_SAMPLES: dict[str, int] = {
    "khanhha_original": 1_695,
    "khanhha_noisy1": 1_695,
    "khanhha_noisy2": 1_695,
    "road420": 420,
    "facade390": 390,
    "concrete3k": 3_000,
}
MILESTONES: tuple[tuple[str, int, str], ...] = (
    ("epoch20", 20, "Frangi — époque 20"),
    ("epoch25_best", 25, "Frangi — époque 25 (best validation)"),
    ("epoch30", 30, "Frangi — époque 30"),
    ("epoch55", 55, "Frangi — époque 55"),
    ("epoch70", 70, "Frangi — époque 70"),
)
METRICS: tuple[str, ...] = ("precision", "recall", "dice", "iou", "wasserstein")
HIGHER_IS_BETTER: dict[str, bool] = {
    "precision": True,
    "recall": True,
    "dice": True,
    "iou": True,
    "wasserstein": False,
}

# Values transcribed from CrackSAM.pdf.  Table 6 contains IoU for the six
# evaluation configurations; Table 2 supplies Pr/Re/F1 for qv LoRA rank 4 on
# the clean Khanhha test set.  Keeping the values beside their provenance makes
# accidental mixing with newly computed SAM 2 metrics less likely.
PAPER_MODELS: dict[str, dict[str, Any]] = {
    "paper_adapter_d32": {
        "label": "CrackSAM originel — Adapter d=32 (SAM 1, ViT-H)",
        "source": "CrackSAM.pdf, Table 6",
        "iou": {
            "khanhha_original": 0.6495,
            "khanhha_noisy1": 0.5466,
            "khanhha_noisy2": 0.4763,
            "road420": 0.6149,
            "facade390": 0.4718,
            "concrete3k": 0.6718,
        },
        "precision": {"khanhha_original": 0.7676},
        "recall": {"khanhha_original": 0.7965},
        "dice": {"khanhha_original": 0.7704},
        "secondary_source": "CrackSAM.pdf, Table 1 (Adapter d=32)",
    },
    "paper_lora_qv_r4": {
        "label": "CrackSAM originel — LoRA qv r=4 (SAM 1, ViT-H)",
        "source": "CrackSAM.pdf, Table 6",
        "iou": {
            "khanhha_original": 0.6416,
            "khanhha_noisy1": 0.5782,
            "khanhha_noisy2": 0.4915,
            "road420": 0.6222,
            "facade390": 0.4544,
            "concrete3k": 0.6798,
        },
        "precision": {"khanhha_original": 0.7620},
        "recall": {"khanhha_original": 0.7918},
        "dice": {"khanhha_original": 0.7639},
        "secondary_source": "CrackSAM.pdf, Table 2 (rank=4)",
    },
}

CATEGORY_LABELS: dict[str, str] = {
    "gain_frangi": "Gain Frangi maximal",
    "gain_baseline": "Gain baseline maximal",
    "both_good": "Les deux bons",
    "both_weak": "Les deux faibles",
    "median": "Cas médian",
    "sparse_divergent": "GT vide/clairsemée divergente",
}
SELECTION_CATEGORIES: tuple[str, ...] = (
    "gain_frangi",
    "gain_baseline",
    "both_good",
    "both_weak",
    "median",
)
IMAGE_DIRECTORY_NAMES = {"images", "image", "imgs", "jpegimages"}
CHECKPOINT_MANIFEST_RELATIVE = Path(
    "ISPRS/CrackSAM/results/2026-07-14_checkpoint_manifest.json"
)
WASSERSTEIN_FEASIBILITY_RELATIVE = Path(
    "ISPRS/CrackSAM/results/2026-07-14_wasserstein_feasibility.json"
)


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    variant: str
    epoch: int | None
    path: Path


@dataclass
class RunData:
    spec: RunSpec
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_image: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    exact_root: Path | None = None
    exact_status: str = "absent"


@dataclass(frozen=True)
class SelectedCase:
    dataset: str
    category: str
    case_name: str
    baseline_iou: float
    frangi_iou: float
    delta_iou: float
    target_pixels: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--primary-milestone",
        choices=tuple(item[0] for item in MILESTONES),
        default="epoch25_best",
        help=(
            "Jalon Frangi utilisé pour les statistiques appariées et les "
            "illustrations (défaut: epoch25_best, choisi sur la validation)."
        ),
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=5_000,
        help="Nombre de réplications bootstrap déterministes (défaut: 5000).",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=20_260_714
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=1e-6,
        help="Tolérance absolue pour wins/ties/losses (défaut: 1e-6).",
    )
    parser.add_argument(
        "--copy-prompts",
        type=int,
        default=12,
        help="Nombre cible de prompts .npy à copier avec manifeste (défaut: 12).",
    )
    parser.add_argument(
        "--panel-dpi",
        type=int,
        default=120,
        help="Résolution des panneaux JPEG qualitatifs (défaut: 120).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Échouer si une des six évaluations attendues est incomplète.",
    )
    return parser.parse_args()


def _finite_float(value: Any) -> float:
    if value in (None, "", "None", "null", "nan", "NaN"):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray([_finite_float(value) for value in values], dtype=np.float64)
    return array[np.isfinite(array)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base_seed: int, *parts: str) -> int:
    payload = "\0".join((str(base_seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _prepare_managed_output(output: Path) -> None:
    """Remove only artifacts owned by this generator before a fresh report."""
    output.mkdir(parents=True, exist_ok=True)
    for name in ("tables", "figures", "prompts_npy", "wasserstein_audit"):
        path = output / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    for name in ("RAPPORT_FRANGI_MILESTONES.md", "report_manifest.json"):
        (output / name).unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def _field_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _field_order(rows)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        if fields:
            writer = csv.DictWriter(
                output,
                fieldnames=fields,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for source in rows:
                row = {
                    key: (
                        ""
                        if isinstance(value, float) and not math.isfinite(value)
                        else value
                    )
                    for key, value in source.items()
                }
                writer.writerow(row)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: str, limit: int = 100) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.replace("\\", "/"))
    normalized = normalized.strip("._") or "case"
    if len(normalized) <= limit:
        return normalized
    suffix = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{normalized[: limit - 11]}_{suffix}"


def _display_case_name(value: str, width: int = 30) -> str:
    """Compact a path-like case name for a plot title without losing identity."""
    name = Path(value.replace("\\", "/")).name
    if len(name) > 2 * width:
        name = f"{name[: width - 2]}…{name[-width + 1 :]}"
    return "\n".join(textwrap.wrap(name, width=width, break_long_words=True))


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    stat = path.stat()
    display_path = (
        path.relative_to(relative_to).as_posix()
        if relative_to is not None and path.is_relative_to(relative_to)
        else str(path)
    )
    return {
        "path": display_path,
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def load_checkpoint_manifest(
    warnings: list[str],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    repository_root = Path(__file__).resolve().parents[2]
    path = repository_root / CHECKPOINT_MANIFEST_RELATIVE
    if not path.is_file():
        warnings.append(f"Manifeste versionné des checkpoints absent: {path}")
        return path, {}, []
    try:
        value = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Manifeste des checkpoints invalide {path}: {exc}")
        return path, {}, []
    if not isinstance(value, Mapping) or value.get("format_version") != 1:
        warnings.append(f"Format de manifeste checkpoint non pris en charge: {path}")
        return path, {}, []

    rows: list[dict[str, Any]] = []
    base = value.get("base_checkpoint_dependency")
    if isinstance(base, Mapping):
        base_row = {
            "id": "sam2_hiera_large_base_dependency",
            "variant": "foundation",
            "role": "base dependency (non versionnée)",
            "epoch": None,
            "global_step": None,
            "size_bytes": base.get("size_bytes"),
            "sha256": base.get("sha256"),
            "run_contract_sha256": None,
            "vm_path": base.get("vm_path"),
            "local_backup_path": base.get("local_backup_path"),
            "equivalent_alias": None,
            "local_present": False,
            "local_verified": False,
            "vm_present": False,
            "vm_verified": False,
        }
        base_relative = base.get("local_backup_path")
        base_path = repository_root / str(base_relative) if base_relative else None
        base_row["local_present"] = bool(base_path and base_path.is_file())
        if base_path and base_path.is_file():
            base_row["local_verified"] = bool(
                base_path.stat().st_size == base.get("size_bytes")
                and _sha256(base_path) == base.get("sha256")
            )
            if not base_row["local_verified"]:
                warnings.append(
                    f"Checkpoint de fondation différent du manifeste: {base_relative}"
                )
        vm_value = base.get("vm_path")
        vm_path = Path(str(vm_value)).expanduser() if vm_value else None
        base_row["vm_present"] = bool(vm_path and vm_path.is_file())
        if vm_path and vm_path.is_file():
            base_row["vm_verified"] = bool(
                vm_path.stat().st_size == base.get("size_bytes")
                and _sha256(vm_path) == base.get("sha256")
            )
            if not base_row["vm_verified"]:
                warnings.append(
                    f"Checkpoint de fondation VM différent du manifeste: {vm_path}"
                )
        rows.append(base_row)
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list):
        warnings.append(f"Liste checkpoints absente du manifeste: {path}")
        checkpoints = []
    for entry in checkpoints:
        if not isinstance(entry, Mapping):
            continue
        row = {
            key: entry.get(key)
            for key in (
                "id",
                "variant",
                "role",
                "epoch",
                "global_step",
                "size_bytes",
                "sha256",
                "run_contract_sha256",
                "vm_path",
                "local_backup_path",
                "equivalent_alias",
            )
        }
        local_relative = entry.get("local_backup_path")
        local_path = repository_root / str(local_relative) if local_relative else None
        row["local_present"] = bool(local_path and local_path.is_file())
        row["local_verified"] = False
        row["vm_present"] = False
        row["vm_verified"] = False
        if local_path and local_path.is_file():
            expected_size = entry.get("size_bytes")
            expected_hash = entry.get("sha256")
            size_matches = expected_size == local_path.stat().st_size
            hash_matches = (
                isinstance(expected_hash, str) and _sha256(local_path) == expected_hash
            )
            row["local_verified"] = bool(size_matches and hash_matches)
            if not row["local_verified"]:
                warnings.append(
                    f"Checkpoint local différent du manifeste: {local_relative}"
                )
        vm_value = entry.get("vm_path")
        vm_path = Path(str(vm_value)).expanduser() if vm_value else None
        row["vm_present"] = bool(vm_path and vm_path.is_file())
        if vm_path and vm_path.is_file():
            expected_size = entry.get("size_bytes")
            expected_hash = entry.get("sha256")
            size_matches = expected_size == vm_path.stat().st_size
            hash_matches = (
                isinstance(expected_hash, str) and _sha256(vm_path) == expected_hash
            )
            row["vm_verified"] = bool(size_matches and hash_matches)
            if not row["vm_verified"]:
                warnings.append(f"Checkpoint VM différent du manifeste: {vm_path}")
        rows.append(row)
    return path, dict(value), rows


def load_wasserstein_feasibility_manifest(
    warnings: list[str],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Load and flatten the versioned cross-run dense-EMD feasibility scan."""
    repository_root = Path(__file__).resolve().parents[2]
    path = repository_root / WASSERSTEIN_FEASIBILITY_RELATIVE
    if not path.is_file():
        warnings.append(f"Manifeste versionné de faisabilité Wasserstein absent: {path}")
        return path, {}, []
    try:
        value = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Manifeste de faisabilité Wasserstein invalide {path}: {exc}")
        return path, {}, []
    if not isinstance(value, Mapping) or value.get("format_version") != 1:
        warnings.append(
            f"Format de manifeste de faisabilité Wasserstein non pris en charge: {path}"
        )
        return path, {}, []

    run_order_value = value.get("run_order", [])
    run_order = (
        [str(run) for run in run_order_value]
        if isinstance(run_order_value, list)
        else []
    )
    rows: list[dict[str, Any]] = []
    threshold_scan = value.get("threshold_scan", [])
    if not isinstance(threshold_scan, list):
        warnings.append(f"Liste threshold_scan absente ou invalide dans {path}")
        threshold_scan = []
    for threshold in threshold_scan:
        if not isinstance(threshold, Mapping):
            continue
        ideal_range = threshold.get("ideal_wall_time_hours_8_workers_range", [])
        real_range = threshold.get("estimated_real_wall_time_hours_range", [])
        runnable_counts = threshold.get("runnable_counts_by_run_order", [])
        common_by_dataset = threshold.get("strict_common_by_dataset", {})
        row: dict[str, Any] = {
            "per_case_threshold_gib": threshold.get("per_case_threshold_gib"),
            "cases_per_run": value.get("cases_per_run"),
            "strict_common_cases": threshold.get("strict_common_cases"),
            "strict_common_fraction": threshold.get("strict_common_fraction"),
            "union_excluded_cases": threshold.get("union_excluded_cases"),
            "ideal_wall_time_hours_8_workers_low": (
                ideal_range[0]
                if isinstance(ideal_range, list) and len(ideal_range) >= 1
                else None
            ),
            "ideal_wall_time_hours_8_workers_high": (
                ideal_range[1]
                if isinstance(ideal_range, list) and len(ideal_range) >= 2
                else None
            ),
            "estimated_real_wall_time_hours_low": (
                real_range[0]
                if isinstance(real_range, list) and len(real_range) >= 1
                else None
            ),
            "estimated_real_wall_time_hours_high": (
                real_range[1]
                if isinstance(real_range, list) and len(real_range) >= 2
                else None
            ),
        }
        if isinstance(runnable_counts, list):
            for index, run in enumerate(run_order):
                row[f"runnable_{run}"] = (
                    runnable_counts[index] if index < len(runnable_counts) else None
                )
            if len(runnable_counts) != len(run_order):
                warnings.append(
                    "Faisabilité Wasserstein: le nombre de comptes exécutables ne "
                    f"correspond pas à run_order au seuil {threshold.get('per_case_threshold_gib')}."
                )
        if isinstance(common_by_dataset, Mapping):
            for dataset in DATASETS:
                row[f"strict_common_{dataset}"] = common_by_dataset.get(dataset)
        rows.append(row)
    if not rows:
        warnings.append(f"Aucun seuil exploitable dans le manifeste Wasserstein: {path}")
    return path, dict(value), rows


def _safe_relative_case(case_name: str, *, strip_image_dir: bool = False) -> Path:
    relative = Path(case_name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Nom de cas non sûr: {case_name!r}")
    parts = [part for part in relative.parts if part not in ("", ".")]
    if strip_image_dir and parts and parts[0].casefold() in IMAGE_DIRECTORY_NAMES:
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Nom de cas vide: {case_name!r}")
    return Path(*parts)


def _prediction_path(run: RunData, dataset: str, case_name: str) -> Path:
    relative = _safe_relative_case(case_name)
    return run.spec.path / dataset / "predictions" / relative.with_suffix(".png")


def _prompt_cache_dir(prompt_root: Path, dataset: str) -> Path:
    mapping = {
        "khanhha_original": ("khanhha", "test_original"),
        "khanhha_noisy1": ("khanhha", "test_noisy1"),
        "khanhha_noisy2": ("khanhha", "test_noisy2"),
        "road420": ("road420", "test_original"),
        "facade390": ("facade390", "test_original"),
        "concrete3k": ("concrete3k", "test_original"),
    }
    return prompt_root.joinpath(*mapping[dataset])


def _prompt_path(prompt_root: Path, dataset: str, case_name: str) -> Path:
    relative = _safe_relative_case(case_name, strip_image_dir=True)
    return _prompt_cache_dir(prompt_root, dataset) / relative.parent / f"{relative.name}.npy"


def _find_exact_root(run_root: Path) -> Path | None:
    candidates = (
        run_root / "wasserstein_exact",
        run_root / "exact_wasserstein",
        run_root.parent / f"{run_root.name}_wasserstein_exact",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _evaluation_uses_exact_wasserstein(run_root: Path) -> bool:
    """Return true only when the immutable inference contract says exact EMD."""
    contract_path = run_root / "evaluation_contract.json"
    if not contract_path.is_file():
        return False
    try:
        contract = _read_json(contract_path)
    except (OSError, json.JSONDecodeError):
        return False
    wasserstein = contract.get("wasserstein") if isinstance(contract, Mapping) else None
    return bool(
        isinstance(wasserstein, Mapping)
        and wasserstein.get("exact") is True
        and wasserstein.get("skip") is not True
        and wasserstein.get("max_points") is None
    )


def _load_summary_fallback(root: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    summary_csv = root / "summary.csv"
    rows: list[Mapping[str, Any]] = []
    if summary_csv.is_file():
        rows = _read_csv(summary_csv)
    elif (root / "summary.json").is_file():
        value = _read_json(root / "summary.json")
        if isinstance(value, list):
            rows = [row for row in value if isinstance(row, Mapping)]
        elif isinstance(value, Mapping):
            rows = [value]
    for row in rows:
        dataset = str(row.get("dataset", ""))
        if dataset not in DATASETS:
            continue
        normalized = dict(row)
        for metric in METRICS:
            normalized[metric] = _finite_float(row.get(metric))
            normalized[f"{metric}_std"] = _finite_float(row.get(f"{metric}_std"))
            normalized[f"{metric}_finite_samples"] = int(
                _finite_float(row.get(f"{metric}_finite_samples"))
                if math.isfinite(_finite_float(row.get(f"{metric}_finite_samples")))
                else 0
            )
        samples = _finite_float(row.get("samples"))
        normalized["samples"] = int(samples) if math.isfinite(samples) else 0
        summaries[dataset] = normalized
    return summaries


def _load_per_image(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for raw in _read_csv(path):
        case_name = str(raw.get("case_name", ""))
        if not case_name:
            continue
        row: dict[str, Any] = {"case_name": case_name}
        for metric in METRICS:
            row[metric] = _finite_float(raw.get(metric))
        row["inference_seconds"] = _finite_float(raw.get("inference_seconds"))
        rows[case_name] = row
    return rows


def _summary_from_rows(
    dataset: str,
    rows: Mapping[str, Mapping[str, Any]],
    *,
    exact_available: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"dataset": dataset, "samples": len(rows)}
    for metric in METRICS:
        values = _finite(row.get(metric) for row in rows.values())
        summary[metric] = float(np.mean(values)) if values.size else math.nan
        summary[f"{metric}_std"] = float(np.std(values)) if values.size else math.nan
        summary[f"{metric}_finite_samples"] = int(values.size)
    finite_w = summary["wasserstein_finite_samples"]
    summary["wasserstein_exact"] = bool(exact_available and finite_w)
    summary["wasserstein_complete"] = bool(
        exact_available and len(rows) > 0 and finite_w == len(rows)
    )
    summary["wasserstein_missing_samples"] = int(len(rows) - finite_w)
    return summary


def load_run(spec: RunSpec, warnings: list[str]) -> RunData | None:
    if not spec.path.is_dir():
        warnings.append(f"Évaluation absente pour {spec.label}: {spec.path}")
        return None
    run = RunData(spec=spec)
    fallback = _load_summary_fallback(spec.path)
    base_wasserstein_is_exact = _evaluation_uses_exact_wasserstein(spec.path)
    exact_root = _find_exact_root(spec.path)
    run.exact_root = exact_root
    if base_wasserstein_is_exact:
        run.exact_status = "exact intégré à l’évaluation"
    elif exact_root is None:
        run.exact_status = "absent"
    elif (exact_root / "summary.csv").is_file() or (exact_root / "summary.json").is_file():
        run.exact_status = "publié"
    elif (exact_root / "support_scan.json").is_file():
        run.exact_status = "scan seulement / calcul non publié"
    else:
        run.exact_status = "répertoire présent mais résultat non publié"

    exact_fallback = _load_summary_fallback(exact_root) if exact_root else {}
    for dataset in DATASETS:
        base_rows = _load_per_image(spec.path / dataset / "per_image.csv")
        capped_wasserstein_count = 0
        if not base_wasserstein_is_exact:
            capped_wasserstein_count = sum(
                math.isfinite(_finite_float(row.get("wasserstein")))
                for row in base_rows.values()
            )
            # A regular evaluation may contain the deterministic max_points
            # approximation.  It must never enter a column labelled "exact".
            for row in base_rows.values():
                row["wasserstein"] = math.nan
            if dataset in fallback:
                fallback[dataset]["wasserstein"] = math.nan
                fallback[dataset]["wasserstein_std"] = math.nan
                fallback[dataset]["wasserstein_finite_samples"] = 0
                fallback[dataset]["wasserstein_exact"] = False
                fallback[dataset]["wasserstein_complete"] = False
        if capped_wasserstein_count:
            warnings.append(
                f"{spec.key}/{dataset}: {capped_wasserstein_count} distances Wasserstein "
                "plafonnées ignorées; seules les valeurs exactes séparées sont publiées."
            )
        exact_rows = (
            _load_per_image(exact_root / dataset / "per_image.csv")
            if exact_root is not None
            else {}
        )
        if base_rows and exact_rows:
            # Only Wasserstein is imported from the exact run.  The segmentation
            # values remain tied to the original inference CSV.
            for case_name, row in base_rows.items():
                if case_name in exact_rows:
                    row["wasserstein"] = exact_rows[case_name]["wasserstein"]
        run.per_image[dataset] = base_rows
        if base_rows:
            run.summaries[dataset] = _summary_from_rows(
                dataset,
                base_rows,
                exact_available=base_wasserstein_is_exact or bool(exact_rows),
            )
        elif dataset in fallback:
            run.summaries[dataset] = fallback[dataset]
            if dataset in exact_fallback:
                exact = exact_fallback[dataset]
                for key in (
                    "wasserstein",
                    "wasserstein_std",
                    "wasserstein_finite_samples",
                    "wasserstein_complete",
                    "wasserstein_missing_samples",
                    "wasserstein_exact",
                ):
                    if key in exact:
                        run.summaries[dataset][key] = exact[key]
        else:
            warnings.append(
                f"Aucune métrique pour {spec.label} / {DATASET_LABELS[dataset]}."
            )

    if not run.summaries:
        warnings.append(f"Évaluation inutilisable (aucun résumé): {spec.path}")
        return None
    return run


def wasserstein_feasibility_rows(
    runs: Sequence[RunData], warnings: list[str]
) -> list[dict[str, Any]]:
    """Summarize exact dense-EMD feasibility without publishing partial means."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        root = run.exact_root
        row: dict[str, Any] = {
            "run": run.spec.key,
            "epoch": run.spec.epoch,
            "exact_root": str(root) if root else "",
            "exact_status": run.exact_status,
            "scan_available": False,
            "exact_summary_published": False,
        }
        if root is None:
            rows.append(row)
            continue
        support_path = root / "support_scan.json"
        oversized_path = root / "oversized.json"
        contract_path = root / "exact_wasserstein_contract.json"
        progress_path = root / "progress.jsonl"
        summary_path = root / "summary.csv"
        row["exact_summary_published"] = summary_path.is_file()

        support: Mapping[str, Any] = {}
        oversized: Mapping[str, Any] = {}
        contract: Mapping[str, Any] = {}
        for path, label in (
            (support_path, "support_scan"),
            (oversized_path, "oversized"),
            (contract_path, "exact_wasserstein_contract"),
        ):
            if not path.is_file():
                continue
            try:
                value = _read_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append(f"{run.spec.key}: {label} invalide: {exc}")
                continue
            if not isinstance(value, Mapping):
                warnings.append(f"{run.spec.key}: {label} n’est pas un objet JSON")
                continue
            if label == "support_scan":
                support = value
            elif label == "oversized":
                oversized = value
            else:
                contract = value

        estimated = support.get("estimated_gib", {})
        if not isinstance(estimated, Mapping):
            estimated = {}
        tasks = int(_finite_float(support.get("tasks"))) if math.isfinite(
            _finite_float(support.get("tasks"))
        ) else int(_finite_float(contract.get("tasks"))) if math.isfinite(
            _finite_float(contract.get("tasks"))
        ) else 0
        oversized_count = int(_finite_float(oversized.get("count"))) if math.isfinite(
            _finite_float(oversized.get("count"))
        ) else 0
        row.update(
            {
                "scan_available": bool(support),
                "tasks_total": tasks,
                "estimated_gib_p50": _finite_float(estimated.get("p50")),
                "estimated_gib_p99": _finite_float(estimated.get("p99")),
                "estimated_gib_max": _finite_float(estimated.get("max")),
                "memory_budget_gib": (
                    _finite_float(oversized.get("memory_budget_bytes")) / 1024**3
                    if math.isfinite(_finite_float(oversized.get("memory_budget_bytes")))
                    else _finite_float(contract.get("memory_budget_gb"))
                ),
                "oversized_excluded": oversized_count,
                "runnable_tasks": max(0, tasks - oversized_count),
                "empty_prediction": support.get("empty_prediction", ""),
                "empty_target": support.get("empty_target", ""),
                "matrix_bytes_per_entry_estimate": support.get(
                    "matrix_bytes_per_entry_estimate",
                    contract.get("matrix_bytes_per_entry_estimate", ""),
                ),
            }
        )
        oversized_tasks = oversized.get("tasks", [])
        if isinstance(oversized_tasks, list):
            row["oversized_cases"] = "; ".join(
                f"{item.get('dataset')}/{item.get('case_name')}"
                for item in oversized_tasks
                if isinstance(item, Mapping)
            )

        completed: dict[tuple[str, str], Mapping[str, Any]] = {}
        failed = 0
        if progress_path.is_file():
            with progress_path.open("rb") as progress:
                for line_number, raw_line in enumerate(progress, start=1):
                    if not raw_line.strip():
                        continue
                    try:
                        entry = json.loads(raw_line)
                    except json.JSONDecodeError:
                        # A running process may leave only its final line
                        # truncated; prior durable entries remain valid.
                        if line_number > 1 and not raw_line.endswith(b"\n"):
                            break
                        warnings.append(
                            f"{run.spec.key}: ligne de progrès Wasserstein invalide {line_number}"
                        )
                        continue
                    if not isinstance(entry, Mapping):
                        continue
                    if entry.get("status") == "complete":
                        key = (str(entry.get("dataset", "")), str(entry.get("case_name", "")))
                        completed[key] = entry
                    elif entry.get("status") == "failed":
                        failed += 1
        seconds = _finite(entry.get("seconds") for entry in completed.values())
        runnable = int(row.get("runnable_tasks", 0))
        row.update(
            {
                "progress_complete_unique": len(completed),
                "progress_failed_entries": failed,
                "progress_fraction_of_runnable": (
                    len(completed) / runnable if runnable else math.nan
                ),
                "progress_compute_seconds_sum": (
                    float(np.sum(seconds)) if seconds.size else math.nan
                ),
                "progress_task_seconds_p50": (
                    float(np.percentile(seconds, 50)) if seconds.size else math.nan
                ),
                "progress_task_seconds_p99": (
                    float(np.percentile(seconds, 99)) if seconds.size else math.nan
                ),
                "progress_task_seconds_max": (
                    float(np.max(seconds)) if seconds.size else math.nan
                ),
                "progress_journal_bytes": (
                    progress_path.stat().st_size if progress_path.is_file() else 0
                ),
                "progress_wall_elapsed_seconds": (
                    max(0.0, progress_path.stat().st_mtime - contract_path.stat().st_mtime)
                    if progress_path.is_file() and contract_path.is_file()
                    else math.nan
                ),
            }
        )
        recorded_contract_hash = contract.get("evaluation_contract_sha256")
        evaluation_contract = run.spec.path / "evaluation_contract.json"
        observed_contract_hash = (
            _sha256(evaluation_contract) if evaluation_contract.is_file() else None
        )
        row["evaluation_contract_sha256_recorded"] = recorded_contract_hash
        row["evaluation_contract_sha256_observed"] = observed_contract_hash
        row["evaluation_contract_matches"] = bool(
            recorded_contract_hash
            and observed_contract_hash
            and recorded_contract_hash == observed_contract_hash
        )
        rows.append(row)
    return rows


def copy_wasserstein_audit(
    runs: Sequence[RunData], output: Path, warnings: list[str]
) -> list[dict[str, Any]]:
    """Archive the exact-run feasibility evidence without deriving partial means."""
    audit_root = output / "wasserstein_audit"
    rows: list[dict[str, Any]] = []
    required_names = (
        "support_scan.json",
        "oversized.json",
        "exact_wasserstein_contract.json",
    )
    for run in runs:
        if run.exact_root is None:
            continue
        names = list(required_names)
        if run.spec.key == "baseline_best":
            names.append("progress.jsonl")
        destination_root = audit_root / run.spec.key
        destination_root.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = run.exact_root / name
            row: dict[str, Any] = {
                "run": run.spec.key,
                "artifact": name,
                "source_path": str(source),
                "available": source.is_file(),
                "contains_partial_metric_values": name == "progress.jsonl",
                "partial_mean_published": False,
            }
            destination = destination_root / name
            if not source.is_file():
                destination.unlink(missing_ok=True)
                warnings.append(
                    f"Audit Wasserstein absent pour {run.spec.key}: {source}"
                )
                rows.append(row)
                continue
            shutil.copy2(source, destination)
            source_hash = _sha256(source)
            destination_record = _file_record(destination, relative_to=output)
            row.update(
                {
                    "source_size_bytes": source.stat().st_size,
                    "source_sha256": source_hash,
                    "copied_path": destination_record["path"],
                    "copied_size_bytes": destination_record["size_bytes"],
                    "copied_sha256": destination_record["sha256"],
                    "copy_verified": source_hash == destination_record["sha256"],
                }
            )
            if not row["copy_verified"]:
                warnings.append(
                    f"Copie d’audit Wasserstein non vérifiée pour {run.spec.key}/{name}"
                )
            rows.append(row)
    return rows


def resolve_run_specs(artifact_root: Path) -> list[RunSpec]:
    baseline_path = artifact_root / "baseline_r4" / "final_evaluation"
    specs = [
        RunSpec(
            key="baseline_best",
            label="Baseline SAM 2 + LoRA r=4 — meilleure époque validation",
            variant="baseline",
            epoch=None,
            path=baseline_path,
        )
    ]
    comparison_root = artifact_root / "frangi_r4" / "milestone_comparison"
    frangi_root = artifact_root / "frangi_r4"
    for key, epoch, label in MILESTONES:
        path = comparison_root / key
        if not path.is_dir():
            legacy = frangi_root / f"milestone_epoch{epoch}_evaluation"
            if legacy.is_dir():
                path = legacy
        specs.append(RunSpec(key, label, "frangi", epoch, path))
    return specs


def aggregate_summary(run: RunData, mode: str) -> dict[str, Any]:
    if mode not in ("macro", "weighted"):
        raise ValueError(mode)
    result: dict[str, Any] = {
        "dataset": "MACRO_6_DATASETS" if mode == "macro" else "PONDERE_IMAGES",
        "samples": sum(int(run.summaries.get(ds, {}).get("samples", 0)) for ds in DATASETS),
    }
    for metric in METRICS:
        if mode == "macro":
            values = _finite(run.summaries.get(ds, {}).get(metric) for ds in DATASETS)
            result[metric] = float(np.mean(values)) if values.size else math.nan
            result[f"{metric}_std"] = float(np.std(values)) if values.size else math.nan
            result[f"{metric}_finite_samples"] = sum(
                int(run.summaries.get(ds, {}).get(f"{metric}_finite_samples", 0))
                for ds in DATASETS
            )
        else:
            values: list[np.ndarray] = []
            for ds in DATASETS:
                if run.per_image.get(ds):
                    current = _finite(row.get(metric) for row in run.per_image[ds].values())
                    if current.size:
                        values.append(current)
            if values:
                joined = np.concatenate(values)
                result[metric] = float(np.mean(joined))
                result[f"{metric}_std"] = float(np.std(joined))
                result[f"{metric}_finite_samples"] = int(joined.size)
            else:
                numerator = 0.0
                denominator = 0
                for ds in DATASETS:
                    summary = run.summaries.get(ds, {})
                    value = _finite_float(summary.get(metric))
                    count = int(summary.get(f"{metric}_finite_samples", 0))
                    if math.isfinite(value) and count:
                        numerator += value * count
                        denominator += count
                result[metric] = numerator / denominator if denominator else math.nan
                result[f"{metric}_std"] = math.nan
                result[f"{metric}_finite_samples"] = denominator
    result["wasserstein_exact"] = any(
        bool(run.summaries.get(ds, {}).get("wasserstein_exact")) for ds in DATASETS
    )
    result["wasserstein_complete"] = all(
        bool(run.summaries.get(ds, {}).get("wasserstein_complete"))
        for ds in DATASETS
        if ds in run.summaries
    ) and len(run.summaries) == len(DATASETS)
    result["wasserstein_missing_samples"] = (
        int(result["samples"]) - int(result["wasserstein_finite_samples"])
    )
    return result


def metric_summary_rows(runs: Sequence[RunData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        summaries = [run.summaries[ds] for ds in DATASETS if ds in run.summaries]
        summaries.extend((aggregate_summary(run, "macro"), aggregate_summary(run, "weighted")))
        for summary in summaries:
            row: dict[str, Any] = {
                "run": run.spec.key,
                "label": run.spec.label,
                "variant": run.spec.variant,
                "epoch": run.spec.epoch,
                "evaluation_root": str(run.spec.path),
                "dataset": summary["dataset"],
                "samples": summary.get("samples", 0),
                "wasserstein_source": run.exact_status,
            }
            for metric in METRICS:
                row[metric] = summary.get(metric, math.nan)
                row[f"{metric}_std"] = summary.get(f"{metric}_std", math.nan)
                row[f"{metric}_finite_samples"] = summary.get(
                    f"{metric}_finite_samples", 0
                )
            row["wasserstein_complete"] = summary.get("wasserstein_complete", False)
            row["wasserstein_missing_samples"] = summary.get(
                "wasserstein_missing_samples", summary.get("samples", 0)
            )
            rows.append(row)
    return rows


def milestone_delta_rows(
    baseline: RunData | None, frangi_runs: Sequence[RunData]
) -> list[dict[str, Any]]:
    if baseline is None:
        return []
    rows: list[dict[str, Any]] = []
    for frangi in frangi_runs:
        wasserstein_pairs: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        for dataset in DATASETS:
            baseline_cases = baseline.per_image.get(dataset, {})
            frangi_cases = frangi.per_image.get(dataset, {})
            common_names = sorted(set(baseline_cases) & set(frangi_cases))
            finite_pairs = [
                (
                    _finite_float(baseline_cases[name].get("wasserstein")),
                    _finite_float(frangi_cases[name].get("wasserstein")),
                )
                for name in common_names
                if math.isfinite(
                    _finite_float(baseline_cases[name].get("wasserstein"))
                )
                and math.isfinite(
                    _finite_float(frangi_cases[name].get("wasserstein"))
                )
            ]
            if finite_pairs:
                pair_array = np.asarray(finite_pairs, dtype=np.float64)
                base_values = pair_array[:, 0]
                frangi_values = pair_array[:, 1]
            else:
                base_values = np.empty(0, dtype=np.float64)
                frangi_values = np.empty(0, dtype=np.float64)
            wasserstein_pairs[dataset] = (
                base_values,
                frangi_values,
                len(common_names),
            )

        pairs = [
            (ds, baseline.summaries.get(ds), frangi.summaries.get(ds))
            for ds in DATASETS
        ]
        pairs.append(
            (
                "MACRO_6_DATASETS",
                aggregate_summary(baseline, "macro"),
                aggregate_summary(frangi, "macro"),
            )
        )
        pairs.append(
            (
                "PONDERE_IMAGES",
                aggregate_summary(baseline, "weighted"),
                aggregate_summary(frangi, "weighted"),
            )
        )
        for dataset, baseline_summary, frangi_summary in pairs:
            if baseline_summary is None or frangi_summary is None:
                continue
            row: dict[str, Any] = {
                "milestone": frangi.spec.key,
                "epoch": frangi.spec.epoch,
                "dataset": dataset,
                "baseline_samples": baseline_summary.get("samples", 0),
                "frangi_samples": frangi_summary.get("samples", 0),
            }
            for metric in METRICS:
                base = _finite_float(baseline_summary.get(metric))
                new = _finite_float(frangi_summary.get(metric))
                row[f"baseline_{metric}"] = base
                row[f"frangi_{metric}"] = new
                row[f"delta_{metric}_frangi_minus_baseline"] = (
                    new - base if math.isfinite(base) and math.isfinite(new) else math.nan
                )
                row[f"improvement_{metric}"] = (
                    (new - base if HIGHER_IS_BETTER[metric] else base - new)
                    if math.isfinite(base) and math.isfinite(new)
                    else math.nan
                )

            # Wasserstein must be paired.  Replace the independent summary
            # means above with means over the strict common finite support.
            if dataset in DATASETS:
                base_w, frangi_w, total_w = wasserstein_pairs[dataset]
                covered_datasets = int(bool(base_w.size))
            else:
                available = [
                    (base_w, frangi_w)
                    for base_w, frangi_w, _ in wasserstein_pairs.values()
                    if base_w.size
                ]
                total_w = sum(item[2] for item in wasserstein_pairs.values())
                covered_datasets = len(available)
                if dataset == "PONDERE_IMAGES" and available:
                    base_w = np.concatenate([item[0] for item in available])
                    frangi_w = np.concatenate([item[1] for item in available])
                elif dataset == "MACRO_6_DATASETS" and available:
                    base_w = np.asarray([np.mean(item[0]) for item in available])
                    frangi_w = np.asarray([np.mean(item[1]) for item in available])
                else:
                    base_w = np.empty(0, dtype=np.float64)
                    frangi_w = np.empty(0, dtype=np.float64)
            base_w_mean = float(np.mean(base_w)) if base_w.size else math.nan
            frangi_w_mean = float(np.mean(frangi_w)) if frangi_w.size else math.nan
            row["baseline_wasserstein"] = base_w_mean
            row["frangi_wasserstein"] = frangi_w_mean
            row["delta_wasserstein_frangi_minus_baseline"] = (
                frangi_w_mean - base_w_mean
                if math.isfinite(base_w_mean) and math.isfinite(frangi_w_mean)
                else math.nan
            )
            row["improvement_wasserstein"] = (
                base_w_mean - frangi_w_mean
                if math.isfinite(base_w_mean) and math.isfinite(frangi_w_mean)
                else math.nan
            )
            common_w = int(base_w.size)
            if dataset == "MACRO_6_DATASETS" and available:
                common_w = sum(item[0].size for item in available)
            row["wasserstein_common_finite"] = common_w
            row["wasserstein_total_pairs"] = total_w
            row["wasserstein_common_coverage"] = (
                common_w / total_w if total_w else math.nan
            )
            row["wasserstein_datasets_covered"] = covered_datasets
            row["wasserstein_datasets_total"] = 1 if dataset in DATASETS else len(DATASETS)
            rows.append(row)
    return rows


def per_image_comparison_rows(
    baseline: RunData | None, frangi_runs: Sequence[RunData]
) -> list[dict[str, Any]]:
    if baseline is None:
        return []
    rows: list[dict[str, Any]] = []
    for frangi in frangi_runs:
        for dataset in DATASETS:
            baseline_cases = baseline.per_image.get(dataset, {})
            frangi_cases = frangi.per_image.get(dataset, {})
            for case_name in sorted(set(baseline_cases) & set(frangi_cases)):
                base = baseline_cases[case_name]
                new = frangi_cases[case_name]
                row: dict[str, Any] = {
                    "milestone": frangi.spec.key,
                    "epoch": frangi.spec.epoch,
                    "dataset": dataset,
                    "case_name": case_name,
                }
                for metric in METRICS:
                    base_value = _finite_float(base.get(metric))
                    new_value = _finite_float(new.get(metric))
                    delta = (
                        new_value - base_value
                        if math.isfinite(base_value) and math.isfinite(new_value)
                        else math.nan
                    )
                    row[f"baseline_{metric}"] = base_value
                    row[f"frangi_{metric}"] = new_value
                    row[f"delta_{metric}_frangi_minus_baseline"] = delta
                    row[f"improvement_{metric}"] = (
                        delta if HIGHER_IS_BETTER[metric] else -delta
                    )
                rows.append(row)
    return rows


def _bootstrap_samples(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bootstrap means and medians without allocating a huge index cube."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.empty(0), np.empty(0)
    if values.size == 1:
        return (
            np.full(repetitions, values[0], dtype=np.float64),
            np.full(repetitions, values[0], dtype=np.float64),
        )
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    medians = np.empty(repetitions, dtype=np.float64)
    # Bound temporary sampled values to roughly 32 MiB.
    batch_size = max(1, min(repetitions, (32 * 1024**2) // (8 * values.size)))
    offset = 0
    while offset < repetitions:
        count = min(batch_size, repetitions - offset)
        indices = rng.integers(0, values.size, size=(count, values.size))
        sampled = values[indices]
        means[offset : offset + count] = np.mean(sampled, axis=1)
        medians[offset : offset + count] = np.median(sampled, axis=1)
        offset += count
    return means, medians


def _ci95(samples: np.ndarray) -> tuple[float, float]:
    if not samples.size:
        return math.nan, math.nan
    low, high = np.percentile(samples, (2.5, 97.5))
    return float(low), float(high)


def _paired_row(
    *,
    metric: str,
    dataset: str,
    aggregation: str,
    deltas: np.ndarray,
    total_pairs: int,
    baseline_finite: int,
    frangi_finite: int,
    bootstrap_means: np.ndarray,
    bootstrap_medians: np.ndarray,
    tie_tolerance: float,
) -> dict[str, Any]:
    deltas = np.asarray(deltas, dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    improvements = deltas if HIGHER_IS_BETTER[metric] else -deltas
    wins = int(np.count_nonzero(improvements > tie_tolerance))
    losses = int(np.count_nonzero(improvements < -tie_tolerance))
    ties = int(improvements.size - wins - losses)
    mean_low, mean_high = _ci95(bootstrap_means)
    median_low, median_high = _ci95(bootstrap_medians)
    return {
        "metric": metric,
        "dataset": dataset,
        "aggregation": aggregation,
        "direction_better": "higher" if HIGHER_IS_BETTER[metric] else "lower",
        "total_paired_cases": total_pairs,
        "baseline_finite": baseline_finite,
        "frangi_finite": frangi_finite,
        "common_finite": int(deltas.size),
        "common_coverage": float(deltas.size / total_pairs) if total_pairs else math.nan,
        "mean_delta_frangi_minus_baseline": (
            float(np.mean(deltas)) if deltas.size else math.nan
        ),
        "mean_delta_ci95_low": mean_low,
        "mean_delta_ci95_high": mean_high,
        "median_delta_frangi_minus_baseline": (
            float(np.median(deltas)) if deltas.size else math.nan
        ),
        "median_delta_ci95_low": median_low,
        "median_delta_ci95_high": median_high,
        "mean_improvement_positive_is_better": (
            float(np.mean(improvements)) if improvements.size else math.nan
        ),
        "wins_frangi": wins,
        "ties": ties,
        "losses_frangi": losses,
        "win_rate": float(wins / improvements.size) if improvements.size else math.nan,
        "tie_rate": float(ties / improvements.size) if improvements.size else math.nan,
        "loss_rate": float(losses / improvements.size) if improvements.size else math.nan,
        "tie_tolerance": tie_tolerance,
    }


def paired_statistics(
    baseline: RunData | None,
    primary: RunData | None,
    *,
    repetitions: int,
    base_seed: int,
    tie_tolerance: float,
) -> list[dict[str, Any]]:
    if baseline is None or primary is None:
        return []
    rows: list[dict[str, Any]] = []
    for metric in ("iou", "wasserstein"):
        groups: list[tuple[str, np.ndarray]] = []
        coverage: list[tuple[int, int, int]] = []
        for dataset in DATASETS:
            baseline_cases = baseline.per_image.get(dataset, {})
            primary_cases = primary.per_image.get(dataset, {})
            common_names = sorted(set(baseline_cases) & set(primary_cases))
            baseline_finite = sum(
                math.isfinite(_finite_float(row.get(metric)))
                for row in baseline_cases.values()
            )
            frangi_finite = sum(
                math.isfinite(_finite_float(row.get(metric)))
                for row in primary_cases.values()
            )
            deltas = np.asarray(
                [
                    _finite_float(primary_cases[name].get(metric))
                    - _finite_float(baseline_cases[name].get(metric))
                    for name in common_names
                    if math.isfinite(_finite_float(primary_cases[name].get(metric)))
                    and math.isfinite(_finite_float(baseline_cases[name].get(metric)))
                ],
                dtype=np.float64,
            )
            coverage.append((len(common_names), baseline_finite, frangi_finite))
            if deltas.size:
                means, medians = _bootstrap_samples(
                    deltas,
                    repetitions=repetitions,
                    seed=_stable_seed(base_seed, primary.spec.key, metric, dataset),
                )
                groups.append((dataset, deltas))
            else:
                means, medians = np.empty(0), np.empty(0)
            dataset_row = _paired_row(
                metric=metric,
                dataset=dataset,
                aggregation="dataset",
                deltas=deltas,
                total_pairs=len(common_names),
                baseline_finite=baseline_finite,
                frangi_finite=frangi_finite,
                bootstrap_means=means,
                bootstrap_medians=medians,
                tie_tolerance=tie_tolerance,
            )
            dataset_row["datasets_with_common_finite"] = int(bool(deltas.size))
            dataset_row["datasets_total"] = 1
            rows.append(dataset_row)

        delta_arrays = [group for _, group in groups]
        all_deltas = (
            np.concatenate(delta_arrays) if delta_arrays else np.empty(0, dtype=np.float64)
        )
        if all_deltas.size:
            means, medians = _bootstrap_samples(
                all_deltas,
                repetitions=repetitions,
                seed=_stable_seed(base_seed, primary.spec.key, metric, "weighted"),
            )
        else:
            means, medians = np.empty(0), np.empty(0)
        weighted_row = _paired_row(
            metric=metric,
            dataset="PONDERE_IMAGES",
            aggregation="weighted cases",
            deltas=all_deltas,
            total_pairs=sum(item[0] for item in coverage),
            baseline_finite=sum(item[1] for item in coverage),
            frangi_finite=sum(item[2] for item in coverage),
            bootstrap_means=means,
            bootstrap_medians=medians,
            tie_tolerance=tie_tolerance,
        )
        weighted_row["datasets_with_common_finite"] = len(groups)
        weighted_row["datasets_total"] = len(DATASETS)
        rows.append(weighted_row)

        if groups:
            # Stratified bootstrap: resample within each available dataset,
            # then average the dataset statistics.  Datasets with zero common
            # exact-Wasserstein values remain visible in coverage but cannot
            # contribute a numeric statistic.
            group_means: list[np.ndarray] = []
            group_medians: list[np.ndarray] = []
            for dataset, group in groups:
                boot_mean, boot_median = _bootstrap_samples(
                    group,
                    repetitions=repetitions,
                    seed=_stable_seed(
                        base_seed, primary.spec.key, metric, "macro", dataset
                    ),
                )
                group_means.append(boot_mean)
                group_medians.append(boot_median)
            macro_boot_means = np.mean(np.stack(group_means), axis=0)
            macro_boot_medians = np.median(np.stack(group_medians), axis=0)
        else:
            macro_boot_means = np.empty(0)
            macro_boot_medians = np.empty(0)
        macro_row = _paired_row(
            metric=metric,
            dataset="MACRO_6_DATASETS",
            aggregation="macro available datasets (stratified bootstrap)",
            deltas=all_deltas,
            total_pairs=sum(item[0] for item in coverage),
            baseline_finite=sum(item[1] for item in coverage),
            frangi_finite=sum(item[2] for item in coverage),
            bootstrap_means=macro_boot_means,
            bootstrap_medians=macro_boot_medians,
            tie_tolerance=tie_tolerance,
        )
        if groups:
            macro_row["mean_delta_frangi_minus_baseline"] = float(
                np.mean([np.mean(group) for _, group in groups])
            )
            macro_row["median_delta_frangi_minus_baseline"] = float(
                np.median([np.median(group) for _, group in groups])
            )
            macro_row["mean_improvement_positive_is_better"] = (
                macro_row["mean_delta_frangi_minus_baseline"]
                if HIGHER_IS_BETTER[metric]
                else -macro_row["mean_delta_frangi_minus_baseline"]
            )
        macro_row["datasets_with_common_finite"] = len(groups)
        macro_row["datasets_total"] = len(DATASETS)
        rows.append(macro_row)
    return rows


def _target_pixel_count(data_root: Path, dataset: str, case_name: str) -> int:
    from cracksam2.data import resolve_sample_paths

    _, mask_path = resolve_sample_paths(
        _dataset_root(data_root, dataset), case_name, split="test_vol"
    )
    with Image.open(mask_path) as mask_file:
        raw = np.asarray(mask_file)
        if (
            raw.ndim == 3
            and raw.shape[2] == 4
            and raw[..., 3].min() != raw[..., 3].max()
        ):
            gray = raw[..., 3]
        else:
            gray = np.asarray(mask_file.convert("L"))
    target = np.asarray(gray, dtype=np.float32) / 255.0 > 0.5
    if target.shape != (448, 448):
        target = np.asarray(
            Image.fromarray(target.astype(np.uint8)).resize(
                (448, 448), resample=Image.Resampling.NEAREST
            )
        ).astype(bool)
    return int(np.count_nonzero(target))


def select_cases(
    baseline: RunData | None,
    primary: RunData | None,
    *,
    data_root: Path,
    warnings: list[str],
) -> list[SelectedCase]:
    if baseline is None or primary is None:
        return []
    selected: list[SelectedCase] = []
    pixel_cache: dict[tuple[str, str], int] = {}
    for dataset in DATASETS:
        baseline_cases = baseline.per_image.get(dataset, {})
        frangi_cases = primary.per_image.get(dataset, {})
        candidates: list[tuple[str, float, float, float, int | None]] = []
        mask_errors = 0
        for case_name in sorted(set(baseline_cases) & set(frangi_cases)):
            base = _finite_float(baseline_cases[case_name].get("iou"))
            new = _finite_float(frangi_cases[case_name].get("iou"))
            if math.isfinite(base) and math.isfinite(new):
                # The three Khanhha conditions share targets, so cache by the
                # physical dataset root and case rather than by noise variant.
                cache_dataset = "khanhha" if dataset.startswith("khanhha_") else dataset
                cache_key = (cache_dataset, case_name)
                try:
                    if cache_key not in pixel_cache:
                        pixel_cache[cache_key] = _target_pixel_count(
                            data_root, dataset, case_name
                        )
                    target_pixels: int | None = pixel_cache[cache_key]
                except Exception:
                    target_pixels = None
                    mask_errors += 1
                candidates.append((case_name, base, new, new - base, target_pixels))
        if not candidates:
            continue
        if mask_errors:
            warnings.append(
                f"{dataset}: cible illisible pour {mask_errors}/{len(candidates)} cas "
                "pendant la sélection qualitative."
            )
        substantial = [item for item in candidates if item[4] is not None and item[4] > 32]
        if not substantial:
            substantial = candidates
            warnings.append(
                f"{dataset}: aucun cas à GT >32 pixels lisible; sélection numérique de repli."
            )
        deltas = np.asarray([item[3] for item in substantial])
        median_delta = float(np.median(deltas))
        rankings: dict[str, list[tuple[str, float, float, float, int | None]]] = {
            "gain_frangi": sorted(substantial, key=lambda x: (-x[3], x[0])),
            "gain_baseline": sorted(substantial, key=lambda x: (x[3], x[0])),
            "both_good": sorted(substantial, key=lambda x: (-min(x[1], x[2]), x[0])),
            "both_weak": sorted(substantial, key=lambda x: (max(x[1], x[2]), x[0])),
            "median": sorted(
                substantial, key=lambda x: (abs(x[3] - median_delta), x[0])
            ),
        }
        already_used: set[str] = set()
        for category in SELECTION_CATEGORIES:
            choice = next(
                (item for item in rankings[category] if item[0] not in already_used),
                rankings[category][0],
            )
            already_used.add(choice[0])
            selected.append(
                SelectedCase(
                    dataset,
                    category,
                    choice[0],
                    choice[1],
                    choice[2],
                    choice[3],
                    choice[4],
                )
            )
        sparse = [
            item
            for item in candidates
            if item[4] is not None and item[4] <= 32 and item[0] not in already_used
        ]
        if sparse:
            choice = max(sparse, key=lambda item: (abs(item[3]), item[0]))
            selected.append(
                SelectedCase(
                    dataset,
                    "sparse_divergent",
                    choice[0],
                    choice[1],
                    choice[2],
                    choice[3],
                    choice[4],
                )
            )
    return selected


def _load_prediction(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    if array.shape != (448, 448):
        array = np.asarray(
            Image.fromarray(array).resize((448, 448), resample=Image.Resampling.NEAREST)
        )
    return array > 127


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _load_prompt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    logits = np.squeeze(logits)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError(f"Prompt invalide {path}: shape={logits.shape}")
    probability = _sigmoid(logits)
    if probability.shape != (448, 448):
        probability = np.asarray(
            Image.fromarray(probability, mode="F").resize(
                (448, 448), resample=Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
    return logits, probability


def _dataset_root(data_root: Path, dataset: str) -> Path:
    if dataset.startswith("khanhha_"):
        return data_root / "khanhha" / "test"
    return data_root / dataset


def _load_input_and_target(
    data_root: Path, dataset: str, case_name: str
) -> tuple[np.ndarray, np.ndarray]:
    # Import only for qualitative generation so metric-only reports still work
    # in minimal environments where OpenCV/torch data dependencies are absent.
    from cracksam2.data import apply_noise_perturbation, resolve_sample_paths

    image_path, mask_path = resolve_sample_paths(
        _dataset_root(data_root, dataset), case_name, split="test_vol"
    )
    with Image.open(image_path) as image_file:
        image = np.asarray(image_file.convert("RGB"), dtype=np.uint8).copy()
    noise = {
        "khanhha_noisy1": "noisy1",
        "khanhha_noisy2": "noisy2",
    }.get(dataset, "original")
    image = apply_noise_perturbation(image, noise, output_size=(448, 448))
    if image.shape[:2] != (448, 448):
        image = np.asarray(
            Image.fromarray(image).resize((448, 448), resample=Image.Resampling.BICUBIC)
        )
    with Image.open(mask_path) as mask_file:
        raw = np.asarray(mask_file)
        if (
            raw.ndim == 3
            and raw.shape[2] == 4
            and raw[..., 3].min() != raw[..., 3].max()
        ):
            gray = raw[..., 3]
        else:
            gray = np.asarray(mask_file.convert("L"))
    target = np.asarray(gray, dtype=np.float32) / 255.0 > 0.5
    if target.shape != (448, 448):
        target = np.asarray(
            Image.fromarray(target.astype(np.uint8)).resize(
                (448, 448), resample=Image.Resampling.NEAREST
            )
        ).astype(bool)
    return image, target


def _error_rgb(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = np.zeros((*target.shape, 3), dtype=np.uint8)
    result[np.logical_and(prediction, target)] = (40, 190, 70)  # TP: green
    result[np.logical_and(prediction, ~target)] = (235, 60, 55)  # FP: red
    result[np.logical_and(~prediction, target)] = (45, 200, 235)  # FN: cyan
    return result


def generate_case_panels(
    selections: Sequence[SelectedCase],
    *,
    baseline: RunData,
    primary: RunData,
    data_root: Path,
    prompt_root: Path,
    output: Path,
    dpi: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selection in selections:
        panel_path = (
            output
            / "figures"
            / "cases"
            / selection.dataset
            / f"{selection.category}__{_slug(selection.case_name)}.jpg"
        )
        row: dict[str, Any] = {
            "dataset": selection.dataset,
            "category": selection.category,
            "category_label": CATEGORY_LABELS[selection.category],
            "case_name": selection.case_name,
            "baseline_iou_csv": selection.baseline_iou,
            "frangi_iou_csv": selection.frangi_iou,
            "delta_iou": selection.delta_iou,
            "target_pixels_at_selection": selection.target_pixels,
            "panel": panel_path.relative_to(output).as_posix(),
            "generated": False,
        }
        try:
            image, target = _load_input_and_target(
                data_root, selection.dataset, selection.case_name
            )
            baseline_path = _prediction_path(baseline, selection.dataset, selection.case_name)
            frangi_path = _prediction_path(primary, selection.dataset, selection.case_name)
            prompt_path = _prompt_path(prompt_root, selection.dataset, selection.case_name)
            baseline_prediction = _load_prediction(baseline_path)
            frangi_prediction = _load_prediction(frangi_path)
            _, prompt_probability = _load_prompt(prompt_path)
            row.update(
                {
                    "baseline_prediction": str(baseline_path),
                    "frangi_prediction": str(frangi_path),
                    "prompt_source": str(prompt_path),
                    "target_pixels": int(np.count_nonzero(target)),
                    "baseline_prediction_pixels": int(np.count_nonzero(baseline_prediction)),
                    "frangi_prediction_pixels": int(np.count_nonzero(frangi_prediction)),
                }
            )
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            figure, axes = plt.subplots(1, 8, figsize=(18, 3.0), constrained_layout=False)
            axes[0].imshow(image)
            axes[0].set_title("Entrée réelle", fontsize=8)
            axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Vérité terrain", fontsize=8)
            axes[2].imshow(np.zeros_like(target), cmap="gray", vmin=0, vmax=1)
            axes[2].text(
                0.5,
                0.5,
                "∅\nmask_input=None",
                transform=axes[2].transAxes,
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
            axes[2].set_title("Prompt baseline\n(absent)", fontsize=8)
            axes[3].imshow(prompt_probability, cmap="magma", vmin=0, vmax=1)
            axes[3].set_title("Prompt Frangi\nsigmoid(logits)", fontsize=8)
            axes[4].imshow(baseline_prediction, cmap="gray", vmin=0, vmax=1)
            axes[4].set_title(f"Préd. baseline\nIoU {selection.baseline_iou:.3f}", fontsize=8)
            axes[5].imshow(_error_rgb(baseline_prediction, target))
            axes[5].set_title("Erreurs baseline", fontsize=8)
            axes[6].imshow(frangi_prediction, cmap="gray", vmin=0, vmax=1)
            axes[6].set_title(f"Préd. Frangi\nIoU {selection.frangi_iou:.3f}", fontsize=8)
            axes[7].imshow(_error_rgb(frangi_prediction, target))
            axes[7].set_title("Erreurs Frangi", fontsize=8)
            for axis in axes:
                axis.axis("off")
            figure.suptitle(
                f"{DATASET_LABELS[selection.dataset]} — "
                f"{CATEGORY_LABELS[selection.category]} — ΔIoU={selection.delta_iou:+.4f}\n"
                f"{_display_case_name(selection.case_name, width=55)}",
                fontsize=9,
                linespacing=1.15,
            )
            figure.text(
                0.5,
                0.015,
                "Carte d’erreur : vrai positif = vert · faux positif = rouge · faux négatif = cyan",
                ha="center",
                fontsize=8,
            )
            figure.subplots_adjust(left=0.01, right=0.995, top=0.72, bottom=0.12, wspace=0.05)
            figure.savefig(
                panel_path,
                dpi=dpi,
                format="jpeg",
                pil_kwargs={"quality": 88, "optimize": True, "progressive": True},
            )
            plt.close(figure)
            row["generated"] = True
        except Exception as exc:  # qualitative assets must not erase numeric report
            warnings.append(
                f"Panneau non généré pour {selection.dataset}/{selection.case_name}: {exc}"
            )
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def copy_prompt_examples(
    selections: Sequence[SelectedCase],
    *,
    prompt_root: Path,
    output: Path,
    requested: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if requested <= 0:
        return []
    # Two contrasting examples per dataset are considered first, then the
    # remaining categories fill any holes caused by missing prompt files.
    priority = {
        "gain_frangi": 0,
        "gain_baseline": 1,
        "median": 2,
        "both_good": 3,
        "both_weak": 4,
        "sparse_divergent": 5,
    }
    ordered = sorted(
        selections,
        key=lambda item: (priority[item.category], DATASETS.index(item.dataset), item.case_name),
    )
    manifest: list[dict[str, Any]] = []
    used_sources: set[Path] = set()
    for selection in ordered:
        if len(manifest) >= requested:
            break
        source = _prompt_path(prompt_root, selection.dataset, selection.case_name)
        if source in used_sources or not source.is_file():
            if not source.is_file():
                warnings.append(f"Prompt source absent: {source}")
            continue
        used_sources.add(source)
        try:
            logits = np.asarray(np.load(source, allow_pickle=False), dtype=np.float32)
            squeezed = np.squeeze(logits)
            if squeezed.ndim != 2 or not np.isfinite(squeezed).all():
                raise ValueError(f"shape={logits.shape}, valeurs finies requises")
            probability = _sigmoid(squeezed)
            destination = (
                output
                / "prompts_npy"
                / selection.dataset
                / f"{selection.category}__{_slug(selection.case_name)}.npy"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest.append(
                {
                    "index": len(manifest) + 1,
                    "dataset": selection.dataset,
                    "category": selection.category,
                    "case_name": selection.case_name,
                    "delta_iou": selection.delta_iou,
                    "source": str(source),
                    "copied_path": destination.relative_to(output).as_posix(),
                    "sha256": _sha256(destination),
                    "dtype": str(logits.dtype),
                    "shape": list(logits.shape),
                    "logit_min": float(np.min(squeezed)),
                    "logit_max": float(np.max(squeezed)),
                    "logit_mean": float(np.mean(squeezed)),
                    "probability_min": float(np.min(probability)),
                    "probability_max": float(np.max(probability)),
                    "probability_mean": float(np.mean(probability)),
                    "probability_nonzero_0_5": float(np.mean(probability >= 0.5)),
                }
            )
        except Exception as exc:
            warnings.append(f"Prompt non copié {source}: {exc}")
    if len(manifest) < requested:
        warnings.append(
            f"Seulement {len(manifest)}/{requested} prompts ont pu être copiés."
        )
    _write_json(
        output / "prompts_npy" / "manifest.json",
        {
            "format_version": 1,
            "description": (
                "Pseudo-logits Frangi réellement injectés comme mask_input; "
                "la visualisation applique sigmoid mais les .npy restent inchangés."
            ),
            "requested": requested,
            "copied": len(manifest),
            "files": manifest,
        },
    )
    _write_csv(output / "tables" / "prompt_manifest.csv", manifest)
    return manifest


def plot_prompt_gallery(
    manifest: Sequence[Mapping[str, Any]], output: Path, warnings: list[str]
) -> Path | None:
    if not manifest:
        return None
    columns = 4
    rows_count = math.ceil(len(manifest) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(13.5, 3.2 * rows_count),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    try:
        for axis, entry in zip(axes_array, manifest):
            path = output / str(entry["copied_path"])
            logits = np.squeeze(np.load(path, allow_pickle=False))
            probability = _sigmoid(logits)
            axis.imshow(probability, cmap="magma", vmin=0, vmax=1)
            axis.set_title(
                f"{DATASET_SHORT_LABELS[str(entry['dataset'])]} · "
                f"{CATEGORY_LABELS[str(entry['category'])]}\n"
                f"{_display_case_name(str(entry['case_name']), width=27)}\n"
                f"ΔIoU={float(entry['delta_iou']):+.3f}",
                fontsize=7,
                linespacing=1.15,
                pad=5,
            )
            axis.axis("off")
        for axis in axes_array[len(manifest) :]:
            axis.axis("off")
        figure.suptitle(
            "Galerie des prompts Frangi-similarité — probabilité sigmoid(pseudo-logits)",
            fontsize=12,
        )
        path = output / "figures" / "prompt_gallery.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            path,
            dpi=140,
            format="jpeg",
            pil_kwargs={"quality": 90, "optimize": True, "progressive": True},
        )
        return path
    except Exception as exc:
        warnings.append(f"Galerie des prompts non générée: {exc}")
        return None
    finally:
        plt.close(figure)


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_milestone_evolution(
    baseline: RunData | None, frangi_runs: Sequence[RunData], output: Path
) -> Path | None:
    if not frangi_runs:
        return None
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, dataset in zip(axes.ravel(), DATASETS):
        points = sorted(
            (
                (run.spec.epoch, _finite_float(run.summaries.get(dataset, {}).get("iou")))
                for run in frangi_runs
            ),
            key=lambda pair: pair[0] if pair[0] is not None else -1,
        )
        x = [point[0] for point in points if point[0] is not None and math.isfinite(point[1])]
        y = [point[1] for point in points if point[0] is not None and math.isfinite(point[1])]
        if x:
            axis.plot(x, y, marker="o", color="#b2182b", linewidth=2, label="Frangi")
        if baseline and dataset in baseline.summaries:
            value = _finite_float(baseline.summaries[dataset].get("iou"))
            if math.isfinite(value):
                axis.axhline(value, color="#2166ac", linestyle="--", label="Baseline best")
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Époque Frangi")
        axis.set_ylabel("IoU moyen par image")
    handles = [
        Line2D([0], [0], color="#b2182b", marker="o", label="Frangi"),
        Line2D([0], [0], color="#2166ac", linestyle="--", label="Baseline best"),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2)
    figure.suptitle("Évolution des cinq jalons Frangi face à la meilleure baseline", y=1.01)
    figure.tight_layout()
    return _save_figure(figure, output / "figures" / "milestone_iou_evolution.png")


def plot_delta_heatmap(
    baseline: RunData | None, frangi_runs: Sequence[RunData], output: Path
) -> Path | None:
    if baseline is None or not frangi_runs:
        return None
    matrix = np.full((len(frangi_runs), len(DATASETS)), np.nan)
    for row_index, run in enumerate(frangi_runs):
        for column_index, dataset in enumerate(DATASETS):
            base = _finite_float(baseline.summaries.get(dataset, {}).get("iou"))
            new = _finite_float(run.summaries.get(dataset, {}).get("iou"))
            if math.isfinite(base) and math.isfinite(new):
                matrix[row_index, column_index] = new - base
    if not np.isfinite(matrix).any():
        return None
    scale = max(float(np.nanmax(np.abs(matrix))), 1e-4)
    figure, axis = plt.subplots(figsize=(10.5, 4.5))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axis.set_xticks(range(len(DATASETS)), [DATASET_SHORT_LABELS[ds] for ds in DATASETS], rotation=25, ha="right")
    axis.set_yticks(range(len(frangi_runs)), [run.spec.key for run in frangi_runs])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if math.isfinite(value):
                axis.text(column, row, f"{value:+.4f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="ΔIoU Frangi − baseline")
    axis.set_title("Écart d’IoU de chaque jalon Frangi par rapport à la baseline best")
    figure.tight_layout()
    return _save_figure(figure, output / "figures" / "delta_iou_heatmap.png")


def plot_macro_metrics(runs: Sequence[RunData], output: Path) -> Path | None:
    if not runs:
        return None
    metrics = ("precision", "recall", "dice", "iou")
    width = 0.8 / len(runs)
    x = np.arange(len(metrics))
    figure, axis = plt.subplots(figsize=(12, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    any_value = False
    for index, (run, color) in enumerate(zip(runs, colors)):
        macro = aggregate_summary(run, "macro")
        values = [_finite_float(macro.get(metric)) for metric in metrics]
        if any(math.isfinite(value) for value in values):
            any_value = True
        axis.bar(
            x - 0.4 + width / 2 + index * width,
            values,
            width=width,
            color=color,
            label=run.spec.key,
        )
    if not any_value:
        plt.close(figure)
        return None
    axis.set_xticks(x, ["Précision", "Rappel", "Dice/F1", "IoU"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Moyenne macro des six configurations")
    axis.set_title("Vue d’ensemble des métriques par jalon")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    return _save_figure(figure, output / "figures" / "macro_metric_overview.png")


def plot_paper_comparison(
    baseline: RunData | None, primary: RunData | None, output: Path
) -> Path | None:
    series: list[tuple[str, list[float], str]] = []
    if baseline:
        series.append(
            (
                "SAM 2 baseline best",
                [_finite_float(baseline.summaries.get(ds, {}).get("iou")) for ds in DATASETS],
                "#2166ac",
            )
        )
    if primary:
        series.append(
            (
                f"SAM 2 Frangi {primary.spec.key}",
                [_finite_float(primary.summaries.get(ds, {}).get("iou")) for ds in DATASETS],
                "#b2182b",
            )
        )
    paper_colors = ("#4d9221", "#762a83")
    for color, model in zip(paper_colors, PAPER_MODELS.values()):
        series.append((str(model["label"]), [model["iou"][ds] for ds in DATASETS], color))
    if not series:
        return None
    x = np.arange(len(DATASETS))
    width = 0.82 / len(series)
    figure, axis = plt.subplots(figsize=(13, 6))
    for index, (label, values, color) in enumerate(series):
        axis.bar(x - 0.41 + width / 2 + index * width, values, width, label=label, color=color)
    axis.set_xticks(x, [DATASET_SHORT_LABELS[ds] for ds in DATASETS], rotation=20, ha="right")
    axis.set_ylim(0, 0.85)
    axis.set_ylabel("IoU moyen par image")
    axis.set_title("Comparaison avec les deux configurations publiées de CrackSAM originel")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    return _save_figure(figure, output / "figures" / "paper_iou_comparison.png")


def plot_delta_distributions(
    baseline: RunData | None, primary: RunData | None, output: Path
) -> Path | None:
    if baseline is None or primary is None:
        return None
    values: list[np.ndarray] = []
    labels: list[str] = []
    for dataset in DATASETS:
        base = baseline.per_image.get(dataset, {})
        new = primary.per_image.get(dataset, {})
        delta = _finite(
            _finite_float(new[name].get("iou")) - _finite_float(base[name].get("iou"))
            for name in sorted(set(base) & set(new))
            if math.isfinite(_finite_float(new[name].get("iou")))
            and math.isfinite(_finite_float(base[name].get("iou")))
        )
        if delta.size:
            values.append(delta)
            labels.append(DATASET_SHORT_LABELS[dataset])
    if not values:
        return None
    figure, axis = plt.subplots(figsize=(11, 5.5))
    box = axis.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#ef8a62")
        patch.set_alpha(0.75)
    axis.axhline(0, color="black", linewidth=1, linestyle="--")
    axis.set_ylabel("ΔIoU par image (Frangi − baseline)")
    axis.set_title(f"Distribution appariée des écarts — {primary.spec.key}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, output / "figures" / "paired_delta_iou_distributions.png")


def plot_iou_scatter(
    baseline: RunData | None, primary: RunData | None, output: Path
) -> Path | None:
    if baseline is None or primary is None:
        return None
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    found = False
    for axis, dataset in zip(axes.ravel(), DATASETS):
        base = baseline.per_image.get(dataset, {})
        new = primary.per_image.get(dataset, {})
        pairs = [
            (_finite_float(base[name].get("iou")), _finite_float(new[name].get("iou")))
            for name in sorted(set(base) & set(new))
            if math.isfinite(_finite_float(base[name].get("iou")))
            and math.isfinite(_finite_float(new[name].get("iou")))
        ]
        if pairs:
            found = True
            array = np.asarray(pairs)
            axis.hexbin(array[:, 0], array[:, 1], gridsize=35, mincnt=1, cmap="viridis", bins="log")
        axis.plot((0, 1), (0, 1), "--", color="gray", linewidth=1)
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.15)
        axis.set_xlabel("IoU baseline")
        axis.set_ylabel("IoU Frangi")
    if not found:
        plt.close(figure)
        return None
    figure.suptitle(f"IoU apparié image par image — {primary.spec.key}")
    figure.tight_layout()
    return _save_figure(figure, output / "figures" / "paired_iou_scatter.png")


def plot_wasserstein_intersection(
    baseline: RunData | None, primary: RunData | None, output: Path
) -> Path | None:
    if baseline is None or primary is None:
        return None
    labels: list[str] = []
    base_means: list[float] = []
    new_means: list[float] = []
    coverage: list[str] = []
    for dataset in DATASETS:
        base = baseline.per_image.get(dataset, {})
        new = primary.per_image.get(dataset, {})
        pairs = [
            (
                _finite_float(base[name].get("wasserstein")),
                _finite_float(new[name].get("wasserstein")),
            )
            for name in sorted(set(base) & set(new))
            if math.isfinite(_finite_float(base[name].get("wasserstein")))
            and math.isfinite(_finite_float(new[name].get("wasserstein")))
        ]
        if pairs:
            array = np.asarray(pairs)
            labels.append(DATASET_SHORT_LABELS[dataset])
            base_means.append(float(np.mean(array[:, 0])))
            new_means.append(float(np.mean(array[:, 1])))
            coverage.append(f"n={len(pairs)}")
    if not labels:
        return None
    x = np.arange(len(labels))
    width = 0.37
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(x - width / 2, base_means, width, label="Baseline", color="#2166ac")
    axis.bar(x + width / 2, new_means, width, label="Frangi", color="#b2182b")
    axis.set_xticks(x, [f"{label}\n{n}" for label, n in zip(labels, coverage)])
    axis.set_ylabel("Wasserstein exact moyen (plus petit = meilleur)")
    axis.set_title("Wasserstein sur l’intersection stricte des cas finis")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return _save_figure(figure, output / "figures" / "wasserstein_common_intersection.png")


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for raw in _read_csv(path):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            number = _finite_float(value)
            row[key] = number if math.isfinite(number) else value
        result.append(row)
    return result


def load_histories(
    artifact_root: Path, warnings: list[str]
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    histories: dict[str, dict[str, list[dict[str, Any]]]] = {}
    combined: list[dict[str, Any]] = []
    for variant in ("baseline_r4", "frangi_r4"):
        histories[variant] = {}
        for split in ("train", "validation"):
            path = artifact_root / variant / f"{split}.csv"
            rows = _load_history(path)
            histories[variant][split] = rows
            if not rows:
                warnings.append(f"Historique {split} absent: {path}")
            for row in rows:
                combined.append({"variant": variant, "split": split, **row})
    return histories, combined


def plot_training_curves(
    histories: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], output: Path
) -> Path | None:
    if not any(histories.get(variant, {}).get("train") for variant in histories):
        return None
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"baseline_r4": "#2166ac", "frangi_r4": "#b2182b"}
    labels = {"baseline_r4": "Baseline", "frangi_r4": "Frangi"}
    panels = (
        (axes[0, 0], "train", "loss", "Perte entraînement"),
        (axes[0, 1], "validation", "loss", "Perte validation"),
        (axes[1, 0], "validation", "dice", "Dice validation"),
        (axes[1, 1], "validation", "iou", "IoU validation"),
    )
    for axis, split, metric, title in panels:
        for variant in ("baseline_r4", "frangi_r4"):
            rows = histories.get(variant, {}).get(split, ())
            points = [
                (_finite_float(row.get("epoch")), _finite_float(row.get(metric)))
                for row in rows
            ]
            points = [point for point in points if all(math.isfinite(value) for value in point)]
            if points:
                x, y = zip(*points)
                axis.plot(x, y, color=colors[variant], label=labels[variant], linewidth=1.8)
                if split == "validation" and metric in ("dice", "iou"):
                    best_index = int(np.argmax(y))
                    axis.scatter([x[best_index]], [y[best_index]], color=colors[variant], s=35, zorder=3)
        axis.set_title(title)
        axis.set_xlabel("Époque")
        axis.grid(alpha=0.25)
        if metric in ("dice", "iou"):
            axis.set_ylim(0, 1)
        axis.legend()
    figure.suptitle("Dynamiques d’entraînement et de validation")
    figure.tight_layout()
    return _save_figure(figure, output / "figures" / "training_validation_curves.png")


def best_validation_rows(
    histories: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in ("baseline_r4", "frangi_r4"):
        validation = histories.get(variant, {}).get("validation", ())
        valid = [row for row in validation if math.isfinite(_finite_float(row.get("dice")))]
        if not valid:
            continue
        best = max(valid, key=lambda row: _finite_float(row.get("dice")))
        rows.append(
            {
                "variant": variant,
                "best_selection_metric": "validation dice",
                **best,
                "validation_rows": len(validation),
            }
        )
    return rows


def robustness_generalization_rows(runs: Sequence[RunData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        values = {
            dataset: _finite_float(run.summaries.get(dataset, {}).get("iou"))
            for dataset in DATASETS
        }
        clean = values["khanhha_original"]
        zero_shot = _finite(values[dataset] for dataset in ("road420", "facade390", "concrete3k"))
        rows.append(
            {
                "run": run.spec.key,
                "clean_iou": clean,
                "noisy1_iou": values["khanhha_noisy1"],
                "noisy1_drop_from_clean": (
                    values["khanhha_noisy1"] - clean
                    if math.isfinite(clean) and math.isfinite(values["khanhha_noisy1"])
                    else math.nan
                ),
                "noisy2_iou": values["khanhha_noisy2"],
                "noisy2_drop_from_clean": (
                    values["khanhha_noisy2"] - clean
                    if math.isfinite(clean) and math.isfinite(values["khanhha_noisy2"])
                    else math.nan
                ),
                "zero_shot_macro_iou": float(np.mean(zero_shot)) if zero_shot.size else math.nan,
            }
        )
    for key, model in PAPER_MODELS.items():
        values = model["iou"]
        clean = values["khanhha_original"]
        rows.append(
            {
                "run": key,
                "clean_iou": clean,
                "noisy1_iou": values["khanhha_noisy1"],
                "noisy1_drop_from_clean": values["khanhha_noisy1"] - clean,
                "noisy2_iou": values["khanhha_noisy2"],
                "noisy2_drop_from_clean": values["khanhha_noisy2"] - clean,
                "zero_shot_macro_iou": float(
                    np.mean([values[ds] for ds in ("road420", "facade390", "concrete3k")])
                ),
            }
        )
    return rows


def paper_reference_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, model in PAPER_MODELS.items():
        for dataset in DATASETS:
            rows.append(
                {
                    "model": key,
                    "label": model["label"],
                    "dataset": dataset,
                    "precision": model.get("precision", {}).get(dataset, math.nan),
                    "recall": model.get("recall", {}).get(dataset, math.nan),
                    "dice": model.get("dice", {}).get(dataset, math.nan),
                    "iou": model["iou"][dataset],
                    "iou_source": model["source"],
                    "pr_re_f1_source": model.get("secondary_source", ""),
                }
            )
    return rows


def paper_comparison_rows(runs: Sequence[RunData]) -> list[dict[str, Any]]:
    """Compare every local run with both published models wherever defined."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        for paper_key, model in PAPER_MODELS.items():
            scopes: list[tuple[str, Mapping[str, Any]]] = [
                (dataset, run.summaries.get(dataset, {})) for dataset in DATASETS
            ]
            scopes.extend(
                (
                    ("MACRO_6_DATASETS", aggregate_summary(run, "macro")),
                    ("PONDERE_IMAGES", aggregate_summary(run, "weighted")),
                )
            )
            for dataset, run_summary in scopes:
                row: dict[str, Any] = {
                    "run": run.spec.key,
                    "run_label": run.spec.label,
                    "epoch": run.spec.epoch,
                    "paper_model": paper_key,
                    "paper_label": model["label"],
                    "dataset": dataset,
                }
                for metric in ("precision", "recall", "dice", "iou"):
                    run_value = _finite_float(run_summary.get(metric))
                    if dataset in DATASETS:
                        paper_value = _finite_float(model.get(metric, {}).get(dataset))
                    elif metric == "iou":
                        published = np.asarray(
                            [model["iou"][name] for name in DATASETS], dtype=np.float64
                        )
                        if dataset == "MACRO_6_DATASETS":
                            paper_value = float(np.mean(published))
                        else:
                            weights = np.asarray(
                                [DATASET_SAMPLES[name] for name in DATASETS],
                                dtype=np.float64,
                            )
                            paper_value = float(np.average(published, weights=weights))
                    else:
                        paper_value = math.nan
                    row[f"sam2_{metric}"] = run_value
                    row[f"paper_{metric}"] = paper_value
                    row[f"delta_{metric}_sam2_minus_paper"] = (
                        run_value - paper_value
                        if math.isfinite(run_value) and math.isfinite(paper_value)
                        else math.nan
                    )
                row["iou_source"] = model["source"]
                row["pr_re_f1_source"] = model.get("secondary_source", "")
                rows.append(row)
    return rows


def plot_paper_delta_heatmap(
    rows: Sequence[Mapping[str, Any]], output: Path
) -> Path | None:
    labels: list[str] = []
    matrix_rows: list[list[float]] = []
    for run_key in ("baseline_best", *(item[0] for item in MILESTONES)):
        for paper_key, short_paper in (
            ("paper_adapter_d32", "Adapter d32"),
            ("paper_lora_qv_r4", "LoRA qv r4"),
        ):
            selected = {
                str(row.get("dataset")): _finite_float(
                    row.get("delta_iou_sam2_minus_paper")
                )
                for row in rows
                if row.get("run") == run_key and row.get("paper_model") == paper_key
            }
            if not any(dataset in selected for dataset in DATASETS):
                continue
            labels.append(f"{run_key} − {short_paper}")
            matrix_rows.append([selected.get(dataset, math.nan) for dataset in DATASETS])
    if not matrix_rows:
        return None
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    scale = max(float(np.nanmax(np.abs(matrix))), 1e-4)
    figure, axis = plt.subplots(
        figsize=(11.5, max(5.5, 0.43 * len(labels) + 2.0)), constrained_layout=True
    )
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axis.set_xticks(
        range(len(DATASETS)),
        [DATASET_SHORT_LABELS[dataset] for dataset in DATASETS],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if math.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    figure.colorbar(image, ax=axis, label="ΔIoU SAM 2 − CrackSAM publié")
    axis.set_title(
        "Comparaison exhaustive des six runs avec les deux CrackSAM publiés"
    )
    return _save_figure(figure, output / "figures" / "paper_delta_iou_heatmap.png")


def delta_quantile_rows(
    baseline: RunData | None, primary: RunData | None
) -> list[dict[str, Any]]:
    if baseline is None or primary is None:
        return []
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        base = baseline.per_image.get(dataset, {})
        new = primary.per_image.get(dataset, {})
        deltas = _finite(
            _finite_float(new[name].get("iou")) - _finite_float(base[name].get("iou"))
            for name in sorted(set(base) & set(new))
            if math.isfinite(_finite_float(new[name].get("iou")))
            and math.isfinite(_finite_float(base[name].get("iou")))
        )
        if deltas.size:
            quantiles = np.percentile(deltas, (0, 5, 25, 50, 75, 95, 100))
            rows.append(
                {
                    "dataset": dataset,
                    "n": int(deltas.size),
                    "min": float(quantiles[0]),
                    "p05": float(quantiles[1]),
                    "p25": float(quantiles[2]),
                    "median": float(quantiles[3]),
                    "p75": float(quantiles[4]),
                    "p95": float(quantiles[5]),
                    "max": float(quantiles[6]),
                }
            )
    return rows


def _fmt(value: Any, digits: int = 4, signed: bool = False) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "n.d."
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    number = _finite_float(value)
    return "n.d." if not math.isfinite(number) else f"{100 * number:.{digits}f} %"


def _range_text(low: Any, high: Any, *, digits: int = 1, suffix: str = "") -> str:
    low_number = _finite_float(low)
    high_number = _finite_float(high)
    if not (math.isfinite(low_number) and math.isfinite(high_number)):
        return "n.d."
    return f"{low_number:.{digits}f}–{high_number:.{digits}f}{suffix}"


def _count_fmt(value: Any) -> str:
    number = _finite_float(value)
    return "n.d." if not math.isfinite(number) else f"{int(number):,}".replace(",", " ")


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_Aucune donnée disponible._"
    lines = [
        "| " + " | ".join(_md_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_md_escape(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _relative_link(output: Path, path: Path | None) -> str | None:
    return None if path is None else path.relative_to(output).as_posix()


def build_report(
    *,
    args: argparse.Namespace,
    runs: Sequence[RunData],
    baseline: RunData | None,
    frangi_runs: Sequence[RunData],
    primary: RunData | None,
    metric_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
    paper_comparisons: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    best_validation: Sequence[Mapping[str, Any]],
    robustness_rows: Sequence[Mapping[str, Any]],
    wasserstein_feasibility: Sequence[Mapping[str, Any]],
    wasserstein_manifest_path: Path,
    wasserstein_manifest: Mapping[str, Any],
    wasserstein_threshold_rows: Sequence[Mapping[str, Any]],
    wasserstein_audit_rows: Sequence[Mapping[str, Any]],
    quantile_rows: Sequence[Mapping[str, Any]],
    panel_rows: Sequence[Mapping[str, Any]],
    prompt_manifest: Sequence[Mapping[str, Any]],
    checkpoint_manifest_path: Path,
    checkpoint_manifest: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    figures: Mapping[str, Path | None],
    warnings: Sequence[str],
) -> str:
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    primary_label = primary.spec.label if primary else "indisponible"
    wasserstein_run_order_value = wasserstein_manifest.get("run_order", [])
    wasserstein_run_order = (
        [str(run) for run in wasserstein_run_order_value]
        if isinstance(wasserstein_run_order_value, list)
        else []
    )
    live_attempt_value = wasserstein_manifest.get("live_attempt", {})
    live_attempt = (
        live_attempt_value if isinstance(live_attempt_value, Mapping) else {}
    )
    wasserstein_threshold_lookup: dict[float, Mapping[str, Any]] = {}
    for row in wasserstein_threshold_rows:
        threshold = _finite_float(row.get("per_case_threshold_gib"))
        if math.isfinite(threshold):
            wasserstein_threshold_lookup[threshold] = row
    threshold_8 = wasserstein_threshold_lookup.get(8.0, {})
    threshold_140 = wasserstein_threshold_lookup.get(140.0, {})
    sections: list[str] = [
        "# Rapport exhaustif — CrackSAM 2 baseline vs guidage Frangi-similarité",
        "",
        f"> Généré automatiquement le **{generated}**. Comparaison qualitative et "
        f"appariée principale : **{primary_label}**. Les cinq jalons Frangi restent "
        "tous évalués dans les tableaux et graphiques.",
        "",
        "## Résumé exécutif",
        "",
    ]

    if baseline and primary:
        baseline_macro = aggregate_summary(baseline, "macro")
        primary_macro = aggregate_summary(primary, "macro")
        delta = _finite_float(primary_macro.get("iou")) - _finite_float(baseline_macro.get("iou"))
        weighted_pair = next(
            (
                row
                for row in paired_rows
                if row.get("metric") == "iou" and row.get("dataset") == "PONDERE_IMAGES"
            ),
            None,
        )
        sections.extend(
            [
                f"La baseline atteint un IoU macro de **{_fmt(baseline_macro.get('iou'))}** "
                f"sur les six configurations, contre **{_fmt(primary_macro.get('iou'))}** "
                f"pour le jalon principal Frangi (Δ={_fmt(delta, signed=True)}).",
                "",
            ]
        )
        if weighted_pair:
            sections.extend(
                [
                    "Sur les observations appariées, le delta moyen pondéré est "
                    f"**{_fmt(weighted_pair.get('mean_delta_frangi_minus_baseline'), signed=True)}** "
                    f"(IC bootstrap 95 % [{_fmt(weighted_pair.get('mean_delta_ci95_low'), signed=True)}, "
                    f"{_fmt(weighted_pair.get('mean_delta_ci95_high'), signed=True)}]), avec "
                    f"**{weighted_pair.get('wins_frangi', 0)} gains**, "
                    f"**{weighted_pair.get('ties', 0)} égalités** et "
                    f"**{weighted_pair.get('losses_frangi', 0)} pertes** pour Frangi.",
                    "",
                ]
            )
    else:
        sections.extend(
            [
                "La comparaison principale n’est pas calculable car la baseline ou le jalon "
                "Frangi demandé est absent. Les résultats partiels disponibles sont néanmoins publiés.",
                "",
            ]
        )

    sections.extend(
        [
            "Le point essentiel d’interprétation est que **epoch25_best est sélectionné par le "
            "Dice de validation**, pas en recherchant le meilleur résultat sur les jeux de test. "
            "Les autres jalons servent à documenter la stabilité et une éventuelle dérive, sans "
            "modifier a posteriori le modèle principal.",
            "",
            "## Protocole, sources et garanties",
            "",
            "- Baseline : SAM 2 Hiera Large + LoRA q/v rang 4, meilleure époque selon la validation, sans `mask_input`.",
            "- Variante : même architecture et même protocole, avec pseudo-logits Frangi-similarité statiques de forme 1×256×256 comme `mask_input`.",
            "- Tests : Khanhha original, deux perturbations déterministes, puis Road420, Facade390 et Concrete3k en zero-shot.",
            "- Métriques de segmentation : moyenne arithmétique des métriques calculées image par image, conformément aux CSV d’évaluation.",
            "- Wasserstein : distance sur masque direct. Aucune valeur plafonnée à 2 000 points n’est présentée comme exacte ; seules les sorties de `wasserstein_exact/` sont intégrées.",
            "- Incertitude : bootstrap percentile 95 % déterministe, "
            f"{args.bootstrap_repetitions:,} réplications, graine {args.bootstrap_seed}; bootstrap stratifié pour le macro.",
            f"- Égalité appariée : |amélioration| ≤ {args.tie_tolerance:g}.",
            "",
            "### Racines utilisées",
            "",
            md_table(
                ("Rôle", "Chemin"),
                (
                    ("Données", args.data_root),
                    ("Prompts", args.prompt_root),
                    ("Artefacts", args.artifact_root),
                    ("Rapport", args.output),
                ),
            ),
            "",
            "### Inventaire des évaluations",
            "",
            md_table(
                ("Run", "Époque", "Datasets", "Cas", "Wasserstein exact", "Racine"),
                [
                    (
                        run.spec.key,
                        run.spec.epoch if run.spec.epoch is not None else "best val",
                        f"{len(run.summaries)}/6",
                        sum(len(run.per_image.get(ds, {})) for ds in DATASETS),
                        run.exact_status,
                        run.spec.path,
                    )
                    for run in runs
                ],
            ),
            "",
            "## Référence : CrackSAM originel publié",
            "",
            "Ces nombres sont des valeurs publiées dans `CrackSAM.pdf`, et non des "
            "réévaluations locales. Le backbone (SAM 1 ViT-H) diffère du SAM 2 Hiera Large "
            "utilisé ici : la comparaison mesure un niveau de performance, pas une ablation "
            "architecturale strictement contrôlée.",
            "",
            md_table(
                ("Modèle papier", "Pr clean", "Re clean", "F1 clean", *[DATASET_SHORT_LABELS[ds] for ds in DATASETS]),
                [
                    (
                        model["label"],
                        _fmt(model.get("precision", {}).get("khanhha_original")),
                        _fmt(model.get("recall", {}).get("khanhha_original")),
                        _fmt(model.get("dice", {}).get("khanhha_original")),
                        *[_fmt(model["iou"][ds]) for ds in DATASETS],
                    )
                    for model in PAPER_MODELS.values()
                ],
            ),
            "",
            "Sources : IoU des six configurations pour les deux modèles dans la Table 6 ; "
            "Pr/Re/F1 du LoRA qv rang 4 dans la Table 2 ; Pr/Re/F1 de l’Adapter "
            "d=32 dans la Table 1 (ablation du milieu de l’Adapter).",
            "",
        ]
    )
    paper_figure = figures.get("paper")
    if paper_figure:
        sections.extend([f"![Comparaison avec CrackSAM originel]({_relative_link(args.output, paper_figure)})", ""])
    paper_delta_figure = figures.get("paper_delta")
    if paper_delta_figure:
        sections.extend(
            [
                "### Deltas exhaustifs face aux deux références publiées",
                "",
                f"![Heatmap des deltas papier]({_relative_link(args.output, paper_delta_figure)})",
                "",
            ]
        )
    for paper_key, paper_short in (
        ("paper_adapter_d32", "Adapter d=32"),
        ("paper_lora_qv_r4", "LoRA qv r=4"),
    ):
        sections.extend(
            [
                f"#### ΔIoU SAM 2 − CrackSAM originel {paper_short}",
                "",
                md_table(
                    (
                        "Run",
                        *[DATASET_SHORT_LABELS[dataset] for dataset in DATASETS],
                        "Macro",
                        "Pondéré",
                    ),
                    [
                        (
                            run.spec.key,
                            *[
                                _fmt(
                                    next(
                                        (
                                            row.get("delta_iou_sam2_minus_paper")
                                            for row in paper_comparisons
                                            if row.get("run") == run.spec.key
                                            and row.get("paper_model") == paper_key
                                            and row.get("dataset") == dataset
                                        ),
                                        math.nan,
                                    ),
                                    signed=True,
                                )
                                for dataset in DATASETS
                            ],
                            _fmt(
                                next(
                                    (
                                        row.get("delta_iou_sam2_minus_paper")
                                        for row in paper_comparisons
                                        if row.get("run") == run.spec.key
                                        and row.get("paper_model") == paper_key
                                        and row.get("dataset") == "MACRO_6_DATASETS"
                                    ),
                                    math.nan,
                                ),
                                signed=True,
                            ),
                            _fmt(
                                next(
                                    (
                                        row.get("delta_iou_sam2_minus_paper")
                                        for row in paper_comparisons
                                        if row.get("run") == run.spec.key
                                        and row.get("paper_model") == paper_key
                                        and row.get("dataset") == "PONDERE_IMAGES"
                                    ),
                                    math.nan,
                                ),
                                signed=True,
                            ),
                        )
                        for run in runs
                    ],
                ),
                "",
            ]
        )
    sections.extend(
        [
            "#### Métriques publiées sur Khanhha propre : valeurs et deltas",
            "",
            md_table(
                (
                    "Run",
                    "Référence papier",
                    "Pr SAM2/papier/Δ",
                    "Re SAM2/papier/Δ",
                    "F1 SAM2/papier/Δ",
                    "IoU SAM2/papier/Δ",
                ),
                [
                    (
                        row.get("run"),
                        row.get("paper_model"),
                        f"{_fmt(row.get('sam2_precision'))} / {_fmt(row.get('paper_precision'))} / "
                        f"{_fmt(row.get('delta_precision_sam2_minus_paper'), signed=True)}",
                        f"{_fmt(row.get('sam2_recall'))} / {_fmt(row.get('paper_recall'))} / "
                        f"{_fmt(row.get('delta_recall_sam2_minus_paper'), signed=True)}",
                        f"{_fmt(row.get('sam2_dice'))} / {_fmt(row.get('paper_dice'))} / "
                        f"{_fmt(row.get('delta_dice_sam2_minus_paper'), signed=True)}",
                        f"{_fmt(row.get('sam2_iou'))} / {_fmt(row.get('paper_iou'))} / "
                        f"{_fmt(row.get('delta_iou_sam2_minus_paper'), signed=True)}",
                    )
                    for row in paper_comparisons
                    if row.get("dataset") == "khanhha_original"
                ],
            ),
            "",
        ]
    )

    sections.extend(["## Dynamique d’entraînement et choix des poids", ""])
    training_figure = figures.get("training")
    if training_figure:
        sections.extend([f"![Courbes d’entraînement]({_relative_link(args.output, training_figure)})", ""])
    sections.extend(
        [
            md_table(
                ("Variante", "Époque best", "Dice val", "IoU val", "Pr val", "Re val", "Pas global"),
                [
                    (
                        row.get("variant"),
                        int(_finite_float(row.get("epoch"))) if math.isfinite(_finite_float(row.get("epoch"))) else "n.d.",
                        _fmt(row.get("dice")),
                        _fmt(row.get("iou")),
                        _fmt(row.get("precision")),
                        _fmt(row.get("recall")),
                        int(_finite_float(row.get("global_step"))) if math.isfinite(_finite_float(row.get("global_step"))) else "n.d.",
                    )
                    for row in best_validation
                ],
            ),
            "",
            "### Manifeste des poids conservés",
            "",
            f"Source versionnée : `{checkpoint_manifest_path}` (format "
            f"{checkpoint_manifest.get('format_version', 'n.d.')}, vérifié le "
            f"{checkpoint_manifest.get('verified_at_utc', 'n.d.')}).",
            "",
            md_table(
                (
                    "ID",
                    "Variante/rôle",
                    "Époque",
                    "Pas",
                    "Taille",
                    "SHA-256",
                    "Chemin VM ici",
                    "Backup local ici",
                    "Audit manifeste",
                ),
                [
                    (
                        row.get("id"),
                        f"{row.get('variant')} / {row.get('role')}",
                        row.get("epoch") if row.get("epoch") is not None else "—",
                        row.get("global_step") if row.get("global_step") is not None else "—",
                        f"{int(row.get('size_bytes', 0)):,}" if row.get("size_bytes") else "n.d.",
                        str(row.get("sha256", ""))[:16] + "…",
                        "présent et vérifié" if row.get("vm_verified") else (
                            "présent, non conforme" if row.get("vm_present") else "non visible ici"
                        ),
                        "présent et vérifié" if row.get("local_verified") else (
                            "présent, non conforme" if row.get("local_present") else "non monté ici"
                        ),
                        checkpoint_manifest.get("verified_at_utc", "n.d."),
                    )
                    for row in checkpoint_rows
                ],
            ),
            "",
            "Les deux colonnes « ici » décrivent uniquement le système de fichiers visible "
            "pendant cette génération. `non monté ici` ne signifie donc pas que la sauvegarde "
            "a disparu : les chemins VM et backups locaux ont été contrôlés séparément à la "
            f"date `verified_at_utc={checkpoint_manifest.get('verified_at_utc', 'n.d.')}` du "
            "manifeste versionné. Lorsqu’un fichier est visible ici, sa taille et son SHA-256 "
            "sont recalculés.",
            "",
            "Le checkpoint de fondation SAM 2 (~898 Mo) est une dépendance identifiée par "
            "SHA-256 et n’est pas destiné à Git. Les poids adaptateurs conservés font environ "
            "5,7 Mo chacun. Les alias `best.pt`/époque 25 et `latest.pt`/époque 70 sont "
            "documentés dans le manifeste sans dupliquer leur contenu.",
            "",
            "## Comparaison exhaustive des jalons",
            "",
            "### IoU par dataset et par jalon",
            "",
        ]
    )
    iou_rows: list[tuple[Any, ...]] = []
    for run in runs:
        macro = aggregate_summary(run, "macro")
        weighted = aggregate_summary(run, "weighted")
        iou_rows.append(
            (
                run.spec.key,
                *[_fmt(run.summaries.get(ds, {}).get("iou")) for ds in DATASETS],
                _fmt(macro.get("iou")),
                _fmt(weighted.get("iou")),
            )
        )
    sections.extend(
        [
            md_table(
                ("Run", *[DATASET_SHORT_LABELS[ds] for ds in DATASETS], "Macro", "Pondéré"),
                iou_rows,
            ),
            "",
        ]
    )
    for key in ("milestones", "heatmap", "macro"):
        figure_path = figures.get(key)
        if figure_path:
            sections.extend([f"![{key}]({_relative_link(args.output, figure_path)})", ""])

    sections.extend(["### Métriques complètes pour chaque évaluation", ""])
    for run in runs:
        sections.extend([f"#### {run.spec.label}", ""])
        detailed_rows = []
        for dataset in DATASETS:
            summary = run.summaries.get(dataset)
            if not summary:
                continue
            w_count = int(summary.get("wasserstein_finite_samples", 0))
            detailed_rows.append(
                (
                    DATASET_LABELS[dataset],
                    summary.get("samples", 0),
                    _fmt(summary.get("precision")),
                    _fmt(summary.get("recall")),
                    _fmt(summary.get("dice")),
                    _fmt(summary.get("iou")),
                    _fmt(summary.get("wasserstein")),
                    f"{w_count}/{summary.get('samples', 0)}",
                )
            )
        for mode, label in (("macro", "Macro six datasets"), ("weighted", "Pondéré images")):
            summary = aggregate_summary(run, mode)
            detailed_rows.append(
                (
                    label,
                    summary.get("samples", 0),
                    _fmt(summary.get("precision")),
                    _fmt(summary.get("recall")),
                    _fmt(summary.get("dice")),
                    _fmt(summary.get("iou")),
                    _fmt(summary.get("wasserstein")),
                    f"{summary.get('wasserstein_finite_samples', 0)}/{summary.get('samples', 0)}",
                )
            )
        sections.extend(
            [
                md_table(
                    ("Dataset", "n", "Pr", "Re", "Dice", "IoU", "W exact", "Couverture W"),
                    detailed_rows,
                ),
                "",
            ]
        )

    sections.extend(["### Deltas de chaque jalon face à la baseline best", ""])
    for frangi in frangi_runs:
        rows_for_run = [
            row
            for row in delta_rows
            if row.get("milestone") == frangi.spec.key
            and row.get("dataset") in (*DATASETS, "MACRO_6_DATASETS", "PONDERE_IMAGES")
        ]
        sections.extend(
            [
                f"#### {frangi.spec.key}",
                "",
                md_table(
                    (
                        "Dataset",
                        "ΔPr",
                        "ΔRe",
                        "ΔDice",
                        "ΔIoU",
                        "Amélioration W",
                        "Couverture W commune",
                    ),
                    [
                        (
                            DATASET_LABELS.get(str(row["dataset"]), row["dataset"]),
                            _fmt(row.get("delta_precision_frangi_minus_baseline"), signed=True),
                            _fmt(row.get("delta_recall_frangi_minus_baseline"), signed=True),
                            _fmt(row.get("delta_dice_frangi_minus_baseline"), signed=True),
                            _fmt(row.get("delta_iou_frangi_minus_baseline"), signed=True),
                            _fmt(row.get("improvement_wasserstein"), signed=True),
                            f"{row.get('wasserstein_common_finite', 0)}/"
                            f"{row.get('wasserstein_total_pairs', 0)} · "
                            f"{row.get('wasserstein_datasets_covered', 0)}/"
                            f"{row.get('wasserstein_datasets_total', 1)} ds",
                        )
                        for row in rows_for_run
                    ],
                ),
                "",
            ]
        )

    sections.extend(
        [
            "## Statistiques appariées du jalon principal",
            "",
            "Les cas sont joints par `(dataset, case_name)`. Pour l’IoU, Δ = Frangi − "
            "baseline. Pour Wasserstein, la colonne delta garde aussi Frangi − baseline, "
            "donc une valeur négative est favorable ; la colonne d’amélioration remet le "
            "signe dans le sens « positif = meilleur ». Les wins/ties/losses utilisent ce "
            "sens favorable.",
            "",
        ]
    )
    for metric in ("iou", "wasserstein"):
        current = [row for row in paired_rows if row.get("metric") == metric]
        sections.extend(
            [
                f"### {metric.upper()}",
                "",
                md_table(
                    (
                        "Dataset/agrégation",
                        "Datasets couverts",
                        "n commun",
                        "Couverture",
                        "Δ moyen [IC95]",
                        "Δ médian [IC95]",
                        "G/E/P Frangi",
                    ),
                    [
                        (
                            DATASET_LABELS.get(str(row["dataset"]), row["dataset"]),
                            f"{row.get('datasets_with_common_finite', 0)}/{row.get('datasets_total', 1)}",
                            row.get("common_finite", 0),
                            _pct(row.get("common_coverage")),
                            f"{_fmt(row.get('mean_delta_frangi_minus_baseline'), signed=True)} "
                            f"[{_fmt(row.get('mean_delta_ci95_low'), signed=True)}, "
                            f"{_fmt(row.get('mean_delta_ci95_high'), signed=True)}]",
                            f"{_fmt(row.get('median_delta_frangi_minus_baseline'), signed=True)} "
                            f"[{_fmt(row.get('median_delta_ci95_low'), signed=True)}, "
                            f"{_fmt(row.get('median_delta_ci95_high'), signed=True)}]",
                            f"{row.get('wins_frangi', 0)}/{row.get('ties', 0)}/{row.get('losses_frangi', 0)}",
                        )
                        for row in current
                    ],
                ),
                "",
            ]
        )
    for key in ("distribution", "scatter", "wasserstein"):
        figure_path = figures.get(key)
        if figure_path:
            sections.extend([f"![{key}]({_relative_link(args.output, figure_path)})", ""])

    sections.extend(
        [
            "### Quantiles des deltas IoU par image",
            "",
            md_table(
                ("Dataset", "n", "Min", "P05", "P25", "Médiane", "P75", "P95", "Max"),
                [
                    (
                        DATASET_LABELS.get(str(row["dataset"]), row["dataset"]),
                        row.get("n"),
                        _fmt(row.get("min"), signed=True),
                        _fmt(row.get("p05"), signed=True),
                        _fmt(row.get("p25"), signed=True),
                        _fmt(row.get("median"), signed=True),
                        _fmt(row.get("p75"), signed=True),
                        _fmt(row.get("p95"), signed=True),
                        _fmt(row.get("max"), signed=True),
                    )
                    for row in quantile_rows
                ],
            ),
            "",
            "## Robustesse aux perturbations et généralisation zero-shot",
            "",
            md_table(
                ("Run", "IoU clean", "IoU bruit 1", "Δ bruit 1", "IoU bruit 2", "Δ bruit 2", "IoU zero-shot macro"),
                [
                    (
                        row.get("run"),
                        _fmt(row.get("clean_iou")),
                        _fmt(row.get("noisy1_iou")),
                        _fmt(row.get("noisy1_drop_from_clean"), signed=True),
                        _fmt(row.get("noisy2_iou")),
                        _fmt(row.get("noisy2_drop_from_clean"), signed=True),
                        _fmt(row.get("zero_shot_macro_iou")),
                    )
                    for row in robustness_rows
                ],
            ),
            "",
            "Les deltas de bruit sont calculés à l’intérieur d’un même modèle. Ils "
            "quantifient la dégradation par rapport au test propre ; ils ne doivent pas être "
            "confondus avec les deltas Frangi − baseline.",
            "",
            "## Analyse qualitative : gains, échecs et cas typiques",
            "",
            "Pour chacun des six datasets, cinq cas réels à vérité terrain substantielle "
            "(**plus de 32 pixels positifs après redimensionnement 448×448**) sont sélectionnés "
            "sans jugement manuel : plus grand gain Frangi, plus grand gain baseline, meilleur "
            "cas où les deux réussissent, plus faible cas où les deux échouent, et cas le plus "
            "proche du delta médian. Lorsqu’il existe, un sixième cas à GT vide ou clairsemée "
            "(≤32 pixels) maximisant |ΔIoU| est ajouté. Cette règle expose volontairement les "
            "succès comme les contre-exemples sans laisser les masques vides monopoliser les extrema.",
            "",
            "Dans chaque panneau, la tuile « prompt baseline » représente fidèlement "
            "l’absence de tenseur (`mask_input=None`) ; ce n’est pas un prompt nul réellement "
            "injecté. La carte Frangi est la sigmoid des pseudo-logits réellement chargés. "
            "Vert = vrai positif, rouge = faux positif, cyan = faux négatif.",
            "",
        ]
    )
    for dataset in DATASETS:
        current = [row for row in panel_rows if row.get("dataset") == dataset]
        if not current:
            continue
        sections.extend([f"### {DATASET_LABELS[dataset]}", ""])
        for row in current:
            sections.extend(
                [
                    f"#### {row.get('category_label')} — `{row.get('case_name')}`",
                    "",
                    f"IoU baseline {_fmt(row.get('baseline_iou_csv'))}, IoU Frangi "
                    f"{_fmt(row.get('frangi_iou_csv'))}, ΔIoU "
                    f"{_fmt(row.get('delta_iou'), signed=True)} ; GT = "
                    f"{row.get('target_pixels', row.get('target_pixels_at_selection', 'n.d.'))} pixels.",
                    "",
                ]
            )
            if row.get("generated"):
                sections.extend([f"![Panneau qualitatif]({row.get('panel')})", ""])
            else:
                sections.extend([f"_Panneau indisponible : {_md_escape(row.get('error', 'raison inconnue'))}._", ""])

    sections.extend(
        [
            "## Galerie et archivage des prompts Frangi-similarité",
            "",
            f"**{len(prompt_manifest)} prompts `.npy`** sont copiés sans modification dans "
            "`prompts_npy/`. Le SHA-256, la forme, le type et les statistiques en espace "
            "logit/probabilité sont enregistrés dans `prompts_npy/manifest.json` et "
            "`tables/prompt_manifest.csv`. Cela permet d’auditer exactement ce qui a été "
            "ajouté à la variante Frangi par rapport à la baseline sans prompt.",
            "",
        ]
    )
    gallery = figures.get("prompts")
    if gallery:
        sections.extend([f"![Galerie des prompts]({_relative_link(args.output, gallery)})", ""])
    if prompt_manifest:
        sections.extend(
            [
                md_table(
                    ("Dataset", "Cas", "Catégorie", "ΔIoU", "Fichier", "SHA-256 (12 car.)"),
                    [
                        (
                            DATASET_SHORT_LABELS[str(row["dataset"])],
                            row["case_name"],
                            CATEGORY_LABELS[str(row["category"])],
                            _fmt(row.get("delta_iou"), signed=True),
                            row["copied_path"],
                            str(row["sha256"])[:12],
                        )
                        for row in prompt_manifest
                    ],
                ),
                "",
            ]
        )

    sections.extend(
        [
            "## Wasserstein : couverture et limites de calcul",
            "",
            "### Faisabilité transversale du transport dense exact",
            "",
            f"Source versionnée : `{wasserstein_manifest_path}` (format "
            f"{wasserstein_manifest.get('format_version', 'n.d.')}, générée le "
            f"{wasserstein_manifest.get('generated_at_utc', 'n.d.')}). Métrique planifiée : "
            f"`{wasserstein_manifest.get('metric', 'n.d.')}`.",
            "",
            "Le tableau suivant applique le **même seuil d’admissibilité mémoire par cas** "
            "aux six runs. Les six colonnes de runs donnent le nombre de cas exécutables "
            "individuellement ; « commun » impose l’intersection stricte des cas admissibles "
            "dans les six runs. Les ETA idéales supposent huit workers parfaitement remplis "
            "et ne constituent pas des mesures de durée.",
            "",
            md_table(
                (
                    "Seuil/cas",
                    *wasserstein_run_order,
                    "Commun",
                    "Couverture",
                    "Union exclue",
                    "ETA idéale (8 workers)",
                    "ETA réelle estimée",
                ),
                [
                    (
                        f"{_fmt(row.get('per_case_threshold_gib'), 2)} GiB",
                        *[
                            row.get(f"runnable_{run}", "n.d.")
                            for run in wasserstein_run_order
                        ],
                        f"{row.get('strict_common_cases', 'n.d.')}/"
                        f"{row.get('cases_per_run', 'n.d.')}",
                        _pct(row.get("strict_common_fraction"), 3),
                        row.get("union_excluded_cases", "n.d."),
                        _range_text(
                            row.get("ideal_wall_time_hours_8_workers_low"),
                            row.get("ideal_wall_time_hours_8_workers_high"),
                            suffix=" h",
                        ),
                        _range_text(
                            row.get("estimated_real_wall_time_hours_low"),
                            row.get("estimated_real_wall_time_hours_high"),
                            suffix=" h",
                        ),
                    )
                    for row in wasserstein_threshold_rows
                ],
            ),
            "",
            "Intersection stricte détaillée par dataset lorsque le scan versionné la "
            "fournit :",
            "",
            md_table(
                ("Seuil/cas", *[DATASET_SHORT_LABELS[dataset] for dataset in DATASETS]),
                [
                    (
                        f"{_fmt(row.get('per_case_threshold_gib'), 2)} GiB",
                        *[
                            row.get(f"strict_common_{dataset}", "n.d.")
                            for dataset in DATASETS
                        ],
                    )
                    for row in wasserstein_threshold_rows
                    if any(
                        row.get(f"strict_common_{dataset}") is not None
                        for dataset in DATASETS
                    )
                ],
            ),
            "",
            f"Le compromis **8 GiB/cas** conserve "
            f"**{threshold_8.get('strict_common_cases', 'n.d.')}/"
            f"{threshold_8.get('cases_per_run', 'n.d.')} cas communs** "
            f"({_pct(threshold_8.get('strict_common_fraction'), 3)}) pour une ETA idéale "
            f"de {_range_text(threshold_8.get('ideal_wall_time_hours_8_workers_low'), threshold_8.get('ideal_wall_time_hours_8_workers_high'), suffix=' h')}. "
            f"À **140 GiB/cas**, le support commun atteint "
            f"**{threshold_140.get('strict_common_cases', 'n.d.')}/"
            f"{threshold_140.get('cases_per_run', 'n.d.')}** "
            f"({_pct(threshold_140.get('strict_common_fraction'), 3)}), mais l’ETA réelle "
            f"est {_range_text(threshold_140.get('estimated_real_wall_time_hours_low'), threshold_140.get('estimated_real_wall_time_hours_high'), suffix=' h')}.",
            "",
            "### Tentative réelle à 140 GiB : état durable, non publiable comme moyenne",
            "",
            md_table(
                (
                    "Run",
                    "Exécutables",
                    "Terminés durables",
                    "Échecs",
                    "Journal reprenable",
                    "Résumé publié",
                    "Cas actif max",
                    "RSS observée",
                ),
                [
                    (
                        live_attempt.get("run", "n.d."),
                        live_attempt.get("runnable_cases", "n.d."),
                        live_attempt.get("durably_completed_cases", "n.d."),
                        live_attempt.get("failed_cases", "n.d."),
                        "oui" if live_attempt.get("journal_is_resumable") else "non",
                        "oui" if live_attempt.get("published_summary") else "non",
                        f"{_fmt(live_attempt.get('largest_active_case_estimated_gib'), 2)} GiB; "
                        f"≥{_fmt(live_attempt.get('largest_active_case_elapsed_seconds_lower_bound'), 0)} s; "
                        + (
                            "terminé"
                            if live_attempt.get("largest_active_case_completed_before_stop")
                            else "incomplet à l’arrêt"
                        ),
                        f"{_fmt(live_attempt.get('observed_worker_rss_gib'), 2)} GiB",
                    )
                ]
                if live_attempt
                else [],
            ),
            "",
            f"Motif d’arrêt consigné : {_md_escape(live_attempt.get('stop_reason', 'n.d.'))} "
            "Le journal baseline reste auditable et reprenable, mais **les "
            f"{_count_fmt(live_attempt.get('durably_completed_cases'))} distances déjà "
            "terminées ne sont jamais moyennées ni présentées comme résultat final**, "
            "car elles forment un sous-échantillon déterminé par le coût de calcul et ne sont "
            "pas comparables aux cinq jalons encore non calculés.",
            "",
            "### Faisabilité et progression observées dans chaque répertoire de run",
            "",
            md_table(
                (
                    "Run",
                    "Tâches",
                    "Mémoire p50/p99/max",
                    "Budget",
                    "Exclues",
                    "Progrès durable",
                    "Fenêtre écoulée",
                    "Résumé exact publié",
                    "Contrat lié",
                ),
                [
                    (
                        row.get("run"),
                        row.get("tasks_total", "n.d."),
                        f"{_fmt(row.get('estimated_gib_p50'), 2)} / "
                        f"{_fmt(row.get('estimated_gib_p99'), 2)} / "
                        f"{_fmt(row.get('estimated_gib_max'), 2)} GiB",
                        f"{_fmt(row.get('memory_budget_gib'), 1)} GiB",
                        row.get("oversized_excluded", 0),
                        f"{row.get('progress_complete_unique', 0)}/"
                        f"{row.get('runnable_tasks', 0)} "
                        f"({_pct(row.get('progress_fraction_of_runnable'))})",
                        (
                            f"{_finite_float(row.get('progress_wall_elapsed_seconds')) / 3600:.2f} h"
                            if math.isfinite(
                                _finite_float(row.get("progress_wall_elapsed_seconds"))
                            )
                            else "n.d."
                        ),
                        "oui" if row.get("exact_summary_published") else "non",
                        "oui" if row.get("evaluation_contract_matches") else "non/n.d.",
                    )
                    for row in wasserstein_feasibility
                ],
            ),
            "",
            "Les quantiles mémoire proviennent du scan de support complet avant calcul. Le "
            "progrès est le nombre de clés `(dataset, case_name)` complètes et durables dans "
            "`progress.jsonl`, rapporté aux tâches exécutables après exclusions. La fenêtre "
            "écoulée est mesurée entre la création du contrat exact et la dernière écriture du "
            "journal ; la somme des temps CPU/tâche reste disponible dans le CSV. **Aucune "
            "moyenne des distances du journal partiel n’est calculée ni publiée comme résultat.**",
            "",
            "### Artefacts d’audit recopiés avec empreintes",
            "",
            "Les scans de support, listes d’exclusions et contrats exacts disponibles sont "
            "recopiés sous `wasserstein_audit/<run>/`. Le journal `progress.jsonl` de la "
            "baseline est également conservé comme preuve de progression durable ; sa "
            "présence n’autorise aucune statistique partielle. Les SHA-256 source et copie "
            "doivent coïncider.",
            "",
            md_table(
                ("Run", "Artefact", "Disponible", "Copie", "Taille", "SHA-256", "Vérifiée"),
                [
                    (
                        row.get("run"),
                        row.get("artifact"),
                        "oui" if row.get("available") else "non",
                        row.get("copied_path", "—"),
                        f"{int(row.get('copied_size_bytes', 0)):,}"
                        if row.get("copied_size_bytes") is not None
                        else "—",
                        str(row.get("copied_sha256", ""))[:16] + "…"
                        if row.get("copied_sha256")
                        else "—",
                        "oui" if row.get("copy_verified") else "non/n.d.",
                    )
                    for row in wasserstein_audit_rows
                ],
            ),
            "",
            "La distance exacte est potentiellement très coûteuse car le transport dense "
            "croît avec le produit du nombre de pixels actifs des deux masques. Les résultats "
            "ne sont comparés **que sur l’intersection des cas possédant une valeur finie dans "
            "les deux modèles**. Les colonnes `baseline_finite`, `frangi_finite`, "
            "`common_finite` et `common_coverage` de `tables/paired_statistics.csv` rendent "
            "toute incomplétude visible. Une absence de valeur n’est ni remplacée par zéro ni "
            "par la distance plafonnée utilisée pendant certains diagnostics rapides.",
            "",
            "## Limites d’interprétation",
            "",
            "- Les moyennes par image donnent le même poids à chaque image, quelle que soit la surface annotée. Les agrégats macro donnent ensuite le même poids à chaque configuration, tandis que les agrégats pondérés donnent le même poids à chaque image/configuration.",
            "- Les trois variantes Khanhha (propre et deux bruits) partagent les mêmes scènes. Le total pondéré les considère comme trois conditions expérimentales, pas comme trois images indépendantes du point de vue sémantique.",
            "- Les IC bootstrap sont descriptifs de cet échantillon de test ; ils ne corrigent ni les comparaisons multiples entre jalons, ni les décalages de domaine.",
            "- Les comparaisons avec le papier ne sont pas parfaitement contrôlées : SAM 1 ViT-H, SAM 2 Hiera Large, code, poids de fondation et entraînements diffèrent.",
            "- Les illustrations sont sélectionnées par une règle sur l’IoU. Elles sont informatives mais ne remplacent pas l’inspection de toutes les prédictions, disponible dans les artefacts d’évaluation.",
            "- Un IoU faible peut provenir d’une erreur de classe, d’une épaisseur de masque différente ou d’une annotation discutable ; les cartes FP/FN aident à distinguer ces situations sans prétendre trancher automatiquement la qualité de l’annotation.",
            "",
            "## Fichiers tabulaires et reproductibilité",
            "",
            "- `tables/metric_summary.csv` : métriques, écarts-types et couvertures pour les six datasets et les deux agrégations.",
            "- `tables/milestone_deltas_vs_baseline.csv` : deltas de chaque jalon contre la baseline best.",
            "- `tables/per_image_all_milestones.csv` : jointure exhaustive image par image pour les cinq jalons.",
            "- `tables/paired_statistics.csv` : statistiques appariées, IC bootstrap et couverture Wasserstein du jalon principal.",
            "- `tables/wasserstein_feasibility.csv` : scan mémoire dense, exclusions et progression durable, sans moyenne partielle.",
            "- `tables/wasserstein_threshold_feasibility.csv` : neuf seuils mémoire communs aux six runs, support strict et ETA de planification.",
            "- `tables/wasserstein_audit_files.csv` et `wasserstein_audit/<run>/` : inventaire, copies et empreintes des preuves de faisabilité exactes.",
            "- `tables/delta_iou_quantiles.csv` : quantiles des deltas du jalon principal.",
            "- `tables/training_history.csv` et `tables/best_validation.csv` : historiques et sélection validation.",
            "- `tables/paper_reference_values.csv` : transcription traçable des Tables 1, 2 et 6 du papier.",
            "- `tables/paper_comparison_all_runs.csv` : valeurs et deltas de chacun des six runs contre les deux modèles publiés.",
            "- `tables/checkpoint_manifest.csv` : vue tabulaire du manifeste versionné des poids et vérification locale.",
            "- `tables/selected_cases.csv` et `tables/prompt_manifest.csv` : provenance des illustrations et prompts.",
            "- `report_manifest.json` : arguments, runs trouvés, manifeste des checkpoints, avertissements et SHA-256 de toutes les sorties (hors manifeste lui-même).",
            "",
        ]
    )
    if warnings:
        sections.extend(
            [
                "## Avertissements de complétude",
                "",
                *[f"- {warning}" for warning in sorted(set(warnings))],
                "",
            ]
        )
    else:
        sections.extend(["## Avertissements de complétude", "", "Aucun avertissement.", ""])
    return "\n".join(sections).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions doit être positif")
    if args.tie_tolerance < 0:
        raise ValueError("--tie-tolerance ne peut pas être négatif")
    if args.copy_prompts < 0:
        raise ValueError("--copy-prompts ne peut pas être négatif")
    if args.panel_dpi < 72:
        raise ValueError("--panel-dpi doit être au moins 72")

    args.data_root = args.data_root.expanduser().resolve()
    args.prompt_root = args.prompt_root.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    _prepare_managed_output(args.output)
    (args.output / "tables").mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    checkpoint_manifest_path, checkpoint_manifest, checkpoint_rows = (
        load_checkpoint_manifest(warnings)
    )
    wasserstein_manifest_path, wasserstein_manifest, wasserstein_threshold_rows = (
        load_wasserstein_feasibility_manifest(warnings)
    )
    specs = resolve_run_specs(args.artifact_root)
    loaded = [load_run(spec, warnings) for spec in specs]
    runs = [run for run in loaded if run is not None]
    if not runs:
        raise FileNotFoundError(
            f"Aucune évaluation exploitable sous {args.artifact_root}. "
            "Attendre la publication des summary/per_image CSV."
        )
    runs_by_key = {run.spec.key: run for run in runs}
    baseline = runs_by_key.get("baseline_best")
    frangi_runs = [runs_by_key[key] for key, _, _ in MILESTONES if key in runs_by_key]
    primary = runs_by_key.get(args.primary_milestone)
    if primary is None and frangi_runs:
        # Fallback is explicit and deterministic.  It does not search test IoU.
        primary = frangi_runs[0]
        warnings.append(
            f"Jalon principal {args.primary_milestone} absent; repli sur {primary.spec.key} "
            "(premier jalon chronologique disponible, sans sélection sur le test)."
        )

    expected_keys = {"baseline_best", *(item[0] for item in MILESTONES)}
    missing_keys = sorted(expected_keys - set(runs_by_key))
    incomplete_runs = [run.spec.key for run in runs if len(run.summaries) != len(DATASETS)]
    if args.strict and (missing_keys or incomplete_runs):
        raise RuntimeError(
            f"Mode strict: runs absents={missing_keys}, runs incomplets={incomplete_runs}"
        )

    metric_rows = metric_summary_rows(runs)
    delta_rows = milestone_delta_rows(baseline, frangi_runs)
    per_image_rows = per_image_comparison_rows(baseline, frangi_runs)
    paired_rows = paired_statistics(
        baseline,
        primary,
        repetitions=args.bootstrap_repetitions,
        base_seed=args.bootstrap_seed,
        tie_tolerance=args.tie_tolerance,
    )
    quantile_rows = delta_quantile_rows(baseline, primary)
    histories, combined_history = load_histories(args.artifact_root, warnings)
    best_validation = best_validation_rows(histories)
    robustness_rows = robustness_generalization_rows(runs)
    wasserstein_feasibility = wasserstein_feasibility_rows(runs, warnings)
    wasserstein_audit_rows = copy_wasserstein_audit(runs, args.output, warnings)
    paper_rows = paper_reference_rows()
    paper_comparisons = paper_comparison_rows(runs)

    _write_csv(args.output / "tables" / "metric_summary.csv", metric_rows)
    _write_csv(args.output / "tables" / "milestone_deltas_vs_baseline.csv", delta_rows)
    _write_csv(args.output / "tables" / "per_image_all_milestones.csv", per_image_rows)
    _write_csv(args.output / "tables" / "paired_statistics.csv", paired_rows)
    _write_csv(args.output / "tables" / "delta_iou_quantiles.csv", quantile_rows)
    _write_csv(args.output / "tables" / "training_history.csv", combined_history)
    _write_csv(args.output / "tables" / "best_validation.csv", best_validation)
    _write_csv(args.output / "tables" / "robustness_generalization.csv", robustness_rows)
    _write_csv(
        args.output / "tables" / "wasserstein_feasibility.csv",
        wasserstein_feasibility,
    )
    _write_csv(
        args.output / "tables" / "wasserstein_threshold_feasibility.csv",
        wasserstein_threshold_rows,
    )
    _write_csv(
        args.output / "tables" / "wasserstein_audit_files.csv",
        wasserstein_audit_rows,
    )
    _write_csv(args.output / "tables" / "paper_reference_values.csv", paper_rows)
    _write_csv(
        args.output / "tables" / "paper_comparison_all_runs.csv",
        paper_comparisons,
    )
    _write_csv(args.output / "tables" / "checkpoint_manifest.csv", checkpoint_rows)

    figures: dict[str, Path | None] = {
        "milestones": plot_milestone_evolution(baseline, frangi_runs, args.output),
        "heatmap": plot_delta_heatmap(baseline, frangi_runs, args.output),
        "macro": plot_macro_metrics(runs, args.output),
        "paper": plot_paper_comparison(baseline, primary, args.output),
        "paper_delta": plot_paper_delta_heatmap(paper_comparisons, args.output),
        "distribution": plot_delta_distributions(baseline, primary, args.output),
        "scatter": plot_iou_scatter(baseline, primary, args.output),
        "wasserstein": plot_wasserstein_intersection(baseline, primary, args.output),
        "training": plot_training_curves(histories, args.output),
    }

    selections = select_cases(
        baseline,
        primary,
        data_root=args.data_root,
        warnings=warnings,
    )
    panel_rows: list[dict[str, Any]] = []
    if baseline is not None and primary is not None:
        panel_rows = generate_case_panels(
            selections,
            baseline=baseline,
            primary=primary,
            data_root=args.data_root,
            prompt_root=args.prompt_root,
            output=args.output,
            dpi=args.panel_dpi,
            warnings=warnings,
        )
    _write_csv(args.output / "tables" / "selected_cases.csv", panel_rows)

    prompt_manifest = copy_prompt_examples(
        selections,
        prompt_root=args.prompt_root,
        output=args.output,
        requested=args.copy_prompts,
        warnings=warnings,
    )
    figures["prompts"] = plot_prompt_gallery(prompt_manifest, args.output, warnings)

    report = build_report(
        args=args,
        runs=runs,
        baseline=baseline,
        frangi_runs=frangi_runs,
        primary=primary,
        metric_rows=metric_rows,
        delta_rows=delta_rows,
        paper_comparisons=paper_comparisons,
        paired_rows=paired_rows,
        best_validation=best_validation,
        robustness_rows=robustness_rows,
        wasserstein_feasibility=wasserstein_feasibility,
        wasserstein_manifest_path=wasserstein_manifest_path,
        wasserstein_manifest=wasserstein_manifest,
        wasserstein_threshold_rows=wasserstein_threshold_rows,
        wasserstein_audit_rows=wasserstein_audit_rows,
        quantile_rows=quantile_rows,
        panel_rows=panel_rows,
        prompt_manifest=prompt_manifest,
        checkpoint_manifest_path=checkpoint_manifest_path,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_rows=checkpoint_rows,
        figures=figures,
        warnings=warnings,
    )
    report_path = args.output / "RAPPORT_FRANGI_MILESTONES.md"
    _atomic_text(report_path, report)

    managed_output_paths = {report_path}
    for directory_name in (
        "tables",
        "figures",
        "prompts_npy",
        "wasserstein_audit",
    ):
        directory = args.output / directory_name
        if directory.is_dir():
            managed_output_paths.update(
                path
                for path in directory.rglob("*")
                if path.is_file() and not path.name.endswith(".tmp")
            )
    output_files = [
        _file_record(path, relative_to=args.output)
        for path in sorted(managed_output_paths)
    ]
    manifest = {
        "format_version": 1,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "command_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "primary_milestone_requested": args.primary_milestone,
        "primary_milestone_used": primary.spec.key if primary else None,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "generator": _file_record(Path(__file__).resolve()),
        "checkpoint_manifest": {
            **(
                _file_record(checkpoint_manifest_path)
                if checkpoint_manifest_path.is_file()
                else {"path": str(checkpoint_manifest_path), "missing": True}
            ),
            "format_version": checkpoint_manifest.get("format_version"),
            "verified_at_utc": checkpoint_manifest.get("verified_at_utc"),
            "entries": checkpoint_rows,
        },
        "wasserstein_feasibility_manifest": {
            **(
                _file_record(wasserstein_manifest_path)
                if wasserstein_manifest_path.is_file()
                else {"path": str(wasserstein_manifest_path), "missing": True}
            ),
            "format_version": wasserstein_manifest.get("format_version"),
            "generated_at_utc": wasserstein_manifest.get("generated_at_utc"),
            "metric": wasserstein_manifest.get("metric"),
            "thresholds": len(wasserstein_threshold_rows),
            "live_attempt": wasserstein_manifest.get("live_attempt"),
            "publication_policy": wasserstein_manifest.get("publication_policy"),
        },
        "wasserstein_audit": {
            "files_expected_or_copied": len(wasserstein_audit_rows),
            "files_copied": sum(bool(row.get("available")) for row in wasserstein_audit_rows),
            "copies_verified": sum(
                bool(row.get("copy_verified")) for row in wasserstein_audit_rows
            ),
            "files": wasserstein_audit_rows,
            "partial_mean_published": False,
        },
        "runs": [
            {
                "key": run.spec.key,
                "label": run.spec.label,
                "epoch": run.spec.epoch,
                "path": run.spec.path,
                "datasets": sorted(run.summaries),
                "per_image_cases": {
                    dataset: len(run.per_image.get(dataset, {})) for dataset in DATASETS
                },
                "exact_wasserstein_root": run.exact_root,
                "exact_wasserstein_status": run.exact_status,
            }
            for run in runs
        ],
        "figures": {
            key: _relative_link(args.output, value) for key, value in figures.items()
        },
        "qualitative_panels_generated": sum(bool(row.get("generated")) for row in panel_rows),
        "prompts_copied": len(prompt_manifest),
        "output_files_hashed": len(output_files),
        "output_bytes_hashed": sum(record["size_bytes"] for record in output_files),
        "outputs": output_files,
        "warnings": sorted(set(warnings)),
    }
    _write_json(args.output / "report_manifest.json", manifest)
    print(f"Rapport: {report_path}")
    print(f"Runs exploités: {len(runs)}/6; panneaux: {manifest['qualitative_panels_generated']}; prompts: {len(prompt_manifest)}")
    if warnings:
        print(f"Avertissements: {len(set(warnings))} (voir le rapport et report_manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
