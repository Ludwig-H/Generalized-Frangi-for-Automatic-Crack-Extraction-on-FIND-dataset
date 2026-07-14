#!/usr/bin/env python3
"""Compute exact direct-mask Wasserstein distances from saved predictions.

Inference and optimal transport are deliberately separated: SAM 2 writes
binary prediction PNGs on the GPU, then POT's single-threaded exact network
simplex is parallelized across the VM's CPU cores.  A memory-aware scheduler
bounds the sum of dense cost/transport matrices running concurrently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

# Dense cost construction can otherwise start a full BLAS thread pool inside
# every process.  Exact EMD itself is single-threaded, so one BLAS thread per
# process prevents severe oversubscription on the 48-vCPU experiment VM.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import cv2
import numpy as np
import ot
from PIL import Image

from cracksam2.data import read_sample_list, resolve_sample_paths
from cracksam2.metrics import wasserstein_mask_distance


LIST_ROOT = Path(__file__).parent / "CrackSAM" / "CrackSAM" / "lists"
# Measured with POT 0.9.7 on the target VM: cost construction plus the network
# simplex peaks near 46 bytes per dense transport arc.  Round upward so the
# scheduler leaves enough room for worker imports and image buffers.
MATRIX_BYTES_PER_ENTRY = 48


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    list_file: Path


@dataclass(frozen=True)
class ExactTask:
    dataset: str
    case_name: str
    prediction_path: str
    target_path: str
    prediction_points: int
    target_points: int
    cost_entries: int
    estimated_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=(
            "khanhha_original",
            "khanhha_noisy1",
            "khanhha_noisy2",
            "road420",
            "facade390",
            "concrete3k",
        ),
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--memory-budget-gb", type=float, default=140.0)
    parser.add_argument(
        "--skip-oversized",
        action="store_true",
        help=(
            "Compute every exact case that fits the memory budget and publish "
            "explicitly incomplete summaries for larger cases. By default any "
            "oversized case aborts before computation."
        ),
    )
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def dataset_specs(data_root: Path) -> list[DatasetSpec]:
    khanhha_list = LIST_ROOT / "lists_khanhha" / "test_vol.txt"
    return [
        DatasetSpec("khanhha_original", data_root / "khanhha" / "test", khanhha_list),
        DatasetSpec("khanhha_noisy1", data_root / "khanhha" / "test", khanhha_list),
        DatasetSpec("khanhha_noisy2", data_root / "khanhha" / "test", khanhha_list),
        DatasetSpec(
            "road420", data_root / "road420", LIST_ROOT / "lists_road420" / "test_vol.txt"
        ),
        DatasetSpec(
            "facade390",
            data_root / "facade390",
            LIST_ROOT / "lists_facade390" / "test_vol.txt",
        ),
        DatasetSpec(
            "concrete3k",
            data_root / "concrete3k",
            LIST_ROOT / "lists_concrete3k" / "test_vol.txt",
        ),
    ]


def _prediction_path(root: Path, case_name: str) -> Path:
    relative = Path(case_name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe case name: {case_name!r}")
    return root / relative.with_suffix(".png")


def _load_prediction(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        prediction = np.asarray(image.convert("L"), dtype=np.uint8) > 127
    if prediction.shape != (448, 448):
        prediction = cv2.resize(
            prediction.astype(np.uint8), (448, 448), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return prediction


def _load_target(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == 4 and np.unique(array[..., 3]).size > 1:
            gray = array[..., 3]
        else:
            gray = np.asarray(image.convert("L"))
    target = gray.astype(np.float32) / 255.0 > 0.5
    if target.shape != (448, 448):
        target = cv2.resize(
            target.astype(np.uint8), (448, 448), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _read_completed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return completed
    size = path.stat().st_size
    valid_end = 0
    with path.open("rb") as source:
        while True:
            line_start = source.tell()
            raw = source.readline()
            if not raw:
                break
            line_end = source.tell()
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                if line_end == size and not raw.endswith(b"\n"):
                    valid_end = line_start
                    break
                raise
            if row.get("status") == "complete":
                completed[(str(row["dataset"]), str(row["case_name"]))] = row
            valid_end = line_end
    if valid_end < size:
        with path.open("r+b") as output:
            output.truncate(valid_end)
            output.flush()
            os.fsync(output.fileno())
    return completed


def scan_tasks(
    specs: Iterable[DatasetSpec],
    evaluation_root: Path,
    max_cases: int | None,
) -> list[ExactTask]:
    tasks: list[ExactTask] = []
    for spec in specs:
        names = read_sample_list(spec.list_file)
        if max_cases is not None:
            names = names[:max_cases]
        prediction_root = evaluation_root / spec.name / "predictions"
        for case_name in names:
            prediction_path = _prediction_path(prediction_root, case_name)
            if not prediction_path.is_file():
                raise FileNotFoundError(f"Prediction missing: {prediction_path}")
            _, target_path = resolve_sample_paths(
                spec.root, case_name, split="test_vol"
            )
            prediction_points = int(np.count_nonzero(_load_prediction(prediction_path)))
            target_points = int(np.count_nonzero(_load_target(target_path)))
            cost_entries = prediction_points * target_points
            estimated_bytes = cost_entries * MATRIX_BYTES_PER_ENTRY
            tasks.append(
                ExactTask(
                    dataset=spec.name,
                    case_name=case_name,
                    prediction_path=str(prediction_path),
                    target_path=str(target_path),
                    prediction_points=prediction_points,
                    target_points=target_points,
                    cost_entries=cost_entries,
                    estimated_bytes=estimated_bytes,
                )
            )
    return tasks


def scan_summary(tasks: list[ExactTask]) -> dict[str, Any]:
    def quantiles(values: list[int]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            name: float(np.quantile(array, probability))
            for name, probability in (
                ("min", 0.0),
                ("p50", 0.5),
                ("p90", 0.9),
                ("p95", 0.95),
                ("p99", 0.99),
                ("max", 1.0),
            )
        }

    return {
        "tasks": len(tasks),
        "matrix_bytes_per_entry_estimate": MATRIX_BYTES_PER_ENTRY,
        "blas_threads_per_worker": 1,
        "prediction_points": quantiles([task.prediction_points for task in tasks]),
        "target_points": quantiles([task.target_points for task in tasks]),
        "cost_entries": quantiles([task.cost_entries for task in tasks]),
        "estimated_gib": quantiles(
            [task.estimated_bytes / (1024**3) for task in tasks]
        ),
        "empty_prediction": sum(task.prediction_points == 0 for task in tasks),
        "empty_target": sum(task.target_points == 0 for task in tasks),
    }


def _exact_worker(task: ExactTask) -> dict[str, Any]:
    started = time.perf_counter()
    prediction = _load_prediction(task.prediction_path).astype(np.float64)
    target = _load_target(task.target_path).astype(np.float64)
    distance = wasserstein_mask_distance(prediction, target, max_points=None)
    return {
        "status": "complete",
        "dataset": task.dataset,
        "case_name": task.case_name,
        "wasserstein": float(distance),
        "prediction_points": task.prediction_points,
        "target_points": task.target_points,
        "cost_entries": task.cost_entries,
        "estimated_bytes": task.estimated_bytes,
        "seconds": time.perf_counter() - started,
    }


def run_tasks(
    tasks: list[ExactTask],
    *,
    workers: int,
    memory_budget: int,
    journal: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    completed = _read_completed(journal)
    pending = [
        task for task in tasks if (task.dataset, task.case_name) not in completed
    ]
    pending.sort(key=lambda task: task.estimated_bytes, reverse=True)
    oversized = [task for task in pending if task.estimated_bytes > memory_budget]
    if oversized:
        worst = oversized[0]
        raise MemoryError(
            "At least one exact dense transport problem exceeds the complete memory "
            f"budget: {worst.dataset}/{worst.case_name}, estimated "
            f"{worst.estimated_bytes / 1024**3:.2f} GiB > "
            f"{memory_budget / 1024**3:.2f} GiB"
        )

    active: dict[Future, tuple[ExactTask, int]] = {}
    active_bytes = 0
    completed_count = len(completed)
    total = len(tasks)
    # ``spawn`` avoids inheriting OpenCV/POT/BLAS thread state into the workers.
    # This is also safe when the scan is launched after a CUDA inference process.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        while pending or active:
            submitted = False
            index = 0
            while len(active) < workers and index < len(pending):
                task = pending[index]
                required = max(task.estimated_bytes, 1)
                if not active or active_bytes + required <= memory_budget:
                    pending.pop(index)
                    future = executor.submit(_exact_worker, task)
                    active[future] = (task, required)
                    active_bytes += required
                    submitted = True
                else:
                    index += 1
            if not active:
                raise RuntimeError("Memory-aware scheduler made no progress")
            if submitted and len(active) < workers and pending:
                # No other pending task fits; wait for memory to be released.
                pass
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task, reserved = active.pop(future)
                active_bytes -= reserved
                try:
                    row = future.result()
                except BaseException as exc:
                    _append_jsonl(
                        journal,
                        {
                            "status": "failed",
                            "dataset": task.dataset,
                            "case_name": task.case_name,
                            "error": repr(exc),
                        },
                    )
                    raise
                _append_jsonl(journal, row)
                completed[(task.dataset, task.case_name)] = row
                completed_count += 1
                print(
                    f"[{completed_count}/{total}] {task.dataset}/{task.case_name}: "
                    f"W={row['wasserstein']:.6f} in {row['seconds']:.2f}s; "
                    f"active={len(active)}, reserved={active_bytes / 1024**3:.2f} GiB",
                    flush=True,
                )
    return completed


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish_results(
    specs: list[DatasetSpec],
    evaluation_root: Path,
    output_root: Path,
    completed: dict[tuple[str, str], dict[str, Any]],
    *,
    allow_incomplete: bool = False,
) -> None:
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        source_rows = _read_csv(evaluation_root / spec.name / "per_image.csv")
        rows: list[dict[str, Any]] = []
        for source in source_rows:
            key = (spec.name, source["case_name"])
            if key not in completed and not allow_incomplete:
                raise RuntimeError(f"Exact result missing for {key}")
            row = dict(source)
            row["wasserstein"] = (
                completed[key]["wasserstein"] if key in completed else None
            )
            rows.append(row)
        result_dir = output_root / spec.name
        _write_csv(result_dir / "per_image.csv", rows)
        summary: dict[str, Any] = {
            "dataset": spec.name,
            "samples": len(rows),
            "wasserstein_exact": True,
            "wasserstein_max_points": None,
        }
        for metric in ("precision", "recall", "dice", "iou", "wasserstein"):
            values = np.asarray(
                [
                    np.nan
                    if row[metric] in (None, "")
                    else float(row[metric])
                    for row in rows
                ],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            summary[metric] = float(np.mean(finite)) if finite.size else None
            summary[f"{metric}_std"] = (
                float(np.std(finite)) if finite.size else None
            )
            summary[f"{metric}_finite_samples"] = int(finite.size)
        summary["wasserstein_complete"] = (
            summary["wasserstein_finite_samples"] == len(rows)
        )
        summary["wasserstein_missing_samples"] = (
            len(rows) - summary["wasserstein_finite_samples"]
        )
        summaries.append(summary)
        _write_json(result_dir / "summary.json", summary)
    _write_csv(output_root / "summary.csv", summaries)
    _write_json(output_root / "summary.json", summaries)


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.memory_budget_gb <= 0:
        raise ValueError("--memory-budget-gb must be positive")
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    args.evaluation_root = args.evaluation_root.expanduser().resolve()
    output_root = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.evaluation_root / "wasserstein_exact"
    )
    specs = dataset_specs(args.data_root.expanduser())
    if args.datasets:
        selected = set(args.datasets)
        specs = [spec for spec in specs if spec.name in selected]

    evaluation_contract = args.evaluation_root / "evaluation_contract.json"
    if not evaluation_contract.is_file():
        raise FileNotFoundError(f"Evaluation contract missing: {evaluation_contract}")
    contract = {
        "format_version": 1,
        "algorithm": "POT ot.emd2 exact dense Euclidean direct-mask",
        "pot_version": ot.__version__,
        "evaluation_contract_sha256": _sha256(evaluation_contract),
        "datasets": [spec.name for spec in specs],
        "max_cases": args.max_cases,
        "matrix_bytes_per_entry_estimate": MATRIX_BYTES_PER_ENTRY,
        "blas_threads_per_worker": 1,
        "memory_budget_gb": args.memory_budget_gb,
        "oversized_policy": "skip" if args.skip_oversized else "error",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = output_root / "exact_wasserstein_contract.json"
    if contract_path.is_file():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != contract:
            raise RuntimeError("Exact Wasserstein output contract mismatch")
    else:
        _write_json(contract_path, contract)

    tasks = scan_tasks(specs, args.evaluation_root, args.max_cases)
    summary = scan_summary(tasks)
    _write_json(output_root / "support_scan.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    memory_budget = int(args.memory_budget_gb * 1024**3)
    oversized = [task for task in tasks if task.estimated_bytes > memory_budget]
    oversized.sort(key=lambda task: task.estimated_bytes, reverse=True)
    _write_json(
        output_root / "oversized.json",
        {
            "count": len(oversized),
            "memory_budget_bytes": memory_budget,
            "tasks": [asdict(task) for task in oversized],
        },
    )
    if args.scan_only:
        return 0

    if oversized and not args.skip_oversized:
        worst = oversized[0]
        raise MemoryError(
            f"{len(oversized)} exact dense transport problem(s) exceed the "
            f"memory budget; worst is {worst.dataset}/{worst.case_name}, "
            f"estimated {worst.estimated_bytes / 1024**3:.2f} GiB > "
            f"{args.memory_budget_gb:.2f} GiB. Re-run with --skip-oversized "
            "only if explicitly incomplete summaries are acceptable."
        )
    oversized_keys = {(task.dataset, task.case_name) for task in oversized}
    runnable = [
        task for task in tasks if (task.dataset, task.case_name) not in oversized_keys
    ]

    completed = run_tasks(
        runnable,
        workers=min(args.workers, os.cpu_count() or args.workers),
        memory_budget=memory_budget,
        journal=output_root / "progress.jsonl",
    )
    publish_results(
        specs,
        args.evaluation_root,
        output_root,
        completed,
        allow_incomplete=bool(oversized),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
