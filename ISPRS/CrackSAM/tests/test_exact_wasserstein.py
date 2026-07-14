from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

import compute_exact_wasserstein as exact  # noqa: E402


def _save_mask(path: Path, points: list[tuple[int, int]]) -> None:
    mask = np.zeros((448, 448), dtype=np.uint8)
    for y, x in points:
        mask[y, x] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def _task(tmp_path: Path) -> exact.ExactTask:
    prediction = tmp_path / "prediction.png"
    target = tmp_path / "target.png"
    _save_mask(prediction, [(10, 20)])
    _save_mask(target, [(13, 24)])
    return exact.ExactTask(
        dataset="mini",
        case_name="case.png",
        prediction_path=str(prediction),
        target_path=str(target),
        prediction_points=1,
        target_points=1,
        cost_entries=1,
        estimated_bytes=exact.MATRIX_BYTES_PER_ENTRY,
    )


def test_exact_worker_uses_complete_support_and_euclidean_ground_cost(
    tmp_path: Path,
) -> None:
    row = exact._exact_worker(_task(tmp_path))

    assert row["status"] == "complete"
    assert row["wasserstein"] == pytest.approx(5.0)
    assert row["prediction_points"] == 1
    assert row["target_points"] == 1


def test_run_tasks_is_resumable_and_repairs_truncated_journal(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    journal = tmp_path / "progress.jsonl"

    first = exact.run_tasks(
        [task], workers=1, memory_budget=1024, journal=journal
    )
    assert first[("mini", "case.png")]["wasserstein"] == pytest.approx(5.0)
    original = journal.read_bytes()

    with journal.open("ab") as output:
        output.write(b'{"status": "complete"')
    second = exact.run_tasks(
        [task], workers=1, memory_budget=1024, journal=journal
    )

    assert second[("mini", "case.png")]["wasserstein"] == pytest.approx(5.0)
    assert journal.read_bytes() == original


def test_run_tasks_rejects_a_problem_larger_than_memory_budget(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    oversized = exact.ExactTask(
        **{**task.__dict__, "estimated_bytes": 1025}
    )

    with pytest.raises(MemoryError, match="exceeds the complete memory budget"):
        exact.run_tasks(
            [oversized],
            workers=1,
            memory_budget=1024,
            journal=tmp_path / "progress.jsonl",
        )


def test_publish_results_preserves_segmentation_metrics_and_adds_exact_distance(
    tmp_path: Path,
) -> None:
    evaluation_root = tmp_path / "evaluation"
    dataset_root = evaluation_root / "mini"
    dataset_root.mkdir(parents=True)
    source_row = {
        "case_name": "case.png",
        "precision": 0.8,
        "recall": 0.6,
        "dice": 0.7,
        "iou": 0.5,
        "wasserstein": "",
        "inference_seconds": 0.01,
    }
    with (dataset_root / "per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(source_row))
        writer.writeheader()
        writer.writerow(source_row)

    spec = exact.DatasetSpec("mini", tmp_path / "data", tmp_path / "list.txt")
    completed = {
        ("mini", "case.png"): {
            "status": "complete",
            "dataset": "mini",
            "case_name": "case.png",
            "wasserstein": 5.0,
        }
    }
    output_root = tmp_path / "exact"
    exact.publish_results([spec], evaluation_root, output_root, completed)

    summary = json.loads(
        (output_root / "mini" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["wasserstein_exact"] is True
    assert summary["wasserstein"] == pytest.approx(5.0)
    assert summary["dice"] == pytest.approx(0.7)
    with (output_root / "mini" / "per_image.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        row = next(csv.DictReader(source))
    assert float(row["wasserstein"]) == pytest.approx(5.0)
