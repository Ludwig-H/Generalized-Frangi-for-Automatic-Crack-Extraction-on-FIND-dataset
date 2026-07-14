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
        prediction_sha256=exact._sha256(prediction),
        target_sha256=exact._sha256(target),
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


def test_exact_worker_rejects_a_prediction_changed_after_scan(tmp_path: Path) -> None:
    task = _task(tmp_path)
    _save_mask(Path(task.prediction_path), [(11, 21)])

    with pytest.raises(RuntimeError, match="Prediction changed after exact scan"):
        exact._exact_worker(task)


def test_target_loader_uses_nonconstant_rgba_alpha_as_mask(tmp_path: Path) -> None:
    rgba = np.full((448, 448, 4), 255, dtype=np.uint8)
    rgba[..., 3] = 0
    rgba[17, 23, 3] = 255
    path = tmp_path / "target.png"
    Image.fromarray(rgba).save(path)

    target = exact._load_target(path)

    assert target.sum() == 1
    assert target[17, 23]


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


def test_completed_journal_repairs_a_valid_final_line_without_newline(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "progress.jsonl"
    first = {
        "status": "complete",
        "dataset": "mini",
        "case_name": "first.png",
        "wasserstein": 1.0,
    }
    journal.write_text(json.dumps(first), encoding="utf-8")

    completed = exact._read_completed(journal)
    assert completed[("mini", "first.png")]["wasserstein"] == 1.0
    assert journal.read_bytes().endswith(b"\n")

    second = {**first, "case_name": "second.png", "wasserstein": 2.0}
    exact._append_jsonl(journal, second)
    repaired = exact._read_completed(journal)

    assert repaired[("mini", "second.png")]["wasserstein"] == 2.0
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 2


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


def test_run_tasks_rejects_completed_rows_for_changed_mask_content(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    journal = tmp_path / "progress.jsonl"
    completed_row = exact._exact_worker(task)
    exact._append_jsonl(journal, completed_row)
    changed = exact.ExactTask(
        **{**task.__dict__, "prediction_sha256": "0" * 64}
    )

    with pytest.raises(RuntimeError, match="task identity mismatch"):
        exact.run_tasks(
            [changed],
            workers=1,
            memory_budget=1024,
            journal=journal,
        )


def _write_evaluation_contract(
    path: Path,
    spec: exact.DatasetSpec,
    *,
    save_predictions: bool = True,
    max_samples: int | None = None,
    root: Path | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "save_predictions": save_predictions,
                "max_samples": max_samples,
                "datasets": [
                    {
                        "name": spec.name,
                        "root": str((root or spec.root).resolve()),
                        "list_file": {
                            **exact._file_identity(spec.list_file),
                            "path": str(spec.list_file.resolve()),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_evaluation_contract_binds_data_root_and_list(tmp_path: Path) -> None:
    list_file = tmp_path / "test_vol.txt"
    list_file.write_text("case.png\n", encoding="utf-8")
    spec = exact.DatasetSpec("mini", tmp_path / "data", list_file)
    contract_path = tmp_path / "evaluation_contract.json"
    _write_evaluation_contract(contract_path, spec)

    observed = exact.validate_evaluation_contract(contract_path, [spec], None)

    assert observed["datasets"][0]["name"] == "mini"


def test_evaluation_contract_rejects_a_different_data_root(tmp_path: Path) -> None:
    list_file = tmp_path / "test_vol.txt"
    list_file.write_text("case.png\n", encoding="utf-8")
    spec = exact.DatasetSpec("mini", tmp_path / "data", list_file)
    contract_path = tmp_path / "evaluation_contract.json"
    _write_evaluation_contract(contract_path, spec, root=tmp_path / "other-data")

    with pytest.raises(ValueError, match="data root mismatch"):
        exact.validate_evaluation_contract(contract_path, [spec], None)


def test_evaluation_contract_requires_saved_predictions(tmp_path: Path) -> None:
    list_file = tmp_path / "test_vol.txt"
    list_file.write_text("case.png\n", encoding="utf-8")
    spec = exact.DatasetSpec("mini", tmp_path / "data", list_file)
    contract_path = tmp_path / "evaluation_contract.json"
    _write_evaluation_contract(contract_path, spec, save_predictions=False)

    with pytest.raises(ValueError, match="--save-predictions"):
        exact.validate_evaluation_contract(contract_path, [spec], None)


def test_evaluation_contract_rejects_exact_cases_beyond_smoke_run(
    tmp_path: Path,
) -> None:
    list_file = tmp_path / "test_vol.txt"
    list_file.write_text("case.png\n", encoding="utf-8")
    spec = exact.DatasetSpec("mini", tmp_path / "data", list_file)
    contract_path = tmp_path / "evaluation_contract.json"
    _write_evaluation_contract(contract_path, spec, max_samples=1)

    with pytest.raises(ValueError, match="exceeds the evaluation selection"):
        exact.validate_evaluation_contract(contract_path, [spec], None)


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


def test_publish_results_marks_an_allowed_missing_exact_case(
    tmp_path: Path,
) -> None:
    evaluation_root = tmp_path / "evaluation"
    dataset_root = evaluation_root / "mini"
    dataset_root.mkdir(parents=True)
    source_row = {
        "case_name": "oversized.png",
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
    output_root = tmp_path / "exact"
    exact.publish_results(
        [spec], evaluation_root, output_root, {}, allow_incomplete=True
    )

    summary = json.loads(
        (output_root / "mini" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["wasserstein_complete"] is False
    assert summary["wasserstein_missing_samples"] == 1
    assert summary["wasserstein_finite_samples"] == 0


def test_publish_results_limits_rows_for_max_cases(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "evaluation"
    dataset_root = evaluation_root / "mini"
    dataset_root.mkdir(parents=True)
    source_rows = [
        {
            "case_name": f"case-{index}.png",
            "precision": 0.8,
            "recall": 0.6,
            "dice": 0.7,
            "iou": 0.5,
            "wasserstein": "",
            "inference_seconds": 0.01,
        }
        for index in range(2)
    ]
    with (dataset_root / "per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    spec = exact.DatasetSpec("mini", tmp_path / "data", tmp_path / "list.txt")
    completed = {
        ("mini", "case-0.png"): {
            "status": "complete",
            "dataset": "mini",
            "case_name": "case-0.png",
            "wasserstein": 5.0,
        }
    }
    output_root = tmp_path / "exact"
    exact.publish_results(
        [spec],
        evaluation_root,
        output_root,
        completed,
        max_cases=1,
    )

    summary = json.loads(
        (output_root / "mini" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["samples"] == 1
    assert summary["wasserstein_complete"] is True
    with (output_root / "mini" / "per_image.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        assert len(list(csv.DictReader(source))) == 1
