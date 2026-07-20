from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset


CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

import evaluate_sam2 as evaluate  # noqa: E402


def _row(case_name: str, *, dice: float = 1.0) -> dict[str, object]:
    return {
        "case_name": case_name,
        "precision": 1.0,
        "recall": 1.0,
        "dice": dice,
        "iou": 1.0,
        "wasserstein": None,
        "inference_seconds": 0.01,
    }


def test_progress_is_fsynced_deduplicated_and_repairs_trailing_fragment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress = tmp_path / "progress.jsonl"
    fsync_calls: list[int] = []
    monkeypatch.setattr(
        evaluate.os, "fsync", lambda descriptor: fsync_calls.append(descriptor)
    )

    evaluate._append_progress_batch(progress, "mini", [_row("a.png")])
    evaluate._append_progress_batch(progress, "mini", [_row("a.png", dice=0.75)])
    with progress.open("ab") as output:
        output.write(b'{"format_version": 1, "rows":')

    rows = evaluate._read_progress_rows(progress, expected_dataset="mini")

    assert list(rows) == ["a.png"]
    assert rows["a.png"]["dice"] == pytest.approx(0.75)
    assert progress.read_bytes().endswith(b"\n")
    repaired_lines = progress.read_text(encoding="utf-8").splitlines()
    assert len(repaired_lines) == 2
    assert all(json.loads(line)["dataset"] == "mini" for line in repaired_lines)
    assert len(fsync_calls) == 3


def test_evaluation_contract_is_immutable_and_rejects_orphan_results(
    tmp_path: Path,
) -> None:
    contract = {
        "format_version": 1,
        "adapter_checkpoint": {"sha256": "adapter"},
        "base_checkpoint": {"sha256": "base"},
        "variant": "baseline",
        "datasets": ["mini"],
        "threshold": 0.5,
        "wasserstein": {"skip": True, "max_points": 2000},
        "max_samples": None,
        "amp_dtype": "bfloat16",
    }
    path = evaluate._ensure_evaluation_contract(tmp_path / "fresh", contract)
    assert json.loads(path.read_text(encoding="utf-8")) == contract
    assert evaluate._ensure_evaluation_contract(tmp_path / "fresh", contract) == path

    incompatible = {**contract, "threshold": 0.6}
    with pytest.raises(RuntimeError, match="incompatible contract"):
        evaluate._ensure_evaluation_contract(tmp_path / "fresh", incompatible)

    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "summary.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="without evaluation_contract"):
        evaluate._ensure_evaluation_contract(orphan, contract)


class _FakeDataset(Dataset):
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.sample_names = ["a.png", "b.png"]

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "case_name": self.sample_names[index],
            "image": torch.zeros((3, 4, 4), dtype=torch.float32),
            "mask": torch.zeros((1, 4, 4), dtype=torch.float32),
        }


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.samples_seen = 0

    def forward(
        self,
        images: torch.Tensor,
        *,
        mask_input: torch.Tensor | None,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        self.samples_seen += images.shape[0]
        return {
            "logits": torch.full(
                (images.shape[0], 1, images.shape[-2], images.shape[-1]), -10.0
            )
        }


class _FailIfCalledModel(torch.nn.Module):
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("completed evaluation should not invoke the model")


def test_evaluate_spec_resumes_remaining_cases_and_rebuilds_final_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "CrackSegmentationDataset", _FakeDataset)
    spec = evaluate.EvaluationSpec(
        "mini", tmp_path / "data", tmp_path / "list.txt"
    )
    result_dir = tmp_path / "results" / "mini"
    evaluate._append_progress_batch(
        result_dir / "progress.jsonl", "mini", [_row("a.png")]
    )
    arguments = {
        "checkpoint_variant": "baseline",
        "prompt_condition": "none",
        "output_root": tmp_path / "results",
        "device": torch.device("cpu"),
        "amp_dtype": "none",
        "batch_size": 2,
        "num_workers": 0,
        "threshold": 0.5,
        "wasserstein_max_points": 2000,
        "skip_wasserstein": True,
        "save_predictions": False,
        "max_samples": None,
    }

    model = _FakeModel()
    summary = evaluate.evaluate_spec(model, spec, **arguments)

    assert model.samples_seen == 1
    assert summary["samples"] == 2
    assert summary["wasserstein"] is None
    assert summary["wasserstein_std"] is None
    assert summary["wasserstein_finite_samples"] == 0
    assert "NaN" not in (result_dir / "summary.json").read_text(encoding="utf-8")

    (result_dir / "per_image.csv").unlink()
    (result_dir / "summary.json").unlink()
    resumed = evaluate.evaluate_spec(_FailIfCalledModel(), spec, **arguments)

    assert resumed["samples"] == 2
    assert (result_dir / "per_image.csv").is_file()
    assert (result_dir / "summary.json").is_file()
    progress_lines = (result_dir / "progress.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(progress_lines) == 2
