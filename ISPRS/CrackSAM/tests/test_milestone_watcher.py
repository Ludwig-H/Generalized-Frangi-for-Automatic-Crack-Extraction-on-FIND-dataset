from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch


CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from workflows import watch_training_milestone as watcher  # noqa: E402


def test_validation_completed_requires_the_exact_epoch(tmp_path: Path) -> None:
    path = tmp_path / "validation.csv"
    assert watcher.validation_completed(path, 20) is False

    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("epoch", "dice"))
        writer.writeheader()
        writer.writerow({"epoch": 20, "dice": 0.75})

    assert watcher.validation_completed(path, 20) is True
    assert watcher.validation_completed(path, 30) is False


def test_atomic_copy_preserves_a_loadable_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "latest.pt"
    destination = tmp_path / "milestone.pt"
    torch.save(
        {
            "format_version": 1,
            "epoch": 30,
            "next_batch": 0,
            "global_step": 123,
        },
        source,
    )

    watcher.atomic_copy(source, destination)

    epoch, batch, step, _ = watcher.load_position(destination)
    assert (epoch, batch, step) == (30, 0, 123)
    assert not list(tmp_path.glob(".*.tmp"))
