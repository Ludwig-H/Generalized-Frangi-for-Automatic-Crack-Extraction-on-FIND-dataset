from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from protocol.build_smoke_protocol import build_smoke_protocol  # noqa: E402


def _write_source_protocol(
    root: Path, *, groups_per_fold: int = 3
) -> tuple[Path, Path]:
    names: list[str] = []
    rows: list[dict[str, object]] = []
    for round_index in range(groups_per_fold):
        for fold in range(5):
            group = f"source-{fold}-{round_index}"
            for crop in range(2):
                name = f"{group}-crop-{crop}.png"
                names.append(name)
                rows.append(
                    {
                        "sample_name": name,
                        "source_family": "synthetic",
                        "physical_group": group,
                        "oof_fold": fold,
                    }
                )
    train_list = root / "train.txt"
    fold_csv = root / "folds.csv"
    train_list.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    with fold_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return train_list, fold_csv


def test_smoke_protocol_is_deterministic_group_safe_and_explicitly_non_scientific(
    tmp_path: Path,
) -> None:
    train_list, fold_csv = _write_source_protocol(tmp_path)
    first = build_smoke_protocol(
        train_list, fold_csv, tmp_path / "first", samples_per_fold=2
    )
    second = build_smoke_protocol(
        train_list, fold_csv, tmp_path / "second", samples_per_fold=2
    )

    assert first == second
    assert first["scientific_result"] is False
    assert first["sample_count"] == 10
    assert first["physical_group_count"] == 10
    assert first["fold_sample_counts"] == {str(fold): 2 for fold in range(5)}
    assert (tmp_path / "first" / "train.txt").read_bytes() == (
        tmp_path / "second" / "train.txt"
    ).read_bytes()
    with (tmp_path / "first" / "train_group_folds.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert len({row["physical_group"] for row in rows}) == 10
    assert json.loads((tmp_path / "first" / "manifest.json").read_text()) == first


def test_smoke_protocol_refuses_insufficient_distinct_groups(tmp_path: Path) -> None:
    train_list, fold_csv = _write_source_protocol(tmp_path, groups_per_fold=1)
    with pytest.raises(ValueError, match="Not enough distinct groups"):
        build_smoke_protocol(
            train_list, fold_csv, tmp_path / "smoke", samples_per_fold=2
        )
