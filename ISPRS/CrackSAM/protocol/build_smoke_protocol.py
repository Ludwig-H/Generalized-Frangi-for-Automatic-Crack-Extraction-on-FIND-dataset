#!/usr/bin/env python3
"""Build a tiny, deterministic five-fold protocol for a real GPU smoke test.

The smoke protocol is deliberately separate from the scientific protocol.  It
selects one crop from each of several distinct physical groups per fold, which
is enough to exercise cache extraction, SAM 2 loading, residual training,
resume, and OOF evaluation without pretending to produce a useful estimate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

REQUIRED_COLUMNS = ("sample_name", "source_family", "physical_group", "oof_fold")
SMOKE_SCHEMA = "cracksam2.frangigraph-smoke-protocol"
SMOKE_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-list",
        type=Path,
        default=root / "cracksam_paper" / "lists" / "lists_khanhha" / "train.txt",
    )
    parser.add_argument(
        "--fold-csv",
        type=Path,
        default=root / "frangigraph_v1" / "train_group_folds.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-fold", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_sample_names(path: Path) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Training list must be non-empty and unique: {path}")
    return names


def read_fold_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(REQUIRED_COLUMNS).issubset(
            reader.fieldnames
        ):
            raise ValueError(f"Fold CSV must contain {list(REQUIRED_COLUMNS)}")
        rows: dict[str, dict[str, str]] = {}
        group_folds: dict[str, str] = {}
        for line_number, raw in enumerate(reader, start=2):
            row = {name: (raw.get(name) or "").strip() for name in REQUIRED_COLUMNS}
            if any(not row[name] for name in REQUIRED_COLUMNS):
                raise ValueError(f"Empty fold field at {path}:{line_number}")
            name = row["sample_name"]
            if name in rows:
                raise ValueError(f"Duplicate sample in fold CSV: {name!r}")
            fold = row["oof_fold"]
            if fold not in {"0", "1", "2", "3", "4"}:
                raise ValueError(f"Invalid fold {fold!r} for {name!r}")
            group = row["physical_group"]
            previous = group_folds.setdefault(group, fold)
            if previous != fold:
                raise ValueError(f"Physical group crosses folds: {group!r}")
            rows[name] = row
    if not rows:
        raise ValueError(f"Fold CSV is empty: {path}")
    return rows


def select_smoke_rows(
    ordered_names: Sequence[str],
    rows_by_name: Mapping[str, Mapping[str, str]],
    *,
    samples_per_fold: int,
) -> list[dict[str, str]]:
    """Select one crop per physical group in canonical list order."""
    if samples_per_fold <= 0:
        raise ValueError("samples_per_fold must be positive")
    missing = [name for name in ordered_names if name not in rows_by_name]
    unexpected = sorted(set(rows_by_name) - set(ordered_names))
    if missing or unexpected:
        raise ValueError(
            "Fold CSV differs from the training list; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    selected: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    used_groups: set[str] = set()
    for name in ordered_names:
        row = rows_by_name[name]
        fold = str(row["oof_fold"])
        group = str(row["physical_group"])
        if counts[fold] >= samples_per_fold or group in used_groups:
            continue
        selected.append({key: str(row[key]) for key in REQUIRED_COLUMNS})
        counts[fold] += 1
        used_groups.add(group)
        if all(counts[str(fold_index)] == samples_per_fold for fold_index in range(5)):
            break
    deficient = {
        str(fold): counts[str(fold)]
        for fold in range(5)
        if counts[str(fold)] != samples_per_fold
    }
    if deficient:
        raise ValueError(f"Not enough distinct groups for smoke folds: {deficient}")
    return selected


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_smoke_protocol(
    train_list: Path,
    fold_csv: Path,
    output: Path,
    *,
    samples_per_fold: int,
) -> dict[str, object]:
    names = read_sample_names(train_list)
    rows = select_smoke_rows(
        names,
        read_fold_rows(fold_csv),
        samples_per_fold=samples_per_fold,
    )
    selected_names = [row["sample_name"] for row in rows]
    output.mkdir(parents=True, exist_ok=True)
    mini_list = output / "train.txt"
    mini_folds = output / "train_group_folds.csv"
    _write_text_atomic(mini_list, "".join(f"{name}\n" for name in selected_names))

    csv_lines: list[str] = []
    header = ",".join(REQUIRED_COLUMNS)
    csv_lines.append(header)
    for row in rows:
        # FIND sample identifiers and parsed group names contain no CSV control
        # characters; reject them instead of introducing dialect ambiguity.
        if any(
            "," in row[name] or "\n" in row[name] or "\r" in row[name]
            for name in REQUIRED_COLUMNS
        ):
            raise ValueError("Smoke protocol fields cannot contain CSV delimiters")
        csv_lines.append(",".join(row[name] for name in REQUIRED_COLUMNS))
    _write_text_atomic(mini_folds, "\n".join(csv_lines) + "\n")

    manifest: dict[str, object] = {
        "schema": SMOKE_SCHEMA,
        "schema_version": SMOKE_SCHEMA_VERSION,
        "scientific_result": False,
        "purpose": "real-data GPU pipeline smoke test only",
        "selection": "first canonical crop from distinct physical groups per fold",
        "samples_per_fold": samples_per_fold,
        "sample_count": len(rows),
        "physical_group_count": len({row["physical_group"] for row in rows}),
        "fold_sample_counts": dict(
            sorted(Counter(row["oof_fold"] for row in rows).items())
        ),
        "source": {
            "train_list_sha256": sha256_file(train_list),
            "fold_csv_sha256": sha256_file(fold_csv),
        },
        "outputs": {
            "train_list": {"name": mini_list.name, "sha256": sha256_file(mini_list)},
            "fold_csv": {"name": mini_folds.name, "sha256": sha256_file(mini_folds)},
        },
    }
    _write_text_atomic(
        output / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = build_smoke_protocol(
        args.train_list,
        args.fold_csv,
        args.output,
        samples_per_fold=args.samples_per_fold,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
