#!/usr/bin/env python3
"""Audit the historical CrackSAM lists and build group-safe training folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PAPER_LISTS = {
    "train": Path("lists/lists_khanhha/train.txt"),
    "validation": Path("lists/lists_khanhha/val_vol.txt"),
    "test": Path("lists/lists_khanhha/test_vol.txt"),
    "road420": Path("lists/lists_road420/test_vol.txt"),
    "facade390": Path("lists/lists_facade390/test_vol.txt"),
    "concrete3k": Path("lists/lists_concrete3k/test_vol.txt"),
}
FOLD_ALGORITHM = "source-stratified-largest-group-first-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path(__file__).parent / "cracksam_paper",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "frangigraph_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def _clean_stem(sample_name: str) -> str:
    stem = Path(sample_name.replace("\\", "/")).stem
    # Several FIND names end in ``.jpg.jpg``.
    if stem.lower().endswith((".jpg", ".jpeg", ".png")):
        stem = Path(stem).stem
    return stem


def source_family(sample_name: str) -> str:
    return _clean_stem(sample_name).split("_", 1)[0]


def physical_source_group(sample_name: str) -> str:
    """Conservatively map known crop names back to their physical source image."""
    stem = _clean_stem(sample_name)
    rules = (
        ("CRACK500_", 3),
        ("Eugen_Muller_", 3),
        ("GAPS384_", 3),
        ("Rissbilder_for_Florian_", 4),
        ("Volker_", 2),
        ("noncrack_noncrack_concrete_wall_", 5),
    )
    for prefix, parts in rules:
        if stem.startswith(prefix):
            return "_".join(stem.split("_")[:parts])
    if stem.startswith("DeepCrack_"):
        return stem.split("-", 1)[0]
    # Concrete3k crops use ``source_index.jpg`` (for example ``017_24.jpg``).
    if re.fullmatch(r"[0-9]{3}_[0-9]+", stem):
        return stem.split("_", 1)[0]
    return stem


def read_names(path: Path) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"List must be non-empty and unique: {path}")
    return names


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()


def assign_group_folds(
    sample_names: Iterable[str], *, folds: int, seed: int
) -> dict[str, int]:
    """Assign whole physical groups to balanced, source-stratified folds."""
    if folds < 2:
        raise ValueError("At least two folds are required")
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in sample_names:
        grouped[physical_source_group(name)].append(name)
    if len(grouped) < folds:
        raise ValueError("There are fewer physical groups than folds")

    groups_by_source: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for group, members in grouped.items():
        families = {source_family(name) for name in members}
        if len(families) != 1:
            raise ValueError(f"Physical group spans source families: {group}")
        groups_by_source[next(iter(families))].append((group, members))

    fold_total = [0] * folds
    assignment: dict[str, int] = {}
    for family in sorted(groups_by_source):
        fold_family = [0] * folds
        ordered = sorted(
            groups_by_source[family],
            key=lambda item: (-len(item[1]), _stable_key(seed, item[0])),
        )
        for group, members in ordered:
            chosen = min(
                range(folds),
                key=lambda fold: (
                    fold_family[fold],
                    fold_total[fold],
                    _stable_key(seed, f"{group}/{fold}"),
                ),
            )
            assignment[group] = chosen
            fold_family[chosen] += len(members)
            fold_total[chosen] += len(members)
    return assignment


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def build_protocol(
    paper_root: Path, output: Path, *, folds: int, seed: int
) -> dict[str, Any]:
    lists: dict[str, list[str]] = {}
    list_manifest: dict[str, Any] = {}
    for split, relative in PAPER_LISTS.items():
        path = paper_root / relative
        names = read_names(path)
        lists[split] = names
        groups = {physical_source_group(name) for name in names}
        list_manifest[split] = {
            "path": str(relative),
            "samples": len(names),
            "physical_groups": len(groups),
            "sha256": _sha256(path),
        }

    khanhha_splits = ("train", "validation", "test")
    group_sets = {
        split: {physical_source_group(name) for name in lists[split]}
        for split in khanhha_splits
    }
    overlap: dict[str, Any] = {}
    for index, left in enumerate(khanhha_splits):
        for right in khanhha_splits[index + 1 :]:
            common = sorted(group_sets[left] & group_sets[right])
            overlap[f"{left}__{right}"] = {
                "physical_groups": len(common),
                "examples": common[:20],
            }

    assignment = assign_group_folds(lists["train"], folds=folds, seed=seed)
    rows = [
        {
            "sample_name": name,
            "source_family": source_family(name),
            "physical_group": physical_source_group(name),
            "oof_fold": assignment[physical_source_group(name)],
        }
        for name in lists["train"]
    ]
    rows.sort(key=lambda row: (int(row["oof_fold"]), str(row["sample_name"])))
    folds_path = output / "train_group_folds.csv"
    _atomic_csv(folds_path, rows)

    fold_stats: dict[str, Any] = {}
    for fold in range(folds):
        selected = [row for row in rows if row["oof_fold"] == fold]
        fold_stats[str(fold)] = {
            "samples": len(selected),
            "physical_groups": len({row["physical_group"] for row in selected}),
            "source_samples": dict(
                sorted(Counter(row["source_family"] for row in selected).items())
            ),
        }

    manifest = {
        "format_version": 1,
        "purpose": "FrangiGraph-Residual development protocol",
        "grouping_algorithm": "known-source-name-parser-v1",
        "fold_algorithm": FOLD_ALGORITHM,
        "fold_seed": seed,
        "folds": folds,
        "lists": list_manifest,
        "historical_khanhha_group_overlap": overlap,
        "train_group_folds": {
            "path": folds_path.name,
            "sha256": _sha256(folds_path),
            "fold_stats": fold_stats,
        },
        "rules": {
            "residual_training": (
                "for held-out folds 0-3 train only on the other folds among 0-3; "
                "for held-out fold 4 train on folds 0-3"
            ),
            "gate_fit": "OOF predictions from folds 0-3; fold 4 excluded upstream",
            "gate_threshold": "OOF predictions from fold 4 only",
            "final_residual": "train on all Khanhha train groups after choices are frozen",
            "historical_tests": "exploratory only; never fit residual or gate",
            "confirmatory_holdout": "not yet collected",
            "baseline_scope": (
                "historical baseline saw all train folds; this cycle is exploratory "
                "until baseline OOF or an independent holdout exists"
            ),
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = build_protocol(
        args.paper_root.expanduser().resolve(),
        args.output.expanduser().resolve(),
        folds=args.folds,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
