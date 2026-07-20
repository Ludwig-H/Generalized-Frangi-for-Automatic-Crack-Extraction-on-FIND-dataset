"""Leak-safe data and normalization for FrangiGraph-Residual training.

The seven-channel graph raster is loaded from the immutable schema-v2 cache.
Only the Hessian magnitude is learned-normalized: it is transformed with
``log1p`` and standardized with statistics computed on the training folds.
All other transformations are fixed and stated explicitly in the checkpoint.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import (
    CrackSegmentationDataset,
    normalize_noise_mode,
    read_sample_list,
    sample_names_sha256,
)
from .graph_cache import (
    GRAPH_CACHE_MANIFEST,
    graph_cache_relative_path,
    load_graph_raster,
    sha256_file,
    validate_completed_graph_cache,
    write_json_atomic,
)
from .graph_types import (
    FRANGI_RASTER_CHANNEL_INDEX,
    FRANGI_RASTER_CHANNEL_SEMANTICS,
    FRANGI_RASTER_CHANNELS,
    FRANGI_RASTER_SCHEMA_VERSION,
    no_frangi_evidence_raster,
    validate_frangi_raster,
)

NORMALIZATION_FORMAT_VERSION = 1
REQUIRED_FOLDS = frozenset(range(5))


@dataclass(frozen=True, slots=True)
class FoldSplit:
    """One group-safe out-of-fold training/validation split."""

    held_out_fold: int
    train_names: tuple[str, ...]
    validation_names: tuple[str, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    excluded_training_folds: tuple[int, ...] = ()
    excluded_training_names: tuple[str, ...] = ()
    excluded_training_groups: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphCacheIndex:
    """Validated manifest and direct case-name lookup for a graph cache."""

    root: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    entries: Mapping[str, Mapping[str, object]]

    @property
    def image_size(self) -> tuple[int, int]:
        values = self.manifest["image_size"]
        return int(values[0]), int(values[1])  # type: ignore[index]

    @property
    def max_scale(self) -> float:
        frangi = self.manifest.get("frangi")
        if not isinstance(frangi, Mapping):
            raise ValueError("graph cache manifest has no Frangi parameters")
        scales = frangi.get("scales")
        if not isinstance(scales, list) or not scales:
            raise ValueError("graph cache manifest has no non-empty Frangi scale list")
        values = tuple(float(value) for value in scales)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("graph cache Frangi scales must be finite and positive")
        return max(values)

    def entry(self, case_name: str) -> Mapping[str, object]:
        try:
            return self.entries[case_name]
        except KeyError as exc:
            raise KeyError(f"case absent from graph cache: {case_name!r}") from exc


@dataclass(frozen=True, slots=True)
class FrangiRasterNormalization:
    """Explicit affine preprocessing fitted on training rasters only.

    The operation is ``(transform(x) - offset) / scale`` per channel, where
    ``transform`` is ``log1p`` only for ``hessian_magnitude`` and the identity
    for every other channel.  ``winning_scale`` is therefore divided by the
    largest extraction scale through its affine scale value.
    """

    channel_names: tuple[str, ...]
    log1p_channels: tuple[str, ...]
    offset: tuple[float, ...]
    scale: tuple[float, ...]
    fit_sample_count: int
    fit_sample_names_sha256: str
    graph_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.channel_names != FRANGI_RASTER_CHANNELS:
            raise ValueError("normalization channel order differs from schema v2")
        if self.log1p_channels != ("hessian_magnitude",):
            raise ValueError("only hessian_magnitude may use log1p in version 1")
        channel_count = len(FRANGI_RASTER_CHANNELS)
        if len(self.offset) != channel_count or len(self.scale) != channel_count:
            raise ValueError("normalization offset/scale must contain seven values")
        if any(not math.isfinite(value) for value in self.offset):
            raise ValueError("normalization offsets must be finite")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.scale):
            raise ValueError("normalization scales must be finite and positive")
        if self.fit_sample_count <= 0:
            raise ValueError("normalization needs at least one training sample")
        for field_name, digest in (
            ("fit_sample_names_sha256", self.fit_sample_names_sha256),
            ("graph_manifest_sha256", self.graph_manifest_sha256),
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")

    @property
    def max_scale(self) -> float:
        return self.scale[FRANGI_RASTER_CHANNEL_INDEX["winning_scale"]]

    def transform(self, raster: np.ndarray) -> np.ndarray:
        """Validate and normalize one cached raster without changing it in place."""
        validate_frangi_raster(raster)
        output = raster.astype(np.float32, copy=True)

        # Clamp only the channels whose cache contract is mathematically bounded.
        output[FRANGI_RASTER_CHANNEL_INDEX["similarity"]] = np.clip(
            output[FRANGI_RASTER_CHANNEL_INDEX["similarity"]], 0.0, 1.0
        )
        output[FRANGI_RASTER_CHANNEL_INDEX["support"]] = np.clip(
            output[FRANGI_RASTER_CHANNEL_INDEX["support"]], 0.0, 1.0
        )
        output[FRANGI_RASTER_CHANNEL_INDEX["orientation_sin2"]] = np.clip(
            output[FRANGI_RASTER_CHANNEL_INDEX["orientation_sin2"]], -1.0, 1.0
        )
        output[FRANGI_RASTER_CHANNEL_INDEX["orientation_cos2"]] = np.clip(
            output[FRANGI_RASTER_CHANNEL_INDEX["orientation_cos2"]], -1.0, 1.0
        )
        output[FRANGI_RASTER_CHANNEL_INDEX["distance_to_skeleton"]] = np.clip(
            output[FRANGI_RASTER_CHANNEL_INDEX["distance_to_skeleton"]], 0.0, 1.0
        )

        winning_scale_index = FRANGI_RASTER_CHANNEL_INDEX["winning_scale"]
        if float(output[winning_scale_index].max()) > self.max_scale + 1e-5:
            raise ValueError(
                "winning_scale exceeds the largest scale published by the cache"
            )

        magnitude_index = FRANGI_RASTER_CHANNEL_INDEX["hessian_magnitude"]
        output[magnitude_index] = np.log1p(output[magnitude_index])
        offsets = np.asarray(self.offset, dtype=np.float32)[:, None, None]
        scales = np.asarray(self.scale, dtype=np.float32)[:, None, None]
        output = (output - offsets) / scales
        if not np.isfinite(output).all():
            raise ValueError("normalized graph raster contains NaN or infinity")
        return np.ascontiguousarray(output, dtype=np.float32)

    def preprocessing_contract(self) -> dict[str, object]:
        """Return the evaluator-facing, raw and unambiguous channel contract."""
        return {
            "format_version": NORMALIZATION_FORMAT_VERSION,
            "channel_names": list(self.channel_names),
            "log1p_channels": list(self.log1p_channels),
            "offset": list(self.offset),
            "scale": list(self.scale),
            "formula": "(transform(raw) - offset) / scale",
            "bounded_raw_channels": {
                "similarity": [0.0, 1.0],
                "support": [0.0, 1.0],
                "orientation_sin2": [-1.0, 1.0],
                "orientation_cos2": [-1.0, 1.0],
                "distance_to_skeleton": [0.0, 1.0],
            },
            "winning_scale_divisor": self.max_scale,
            "fit_sample_count": self.fit_sample_count,
            "fit_sample_names_sha256": self.fit_sample_names_sha256,
            "graph_manifest_sha256": self.graph_manifest_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return self.preprocessing_contract()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FrangiRasterNormalization":
        if int(value.get("format_version", -1)) != NORMALIZATION_FORMAT_VERSION:
            raise ValueError("unsupported raster normalization format")
        return cls(
            channel_names=tuple(str(item) for item in value.get("channel_names", [])),
            log1p_channels=tuple(str(item) for item in value.get("log1p_channels", [])),
            offset=tuple(float(item) for item in value.get("offset", [])),
            scale=tuple(float(item) for item in value.get("scale", [])),
            fit_sample_count=int(value.get("fit_sample_count", 0)),
            fit_sample_names_sha256=str(value.get("fit_sample_names_sha256", "")),
            graph_manifest_sha256=str(value.get("graph_manifest_sha256", "")),
        )


def load_group_safe_fold(
    fold_csv: str | os.PathLike[str],
    full_sample_names: Sequence[str],
    held_out_fold: int,
    *,
    exclude_training_folds: Sequence[int] = (),
) -> FoldSplit:
    """Read and strictly validate a five-part group-safe training split.

    ``held_out_fold`` always supplies validation and is never used for
    training.  ``exclude_training_folds`` removes additional folds from the
    training side without adding them to validation.  The latter is intended
    for an external calibration split (for example fold 4 while producing OOF
    predictions for folds 0--3).
    """
    if held_out_fold not in REQUIRED_FOLDS:
        raise ValueError("held_out_fold must be one of 0, 1, 2, 3, 4")
    excluded_values = tuple(exclude_training_folds)
    invalid_exclusions = sorted(
        {
            value
            for value in excluded_values
            if not isinstance(value, int) or value not in REQUIRED_FOLDS
        },
        key=str,
    )
    if invalid_exclusions:
        raise ValueError(
            "exclude_training_folds must contain only folds 0, 1, 2, 3, 4; "
            f"invalid={invalid_exclusions}"
        )
    duplicate_exclusions = sorted(
        {value for value in excluded_values if excluded_values.count(value) > 1}
    )
    if duplicate_exclusions:
        raise ValueError(
            "exclude_training_folds contains duplicate folds: "
            f"{duplicate_exclusions}"
        )
    if held_out_fold in excluded_values:
        raise ValueError(
            "held_out_fold is already excluded from training and cannot also "
            "appear in exclude_training_folds"
        )
    excluded_folds = tuple(sorted(excluded_values))
    path = Path(fold_csv)
    try:
        with path.open(newline="", encoding="utf-8-sig") as source:
            rows = list(csv.DictReader(source))
    except OSError as exc:
        raise FileNotFoundError(f"fold assignment not found: {path}") from exc
    required_columns = {"sample_name", "source_family", "physical_group", "oof_fold"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError(f"invalid fold assignment columns in {path}")

    expected = list(full_sample_names)
    if len(expected) != len(set(expected)):
        raise ValueError("the canonical training list contains duplicate names")
    by_name: dict[str, dict[str, str]] = {}
    group_fold: dict[str, int] = {}
    observed_folds: set[int] = set()
    for row in rows:
        name = row["sample_name"]
        group = row["physical_group"]
        if not name or not group:
            raise ValueError("fold rows need non-empty sample and physical-group names")
        if name in by_name:
            raise ValueError(f"duplicate sample in fold assignment: {name}")
        try:
            fold = int(row["oof_fold"])
        except ValueError as exc:
            raise ValueError(f"non-integer fold for sample {name!r}") from exc
        if fold not in REQUIRED_FOLDS:
            raise ValueError(f"invalid fold {fold} for sample {name!r}")
        previous = group_fold.setdefault(group, fold)
        if previous != fold:
            raise ValueError(f"physical group crosses folds: {group!r}")
        observed_folds.add(fold)
        by_name[name] = row

    if observed_folds != REQUIRED_FOLDS:
        raise ValueError(
            f"fold assignment must contain all five folds: {observed_folds}"
        )
    missing = sorted(set(expected) - set(by_name))
    unexpected = sorted(set(by_name) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "fold assignment differs from canonical train list; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    training_excluded_folds = {held_out_fold, *excluded_folds}
    train_names = tuple(
        name
        for name in expected
        if int(by_name[name]["oof_fold"]) not in training_excluded_folds
    )
    validation_names = tuple(
        name for name in expected if int(by_name[name]["oof_fold"]) == held_out_fold
    )
    excluded_training_names = tuple(
        name
        for name in expected
        if int(by_name[name]["oof_fold"]) in excluded_folds
    )
    train_groups = tuple(
        sorted({by_name[name]["physical_group"] for name in train_names})
    )
    validation_groups = tuple(
        sorted({by_name[name]["physical_group"] for name in validation_names})
    )
    excluded_training_groups = tuple(
        sorted(
            {
                by_name[name]["physical_group"]
                for name in excluded_training_names
            }
        )
    )
    if not train_names:
        raise ValueError("training split is empty after fold exclusions")
    if not validation_names:
        raise ValueError("held-out validation fold must be non-empty")
    overlap = set(train_groups).intersection(validation_groups)
    if overlap:
        raise ValueError(
            f"physical-group leakage across selected fold: {sorted(overlap)[:5]}"
        )
    excluded_overlap = set(train_groups).intersection(excluded_training_groups)
    if excluded_overlap:
        raise ValueError(
            "physical-group leakage into additionally excluded folds: "
            f"{sorted(excluded_overlap)[:5]}"
        )
    return FoldSplit(
        held_out_fold=held_out_fold,
        train_names=train_names,
        validation_names=validation_names,
        train_groups=train_groups,
        validation_groups=validation_groups,
        excluded_training_folds=excluded_folds,
        excluded_training_names=excluded_training_names,
        excluded_training_groups=excluded_training_groups,
    )


def _manifest_contract(manifest: Mapping[str, object]) -> dict[str, object]:
    mutable = {"status", "entries", "aggregate"}
    return {key: value for key, value in manifest.items() if key not in mutable}


def load_graph_cache_index(
    cache_dir: str | os.PathLike[str],
    full_sample_names: Sequence[str],
    *,
    image_size: Sequence[int] = (448, 448),
    noise_mode: str | None = "original",
    verify_files: bool = True,
) -> GraphCacheIndex:
    """Validate cache schema, split identity, provenance metadata and files."""
    root = Path(cache_dir).expanduser().resolve()
    manifest_path = root / GRAPH_CACHE_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid graph cache manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("graph cache manifest must be a JSON object")
    expected_size = [int(value) for value in image_size]
    externally_expected = {
        "schema_version": FRANGI_RASTER_SCHEMA_VERSION,
        "status": "complete",
        "sample_count": len(full_sample_names),
        "sample_names_sha256": sample_names_sha256(full_sample_names),
        "image_size": expected_size,
        "noise": normalize_noise_mode(noise_mode),
        "channels": list(FRANGI_RASTER_CHANNELS),
        "channel_semantics": dict(FRANGI_RASTER_CHANNEL_SEMANTICS),
    }
    mismatches = {
        key: {"observed": manifest.get(key), "expected": value}
        for key, value in externally_expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"graph cache is incompatible with training data: {mismatches}"
        )
    if not isinstance(manifest.get("extractor_sha256"), str):
        raise ValueError("graph cache manifest lacks extractor_sha256")
    if not isinstance(manifest.get("frangi"), dict):
        raise ValueError("graph cache manifest lacks Frangi parameters")

    # This checks every .npz schema/provenance/shape and its published SHA-256.
    validate_completed_graph_cache(
        root,
        _manifest_contract(manifest),
        verify_files=verify_files,
    )
    entries_value = manifest.get("entries")
    if not isinstance(entries_value, list):
        raise ValueError("graph cache manifest entries must be a list")
    entries: dict[str, Mapping[str, object]] = {}
    for entry in entries_value:
        if not isinstance(entry, dict):
            raise ValueError("graph cache entries must be JSON objects")
        name = str(entry.get("case_name", ""))
        if not name or name in entries:
            raise ValueError(f"duplicate or empty graph cache case: {name!r}")
        entries[name] = entry
    if set(entries) != set(full_sample_names):
        raise ValueError("graph cache entries differ from the canonical training list")
    return GraphCacheIndex(
        root=root,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        entries=entries,
    )


def _load_indexed_raster(index: GraphCacheIndex, case_name: str) -> np.ndarray:
    entry = index.entry(case_name)
    path = index.root / graph_cache_relative_path(case_name)
    sample = load_graph_raster(
        path,
        expected_case_name=case_name,
        expected_image_sha256=str(entry["image_sha256"]),
        expected_mask_sha256=str(entry["mask_sha256"]),
        expected_image_size=index.image_size,
    )
    return sample.raster


def fit_frangi_raster_normalization(
    index: GraphCacheIndex,
    train_sample_names: Sequence[str],
) -> FrangiRasterNormalization:
    """Fit log-Hessian mean/std using pixels from training folds only."""
    names = tuple(train_sample_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("normalization training names must be unique and non-empty")
    unknown = sorted(set(names) - set(index.entries))
    if unknown:
        raise ValueError(f"normalization names absent from graph cache: {unknown[:5]}")

    # Parallel/Welford merge of per-raster moments keeps float64 precision while
    # avoiding a giant concatenated array.
    count = 0
    mean = 0.0
    squared_deviation = 0.0
    magnitude_index = FRANGI_RASTER_CHANNEL_INDEX["hessian_magnitude"]
    for name in names:
        magnitude = _load_indexed_raster(index, name)[magnitude_index]
        values = np.log1p(magnitude.astype(np.float64, copy=False))
        batch_count = int(values.size)
        batch_mean = float(values.mean(dtype=np.float64))
        batch_m2 = float(np.square(values - batch_mean).sum(dtype=np.float64))
        if count == 0:
            count, mean, squared_deviation = batch_count, batch_mean, batch_m2
            continue
        delta = batch_mean - mean
        combined = count + batch_count
        mean += delta * batch_count / combined
        squared_deviation += batch_m2 + delta * delta * count * batch_count / combined
        count = combined
    variance = max(0.0, squared_deviation / count)
    standard_deviation = math.sqrt(variance)
    if standard_deviation < 1e-6:
        standard_deviation = 1.0

    offsets = [0.0] * len(FRANGI_RASTER_CHANNELS)
    scales = [1.0] * len(FRANGI_RASTER_CHANNELS)
    offsets[magnitude_index] = mean
    scales[magnitude_index] = standard_deviation
    scales[FRANGI_RASTER_CHANNEL_INDEX["winning_scale"]] = index.max_scale
    return FrangiRasterNormalization(
        channel_names=FRANGI_RASTER_CHANNELS,
        log1p_channels=("hessian_magnitude",),
        offset=tuple(offsets),
        scale=tuple(scales),
        fit_sample_count=len(names),
        fit_sample_names_sha256=sample_names_sha256(names),
        graph_manifest_sha256=index.manifest_sha256,
    )


def load_or_fit_frangi_raster_normalization(
    path: str | os.PathLike[str],
    index: GraphCacheIndex,
    train_sample_names: Sequence[str],
) -> FrangiRasterNormalization:
    """Reuse an exact fitted contract or atomically publish newly fitted stats."""
    destination = Path(path)
    names = tuple(train_sample_names)
    if destination.is_file():
        try:
            value = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid normalization file: {destination}") from exc
        if not isinstance(value, dict):
            raise ValueError("normalization file must contain a JSON object")
        normalization = FrangiRasterNormalization.from_dict(value)
        expected = {
            "fit_sample_count": len(names),
            "fit_sample_names_sha256": sample_names_sha256(names),
            "graph_manifest_sha256": index.manifest_sha256,
            "max_scale": index.max_scale,
        }
        observed = {
            "fit_sample_count": normalization.fit_sample_count,
            "fit_sample_names_sha256": normalization.fit_sample_names_sha256,
            "graph_manifest_sha256": normalization.graph_manifest_sha256,
            "max_scale": normalization.max_scale,
        }
        if observed != expected:
            raise ValueError(
                "saved normalization was fitted on different data/cache: "
                f"observed={observed}, expected={expected}"
            )
        return normalization
    normalization = fit_frangi_raster_normalization(index, names)
    write_json_atomic(destination, normalization.to_dict())
    return normalization


class FrangiGraphRasterDataset(Dataset):
    """Crack images/masks paired with validated and normalized graph rasters."""

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        list_file: str | os.PathLike[str],
        sample_names: Sequence[str],
        cache_index: GraphCacheIndex,
        normalization: FrangiRasterNormalization,
        *,
        split: str = "train",
        verify_source_files: bool = True,
        raster_condition: str = "correct",
    ) -> None:
        full_names = read_sample_list(list_file)
        selected = tuple(sample_names)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("selected sample names must be unique and non-empty")
        unknown = sorted(set(selected) - set(full_names))
        if unknown:
            raise ValueError(f"selected samples absent from list: {unknown[:5]}")
        if normalization.graph_manifest_sha256 != cache_index.manifest_sha256:
            raise ValueError("normalization belongs to a different graph cache")
        self.base = CrackSegmentationDataset(
            data_root,
            list_file=list_file,
            split=split,
            image_size=cache_index.image_size,
            augment=False,
            noise_mode=str(cache_index.manifest["noise"]),
        )
        self.base.sample_names = list(selected)
        self.sample_names = selected
        self.cache_index = cache_index
        self.normalization = normalization
        if raster_condition not in ("correct", "no_evidence"):
            raise ValueError(
                "raster_condition must be 'correct' or 'no_evidence'"
            )
        self.raster_condition = raster_condition
        if verify_source_files:
            self._verify_source_provenance()

    def _verify_source_provenance(self) -> None:
        for name in self.sample_names:
            image_path, mask_path = self.base._paths(name)
            entry = self.cache_index.entry(name)
            if sha256_file(image_path) != entry.get("image_sha256"):
                raise ValueError(f"source image changed after graph caching: {name}")
            if sha256_file(mask_path) != entry.get("mask_sha256"):
                raise ValueError(f"source mask changed after graph caching: {name}")

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base[index]
        case_name = str(sample["case_name"])
        if self.raster_condition == "correct":
            raster = _load_indexed_raster(self.cache_index, case_name)
        else:
            raster = no_frangi_evidence_raster(*self.cache_index.image_size)
        sample["frangi_raster"] = torch.from_numpy(self.normalization.transform(raster))
        return sample


__all__ = [
    "FoldSplit",
    "FrangiGraphRasterDataset",
    "FrangiRasterNormalization",
    "GraphCacheIndex",
    "NORMALIZATION_FORMAT_VERSION",
    "fit_frangi_raster_normalization",
    "load_graph_cache_index",
    "load_group_safe_fold",
    "load_or_fit_frangi_raster_normalization",
]
