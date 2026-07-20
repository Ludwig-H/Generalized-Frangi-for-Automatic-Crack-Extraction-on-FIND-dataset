"""Atomic, resumable storage for schema-v2 Frangi raster samples."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping, Sequence
import zipfile

import numpy as np

from .data import normalize_noise_mode, sample_names_sha256
from .graph_types import (
    FRANGI_GATE_FEATURE_SEMANTICS,
    FRANGI_RASTER_CHANNEL_SEMANTICS,
    FRANGI_RASTER_CHANNELS,
    FRANGI_RASTER_SCHEMA_VERSION,
    FrangiRasterSample,
)


GRAPH_CACHE_MANIFEST = ".cracksam2-frangi-graph-v2.json"
GRAPH_CACHE_IN_PROGRESS = ".cracksam2-frangi-graph-v2.in-progress.json"


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash one file without loading it in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha256(
    paths: Sequence[str | os.PathLike[str]],
    *,
    relative_to: str | os.PathLike[str],
) -> str:
    """Hash extractor sources together with stable repository-relative names."""
    root = Path(relative_to).resolve()
    resolved = sorted(Path(path).resolve() for path in paths)
    digest = hashlib.sha256()
    for path in resolved:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Implementation source is outside {root}: {path}") from exc
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def graph_cache_relative_path(case_name: str) -> Path:
    """Map a split identifier to ``relative/name.ext.npz`` without traversal."""
    normalized = case_name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or (relative.parts and relative.parts[0].endswith(":"))
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise ValueError(f"Unsafe sample path in split list: {case_name!r}")
    return Path(*relative.parts[:-1]) / f"{relative.name}.npz"


def write_json_atomic(path: str | os.PathLike[str], value: object) -> Path:
    """Publish JSON with fsync + rename so Spot interruption cannot truncate it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            json.dump(value, temporary, indent=2, sort_keys=True, ensure_ascii=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def save_graph_raster_atomic(
    path: str | os.PathLike[str], sample: FrangiRasterSample
) -> Path:
    """Atomically save one self-describing raster as an unpickled ``.npz``."""
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError(f"Graph cache path must end in .npz: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            np.savez_compressed(
                temporary,
                schema_version=np.asarray(FRANGI_RASTER_SCHEMA_VERSION, dtype=np.int64),
                channel_names=np.asarray(FRANGI_RASTER_CHANNELS, dtype=np.str_),
                case_name=np.asarray(sample.case_name, dtype=np.str_),
                raster=sample.raster,
                image_sha256=np.asarray(sample.image_sha256, dtype=np.str_),
                mask_sha256=np.asarray(sample.mask_sha256, dtype=np.str_),
                elapsed_seconds=np.asarray(sample.elapsed_seconds, dtype=np.float64),
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _scalar_string(archive: np.lib.npyio.NpzFile, key: str) -> str:
    value = archive[key]
    if value.shape != () or value.dtype.kind not in ("U", "S"):
        raise ValueError(f"Invalid scalar string field {key!r}")
    return str(value.item())


def load_graph_raster(
    path: str | os.PathLike[str],
    *,
    expected_case_name: str | None = None,
    expected_image_sha256: str | None = None,
    expected_mask_sha256: str | None = None,
    expected_image_size: Sequence[int] | None = None,
) -> FrangiRasterSample:
    """Load one cache entry and reject schema, provenance or shape drift."""
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "schema_version",
                "channel_names",
                "case_name",
                "raster",
                "image_sha256",
                "mask_sha256",
                "elapsed_seconds",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"Missing graph cache fields: {sorted(missing)}")
            schema = archive["schema_version"]
            if schema.shape != () or int(schema.item()) != FRANGI_RASTER_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported graph cache schema: {schema!r}; "
                    f"expected {FRANGI_RASTER_SCHEMA_VERSION}"
                )
            channels = tuple(str(value) for value in archive["channel_names"].tolist())
            if channels != FRANGI_RASTER_CHANNELS:
                raise ValueError(
                    f"Graph cache channels differ: {channels}; expected {FRANGI_RASTER_CHANNELS}"
                )
            sample = FrangiRasterSample(
                case_name=_scalar_string(archive, "case_name"),
                raster=np.asarray(archive["raster"]).copy(),
                image_sha256=_scalar_string(archive, "image_sha256"),
                mask_sha256=_scalar_string(archive, "mask_sha256"),
                elapsed_seconds=float(archive["elapsed_seconds"].item()),
            )
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid Frangi graph cache entry: {source}: {exc}") from exc

    expected = {
        "case_name": expected_case_name,
        "image_sha256": expected_image_sha256,
        "mask_sha256": expected_mask_sha256,
    }
    for field, expected_value in expected.items():
        if expected_value is not None and getattr(sample, field) != expected_value:
            raise ValueError(
                f"Graph cache {field} mismatch for {source}: "
                f"{getattr(sample, field)!r} != {expected_value!r}"
            )
    if expected_image_size is not None:
        size = tuple(int(value) for value in expected_image_size)
        if len(size) != 2 or sample.image_size != size:
            raise ValueError(
                f"Graph cache image size mismatch for {source}: "
                f"{sample.image_size} != {size}"
            )
    return sample


def build_graph_cache_contract(
    sample_names: Sequence[str],
    *,
    image_size: Sequence[int],
    noise_mode: str | None,
    frangi_parameters: Mapping[str, object],
    extractor_sha256: str,
) -> dict[str, object]:
    """Build the immutable part shared by in-progress and complete manifests."""
    names = list(sample_names)
    if len(set(names)) != len(names):
        raise ValueError("A graph-cache split cannot contain duplicate sample names")
    size = tuple(int(value) for value in image_size)
    if len(size) != 2 or any(value <= 0 for value in size):
        raise ValueError("image_size must contain two positive integers")
    if len(extractor_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in extractor_sha256
    ):
        raise ValueError("extractor_sha256 must be a lowercase SHA-256 digest")
    return {
        "schema_version": FRANGI_RASTER_SCHEMA_VERSION,
        "sample_count": len(names),
        "sample_names_sha256": sample_names_sha256(names),
        "image_size": list(size),
        "noise": normalize_noise_mode(noise_mode),
        "channels": list(FRANGI_RASTER_CHANNELS),
        "channel_semantics": dict(FRANGI_RASTER_CHANNEL_SEMANTICS),
        "gate_feature_semantics": dict(FRANGI_GATE_FEATURE_SEMANTICS),
        "frangi": dict(frangi_parameters),
        "extractor_sha256": extractor_sha256,
    }


def assert_manifest_compatible(
    observed: Mapping[str, object], expected_contract: Mapping[str, object]
) -> None:
    """Reject reuse when any source split, preprocessing or extractor key drifts."""
    mismatches = {
        key: {"observed": observed.get(key), "expected": value}
        for key, value in expected_contract.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frangi graph cache manifest is incompatible: {mismatches}")


def graph_cache_record(
    cache_root: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
    sample: FrangiRasterSample,
) -> dict[str, object]:
    """Build one final-manifest entry after the file has been published."""
    root = Path(cache_root).resolve()
    path = Path(cache_path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Cache entry is outside {root}: {path}") from exc
    return {
        "case_name": sample.case_name,
        "cache_file": relative.as_posix(),
        "cache_sha256": sha256_file(path),
        "cache_bytes": path.stat().st_size,
        "image_sha256": sample.image_sha256,
        "mask_sha256": sample.mask_sha256,
        "raster_shape": list(sample.raster.shape),
        "statistics": sample.statistics(),
    }


def aggregate_graph_cache_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate compact per-sample statistics without reopening raster arrays."""
    if not records:
        return {
            "samples": 0,
            "cache_bytes": 0,
            "support_density_mean": 0.0,
            "skeleton_density_mean": 0.0,
            "elapsed_seconds_total": 0.0,
            "nonfinite_values": 0,
            "gate_features": {
                "similarity_mean": 0.0,
                "strength_mean": 0.0,
                "density_mean": 0.0,
                "stability": None,
                "stability_available": False,
            },
        }
    statistics = [record["statistics"] for record in records]
    gate_features = [value["gate_features"] for value in statistics]
    return {
        "samples": len(records),
        "cache_bytes": int(sum(int(record["cache_bytes"]) for record in records)),
        "support_density_mean": float(
            np.mean([float(value["support_density"]) for value in statistics])
        ),
        "skeleton_density_mean": float(
            np.mean([float(value["skeleton_density"]) for value in statistics])
        ),
        "elapsed_seconds_total": float(
            sum(float(value["elapsed_seconds"]) for value in statistics)
        ),
        "nonfinite_values": int(
            sum(int(value["nonfinite_values"]) for value in statistics)
        ),
        "gate_features": {
            "similarity_mean": float(
                np.mean([float(value["similarity"]) for value in gate_features])
            ),
            "strength_mean": float(
                np.mean([float(value["strength"]) for value in gate_features])
            ),
            "density_mean": float(
                np.mean([float(value["density"]) for value in gate_features])
            ),
            "stability": None,
            "stability_available": False,
        },
    }


def validate_completed_graph_cache(
    cache_dir: str | os.PathLike[str],
    expected_contract: Mapping[str, object],
    *,
    verify_files: bool = True,
) -> dict[str, object]:
    """Validate a published manifest and, optionally, every referenced ``.npz``."""
    root = Path(cache_dir)
    manifest_path = root / GRAPH_CACHE_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid graph cache manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Graph cache manifest must be a JSON object: {manifest_path}")
    assert_manifest_compatible(manifest, expected_contract)
    if manifest.get("status") != "complete":
        raise ValueError(f"Graph cache is not complete: {manifest_path}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != expected_contract["sample_count"]:
        raise ValueError("Graph cache manifest has an invalid entry count")
    if verify_files:
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Graph cache manifest entry must be an object")
            relative = graph_cache_relative_path(str(entry.get("case_name", "")))
            if relative.as_posix() != entry.get("cache_file"):
                raise ValueError("Graph cache entry path does not match its case name")
            path = root / relative
            sample = load_graph_raster(
                path,
                expected_case_name=str(entry["case_name"]),
                expected_image_sha256=str(entry["image_sha256"]),
                expected_mask_sha256=str(entry["mask_sha256"]),
                expected_image_size=expected_contract["image_size"],
            )
            if sha256_file(path) != entry.get("cache_sha256"):
                raise ValueError(f"Graph cache file SHA-256 mismatch: {path}")
            if list(sample.raster.shape) != entry.get("raster_shape"):
                raise ValueError(f"Graph cache raster shape mismatch: {path}")
    return manifest


__all__ = [
    "GRAPH_CACHE_IN_PROGRESS",
    "GRAPH_CACHE_MANIFEST",
    "aggregate_graph_cache_records",
    "assert_manifest_compatible",
    "build_graph_cache_contract",
    "graph_cache_record",
    "graph_cache_relative_path",
    "implementation_sha256",
    "load_graph_raster",
    "save_graph_raster_atomic",
    "sha256_file",
    "validate_completed_graph_cache",
    "write_json_atomic",
]
