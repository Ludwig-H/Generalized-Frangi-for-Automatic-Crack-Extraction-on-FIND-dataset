from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

import precompute_frangi_graph_cache as precompute  # noqa: E402
from cracksam2.graph_cache import (  # noqa: E402
    GRAPH_CACHE_IN_PROGRESS,
    GRAPH_CACHE_MANIFEST,
    aggregate_graph_cache_records,
    assert_manifest_compatible,
    build_graph_cache_contract,
    graph_cache_record,
    graph_cache_relative_path,
    load_graph_raster,
    save_graph_raster_atomic,
    validate_completed_graph_cache,
    write_json_atomic,
)
from cracksam2.graph_types import (  # noqa: E402
    FRANGI_RASTER_CHANNELS,
    FrangiRasterSample,
    no_frangi_evidence_raster,
    validate_frangi_raster,
)


def _empty_raster(height: int = 6, width: int = 8) -> np.ndarray:
    return no_frangi_evidence_raster(height, width)


def test_no_evidence_is_not_an_all_zero_distance_map() -> None:
    raster = no_frangi_evidence_raster(3, 5)
    np.testing.assert_array_equal(raster[:6], 0.0)
    np.testing.assert_array_equal(raster[6], 1.0)


def _sample(case_name: str = "nested/sample.jpg") -> FrangiRasterSample:
    raster = _empty_raster()
    raster[0, 2, 3] = 0.75
    raster[1, 2, 3] = 1.0
    raster[2, 2, 3] = 0.125
    raster[3, 2, 3] = 1.0
    raster[4, 2, 3] = 0.0
    raster[5, 2, 3] = 1.0
    raster[6, 2, 3] = 0.0
    return FrangiRasterSample(
        case_name=case_name,
        raster=raster,
        image_sha256="a" * 64,
        mask_sha256="b" * 64,
        elapsed_seconds=0.25,
    )


def test_graph_raster_atomic_round_trip_and_provenance_rejection(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "sample.jpg.npz"
    sample = _sample()

    save_graph_raster_atomic(destination, sample)
    loaded = load_graph_raster(
        destination,
        expected_case_name=sample.case_name,
        expected_image_sha256=sample.image_sha256,
        expected_mask_sha256=sample.mask_sha256,
        expected_image_size=sample.image_size,
    )

    np.testing.assert_array_equal(loaded.raster, sample.raster)
    assert loaded.elapsed_seconds == pytest.approx(0.25)
    assert loaded.statistics()["gate_features"] == {
        "similarity": pytest.approx(0.75),
        "strength": pytest.approx(0.125),
        "density": pytest.approx(1.0 / 48.0),
        "stability": None,
        "stability_available": False,
    }
    assert list(destination.parent.glob("*.tmp")) == []
    with pytest.raises(ValueError, match="image_sha256 mismatch"):
        load_graph_raster(destination, expected_image_sha256="c" * 64)


def test_graph_raster_validation_catches_invalid_orientation() -> None:
    raster = _empty_raster()
    raster[3, 1, 1] = 1.0
    raster[4, 1, 1] = 0.5
    raster[5, 1, 1] = 0.5

    with pytest.raises(ValueError, match="unit direction"):
        validate_frangi_raster(raster)


@pytest.mark.parametrize(
    "case_name",
    ("../escape.png", "/absolute.png", "a/../../b.png", "C:\\absolute.png"),
)
def test_graph_cache_path_rejects_traversal(case_name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        graph_cache_relative_path(case_name)


def test_complete_manifest_validates_files_and_frangi_parameters(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    sample = _sample("sample.jpg")
    path = cache / graph_cache_relative_path(sample.case_name)
    save_graph_raster_atomic(path, sample)
    contract = build_graph_cache_contract(
        [sample.case_name],
        image_size=sample.image_size,
        noise_mode="none",
        frangi_parameters={"scales": [1.0], "R": 1},
        extractor_sha256="c" * 64,
    )
    record = graph_cache_record(cache, path, sample)
    manifest = {
        **contract,
        "status": "complete",
        "entries": [record],
        "aggregate": aggregate_graph_cache_records([record]),
    }
    write_json_atomic(cache / GRAPH_CACHE_MANIFEST, manifest)

    validated = validate_completed_graph_cache(cache, contract)
    assert validated["aggregate"]["samples"] == 1
    assert validated["channels"] == list(FRANGI_RASTER_CHANNELS)
    assert validated["aggregate"]["gate_features"]["stability"] is None
    assert not validated["aggregate"]["gate_features"]["stability_available"]

    incompatible = {**contract, "frangi": {"scales": [2.0], "R": 1}}
    with pytest.raises(ValueError, match="incompatible"):
        assert_manifest_compatible(manifest, incompatible)


def _write_pair(root: Path, stem: str, value: int) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    image = np.full((5, 7, 3), value, dtype=np.uint8)
    mask = np.zeros((5, 7), dtype=np.uint8)
    Image.fromarray(image).save(root / "images" / f"{stem}.png")
    Image.fromarray(mask).save(root / "masks" / f"{stem}.png")


def test_precompute_resumes_after_interruption_without_recomputing_valid_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    _write_pair(data, "first", 30)
    _write_pair(data, "second", 220)
    split = tmp_path / "split.txt"
    split.write_text("first.png\nsecond.png\n", encoding="utf-8")
    cache = tmp_path / "cache"
    args = argparse.Namespace(
        data_root=data,
        list_file=split,
        cache_dir=cache,
        image_dir=None,
        mask_dir=None,
        noise="none",
        image_size=8,
        device="cpu",
        scales=(1.0,),
        radius=1,
        ss=1.0,
        si=0.25,
        sa=0.3,
        tau=0.5,
        min_rel_size=1000.0,
        graph_order=1,
        overwrite=False,
        limit=None,
        failure_log=None,
        quiet=True,
    )
    monkeypatch.setattr(precompute, "_extractor_sha256", lambda: "d" * 64)
    first_run_calls: list[int] = []

    def interrupted_generator(image: np.ndarray, **_: object) -> np.ndarray:
        value = int(round(float(image.mean())))
        first_run_calls.append(value)
        if value > 100:
            raise RuntimeError("simulated Spot preemption")
        return _empty_raster(*image.shape[:2])

    monkeypatch.setattr(precompute, "generate_frangi_raster", interrupted_generator)
    assert precompute.run(args) == 1
    assert len(first_run_calls) == 2
    assert (cache / "first.png.npz").is_file()
    assert (cache / GRAPH_CACHE_IN_PROGRESS).is_file()
    assert not (cache / GRAPH_CACHE_MANIFEST).exists()

    resumed_calls: list[int] = []

    def resumed_generator(image: np.ndarray, **_: object) -> np.ndarray:
        resumed_calls.append(int(round(float(image.mean()))))
        return _empty_raster(*image.shape[:2])

    monkeypatch.setattr(precompute, "generate_frangi_raster", resumed_generator)
    assert precompute.run(args) == 0
    assert len(resumed_calls) == 1
    assert resumed_calls[0] > 100
    assert not (cache / GRAPH_CACHE_IN_PROGRESS).exists()
    manifest = json.loads((cache / GRAPH_CACHE_MANIFEST).read_text(encoding="utf-8"))
    assert [entry["case_name"] for entry in manifest["entries"]] == [
        "first.png",
        "second.png",
    ]
    validate_completed_graph_cache(cache, {
        key: value for key, value in manifest.items()
        if key not in {"status", "entries", "aggregate"}
    })
