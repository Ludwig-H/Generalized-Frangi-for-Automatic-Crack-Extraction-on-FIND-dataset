from __future__ import annotations

import http.cookiejar
import sys
import urllib.request
import zipfile
from pathlib import Path

import pytest

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from prepare_cracksam2_data import (  # noqa: E402
    DatasetSpec,
    ExtractionRule,
    PreparationError,
    _download_url,
    prepare_dataset,
)


def _mini_spec(name: str = "mini") -> DatasetSpec:
    return DatasetSpec(
        name=name,
        archive_name=f"{name}.zip",
        rules=(
            ExtractionRule(("images",), ("images",)),
            ExtractionRule(("labels",), ("masks",)),
        ),
        expected_counts={"images": 1, "masks": 1},
        pairs=(("images", "masks"),),
    )


def test_selective_extraction_mapping_and_idempotency(tmp_path: Path) -> None:
    archive = tmp_path / "mini.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("wrapper/images/sample.jpg", b"image")
        target.writestr("wrapper/labels/sample.jpg", b"mask")
        target.writestr("wrapper/images_HR/sample.jpg", b"large")
        target.writestr("wrapper/pseudo_color_labels/sample.png", b"unused")

    destination = prepare_dataset(archive, tmp_path / "output", _mini_spec())
    assert (destination / "images/sample.jpg").read_bytes() == b"image"
    assert (destination / "masks/sample.jpg").read_bytes() == b"mask"
    assert not (destination / "images_HR").exists()

    marker_mtime = (destination / ".cracksam2-prepared.json").stat().st_mtime_ns
    assert prepare_dataset(archive, tmp_path / "output", _mini_spec()) == destination
    assert (destination / ".cracksam2-prepared.json").stat().st_mtime_ns == marker_mtime


def test_zip_slip_is_rejected_before_writing_files(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("images/sample.jpg", b"image")
        target.writestr("labels/sample.jpg", b"mask")
        target.writestr("../escaped.txt", b"unsafe")

    output = tmp_path / "output"
    with pytest.raises(PreparationError, match="unsafe ZIP member"):
        prepare_dataset(archive, output, _mini_spec())

    assert not (tmp_path / "escaped.txt").exists()
    assert not (output / "mini").exists()
    assert list(output.glob(".mini-*")) == []


def test_facade_suffix_can_be_canonicalized_for_linux_lists(tmp_path: Path) -> None:
    archive = tmp_path / "facade.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("images/DJ_Wall_1.jpg", b"image")
        target.writestr("masks/DJ_Wall_1.jpg", b"mask")

    base = _mini_spec("facade")
    spec = DatasetSpec(
        name=base.name,
        archive_name=base.archive_name,
        rules=(
            ExtractionRule(("images",), ("images",)),
            ExtractionRule(("masks",), ("masks",)),
        ),
        expected_counts=base.expected_counts,
        pairs=base.pairs,
        canonical_suffix=".JPG",
    )
    destination = prepare_dataset(archive, tmp_path / "output", spec)

    assert (destination / "images/DJ_Wall_1.JPG").is_file()
    assert (destination / "masks/DJ_Wall_1.JPG").is_file()


def test_download_url_uses_cookie_aware_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"redirected SharePoint archive"
    captured: dict[str, object] = {}

    class Headers:
        @staticmethod
        def get_content_type() -> str:
            return "application/zip"

    class Response:
        headers = Headers()

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            if captured.get("read"):
                return b""
            captured["read"] = True
            return payload

    class Opener:
        def open(self, request: urllib.request.Request, timeout: int) -> Response:
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

    def build_opener(*handlers: object) -> Opener:
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)

    destination = tmp_path / "concrete3k.zip"
    _download_url("https://sharepoint.example/archive", destination)

    handlers = captured["handlers"]
    assert isinstance(handlers, tuple)
    assert len(handlers) == 1
    cookie_processor = handlers[0]
    assert isinstance(cookie_processor, urllib.request.HTTPCookieProcessor)
    assert isinstance(cookie_processor.cookiejar, http.cookiejar.CookieJar)
    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    assert request.get_header("User-agent") == "CrackSAM2-data/1"
    assert captured["timeout"] == 60
    assert destination.read_bytes() == payload
