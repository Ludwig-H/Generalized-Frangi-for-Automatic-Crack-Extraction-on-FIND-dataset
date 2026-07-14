#!/usr/bin/env python3
"""Download and prepare the four datasets used by the CrackSAM 2 benchmark.

Only the subsets needed by CrackSAM 2 are extracted. Dataset directories are
assembled in a temporary sibling directory, validated, and then published with
an atomic rename.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


CHUNK_SIZE = 8 * 1024 * 1024
PREPARATION_VERSION = 1


class PreparationError(RuntimeError):
    """Raised when a dataset cannot be downloaded, extracted, or validated."""


@dataclass(frozen=True)
class ExtractionRule:
    source: tuple[str, ...]
    destination: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    archive_name: str
    rules: tuple[ExtractionRule, ...]
    expected_counts: Mapping[str, int]
    pairs: tuple[tuple[str, str], ...]
    google_drive_id: str | None = None
    download_url: str | None = None
    canonical_suffix: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "khanhha": DatasetSpec(
        name="khanhha",
        archive_name="crack_segmentation_dataset.zip",
        google_drive_id="1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP",
        rules=(
            ExtractionRule(("train", "images"), ("train", "images")),
            ExtractionRule(("train", "masks"), ("train", "masks")),
            ExtractionRule(("test", "images"), ("test", "images")),
            ExtractionRule(("test", "masks"), ("test", "masks")),
        ),
        expected_counts={
            "train/images": 9603,
            "train/masks": 9603,
            "test/images": 1695,
            "test/masks": 1695,
        },
        pairs=(("train/images", "train/masks"), ("test/images", "test/masks")),
    ),
    "road420": DatasetSpec(
        name="road420",
        archive_name="Road420.zip",
        google_drive_id="1khUfS2uDZb5eDOhpL1qJPYsOxso7Limu",
        rules=(
            ExtractionRule(("images",), ("images",)),
            ExtractionRule(("masks",), ("masks",)),
        ),
        expected_counts={"images": 420, "masks": 420},
        pairs=(("images", "masks"),),
    ),
    "facade390": DatasetSpec(
        name="facade390",
        archive_name="Facade390.zip",
        google_drive_id="1P1b15kRQpVcT7cNDzZB_1vFTrN0WKPB_",
        rules=(
            ExtractionRule(("images",), ("images",)),
            ExtractionRule(("masks",), ("masks",)),
        ),
        expected_counts={"images": 390, "masks": 390},
        pairs=(("images", "masks"),),
        # The upstream CrackSAM list uses .JPG while the ZIP stores .jpg.
        canonical_suffix=".JPG",
    ),
    "concrete3k": DatasetSpec(
        name="concrete3k",
        archive_name="concrete3k.zip",
        download_url=(
            "https://chdeducn-my.sharepoint.com/:u:/g/personal/"
            "2018024008_chd_edu_cn/"
            "EdzjOhykuQxDjRgs6k-5PU0BtJntPGtTo445f4lBv5HV4Q"
            "?e=MCOv5W&download=1"
        ),
        rules=(
            ExtractionRule(("images",), ("images",)),
            ExtractionRule(("labels",), ("masks",)),
        ),
        expected_counts={"images": 3000, "masks": 3000},
        pairs=(("images", "masks"),),
    ),
}


def _safe_member_parts(info: zipfile.ZipInfo) -> tuple[str, ...]:
    raw_name = info.filename.replace("\\", "/")
    path = PurePosixPath(raw_name)
    parts = path.parts
    if (
        not raw_name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in parts)
        or (parts and ":" in parts[0])
    ):
        raise PreparationError(f"unsafe ZIP member path: {info.filename!r}")
    unix_mode = info.external_attr >> 16
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise PreparationError(f"symbolic link forbidden in ZIP: {info.filename!r}")
    return tuple(parts)


def _match_rule(
    parts: tuple[str, ...], rules: Sequence[ExtractionRule]
) -> tuple[ExtractionRule, tuple[str, ...]] | None:
    for rule in rules:
        width = len(rule.source)
        for root_offset in (0, 1):
            if parts[root_offset : root_offset + width] != rule.source:
                continue
            relative = parts[root_offset + width :]
            if relative:
                return rule, relative
    return None


def _canonical_destination(
    spec: DatasetSpec, rule: ExtractionRule, relative: tuple[str, ...]
) -> Path:
    destination = Path(*rule.destination, *relative)
    if spec.canonical_suffix and destination.suffix.lower() == ".jpg":
        destination = destination.with_suffix(spec.canonical_suffix)
    return destination


def _iter_relative_files(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise PreparationError(f"missing expected directory: {directory}")
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }


def validate_dataset(dataset_dir: Path, spec: DatasetSpec) -> dict[str, int]:
    """Validate file counts and one-to-one image/mask correspondence."""

    counts: dict[str, int] = {}
    files_by_directory: dict[str, set[str]] = {}
    for relative_dir, expected in spec.expected_counts.items():
        files = _iter_relative_files(dataset_dir / relative_dir)
        actual = len(files)
        if actual != expected:
            raise PreparationError(
                f"{spec.name}: {relative_dir} contains {actual} files; "
                f"expected {expected}"
            )
        counts[relative_dir] = actual
        files_by_directory[relative_dir] = files

    for images_dir, masks_dir in spec.pairs:
        images = files_by_directory[images_dir]
        masks = files_by_directory[masks_dir]
        if images != masks:
            missing_masks = sorted(images - masks)[:5]
            missing_images = sorted(masks - images)[:5]
            raise PreparationError(
                f"{spec.name}: image/mask names differ for {images_dir} and "
                f"{masks_dir}; missing masks={missing_masks}, "
                f"missing images={missing_images}"
            )
    return counts


def _extract_selected(archive: Path, staging: Path, spec: DatasetSpec) -> int:
    selected: list[tuple[zipfile.ZipInfo, Path]] = []
    destinations: set[Path] = set()
    try:
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                parts = _safe_member_parts(info)
                if info.is_dir():
                    continue
                match = _match_rule(parts, spec.rules)
                if match is None:
                    continue
                rule, relative = match
                destination = _canonical_destination(spec, rule, relative)
                if destination in destinations:
                    raise PreparationError(
                        f"{spec.name}: duplicate destination in ZIP: {destination}"
                    )
                destinations.add(destination)
                selected.append((info, destination))

            if not selected:
                raise PreparationError(
                    f"{spec.name}: no expected files found in {archive}"
                )

            for info, relative_destination in selected:
                destination = staging / relative_destination
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info) as input_file, destination.open("xb") as output_file:
                    shutil.copyfileobj(input_file, output_file, length=CHUNK_SIZE)
    except zipfile.BadZipFile as exc:
        raise PreparationError(f"invalid ZIP archive {archive}: {exc}") from exc
    return len(selected)


def prepare_dataset(archive: Path, output: Path, spec: DatasetSpec) -> Path:
    """Selectively extract and atomically publish one dataset."""

    destination = output / spec.name
    if destination.exists():
        validate_dataset(destination, spec)
        print(f"[{spec.name}] already prepared: {destination}")
        return destination

    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{spec.name}-", dir=output))
    try:
        selected = _extract_selected(archive, staging, spec)
        counts = validate_dataset(staging, spec)
        marker = {
            "dataset": spec.name,
            "preparation_version": PREPARATION_VERSION,
            "source_archive": archive.name,
            "selected_files": selected,
            "counts": counts,
        }
        (staging / ".cracksam2-prepared.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            os.replace(staging, destination)
        except OSError:
            if destination.exists():
                validate_dataset(destination, spec)
                shutil.rmtree(staging)
            else:
                raise
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"[{spec.name}] prepared: {destination}")
    return destination


def _download_google_drive(file_id: str, destination: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise PreparationError(
            "gdown is required for Google Drive downloads; install it with "
            "`python -m pip install gdown`"
        ) from exc
    result = gdown.download(id=file_id, output=str(destination), quiet=False)
    if result is None or not destination.is_file():
        raise PreparationError(f"gdown failed for Google Drive file {file_id}")


def _download_url(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "CrackSAM2-data/1"})
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookie_jar)
    )
    try:
        with opener.open(request, timeout=60) as response:
            content_type = response.headers.get_content_type()
            if content_type == "text/html":
                raise PreparationError(
                    f"download URL returned HTML instead of an archive: {url}"
                )
            with destination.open("xb") as output_file:
                shutil.copyfileobj(response, output_file, length=CHUNK_SIZE)
    except OSError as exc:
        raise PreparationError(f"download failed for {url}: {exc}") from exc


def download_archive(spec: DatasetSpec, destination: Path) -> None:
    """Download to a temporary sibling and replace the final archive atomically."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.part"
    )
    try:
        if spec.google_drive_id:
            _download_google_drive(spec.google_drive_id, temporary)
        elif spec.download_url:
            _download_url(spec.download_url, temporary)
        else:
            raise PreparationError(f"no download source configured for {spec.name}")
        if not zipfile.is_zipfile(temporary):
            raise PreparationError(
                f"downloaded file for {spec.name} is not a valid ZIP archive"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_datasets(values: Iterable[str]) -> list[str]:
    selected: list[str] = []
    for value in values:
        for name in value.split(","):
            normalized = name.strip().lower()
            if normalized and normalized not in selected:
                selected.append(normalized)
    unknown = sorted(set(selected) - DATASETS.keys())
    if unknown:
        raise PreparationError(
            f"unknown datasets {unknown}; choices are {sorted(DATASETS)}"
        )
    if not selected:
        raise PreparationError("at least one dataset must be selected")
    return selected


def _parse_archive_overrides(values: Iterable[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise PreparationError(
                f"invalid --archive value {value!r}; expected DATASET=/path/file.zip"
            )
        name, raw_path = value.split("=", 1)
        name = name.strip().lower()
        if name not in DATASETS:
            raise PreparationError(f"unknown dataset in --archive: {name!r}")
        if name in overrides:
            raise PreparationError(f"duplicate --archive override for {name}")
        overrides[name] = Path(raw_path).expanduser().resolve()
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination root (one subdirectory per dataset)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        metavar="NAME",
        help="datasets to prepare, space- or comma-separated (default: all)",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="download/cache directory (default: OUTPUT/_archives)",
    )
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="use a local archive for one dataset; may be repeated",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="never use the network; require local/cached archives",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="retain archives downloaded during this invocation",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    archive_dir = (
        args.archive_dir.expanduser().resolve()
        if args.archive_dir
        else output / "_archives"
    )
    selected = _parse_datasets(args.datasets)
    overrides = _parse_archive_overrides(args.archive)

    for name in selected:
        spec = DATASETS[name]
        destination = output / name
        if destination.exists():
            validate_dataset(destination, spec)
            print(f"[{name}] already prepared: {destination}")
            continue

        archive = overrides.get(name, archive_dir / spec.archive_name)
        downloaded = False
        if not archive.is_file():
            if name in overrides:
                raise PreparationError(f"local archive does not exist: {archive}")
            if args.skip_download:
                raise PreparationError(
                    f"archive missing with --skip-download: {archive}"
                )
            print(f"[{name}] downloading {spec.archive_name} to {archive_dir}")
            download_archive(spec, archive)
            downloaded = True
        elif not zipfile.is_zipfile(archive):
            if name in overrides or args.skip_download:
                raise PreparationError(f"local archive is not a valid ZIP: {archive}")
            print(f"[{name}] replacing invalid cached archive: {archive}")
            download_archive(spec, archive)
            downloaded = True

        prepare_dataset(archive, output, spec)
        if downloaded and not args.keep_archives:
            archive.unlink()
            print(f"[{name}] removed downloaded archive: {archive}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except PreparationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
