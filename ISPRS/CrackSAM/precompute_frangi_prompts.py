#!/usr/bin/env python3
"""Precompute static Frangi-Graph mask prompts for one CrackSAM split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from cracksam2.data import (
    FRANGI_PROMPT_EPS,
    PROMPT_CACHE_MANIFEST,
    PROMPT_CACHE_VERSION,
    apply_noise_perturbation,
    normalize_noise_mode,
    read_sample_list,
    resolve_sample_paths,
    sample_names_sha256,
)
from cracksam2.frangi import (
    DEFAULT_FRANGI_SCALES,
    generate_frangi_prompt,
)


IMAGE_SIZE = (448, 448)
PROMPT_SIZE = (256, 256)
IN_PROGRESS_MANIFEST = ".cracksam2-frangi.in-progress.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--image-dir", default="images")
    parser.add_argument(
        "--noise",
        choices=("none", "noisy1", "noisy2"),
        default="none",
        help="Apply the paper's test perturbation before computing Frangi.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--failure-log", type=Path)
    return parser.parse_args()


def _apply_noise(image: np.ndarray, noise: str) -> np.ndarray:
    return apply_noise_perturbation(image, noise, output_size=IMAGE_SIZE)


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _manifest(names: list[str], noise: str, status: str) -> dict[str, object]:
    return {
        "format_version": PROMPT_CACHE_VERSION,
        "status": status,
        "samples": len(names),
        "sample_names_sha256": sample_names_sha256(names),
        "image_size": list(IMAGE_SIZE),
        "prompt_size": list(PROMPT_SIZE),
        "noise": normalize_noise_mode(noise),
        "frangi": {
            "scales": [float(value) for value in DEFAULT_FRANGI_SCALES],
            "R": 3,
            "K": 1,
            "eps": FRANGI_PROMPT_EPS,
        },
    }


def _manifest_matches(observed: object, expected: dict[str, object]) -> bool:
    if not isinstance(observed, dict):
        return False
    return all(observed.get(key) == value for key, value in expected.items())


def _valid_cached_prompt(path: Path) -> bool:
    try:
        prompt = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    return (
        prompt.shape == (1, *PROMPT_SIZE)
        and prompt.dtype == np.float32
        and np.isfinite(prompt).all()
    )


def main() -> int:
    args = parse_args()
    names = read_sample_list(args.list_file)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        names = names[: args.limit]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    desired_in_progress = _manifest(names, args.noise, "in_progress")
    desired_complete = _manifest(names, args.noise, "complete")
    complete_path = args.cache_dir / PROMPT_CACHE_MANIFEST
    in_progress_path = args.cache_dir / IN_PROGRESS_MANIFEST

    existing_manifest_path = next(
        (path for path in (complete_path, in_progress_path) if path.is_file()), None
    )
    if existing_manifest_path is not None and not args.overwrite:
        observed = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        expected = (
            desired_complete
            if existing_manifest_path == complete_path
            else desired_in_progress
        )
        if not _manifest_matches(observed, expected):
            raise RuntimeError(
                f"Existing cache manifest is incompatible: {existing_manifest_path}. "
                "Use --overwrite to rebuild this cache."
            )
    elif existing_manifest_path is None and not args.overwrite:
        orphan = next(args.cache_dir.rglob("*.npy"), None)
        if orphan is not None:
            raise RuntimeError(
                f"Unversioned prompt files found below {args.cache_dir}. "
                "Use --overwrite to rebuild them with provenance metadata."
            )

    complete_path.unlink(missing_ok=True)
    _write_json_atomic(in_progress_path, desired_in_progress)

    for name in tqdm(names, unit="image", desc=f"Frangi ({args.noise})"):
        relative = Path(name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe sample path in split list: {name!r}")
        source, _ = resolve_sample_paths(
            args.data_root,
            name,
            image_dir=args.image_dir,
            mask_dir="masks",
        )
        destination = args.cache_dir / relative.parent / f"{relative.name}.npy"
        if destination.exists() and not args.overwrite and _valid_cached_prompt(destination):
            continue
        try:
            with Image.open(source) as source_image:
                image = np.asarray(source_image.convert("RGB"))
            image = _apply_noise(image, args.noise)
            if image.shape[:2] != IMAGE_SIZE:
                image = cv2.resize(
                    image,
                    (IMAGE_SIZE[1], IMAGE_SIZE[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
            generate_frangi_prompt(
                image,
                output_path=destination,
                prompt_size=PROMPT_SIZE,
                scales=DEFAULT_FRANGI_SCALES,
                R=3,
                K=1,
                eps=FRANGI_PROMPT_EPS,
                device=args.device,
            )
        except Exception as exc:  # keep an exact per-sample recovery log
            failures.append({"sample": name, "error": repr(exc)})
            tqdm.write(f"FAILED {name}: {exc}")

    failure_log = args.failure_log or args.cache_dir / "failures.json"
    _write_json_atomic(failure_log, failures)
    if failures:
        print(f"{len(failures)} prompt(s) failed; see {failure_log}")
        return 1
    _write_json_atomic(complete_path, desired_complete)
    in_progress_path.unlink(missing_ok=True)
    print(f"Cached {len(names)} prompts in {args.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
