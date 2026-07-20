#!/usr/bin/env python3
"""Precompute the resumable seven-channel Frangi raster cache (schema v2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image
import scipy
import torch
from tqdm import tqdm

import cracksam2.frangi as frangi_module
import cracksam2.graph_types as graph_types_module
import cracksam2.data as data_module
from cracksam2.data import (
    apply_noise_perturbation,
    normalize_noise_mode,
    read_sample_list,
    resolve_sample_paths,
)
from cracksam2.frangi import DEFAULT_FRANGI_SCALES, generate_frangi_raster
from cracksam2.graph_cache import (
    GRAPH_CACHE_IN_PROGRESS,
    GRAPH_CACHE_MANIFEST,
    aggregate_graph_cache_records,
    assert_manifest_compatible,
    build_graph_cache_contract,
    graph_cache_record,
    graph_cache_relative_path,
    implementation_sha256,
    load_graph_raster,
    save_graph_raster_atomic,
    sha256_file,
    validate_completed_graph_cache,
    write_json_atomic,
)
from cracksam2.graph_types import FrangiRasterSample


CRACKSAM_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--image-dir")
    parser.add_argument("--mask-dir")
    parser.add_argument(
        "--noise",
        choices=("none", "original", "noisy1", "noisy2"),
        default="none",
    )
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scales", nargs="+", type=float, default=DEFAULT_FRANGI_SCALES)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--ss", type=float, default=1.0)
    parser.add_argument("--si", type=float, default=0.25)
    parser.add_argument("--sa", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.18)
    parser.add_argument("--min-rel-size", type=float, default=120.0)
    parser.add_argument("--graph-order", type=int, choices=(1, 2), default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--failure-log", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _frangi_parameters(args: argparse.Namespace) -> dict[str, object]:
    return {
        "scales": [float(value) for value in args.scales],
        "R": int(args.radius),
        "ss": float(args.ss),
        "si": float(args.si),
        "sa": float(args.sa),
        "tau": float(args.tau),
        "min_rel_size": float(args.min_rel_size),
        "K": int(args.graph_order),
        "compute_centrality": True,
        "hessian_scale_normalization": "sigma_squared",
        "distance_normalization": "image_diagonal",
        "runtime": _runtime_contract(args.device),
    }


def _runtime_contract(device_name: str) -> dict[str, object]:
    """Record the numerical backend so a resumed cache cannot mix backends."""
    device = torch.device(device_name)
    runtime: dict[str, object] = {
        "device": str(device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA cache requested but Torch cannot access CUDA")
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        runtime.update(
            {
                "cuda_device_index": int(device_index),
                "cuda_device_name": torch.cuda.get_device_name(device_index),
                "cuda_capability": list(torch.cuda.get_device_capability(device_index)),
            }
        )
    return runtime


def _extractor_sha256() -> str:
    return implementation_sha256(
        (
            Path(__file__),
            Path(data_module.__file__),
            Path(frangi_module.__file__),
            Path(graph_types_module.__file__),
            REPOSITORY_ROOT / "ISPRS" / "src" / "frangi_hessian.py",
            REPOSITORY_ROOT / "ISPRS" / "src" / "graph_extraction.py",
        ),
        relative_to=REPOSITORY_ROOT,
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid graph cache manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Graph cache manifest must be an object: {path}")
    return value


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if (args.image_dir is None) != (args.mask_dir is None):
        raise ValueError("--image-dir and --mask-dir must be provided together")
    if not args.scales or any(float(value) <= 0.0 for value in args.scales):
        raise ValueError("--scales must contain positive values")
    if args.radius <= 0:
        raise ValueError("--radius must be positive")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError("--tau must lie in (0, 1]")
    if args.min_rel_size <= 0.0:
        raise ValueError("--min-rel-size must be positive")


def _load_processed_image(path: Path, noise: str, image_size: int) -> np.ndarray:
    with Image.open(path) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.uint8)
    image = apply_noise_perturbation(image, noise, output_size=image_size)
    if image.shape[:2] != (image_size, image_size):
        image = cv2.resize(
            image,
            (image_size, image_size),
            interpolation=cv2.INTER_CUBIC,
        )
    return np.ascontiguousarray(image)


def _reuse_sample(
    path: Path,
    *,
    case_name: str,
    image_sha256: str,
    mask_sha256: str,
    image_size: int,
) -> FrangiRasterSample | None:
    if not path.is_file():
        return None
    try:
        return load_graph_raster(
            path,
            expected_case_name=case_name,
            expected_image_sha256=image_sha256,
            expected_mask_sha256=mask_sha256,
            expected_image_size=(image_size, image_size),
        )
    except ValueError:
        return None


def run(args: argparse.Namespace) -> int:
    """Generate or resume a cache. Exposed separately for interruption tests."""
    _validate_arguments(args)
    names = read_sample_list(args.list_file)
    if args.limit is not None:
        names = names[: args.limit]
    noise = normalize_noise_mode(args.noise)
    parameters = _frangi_parameters(args)
    contract = build_graph_cache_contract(
        names,
        image_size=(args.image_size, args.image_size),
        noise_mode=noise,
        frangi_parameters=parameters,
        extractor_sha256=_extractor_sha256(),
    )

    cache_dir = args.cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    complete_path = cache_dir / GRAPH_CACHE_MANIFEST
    in_progress_path = cache_dir / GRAPH_CACHE_IN_PROGRESS
    existing_paths = [path for path in (complete_path, in_progress_path) if path.is_file()]
    if existing_paths and not args.overwrite:
        for path in existing_paths:
            assert_manifest_compatible(_read_json(path), contract)
    elif not existing_paths and not args.overwrite:
        orphan = next(cache_dir.rglob("*.npz"), None)
        if orphan is not None:
            raise RuntimeError(
                f"Unversioned graph cache file found: {orphan}. "
                "Use --overwrite to rebuild it with provenance metadata."
            )

    # A complete manifest is withdrawn while its entries are checked/repaired;
    # every individual .npz remains atomic and reusable after Spot preemption.
    complete_path.unlink(missing_ok=True)
    write_json_atomic(in_progress_path, {**contract, "status": "in_progress"})

    failures: list[dict[str, str]] = []
    records_by_name: dict[str, dict[str, object]] = {}
    progress = tqdm(
        names,
        unit="image",
        desc=f"Frangi raster v2 ({noise})",
        disable=bool(args.quiet),
    )
    for case_name in progress:
        try:
            image_path, mask_path = resolve_sample_paths(
                args.data_root,
                case_name,
                image_dir=args.image_dir,
                mask_dir=args.mask_dir,
            )
            image_digest = sha256_file(image_path)
            mask_digest = sha256_file(mask_path)
            cache_path = cache_dir / graph_cache_relative_path(case_name)
            sample = None
            if not args.overwrite:
                sample = _reuse_sample(
                    cache_path,
                    case_name=case_name,
                    image_sha256=image_digest,
                    mask_sha256=mask_digest,
                    image_size=args.image_size,
                )
            if sample is None:
                image = _load_processed_image(image_path, noise, args.image_size)
                started = time.perf_counter()
                raster = generate_frangi_raster(
                    image,
                    scales=tuple(float(value) for value in args.scales),
                    R=args.radius,
                    ss=args.ss,
                    si=args.si,
                    sa=args.sa,
                    tau=args.tau,
                    min_rel_size=args.min_rel_size,
                    K=args.graph_order,
                    device=args.device,
                )
                sample = FrangiRasterSample(
                    case_name=case_name,
                    raster=raster,
                    image_sha256=image_digest,
                    mask_sha256=mask_digest,
                    elapsed_seconds=time.perf_counter() - started,
                )
                save_graph_raster_atomic(cache_path, sample)
            records_by_name[case_name] = graph_cache_record(
                cache_dir, cache_path, sample
            )
        except Exception as exc:  # retain exact per-sample recovery information
            failures.append({"sample": case_name, "error": repr(exc)})
            if not args.quiet:
                tqdm.write(f"FAILED {case_name}: {exc}")

    failure_log = args.failure_log or cache_dir / "failures.json"
    write_json_atomic(failure_log, failures)
    if failures:
        print(f"{len(failures)} raster(s) failed; resumable state: {in_progress_path}")
        return 1

    records = [records_by_name[name] for name in names]
    complete_manifest = {
        **contract,
        "status": "complete",
        "entries": records,
        "aggregate": aggregate_graph_cache_records(records),
    }
    write_json_atomic(complete_path, complete_manifest)
    in_progress_path.unlink(missing_ok=True)
    validate_completed_graph_cache(cache_dir, contract, verify_files=False)
    print(f"Cached {len(records)} Frangi raster(s) in {cache_dir}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
