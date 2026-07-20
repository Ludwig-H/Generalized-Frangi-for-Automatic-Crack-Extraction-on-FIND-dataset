#!/usr/bin/env python3
"""Evaluate a CrackSAM 2 checkpoint on the six-paper protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from cracksam2.data import PROMPT_CACHE_MANIFEST, CrackSegmentationDataset
from cracksam2.metrics import evaluate_masks, segmentation_metrics
from cracksam2.model import build_cracksam2, load_adapter_state_dict


LIST_ROOT = Path(__file__).parent / "protocol" / "cracksam_paper" / "lists"


@dataclass(frozen=True)
class EvaluationSpec:
    name: str
    root: Path
    list_file: Path
    noise: str = "original"
    prompt_cache: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-cache-root", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=(
            "khanhha_original",
            "khanhha_noisy1",
            "khanhha_noisy2",
            "road420",
            "facade390",
            "concrete3k",
        ),
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    wasserstein_group = parser.add_mutually_exclusive_group()
    wasserstein_group.add_argument(
        "--wasserstein-max-points",
        type=int,
        default=2_000,
        help="Deterministically cap each mask support (default: 2000).",
    )
    wasserstein_group.add_argument(
        "--wasserstein-exact",
        action="store_true",
        help="Use the complete pixel support; this can require prohibitive memory.",
    )
    parser.add_argument("--skip-wasserstein", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16", "none"), default="bfloat16"
    )
    return parser.parse_args()


def default_specs(data_root: Path, prompt_root: Path | None) -> list[EvaluationSpec]:
    def prompt(*parts: str) -> Path | None:
        return None if prompt_root is None else prompt_root.joinpath(*parts)

    khanhha_list = LIST_ROOT / "lists_khanhha" / "test_vol.txt"
    return [
        EvaluationSpec(
            "khanhha_original",
            data_root / "khanhha" / "test",
            khanhha_list,
            "original",
            prompt("khanhha", "test_original"),
        ),
        EvaluationSpec(
            "khanhha_noisy1",
            data_root / "khanhha" / "test",
            khanhha_list,
            "noisy1",
            prompt("khanhha", "test_noisy1"),
        ),
        EvaluationSpec(
            "khanhha_noisy2",
            data_root / "khanhha" / "test",
            khanhha_list,
            "noisy2",
            prompt("khanhha", "test_noisy2"),
        ),
        EvaluationSpec(
            "road420",
            data_root / "road420",
            LIST_ROOT / "lists_road420" / "test_vol.txt",
            "original",
            prompt("road420", "test_original"),
        ),
        EvaluationSpec(
            "facade390",
            data_root / "facade390",
            LIST_ROOT / "lists_facade390" / "test_vol.txt",
            "original",
            prompt("facade390", "test_original"),
        ),
        EvaluationSpec(
            "concrete3k",
            data_root / "concrete3k",
            LIST_ROOT / "lists_concrete3k" / "test_vol.txt",
            "original",
            prompt("concrete3k", "test_original"),
        ),
    ]


def _autocast(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty result table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = (
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    with temporary.open("w", encoding="utf-8") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    path = path.expanduser()
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "sha256": _sha256(path)}


def _ensure_evaluation_contract(
    output_root: Path, contract: dict[str, Any]
) -> Path:
    """Create or strictly verify the immutable evaluation protocol."""
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "evaluation_contract.json"
    expected = _json_safe(contract)
    if path.is_file():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid evaluation contract: {path}") from exc
        if observed != expected:
            differing = sorted(
                key
                for key in set(observed) | set(expected)
                if observed.get(key) != expected.get(key)
            )
            raise RuntimeError(
                "Evaluation output is bound to an incompatible contract; "
                f"differing fields: {differing}"
            )
        return path

    stale_artifacts = sorted(
        candidate
        for pattern in (
            "progress.jsonl",
            "per_image.csv",
            "summary.json",
            "summary.csv",
        )
        for candidate in output_root.rglob(pattern)
    )
    if stale_artifacts:
        preview = ", ".join(str(candidate) for candidate in stale_artifacts[:3])
        raise RuntimeError(
            "Evaluation artifacts exist without evaluation_contract.json; "
            f"refusing unsafe resume ({preview})"
        )
    _write_json(path, expected)
    return path


_PROGRESS_NUMERIC_FIELDS = (
    "precision",
    "recall",
    "dice",
    "iou",
    "wasserstein",
    "inference_seconds",
)


def _normalize_progress_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("Progress rows must be JSON objects")
    case_name = row.get("case_name")
    if not isinstance(case_name, str) or not case_name:
        raise ValueError("Progress row has no valid case_name")
    normalized: dict[str, Any] = {"case_name": case_name}
    for field in _PROGRESS_NUMERIC_FIELDS:
        value = row.get(field)
        if value is None and field == "wasserstein":
            normalized[field] = None
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field!r} for progress case {case_name!r}") from exc
        if not math.isfinite(number):
            if field == "wasserstein":
                normalized[field] = None
                continue
            raise ValueError(f"Non-finite {field!r} for progress case {case_name!r}")
        normalized[field] = number
    return normalized


def _append_progress_batch(
    path: Path, dataset_name: str, rows: list[dict[str, Any]]
) -> None:
    """Durably append one completed inference batch to the progress journal."""
    if not rows:
        raise ValueError("Cannot append an empty progress batch")
    entry = {
        "format_version": 1,
        "dataset": dataset_name,
        "rows": [_normalize_progress_row(row) for row in rows],
    }
    payload = json.dumps(
        _json_safe(entry), sort_keys=True, ensure_ascii=True, allow_nan=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(payload + "\n")
        output.flush()
        os.fsync(output.fileno())


def _read_progress_rows(
    path: Path, *, expected_dataset: str
) -> dict[str, dict[str, Any]]:
    """Read and deduplicate a journal, repairing only a truncated final line."""
    if not path.is_file():
        return {}

    rows_by_case: dict[str, dict[str, Any]] = {}
    size = path.stat().st_size
    valid_end = 0
    append_newline = False
    with path.open("rb") as source:
        line_number = 0
        while True:
            line_start = source.tell()
            raw_line = source.readline()
            if not raw_line:
                break
            line_number += 1
            line_end = source.tell()
            if not raw_line.strip():
                valid_end = line_end
                continue
            try:
                entry = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if line_end == size and not raw_line.endswith(b"\n"):
                    valid_end = line_start
                    break
                raise ValueError(
                    f"Invalid progress journal {path} at line {line_number}"
                ) from exc
            if not isinstance(entry, dict) or entry.get("format_version") != 1:
                raise ValueError(
                    f"Unsupported progress entry in {path} at line {line_number}"
                )
            if entry.get("dataset") != expected_dataset:
                raise ValueError(
                    f"Progress dataset mismatch in {path} at line {line_number}: "
                    f"{entry.get('dataset')!r} != {expected_dataset!r}"
                )
            batch_rows = entry.get("rows")
            if not isinstance(batch_rows, list) or not batch_rows:
                raise ValueError(
                    f"Empty or invalid progress batch in {path} at line {line_number}"
                )
            for row in batch_rows:
                normalized = _normalize_progress_row(row)
                rows_by_case[normalized["case_name"]] = normalized
            valid_end = line_end
            append_newline = not raw_line.endswith(b"\n")

    if valid_end < size:
        with path.open("r+b") as output:
            output.truncate(valid_end)
            output.flush()
            os.fsync(output.fileno())
    elif append_newline:
        with path.open("ab") as output:
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
    return rows_by_case


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    spec: EvaluationSpec,
    variant: str,
    wasserstein_max_points: int | None,
    skip_wasserstein: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset": spec.name,
        "variant": variant,
        "noise": spec.noise,
        "samples": len(rows),
        "wasserstein_exact": wasserstein_max_points is None and not skip_wasserstein,
        "wasserstein_max_points": wasserstein_max_points,
    }
    for metric in ("precision", "recall", "dice", "iou", "wasserstein"):
        values = np.asarray(
            [np.nan if row[metric] is None else row[metric] for row in rows],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        summary[metric] = float(np.mean(finite)) if finite.size else None
        summary[f"{metric}_std"] = float(np.std(finite)) if finite.size else None
        summary[f"{metric}_finite_samples"] = int(finite.size)
    inference_times = np.asarray(
        [row["inference_seconds"] for row in rows], dtype=np.float64
    )
    summary["mean_inference_seconds"] = float(np.mean(inference_times))
    return summary


def _save_prediction(root: Path, case_name: str, prediction: np.ndarray) -> None:
    relative = Path(case_name.replace("\\", "/"))
    destination = root / relative.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((prediction.astype(np.uint8) * 255)).save(destination)


def evaluate_spec(
    model: torch.nn.Module,
    spec: EvaluationSpec,
    *,
    variant: str,
    output_root: Path,
    device: torch.device,
    amp_dtype: str,
    batch_size: int,
    num_workers: int,
    threshold: float,
    wasserstein_max_points: int | None,
    skip_wasserstein: bool,
    save_predictions: bool,
    max_samples: int | None,
) -> dict[str, Any]:
    use_frangi = variant == "frangi"
    if use_frangi and spec.prompt_cache is None:
        raise ValueError(f"No Frangi prompt cache configured for {spec.name}")
    dataset = CrackSegmentationDataset(
        spec.root,
        list_file=spec.list_file,
        split="test_vol",
        image_size=448,
        prompt_size=256,
        noise_mode=spec.noise,
        augment=False,
        prompt_cache_dir=spec.prompt_cache if use_frangi else None,
    )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max-samples must be positive")
        dataset.sample_names = dataset.sample_names[:max_samples]
    selected_names = list(dataset.sample_names)
    if len(selected_names) != len(set(selected_names)):
        raise ValueError(f"Dataset {spec.name} contains duplicate sample names")

    result_dir = output_root / spec.name
    progress_path = result_dir / "progress.jsonl"
    rows_by_case = _read_progress_rows(
        progress_path, expected_dataset=spec.name
    )
    unexpected = sorted(set(rows_by_case) - set(selected_names))
    if unexpected:
        raise RuntimeError(
            f"Progress for {spec.name} contains cases outside the selected split: "
            f"{unexpected[:5]}"
        )
    prediction_root = result_dir / "predictions"
    completed_names = set(rows_by_case)
    if save_predictions:
        completed_names = {
            name
            for name in completed_names
            if (
                prediction_root
                / Path(name.replace("\\", "/")).with_suffix(".png")
            ).is_file()
        }
    dataset.sample_names = [
        name for name in selected_names if name not in completed_names
    ]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=spec.name, unit="batch"):
            images = batch["image"].to(device, non_blocking=True)
            prompts = (
                batch["prompt"].to(device, non_blocking=True) if use_frangi else None
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with _autocast(device, amp_dtype):
                output = model(images, mask_input=prompts, output_size=(448, 448))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            predictions = (torch.sigmoid(output["logits"].float()) > threshold).cpu().numpy()
            targets = batch["mask"].numpy()
            per_sample_time = elapsed / images.shape[0]
            batch_rows: list[dict[str, Any]] = []
            for case_name, prediction, target in zip(
                batch["case_name"], predictions, targets
            ):
                prediction_2d = prediction[0].astype(np.float32)
                target_2d = target[0].astype(np.float32)
                if skip_wasserstein:
                    values = segmentation_metrics(
                        prediction_2d, target_2d, threshold=threshold
                    )
                    values["wasserstein"] = None
                else:
                    values = dict(
                        evaluate_masks(
                            prediction_2d,
                            target_2d,
                            threshold=threshold,
                            max_points=wasserstein_max_points,
                        )
                    )
                row = {
                    "case_name": case_name,
                    **values,
                    "inference_seconds": per_sample_time,
                }
                batch_rows.append(row)
                if save_predictions:
                    _save_prediction(prediction_root, case_name, prediction_2d)
            _append_progress_batch(progress_path, spec.name, batch_rows)
            rows_by_case.update(
                (row["case_name"], _normalize_progress_row(row)) for row in batch_rows
            )

    missing = [name for name in selected_names if name not in rows_by_case]
    if missing:
        raise RuntimeError(
            f"Evaluation for {spec.name} ended with {len(missing)} missing cases"
        )
    rows = [rows_by_case[name] for name in selected_names]
    summary = _summarize_rows(
        rows,
        spec=spec,
        variant=variant,
        wasserstein_max_points=wasserstein_max_points,
        skip_wasserstein=skip_wasserstein,
    )
    _write_csv(result_dir / "per_image.csv", rows)
    _write_json(result_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but no CUDA device is available")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    if args.wasserstein_max_points is not None and args.wasserstein_max_points <= 0:
        raise ValueError("wasserstein-max-points must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    checkpoint = torch.load(
        args.adapter_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        observed_version = (
            checkpoint.get("format_version")
            if isinstance(checkpoint, dict)
            else type(checkpoint).__name__
        )
        raise ValueError(
            "unsupported adapter checkpoint format_version: "
            f"expected 1, got {observed_version!r}"
        )
    variant = str(checkpoint.get("variant", ""))
    if variant not in ("baseline", "frangi"):
        raise ValueError("adapter checkpoint does not identify baseline/frangi variant")
    lora = checkpoint.get("lora", {})
    rank = int(lora.get("rank", 0))
    alpha = float(lora.get("alpha", rank))
    config = checkpoint.get("model_config")
    if not config or rank not in (4, 8):
        raise ValueError("adapter checkpoint is missing its SAM 2/LoRA metadata")
    if variant == "frangi" and args.prompt_cache_root is None:
        raise ValueError("Frangi evaluation requires --prompt-cache-root")
    expected_base = checkpoint.get("base_checkpoint")
    observed_base = _file_identity(args.sam2_checkpoint)
    if expected_base:
        if observed_base != expected_base:
            raise RuntimeError(
                f"SAM 2 base checkpoint mismatch: {observed_base} != {expected_base}"
            )
    if "adapter" not in checkpoint:
        raise ValueError("adapter checkpoint has no adapter state")

    specs = default_specs(args.data_root, args.prompt_cache_root)
    selected = set(args.datasets or (spec.name for spec in specs))
    specs = [spec for spec in specs if spec.name in selected]
    wasserstein_max_points = (
        None if args.wasserstein_exact else args.wasserstein_max_points
    )
    dataset_contracts = [
        {
            "name": spec.name,
            "root": str(spec.root.expanduser().resolve()),
            "list_file": {
                **_file_identity(spec.list_file),
                "path": str(spec.list_file.expanduser().resolve()),
            },
            "noise": spec.noise,
            "prompt_cache": (
                None
                if spec.prompt_cache is None
                else {
                    "path": str(spec.prompt_cache.expanduser().resolve()),
                    "manifest": _file_identity(
                        spec.prompt_cache / PROMPT_CACHE_MANIFEST
                    ),
                }
            ),
        }
        for spec in specs
    ]
    contract = {
        "format_version": 1,
        "adapter_checkpoint": _file_identity(args.adapter_checkpoint),
        "base_checkpoint": observed_base,
        "variant": variant,
        "datasets": dataset_contracts,
        "threshold": args.threshold,
        "wasserstein": {
            "skip": args.skip_wasserstein,
            "exact": wasserstein_max_points is None and not args.skip_wasserstein,
            "max_points": wasserstein_max_points,
        },
        "max_samples": args.max_samples,
        "amp": {
            "dtype": args.amp_dtype,
            "enabled": device.type == "cuda" and args.amp_dtype != "none",
        },
        "device_type": device.type,
        "batch_size": args.batch_size,
        "save_predictions": args.save_predictions,
        "software": {"torch": torch.__version__, "numpy": np.__version__},
        "code": {
            path.name: _file_identity(path)
            for path in (
                Path(__file__),
                Path(__file__).parent / "cracksam2" / "data.py",
                Path(__file__).parent / "cracksam2" / "metrics.py",
                Path(__file__).parent / "cracksam2" / "model.py",
            )
        },
    }
    _ensure_evaluation_contract(args.output, contract)

    model, report = build_cracksam2(
        args.sam2_checkpoint,
        rank=rank,
        alpha=alpha,
        config=config,
        device=device,
    )
    load_adapter_state_dict(model, checkpoint["adapter"], strict=True)
    print(
        f"Loaded {variant} checkpoint with {report.trainable_parameters:,} LoRA parameters"
    )

    summaries = [
        evaluate_spec(
            model,
            spec,
            variant=variant,
            output_root=args.output,
            device=device,
            amp_dtype=args.amp_dtype,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
            wasserstein_max_points=wasserstein_max_points,
            skip_wasserstein=args.skip_wasserstein,
            save_predictions=args.save_predictions,
            max_samples=args.max_samples,
        )
        for spec in specs
    ]
    _write_csv(args.output / "summary.csv", summaries)
    _write_json(args.output / "summary.json", summaries)
    print(json.dumps(_json_safe(summaries), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
