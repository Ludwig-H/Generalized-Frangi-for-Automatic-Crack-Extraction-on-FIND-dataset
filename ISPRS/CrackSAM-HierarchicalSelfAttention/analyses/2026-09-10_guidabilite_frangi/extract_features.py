#!/usr/bin/env python3
"""Extract compact, unguided baseline SAM 2 + LoRA features; no training.

The final H is useful for a separate confidence pass. ``--include-pre-global``
additionally pools the input of the last global Hiera attention, which would be
available before that attention in a single-pass implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import nullcontext

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CRACKSAM = REPO / "ISPRS/CrackSAM"
FOUNDATION_SHA = "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"
ADAPTER_SHA = "d154d60a82ec2a0af4540559785a483818612319350c04a1b06035053b6f6a04"
CONFIG = "configs/sam2/sam2_hiera_l.yaml"
DIMENSIONS = {"mean": 256, "std": 256, "grid2": 1024, "multiscale_mean": 352}
DATASETS = {
    "khanhha_original": ("khanhha/test", "khanhha", "original"),
    "khanhha_noisy1": ("khanhha/test", "khanhha", "noisy1"),
    "khanhha_noisy2": ("khanhha/test", "khanhha", "noisy2"),
    "road420": ("road420", "road420", "original"),
    "facade390": ("facade390", "facade390", "original"),
    "concrete3k": ("concrete3k", "concrete3k", "original"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_baseline_payload(path):
    """Keep weights-only loading; allow the historical TorchVersion metadata."""
    import torch
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".tmp-", delete=False) as f:
        temporary = f.name
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".tmp-", delete=False) as f:
        temporary = f.name
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def ensure_contract(path, contract):
    """Never mix shards from different weights, inputs, code or precision."""
    digest = canonical_sha(contract)
    if Path(path).exists():
        previous = json.loads(Path(path).read_text())
        if previous != {"sha256": digest, "contract": contract}:
            raise ValueError("Extraction contract changed: use a new output directory.")
    else:
        atomic_json(path, {"sha256": digest, "contract": contract})
    return digest


def read_cases(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty cases CSV")
    seen = set()
    for row in rows:
        expected = row["dataset"] + "::" + row["case_name"]
        if row["image_id"] != expected or expected in seen:
            raise ValueError(f"Invalid or duplicate image_id: {row['image_id']}")
        if row["dataset"] not in DATASETS:
            raise ValueError(f"Unknown dataset: {row['dataset']}")
        seen.add(expected)
    return rows


def select_verification(rows, count):
    """Round-robin datasets, then SHA256 order; independent of gain/loss labels."""
    buckets = {}
    for row in rows:
        buckets.setdefault(row["dataset"], []).append(row["image_id"])
    for values in buckets.values():
        values.sort(key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    selected = []
    while len(selected) < min(count, len(rows)):
        for name in sorted(buckets):
            if buckets[name] and len(selected) < count:
                selected.append(buckets[name].pop(0))
    return set(selected)


def expected_scores(path):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("milestone", "epoch25_best") != "epoch25_best":
                continue
            key = row["dataset"] + "::" + row["case_name"]
            value = float(row["baseline_iou"])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid reference IoU: {key}")
            if key in result:
                raise ValueError(f"Duplicate reference IoU: {key}")
            result[key] = value
    return result


class CaseDataset:
    """Select CSV cases while reusing historical loading and noise transforms."""
    def __init__(self, rows, data_root):
        from cracksam2.data import CrackSegmentationDataset
        self.rows = rows
        self.datasets = {}
        self.indices = {}
        for name in dict.fromkeys(row["dataset"] for row in rows):
            relative, list_name, noise = DATASETS[name]
            dataset = CrackSegmentationDataset(
                data_root / relative,
                list_file=CRACKSAM / "protocol/cracksam_paper/lists" /
                          f"lists_{list_name}/test_vol.txt",
                split="test_vol", image_size=448, noise_mode=noise,
                augment=False, prompt_cache_dir=None,
            )
            self.datasets[name] = dataset
            self.indices[name] = {value: index for index, value in enumerate(dataset.sample_names)}
        for row in rows:
            if row["case_name"] not in self.indices[row["dataset"]]:
                raise ValueError(f"Case outside historical test split: {row['image_id']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample = self.datasets[row["dataset"]][self.indices[row["dataset"]][row["case_name"]]]
        sample["image_id"] = row["image_id"]
        return sample

    def fingerprint(self):
        files, samples = {}, []
        for row in self.rows:
            dataset = self.datasets[row["dataset"]]
            paths = dataset._paths(row["case_name"])
            identifiers = []
            for path in paths:
                key = str(path.resolve())
                if key not in files:
                    files[key] = {"sha256": sha256(path), "bytes": path.stat().st_size}
                identifiers.append(key)
            samples.append({"image_id": row["image_id"], "image": identifiers[0], "mask": identifiers[1]})
        return {"files": files, "samples": samples}


def pool_features(features):
    """Channel-major 2x2 grid, population std, and final/32/64 channel means."""
    import torch
    import torch.nn.functional as F
    h = features.image_embeddings.float()
    high = features.high_resolution_features
    expected = ((32, 256, 256), (64, 128, 128))
    if tuple(h.shape[1:]) != (256, 64, 64) or len(high) != 2:
        raise ValueError("Unexpected SAM 2 Hiera-L feature shapes")
    if any(tuple(value.shape[1:]) != shape for value, shape in zip(high, expected)):
        raise ValueError("Unexpected high-resolution feature shapes")
    mean = h.mean((-2, -1))
    values = {
        "mean": mean,
        "std": h.std((-2, -1), unbiased=False),
        "grid2": F.adaptive_avg_pool2d(h, (2, 2)).flatten(1),
        "multiscale_mean": torch.cat([mean] + [value.float().mean((-2, -1)) for value in high], dim=1),
    }
    result = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in values.items()}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Non-finite features")
    return result


def install_pre_global_hook(model):
    """Capture attention input after norm1, before qkv, in a global Hiera block."""
    blocks = model.sam2.image_encoder.trunk.blocks
    indices = [i for i, block in enumerate(blocks) if getattr(block, "window_size", None) == 0]
    if not indices:
        raise ValueError("No unambiguous global Hiera block (window_size == 0)")
    index = indices[-1]
    state = {}

    def capture(module, inputs):
        h = inputs[0]
        if h.ndim != 4:
            raise ValueError("Global Hiera attention input must be BHWC")
        state["mean"] = h.float().mean((1, 2)).detach().cpu().numpy().astype(np.float32)
        state["shape"] = list(h.shape[1:])

    handle = blocks[index].attn.register_forward_pre_hook(capture)
    description = {"module": f"sam2.image_encoder.trunk.blocks.{index}.attn",
                   "location": "forward_pre_hook: normalized BHWC input before qkv/global attention",
                   "block_index": index, "pooling": "spatial mean over H,W"}
    return state, handle, description


def validate_shard(path, expected_ids, digest):
    with np.load(path, allow_pickle=False) as f:
        data = {key: f[key] for key in f.files}
    if data.get("contract_sha256", np.array("")).item() != digest:
        raise ValueError(f"Wrong shard contract: {path}")
    if data["ids"].tolist() != expected_ids:
        raise ValueError(f"Wrong shard IDs/order: {path}")
    for key, columns in DIMENSIONS.items():
        if data[key].shape != (len(expected_ids), columns) or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key} shard: {path}")
    if "pre_global_mean" in data:
        value = data["pre_global_mean"]
        if value.ndim != 2 or value.shape[0] != len(expected_ids) or not np.isfinite(value).all():
            raise ValueError(f"Invalid pre-global shard: {path}")
    return data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=HERE / "tables/categories.csv")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "none"), default="bfloat16")
    parser.add_argument("--include-pre-global", action="store_true")
    parser.add_argument("--verify-count", type=int, default=18)
    parser.add_argument("--verify-tolerance", type=float, default=0.01,
                        help="Maximum absolute per-image IoU difference; abort above it.")
    parser.add_argument("--reference-csv", type=Path, default=CRACKSAM /
                        "results/frangi_milestone_report/tables/per_image_all_milestones.csv")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0 or args.verify_count < 0 or args.verify_tolerance < 0:
        parser.error("Invalid batch size, workers or verification options")
    return args


def main(argv=None):
    args = parse_args(argv)
    rows = read_cases(args.cases)
    for path, required in ((args.foundation, FOUNDATION_SHA), (args.adapter, ADAPTER_SHA)):
        if sha256(path) != required:
            raise ValueError(f"Historical checkpoint SHA256 mismatch: {path}")
    sys.path.insert(0, str(CRACKSAM))
    import torch
    from torch.utils.data import DataLoader, Subset
    from cracksam2.model import build_cracksam2, load_adapter_state_dict
    from cracksam2.metrics import segmentation_metrics

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu explicitly.")
    torch.manual_seed(20260910)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = CaseDataset(rows, args.data_root.resolve())
    print(f"Hashing source images/masks for {len(rows)} observations...", flush=True)
    inputs = dataset.fingerprint()
    versions = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python-headless", "scipy", "pillow", "SAM-2"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    code_paths = [Path(__file__), CRACKSAM / "cracksam2/model.py", CRACKSAM / "cracksam2/data.py",
                  CRACKSAM / "cracksam2/metrics.py"]
    import sam2
    sam_root = Path(sam2.__file__).parent
    for relative in ("modeling/backbones/hieradet.py", "modeling/backbones/image_encoder.py", "modeling/sam2_base.py"):
        code_paths.append(sam_root / relative)
    payload = load_baseline_payload(args.adapter)
    if payload.get("variant") != "baseline" or payload.get("format_version") != 1:
        raise ValueError("Expected historical baseline LoRA checkpoint, format_version=1")
    lora = payload["lora"]
    if lora["rank"] != 4:
        raise ValueError("Expected rank-4 historical LoRA")
    model, _ = build_cracksam2(args.foundation, rank=lora["rank"], alpha=lora["alpha"],
                              config=CONFIG, device=device)
    load_adapter_state_dict(model, payload["adapter"], strict=True)
    model.requires_grad_(False).eval()
    hook_state, hook_handle, hook_description = {}, None, None
    if args.include_pre_global:
        hook_state, hook_handle, hook_description = install_pre_global_hook(model)
    verify_ids = select_verification(rows, args.verify_count)
    reference = expected_scores(args.reference_csv) if verify_ids else {}
    if verify_ids - reference.keys():
        raise ValueError("Missing historical verification scores")
    amp = args.amp_dtype if device.type == "cuda" else "none"
    contract = {
        "format_version": 1, "cases_sha256": sha256(args.cases),
        "input_manifest_sha256": canonical_sha(inputs), "foundation_sha256": FOUNDATION_SHA,
        "adapter_sha256": ADAPTER_SHA, "sam2_config": CONFIG, "lora": {"rank": 4, "alpha": lora["alpha"]},
        "versions": versions, "code_sha256": {str(path): sha256(path) for path in code_paths},
        "device": str(device), "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "amp_dtype": amp, "batch_size": args.batch_size, "seed": 20260910,
        "preprocessing": "CrackSegmentationDataset RGB/noise/cubic448, encode_images bilinear1024/ImageNet/no_mem_embed",
        "pooling": {"mean": "final H spatial mean", "std": "final H population std",
                    "grid2": "final H adaptive 2x2 mean, channel-major flattened",
                    "multiscale_mean": "concat(final H mean, high_res[0] mean, high_res[1] mean)"},
        "pre_global": hook_description,
        "verification": {"ids": sorted(verify_ids), "reference_sha256": sha256(args.reference_csv) if verify_ids else None,
                         "threshold": 0.5, "absolute_iou_tolerance": args.verify_tolerance},
    }
    output = args.output_dir
    digest = ensure_contract(output / "contract.json", contract)
    atomic_json(output / "input_manifest.json", inputs)
    batches = [list(range(i, min(i + args.batch_size, len(rows)))) for i in range(0, len(rows), args.batch_size)]
    pending = []
    for number, indices in enumerate(batches):
        shard = output / "shards" / f"{number:06d}.npz"
        if shard.exists():
            validate_shard(shard, [rows[i]["image_id"] for i in indices], digest)
        else:
            pending.append(number)
    pending_indices = [i for number in pending for i in batches[number]]
    loader = DataLoader(Subset(dataset, pending_indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    print(f"{len(pending)}/{len(batches)} batches to extract; no training.", flush=True)
    try:
        with torch.inference_mode():
            for number, batch in zip(pending, loader):
                ids = list(batch["image_id"])
                expected_ids = [rows[i]["image_id"] for i in batches[number]]
                if ids != expected_ids:
                    raise ValueError("Batch boundary changed during resume")
                images = batch["image"].to(device, non_blocking=True)
                context = torch.autocast("cuda", dtype=torch.bfloat16) if amp != "none" else nullcontext()
                with context:
                    features = model.encode_images(images)
                    needs_verify = any(value in verify_ids for value in ids)
                    decoded = model.decode_features(features, mask_input=None)["logits"] if needs_verify else None
                arrays = pool_features(features)
                arrays["ids"] = np.asarray(ids, dtype=str)
                arrays["contract_sha256"] = np.asarray(digest)
                if args.include_pre_global:
                    arrays["pre_global_mean"] = hook_state["mean"]
                    arrays["pre_global_input_shape"] = np.asarray(hook_state["shape"], dtype=np.int64)
                verified, observed, expected = [], [], []
                if needs_verify:
                    probabilities = torch.sigmoid(decoded).float().cpu()
                    for j, image_id in enumerate(ids):
                        if image_id not in verify_ids:
                            continue
                        iou = segmentation_metrics(probabilities[j], batch["mask"][j])["iou"]
                        verified.append(image_id)
                        observed.append(iou)
                        expected.append(reference[image_id])
                arrays["verify_ids"] = np.asarray(verified, dtype=str)
                arrays["verify_observed"] = np.asarray(observed, dtype=np.float64)
                arrays["verify_expected"] = np.asarray(expected, dtype=np.float64)
                if any(abs(a - b) > args.verify_tolerance for a, b in zip(observed, expected)):
                    atomic_json(output / "verification_failed.json", {"ids": verified, "observed": observed,
                                                                      "expected": expected, "contract_sha256": digest})
                    raise RuntimeError("Baseline IoU verification failed; inspect verification_failed.json")
                atomic_npz(output / "shards" / f"{number:06d}.npz", arrays)
                print(f"batch {number + 1}/{len(batches)} ({len(ids)} images)", flush=True)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    shards = [validate_shard(output / "shards" / f"{number:06d}.npz",
                             [rows[i]["image_id"] for i in indices], digest)
              for number, indices in enumerate(batches)]
    keys = ["ids", *DIMENSIONS] + (["pre_global_mean"] if args.include_pre_global else [])
    combined = {key: np.concatenate([part[key] for part in shards], axis=0) for key in keys}
    atomic_npz(output / "features.npz", combined)
    verification = [{"image_id": image_id, "observed_iou": float(observed), "historical_iou": float(expected),
                     "absolute_error": abs(float(observed) - float(expected))}
                    for shard in shards for image_id, observed, expected in
                    zip(shard["verify_ids"].tolist(), shard["verify_observed"], shard["verify_expected"])]
    if set(value["image_id"] for value in verification) != verify_ids:
        raise RuntimeError("Verification records incomplete")
    for value in verification:
        if (value["historical_iou"] != reference[value["image_id"]]
                or not np.isfinite(value["observed_iou"])
                or value["absolute_error"] > args.verify_tolerance):
            raise RuntimeError("Resumed baseline verification is inconsistent")
    atomic_json(output / "metadata.json", {
        "status": "complete", "contract_sha256": digest, "contract": contract,
        "features_sha256": sha256(output / "features.npz"), "n_images": len(rows),
        "shapes": {key: list(value.shape) for key, value in combined.items()},
        "dtypes": {key: str(value.dtype) for key, value in combined.items()},
        "verification": verification, "verified": bool(verify_ids),
        "pre_global_input_shape": (shards[0]["pre_global_input_shape"].tolist()
                                   if args.include_pre_global else None),
        "limitations": ["Frozen historical baseline features; no retraining or Frangi input.",
                        "Final H is available after encoding: earlier attention requires a separate pass or pre-global features.",
                        "Image pooling discards some spatial information; no negative probe establishes absence of information."],
    })
    print(f"Complete: {output / 'features.npz'}", flush=True)


if __name__ == "__main__":
    main()
