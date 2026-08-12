#!/usr/bin/env python3
"""Cache les logits du CrackSAM Khánh Hà **gelé** sur les RGB d'IRT-Crack.

Chaîne exacte, et pourquoi :

1. l'image est lue en RGB puis redimensionnée en ``448×448`` par interpolation
   bicubique — c'est mot pour mot ce que fait ``cracksam2/data.py`` à
   l'entraînement, donc la baseline voit la distribution sur laquelle elle a été
   apprise ;
2. SAM 2 produit des logits ``448²`` (``mask_input=None``, aucun prompt) ;
3. ces logits sont ré-échantillonnés bilinéairement vers la **résolution native**
   ``480×640`` et cachés ainsi.

Le rééchantillonnage ne porte donc que sur des logits lisses : la vérité terrain
n'est jamais rééchantillonnée, et toutes les métriques sont calculées à la
résolution d'annotation.

Aucun gradient n'est calculé ; le modèle est en ``eval`` et sous ``no_grad``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_residual._repo import ensure_repository_on_path  # noqa: E402

ensure_repository_on_path()

from thermal_residual.cache import CacheWriter  # noqa: E402
from thermal_residual.constants import BASELINE_CACHE_VERSION, BASELINE_INPUT_SIZE  # noqa: E402
from thermal_residual.manifest import manifest_digest, read_manifest  # noqa: E402
from thermal_residual.provenance import sha256_file  # noqa: E402

from cracksam2.model import build_cracksam2, load_adapter_state_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint", required=True, type=Path, help="poids SAM 2 Hiera-L")
    parser.add_argument(
        "--lora-checkpoint",
        required=True,
        type=Path,
        help="checkpoint CrackSAM (clé « adapter »), par exemple le barreau tol3 de GeoLoRA",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--input-size",
        type=int,
        default=BASELINE_INPUT_SIZE[0],
        help="côté du carré donné à SAM (448 = distribution d'entraînement Khánh Hà)",
    )
    parser.add_argument("--limit", type=int, default=0, help="ne traiter que les N premiers (débogage)")
    return parser.parse_args()


def load_batch(paths: list[Path], size: int) -> torch.Tensor:
    import cv2

    images = []
    for path in paths:
        array = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if array is None:
            raise FileNotFoundError(f"image illisible : {path}")
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        if array.shape[:2] != (size, size):
            array = cv2.resize(array, (size, size), interpolation=cv2.INTER_CUBIC)
        images.append(array.transpose(2, 0, 1).astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(images))


def main() -> int:
    args = parse_args()
    samples = read_manifest(args.manifest)
    if args.limit:
        samples = samples[: args.limit]

    device = torch.device(args.device)
    checkpoint_sha = sha256_file(args.lora_checkpoint)
    print(f"checkpoint LoRA : {args.lora_checkpoint} (sha256 {checkpoint_sha[:16]}…)")

    model, report = build_cracksam2(args.sam2_checkpoint, rank=args.rank, device=str(device))
    payload = torch.load(args.lora_checkpoint, map_location="cpu", weights_only=False)
    adapter = payload.get("adapter") if isinstance(payload, dict) else None
    if adapter is None:
        raise SystemExit(
            "le checkpoint ne contient pas de clé « adapter » : "
            f"clés vues = {sorted(payload)[:12] if isinstance(payload, dict) else type(payload)}"
        )
    load_adapter_state_dict(model, adapter, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    writer = CacheWriter(
        args.output,
        schema_version=BASELINE_CACHE_VERSION,
        kind="baseline_logits",
        parameters={
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_path": str(args.lora_checkpoint),
            "sam2_checkpoint_sha256": sha256_file(args.sam2_checkpoint),
            "rank": args.rank,
            "input_size": [args.input_size, args.input_size],
            "resample_to": "native",
            "mask_input": None,
            "model_config": {
                "lora_rank": report.rank,
                "lora_alpha": report.alpha,
                "trainable_parameters": report.trainable_parameters,
            },
        },
        extra={
            "dataset_manifest": str(args.manifest),
            "dataset_manifest_sha256": manifest_digest(samples),
        },
    )

    pending = [sample for sample in samples if not writer.has(sample.sample_id)]
    print(f"{len(samples) - len(pending)} déjà en cache, {len(pending)} à calculer")
    started = time.time()

    with torch.no_grad():
        for offset in range(0, len(pending), args.batch_size):
            chunk = pending[offset : offset + args.batch_size]
            images = load_batch([s.rgb_path for s in chunk], args.input_size).to(device)
            outputs = model(images)
            logits = outputs["logits"].float()
            for index, sample in enumerate(chunk):
                native = F.interpolate(
                    logits[index : index + 1],
                    size=(sample.height, sample.width),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                writer.write(
                    sample.sample_id,
                    {"baseline_logits": native.cpu().numpy().astype(np.float32)},
                    {
                        "source_rgb_sha256": sample.rgb_sha256,
                        "height": sample.height,
                        "width": sample.width,
                    },
                )
            done = offset + len(chunk)
            rate = done / max(1e-6, time.time() - started)
            print(f"  {done}/{len(pending)} — {rate:.1f} img/s", end="\r", flush=True)

    manifest_path = writer.finalize()
    print(f"\ncache écrit : {manifest_path}")
    print(f"{writer.manifest['count']} entrées, {len(writer.manifest['errors'])} erreur(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
