"""Boucle d'entraînement du correcteur, reprenable et instrumentée.

CrackSAM n'est pas chargé : l'entraînement lit des logits cachés, donc une époque
coûte quelques secondes et la matrice complète — sept bras, trois graines — tient
dans une session Spot.

Choix pré-enregistrés, identiques pour tous les bras :

* taux d'apprentissage **constant** — le poly-LR ``power=6`` de GeoLoRA s'était
  effondré à ``3·10⁻²⁷`` et donnait ``best epoch = 0`` ;
* sélection du checkpoint sur la **validation non augmentée** ;
* aucune donnée de test n'est ouverte par ce module.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ._repo import git_commit
from .cache import CacheHandle
from .constants import (
    CHECKPOINT_FORMAT_VERSION,
    DECISION_THRESHOLD,
    EVIDENCE_SOURCE_FRANGI,
    PRIMARY_TOLERANCE,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
)
from .data import EvidencePermutation, FlipAugmentation, IRTResidualDataset, collate
from .losses import LossWeights, corrector_loss
from .manifest import IRTSample
from .metrics import tolerant_scores
from .model import ThermalSignedAbstentionAdapter, build_adapter
from .provenance import atomic_write_json, sha256_json

#: Métriques admissibles pour la sélection de checkpoint.
SELECTION_METRICS = ("iou", f"iou_buffered_tol{PRIMARY_TOLERANCE}")


@dataclass
class TrainingConfig:
    """Hyperparamètres. Aucun n'est réglé par bras."""

    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 8
    max_epochs: int = 100
    early_stopping_patience: int = 15
    gradient_clip_norm: float = 1.0
    amp: bool = True
    num_workers: int = 0
    selection_metric: str = f"iou_buffered_tol{PRIMARY_TOLERANCE}"
    seeds: tuple[int, ...] = (13, 37, 73)

    def __post_init__(self) -> None:
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(
                f"selection_metric doit être dans {SELECTION_METRICS}, "
                f"reçu {self.selection_metric!r}"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "TrainingConfig":
        payload = dict(payload or {})
        return cls(
            learning_rate=float(payload.get("learning_rate", 3.0e-4)),
            weight_decay=float(payload.get("weight_decay", 1.0e-4)),
            batch_size=int(payload.get("batch_size", 8)),
            max_epochs=int(payload.get("max_epochs", 100)),
            early_stopping_patience=int(payload.get("early_stopping_patience", 15)),
            gradient_clip_norm=float(payload.get("gradient_clip_norm", 1.0)),
            amp=bool(payload.get("amp", True)),
            num_workers=int(payload.get("num_workers", 0)),
            selection_metric=str(
                payload.get("selection_metric", f"iou_buffered_tol{PRIMARY_TOLERANCE}")
            ),
            seeds=tuple(int(value) for value in payload.get("seeds", (13, 37, 73))),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "gradient_clip_norm": self.gradient_clip_norm,
            "amp": self.amp,
            "num_workers": self.num_workers,
            "selection_metric": self.selection_metric,
            "seeds": list(self.seeds),
        }


@dataclass
class ArmSpecification:
    """Ce qui distingue un bras : sa source d'évidence, sa tête, sa portée."""

    name: str
    evidence_source: str = EVIDENCE_SOURCE_FRANGI
    permuted: bool = False
    trained: bool = True
    model: dict[str, Any] = field(default_factory=dict)

    @property
    def has_abstention(self) -> bool:
        return str(self.model.get("head", "signed_abstention")) == "signed_abstention"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _validation_scores(
    model: ThermalSignedAbstentionAdapter,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """IoU stricte et IoU tolérante à 3 px sur la validation, sans augmentation."""

    model.eval()
    strict: list[float] = []
    buffered: list[float] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["baseline_logits"].to(device),
                batch["thermal_evidence"].to(device),
                batch["modality_present"].to(device),
            )
            probabilities = torch.sigmoid(outputs["logits"]).cpu().numpy()
            targets = batch["mask"].numpy()
            for index in range(probabilities.shape[0]):
                prediction = probabilities[index, 0] > DECISION_THRESHOLD
                truth = targets[index, 0] > 0.5
                strict.append(tolerant_scores(prediction, truth, 0)["iou_buffered"])
                buffered.append(
                    tolerant_scores(prediction, truth, PRIMARY_TOLERANCE)["iou_buffered"]
                )
    return {
        "iou": float(np.mean(strict)) if strict else 0.0,
        f"iou_buffered_tol{PRIMARY_TOLERANCE}": float(np.mean(buffered)) if buffered else 0.0,
    }


def train_arm(
    *,
    arm: ArmSpecification,
    samples: Sequence[IRTSample],
    assignment: Mapping[str, str],
    baseline_cache: CacheHandle,
    thermal_cache: CacheHandle,
    training: TrainingConfig,
    weights: LossWeights,
    output_dir: str | Path,
    seed: int,
    device: str | torch.device = "cuda",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Entraîne un bras et écrit ``best.pt``, ``latest.pt`` et ``training.json``."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    set_seed(seed)

    train_samples = [s for s in samples if assignment[s.sample_id] == SPLIT_TRAIN]
    validation_samples = [s for s in samples if assignment[s.sample_id] == SPLIT_VALIDATION]
    if not train_samples or not validation_samples:
        raise ValueError("les splits train et validation ne peuvent pas être vides")

    permutation = (
        EvidencePermutation(samples, assignment, seed=seed) if arm.permuted else None
    )
    train_dataset = IRTResidualDataset(
        train_samples,
        baseline_cache,
        thermal_cache,
        evidence_source=arm.evidence_source,
        permutation=permutation,
        augmentation=FlipAugmentation(),
        seed=seed,
    )
    validation_dataset = IRTResidualDataset(
        validation_samples,
        baseline_cache,
        thermal_cache,
        evidence_source=arm.evidence_source,
        permutation=permutation,
        augmentation=None,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=training.num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        collate_fn=collate,
    )

    model = build_adapter(arm.model).to(resolved_device)
    parameter_count = model.trainable_parameters()
    if parameter_count >= 100_000:
        raise ValueError(
            f"le correcteur doit rester sous 100 000 paramètres, il en a {parameter_count}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    use_amp = bool(training.amp and resolved_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    patience = 0
    started = time.time()

    for epoch in range(training.max_epochs):
        train_dataset.set_epoch(epoch)
        validation_dataset.set_epoch(epoch)
        model.train()
        journals: list[dict[str, float]] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    batch["baseline_logits"].to(resolved_device),
                    batch["thermal_evidence"].to(resolved_device),
                    batch["modality_present"].to(resolved_device),
                )
                loss, journal = corrector_loss(
                    outputs,
                    batch["mask"].to(resolved_device),
                    weights,
                    has_abstention=arm.has_abstention,
                )
            scaler.scale(loss).backward()
            if training.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            journals.append(journal)

        scores = _validation_scores(model, validation_loader, resolved_device)
        record = {
            "epoch": epoch,
            "train": {
                key: float(np.mean([entry[key] for entry in journals])) for key in journals[0]
            },
            "validation": scores,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)

        current = scores[training.selection_metric]
        improved = current > best_score
        if improved:
            best_score = current
            best_epoch = epoch
            best_metrics = dict(scores)
            patience = 0
        else:
            patience += 1

        payload = _checkpoint_payload(
            model=model,
            arm=arm,
            training=training,
            weights=weights,
            seed=seed,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            provenance=provenance,
            epoch=epoch,
            optimizer=optimizer,
        )
        torch.save(payload, output / "latest.pt")
        if improved:
            torch.save(payload, output / "best.pt")
        atomic_write_json(
            output / "training.json",
            {
                "arm": arm.name,
                "seed": seed,
                "device": str(resolved_device),
                "trainable_parameters": parameter_count,
                "model_config": model.config(),
                "training_config": training.to_json(),
                "loss_weights": weights.to_json(),
                "evidence_source": arm.evidence_source,
                "permuted": arm.permuted,
                "best_epoch": best_epoch,
                "best_validation": best_metrics,
                "history": history,
                "git_commit": git_commit(),
                "provenance": dict(provenance or {}),
            },
        )
        if patience >= training.early_stopping_patience:
            break

    return {
        "arm": arm.name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "epochs_run": len(history),
        "output_dir": str(output),
    }


def _checkpoint_payload(
    *,
    model: ThermalSignedAbstentionAdapter,
    arm: ArmSpecification,
    training: TrainingConfig,
    weights: LossWeights,
    seed: int,
    best_epoch: int,
    best_metrics: Mapping[str, float],
    provenance: Mapping[str, Any] | None,
    epoch: int,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Checkpoint du **correcteur seul** : SAM n'y est jamais sérialisé."""

    record = dict(provenance or {})
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config(),
        "training_config": training.to_json(),
        "loss_weights": weights.to_json(),
        "arm": {
            "name": arm.name,
            "evidence_source": arm.evidence_source,
            "permuted": arm.permuted,
            "model": dict(arm.model),
        },
        "baseline_checkpoint_sha256": record.get("baseline_checkpoint_sha256", "unknown"),
        "baseline_cache_manifest_sha256": record.get("baseline_cache_manifest_sha256", "unknown"),
        "thermal_cache_manifest_sha256": record.get("thermal_cache_manifest_sha256", "unknown"),
        "dataset_manifest_sha256": record.get("dataset_manifest_sha256", "unknown"),
        "split_sha256": record.get("split_sha256", "unknown"),
        "git_commit": git_commit(),
        "seed": int(seed),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_validation_metrics": dict(best_metrics),
        "config_digest": sha256_json(
            {
                "model": model.config(),
                "training": training.to_json(),
                "loss": weights.to_json(),
                "arm": arm.name,
                "seed": seed,
            }
        ),
    }


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"version de checkpoint inattendue : {payload.get('format_version')}"
        )
    return payload


def write_run_summary(path: str | Path, summary: Mapping[str, Any]) -> Path:
    return atomic_write_json(path, json.loads(json.dumps(summary, default=str)))


__all__ = [
    "ArmSpecification",
    "SELECTION_METRICS",
    "TrainingConfig",
    "load_checkpoint",
    "set_seed",
    "train_arm",
    "write_run_summary",
]
