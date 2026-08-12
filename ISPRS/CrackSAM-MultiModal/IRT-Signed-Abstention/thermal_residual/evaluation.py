"""Évaluation par image d'un bras, et diagnostics du correcteur.

Chaque bras produit un CSV par image. Les comparaisons sont ensuite **appariées**
sur ces CSV (``thermal_residual.stats``), jamais sur des moyennes agrégées.

Le bras A0 n'a pas de correcteur : il s'évalue directement sur les logits cachés,
ce qui garantit que la baseline rapportée est exactement celle que les autres
bras corrigent — aucune ré-inférence, aucune dérive possible.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .cache import CacheHandle
from .constants import DECISION_THRESHOLD, TOLERANCES
from .data import EvidencePermutation, IRTResidualDataset, collate
from .manifest import IRTSample
from .metrics import action_statistics, aggregate, image_metrics
from .model import ThermalSignedAbstentionAdapter
from .provenance import atomic_write_json


def _dataset(
    samples: Sequence[IRTSample],
    baseline_cache: CacheHandle,
    thermal_cache: CacheHandle,
    *,
    evidence_source: str,
    permutation: EvidencePermutation | None,
    seed: int,
) -> IRTResidualDataset:
    dataset = IRTResidualDataset(
        samples,
        baseline_cache,
        thermal_cache,
        evidence_source=evidence_source,
        permutation=permutation,
        augmentation=None,
        seed=seed,
    )
    # À l'évaluation la permutation est **figée** : époque 0, graine du run.
    dataset.set_epoch(0)
    return dataset


def evaluate_arm(
    *,
    samples: Sequence[IRTSample],
    baseline_cache: CacheHandle,
    thermal_cache: CacheHandle,
    model: ThermalSignedAbstentionAdapter | None,
    evidence_source: str,
    permuted: bool,
    assignment: Mapping[str, str],
    seed: int,
    device: str | torch.device = "cpu",
    batch_size: int = 4,
    tolerances: Sequence[int] = TOLERANCES,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Métriques par image, diagnostics d'actions et appariement permuté."""

    from torch.utils.data import DataLoader

    resolved_device = torch.device(device)
    permutation = (
        EvidencePermutation(
            [s for s in samples], assignment, seed=seed
        )
        if permuted
        else None
    )
    dataset = _dataset(
        samples,
        baseline_cache,
        thermal_cache,
        evidence_source=evidence_source,
        permutation=permutation,
        seed=seed,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    if model is not None:
        model = model.to(resolved_device).eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            baseline_logits = batch["baseline_logits"].to(resolved_device)
            if model is None:
                logits = baseline_logits
                residual = torch.zeros_like(baseline_logits)
                hard_action = torch.full_like(baseline_logits, 2.0)
                active = torch.zeros_like(baseline_logits)
            else:
                outputs = model(
                    baseline_logits,
                    batch["thermal_evidence"].to(resolved_device),
                    batch["modality_present"].to(resolved_device),
                )
                logits = outputs["logits"]
                residual = outputs["residual_logits"]
                hard_action = outputs["hard_action"].to(logits.dtype)
                active = outputs["active_probability"]

            probabilities = torch.sigmoid(logits).cpu().numpy()
            baseline_probabilities = torch.sigmoid(baseline_logits).cpu().numpy()
            residual_numpy = residual.cpu().numpy()
            action_numpy = hard_action.cpu().numpy()
            active_numpy = active.cpu().numpy()
            targets = batch["mask"].numpy()

            for index, sample_id in enumerate(batch["sample_id"]):
                truth = targets[index, 0]
                row: dict[str, Any] = {"sample_id": sample_id}
                row.update(
                    image_metrics(
                        probabilities[index, 0],
                        truth,
                        threshold=threshold,
                        tolerances=tolerances,
                    )
                )
                row.update(
                    {
                        f"baseline_{key}": value
                        for key, value in image_metrics(
                            baseline_probabilities[index, 0],
                            truth,
                            threshold=threshold,
                            tolerances=tolerances,
                        ).items()
                    }
                )
                row.update(
                    action_statistics(
                        action_numpy[index], residual_numpy[index], active_numpy[index]
                    )
                )
                row["evidence_source_id"] = batch["evidence_source_id"][index]
                row["improves_baseline"] = float(row["iou"] > row["baseline_iou"])
                row.update(_error_deltas(probabilities[index, 0], baseline_probabilities[index, 0], truth, threshold))
                rows.append(row)

    return {
        "rows": rows,
        "summary": aggregate(rows),
        "permutation": dataset.permutation_mapping(),
        "count": len(rows),
    }


def _error_deltas(
    probability: np.ndarray,
    baseline_probability: np.ndarray,
    truth: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Ce que le correcteur fait des erreurs de la baseline — §9.2.

    Deux nombres qui séparent « il répare » de « il déplace le problème » :
    la fraction des faux négatifs de la baseline devenus vrais positifs, et la
    fraction de ses faux positifs devenus vrais négatifs.
    """

    reference = np.asarray(truth) > 0.5
    baseline = np.asarray(baseline_probability) > threshold
    candidate = np.asarray(probability) > threshold

    false_negatives = reference & ~baseline
    false_positives = ~reference & baseline
    recovered = float((false_negatives & candidate).sum()) / float(max(1, int(false_negatives.sum())))
    removed = float((false_positives & ~candidate).sum()) / float(max(1, int(false_positives.sum())))
    introduced = float((~reference & ~baseline & candidate).sum()) / float(
        max(1, int((~reference & ~baseline).sum()))
    )
    lost = float((reference & baseline & ~candidate).sum()) / float(
        max(1, int((reference & baseline).sum()))
    )
    return {
        "recovered_false_negatives": recovered,
        "removed_false_positives": removed,
        "introduced_false_positives": introduced,
        "lost_true_positives": lost,
    }


def write_per_image_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return destination
    columns = list(rows[0])
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in columns} for row in rows])
    temporary.replace(destination)
    return destination


def read_per_image_csv(path: str | Path) -> dict[str, dict[str, float]]:
    """Relit un CSV par image, en ne gardant que les colonnes numériques."""

    table: dict[str, dict[str, float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values: dict[str, float] = {}
            for key, value in row.items():
                if key in ("sample_id", "evidence_source_id"):
                    continue
                try:
                    values[key] = float(value)
                except (TypeError, ValueError):
                    continue
            table[row["sample_id"]] = values
    return table


def write_summary(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return atomic_write_json(path, dict(payload))


__all__ = [
    "evaluate_arm",
    "read_per_image_csv",
    "write_per_image_csv",
    "write_summary",
]
