"""Plafond d'une correction bornée : ce que ``delta_max`` rend inatteignable.

La correction vaut ``Δz = δ_max(π⁺ − π⁻)``, donc ``|Δz| ≤ δ_max``. La prédiction
étant ``z₀ + Δz > 0``, **un pixel n'est corrigeable que si ``|z₀| < δ_max``**.
Avec la valeur recommandée ``δ_max = 4``, cela restreint la correction aux pixels
où ``p₀ ∈ (0,018 ; 0,982)`` : tout faux négatif *confiant* de la baseline est hors
d'atteinte, quelle que soit la qualité de l'évidence thermique.

Ce module mesure le plafond avant de dépenser du GPU. L'oracle borné applique
``+δ_max`` là où la vérité dit fissure et ``−δ_max`` ailleurs : c'est la
meilleure correction qu'un correcteur borné puisse produire, tous encodeurs
confondus. Si son IoU dépasse à peine celle de la baseline, la campagne ne peut
pas conclure — elle ne saurait pas distinguer « la thermique n'aide pas » de
« ``δ_max`` est trop petit ».

C'est le pendant, pour la borne d'amplitude, de l'oracle de source de
CrackSAM-GFA : une porte chiffrée qu'on franchit ou non avant d'entraîner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .constants import DECISION_THRESHOLD, PRIMARY_TOLERANCE, TOLERANCES
from .metrics import tolerant_scores


@dataclass(frozen=True)
class CeilingReport:
    """Plafond mesuré pour une valeur de ``delta_max``."""

    delta_max: float
    count: int
    baseline_iou: float
    oracle_iou: float
    baseline_iou_tolerant: float
    oracle_iou_tolerant: float
    unreachable_error_fraction: float
    unreachable_false_negative_fraction: float
    unreachable_false_positive_fraction: float
    logit_abs_quantiles: dict[str, float]

    @property
    def headroom(self) -> float:
        """Marge que la méthode peut espérer, en IoU tolérante primaire."""

        return self.oracle_iou_tolerant - self.baseline_iou_tolerant

    def to_json(self) -> dict[str, Any]:
        return {
            "delta_max": self.delta_max,
            "count": self.count,
            "baseline_iou": self.baseline_iou,
            "oracle_iou": self.oracle_iou,
            f"baseline_iou_buffered_tol{PRIMARY_TOLERANCE}": self.baseline_iou_tolerant,
            f"oracle_iou_buffered_tol{PRIMARY_TOLERANCE}": self.oracle_iou_tolerant,
            "headroom": self.headroom,
            "unreachable_error_fraction": self.unreachable_error_fraction,
            "unreachable_false_negative_fraction": self.unreachable_false_negative_fraction,
            "unreachable_false_positive_fraction": self.unreachable_false_positive_fraction,
            "logit_abs_quantiles": dict(self.logit_abs_quantiles),
        }


def bounded_oracle_logits(
    baseline_logits: np.ndarray, mask: np.ndarray, delta_max: float
) -> np.ndarray:
    """La meilleure correction bornée possible : ``±δ_max`` selon la vérité."""

    target = np.asarray(mask, dtype=np.float32) > 0.5
    return np.asarray(baseline_logits, dtype=np.float32) + np.where(
        target, float(delta_max), -float(delta_max)
    ).astype(np.float32)


def measure_ceiling(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    delta_max: float,
    *,
    threshold: float = DECISION_THRESHOLD,
    tolerance: int = PRIMARY_TOLERANCE,
) -> CeilingReport:
    """Plafond sur un ensemble de couples ``(logits baseline, masque)``."""

    baseline_strict: list[float] = []
    oracle_strict: list[float] = []
    baseline_tolerant: list[float] = []
    oracle_tolerant: list[float] = []
    unreachable = 0
    unreachable_fn = 0
    unreachable_fp = 0
    errors = 0
    false_negatives = 0
    false_positives = 0
    magnitudes: list[np.ndarray] = []
    count = 0

    decision = float(np.log(threshold / (1.0 - threshold)))

    for logits, mask in pairs:
        count += 1
        z0 = np.asarray(logits, dtype=np.float32).reshape(logits.shape[-2:])
        truth = np.asarray(mask, dtype=np.float32).reshape(z0.shape) > 0.5
        prediction = z0 > decision

        oracle = bounded_oracle_logits(z0, truth, delta_max) > decision
        baseline_strict.append(tolerant_scores(prediction, truth, 0)["iou_buffered"])
        oracle_strict.append(tolerant_scores(oracle, truth, 0)["iou_buffered"])
        baseline_tolerant.append(tolerant_scores(prediction, truth, tolerance)["iou_buffered"])
        oracle_tolerant.append(tolerant_scores(oracle, truth, tolerance)["iou_buffered"])

        wrong = prediction != truth
        # Un pixel faux reste faux si la correction bornée ne franchit pas le seuil.
        out_of_reach = wrong & (np.abs(z0 - decision) >= float(delta_max))
        errors += int(wrong.sum())
        unreachable += int(out_of_reach.sum())
        missed = truth & ~prediction
        spurious = ~truth & prediction
        false_negatives += int(missed.sum())
        false_positives += int(spurious.sum())
        unreachable_fn += int((missed & out_of_reach).sum())
        unreachable_fp += int((spurious & out_of_reach).sum())
        magnitudes.append(np.abs(z0).ravel()[::37])  # sous-échantillonnage déterministe

    stacked = np.concatenate(magnitudes) if magnitudes else np.zeros(1, dtype=np.float32)
    quantiles = np.percentile(stacked, [50.0, 90.0, 95.0, 99.0, 99.9])
    return CeilingReport(
        delta_max=float(delta_max),
        count=count,
        baseline_iou=float(np.mean(baseline_strict)) if baseline_strict else 0.0,
        oracle_iou=float(np.mean(oracle_strict)) if oracle_strict else 0.0,
        baseline_iou_tolerant=float(np.mean(baseline_tolerant)) if baseline_tolerant else 0.0,
        oracle_iou_tolerant=float(np.mean(oracle_tolerant)) if oracle_tolerant else 0.0,
        unreachable_error_fraction=float(unreachable) / float(max(1, errors)),
        unreachable_false_negative_fraction=float(unreachable_fn) / float(max(1, false_negatives)),
        unreachable_false_positive_fraction=float(unreachable_fp) / float(max(1, false_positives)),
        logit_abs_quantiles={
            "p50": float(quantiles[0]),
            "p90": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "p99": float(quantiles[3]),
            "p999": float(quantiles[4]),
        },
    )


def sweep(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    delta_values: Sequence[float],
    **kwargs: Any,
) -> list[CeilingReport]:
    """Plafond pour plusieurs ``delta_max``, afin de choisir la borne sur mesure."""

    return [measure_ceiling(pairs, value, **kwargs) for value in delta_values]


def recommend_delta_max(report: CeilingReport, *, coverage: float = 0.99) -> float:
    """Borne suggérée : le quantile de ``|z₀|`` couvrant ``coverage`` des pixels.

    Ce n'est qu'une suggestion chiffrée. La décision reste explicite et doit être
    écrite dans la configuration avant le premier entraînement, pas ajustée après
    avoir vu les résultats.
    """

    key = {0.5: "p50", 0.9: "p90", 0.95: "p95", 0.99: "p99", 0.999: "p999"}.get(coverage)
    if key is None:
        raise ValueError("coverage doit valoir 0.5, 0.9, 0.95, 0.99 ou 0.999")
    return float(report.logit_abs_quantiles[key])


__all__ = [
    "CeilingReport",
    "TOLERANCES",
    "bounded_oracle_logits",
    "measure_ceiling",
    "recommend_delta_max",
    "sweep",
]
