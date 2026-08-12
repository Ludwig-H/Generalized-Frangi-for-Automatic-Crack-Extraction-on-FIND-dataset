"""Statistique appariée : bootstrap sur les deltas par image — §9.3.

Toutes les comparaisons de cette étude sont **appariées** : chaque image est vue
par tous les bras, donc la variance inter-images — de très loin la plus grande —
est éliminée par la différence. Le bootstrap est fait sur les deltas, avec
rééchantillonnage des images, 10 000 réplications, et un intervalle par
percentiles.

La variance entre graines est rapportée **séparément** : elle mesure autre chose
(l'instabilité de l'optimisation) et l'agréger au bootstrap donnerait un
intervalle qui ne répond à aucune question précise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .constants import BOOTSTRAP_REPLICATES


@dataclass(frozen=True)
class PairedDelta:
    """Résultat d'une comparaison appariée."""

    comparison: str
    metric: str
    count: int
    mean: float
    median: float
    ci_low: float
    ci_high: float
    wins: int
    losses: int
    ties: int
    replicates: int

    @property
    def excludes_zero(self) -> bool:
        return (self.ci_low > 0.0) or (self.ci_high < 0.0)

    def verdict(self) -> str:
        if not self.excludes_zero:
            return "indiscernable"
        return "favorable" if self.mean > 0 else "défavorable"

    def to_json(self) -> dict[str, object]:
        return {
            "comparison": self.comparison,
            "metric": self.metric,
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "ci95_low": self.ci_low,
            "ci95_high": self.ci_high,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "replicates": self.replicates,
            "excludes_zero": self.excludes_zero,
            "verdict": self.verdict(),
        }


def paired_bootstrap(
    deltas: Sequence[float],
    *,
    comparison: str = "",
    metric: str = "",
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 20260812,
    confidence: float = 0.95,
) -> PairedDelta:
    """Intervalle par percentiles du delta moyen, rééchantillonné par image."""

    values = np.asarray(list(deltas), dtype=np.float64)
    if values.size == 0:
        raise ValueError("aucun delta à rééchantillonner")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(replicates), values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.percentile(means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return PairedDelta(
        comparison=comparison,
        metric=metric,
        count=int(values.size),
        mean=float(values.mean()),
        median=float(np.median(values)),
        ci_low=float(low),
        ci_high=float(high),
        wins=int(np.count_nonzero(values > 0)),
        losses=int(np.count_nonzero(values < 0)),
        ties=int(np.count_nonzero(values == 0)),
        replicates=int(replicates),
    )


def compare_arms(
    left: Mapping[str, Mapping[str, float]],
    right: Mapping[str, Mapping[str, float]],
    metric: str,
    *,
    comparison: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 20260812,
) -> PairedDelta:
    """Delta apparié ``left − right`` sur les images communes aux deux bras."""

    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError(f"aucune image commune entre les deux bras de « {comparison} »")
    deltas = [float(left[name][metric]) - float(right[name][metric]) for name in shared]
    return paired_bootstrap(
        deltas,
        comparison=comparison,
        metric=metric,
        replicates=replicates,
        seed=seed,
    )


def seed_variance(values: Sequence[float]) -> dict[str, float]:
    """Moyenne, écart-type et étendue d'une métrique à travers les graines."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        # Écart-type d'échantillon : une seule graine ne mesure aucune variance.
        "std": float(array.std(ddof=1)) if array.size > 1 else float("nan"),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def detection_floor(deltas: Sequence[float], *, confidence: float = 0.95) -> float:
    """Demi-largeur de l'IC normal du delta moyen : le plancher de détection.

    Utile pour dire, **avant** de conclure, si l'effet recherché est mesurable au
    ``N`` disponible. Sur 90 images de test, il vaut environ ``1,96·σ/√90``.
    """

    values = np.asarray(list(deltas), dtype=np.float64)
    if values.size < 2:
        return float("nan")
    from math import sqrt

    quantile = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else float(
        __import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.5 + confidence / 2.0)
    )
    return float(quantile * values.std(ddof=1) / sqrt(values.size))


__all__ = [
    "PairedDelta",
    "compare_arms",
    "detection_floor",
    "paired_bootstrap",
    "seed_variance",
]
