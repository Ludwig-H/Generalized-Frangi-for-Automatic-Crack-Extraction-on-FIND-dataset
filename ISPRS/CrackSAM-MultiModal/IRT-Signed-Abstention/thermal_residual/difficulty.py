"""Poids de difficulté par image, calculés sur le **train** seul.

Diagnostic qui les motive : le gain de l'évidence thermique se concentre sur le
tiers du test où la baseline échoue (`+0,0117` d'IoU à 3 px, IC95 excluant zéro)
et s'inverse sur le tiers moyen (`−0,0038`, IC95 excluant zéro aussi). Une perte
qui moyenne uniformément sur les images est donc dominée par des images où il n'y
a rien à gagner et tout à perdre.

Le poids d'une image est sa **marge de progression** : ``1 − IoU_baseline``, à
tolérance primaire, plafonnée pour qu'aucune image ne pèse plus que ``max_ratio``
fois la moyenne. Il n'utilise que des logits déjà cachés et des masques
d'entraînement — rien du test, rien que l'entraînement ne voie déjà.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .cache import CacheHandle
from .constants import DECISION_THRESHOLD, PRIMARY_TOLERANCE
from .data import load_mask
from .manifest import IRTSample
from .metrics import tolerant_scores


def baseline_headroom(
    samples: Sequence[IRTSample],
    baseline_cache: CacheHandle,
    *,
    tolerance: int = PRIMARY_TOLERANCE,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """``1 − IoU_baseline`` par image : ce qu'il reste à gagner."""

    headroom: dict[str, float] = {}
    for sample in samples:
        logits = np.asarray(baseline_cache.entry(sample.sample_id)["baseline_logits"])
        prediction = logits.reshape(logits.shape[-2:]) > float(np.log(threshold / (1.0 - threshold)))
        truth = load_mask(sample.mask_path) > 0.5
        headroom[sample.sample_id] = 1.0 - tolerant_scores(prediction, truth, tolerance)["iou_buffered"]
    return headroom


def difficulty_weights(
    headroom: Mapping[str, float], *, floor: float = 0.1, max_ratio: float = 5.0
) -> dict[str, float]:
    """Normalise les marges en poids de moyenne 1, bornés par ``max_ratio``.

    ``floor`` empêche qu'une image parfaitement segmentée disparaisse
    complètement de l'entraînement : elle continue de contraindre le correcteur à
    ne pas la casser.
    """

    if not headroom:
        return {}
    raw = {name: max(float(value), float(floor)) for name, value in headroom.items()}
    mean = float(np.mean(list(raw.values()))) or 1.0
    weights = {name: value / mean for name, value in raw.items()}
    ceiling = float(max_ratio)
    return {name: min(value, ceiling) for name, value in weights.items()}


__all__ = ["baseline_headroom", "difficulty_weights"]
