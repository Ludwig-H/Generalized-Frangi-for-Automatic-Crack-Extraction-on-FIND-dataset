"""Métriques de segmentation et diagnostics du correcteur — §9.

Les définitions **ne sont pas nouvelles** : ce sont celles de
``ISPRS/CrackSAM-GeoLoRA/scripts/05_tolerant_iou.py`` (IoU tolérante) et
``03_evaluate.py`` (composantes, couverture de squelette). Ces deux fichiers sont
des scripts numérotés, pas des modules importables ; les définitions sont donc
reprises ici et leur **équivalence est prouvée par test**
(``tests/test_metrics.py``), qui charge le script d'origine par chemin et compare
les deux implémentations sur des masques tirés au hasard.

Convention retenue pour conclure : ``iou_buffered``. Elle ne dilate jamais les
deux masques à la fois, donc elle mesure une distance d'appariement plutôt qu'un
recouvrement de versions épaissies, qui sature vers 1. La tolérance primaire de
cette campagne est ``k = 3``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi

from .constants import ACTION_NAMES, DECISION_THRESHOLD, TOLERANCES

_CONNECTIVITY = np.ones((3, 3), dtype=int)


# --------------------------------------------------------------------------- #
# Tolérance
# --------------------------------------------------------------------------- #


def dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    """Dilatation **euclidienne** : tout pixel à distance ``≤ radius`` du support."""

    binary = np.asarray(mask, dtype=bool)
    if radius <= 0 or not binary.any():
        return binary
    return ndi.distance_transform_edt(~binary) <= float(radius)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    if not left.any() and not right.any():
        return 1.0
    if not left.any() or not right.any():
        return 0.0
    return float(np.count_nonzero(left & right)) / float(np.count_nonzero(left | right))


def tolerant_scores(
    prediction: np.ndarray, truth: np.ndarray, radius: float
) -> dict[str, float]:
    """Précision, rappel, F1 et IoU tolérants, plus l'IoU « dilate_both »."""

    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    both = _iou(dilate(prediction, radius), dilate(truth, radius))

    if not prediction.any() and not truth.any():
        return {
            "iou_dilate_both": both,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou_buffered": 1.0,
        }
    if not prediction.any() or not truth.any():
        return {
            "iou_dilate_both": both,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou_buffered": 0.0,
        }

    precision = float(np.count_nonzero(prediction & dilate(truth, radius))) / float(
        np.count_nonzero(prediction)
    )
    recall = float(np.count_nonzero(truth & dilate(prediction, radius))) / float(
        np.count_nonzero(truth)
    )
    denominator = precision + recall
    f1 = 0.0 if denominator <= 0 else 2.0 * precision * recall / denominator
    buffered = 0.0 if f1 <= 0 else f1 / (2.0 - f1)
    return {
        "iou_dilate_both": both,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_buffered": buffered,
    }


# --------------------------------------------------------------------------- #
# Topologie
# --------------------------------------------------------------------------- #


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    return np.asarray(skeletonize(np.asarray(mask, dtype=bool)), dtype=bool)


def topology(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Composantes connexes et couverture du squelette de la vérité terrain."""

    predicted = np.asarray(prediction, dtype=bool)
    reference = np.asarray(truth, dtype=bool)
    _, predicted_components = ndi.label(predicted, structure=_CONNECTIVITY)
    _, reference_components = ndi.label(reference, structure=_CONNECTIVITY)
    if reference.any():
        reference_skeleton = _skeletonize(reference)
        covered = float((reference_skeleton & predicted).sum()) / float(
            max(int(reference_skeleton.sum()), 1)
        )
    else:
        covered = 1.0 if not predicted.any() else 0.0
    return {
        "components_pred": float(predicted_components),
        "components_true": float(reference_components),
        "skeleton_covered": covered,
    }


def cldice(prediction: np.ndarray, truth: np.ndarray) -> float:
    """clDice binaire : moyenne harmonique des deux couvertures de squelette."""

    predicted = np.asarray(prediction, dtype=bool)
    reference = np.asarray(truth, dtype=bool)
    if not predicted.any() and not reference.any():
        return 1.0
    if not predicted.any() or not reference.any():
        return 0.0
    predicted_skeleton = _skeletonize(predicted)
    reference_skeleton = _skeletonize(reference)
    precision = float((predicted_skeleton & reference).sum()) / float(
        max(int(predicted_skeleton.sum()), 1)
    )
    sensitivity = float((reference_skeleton & predicted).sum()) / float(
        max(int(reference_skeleton.sum()), 1)
    )
    denominator = precision + sensitivity
    return 0.0 if denominator <= 0 else 2.0 * precision * sensitivity / denominator


# --------------------------------------------------------------------------- #
# Agrégation par image
# --------------------------------------------------------------------------- #


def image_metrics(
    probability: np.ndarray,
    truth: np.ndarray,
    *,
    threshold: float = DECISION_THRESHOLD,
    tolerances: Sequence[int] = TOLERANCES,
) -> dict[str, float]:
    """Toutes les métriques de segmentation d'une image."""

    prediction = np.asarray(probability, dtype=np.float32) > threshold
    reference = np.asarray(truth, dtype=np.float32) > 0.5

    strict = tolerant_scores(prediction, reference, 0)
    values: dict[str, float] = {
        "iou": strict["iou_buffered"],
        "dice": strict["f1"],
        "precision": strict["precision"],
        "recall": strict["recall"],
        "cldice": cldice(prediction, reference),
        "crack_fraction_true": float(reference.mean()),
        "crack_fraction_pred": float(prediction.mean()),
    }
    values.update(topology(prediction, reference))
    for radius in tolerances:
        scores = tolerant_scores(prediction, reference, radius)
        for name, score in scores.items():
            values[f"{name}_tol{radius}"] = score
    return values


def action_statistics(
    hard_action: np.ndarray, residual: np.ndarray, active: np.ndarray | None = None
) -> dict[str, float]:
    """Fractions d'actions, amplitude et quantiles de ``|Δz|`` — §9.2."""

    actions = np.asarray(hard_action).reshape(-1)
    magnitude = np.abs(np.asarray(residual, dtype=np.float64)).reshape(-1)
    total = max(1, actions.size)
    statistics = {
        f"fraction_{name}": float(np.count_nonzero(actions == index)) / total
        for index, name in enumerate(ACTION_NAMES)
    }
    quantiles = np.percentile(magnitude, [50.0, 90.0, 99.0]) if magnitude.size else np.zeros(3)
    statistics.update(
        {
            "residual_abs_mean": float(magnitude.mean()) if magnitude.size else 0.0,
            "residual_abs_max": float(magnitude.max()) if magnitude.size else 0.0,
            "residual_abs_p50": float(quantiles[0]),
            "residual_abs_p90": float(quantiles[1]),
            "residual_abs_p99": float(quantiles[2]),
            "residual_nonzero_fraction": float(np.count_nonzero(magnitude > 0.0)) / total,
        }
    )
    if active is not None:
        statistics["active_probability_mean"] = float(np.asarray(active).mean())
    return statistics


def aggregate(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Moyenne macro (par image) de toutes les colonnes numériques."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        return {}
    keys = [
        key
        for key in materialized[0]
        if all(isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool) for row in materialized)
    ]
    return {key: float(np.mean([float(row[key]) for row in materialized])) for key in keys}


__all__ = [
    "action_statistics",
    "aggregate",
    "cldice",
    "dilate",
    "image_metrics",
    "tolerant_scores",
    "topology",
]
