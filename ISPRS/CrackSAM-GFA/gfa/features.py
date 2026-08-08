"""Statistiques par fragment, données à l'arbitre sans aucune prémultiplication.

Deux étages volontairement séparés :

``evidence_features``
    Ne dépend que de l'image et des filtres — calculable sur CPU, en parallèle,
    sans GPU ni SAM 2.
``sam_features``
    Ne dépend que de ``z0`` et des features haute résolution de SAM 2 —
    calculable sur GPU, indépendamment du premier étage.

Le vecteur final est leur concaténation, dans un ordre gelé.

Le point important : chaque canal entre séparément, avec en plus son
**asymétrie de flancs**. Une vallée sombre bilatérale et une marche d'ombre
unilatérale ont la même réponse sur l'axe ; elles ne diffèrent qu'en comparant
les deux flancs au même rayon. C'est la leçon de Steger reprise au §3.2 du
rapport.
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np

from .evidence import CHANNELS, SUPPORT_SOURCES, Evidence
from .fragments import Fragment

#: Distance des flancs, en multiples du rayon de corridor.
FLANK_FACTOR: Final[float] = 1.5

_STATS_PER_CHANNEL: Final[int] = 6  # axe: moy/med/p90 ; flancs: g/d ; asymétrie
EVIDENCE_DIM: Final[int] = len(CHANNELS) * _STATS_PER_CHANNEL + 10
SAM_DIM: Final[int] = 11
FEATURE_DIM: Final[int] = EVIDENCE_DIM + SAM_DIM


def _sample(plane: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    height, width = plane.shape
    rows = np.clip(np.round(rows).astype(int), 0, height - 1)
    cols = np.clip(np.round(cols).astype(int), 0, width - 1)
    return plane[rows, cols]


def _flank_coordinates(
    fragment: Fragment, shape: tuple[int, int]
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Coordonnées des deux flancs, à ``FLANK_FACTOR * radius`` de l'axe."""

    pixels = fragment.pixels.astype(np.float64)
    if pixels.shape[0] >= 3:
        # Tangente locale lissée par différences centrées.
        tangent = np.gradient(pixels, axis=0)
    else:
        delta = pixels[-1] - pixels[0]
        tangent = np.repeat(delta[None, :], pixels.shape[0], axis=0)
    norm = np.hypot(tangent[:, 0], tangent[:, 1])
    norm[norm < 1e-8] = 1.0
    tangent = tangent / norm[:, None]
    # Normale = tangente tournée de 90 degrés.
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    offset = FLANK_FACTOR * float(fragment.radius)
    left = (pixels[:, 0] + offset * normal[:, 0], pixels[:, 1] + offset * normal[:, 1])
    right = (pixels[:, 0] - offset * normal[:, 0], pixels[:, 1] - offset * normal[:, 1])
    return left, right


def evidence_features(evidence: Evidence, fragment: Fragment) -> np.ndarray:
    """Vecteur ``(EVIDENCE_DIM,)`` calculé sans GT et sans SAM 2."""

    shape = evidence.shape
    rows = fragment.pixels[:, 0]
    cols = fragment.pixels[:, 1]
    (left_rows, left_cols), (right_rows, right_cols) = _flank_coordinates(
        fragment, shape
    )

    values: list[float] = []
    for index in range(len(CHANNELS)):
        plane = evidence.planes[index]
        axis = plane[rows, cols]
        left = _sample(plane, left_rows, left_cols)
        right = _sample(plane, right_rows, right_cols)
        left_mean = float(np.mean(left))
        right_mean = float(np.mean(right))
        # Asymétrie normalisée : 0 pour une vallée symétrique, 1 pour une marche
        # parfaitement unilatérale. C'est le discriminant ligne/marche.
        denominator = abs(left_mean) + abs(right_mean) + 1e-6
        asymmetry = abs(left_mean - right_mean) / denominator
        values.extend(
            [
                float(np.mean(axis)),
                float(np.median(axis)),
                float(np.quantile(axis, 0.9)),
                left_mean,
                right_mean,
                asymmetry,
            ]
        )

    # Géométrie du fragment.
    tangent = fragment.tangent()
    angle = float(np.arctan2(tangent[0], tangent[1]))
    theta_axis = np.arctan2(
        evidence.plane("sin2theta")[rows, cols],
        evidence.plane("cos2theta")[rows, cols],
    )
    # Cohérence circulaire de l'orientation double-angle le long de l'axe : une
    # frontière d'ombre est très cohérente, une fissure texturée l'est moins.
    coherence = float(
        np.hypot(np.mean(np.cos(theta_axis)), np.mean(np.sin(theta_axis)))
    )
    source_flags = [
        1.0 if name in fragment.sources else 0.0 for name in SUPPORT_SOURCES
    ]
    values.extend(
        [
            float(fragment.length) / 32.0,
            float(fragment.radius) / 6.0,
            float(fragment.straightness()),
            float(np.cos(2.0 * angle)),
            float(np.sin(2.0 * angle)),
            coherence,
            *source_flags,
        ]
    )
    return np.asarray(values, dtype=np.float32)


def sam_features(
    z0: np.ndarray,
    fragment: Fragment,
    high_res_mean: Sequence[float] | None = None,
) -> np.ndarray:
    """Vecteur ``(SAM_DIM,)`` décrivant ce que la baseline pense déjà de la bande.

    C'est l'information qui permet à l'arbitre de raisonner en **utilité
    marginale** : une bande où ``z0`` est déjà franchement positive n'a pas
    besoin d'être ajoutée, et une bande où ``z0`` est franchement négative ne
    peut pas être retirée utilement.
    """

    shape = z0.shape
    rows = fragment.pixels[:, 0]
    cols = fragment.pixels[:, 1]
    corridor = fragment.corridor(shape)
    axis_values = z0[rows, cols]
    corridor_values = z0[corridor]
    (left_rows, left_cols), (right_rows, right_cols) = _flank_coordinates(
        fragment, shape
    )
    left = _sample(z0, left_rows, left_cols)
    right = _sample(z0, right_rows, right_cols)

    positive_fraction = float(np.mean(corridor_values > 0.0))
    values = [
        float(np.mean(axis_values)) / 8.0,
        float(np.median(axis_values)) / 8.0,
        float(np.quantile(axis_values, 0.1)) / 8.0,
        float(np.quantile(axis_values, 0.9)) / 8.0,
        float(np.mean(corridor_values)) / 8.0,
        float(np.std(corridor_values)) / 8.0,
        positive_fraction,
        # Marge au seuil : proche de 0 signifie que la bande est indécise, donc
        # qu'une correction bornée peut la faire basculer.
        float(np.mean(np.abs(corridor_values))) / 8.0,
        float(np.mean(left)) / 8.0,
        float(np.mean(right)) / 8.0,
        float(corridor.sum()) / float(shape[0] * shape[1]),
    ]
    if high_res_mean is not None:
        values = list(values) + [float(v) for v in high_res_mean]
    return np.asarray(values, dtype=np.float32)


def stack_features(
    evidence: Evidence,
    fragments: Sequence[Fragment],
    z0: np.ndarray | None = None,
) -> np.ndarray:
    """Matrice ``(n_fragments, D)``. ``D = EVIDENCE_DIM`` si ``z0`` est absent."""

    if not fragments:
        width = EVIDENCE_DIM if z0 is None else FEATURE_DIM
        return np.zeros((0, width), dtype=np.float32)
    rows = []
    for fragment in fragments:
        vector = evidence_features(evidence, fragment)
        if z0 is not None:
            vector = np.concatenate([vector, sam_features(z0, fragment)])
        rows.append(vector)
    return np.stack(rows).astype(np.float32)
