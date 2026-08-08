"""Les deux plafonds à mesurer AVANT d'entraîner quoi que ce soit.

Le rapport du 8 août 2026 l'exige explicitement (§10) : « Avant de l'entraîner,
deux plafonds […] doivent dépasser des seuils pré-enregistrés. »

L'oracle de **source** est la porte décisive : il majore ce qu'un arbitre peut
atteindre avec cette famille de fragments, ce corridor et cet ensemble
d'actions. Le seuil pré-enregistré est un gain d'IoU groupé ``>= +0,01``.

Deux quantités sont produites, et la distinction est essentielle pour qu'un
résultat négatif soit concluant :

``achievable``
    Gain réellement atteignable par une assignation d'actions par fragment,
    obtenu par montée de coordonnées. C'est une **borne inférieure** du plafond.
``upper_bound``
    Gain obtenu en choisissant librement le label de chaque pixel *à l'intérieur
    de l'union des corridors*. Aucune assignation d'actions par fragment ne peut
    faire mieux, donc c'est une **borne supérieure** stricte.

Si ``upper_bound < 0,01`` le no-go est concluant sans réserve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Sequence

import numpy as np

from .fragments import Fragment

Action = Literal["abstain", "add", "remove"]
ACTIONS: Final[tuple[Action, ...]] = ("abstain", "add", "remove")

#: Amplitude bornée d'une correction, en nats de logit (§2.3 du DESIGN).
DELTA: Final[float] = 4.0


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    """IoU binaire, conventions identiques à ``cracksam2.metrics``.

    Deux masques vides valent ``1``; un seul vide vaut ``0``.
    """

    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    pred_count = int(pred.sum())
    truth_count = int(truth.sum())
    if pred_count == 0 and truth_count == 0:
        return 1.0
    if pred_count == 0 or truth_count == 0:
        return 0.0
    intersection = int(np.count_nonzero(pred & truth))
    return intersection / float(pred_count + truth_count - intersection)


def apply_actions(
    z0: np.ndarray,
    fragments: Sequence[Fragment],
    actions: Sequence[Action],
    delta: float = DELTA,
) -> np.ndarray:
    """Applique ``torch.where(B, z0 + dz, z0)`` en NumPy.

    Hors de ``B`` la sortie est ``z0`` **bit à bit** : c'est la garantie C1/C2
    du pré-enregistrement, et elle est vérifiée par les tests.
    """

    if len(fragments) != len(actions):
        raise ValueError("un fragment doit recevoir exactement une action")
    shape = z0.shape
    accumulator = np.zeros(shape, dtype=np.float32)
    accepted = np.zeros(shape, dtype=bool)
    for fragment, action in zip(fragments, actions):
        if action == "abstain":
            continue
        corridor = fragment.corridor(shape)
        accepted |= corridor
        accumulator[corridor] += float(delta) if action == "add" else -float(delta)
    if not accepted.any():
        return z0.copy()
    bounded = np.clip(accumulator, -float(delta), float(delta))
    return np.where(accepted, z0 + bounded, z0).astype(np.float32)


@dataclass(frozen=True)
class SourceOracleResult:
    baseline_iou: float
    achievable_iou: float
    upper_bound_iou: float
    actions: tuple[Action, ...]
    n_fragments: int
    n_add: int
    n_remove: int
    corridor_pixels: int

    @property
    def achievable_gain(self) -> float:
        return self.achievable_iou - self.baseline_iou

    @property
    def upper_bound_gain(self) -> float:
        return self.upper_bound_iou - self.baseline_iou


def source_oracle(
    z0: np.ndarray,
    target: np.ndarray,
    fragments: Sequence[Fragment],
    delta: float = DELTA,
    passes: int = 3,
) -> SourceOracleResult:
    """Plafond de la famille de candidats sur une image.

    Le corridor de chaque fragment est **fixé sans GT** : l'oracle choisit une
    action, il ne recoupe, ne déplace, ne réoriente ni ne redimensionne aucun
    fragment. C'est la condition posée au §10 du rapport.
    """

    truth = np.asarray(target, dtype=bool)
    baseline = binary_iou(z0 > 0.0, truth)

    if not fragments:
        return SourceOracleResult(
            baseline_iou=baseline,
            achievable_iou=baseline,
            upper_bound_iou=baseline,
            actions=(),
            n_fragments=0,
            n_add=0,
            n_remove=0,
            corridor_pixels=0,
        )

    shape = z0.shape
    union = np.zeros(shape, dtype=bool)
    for fragment in fragments:
        union |= fragment.corridor(shape)

    # Borne supérieure : label libre par pixel dans l'union des corridors.
    # Aucune assignation d'actions ne peut la dépasser.
    upper = np.where(union, truth, z0 > 0.0)
    upper_bound = binary_iou(upper, truth)

    # Montée de coordonnées : chaque fragment choisit son action au vu des
    # autres, jusqu'à stabilisation. Initialisation à « s'abstenir », donc le
    # point de départ est exactement z0.
    actions: list[Action] = ["abstain"] * len(fragments)
    best = baseline
    for _ in range(int(passes)):
        changed = False
        for index in range(len(fragments)):
            current = actions[index]
            local_best, local_action = best, current
            for candidate in ACTIONS:
                if candidate == current:
                    continue
                actions[index] = candidate
                score = binary_iou(
                    apply_actions(z0, fragments, actions, delta) > 0.0, truth
                )
                if score > local_best + 1e-12:
                    local_best, local_action = score, candidate
            actions[index] = local_action
            if local_action != current:
                changed = True
                best = local_best
        if not changed:
            break

    return SourceOracleResult(
        baseline_iou=baseline,
        achievable_iou=best,
        upper_bound_iou=upper_bound,
        actions=tuple(actions),
        n_fragments=len(fragments),
        n_add=sum(1 for a in actions if a == "add"),
        n_remove=sum(1 for a in actions if a == "remove"),
        corridor_pixels=int(union.sum()),
    )


def marginal_utility(
    z0: np.ndarray,
    target: np.ndarray,
    fragments: Sequence[Fragment],
    delta: float = DELTA,
) -> np.ndarray:
    """Utilité marginale ``(n, 3)`` de chaque action, fragment par fragment.

    C'est l'étiquette d'entraînement de l'arbitre : « cette bande, seule,
    améliore-t-elle ``z0`` ? ». Elle est calculée face à ``z0`` et non face à la
    prédiction courante, donc elle ne dépend pas d'un ordre d'itération.
    """

    truth = np.asarray(target, dtype=bool)
    baseline = binary_iou(z0 > 0.0, truth)
    utilities = np.zeros((len(fragments), len(ACTIONS)), dtype=np.float32)
    for index, fragment in enumerate(fragments):
        for action_index, action in enumerate(ACTIONS):
            if action == "abstain":
                continue
            solo: list[Action] = ["abstain"] * len(fragments)
            solo[index] = action
            score = binary_iou(
                apply_actions(z0, fragments, solo, delta) > 0.0, truth
            )
            utilities[index, action_index] = score - baseline
    return utilities


def sample_points_geodesic(pixels: np.ndarray, count: int) -> np.ndarray:
    """Échantillonne ``count`` points régulièrement le long d'une polyligne.

    Le rapport (§3.7, leçon TPP-SAM) demande un échantillonnage géodésique
    plutôt qu'un top-K raster : deux pixels voisins de forte réponse ne sont pas
    deux prompts informatifs.
    """

    n = int(pixels.shape[0])
    if n == 0 or count <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    if count >= n:
        return pixels.astype(np.int32)
    indices = np.linspace(0, n - 1, int(count)).round().astype(int)
    return pixels[np.unique(indices)].astype(np.int32)
