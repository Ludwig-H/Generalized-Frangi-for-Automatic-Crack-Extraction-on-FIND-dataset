"""Perte de continuité ``soft-clDice``, ajoutée à la perte CrackSAM existante.

Le mode d'échec que l'arbitrage par fragments tentait de réparer après coup est
la **rupture** : un réseau de fissures prédit en morceaux obtient un IoU
honorable mais une topologie fausse. ``soft-clDice`` pénalise explicitement ce
défaut en comparant les squelettes plutôt que les surfaces.

Référence : Shit et al., *clDice — a Novel Topology-Preserving Loss Function for
Tubular Structure Segmentation*, CVPR 2021,
`DOI 10.1109/CVPR46437.2021.01629 <https://doi.org/10.1109/CVPR46437.2021.01629>`_.

La squelettisation douce est différentiable : elle enchaîne des érosions et
dilatations morphologiques réalisées par ``min``/``max`` pooling.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

#: Itérations de squelettisation douce. Doit dépasser le demi-rayon des
#: structures : les fissures Khánh Hà font ``≈19 px`` de large à 448, donc un
#: rayon de 10 est le minimum utile.
DEFAULT_ITERATIONS: Final[int] = 10


def _soft_erode(mask: torch.Tensor) -> torch.Tensor:
    # L'érosion est une min-pooling, obtenue par max-pooling du complément.
    return -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)


def _soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def _soft_open(mask: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(mask))


def soft_skeleton(mask: torch.Tensor, iterations: int = DEFAULT_ITERATIONS):
    """Squelette doux différentiable d'un masque de probabilité dans ``[0,1]``."""

    opened = _soft_open(mask)
    skeleton = F.relu(mask - opened)
    for _ in range(int(iterations)):
        mask = _soft_erode(mask)
        opened = _soft_open(mask)
        delta = F.relu(mask - opened)
        # Union douce, bornée : évite que le squelette dépasse 1 par accumulation.
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    iterations: int = DEFAULT_ITERATIONS,
    smooth: float = 1.0,
) -> torch.Tensor:
    """``1 − clDice``. Entrées ``(B,1,H,W)`` : probabilités et cible binaire."""

    if probabilities.shape != targets.shape:
        raise ValueError(
            f"formes incompatibles : {probabilities.shape} et {targets.shape}"
        )
    prediction_skeleton = soft_skeleton(probabilities, iterations)
    target_skeleton = soft_skeleton(targets, iterations)

    # Précision topologique : le squelette prédit tombe-t-il dans la cible ?
    precision = (torch.sum(prediction_skeleton * targets, dim=(1, 2, 3)) + smooth) / (
        torch.sum(prediction_skeleton, dim=(1, 2, 3)) + smooth
    )
    # Sensibilité topologique : le squelette cible est-il couvert ?
    sensitivity = (
        torch.sum(target_skeleton * probabilities, dim=(1, 2, 3)) + smooth
    ) / (torch.sum(target_skeleton, dim=(1, 2, 3)) + smooth)

    cldice = 2.0 * precision * sensitivity / (precision + sensitivity)
    return (1.0 - cldice).mean()


def consistency_loss(
    clean_logits: torch.Tensor, shadowed_logits: torch.Tensor
) -> torch.Tensor:
    """Cohérence propre / ombré : la prédiction doit résister à l'ombre.

    Elle transforme en objectif optimisé ce que l'étude anti-ombre ne pouvait
    poser que comme hypothèse. La cible est détachée : c'est la version ombrée
    qui doit rejoindre la version propre, et non l'inverse.
    """

    return F.mse_loss(
        torch.sigmoid(shadowed_logits), torch.sigmoid(clean_logits).detach()
    )


def soft_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilatation douce par max-pooling, différentiable.

    Approximation carrée de la dilatation euclidienne employée par la métrique.
    L'écart est sans conséquence ici : le rayon n'est pas une grandeur à
    estimer, c'est une tolérance choisie.
    """

    if radius <= 0:
        return mask
    size = 2 * int(radius) + 1
    return F.max_pool2d(mask, kernel_size=size, stride=1, padding=int(radius))


def tolerant_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    radius: int,
    smooth: float = 1.0,
) -> torch.Tensor:
    """``1 − F1`` tolérant : ne pénalise plus le placement à moins de ``radius``.

    Version douce et différentiable de la métrique ``buffered`` de
    ``scripts/05_tolerant_iou.py`` :

    * un pixel prédit compte s'il tombe à moins de ``radius`` de la vérité ;
    * un pixel vrai est couvert s'il a une prédiction à moins de ``radius``.

    Motivation mesurée : sur ce corpus, dilater la vérité terrain d'un seul
    pixel fait chuter son IoU contre elle-même à ``0,881``. Faire payer au
    réseau une erreur de frontière d'un pixel détourne sa capacité de ce qui
    compte — les branches manquées et les ruptures.
    """

    if probabilities.shape != targets.shape:
        raise ValueError(
            f"formes incompatibles : {probabilities.shape} et {targets.shape}"
        )
    dilated_targets = soft_dilate(targets, radius)
    dilated_predictions = soft_dilate(probabilities, radius)

    precision = (
        torch.sum(probabilities * dilated_targets, dim=(1, 2, 3)) + smooth
    ) / (torch.sum(probabilities, dim=(1, 2, 3)) + smooth)
    recall = (
        torch.sum(targets * dilated_predictions, dim=(1, 2, 3)) + smooth
    ) / (torch.sum(targets, dim=(1, 2, 3)) + smooth)

    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return (1.0 - f1).mean()
