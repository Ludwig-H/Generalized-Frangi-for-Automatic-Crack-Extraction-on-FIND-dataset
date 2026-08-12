"""Fonction de coût du correcteur — §6 de la spécification.

.. math::
   \\mathcal L = \\mathcal L_{\\mathrm{seg}}
   + \\lambda_\\Delta \\mathcal L_\\Delta
   + \\lambda_{\\mathrm{active}} \\mathcal L_{\\mathrm{active}}
   + \\lambda_{\\mathrm{conflict}} \\mathcal L_{\\mathrm{conflict}}

Les trois termes de segmentation sont les implémentations **maintenues** du
dépôt : ``binary_dice_loss`` de ``cracksam2.losses`` et ``tolerant_loss`` de
``geolora.losses``, celle-là même dont le barreau ``tol3`` a battu la baseline
d'IoU stricte sur Khánh Hà. Rien n'est réécrit ici.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from . import _repo  # noqa: F401

from cracksam2.losses import binary_dice_loss  # noqa: E402
from geolora.losses import tolerant_loss  # noqa: E402


@dataclass(frozen=True)
class LossWeights:
    """Pondérations gelées. Toute modification se décide sur la validation."""

    bce: float = 0.5
    dice: float = 0.5
    tolerant: float = 0.25
    tolerant_radius: int = 3
    residual_l1: float = 1.0e-3
    active: float = 1.0e-4
    conflict: float = 1.0e-3

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "LossWeights":
        payload = dict(payload or {})
        regularization = dict(payload.pop("regularization", {}) or {})
        return cls(
            bce=float(payload.get("bce_weight", 0.5)),
            dice=float(payload.get("dice_weight", 0.5)),
            tolerant=float(payload.get("tolerant_weight", 0.25)),
            tolerant_radius=int(payload.get("tolerant_radius", 3)),
            residual_l1=float(regularization.get("residual_l1_weight", 1.0e-3)),
            active=float(regularization.get("active_weight", 1.0e-4)),
            conflict=float(regularization.get("conflict_weight", 1.0e-3)),
        )

    def to_json(self) -> dict[str, float]:
        return {
            "bce_weight": self.bce,
            "dice_weight": self.dice,
            "tolerant_weight": self.tolerant,
            "tolerant_radius": self.tolerant_radius,
            "residual_l1_weight": self.residual_l1,
            "active_weight": self.active,
            "conflict_weight": self.conflict,
        }


def segmentation_loss(
    logits: torch.Tensor, targets: torch.Tensor, weights: LossWeights
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """BCE + Dice + perte tolérante, chacune rendue séparément."""

    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = binary_dice_loss(logits, targets)
    probabilities = torch.sigmoid(logits)
    tolerant = tolerant_loss(probabilities, targets, radius=weights.tolerant_radius)
    total = weights.bce * bce + weights.dice * dice + weights.tolerant * tolerant
    return total, {"bce": bce, "dice": dice, "tolerant": tolerant}


def corrector_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    weights: LossWeights,
    *,
    has_abstention: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Coût complet et journal détaillé de ses composantes.

    ``has_abstention`` désactive le terme d'activité pour les têtes qui n'ont pas
    d'action « s'abstenir » : il y vaudrait ``1`` partout, donc une constante
    ajoutée au coût, ce qui rendrait les valeurs incomparables entre bras sans
    rien changer aux gradients.
    """

    logits = outputs["logits"]
    residual = outputs["residual_logits"]
    segmentation, components = segmentation_loss(logits, targets, weights)

    residual_l1 = residual.abs().mean()
    conflict = (
        outputs["reinforce_probability"] * outputs["suppress_probability"]
    ).mean()
    active = outputs["active_probability"].mean()

    total = segmentation + weights.residual_l1 * residual_l1 + weights.conflict * conflict
    if has_abstention:
        total = total + weights.active * active

    journal = {
        "loss": float(total.detach()),
        "segmentation": float(segmentation.detach()),
        "bce": float(components["bce"].detach()),
        "dice": float(components["dice"].detach()),
        "tolerant": float(components["tolerant"].detach()),
        "residual_l1": float(residual_l1.detach()),
        "conflict": float(conflict.detach()),
        "active": float(active.detach()),
    }
    return total, journal


__all__ = ["LossWeights", "corrector_loss", "segmentation_loss"]
