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

from cracksam2.losses import binary_dice_loss_per_image  # noqa: E402
from geolora.losses import soft_dilate  # noqa: E402


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


def tolerant_loss_per_image(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    radius: int,
    smooth: float = 1.0,
) -> torch.Tensor:
    """``1 − F1`` tolérant, **une valeur par image**.

    Réécriture stricte de ``geolora.losses.tolerant_loss`` sans sa moyenne
    finale ; ``test_losses.py`` vérifie que la moyenne des deux coïncide.
    """

    if probabilities.shape != targets.shape:
        raise ValueError(f"formes incompatibles : {probabilities.shape} et {targets.shape}")
    dilated_targets = soft_dilate(targets, radius)
    dilated_predictions = soft_dilate(probabilities, radius)
    precision = (torch.sum(probabilities * dilated_targets, dim=(1, 2, 3)) + smooth) / (
        torch.sum(probabilities, dim=(1, 2, 3)) + smooth
    )
    recall = (torch.sum(targets * dilated_predictions, dim=(1, 2, 3)) + smooth) / (
        torch.sum(targets, dim=(1, 2, 3)) + smooth
    )
    return 1.0 - 2.0 * precision * recall / (precision + recall + 1e-8)


def _weighted_mean(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    if sample_weights is None:
        return values.mean()
    w = sample_weights.to(device=values.device, dtype=values.dtype)
    return (values * w).sum() / w.sum().clamp_min(1e-8)


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: LossWeights,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """BCE + Dice + perte tolérante, chacune rendue séparément.

    ``sample_weights`` pondère **par image**. Sans lui, le comportement est
    exactement la moyenne uniforme d'origine.
    """

    targets = targets.to(dtype=logits.dtype)
    bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = _weighted_mean(bce_map.mean(dim=(1, 2, 3)), sample_weights)
    dice = _weighted_mean(binary_dice_loss_per_image(logits, targets), sample_weights)
    probabilities = torch.sigmoid(logits)
    tolerant = _weighted_mean(
        tolerant_loss_per_image(probabilities, targets, radius=weights.tolerant_radius),
        sample_weights,
    )
    total = weights.bce * bce + weights.dice * dice + weights.tolerant * tolerant
    return total, {"bce": bce, "dice": dice, "tolerant": tolerant}


def corrector_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    weights: LossWeights,
    *,
    has_abstention: bool = True,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Coût complet et journal détaillé de ses composantes.

    ``has_abstention`` désactive le terme d'activité pour les têtes qui n'ont pas
    d'action « s'abstenir » : il y vaudrait ``1`` partout, donc une constante
    ajoutée au coût, ce qui rendrait les valeurs incomparables entre bras sans
    rien changer aux gradients.
    """

    logits = outputs["logits"]
    residual = outputs["residual_logits"]
    segmentation, components = segmentation_loss(logits, targets, weights, sample_weights)

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
