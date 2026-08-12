"""Audit de recalage **ancré sur la vérité terrain**, avec contrôle RGB.

Le premier estimateur écrit pour cette étude — corrélation croisée des gradients
entre RGB et thermique — s'est révélé inutilisable sur données réelles : 63 % de
ses estimations saturaient au bord de la fenêtre de recherche. Il mesurait sa
propre fenêtre, pas un décalage.

Celui-ci procède autrement, et se valide lui-même :

1. on prend le masque de vérité terrain comme **ancre** ;
2. on cherche le décalage entier qui maximise le contraste
   ``|moyenne sur la fissure − moyenne hors fissure|`` du champ testé ;
3. on applique **le même estimateur au RGB**. Si le RGB tombe à ``0 px``, alors
   l'estimateur fonctionne et l'annotation est bien alignée sur le visible ;
   tout écart mesuré sur la thermique est alors imputable à la thermique.

Sans ce contrôle, un décalage mesuré ne prouve rien : il peut venir de
l'estimateur, du contraste faible, ou de la fenêtre de recherche.

.. note::
   La vérité terrain n'est utilisée **que** pour cet audit, jamais pour
   construire l'évidence. Aucun décalage estimé ici n'est appliqué aux données
   d'entraînement : ce serait faire entrer l'annotation dans l'entrée du modèle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ShiftEstimate:
    """Décalage entier maximisant le contraste fissure / fond."""

    dy: int
    dx: int
    contrast: float
    saturated: bool

    @property
    def norm(self) -> float:
        return float(np.hypot(self.dy, self.dx))


def best_contrast_shift(
    field: np.ndarray, mask: np.ndarray, radius: int = 10
) -> ShiftEstimate:
    """Décalage entier de ``field`` qui aligne le mieux son contraste sur ``mask``.

    Le décalage est circulaire (``np.roll``) : sur des structures fines et un
    décalage de quelques pixels, l'effet de bord est négligeable devant le
    signal, et cela évite d'avoir à choisir une valeur de remplissage.
    """

    values = np.asarray(field, dtype=np.float32)
    target = np.asarray(mask, dtype=bool)
    if values.shape != target.shape:
        raise ValueError(f"formes incompatibles : {values.shape} et {target.shape}")
    if not target.any() or target.all():
        return ShiftEstimate(0, 0, 0.0, False)

    best = ShiftEstimate(0, 0, -1.0, False)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(values, dy, axis=0), dx, axis=1)
            contrast = float(abs(shifted[target].mean() - shifted[~target].mean()))
            if contrast > best.contrast:
                best = ShiftEstimate(dy, dx, contrast, abs(dy) == radius or abs(dx) == radius)
    return best


@dataclass(frozen=True)
class RegistrationReport:
    """Verdict de recalage d'une modalité, avec son contrôle."""

    name: str
    count: int
    median_norm: float
    fraction_exact: float
    fraction_within_2: float
    saturation_rate: float
    median_dy: float
    median_dx: float
    std_dy: float
    std_dx: float
    mean_contrast: float

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "median_shift_px": self.median_norm,
            "fraction_exact": self.fraction_exact,
            "fraction_within_2px": self.fraction_within_2,
            "saturation_rate": self.saturation_rate,
            "median_dy": self.median_dy,
            "median_dx": self.median_dx,
            "std_dy": self.std_dy,
            "std_dx": self.std_dx,
            "mean_contrast": self.mean_contrast,
        }


def summarize(name: str, estimates: Sequence[ShiftEstimate]) -> RegistrationReport:
    if not estimates:
        raise ValueError("aucune estimation à résumer")
    dy = np.array([e.dy for e in estimates], dtype=float)
    dx = np.array([e.dx for e in estimates], dtype=float)
    norms = np.hypot(dy, dx)
    return RegistrationReport(
        name=name,
        count=len(estimates),
        median_norm=float(np.median(norms)),
        fraction_exact=float((norms == 0).mean()),
        fraction_within_2=float((norms <= 2).mean()),
        saturation_rate=float(np.mean([e.saturated for e in estimates])),
        median_dy=float(np.median(dy)),
        median_dx=float(np.median(dx)),
        std_dy=float(dy.std()),
        std_dx=float(dx.std()),
        mean_contrast=float(np.mean([e.contrast for e in estimates])),
    )


def verdict(
    thermal: RegistrationReport,
    control: RegistrationReport,
    *,
    threshold_px: float = 3.0,
    control_max_px: float = 1.0,
) -> dict[str, Any]:
    """Conclut, et **refuse de conclure** si le contrôle RGB n'est pas propre.

    Un décalage thermique n'est interprétable que si le même estimateur trouve le
    RGB aligné. Sinon le verdict est ``"non concluant"`` : c'est l'estimateur qui
    est en cause, pas nécessairement les données.
    """

    control_ok = control.median_norm <= control_max_px
    if not control_ok:
        decision = "non concluant"
    elif thermal.median_norm <= threshold_px:
        decision = "accepté"
    else:
        decision = "REJETÉ"
    return {
        "decision": decision,
        "threshold_px": threshold_px,
        "control_is_clean": control_ok,
        "control": control.to_json(),
        "thermal": thermal.to_json(),
        "explanation": (
            "le contrôle RGB tombe à 0 px, donc l'estimateur fonctionne et "
            "l'écart mesuré sur la thermique lui est imputable"
            if control_ok
            else "le contrôle RGB n'est pas aligné : l'estimateur ou l'annotation "
            "sont en cause, aucun verdict sur la thermique n'est justifié"
        ),
    }


def audit_pairs(
    pairs: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]], *, radius: int = 10
) -> dict[str, Any]:
    """Audite une suite de ``(rgb_gris, thermique, masque)``."""

    control: list[ShiftEstimate] = []
    thermal: list[ShiftEstimate] = []
    for grayscale, decoded, mask in pairs:
        binary = np.asarray(mask, dtype=bool)
        if not binary.any():
            continue
        control.append(best_contrast_shift(grayscale, binary, radius))
        thermal.append(best_contrast_shift(decoded, binary, radius))
    if not control:
        raise ValueError("aucune paire exploitable : tous les masques sont vides")
    return verdict(summarize("thermique", thermal), summarize("rgb (contrôle)", control))


__all__ = [
    "RegistrationReport",
    "ShiftEstimate",
    "audit_pairs",
    "best_contrast_shift",
    "summarize",
    "verdict",
]
