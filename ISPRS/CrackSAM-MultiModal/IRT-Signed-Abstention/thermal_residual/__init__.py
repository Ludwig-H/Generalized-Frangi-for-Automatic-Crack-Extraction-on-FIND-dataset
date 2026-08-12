"""CrackSAM-IRT — correction thermique signée avec abstention.

Un CrackSAM 2 + LoRA entraîné sur Khánh Hà reste **gelé**. Un petit correcteur
(≈ 21 000 paramètres) lit ses logits et une évidence Frangi calculée sur la
modalité thermique — que SAM ne voit jamais — puis choisit par pixel entre
renforcer, supprimer et s'abstenir.

Spécification : ``README.md`` du dossier parent. Écarts assumés et corrections
apportées : ``ERRATA.md``.
"""

from __future__ import annotations

from . import _repo  # noqa: F401  (bootstrap sys.path avant tout autre import)

__all__ = [
    "cache",
    "config",
    "constants",
    "data",
    "evaluation",
    "losses",
    "manifest",
    "metrics",
    "model",
    "provenance",
    "splits",
    "stats",
    "thermal_decode",
    "thermal_frangi",
    "training",
]
