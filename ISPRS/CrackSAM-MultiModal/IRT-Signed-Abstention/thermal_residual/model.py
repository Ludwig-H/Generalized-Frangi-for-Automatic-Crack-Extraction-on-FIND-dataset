"""Correcteur résiduel signé à abstention.

La baseline CrackSAM est gelée et n'apparaît ici que par ses **logits**, toujours
détachés. Le correcteur produit par pixel trois probabilités — renforcer,
supprimer, s'abstenir — et la correction

.. math:: \\Delta z = m\\,\\delta_{\\max}(\\pi^+ - \\pi^-)

qui est nulle **exactement** à l'initialisation et lorsque la thermique est
absente. Deux têtes de contrôle partagent le même encodeur : ``signed`` (deux
actions, sans abstention) et ``positive_only`` (renforcement seul).

.. warning::
   ``positive_only`` ne peut pas satisfaire l'identité bit-à-bit à
   l'initialisation, et ce n'est pas un défaut d'implémentation : une correction
   à la fois **non négative**, **nulle** au point d'initialisation et **de
   gradient non nul** en ce point n'existe pas — la nullité y serait un minimum,
   donc un point critique. Ce bras part de ``σ(−8) ≈ 3,4·10⁻⁴``, soit
   ``|Δz| ≤ 1,3·10⁻³`` : identité à ``10⁻³`` près, pas au bit près.
"""

from __future__ import annotations

from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    ACTION_ABSTAIN,
    ACTION_NAMES,
    ACTION_REINFORCE,
    ACTION_SUPPRESS,
    CORRECTOR_INPUT_CHANNELS,
    EVIDENCE_CHANNELS,
    HEAD_POSITIVE_ONLY,
    HEAD_SIGNED,
    HEAD_SIGNED_ABSTENTION,
    HEADS,
    LOGIT_CLIP,
    SCOPE_EVIDENCE_UNION,
    SCOPE_GLOBAL,
    SCOPES,
)

#: Biais d'initialisation de la tête à trois actions : ``π⁺ = π⁻`` exactement,
#: donc ``Δz = 0`` exactement, avec des dérivées non nulles.
IDENTITY_BIAS_THREE: Final[tuple[float, float, float]] = (-2.0, -2.0, 2.0)

#: Tête à deux actions : ``softmax(0, 0)`` donne ``π⁺ = π⁻ = 0,5``.
IDENTITY_BIAS_TWO: Final[tuple[float, float]] = (0.0, 0.0)

#: Tête positive seule : voir l'avertissement du module.
NEAR_IDENTITY_BIAS_POSITIVE: Final[float] = -8.0

#: Index du canal ``similarity_max`` dans le tenseur d'évidence.
SIMILARITY_MAX_INDEX: Final[int] = EVIDENCE_CHANNELS.index("similarity_max")

#: Index du canal ``support_union``.
SUPPORT_INDEX: Final[int] = EVIDENCE_CHANNELS.index("support_union")

EPSILON: Final[float] = 1e-6


def binary_entropy(probability: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    """Entropie binaire, en nats, bornée et sans ``log(0)``."""

    return -(
        probability * torch.log(probability + eps)
        + (1.0 - probability) * torch.log1p(-probability + eps)
    )


def _head_channels(head: str) -> int:
    if head == HEAD_SIGNED_ABSTENTION:
        return 3
    if head == HEAD_SIGNED:
        return 2
    return 1


class ThermalSignedAbstentionAdapter(nn.Module):
    """Petit correcteur convolutif dilaté, strictement inférieur à 100 000 paramètres.

    Parameters
    ----------
    evidence_channels:
        Nombre de plans d'évidence. Quatre dans tous les bras, y compris les
        contrôles, pour que la capacité d'entrée soit identique.
    hidden_channels:
        Largeur de l'encodeur.
    delta_max:
        Amplitude maximale de la correction en logits.
    head:
        ``signed_abstention`` (protocole), ``signed`` (contrôle A5) ou
        ``positive_only`` (contrôle A6).
    correction_scope:
        ``global`` (protocole causal) ou ``evidence_union`` (ablation §5.6).
    """

    def __init__(
        self,
        evidence_channels: int = len(EVIDENCE_CHANNELS),
        hidden_channels: int = 32,
        delta_max: float = 12.0,
        logit_clip: float = LOGIT_CLIP,
        head: str = HEAD_SIGNED_ABSTENTION,
        correction_scope: str = SCOPE_GLOBAL,
        support_dilation: int = 3,
        baseline_scope_threshold: float = 0.35,
    ) -> None:
        super().__init__()
        if evidence_channels < 1:
            raise ValueError("evidence_channels doit être positif")
        if hidden_channels < 1:
            raise ValueError("hidden_channels doit être positif")
        if delta_max <= 0.0:
            raise ValueError("delta_max doit être strictement positif")
        if logit_clip <= 0.0:
            raise ValueError("logit_clip doit être strictement positif")
        if head not in HEADS:
            raise ValueError(f"tête inconnue : {head!r} (attendu {HEADS})")
        if correction_scope not in SCOPES:
            raise ValueError(f"portée inconnue : {correction_scope!r} (attendu {SCOPES})")
        if support_dilation < 0:
            raise ValueError("support_dilation ne peut pas être négatif")

        self.evidence_channels = int(evidence_channels)
        self.hidden_channels = int(hidden_channels)
        self.delta_max = float(delta_max)
        self.logit_clip = float(logit_clip)
        self.head = str(head)
        self.correction_scope = str(correction_scope)
        self.support_dilation = int(support_dilation)
        self.baseline_scope_threshold = float(baseline_scope_threshold)

        input_channels = CORRECTOR_INPUT_CHANNELS - len(EVIDENCE_CHANNELS) + self.evidence_channels
        groups = min(8, self.hidden_channels)
        layers: list[nn.Module] = []
        in_channels = input_channels
        for dilation in (1, 2, 4):
            layers.extend(
                (
                    nn.Conv2d(
                        in_channels,
                        self.hidden_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.GroupNorm(groups, self.hidden_channels),
                    nn.GELU(),
                )
            )
            in_channels = self.hidden_channels
        self.encoder = nn.Sequential(*layers)
        self.action_head = nn.Conv2d(self.hidden_channels, _head_channels(self.head), kernel_size=1)
        self.reset_identity()

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def reset_identity(self) -> None:
        """Initialise la tête pour que ``Δz`` soit nul avec des gradients vivants.

        Les **poids** sont mis à zéro, pas le biais : la sortie de la tête est
        alors constante et le terme ``π⁺ − π⁻`` s'annule par symétrie, tandis que
        ``∂(π⁺−π⁻)/∂q`` reste non nul. Mettre simultanément à zéro une porte
        scalaire et une projection — l'erreur mesurée dans GeoLoRA, où ``gamma``
        restait à ``0,0000`` — annulerait les deux gradients.
        """

        nn.init.zeros_(self.action_head.weight)
        with torch.no_grad():
            if self.head == HEAD_SIGNED_ABSTENTION:
                self.action_head.bias.copy_(torch.tensor(IDENTITY_BIAS_THREE))
            elif self.head == HEAD_SIGNED:
                self.action_head.bias.copy_(torch.tensor(IDENTITY_BIAS_TWO))
            else:
                self.action_head.bias.fill_(NEAR_IDENTITY_BIAS_POSITIVE)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config(self) -> dict[str, Any]:
        return {
            "evidence_channels": self.evidence_channels,
            "hidden_channels": self.hidden_channels,
            "delta_max": self.delta_max,
            "logit_clip": self.logit_clip,
            "head": self.head,
            "correction_scope": self.correction_scope,
            "support_dilation": self.support_dilation,
            "baseline_scope_threshold": self.baseline_scope_threshold,
            "trainable_parameters": self.trainable_parameters(),
        }

    # ------------------------------------------------------------------ #
    # Portée
    # ------------------------------------------------------------------ #

    def build_scope(
        self, baseline_probability: torch.Tensor, thermal_evidence: torch.Tensor
    ) -> torch.Tensor:
        """Masque ``Ω`` de la portée ``evidence_union`` — §5.6.

        ``Ω = Dilate(support, r) ∪ {p₀ > τ}``. La dilatation est un
        ``max_pool2d`` : approximation carrée d'un disque, sans conséquence pour
        une tolérance choisie plutôt qu'estimée.
        """

        support = thermal_evidence[:, SUPPORT_INDEX : SUPPORT_INDEX + 1]
        if self.support_dilation > 0:
            size = 2 * self.support_dilation + 1
            support = F.max_pool2d(support, kernel_size=size, stride=1, padding=self.support_dilation)
        foreground = (baseline_probability > self.baseline_scope_threshold).to(support.dtype)
        return torch.clamp(support + foreground, max=1.0)

    # ------------------------------------------------------------------ #
    # Passe avant
    # ------------------------------------------------------------------ #

    def forward(
        self,
        baseline_logits: torch.Tensor,
        thermal_evidence: torch.Tensor,
        modality_present: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if baseline_logits.ndim != 4 or baseline_logits.shape[1] != 1:
            raise ValueError(
                f"baseline_logits doit être (B,1,H,W), reçu {tuple(baseline_logits.shape)}"
            )
        if thermal_evidence.ndim != 4 or thermal_evidence.shape[1] != self.evidence_channels:
            raise ValueError(
                f"thermal_evidence doit être (B,{self.evidence_channels},H,W), "
                f"reçu {tuple(thermal_evidence.shape)}"
            )
        if thermal_evidence.shape[0] != baseline_logits.shape[0] or (
            thermal_evidence.shape[-2:] != baseline_logits.shape[-2:]
        ):
            raise ValueError("évidence et logits doivent partager lot et résolution")
        if modality_present.ndim != 1 or modality_present.shape[0] != baseline_logits.shape[0]:
            raise ValueError("modality_present doit être un vecteur booléen de taille B")
        if modality_present.dtype is not torch.bool:
            raise TypeError("modality_present doit avoir dtype torch.bool")

        z0 = baseline_logits.detach()
        p0 = torch.sigmoid(z0)
        entropy = binary_entropy(p0)

        features = torch.cat(
            (
                z0.clamp(-self.logit_clip, self.logit_clip) / self.logit_clip,
                p0,
                entropy,
                thermal_evidence.to(dtype=z0.dtype),
            ),
            dim=1,
        )
        hidden = self.encoder(features)
        action_logits = self.action_head(hidden)

        present = modality_present.to(device=z0.device, dtype=z0.dtype).reshape(-1, 1, 1, 1)

        if self.head == HEAD_POSITIVE_ONLY:
            gate = torch.sigmoid(action_logits)
            similarity = thermal_evidence[
                :, SIMILARITY_MAX_INDEX : SIMILARITY_MAX_INDEX + 1
            ].to(dtype=z0.dtype)
            delta = self.delta_max * gate * similarity
            probabilities = torch.cat(
                (gate, torch.zeros_like(gate), 1.0 - gate), dim=1
            )
        else:
            probabilities = torch.softmax(action_logits, dim=1)
            reinforce = probabilities[:, ACTION_REINFORCE : ACTION_REINFORCE + 1]
            suppress = probabilities[:, ACTION_SUPPRESS : ACTION_SUPPRESS + 1]
            delta = self.delta_max * (reinforce - suppress)
            if self.head == HEAD_SIGNED:
                probabilities = torch.cat(
                    (reinforce, suppress, torch.zeros_like(reinforce)), dim=1
                )

        delta = present * delta
        scope = None
        if self.correction_scope == SCOPE_EVIDENCE_UNION:
            scope = self.build_scope(p0, thermal_evidence)
            delta = delta * scope

        logits = z0 + delta
        reinforce_probability = probabilities[:, ACTION_REINFORCE : ACTION_REINFORCE + 1]
        suppress_probability = probabilities[:, ACTION_SUPPRESS : ACTION_SUPPRESS + 1]
        abstain_probability = probabilities[:, ACTION_ABSTAIN : ACTION_ABSTAIN + 1]

        return {
            "logits": logits,
            "baseline_logits": z0,
            "residual_logits": delta,
            "action_logits": action_logits,
            "action_probabilities": probabilities,
            "reinforce_probability": reinforce_probability,
            "suppress_probability": suppress_probability,
            "abstain_probability": abstain_probability,
            "active_probability": 1.0 - abstain_probability,
            "hard_action": probabilities.argmax(dim=1, keepdim=True),
            "correction_scope": (
                scope if scope is not None else torch.ones_like(delta)
            ),
        }


def build_adapter(config: dict[str, Any]) -> ThermalSignedAbstentionAdapter:
    """Construit le correcteur depuis la section ``model`` d'une configuration."""

    return ThermalSignedAbstentionAdapter(
        evidence_channels=int(config.get("evidence_channels", len(EVIDENCE_CHANNELS))),
        hidden_channels=int(config.get("hidden_channels", 32)),
        delta_max=float(config.get("delta_max", 12.0)),
        logit_clip=float(config.get("logit_clip", LOGIT_CLIP)),
        head=str(config.get("head", HEAD_SIGNED_ABSTENTION)),
        correction_scope=str(config.get("correction_scope", SCOPE_GLOBAL)),
        support_dilation=int(config.get("support_dilation", 3)),
        baseline_scope_threshold=float(config.get("baseline_scope_threshold", 0.35)),
    )


__all__ = [
    "ACTION_NAMES",
    "IDENTITY_BIAS_THREE",
    "IDENTITY_BIAS_TWO",
    "NEAR_IDENTITY_BIAS_POSITIVE",
    "ThermalSignedAbstentionAdapter",
    "binary_entropy",
    "build_adapter",
]
