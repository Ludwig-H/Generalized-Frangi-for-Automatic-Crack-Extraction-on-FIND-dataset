"""Arbitre local : ajouter / retirer / s'abstenir, une décision par fragment.

Garanties structurelles, vérifiées par les tests :

1. à l'initialisation, ``gamma = 0`` rend la correction **exactement** nulle,
   donc le modèle non entraîné vaut ``z0`` bit à bit ;
2. le biais de classification favorise ``s'abstenir``, donc l'enveloppe acceptée
   démarre quasi vide ;
3. les amplitudes sont **bornées** dans ``[0, A]`` et **non négatives** : le
   signe de la correction vient de l'action, jamais de l'amplitude ;
4. hors de l'enveloppe acceptée ``B``, la sortie est ``z0`` par ``torch.where``,
   et non par une confiance multipliée — c'est la différence exacte que le
   rapport du 8 août exige (§10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Amplitude maximale d'une correction, en nats de logit (§2.3 du DESIGN).
MAX_DELTA: Final[float] = 4.0

#: Indices des actions dans la tête de classification.
ABSTAIN: Final[int] = 0
ADD: Final[int] = 1
REMOVE: Final[int] = 2


@dataclass(frozen=True)
class ArbiterOutput:
    action_logits: torch.Tensor  # (n, 3)
    add_amplitude: torch.Tensor  # (n,) dans [0, MAX_DELTA]
    remove_amplitude: torch.Tensor  # (n,) dans [0, MAX_DELTA]

    def hard_actions(self) -> torch.Tensor:
        return torch.argmax(self.action_logits, dim=-1)


class FragmentArbiter(nn.Module):
    """MLP par fragment + attention entre fragments de la **même** image.

    L'attention est motivée par une observation du rapport : une frontière
    d'ombre est une structure longue, cohérente et fragmentée en plusieurs
    bandes alignées, alors qu'un réseau de fissures est connecté mais
    d'orientation variable. Un fragment isolé ne peut pas faire cette
    distinction ; l'ensemble des fragments d'une image le peut.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_delta: float = MAX_DELTA,
        abstain_bias: float = 4.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim doit être divisible par num_heads")
        self.max_delta = float(max_delta)
        self.input_norm = nn.LayerNorm(input_dim)
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.action_head = nn.Linear(hidden_dim, 3)
        self.add_head = nn.Linear(hidden_dim, 1)
        self.remove_head = nn.Linear(hidden_dim, 1)
        # Gain global initialisé à zéro : garantit dz == 0 au départ.
        self.gamma = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        with torch.no_grad():
            self.action_head.bias[ABSTAIN] = float(abstain_bias)
        nn.init.zeros_(self.add_head.weight)
        nn.init.zeros_(self.add_head.bias)
        nn.init.zeros_(self.remove_head.weight)
        nn.init.zeros_(self.remove_head.bias)

    def forward(
        self, features: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> ArbiterOutput:
        """``features`` : ``(B, n, D)``. ``padding_mask`` : ``True`` = à ignorer."""

        if features.ndim != 3:
            raise ValueError(f"features doit être (B, n, D), reçu {features.shape}")
        hidden = self.embed(self.input_norm(features))
        hidden = self.context(hidden, src_key_padding_mask=padding_mask)
        gain = torch.clamp(self.gamma, min=0.0)
        add = gain * self.max_delta * torch.sigmoid(self.add_head(hidden)).squeeze(-1)
        remove = (
            gain * self.max_delta * torch.sigmoid(self.remove_head(hidden)).squeeze(-1)
        )
        return ArbiterOutput(
            action_logits=self.action_head(hidden),
            add_amplitude=torch.clamp(add, 0.0, self.max_delta),
            remove_amplitude=torch.clamp(remove, 0.0, self.max_delta),
        )


def compose_logits(
    z0: torch.Tensor,
    corridors: torch.Tensor,
    output: ArbiterOutput,
    hard: bool = True,
    max_delta: float = MAX_DELTA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``z = where(B, z0 + dz, z0)`` et renvoie ``(z, B)``.

    ``corridors`` : ``(B, n, H, W)`` booléen, un corridor par fragment.

    En mode ``hard`` (évaluation), l'action est l'argmax et l'identité hors
    ``B`` est exacte. En mode doux (entraînement), les probabilités pondèrent la
    correction pour laisser passer le gradient, mais ``B`` reste défini par
    l'argmax : la sélection n'est jamais une confiance continue.
    """

    if corridors.ndim != 4:
        raise ValueError(f"corridors doit être (B, n, H, W), reçu {corridors.shape}")
    probabilities = torch.softmax(output.action_logits, dim=-1)
    chosen = output.hard_actions()

    if hard:
        add_weight = (chosen == ADD).to(z0.dtype)
        remove_weight = (chosen == REMOVE).to(z0.dtype)
    else:
        add_weight = probabilities[..., ADD]
        remove_weight = probabilities[..., REMOVE]

    signed = add_weight * output.add_amplitude - remove_weight * output.remove_amplitude
    masks = corridors.to(z0.dtype)
    delta = torch.einsum("bn,bnhw->bhw", signed, masks)
    delta = torch.clamp(delta, -float(max_delta), float(max_delta))

    accepted = (chosen != ABSTAIN).to(masks.dtype)
    envelope = torch.einsum("bn,bnhw->bhw", accepted, masks) > 0.0
    return torch.where(envelope, z0 + delta, z0), envelope


def arbiter_loss(
    output: ArbiterOutput,
    target_actions: torch.Tensor,
    utilities: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    abstain_weight: float = 0.25,
) -> torch.Tensor:
    """Entropie croisée pondérée par l'utilité marginale absolue.

    Une bande dont les trois actions changent l'IoU de façon négligeable ne doit
    pas peser autant qu'une bande décisive : la pondération par ``|utilité|``
    évite que l'arbitre apprenne surtout à trancher des cas sans enjeu. Les
    fragments dont l'action optimale est « s'abstenir » sont sous-pondérés pour
    la même raison, car ils dominent numériquement.
    """

    logits = output.action_logits.reshape(-1, 3)
    targets = target_actions.reshape(-1)
    losses = F.cross_entropy(logits, targets, reduction="none")

    best_utility = utilities.reshape(-1, 3).max(dim=-1).values.abs()
    weights = best_utility.clamp(min=0.0)
    weights = torch.where(
        targets == ABSTAIN, weights * float(abstain_weight) + 1e-3, weights + 1e-3
    )
    if padding_mask is not None:
        weights = weights * (~padding_mask.reshape(-1)).to(weights.dtype)
    return (losses * weights).sum() / weights.sum().clamp(min=1e-6)
