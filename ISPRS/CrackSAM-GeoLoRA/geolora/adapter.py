"""Adapter géométrique parallèle, injecté à trois échelles, initialisé à zéro.

Conception dictée par trois échecs mesurés :

* le pseudo-masque dense a coûté ``−0,0979`` d'IoU en causal — la géométrie
  n'entre donc **jamais** par ``mask_input`` ;
* une moyenne géométrique est équivariante sous perturbation multiplicative —
  les onze canaux restent donc **séparés** jusqu'à l'encodeur ;
* les corridors mono-échelle ne pouvaient atteindre qu'un tiers de la vérité
  terrain — l'injection est donc **multi-échelle**.

Les projections de sortie sont initialisées à zéro. À l'initialisation, le
modèle composé est donc **exactement** la baseline gelée : c'est la garantie qui
a tenu dans CrackSAM-GFA, et l'entraînement part d'un point dont on connaît la
valeur au bit près.

Points d'injection, mesurés sur SAM 2 Hiera-L à 448 px d'entrée (le trunk opère
en interne à 1024) :

===================  ==================
Sortie de SAM 2      Forme
===================  ==================
``high_res[0]``      ``(B, 32, 256, 256)``
``high_res[1]``      ``(B, 64, 128, 128)``
``image_embeddings`` ``(B, 256, 64, 64)``
===================  ==================
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Nombre de canaux d'évidence attendus en entrée.
IN_CHANNELS: Final[int] = 11

#: (canaux, résolution) de chaque point d'injection.
INJECTION_SPEC: Final[tuple[tuple[int, int], ...]] = ((32, 256), (64, 128), (256, 64))


def _block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.GELU(),
    )


class GeometryAdapter(nn.Module):
    """Encode les onze canaux et produit une correction additive par échelle."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        width: int = 48,
        spec: Sequence[tuple[int, int]] = INJECTION_SPEC,
    ) -> None:
        super().__init__()
        self.spec = tuple(spec)
        self.stem = nn.Sequential(_block(in_channels, width), _block(width, width))
        self.down1 = _block(width, width * 2, stride=2)
        self.down2 = _block(width * 2, width * 4, stride=2)

        widths = (width, width * 2, width * 4)
        # Projections finales à zéro : dz == 0 à l'initialisation.
        self.projections = nn.ModuleList(
            nn.Conv2d(w, channels, 1) for w, (channels, _) in zip(widths, self.spec)
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        # ATTENTION : le gain NE DOIT PAS être initialisé à zéro en même temps
        # que les projections. La sortie valant ``gamma * projection(x)``, deux
        # facteurs nuls annulent les DEUX gradients — c'est un point selle dont
        # l'adapter ne sort jamais. Mesuré : gamma restait à 0,0000 et la
        # variante géométrique était numériquement identique à ``cldice``.
        # Les projections nulles suffisent à garantir dz == 0 à l'initialisation.
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, evidence: torch.Tensor) -> list[torch.Tensor]:
        if evidence.ndim != 4:
            raise ValueError(f"evidence doit être (B,C,H,W), reçu {evidence.shape}")
        finest = self.spec[0][1]
        hidden = F.interpolate(
            evidence, size=(finest, finest), mode="bilinear", align_corners=False
        )
        level0 = self.stem(hidden)
        level1 = self.down1(level0)
        level2 = self.down2(level1)

        outputs = []
        for feature, projection, (channels, size) in zip(
            (level0, level1, level2), self.projections, self.spec
        ):
            corrected = projection(feature)
            if corrected.shape[-1] != size:
                corrected = F.interpolate(
                    corrected, size=(size, size), mode="bilinear", align_corners=False
                )
            outputs.append(self.gamma * corrected)
        return outputs

    @torch.no_grad()
    def activation(self) -> float:
        """Amplitude moyenne des projections : ``0`` = adapter inactif.

        C'est le bon indicateur, et non ``gamma`` : les projections partent de
        zéro et ne s'en écartent que si l'évidence sert à quelque chose.
        """

        weights = [p.weight.abs().mean() for p in self.projections]
        return float(torch.stack(weights).mean().item())


class GeoGuidedCrackSAM2(nn.Module):
    """SAM 2 + LoRA, augmenté d'une évidence géométrique additive.

    ``evidence=None`` reproduit **exactement** la voie baseline, ce qui rend le
    contrôle « sans évidence » gratuit et bit à bit.
    """

    def __init__(self, backbone: nn.Module, adapter: GeometryAdapter) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    def forward(
        self,
        images: torch.Tensor,
        evidence: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone.encode_images(images)
        if evidence is not None:
            corrections = self.adapter(evidence.to(images.dtype))
            high_res = [
                level + correction.to(level.dtype)
                for level, correction in zip(
                    features.high_resolution_features, corrections[:2]
                )
            ]
            embeddings = features.image_embeddings + corrections[2].to(
                features.image_embeddings.dtype
            )
            features = replace(
                features,
                image_embeddings=embeddings,
                high_resolution_features=tuple(high_res),
            )
        # mask_input reste None en toutes circonstances : cette interface est un
        # contrôle négatif, jamais un chemin d'entraînement.
        return self.backbone.decode_features(
            features, mask_input=None, output_size=output_size
        )
