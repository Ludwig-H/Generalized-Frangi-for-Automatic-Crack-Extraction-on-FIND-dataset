"""Garanties structurelles de CrackSAM-GeoLoRA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geolora.adapter import (  # noqa: E402
    INJECTION_SPEC,
    GeometryAdapter,
    GeoGuidedCrackSAM2,
)
from geolora.losses import soft_cldice_loss, soft_skeleton  # noqa: E402


@dataclass
class _Features:
    image_embeddings: torch.Tensor
    high_resolution_features: tuple[torch.Tensor, ...]
    input_size: tuple[int, int]


class _FakeBackbone(torch.nn.Module):
    """Reproduit la seule interface utilisée : encode puis decode."""

    def __init__(self, scale: int = 8) -> None:
        super().__init__()
        self.scale = scale
        self.seen_mask_input: object = "jamais appelé"

    def encode_images(self, images: torch.Tensor) -> _Features:
        batch = images.shape[0]
        sizes = [size // self.scale for _, size in INJECTION_SPEC]
        channels = [channel for channel, _ in INJECTION_SPEC]
        torch.manual_seed(0)
        high = tuple(
            torch.ones(batch, channels[i], sizes[i], sizes[i]) for i in range(2)
        )
        embeddings = torch.ones(batch, channels[2], sizes[2], sizes[2])
        return _Features(embeddings, high, (448, 448))

    def decode_features(self, features, mask_input=None, output_size=None):
        self.seen_mask_input = mask_input
        pooled = features.image_embeddings.mean(dim=1, keepdim=True)
        total = pooled + sum(f.mean() for f in features.high_resolution_features)
        return {"logits": total}


def _spec_scaled(scale: int = 8):
    return tuple((channel, size // scale) for channel, size in INJECTION_SPEC)


def test_adapter_is_exactly_zero_at_initialisation():
    adapter = GeometryAdapter(spec=_spec_scaled())
    evidence = torch.randn(2, 11, 28, 28)
    for correction in adapter(evidence):
        assert torch.equal(correction, torch.zeros_like(correction))
    assert adapter.activation() == 0.0
    # Le gain, lui, ne doit PAS être nul : gamma == 0 avec des projections
    # nulles annule les deux gradients et fige l'adapter (constaté en réel).
    assert float(adapter.gamma.item()) != 0.0


def test_adapter_emits_one_correction_per_injection_point():
    adapter = GeometryAdapter(spec=_spec_scaled())
    corrections = adapter(torch.randn(3, 11, 28, 28))
    assert len(corrections) == len(INJECTION_SPEC)
    for correction, (channels, size) in zip(corrections, _spec_scaled()):
        assert correction.shape == (3, channels, size, size)


def test_untrained_composition_equals_the_frozen_baseline():
    backbone = _FakeBackbone()
    model = GeoGuidedCrackSAM2(backbone, GeometryAdapter(spec=_spec_scaled()))
    images = torch.rand(2, 3, 448, 448)
    evidence = torch.rand(2, 11, 28, 28)
    with torch.no_grad():
        without = model(images)["logits"]
        with_geometry = model(images, evidence=evidence)["logits"]
    assert torch.equal(without, with_geometry)


def test_geometry_never_reaches_mask_input():
    backbone = _FakeBackbone()
    model = GeoGuidedCrackSAM2(backbone, GeometryAdapter(spec=_spec_scaled()))
    with torch.no_grad():
        model(torch.rand(1, 3, 448, 448), evidence=torch.rand(1, 11, 28, 28))
    assert backbone.seen_mask_input is None


def test_adapter_moves_the_output_once_the_gain_opens():
    torch.manual_seed(0)
    backbone = _FakeBackbone()
    adapter = GeometryAdapter(spec=_spec_scaled())
    with torch.no_grad():
        for projection in adapter.projections:
            torch.nn.init.normal_(projection.weight, std=0.1)
    model = GeoGuidedCrackSAM2(backbone, adapter)
    images = torch.rand(1, 3, 448, 448)
    with torch.no_grad():
        without = model(images)["logits"]
        with_geometry = model(images, evidence=torch.rand(1, 11, 28, 28))["logits"]
    assert not torch.equal(without, with_geometry)


def test_evidence_none_stays_bit_exact_even_when_adapter_is_active():
    torch.manual_seed(1)
    adapter = GeometryAdapter(spec=_spec_scaled())
    with torch.no_grad():
        for projection in adapter.projections:
            torch.nn.init.normal_(projection.weight, std=0.5)
    model = GeoGuidedCrackSAM2(_FakeBackbone(), adapter)
    images = torch.rand(2, 3, 448, 448)
    reference = _FakeBackbone()
    with torch.no_grad():
        expected = reference.decode_features(reference.encode_images(images))["logits"]
        observed = model(images)["logits"]
    assert torch.equal(expected, observed)


def test_soft_skeleton_thins_a_thick_bar():
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 26:38, 8:56] = 1.0  # barre de 12 px de large
    skeleton = soft_skeleton(mask, iterations=10)
    assert skeleton.sum() < mask.sum() / 3.0
    assert skeleton.sum() > 0.0


def test_cldice_is_near_zero_for_identical_masks():
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 28:34, 6:58] = 1.0
    assert soft_cldice_loss(mask, mask) < 0.05


def test_cldice_penalises_a_broken_structure_more_than_a_thinner_one():
    target = torch.zeros(1, 1, 64, 64)
    target[..., 28:34, 6:58] = 1.0

    thinner = torch.zeros_like(target)
    thinner[..., 29:33, 6:58] = 1.0  # même topologie, plus fin

    broken = target.clone()
    broken[..., :, 28:36] = 0.0  # rupture franche au milieu

    assert soft_cldice_loss(broken, target) > soft_cldice_loss(thinner, target)


def test_cldice_gradient_flows():
    logits = torch.randn(1, 1, 48, 48, requires_grad=True)
    target = torch.zeros(1, 1, 48, 48)
    target[..., 20:28, 4:44] = 1.0
    loss = soft_cldice_loss(torch.sigmoid(logits), target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0.0


def test_evidence_channels_are_bounded_and_finite():
    from geolora.evidence import CHANNELS, N_CHANNELS, compute_evidence

    rng = np.random.default_rng(0)
    image = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
    image[100:110, :, :] = 25  # une bande sombre de 10 px, largeur cible
    evidence = compute_evidence(image)
    assert evidence.planes.shape == (N_CHANNELS, 224, 224)
    assert np.isfinite(evidence.planes).all()
    for name in CHANNELS:
        plane = evidence.plane(name)
        lower = -1.0 if name in {"cos2theta", "sin2theta"} else 0.0
        assert plane.min() >= lower - 1e-6, name
        assert plane.max() <= 1.0 + 1e-6, name


@pytest.mark.parametrize("width", [4, 10, 20])
def test_adapter_parameter_count_is_comparable_to_lora(width):
    adapter = GeometryAdapter(width=width * 4, spec=_spec_scaled())
    total = sum(p.numel() for p in adapter.parameters())
    # La LoRA baseline compte 453 248 paramètres. L'adapter doit rester du même
    # ordre, faute de quoi un gain pourrait venir de la capacité.
    assert total < 2_000_000, total


def test_adapter_gradients_are_not_both_dead_at_initialisation():
    """Régression : ``gamma=0`` et projections nulles annulaient tout gradient.

    Ce test aurait épargné une heure de GPU : la variante géométrique était
    numériquement identique à sa version sans géométrie.
    """

    adapter = GeometryAdapter(spec=_spec_scaled())
    evidence = torch.randn(2, 11, 28, 28)
    loss = sum(c.pow(2).sum() for c in adapter(evidence))
    # Une sortie nulle a un gradient nul pour une perte quadratique ; on ajoute
    # donc une perturbation linéaire pour sonder la direction de descente.
    loss = loss + sum(c.sum() for c in adapter(evidence))
    loss.backward()
    gradients = [
        p.weight.grad.abs().sum().item()
        for p in adapter.projections
        if p.weight.grad is not None
    ]
    assert gradients and max(gradients) > 0.0, "les projections ne reçoivent aucun gradient"
