"""Pondération par la difficulté — le correctif issu du diagnostic.

La campagne du 12 août a montré que le gain de l'évidence thermique se concentre
sur le tiers du test où la baseline échoue et **s'inverse** ailleurs. Ces tests
fixent les propriétés du correctif : équivalence exacte quand les poids sont
uniformes, et effet mesurable quand ils ne le sont pas.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from geolora.losses import tolerant_loss
from thermal_residual.difficulty import baseline_headroom, difficulty_weights
from thermal_residual.losses import (
    LossWeights,
    segmentation_loss,
    tolerant_loss_per_image,
)


def _batch(batch: int = 4, size: tuple[int, int] = (24, 32)):
    torch.manual_seed(2)
    logits = torch.randn(batch, 1, *size) * 2.0
    targets = (torch.rand(batch, 1, *size) > 0.7).float()
    return logits, targets


def test_perte_tolerante_par_image_coincide_avec_celle_de_geolora() -> None:
    """La version par image doit être la même perte, sans sa moyenne finale."""

    logits, targets = _batch()
    probabilities = torch.sigmoid(logits)
    per_image = tolerant_loss_per_image(probabilities, targets, radius=3)
    assert per_image.shape == (logits.shape[0],)
    assert torch.allclose(per_image.mean(), tolerant_loss(probabilities, targets, radius=3), atol=1e-6)


def test_poids_uniformes_ne_changent_rien() -> None:
    logits, targets = _batch()
    weights = LossWeights()
    without, _ = segmentation_loss(logits, targets, weights)
    with_ones, _ = segmentation_loss(logits, targets, weights, torch.ones(logits.shape[0]))
    assert torch.allclose(without, with_ones, atol=1e-6)


def test_les_poids_deplacent_bien_la_perte() -> None:
    """Pondérer vers une image doit rapprocher la perte de celle de cette image."""

    logits, targets = _batch()
    weights = LossWeights()
    focused = torch.zeros(logits.shape[0])
    focused[0] = 1.0
    pooled, _ = segmentation_loss(logits, targets, weights, focused)
    alone, _ = segmentation_loss(logits[:1], targets[:1], weights)
    assert torch.allclose(pooled, alone, atol=1e-5)


def test_normalisation_des_poids() -> None:
    raw = {"a": 0.0, "b": 0.5, "c": 1.0}
    w = difficulty_weights(raw, floor=0.1, max_ratio=5.0)
    assert set(w) == set(raw)
    assert all(value > 0 for value in w.values())
    assert w["c"] > w["b"] > w["a"], "une image plus difficile doit peser plus"
    assert max(w.values()) <= 5.0 + 1e-9
    assert abs(float(np.mean(list(w.values()))) - 1.0) < 0.5


def test_le_plancher_empeche_une_image_de_disparaitre() -> None:
    w = difficulty_weights({"parfaite": 0.0, "ratee": 1.0}, floor=0.1)
    assert w["parfaite"] > 0.0, "une image parfaite doit continuer à contraindre le modèle"
    assert w["ratee"] / w["parfaite"] == pytest.approx(10.0, rel=0.01)


def test_poids_vides() -> None:
    assert difficulty_weights({}) == {}


def test_marge_calculee_depuis_les_logits_caches(caches, fake_manifest) -> None:
    baseline, _ = caches
    headroom = baseline_headroom(fake_manifest[:4], baseline)
    assert set(headroom) == {s.sample_id for s in fake_manifest[:4]}
    assert all(0.0 <= value <= 1.0 for value in headroom.values())
    # Les logits fabriqués par la fixture effacent la moitié droite du masque :
    # la marge doit donc être franchement positive.
    assert min(headroom.values()) > 0.05


def test_le_dataset_porte_les_poids(caches, fake_manifest) -> None:
    from thermal_residual.data import IRTResidualDataset, collate

    baseline, thermal = caches
    weights = {s.sample_id: 1.0 + index for index, s in enumerate(fake_manifest)}
    dataset = IRTResidualDataset(fake_manifest, baseline, thermal, sample_weights=weights)
    batch = collate([dataset[i] for i in range(3)])
    assert batch["sample_weight"].shape == (3,)
    assert float(batch["sample_weight"][2]) == pytest.approx(3.0)

    unweighted = IRTResidualDataset(fake_manifest, baseline, thermal)
    assert float(unweighted[0]["sample_weight"]) == pytest.approx(1.0)
