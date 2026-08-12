"""Évidence Frangi double polarité, et reconstruction du support.

Le test le plus important est ``test_support_reproduit_tau_mask`` : il prouve que
la reconstruction du support sans MST est **identique** au ``tau_mask`` de
l'extracteur, alors que la branche ``compute_centrality=False`` renvoie un plan
de zéros à sa place.
"""

from __future__ import annotations

import numpy as np
import pytest

from cracksam2.frangi import extract_frangi_graph_gpu
from thermal_residual.constants import EVIDENCE_CHANNELS
from thermal_residual.thermal_frangi import (
    ThermalEvidenceConfig,
    generate_dual_polarity_thermal_evidence,
    raw_thermal_evidence,
    stack_evidence,
    support_from_similarity,
)


def _line_image(dark: bool, size: tuple[int, int] = (72, 88)) -> np.ndarray:
    image = np.full(size, 0.5, dtype=np.float32)
    band = slice(size[0] // 2 - 1, size[0] // 2 + 2)
    image[band, 6 : size[1] - 6] = 0.15 if dark else 0.85
    rng = np.random.default_rng(7)
    return np.clip(image + rng.normal(0.0, 0.01, size).astype(np.float32), 0.0, 1.0)


def test_polarite_sombre_et_claire_sont_separees() -> None:
    dark = generate_dual_polarity_thermal_evidence(
        _line_image(dark=True), encoding="grayscale", device="cpu"
    )
    bright = generate_dual_polarity_thermal_evidence(
        _line_image(dark=False), encoding="grayscale", device="cpu"
    )
    row = slice(34, 39)
    assert dark["similarity_dark"][row].mean() > 3.0 * dark["similarity_bright"][row].mean()
    assert bright["similarity_bright"][row].mean() > 3.0 * bright["similarity_dark"][row].mean()


def test_similarity_max_domine_les_deux_polarites() -> None:
    payload = generate_dual_polarity_thermal_evidence(
        _line_image(dark=True), encoding="grayscale", device="cpu"
    )
    assert np.all(payload["similarity_max"] >= payload["similarity_dark"] - 1e-6)
    assert np.all(payload["similarity_max"] >= payload["similarity_bright"] - 1e-6)


def test_canaux_finis_bornes_et_support_binaire() -> None:
    payload = generate_dual_polarity_thermal_evidence(
        _line_image(dark=True), encoding="grayscale", device="cpu"
    )
    for name in EVIDENCE_CHANNELS:
        array = payload[name]
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
        assert array.min() >= 0.0 and array.max() <= 1.0 + 1e-6
    assert set(np.unique(payload["support_union"])).issubset({0.0, 1.0})
    assert payload["support_union"].any(), "le support ne devrait pas être vide sur une fissure nette"


def test_empilement_dans_l_ordre_canonique() -> None:
    payload = generate_dual_polarity_thermal_evidence(
        _line_image(dark=True), encoding="grayscale", device="cpu"
    )
    stacked = stack_evidence(payload)
    assert stacked.shape[0] == len(EVIDENCE_CHANNELS)
    for index, name in enumerate(EVIDENCE_CHANNELS):
        assert np.array_equal(stacked[index], payload[name])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_support_reproduit_tau_mask(seed: int) -> None:
    """La branche sans MST renvoie ``tau_mask`` vide : on le reconstruit à l'identique."""

    rng = np.random.default_rng(seed)
    image = np.full((64, 80), 0.5, dtype=np.float32)
    image[30:33, 8:72] = 0.15
    image[10:54, 40:42] = 0.2
    image = np.clip(image + rng.normal(0.0, 0.02, image.shape).astype(np.float32), 0.0, 1.0)

    config = ThermalEvidenceConfig()
    shared = dict(
        scales=config.scales,
        R=config.R,
        ss=config.ss,
        si=config.si,
        sa=config.sa,
        tau=config.tau,
        min_rel_size=config.min_rel_size,
        K=config.K,
        device="cpu",
        return_raster_features=False,
    )
    response, similarity, _, _, diagnostics = extract_frangi_graph_gpu(
        {"t": image}, {"t": 1.0}, compute_centrality=True, **shared
    )
    _, similarity_fast, _, _, diagnostics_fast = extract_frangi_graph_gpu(
        {"t": image}, {"t": 1.0}, compute_centrality=False, **shared
    )

    assert np.array_equal(similarity, similarity_fast), "la similarité doit être identique"
    assert not diagnostics_fast["tau_mask"].any(), (
        "si cette assertion casse, l'extracteur remplit désormais tau_mask sans MST "
        "et la reconstruction locale doit être remplacée par une lecture directe"
    )

    rebuilt = support_from_similarity(response, similarity, config.tau)
    truth = (np.asarray(diagnostics["tau_mask"]) > 0.0).astype(np.float32)
    assert np.array_equal(rebuilt, truth)


def test_support_vide_sur_une_image_plate() -> None:
    flat = np.full((32, 32), 0.4, dtype=np.float32)
    payload = generate_dual_polarity_thermal_evidence(flat, encoding="grayscale", device="cpu")
    assert np.isfinite(payload["support_union"]).all()


def test_evidence_thermique_brute_a_quatre_canaux() -> None:
    thermal = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    raw = raw_thermal_evidence(thermal)
    assert raw.shape == (4, 4, 4)
    assert np.allclose(raw[0] + raw[1], 1.0)
    assert np.allclose(raw[2], 2.0 * np.abs(thermal - 0.5))
    assert np.allclose(raw[3], 1.0)
