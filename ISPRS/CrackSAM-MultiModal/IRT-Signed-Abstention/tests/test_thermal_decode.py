"""Décodage thermique — §13.6.

Le test central n'est pas « le décodage marche » mais « la conversion standard en
niveaux de gris est **fausse** sur une palette JET, et voici de combien ».
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_residual.thermal_decode import (
    CHROMA_THRESHOLD,
    ThermalDecodeError,
    colormap_palette,
    decode_thermal,
    naive_opencv_grayscale,
    robust_normalize,
)


def _jet_image(scalar: np.ndarray) -> np.ndarray:
    colors, scalars = colormap_palette("jet")
    indices = np.clip(np.searchsorted(scalars, scalar.ravel()), 0, len(scalars) - 1)
    return colors[indices].reshape(*scalar.shape, 3).astype(np.float32)


def test_recuperation_monotone_des_256_indices() -> None:
    """Un dégradé de 256 niveaux encodé en JET se redécode de façon croissante."""

    reference = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :]
    decoding = decode_thermal(_jet_image(reference), encoding="jet")
    recovered = decoding.scalar[0]
    assert np.all(np.diff(recovered) >= -1e-6), "le décodage n'est pas monotone"
    assert np.abs(recovered - reference[0]).max() < 0.01


def test_erreur_de_palette_faible_sur_une_image_jet() -> None:
    rng = np.random.default_rng(0)
    scalar = rng.random((32, 40)).astype(np.float32)
    decoding = decode_thermal(_jet_image(scalar), encoding="jet")
    assert decoding.palette_error_mean < 1e-3
    assert np.abs(decoding.scalar - scalar).mean() < 0.01


def test_le_gris_naif_est_non_monotone_et_diverge() -> None:
    """La démonstration du piège : le vert médian bat le rouge maximal."""

    reference = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :]
    image = _jet_image(reference)
    naive = naive_opencv_grayscale(image)[0]
    correct = decode_thermal(image, encoding="jet").scalar[0]

    assert np.any(np.diff(naive) < -1e-3), "le gris naïf devrait être non monotone"
    assert naive[128] > naive[-1], "le vert médian devrait dépasser le rouge maximal"
    assert np.abs(naive - correct).mean() > 0.1, "les deux décodages devraient diverger nettement"


def test_image_monochrome_stable() -> None:
    rng = np.random.default_rng(1)
    gray = rng.random((24, 24)).astype(np.float32)
    image = np.repeat(gray[..., None], 3, axis=-1)
    decoding = decode_thermal(image, encoding="auto")
    assert decoding.encoding == "grayscale"
    assert np.abs(decoding.scalar - gray).max() < 1e-6
    assert decoding.chroma_p99 < CHROMA_THRESHOLD


def test_auto_detecte_le_jet() -> None:
    scalar = np.linspace(0.0, 1.0, 40, dtype=np.float32)[None, :].repeat(20, axis=0)
    decoding = decode_thermal(_jet_image(scalar), encoding="auto")
    assert decoding.encoding == "jet"
    assert decoding.requested_encoding == "auto"


def test_normalisation_robuste_borne_et_survit_au_constant() -> None:
    values = np.concatenate([np.full(100, 0.5, np.float32), np.array([-5.0, 12.0], np.float32)])
    normalized, statistics = robust_normalize(values)
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    assert statistics["p01"] <= statistics["p99"]

    constant, _ = robust_normalize(np.full((8, 8), 0.3, np.float32))
    assert np.isfinite(constant).all()
    assert constant.max() <= 1.0


def test_encodage_inconnu_rejete() -> None:
    with pytest.raises(ThermalDecodeError, match="encodage inconnu"):
        decode_thermal(np.zeros((4, 4, 3), np.float32), encoding="viridis")


def test_decodage_depuis_un_fichier(fake_dataset) -> None:
    path = sorted((fake_dataset / "02-Infrared images").glob("*.png"))[0]
    decoding = decode_thermal(path, encoding="auto")
    assert decoding.encoding == "jet"
    assert decoding.normalized.shape == (48, 64)
    assert np.isfinite(decoding.normalized).all()
