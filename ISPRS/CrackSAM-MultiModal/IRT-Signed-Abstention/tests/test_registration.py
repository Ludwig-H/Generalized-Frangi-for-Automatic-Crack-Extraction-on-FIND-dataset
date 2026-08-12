"""Audit de recalage : l'estimateur doit se valider lui-même.

Le premier estimateur écrit pour cette étude saturait au bord de sa fenêtre dans
63 % des cas et rendait un chiffre qui n'était que la taille de cette fenêtre.
Ces tests fixent les deux propriétés qui manquaient : détecter un décalage connu,
et **refuser de conclure** quand le contrôle est mauvais.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_residual.registration import (
    audit_pairs,
    best_contrast_shift,
    summarize,
    verdict,
)


def _scene(shift: tuple[int, int] = (0, 0), size: tuple[int, int] = (48, 64)):
    mask = np.zeros(size, dtype=bool)
    mask[20:22, 6 : size[1] - 6] = True
    field = np.full(size, 0.3, dtype=np.float32)
    shifted = np.roll(np.roll(mask, shift[0], axis=0), shift[1], axis=1)
    field[shifted] = 0.9
    return field, mask


@pytest.mark.parametrize("shift", [(0, 0), (3, 0), (0, -4), (2, 5)])
def test_retrouve_un_decalage_connu(shift) -> None:
    field, mask = _scene(shift)
    estimate = best_contrast_shift(field, mask, radius=8)
    # Le champ est décalé de ``shift`` ; le ramener demande le décalage opposé.
    assert (estimate.dy, estimate.dx) == (-shift[0], -shift[1])
    assert not estimate.saturated
    assert estimate.contrast > 0.3


def test_signale_la_saturation() -> None:
    field, mask = _scene((0, 9))
    estimate = best_contrast_shift(field, mask, radius=4)
    assert estimate.saturated, "un décalage hors fenêtre doit être signalé"


def test_masque_vide_ou_plein_rend_zero() -> None:
    field = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    assert best_contrast_shift(field, np.zeros((16, 16), bool)).contrast == 0.0
    assert best_contrast_shift(field, np.ones((16, 16), bool)).contrast == 0.0


def test_verdict_refuse_de_conclure_si_le_controle_est_mauvais() -> None:
    aligned = summarize("thermique", [best_contrast_shift(*_scene((0, 0)))])
    drifting = summarize("rgb (contrôle)", [best_contrast_shift(*_scene((0, 6)))])
    result = verdict(aligned, drifting)
    assert result["decision"] == "non concluant"
    assert result["control_is_clean"] is False


def test_verdict_accepte_et_rejette_selon_le_seuil() -> None:
    control = summarize("rgb (contrôle)", [best_contrast_shift(*_scene((0, 0)))])
    close = summarize("thermique", [best_contrast_shift(*_scene((1, 1)))])
    far = summarize("thermique", [best_contrast_shift(*_scene((0, 7)))])
    assert verdict(close, control, threshold_px=3.0)["decision"] == "accepté"
    assert verdict(far, control, threshold_px=3.0)["decision"] == "REJETÉ"


def test_audit_complet_sur_des_paires() -> None:
    pairs = []
    for offset in range(4):
        rgb, mask = _scene((0, 0))
        thermal, _ = _scene((0, 5))
        pairs.append((rgb, thermal, mask))
    result = audit_pairs(pairs, radius=8)
    assert result["control_is_clean"] is True
    assert result["decision"] == "REJETÉ"
    assert result["thermal"]["median_shift_px"] == pytest.approx(5.0)
    assert result["control"]["fraction_exact"] == pytest.approx(1.0)
