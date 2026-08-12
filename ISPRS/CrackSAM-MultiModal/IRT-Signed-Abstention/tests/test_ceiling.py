"""Plafond d'une correction bornée.

Le point que ces tests fixent : ``delta_max`` n'est pas un détail de
régularisation, c'est **une borne de faisabilité**. Un faux négatif confiant est
hors d'atteinte par construction, et il faut le savoir avant d'entraîner quoi que
ce soit.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_residual.ceiling import (
    bounded_oracle_logits,
    measure_ceiling,
    recommend_delta_max,
    sweep,
)


def _pair(confidence: float, size: tuple[int, int] = (32, 40)):
    """Une fissure horizontale que la baseline manque, avec la confiance donnée."""

    mask = np.zeros(size, dtype=np.float32)
    mask[15:17, 4 : size[1] - 4] = 1.0
    logits = np.full(size, -confidence, dtype=np.float32)
    return logits[None, ...], mask


def test_un_faux_negatif_confiant_est_hors_de_portee() -> None:
    """La démonstration : à ``δ_max = 4``, un logit à −6 ne peut pas être corrigé."""

    pairs = [_pair(6.0)]
    reachable = measure_ceiling(pairs, delta_max=8.0)
    unreachable = measure_ceiling(pairs, delta_max=4.0)

    assert reachable.oracle_iou == pytest.approx(1.0)
    assert unreachable.oracle_iou == pytest.approx(0.0)
    assert unreachable.unreachable_false_negative_fraction == pytest.approx(1.0)
    assert reachable.unreachable_false_negative_fraction == pytest.approx(0.0)


def test_l_oracle_borne_ne_depasse_jamais_delta_max() -> None:
    logits, mask = _pair(2.0)
    corrected = bounded_oracle_logits(logits[0], mask, 4.0)
    assert np.abs(corrected - logits[0]).max() == pytest.approx(4.0)


def test_le_plafond_croit_avec_delta_max() -> None:
    pairs = [_pair(value) for value in (1.0, 3.0, 5.0, 7.0)]
    reports = sweep(pairs, [1.0, 4.0, 8.0, 16.0])
    oracles = [report.oracle_iou for report in reports]
    assert oracles == sorted(oracles), "un plafond ne peut pas décroître quand la borne croît"
    assert oracles[0] < oracles[-1]

    unreachable = [report.unreachable_error_fraction for report in reports]
    assert unreachable == sorted(unreachable, reverse=True)


def test_la_marge_est_nulle_quand_la_baseline_est_parfaite() -> None:
    mask = np.zeros((24, 24), dtype=np.float32)
    mask[10:12, 4:20] = 1.0
    logits = np.where(mask > 0.5, 5.0, -5.0).astype(np.float32)[None, ...]
    report = measure_ceiling([(logits, mask)], delta_max=4.0)
    assert report.baseline_iou == pytest.approx(1.0)
    assert report.headroom == pytest.approx(0.0)


def test_quantiles_et_recommandation() -> None:
    pairs = [_pair(value) for value in (1.0, 3.0, 9.0)]
    report = measure_ceiling(pairs, delta_max=4.0)
    quantiles = report.logit_abs_quantiles
    assert quantiles["p50"] <= quantiles["p99"] <= quantiles["p999"]
    assert recommend_delta_max(report, coverage=0.99) == quantiles["p99"]
    with pytest.raises(ValueError, match="coverage"):
        recommend_delta_max(report, coverage=0.42)
