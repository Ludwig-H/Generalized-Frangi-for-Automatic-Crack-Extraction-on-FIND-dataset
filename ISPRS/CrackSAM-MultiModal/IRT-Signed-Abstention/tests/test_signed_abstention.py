"""Sens de la correction, bornage, budget paramétrique — §13.4, §13.5."""

from __future__ import annotations

import pytest
import torch

from thermal_residual.constants import (
    ACTION_ABSTAIN,
    ACTION_REINFORCE,
    ACTION_SUPPRESS,
    HEAD_POSITIVE_ONLY,
    HEAD_SIGNED,
    HEAD_SIGNED_ABSTENTION,
)
from thermal_residual.model import ThermalSignedAbstentionAdapter, binary_entropy


def _inputs(batch: int = 2, size: tuple[int, int] = (16, 20)):
    torch.manual_seed(11)
    logits = torch.randn(batch, 1, *size) * 3.0
    evidence = torch.rand(batch, 4, *size)
    present = torch.ones(batch, dtype=torch.bool)
    return logits, evidence, present


def _force_action(model: ThermalSignedAbstentionAdapter, action: int) -> None:
    """Fixe le biais pour rendre une action quasi certaine, poids toujours nuls."""

    with torch.no_grad():
        model.action_head.weight.zero_()
        bias = torch.full_like(model.action_head.bias, -12.0)
        bias[action] = 12.0
        model.action_head.bias.copy_(bias)


@pytest.mark.parametrize(
    "action,sign",
    [(ACTION_REINFORCE, +1), (ACTION_SUPPRESS, -1), (ACTION_ABSTAIN, 0)],
)
def test_sens_de_la_correction(action: int, sign: int) -> None:
    model = ThermalSignedAbstentionAdapter().eval()
    _force_action(model, action)
    logits, evidence, present = _inputs()
    residual = model(logits, evidence, present)["residual_logits"]
    if sign > 0:
        assert residual.min() > 3.9
    elif sign < 0:
        assert residual.max() < -3.9
    else:
        assert residual.abs().max() < 1e-4


def test_bornage_par_delta_max() -> None:
    for delta_max in (1.0, 4.0, 10.0):
        model = ThermalSignedAbstentionAdapter(delta_max=delta_max).eval()
        _force_action(model, ACTION_REINFORCE)
        logits, evidence, present = _inputs()
        residual = model(logits, evidence, present)["residual_logits"]
        assert residual.abs().max() <= delta_max + 1e-6


def test_budget_parametrique_sous_cent_mille() -> None:
    for head in (HEAD_SIGNED_ABSTENTION, HEAD_SIGNED, HEAD_POSITIVE_ONLY):
        model = ThermalSignedAbstentionAdapter(head=head)
        assert model.trainable_parameters() < 100_000
        assert model.trainable_parameters() > 10_000


def test_les_trois_tetes_ont_une_capacite_comparable() -> None:
    counts = {
        head: ThermalSignedAbstentionAdapter(head=head).trainable_parameters()
        for head in (HEAD_SIGNED_ABSTENTION, HEAD_SIGNED, HEAD_POSITIVE_ONLY)
    }
    spread = max(counts.values()) - min(counts.values())
    assert spread <= 100, f"écart de capacité trop grand entre les têtes : {counts}"


def test_probabilites_normalisees_et_actions_exclusives() -> None:
    model = ThermalSignedAbstentionAdapter().eval()
    logits, evidence, present = _inputs()
    outputs = model(logits, evidence, present)
    total = (
        outputs["reinforce_probability"]
        + outputs["suppress_probability"]
        + outputs["abstain_probability"]
    )
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    assert outputs["hard_action"].min() >= 0 and outputs["hard_action"].max() <= 2


def test_portee_evidence_union_restreint_la_correction() -> None:
    model = ThermalSignedAbstentionAdapter(correction_scope="evidence_union").eval()
    _force_action(model, ACTION_REINFORCE)
    logits = torch.full((1, 1, 16, 16), -8.0)
    evidence = torch.zeros(1, 4, 16, 16)
    evidence[0, 3, 8, 8] = 1.0  # support ponctuel
    present = torch.ones(1, dtype=torch.bool)
    outputs = model(logits, evidence, present)
    residual = outputs["residual_logits"][0, 0]
    assert residual[8, 8] > 3.9
    assert residual[0, 0].abs() < 1e-6, "hors de Ω la correction doit être exactement nulle"


def test_entropie_binaire_maximale_en_un_demi() -> None:
    probability = torch.tensor([0.0, 0.5, 1.0])
    entropy = binary_entropy(probability)
    assert entropy[1] > entropy[0] and entropy[1] > entropy[2]
    assert torch.isfinite(entropy).all()


def test_formes_et_types_invalides_rejetes() -> None:
    model = ThermalSignedAbstentionAdapter()
    logits, evidence, present = _inputs()
    with pytest.raises(ValueError, match=r"\(B,1,H,W\)"):
        model(logits[:, 0], evidence, present)
    with pytest.raises(ValueError, match="thermal_evidence"):
        model(logits, evidence[:, :2], present)
    with pytest.raises(TypeError, match="torch.bool"):
        model(logits, evidence, present.float())
