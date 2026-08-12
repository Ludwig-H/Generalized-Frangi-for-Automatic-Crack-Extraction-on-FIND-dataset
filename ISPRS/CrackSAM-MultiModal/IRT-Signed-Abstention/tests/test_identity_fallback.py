"""Identité bit-à-bit et gradients vivants — §13.1, §13.2, §13.3.

Deux propriétés qui doivent tenir **ensemble**, et dont le couplage a déjà coûté
une itération à ce projet : la sortie doit valoir exactement la baseline à
l'initialisation, sans que cela n'annule les gradients de la branche auxiliaire.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from thermal_residual.constants import HEAD_POSITIVE_ONLY, HEAD_SIGNED, HEAD_SIGNED_ABSTENTION
from thermal_residual.model import NEAR_IDENTITY_BIAS_POSITIVE, ThermalSignedAbstentionAdapter

BATCH, HEIGHT, WIDTH = 3, 16, 24


def _inputs():
    torch.manual_seed(5)
    logits = torch.randn(BATCH, 1, HEIGHT, WIDTH) * 4.0
    evidence = torch.rand(BATCH, 4, HEIGHT, WIDTH)
    return logits, evidence


@pytest.mark.parametrize("head", [HEAD_SIGNED_ABSTENTION, HEAD_SIGNED])
def test_identite_bit_a_bit_a_l_initialisation(head: str) -> None:
    model = ThermalSignedAbstentionAdapter(head=head).eval()
    logits, evidence = _inputs()
    outputs = model(logits, evidence, torch.ones(BATCH, dtype=torch.bool))
    assert torch.equal(outputs["logits"], logits)
    assert int(torch.count_nonzero(outputs["residual_logits"])) == 0


def test_positive_only_est_proche_mais_pas_exact() -> None:
    """L'impossibilité est mathématique, pas un défaut : on la mesure au lieu de la nier."""

    model = ThermalSignedAbstentionAdapter(head=HEAD_POSITIVE_ONLY, delta_max=4.0).eval()
    logits, evidence = _inputs()
    residual = model(logits, evidence, torch.ones(BATCH, dtype=torch.bool))["residual_logits"]
    bound = 4.0 * torch.sigmoid(torch.tensor(NEAR_IDENTITY_BIAS_POSITIVE)).item()
    assert residual.abs().max() <= bound + 1e-6
    assert bound < 2e-3


@pytest.mark.parametrize("head", [HEAD_SIGNED_ABSTENTION, HEAD_SIGNED, HEAD_POSITIVE_ONLY])
def test_repli_exact_sans_thermique(head: str) -> None:
    model = ThermalSignedAbstentionAdapter(head=head).eval()
    logits, evidence = _inputs()
    outputs = model(logits, evidence, torch.zeros(BATCH, dtype=torch.bool))
    assert torch.equal(outputs["logits"], logits)
    assert int(torch.count_nonzero(outputs["residual_logits"])) == 0


def test_repli_par_image_dans_un_lot_mixte() -> None:
    model = ThermalSignedAbstentionAdapter().eval()
    with torch.no_grad():
        model.action_head.bias.copy_(torch.tensor([6.0, -6.0, -6.0]))
    logits, evidence = _inputs()
    present = torch.tensor([True, False, True])
    residual = model(logits, evidence, present)["residual_logits"]
    assert residual[0].abs().max() > 1.0
    assert int(torch.count_nonzero(residual[1])) == 0
    assert residual[2].abs().max() > 1.0


def test_gradients_non_nuls_a_l_initialisation() -> None:
    model = ThermalSignedAbstentionAdapter().train()
    logits, evidence = _inputs()
    targets = (torch.rand_like(logits) > 0.5).float()
    outputs = model(logits, evidence, torch.ones(BATCH, dtype=torch.bool))
    F.binary_cross_entropy_with_logits(outputs["logits"], targets).backward()

    assert model.action_head.weight.grad is not None
    assert int(torch.count_nonzero(model.action_head.weight.grad)) > 0


def test_l_encodeur_recoit_du_gradient_des_le_second_pas() -> None:
    """Conséquence assumée de ``weight = 0`` : un pas de retard, pas un gel.

    Au pas 0 la dérivée par rapport aux features vaut ``Wᵀ = 0``, donc l'encodeur
    ne reçoit rien ; mais le gradient de ``W`` lui-même est non nul, donc dès le
    pas 1 l'encodeur apprend. C'est la différence avec le couplage mort de
    GeoLoRA, où les deux facteurs étaient nuls et le restaient.
    """

    model = ThermalSignedAbstentionAdapter().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logits, evidence = _inputs()
    targets = (torch.rand_like(logits) > 0.5).float()

    outputs = model(logits, evidence, torch.ones(BATCH, dtype=torch.bool))
    F.binary_cross_entropy_with_logits(outputs["logits"], targets).backward()
    assert int(torch.count_nonzero(model.encoder[0].weight.grad)) == 0
    optimizer.step()
    model.zero_grad(set_to_none=True)

    outputs = model(logits, evidence, torch.ones(BATCH, dtype=torch.bool))
    F.binary_cross_entropy_with_logits(outputs["logits"], targets).backward()
    assert int(torch.count_nonzero(model.encoder[0].weight.grad)) > 0


def test_repli_reste_exact_apres_entrainement() -> None:
    """La garantie doit survivre à l'optimisation, pas seulement à l'initialisation."""

    model = ThermalSignedAbstentionAdapter().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    logits, evidence = _inputs()
    targets = (torch.rand_like(logits) > 0.5).float()
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(logits, evidence, torch.ones(BATCH, dtype=torch.bool))
        F.binary_cross_entropy_with_logits(outputs["logits"], targets).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        trained = model(logits, evidence, torch.ones(BATCH, dtype=torch.bool))
        absent = model(logits, evidence, torch.zeros(BATCH, dtype=torch.bool))
    assert trained["residual_logits"].abs().max() > 1e-3, "le modèle n'a rien appris : test vide"
    assert torch.equal(absent["logits"], logits)


def test_la_baseline_ne_recoit_aucun_gradient() -> None:
    model = ThermalSignedAbstentionAdapter().train()
    logits, evidence = _inputs()
    logits.requires_grad_(True)
    targets = (torch.rand_like(logits) > 0.5).float()
    outputs = model(logits, evidence, torch.ones(BATCH, dtype=torch.bool))
    F.binary_cross_entropy_with_logits(outputs["logits"], targets).backward()
    assert logits.grad is None, "les logits baseline doivent être détachés"
