from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from cracksam2.losses import (  # noqa: E402
    binary_dice_loss_per_image,
    cracksam_loss_per_image,
    frangi_evidence_target,
    masked_balanced_evidence_loss,
    residual_training_loss,
    soft_cldice_loss_per_image,
    soft_skeletonize,
)


def test_per_image_loss_cannot_hide_one_bad_image() -> None:
    targets = torch.zeros(2, 1, 8, 8)
    logits = torch.full_like(targets, -8.0)
    logits[1] = 8.0

    combined, ce, dice = cracksam_loss_per_image(logits, targets)

    assert combined.shape == ce.shape == dice.shape == (2,)
    # The historical two-class Dice assigns a small finite penalty to an
    # almost-empty foreground prediction; the failed image must remain much
    # worse and visible as its own value.
    assert combined[0] < 0.2
    assert combined[1] > 1.0
    assert binary_dice_loss_per_image(logits, targets).shape == (2,)


def test_soft_skeleton_and_cldice_have_one_value_per_image() -> None:
    targets = torch.zeros(2, 1, 9, 9)
    targets[:, :, 4, 1:8] = 1.0
    logits = torch.where(targets > 0, 8.0, -8.0)

    skeleton = soft_skeletonize(targets, iterations=5)
    loss = soft_cldice_loss_per_image(logits, targets, iterations=5)

    assert skeleton.shape == targets.shape
    assert torch.count_nonzero(skeleton) > 0
    assert loss.shape == (2,)
    assert torch.all(loss < 0.01)


def test_equal_candidate_has_no_safety_penalty_at_zero_margin() -> None:
    targets = torch.randint(0, 2, (3, 1, 8, 8)).float()
    baseline = torch.randn_like(targets)
    result = residual_training_loss(
        baseline.clone().requires_grad_(True),
        baseline,
        targets,
        topology_weight=0.0,
        safety_weight=1.0,
        safety_margin=0.0,
        skeleton_iterations=3,
    )

    assert result["degradation"].item() == pytest.approx(0.0)
    assert result["degraded_fraction"].item() == pytest.approx(0.0)
    assert result["topology"].item() == pytest.approx(0.0)


def test_worse_candidate_is_penalized_and_receives_gradients() -> None:
    targets = torch.zeros(2, 1, 8, 8)
    baseline = torch.full_like(targets, -8.0)
    candidate = torch.full_like(targets, 2.0, requires_grad=True)
    result = residual_training_loss(
        candidate,
        baseline,
        targets,
        topology_weight=0.1,
        safety_weight=2.0,
        skeleton_iterations=3,
    )

    assert result["degradation"] > 0
    assert result["degraded_fraction"].item() == pytest.approx(1.0)
    result["loss"].backward()
    assert candidate.grad is not None
    assert torch.count_nonzero(candidate.grad) > 0


def test_residual_loss_rejects_negative_weights() -> None:
    tensor = torch.zeros(1, 1, 4, 4)
    with pytest.raises(ValueError, match="topology_weight"):
        residual_training_loss(tensor, tensor, tensor, topology_weight=-0.1)


def test_evidence_target_uses_a_bounded_spatial_tolerance() -> None:
    target = torch.zeros(1, 1, 9, 9)
    target[:, :, 4, 4] = 1.0

    exact = frangi_evidence_target(target, tolerance=0)
    tolerant = frangi_evidence_target(target, tolerance=2)

    assert torch.count_nonzero(exact) == 1
    assert torch.count_nonzero(tolerant) == 25
    assert tolerant[0, 0, 2, 2] == 1
    assert tolerant[0, 0, 1, 1] == 0


def test_evidence_loss_balances_positive_and_negative_support_per_image() -> None:
    logits = torch.zeros(1, 1, 5, 5, requires_grad=True)
    support = torch.zeros_like(logits)
    support[:, :, 2, 2] = 1.0
    support[:, :, 0, :] = 1.0
    target = torch.zeros_like(logits)
    target[:, :, 2, 2] = 1.0

    result = masked_balanced_evidence_loss(
        logits, support, target, tolerance=0
    )

    assert result["loss"].item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert result["positive_fraction"].item() == pytest.approx(1.0 / 6.0)
    result["loss"].backward()
    assert logits.grad is not None
    # The one positive and all five negatives each receive half of the class-
    # balanced loss, rather than the positive being overwhelmed five-to-one.
    positive_gradient = abs(float(logits.grad[0, 0, 2, 2]))
    negative_gradient_sum = float(logits.grad[0, 0, 0].abs().sum())
    assert positive_gradient == pytest.approx(negative_gradient_sum)


def test_empty_evidence_support_has_a_finite_differentiable_zero_loss() -> None:
    logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    result = masked_balanced_evidence_loss(
        logits,
        torch.zeros_like(logits),
        torch.zeros_like(logits),
    )

    assert result["loss"].item() == 0.0
    assert result["support_fraction"].item() == 0.0
    result["loss"].backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_evidence_target_is_aggregated_on_the_thresholded_feature_grid() -> None:
    logits = torch.full((1, 1, 4, 4), -2.0, requires_grad=True)
    support = torch.zeros_like(logits)
    support[:, :, 1, 1] = 1.0
    target = torch.zeros(1, 1, 8, 8)
    # A one-pixel annotation near the edge of the corresponding 2x2 output
    # block must supervise the single selector cell, not four mixed logits.
    target[:, :, 3, 3] = 1.0

    result = masked_balanced_evidence_loss(
        logits, support, target, tolerance=0
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(-2.0), torch.tensor(1.0)
    )

    assert result["loss"].item() == pytest.approx(expected.item())
    assert result["positive_fraction"].item() == 1.0


def test_residual_loss_adds_weighted_evidence_supervision() -> None:
    target = torch.zeros(1, 1, 4, 4)
    candidate = torch.zeros_like(target, requires_grad=True)
    evidence_logits = torch.zeros_like(target, requires_grad=True)
    support = torch.ones_like(target)

    without = residual_training_loss(
        candidate,
        torch.zeros_like(target),
        target,
        topology_weight=0.0,
        safety_weight=0.0,
    )
    with_evidence = residual_training_loss(
        candidate,
        torch.zeros_like(target),
        target,
        topology_weight=0.0,
        safety_weight=0.0,
        evidence_logits=evidence_logits,
        evidence_support=support,
        evidence_weight=0.25,
        evidence_target_tolerance=0,
    )

    assert with_evidence["loss"] > without["loss"]
    assert with_evidence["evidence"].item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item()
    )
