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
