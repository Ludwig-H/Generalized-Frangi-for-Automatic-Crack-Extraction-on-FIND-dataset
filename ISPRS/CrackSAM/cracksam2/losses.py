"""Loss and learning-rate policy used in the CrackSAM experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    foreground = torch.sigmoid(logits)
    targets = targets.float()
    probabilities = torch.cat((1.0 - foreground, foreground), dim=1)
    one_hot_targets = torch.cat((1.0 - targets, targets), dim=1)
    # Match CrackSAM's original Dice implementation: aggregate over batch and
    # space for each class, then average background and crack losses.
    reduce_dims = (0, *range(2, logits.ndim))
    intersection = (probabilities * one_hot_targets).sum(dim=reduce_dims)
    denominator = probabilities.square().sum(
        dim=reduce_dims
    ) + one_hot_targets.square().sum(dim=reduce_dims)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def cracksam_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ce_weight: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return 0.2 binary CE + 0.8 Dice, along with both components."""

    if not 0.0 <= ce_weight <= 1.0:
        raise ValueError(f"ce_weight must be in [0,1], got {ce_weight}")
    if targets.ndim == logits.ndim - 1:
        targets = targets.unsqueeze(1)
    if logits.shape != targets.shape:
        raise ValueError(f"logits {logits.shape} and targets {targets.shape} differ")
    ce = F.binary_cross_entropy_with_logits(logits, targets.float())
    dice = binary_dice_loss(logits, targets)
    return ce_weight * ce + (1.0 - ce_weight) * dice, ce, dice


def warmup_poly_lr(
    step: int,
    total_steps: int,
    base_lr: float = 4e-4,
    warmup_steps: int = 300,
    power: float = 6.0,
) -> float:
    """Linear warmup followed by the roadmap's polynomial decay."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
    clamped_step = min(step, total_steps)
    if warmup_steps and clamped_step < warmup_steps:
        return base_lr * (clamped_step + 1) / warmup_steps
    progress = (clamped_step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * max(0.0, 1.0 - progress) ** power


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
