"""Loss and learning-rate policy used in the CrackSAM experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _matching_targets(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == logits.ndim - 1:
        targets = targets.unsqueeze(1)
    if logits.shape != targets.shape:
        raise ValueError(f"logits {logits.shape} and targets {targets.shape} differ")
    return targets.float()


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


def binary_dice_loss_per_image(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Return one background/foreground Dice loss per image.

    Unlike :func:`binary_dice_loss`, no image in a batch can compensate for a
    failure on another image.  This is the safer definition for the residual
    correction and its per-image degradation penalty.
    """
    targets = _matching_targets(logits, targets)
    foreground = torch.sigmoid(logits)
    probabilities = torch.cat((1.0 - foreground, foreground), dim=1)
    one_hot_targets = torch.cat((1.0 - targets, targets), dim=1)
    reduce_dims = tuple(range(2, logits.ndim))
    intersection = (probabilities * one_hot_targets).sum(dim=reduce_dims)
    denominator = probabilities.square().sum(
        dim=reduce_dims
    ) + one_hot_targets.square().sum(dim=reduce_dims)
    per_class = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    return per_class.mean(dim=1)


def cracksam_loss_per_image(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ce_weight: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CrackSAM loss, CE and Dice as vectors with one value per image."""
    if not 0.0 <= ce_weight <= 1.0:
        raise ValueError(f"ce_weight must be in [0,1], got {ce_weight}")
    targets = _matching_targets(logits, targets)
    ce_map = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    ce = ce_map.flatten(1).mean(dim=1)
    dice = binary_dice_loss_per_image(logits, targets)
    return ce_weight * ce + (1.0 - ce_weight) * dice, ce, dice


def _soft_erode(mask: torch.Tensor) -> torch.Tensor:
    eroded_vertical = -F.max_pool2d(-mask, (3, 1), stride=1, padding=(1, 0))
    eroded_horizontal = -F.max_pool2d(-mask, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(eroded_vertical, eroded_horizontal)


def _soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(mask, 3, stride=1, padding=1)


def _soft_open(mask: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(mask))


def soft_skeletonize(mask: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """Differentiable morphological skeleton approximation used by clDice."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape (B,1,H,W), got {mask.shape}")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    opened = _soft_open(mask)
    skeleton = F.relu(mask - opened)
    current = mask
    for _ in range(iterations - 1):
        current = _soft_erode(current)
        opened = _soft_open(current)
        delta = F.relu(current - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss_per_image(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iterations: int = 20,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Return one topology-aware clDice loss per image."""
    targets = _matching_targets(logits, targets)
    probabilities = torch.sigmoid(logits)
    predicted_skeleton = soft_skeletonize(probabilities, iterations=iterations)
    target_skeleton = soft_skeletonize(targets, iterations=iterations)
    reduce_dims = tuple(range(1, logits.ndim))
    topology_precision = (
        (predicted_skeleton * targets).sum(dim=reduce_dims) + smooth
    ) / (predicted_skeleton.sum(dim=reduce_dims) + smooth)
    topology_sensitivity = (
        (target_skeleton * probabilities).sum(dim=reduce_dims) + smooth
    ) / (target_skeleton.sum(dim=reduce_dims) + smooth)
    return 1.0 - (
        2.0
        * topology_precision
        * topology_sensitivity
        / (topology_precision + topology_sensitivity + smooth)
    )


def residual_training_loss(
    candidate_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ce_weight: float = 0.2,
    topology_weight: float = 0.1,
    safety_weight: float = 1.0,
    safety_margin: float = 0.0,
    skeleton_iterations: int = 20,
) -> dict[str, torch.Tensor]:
    """Loss for the residual candidate with an image-level safety penalty.

    The baseline is detached.  Whenever the candidate segmentation loss is
    worse by more than ``-safety_margin``, the extra error is penalized for that
    image instead of being hidden by easier images in the same batch.
    """
    if candidate_logits.shape != baseline_logits.shape:
        raise ValueError("candidate and baseline logits must have equal shape")
    for name, value in (
        ("topology_weight", topology_weight),
        ("safety_weight", safety_weight),
        ("safety_margin", safety_margin),
    ):
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
    candidate_loss, candidate_ce, candidate_dice = cracksam_loss_per_image(
        candidate_logits, targets, ce_weight=ce_weight
    )
    with torch.no_grad():
        baseline_loss, _, _ = cracksam_loss_per_image(
            baseline_logits.detach(), targets, ce_weight=ce_weight
        )
    if topology_weight > 0.0:
        topology = soft_cldice_loss_per_image(
            candidate_logits,
            targets,
            iterations=skeleton_iterations,
        )
    else:
        # Full-resolution soft skeletonization is deliberately skipped when
        # disabled; at 448 px it is materially more expensive than BCE/Dice.
        topology = torch.zeros_like(candidate_loss)
    degradation = F.relu(candidate_loss - baseline_loss + safety_margin)
    total_per_image = (
        candidate_loss
        + topology_weight * topology
        + safety_weight * degradation
    )
    return {
        "loss": total_per_image.mean(),
        "segmentation": candidate_loss.mean(),
        "ce": candidate_ce.mean(),
        "dice": candidate_dice.mean(),
        "topology": topology.mean(),
        "degradation": degradation.mean(),
        "degraded_fraction": (degradation > 0).float().mean(),
    }


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
