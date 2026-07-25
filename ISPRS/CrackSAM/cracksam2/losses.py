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


def frangi_evidence_target(
    targets: torch.Tensor,
    *,
    tolerance: int = 3,
) -> torch.Tensor:
    """Return the train-only label for Frangi support verification.

    A candidate is labelled compatible when it falls on the annotated crack or
    within a small tolerance for raster/skeleton alignment.  This target does
    not encode marginal utility over SAM: the segmentation loss must learn that
    distinction.  It is consumed only during training and is never an inference
    input or cache channel.
    """

    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    if targets.ndim != 4 or targets.shape[1] != 1:
        raise ValueError(f"targets must have shape (B,1,H,W), got {targets.shape}")
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    binary = (targets > 0.5).to(dtype=torch.float32)
    if tolerance:
        binary = F.max_pool2d(
            binary,
            kernel_size=2 * tolerance + 1,
            stride=1,
            padding=tolerance,
        )
    return binary


def masked_balanced_evidence_loss(
    evidence_logits: torch.Tensor,
    evidence_support: torch.Tensor,
    targets: torch.Tensor,
    *,
    tolerance: int = 3,
) -> dict[str, torch.Tensor]:
    """Balanced compatibility BCE evaluated only at Frangi support cells.

    Positives and negatives contribute equally within each image when both are
    present.  An empty support has a finite differentiable zero loss, which is
    required for images where Frangi abstains entirely.
    """

    if evidence_logits.ndim != 4 or evidence_logits.shape[1] != 1:
        raise ValueError("evidence_logits must have shape (B,1,H,W)")
    if evidence_support.shape != evidence_logits.shape:
        raise ValueError("evidence_support and evidence_logits must have equal shape")
    labels = frangi_evidence_target(targets, tolerance=tolerance).to(
        device=evidence_logits.device,
        dtype=evidence_logits.dtype,
    )
    if labels.shape[0] != evidence_logits.shape[0]:
        raise ValueError("evidence logits and targets must have equal batch sizes")
    if labels.shape[-2:] != evidence_logits.shape[-2:]:
        if (
            labels.shape[-2] >= evidence_logits.shape[-2]
            and labels.shape[-1] >= evidence_logits.shape[-1]
        ):
            # Match the selector's support max-pooling: a feature cell is a
            # positive candidate when any tolerated GT pixel falls inside it.
            labels = F.adaptive_max_pool2d(labels, evidence_logits.shape[-2:])
        else:
            labels = F.interpolate(
                labels, size=evidence_logits.shape[-2:], mode="nearest"
            )
    if labels.shape != evidence_logits.shape:
        raise ValueError(
            "evidence logits and segmentation targets must have equal shape"
        )
    valid = evidence_support.to(device=evidence_logits.device) > 0.5
    positive = valid & (labels > 0.5)
    negative = valid & ~positive
    loss_map = F.binary_cross_entropy_with_logits(
        evidence_logits, labels, reduction="none"
    )
    per_image: list[torch.Tensor] = []
    for index in range(evidence_logits.shape[0]):
        positive_loss = loss_map[index][positive[index]]
        negative_loss = loss_map[index][negative[index]]
        if positive_loss.numel() and negative_loss.numel():
            value = 0.5 * (positive_loss.mean() + negative_loss.mean())
        elif positive_loss.numel():
            value = positive_loss.mean()
        elif negative_loss.numel():
            value = negative_loss.mean()
        else:
            value = evidence_logits[index].sum() * 0.0
        per_image.append(value)
    valid_count = valid.float().sum()
    positive_fraction = torch.where(
        valid_count > 0,
        positive.float().sum() / valid_count.clamp_min(1.0),
        valid_count,
    )
    return {
        "loss": torch.stack(per_image).mean(),
        "positive_fraction": positive_fraction,
        "support_fraction": valid.float().mean(),
    }


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
    evidence_logits: torch.Tensor | None = None,
    evidence_support: torch.Tensor | None = None,
    evidence_weight: float = 0.0,
    evidence_target_tolerance: int = 3,
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
        ("evidence_weight", evidence_weight),
    ):
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
    if evidence_target_tolerance < 0:
        raise ValueError("evidence_target_tolerance cannot be negative")
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
    if (evidence_logits is None) != (evidence_support is None):
        raise ValueError(
            "evidence_logits and evidence_support must be provided together"
        )
    if evidence_logits is None:
        evidence = candidate_logits.sum() * 0.0
        evidence_positive_fraction = evidence.detach()
        evidence_support_fraction = evidence.detach()
    else:
        evidence_values = masked_balanced_evidence_loss(
            evidence_logits,
            evidence_support,
            targets,
            tolerance=evidence_target_tolerance,
        )
        evidence = evidence_values["loss"]
        evidence_positive_fraction = evidence_values["positive_fraction"]
        evidence_support_fraction = evidence_values["support_fraction"]
    total_per_image = (
        candidate_loss
        + topology_weight * topology
        + safety_weight * degradation
    )
    return {
        "loss": total_per_image.mean() + evidence_weight * evidence,
        "segmentation": candidate_loss.mean(),
        "ce": candidate_ce.mean(),
        "dice": candidate_dice.mean(),
        "topology": topology.mean(),
        "degradation": degradation.mean(),
        "degraded_fraction": (degradation > 0).float().mean(),
        "evidence": evidence,
        "evidence_positive_fraction": evidence_positive_fraction,
        "evidence_support_fraction": evidence_support_fraction,
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
