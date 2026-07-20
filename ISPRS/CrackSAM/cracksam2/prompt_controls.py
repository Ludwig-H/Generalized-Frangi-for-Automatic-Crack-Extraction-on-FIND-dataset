"""Deterministic inference controls for the historical Frangi mask prompt.

These utilities are deliberately independent from SAM 2.  They make the
causal prompt ablation reproducible across batch sizes and interrupted Spot
sessions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Literal

import torch

from .data import FRANGI_BACKGROUND_LOGIT


PromptCondition = Literal[
    "none",
    "frangi",
    "zero_logit",
    "permuted",
    "shifted",
]

PROMPT_CONDITIONS: tuple[PromptCondition, ...] = (
    "none",
    "frangi",
    "zero_logit",
    "permuted",
    "shifted",
)
PROMPT_CACHE_CONDITIONS = frozenset({"frangi", "permuted", "shifted"})
PROMPT_PERMUTATION_ALGORITHM = "sha256-sort-circular-successor-v1"


def default_prompt_condition(checkpoint_variant: str) -> PromptCondition:
    """Preserve the historical evaluator behaviour when no condition is given."""
    if checkpoint_variant == "baseline":
        return "none"
    if checkpoint_variant == "frangi":
        return "frangi"
    raise ValueError(
        "checkpoint_variant must be 'baseline' or 'frangi', "
        f"got {checkpoint_variant!r}"
    )


def condition_needs_cache(condition: PromptCondition | str) -> bool:
    """Return whether a condition consumes a sample-specific Frangi cache."""
    if condition not in PROMPT_CONDITIONS:
        raise ValueError(f"Unknown prompt condition: {condition!r}")
    return condition in PROMPT_CACHE_CONDITIONS


def deterministic_prompt_name_map(
    sample_names: Sequence[str], *, seed: int
) -> dict[str, str]:
    """Build a deterministic derangement from targets to prompt sources.

    Names are sorted by a seeded SHA-256 key and each target receives the next
    name in that circular order.  Consequently the mapping is independent of
    input order, has no fixed point, and remains stable after an interrupted
    evaluation resumes.
    """
    names = list(sample_names)
    if len(names) < 2:
        raise ValueError("At least two distinct samples are required for permutation")
    if len(names) != len(set(names)):
        raise ValueError("Sample names must be unique for prompt permutation")

    seed_bytes = str(int(seed)).encode("ascii") + b"\0"
    ordered = sorted(
        names,
        key=lambda name: hashlib.sha256(seed_bytes + name.encode("utf-8")).digest(),
    )
    sources = ordered[1:] + ordered[:1]
    mapping = dict(zip(ordered, sources, strict=True))
    if any(target == source for target, source in mapping.items()):
        raise AssertionError("Prompt permutation unexpectedly contains a fixed point")
    return mapping


def prompt_name_map_sha256(mapping: dict[str, str]) -> str:
    """Hash a mapping independently of dictionary insertion order."""
    digest = hashlib.sha256()
    for target, source in sorted(mapping.items()):
        digest.update(target.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def shift_prompt_without_wrap(
    prompt: torch.Tensor,
    *,
    dy: int,
    dx: int,
    fill_value: float = FRANGI_BACKGROUND_LOGIT,
) -> torch.Tensor:
    """Translate the final two dimensions without circular wraparound."""
    if prompt.ndim < 2:
        raise ValueError("prompt must have at least two dimensions")
    height, width = (int(prompt.shape[-2]), int(prompt.shape[-1]))
    dy, dx = int(dy), int(dx)
    if dy == 0 and dx == 0:
        raise ValueError("Prompt shift cannot be (0, 0)")
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError(
            f"Prompt shift {(dy, dx)} must be smaller than {(height, width)}"
        )

    shifted = torch.full_like(prompt, float(fill_value))
    source_y = slice(max(0, -dy), min(height, height - dy))
    target_y = slice(max(0, dy), min(height, height + dy))
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_x = slice(max(0, dx), min(width, width + dx))
    shifted[..., target_y, target_x] = prompt[..., source_y, source_x]
    return shifted


def prepare_mask_input(
    condition: PromptCondition | str,
    *,
    batch_size: int,
    device: torch.device,
    cached_prompt: torch.Tensor | None,
    prompt_size: tuple[int, int] = (256, 256),
    shift: tuple[int, int] = (32, 32),
) -> torch.Tensor | None:
    """Materialize the exact mask input for one causal condition."""
    if condition not in PROMPT_CONDITIONS:
        raise ValueError(f"Unknown prompt condition: {condition!r}")
    if condition == "none":
        return None
    if condition == "zero_logit":
        return torch.zeros(
            (batch_size, 1, *prompt_size), dtype=torch.float32, device=device
        )
    if cached_prompt is None:
        raise ValueError(f"Condition {condition!r} requires a cached Frangi prompt")
    prompt = cached_prompt.to(device, non_blocking=True)
    if condition in ("frangi", "permuted"):
        return prompt
    if condition == "shifted":
        return shift_prompt_without_wrap(prompt, dy=shift[0], dx=shift[1])
    raise AssertionError(f"Unhandled prompt condition: {condition}")


__all__ = [
    "PROMPT_CACHE_CONDITIONS",
    "PROMPT_CONDITIONS",
    "PROMPT_PERMUTATION_ALGORITHM",
    "PromptCondition",
    "condition_needs_cache",
    "default_prompt_condition",
    "deterministic_prompt_name_map",
    "prepare_mask_input",
    "prompt_name_map_sha256",
    "shift_prompt_without_wrap",
]
