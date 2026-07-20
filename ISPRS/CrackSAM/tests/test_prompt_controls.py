from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from cracksam2.data import FRANGI_BACKGROUND_LOGIT  # noqa: E402
from cracksam2.prompt_controls import (  # noqa: E402
    default_prompt_condition,
    deterministic_prompt_name_map,
    prepare_mask_input,
    prompt_name_map_sha256,
    shift_prompt_without_wrap,
)


def test_default_condition_only_preserves_historical_cli_behaviour() -> None:
    assert default_prompt_condition("baseline") == "none"
    assert default_prompt_condition("frangi") == "frangi"
    with pytest.raises(ValueError, match="checkpoint_variant"):
        default_prompt_condition("other")


def test_prompt_permutation_is_order_independent_deterministic_derangement() -> None:
    names = ["a.png", "b.png", "c.png", "d.png"]
    first = deterministic_prompt_name_map(names, seed=3407)
    repeated = deterministic_prompt_name_map(list(reversed(names)), seed=3407)

    assert first == repeated
    assert set(first) == set(names)
    assert set(first.values()) == set(names)
    assert all(target != source for target, source in first.items())
    assert prompt_name_map_sha256(first) == prompt_name_map_sha256(repeated)
    assert deterministic_prompt_name_map(names, seed=99) != first


def test_prompt_permutation_rejects_an_invalid_split() -> None:
    with pytest.raises(ValueError, match="two distinct"):
        deterministic_prompt_name_map(["only.png"], seed=1)
    with pytest.raises(ValueError, match="unique"):
        deterministic_prompt_name_map(["same.png", "same.png"], seed=1)


def test_shift_never_wraps_and_uses_frangi_background() -> None:
    prompt = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
    shifted = shift_prompt_without_wrap(prompt, dy=1, dx=-1)

    expected = torch.full_like(prompt, FRANGI_BACKGROUND_LOGIT)
    expected[..., 1:, :3] = prompt[..., :3, 1:]
    torch.testing.assert_close(shifted, expected)
    assert shifted[..., 0, -1] == pytest.approx(FRANGI_BACKGROUND_LOGIT)
    assert not torch.any(shifted == prompt[..., -1, 0])


def test_none_and_zero_logit_are_distinct_mask_encoder_inputs() -> None:
    device = torch.device("cpu")
    assert (
        prepare_mask_input(
            "none", batch_size=2, device=device, cached_prompt=None
        )
        is None
    )
    zero = prepare_mask_input(
        "zero_logit", batch_size=2, device=device, cached_prompt=None
    )
    assert zero is not None
    assert zero.shape == (2, 1, 256, 256)
    assert torch.count_nonzero(zero) == 0


def test_cached_conditions_use_the_given_tensor_and_shift_is_non_circular() -> None:
    cached = torch.ones((2, 1, 6, 7), dtype=torch.float32)
    frangi = prepare_mask_input(
        "frangi", batch_size=2, device=torch.device("cpu"), cached_prompt=cached
    )
    permuted = prepare_mask_input(
        "permuted", batch_size=2, device=torch.device("cpu"), cached_prompt=cached
    )
    shifted = prepare_mask_input(
        "shifted",
        batch_size=2,
        device=torch.device("cpu"),
        cached_prompt=cached,
        shift=(1, 2),
    )

    torch.testing.assert_close(frangi, cached)
    torch.testing.assert_close(permuted, cached)
    assert shifted is not None
    assert torch.all(shifted[..., 0, :] == FRANGI_BACKGROUND_LOGIT)
    assert torch.all(shifted[..., :, :2] == FRANGI_BACKGROUND_LOGIT)
