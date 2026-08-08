"""Garanties structurelles de CrackSAM-GFA.

Ces tests portent sur les propriétés que l'expérience historique n'avait pas :
une voie baseline exacte, une abstention réelle et une correction bornée qui ne
peut pas déborder de l'enveloppe acceptée.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gfa.arbiter import (  # noqa: E402
    ABSTAIN,
    ADD,
    REMOVE,
    FragmentArbiter,
    compose_logits,
)
from gfa.evidence import CHANNELS, N_CHANNELS, Evidence, compute_evidence  # noqa: E402
from gfa.features import EVIDENCE_DIM, FEATURE_DIM, stack_features  # noqa: E402
from gfa.fragments import Fragment, propose_fragments  # noqa: E402
from gfa.oracles import apply_actions, binary_iou, source_oracle  # noqa: E402


def _segment(row: int, col0: int, col1: int, radius: float = 2.0) -> Fragment:
    cols = np.arange(col0, col1, dtype=np.int32)
    pixels = np.stack([np.full_like(cols, row), cols], axis=1)
    return Fragment(pixels=pixels.astype(np.int32), radius=radius, sources=("ofs",))


def test_untrained_arbiter_returns_z0_bit_for_bit():
    torch.manual_seed(0)
    arbiter = FragmentArbiter(input_dim=FEATURE_DIM)
    arbiter.eval()
    features = torch.randn(2, 5, FEATURE_DIM)
    z0 = torch.randn(2, 32, 32)
    corridors = torch.rand(2, 5, 32, 32) > 0.5

    with torch.no_grad():
        output = arbiter(features)
        z, _ = compose_logits(z0, corridors, output, hard=True)

    # gamma initialisé à zéro : la correction est exactement nulle.
    assert torch.equal(z, z0)


def test_untrained_arbiter_abstains_on_every_fragment():
    arbiter = FragmentArbiter(input_dim=FEATURE_DIM)
    arbiter.eval()
    with torch.no_grad():
        output = arbiter(torch.randn(3, 7, FEATURE_DIM))
    assert torch.all(output.hard_actions() == ABSTAIN)


def test_amplitudes_stay_non_negative_and_bounded_after_training_steps():
    torch.manual_seed(0)
    arbiter = FragmentArbiter(input_dim=FEATURE_DIM)
    optimiser = torch.optim.SGD(arbiter.parameters(), lr=5.0)
    features = torch.randn(4, 6, FEATURE_DIM)
    for _ in range(20):
        output = arbiter(features)
        # Pousse volontairement les amplitudes vers le haut.
        loss = -(output.add_amplitude.sum() + output.remove_amplitude.sum())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        output = arbiter(features)
    assert torch.all(output.add_amplitude >= 0.0)
    assert torch.all(output.remove_amplitude >= 0.0)
    assert torch.all(output.add_amplitude <= arbiter.max_delta + 1e-6)
    assert torch.all(output.remove_amplitude <= arbiter.max_delta + 1e-6)


def test_correction_never_escapes_the_accepted_envelope():
    torch.manual_seed(1)
    arbiter = FragmentArbiter(input_dim=FEATURE_DIM)
    with torch.no_grad():
        arbiter.gamma.fill_(1.0)
        arbiter.action_head.bias[ADD] = 10.0
        arbiter.action_head.bias[ABSTAIN] = 0.0
        arbiter.add_head.bias.fill_(3.0)

    z0 = torch.randn(1, 24, 24)
    corridors = torch.zeros(1, 2, 24, 24, dtype=torch.bool)
    corridors[0, 0, 4:8, 4:20] = True
    corridors[0, 1, 14:18, 4:20] = True

    with torch.no_grad():
        z, envelope = compose_logits(z0, corridors, arbiter(torch.randn(1, 2, FEATURE_DIM)))

    assert envelope.any(), "le test doit exercer une enveloppe non vide"
    assert torch.equal(z[~envelope], z0[~envelope])
    assert not torch.equal(z[envelope], z0[envelope])


def test_empty_fragment_set_is_a_legal_output_equal_to_z0():
    z0 = np.random.default_rng(0).normal(size=(16, 16)).astype(np.float32)
    assert np.array_equal(apply_actions(z0, [], []), z0)


def test_all_abstain_is_bit_identical():
    z0 = np.random.default_rng(1).normal(size=(32, 32)).astype(np.float32)
    fragments = [_segment(8, 2, 20), _segment(20, 4, 24)]
    result = apply_actions(z0, fragments, ["abstain", "abstain"])
    assert np.array_equal(result, z0)


def test_add_and_remove_move_logits_in_the_expected_direction():
    z0 = np.zeros((32, 32), dtype=np.float32)
    fragment = _segment(10, 5, 25)
    corridor = fragment.corridor(z0.shape)
    added = apply_actions(z0, [fragment], ["add"])
    removed = apply_actions(z0, [fragment], ["remove"])
    assert np.all(added[corridor] > 0.0)
    assert np.all(removed[corridor] < 0.0)
    assert np.array_equal(added[~corridor], z0[~corridor])


def test_source_oracle_orders_baseline_achievable_upper_bound():
    rng = np.random.default_rng(3)
    target = np.zeros((64, 64), dtype=bool)
    target[30:33, 8:56] = True
    z0 = np.where(target, 0.4, -0.6).astype(np.float32)
    z0 += rng.normal(0.0, 0.2, size=z0.shape).astype(np.float32)
    fragments = [_segment(31, 8, 30), _segment(31, 30, 56), _segment(50, 8, 30)]

    result = source_oracle(z0, target, fragments)
    assert result.baseline_iou <= result.achievable_iou + 1e-9
    assert result.achievable_iou <= result.upper_bound_iou + 1e-9
    assert result.n_fragments == 3


def test_source_oracle_without_fragments_reports_no_gain():
    target = np.zeros((32, 32), dtype=bool)
    target[10:14, 4:28] = True
    z0 = np.where(target, 1.0, -1.0).astype(np.float32)
    result = source_oracle(z0, target, [])
    assert result.achievable_gain == pytest.approx(0.0)
    assert result.upper_bound_gain == pytest.approx(0.0)


def test_binary_iou_matches_cracksam2_empty_conventions():
    empty = np.zeros((8, 8), dtype=bool)
    assert binary_iou(empty, empty) == 1.0
    filled = np.ones((8, 8), dtype=bool)
    assert binary_iou(empty, filled) == 0.0
    assert binary_iou(filled, empty) == 0.0


def test_evidence_channels_are_finite_and_ordered():
    rng = np.random.default_rng(5)
    image = (rng.random((96, 96, 3)) * 255).astype(np.uint8)
    image[40:44, :, :] = 20  # une bande sombre horizontale
    evidence = compute_evidence(image)
    assert evidence.planes.shape == (N_CHANNELS, 96, 96)
    assert np.isfinite(evidence.planes).all()
    for name in CHANNELS:
        if name in {"cos2theta", "sin2theta", "scale"}:
            continue
        plane = evidence.plane(name)
        assert plane.min() >= -1e-6, name
        assert plane.max() <= 1.0 + 1e-6, name


def test_uniform_image_produces_an_empty_support():
    """L'abstention doit être réelle : pas de structure, pas de fragment."""

    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    evidence = compute_evidence(image)
    thresholds = {"frangi_sim": 0.5, "phase_sym": 0.3, "ofs": 0.2, "dbic": 0.5}
    assert propose_fragments(evidence, thresholds) == []


def test_feature_dimensions_are_consistent():
    rng = np.random.default_rng(7)
    planes = rng.random((N_CHANNELS, 48, 48)).astype(np.float32)
    evidence = Evidence(planes=planes, noise_sigma=0.01)
    fragments = [_segment(20, 4, 24), _segment(30, 6, 30)]
    without_z0 = stack_features(evidence, fragments)
    assert without_z0.shape == (2, EVIDENCE_DIM)
    z0 = rng.normal(size=(48, 48)).astype(np.float32)
    with_z0 = stack_features(evidence, fragments, z0=z0)
    assert with_z0.shape == (2, FEATURE_DIM)
    assert np.isfinite(with_z0).all()
