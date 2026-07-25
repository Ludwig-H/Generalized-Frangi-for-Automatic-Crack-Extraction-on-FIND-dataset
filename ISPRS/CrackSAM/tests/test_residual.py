from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from cracksam2.graph_types import FRANGI_RASTER_CHANNEL_INDEX, FRANGI_RASTER_CHANNELS  # noqa: E402
from cracksam2.evidence_selection import PROFILE_CHANNELS  # noqa: E402
from cracksam2.losses import residual_training_loss  # noqa: E402
from cracksam2.model import CrackSAM2  # noqa: E402
from cracksam2.residual import (  # noqa: E402
    VERIFIED_ADAPTER_MODE,
    FrangiGraphResidual,
    select_residual_logits,
)


class _FakePromptEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.mask_input_size = (4, 4)

    def forward(self, *, points, boxes, masks):
        del points, boxes
        batch_size = 1 if masks is None else masks.shape[0]
        sparse = self.anchor * torch.zeros(batch_size, 0, 3)
        dense = self.anchor * torch.zeros(batch_size, 3, 2, 2)
        if masks is not None:
            pooled = F.adaptive_avg_pool2d(masks, (2, 2))
            dense = dense + pooled.expand(-1, 3, -1, -1)
        return sparse, dense

    def get_dense_pe(self):
        return self.anchor * torch.zeros(1, 3, 2, 2)


class _FakeMaskDecoder(nn.Module):
    def forward(
        self,
        *,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        repeat_image,
        high_res_features,
    ):
        del (
            image_pe,
            sparse_prompt_embeddings,
            multimask_output,
            repeat_image,
            high_res_features,
        )
        low_res = image_embeddings.mean(dim=1, keepdim=True)
        low_res = low_res + dense_prompt_embeddings.mean(dim=1, keepdim=True)
        score = low_res.mean(dim=(2, 3))
        return low_res, score, None, score - 0.25


class _FakeFullSAM2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_size = 8
        self.directly_add_no_mem_embed = False
        self.sam_prompt_encoder = _FakePromptEncoder()
        self.sam_mask_decoder = _FakeMaskDecoder()
        self.forward_image_calls = 0

    def forward_image(self, images):
        self.forward_image_calls += 1
        return images

    @staticmethod
    def _sequence(features):
        return features.flatten(2).permute(2, 0, 1)

    def _prepare_backbone_features(self, backbone_out):
        high_resolution = F.adaptive_avg_pool2d(backbone_out, (4, 4))
        embeddings = F.adaptive_avg_pool2d(backbone_out, (2, 2))
        return (
            None,
            [self._sequence(high_resolution), self._sequence(embeddings)],
            None,
            [(4, 4), (2, 2)],
        )


@pytest.mark.parametrize("with_mask", [False, True])
def test_factorized_encode_decode_matches_forward_exactly(with_mask):
    torch.manual_seed(7)
    baseline = CrackSAM2(_FakeFullSAM2()).eval()
    images = torch.rand(2, 3, 7, 9)
    masks = torch.randn(2, 1, 5, 6) if with_mask else None

    direct = baseline(images, mask_input=masks, output_size=(6, 10))
    features = baseline.encode_images(images)
    factorized = baseline.decode_features(
        features,
        mask_input=masks,
        output_size=(6, 10),
    )

    assert features.input_size == (7, 9)
    assert features.batch_size == 2
    for key in direct:
        torch.testing.assert_close(direct[key], factorized[key], rtol=0, atol=0)


def test_encoded_features_can_feed_multiple_decodes_without_reencoding():
    baseline = CrackSAM2(_FakeFullSAM2()).eval()
    images = torch.rand(2, 3, 8, 8)
    masks = torch.ones(2, 1, 4, 4)

    features = baseline.encode_images(images)
    no_prompt = baseline.decode_features(features)
    prompted = baseline.decode_features(features, mask_input=masks)

    assert baseline.sam2.forward_image_calls == 1
    assert not torch.equal(no_prompt["logits"], prompted["logits"])


def _build_residual_model() -> FrangiGraphResidual:
    baseline = CrackSAM2(_FakeFullSAM2()).eval()
    return FrangiGraphResidual(
        baseline,
        raster_channels=2,
        high_resolution_channels=(3,),
        hidden_channels=4,
    )


def test_residual_initialization_is_exactly_neutral_and_baseline_is_frozen():
    torch.manual_seed(11)
    images = torch.rand(2, 3, 7, 9)
    raster = torch.rand(2, 2, 5, 6)
    model = _build_residual_model()
    reference = model.baseline(images)["logits"]
    model.baseline.sam2.forward_image_calls = 0

    model.train()
    output = model(images, raster)

    assert model.training
    assert model.adapter.training
    assert not model.baseline.training
    assert model.baseline.sam2.forward_image_calls == 1
    assert all(not parameter.requires_grad for parameter in model.baseline.parameters())
    assert all(parameter.requires_grad for parameter in model.adapter.parameters())
    assert torch.count_nonzero(output["residual_logits"]).item() == 0
    torch.testing.assert_close(output["baseline_logits"], reference, rtol=0, atol=0)
    torch.testing.assert_close(output["candidate_logits"], reference, rtol=0, atol=0)
    torch.testing.assert_close(output["logits"], reference, rtol=0, atol=0)


def test_boolean_confidence_gate_has_an_exact_per_image_fallback():
    baseline = torch.randn(3, 1, 4, 5)
    candidate = baseline + torch.randn_like(baseline)
    accepted = torch.tensor([True, False, True])

    selected = select_residual_logits(baseline, candidate, accepted)

    assert torch.equal(selected[0], candidate[0])
    assert torch.equal(selected[1], baseline[1])
    assert torch.equal(selected[2], candidate[2])
    assert select_residual_logits(baseline, candidate, False) is baseline
    assert select_residual_logits(baseline, candidate, True) is candidate


def test_residual_wrapper_falls_back_after_adapter_has_changed():
    model = _build_residual_model().eval()
    with torch.no_grad():
        model.adapter.output_projection.bias.fill_(0.75)

    output = model(
        torch.rand(2, 3, 8, 8),
        torch.rand(2, 2, 8, 8),
        accept_residual=False,
    )

    assert torch.count_nonzero(output["residual_logits"]).item() > 0
    assert output["logits"] is output["baseline_logits"]
    torch.testing.assert_close(
        output["logits"], output["baseline_logits"], rtol=0, atol=0
    )


def test_only_adapter_receives_gradients_during_residual_training():
    torch.manual_seed(17)
    model = _build_residual_model().train()
    images = torch.rand(2, 3, 8, 8)
    raster = torch.rand(2, 2, 8, 8)

    output = model(images, raster)
    output["candidate_logits"].sum().backward()

    assert all(parameter.grad is None for parameter in model.baseline.parameters())
    assert model.adapter.output_projection.weight.grad is not None
    assert torch.count_nonzero(model.adapter.output_projection.weight.grad).item() > 0


def test_residual_adapter_rejects_wrong_raster_channel_count():
    model = _build_residual_model()
    with pytest.raises(ValueError, match="raster must have shape"):
        model(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))


def _build_verified_model(*, evidence_dilation: int = 0) -> FrangiGraphResidual:
    return FrangiGraphResidual(
        CrackSAM2(_FakeFullSAM2()).eval(),
        raster_channels=len(FRANGI_RASTER_CHANNELS),
        high_resolution_channels=(3,),
        hidden_channels=4,
        adapter_mode=VERIFIED_ADAPTER_MODE,
        profile_radii=(1.0,),
        evidence_dilation=evidence_dilation,
        evidence_threshold=0.5,
    )


def test_verified_residual_is_neutral_and_exposes_local_evidence_diagnostics() -> None:
    model = _build_verified_model().eval()
    images = torch.rand(2, 3, 8, 8)
    raster = torch.zeros(2, len(FRANGI_RASTER_CHANNELS), 8, 8)
    raster[:, FRANGI_RASTER_CHANNEL_INDEX["support"]] = 1.0
    raster[:, FRANGI_RASTER_CHANNEL_INDEX["orientation_cos2"]] = 1.0

    output = model(images, raster)

    assert torch.equal(output["candidate_logits"], output["baseline_logits"])
    assert output["photometric_profiles"].shape == (
        2,
        len(PROFILE_CHANNELS),
        8,
        8,
    )
    assert output["evidence_selected"].shape == output["baseline_logits"].shape
    for field in (
        "evidence_score_fusion",
        "evidence_support_fusion",
        "evidence_selected_fusion",
        "correction_envelope_fusion",
    ):
        assert output[field].shape == (2, 1, 4, 4)
    assert output["photometric_profiles_fusion"].shape == (
        2,
        len(PROFILE_CHANNELS),
        4,
        4,
    )
    assert torch.count_nonzero(output["residual_logits"]) == 0


def test_verified_residual_rejects_empty_or_low_confidence_evidence_exactly() -> None:
    model = _build_verified_model().eval()
    with torch.no_grad():
        model.adapter.output_projection.bias.fill_(0.75)
        model.adapter.verifier[-1].weight.zero_()
        model.adapter.verifier[-1].bias.fill_(-20.0)
    images = torch.rand(1, 3, 8, 8)
    empty = torch.zeros(1, len(FRANGI_RASTER_CHANNELS), 8, 8)

    no_evidence = model(images, empty)
    assert torch.equal(no_evidence["candidate_logits"], no_evidence["baseline_logits"])
    assert torch.count_nonzero(no_evidence["correction_envelope"]) == 0

    supported = empty.clone()
    supported[:, FRANGI_RASTER_CHANNEL_INDEX["support"]] = 1.0
    supported[:, FRANGI_RASTER_CHANNEL_INDEX["orientation_cos2"]] = 1.0
    rejected = model(images, supported)
    assert torch.equal(rejected["candidate_logits"], rejected["baseline_logits"])
    assert torch.count_nonzero(rejected["evidence_selected"]) == 0


def test_verified_residual_reports_orientation_coherence_after_support_pooling(
) -> None:
    model = _build_verified_model().eval()
    images = torch.rand(1, 3, 8, 8)
    raster = torch.zeros(1, len(FRANGI_RASTER_CHANNELS), 8, 8)
    support_index = FRANGI_RASTER_CHANNEL_INDEX["support"]
    cos_index = FRANGI_RASTER_CHANNEL_INDEX["orientation_cos2"]
    raster[:, support_index, 0, :2] = 1.0
    # Orthogonal unoriented normals have opposite double-angle encodings. They
    # cancel inside the same 2x2 feature cell and must not invent theta=0.
    raster[:, cos_index, 0, 0] = 1.0
    raster[:, cos_index, 0, 1] = -1.0

    output = model(images, raster)
    coherence_index = PROFILE_CHANNELS.index("orientation_coherence")

    assert output["evidence_support_fusion"][0, 0, 0, 0] == 1.0
    cell_profiles = output["photometric_profiles_fusion"][0, :, 0, 0]
    assert cell_profiles[coherence_index] == pytest.approx(0.0, abs=1e-7)
    assert torch.count_nonzero(cell_profiles) == 0


def test_verified_residual_correction_is_exactly_zero_outside_selected_region() -> None:
    model = _build_verified_model(evidence_dilation=0).eval()
    with torch.no_grad():
        model.adapter.output_projection.bias.fill_(0.75)
        model.adapter.verifier[-1].weight.zero_()
        model.adapter.verifier[-1].bias.fill_(20.0)
    raster = torch.zeros(1, len(FRANGI_RASTER_CHANNELS), 8, 8)
    support_index = FRANGI_RASTER_CHANNEL_INDEX["support"]
    # Adaptive max reduction must preserve a one-pixel Frangi skeleton.
    raster[:, support_index, 1, 1] = 1.0
    raster[:, FRANGI_RASTER_CHANNEL_INDEX["orientation_cos2"]] = 1.0

    output = model(torch.rand(1, 3, 8, 8), raster)
    envelope = output["correction_envelope"].bool()

    assert torch.count_nonzero(envelope) > 0
    assert torch.count_nonzero(~envelope) > 0
    assert torch.count_nonzero(output["residual_logits"][~envelope]) == 0
    assert torch.count_nonzero(output["residual_logits"][envelope]) > 0
    torch.testing.assert_close(
        output["candidate_logits"][~envelope],
        output["baseline_logits"][~envelope],
        rtol=0,
        atol=0,
    )


def test_verified_residual_trains_proposal_and_verifier_without_baseline_gradients(
) -> None:
    model = _build_verified_model().train()
    images = torch.rand(2, 3, 8, 8)
    raster = torch.zeros(2, len(FRANGI_RASTER_CHANNELS), 8, 8)
    raster[:, FRANGI_RASTER_CHANNEL_INDEX["support"]] = 1.0
    raster[:, FRANGI_RASTER_CHANNEL_INDEX["orientation_cos2"]] = 1.0
    targets = torch.zeros(2, 1, 8, 8)
    targets[:, :, :, 3:5] = 1.0

    output = model(images, raster)
    losses = residual_training_loss(
        output["candidate_logits"],
        output["baseline_logits"],
        targets,
        topology_weight=0.0,
        evidence_logits=output["evidence_logits"],
        evidence_support=output["evidence_support"],
        evidence_weight=0.25,
        evidence_target_tolerance=0,
    )
    losses["loss"].backward()

    assert model.adapter.output_projection.weight.grad is not None
    assert torch.count_nonzero(model.adapter.output_projection.weight.grad) > 0
    assert model.adapter.verifier[-1].weight.grad is not None
    assert torch.count_nonzero(model.adapter.verifier[-1].weight.grad) > 0
    assert all(parameter.grad is None for parameter in model.baseline.parameters())


def test_verified_photometric_profiles_remain_finite_under_amp() -> None:
    model = _build_verified_model().eval()
    images = torch.rand(1, 3, 8, 8)
    raster = torch.zeros(1, len(FRANGI_RASTER_CHANNELS), 8, 8)
    raster[:, FRANGI_RASTER_CHANNEL_INDEX["support"]] = 1.0
    raster[:, FRANGI_RASTER_CHANNEL_INDEX["orientation_cos2"]] = 1.0

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(images, raster)

    assert torch.isfinite(output["photometric_profiles"]).all()
    assert torch.isfinite(output["candidate_logits"]).all()
