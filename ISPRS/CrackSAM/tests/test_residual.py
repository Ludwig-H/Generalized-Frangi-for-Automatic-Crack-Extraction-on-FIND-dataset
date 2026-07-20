from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from cracksam2.model import CrackSAM2  # noqa: E402
from cracksam2.residual import FrangiGraphResidual, select_residual_logits


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
