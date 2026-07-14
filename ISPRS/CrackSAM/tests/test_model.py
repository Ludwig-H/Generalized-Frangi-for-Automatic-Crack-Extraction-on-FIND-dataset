from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from cracksam2.losses import cracksam_loss, warmup_poly_lr
from cracksam2.model import LoRALinear, LoRAQKV, inject_lora_qv


def test_lora_linear_is_identity_at_initialization_and_trains_only_delta():
    base = nn.Linear(5, 7)
    inputs = torch.randn(2, 3, 5)
    reference = base(inputs).detach()
    layer = LoRALinear(base, rank=2)

    torch.testing.assert_close(layer(inputs), reference)
    assert not layer.base.weight.requires_grad
    assert layer.lora_a.weight.requires_grad
    assert layer.lora_b.weight.requires_grad


def test_hiera_qkv_lora_changes_only_query_and_value_slices():
    base = nn.Linear(4, 18, bias=False)
    layer = LoRAQKV(base, rank=2)
    nn.init.ones_(layer.lora_b_q.weight)
    nn.init.ones_(layer.lora_b_v.weight)
    inputs = torch.randn(2, 4)

    delta = layer(inputs) - base(inputs)
    query, key, value = delta.chunk(3, dim=-1)
    assert torch.count_nonzero(query).item() > 0
    assert torch.count_nonzero(value).item() > 0
    assert torch.count_nonzero(key).item() == 0


class _Attention(nn.Module):
    def __init__(self, fused: bool = False):
        super().__init__()
        if fused:
            self.qkv = nn.Linear(8, 24)
        else:
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)
            self.out_proj = nn.Linear(8, 8)


class _FakeSAM2(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = nn.ModuleList(
            [nn.ModuleDict({"attn": _Attention(fused=True)}) for _ in range(3)]
        )
        # ModuleDict does not expose keys as attributes; use simple modules like Hiera.
        hiera_blocks = nn.ModuleList()
        for entry in blocks:
            block = nn.Module()
            block.attn = entry["attn"]
            hiera_blocks.append(block)
        self.image_encoder = nn.Module()
        self.image_encoder.trunk = nn.Module()
        self.image_encoder.trunk.blocks = hiera_blocks
        self.sam_mask_decoder = nn.Sequential(_Attention(), _Attention())


def test_injection_freezes_base_and_targets_every_expected_projection():
    model = _FakeSAM2()
    report = inject_lora_qv(model, rank=4)

    assert len(report.hiera_qv_projections) == 3
    assert len(report.decoder_q_projections) == 2
    assert len(report.decoder_v_projections) == 2
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)


def test_cracksam_loss_uses_requested_weighting():
    logits = torch.tensor([[[[0.2, -0.3], [1.0, -2.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    total, ce, dice = cracksam_loss(logits, target, ce_weight=0.2)
    torch.testing.assert_close(total, 0.2 * ce + 0.8 * dice)


def test_warmup_poly_schedule_boundaries():
    assert warmup_poly_lr(0, 1000, warmup_steps=300) == pytest.approx(4e-4 / 300)
    assert warmup_poly_lr(299, 1000, warmup_steps=300) == pytest.approx(4e-4)
    assert warmup_poly_lr(300, 1000, warmup_steps=300) == pytest.approx(4e-4)
    assert warmup_poly_lr(1000, 1000, warmup_steps=300) == 0.0
