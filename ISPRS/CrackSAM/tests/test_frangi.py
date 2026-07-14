from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from cracksam2.frangi import (  # noqa: E402
    DEFAULT_FRANGI_SCALES,
    extract_frangi_graph_gpu,
    generate_frangi_prompt,
    probability_to_logits,
    rgb_to_grayscale,
    save_prompt_atomic,
)


def test_roadmap_defaults_are_explicit() -> None:
    signature = inspect.signature(extract_frangi_graph_gpu)
    assert signature.parameters["scales"].default == (1.0, 3.0, 5.0, 9.0, 15.0)
    assert signature.parameters["R"].default == 3
    assert signature.parameters["K"].default == 1
    assert DEFAULT_FRANGI_SCALES == (1.0, 3.0, 5.0, 9.0, 15.0)


def test_rgb_to_grayscale_supports_hwc_and_chw_uint8() -> None:
    rgb_hwc = np.zeros((5, 6, 3), dtype=np.uint8)
    rgb_hwc[..., 0] = 255
    gray_hwc = rgb_to_grayscale(rgb_hwc)
    gray_chw = rgb_to_grayscale(torch.from_numpy(rgb_hwc).permute(2, 0, 1))

    assert gray_hwc.shape == (5, 6)
    assert gray_hwc.dtype == torch.float32
    torch.testing.assert_close(gray_hwc, gray_chw)
    torch.testing.assert_close(gray_hwc, torch.full((5, 6), 0.2989))


def test_rgb_to_grayscale_does_not_mutate_float_input() -> None:
    image = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)
    original = image.clone()

    gray = rgb_to_grayscale(image)

    torch.testing.assert_close(image, original)
    torch.testing.assert_close(gray, torch.tensor([[0.0, 1.0]]))


def test_probability_to_logits_clips_and_preserves_container_type() -> None:
    probabilities = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    logits = probability_to_logits(probabilities)

    assert isinstance(logits, np.ndarray)
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()
    assert logits[1] == pytest.approx(0.0, abs=1e-7)
    assert logits[0] == pytest.approx(-logits[2], rel=2e-3)

    tensor_logits = probability_to_logits(torch.from_numpy(probabilities))
    assert isinstance(tensor_logits, torch.Tensor)
    assert tensor_logits.dtype == torch.float32
    assert torch.isfinite(tensor_logits).all()


def test_atomic_prompt_writer_round_trips_float32(tmp_path) -> None:
    destination = tmp_path / "nested" / "sample.npy"
    prompt = np.arange(12, dtype=np.float64).reshape(1, 3, 4)

    assert save_prompt_atomic(destination, prompt) == destination
    cached = np.load(destination, allow_pickle=False)

    assert cached.dtype == np.float32
    np.testing.assert_array_equal(cached, prompt.astype(np.float32))
    assert list(destination.parent.glob("*.tmp")) == []


def test_cpu_extraction_preserves_return_contract() -> None:
    image = torch.ones((16, 16), dtype=torch.float32)
    image[:, 7:9] = 0.0

    response, similarity, centrality, timings, diagnostics = extract_frangi_graph_gpu(
        {"visible": image},
        {"visible": 1.0},
        scales=(1.0,),
        R=1,
        tau=0.5,
        min_rel_size=1000.0,
        K=1,
        device="cpu",
    )

    for output in (response, similarity, centrality):
        assert output.shape == image.shape
        assert output.dtype == np.float32
        assert np.isfinite(output).all()
    assert 0.0 <= similarity.min() <= similarity.max() <= 1.0
    assert similarity.max() > 0.0
    assert "Total" in timings
    assert set(diagnostics) == {"tau_mask", "comp_mask"}


def test_generate_prompt_has_sam_mask_shape_and_can_cache(tmp_path) -> None:
    image = np.full((12, 12, 3), 255, dtype=np.uint8)
    destination = tmp_path / "blank.npy"

    prompt = generate_frangi_prompt(
        image,
        destination,
        prompt_size=(8, 10),
        scales=(1.0,),
        R=1,
        device="cpu",
    )

    assert prompt.shape == (1, 8, 10)
    assert prompt.dtype == np.float32
    assert np.isfinite(prompt).all()
    np.testing.assert_array_equal(np.load(destination, allow_pickle=False), prompt)


def test_similarity_only_path_matches_full_graph() -> None:
    image = torch.ones((16, 16), dtype=torch.float32)
    image[:, 7:9] = 0.0
    arguments = {
        "scales": (1.0,),
        "R": 1,
        "tau": 0.5,
        "min_rel_size": 1000.0,
        "K": 1,
        "device": "cpu",
    }
    full = extract_frangi_graph_gpu(
        {"visible": image}, {"visible": 1.0}, compute_centrality=True, **arguments
    )
    similarity_only = extract_frangi_graph_gpu(
        {"visible": image}, {"visible": 1.0}, compute_centrality=False, **arguments
    )

    np.testing.assert_array_equal(full[1], similarity_only[1])
    assert similarity_only[3]["4. MST (CPU)"] == 0.0
    assert similarity_only[3]["5. Betweenness (GPU)"] == 0.0
