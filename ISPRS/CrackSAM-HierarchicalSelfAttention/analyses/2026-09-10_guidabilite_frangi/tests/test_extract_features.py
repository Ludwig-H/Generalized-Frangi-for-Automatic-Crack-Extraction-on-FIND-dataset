"""Guard provenance, label-independent sampling, and feature tensor semantics."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = spec_from_file_location("extract_features", Path(__file__).parents[1] / "extract_features.py")
extract = module_from_spec(SPEC)
SPEC.loader.exec_module(extract)


def test_weights_only_loader_accepts_historical_torch_version_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    from torch.torch_version import TorchVersion

    path = tmp_path / "baseline.pt"
    tensor = torch.tensor([1., 2.])
    torch.save({"adapter": {"test": tensor}, "version": TorchVersion("2.7.0")}, path)
    version_allowed_before = TorchVersion in torch.serialization.get_safe_globals()
    payload = extract.load_baseline_payload(path)
    torch.testing.assert_close(payload["adapter"]["test"], tensor)
    assert str(payload["version"]) == "2.7.0"
    assert (TorchVersion in torch.serialization.get_safe_globals()) == version_allowed_before


def test_resume_rejects_changed_weights_or_image_manifest(tmp_path):
    path = tmp_path / "contract.json"
    original = {"weights": "a", "inputs": "same"}
    digest = extract.ensure_contract(path, original)
    assert extract.ensure_contract(path, original) == digest
    for changed in ({"weights": "b", "inputs": "same"}, {"weights": "a", "inputs": "other"}):
        with pytest.raises(ValueError, match="contract changed"):
            extract.ensure_contract(path, changed)
    assert extract.ensure_contract(path, original) == digest


def test_verification_sampling_is_balanced_and_label_independent():
    rows = [{"dataset": dataset, "image_id": f"{dataset}::{i}", "category": "gain"}
            for dataset in ("a", "b", "c") for i in range(10)]
    original = extract.select_verification(rows, 6)
    changed = [dict(row, category="loss") for row in reversed(rows)]
    assert extract.select_verification(changed, 6) == original
    assert all(sum(value.startswith(dataset + "::") for value in original) == 2 for dataset in ("a", "b", "c"))


def test_cases_reject_duplicate_image_and_mismatched_identity(tmp_path):
    path = tmp_path / "cases.csv"
    header = "image_id,dataset,case_name\n"
    row = "road420::example.jpg,road420,example.jpg\n"
    path.write_text(header + row + row)
    with pytest.raises(ValueError, match="duplicate"):
        extract.read_cases(path)
    path.write_text(header + "wrong,road420,example.jpg\n")
    with pytest.raises(ValueError, match="Invalid"):
        extract.read_cases(path)


def test_resume_shard_rejects_reordered_images_and_nonfinite_values(tmp_path):
    path = tmp_path / "shard.npz"
    arrays = {key: np.zeros((2, dimensions), dtype=np.float32) for key, dimensions in extract.DIMENSIONS.items()}
    arrays.update(ids=np.asarray(["a", "b"]), contract_sha256=np.asarray("abc"))
    extract.atomic_npz(path, arrays)
    extract.validate_shard(path, ["a", "b"], "abc")
    with pytest.raises(ValueError, match="IDs/order"):
        extract.validate_shard(path, ["b", "a"], "abc")
    arrays["mean"][0, 0] = np.nan
    extract.atomic_npz(path, arrays)
    with pytest.raises(ValueError, match="Invalid mean"):
        extract.validate_shard(path, ["a", "b"], "abc")


def test_pooling_preserves_channel_and_quadrant_order():
    torch = pytest.importorskip("torch")
    pattern = torch.tensor([[1., 2.], [3., 4.]]).repeat_interleave(32, 0).repeat_interleave(32, 1)
    h = pattern[None, None].expand(1, 256, 64, 64).clone()
    h[:, 1] += 10
    features = SimpleNamespace(image_embeddings=h, high_resolution_features=(
        torch.full((1, 32, 256, 256), 7.), torch.full((1, 64, 128, 128), 9.)))
    values = extract.pool_features(features)
    np.testing.assert_allclose(values["mean"][0, :2], [2.5, 12.5])
    np.testing.assert_allclose(values["std"][0, :2], np.sqrt(1.25))
    np.testing.assert_allclose(values["grid2"][0, :8], [1, 2, 3, 4, 11, 12, 13, 14])
    np.testing.assert_allclose(values["multiscale_mean"][0, 256:288], 7)
    np.testing.assert_allclose(values["multiscale_mean"][0, 288:], 9)


def test_pre_global_hook_observes_input_not_attention_output():
    torch = pytest.importorskip("torch")

    class Attention(torch.nn.Module):
        def forward(self, h):
            return h * 7

    blocks = [SimpleNamespace(window_size=window, attn=Attention()) for window in (0, 4, 0, 4)]
    model = SimpleNamespace(sam2=SimpleNamespace(image_encoder=SimpleNamespace(trunk=SimpleNamespace(blocks=blocks))))
    state, handle, description = extract.install_pre_global_hook(model)
    try:
        output = blocks[2].attn(torch.ones((2, 3, 3, 8)))
        assert description["block_index"] == 2
        np.testing.assert_array_equal(state["mean"], np.ones((2, 8)))
        assert torch.all(output == 7)
    finally:
        handle.remove()
