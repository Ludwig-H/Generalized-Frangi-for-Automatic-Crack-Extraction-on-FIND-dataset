from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from cracksam2.data import (  # noqa: E402
    PROMPT_CACHE_MANIFEST,
    CrackSegmentationDataset,
    apply_geometric_transform,
    apply_noise_perturbation,
    resolve_sample_paths,
    sample_names_sha256,
)


def _write_pair(root, image_name="sample.JPG", mask_name="sample.png"):
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[1, 2] = (255, 128, 64)
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1, 2] = 255
    Image.fromarray(image).save(root / "images" / image_name)
    Image.fromarray(mask).save(root / "masks" / mask_name)
    return image, mask


def test_resolves_existing_list_and_returns_rgb_float_binary_mask(tmp_path):
    root = tmp_path / "dataset"
    _write_pair(root)
    list_file = tmp_path / "test_vol.txt"
    list_file.write_text("sample.JPG\n", encoding="utf-8")

    image_path, mask_path = resolve_sample_paths(root, "sample.JPG")
    assert image_path.name == "sample.JPG"
    assert mask_path.name == "sample.png"

    dataset = CrackSegmentationDataset(
        root, list_file=list_file, image_size=(8, 10)
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 8, 10)
    assert sample["image"].dtype == torch.float32
    assert 0.0 <= sample["image"].min() <= sample["image"].max() <= 1.0
    assert sample["mask"].shape == (1, 8, 10)
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
    assert sample["case_name"] == "sample.JPG"


def test_geometric_transform_keeps_mask_and_prompt_registered():
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    mask = np.zeros((3, 3), dtype=np.uint8)
    prompt = np.zeros((3, 3), dtype=np.float32)
    image[0, 1] = 255
    mask[0, 1] = 1
    prompt[0, 1] = 0.75

    transformed_image, transformed_mask, transformed_prompt = apply_geometric_transform(
        image,
        mask,
        prompt,
        rotation_k=1,
        horizontal_flip=True,
        vertical_flip=True,
    )
    image_position = np.argwhere(transformed_image[..., 0] > 0)
    mask_position = np.argwhere(transformed_mask > 0)
    prompt_position = np.argwhere(transformed_prompt > 0)
    np.testing.assert_array_equal(image_position, mask_position)
    np.testing.assert_array_equal(mask_position, prompt_position)
    assert transformed_prompt[tuple(prompt_position[0])] == 0.75


def test_paper_noise_perturbations():
    constant = np.full((32, 24, 3), 100, dtype=np.uint8)
    noisy1 = apply_noise_perturbation(constant, "noisy1")
    assert noisy1.shape == constant.shape
    np.testing.assert_allclose(noisy1, 50, atol=1)

    noisy2 = apply_noise_perturbation(constant, "noisy2", output_size=448)
    assert noisy2.shape == (448, 448, 3)
    np.testing.assert_allclose(noisy2, 100, atol=1)


def test_frangi_prompt_is_cached_as_npy_and_reused(tmp_path):
    root = tmp_path / "dataset"
    _write_pair(root, image_name="nested.jpg", mask_name="nested.jpg")
    list_file = tmp_path / "train.txt"
    list_file.write_text("nested.jpg\n", encoding="utf-8")
    calls = []

    def prompt_generator(image):
        calls.append(image.shape)
        return image[..., 0].astype(np.float32) / 255.0

    cache_dir = tmp_path / "prompt-cache"
    dataset = CrackSegmentationDataset(
        root,
        list_file=list_file,
        image_size=8,
        prompt_cache_dir=cache_dir,
        prompt_generator=prompt_generator,
    )
    first = dataset[0]
    second = dataset[0]

    assert calls == [(4, 6, 3)]
    assert (cache_dir / "nested.jpg.npy").is_file()
    assert first["prompt"].shape == (1, 256, 256)
    torch.testing.assert_close(first["prompt"], second["prompt"])


def test_completed_prompt_cache_manifest_is_required_and_validated(tmp_path):
    root = tmp_path / "dataset"
    _write_pair(root)
    list_file = tmp_path / "test.txt"
    list_file.write_text("sample.JPG\n", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "sample.JPG.npy", np.zeros((1, 256, 256), dtype=np.float32))

    with pytest.raises(FileNotFoundError, match="manifest"):
        CrackSegmentationDataset(root, list_file=list_file, prompt_cache_dir=cache)

    manifest = {
        "format_version": 1,
        "status": "complete",
        "samples": 1,
        "sample_names_sha256": sample_names_sha256(["sample.JPG"]),
        "image_size": [448, 448],
        "prompt_size": [256, 256],
        "noise": "original",
        "frangi": {
            "scales": [1.0, 3.0, 5.0, 9.0, 15.0],
            "R": 3,
            "K": 1,
            "eps": 1e-5,
        },
    }
    (cache / PROMPT_CACHE_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    dataset = CrackSegmentationDataset(
        root, list_file=list_file, prompt_cache_dir=cache
    )
    assert dataset[0]["prompt"].shape == (1, 256, 256)

    manifest["noise"] = "noisy1"
    (cache / PROMPT_CACHE_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incompatible"):
        CrackSegmentationDataset(root, list_file=list_file, prompt_cache_dir=cache)


def test_augmentation_is_stateless_per_epoch_and_sample(tmp_path):
    root = tmp_path / "dataset"
    _write_pair(root, image_name="sample.png", mask_name="sample.png")
    list_file = tmp_path / "train.txt"
    list_file.write_text("sample.png\n", encoding="utf-8")
    dataset = CrackSegmentationDataset(
        root,
        list_file=list_file,
        image_size=8,
        augment=True,
        augmentation_seed=123,
    )

    dataset.set_epoch(0)
    first = dataset[0]
    repeated = dataset[0]
    torch.testing.assert_close(first["image"], repeated["image"])
    torch.testing.assert_close(first["mask"], repeated["mask"])

    dataset.set_epoch(1)
    next_epoch = dataset[0]
    assert not torch.equal(first["mask"], next_epoch["mask"])
