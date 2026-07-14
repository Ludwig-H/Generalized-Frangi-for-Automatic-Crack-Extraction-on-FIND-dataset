from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

import precompute_frangi_prompts as precompute  # noqa: E402
from cracksam2.data import PROMPT_CACHE_MANIFEST  # noqa: E402


def test_precompute_resizes_before_frangi_and_publishes_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    Image.fromarray(np.zeros((11, 17, 3), dtype=np.uint8)).save(
        root / "images" / "SAMPLE.JPG"
    )
    Image.fromarray(np.zeros((11, 17), dtype=np.uint8)).save(
        root / "masks" / "SAMPLE.JPG"
    )
    list_file = tmp_path / "test.txt"
    list_file.write_text("sample.jpg\n", encoding="utf-8")
    cache = tmp_path / "cache"
    observed: list[tuple[int, ...]] = []

    def fake_generate(image: np.ndarray, output_path: Path, **kwargs) -> np.ndarray:
        observed.append(image.shape)
        prompt = np.zeros((1, 256, 256), dtype=np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, prompt, allow_pickle=False)
        return prompt

    monkeypatch.setattr(precompute, "generate_frangi_prompt", fake_generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precompute_frangi_prompts.py",
            "--data-root",
            str(root),
            "--list-file",
            str(list_file),
            "--cache-dir",
            str(cache),
            "--device",
            "cpu",
        ],
    )

    assert precompute.main() == 0
    assert observed == [(448, 448, 3)]
    assert (cache / "sample.jpg.npy").is_file()
    manifest = json.loads((cache / PROMPT_CACHE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["samples"] == 1
    assert manifest["noise"] == "original"
    assert manifest["frangi"] == {
        "K": 1,
        "R": 3,
        "eps": 1e-5,
        "scales": [1.0, 3.0, 5.0, 9.0, 15.0],
    }

    assert precompute.main() == 0
    assert observed == [(448, 448, 3)]
