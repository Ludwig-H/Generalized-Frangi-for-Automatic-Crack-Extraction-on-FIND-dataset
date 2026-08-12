"""Fixtures : un faux IRT-Crack, complet et minuscule, écrit sur disque.

Aucun test n'a besoin du vrai jeu de données ni d'un GPU. Le faux jeu reproduit
tout ce qui compte : arborescence à quatre dossiers, thermique en fausses
couleurs JET, masques ``.jpg`` face à des images ``.png``, et une fissure
réellement présente dans les deux modalités.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_residual.thermal_decode import colormap_palette  # noqa: E402

HEIGHT, WIDTH = 48, 64
SAMPLE_COUNT = 12


def _crack_mask(index: int) -> np.ndarray:
    """Un trait horizontal et un trait vertical, décalés selon l'échantillon."""

    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    row = 8 + (index * 3) % (HEIGHT - 16)
    column = 10 + (index * 5) % (WIDTH - 20)
    mask[row : row + 2, 5 : WIDTH - 5] = True
    mask[8 : HEIGHT - 8, column : column + 2] = True
    return mask


def _encode_jet(scalar: np.ndarray) -> np.ndarray:
    colors, scalars = colormap_palette("jet")
    indices = np.clip(np.searchsorted(scalars, scalar.ravel()), 0, len(scalars) - 1)
    return (colors[indices].reshape(*scalar.shape, 3) * 255.0).round().astype(np.uint8)


@pytest.fixture(scope="session")
def fake_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Écrit un faux IRT-Crack et retourne sa racine."""

    root = tmp_path_factory.mktemp("irt-crack")
    visible = root / "01-Visible images"
    infrared = root / "02-Infrared images"
    truth = root / "04-Ground truth"
    for directory in (visible, infrared, truth):
        directory.mkdir(parents=True)

    rng = np.random.default_rng(20260812)
    for index in range(SAMPLE_COUNT):
        name = f"LAB{index:05d}"
        mask = _crack_mask(index)

        rgb = np.full((HEIGHT, WIDTH, 3), 150, dtype=np.float32)
        rgb += rng.normal(0.0, 6.0, rgb.shape)
        rgb[mask] -= 70.0
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(visible / f"{name}.png")

        # La fissure est CHAUDE : c'est la polarité « bright » après décodage.
        thermal = np.full((HEIGHT, WIDTH), 0.35, dtype=np.float32)
        thermal += rng.normal(0.0, 0.01, thermal.shape).astype(np.float32)
        thermal[mask] += 0.45
        Image.fromarray(_encode_jet(np.clip(thermal, 0.0, 1.0))).save(infrared / f"{name}.png")

        Image.fromarray((mask * 255).astype(np.uint8)).save(truth / f"{name}.jpg", quality=100)

    return root


@pytest.fixture(scope="session")
def fake_manifest(fake_dataset: Path, tmp_path_factory: pytest.TempPathFactory):
    from thermal_residual.manifest import build_manifest

    return build_manifest(fake_dataset)


@pytest.fixture(scope="session")
def fake_split(fake_manifest):
    from thermal_residual.splits import build_split

    return build_split(fake_manifest, test_size=4, validation_fraction=0.25)
