"""Appariement, fuite entre splits, déterminisme — §13.7."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from thermal_residual.manifest import (
    ManifestError,
    build_manifest,
    discover_layout,
    manifest_digest,
    read_manifest,
    read_official_split,
    write_manifest,
)
from thermal_residual.splits import assert_disjoint, build_split, read_split, write_split


def test_decouvre_les_quatre_dossiers(fake_dataset: Path) -> None:
    layout = discover_layout(fake_dataset)
    assert layout.visible.name.startswith("01")
    assert layout.infrared.name.startswith("02")
    assert layout.ground_truth.name.startswith("04")


def test_apparie_png_et_jpg(fake_manifest) -> None:
    assert len(fake_manifest) == 12
    for sample in fake_manifest:
        assert sample.rgb_path.suffix == ".png"
        assert sample.mask_path.suffix == ".jpg"
        assert sample.height > 0 and sample.width > 0
        assert len(sample.rgb_sha256) == 64


def test_detecte_un_fichier_manquant(fake_dataset: Path, tmp_path: Path) -> None:
    copy = tmp_path / "amputé"
    shutil.copytree(fake_dataset, copy)
    next(iter((copy / "02-Infrared images").glob("*.png"))).unlink()
    with pytest.raises(ManifestError, match="thermique manquante"):
        build_manifest(copy)


def test_refuse_un_masque_vide(fake_dataset: Path, tmp_path: Path) -> None:
    import numpy as np
    from PIL import Image

    copy = tmp_path / "masque-vide"
    shutil.copytree(fake_dataset, copy)
    target = sorted((copy / "04-Ground truth").glob("*.jpg"))[0]
    Image.fromarray(np.zeros((48, 64), dtype=np.uint8)).save(target)

    with pytest.raises(ManifestError, match="masque vide"):
        build_manifest(copy)
    assert len(build_manifest(copy, allow_empty_masks=True)) == 12


def test_refuse_une_racine_ambigue(fake_dataset: Path, tmp_path: Path) -> None:
    copy = tmp_path / "ambigu"
    shutil.copytree(fake_dataset, copy)
    source = sorted((copy / "01-Visible images").glob("*.png"))[0]
    shutil.copy(source, source.with_suffix(".bmp"))
    with pytest.raises(ManifestError, match="ambiguë"):
        build_manifest(copy)


def test_manifeste_invariant_a_l_ordre(fake_dataset: Path) -> None:
    first = build_manifest(fake_dataset)
    second = build_manifest(fake_dataset)
    assert [s.sample_id for s in first] == [s.sample_id for s in second]
    assert manifest_digest(first) == manifest_digest(second)


def test_aller_retour_csv(fake_manifest, tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "manifest.csv", fake_manifest)
    again = read_manifest(path)
    assert manifest_digest(again) == manifest_digest(fake_manifest)


def test_splits_disjoints_et_deterministes(fake_manifest) -> None:
    first = build_split(fake_manifest, test_size=4, validation_fraction=0.25)
    second = build_split(list(reversed(fake_manifest)), test_size=4, validation_fraction=0.25)
    assert_disjoint(first)
    assert first.assignment == second.assignment, "le split dépend de l'ordre d'énumération"
    counts = first.counts()
    assert counts["test"] == 4
    assert counts["train"] + counts["validation"] == 8
    assert counts["validation"] >= 1


def test_split_serialise(fake_split, tmp_path: Path) -> None:
    path = write_split(tmp_path / "split.json", fake_split)
    again = read_split(path)
    assert again.assignment == fake_split.assignment
    assert again.origin == fake_split.origin


def test_split_officiel_lu_depuis_des_listes(tmp_path: Path) -> None:
    directory = tmp_path / "00_List"
    directory.mkdir()
    (directory / "train_val.txt").write_text("a/LAB00001.png 0\nLAB00002.png\n", encoding="utf-8")
    (directory / "test.txt").write_text("LAB00003.png\n", encoding="utf-8")
    assignments = read_official_split(directory)
    assert assignments == {"LAB00001": "train", "LAB00002": "train", "LAB00003": "test"}


def test_split_officiel_contradictoire_rejete(tmp_path: Path) -> None:
    directory = tmp_path / "00_List"
    directory.mkdir()
    (directory / "train_val.txt").write_text("LAB00001.png\n", encoding="utf-8")
    (directory / "test.txt").write_text("LAB00001.png\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="plusieurs splits"):
        read_official_split(directory)


def test_split_absent_donne_un_dictionnaire_vide() -> None:
    assert read_official_split(None) == {}
