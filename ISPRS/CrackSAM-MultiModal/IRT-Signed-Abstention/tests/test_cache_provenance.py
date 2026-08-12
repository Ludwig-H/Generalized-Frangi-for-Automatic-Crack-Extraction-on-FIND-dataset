"""Provenance des caches — §13.9.

Un cache est refusé, jamais réutilisé silencieusement, dès qu'une de ses sources
a bougé. Les quatre causes de refus exigées par la spécification sont testées
une par une.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from thermal_residual.cache import (
    CacheWriter,
    entry_path,
    extractor_digest,
    open_cache,
    validate_baseline_cache,
    validate_thermal_cache,
)
from thermal_residual.constants import BASELINE_CACHE_VERSION, EVIDENCE_CHANNELS, THERMAL_CACHE_VERSION
from thermal_residual.provenance import ProvenanceError, load_npz

THERMAL_PARAMETERS = {"encoding": "auto", "scales": [1.0, 3.0], "R": 3, "tau": 0.18, "K": 1}


@pytest.fixture()
def baseline_cache(tmp_path: Path, fake_manifest):
    writer = CacheWriter(
        tmp_path / "baseline",
        schema_version=BASELINE_CACHE_VERSION,
        kind="baseline_logits",
        parameters={"checkpoint_sha256": "a" * 64},
    )
    for sample in fake_manifest:
        writer.write(
            sample.sample_id,
            {"baseline_logits": np.zeros((1, sample.height, sample.width), np.float32)},
            {"source_rgb_sha256": sample.rgb_sha256},
        )
    writer.finalize()
    return open_cache(tmp_path / "baseline")


@pytest.fixture()
def thermal_cache(tmp_path: Path, fake_manifest):
    writer = CacheWriter(
        tmp_path / "thermal",
        schema_version=THERMAL_CACHE_VERSION,
        kind="thermal_evidence",
        parameters={**THERMAL_PARAMETERS, "extractor_sha256": extractor_digest()},
        extra={"channels": list(EVIDENCE_CHANNELS)},
    )
    for sample in fake_manifest:
        arrays = {name: np.zeros((sample.height, sample.width), np.float32) for name in EVIDENCE_CHANNELS}
        arrays["thermal_decoded"] = np.zeros((sample.height, sample.width), np.float32)
        writer.write(sample.sample_id, arrays, {"source_thermal_sha256": sample.thermal_sha256})
    writer.finalize()
    return open_cache(tmp_path / "thermal")


def test_cache_valide_accepte(baseline_cache, thermal_cache, fake_manifest) -> None:
    validate_baseline_cache(baseline_cache, fake_manifest, checkpoint_sha256="a" * 64)
    validate_thermal_cache(thermal_cache, fake_manifest, extractor_config=THERMAL_PARAMETERS)


def test_refus_si_l_image_source_a_change(baseline_cache, fake_manifest) -> None:
    altered = [replace(fake_manifest[0], rgb_sha256="b" * 64)] + list(fake_manifest[1:])
    with pytest.raises(ProvenanceError, match="a changé"):
        validate_baseline_cache(baseline_cache, altered)


def test_refus_si_le_checkpoint_differe(baseline_cache, fake_manifest) -> None:
    with pytest.raises(ProvenanceError, match="checkpoint"):
        validate_baseline_cache(baseline_cache, fake_manifest, checkpoint_sha256="c" * 64)


def test_refus_si_la_configuration_frangi_differe(thermal_cache, fake_manifest) -> None:
    with pytest.raises(ProvenanceError, match="configuration Frangi"):
        validate_thermal_cache(
            thermal_cache, fake_manifest, extractor_config={**THERMAL_PARAMETERS, "tau": 0.5}
        )


def test_refus_si_le_code_de_l_extracteur_a_change(tmp_path: Path, fake_manifest) -> None:
    writer = CacheWriter(
        tmp_path / "périmé",
        schema_version=THERMAL_CACHE_VERSION,
        kind="thermal_evidence",
        parameters={**THERMAL_PARAMETERS, "extractor_sha256": "0" * 64},
        extra={"channels": list(EVIDENCE_CHANNELS)},
    )
    for sample in fake_manifest:
        writer.write(
            sample.sample_id,
            {name: np.zeros((4, 4), np.float32) for name in EVIDENCE_CHANNELS},
            {"source_thermal_sha256": sample.thermal_sha256},
        )
    writer.finalize()
    handle = open_cache(tmp_path / "périmé")
    with pytest.raises(ProvenanceError, match="extracteur a changé"):
        validate_thermal_cache(handle, fake_manifest, extractor_config=THERMAL_PARAMETERS)
    validate_thermal_cache(
        handle, fake_manifest, extractor_config=THERMAL_PARAMETERS, check_extractor_digest=False
    )


def test_refus_si_un_echantillon_manque(baseline_cache, fake_manifest, tmp_path: Path) -> None:
    manifest_path = Path(baseline_cache.root) / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["entries"].pop(fake_manifest[0].sample_id)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="absent"):
        validate_baseline_cache(open_cache(manifest_path), fake_manifest)


def test_refus_si_l_ordre_des_canaux_change(thermal_cache, fake_manifest) -> None:
    manifest_path = Path(thermal_cache.root) / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["channels"] = list(reversed(payload["channels"]))
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="ordre des canaux"):
        validate_thermal_cache(open_cache(manifest_path), fake_manifest)


def test_reprise_ne_recalcule_pas(tmp_path: Path, fake_manifest) -> None:
    parameters = {"checkpoint_sha256": "a" * 64}
    first = CacheWriter(
        tmp_path / "reprise", schema_version=BASELINE_CACHE_VERSION, kind="baseline_logits", parameters=parameters
    )
    first.write(
        fake_manifest[0].sample_id,
        {"baseline_logits": np.ones((1, 4, 4), np.float32)},
        {"source_rgb_sha256": fake_manifest[0].rgb_sha256},
    )
    first.finalize()

    second = CacheWriter(
        tmp_path / "reprise", schema_version=BASELINE_CACHE_VERSION, kind="baseline_logits", parameters=parameters
    )
    assert second.has(fake_manifest[0].sample_id)
    assert not second.has(fake_manifest[1].sample_id)

    # Un changement de paramètres repart de zéro plutôt que de mélanger.
    third = CacheWriter(
        tmp_path / "reprise",
        schema_version=BASELINE_CACHE_VERSION,
        kind="baseline_logits",
        parameters={"checkpoint_sha256": "d" * 64},
    )
    assert not third.has(fake_manifest[0].sample_id)


def test_ecriture_sans_pickle_et_finie(tmp_path: Path) -> None:
    writer = CacheWriter(
        tmp_path / "npz", schema_version=BASELINE_CACHE_VERSION, kind="baseline_logits", parameters={}
    )
    with pytest.raises(ValueError, match="non finies"):
        writer.write("nan", {"baseline_logits": np.full((2, 2), np.nan, np.float32)}, {})
    writer.write("ok", {"baseline_logits": np.zeros((2, 2), np.float32)}, {})
    arrays = load_npz(entry_path(tmp_path / "npz", "ok"))
    assert arrays["baseline_logits"].shape == (2, 2)
