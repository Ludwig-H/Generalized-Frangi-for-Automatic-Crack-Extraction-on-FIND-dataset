"""Bout en bout sur le faux jeu : caches → entraînement → évaluation — §13.10.

Aucun GPU, aucun SAM, aucune donnée réelle. Les logits « baseline » sont
fabriqués : une version dégradée du masque, où une partie de la fissure a été
effacée. La thermique, elle, la montre — c'est exactement la situation que le
correcteur est censé exploiter, en miniature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from thermal_residual.cache import (
    CacheWriter,
    extractor_digest,
    open_cache,
    validate_baseline_cache,
    validate_thermal_cache,
)
from thermal_residual.constants import (
    BASELINE_CACHE_VERSION,
    EVIDENCE_CHANNELS,
    SPLIT_TEST,
    THERMAL_CACHE_VERSION,
)
from thermal_residual.data import IRTResidualDataset, collate
from thermal_residual.evaluation import evaluate_arm, read_per_image_csv, write_per_image_csv
from thermal_residual.losses import LossWeights, corrector_loss
from thermal_residual.model import ThermalSignedAbstentionAdapter
from thermal_residual.thermal_frangi import (
    ThermalEvidenceConfig,
    generate_dual_polarity_thermal_evidence,
)
from thermal_residual.training import ArmSpecification, TrainingConfig, train_arm

THERMAL_CONFIG = ThermalEvidenceConfig(encoding="auto", scales=(1.0, 3.0, 5.0), R=3, tau=0.18)


@pytest.fixture(scope="module")
def caches(tmp_path_factory, fake_manifest):
    """Construit les deux caches, avec le vrai extracteur Frangi sur CPU."""

    root = tmp_path_factory.mktemp("caches")

    thermal_writer = CacheWriter(
        root / "thermal",
        schema_version=THERMAL_CACHE_VERSION,
        kind="thermal_evidence",
        parameters={**THERMAL_CONFIG.to_json(), "extractor_sha256": extractor_digest()},
        extra={"channels": list(EVIDENCE_CHANNELS)},
    )
    baseline_writer = CacheWriter(
        root / "baseline",
        schema_version=BASELINE_CACHE_VERSION,
        kind="baseline_logits",
        parameters={"checkpoint_sha256": "f" * 64},
    )

    rng = np.random.default_rng(4)
    from thermal_residual.data import load_mask

    for sample in fake_manifest:
        evidence = generate_dual_polarity_thermal_evidence(
            sample.thermal_path, device="cpu", config=THERMAL_CONFIG
        )
        decoding = evidence.pop("decoding")
        thermal_writer.write(
            sample.sample_id,
            {
                name: np.asarray(evidence[name], dtype=np.float32)
                for name in ("thermal_decoded", *EVIDENCE_CHANNELS)
            },
            {"source_thermal_sha256": sample.thermal_sha256, "decoding": decoding.to_json()},
        )

        mask = load_mask(sample.mask_path)
        degraded = mask.copy()
        degraded[:, degraded.shape[1] // 2 :] = 0.0  # la moitié droite est perdue
        logits = 6.0 * degraded - 3.0 + rng.normal(0.0, 0.2, degraded.shape).astype(np.float32)
        baseline_writer.write(
            sample.sample_id,
            {"baseline_logits": logits[None, ...].astype(np.float32)},
            {"source_rgb_sha256": sample.rgb_sha256},
        )

    thermal_writer.finalize()
    baseline_writer.finalize()
    return open_cache(root / "baseline"), open_cache(root / "thermal")


def test_les_caches_sont_valides(caches, fake_manifest) -> None:
    baseline, thermal = caches
    validate_baseline_cache(baseline, fake_manifest, checkpoint_sha256="f" * 64)
    validate_thermal_cache(thermal, fake_manifest, extractor_config=THERMAL_CONFIG.to_json())


def test_le_dataset_aligne_evidence_logits_et_masque(caches, fake_manifest) -> None:
    baseline, thermal = caches
    dataset = IRTResidualDataset(fake_manifest, baseline, thermal)
    item = dataset[0]
    assert item["baseline_logits"].shape == (1, 48, 64)
    assert item["thermal_evidence"].shape == (4, 48, 64)
    assert item["mask"].shape == (1, 48, 64)
    assert bool(item["modality_present"]) is True

    batch = collate([dataset[i] for i in range(3)])
    assert batch["baseline_logits"].shape == (3, 1, 48, 64)


def test_source_zeros_annule_l_evidence_sans_changer_la_forme(caches, fake_manifest) -> None:
    baseline, thermal = caches
    dataset = IRTResidualDataset(fake_manifest, baseline, thermal, evidence_source="zeros")
    item = dataset[0]
    assert item["thermal_evidence"].shape == (4, 48, 64)
    assert float(item["thermal_evidence"].abs().max()) == 0.0


def test_source_brute_a_la_meme_capacite(caches, fake_manifest) -> None:
    baseline, thermal = caches
    dataset = IRTResidualDataset(fake_manifest, baseline, thermal, evidence_source="raw_thermal")
    evidence = dataset[0]["thermal_evidence"]
    assert evidence.shape == (4, 48, 64)
    assert torch.isfinite(evidence).all()


def test_la_perte_diminue_et_le_mini_lot_est_sur_appris(caches, fake_manifest) -> None:
    baseline, thermal = caches
    dataset = IRTResidualDataset(fake_manifest[:4], baseline, thermal)
    batch = collate([dataset[index] for index in range(4)])

    model = ThermalSignedAbstentionAdapter()
    assert model.trainable_parameters() < 100_000
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    weights = LossWeights()

    history = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["baseline_logits"], batch["thermal_evidence"], batch["modality_present"]
        )
        loss, journal = corrector_loss(outputs, batch["mask"], weights)
        loss.backward()
        optimizer.step()
        history.append(journal["loss"])

    assert history[-1] < history[0], "la perte doit diminuer"
    assert history[-1] < 0.9 * history[0], "le mini-lot doit être sur-appris"
    assert batch["baseline_logits"].grad is None


def test_entrainement_complet_et_evaluation(caches, fake_manifest, fake_split, tmp_path: Path) -> None:
    baseline, thermal = caches
    trainable = [s for s in fake_manifest if fake_split.assignment[s.sample_id] != SPLIT_TEST]
    test_samples = [s for s in fake_manifest if fake_split.assignment[s.sample_id] == SPLIT_TEST]

    summary = train_arm(
        arm=ArmSpecification(name="frangi_signed_abstention"),
        samples=trainable,
        assignment=fake_split.assignment,
        baseline_cache=baseline,
        thermal_cache=thermal,
        training=TrainingConfig(max_epochs=4, batch_size=2, amp=False, early_stopping_patience=99),
        weights=LossWeights(),
        output_dir=tmp_path / "run",
        seed=13,
        device="cpu",
        provenance={"dataset_manifest_sha256": "test"},
    )
    assert summary["best_epoch"] >= 0
    assert (tmp_path / "run" / "best.pt").is_file()
    assert (tmp_path / "run" / "latest.pt").is_file()
    assert (tmp_path / "run" / "training.json").is_file()

    payload = torch.load(tmp_path / "run" / "best.pt", map_location="cpu", weights_only=False)
    assert "model_state_dict" in payload
    assert not any("sam" in key.lower() for key in payload["model_state_dict"]), (
        "SAM ne doit jamais être sérialisé dans le checkpoint du correcteur"
    )

    model = ThermalSignedAbstentionAdapter()
    model.load_state_dict(payload["model_state_dict"])
    result = evaluate_arm(
        samples=test_samples,
        baseline_cache=baseline,
        thermal_cache=thermal,
        model=model,
        evidence_source="frangi",
        permuted=False,
        assignment=fake_split.assignment,
        seed=13,
        device="cpu",
    )
    assert result["count"] == len(test_samples)
    row = result["rows"][0]
    for key in ("iou", "iou_buffered_tol3", "baseline_iou", "fraction_abstain", "residual_abs_mean"):
        assert key in row

    path = write_per_image_csv(tmp_path / "per_image.csv", result["rows"])
    table = read_per_image_csv(path)
    assert set(table) == {s.sample_id for s in test_samples}


def test_le_bras_baseline_reproduit_exactement_les_logits_caches(
    caches, fake_manifest, fake_split
) -> None:
    baseline, thermal = caches
    test_samples = [s for s in fake_manifest if fake_split.assignment[s.sample_id] == SPLIT_TEST]
    result = evaluate_arm(
        samples=test_samples,
        baseline_cache=baseline,
        thermal_cache=thermal,
        model=None,
        evidence_source="zeros",
        permuted=False,
        assignment=fake_split.assignment,
        seed=13,
        device="cpu",
    )
    for row in result["rows"]:
        assert row["iou"] == pytest.approx(row["baseline_iou"])
        assert row["residual_abs_max"] == 0.0
