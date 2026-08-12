"""Les sept configurations : capacité égale, hyperparamètres identiques.

« Aucun réglage par bras » est une prétention facile à écrire et facile à
violer. Ces tests la rendent mécanique : le bloc partagé doit être **identique à
l'octet près** dans les sept fichiers, et les bras ne peuvent différer que par ce
qui les définit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thermal_residual._repo import STUDY_ROOT
from thermal_residual.config import ConfigError, load_ablation_matrix, load_arm_config, parse_arm_config
from thermal_residual.constants import ARM_IDENTIFIERS
from thermal_residual.model import build_adapter

CONFIG_DIR = STUDY_ROOT / "configs"
SHARED_MARKER = "# Bloc partagé"


def _config_files() -> list[Path]:
    return sorted(p for p in CONFIG_DIR.glob("*.yaml") if not p.name.startswith("ablation_matrix"))


def test_tous_les_bras_declares_existent() -> None:
    configs = {load_arm_config(path).identifier for path in _config_files()}
    assert configs == set(ARM_IDENTIFIERS)


def test_bloc_partage_identique_a_l_octet_pres() -> None:
    blocks = {}
    for path in _config_files():
        text = path.read_text(encoding="utf-8")
        assert SHARED_MARKER in text, f"{path.name} n'a pas de bloc partagé"
        blocks[path.name] = text[text.index(SHARED_MARKER) :]
    distinct = set(blocks.values())
    assert len(distinct) == 1, (
        "les hyperparamètres diffèrent entre bras : " + ", ".join(sorted(blocks))
    )


def test_identifiant_et_nom_sont_coherents() -> None:
    for path in _config_files():
        config = load_arm_config(path)
        assert config.arm.name == ARM_IDENTIFIERS[config.identifier]


def test_capacite_identique_entre_les_bras_comparables() -> None:
    """A1, A2, A3 et A4 doivent avoir exactement le même nombre de paramètres."""

    counts = {}
    for path in _config_files():
        config = load_arm_config(path)
        counts[config.identifier] = build_adapter(config.arm.model).trainable_parameters()
    comparable = {key: counts[key] for key in ("A1", "A2", "A3", "A4")}
    assert len(set(comparable.values())) == 1, f"capacités différentes : {comparable}"
    assert all(value < 100_000 for value in counts.values())


def test_a3_ne_differe_de_a2_que_par_la_permutation() -> None:
    a2 = load_arm_config(CONFIG_DIR / "irt_signed_abstention_v1.yaml")
    a3 = load_arm_config(CONFIG_DIR / "irt_frangi_permuted.yaml")
    assert a2.arm.model == a3.arm.model
    assert a2.arm.evidence_source == a3.arm.evidence_source
    assert a2.arm.permuted is False and a3.arm.permuted is True
    assert a2.training.to_json() == a3.training.to_json()
    assert a2.loss.to_json() == a3.loss.to_json()


def test_a4_ne_differe_de_a2_que_par_la_representation() -> None:
    a2 = load_arm_config(CONFIG_DIR / "irt_signed_abstention_v1.yaml")
    a4 = load_arm_config(CONFIG_DIR / "irt_raw_thermal.yaml")
    assert a2.arm.model == a4.arm.model
    assert a2.arm.evidence_source == "frangi" and a4.arm.evidence_source == "raw_thermal"
    assert a2.arm.permuted == a4.arm.permuted is False


def test_a0_n_entraine_rien() -> None:
    a0 = load_arm_config(CONFIG_DIR / "irt_baseline.yaml")
    assert a0.arm.trained is False


def test_tolerance_primaire_gelee_a_trois_pixels() -> None:
    for path in _config_files():
        config = load_arm_config(path)
        assert config.loss.tolerant_radius == 3
        assert config.training.selection_metric == "iou_buffered_tol3"


def test_bornes_fixees_par_la_mesure_et_non_par_la_specification() -> None:
    """``delta_max`` et ``logit_clip`` viennent de la porte de plafond, pas du §5.

    Mesuré sur la validation d'IRT-Crack avec la baseline ``tol3`` gelée :
    ``|z₀|`` a pour médiane ``12,27`` et pour q99 ``16,90``. La valeur ``4,0``
    recommandée par la spécification laissait ``18,9 %`` des erreurs hors de
    portée, et son ``clip(z₀, −10, 10)`` saturait **plus d'un pixel sur deux**.
    """

    for path in _config_files():
        config = load_arm_config(path)
        assert config.arm.model["delta_max"] == 12.0
        assert config.arm.model["logit_clip"] == 20.0


def test_matrice_d_ablations_resout_les_chemins() -> None:
    protocol = load_ablation_matrix(CONFIG_DIR / "ablation_matrix.yaml")
    assert [entry["identifier"] for entry in protocol["arms"]] == ["A%d" % i for i in range(7)]
    assert protocol["seeds"] == [13, 37, 73]
    assert "A2-A1" in protocol["comparisons"]
    assert "A2-A3" in protocol["comparisons"]
    assert "A2-A4" in protocol["comparisons"]
    for entry in protocol["arms"]:
        assert Path(entry["config"]).is_file()


def test_configuration_invalide_rejetee() -> None:
    with pytest.raises(ConfigError, match="identifiant"):
        parse_arm_config({"identifier": "Z9"})
    with pytest.raises(ConfigError, match="doit s'appeler"):
        parse_arm_config({"identifier": "A2", "name": "autre_chose"})
    with pytest.raises(ConfigError, match="tête inconnue"):
        parse_arm_config({"identifier": "A2", "arm": {"model": {"head": "magique"}}})
    with pytest.raises(ConfigError, match="source d'évidence"):
        parse_arm_config({"identifier": "A2", "arm": {"evidence_source": "radar"}})


def test_empreinte_de_configuration_stable() -> None:
    first = load_arm_config(CONFIG_DIR / "irt_signed_abstention_v1.yaml")
    second = load_arm_config(CONFIG_DIR / "irt_signed_abstention_v1.yaml")
    assert first.digest() == second.digest()
    other = load_arm_config(CONFIG_DIR / "irt_raw_thermal.yaml")
    assert first.digest() != other.digest()


def test_a7_ne_differe_de_a2_que_par_la_ponderation() -> None:
    """Le correctif ne doit changer *que* la pondération de la perte."""

    a2 = load_arm_config(CONFIG_DIR / "irt_signed_abstention_v1.yaml")
    a7 = load_arm_config(CONFIG_DIR / "irt_frangi_difficulty_weighted.yaml")
    assert a2.arm.model == a7.arm.model
    assert a2.arm.evidence_source == a7.arm.evidence_source
    assert a2.arm.permuted == a7.arm.permuted is False
    assert a2.training.to_json() == a7.training.to_json()
    assert a2.loss.to_json() == a7.loss.to_json()
    assert a2.arm.difficulty_weighted is False and a7.arm.difficulty_weighted is True


def test_a8_est_le_controle_permute_de_a7() -> None:
    a7 = load_arm_config(CONFIG_DIR / "irt_frangi_difficulty_weighted.yaml")
    a8 = load_arm_config(CONFIG_DIR / "irt_frangi_difficulty_weighted_permuted.yaml")
    assert a7.arm.model == a8.arm.model
    assert a7.arm.difficulty_weighted == a8.arm.difficulty_weighted is True
    assert a7.arm.permuted is False and a8.arm.permuted is True


def test_matrice_du_correctif() -> None:
    protocol = load_ablation_matrix(CONFIG_DIR / "ablation_matrix_weighted.yaml")
    identifiers = [entry["identifier"] for entry in protocol["arms"]]
    assert "A7" in identifiers and "A8" in identifiers
    assert "A7-A8" in protocol["comparisons"], "le contrôle permuté doit être comparé"
    assert "A7-A1" in protocol["comparisons"]
