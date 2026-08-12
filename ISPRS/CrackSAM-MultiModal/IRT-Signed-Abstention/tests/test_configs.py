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
    return sorted(path for path in CONFIG_DIR.glob("*.yaml") if path.name != "ablation_matrix.yaml")


def test_les_sept_bras_existent() -> None:
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


def test_matrice_d_ablations_resout_les_chemins() -> None:
    protocol = load_ablation_matrix(CONFIG_DIR / "ablation_matrix.yaml")
    assert [entry["identifier"] for entry in protocol["arms"]] == sorted(ARM_IDENTIFIERS)
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
