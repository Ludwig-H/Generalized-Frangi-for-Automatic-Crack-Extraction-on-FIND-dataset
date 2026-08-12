"""Métriques : équivalence prouvée avec l'implémentation GeoLoRA, et sanité.

``05_tolerant_iou.py`` est un script numéroté, pas un module importable. Plutôt
que de dupliquer sa définition sans filet, on la charge **par chemin** et on
compare les deux implémentations sur des masques tirés au hasard : la
duplication devient une équivalence vérifiée.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from thermal_residual._repo import REPOSITORY_ROOT
from thermal_residual.metrics import (
    action_statistics,
    aggregate,
    cldice,
    dilate,
    image_metrics,
    tolerant_scores,
    topology,
)

_SCRIPT = REPOSITORY_ROOT / "ISPRS" / "CrackSAM-GeoLoRA" / "scripts" / "05_tolerant_iou.py"


def _load_reference():
    if not _SCRIPT.is_file():
        pytest.skip(f"script de référence absent : {_SCRIPT}")
    spec = importlib.util.spec_from_file_location("geolora_tolerant_iou", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _random_pair(seed: int, size: tuple[int, int] = (48, 56)):
    rng = np.random.default_rng(seed)
    truth = np.zeros(size, dtype=bool)
    row = rng.integers(4, size[0] - 6)
    truth[row : row + 2, 4 : size[1] - 4] = True
    prediction = np.zeros(size, dtype=bool)
    shift = int(rng.integers(-3, 4))
    prediction[row + shift : row + shift + 3, 6 : size[1] - 2] = True
    if rng.random() < 0.3:
        prediction[rng.integers(0, size[0]), rng.integers(0, size[1])] = True
    return prediction, truth


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("radius", [0, 1, 3, 5])
def test_equivalence_avec_le_script_geolora(seed: int, radius: int) -> None:
    reference = _load_reference()
    prediction, truth = _random_pair(seed)
    ours = tolerant_scores(prediction, truth, radius)
    theirs = reference.tolerant_scores(prediction, truth, radius)
    assert set(ours) == set(theirs)
    for key in ours:
        assert ours[key] == pytest.approx(theirs[key], abs=1e-12), key


def test_dilatation_euclidienne_et_non_carree() -> None:
    mask = np.zeros((21, 21), dtype=bool)
    mask[10, 10] = True
    dilated = dilate(mask, 3)
    assert dilated[10, 13] and dilated[13, 10]
    assert not dilated[13, 13], "un coin à distance 4,24 ne doit pas être inclus"


def test_cas_vides_explicites() -> None:
    empty = np.zeros((8, 8), dtype=bool)
    full = np.ones((8, 8), dtype=bool)
    assert tolerant_scores(empty, empty, 3)["iou_buffered"] == 1.0
    assert tolerant_scores(full, empty, 3)["iou_buffered"] == 0.0
    assert tolerant_scores(empty, full, 3)["iou_buffered"] == 0.0
    assert cldice(empty, empty) == 1.0
    assert cldice(empty, full) == 0.0


def test_tolerance_pardonne_le_placement() -> None:
    """Un décalage de 2 px est effacé dès ``k = 2`` — §3.2 du rapport GeoLoRA."""

    truth = np.zeros((20, 60), dtype=bool)
    truth[10, 5:55] = True
    shifted = np.zeros_like(truth)
    shifted[12, 5:55] = True

    assert tolerant_scores(shifted, truth, 0)["iou_buffered"] < 0.3
    assert tolerant_scores(shifted, truth, 2)["iou_buffered"] == pytest.approx(1.0)


def test_la_tolerance_ne_penalise_qu_une_rupture_plus_large_qu_elle() -> None:
    """Caractérisation exacte, et elle nuance le rapport GeoLoRA.

    Ce rapport écrit « une rupture reste pénalisée à toutes les tolérances ».
    Ce n'est vrai **que** pour une rupture plus large que la tolérance : les
    pixels d'une rupture de 4 px ont tous une prédiction à moins de 5 px, donc
    ``iou_buffered`` vaut exactement 1 à ``k = 5``. La propriété utile est donc
    « une rupture plus large que ``k`` reste pénalisée », et c'est elle qui rend
    la métrique capable de juger la continuité au bon ordre de grandeur.
    """

    truth = np.zeros((20, 60), dtype=bool)
    truth[10, 5:55] = True

    narrow = truth.copy()
    narrow[10, 28:32] = False  # rupture de 4 px, plus courte que k = 5
    assert tolerant_scores(narrow, truth, 5)["iou_buffered"] == pytest.approx(1.0)
    assert tolerant_scores(narrow, truth, 0)["iou_buffered"] < 1.0

    wide = truth.copy()
    wide[10, 20:40] = False  # rupture de 20 px, bien plus large que k
    for radius in (0, 1, 3, 5, 8):
        assert tolerant_scores(wide, truth, radius)["iou_buffered"] < 1.0


def test_topologie_compte_les_composantes() -> None:
    prediction = np.zeros((20, 20), dtype=bool)
    prediction[2:5, 2:8] = True
    prediction[12:15, 12:18] = True
    truth = np.zeros((20, 20), dtype=bool)
    truth[2:5, 2:18] = True
    values = topology(prediction, truth)
    assert values["components_pred"] == 2
    assert values["components_true"] == 1
    assert 0.0 < values["skeleton_covered"] < 1.0


def test_image_metrics_complet_et_coherent() -> None:
    prediction, truth = _random_pair(0)
    values = image_metrics(prediction.astype(np.float32), truth.astype(np.float32))
    assert values["iou"] == pytest.approx(values["iou_buffered_tol0"])
    assert values["iou_buffered_tol3"] >= values["iou_buffered_tol0"]
    assert 0.0 <= values["cldice"] <= 1.0
    assert values["crack_fraction_true"] > 0.0


def test_statistiques_d_actions() -> None:
    actions = np.array([[0, 0, 1], [2, 2, 2]])
    residual = np.array([[2.0, -1.0, 0.0], [0.0, 0.0, 0.5]])
    values = action_statistics(actions, residual)
    assert values["fraction_reinforce"] == pytest.approx(2 / 6)
    assert values["fraction_abstain"] == pytest.approx(3 / 6)
    assert values["residual_abs_max"] == pytest.approx(2.0)
    assert values["residual_nonzero_fraction"] == pytest.approx(3 / 6)


def test_agregation_ignore_les_colonnes_non_numeriques() -> None:
    rows = [{"sample_id": "a", "iou": 0.4}, {"sample_id": "b", "iou": 0.6}]
    assert aggregate(rows) == {"iou": pytest.approx(0.5)}
    assert aggregate([]) == {}
