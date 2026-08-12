"""Contrôle permuté — §13.8.

Un contrôle permuté mal fait est pire qu'aucun contrôle : il donne l'illusion
d'avoir testé la causalité. Les quatre propriétés vérifiées ici sont celles qui
peuvent silencieusement rendre le contrôle vide.
"""

from __future__ import annotations

import numpy as np

from thermal_residual.constants import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION
from thermal_residual.data import EvidencePermutation, derangement


def test_derangement_sans_point_fixe() -> None:
    rng = np.random.default_rng(0)
    names = [f"s{i}" for i in range(20)]
    mapping = derangement(names, rng)
    assert set(mapping) == set(names)
    assert set(mapping.values()) == set(names)
    assert all(source != target for source, target in mapping.items())


def test_derangement_impossible_sur_un_seul_element() -> None:
    rng = np.random.default_rng(0)
    assert derangement(["seul"], rng) == {"seul": "seul"}


def test_aucune_permutation_entre_splits(fake_manifest, fake_split) -> None:
    permutation = EvidencePermutation(fake_manifest, fake_split.assignment, seed=13)
    mapping = permutation.mapping(epoch=0)
    for source, target in mapping.items():
        assert fake_split.assignment[source] == fake_split.assignment[target], (
            f"« {source} » ({fake_split.assignment[source]}) est apparié à "
            f"« {target} » ({fake_split.assignment[target]})"
        )


def test_couverture_complete_des_trois_splits(fake_manifest, fake_split) -> None:
    permutation = EvidencePermutation(fake_manifest, fake_split.assignment, seed=13)
    mapping = permutation.mapping(epoch=0)
    assert set(mapping) == {sample.sample_id for sample in fake_manifest}
    for split in (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST):
        members = [s for s in mapping if fake_split.assignment[s] == split]
        if len(members) >= 2:
            assert all(mapping[name] != name for name in members)


def test_deterministe_a_graine_fixe(fake_manifest, fake_split) -> None:
    first = EvidencePermutation(fake_manifest, fake_split.assignment, seed=37).mapping(0)
    second = EvidencePermutation(fake_manifest, fake_split.assignment, seed=37).mapping(0)
    assert first == second


def test_change_avec_l_epoque_et_avec_la_graine(fake_manifest, fake_split) -> None:
    permutation = EvidencePermutation(fake_manifest, fake_split.assignment, seed=37)
    epochs = [permutation.mapping(epoch) for epoch in range(6)]
    assert any(epochs[0] != other for other in epochs[1:]), (
        "la permutation doit être re-tirée à chaque époque"
    )
    other_seed = EvidencePermutation(fake_manifest, fake_split.assignment, seed=73).mapping(0)
    assert other_seed != epochs[0]


def test_reste_dans_la_strate_horaire(fake_manifest, fake_split) -> None:
    from dataclasses import replace

    strata = ("morning", "evening")
    labelled = [
        replace(sample, time_stratum=strata[index % 2])
        for index, sample in enumerate(fake_manifest)
    ]
    by_id = {sample.sample_id: sample.time_stratum for sample in labelled}
    mapping = EvidencePermutation(labelled, fake_split.assignment, seed=13).mapping(0)
    for source, target in mapping.items():
        assert by_id[source] == by_id[target]
