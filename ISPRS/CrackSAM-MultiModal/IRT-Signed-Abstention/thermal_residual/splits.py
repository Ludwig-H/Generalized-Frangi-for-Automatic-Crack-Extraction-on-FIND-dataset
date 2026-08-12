"""Splits déterministes, stratifiés, et invariants à l'ordre d'énumération.

Deux niveaux :

1. **train / test.** Le split « officiel » 358/90 d'IRT-Crack est cité par le
   benchmark IRFusionFormer, mais il n'est distribué ni sur Zenodo ni dans le
   dépôt GitHub : il vit dans un dossier ``00_List`` sur un Google Drive. Quand
   il est fourni, il est utilisé tel quel et le split est marqué ``official``.
   Sinon un split ``derived`` de même effectif est construit ici, et le rapport
   doit dire lequel a servi — les chiffres publiés ne sont alors comparables
   qu'en ordre de grandeur.
2. **train interne / validation**, 80/20 dans le train, stratifié par strate
   horaire puis par quartile de fraction de pixels fissure.

Le déterminisme ne repose jamais sur ``random`` ni sur l'ordre du système de
fichiers : les échantillons d'une strate sont ordonnés par
``sha256(sel + sample_id)``, ce qui est stable d'une machine à l'autre.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from .constants import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION
from .manifest import IRTSample, ManifestError
from .provenance import atomic_write_json

#: Effectif de test annoncé par le benchmark ; utilisé par le split dérivé.
DEFAULT_TEST_SIZE = 90

#: Fraction de validation prélevée dans le train — §7.3.
DEFAULT_VALIDATION_FRACTION = 0.2

#: Sels fixes. Les changer change le split, donc invalide toute comparaison.
TEST_SALT = "irt-signed-abstention/test/v1"
VALIDATION_SALT = "irt-signed-abstention/validation/v1"


def _order_key(salt: str, sample_id: str) -> str:
    return hashlib.sha256(f"{salt}|{sample_id}".encode("utf-8")).hexdigest()


def crack_fraction(mask_path: str | Path, threshold: float = 0.5) -> float:
    """Fraction de pixels fissure d'un masque, dans ``[0, 1]``."""

    with Image.open(mask_path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return float((array > threshold).mean())


def compute_crack_fractions(samples: Iterable[IRTSample]) -> dict[str, float]:
    return {sample.sample_id: crack_fraction(sample.mask_path) for sample in samples}


def _quartile_labels(values: Mapping[str, float], identifiers: Sequence[str]) -> dict[str, int]:
    """Attribue un quartile à chaque identifiant, par rang et non par valeur.

    Le rang évite qu'une distribution très concentrée (beaucoup de fractions
    identiques) place tout le monde dans le même quartile.
    """

    ordered = sorted(identifiers, key=lambda name: (values.get(name, 0.0), name))
    labels: dict[str, int] = {}
    total = max(1, len(ordered))
    for rank, name in enumerate(ordered):
        labels[name] = min(3, (rank * 4) // total)
    return labels


def _stratified_take(
    identifiers: Sequence[str],
    strata: Mapping[str, tuple[str, int]],
    count: int,
    salt: str,
) -> list[str]:
    """Prélève ``count`` identifiants en respectant la composition des strates.

    Le quota par strate est proportionnel ; le reste est attribué aux strates
    dont la partie fractionnaire est la plus grande, ordre lexicographique en
    cas d'égalité. Aucun tirage aléatoire n'intervient.
    """

    if count <= 0:
        return []
    if count >= len(identifiers):
        return sorted(identifiers)

    groups: dict[tuple[str, int], list[str]] = {}
    for name in identifiers:
        groups.setdefault(strata[name], []).append(name)

    total = len(identifiers)
    exact = {key: count * len(members) / total for key, members in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = count - sum(quotas.values())
    ordering = sorted(
        groups, key=lambda key: (-(exact[key] - quotas[key]), str(key))
    )
    index = 0
    while remainder > 0 and ordering:
        key = ordering[index % len(ordering)]
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remainder -= 1
        index += 1
        if index > 4 * len(ordering) + count:  # garde-fou : strates saturées
            break

    selected: list[str] = []
    for key, members in groups.items():
        ordered = sorted(members, key=lambda name: _order_key(salt, name))
        selected.extend(ordered[: quotas[key]])

    # Complète si des strates étaient saturées.
    if len(selected) < count:
        remaining = sorted(
            set(identifiers) - set(selected), key=lambda name: _order_key(salt, name)
        )
        selected.extend(remaining[: count - len(selected)])
    return sorted(selected)


@dataclass(frozen=True)
class SplitAssignment:
    """Affectation figée de chaque échantillon à un split."""

    assignment: dict[str, str]
    origin: str  # "official" ou "derived"
    crack_fractions: dict[str, float]
    validation_fraction: float
    test_size: int

    def identifiers(self, split: str) -> list[str]:
        return sorted(name for name, value in self.assignment.items() if value == split)

    def counts(self) -> dict[str, int]:
        return {
            split: len(self.identifiers(split))
            for split in (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)
        }

    def to_json(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "validation_fraction": self.validation_fraction,
            "test_size": self.test_size,
            "test_salt": TEST_SALT,
            "validation_salt": VALIDATION_SALT,
            "counts": self.counts(),
            "assignment": dict(sorted(self.assignment.items())),
            "crack_fractions": {
                name: round(value, 8)
                for name, value in sorted(self.crack_fractions.items())
            },
        }


def build_split(
    samples: Sequence[IRTSample],
    *,
    crack_fractions: Mapping[str, float] | None = None,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    test_size: int = DEFAULT_TEST_SIZE,
) -> SplitAssignment:
    """Construit l'affectation train / validation / test."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction doit être dans (0, 1)")
    identifiers = [sample.sample_id for sample in samples]
    if len(set(identifiers)) != len(identifiers):
        raise ManifestError("identifiants dupliqués dans le manifeste")

    fractions = dict(crack_fractions) if crack_fractions else compute_crack_fractions(samples)
    quartiles = _quartile_labels(fractions, identifiers)
    strata = {
        sample.sample_id: (sample.time_stratum, quartiles[sample.sample_id])
        for sample in samples
    }

    official = {
        sample.sample_id: sample.official_split
        for sample in samples
        if sample.official_split in ("train", "test")
    }
    has_official = len(official) == len(identifiers)

    if has_official:
        origin = "official"
        test_identifiers = sorted(
            name for name, split in official.items() if split == "test"
        )
        pool = sorted(name for name, split in official.items() if split == "train")
    else:
        origin = "derived"
        test_identifiers = _stratified_take(identifiers, strata, test_size, TEST_SALT)
        pool = sorted(set(identifiers) - set(test_identifiers))

    validation_count = max(1, int(round(len(pool) * validation_fraction)))
    validation_identifiers = _stratified_take(
        pool, strata, validation_count, VALIDATION_SALT
    )

    assignment: dict[str, str] = {}
    for name in identifiers:
        if name in test_identifiers:
            assignment[name] = SPLIT_TEST
        elif name in validation_identifiers:
            assignment[name] = SPLIT_VALIDATION
        else:
            assignment[name] = SPLIT_TRAIN

    return SplitAssignment(
        assignment=assignment,
        origin=origin,
        crack_fractions=fractions,
        validation_fraction=float(validation_fraction),
        test_size=len(test_identifiers),
    )


def write_split(path: str | Path, split: SplitAssignment) -> Path:
    return atomic_write_json(path, split.to_json())


def read_split(path: str | Path) -> SplitAssignment:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SplitAssignment(
        assignment=dict(payload["assignment"]),
        origin=str(payload["origin"]),
        crack_fractions={k: float(v) for k, v in payload.get("crack_fractions", {}).items()},
        validation_fraction=float(payload["validation_fraction"]),
        test_size=int(payload["test_size"]),
    )


def assert_disjoint(split: SplitAssignment) -> None:
    """Vérifie qu'aucun identifiant n'appartient à deux splits."""

    seen: dict[str, str] = {}
    for name, value in split.assignment.items():
        if name in seen and seen[name] != value:
            raise ManifestError(f"« {name} » est affecté à {seen[name]} et {value}")
        seen[name] = value
    groups = {
        name: set(split.identifiers(name))
        for name in (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)
    }
    for left in groups:
        for right in groups:
            if left >= right:
                continue
            overlap = groups[left] & groups[right]
            if overlap:
                raise ManifestError(
                    f"intersection non vide entre {left} et {right} : {sorted(overlap)[:5]}"
                )


__all__ = [
    "DEFAULT_TEST_SIZE",
    "DEFAULT_VALIDATION_FRACTION",
    "SplitAssignment",
    "TEST_SALT",
    "VALIDATION_SALT",
    "assert_disjoint",
    "build_split",
    "compute_crack_fractions",
    "crack_fraction",
    "read_split",
    "write_split",
]
