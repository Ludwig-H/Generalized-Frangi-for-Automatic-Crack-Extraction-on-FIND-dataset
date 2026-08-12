"""Jeu de données sur caches, et contrôle permuté.

Le correcteur ne voit jamais ni SAM ni une image : il lit des logits cachés, une
évidence cachée et un masque. Une époque coûte donc quelques secondes, ce qui
rend abordables sept bras et trois graines.

Le contrôle permuté (A3) est implémenté ici et non dans le modèle : c'est un
appariement, pas une architecture. Ses garanties, testées :

* la permutation ne franchit **jamais** une frontière de split ;
* elle reste dans la même strate horaire quand celle-ci est renseignée ;
* elle est sans point fixe dès que la strate compte au moins deux échantillons ;
* elle est re-tirée à chaque époque à l'entraînement, **figée** à l'évaluation et
  déterminée par la graine du run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .cache import CacheHandle, stack_evidence_from_entry
from .constants import (
    EVIDENCE_CHANNELS,
    EVIDENCE_SOURCE_FRANGI,
    EVIDENCE_SOURCE_RAW_THERMAL,
    EVIDENCE_SOURCE_ZEROS,
    EVIDENCE_SOURCES,
)
from .manifest import IRTSample
from .thermal_frangi import raw_thermal_evidence


# --------------------------------------------------------------------------- #
# Contrôle permuté
# --------------------------------------------------------------------------- #


def derangement(identifiers: Sequence[str], rng: np.random.Generator) -> dict[str, str]:
    """Permutation sans point fixe d'une liste d'identifiants.

    Un mélange suivi d'une rotation d'un cran ne laisse aucun point fixe et ne
    demande aucun rejet. Une liste d'un seul élément ne peut évidemment pas être
    dérangée : elle est rendue à l'identique, et le test le sait.
    """

    names = list(identifiers)
    if len(names) < 2:
        return {name: name for name in names}
    order = list(rng.permutation(len(names)))
    return {names[order[i]]: names[order[(i + 1) % len(order)]] for i in range(len(order))}


class EvidencePermutation:
    """Appariement ``sample_id → thermal_sample_id`` du bras permuté."""

    def __init__(
        self,
        samples: Sequence[IRTSample],
        assignment: Mapping[str, str],
        *,
        seed: int,
        respect_strata: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.respect_strata = bool(respect_strata)
        self._groups: dict[tuple[str, str], list[str]] = {}
        for sample in samples:
            split = assignment[sample.sample_id]
            stratum = sample.time_stratum if respect_strata else "all"
            self._groups.setdefault((split, stratum), []).append(sample.sample_id)
        for members in self._groups.values():
            members.sort()

    def mapping(self, epoch: int) -> dict[str, str]:
        """Appariement pour une époque donnée. Déterministe en ``(seed, epoch)``."""

        result: dict[str, str] = {}
        for index, (key, members) in enumerate(sorted(self._groups.items())):
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, int(epoch), index])
            )
            result.update(derangement(members, rng))
        return result


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FlipAugmentation:
    """Retournements horizontaux et verticaux, appliqués de façon covariante.

    Aucune augmentation photométrique : elle détruirait la polarité thermique.
    Aucune rotation arbitraire : elle exigerait une interpolation, donc un
    rééchantillonnage du masque.
    """

    horizontal: bool = True
    vertical: bool = True

    def draw(self, rng: np.random.Generator) -> tuple[bool, bool]:
        return (
            bool(self.horizontal and rng.random() < 0.5),
            bool(self.vertical and rng.random() < 0.5),
        )

    @staticmethod
    def apply(array: np.ndarray, flip_horizontal: bool, flip_vertical: bool) -> np.ndarray:
        if flip_horizontal:
            array = array[..., ::-1]
        if flip_vertical:
            array = array[..., ::-1, :]
        return np.ascontiguousarray(array)


# --------------------------------------------------------------------------- #
# Jeu de données
# --------------------------------------------------------------------------- #


def load_mask(path: str | Path, threshold: float = 0.5) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return (array > threshold).astype(np.float32)


class IRTResidualDataset(Dataset):
    """Logits baseline, évidence et masque, tous cachés et alignés."""

    def __init__(
        self,
        samples: Sequence[IRTSample],
        baseline_cache: CacheHandle,
        thermal_cache: CacheHandle,
        *,
        evidence_source: str = EVIDENCE_SOURCE_FRANGI,
        permutation: EvidencePermutation | None = None,
        augmentation: FlipAugmentation | None = None,
        modality_present: bool = True,
        seed: int = 0,
        preload: bool = False,
        sample_weights: Mapping[str, float] | None = None,
    ) -> None:
        if evidence_source not in EVIDENCE_SOURCES:
            raise ValueError(
                f"source d'évidence inconnue : {evidence_source!r} (attendu {EVIDENCE_SOURCES})"
            )
        self.samples = list(samples)
        self.by_id = {sample.sample_id: sample for sample in self.samples}
        self.baseline_cache = baseline_cache
        self.thermal_cache = thermal_cache
        self.evidence_source = evidence_source
        self.permutation = permutation
        self.augmentation = augmentation
        self.modality_present = bool(modality_present)
        self.sample_weights = dict(sample_weights or {})
        self.seed = int(seed)
        self.epoch = 0
        self._mapping: dict[str, str] = {}
        self._memory: dict[str, dict[str, np.ndarray]] = {}
        self.set_epoch(0)
        if preload:
            for sample in self.samples:
                self._load_raw(sample.sample_id)

    # -- cycle de vie -------------------------------------------------- #

    def set_epoch(self, epoch: int) -> None:
        """Re-tire la permutation. À appeler avant chaque époque d'entraînement."""

        self.epoch = int(epoch)
        if self.permutation is not None:
            self._mapping = self.permutation.mapping(self.epoch)
        else:
            self._mapping = {}

    def evidence_source_id(self, sample_id: str) -> str:
        return self._mapping.get(sample_id, sample_id)

    def permutation_mapping(self) -> dict[str, str]:
        return dict(self._mapping)

    # -- lecture -------------------------------------------------------- #

    def _load_raw(self, sample_id: str) -> dict[str, np.ndarray]:
        cached = self._memory.get(sample_id)
        if cached is not None:
            return cached
        sample = self.by_id[sample_id]
        baseline = self.baseline_cache.entry(sample_id)["baseline_logits"]
        thermal = self.thermal_cache.entry(sample_id)
        payload = {
            "baseline_logits": np.asarray(baseline, dtype=np.float32),
            "thermal_decoded": np.asarray(thermal["thermal_decoded"], dtype=np.float32),
            "evidence": stack_evidence_from_entry(thermal),
            "mask": load_mask(sample.mask_path)[None, ...],
        }
        self._memory[sample_id] = payload
        return payload

    def _evidence_for(self, sample_id: str) -> np.ndarray:
        if self.evidence_source == EVIDENCE_SOURCE_ZEROS:
            reference = self._load_raw(sample_id)["evidence"]
            return np.zeros_like(reference)
        source_id = self.evidence_source_id(sample_id)
        payload = self._load_raw(source_id)
        if self.evidence_source == EVIDENCE_SOURCE_RAW_THERMAL:
            return raw_thermal_evidence(payload["thermal_decoded"])
        return payload["evidence"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        payload = self._load_raw(sample.sample_id)
        logits = payload["baseline_logits"]
        mask = payload["mask"]
        evidence = self._evidence_for(sample.sample_id)

        if evidence.shape[-2:] != logits.shape[-2:]:
            raise ValueError(
                f"« {sample.sample_id} » : évidence {evidence.shape[-2:]} et logits "
                f"{logits.shape[-2:]} ne sont pas alignés"
            )

        if self.augmentation is not None:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, index])
            )
            horizontal, vertical = self.augmentation.draw(rng)
            logits = self.augmentation.apply(logits, horizontal, vertical)
            evidence = self.augmentation.apply(evidence, horizontal, vertical)
            mask = self.augmentation.apply(mask, horizontal, vertical)

        return {
            "sample_id": sample.sample_id,
            "evidence_source_id": self.evidence_source_id(sample.sample_id),
            "baseline_logits": torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32)),
            "thermal_evidence": torch.from_numpy(np.ascontiguousarray(evidence, dtype=np.float32)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)),
            "modality_present": torch.tensor(self.modality_present, dtype=torch.bool),
            "sample_weight": torch.tensor(
                float(self.sample_weights.get(sample.sample_id, 1.0)), dtype=torch.float32
            ),
        }


def collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Assemble un lot en conservant les identifiants sous forme de listes."""

    return {
        "sample_id": [item["sample_id"] for item in batch],
        "evidence_source_id": [item["evidence_source_id"] for item in batch],
        "baseline_logits": torch.stack([item["baseline_logits"] for item in batch]),
        "thermal_evidence": torch.stack([item["thermal_evidence"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "modality_present": torch.stack([item["modality_present"] for item in batch]),
        "sample_weight": torch.stack([item["sample_weight"] for item in batch]),
    }


__all__ = [
    "EVIDENCE_CHANNELS",
    "EvidencePermutation",
    "FlipAugmentation",
    "IRTResidualDataset",
    "collate",
    "derangement",
    "load_mask",
]
