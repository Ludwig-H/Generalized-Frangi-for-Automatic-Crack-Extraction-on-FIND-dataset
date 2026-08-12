"""Manifeste IRT-Crack : découverte, appariement, validation, sérialisation.

Aucun script d'entraînement ne déduit un appariement d'un ordre de fichiers. Le
manifeste est construit une fois, validé, écrit en CSV, puis relu partout.

Contrôles bloquants (§7.2) : fichier manquant, identifiants dupliqués, deux
identifiants pointant sur le même fichier, masque vide non autorisé, dimensions
incohérentes entre modalités, échantillon présent dans plusieurs splits.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

from .constants import (
    FUSION_DIRECTORY_CANDIDATES,
    GROUND_TRUTH_DIRECTORY_CANDIDATES,
    IMAGE_SUFFIXES,
    INFRARED_DIRECTORY_CANDIDATES,
    TIME_STRATA,
    VISIBLE_DIRECTORY_CANDIDATES,
)
from .provenance import sha256_file


class ManifestError(RuntimeError):
    """Le jeu de données ne satisfait pas les contrats d'appariement."""


@dataclass(frozen=True)
class IRTSample:
    """Une paire RGB / thermique / masque d'IRT-Crack."""

    sample_id: str
    rgb_path: Path
    thermal_path: Path
    mask_path: Path
    official_split: str
    time_stratum: str
    height: int
    width: int
    rgb_sha256: str
    thermal_sha256: str
    mask_sha256: str

    def __post_init__(self) -> None:
        if self.time_stratum not in TIME_STRATA:
            raise ManifestError(
                f"strate horaire inconnue pour {self.sample_id} : {self.time_stratum!r}"
            )


MANIFEST_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(IRTSample))


# --------------------------------------------------------------------------- #
# Découverte de l'arborescence
# --------------------------------------------------------------------------- #


def _find_directory(root: Path, candidates: Sequence[str]) -> Path | None:
    """Trouve un sous-dossier par nom canonique, puis par recherche insensible.

    Les ré-hébergements d'IRT-Crack renomment les dossiers (tirets bas contre
    espaces, singulier contre pluriel). On accepte les variantes connues, puis
    on tombe sur une correspondance par préfixe numérique (``01``, ``02``, ``04``).
    """

    for name in candidates:
        candidate = root / name
        if candidate.is_dir():
            return candidate

    lowered = {entry.name.lower(): entry for entry in root.iterdir() if entry.is_dir()}
    for name in candidates:
        entry = lowered.get(name.lower())
        if entry is not None:
            return entry

    prefix = candidates[0].split("-")[0].split("_")[0]
    if prefix.isdigit():
        matches = sorted(
            entry for entry in root.iterdir() if entry.is_dir() and entry.name.startswith(prefix)
        )
        if len(matches) == 1:
            return matches[0]
    return None


@dataclass(frozen=True)
class DatasetLayout:
    """Les quatre dossiers d'IRT-Crack, une fois résolus."""

    root: Path
    visible: Path
    infrared: Path
    ground_truth: Path
    fusion: Path | None


def discover_layout(dataset_root: str | Path) -> DatasetLayout:
    """Résout l'arborescence d'IRT-Crack, en descendant un niveau si besoin."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"racine de jeu de données introuvable : {root}")

    search_roots = [root]
    search_roots.extend(sorted(entry for entry in root.iterdir() if entry.is_dir()))

    for candidate_root in search_roots:
        visible = _find_directory(candidate_root, VISIBLE_DIRECTORY_CANDIDATES)
        infrared = _find_directory(candidate_root, INFRARED_DIRECTORY_CANDIDATES)
        ground_truth = _find_directory(candidate_root, GROUND_TRUTH_DIRECTORY_CANDIDATES)
        if visible and infrared and ground_truth:
            return DatasetLayout(
                root=candidate_root,
                visible=visible,
                infrared=infrared,
                ground_truth=ground_truth,
                fusion=_find_directory(candidate_root, FUSION_DIRECTORY_CANDIDATES),
            )

    raise ManifestError(
        "impossible de trouver les dossiers visible / infrarouge / vérité terrain "
        f"sous {root}. Dossiers vus : "
        + ", ".join(sorted(entry.name for entry in root.iterdir() if entry.is_dir()))
    )


def _index_by_stem(directory: Path) -> dict[str, Path]:
    """Indexe un dossier d'images par racine de nom de fichier.

    Deux fichiers de même racine et d'extensions différentes (``LAB00162.png`` et
    ``LAB00162.jpg``) sont une ambiguïté bloquante : le contrat d'appariement
    interdit de choisir à la place de l'utilisateur.
    """

    index: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem
        if stem in index:
            raise ManifestError(
                f"racine de nom ambiguë dans {directory} : {index[stem].name} et {path.name}"
            )
        index[stem] = path
    return index


# --------------------------------------------------------------------------- #
# Split officiel
# --------------------------------------------------------------------------- #


def read_official_split(path: str | Path | None) -> dict[str, str]:
    """Lit un split officiel, s'il est disponible.

    Deux formats acceptés :

    * un dossier contenant ``train_val.txt`` et ``test.txt`` — c'est la forme
      distribuée par le benchmark IRFusionFormer (Google Drive, dossier
      ``00_List``) ; chaque ligne commence par un chemin ;
    * un JSON ``{"train": [...], "test": [...]}``.

    Retourne ``{sample_id: "train"|"test"}``. Un dictionnaire vide signifie
    « split officiel indisponible » et **n'est pas une erreur** : le split
    officiel d'IRT-Crack n'est pas publié sur Zenodo.
    """

    if path is None:
        return {}
    source = Path(path).expanduser()
    if not source.exists():
        raise ManifestError(f"split officiel introuvable : {source}")

    assignments: dict[str, str] = {}

    def assign(sample_id: str, split: str) -> None:
        previous = assignments.get(sample_id)
        if previous is not None and previous != split:
            raise ManifestError(
                f"« {sample_id} » apparaît dans plusieurs splits officiels : "
                f"{previous} et {split}"
            )
        assignments[sample_id] = split

    if source.is_dir():
        pairs = (("train_val.txt", "train"), ("test.txt", "test"))
        found = False
        for filename, split in pairs:
            listing = source / filename
            if not listing.is_file():
                continue
            found = True
            for line in listing.read_text(encoding="utf-8").splitlines():
                token = line.strip().split()[0] if line.strip() else ""
                if not token:
                    continue
                assign(Path(token).stem, split)
        if not found:
            raise ManifestError(
                f"{source} ne contient ni train_val.txt ni test.txt"
            )
        return assignments

    import json

    payload = json.loads(source.read_text(encoding="utf-8"))
    for split in ("train", "test"):
        for token in payload.get(split, []):
            assign(Path(str(token)).stem, split)
    return assignments


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return int(height), int(width)


def _mask_is_empty(path: Path, threshold: float = 0.5) -> bool:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return not bool((array > threshold).any())


def build_manifest(
    dataset_root: str | Path,
    *,
    official_split: str | Path | None = None,
    time_strata: dict[str, str] | None = None,
    allow_empty_masks: bool = False,
) -> list[IRTSample]:
    """Construit et valide le manifeste complet d'IRT-Crack."""

    layout = discover_layout(dataset_root)
    visible_index = _index_by_stem(layout.visible)
    infrared_index = _index_by_stem(layout.infrared)
    mask_index = _index_by_stem(layout.ground_truth)

    if not visible_index:
        raise ManifestError(f"aucune image visible dans {layout.visible}")

    official = read_official_split(official_split)
    strata = dict(time_strata or {})

    problems: list[str] = []
    samples: list[IRTSample] = []
    seen_paths: dict[Path, str] = {}

    for sample_id in sorted(visible_index):
        rgb_path = visible_index[sample_id]
        thermal_path = infrared_index.get(sample_id)
        mask_path = mask_index.get(sample_id)
        if thermal_path is None:
            problems.append(f"{sample_id}: thermique manquante dans {layout.infrared}")
            continue
        if mask_path is None:
            problems.append(f"{sample_id}: masque manquant dans {layout.ground_truth}")
            continue

        for role, path in (("rgb", rgb_path), ("thermal", thermal_path), ("mask", mask_path)):
            resolved = path.resolve()
            owner = seen_paths.get(resolved)
            if owner is not None and owner != sample_id:
                problems.append(
                    f"{sample_id}: le fichier {role} {resolved} est déjà utilisé par « {owner} »"
                )
            seen_paths[resolved] = sample_id

        rgb_size = _image_size(rgb_path)
        thermal_size = _image_size(thermal_path)
        mask_size = _image_size(mask_path)
        if not (rgb_size == thermal_size == mask_size):
            problems.append(
                f"{sample_id}: dimensions incohérentes rgb={rgb_size} "
                f"thermique={thermal_size} masque={mask_size}"
            )
            continue

        if _mask_is_empty(mask_path) and not allow_empty_masks:
            problems.append(f"{sample_id}: masque vide (utiliser --allow-empty-masks pour l'accepter)")
            continue

        samples.append(
            IRTSample(
                sample_id=sample_id,
                rgb_path=rgb_path,
                thermal_path=thermal_path,
                mask_path=mask_path,
                official_split=official.get(sample_id, "unknown"),
                time_stratum=strata.get(sample_id, "unknown"),
                height=rgb_size[0],
                width=rgb_size[1],
                rgb_sha256=sha256_file(rgb_path),
                thermal_sha256=sha256_file(thermal_path),
                mask_sha256=sha256_file(mask_path),
            )
        )

    orphans = sorted(set(infrared_index) - set(visible_index))
    problems.extend(f"{stem}: thermique sans image visible appariée" for stem in orphans)

    if problems:
        raise ManifestError(
            "manifeste invalide — "
            + f"{len(problems)} problème(s) :\n  - "
            + "\n  - ".join(problems[:40])
            + ("\n  - …" if len(problems) > 40 else "")
        )
    if not samples:
        raise ManifestError("aucun échantillon valide n'a été construit")
    return samples


# --------------------------------------------------------------------------- #
# Sérialisation
# --------------------------------------------------------------------------- #


def write_manifest(path: str | Path, samples: Iterable[IRTSample]) -> Path:
    """Écrit le manifeste en CSV, colonnes dans l'ordre de :class:`IRTSample`."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for sample in samples:
            row = asdict(sample)
            row["rgb_path"] = str(sample.rgb_path)
            row["thermal_path"] = str(sample.thermal_path)
            row["mask_path"] = str(sample.mask_path)
            writer.writerow(row)
    temporary.replace(destination)
    return destination


def read_manifest(path: str | Path) -> list[IRTSample]:
    """Relit un manifeste CSV et rejette toute colonne manquante."""

    source = Path(path)
    with open(source, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ManifestError(f"colonnes manquantes dans {source} : {sorted(missing)}")
        samples = [
            IRTSample(
                sample_id=row["sample_id"],
                rgb_path=Path(row["rgb_path"]),
                thermal_path=Path(row["thermal_path"]),
                mask_path=Path(row["mask_path"]),
                official_split=row["official_split"],
                time_stratum=row["time_stratum"],
                height=int(row["height"]),
                width=int(row["width"]),
                rgb_sha256=row["rgb_sha256"],
                thermal_sha256=row["thermal_sha256"],
                mask_sha256=row["mask_sha256"],
            )
            for row in reader
        ]
    identifiers = [sample.sample_id for sample in samples]
    duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
    if duplicates:
        raise ManifestError(f"identifiants dupliqués dans {source} : {duplicates}")
    return samples


def manifest_digest(samples: Sequence[IRTSample]) -> str:
    """Empreinte du manifeste : identifiants et hachages sources, dans l'ordre."""

    from .provenance import sha256_json

    return sha256_json(
        [
            [s.sample_id, s.rgb_sha256, s.thermal_sha256, s.mask_sha256]
            for s in samples
        ]
    )


def iter_samples(samples: Iterable[IRTSample]) -> Iterator[IRTSample]:
    yield from samples


__all__ = [
    "DatasetLayout",
    "IRTSample",
    "MANIFEST_COLUMNS",
    "ManifestError",
    "build_manifest",
    "discover_layout",
    "manifest_digest",
    "read_manifest",
    "read_official_split",
    "write_manifest",
]
