"""Caches de logits baseline et d'évidence thermique, avec provenance stricte.

Deux caches indépendants et reprenables :

``baseline``
    un ``.npz`` par image contenant ``baseline_logits`` ``float32[1,H,W]``, à la
    **résolution native** de l'image. CrackSAM 2 est chargé une seule fois, puis
    ne l'est plus jamais : l'entraînement du correcteur lit ce cache.

``thermal``
    un ``.npz`` par image contenant la thermique décodée et les quatre canaux
    Frangi double polarité.

Un cache est **refusé** — jamais réutilisé silencieusement — si le SHA-256 d'une
source a changé, si le checkpoint diffère, si la configuration Frangi diffère, ou
si l'ensemble des identifiants ne correspond plus au manifeste du jeu de données.
Chaque entrée est écrite atomiquement, donc une préemption Spot laisse un cache
partiel valide et non un cache corrompu.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ._repo import REPOSITORY_ROOT
from .constants import (
    BASELINE_CACHE_VERSION,
    EVIDENCE_CHANNELS,
    THERMAL_CACHE_VERSION,
)
from .manifest import IRTSample
from .provenance import (
    ProvenanceError,
    array_statistics,
    atomic_write_json,
    atomic_write_npz,
    base_manifest,
    load_npz,
    require,
    sha256_file,
    sha256_json,
)

MANIFEST_NAME = "manifest.json"

#: Fichiers dont le hachage identifie l'extracteur Frangi. Un changement dans
#: l'un d'eux invalide le cache thermique.
_EXTRACTOR_SOURCES: tuple[Path, ...] = (
    REPOSITORY_ROOT / "ISPRS" / "src" / "graph_extraction.py",
    REPOSITORY_ROOT / "ISPRS" / "src" / "frangi_hessian.py",
    REPOSITORY_ROOT / "ISPRS" / "CrackSAM" / "cracksam2" / "frangi.py",
    Path(__file__).resolve().parent / "thermal_frangi.py",
    Path(__file__).resolve().parent / "thermal_decode.py",
)


def extractor_digest() -> str:
    """SHA-256 combiné des sources de l'extracteur et du décodeur thermique."""

    return sha256_json(
        [[path.name, sha256_file(path)] for path in _EXTRACTOR_SOURCES if path.is_file()]
    )


def entry_path(root: str | Path, sample_id: str) -> Path:
    """Chemin du ``.npz`` d'un échantillon. Les identifiants sont plats."""

    safe = sample_id.replace("/", "__")
    return Path(root) / "entries" / f"{safe}.npz"


@dataclass(frozen=True)
class CacheHandle:
    """Un cache ouvert : sa racine et son manifeste relu."""

    root: Path
    manifest: dict[str, Any]

    @property
    def sample_ids(self) -> list[str]:
        return sorted(self.manifest.get("entries", {}))

    def entry(self, sample_id: str) -> dict[str, np.ndarray]:
        path = entry_path(self.root, sample_id)
        if not path.is_file():
            raise ProvenanceError(f"entrée de cache manquante : {path}")
        return load_npz(path)


def open_cache(manifest_path: str | Path) -> CacheHandle:
    """Ouvre un cache depuis son manifeste JSON."""

    import json

    path = Path(manifest_path)
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.is_file():
        raise ProvenanceError(f"manifeste de cache introuvable : {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CacheHandle(root=path.parent, manifest=payload)


# --------------------------------------------------------------------------- #
# Vérifications de provenance
# --------------------------------------------------------------------------- #


def _check_coverage(handle: CacheHandle, samples: Sequence[IRTSample]) -> None:
    entries = handle.manifest.get("entries", {})
    missing = [sample.sample_id for sample in samples if sample.sample_id not in entries]
    require(
        not missing,
        f"{len(missing)} échantillon(s) absent(s) du cache {handle.root} : {missing[:5]}",
    )


def validate_baseline_cache(
    handle: CacheHandle,
    samples: Sequence[IRTSample],
    *,
    checkpoint_sha256: str | None = None,
) -> None:
    """Refuse un cache de logits dont la provenance ne correspond plus."""

    require(
        int(handle.manifest.get("schema_version", -1)) == BASELINE_CACHE_VERSION,
        f"version de schéma inattendue pour le cache baseline : "
        f"{handle.manifest.get('schema_version')} (attendu {BASELINE_CACHE_VERSION})",
    )
    require(handle.manifest.get("kind") == "baseline_logits", "ce cache n'est pas un cache de logits")
    _check_coverage(handle, samples)

    parameters = handle.manifest.get("parameters", {})
    if checkpoint_sha256 is not None:
        require(
            parameters.get("checkpoint_sha256") == checkpoint_sha256,
            "le checkpoint CrackSAM du cache diffère de celui demandé "
            f"({parameters.get('checkpoint_sha256')} ≠ {checkpoint_sha256})",
        )
    entries = handle.manifest["entries"]
    for sample in samples:
        recorded = entries[sample.sample_id]
        require(
            recorded.get("source_rgb_sha256") == sample.rgb_sha256,
            f"« {sample.sample_id} » : l'image RGB source a changé depuis le cache",
        )


def validate_thermal_cache(
    handle: CacheHandle,
    samples: Sequence[IRTSample],
    *,
    extractor_config: Mapping[str, Any] | None = None,
    check_extractor_digest: bool = True,
) -> None:
    """Refuse un cache thermique dont la configuration ou les sources ont bougé."""

    require(
        int(handle.manifest.get("schema_version", -1)) == THERMAL_CACHE_VERSION,
        f"version de schéma inattendue pour le cache thermique : "
        f"{handle.manifest.get('schema_version')} (attendu {THERMAL_CACHE_VERSION})",
    )
    require(handle.manifest.get("kind") == "thermal_evidence", "ce cache n'est pas un cache thermique")
    _check_coverage(handle, samples)

    parameters = handle.manifest.get("parameters", {})
    if extractor_config is not None:
        stored = {key: parameters.get(key) for key in extractor_config}
        require(
            sha256_json(stored) == sha256_json(dict(extractor_config)),
            f"la configuration Frangi du cache diffère : {stored} ≠ {dict(extractor_config)}",
        )
    if check_extractor_digest:
        current = extractor_digest()
        recorded = parameters.get("extractor_sha256")
        require(
            recorded == current,
            "le code de l'extracteur a changé depuis la construction du cache "
            f"({recorded} ≠ {current})",
        )
    require(
        tuple(handle.manifest.get("channels", ())) == EVIDENCE_CHANNELS,
        "l'ordre des canaux d'évidence du cache ne correspond pas à l'ordre canonique",
    )
    entries = handle.manifest["entries"]
    for sample in samples:
        recorded = entries[sample.sample_id]
        require(
            recorded.get("source_thermal_sha256") == sample.thermal_sha256,
            f"« {sample.sample_id} » : la thermique source a changé depuis le cache",
        )


# --------------------------------------------------------------------------- #
# Écriture
# --------------------------------------------------------------------------- #


class CacheWriter:
    """Écrivain de cache reprenable : une entrée écrite n'est jamais recalculée."""

    def __init__(
        self,
        root: str | Path,
        *,
        schema_version: int,
        kind: str,
        parameters: Mapping[str, Any],
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "entries").mkdir(parents=True, exist_ok=True)
        self.manifest = base_manifest(schema_version, kind, parameters)
        self.manifest.update(dict(extra or {}))
        existing = self.root / MANIFEST_NAME
        if existing.is_file():
            import json

            previous = json.loads(existing.read_text(encoding="utf-8"))
            if previous.get("schema_version") == schema_version and sha256_json(
                previous.get("parameters", {})
            ) == sha256_json(dict(parameters)):
                self.manifest["entries"] = dict(previous.get("entries", {}))
                self.manifest["errors"] = list(previous.get("errors", []))

    def has(self, sample_id: str) -> bool:
        return sample_id in self.manifest["entries"] and entry_path(self.root, sample_id).is_file()

    def write(
        self,
        sample_id: str,
        arrays: Mapping[str, np.ndarray],
        metadata: Mapping[str, Any],
    ) -> Path:
        path = atomic_write_npz(entry_path(self.root, sample_id), arrays)
        record = dict(metadata)
        record["statistics"] = {
            name: array_statistics(array) for name, array in arrays.items()
        }
        record["shapes"] = {name: list(np.asarray(array).shape) for name, array in arrays.items()}
        self.manifest["entries"][sample_id] = record
        return path

    def record_error(self, sample_id: str, message: str) -> None:
        self.manifest["errors"].append({"sample_id": sample_id, "message": message})

    def finalize(self) -> Path:
        entries = self.manifest["entries"]
        self.manifest["count"] = len(entries)
        numeric: dict[str, list[float]] = {}
        for record in entries.values():
            for channel, statistics in record.get("statistics", {}).items():
                numeric.setdefault(channel, []).append(float(statistics.get("mean", 0.0)))
        self.manifest["statistics"] = {
            channel: {
                "mean_of_means": float(np.mean(values)),
                "min_of_means": float(np.min(values)),
                "max_of_means": float(np.max(values)),
            }
            for channel, values in numeric.items()
        }
        return atomic_write_json(self.root / MANIFEST_NAME, self.manifest)


def stack_evidence_from_entry(
    entry: Mapping[str, np.ndarray], channels: Iterable[str] = EVIDENCE_CHANNELS
) -> np.ndarray:
    """Empile les canaux d'une entrée dans l'ordre canonique."""

    return np.stack([np.asarray(entry[name], dtype=np.float32) for name in channels])


__all__ = [
    "MANIFEST_NAME",
    "CacheHandle",
    "CacheWriter",
    "entry_path",
    "extractor_digest",
    "open_cache",
    "stack_evidence_from_entry",
    "validate_baseline_cache",
    "validate_thermal_cache",
]
