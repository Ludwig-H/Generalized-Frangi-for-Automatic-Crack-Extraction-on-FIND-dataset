"""Provenance : hachage, écriture atomique, manifestes de cache.

Règles héritées du dépôt (``cracksam2/frangi.py``, ``graph_cache.py``) :

* aucun ``pickle`` — les tableaux sont écrits en ``.npz`` avec
  ``allow_pickle=False`` et relus de même ;
* toute écriture passe par un fichier temporaire puis ``os.replace``, pour
  qu'une préemption Spot ne laisse jamais un artefact tronqué ;
* un cache porte le SHA-256 de chacune de ses sources ; un cache dont une source
  a changé est **refusé**, jamais réutilisé silencieusement.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._repo import git_commit

_HASH_CHUNK = 1 << 20


class ProvenanceError(RuntimeError):
    """Un artefact ne correspond pas à la provenance déclarée."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    """SHA-256 hexadécimal du contenu d'un fichier."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    """SHA-256 d'une structure JSON, sérialisée de façon canonique."""

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> Path:
    """Écrit ``payload`` de façon atomique, avec ``fsync`` avant le renommage."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def atomic_write_npz(
    path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray]
) -> Path:
    """Écrit un ``.npz`` compressé, sans pickle, après contrôle de finitude."""

    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError(f"le cache doit être un .npz : {destination}")
    prepared: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        value = np.asarray(array)
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"refus d'écrire le canal « {name} » : valeurs non finies")
        prepared[name] = value

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp.npz"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **prepared)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def load_npz(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Relit un ``.npz`` en interdisant explicitement le dépicklage."""

    with np.load(path, allow_pickle=False) as handle:
        return {name: np.asarray(handle[name]) for name in handle.files}


def array_statistics(array: np.ndarray) -> dict[str, float]:
    """Statistiques min/max/moyenne d'un canal, pour le manifeste."""

    values = np.asarray(array, dtype=np.float64)
    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def base_manifest(schema_version: int, kind: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Squelette commun à tous les manifestes de cache."""

    return {
        "schema_version": int(schema_version),
        "kind": str(kind),
        "git_commit": git_commit(),
        "created_utc": utc_timestamp(),
        "parameters": dict(parameters),
        "entries": {},
        "errors": [],
        "statistics": {},
    }


def require(condition: bool, message: str) -> None:
    """Lève :class:`ProvenanceError` quand ``condition`` est fausse."""

    if not condition:
        raise ProvenanceError(message)


__all__ = [
    "ProvenanceError",
    "array_statistics",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_npz",
    "base_manifest",
    "load_npz",
    "require",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "utc_timestamp",
]
