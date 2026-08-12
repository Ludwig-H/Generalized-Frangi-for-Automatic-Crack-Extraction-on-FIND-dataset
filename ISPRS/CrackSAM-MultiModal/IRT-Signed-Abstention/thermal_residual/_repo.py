"""Résolution des chemins du dépôt et importation des paquets maintenus.

Le README de spécification écrit ``from ISPRS.CrackSAM.cracksam2.frangi import …``.
Cette ligne **ne peut pas fonctionner** : ``CrackSAM``, ``CrackSAM-GeoLoRA`` et
``CrackSAM-MultiModal`` contiennent des tirets, qui sont interdits dans un nom de
module Python. Les paquets maintenus s'importent donc en insérant leur dossier
parent dans ``sys.path``, exactement comme le font déjà les tests de
``ISPRS/CrackSAM/tests``.

Ce module est le **seul** endroit du paquet qui touche à ``sys.path``.
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Final

#: Racine du dépôt : ``.../IRT-Signed-Abstention/thermal_residual/_repo.py``
#: remonte de cinq niveaux (thermal_residual → IRT-Signed-Abstention →
#: CrackSAM-MultiModal → ISPRS → racine).
REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[4]

#: Dossier autonome de cette étude.
STUDY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: Dossiers à insérer dans ``sys.path`` pour importer les paquets maintenus.
_IMPORT_ROOTS: Final[tuple[Path, ...]] = (
    REPOSITORY_ROOT,                          # ISPRS.src (paquet-espace-de-noms)
    REPOSITORY_ROOT / "ISPRS" / "CrackSAM",   # cracksam2
    REPOSITORY_ROOT / "ISPRS" / "CrackSAM-GeoLoRA",  # geolora
)


def ensure_repository_on_path() -> tuple[Path, ...]:
    """Insère les dossiers d'import du dépôt en tête de ``sys.path``.

    Idempotent : un dossier déjà présent n'est pas réinséré. Retourne les
    dossiers effectivement disponibles, dans l'ordre d'insertion.
    """

    available: list[Path] = []
    for root in _IMPORT_ROOTS:
        if not root.is_dir():
            continue
        available.append(root)
        text = str(root)
        if text not in sys.path:
            sys.path.insert(0, text)
    return tuple(available)


ensure_repository_on_path()


@lru_cache(maxsize=1)
def git_commit() -> str:
    """Commit courant du dépôt, ou ``"unknown"`` hors dépôt Git.

    La valeur est suffixée par ``+dirty`` quand l'arbre de travail contient des
    modifications non commitées : un artefact produit depuis un arbre sale ne
    doit pas se présenter comme reproductible.
    """

    try:
        head = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    if not head:
        return "unknown"
    try:
        status = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return head
    return f"{head}+dirty" if status else head


__all__ = [
    "REPOSITORY_ROOT",
    "STUDY_ROOT",
    "ensure_repository_on_path",
    "git_commit",
]
