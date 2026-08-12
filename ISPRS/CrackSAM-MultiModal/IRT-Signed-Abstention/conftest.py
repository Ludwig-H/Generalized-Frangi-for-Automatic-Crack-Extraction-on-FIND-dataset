"""Amorçage ``sys.path`` pour pytest.

Les dossiers parents de ce paquet contiennent des tirets, donc ``thermal_residual``
n'est pas importable depuis la racine du dépôt sans aide. Ce ``conftest.py``
permet d'écrire ::

    python -m pytest "ISPRS/CrackSAM-MultiModal/IRT-Signed-Abstention/tests"

depuis n'importe quel répertoire de travail.
"""

from __future__ import annotations

import sys
from pathlib import Path

_STUDY_ROOT = Path(__file__).resolve().parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))
