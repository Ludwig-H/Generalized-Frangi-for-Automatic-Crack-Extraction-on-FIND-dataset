"""CrackSAM-GFA — guidage géométrique de SAM 2 par arbitrage de fragments.

Les modules sont importés paresseusement : ``evidence`` tire ``scikit-image`` et
les filtres de l'étude anti-ombre, ``arbiter`` tire PyTorch. Un script qui n'a
besoin que de ``fragments`` ou ``oracles`` ne doit pas payer ces dépendances.
"""

from __future__ import annotations

__all__ = ["arbiter", "evidence", "features", "fragments", "oracles"]
