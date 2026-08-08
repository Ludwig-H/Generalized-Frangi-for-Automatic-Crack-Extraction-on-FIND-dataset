"""Onze canaux d'évidence géométrique, gardés strictement séparés.

Contrainte C3 du pré-enregistrement : aucun produit, aucune moyenne géométrique
n'est calculé ici. Les canaux sont empilés tels quels et c'est l'arbitre, en
aval, qui décide. Le banc filtre-seul du 8 août 2026 a montré qu'un « ET »
multiplicatif transmet l'effondrement de Frangi sous l'ombre à toutes les
cartes ``verified_frangi_*`` (rétention ``≈10 %`` sur le phantom de traversée).

Les filtres proviennent de ``anti_shadow_filters`` — le code CPU validé de
l'étude, dont le port avait été contrôlé contre l'implémentation Torch d'origine
(corrélation de rang > 0,9999999999). On le réutilise sans le réécrire.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

import numpy as np

_STUDY = (
    Path(__file__).resolve().parents[2]
    / "CrackSAM"
    / "results"
    / "2026-08-08_guidage_geometrique_anti_ombre"
)
if str(_STUDY) not in sys.path:
    sys.path.insert(0, str(_STUDY))

import anti_shadow_filters as asf  # noqa: E402


#: Ordre canonique des canaux. Il est gelé : les vecteurs de features par
#: fragment et les checkpoints en dépendent.
CHANNELS: Final[tuple[str, ...]] = (
    "frangi_sim",   # 0 proposeur haute précision, sensible à l'ombre
    "ofs",          # 1 symétrie de flux orienté : veto de marche, abstention
    "ofa",          # 2 antisymétrie de flux : évidence POSITIVE de marche
    "even_odd",     # 3 paire/impair à même sigma
    "dbic",         # 4 BIC(H0) - BIC(H1), ligne contre marche
    "phase_sym",    # 5 symétrie de phase sombre, meilleur rappel squelette
    "abs_energy",   # 6 énergie en unités de bruit robuste (SNR, pas max image)
    "cos2theta",    # 7 orientation double-angle
    "sin2theta",    # 8 orientation double-angle
    "scale",        # 9 rayon gagnant OFS, en pixels
    "profile",      # 10 profil bilatéral vallée/marche
)

N_CHANNELS: Final[int] = len(CHANNELS)

#: Sources autorisées à contribuer au support candidat (§3.2 du DESIGN).
#: Volontairement plus large que Frangi seul : sur ``DJ_Wall_66`` l'historique
#: est dominé par le linteau sombre et manque la fissure que la voie sans Frangi
#: retrouve.
SUPPORT_SOURCES: Final[tuple[str, ...]] = ("frangi_sim", "phase_sym", "ofs", "dbic")

#: Échelle du canal ``scale`` : les rayons OFS explorés.
OFS_RADII: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Evidence:
    """Tenseur d'évidence ``(11, H, W)`` en float32, plus le bruit estimé."""

    planes: np.ndarray
    noise_sigma: float

    def __post_init__(self) -> None:
        if self.planes.ndim != 3 or self.planes.shape[0] != N_CHANNELS:
            raise ValueError(
                f"planes doit avoir la forme ({N_CHANNELS}, H, W), "
                f"reçu {self.planes.shape}"
            )

    def plane(self, name: str) -> np.ndarray:
        return self.planes[CHANNELS.index(name)]

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.planes.shape[1]), int(self.planes.shape[2]))


def _finite(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def compute_evidence(image: np.ndarray) -> Evidence:
    """Calcule les onze canaux pour une image RGB ``uint8`` ou float ``[0,1]``.

    Aucune normalisation par le maximum spatial n'est appliquée en dehors des
    deux contrôles historiques qui, par construction, la conservent :
    ``frangi_sim`` reproduit exactement la chaîne relative de juillet, ce qui en
    fait le comparateur pertinent et non une confiance absolue.
    """

    log_luminance = asf.log_luminance(image)
    noise_sigma = float(asf._noise_from_highpass(log_luminance))

    frangi_sim, _ = asf.historical_frangi_similarity_cpu(image)
    ofs, ofs_aux = asf.oriented_flux_symmetry(log_luminance)
    phase_sym, phase_aux = asf.dark_phase_symmetry(log_luminance)
    even_odd, _ = asf.even_odd_derivative_pair(log_luminance)
    profile, _ = asf.paired_profile(log_luminance)
    dbic, _ = asf.line_step_bic(log_luminance)

    # OFA : l'antisymétrie du flux au rayon gagnant. Une marche d'ombre doit
    # produire un OFA élevé et un OFS faible ; c'est ce couple qui autorise
    # l'action « retirer » plutôt que la seule abstention.
    ofa_raw = _finite(ofs_aux["ofs_antisymmetry"])
    ofa = _confidence_snr(ofa_raw, noise_sigma)

    # Énergie absolue : la courbure transverse sombre exprimée en unités du
    # plancher de bruit robuste, borné. Ce n'est PAS un rang dans l'image : le
    # rapport interdit la normalisation par le maximum spatial (§2.2), qui fait
    # passer « le meilleur élément d'une mauvaise image » pour une évidence
    # forte, mais demande explicitement d'« estimer un niveau de bruit robuste ».
    # Le MAD est un plancher, pas un maximum : une image sans structure reste
    # donc faible partout. C'est la même construction que celle employée à
    # l'intérieur d'``oriented_flux_symmetry`` pour son propre terme de bruit.
    dark_flux = _finite(ofs_aux["ofs_flux"])
    abs_energy = _confidence_snr(dark_flux, asf._robust_sigma(dark_flux))

    # Orientation en double angle : continue à travers theta = 0 ~ pi.
    theta = _finite(phase_aux["phase_orientation"])
    cos2theta = np.cos(2.0 * theta).astype(np.float32)
    sin2theta = np.sin(2.0 * theta).astype(np.float32)

    scale = _finite(ofs_aux["ofs_radius"])

    planes = np.stack(
        [
            _finite(frangi_sim),
            _finite(ofs),
            ofa,
            _finite(even_odd),
            _finite(dbic),
            _finite(phase_sym),
            abs_energy,
            cos2theta,
            sin2theta,
            scale,
            _finite(profile),
        ]
    ).astype(np.float32)
    return Evidence(planes=planes, noise_sigma=noise_sigma)


def _confidence_snr(raw: np.ndarray, noise_sigma: float, k: float = 2.5) -> np.ndarray:
    """Écrase une grandeur positive en ``[0,1]`` par rapport au bruit de l'image.

    ``x / (x + k*sigma_n)`` est monotone, vaut ``0`` pour un signal nul et ``0,5``
    à ``k`` écarts de bruit. Le niveau de référence est le bruit estimé sur le
    résidu haute fréquence, jamais le maximum spatial : une image pauvre ne peut
    donc pas fabriquer une confiance forte pour son « moins mauvais » pixel.
    """

    positive = np.maximum(_finite(raw), 0.0)
    scale = max(float(k) * float(noise_sigma), 1e-8)
    result = positive / (positive + scale)
    return _finite(result).clip(0.0, 1.0).astype(np.float32)


def stack_to_float16(evidence: Evidence) -> np.ndarray:
    """Compresse pour stockage. Les canaux sont bornés sauf ``scale``."""

    return evidence.planes.astype(np.float16)
