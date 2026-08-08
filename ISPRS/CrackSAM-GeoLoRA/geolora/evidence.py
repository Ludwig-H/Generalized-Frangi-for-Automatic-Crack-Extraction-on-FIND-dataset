"""Évidence géométrique **réaccordée à la largeur réelle des fissures**.

L'échec de CrackSAM-GFA avait une cause d'échelle : les filtres, hérités d'une
étude conçue pour des vallées de 1 à 3 px, étaient appliqués à des annotations
Khánh Hà de `19,1 px` de large. Ici les paramètres sont explicitement dérivés de
cette mesure.

Choix de résolution : l'évidence est calculée à **224 px**. Une fissure de
`19,1 px` à 448 y devient `9,6 px`, c'est-à-dire pile dans la plage où les
filtres de l'étude anti-ombre ont été validés. Calculer à 448 imposerait des
noyaux de Gabor de `143 × 143` px pour couvrir la même structure, soit deux
ordres de grandeur de coût en plus, sans information supplémentaire sur des
objets aussi épais.
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

#: Résolution de calcul de l'évidence.
EVIDENCE_SIZE: Final[int] = 224

#: Largeur médiane mesurée sur 400 scènes de test, à 448 px.
MEASURED_WIDTH_448: Final[float] = 19.1
#: La même largeur ramenée à la résolution de calcul.
TARGET_WIDTH: Final[float] = MEASURED_WIDTH_448 * EVIDENCE_SIZE / 448.0  # ≈ 9,6 px

# Les échelles couvrent des largeurs de structure d'environ 3 à 24 px à 224,
# soit 6 à 48 px à 448 : la largeur cible est au centre de la plage, et non
# à son extrémité comme c'était le cas dans CrackSAM-GFA.
FRANGI_SCALES: Final[tuple[float, ...]] = (1.5, 3.0, 5.0, 8.0, 12.0)
OFS_RADII: Final[tuple[int, ...]] = (2, 3, 4, 6, 8)
PHASE_WAVELENGTHS: Final[tuple[float, ...]] = (5.0, 8.0, 12.0, 18.0)
PROFILE_SIGMAS: Final[tuple[float, ...]] = (1.2, 2.0, 3.0, 4.5, 6.0)

CHANNELS: Final[tuple[str, ...]] = (
    "frangi_sim",
    "ofs",
    "ofa",
    "even_odd",
    "dbic",
    "phase_sym",
    "abs_energy",
    "cos2theta",
    "sin2theta",
    "scale",
    "profile",
)
N_CHANNELS: Final[int] = len(CHANNELS)


@dataclass(frozen=True)
class Evidence:
    planes: np.ndarray
    noise_sigma: float

    def plane(self, name: str) -> np.ndarray:
        return self.planes[CHANNELS.index(name)]


def _finite(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def _confidence_snr(raw: np.ndarray, scale: float, k: float = 2.5) -> np.ndarray:
    positive = np.maximum(_finite(raw), 0.0)
    denominator = max(float(k) * float(scale), 1e-8)
    return (
        _finite(positive / (positive + denominator)).clip(0.0, 1.0).astype(np.float32)
    )


def compute_evidence(image: np.ndarray) -> Evidence:
    """Onze canaux ``(11, 224, 224)``, jamais multipliés entre eux.

    ``image`` doit déjà être à ``EVIDENCE_SIZE``.
    """

    log_luminance = asf.log_luminance(image)
    noise_sigma = float(asf._noise_from_highpass(log_luminance))

    frangi_sim, _ = asf.historical_frangi_similarity_cpu(
        image, scales=FRANGI_SCALES, radius=3
    )
    ofs, ofs_aux = asf.oriented_flux_symmetry(log_luminance, radii=OFS_RADII)
    phase_sym, phase_aux = asf.dark_phase_symmetry(
        log_luminance, wavelengths=PHASE_WAVELENGTHS
    )
    even_odd, _ = asf.even_odd_derivative_pair(log_luminance, sigmas=PROFILE_SIGMAS)
    profile, _ = asf.paired_profile(log_luminance, sigmas=PROFILE_SIGMAS)
    dbic, _ = asf.line_step_bic(log_luminance, sigmas=PROFILE_SIGMAS)

    ofa = _confidence_snr(_finite(ofs_aux["ofs_antisymmetry"]), noise_sigma)
    dark_flux = _finite(ofs_aux["ofs_flux"])
    abs_energy = _confidence_snr(dark_flux, asf._robust_sigma(dark_flux))

    theta = _finite(phase_aux["phase_orientation"])
    planes = np.stack(
        [
            _finite(frangi_sim),
            _finite(ofs),
            ofa,
            _finite(even_odd),
            _finite(dbic),
            _finite(phase_sym),
            abs_energy,
            np.cos(2.0 * theta).astype(np.float32),
            np.sin(2.0 * theta).astype(np.float32),
            # Rayon gagnant ramené dans [0,1] : l'échelle doit entrer dans le
            # réseau comme une grandeur bornée, pas comme un nombre de pixels.
            (_finite(ofs_aux["ofs_radius"]) / float(max(OFS_RADII))).clip(0.0, 1.0),
            _finite(profile),
        ]
    ).astype(np.float32)
    return Evidence(planes=planes, noise_sigma=noise_sigma)
