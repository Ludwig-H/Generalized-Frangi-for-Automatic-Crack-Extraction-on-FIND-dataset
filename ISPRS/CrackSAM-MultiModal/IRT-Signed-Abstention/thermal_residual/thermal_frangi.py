"""Évidence Frangi **double polarité** calculée sur la modalité thermique.

Une fissure peut apparaître plus chaude ou plus froide que son environnement
selon l'heure et l'inertie thermique de la chaussée. La version principale
n'impose donc aucune polarité : la similarité Frangi est calculée sur l'image
décodée *et* sur son complément, et les deux cartes sont fournies au correcteur,
qui décide.

L'extracteur n'est pas réécrit : c'est celui de ``cracksam2.frangi``, appelé avec
``compute_centrality=False`` et ``return_raster_features=False``. Ni MST ni
centralité dans ce MVP — c'est une contrainte explicite du protocole (§4.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import _repo  # noqa: F401  (insère les paquets maintenus dans sys.path)
from .constants import EVIDENCE_CHANNELS
from .thermal_decode import ThermalDecoding, decode_thermal

from cracksam2.frangi import extract_frangi_graph_gpu  # noqa: E402

#: Échelles par défaut, celles de CrackSAM 2 (``DEFAULT_FRANGI_SCALES``).
DEFAULT_SCALES: tuple[float, ...] = (1.0, 3.0, 5.0, 9.0, 15.0)
DEFAULT_R: int = 3
DEFAULT_TAU: float = 0.18


@dataclass(frozen=True)
class ThermalEvidenceConfig:
    """Paramètres gelés de l'extraction. Ils sont écrits dans le manifeste."""

    encoding: str = "auto"
    scales: tuple[float, ...] = DEFAULT_SCALES
    R: int = DEFAULT_R
    ss: float = 1.0
    si: float = 0.25
    sa: float = 0.3
    tau: float = DEFAULT_TAU
    min_rel_size: float = 120.0
    K: int = 1

    def to_json(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "scales": list(self.scales),
            "R": self.R,
            "ss": self.ss,
            "si": self.si,
            "sa": self.sa,
            "tau": self.tau,
            "min_rel_size": self.min_rel_size,
            "K": self.K,
            "compute_centrality": False,
            "return_raster_features": False,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ThermalEvidenceConfig":
        return cls(
            encoding=str(payload.get("encoding", "auto")),
            scales=tuple(float(value) for value in payload.get("scales", DEFAULT_SCALES)),
            R=int(payload.get("R", DEFAULT_R)),
            ss=float(payload.get("ss", 1.0)),
            si=float(payload.get("si", 0.25)),
            sa=float(payload.get("sa", 0.3)),
            tau=float(payload.get("tau", DEFAULT_TAU)),
            min_rel_size=float(payload.get("min_rel_size", 120.0)),
            K=int(payload.get("K", 1)),
        )


#: Seuil relatif de candidature des nœuds, codé en dur dans l'extracteur
#: (``τ_1 = max_S_global.max() * 0.01``, ``ISPRS/src/graph_extraction.py:150``).
CANDIDATE_RELATIVE_THRESHOLD: float = 0.01


def support_from_similarity(
    frangi_response: np.ndarray, similarity: np.ndarray, tau: float
) -> np.ndarray:
    """Reproduit ``tau_mask`` sans passer par le MST.

    L'extracteur ne remplit ``diagnostics["tau_mask"]`` que sur la branche
    ``compute_centrality=True`` : avec ``compute_centrality=False`` il renvoie un
    plan de zéros (``ISPRS/src/graph_extraction.py:287``). La spécification
    demandant à la fois le support **et** l'absence de MST, la règle de
    seuillage des nœuds est réimplémentée ici, à l'identique :

    1. les candidats sont les pixels où la réponse Frangi dépasse
       ``0,01 × max`` — c'est ``N`` ;
    2. les nœuds retenus sont ceux dont la similarité survit au seuillage des
       arêtes, donc ceux où ``similarity > 0`` ;
    3. si ces nœuds sont plus nombreux que ``max(1, int(N·τ))``, seuls les
       ``int(N·τ)`` plus fortes similarités sont conservées, à égalité incluse.

    Le test ``test_thermal_frangi.py::test_support_reproduit_tau_mask`` compare
    cette reconstruction au ``tau_mask`` réel obtenu avec ``compute_centrality=True``.
    """

    response = np.asarray(frangi_response, dtype=np.float32)
    values = np.asarray(similarity, dtype=np.float32)
    support = np.zeros(values.shape, dtype=np.float32)

    peak = float(response.max()) if response.size else 0.0
    if peak <= 0.0:
        return support
    candidate_count = int(np.count_nonzero(response > peak * CANDIDATE_RELATIVE_THRESHOLD))
    if candidate_count == 0:
        return support

    node_values = values[values > 0.0]
    if node_values.size == 0:
        return support

    keep = max(1, int(candidate_count * float(tau)))
    if node_values.size > keep:
        threshold = float(np.partition(node_values, -keep)[-keep])
        support[values >= threshold] = 1.0
        # ``values >= threshold`` inclut les pixels nuls si le seuil est nul.
        support[values <= 0.0] = 0.0
    else:
        support[values > 0.0] = 1.0
    return support


def _similarity(
    image: np.ndarray, config: ThermalEvidenceConfig, device: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Similarité Frangi et masque de support d'une image scalaire ``[0,1]``."""

    response, similarity, _, _, _ = extract_frangi_graph_gpu(
        {"thermal": np.asarray(image, dtype=np.float32)},
        {"thermal": 1.0},
        scales=config.scales,
        R=config.R,
        ss=config.ss,
        si=config.si,
        sa=config.sa,
        tau=config.tau,
        min_rel_size=config.min_rel_size,
        K=config.K,
        device=device,
        compute_centrality=False,
        return_raster_features=False,
    )
    similarity = np.asarray(similarity, dtype=np.float32)
    support = support_from_similarity(response, similarity, config.tau)
    return similarity, support


def generate_dual_polarity_thermal_evidence(
    thermal_image: np.ndarray,
    *,
    encoding: str = "auto",
    scales: Sequence[float] = DEFAULT_SCALES,
    R: int = DEFAULT_R,
    tau: float = DEFAULT_TAU,
    device: str | None = "cuda",
    config: ThermalEvidenceConfig | None = None,
) -> dict[str, np.ndarray]:
    """Retourne la thermique décodée et les quatre canaux Frangi recalés.

    Les clés retournées sont, dans cet ordre :
    ``thermal_decoded``, ``similarity_dark``, ``similarity_bright``,
    ``similarity_max``, ``support_union``. Toutes sont ``float32[H, W]``, finies,
    à la résolution de l'image d'entrée. ``support_union`` est binaire.

    Le dictionnaire contient en plus ``decoding`` : le
    :class:`~thermal_residual.thermal_decode.ThermalDecoding` complet, pour
    l'audit. Il n'est pas écrit dans le ``.npz``.
    """

    resolved = config or ThermalEvidenceConfig(
        encoding=encoding, scales=tuple(float(s) for s in scales), R=int(R), tau=float(tau)
    )
    decoding: ThermalDecoding = decode_thermal(thermal_image, encoding=resolved.encoding)
    normalized = decoding.normalized

    similarity_dark, support_dark = _similarity(normalized, resolved, device)
    similarity_bright, support_bright = _similarity(1.0 - normalized, resolved, device)

    similarity_max = np.maximum(similarity_dark, similarity_bright)
    support_union = np.maximum(support_dark, support_bright).astype(np.float32)

    payload = {
        "thermal_decoded": normalized.astype(np.float32),
        "similarity_dark": similarity_dark,
        "similarity_bright": similarity_bright,
        "similarity_max": similarity_max,
        "support_union": support_union,
        "thermal_raw": decoding.scalar.astype(np.float32),
    }
    for name, array in payload.items():
        if not np.isfinite(array).all():
            raise ValueError(f"le canal « {name} » contient des valeurs non finies")
    payload["decoding"] = decoding  # type: ignore[assignment]
    return payload


def stack_evidence(payload: Mapping[str, np.ndarray]) -> np.ndarray:
    """Empile les quatre canaux dans l'ordre canonique :data:`EVIDENCE_CHANNELS`."""

    return np.stack([np.asarray(payload[name], dtype=np.float32) for name in EVIDENCE_CHANNELS])


def raw_thermal_evidence(thermal_decoded: np.ndarray) -> np.ndarray:
    """Contrôle A4 — la thermique brute, à capacité d'entrée identique (§8.2).

    ``[T, 1−T, 2·|T−0,5|, 1]``. Le quatrième canal est constant, exactement comme
    ``support_union`` peut l'être : le correcteur reçoit quatre plans dans les
    deux bras, donc le même nombre de paramètres d'entrée.
    """

    thermal = np.asarray(thermal_decoded, dtype=np.float32)
    return np.stack(
        (
            thermal,
            1.0 - thermal,
            2.0 * np.abs(thermal - 0.5),
            np.ones_like(thermal),
        )
    ).astype(np.float32)


__all__ = [
    "DEFAULT_R",
    "DEFAULT_SCALES",
    "DEFAULT_TAU",
    "ThermalEvidenceConfig",
    "generate_dual_polarity_thermal_evidence",
    "raw_thermal_evidence",
    "stack_evidence",
]
