"""Décodage physique d'une thermique : niveaux de gris ou fausses couleurs JET.

`ISPRS/implementation_notes.md` pose la règle du dépôt : une thermique encodée en
JET **ne doit jamais** passer par une conversion standard en niveaux de gris. Le
vert médian y devient plus lumineux que le rouge maximal, ce qui inverse
localement l'ordre des températures et corrompt la Hessienne. Le décodage
correct est une recherche du plus proche élément de la palette.

Le module accepte ``encoding`` dans ``{"auto", "grayscale", "jet"}`` et rapporte
toujours de quoi auditer la décision : chroma, erreur de palette, percentiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image

#: Nombre d'échantillons de la palette de référence. 2048 dépasse largement les
#: 256 niveaux d'une image 8 bits : l'erreur de quantification vient de l'image,
#: pas de la table.
PALETTE_SAMPLES: Final[int] = 2048

#: Au-dessus de ce chroma (max-min des canaux, dans [0,1]) au 99ᵉ percentile,
#: ``auto`` conclut à une image en fausses couleurs.
CHROMA_THRESHOLD: Final[float] = 0.02

#: Erreur moyenne à la palette au-delà de laquelle un décodage JET est déclaré
#: douteux. Une image JPEG bien encodée reste très en dessous.
PALETTE_ERROR_WARNING: Final[float] = 0.06

ENCODINGS: Final[tuple[str, ...]] = ("auto", "grayscale", "jet")

EPSILON: Final[float] = 1e-6


class ThermalDecodeError(RuntimeError):
    """Le décodage thermique ne peut pas être effectué de façon sûre."""


@lru_cache(maxsize=4)
def colormap_palette(name: str = "jet", samples: int = PALETTE_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """Palette ``(couleurs RGB float32[N,3], scalaires float32[N])``."""

    import matplotlib

    colormap = matplotlib.colormaps[name]
    scalars = np.linspace(0.0, 1.0, int(samples), dtype=np.float64)
    colors = np.asarray(colormap(scalars))[:, :3]
    return colors.astype(np.float32), scalars.astype(np.float32)


@dataclass(frozen=True)
class ThermalDecoding:
    """Résultat d'un décodage, avec de quoi l'auditer."""

    scalar: np.ndarray
    """Valeur décodée brute, ``float32[H, W]`` dans ``[0, 1]``."""

    normalized: np.ndarray
    """Valeur normalisée robuste (percentiles 1 % / 99 %), ``float32[H, W]``."""

    encoding: str
    """Décodage réellement appliqué : ``"grayscale"`` ou ``"jet"``."""

    requested_encoding: str
    chroma_mean: float
    chroma_p99: float
    palette_error_mean: float
    palette_error_p99: float
    percentiles: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "requested_encoding": self.requested_encoding,
            "chroma_mean": self.chroma_mean,
            "chroma_p99": self.chroma_p99,
            "palette_error_mean": self.palette_error_mean,
            "palette_error_p99": self.palette_error_p99,
            "percentiles": dict(self.percentiles),
        }


def load_rgb(path: str | Path) -> np.ndarray:
    """Charge une image en ``float32[H, W, 3]`` RGB dans ``[0, 1]``.

    On passe par PIL, donc l'ordre des canaux est RGB. Utiliser ``cv2.imread``
    ici donnerait du BGR et ferait décoder la palette à l'envers.
    """

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return array.astype(np.float32) / 255.0


def _chroma(rgb: np.ndarray) -> np.ndarray:
    return rgb.max(axis=-1) - rgb.min(axis=-1)


def decode_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Moyenne des trois canaux d'une image réellement monochrome.

    Sur une image où ``R = G = B``, la moyenne est exacte et n'introduit aucune
    pondération perceptuelle arbitraire.
    """

    return rgb.mean(axis=-1).astype(np.float32)


def decode_colormap(rgb: np.ndarray, name: str = "jet") -> tuple[np.ndarray, np.ndarray]:
    """Inverse une palette par plus proche voisin.

    Retourne ``(scalaires float32[H, W], distances float32[H, W])``. La distance
    est euclidienne dans l'espace RGB ``[0,1]³`` : c'est le résidu de palette,
    et il sert de contrôle de qualité du décodage.
    """

    from scipy.spatial import cKDTree

    colors, scalars = colormap_palette(name)
    height, width = rgb.shape[:2]
    flat = np.ascontiguousarray(rgb.reshape(-1, 3), dtype=np.float32)
    distances, indices = cKDTree(colors).query(flat, k=1, workers=-1)
    decoded = scalars[np.asarray(indices, dtype=np.int64)].reshape(height, width)
    return decoded.astype(np.float32), np.asarray(distances, dtype=np.float32).reshape(
        height, width
    )


def robust_normalize(
    scalar: np.ndarray, low: float = 1.0, high: float = 99.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalisation robuste par percentiles, image par image — §4.1.

    ``clip((I − Q_low) / (Q_high − Q_low + ε), 0, 1)``. Une image constante
    donne un plan de zéros plutôt qu'une division par zéro.
    """

    values = np.asarray(scalar, dtype=np.float32)
    q_low, q_high = np.percentile(values, [low, high])
    span = float(q_high - q_low)
    normalized = np.clip((values - float(q_low)) / (span + EPSILON), 0.0, 1.0)
    statistics = {
        "p01": float(q_low),
        "p99": float(q_high),
        "p50": float(np.percentile(values, 50.0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "span": span,
    }
    return normalized.astype(np.float32), statistics


def decode_thermal(
    image: np.ndarray | str | Path,
    *,
    encoding: str = "auto",
    colormap: str = "jet",
) -> ThermalDecoding:
    """Décode une thermique et retourne la valeur brute et normalisée.

    ``encoding="auto"`` conclut au JET dès que le 99ᵉ percentile du chroma
    dépasse :data:`CHROMA_THRESHOLD`. Le résidu de palette est calculé dans tous
    les cas où la palette est inversée, et un résidu élevé signale une image qui
    n'a pas été encodée avec cette palette — cas qu'il faut traiter à la main
    plutôt que deviner.
    """

    if encoding not in ENCODINGS:
        raise ThermalDecodeError(
            f"encodage inconnu : {encoding!r} (attendu {ENCODINGS})"
        )

    rgb = load_rgb(image) if isinstance(image, (str, Path)) else np.asarray(image, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ThermalDecodeError(f"image thermique de forme inattendue : {rgb.shape}")
    rgb = rgb[..., :3]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    chroma = _chroma(rgb)
    chroma_mean = float(chroma.mean())
    chroma_p99 = float(np.percentile(chroma, 99.0))

    resolved = encoding
    if encoding == "auto":
        resolved = "jet" if chroma_p99 >= CHROMA_THRESHOLD else "grayscale"

    if resolved == "grayscale":
        scalar = decode_grayscale(rgb)
        palette_error = np.zeros_like(scalar)
    else:
        scalar, palette_error = decode_colormap(rgb, colormap)

    normalized, percentiles = robust_normalize(scalar)
    return ThermalDecoding(
        scalar=scalar,
        normalized=normalized,
        encoding=resolved,
        requested_encoding=encoding,
        chroma_mean=chroma_mean,
        chroma_p99=chroma_p99,
        palette_error_mean=float(palette_error.mean()),
        palette_error_p99=float(np.percentile(palette_error, 99.0)),
        percentiles=percentiles,
    )


def naive_opencv_grayscale(rgb: np.ndarray) -> np.ndarray:
    """La conversion **interdite**, fournie uniquement pour la démonstrer fausse.

    ``0.299 R + 0.587 G + 0.114 B`` est ce que fait ``cv2.cvtColor(...,
    COLOR_BGR2GRAY)``. Sur une palette JET elle rend le vert médian plus clair
    que le rouge maximal. Le test ``test_thermal_decode`` s'en sert pour mesurer
    la non-monotonie ; aucun code de production ne doit l'appeler.
    """

    weights = np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
    return (np.asarray(rgb, dtype=np.float32)[..., :3] * weights).sum(axis=-1)


__all__ = [
    "CHROMA_THRESHOLD",
    "ENCODINGS",
    "PALETTE_ERROR_WARNING",
    "PALETTE_SAMPLES",
    "ThermalDecodeError",
    "ThermalDecoding",
    "colormap_palette",
    "decode_colormap",
    "decode_grayscale",
    "decode_thermal",
    "load_rgb",
    "naive_opencv_grayscale",
    "robust_normalize",
]
