"""Typed contract for the minimal Frangi raster cache (schema version 2)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Final

import numpy as np


FRANGI_RASTER_SCHEMA_VERSION: Final[int] = 2
FRANGI_RASTER_CHANNELS: Final[tuple[str, ...]] = (
    "similarity",
    "support",
    "hessian_magnitude",
    "winning_scale",
    "orientation_sin2",
    "orientation_cos2",
    "distance_to_skeleton",
)
FRANGI_RASTER_CHANNEL_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(FRANGI_RASTER_CHANNELS)
}
FRANGI_RASTER_CHANNEL_SEMANTICS: Final[dict[str, str]] = {
    "similarity": "Historical node_sim_max in [0, 1], without logit conversion.",
    "support": "Binary mask of graph nodes retained after edge and node pruning.",
    "hessian_magnitude": (
        "Spectral norm of the raw fused Hessian at the winning scale, multiplied "
        "by sigma^2 and measured before per-image maximum normalization."
    ),
    "winning_scale": (
        "Gaussian derivative scale whose positive Hessian response is largest "
        "after the extractor's per-scale spatial normalization; 0 means no "
        "positive response. It is not a calibrated crack width."
    ),
    "orientation_sin2": "sin(2 theta) at the winning scale; 0 means no positive response.",
    "orientation_cos2": "cos(2 theta) at the winning scale; 0 means no positive response.",
    "distance_to_skeleton": (
        "Euclidean distance to the accepted rasterized graph structure "
        "(MST edges when K=1), divided by the image diagonal; 1 everywhere "
        "when no skeleton exists."
    ),
}
FRANGI_GATE_FEATURE_SEMANTICS: Final[dict[str, str]] = {
    "similarity": (
        "Mean historical node_sim_max on retained graph support, in [0, 1]; "
        "0 when support is empty."
    ),
    "strength": (
        "Mean absolute scale-normalized Hessian magnitude on retained graph "
        "support; 0 when support is empty."
    ),
    "density": "Fraction of input pixels belonging to retained graph support.",
    "stability": (
        "Unavailable in schema v2-minimal: the extractor does not yet measure "
        "normal displacement or persistence across scales."
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _validate_sha256(value: str, field: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256 digest")


def validate_frangi_raster(raster: np.ndarray) -> None:
    """Validate numerical and semantic invariants of a seven-channel raster."""
    if not isinstance(raster, np.ndarray):
        raise TypeError("raster must be a NumPy array")
    if raster.dtype != np.float32:
        raise ValueError(f"raster must use float32, got {raster.dtype}")
    if raster.ndim != 3 or raster.shape[0] != len(FRANGI_RASTER_CHANNELS):
        raise ValueError(
            "raster must have shape "
            f"({len(FRANGI_RASTER_CHANNELS)}, height, width), got {raster.shape}"
        )
    if raster.shape[1] <= 0 or raster.shape[2] <= 0:
        raise ValueError("raster height and width must be positive")
    if not np.isfinite(raster).all():
        raise ValueError("raster contains NaN or infinity")

    similarity = raster[FRANGI_RASTER_CHANNEL_INDEX["similarity"]]
    support = raster[FRANGI_RASTER_CHANNEL_INDEX["support"]]
    magnitude = raster[FRANGI_RASTER_CHANNEL_INDEX["hessian_magnitude"]]
    scale = raster[FRANGI_RASTER_CHANNEL_INDEX["winning_scale"]]
    sin2 = raster[FRANGI_RASTER_CHANNEL_INDEX["orientation_sin2"]]
    cos2 = raster[FRANGI_RASTER_CHANNEL_INDEX["orientation_cos2"]]
    distance = raster[FRANGI_RASTER_CHANNEL_INDEX["distance_to_skeleton"]]

    tolerance = 1e-6
    if similarity.min() < -tolerance or similarity.max() > 1.0 + tolerance:
        raise ValueError("similarity must lie in [0, 1]")
    if not np.logical_or(support == 0.0, support == 1.0).all():
        raise ValueError("support must be binary")
    if magnitude.min() < -tolerance:
        raise ValueError("hessian_magnitude must be non-negative")
    if scale.min() < -tolerance:
        raise ValueError("winning_scale must be non-negative")
    if sin2.min() < -1.0 - tolerance or sin2.max() > 1.0 + tolerance:
        raise ValueError("orientation_sin2 must lie in [-1, 1]")
    if cos2.min() < -1.0 - tolerance or cos2.max() > 1.0 + tolerance:
        raise ValueError("orientation_cos2 must lie in [-1, 1]")
    if distance.min() < -tolerance or distance.max() > 1.0 + tolerance:
        raise ValueError("distance_to_skeleton must lie in [0, 1]")

    oriented = scale > 0.0
    if oriented.any():
        norm = np.square(sin2[oriented]) + np.square(cos2[oriented])
        if not np.allclose(norm, 1.0, rtol=2e-5, atol=2e-5):
            raise ValueError("orientation channels must encode a unit direction")
    if np.any(sin2[~oriented] != 0.0) or np.any(cos2[~oriented] != 0.0):
        raise ValueError("orientation must be zero where winning_scale is zero")


def no_frangi_evidence_raster(height: int, width: int) -> np.ndarray:
    """Return the canonical seven-channel encoding of absent Frangi evidence.

    Distance is one because there is no accepted skeleton anywhere.  Using an
    all-zero tensor would instead encode zero distance to a skeleton at every
    pixel and is therefore forbidden for dropout and causal controls.
    """
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    raster = np.zeros(
        (len(FRANGI_RASTER_CHANNELS), height, width), dtype=np.float32
    )
    raster[FRANGI_RASTER_CHANNEL_INDEX["distance_to_skeleton"]] = 1.0
    validate_frangi_raster(raster)
    return raster


@dataclass(frozen=True, slots=True)
class FrangiRasterSample:
    """One self-describing cache entry stored as an unpickled ``.npz`` file."""

    case_name: str
    raster: np.ndarray
    image_sha256: str
    mask_sha256: str
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not self.case_name or "\x00" in self.case_name:
            raise ValueError("case_name must be a non-empty text identifier")
        _validate_sha256(self.image_sha256, "image_sha256")
        _validate_sha256(self.mask_sha256, "mask_sha256")
        if not math.isfinite(float(self.elapsed_seconds)) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        validate_frangi_raster(self.raster)

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.raster.shape[1]), int(self.raster.shape[2])

    def channel(self, name: str) -> np.ndarray:
        try:
            index = FRANGI_RASTER_CHANNEL_INDEX[name]
        except KeyError as exc:
            raise KeyError(f"Unknown Frangi raster channel: {name!r}") from exc
        return self.raster[index]

    def statistics(self) -> dict[str, object]:
        """Return compact finite statistics suitable for a JSON manifest."""
        pixels = int(self.raster.shape[1] * self.raster.shape[2])
        support_pixels = int(np.count_nonzero(self.channel("support")))
        skeleton_pixels = int(np.count_nonzero(self.channel("distance_to_skeleton") == 0.0))
        support = self.channel("support") > 0.0
        similarity = (
            float(self.channel("similarity")[support].mean())
            if support.any()
            else 0.0
        )
        strength = (
            float(self.channel("hessian_magnitude")[support].mean())
            if support.any()
            else 0.0
        )
        return {
            "pixels": pixels,
            "support_pixels": support_pixels,
            "support_density": support_pixels / pixels,
            "skeleton_pixels": skeleton_pixels,
            "skeleton_density": skeleton_pixels / pixels,
            "similarity_mean": float(self.channel("similarity").mean()),
            "similarity_max": float(self.channel("similarity").max()),
            "hessian_magnitude_mean": float(self.channel("hessian_magnitude").mean()),
            "hessian_magnitude_max": float(self.channel("hessian_magnitude").max()),
            "elapsed_seconds": float(self.elapsed_seconds),
            "nonfinite_values": 0,
            "gate_features": {
                "similarity": similarity,
                "strength": strength,
                "density": support_pixels / pixels,
                "stability": None,
                "stability_available": False,
            },
        }


__all__ = [
    "FRANGI_RASTER_CHANNEL_INDEX",
    "FRANGI_RASTER_CHANNEL_SEMANTICS",
    "FRANGI_RASTER_CHANNELS",
    "FRANGI_RASTER_SCHEMA_VERSION",
    "FRANGI_GATE_FEATURE_SEMANTICS",
    "FrangiRasterSample",
    "no_frangi_evidence_raster",
    "validate_frangi_raster",
]
