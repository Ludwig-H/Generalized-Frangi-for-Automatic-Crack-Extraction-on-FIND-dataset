"""Label-free descriptors from the existing SAM feature archive.

The factory only forms descriptors and declares preprocessing. Its caller fits
standardisation/PCA on training folds for prediction, or on the complete archive
for descriptive UMAP. ``blocks`` lists widths, not channel boundaries.
"""
from __future__ import annotations

import numpy as np


REQUIRED_WIDTHS = {
    "mean": 256,
    "std": 256,
    "grid2": 1024,
    "multiscale_mean": 352,
    "pre_global_mean": 576,
}


def spatial_variance_components(mean, std, grid2):
    """Separate variance between equal quadrants and variance within them.

The cached population variance obeys the law of total variance. Slight negative
residuals caused by float32 pooling are clipped; materially inconsistent moments
are rejected. Calculations use float64 before storing float32 descriptors.
"""
    mean = np.asarray(mean)
    std = np.asarray(std)
    grid2 = np.asarray(grid2)
    if mean.ndim != 2 or std.shape != mean.shape or grid2.shape != (len(mean), 4 * mean.shape[1]):
        raise ValueError("Incompatible shapes for spatial variance decomposition")
    if not all(np.isfinite(value).all() for value in (mean, std, grid2)) or np.any(std < 0):
        raise ValueError("Spatial moments must be finite, with nonnegative std")
    quadrants = grid2.reshape(len(mean), mean.shape[1], 4).astype(np.float64)
    quadrant_mean = quadrants.mean(axis=2, keepdims=True)
    if not np.allclose(quadrant_mean[..., 0], mean, rtol=1e-5, atol=1e-4):
        raise ValueError("Quadrant means disagree with the cached global mean")
    between_variance = np.mean((quadrants - quadrant_mean) ** 2, axis=2)
    total_variance = std.astype(np.float64) ** 2
    within_variance = total_variance - between_variance
    tolerance = 1e-6 + 1e-5 * np.maximum(total_variance, between_variance)
    if np.any(within_variance < -tolerance):
        raise ValueError("Negative within-quadrant variance: inconsistent spatial moments")
    return (
        np.sqrt(between_variance).astype(np.float32),
        np.sqrt(np.maximum(within_variance, 0)).astype(np.float32),
    )


def build_variants(features):
    """Return fourteen fixed variants without receiving categories or scores.

    Raw arrays are reused where possible. ``standard`` requests per-channel
    Z-scores; ``balanced`` requests Z-scores followed by division of each block
    by sqrt(block width); ``pca32`` requests Z-scores then unwhitened PCA(32);
    ``l2`` requests per-image L2 normalisation of the raw vector.
    """
    missing = sorted(set(REQUIRED_WIDTHS) - set(features))
    if missing:
        raise ValueError(f"Missing cached features: {', '.join(missing)}")
    arrays = {name: np.asarray(features[name]) for name in REQUIRED_WIDTHS}
    count = len(arrays["mean"])
    for name, width in REQUIRED_WIDTHS.items():
        value = arrays[name]
        if value.shape != (count, width) or count == 0:
            raise ValueError(f"Invalid shape for {name}: {value.shape}")
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"Non-finite or non-floating feature array: {name}")
    mean, std = arrays["mean"], arrays["std"]
    multiscale = arrays["multiscale_mean"]
    if not np.array_equal(multiscale[:, :256], mean):
        raise ValueError("Multiscale prefix differs from cached mean")
    between, within = spatial_variance_components(mean, std, arrays["grid2"])
    variants = {}

    def add(name, values, title, description, metric="euclidean", transform="standard", **metadata):
        variants[name] = dict(values=values, title=title, description=description,
                              metric=metric, transform=transform, **metadata)

    add("mean", mean, "Moyenne de H", "Moyenne spatiale des 256 canaux finaux ; référence.")
    add("std", std, "Variabilité spatiale de H", "Écart-type spatial des 256 canaux finaux.")
    add("mean_std", np.concatenate((mean, std), axis=1), "Moyenne + variabilité",
        "Concaténation de la moyenne et de l’écart-type des canaux finaux.")
    add("grid2", arrays["grid2"], "Grille 2 × 2",
        "Quatre moyennes par canal, avec leur position dans l’image.")
    add("between_std", between, "Variabilité entre quadrants",
        "Écart-type des quatre moyennes de chaque canal ; variations spatiales grossières.")
    add("within_std", within, "Variabilité dans les quadrants",
        "Racine de la variance totale moins la variance entre quadrants.")
    add("highres32", multiscale[:, 256:288], "Haute résolution : 32 canaux",
        "Moyennes des 32 canaux de la carte 256 × 256.")
    add("highres64", multiscale[:, 288:352], "Haute résolution : 64 canaux",
        "Moyennes des 64 canaux de la carte 128 × 128.")
    add("highres96", multiscale[:, 256:352], "Deux hautes résolutions",
        "Concaténation des moyennes des cartes 256 × 256 et 128 × 128.")
    add("multiscale_mean", multiscale, "Trois résolutions",
        "Moyennes des canaux finaux et des deux cartes de haute résolution.")
    add("pre_global_mean", arrays["pre_global_mean"], "Avant l’attention globale",
        "Moyenne des 576 canaux disponibles avant la dernière attention globale.")
    add("mean_cosine", mean, "Moyenne : distance cosinus",
        "Moyenne brute normalisée par image en norme L2, puis distance cosinus.",
        metric="cosine", transform="l2")
    add("multiscale_balanced", multiscale, "Trois résolutions équilibrées",
        "Canaux standardisés ; chaque bloc est divisé par la racine de sa dimension.",
        transform="balanced", blocks=(256, 32, 64))
    add("mean_pca32", mean, "Moyenne : PCA à 32 dimensions",
        "Standardisation puis PCA non blanchie à 32 dimensions, sans catégories.",
        transform="pca32", n_components=32)
    return variants
