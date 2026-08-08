"""Filtres géométriques de fissures fines avec rejet explicite des ombres.

Ce module est volontairement indépendant de SAM 2. Il produit des cartes
continues dans ``[0, 1]`` destinées à être évaluées comme *propositions*
géométriques. Aucun seuil appris, masque vrai ou poids de réseau n'intervient.

Les filtres principaux opposent une ligne sombre approximativement paire à
une marche d'illumination approximativement impaire :

* ``oriented_flux_symmetry`` : approximation discrète 2D fidèle aux équations
  de Law et Chung (ECCV 2010) ;
* ``dark_phase_symmetry`` : banque de Gabor en quadrature, avec polarité sombre ;
* ``paired_profile`` : deux flancs clairs exigés au même rayon et à la même
  orientation ;
* ``black_tophat`` : fermeture morphologique multiscale limitée aux faibles
  largeurs.

Les normalisations sont robustes au bruit mais n'utilisent jamais le maximum
spatial de l'image. Une image uniforme peut donc réellement produire une carte
nulle, contrairement à la Frangi-similarité historique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi
from scipy import special
from skimage.filters import frangi
from skimage.morphology import disk


EPS = np.finfo(np.float32).eps


@dataclass(frozen=True)
class FilterBankResult:
    """Cartes finales et diagnostics utiles à l'analyse des mécanismes."""

    maps: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


def _as_float_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(f"Image HxW, HxWx1, HxWx3 ou HxWx4 attendue, reçu {array.shape}")
    array = array[..., :3].astype(np.float32, copy=False)
    if array.size and float(np.nanmax(array)) > 1.0:
        array = array / 255.0
    return np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def linear_luminance(image: np.ndarray) -> np.ndarray:
    """Convertit du sRGB en luminance linéaire, plus proche du modèle d'ombre."""

    rgb = _as_float_rgb(image)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    return np.tensordot(linear, np.array((0.2126, 0.7152, 0.0722)), axes=([-1], [0])).astype(
        np.float32
    )


def log_luminance(image: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    return np.log(np.maximum(linear_luminance(image), floor)).astype(np.float32)


def _robust_sigma(values: np.ndarray, floor: float = 1e-5) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(mad / 0.6744897501960817, floor)


def _noise_from_highpass(signal: np.ndarray) -> float:
    highpass = signal - ndi.gaussian_filter(signal, sigma=0.8, mode="reflect")
    return _robust_sigma(highpass, floor=1e-5)


def _confidence(raw: np.ndarray, scale: float) -> np.ndarray:
    positive = np.maximum(np.asarray(raw, dtype=np.float32), 0.0)
    result = positive / (positive + max(float(scale), 1e-8))
    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _hessian_geometry(signal: np.ndarray, sigma: float) -> tuple[np.ndarray, ...]:
    """Retourne courbures transverse/longitudinale et normale non orientée."""

    # Les noyaux discrets de dérivées gaussiennes n'ont pas une somme
    # rigoureusement nulle en précision flottante. Centrer le signal évite qu'un
    # niveau DC produise une pseudo-courbure sur une image uniforme.
    centered = np.asarray(signal, dtype=np.float32) - np.float32(np.median(signal))
    gx = ndi.gaussian_filter(centered, sigma=sigma, order=(0, 1), mode="reflect")
    gy = ndi.gaussian_filter(centered, sigma=sigma, order=(1, 0), mode="reflect")
    hxx = ndi.gaussian_filter(centered, sigma=sigma, order=(0, 2), mode="reflect")
    hxy = ndi.gaussian_filter(centered, sigma=sigma, order=(1, 1), mode="reflect")
    hyy = ndi.gaussian_filter(centered, sigma=sigma, order=(2, 0), mode="reflect")

    trace = hxx + hyy
    discriminant = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4.0 * hxy**2, 0.0))
    lambda_plus = 0.5 * (trace + discriminant)
    lambda_minus = 0.5 * (trace - discriminant)
    plus_is_transverse = np.abs(lambda_plus) >= np.abs(lambda_minus)
    transverse = np.where(plus_is_transverse, lambda_plus, lambda_minus)
    longitudinal = np.where(plus_is_transverse, lambda_minus, lambda_plus)

    theta_plus = 0.5 * np.arctan2(2.0 * hxy, hxx - hyy)
    theta_normal = np.where(plus_is_transverse, theta_plus, theta_plus + np.pi / 2.0)
    normal_x = np.cos(theta_normal)
    normal_y = np.sin(theta_normal)
    anisotropy = np.clip(
        1.0 - np.abs(longitudinal) / (np.abs(transverse) + EPS),
        0.0,
        1.0,
    )
    return transverse, longitudinal, normal_x, normal_y, anisotropy, gx, gy


def relative_frangi(
    image: np.ndarray,
    sigmas: Iterable[float] = (0.8, 1.2, 1.8, 2.6, 3.8),
) -> np.ndarray:
    """Baseline Hessienne relative par image, incluse comme contrôle.

    ``gamma=None`` reproduit précisément le défaut conceptuel que l'étude veut
    mesurer : le contraste est adapté au maximum de chaque image. Cette carte
    ne doit donc pas être interprétée comme une confiance absolue.
    """

    response = frangi(
        linear_luminance(image),
        sigmas=tuple(float(value) for value in sigmas),
        alpha=0.5,
        beta=0.5,
        gamma=None,
        black_ridges=True,
        mode="reflect",
    )
    return np.nan_to_num(response.astype(np.float32), nan=0.0).clip(0.0, 1.0)


def historical_frangi_similarity_cpu(
    image: np.ndarray,
    scales: Iterable[float] = (1.0, 3.0, 5.0, 9.0, 15.0),
    radius: int = 3,
    shape_bandwidth: float = 1.0,
    intensity_bandwidth: float = 0.25,
    angle_bandwidth: float = 0.3,
    retained_fraction: float = 0.18,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Réimplémentation CPU vectorisée du ``node_sim_max`` historique.

    Elle conserve sciemment les normalisations et seuils relatifs du code
    CrackSAM 2 afin de fournir le contrôle pertinent. Le MST, les composantes,
    la centralité et la pénalisation de distance ne sont pas calculés, car ils
    ne l'étaient pas non plus lorsque ``compute_centrality=False``.
    """

    if radius < 1 or not 0.0 < retained_fraction <= 1.0:
        raise ValueError("Rayon positif et fraction retenue dans ]0, 1] requis")
    # Le wrapper CrackSAM historique travaille dans le gris sRGB (et non en
    # luminance linéaire). Garder exactement cette convention est essentiel
    # pour que ce contrôle reste comparable aux prompts archivés de juillet.
    rgb = _as_float_rgb(image)
    signal = np.tensordot(
        rgb,
        np.array((0.2989, 0.5870, 0.1140), dtype=np.float32),
        axes=([-1], [0]),
    ).astype(np.float32)
    scale_geometry: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    strongest_curvature = np.zeros_like(signal, dtype=np.float32)
    for sigma in tuple(float(value) for value in scales):
        hxx = ndi.gaussian_filter(signal, sigma=sigma, order=(0, 2), mode="nearest")
        hxy = ndi.gaussian_filter(signal, sigma=sigma, order=(1, 1), mode="nearest")
        hyy = ndi.gaussian_filter(signal, sigma=sigma, order=(2, 0), mode="nearest")
        trace = hxx + hyy
        discriminant = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4.0 * hxy**2, 0.0))
        spectral_norm = 0.5 * (np.abs(trace) + discriminant)
        normalization = float(np.max(spectral_norm)) + 1e-8
        hxx = hxx / normalization
        hxy = hxy / normalization
        hyy = hyy / normalization

        trace = hxx + hyy
        discriminant = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4.0 * hxy**2, 0.0))
        lambda_plus = 0.5 * (trace + discriminant)
        lambda_minus = 0.5 * (trace - discriminant)
        minus_is_transverse = np.abs(lambda_minus) > np.abs(lambda_plus)
        transverse = np.where(minus_is_transverse, lambda_minus, lambda_plus)
        longitudinal = np.where(minus_is_transverse, lambda_plus, lambda_minus)
        positive = transverse > 0.0
        ratio = np.zeros_like(transverse, dtype=np.float32)
        ratio[positive] = np.abs(longitudinal[positive]) / (transverse[positive] + 1e-8)
        curvature = np.where(positive, transverse, 0.0).astype(np.float32)
        theta = (0.5 * np.arctan2(2.0 * hxy, hxx - hyy)).astype(np.float32)
        strongest_curvature = np.maximum(strongest_curvature, curvature)
        scale_geometry.append((ratio, curvature, theta))

    candidate = strongest_curvature > (float(np.max(strongest_curvature)) * 0.01)
    similarity = np.zeros_like(signal, dtype=np.float32)
    if not np.any(candidate):
        return similarity, {"historical_candidate": candidate, "historical_curvature": strongest_curvature}

    height, width = signal.shape
    offsets = [
        (dy, dx)
        for dy in range(0, radius + 1)
        for dx in range(-radius, radius + 1)
        if (dy > 0 or dx > 0) and 0 < dy * dy + dx * dx <= radius * radius
    ]
    edge_records: list[tuple[tuple[slice, slice], tuple[slice, slice], np.ndarray, np.ndarray]] = []
    all_values: list[np.ndarray] = []
    for dy, dx in offsets:
        y0 = slice(max(0, -dy), min(height, height - dy))
        y1 = slice(max(0, dy), min(height, height + dy))
        x0 = slice(max(0, -dx), min(width, width - dx))
        x1 = slice(max(0, dx), min(width, width + dx))
        pair_mask = candidate[y0, x0] & candidate[y1, x1]
        if not np.any(pair_mask):
            continue
        edge_similarity = np.zeros(pair_mask.shape, dtype=np.float32)
        for ratio, curvature, theta in scale_geometry:
            ratio_sum = ratio[y0, x0] + ratio[y1, x1]
            shape_term = np.exp(-0.5 * (ratio_sum / shape_bandwidth) ** 2)
            curvature_product = curvature[y0, x0] * curvature[y1, x1]
            intensity_term = 1.0 - np.exp(
                -0.5 * (curvature_product / intensity_bandwidth) ** 2
            )
            angle_delta = theta[y0, x0] - theta[y1, x1]
            angle_term = np.exp(
                -0.5 * (np.sin(angle_delta) / angle_bandwidth) ** 2
            )
            edge_similarity = np.maximum(
                edge_similarity,
                (shape_term * intensity_term * angle_term).astype(np.float32),
            )
        values = edge_similarity[pair_mask]
        all_values.append(values)
        edge_records.append((
            (y0, x0),
            (y1, x1),
            pair_mask,
            edge_similarity,
        ))

    if not all_values:
        return similarity, {"historical_candidate": candidate, "historical_curvature": strongest_curvature}
    concatenated = np.concatenate(all_values)
    keep_count = max(1, int(concatenated.size * retained_fraction))
    threshold = float(np.partition(concatenated, -keep_count)[-keep_count])
    for source_slice, target_slice, pair_mask, edge_similarity in edge_records:
        accepted = pair_mask & (edge_similarity >= threshold)
        if not np.any(accepted):
            continue
        source_view = similarity[source_slice]
        target_view = similarity[target_slice]
        np.maximum(source_view, np.where(accepted, edge_similarity, 0.0), out=source_view)
        np.maximum(target_view, np.where(accepted, edge_similarity, 0.0), out=target_view)
    return similarity, {
        "historical_candidate": candidate.astype(np.float32),
        "historical_curvature": strongest_curvature,
    }


def even_odd_derivative_pair(
    signal: np.ndarray,
    sigmas: Iterable[float] = (0.8, 1.2, 1.8, 2.6, 3.8),
    odd_penalty: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Réponse paire de ligne moins réponse impaire de marche, même échelle."""

    scores: list[np.ndarray] = []
    raw_even: list[np.ndarray] = []
    raw_odd: list[np.ndarray] = []
    scales = tuple(float(value) for value in sigmas)
    for sigma in scales:
        transverse, _, nx, ny, anisotropy, gx, gy = _hessian_geometry(signal, sigma)
        even = np.maximum((sigma**2) * transverse, 0.0)
        odd = sigma * np.abs(gx * nx + gy * ny)
        even_noise = _robust_sigma(even)
        threshold = 2.5 * even_noise
        numerator = np.maximum(even - odd_penalty * odd - threshold, 0.0)
        denominator = even + odd_penalty * odd + threshold + EPS
        scores.append((numerator / denominator * anisotropy).astype(np.float32))
        raw_even.append(even.astype(np.float32))
        raw_odd.append(odd.astype(np.float32))

    stack = np.stack(scores)
    if stack.shape[0] > 1:
        adjacent = np.sqrt(np.maximum(stack[:-1] * stack[1:], 0.0))
        persistent = np.max(adjacent, axis=0)
    else:
        persistent = stack[0]
    best_index = np.argmax(stack, axis=0)
    scale_array = np.asarray(scales, dtype=np.float32)
    winning_scale = scale_array[best_index]
    return persistent.clip(0.0, 1.0).astype(np.float32), {
        "even_odd_best": np.max(stack, axis=0).astype(np.float32),
        "even_odd_scale": winning_scale.astype(np.float32),
        "even_response": np.take_along_axis(np.stack(raw_even), best_index[None], axis=0)[0],
        "odd_response": np.take_along_axis(np.stack(raw_odd), best_index[None], axis=0)[0],
    }


def _sample_along_normal(
    signal: np.ndarray,
    normal_x: np.ndarray,
    normal_y: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.indices(signal.shape, dtype=np.float32)
    tangent_x = -normal_y
    tangent_y = normal_x
    center_samples: list[np.ndarray] = []
    negative_samples: list[np.ndarray] = []
    positive_samples: list[np.ndarray] = []
    for tangent_offset in (-1.0, 0.0, 1.0):
        tangent_rows = rows + tangent_offset * tangent_y
        tangent_cols = cols + tangent_offset * tangent_x
        center_samples.append(
            ndi.map_coordinates(signal, (tangent_rows, tangent_cols), order=1, mode="reflect")
        )
        negative_samples.append(
            ndi.map_coordinates(
                signal,
                (tangent_rows - radius * normal_y, tangent_cols - radius * normal_x),
                order=1,
                mode="reflect",
            )
        )
        positive_samples.append(
            ndi.map_coordinates(
                signal,
                (tangent_rows + radius * normal_y, tangent_cols + radius * normal_x),
                order=1,
                mode="reflect",
            )
        )
    return (
        np.mean(center_samples, axis=0),
        np.mean(negative_samples, axis=0),
        np.mean(positive_samples, axis=0),
    )


def paired_profile(
    signal: np.ndarray,
    sigmas: Iterable[float] = (0.8, 1.2, 1.8, 2.6, 3.8),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Exige deux flancs plus clairs au même rayon transversal."""

    profile_noise = _noise_from_highpass(signal)
    scores: list[np.ndarray] = []
    symmetry_maps: list[np.ndarray] = []
    depth_maps: list[np.ndarray] = []
    scales = tuple(float(value) for value in sigmas)
    for sigma in scales:
        transverse, _, nx, ny, anisotropy, _, _ = _hessian_geometry(signal, sigma)
        radius = max(1.25, 1.6 * sigma)
        center, negative, positive = _sample_along_normal(signal, nx, ny, radius)
        d_negative = negative - center
        d_positive = positive - center
        positive_negative = np.maximum(d_negative, 0.0)
        positive_positive = np.maximum(d_positive, 0.0)
        bilateral_depth = np.minimum(positive_negative, positive_positive)
        symmetry = 2.0 * bilateral_depth / (
            np.abs(d_negative) + np.abs(d_positive) + EPS
        )
        depth_confidence = _confidence(bilateral_depth, 2.5 * profile_noise)

        even = np.maximum((sigma**2) * transverse, 0.0)
        even_confidence = _confidence(even, 2.5 * _robust_sigma(even))
        score = symmetry * np.sqrt(depth_confidence * even_confidence) * anisotropy
        scores.append(score.astype(np.float32))
        symmetry_maps.append(symmetry.astype(np.float32))
        depth_maps.append(bilateral_depth.astype(np.float32))

    stack = np.stack(scores)
    if stack.shape[0] > 1:
        adjacent = np.sqrt(np.maximum(stack[:-1] * stack[1:], 0.0))
        final = np.maximum(0.5 * np.max(stack, axis=0), np.max(adjacent, axis=0))
    else:
        final = stack[0]
    best_index = np.argmax(stack, axis=0)
    scale_array = np.asarray(scales, dtype=np.float32)
    return final.clip(0.0, 1.0).astype(np.float32), {
        "profile_symmetry": np.take_along_axis(
            np.stack(symmetry_maps), best_index[None], axis=0
        )[0],
        "profile_depth": np.take_along_axis(
            np.stack(depth_maps), best_index[None], axis=0
        )[0],
        "profile_scale": scale_array[best_index].astype(np.float32),
    }


def line_step_bic(
    signal: np.ndarray,
    sigmas: Iterable[float] = (0.8, 1.2, 1.8, 2.6, 3.8),
    profile_samples: int = 13,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compare illumination seule et illumination plus ligne sombre.

    Un profil de log-luminance est échantillonné selon la normale Hessienne.
    ``H0`` contient un polynôme quadratique et une marche gaussienne lissée ;
    ``H1`` ajoute une tranchée gaussienne sombre centrée sur le candidat. Les
    paramètres non linéaires sont parcourus sur une petite grille et les
    coefficients linéaires sont résolus par moindres carrés. Le coefficient de
    tranchée est contraint positif.

    Le score est le gain ``BIC(H0) - BIC(H1)``, pondéré par la profondeur
    absolue et l'anisotropie. Il reste une vérification géométrique douce, et
    non un classificateur appris.
    """

    if profile_samples < 9 or profile_samples % 2 == 0:
        raise ValueError("profile_samples doit être impair et >= 9")
    scales = tuple(float(value) for value in sigmas)
    if not scales or min(scales) <= 0.0:
        raise ValueError("Les échelles doivent être strictement positives")

    signal = np.asarray(signal, dtype=np.float32)
    height, width = signal.shape
    rows, cols = np.indices(signal.shape, dtype=np.float32)
    coordinates = np.linspace(-3.0, 3.0, profile_samples, dtype=np.float32)
    polynomial = np.stack(
        (np.ones_like(coordinates), coordinates, coordinates**2), axis=1
    )
    identity = np.eye(profile_samples, dtype=np.float64)
    noise = _noise_from_highpass(signal)
    noise_floor = profile_samples * (0.35 * noise) ** 2 + 1e-10

    scores: list[np.ndarray] = []
    delta_maps: list[np.ndarray] = []
    depth_maps: list[np.ndarray] = []
    for sigma in scales:
        _, _, normal_x, normal_y, anisotropy, _, _ = _hessian_geometry(signal, sigma)
        tangent_x = -normal_y
        tangent_y = normal_x
        profile_planes: list[np.ndarray] = []
        for coordinate in coordinates:
            tangent_samples: list[np.ndarray] = []
            for tangent_offset in (-1.0, 0.0, 1.0):
                sample_rows = (
                    rows
                    + coordinate * sigma * normal_y
                    + tangent_offset * tangent_y
                )
                sample_cols = (
                    cols
                    + coordinate * sigma * normal_x
                    + tangent_offset * tangent_x
                )
                tangent_samples.append(
                    ndi.map_coordinates(
                        signal,
                        (sample_rows, sample_cols),
                        order=1,
                        mode="reflect",
                    )
                )
            profile_planes.append(np.mean(tangent_samples, axis=0))
        profiles = np.stack(profile_planes, axis=-1).reshape(-1, profile_samples).astype(
            np.float64
        )

        best_sse_h0 = np.full(profiles.shape[0], np.inf, dtype=np.float64)
        best_sse_h1 = np.full(profiles.shape[0], np.inf, dtype=np.float64)
        best_depth = np.zeros(profiles.shape[0], dtype=np.float64)
        for step_center in (-0.8, 0.0, 0.8):
            for step_softness in (0.05, 0.5, 1.0):
                step = special.ndtr((coordinates - step_center) / step_softness)
                design_h0 = np.column_stack((polynomial, step)).astype(np.float64)
                residual_projection = identity - design_h0 @ np.linalg.pinv(design_h0)
                residuals_h0 = profiles @ residual_projection
                sse_h0 = np.einsum("ij,ij->i", residuals_h0, residuals_h0)
                best_sse_h0 = np.minimum(best_sse_h0, sse_h0)

                for line_width in (0.45, 0.8, 1.2):
                    dark_line = -np.exp(-0.5 * (coordinates / line_width) ** 2)
                    residual_line = residual_projection @ dark_line
                    line_norm = float(np.dot(residual_line, residual_line))
                    if line_norm <= 1e-10:
                        continue
                    depth = np.maximum(
                        (residuals_h0 @ residual_line) / line_norm,
                        0.0,
                    )
                    sse_h1 = np.maximum(sse_h0 - depth**2 * line_norm, 0.0)
                    improved = sse_h1 < best_sse_h1
                    best_sse_h1[improved] = sse_h1[improved]
                    best_depth[improved] = depth[improved]

        delta_bic = (
            profile_samples
            * np.log((best_sse_h0 + noise_floor) / (best_sse_h1 + noise_floor))
            - np.log(float(profile_samples))
        )
        bic_confidence = 1.0 - np.exp(-np.maximum(delta_bic, 0.0) / 6.0)
        depth_confidence = _confidence(best_depth, 2.5 * noise)
        score = (
            bic_confidence.reshape(height, width)
            * depth_confidence.reshape(height, width)
            * anisotropy
        )
        scores.append(score.astype(np.float32))
        delta_maps.append(delta_bic.reshape(height, width).astype(np.float32))
        depth_maps.append(best_depth.reshape(height, width).astype(np.float32))

    stack = np.stack(scores)
    if stack.shape[0] > 1:
        persistent = np.max(
            np.sqrt(np.maximum(stack[:-1] * stack[1:], 0.0)), axis=0
        )
    else:
        persistent = stack[0]
    best_index = np.argmax(stack, axis=0)
    scale_array = np.asarray(scales, dtype=np.float32)
    return persistent.clip(0.0, 1.0).astype(np.float32), {
        "line_step_delta_bic": np.take_along_axis(
            np.stack(delta_maps), best_index[None], axis=0
        )[0],
        "line_step_depth": np.take_along_axis(
            np.stack(depth_maps), best_index[None], axis=0
        )[0],
        "line_step_scale": scale_array[best_index].astype(np.float32),
    }


def _disk_and_ring(radius: int) -> tuple[np.ndarray, np.ndarray, float]:
    footprint = disk(radius).astype(np.float32)
    inner = disk(max(radius - 1, 0)).astype(np.float32)
    if inner.shape != footprint.shape:
        padded = np.zeros_like(footprint)
        offset = (footprint.shape[0] - inner.shape[0]) // 2
        padded[offset : offset + inner.shape[0], offset : offset + inner.shape[1]] = inner
        inner = padded
    ring = np.maximum(footprint - inner, 0.0)
    perimeter = max(float(np.sum(ring)), 1.0)
    return footprint, ring, perimeter


def oriented_flux_symmetry(
    signal: np.ndarray,
    radii: Iterable[int] = (1, 2, 3, 4, 5),
    smoothing_sigma: float = 0.8,
    antisymmetry_penalty: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Oriented Flux Symmetry 2D pour lignes sombres.

    Par le théorème de la divergence, le tenseur OOF est l'intégrale de la
    Hessienne sur un disque. Le vecteur OFA est la moyenne du gradient sur son
    anneau. Les deux quantités utilisent le même rayon et la même normale.
    """

    centered = np.asarray(signal, dtype=np.float32) - np.float32(np.median(signal))
    gx = ndi.gaussian_filter(centered, smoothing_sigma, order=(0, 1), mode="reflect")
    gy = ndi.gaussian_filter(centered, smoothing_sigma, order=(1, 0), mode="reflect")
    hxx = ndi.gaussian_filter(centered, smoothing_sigma, order=(0, 2), mode="reflect")
    hxy = ndi.gaussian_filter(centered, smoothing_sigma, order=(1, 1), mode="reflect")
    hyy = ndi.gaussian_filter(centered, smoothing_sigma, order=(2, 0), mode="reflect")

    score_maps: list[np.ndarray] = []
    flux_maps: list[np.ndarray] = []
    anti_maps: list[np.ndarray] = []
    radius_values = tuple(int(value) for value in radii)
    if not radius_values or min(radius_values) < 1:
        raise ValueError("Les rayons OFS doivent être des entiers strictement positifs")

    for radius in radius_values:
        footprint, ring, perimeter = _disk_and_ring(radius)
        disk_kernel = footprint / perimeter
        ring_kernel = ring / perimeter
        qxx = ndi.convolve(hxx, disk_kernel, mode="reflect")
        qxy = ndi.convolve(hxy, disk_kernel, mode="reflect")
        qyy = ndi.convolve(hyy, disk_kernel, mode="reflect")
        ofa_x = ndi.convolve(gx, ring_kernel, mode="reflect")
        ofa_y = ndi.convolve(gy, ring_kernel, mode="reflect")

        trace = qxx + qyy
        discriminant = np.sqrt(np.maximum((qxx - qyy) ** 2 + 4.0 * qxy**2, 0.0))
        lambda_plus = 0.5 * (trace + discriminant)
        lambda_minus = 0.5 * (trace - discriminant)
        plus_is_transverse = np.abs(lambda_plus) >= np.abs(lambda_minus)
        transverse = np.where(plus_is_transverse, lambda_plus, lambda_minus)
        longitudinal = np.where(plus_is_transverse, lambda_minus, lambda_plus)
        theta_plus = 0.5 * np.arctan2(2.0 * qxy, qxx - qyy)
        theta_normal = np.where(plus_is_transverse, theta_plus, theta_plus + np.pi / 2.0)
        nx = np.cos(theta_normal)
        ny = np.sin(theta_normal)

        dark_flux = np.maximum(transverse, 0.0)
        antisymmetry = np.abs(ofa_x * nx + ofa_y * ny)
        anisotropy = np.clip(
            1.0 - np.abs(longitudinal) / (np.abs(transverse) + EPS), 0.0, 1.0
        )
        noise = 2.5 * _robust_sigma(transverse)
        numerator = np.maximum(
            dark_flux - antisymmetry_penalty * antisymmetry - noise,
            0.0,
        )
        denominator = dark_flux + antisymmetry_penalty * antisymmetry + noise + EPS
        score_maps.append((numerator / denominator * anisotropy).astype(np.float32))
        flux_maps.append(dark_flux.astype(np.float32))
        anti_maps.append(antisymmetry.astype(np.float32))

    stack = np.stack(score_maps)
    best_index = np.argmax(stack, axis=0)
    radii_array = np.asarray(radius_values, dtype=np.float32)
    return np.max(stack, axis=0).clip(0.0, 1.0).astype(np.float32), {
        "ofs_radius": radii_array[best_index].astype(np.float32),
        "ofs_flux": np.take_along_axis(np.stack(flux_maps), best_index[None], axis=0)[0],
        "ofs_antisymmetry": np.take_along_axis(
            np.stack(anti_maps), best_index[None], axis=0
        )[0],
    }


def _gabor_kernel(
    wavelength: float,
    theta: float,
    gamma: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = 0.56 * wavelength
    half_size = int(np.ceil(3.0 * sigma))
    y, x = np.mgrid[-half_size : half_size + 1, -half_size : half_size + 1]
    normal = x * np.cos(theta) + y * np.sin(theta)
    tangent = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-(normal**2 + (gamma * tangent) ** 2) / (2.0 * sigma**2))
    phase = 2.0 * np.pi * normal / wavelength
    even = envelope * np.cos(phase)
    odd = envelope * np.sin(phase)
    even -= np.mean(even)
    even /= np.sqrt(np.sum(even**2)) + EPS
    odd /= np.sqrt(np.sum(odd**2)) + EPS
    return even.astype(np.float32), odd.astype(np.float32)


def dark_phase_symmetry(
    signal: np.ndarray,
    wavelengths: Iterable[float] = (3.0, 4.5, 6.5, 9.0),
    orientations: int = 12,
    noise_k: float = 1.25,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Phase-symétrie sombre par Gabor en quadrature.

    Ce n'est délibérément pas une phase-congruency générique : une marche est
    rejetée par ``|odd|`` et seule la polarité paire correspondant à une ligne
    sombre contribue au numérateur.
    """

    centered = np.asarray(signal, dtype=np.float32) - np.float32(np.median(signal))
    orientation_scores: list[np.ndarray] = []
    orientation_energy: list[np.ndarray] = []
    for orientation_index in range(int(orientations)):
        theta = np.pi * orientation_index / float(orientations)
        numerator = np.zeros_like(signal, dtype=np.float32)
        denominator = np.zeros_like(signal, dtype=np.float32)
        for wavelength in wavelengths:
            even_kernel, odd_kernel = _gabor_kernel(float(wavelength), theta)
            even = ndi.convolve(centered, even_kernel, mode="reflect")
            odd = ndi.convolve(centered, odd_kernel, mode="reflect")
            amplitude = np.hypot(even, odd)
            noise = noise_k * _robust_sigma(amplitude)
            numerator += np.maximum(-even - np.abs(odd) - noise, 0.0).astype(np.float32)
            denominator += amplitude.astype(np.float32)
        score = numerator / (denominator + EPS)
        energy_gate = _confidence(denominator, 2.5 * _robust_sigma(denominator))
        orientation_scores.append((score * energy_gate).astype(np.float32))
        orientation_energy.append(denominator)

    stack = np.stack(orientation_scores)
    best_index = np.argmax(stack, axis=0)
    angle_values = np.arange(int(orientations), dtype=np.float32) * (
        np.pi / float(orientations)
    )
    return np.max(stack, axis=0).clip(0.0, 1.0).astype(np.float32), {
        "phase_orientation": angle_values[best_index].astype(np.float32),
        "phase_energy": np.take_along_axis(
            np.stack(orientation_energy), best_index[None], axis=0
        )[0],
    }


def multiscale_black_tophat(
    signal: np.ndarray,
    radii: Iterable[int] = (2, 3, 4, 5),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Black top-hat log-photométrique, persistant sur deux largeurs voisines."""

    noise = _noise_from_highpass(signal)
    maps: list[np.ndarray] = []
    raw_maps: list[np.ndarray] = []
    radius_values = tuple(int(value) for value in radii)
    for radius in radius_values:
        closed = ndi.grey_closing(signal, footprint=disk(radius), mode="reflect")
        raw = np.maximum(closed - signal, 0.0)
        maps.append(_confidence(np.maximum(raw - 2.5 * noise, 0.0), 2.5 * noise))
        raw_maps.append(raw.astype(np.float32))
    stack = np.stack(maps)
    if len(radius_values) > 1:
        persistent = np.max(np.sqrt(np.maximum(stack[:-1] * stack[1:], 0.0)), axis=0)
    else:
        persistent = stack[0]
    best_index = np.argmax(stack, axis=0)
    radius_array = np.asarray(radius_values, dtype=np.float32)
    return persistent.clip(0.0, 1.0).astype(np.float32), {
        "tophat_radius": radius_array[best_index].astype(np.float32),
        "tophat_raw": np.take_along_axis(np.stack(raw_maps), best_index[None], axis=0)[0],
    }


def morphological_reflectance(
    luminance: np.ndarray,
    closing_radius: int = 10,
    illumination_sigma: float = 5.0,
) -> np.ndarray:
    """Quotient log simple : estimation d'illumination sans effacer les traits fins."""

    safe = np.maximum(np.asarray(luminance, dtype=np.float32), 1e-3)
    crack_suppressed = ndi.grey_closing(
        safe,
        footprint=disk(int(closing_radius)),
        mode="reflect",
    )
    illumination = ndi.gaussian_filter(
        crack_suppressed,
        sigma=float(illumination_sigma),
        mode="reflect",
    )
    return (np.log(safe) - np.log(np.maximum(illumination, 1e-3))).astype(np.float32)


def geometric_mean(maps: Iterable[np.ndarray], floor: float = 1e-4) -> np.ndarray:
    arrays = [np.clip(np.asarray(item, dtype=np.float32), 0.0, 1.0) for item in maps]
    if not arrays:
        raise ValueError("Au moins une carte est nécessaire pour la fusion")
    log_sum = np.zeros_like(arrays[0], dtype=np.float64)
    for array in arrays:
        if array.shape != arrays[0].shape:
            raise ValueError("Toutes les cartes fusionnées doivent avoir la même forme")
        log_sum += np.log(np.maximum(array, floor))
    fused = np.exp(log_sum / len(arrays))
    # Un vrai zéro dans une preuve remet la fusion à zéro : le plancher ne sert
    # qu'à stabiliser numériquement le produit.
    fused[np.minimum.reduce(arrays) <= 0.0] = 0.0
    return fused.astype(np.float32).clip(0.0, 1.0)


def compute_filter_bank(image: np.ndarray) -> FilterBankResult:
    """Calcule toutes les ablations retenues sur une image RGB ou grise."""

    luminance = linear_luminance(image)
    log_signal = np.log(np.maximum(luminance, 1e-3)).astype(np.float32)
    reflectance = morphological_reflectance(luminance)

    derivative, derivative_diag = even_odd_derivative_pair(log_signal)
    profile, profile_diag = paired_profile(log_signal)
    model_bic, model_bic_diag = line_step_bic(log_signal)
    ofs, ofs_diag = oriented_flux_symmetry(log_signal)
    phase, phase_diag = dark_phase_symmetry(log_signal)
    tophat, tophat_diag = multiscale_black_tophat(log_signal)
    ofs_reflectance, ofs_reflectance_diag = oriented_flux_symmetry(reflectance)
    profile_reflectance, profile_reflectance_diag = paired_profile(reflectance)
    historical_similarity, historical_diag = historical_frangi_similarity_cpu(image)

    maps = {
        "frangi_similarity_historical": historical_similarity,
        "frangi_relative": relative_frangi(image),
        "black_tophat": tophat,
        "derivative_pair": derivative,
        "paired_profile": profile,
        "line_step_bic": model_bic,
        "phase_symmetry": phase,
        "ofs": ofs,
        "ofs_reflectance": ofs_reflectance,
        "fusion_ofs_profile": geometric_mean((ofs, profile)),
        "fusion_precision": geometric_mean((ofs, profile, phase)),
        "fusion_reflectance": geometric_mean((ofs_reflectance, profile_reflectance, phase)),
        "verified_frangi_ofs": geometric_mean((historical_similarity, ofs)),
        "verified_frangi_bic": geometric_mean((historical_similarity, model_bic)),
        "verified_frangi_v2": geometric_mean(
            (historical_similarity, derivative, model_bic)
        ),
        "verified_frangi_consensus": geometric_mean(
            (historical_similarity, ofs, profile, phase)
        ),
    }
    diagnostics = {
        "luminance": luminance,
        "log_luminance": log_signal,
        "reflectance": reflectance,
        **derivative_diag,
        **profile_diag,
        **model_bic_diag,
        **ofs_diag,
        **phase_diag,
        **tophat_diag,
        **historical_diag,
        **{f"reflectance_{key}": value for key, value in ofs_reflectance_diag.items()},
        **{f"reflectance_{key}": value for key, value in profile_reflectance_diag.items()},
    }
    return FilterBankResult(maps=maps, diagnostics=diagnostics)


__all__ = [
    "FilterBankResult",
    "compute_filter_bank",
    "dark_phase_symmetry",
    "even_odd_derivative_pair",
    "geometric_mean",
    "historical_frangi_similarity_cpu",
    "linear_luminance",
    "line_step_bic",
    "log_luminance",
    "morphological_reflectance",
    "multiscale_black_tophat",
    "oriented_flux_symmetry",
    "paired_profile",
    "relative_frangi",
]
