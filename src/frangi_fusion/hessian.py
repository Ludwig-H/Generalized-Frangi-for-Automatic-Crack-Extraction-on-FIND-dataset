# src/frangi_fusion/hessian.py

import numpy as np
from typing import Dict, List, Tuple
from skimage.feature import hessian_matrix

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert to float32 in [0,1]. Accepts HxW or HxWxC."""
    if img.ndim == 2:
        g = img.astype(np.float32)
    elif img.ndim == 3:
        c = img.shape[2]
        arr = img.astype(np.float32)
        if c >= 3:
            w = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
            g = arr[..., :3].dot(w)
        elif c == 2:
            g = arr.mean(axis=2)
        else:
            g = arr[..., 0]
    else:
        raise ValueError("Unsupported image shape")
    g -= g.min()
    if g.max() > 0:
        g /= g.max()
    return g

def _order_by_abs(e1: np.ndarray, e2: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """Ensure |e1| <= |e2| pixelwise."""
    swap = np.abs(e1) > np.abs(e2)
    if np.any(swap):
        e1c, e2c = e1.copy(), e2.copy()
        e1c[swap], e2c[swap] = e2[swap], e1[swap]
        return e1c, e2c
    return e1, e2

def _eigvals_from_hessian(Hxx: np.ndarray, Hxy: np.ndarray, Hyy: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """Closed-form eigenvalues of a 2x2 symmetric matrix."""
    tr = (Hxx + Hyy) / 2.0
    disc = np.sqrt(((Hxx - Hyy) / 2.0) ** 2 + Hxy ** 2)
    l1 = tr - disc
    l2 = tr + disc
    return l1, l2

def _hessian_raw(gray: np.ndarray, sigma: float) -> Dict[str,np.ndarray]:
    """
    Hessian with:
      - order='xy' (cohérence des noms Hxx, Hxy, Hyy),
      - use_gaussian_derivatives=True,
      - mode='reflect' pour éviter les artefacts de bord,
      - normalisation d’échelle par sigma**2.
    """
    # Travail en float64 pour la stabilité num.
    g64 = gray.astype(np.float64, copy=False)

    Hxx, Hxy, Hyy = hessian_matrix(
        g64,
        sigma=float(sigma),
        order='xy',                     # <— cohérent avec les noms
        use_gaussian_derivatives=True,
        mode='reflect',
        cval=0.0
    )

    # Normalisation d’échelle (Lindeberg): multiplié par sigma^2
    s2 = float(sigma) ** 2
    Hxx *= s2; Hxy *= s2; Hyy *= s2

    # Valeurs propres (fermées), puis tri par valeur absolue
    e1_raw, e2_raw = _eigvals_from_hessian(Hxx, Hxy, Hyy)
    e1_raw, e2_raw = _order_by_abs(e1_raw, e2_raw)

    # Orientation principale (bornée dans [-pi/2, pi/2])
    theta = 0.5 * np.arctan2(2.0 * Hxy, (Hxx - Hyy) + 1e-12)

    return {
        "Hxx_raw": Hxx, "Hxy_raw": Hxy, "Hyy_raw": Hyy,
        "e1": e1_raw, "e2": e2_raw, "theta": theta
    }

def _normalize_per_matrix_by_maxabs_e2(Hd: Dict[str,np.ndarray]) -> None:
    """
    Normalise λ en divisant par max(|λ2|) sur toute la matrice.
    Garantit λ1, λ2 ∈ [-1,1] et re-ordonne par |.| si besoin.
    """
    denom = float(np.max(np.abs(Hd["e2"])))
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    e1n = Hd["e1"] / denom
    e2n = Hd["e2"] / denom
    e1n, e2n = _order_by_abs(e1n, e2n)
    Hd["e1n"] = e1n
    Hd["e2n"] = e2n
    Hd["eig_norm_denom"] = denom

def compute_hessians_per_scale(modality_gray: np.ndarray, sigmas: List[float]) -> List[Dict[str,np.ndarray]]:
    """
    Pour chaque σ:
      - calcule Hxx,Hxy,Hyy avec normalisation d’échelle sigma**2,
      - calcule e1,e2 (bruts) puis e1n,e2n = λ / max(|λ2|) ∈ [-1,1],
      - conserve θ.
    """
    out = []
    for s in sigmas:
        Hd = _hessian_raw(modality_gray, s)
        _normalize_per_matrix_by_maxabs_e2(Hd)
        Hd["sigma"] = s
        out.append(Hd)
    return out

def fuse_hessians_per_scale(hessians_by_modality: Dict[str, List[Dict[str,np.ndarray]]],
                            weights_by_modality: Dict[str, float]) -> List[Dict[str,np.ndarray]]:
    """
    Fusion par échelle au niveau H brut: H_total = Σ_m w_m H_m.
    Puis recalcul de e1,e2 et normalisation par max(|λ2|) sur la matrice fusionnée.
    """
    first_key = next(iter(hessians_by_modality))
    sigmas = [Hd["sigma"] for Hd in hessians_by_modality[first_key]]

    fused = []
    for sidx, sigma in enumerate(sigmas):
        Hxx_raw = None; Hxy_raw = None; Hyy_raw = None
        for mod, lst in hessians_by_modality.items():
            w = float(weights_by_modality.get(mod, 1.0))
            Hd = lst[sidx]
            if Hxx_raw is None:
                Hxx_raw = w * Hd["Hxx_raw"]; Hxy_raw = w * Hd["Hxy_raw"]; Hyy_raw = w * Hd["Hyy_raw"]
            else:
                Hxx_raw += w * Hd["Hxx_raw"]; Hxy_raw += w * Hd["Hxy_raw"]; Hyy_raw += w * Hd["Hyy_raw"]

        e1_raw, e2_raw = _eigvals_from_hessian(Hxx_raw, Hxy_raw, Hyy_raw)
        e1_raw, e2_raw = _order_by_abs(e1_raw, e2_raw)

        Hd_f = {
            "Hxx_raw": Hxx_raw, "Hxy_raw": Hxy_raw, "Hyy_raw": Hyy_raw,
            "e1": e1_raw, "e2": e2_raw,
            "theta": 0.5 * np.arctan2(2.0 * Hxy_raw, (Hxx_raw - Hyy_raw) + 1e-12),
            "sigma": sigma
        }
        _normalize_per_matrix_by_maxabs_e2(Hd_f)
        fused.append(Hd_f)

    return fused
