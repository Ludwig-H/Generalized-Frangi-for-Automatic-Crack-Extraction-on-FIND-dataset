
import numpy as np
from skimage.morphology import skeletonize, disk, dilation
import warnings

def skeletonize_lee(binary_mask: np.ndarray) -> np.ndarray:
    m = (binary_mask > 0).astype(np.uint8)
    sk = skeletonize(m>0)
    return sk.astype(np.uint8)

def thicken(skel: np.ndarray, pixels: int = 6) -> np.ndarray:
    if pixels <= 1: return skel.astype(np.uint8)
    from skimage.morphology import dilation, disk
    thick = dilation(skel>0, disk(int(pixels)))
    return thick.astype(np.uint8)

def jaccard_index(A: np.ndarray, B: np.ndarray) -> float:
    A = (A>0).astype(np.uint8); B = (B>0).astype(np.uint8)
    inter = np.logical_and(A,B).sum()
    union = np.logical_or(A,B).sum()
    return float(inter) / float(union + 1e-9)

def tversky_index(A: np.ndarray, B: np.ndarray, alpha: float = 1.0, beta: float = 0.5) -> float:
    A = (A>0).astype(np.uint8); B = (B>0).astype(np.uint8)
    inter = np.logical_and(A,B).sum()
    fp = np.logical_and(np.logical_not(A), B).sum()
    fn = np.logical_and(A, np.logical_not(B)).sum()
    return float(inter) / float(inter + alpha*fn + beta*fp + 1e-9)

def wasserstein_distance_skeletons(A: np.ndarray, B: np.ndarray, max_samples: int = 2000) -> float:
    import ot
    Ay, Ax = np.nonzero(A>0); By, Bx = np.nonzero(B>0)
    if len(Ay)==0 and len(By)==0: return 0.0
    if len(Ay)==0: return float(len(By))
    if len(By)==0: return float(len(Ay))
    A_pts = np.column_stack([Ay,Ax]).astype(np.float32)
    B_pts = np.column_stack([By,Bx]).astype(np.float32)
    if A_pts.shape[0] > max_samples:
        idx = np.random.choice(A_pts.shape[0], size=max_samples, replace=False); A_pts = A_pts[idx]
    if B_pts.shape[0] > max_samples:
        idx = np.random.choice(B_pts.shape[0], size=max_samples, replace=False); B_pts = B_pts[idx]
    na, nb = A_pts.shape[0], B_pts.shape[0]
    a = np.ones((na,), dtype=np.float64) / float(na)
    b = np.ones((nb,), dtype=np.float64) / float(nb)
    from scipy.spatial.distance import cdist
    M = cdist(A_pts, B_pts, metric='euclidean').astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emd_cost = ot.emd2(a,b,M)
    return float(emd_cost)
