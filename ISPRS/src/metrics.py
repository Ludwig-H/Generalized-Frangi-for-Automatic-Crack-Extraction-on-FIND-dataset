import torch
import numpy as np
import cv2
import warnings
import ot
from skimage.morphology import skeletonize

def skeletonize_lee(binary_mask: np.ndarray) -> np.ndarray:
    m = (binary_mask > 0).astype(np.uint8)
    
    # Morphological closing/opening to smooth out contour irregularities
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    
    sk = skeletonize(m > 0, method='lee')
    return sk.astype(np.uint8)

def thicken(skel: np.ndarray, pixels: int = 3) -> np.ndarray:
    if pixels <= 1: 
        return skel.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    thick = cv2.dilate((skel > 0).astype(np.uint8), kernel)
    return thick

def compute_metrics(pred_mask, gt_mask):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    A = pred_mask.clone().detach().bool().to(device) if isinstance(pred_mask, torch.Tensor) else torch.from_numpy(pred_mask).bool().to(device)
    B = gt_mask.clone().detach().bool().to(device) if isinstance(gt_mask, torch.Tensor) else torch.from_numpy(gt_mask).bool().to(device)
    
    inter = (A & B).sum().float()
    union = (A | B).sum().float()
    jaccard = (inter / (union + 1e-9)).item()
    
    not_A = ~A
    not_B = ~B
    fp = (not_A & B).sum().float()
    fn = (A & not_B).sum().float()
    tversky = (inter / (inter + 1.0 * fn + 0.5 * fp + 1e-9)).item()
    
    return jaccard, tversky

def wasserstein_distance_skeletons(A, B, max_samples: int = 2000) -> float:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    A_t = A.clone().detach().to(device) if isinstance(A, torch.Tensor) else torch.from_numpy(A).to(device)
    B_t = B.clone().detach().to(device) if isinstance(B, torch.Tensor) else torch.from_numpy(B).to(device)
    
    A_pts = torch.nonzero(A_t > 0).float()
    B_pts = torch.nonzero(B_t > 0).float()
    
    na, nb = A_pts.shape[0], B_pts.shape[0]
    
    if na == 0 and nb == 0: 
        return 0.0
    if na == 0: 
        return float(nb)
    if nb == 0: 
        return float(na)
    
    if na > max_samples:
        idx = torch.randperm(na, device=device)[:max_samples]
        A_pts = A_pts[idx]
        na = max_samples
        
    if nb > max_samples:
        idx = torch.randperm(nb, device=device)[:max_samples]
        B_pts = B_pts[idx]
        nb = max_samples
        
    M_t = torch.cdist(A_pts, B_pts, p=2.0)
    
    M = M_t.cpu().numpy().astype(np.float64)
    a = np.ones((na,), dtype=np.float64) / float(na)
    b = np.ones((nb,), dtype=np.float64) / float(nb)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emd_cost = ot.emd2(a, b, M)
        
    return float(emd_cost)
