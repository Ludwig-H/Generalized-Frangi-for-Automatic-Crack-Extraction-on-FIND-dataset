
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import imageio

def overlay_hessian_orientation(base_gray: np.ndarray, Hd, alpha: float = 0.45) -> np.ndarray:
    e2n = np.abs(Hd["e2n"])
    theta = Hd["theta"]
    s = e2n / (np.nanmax(e2n) + 1e-12)
    h = (theta + np.pi/2.0) / np.pi
    hsv = np.stack([h, np.ones_like(h)*0.9, s], axis=-1)
    rgb = hsv_to_rgb(hsv.clip(0,1))
    if base_gray.ndim==2:
        base_rgb = np.stack([base_gray,base_gray,base_gray], axis=-1).astype(np.float32)/255.0
    else:
        base_rgb = base_gray.astype(np.float32)/255.0
    out = (1-alpha)*base_rgb + alpha*rgb
    return (out.clip(0,1)*255).astype(np.uint8)

def show_clusters_on_image(base_gray: np.ndarray, coords: np.ndarray, labels: np.ndarray, figsize=(6,6)):
    plt.figure(figsize=figsize)
    plt.imshow(base_gray, cmap='gray')
    uniq = np.unique(labels[labels>=0])
    for lab in uniq:
        pts = coords[labels==lab]
        plt.scatter(pts[:,1], pts[:,0], s=2, label=f"cluster {lab}")
    plt.axis('off')
    if uniq.size>0: plt.legend(loc='lower right', fontsize=6)
    plt.show()

def animate_fault_growth(base_gray: np.ndarray, fault_edges: np.ndarray, save_path: str, steps: int = 20):
    if fault_edges.shape[0]==0: return
    idx = np.argsort(fault_edges[:, -1])
    fault_edges = fault_edges[idx]
    frames = []
    H, W = base_gray.shape[:2]
    base_rgb = np.dstack([base_gray, base_gray, base_gray]).astype(np.float32)
    n = fault_edges.shape[0]
    chunk = max(1, n//steps)
    for k in range(0, n, chunk):
        img = base_rgb.copy()
        for e in fault_edges[:k+1]:
            r0,c0,r1,c1,w = e
            rr = np.linspace(r0, r1, num=int(max(abs(r1-r0),abs(c1-c0))+1)).astype(int)
            cc = np.linspace(c0, c1, num=rr.shape[0]).astype(int)
            rr = np.clip(rr, 0, H-1); cc = np.clip(cc, 0, W-1)
            img[rr,cc,0] = 255; img[rr,cc,1] = 0; img[rr,cc,2] = 0
        frames.append(img.astype(np.uint8))
    imageio.mimsave(save_path, frames, duration=0.15)
