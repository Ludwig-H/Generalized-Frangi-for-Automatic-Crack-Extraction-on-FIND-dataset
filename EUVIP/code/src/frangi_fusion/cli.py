import os, argparse, numpy as np, pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from .utils import set_seed, auto_discover_find_structure, load_modalities_and_gt_by_index
from .hessian import to_gray, compute_hessians_per_scale, fuse_hessians_per_scale
from .frangi_graph import build_frangi_similarity_graph, distances_from_similarity, triangle_connectivity_graph
from .graph_utils import largest_connected_component
from .clustering_sparse import hdbscan_from_sparse
from .metrics import skeletonize_lee, jaccard_index, tversky_index, wasserstein_distance_skeletons, thicken

def _process_one(struct, idx, sigmas, beta, c, ctheta, R, K, expZ, crackseg_dir=None):
    dat = load_modalities_and_gt_by_index(struct, idx)
    base = dat["arrays"].get("intensity", next(iter(dat["arrays"].values())))
    Hmods = {}
    if "intensity" in dat["arrays"]:
        Hmods["intensity"] = compute_hessians_per_scale(to_gray(dat["arrays"]["intensity"]), sigmas)
    if "range" in dat["arrays"]:
        Hmods["range"] = compute_hessians_per_scale(to_gray(dat["arrays"]["range"]), sigmas)
    if "fused" in dat["arrays"]:
        Hmods["fused"] = compute_hessians_per_scale(to_gray(dat["arrays"]["fused"]), sigmas)
    weights = {k:1.0 for k in Hmods.keys()}
    fused_H = fuse_hessians_per_scale(Hmods, weights)
    coords, _, S = build_frangi_similarity_graph(fused_H, beta, c, ctheta, R, dark_ridges=True)
    D = distances_from_similarity(S)
    if K == 2:
        D = triangle_connectivity_graph(coords, D)
    D_cc, idx_nodes = largest_connected_component(D)
    if D_cc.shape[0] == 0:
        return None
    labels = hdbscan_from_sparse(D_cc, min_cluster_size=50, min_samples=5, allow_single_cluster=True, expZ=expZ)

    # mask from labels
    mask = np.zeros(base.shape[:2], dtype=np.uint8)
    sub_coords = coords[idx_nodes]
    for lab in np.unique(labels[labels >= 0]):
        pts = sub_coords[labels == lab]
        mask[pts[:, 0], pts[:, 1]] = 1
    sk_fr = skeletonize_lee(mask)
    sk_fr = thicken(sk_fr, pixels=6)

    gt = dat["arrays"].get("label", None)
    res = {"index": int(idx)}
    if gt is not None:
        gt_bin = (gt > 0).astype(np.uint8)
        res["jaccard_fr"] = jaccard_index(sk_fr, gt_bin)
        res["tversky_fr"] = tversky_index(sk_fr, gt_bin, alpha=1.0, beta=0.5)
        res["wasser_fr"]  = wasserstein_distance_skeletons(sk_fr, gt_bin)
    else:
        res["jaccard_fr"] = np.nan
        res["tversky_fr"] = np.nan
        res["wasser_fr"]  = np.nan

    # Optional CrackSegDiff directory for comparison
    if crackseg_dir and os.path.isdir(crackseg_dir):
        import glob, re, imageio
        key = re.findall(r"\d+", os.path.basename(dat["paths"].get("label", '')))
        key = key[-1] if key else os.path.basename(dat["paths"].get("label", ''))
        cands = glob.glob(os.path.join(crackseg_dir, f"*{key}*"))
        if cands:
            pred = imageio.v2.imread(cands[0])
            if pred.ndim == 3: pred = pred[..., 0]
            sk_ck = skeletonize_lee(pred > 0); sk_ck = thicken(sk_ck, pixels=6)
            if gt is not None:
                res["jaccard_ck"] = jaccard_index(sk_ck, gt_bin)
                res["tversky_ck"] = tversky_index(sk_ck, gt_bin, alpha=1.0, beta=0.5)
                res["wasser_ck"]  = wasserstein_distance_skeletons(sk_ck, gt_bin)
    return res

def main():
    ap = argparse.ArgumentParser(description="Batch evaluation of Generalized Frangi on FIND")
    ap.add_argument("--find-root", required=True, help="Path to FIND root (after unzip)")
    ap.add_argument("--cracksegdiff-dir", default=None, help="Optional directory with CrackSegDiff outputs")
    ap.add_argument("--num-images", type=int, default=500)
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--K", type=int, default=1, choices=[1,2])
    ap.add_argument("--expz", type=float, default=2.0)
    ap.add_argument("--out-csv", default="results/find_batch_metrics.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    set_seed(123)
    struct = auto_discover_find_structure(args.find_root)

    sigmas = [1,3,5,7,9]
    beta = 0.5; c = 0.25; ctheta = 0.125; R = args.radius

    n = len(struct["label"]) if struct["label"] else len(struct["intensity"])
    n = max(1, n)
    rng = np.random.default_rng(123)
    indices = rng.choice(n, size=min(args.num_images, n), replace=False)

    with tqdm_joblib(tqdm(total=len(indices), desc="FIND batch")):
        results = Parallel(n_jobs=-1)(delayed(_process_one)(
            struct, int(idx), sigmas, beta, c, ctheta, R, args.K, args.expz, args.cracksegdiff_dir
        ) for idx in indices)

    results = [r for r in results if r is not None]
    df = pd.DataFrame(results).sort_values("index")
    df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
