# scripts/run_noise_test_colab.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from skimage.morphology import binary_closing, binary_opening, disk
from PIL import Image

# --- Setup Paths ---
sys.path.append(os.path.abspath("src"))

from frangi_fusion import (
    auto_discover_find_structure, load_modalities_and_gt_by_index, to_gray,
    compute_hessians_per_scale, fuse_hessians_per_scale,
    build_frangi_similarity_graph, distances_from_similarity, triangle_connectivity_graph,
    largest_connected_component, mst_on_cluster, extract_backbone_centrality,
    skeleton_from_mst_graph, skeletonize_lee, thicken, jaccard_index, tversky_index,
    wasserstein_distance_skeletons
)

# --- Configuration ---
IMG_IDX = 133  # Index of image to test (0-based) - using 133 as in notebook analysis
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30] # Gaussian Sigma added to [0,1] image
N_JOBS = 1

# Algorithm Params
SIGMA = [1, 3, 5]
BETA = 0.5
C = 0.25
C_THETA = 0.125
R = 5
K = 1
DARK_RIDGES = True
MODE = "minus"
THRESHOLD_MASK = 0.75
EXPZ = 1
F_THRESHOLD = 0.40 # Centrality threshold
MIN_CENTRALITY = 0.10
THICKNESS = 3
DIAM_AFFIN = 2

# Weights: IMPORTANT - Use both to test the "safety net" hypothesis
WEIGHTS = {"intensity": 0.5, "range":0.5, "filtered":0.0, "fused":0.0}

def add_gaussian_noise(image, sigma):
    """
    Adds Gaussian noise to an image.
    Image is expected to be [0, 255] uint8 or float.
    Converted to float [0,1], noise added, clipped, back to input range/type.
    """
    if sigma <= 0:
        return image.copy()
    
    # Normalize to [0, 1] float
    img_float = image.astype(np.float32)
    if img_float.max() > 1.0:
        img_float /= 255.0
        
    noise = np.random.normal(0, sigma, img_float.shape).astype(np.float32)
    noisy = img_float + noise
    noisy = np.clip(noisy, 0, 1)
    
    # Return in same range as input if it was uint8
    if image.dtype == np.uint8:
        return (noisy * 255).astype(np.uint8)
    return noisy

def process_noise_level(struct, img_idx, noise_std, target_modality="both"):
    try:
        dat = load_modalities_and_gt_by_index(struct, img_idx)
        
        # Prepare GT
        base_original = dat["arrays"].get("intensity")
        if base_original is None:
             base_original = next(iter(dat["arrays"].values()))

        gt_raw = dat["arrays"].get("label", np.zeros_like(base_original))
        gt = (gt_raw > 0).astype(np.uint8)
        if DIAM_AFFIN > 0:
            gt = binary_closing(gt, footprint=disk(DIAM_AFFIN)).astype(np.uint8)
            gt = binary_opening(gt, footprint=disk(DIAM_AFFIN)).astype(np.uint8)
        sk_gt = skeletonize_lee(gt)
        sk_gt_thick = thicken(sk_gt, pixels=THICKNESS)
        
        # --- CRITICAL STEP: Add Noise according to target_modality ---
        noisy_mods_arrays = {}
        for mod, arr in dat["arrays"].items():
            if mod in WEIGHTS and WEIGHTS[mod] > 0:
                # Decide whether to noise this specific modality
                should_noise = False
                if target_modality == "both":
                    should_noise = True
                elif target_modality == mod:
                    should_noise = True
                
                if should_noise:
                    noisy_mods_arrays[mod] = add_gaussian_noise(arr, noise_std)
                else:
                    noisy_mods_arrays[mod] = arr
            else:
                noisy_mods_arrays[mod] = arr 
        
        # Compute Hessians on NOISY images
        hessians = {}
        for mod, arr in noisy_mods_arrays.items():
            if mod in WEIGHTS and WEIGHTS[mod] > 0:
                 hessians[mod] = compute_hessians_per_scale(to_gray(arr), SIGMA)
        
        # Pipeline
        fused_H = fuse_hessians_per_scale(hessians, WEIGHTS)
        
        coords, _, S = build_frangi_similarity_graph(
            fused_H, BETA, C, C_THETA, R,
            candidate_mask=None, threshold_mask=THRESHOLD_MASK, dark_ridges=DARK_RIDGES
        )
        
        D = distances_from_similarity(S, MODE)
        if K == 2:
            D = triangle_connectivity_graph(coords, D)
            
        D_cc, idx_nodes = largest_connected_component(D)
        
        # Simplified clustering: Skip HDBSCAN, take whole CC (as per notebook)
        labels = np.zeros(D_cc.shape[0], dtype=int)
        
        sk_pred = np.zeros_like(base_original, dtype=np.uint8)
        
        if D_cc.shape[0] > 10: 
            sub_coords = coords[idx_nodes]
            all_edges = []
            
            # Since labels are all 0, this loop runs once for the whole CC
            for lab in np.unique(labels):
                if lab < 0: continue
                cl = np.where(labels == lab)[0]
                if cl.size < 3: continue
                
                mst = mst_on_cluster(D_cc, cl)
                global_indices = idx_nodes[cl]
                S_cluster = S[global_indices, :][:, global_indices]
                
                nodes_kept, skel_graph = extract_backbone_centrality(
                    mst, f_threshold=F_THRESHOLD, S=S_cluster, 
                    take_similarity=True, min_centrality=MIN_CENTRALITY
                )
                
                segs = skeleton_from_mst_graph(skel_graph, sub_coords[cl], nodes_kept, S=S_cluster, take_similarity=True)
                if segs.shape[0] > 0: all_edges.append(segs)
            
            if all_edges:
                fault_edges = np.vstack(all_edges)
                H, W = base_original.shape[:2]
                mask = np.zeros((H, W), dtype=np.uint8)
                for e in fault_edges:
                    r0,c0,r1,c1,_ = e
                    num = int(max(abs(r1-r0),abs(c1-c0))+1)
                    rr = np.linspace(r0, r1, num)
                    cc = np.linspace(c0, c1, num)
                    rr = np.clip(rr.astype(int), 0, H-1)
                    cc = np.clip(cc.astype(int), 0, W-1)
                    mask[rr,cc] = 1
                sk_pred = skeletonize_lee(mask)
        
        sk_pred_thick = thicken(sk_pred, pixels=THICKNESS)
        
        jac = jaccard_index(sk_pred_thick, sk_gt_thick)
        tvs = tversky_index(sk_pred_thick, sk_gt_thick, alpha=1.0, beta=0.5)
        
        return {
            "noise": noise_std,
            "jaccard": jac,
            "tversky": tvs
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error at noise {noise_std}: {e}")
        return {"noise": noise_std, "jaccard": 0, "tversky": 0}

def run_scenario(struct, scenario_name, noise_levels):
    print(f"\n--- Running Scenario: {scenario_name.upper()} ---")
    results = []
    for n in noise_levels:
        # print(f"Processing noise sigma: {n}...")
        res = process_noise_level(struct, IMG_IDX, n, target_modality=scenario_name)
        results.append(res)
        # print(f"  -> Tversky: {res['tversky']:.4f}")
    return pd.DataFrame(results)

def main():
    if not os.path.exists("data_find"):
        print("Warning: 'data_find' directory not found in current path.")
        
    struct = auto_discover_find_structure("data_find")
    if not struct["intensity"]:
        print("No data found in data_find structure. Check paths.")
        return

    print(f"Running Noise Robustness Comparative Test on Image Index {IMG_IDX}")
    print(f"Noise Levels (Gaussian Sigma): {NOISE_LEVELS}")
    print(f"Weights Used: {WEIGHTS}")
    
    # 1. Noise ONLY Intensity
    df_int = run_scenario(struct, "intensity", NOISE_LEVELS)
    print("Results (Noise on Intensity ONLY):")
    print(df_int[["noise", "jaccard", "tversky"]])

    # 2. Noise ONLY Range
    df_rng = run_scenario(struct, "range", NOISE_LEVELS)
    print("Results (Noise on Range ONLY):")
    print(df_rng[["noise", "jaccard", "tversky"]])

    # 3. Noise BOTH
    df_both = run_scenario(struct, "both", NOISE_LEVELS)
    print("Results (Noise on BOTH):")
    print(df_both[["noise", "jaccard", "tversky"]])
    
    # Comparative Plot
    plt.figure(figsize=(10,6))
    plt.plot(df_int['noise'], df_int['tversky'], marker='o', label='Noise Intensity Only', linestyle='--')
    plt.plot(df_rng['noise'], df_rng['tversky'], marker='s', label='Noise Range Only', linestyle='--')
    plt.plot(df_both['noise'], df_both['tversky'], marker='x', label='Noise Both', linewidth=2, color='red')
    
    plt.xlabel('Noise Sigma')
    plt.ylabel('Tversky Score')
    plt.title(f'Robustness Comparison (Img {IMG_IDX})')
    plt.grid(True)
    plt.legend()
    plt.savefig('noise_comparison.png')
    print("\nComparative plot saved to noise_comparison.png")

if __name__ == "__main__":
    main()