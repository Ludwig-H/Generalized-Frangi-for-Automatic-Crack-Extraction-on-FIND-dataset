import json
import os

notebook_path = "FIND_Frangi_Fusion_Avignon_Colab.ipynb"

# Code for the NEW cell to be inserted
new_cell_code = r"""# --- Generate and Save Noisy Datasets (Optional) ---
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from tqdm_joblib import tqdm_joblib

save_noisy_images_on_drive = False # @param {type:"boolean"}
NOISE_SAVE_ROOT = "/content/drive/MyDrive/Datasets/FIND/Noisy"

# --- Re-definition of Noise Functions for Standalone Execution ---
# (Must match the Benchmark logic exactly to ensure seeds are identical)
NOISE_BASE_SEED = 1

def _normalize01(x: np.ndarray):
    x = np.asarray(x).astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if (mx - mn) < 1e-12:
        return np.zeros_like(x, dtype=np.float32), mn, mx
    return (x - mn) / (mx - mn), mn, mx

def _denormalize01(x01: np.ndarray, mn: float, mx: float):
    return x01 * (mx - mn) + mn

def add_speckle_intensity(x: np.ndarray, var: float, rng: np.random.Generator):
    if var <= 0: return np.asarray(x).astype(np.float32)
    x01, mn, mx = _normalize01(x)
    n = rng.normal(0.0, np.sqrt(var), size=x01.shape).astype(np.float32)
    y01 = x01 + x01 * n
    y01 = np.clip(y01, 0.0, 1.0)
    return _denormalize01(y01, mn, mx).astype(np.float32)

def add_gaussian_range(x: np.ndarray, sigma: float, rng: np.random.Generator):
    if sigma <= 0: return np.asarray(x).astype(np.float32)
    x01, mn, mx = _normalize01(x)
    y01 = x01 + rng.normal(0.0, sigma, size=x01.shape).astype(np.float32)
    y01 = np.clip(y01, 0.0, 1.0)
    return _denormalize01(y01, mn, mx).astype(np.float32)

def make_noisy_arrays(arrays: dict, idx: int, level_id: int, speckle_var: float = 0.0, range_sigma: float = 0.0, noise_filtered_like_range: bool = True):
    out = dict(arrays)
    # Critical: Seed logic must match the benchmark exactly
    rng_I = np.random.default_rng(NOISE_BASE_SEED + 100000 * idx + 97 * level_id + 1)
    rng_R = np.random.default_rng(NOISE_BASE_SEED + 100000 * idx + 97 * level_id + 2)
    if "intensity" in out and speckle_var > 0:
        out["intensity"] = add_speckle_intensity(out["intensity"], speckle_var, rng_I)
    if "range" in out and range_sigma > 0:
        out["range"] = add_gaussian_range(out["range"], range_sigma, rng_R)
    if noise_filtered_like_range and ("filtered" in out) and range_sigma > 0:
        out["filtered"] = add_gaussian_range(out["filtered"], range_sigma, rng_R)
    return out

def save_single_image_noisy(idx, struct, exp_name, level_id, lvl, speckle_var, range_sigma):
    try:
        # Load Data
        dat = load_modalities_and_gt_by_index(struct, idx)
        
        # Generate Noisy
        noisy_arrays = make_noisy_arrays(dat["arrays"], idx, level_id, speckle_var, range_sigma, noise_filtered_like_range=True)
        
        # Format Path: Root / exp / tag / imXXXXX_modality.png
        tag = f"{lvl:.4f}".replace(".", "p")
        out_dir = os.path.join(NOISE_SAVE_ROOT, exp_name, tag)
        os.makedirs(out_dir, exist_ok=True) # Race condition handled by OS usually fine, or pre-create
        
        base_name = f"im{idx+1:05d}"
        
        # 1. Save Intensity (Grayscale)
        if "intensity" in noisy_arrays:
            img = noisy_arrays["intensity"]
            # Normalize for visualization 0-255
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_uint8 = (img_norm * 255).astype(np.uint8)
            
            fname = f"{base_name}_intensity.png"
            iio.imwrite(os.path.join(out_dir, fname), img_uint8)
            
        # 2. Save Range (Jet Colormap)
        if "range" in noisy_arrays:
            rng_img = noisy_arrays["range"]
            # Normalize strict 0-1 for colormap
            rng_norm = (rng_img - rng_img.min()) / (rng_img.max() - rng_img.min() + 1e-8)
            
            # Apply Jet
            cmap = plt.get_cmap('jet')
            rgba_img = cmap(rng_norm) # Returns (H, W, 4) floats
            rgb_img = rgba_img[:, :, :3] # Keep RGB
            
            rgb_uint8 = (rgb_img * 255).astype(np.uint8)
            
            fname = f"{base_name}_range.png"
            iio.imwrite(os.path.join(out_dir, fname), rgb_uint8)
            
    except Exception as e:
        print(f"Error saving idx {idx}: {e}")

# --- Execution Block ---
if save_noisy_images_on_drive:
    print(f"Generating and saving noisy images to {NOISE_SAVE_ROOT}...")
    
    # Configuration (Must match benchmark)
    speckle_vars = [0.0, 0.01, 0.05, 0.10, 0.3, 0.5]
    range_sigmas = [0.0, 0.01, 0.05, 0.10, 0.3, 0.5]
    experiments = [
        ("speckle_intensity", speckle_vars),
        ("gauss_range", range_sigmas),
        ("both", range_sigmas)
    ]
    
    excluded_ids = [1, 39, 42, 133, 152, 203, 204, 206, 397, 411, 414, 415, 431, 449, 452, 457, 460, 461, 465, 469, 471, 475, 478]
    excluded_ids = [i-1 for i in excluded_ids]
    indices = [i for i in range(500) if i not in excluded_ids]
    
    for exp_name, levels in experiments:
        print(f"Processing experiment: {exp_name}")
        for level_id, lvl in enumerate(levels):
            # Determine params
            if exp_name == "speckle_intensity": sp, sg = lvl, 0.0
            elif exp_name == "gauss_range": sp, sg = 0.0, lvl
            elif exp_name == "both": sp, sg = lvl, lvl
            
            # Run Parallel Saving
            with tqdm_joblib(tqdm(total=len(indices), desc=f"Saving {exp_name} {lvl}")):
                Parallel(n_jobs=8)(delayed(save_single_image_noisy)(
                    idx, struct, exp_name, level_id, lvl, sp, sg
                ) for idx in indices)
    
    print("Done saving images.")
else:
    print("Skipping image generation (checkbox unchecked).")
"""

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Find the Benchmark cell index
target_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "Robust Noise Benchmark" in source_str or "Consolidated and Robust Noise Benchmark" in source_str:
            target_idx = i
            break

if target_idx != -1:
    # 2. Check if the cell already exists (to avoid duplication if run multiple times)
    prev_cell = nb['cells'][target_idx - 1] if target_idx > 0 else None
    if prev_cell and "save_noisy_images_on_drive" in "".join(prev_cell['source']):
        print("Update existing Save Cell...")
        # Update existing
        new_source = [line + '\n' for line in new_cell_code.splitlines()]
        if new_source: new_source[-1] = new_source[-1].rstrip('\n')
        nb['cells'][target_idx - 1]['source'] = new_source
    else:
        print("Inserting New Save Cell...")
        # Create new cell object
        new_source = [line + '\n' for line in new_cell_code.splitlines()]
        if new_source: new_source[-1] = new_source[-1].rstrip('\n')
        
        new_cell_obj = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "id": "save_noisy_images_cell"
            },
            "outputs": [],
            "source": new_source
        }
        
        # Insert BEFORE target_idx
        nb['cells'].insert(target_idx, new_cell_obj)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print("Notebook modified successfully.")
else:
    print("Target benchmark cell not found.")
