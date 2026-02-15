import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add current directory to path to import steger_gpu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from steger_gpu import StegerHessian

def load_image(path):
    with rasterio.open(path) as src:
        # Read first band
        img = src.read(1)
        # Normalize to 0-1
        img = img.astype(np.float32)
        if img.max() > 0:
            img = (img - img.min()) / (img.max() - img.min())
        return img, src.profile

def save_result(data, profile, output_path):
    # Update profile for output (e.g., float32)
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)

def main():
    # Configuration
    # Using the Avignon image as a proxy for the user's data
    input_path = "data_avignon/Ortho_new_extrait.tif" 
    output_dir = "Gauthier_Cerema/results"
    σ = 3.0  # Adjustable scale parameter (sigma)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        # Try to find any tif in current dir or data_avignon
        possible_files = list(Path("data_avignon").glob("*.tif"))
        if possible_files:
            input_path = str(possible_files[0])
            print(f"Using {input_path} instead.")
        else:
            return

    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Processing on {device}")
    
    # 1. Load Data
    print(f"Loading {input_path}...")
    img_np, profile = load_image(input_path)
    
    # To Tensor (Batch, Channel, H, W)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    
    # 2. Steger Filter
    print(f"Running Steger Filter (σ={σ})...")
    steger = StegerHessian(σ=σ, device=device)
    
    ix, iy, ixx, ixy, iyy = steger.compute_hessian(img_tensor)
    # We no longer need separate eigenvalues call for center computation
    t, valid_mask, nx, ny, l2 = steger.compute_steger_center(ix, iy, ixx, ixy, iyy)
    
    # 3. Process Results
    # Create a "Line Strength" map
    # We use the magnitude of the max curvature (l2) masked by valid points
    line_strength = torch.abs(l2) * valid_mask.float()
    
    # Normalize for visualization
    line_strength_np = line_strength.squeeze().cpu().numpy()
    
    # 4. Save and Plot
    output_path = os.path.join(output_dir, "steger_response.tif")
    save_result(line_strength_np, profile, output_path)
    print(f"Saved result to {output_path}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f"Steger Response (σ={σ})")
    plt.imshow(line_strength_np, cmap='inferno')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Thresholded (> mean + std)")
    thresh = line_strength_np.mean() + line_strength_np.std()
    plt.imshow(line_strength_np > thresh, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "visualization.png")
    plt.savefig(plot_path)
    print(f"Saved visualization to {plot_path}")

if __name__ == "__main__":
    main()
