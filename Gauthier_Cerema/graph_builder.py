import torch
import numpy as np
from scipy import sparse
import sys
import os

# Ensure we can import StegerHessian logic if needed
try:
    from steger_gpu import StegerHessian
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from steger_gpu import StegerHessian

def build_steger_graph(ix, iy, ixx, ixy, iyy, 
                       R=10.0, 
                       τ=0.05, 
                       dark_ridges=True,
                       batch_size=2048,
                       steger_tolerance=1.0): 
    """
    Builds a sparse graph from Steger filter outputs based on simplified proximity logic.
    
    Args:
        ix, iy, ixx, ixy, iyy: Tensor components of the Hessian (fused or single) on GPU.
        R (float): Connection radius in PIXELS (integer grid).
        τ (float): Threshold for lambda2 magnitude (filtered by sign).
        dark_ridges (bool): If True, valleys (lambda > tau). If False, ridges (lambda < -tau).
        steger_tolerance (float): Max distance from pixel center to be accepted.
                            
    Returns:
        nodes_data (dict): 'coords' (subpixel), 'pixels' (integer), 'directions', 'l2'.
        adj_matrix (scipy.sparse.csr_matrix): Weights = Dist_sub / sqrt(|l2_i*l2_j|).
    """
    
    device = ix.device
    
    # --- 1. Compute Steger Center and Eigenvalues (GPU) ---
    
    # Trace & Discriminant
    trace = ixx + iyy
    disc = torch.sqrt((ixx - iyy)**2 + 4 * ixy**2)
    
    # Eigenvalues (λ+, λ-)
    l_plus = (trace + disc) / 2
    l_minus = (trace - disc) / 2
    
    # Angle and Eigenvectors
    theta = 0.5 * torch.atan2(2 * ixy, ixx - iyy)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    # Sort by absolute magnitude to find normal n (max curvature)
    abs_l_plus = torch.abs(l_plus)
    abs_l_minus = torch.abs(l_minus)
    mask_minus_bigger = abs_l_minus > abs_l_plus
    
    lambda_sorted = torch.where(mask_minus_bigger, l_minus, l_plus)
    
    # Normal vector (nx, ny) corresponds to λ2
    nx = torch.where(mask_minus_bigger, -sin_t, cos_t)
    ny = torch.where(mask_minus_bigger, cos_t, sin_t)
    
    # Tangent vector (ux, uy)
    ux = -ny
    uy = nx
    
    # Calculate t (sub-pixel offset along normal)
    dir_deriv_1 = nx * ix + ny * iy
    mask_nonzero = torch.abs(lambda_sorted) > 1e-6
    t = torch.zeros_like(dir_deriv_1)
    t[mask_nonzero] = -dir_deriv_1[mask_nonzero] / lambda_sorted[mask_nonzero]
    
    # --- 2. Node Selection (GPU) ---
    
    # Condition: Steger Position Validity
    steger_pos_check = (torch.abs(t * nx) <= steger_tolerance) & (torch.abs(t * ny) <= steger_tolerance)
    
    # Condition: Magnitude & Sign
    # "Un noeud i est activé si ... lambda2 * dark_ridges > tau"
    # dark_ridges is boolean. If True (Valleys), we want lambda > tau.
    # If False (Ridges), we want lambda < -tau (so lambda * -1 > tau).
    
    if dark_ridges:
        sign_mag_check = lambda_sorted > τ
    else:
        sign_mag_check = lambda_sorted < -τ
        
    valid_mask = steger_pos_check & sign_mag_check & mask_nonzero
    
    # Debug Stats
    n_total = valid_mask.numel()
    n_final = valid_mask.sum().item()
    print(f"--- Steger Extraction ---")
    print(f"Total Pixels: {n_total}")
    print(f"Selected Nodes: {n_final} (Threshold τ={τ}, Tol={steger_tolerance})")
    
    # Get indices
    valid_indices = torch.nonzero(valid_mask) # (N, 4) -> b, c, y, x
    if valid_indices.shape[0] == 0:
        return None, None
        
    y_idx = valid_indices[:, 2]
    x_idx = valid_indices[:, 3]
    
    # Gather values
    t_val = t.squeeze()[y_idx, x_idx]
    nx_val = nx.squeeze()[y_idx, x_idx]
    ny_val = ny.squeeze()[y_idx, x_idx]
    ux_val = ux.squeeze()[y_idx, x_idx]
    uy_val = uy.squeeze()[y_idx, x_idx]
    l2_val = lambda_sorted.squeeze()[y_idx, x_idx]
    
    # Coordinates
    # 1. Pixel coordinates (Integer) for Radius check
    px_int = x_idx.float()
    py_int = y_idx.float()
    pixel_coords = torch.stack([px_int, py_int], dim=1) # (N, 2)
    
    # 2. Sub-pixel coordinates (Float) for Weight calculation
    px_sub = px_int + t_val * nx_val
    py_sub = py_int + t_val * ny_val
    sub_coords = torch.stack([px_sub, py_sub], dim=1) # (N, 2)
    
    directions = torch.stack([ux_val, uy_val], dim=1)
    
    N = sub_coords.shape[0]
    
    # --- 3. Build Edges (Batched) ---
    all_rows = []
    all_cols = []
    all_weights = []
    
    abs_l2 = torch.abs(l2_val)
    
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        
        # Batch source: Pixel Coords
        p_batch = pixel_coords[i:end]
        
        # 1. Radius Check on PIXEL coordinates
        dists_pixel = torch.cdist(p_batch, pixel_coords)
        
        # Global row indices
        global_row_indices = torch.arange(i, end, device=device).unsqueeze(1)
        col_indices = torch.arange(N, device=device).unsqueeze(0)
        
        # Mask: Pixel Dist <= R AND j > i (Upper triangular)
        mask = (dists_pixel <= R) & (col_indices > global_row_indices)
        
        rows_local, cols = torch.nonzero(mask, as_tuple=True)
        
        if rows_local.numel() == 0:
            continue
            
        rows_global = rows_local + i
        
        # 2. Weight Calculation on SUB-PIXEL coordinates
        # Gather sub-pixel points
        sub_i = sub_coords[rows_global]
        sub_j = sub_coords[cols]
        
        # Euclidean distance between sub-pixels
        d_ij_sub = torch.norm(sub_i - sub_j, dim=1)
        
        # Denominator: sqrt(|l2_i * l2_j|)
        l2_i = abs_l2[rows_global]
        l2_j = abs_l2[cols]
        denom = torch.sqrt(l2_i * l2_j) + 1e-8
        
        weight = d_ij_sub / denom
        
        all_rows.append(rows_global.cpu())
        all_cols.append(cols.cpu())
        all_weights.append(weight.cpu())
        
    # --- 4. Sparse Matrix ---
    if not all_rows:
        adj_matrix = sparse.csr_matrix((N, N))
    else:
        rows_final = torch.cat(all_rows).numpy()
        cols_final = torch.cat(all_cols).numpy()
        data_final = torch.cat(all_weights).numpy()
        
        # Symmetric
        full_rows = np.concatenate([rows_final, cols_final])
        full_cols = np.concatenate([cols_final, rows_final])
        full_data = np.concatenate([data_final, data_final])
        
        adj_matrix = sparse.csr_matrix((full_data, (full_rows, full_cols)), shape=(N, N))
        
    nodes_data = {
        "coords": sub_coords.cpu().numpy(),
        "pixel_coords": pixel_coords.cpu().numpy(),
        "directions": directions.cpu().numpy(),
        "l2": l2_val.cpu().numpy()
    }
    
    return nodes_data, adj_matrix

if __name__ == "__main__":
    print("GPU Graph Builder module loaded.")
