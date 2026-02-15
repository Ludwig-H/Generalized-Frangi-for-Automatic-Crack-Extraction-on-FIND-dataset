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
                       steger_tolerance=1.0): # Tolerance for sub-pixel position (increased to 1.0 to recover nodes)
    """
    Builds a sparse graph from Steger filter outputs completely on GPU (where feasible).
    
    Args:
        ix, iy, ixx, ixy, iyy: Tensor components of the Hessian (fused or single) on GPU.
        R (float): Connection radius in pixels.
        τ (float): Threshold for |lambda2|.
        dark_ridges (bool): If True, implies we are looking for ridges/valleys.
        batch_size (int): Number of rows to process at once for distance calculation.
        steger_tolerance (float): Max distance from pixel center to be accepted (default 0.5).
                            
    Returns:
        nodes_data (dict): Contains 'coords' (Nx2), 'directions' (Nx2), 'l2' (N) (as numpy arrays for viz).
        adj_matrix (scipy.sparse.csr_matrix): Sparse adjacency matrix containing DISSIMILARITY weights.
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
    # θ = 0.5 * atan2(2b, a-c)
    theta = 0.5 * torch.atan2(2 * ixy, ixx - iyy)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    # Sort by absolute magnitude to find normal n (max curvature)
    # We want λ2 such that |λ2| >= |λ1|
    abs_l_plus = torch.abs(l_plus)
    abs_l_minus = torch.abs(l_minus)
    
    mask_minus_bigger = abs_l_minus > abs_l_plus
    
    # lambda_sorted is the eigenvalue with max absolute magnitude (λ2)
    lambda_sorted = torch.where(mask_minus_bigger, l_minus, l_plus)
    
    # Normal vector (nx, ny) corresponds to λ2
    # If |λ-| > |λ+|, n = v- = (-sin, cos)
    # Else n = v+ = (cos, sin)
    nx = torch.where(mask_minus_bigger, -sin_t, cos_t)
    ny = torch.where(mask_minus_bigger, cos_t, sin_t)
    
    # Tangent vector (ux, uy) is orthogonal to normal
    # u = (-ny, nx)
    ux = -ny
    uy = nx
    
    # Calculate t (sub-pixel offset along normal)
    # t = - (∇r · n) / λ2
    dir_deriv_1 = nx * ix + ny * iy
    
    # Avoid division by zero
    mask_nonzero = torch.abs(lambda_sorted) > 1e-6
    t = torch.zeros_like(dir_deriv_1)
    t[mask_nonzero] = -dir_deriv_1[mask_nonzero] / lambda_sorted[mask_nonzero]
    
    # --- 2. Node Selection (GPU) ---
    # Filter by magnitude
    mag_check = torch.abs(lambda_sorted) > τ
    
    # Filter by sign
    if dark_ridges:
        # Valleys (dark lines) -> Positive curvature -> lambda > 0
        sign_check = lambda_sorted > 0
    else:
        # Ridges (bright lines) -> Negative curvature -> lambda < 0
        sign_check = lambda_sorted < 0
        
    # Steger condition: sub-pixel offset must be within [-tolerance, tolerance] in both x and y
    # Offset is (t*nx, t*ny)
    steger_pos_check = (torch.abs(t * nx) <= steger_tolerance) & (torch.abs(t * ny) <= steger_tolerance)
    
    # Debug Stats
    n_total = mag_check.numel()
    n_mag = mag_check.sum().item()
    n_sign = sign_check.sum().item()
    n_steger = steger_pos_check.sum().item()
    n_combined = (mag_check & sign_check).sum().item()
    n_final = (mag_check & sign_check & steger_pos_check & mask_nonzero).sum().item()
    
    print(f"--- Debug Node Extraction ---")
    print(f"Total pixels: {n_total}")
    print(f"Pass Magnitude (>{τ}): {n_mag} ({n_mag/n_total:.1%})")
    print(f"Pass Sign (Dark={dark_ridges}): {n_sign} ({n_sign/n_total:.1%})")
    print(f"Pass Mag + Sign: {n_combined} ({n_combined/n_total:.1%})")
    print(f"Pass Steger Position (tol={steger_tolerance}): {n_steger} ({n_steger/n_total:.1%})")
    print(f"Final Nodes: {n_final}")
    
    valid_mask = steger_pos_check & mag_check & sign_check & mask_nonzero
    
    # Get indices (y, x)
    # Note: torch.nonzero returns (N, 4) for (B, C, H, W)
    # We assume B=1, C=1.
    valid_indices = torch.nonzero(valid_mask) # shape (N, 4) -> b, c, y, x
    
    if valid_indices.shape[0] == 0:
        return None, None
        
    y_idx = valid_indices[:, 2]
    x_idx = valid_indices[:, 3]
    
    # Gather values (GPU)
    t_val = t.squeeze()[y_idx, x_idx]
    nx_val = nx.squeeze()[y_idx, x_idx]
    ny_val = ny.squeeze()[y_idx, x_idx]
    ux_val = ux.squeeze()[y_idx, x_idx]
    uy_val = uy.squeeze()[y_idx, x_idx]
    l2_val = lambda_sorted.squeeze()[y_idx, x_idx] # Signed value
    abs_l2_val = torch.abs(l2_val)
    
    # Compute sub-pixel coordinates (GPU)
    px = x_idx.float() + t_val * nx_val
    py = y_idx.float() + t_val * ny_val
    
    coords = torch.stack([px, py], dim=1) # (N, 2)
    directions = torch.stack([ux_val, uy_val], dim=1) # (N, 2)
    
    N = coords.shape[0]
    print(f"Number of Steger nodes extracted: {N}")
    
    # --- 3. Build Edges (GPU Batched Radius Search) ---
    
    all_rows = []
    all_cols = []
    all_sims = []
    
    # Iterate in batches to avoid OOM with NxN matrix
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        
        # Source coordinates for this batch
        c_batch = coords[i:end] # (B, 2)
        
        # Compute distances to ALL nodes (N)
        # cdist is efficient on GPU
        dists = torch.cdist(c_batch, coords) # (B, N)
        
        # Mask: distance < R AND distance > 0 (exclude self)
        # We also enforce j > global_row to create upper triangular part only (undirected graph)
        # Construct global row indices
        global_row_indices = torch.arange(i, end, device=device).unsqueeze(1) # (B, 1)
        col_indices = torch.arange(N, device=device).unsqueeze(0) # (1, N)
        
        # Keep only j > i (avoid duplicates and self-loops)
        mask = (dists <= R) & (col_indices > global_row_indices)
        
        # Get indices of valid edges in this batch
        # rows_local are relative to the batch (0..B-1)
        rows_local, cols = torch.nonzero(mask, as_tuple=True)
        
        if rows_local.numel() == 0:
            continue
            
        rows_global = rows_local + i
        
        # --- 4. Compute Weights (Dissimilarity) on GPU ---
        
        # Gather vectors
        P_i = coords[rows_global]
        P_j = coords[cols]
        vec_ij = P_j - P_i
        
        # Normalize vec_ij
        dist_vals = dists[rows_local, cols]
        # Avoid div by zero (mask ensures dist > 0 effectively since j > i)
        vec_ij_norm = vec_ij / (dist_vals.unsqueeze(1) + 1e-8)
        
        # Directions and Lambda2
        u_i = directions[rows_global]
        u_j = directions[cols]
        l2_i = abs_l2_val[rows_global]
        l2_j = abs_l2_val[cols]
        
        # Cross Product Dissimilarity
        # 2D cross product: x1*y2 - x2*y1
        def cross_2d(v1, v2):
            return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            
        cross_i = torch.abs(cross_2d(vec_ij_norm, u_i))
        cross_j = torch.abs(cross_2d(vec_ij_norm, u_j))
        
        # Dissimilarity formula: Distance * ( (1 + Cross_i)/|L2_i| + (1 + Cross_j)/|L2_j| )
        # We add 1.0 to cross product so that even perfectly aligned segments have a cost 
        # inversely proportional to their intensity (stronger = cheaper).
        
        eps = 1e-6
        term_i = (1.0 + cross_i) / (l2_i + eps)
        term_j = (1.0 + cross_j) / (l2_j + eps)
        
        dissim = dist_vals * (term_i + term_j) / 2.0
        
        # Store results (move to CPU to save GPU RAM for next batches)
        all_rows.append(rows_global.cpu())
        all_cols.append(cols.cpu())
        all_sims.append(dissim.cpu())
    
    # --- 5. Construct Sparse Matrix ---
    
    if not all_rows:
        adj_matrix = sparse.csr_matrix((N, N))
    else:
        # Concatenate all batches
        rows_final = torch.cat(all_rows).numpy()
        cols_final = torch.cat(all_cols).numpy()
        data_final = torch.cat(all_sims).numpy()
        
        # Make symmetric (since we computed j > i)
        full_rows = np.concatenate([rows_final, cols_final])
        full_cols = np.concatenate([cols_final, rows_final])
        full_data = np.concatenate([data_final, data_final])
        
        adj_matrix = sparse.csr_matrix((full_data, (full_rows, full_cols)), shape=(N, N))
    
    # Prepare return data (CPU numpy for visualization)
    nodes_data = {
        "coords": coords.cpu().numpy(),
        "directions": directions.cpu().numpy(),
        "l2": l2_val.cpu().numpy()
    }
    
    return nodes_data, adj_matrix

if __name__ == "__main__":
    print("GPU Graph Builder module loaded.")
