import torch
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
import sys
import os

# Ensure we can import StegerHessian logic if needed, 
# though we might re-implement the specific node extraction here for flexibility.
# Assuming this script is run from a context where we can import steger_gpu
try:
    from steger_gpu import StegerHessian
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from steger_gpu import StegerHessian

def build_steger_graph(ix, iy, ixx, ixy, iyy, 
                       R=10.0, 
                       tau=0.1, 
                       dark_ridges=True):
    """
    Builds a sparse similarity graph from Steger filter outputs.
    
    Args:
        ix, iy, ixx, ixy, iyy: Tensor components of the Hessian (fused or single).
        R (float): Connection radius in pixels.
        tau (float): Threshold for |lambda2|.
        dark_ridges (bool): If True, implies we are looking for ridges/valleys based on lambda2 magnitude.
                            (Note: In current steger_gpu, |lambda2| is always the max absolute eigenvalue).
                            
    Returns:
        nodes_data (dict): Contains 'coords' (Nx2), 'directions' (Nx2), 'lambda2' (N).
        adj_matrix (scipy.sparse.csr_matrix): Sparse adjacency matrix with similarity weights.
    """
    
    # 1. Compute Steger Center and Eigenvalues
    # We can reuse the static methods or instances from StegerHessian if available,
    # or just re-implement the lightweight math here using torch for GPU acceleration.
    
    # Ensure inputs are tensors
    device = ix.device
    
    # Eigenvalues
    trace = ixx + iyy
    disc = torch.sqrt((ixx - iyy)**2 + 4 * ixy**2)
    # With the recent fix, lambda2 is the max absolute eigenvalue
    # But we need the signed values to determine orientation correctly if needed
    # However, steger_gpu's center computation uses ixx, ixy, iyy directly.
    
    # Re-compute raw eigenvalues for direction (theta)
    # The fix in steger_gpu sorted them by abs, which is good for filtering by magnitude.
    # Let's trust the input if it comes from the modified steger_gpu, 
    # but we need to compute 't' and 'nx, ny'.
    
    # Orientation (theta) for the max curvature direction (normal to the ridge)
    # theta = 0.5 * atan2(2*ixy, ixx - iyy)
    theta = 0.5 * torch.atan2(2 * ixy, ixx - iyy)
    nx = torch.cos(theta)
    ny = torch.sin(theta)
    
    # Direction of the line (perpendicular to normal)
    # ux = -ny, uy = nx (Rotation by 90 deg)
    ux = -ny
    uy = nx
    
    # Calculate t (sub-pixel offset along the normal)
    # term1 = nx*ix + ny*iy
    # term2 = nx^2*ixx + 2*nx*ny*ixy + ny^2*iyy  (Approx curvature)
    dir_deriv_1 = nx * ix + ny * iy
    dir_deriv_2 = (nx**2)*ixx + 2*nx*ny*ixy + (ny**2)*iyy
    
    # Avoid div by zero
    mask_nonzero = torch.abs(dir_deriv_2) > 1e-6
    t = torch.zeros_like(dir_deriv_1)
    t[mask_nonzero] = -dir_deriv_1[mask_nonzero] / dir_deriv_2[mask_nonzero]
    
    # 2. Select Nodes (Pixels "à la Steger")
    # Criteria: |t| <= 0.5 AND |lambda2| > tau
    # We need the magnitude of lambda2. 
    # Since we recomputed dir_deriv_2 which IS the curvature in the normal direction (~lambda2),
    # let's use that or the eigenvalues from the matrix.
    # Let's calculate exact eigenvalues again to be safe and consistent with the thresholding.
    l1_raw = (trace - disc) / 2
    l2_raw = (trace + disc) / 2
    # We want the one with max abs value
    abs_l1 = torch.abs(l1_raw)
    abs_l2 = torch.abs(l2_raw)
    lambda_max = torch.maximum(abs_l1, abs_l2)
    
    valid_mask = (torch.abs(t) <= 0.5) & (lambda_max > tau) & mask_nonzero
    
    # Extract data for valid pixels
    # Indices (y, x)
    batch, ch, H, W = ix.shape
    # Assuming batch=1, ch=1 for now
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    
    valid_indices = torch.nonzero(valid_mask.squeeze()) # (N, 2) -> y, x
    if valid_indices.shape[0] == 0:
        return None, None
        
    y_idx = valid_indices[:, 0]
    x_idx = valid_indices[:, 1]
    
    # Gather values
    t_val = t.squeeze()[y_idx, x_idx]
    nx_val = nx.squeeze()[y_idx, x_idx]
    ny_val = ny.squeeze()[y_idx, x_idx]
    ux_val = ux.squeeze()[y_idx, x_idx]
    uy_val = uy.squeeze()[y_idx, x_idx]
    l2_val = lambda_max.squeeze()[y_idx, x_idx]
    
    # Compute sub-pixel coordinates
    # P = (x + t*nx, y + t*ny)  <-- Check Steger formula logic
    # Usually Steger correction is added to (x,y) along the normal direction (nx, ny)
    # The offset is t * n
    
    px = x_idx.float() + t_val * nx_val
    py = y_idx.float() + t_val * ny_val
    
    # Move to CPU for graph construction (scipy cKDTree)
    coords = torch.stack([px, py], dim=1).cpu().numpy() # (N, 2)
    directions = torch.stack([ux_val, uy_val], dim=1).cpu().numpy() # (N, 2)
    l2_scores = l2_val.cpu().numpy()
    
    N = coords.shape[0]
    print(f"Number of Steger nodes extracted: {N}")
    
    # 3. Build Edges (Radius Search)
    tree = cKDTree(coords)
    # query_pairs returns a set of pairs (i, j) with i < j
    pairs = tree.query_pairs(r=R, output_type='ndarray') # (M, 2)
    
    if pairs.shape[0] == 0:
        return {"coords": coords, "l2": l2_scores}, sparse.csr_matrix((N, N))

    i_indices = pairs[:, 0]
    j_indices = pairs[:, 1]
    
    # Vector Pj - Pi
    P_i = coords[i_indices]
    P_j = coords[j_indices]
    vec_ij = P_j - P_i
    
    # Normalize
    dist_ij = np.linalg.norm(vec_ij, axis=1)
    # Avoid zero division (should not happen if i!=j)
    mask_dist = dist_ij > 1e-6
    vec_ij_norm = np.zeros_like(vec_ij)
    vec_ij_norm[mask_dist] = vec_ij[mask_dist] / dist_ij[mask_dist, None]
    
    # Directions
    u_i = directions[i_indices]
    u_j = directions[j_indices]
    
    # 4. Compute Dissimilarity / Similarity
    # Dissimilarity based on cross product
    # Cross product 2D (x1*y2 - x2*y1) gives sin(angle) (z-component)
    # We want alignment -> cross product should be 0.
    
    def cross_2d(v1, v2):
        return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    cross_i = np.abs(cross_2d(vec_ij_norm, u_i))
    cross_j = np.abs(cross_2d(vec_ij_norm, u_j))
    
    # Dissimilarity metric requested: 
    # "Dissimilarité à partir du produit vectoriel... entre vecteur reliant les centres et direction"
    # Let's define dissim = (cross_i + cross_j) / 2
    # This is 0 if perfectly aligned, 1 if perpendicular.
    dissim = (cross_i + cross_j) / 2.0
    
    # The user asked for an "Arête de Similarité".
    # Usually Similarity = exp(- Dissim / scale) or 1 - Dissim.
    # Let's use a simple linear conversion for now, or exponential.
    # Since cross product is in [0, 1], let's try:
    # Similarity = 1 - Dissim (Clip at 0)
    
    similarity = 1.0 - dissim
    similarity = np.clip(similarity, 0.0, 1.0)
    
    # Optionally weight by distance? "reliés par une arête... s'ils sont à moins de R pixels"
    # The user didn't explicitly ask for distance weighting in the similarity formula, 
    # only the geometric alignment. But typically closer is better.
    # For now, let's strictly follow "Dissimilarity from cross product".
    
    # Construct Sparse Matrix
    # Symmetric matrix
    row = np.concatenate([i_indices, j_indices])
    col = np.concatenate([j_indices, i_indices])
    data = np.concatenate([similarity, similarity])
    
    adj_matrix = sparse.csr_matrix((data, (row, col)), shape=(N, N))
    
    nodes_data = {
        "coords": coords,
        "directions": directions,
        "l2": l2_scores
    }
    
    return nodes_data, adj_matrix

if __name__ == "__main__":
    # Test stub
    print("Graph Builder module loaded.")
