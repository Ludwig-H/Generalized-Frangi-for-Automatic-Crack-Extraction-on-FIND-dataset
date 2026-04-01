import json
import os

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")],
        "outputs": []
    })

add_md("""# Extraction de Fissures via Frangi Graph Généralisé sur GPU (A100)
## Benchmark : CrackForest

Ce Colab implémente l'approche non supervisée pour le dataset **CrackForest**, exploitant la puissance du **GPU (PyTorch)** pour accélérer le filtrage Hessien et la construction du graphe de similarité.

Ici, nous n'utilisons qu'une **seule modalité** (le visible RGB converti en niveau de gris).

### Caractéristiques de l'implémentation :
- Chargement des données (.jpg pour l'image et .mat pour la Ground Truth).
- Calculs matriciels Hessiens et Valeurs Propres 100% sur GPU (`torch.Tensor`).
- Construction du graphe creux (Sparse) économe en VRAM.
- Algorithme d'extraction topologique (Arbre Couvrant de Poids Minimum + Centralité).
- **Analyse de sensibilité** des paramètres clés du graphe Généralisé.""")

add_code("""!pip install -q scipy numpy matplotlib pandas gdown POT scikit-image tqdm

import os
import zipfile
from pathlib import Path

# Téléchargement du dataset CrackForest depuis GitHub
if not os.path.exists('CrackForest-dataset'):
    print("Téléchargement du dataset CrackForest depuis GitHub...")
    !git clone https://github.com/Ludwig-H/CrackForest-dataset.git
else:
    print("Dataset déjà présent.")""")

add_md("""## 1. Dataloader

Nous chargeons les 118 premières images et la vérité terrain associée (qui est déjà un squelette binaire dans la structure MATLAB).""")

add_code("""import torch
import numpy as np
import cv2
import scipy.io as sio
from pathlib import Path
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class CrackForestDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = Path(root_dir) / 'CrackForest-dataset' / 'image'
        self.gt_dir = Path(root_dir) / 'CrackForest-dataset' / 'groundTruth'
        
        # 118 premières images
        self.identifiants = [f"{i:03d}" for i in range(1, 119)]
        print(f"Dataset chargé avec {len(self.identifiants)} images.")

    def __len__(self):
        return len(self.identifiants)

    def __getitem__(self, idx):
        id_courant = self.identifiants[idx]
        
        path_img = self.img_dir / f"{id_courant}.jpg"
        path_gt  = self.gt_dir / f"{id_courant}.mat"
        
        # Chargement en N&B
        img_vis = cv2.imread(str(path_img), cv2.IMREAD_GRAYSCALE)
        
        # Normalisation
        vis_t = torch.from_numpy(img_vis).float() / 255.0
        
        # Chargement de la Ground Truth MATLAB
        mat = sio.loadmat(str(path_gt))
        # La structure du dataset contient 'Segmentation' ou 'Boundaries'
        # La vérité terrain est déjà squelettisée dans 'Boundaries', on extrait avec [0,0]['Boundaries']
        gt_data = mat['groundTruth'][0, 0]['Boundaries']
        
        # Binarisation stricte
        gt_clean = (gt_data > 0).astype(np.float32)
        gt_t = torch.from_numpy(gt_clean)
        
        return {
            'id': id_courant,
            'visible': vis_t,
            'gt': gt_t
        }

# Initialisation
dataset = CrackForestDataset('.')""")

add_md("""## 2. Calcul Hessien Multi-échelles sur GPU""")

add_code("""import torch.nn.functional as F
import math

class FrangiHessianGPU:
    def __init__(self, scales, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.scales = scales
        self.device = device
        self.kernels = self._precompute_kernels()

    def _precompute_kernels(self):
        kernels = {}
        for s in self.scales:
            size = int(math.ceil(4 * s)) * 2 + 1
            x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=self.device)
            variance = s ** 2
            g_x = (1 / (math.sqrt(2 * math.pi) * s)) * torch.exp(-x**2 / (2 * variance))
            g_x_1 = -(x / variance) * g_x
            g_x_2 = ((x**2 / (variance**2)) - (1 / variance)) * g_x
            kernels[s] = {'0': g_x, '1': g_x_1, '2': g_x_2}
        return kernels

    def compute_hessian(self, image_tensor, s):
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
            
        image_tensor = image_tensor.to(self.device)
        
        k0 = self.kernels[s]['0'].view(1, 1, 1, -1)
        k1 = self.kernels[s]['1'].view(1, 1, 1, -1)
        k2 = self.kernels[s]['2'].view(1, 1, 1, -1)

        k0_T = k0.transpose(2, 3)
        k1_T = k1.transpose(2, 3)
        k2_T = k2.transpose(2, 3)

        pad_size = k0.shape[3] // 2
        
        def convolve_sep(x, k_h, k_v):
            p_h = F.pad(x, (pad_size, pad_size, 0, 0), mode='replicate')
            t = F.conv2d(p_h, k_h)
            p_v = F.pad(t, (0, 0, pad_size, pad_size), mode='replicate')
            return F.conv2d(p_v, k_v)

        ixx = convolve_sep(image_tensor, k2, k0_T)
        iyy = convolve_sep(image_tensor, k0, k2_T)
        ixy = convolve_sep(image_tensor, k1, k1_T)

        return ixx.squeeze(), ixy.squeeze(), iyy.squeeze()

    def compute_eigenvalues_and_vectors(self, ixx, ixy, iyy):
        trace = ixx + iyy
        disc = torch.sqrt((ixx - iyy)**2 + 4 * ixy**2)
        
        l_plus = (trace + disc) / 2
        l_minus = (trace - disc) / 2
        
        abs_l_plus = torch.abs(l_plus)
        abs_l_minus = torch.abs(l_minus)
        
        mask_minus_bigger = abs_l_minus > abs_l_plus
        
        λ2 = torch.where(mask_minus_bigger, l_minus, l_plus)
        λ1 = torch.where(mask_minus_bigger, l_plus, l_minus)

        θ = 0.5 * torch.atan2(2 * ixy, ixx - iyy)
        return λ1, λ2, θ""")

add_md("""## 3. Construction du Graphe (Frangi Graph)

Application de la réponse Frangi et sparsification.""")

add_code("""from scipy.sparse import coo_matrix

def extract_frangi_graph_gpu(imgs_dict, weights, Σ=[5.0], R=3,
                             ss=1.0, si=0.25, sa=0.3, τ=0.2, min_rel_size=150.0, device='cuda'):
    import time
    t0 = time.time()
    
    fh = FrangiHessianGPU(Σ, device=device)
    
    scale_data = []
    max_S_global = None
    H, W = None, None
    
    # 1. Fusion Multimodale par échelle
    for σ in Σ:
        fused_ixx = None
        
        for mod, w in weights.items():
            if w > 0:
                ixx, ixy, iyy = fh.compute_hessian(imgs_dict[mod], σ)
                if H is None:
                    H, W = ixx.shape
                
                trace = ixx + iyy
                disc = torch.sqrt((ixx - iyy)**2 + 4 * ixy**2)
                spectral_norm_local = (torch.abs(trace) + disc) / 2.0
                max_norm = torch.max(spectral_norm_local) + 1e-8
                
                if fused_ixx is None:
                    fused_ixx = w * (ixx / max_norm)
                    fused_ixy = w * (ixy / max_norm)
                    fused_iyy = w * (iyy / max_norm)
                else:
                    fused_ixx += w * (ixx / max_norm)
                    fused_ixy += w * (ixy / max_norm)
                    fused_iyy += w * (iyy / max_norm)
                    
        λ1, λ2, θ = fh.compute_eigenvalues_and_vectors(fused_ixx, fused_ixy, fused_iyy)
        
        mask_pos = λ2 > 0
        R_B = torch.zeros_like(λ2)
        R_B[mask_pos] = torch.abs(λ1[mask_pos]) / (λ2[mask_pos] + 1e-8)
        
        S_norm = torch.zeros_like(λ2)
        S_norm[mask_pos] = λ2[mask_pos]
        
        if max_S_global is None: max_S_global = S_norm.clone()
        else: max_S_global = torch.max(max_S_global, S_norm)
            
        scale_data.append((R_B, S_norm, θ, mask_pos))
        
    if device == 'cuda': torch.cuda.synchronize()
    t_hessian = time.time()
        
    τ_1 = max_S_global.max() * 0.01 
    candidates_mask = max_S_global > τ_1
    coords = torch.nonzero(candidates_mask).float()
    N = coords.shape[0]
    
    if N == 0:
        return np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W)), {}, {'tau_mask': np.zeros((H, W)), 'comp_mask': np.zeros((H, W))}
    
    index_map = torch.full((H, W), -1, dtype=torch.long, device=device)
    index_map[candidates_mask] = torch.arange(N, device=device)
    
    padded_index_map = torch.nn.functional.pad(index_map.unsqueeze(0).unsqueeze(0).float(), (R, R, R, R), value=-1).long()
    
    patches = padded_index_map[0, 0].unfold(0, 2*R+1, 1).unfold(1, 2*R+1, 1)
    
    y_coords, x_coords = torch.nonzero(candidates_mask, as_tuple=True)
    cand_patches = patches[y_coords, x_coords]
    
    dy, dx = torch.meshgrid(torch.arange(-R, R+1, device=device), torch.arange(-R, R+1, device=device), indexing='ij')
    dist_sq = dx**2 + dy**2
    valid_dist_mask = (dist_sq <= R**2) & (dist_sq > 0)
    half_mask = (dy > 0) | ((dy == 0) & (dx > 0))
    valid_mask = valid_dist_mask & half_mask
    
    valid_neighbors = cand_patches[:, valid_mask]
    
    source_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, valid_neighbors.shape[1])
    valid_pairs_mask = valid_neighbors != -1
    
    i_idx_t = source_idx[valid_pairs_mask]
    j_idx_t = valid_neighbors[valid_pairs_mask]
    
    if len(i_idx_t) == 0:
        return max_S_global.cpu().numpy(), np.zeros((H, W)), np.zeros((H, W)), {}, {'tau_mask': np.zeros((H, W)), 'comp_mask': np.zeros((H, W))}
    
    if device == 'cuda': torch.cuda.synchronize()
    t_unfold = time.time()
    
    S_ij_max = torch.zeros(len(i_idx_t), device=device, dtype=torch.float32)
    
    for R_B, S_norm, θ, mask_pos in scale_data:
        rb_c = R_B[candidates_mask]
        s_c  = S_norm[candidates_mask]
        t_c  = θ[candidates_mask]
        
        rb_sum = rb_c[i_idx_t] + rb_c[j_idx_t]
        S_shape = torch.exp(-0.5 * (rb_sum / ss)**2)
        
        s_prod = s_c[i_idx_t] * s_c[j_idx_t]
        S_int = 1 - torch.exp(-0.5 * (s_prod / si)**2)
        
        dt = t_c[i_idx_t] - t_c[j_idx_t]
        S_align = torch.exp(-0.5 * (torch.sin(dt) / sa)**2)
        
        S_ij = S_shape * S_int * S_align
        S_ij_max = torch.max(S_ij_max, S_ij)
        
    if device == 'cuda': torch.cuda.synchronize()
    t_sim = time.time()
        
    dist_ij_t = torch.norm(coords[i_idx_t] - coords[j_idx_t], dim=1)
    d_ij = (1 - S_ij_max) * dist_ij_t + 1e-8
    
    S_cpu = S_ij_max.cpu().numpy()
    d_cpu = d_ij.cpu().numpy()
    i_cpu = i_idx_t.cpu().numpy()
    j_cpu = j_idx_t.cpu().numpy()
    
    import numpy as np
    from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order, connected_components
    
    num_edges = len(S_cpu)
    num_to_keep_edges = max(1, int(num_edges * τ))
    
    if num_edges > num_to_keep_edges:
        threshold_edge = np.partition(S_cpu, -num_to_keep_edges)[-num_to_keep_edges]
        edge_mask = S_cpu >= threshold_edge
    else:
        edge_mask = np.ones(num_edges, dtype=bool)
    
    i_v = i_cpu[edge_mask]
    j_v = j_cpu[edge_mask]
    S_v = S_cpu[edge_mask]
    d_v = d_cpu[edge_mask]
    
    node_sim_max = np.zeros(N, dtype=np.float32)
    if len(S_v) > 0:
        np.maximum.at(node_sim_max, i_v, S_v)
        np.maximum.at(node_sim_max, j_v, S_v)
        
    N_total = H * W
    num_to_keep_nodes = max(1, int(N_total * τ))
    
    valid_nodes_candidates = np.unique(np.concatenate([i_v, j_v])) if len(i_v) > 0 else np.array([], dtype=np.int32)
    
    if len(valid_nodes_candidates) > num_to_keep_nodes:
        sims_candidates = node_sim_max[valid_nodes_candidates]
        threshold_node = np.partition(sims_candidates, -num_to_keep_nodes)[-num_to_keep_nodes]
        valid_nodes = valid_nodes_candidates[sims_candidates >= threshold_node]
    else:
        valid_nodes = valid_nodes_candidates
        
    if len(valid_nodes) > 0 and len(i_v) > 0:
        keep_node_mask = np.zeros(N, dtype=bool)
        keep_node_mask[valid_nodes] = True
        
        final_edge_mask = keep_node_mask[i_v] & keep_node_mask[j_v]
        i_v = i_v[final_edge_mask]
        j_v = j_v[final_edge_mask]
        S_v = S_v[final_edge_mask]
        d_v = d_v[final_edge_mask]
    else:
        i_v, j_v, S_v, d_v = np.array([]), np.array([]), np.array([]), np.array([])
    
    remap = np.full(N, -1, dtype=np.int32)
    remap[valid_nodes] = np.arange(len(valid_nodes))
    
    i_mapped = remap[i_v]
    j_mapped = remap[j_v]
    
    orig_coords_cpu = coords.cpu().numpy().astype(int)
    coords = coords[valid_nodes]
    N_valid = len(valid_nodes)
    
    S_sparse = coo_matrix((np.concatenate([S_v, S_v]), 
                           (np.concatenate([i_mapped, j_mapped]), np.concatenate([j_mapped, i_mapped]))), 
                          shape=(N_valid, N_valid)).tocsr()
                          
    sparse_dist = coo_matrix((np.concatenate([d_v, d_v]), 
                              (np.concatenate([i_mapped, j_mapped]), np.concatenate([j_mapped, i_mapped]))), 
                             shape=(N_valid, N_valid)).tocsr()
    
    n_comp, labels = connected_components(sparse_dist, directed=False)
    counts = np.bincount(labels)
    
    min_size = N_total / min_rel_size
    valid_components = np.where(counts > min_size)[0]
    
    cent_img = np.zeros((H, W), dtype=np.float32)
    comp_mask = np.zeros((H, W), dtype=np.float32)
    tau_mask = np.zeros((H, W), dtype=np.float32)
    tau_mask[orig_coords_cpu[valid_nodes, 0], orig_coords_cpu[valid_nodes, 1]] = 1.0
    
    if len(valid_components) > 0:
        t_mst = 0
        t_bet_start = time.time()
        
        for comp_id in valid_components:
            mask_comp = (labels == comp_id)
            nodes_comp = np.where(mask_comp)[0]
            
            sparse_dist_comp = sparse_dist[nodes_comp, :][:, nodes_comp]
            S_sparse_comp = S_sparse[nodes_comp, :][:, nodes_comp]
            
            t_mst_start = time.time()
            mst = minimum_spanning_tree(sparse_dist_comp)
            order, preds = breadth_first_order(mst, i_start=0, directed=False, return_predecessors=True)
            t_mst += (time.time() - t_mst_start)
            
            N_L = len(nodes_comp)
            valid_mask_preds = preds >= 0
            p_valid = preds[valid_mask_preds]
            i_valid = np.arange(N_L)[valid_mask_preds]
            
            W_parent_np = np.zeros(N_L, dtype=np.float32)
            
            S_coo_comp = S_sparse_comp.tocoo()
            weights_dict = {(r, c): v for r, c, v in zip(S_coo_comp.row, S_coo_comp.col, S_coo_comp.data)}
            for idx, (p, i) in enumerate(zip(p_valid, i_valid)):
                W_parent_np[i] = weights_dict.get((p, i), 0.0)

            E_mass_np = np.zeros(N_L, dtype=np.float32)
            
            for i in order[::-1]:
                p = preds[i]
                if p >= 0:
                    E_mass_np[p] += E_mass_np[i] + W_parent_np[i]
                    
            M_total = E_mass_np[order[0]]
            
            W_parent = torch.tensor(W_parent_np, dtype=torch.float32, device=device)
            E_mass = torch.tensor(E_mass_np, dtype=torch.float32, device=device)
            
            child_branch_mass = W_parent + E_mass
            p_valid_t = torch.tensor(p_valid, dtype=torch.long, device=device)
            i_valid_t = torch.tensor(i_valid, dtype=torch.long, device=device)
            
            sum_masses_children = torch.zeros(N_L, dtype=torch.float32, device=device)
            sum_masses_children.index_add_(0, p_valid_t, child_branch_mass[i_valid_t])
            
            parent_branch_mass = torch.clamp(M_total - sum_masses_children, min=0.0)
            
            val_child = child_branch_mass * (M_total - child_branch_mass)
            sum_val_child = torch.zeros(N_L, dtype=torch.float32, device=device)
            sum_val_child.index_add_(0, p_valid_t, val_child[i_valid_t])
            
            val_parent = parent_branch_mass * (M_total - parent_branch_mass)
            centrality = (sum_val_child + val_parent) / 2.0
            
            if centrality.max() > 0:
                centrality /= centrality.max()
                
            coords_comp = coords[nodes_comp].cpu().numpy().astype(int)
            cent_img[coords_comp[:, 0], coords_comp[:, 1]] = centrality.cpu().numpy()
            comp_mask[coords_comp[:, 0], coords_comp[:, 1]] = 1.0
            
        if device == 'cuda': torch.cuda.synchronize()
        t_mst_total = t_mst
        t_bet_total = time.time() - t_bet_start - t_mst_total
    else:
        t_mst_total = 0
        t_bet_total = 0
    
    sim_img = np.zeros((H, W), dtype=np.float32)
    sim_img[orig_coords_cpu[:, 0], orig_coords_cpu[:, 1]] = node_sim_max
    
    if device == 'cuda': torch.cuda.synchronize()
    t_end = time.time()
    
    timings = {
        "1. Hessian Fusion": t_hessian - t0,
        "2. Graph Unfold": t_unfold - t_hessian,
        "3. Frangi Similarity": t_sim - t_unfold,
        "4. MST (CPU)": t_mst_total,
        "5. Betweenness (GPU)": t_bet_total,
        "Total": t_end - t0
    }
    
    return max_S_global.cpu().numpy(), sim_img, cent_img, timings, {'tau_mask': tau_mask, 'comp_mask': comp_mask}""")

add_md("""## 4. Visualisation sur un échantillon CrackForest""")

add_code("""import ot
from skimage.morphology import skeletonize, disk, dilation
import warnings

def skeletonize_lee(binary_mask: np.ndarray) -> np.ndarray:
    import cv2
    m = (binary_mask > 0).astype(np.uint8)
    
    # Lissage morphologique pour gommer les irrégularités de contour 
    # (cela élimine la grande majorité des petites branches parasites du squelette)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    
    sk = skeletonize(m>0)
    return sk.astype(np.uint8)

def thicken(skel: np.ndarray, pixels: int = 3) -> np.ndarray:
    if pixels <= 1: return skel.astype(np.uint8)
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels*2+1, pixels*2+1))
    thick = cv2.dilate((skel>0).astype(np.uint8), kernel)
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
    
    if na == 0 and nb == 0: return 0.0
    if na == 0: return float(nb)
    if nb == 0: return float(na)
    
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

# Prendre un échantillon
sample = dataset[10] # e.g. LAB00030 (ici CrackForest 011)
imgs = {'visible': sample['visible']}
weights = {'visible': 1.0}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
frangi_response, similarity_img, centrality, timings, diagnostics = extract_frangi_graph_gpu(imgs, weights, device=device)

skeleton = (centrality > 0.025).astype(np.float32)

gt_arr_sample = sample['gt'].numpy().astype(np.uint8)
sk_gt_sample = skeletonize_lee(gt_arr_sample)
sk_gt_thick_sample = thicken(sk_gt_sample, pixels=3)

pred_sample = skeleton.astype(np.uint8)
sk_pred_thick_sample = thicken(pred_sample, pixels=3)

j_sample, t_sample = compute_metrics(sk_pred_thick_sample, sk_gt_thick_sample)
w_sample = wasserstein_distance_skeletons(sk_pred_thick_sample, sk_gt_thick_sample)

print(f"--- Metrics for sample: {sample['id']} ---")
print(f"Jaccard (IoU): {j_sample:.4f}")
print(f"Tversky:       {t_sample:.4f}")
print(f"Wasserstein:   {w_sample:.4f}")

fig, axes = plt.subplots(2, 4, figsize=(32, 12))

axes[0, 0].imshow(sample['visible'].numpy(), cmap='gray')
axes[0, 0].set_title('Modalité : Visible')

axes[0, 1].imshow(frangi_response, cmap='magma')
axes[0, 1].set_title('Réponse Frangi Multi-échelles (Fused Λ2)')

axes[0, 2].imshow(similarity_img, cmap='magma')
axes[0, 2].set_title('Similarité Frangi-Graph (Max)')

axes[0, 3].imshow(sample['gt'].numpy(), cmap='gray')
axes[0, 3].set_title('Ground Truth (Brut)')

axes[1, 0].imshow(centrality, cmap='hot')
axes[1, 0].set_title('Betweenness Centrality (Graph GPU)')

axes[1, 1].imshow(np.zeros_like(skeleton), cmap='gray')
h, w = skeleton.shape
rgba_tau = np.zeros((h, w, 4), dtype=np.float32)
rgba_tau[diagnostics['tau_mask'] > 0] = [1.0, 1.0, 1.0, 0.3]
rgba_comp = np.zeros((h, w, 4), dtype=np.float32)
rgba_comp[diagnostics['comp_mask'] > 0] = [0.0, 0.5, 1.0, 0.8]
axes[1, 1].imshow(rgba_tau)
axes[1, 1].imshow(rgba_comp)
axes[1, 1].set_title('Filtrage: Noeuds (τ) & Composantes')

axes[1, 2].imshow(skeleton, cmap='gray')
axes[1, 2].set_title('Squelette Prédit (Brut)')

axes[1, 3].imshow(np.zeros_like(skeleton), cmap='gray')
rgba_gt_skel = np.zeros((h, w, 4), dtype=np.float32)
rgba_gt_skel[sk_gt_thick_sample > 0] = [0.0, 1.0, 0.0, 0.4] # Vert transparent

rgba_pred = np.zeros((h, w, 4), dtype=np.float32)
rgba_pred[sk_pred_thick_sample > 0] = [1.0, 0.0, 0.0, 0.4] # Rouge transparent

axes[1, 3].imshow(rgba_gt_skel)
axes[1, 3].imshow(rgba_pred)
axes[1, 3].set_title('Éval (Vert: GT Squelette, Rouge: Pred)')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()""")

add_md("""## 5. Analyse de sensibilité des paramètres

Nous allons faire varier paramètre par paramètre : `R, ss, si, sa, τ`, `τ_c` (le seuil de centralité), `min_rel_size` (taille relative minimale d'une composante) et `Σ` (réduit à `σ_0`).
Les autres paramètres resteront constants.

Nous chargeons d'abord le dataset en RAM pour une exécution ultra-rapide.""")

add_code("""import copy
from tqdm import tqdm
import pandas as pd

print("Chargement complet des 118 images en RAM pour accélérer l'analyse de sensibilité...")
all_data = []
for i in range(len(dataset)):
    all_data.append(dataset[i])
print("Terminé.")

default_params = {
    'R': 3,
    'ss': 1.0,
    'si': 0.25,
    'sa': 0.3,
    'τ': 0.2,
    'σ_0': 5.0,
    'τ_c': 0.025,
    'min_rel_size': 150.0
}

# nb_pas = 15
param_ranges = {
    'R': np.linspace(1, 15, 15, dtype=int).tolist(),
    'ss': np.linspace(0.1, 3.0, 15).tolist(),
    'si': np.linspace(0.05, 1.0, 15).tolist(),
    'sa': np.linspace(0.05, 1.0, 15).tolist(),
    'τ': np.linspace(0.01, 0.5, 15).tolist(),
    'τ_c': np.linspace(0.005, 0.1, 15).tolist(),
    'σ_0': np.linspace(1.0, 15.0, 15).tolist(),
    'min_rel_size': np.linspace(50.0, 500.0, 15).tolist()
}

os.makedirs("sensitivity_results", exist_ok=True)

def evaluate_dataset(params):
    import time
    j_list, t_list, w_list = [], [], []
    individual_results = []
    
    sigma_val = params['σ_0']
    
    for sample in tqdm(all_data, desc=f"Éval images (Σ={sigma_val:.1f}, R={params['R']}, ss={params['ss']:.2f}, si={params['si']:.2f}, sa={params['sa']:.2f}, τ={params['τ']:.2f}, τ_c={params['τ_c']:.3f}, min_rel_size={params['min_rel_size']:.1f})", leave=False):
        imgs_i = {'visible': sample['visible']}
        weights_i = {'visible': 1.0}
        
        _, _, centrality_i, _, _ = extract_frangi_graph_gpu(
            imgs_i, weights_i, 
            Σ=[sigma_val], 
            R=int(params['R']),
            ss=params['ss'], 
            si=params['si'], 
            sa=params['sa'], 
            τ=params['τ'],
            min_rel_size=params['min_rel_size'],
            device=device
        )
        
        pred_i = (centrality_i > params['τ_c']).astype(np.uint8)
        sk_pred_thick_i = thicken(pred_i, pixels=3)
        
        gt_arr_i = sample['gt'].numpy().astype(np.uint8)
        sk_gt_i = skeletonize_lee(gt_arr_i)
        sk_gt_thick_i = thicken(sk_gt_i, pixels=3)
        
        j, t = compute_metrics(sk_pred_thick_i, sk_gt_thick_i)
        w = wasserstein_distance_skeletons(sk_pred_thick_i, sk_gt_thick_i)
        
        j_list.append(j)
        t_list.append(t)
        w_list.append(w)
        individual_results.append({
            'ID': sample['id'], 
            'Jaccard': j, 
            'Tversky': t, 
            'Wasserstein': w
        })
        
    return np.mean(j_list), np.std(j_list), np.mean(t_list), np.std(t_list), np.mean(w_list), np.std(w_list), individual_results

import time
for param_name, values in param_ranges.items():
    print(f"\\n{'='*60}\\n--- Analyse de sensibilité pour {param_name} ---\\n{'='*60}")
    results_summary = []
    all_individual_results = []
    
    t0_param = time.time()
    
    for val in values:
        t0_val = time.time()
        
        current_params = copy.deepcopy(default_params)
        current_params[param_name] = val
        
        j_mean, j_std, t_mean, t_std, w_mean, w_std, ind_res = evaluate_dataset(current_params)
        
        t_val = time.time() - t0_val
        print(f"[{param_name} = {val}] évalué en {t_val:.2f}s : Jaccard={j_mean:.4f}±{j_std:.4f} | Tversky={t_mean:.4f}±{t_std:.4f} | Wasserstein={w_mean:.4f}±{w_std:.4f}")
        
        results_summary.append({
            param_name: val,
            'Jaccard_mean': j_mean, 'Jaccard_std': j_std,
            'Tversky_mean': t_mean, 'Tversky_std': t_std,
            'Wasserstein_mean': w_mean, 'Wasserstein_std': w_std
        })
        
        for res in ind_res:
            res_copy = res.copy()
            res_copy[param_name] = val
            all_individual_results.append(res_copy)
            
    t_param = time.time() - t0_param
    print(f"--- Fin de l'analyse pour {param_name} (Temps total : {t_param:.2f}s) ---")
            
    # Sauvegarde CSV
    df_summary = pd.DataFrame(results_summary)
    df_summary.to_csv(f"sensitivity_results/summary_{param_name}.csv", index=False)
    
    df_ind = pd.DataFrame(all_individual_results)
    df_ind.to_csv(f"sensitivity_results/individual_{param_name}.csv", index=False)
    
    # Affichage des 3 jolies courbes
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Sensibilité : {param_name}", fontsize=16)
    
    x_vals = df_summary[param_name]
    
    # Jaccard
    axes[0].plot(x_vals, df_summary['Jaccard_mean'], 'b-', label='Jaccard')
    axes[0].fill_between(x_vals, 
                         df_summary['Jaccard_mean'] - df_summary['Jaccard_std'],
                         df_summary['Jaccard_mean'] + df_summary['Jaccard_std'], 
                         color='b', alpha=0.2)
    axes[0].set_title('Jaccard (IoU)')
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel('Score')
    axes[0].grid(True)
    
    # Tversky
    axes[1].plot(x_vals, df_summary['Tversky_mean'], 'g-', label='Tversky')
    axes[1].fill_between(x_vals, 
                         df_summary['Tversky_mean'] - df_summary['Tversky_std'],
                         df_summary['Tversky_mean'] + df_summary['Tversky_std'], 
                         color='g', alpha=0.2)
    axes[1].set_title('Tversky')
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel('Score')
    axes[1].grid(True)
    
    # Wasserstein
    axes[2].plot(x_vals, df_summary['Wasserstein_mean'], 'r-', label='Wasserstein')
    axes[2].fill_between(x_vals, 
                         df_summary['Wasserstein_mean'] - df_summary['Wasserstein_std'],
                         df_summary['Wasserstein_mean'] + df_summary['Wasserstein_std'], 
                         color='r', alpha=0.2)
    axes[2].set_title('Wasserstein')
    axes[2].set_xlabel(param_name)
    axes[2].set_ylabel('Distance')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"sensitivity_results/plot_{param_name}.png")
    plt.show()""")

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "A100",
            "name": "Frangi_CrackForest_GPU.ipynb"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("Frangi_CrackForest_GPU.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook CrackForest généré avec succès.")