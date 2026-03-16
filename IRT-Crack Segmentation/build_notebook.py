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
## Benchmark : IRT-Crack Segmentation

Ce Colab implémente l'approche non supervisée décrite dans l'article EUSIPCO **"Multi-Modal, Training-Free Crack Extraction via Generalized Frangi Graph"**, en exploitant la puissance du **GPU (PyTorch)** pour accélérer le filtrage Hessien multi-échelles et la construction du graphe de similarité.

### Caractéristiques de l'implémentation :
- Chargement robuste des données (indexation creuse, asymétrie PNG/JPG).
- Calculs matriciels Hessiens et Valeurs Propres 100% sur GPU (`torch.Tensor`).
- Construction du graphe creux (Sparse) économe en VRAM via les K-Nearest Neighbors (`scipy.spatial.cKDTree`).
- Algorithme d'extraction topologique (Arbre Couvrant de Poids Minimum + Centralité).""")

add_code("""!pip install -q gdown
import os
import zipfile
import gdown

use_zip = True # @param {type:"boolean"}

# Option 1 (Recommandée) : Téléchargement du fichier ZIP (plus rapide)
zip_file_id = '1HhVmtQwB56VMuIBcAjvv-J-BN3o9m2vL'
zip_path = 'IRT-Crack-Dataset.zip'

# Option 2 : Téléchargement dossier par dossier (plus lent)
folder_id = '18yq9IFOSOvO7O95NVtZ3hpG9_KDdpJcO'
dest_dir = 'IRT-Crack-Dataset'

def check_dataset_exists():
    from pathlib import Path
    for path in Path('.').rglob('01-Visible Image'):
        return True
    return False

if not check_dataset_exists():
    if use_zip:
        print("Téléchargement du dataset (ZIP) depuis Google Drive...")
        gdown.download(id=zip_file_id, output=zip_path, quiet=False)
        print("Extraction du ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Extraction terminée.")
        # Nettoyage du zip pour libérer de l'espace
        if os.path.exists(zip_path):
            os.remove(zip_path)
    else:
        print("Téléchargement du dataset (Dossier) depuis Google Drive...")
        gdown.download_folder(id=folder_id, output=dest_dir, quiet=False, use_cookies=False)
        print("Téléchargement terminé.")
else:
    print("Dataset déjà présent.")""")

add_md("""## 1. Dataloader Rigoureux

Nous respectons scrupuleusement la topologie du dataset avec une classe PyTorch personnalisée :
- **Asymétrie des extensions** : `.png` pour les entrées, `.jpg` pour le Ground Truth.
- **Indexation creuse** : Parcours via `pathlib` pour ignorer les trous dans la numérotation.
- **Binarisation stricte** du Ground Truth pour compenser les artefacts JPEG.""")

add_code("""import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class IRTCrackDataset(Dataset):
    def __init__(self, root_dir):
        # La commande gdown peut avoir créé un sous-dossier avec le nom original du drive
        # On recherche le dossier contenant '01-Visible Image'
        self.root_dir = None
        for path in Path(root_dir).rglob('01-Visible Image'):
            self.root_dir = path.parent
            break
            
        if self.root_dir is None:
            raise FileNotFoundError("Structure du dataset non trouvée.")
            
        self.vis_dir = self.root_dir / '01-Visible Image'
        self.ir_dir  = self.root_dir / '02-Infrared Image'
        self.fus_dir = self.root_dir / '03-Fusion Image'
        self.gt_dir  = self.root_dir / '04-Ground Truth'
        
        # Étape de cartographie (Sparse index)
        vis_paths = list(self.vis_dir.glob('*.png'))
        self.identifiants = sorted([p.stem for p in vis_paths])
        print(f"Dataset chargé avec {len(self.identifiants)} images.")

    def __len__(self):
        return len(self.identifiants)

    def __getitem__(self, idx):
        id_courant = self.identifiants[idx]
        
        # Résolution des chemins avec asymétrie d'extension
        path_vis = self.vis_dir / f"{id_courant}.png"
        path_ir  = self.ir_dir / f"{id_courant}.png"
        path_fus = self.fus_dir / f"{id_courant}.png"
        path_gt  = self.gt_dir / f"{id_courant}.jpg"
        
        # Chargement en N&B
        img_vis = cv2.imread(str(path_vis), cv2.IMREAD_GRAYSCALE)
        img_ir  = cv2.imread(str(path_ir), cv2.IMREAD_GRAYSCALE)
        img_fus = cv2.imread(str(path_fus), cv2.IMREAD_GRAYSCALE)
        img_gt  = cv2.imread(str(path_gt), cv2.IMREAD_GRAYSCALE)
        
        # Normalisation
        vis_t = torch.from_numpy(img_vis).float() / 255.0
        ir_t  = torch.from_numpy(img_ir).float() / 255.0
        fus_t = torch.from_numpy(img_fus).float() / 255.0
        
        # Binarisation stricte du Ground Truth (compensation artefacts JPEG)
        gt_clean = (img_gt > 127).astype(np.float32)
        gt_t = torch.from_numpy(gt_clean)
        
        return {
            'id': id_courant,
            'visible': vis_t,
            'infrared': ir_t,
            'fusion': fus_t,
            'gt': gt_t
        }

# Initialisation
# Comme l'extraction ZIP se fait dans '.', on lance la recherche depuis le répertoire courant
dataset = IRTCrackDataset('.')""")

add_md("""## 2. Calcul Hessien Multi-échelles sur GPU

Nous implémentons le filtrage Hessien via des convolutions PyTorch (`F.conv2d`) et dérivons les valeurs propres (Λ1, Λ2) et vecteurs propres (Θ).""")

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
        # Ajout des dimensions batch et canal : (1, 1, H, W)
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
        
        # Pour les structures sombres (dark ridges), on cherche la courbure max positive
        λ2 = torch.where(mask_minus_bigger, l_minus, l_plus)
        λ1 = torch.where(mask_minus_bigger, l_plus, l_minus)

        θ = 0.5 * torch.atan2(2 * ixy, ixx - iyy)
        return λ1, λ2, θ""")

add_md("""## 3. Fusion Multimodale et Construction du Graphe (Frangi Graph)

Cette fonction exécute la pipeline optimisée :
1. **Fusion au niveau Hessien** en sommant pondérément les modalités (Intensité, Range/IR) normalisées.
2. **Réponse Frangi Multi-échelles** pour isoler les "Dark Ridges".
3. **Voisinage via Unfold (GPU)** pour calculer les similarités instantanément sans boucles.
4. **Topologie Hybride (MST CPU + Centralité GPU)** : L'extraction de l'arbre se fait avec SciPy sur CPU, mais le calcul de *Weighted Betweenness Centrality* est transféré et vectorisé sur le GPU PyTorch avec `index_add_`.""")

add_code("""from scipy.sparse import coo_matrix

def extract_frangi_graph_gpu(imgs_dict, weights, Σ=[1, 3, 5, 7], R=10, 
                             ss=2.0, si=0.25, sa=0.125, τ=0.05, device='cuda'):
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
                
                # Normalisation stricte par la norme spectrale maximale (Eq. 1 du papier Eusipco)
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
        
        # Filtre de Frangi (hypothèse: "Dark Ridges" -> l2 > 0)
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
        
    # 2. Étape 1a : Premier seuillage (léger) sur la réponse brute λ2
    τ_1 = max_S_global.max() * 0.01 
    candidates_mask = max_S_global > τ_1
    coords = torch.nonzero(candidates_mask).float() # (N, 2)
    N = coords.shape[0]
    
    if N == 0:
        return np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    
    # 3. Graphe creux (Sparse) généré EXCLUSIVEMENT sur GPU via unfold (Piste 1: Ultra-rapide)
    index_map = torch.full((H, W), -1, dtype=torch.long, device=device)
    index_map[candidates_mask] = torch.arange(N, device=device)
    
    # Padding pour l'extraction des patchs (rayon R)
    padded_index_map = torch.nn.functional.pad(index_map.unsqueeze(0).unsqueeze(0).float(), (R, R, R, R), value=-1).long()
    
    # Fenêtrage glissant pour obtenir tous les voisinages en un coup
    patches = padded_index_map[0, 0].unfold(0, 2*R+1, 1).unfold(1, 2*R+1, 1) # (H, W, 2R+1, 2R+1)
    
    # Récupération des voisinages uniquement pour nos candidats
    y_coords, x_coords = torch.nonzero(candidates_mask, as_tuple=True)
    cand_patches = patches[y_coords, x_coords] # (N, 2R+1, 2R+1)
    
    # Création du masque géométrique valide (demi-cercle pour éviter les arêtes en double)
    dy, dx = torch.meshgrid(torch.arange(-R, R+1, device=device), torch.arange(-R, R+1, device=device), indexing='ij')
    dist_sq = dx**2 + dy**2
    valid_dist_mask = (dist_sq <= R**2) & (dist_sq > 0)
    half_mask = (dy > 0) | ((dy == 0) & (dx > 0))
    valid_mask = valid_dist_mask & half_mask # (2R+1, 2R+1)
    
    # Filtrage des voisins valides
    valid_neighbors = cand_patches[:, valid_mask] # (N, M)
    
    # Création des arêtes
    source_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, valid_neighbors.shape[1])
    valid_pairs_mask = valid_neighbors != -1
    
    i_idx_t = source_idx[valid_pairs_mask]
    j_idx_t = valid_neighbors[valid_pairs_mask]
    
    if len(i_idx_t) == 0:
        return max_S_global.cpu().numpy(), np.zeros((H, W)), np.zeros((H, W)), {}
    
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
        
    # 4. Extraction Topologique (MST CPU + Centralité GPU PyTorch)
    # Transformation en distance (Eq. 6) vectorisée sur GPU
    dist_ij_t = torch.norm(coords[i_idx_t] - coords[j_idx_t], dim=1)
    d_ij = (1 - S_ij_max) * dist_ij_t + 1e-8
    
    S_cpu = S_ij_max.cpu().numpy()
    d_cpu = d_ij.cpu().numpy()
    i_cpu = i_idx_t.cpu().numpy()
    j_cpu = j_idx_t.cpu().numpy()
    
    import numpy as np
    from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order, connected_components
    
    # 1re Optimisation: Calcul ultra-rapide des similarités max par noeud avec NumPy 
    node_sim_max = np.zeros(N, dtype=np.float32)
    np.maximum.at(node_sim_max, i_cpu, S_cpu)
    np.maximum.at(node_sim_max, j_cpu, S_cpu)
        
    # Seuillage strict des top tau % (ex: 5%)
    num_to_keep = max(1, int(N * τ))
    if N > num_to_keep:
        threshold_sim = np.partition(node_sim_max, -num_to_keep)[-num_to_keep]
        valid_nodes = np.where(node_sim_max >= threshold_sim)[0]
    else:
        valid_nodes = np.arange(N)
        
    # Filtrage des arêtes en amont (évite le découpage d'une gigantesque matrice SciPy)
    valid_mask = np.zeros(N, dtype=bool)
    valid_mask[valid_nodes] = True
    edge_mask = valid_mask[i_cpu] & valid_mask[j_cpu]
    
    i_v = i_cpu[edge_mask]
    j_v = j_cpu[edge_mask]
    S_v = S_cpu[edge_mask]
    d_v = d_cpu[edge_mask]
    
    # Remapping des index (0 à N_valid - 1)
    remap = np.full(N, -1, dtype=np.int32)
    remap[valid_nodes] = np.arange(len(valid_nodes))
    
    i_mapped = remap[i_v]
    j_mapped = remap[j_v]
    
    orig_coords_cpu = coords.cpu().numpy().astype(int)
    coords = coords[valid_nodes]
    N_valid = len(valid_nodes)
    
    from scipy.sparse import coo_matrix
    S_sparse = coo_matrix((np.concatenate([S_v, S_v]), 
                           (np.concatenate([i_mapped, j_mapped]), np.concatenate([j_mapped, i_mapped]))), 
                          shape=(N_valid, N_valid)).tocsr()
                          
    sparse_dist = coo_matrix((np.concatenate([d_v, d_v]), 
                              (np.concatenate([i_mapped, j_mapped]), np.concatenate([j_mapped, i_mapped]))), 
                             shape=(N_valid, N_valid)).tocsr()
    
    # Isolation de la plus grande composante connexe
    n_comp, labels = connected_components(sparse_dist, directed=False)
    if n_comp > 1:
        counts = np.bincount(labels)
        largest_comp = np.argmax(counts)
        mask_largest = (labels == largest_comp)
        nodes_largest = np.where(mask_largest)[0]
    else:
        nodes_largest = np.arange(N_valid)
        
    if len(nodes_largest) == 0:
        return max_S_global.cpu().numpy(), np.zeros((H, W)), np.zeros((H, W)), {}
        
    sparse_dist_largest = sparse_dist[nodes_largest, :][:, nodes_largest]
    S_sparse_largest = S_sparse[nodes_largest, :][:, nodes_largest]
    
    # Minimum Spanning Tree (SciPy, très optimisé en C, sur CPU)
    mst = minimum_spanning_tree(sparse_dist_largest)
    order, preds = breadth_first_order(mst, i_start=0, directed=False, return_predecessors=True)
    
    if device == 'cuda': torch.cuda.synchronize()
    t_mst = time.time()
    
    # Weighted Betweenness Centrality implémentée sur GPU via PyTorch
    N_L = len(nodes_largest)
    valid_mask = preds >= 0
    p_valid = preds[valid_mask]
    i_valid = np.arange(N_L)[valid_mask]
    
    W_parent_np = np.zeros(N_L, dtype=np.float32)
    
    # Extraction robuste des poids : on utilise COO pour chercher les correspondances, ou une simple boucle/list comprehension
    # car l'indexation de tableaux dans scipy.sparse CSR peut retourner un produit extérieur (NxN) selon la version.
    S_coo_largest = S_sparse_largest.tocoo()
    # On convertit en dictionnaire pour un accès O(1)
    # Les paires (p, i) sont les arêtes de l'arbre
    import scipy.sparse as sp
    # Le plus performant:
    weights_dict = {(r, c): v for r, c, v in zip(S_coo_largest.row, S_coo_largest.col, S_coo_largest.data)}
    for idx, (p, i) in enumerate(zip(p_valid, i_valid)):
        W_parent_np[i] = weights_dict.get((p, i), 0.0)


    
    # 2e Optimisation : Accumulation sur CPU pur (NumPy très rapide pour les boucles)
    E_mass_np = np.zeros(N_L, dtype=np.float32)
    
    # Accumulation depuis les feuilles
    for i in order[::-1]:
        p = preds[i]
        if p >= 0:
            E_mass_np[p] += E_mass_np[i] + W_parent_np[i]
            
    M_total = E_mass_np[order[0]]
    
    # Transfert vers le GPU (une seule fois) pour les calculs tensoriels finaux
    W_parent = torch.tensor(W_parent_np, dtype=torch.float32, device=device)
    E_mass = torch.tensor(E_mass_np, dtype=torch.float32, device=device)
    
    # Vectorisation des opérations de Betweenness (sur GPU)
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
        
    # Re-projection sur l'image
    cent_img = np.zeros((H, W), dtype=np.float32)
    coords_largest = coords[nodes_largest].cpu().numpy().astype(int)
    cent_img[coords_largest[:, 0], coords_largest[:, 1]] = centrality.cpu().numpy()
    
    # Projection de la similarité max par noeud sur l'image
    sim_img = np.zeros((H, W), dtype=np.float32)
    sim_img[orig_coords_cpu[:, 0], orig_coords_cpu[:, 1]] = node_sim_max
    
    if device == 'cuda': torch.cuda.synchronize()
    t_end = time.time()
    
    timings = {
        "1. Hessian Fusion": t_hessian - t0,
        "2. Graph Unfold": t_unfold - t_hessian,
        "3. Frangi Similarity": t_sim - t_unfold,
        "4. MST (CPU)": t_mst - t_sim,
        "5. Betweenness (GPU)": t_end - t_mst,
        "Total": t_end - t0
    }
    
    return max_S_global.cpu().numpy(), sim_img, cent_img, timings""")

add_md("""## 4. Visualisation Complète (Inspection Visuelle)

Nous allons illustrer le processus complet sur un échantillon pour observer l'apport de la fusion et le rôle de la centralité.""")

add_code("""!pip install -q POT
import ot
from skimage.morphology import skeletonize, disk, dilation
import warnings

def skeletonize_lee(binary_mask: np.ndarray) -> np.ndarray:
    m = (binary_mask > 0).astype(np.uint8)
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
    A = torch.from_numpy(pred_mask).bool().to(device)
    B = torch.from_numpy(gt_mask).bool().to(device)
    
    inter = torch.logical_and(A, B).sum().float()
    union = torch.logical_or(A, B).sum().float()
    jaccard = (inter / (union + 1e-9)).item()
    
    fp = torch.logical_and(torch.logical_not(A), B).sum().float()
    fn = torch.logical_and(A, torch.logical_not(B)).sum().float()
    tversky = (inter / (inter + 1.0 * fn + 0.5 * fp + 1e-9)).item()
    
    return jaccard, tversky

def wasserstein_distance_skeletons(A: np.ndarray, B: np.ndarray, max_samples: int = 2000) -> float:
    Ay, Ax = np.nonzero(A>0); By, Bx = np.nonzero(B>0)
    if len(Ay)==0 and len(By)==0: return 0.0
    if len(Ay)==0: return float(len(By))
    if len(By)==0: return float(len(Ay))
    A_pts = np.column_stack([Ay,Ax]).astype(np.float32)
    B_pts = np.column_stack([By,Bx]).astype(np.float32)
    if A_pts.shape[0] > max_samples:
        idx = np.random.choice(A_pts.shape[0], size=max_samples, replace=False); A_pts = A_pts[idx]
    if B_pts.shape[0] > max_samples:
        idx = np.random.choice(B_pts.shape[0], size=max_samples, replace=False); B_pts = B_pts[idx]
    na, nb = A_pts.shape[0], B_pts.shape[0]
    a = np.ones((na,), dtype=np.float64) / float(na)
    b = np.ones((nb,), dtype=np.float64) / float(nb)
    
    # Calcul de la matrice de coût sur le GPU (Accélération fulgurante de cdist)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A_t = torch.from_numpy(A_pts).to(device)
    B_t = torch.from_numpy(B_pts).to(device)
    M_t = torch.cdist(A_t, B_t, p=2.0)
    M = M_t.cpu().numpy().astype(np.float64)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emd_cost = ot.emd2(a,b,M)
    return float(emd_cost)

# Prendre un échantillon
sample = dataset[10] # e.g. LAB00030
imgs = {
    'visible': sample['visible'],
    'infrared': sample['infrared']
}

# Fusion : 50% Visible, 50% Infrarouge
weights = {'visible': 0.5, 'infrared': 0.5}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
frangi_response, similarity_img, centrality, timings = extract_frangi_graph_gpu(imgs, weights, device=device)

# Seuillage final adaptatif pour extraire le squelette
# On garde les chemins majeurs (centralité élevée)
skeleton = (centrality > 0.01).astype(np.float32)

# --- Metrics and Thickening for the single sample example ---
gt_arr_sample = sample['gt'].numpy().astype(np.uint8)
sk_gt_sample = skeletonize_lee(gt_arr_sample)
sk_gt_thick_sample = thicken(sk_gt_sample, pixels=3)

pred_sample = skeleton.astype(np.uint8)
sk_pred_thick_sample = thicken(pred_sample, pixels=3)

j_sample, t_sample = compute_metrics(sk_pred_thick_sample, sk_gt_thick_sample)
w_sample = wasserstein_distance_skeletons(sk_pred_thick_sample, sk_gt_thick_sample)

print("--- Metrics for sample ---")
print(f"Jaccard (IoU): {j_sample:.4f}")
print(f"Tversky:       {t_sample:.4f}")
print(f"Wasserstein:   {w_sample:.4f}")
print("--- Timings ---")
for k, v in timings.items():
    print(f"{k}: {v*1000:.2f} ms")
print("--------------------------")

fig, axes = plt.subplots(2, 4, figsize=(24, 12))

axes[0, 0].imshow(sample['visible'].numpy(), cmap='gray')
axes[0, 0].set_title('Modalité : Visible')

axes[0, 1].imshow(sample['infrared'].numpy(), cmap='gray')
axes[0, 1].set_title('Modalité : Infrarouge (IR)')

axes[0, 2].imshow(frangi_response, cmap='magma')
axes[0, 2].set_title('Réponse Frangi Multi-échelles (Fused Λ2)')

axes[0, 3].imshow(sample['gt'].numpy(), cmap='gray')
axes[0, 3].set_title('Ground Truth (Binarisé)')

axes[1, 0].imshow(similarity_img, cmap='magma')
axes[1, 0].set_title('Similarité Frangi-Graph (Max)')

axes[1, 1].imshow(centrality, cmap='hot')
axes[1, 1].set_title('Betweenness Centrality (Graph GPU)')

# Squelette Brut
axes[1, 2].imshow(skeleton, cmap='gray')
axes[1, 2].set_title('Squelette Prédit (Brut)')

# Superposition Métriques : GT brute (Blanc) + GT Squelette grossi (Vert) + Pred grossi (Rouge)
# On met un fond noir pour bien faire ressortir les couleurs
axes[1, 3].imshow(np.zeros_like(skeleton), cmap='gray')

# Création de masques RGBA pour maîtriser parfaitement la couleur et la transparence
h, w = skeleton.shape
rgba_gt = np.zeros((h, w, 4), dtype=np.float32)
rgba_gt[sample['gt'].numpy() > 0] = [1.0, 1.0, 1.0, 0.3] # Blanc transparent

rgba_gt_skel = np.zeros((h, w, 4), dtype=np.float32)
rgba_gt_skel[sk_gt_thick_sample > 0] = [0.0, 1.0, 0.0, 0.4] # Vert transparent

rgba_pred = np.zeros((h, w, 4), dtype=np.float32)
rgba_pred[sk_pred_thick_sample > 0] = [1.0, 0.0, 0.0, 0.4] # Rouge transparent

axes[1, 3].imshow(rgba_gt)
axes[1, 3].imshow(rgba_gt_skel)
axes[1, 3].imshow(rgba_pred)
axes[1, 3].set_title('Éval (Blanc: GT, Vert: GT Squelette, Rouge: Pred)')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()""")

add_md("""## 5. Évaluation des Métriques (Jaccard / Tversky)

Calcul des métriques sur un batch d'images pour valider l'approche sur le benchmark.""")

add_code("""# --- Batch Evaluation ---
# Évaluation sur 20 images pour démonstration rapide
num_eval = min(20, len(dataset))
jaccard_scores = []
tversky_scores = []
wasserstein_scores = []

print(f"Évaluation sur {num_eval} images...")
for i in range(num_eval):
    sample_i = dataset[i]
    imgs_i = {'visible': sample_i['visible'], 'infrared': sample_i['infrared']}
    
    _, _, centrality_i, _ = extract_frangi_graph_gpu(imgs_i, weights, device=device)
    
    pred_i = (centrality_i > 0.01).astype(np.uint8)
    sk_pred_thick_i = thicken(pred_i, pixels=3)
    
    gt_arr_i = sample_i['gt'].numpy().astype(np.uint8)
    sk_gt_i = skeletonize_lee(gt_arr_i)
    sk_gt_thick_i = thicken(sk_gt_i, pixels=3)
    
    j, t = compute_metrics(sk_pred_thick_i, sk_gt_thick_i)
    w = wasserstein_distance_skeletons(sk_pred_thick_i, sk_gt_thick_i)
    
    jaccard_scores.append(j)
    tversky_scores.append(t)
    wasserstein_scores.append(w)

print(f"Jaccard (IoU) Moyen : {np.mean(jaccard_scores):.4f} ± {np.std(jaccard_scores):.4f}")
print(f"Tversky Moyen       : {np.mean(tversky_scores):.4f} ± {np.std(tversky_scores):.4f}")
print(f"Wasserstein Moyen   : {np.mean(wasserstein_scores):.4f} ± {np.std(wasserstein_scores):.4f}")""")

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "A100",
            "name": "Frangi_IRT_Crack_GPU.ipynb"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("IRT-Crack Segmentation/Frangi_IRT_Crack_GPU.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook generated successfully.")