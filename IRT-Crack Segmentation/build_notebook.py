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
        l2 = torch.where(mask_minus_bigger, l_minus, l_plus)
        l1 = torch.where(mask_minus_bigger, l_plus, l_minus)

        theta = 0.5 * torch.atan2(2 * ixy, ixx - iyy)
        return l1, l2, theta""")

add_md("""## 3. Fusion Multimodale et Construction du Graphe (Frangi Graph)

Cette fonction exécute la pipeline décrite dans l'article :
1. **Fusion au niveau Hessien** en sommant pondérément les modalités (Intensité, Range/IR) normalisées.
2. **Réponse Frangi Multi-échelles** pour isoler les "Dark Ridges".
3. **Métrique de similarité Frangi** calculée sur un graphe creux (Sparse) généré par `scipy.spatial.cKDTree` pour éviter l'explosion mémoire en O(N²).
4. **Extraction Topologique (MST + Centralité)** calculée efficacement via SciPy.""")

add_code("""from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order, connected_components

def extract_frangi_graph_gpu(imgs_dict, weights, scales=[1, 3, 5, 7], R=10, 
                             ss=2.0, si=0.25, sa=0.125, device='cuda'):
    
    fh = FrangiHessianGPU(scales, device=device)
    
    scale_data = []
    max_S_global = None
    H, W = None, None
    
    # 1. Fusion Multimodale par échelle
    for s in scales:
        fused_ixx = None
        
        for mod, w in weights.items():
            if w > 0:
                ixx, ixy, iyy = fh.compute_hessian(imgs_dict[mod], s)
                if H is None:
                    H, W = ixx.shape
                
                # Normalisation stricte par la norme spectrale maximale (Eq. 1 du papier Eusipco)
                # Max(|lambda1|, |lambda2|) = (|trace| + disc) / 2
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
                    
        l1, l2, theta = fh.compute_eigenvalues_and_vectors(fused_ixx, fused_ixy, fused_iyy)
        
        # Filtre de Frangi (hypothèse: "Dark Ridges" -> l2 > 0)
        mask_pos = l2 > 0
        RB = torch.zeros_like(l2)
        RB[mask_pos] = torch.abs(l1[mask_pos]) / (l2[mask_pos] + 1e-8)
        
        S_norm = torch.zeros_like(l2)
        S_norm[mask_pos] = l2[mask_pos]
        
        if max_S_global is None: max_S_global = S_norm.clone()
        else: max_S_global = torch.max(max_S_global, S_norm)
            
        scale_data.append((RB, S_norm, theta, mask_pos))
        
    # 2. Seuillage pour définir les nœuds du graphe
    thresh = max_S_global.max() * 0.1 # (Étape 1: Dual Thresholding sera raffinée plus tard)
    candidates_mask = max_S_global > thresh
    coords = torch.nonzero(candidates_mask).float() # (N, 2)
    N = coords.shape[0]
    
    if N == 0:
        return np.zeros((H, W)), np.zeros((H, W))
    
    # 3. Graphe creux (Sparse) généré EXCLUSIVEMENT sur GPU via convolutions/décalages (Optimisation)
    # Permet de se passer du kdtree sur CPU et d'éviter les transferts de mémoire lents.
    index_map = torch.full((H, W), -1, dtype=torch.long, device=device)
    index_map[candidates_mask] = torch.arange(N, device=device)
    
    i_indices = []
    j_indices = []
    dists = []
    
    for dy in range(0, R+1):
        for dx in range(-R, R+1):
            if dy == 0 and dx <= 0:
                continue
            d_sq = dx**2 + dy**2
            if d_sq <= R**2:
                y_start, y_end = max(0, -dy), min(H, H-dy)
                x_start, x_end = max(0, -dx), min(W, W-dx)
                
                overlap = candidates_mask[y_start:y_end, x_start:x_end] & candidates_mask[y_start+dy:y_end+dy, x_start+dx:x_end+dx]
                
                if not overlap.any():
                    continue
                    
                y1_ov, x1_ov = torch.nonzero(overlap, as_tuple=True)
                y1 = y1_ov + y_start
                x1 = x1_ov + x_start
                y2 = y1 + dy
                x2 = x1 + dx
                
                idx1 = index_map[y1, x1]
                idx2 = index_map[y2, x2]
                
                i_indices.append(idx1)
                j_indices.append(idx2)
                dists.append(torch.full((len(idx1),), math.sqrt(d_sq), device=device))
    
    if len(i_indices) == 0:
        return max_S_global.cpu().numpy(), np.zeros((H, W))
    
    i_idx_t = torch.cat(i_indices)
    j_idx_t = torch.cat(j_indices)
    dist_ij_t = torch.cat(dists)
    
    S_ij_max = torch.zeros(len(i_idx_t), device=device, dtype=torch.float32)
    
    for RB, S_norm, theta, mask_pos in scale_data:
        rb_c = RB[candidates_mask]
        s_c  = S_norm[candidates_mask]
        t_c  = theta[candidates_mask]
        
        rb_sum = rb_c[i_idx_t] + rb_c[j_idx_t]
        S_shape = torch.exp(-0.5 * (rb_sum / ss)**2)
        
        s_prod = s_c[i_idx_t] * s_c[j_idx_t]
        S_int = 1 - torch.exp(-0.5 * (s_prod / si)**2)
        
        dt = t_c[i_idx_t] - t_c[j_idx_t]
        S_align = torch.exp(-0.5 * (torch.sin(dt) / sa)**2)
        
        S_ij = S_shape * S_int * S_align
        S_ij_max = torch.max(S_ij_max, S_ij)
        
    # Transformation en distance pour le MST (Eq. 6 Eusipco)
    d_ij = (1 - S_ij_max) * dist_ij_t + 1e-8 # epsilon pour garantir la positivité stricte
    
    d_ij_cpu = d_ij.cpu().numpy()
    i_idx_cpu = i_idx_t.cpu().numpy()
    j_idx_cpu = j_idx_t.cpu().numpy()
    S_cpu = S_ij_max.cpu().numpy()
    
    sparse_dist = coo_matrix((d_ij_cpu, (i_idx_cpu, j_idx_cpu)), shape=(N, N)).tocsr()
    sparse_dist = sparse_dist + sparse_dist.T
    
    S_sparse = coo_matrix((S_cpu, (i_idx_cpu, j_idx_cpu)), shape=(N, N)).tocsr()
    S_sparse = S_sparse + S_sparse.T
    
    # 4. Extraction Topologique rigoureuse selon l'article
    # a. Isolation de la plus grande composante connexe
    n_comp, labels = connected_components(sparse_dist, directed=False)
    if n_comp > 1:
        counts = np.bincount(labels)
        largest_comp = np.argmax(counts)
        mask_largest = (labels == largest_comp)
        nodes_largest = np.where(mask_largest)[0]
    else:
        nodes_largest = np.arange(N)
        
    if len(nodes_largest) == 0:
        return max_S_global.cpu().numpy(), np.zeros((H, W))
        
    sparse_dist_largest = sparse_dist[nodes_largest, :][:, nodes_largest]
    S_sparse_largest = S_sparse[nodes_largest, :][:, nodes_largest]
    
    # b. Minimum Spanning Tree
    mst = minimum_spanning_tree(sparse_dist_largest)
    
    # c. Weighted Betweenness Centrality exacte (Eq. 7 Eusipco)
    order, preds = breadth_first_order(mst, i_start=0, directed=False, return_predecessors=True)
    
    N_L = len(nodes_largest)
    E_mass = np.zeros(N_L)
    W_parent = np.zeros(N_L)
    children = {i: [] for i in range(N_L)}
    
    for i in range(N_L):
        p = preds[i]
        if p >= 0:
            children[p].append(i)
            
    # Accumulation des masses (somme des similarités S_ij) depuis les feuilles vers la racine
    for i in order[::-1]:
        p = preds[i]
        if p >= 0:
            w = S_sparse_largest[p, i]
            W_parent[i] = w
            E_mass[p] += E_mass[i] + w
            
    M_total = E_mass[order[0]]
    centrality = np.zeros(N_L)
    
    # Calcul de la centralité pour chaque noeud
    for i in range(N_L):
        masses = []
        sum_masses = 0
        for c in children[i]:
            m_c = W_parent[c] + E_mass[c]
            masses.append(m_c)
            sum_masses += m_c
            
        m_p = M_total - sum_masses
        if m_p > 0:
            masses.append(m_p)
            
        c_val = 0.0
        for m in masses:
            c_val += m * (M_total - m)
        centrality[i] = c_val / 2.0
        
    if centrality.max() > 0:
        centrality /= centrality.max()
        
    # Re-projection sur l'image
    cent_img = np.zeros((H, W), dtype=np.float32)
    coords_largest = coords[nodes_largest].cpu().numpy().astype(int)
    cent_img[coords_largest[:, 0], coords_largest[:, 1]] = centrality
    
    return max_S_global.cpu().numpy(), cent_img""")

add_md("""## 4. Visualisation Complète (Inspection Visuelle)

Nous allons illustrer le processus complet sur un échantillon pour observer l'apport de la fusion et le rôle de la centralité.""")

add_code("""# Prendre un échantillon
sample = dataset[10] # e.g. LAB00030
imgs = {
    'visible': sample['visible'],
    'infrared': sample['infrared']
}

# Fusion : 50% Visible, 50% Infrarouge
weights = {'visible': 0.5, 'infrared': 0.5}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
frangi_response, centrality = extract_frangi_graph_gpu(imgs, weights, device=device)

# Seuillage final adaptatif pour extraire le squelette
# On garde les chemins majeurs (centralité élevée)
skeleton = (centrality > 0.05).astype(np.float32)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(sample['visible'].numpy(), cmap='gray')
axes[0, 0].set_title('Modalité : Visible')

axes[0, 1].imshow(sample['infrared'].numpy(), cmap='gray')
axes[0, 1].set_title('Modalité : Infrarouge (IR)')

axes[0, 2].imshow(sample['gt'].numpy(), cmap='gray')
axes[0, 2].set_title('Ground Truth (Binarisé)')

axes[1, 0].imshow(frangi_response, cmap='magma')
axes[1, 0].set_title('Réponse Frangi Multi-échelles (Fused Λ2)')

axes[1, 1].imshow(centrality, cmap='hot')
axes[1, 1].set_title('Betweenness Centrality (Graph)')

# Superposition : Image fusion en fond + Squelette extrait en Rouge + GT en Vert
fused_bg = sample['fusion'].numpy()
axes[1, 2].imshow(fused_bg, cmap='gray', alpha=0.5)
axes[1, 2].imshow(skeleton, cmap='Reds', alpha=np.where(skeleton > 0, 1.0, 0.0))
axes[1, 2].imshow(sample['gt'].numpy(), cmap='Greens', alpha=np.where(sample['gt'].numpy() > 0, 0.5, 0.0))
axes[1, 2].set_title('Superposition (Rouge: Prédiction, Vert: GT)')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()""")

add_md("""## 5. Évaluation des Métriques (Jaccard / Tversky)

Calcul des métriques sur un batch d'images pour valider l'approche sur le benchmark.""")

add_code("""def compute_metrics(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    jaccard = intersection / (union + 1e-8)
    
    # Tversky (alpha=1, beta=0.5 -> pénalise moins les Faux Positifs, adapté pour les réseaux filaires)
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
    tversky = intersection / (intersection + 1.0 * fn + 0.5 * fp + 1e-8)
    
    return jaccard, tversky

# Évaluation sur 20 images pour démonstration rapide
num_eval = min(20, len(dataset))
jaccard_scores = []
tversky_scores = []

print(f"Évaluation sur {num_eval} images...")
for i in range(num_eval):
    sample = dataset[i]
    imgs = {'visible': sample['visible'], 'infrared': sample['infrared']}
    
    _, centrality = extract_frangi_graph_gpu(imgs, weights, device=device)
    
    # Dilatation du squelette pour comparaison avec le GT (qui a souvent une épaisseur de quelques pixels)
    pred = (centrality > 0.05).astype(np.uint8)
    pred_dilated = cv2.dilate(pred, np.ones((3,3), np.uint8), iterations=1)
    
    gt_arr = sample['gt'].numpy().astype(np.uint8)
    
    j, t = compute_metrics(pred_dilated, gt_arr)
    jaccard_scores.append(j)
    tversky_scores.append(t)

print(f"Jaccard (IoU) Moyen : {np.mean(jaccard_scores):.4f} ± {np.std(jaccard_scores):.4f}")
print(f"Tversky Moyen       : {np.mean(tversky_scores):.4f} ± {np.std(tversky_scores):.4f}")""")

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