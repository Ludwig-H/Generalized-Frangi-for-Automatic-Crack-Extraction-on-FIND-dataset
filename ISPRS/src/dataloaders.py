import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def decode_jet_to_grayscale(img_bgr):
    """
    Decodes a BGR image with a JET colormap back to a linear grayscale representing temperature.
    Uses a 256-color look-up table and nearest-neighbor search for 100% mathematical accuracy.
    """
    # Generate the standard JET colormap from matplotlib
    cmap = plt.get_cmap('jet')
    jet_colors = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r, g, b, _ = cmap(i)
        jet_colors[i] = [int(b * 255), int(g * 255), int(r * 255)] # BGR format
        
    H, W, _ = img_bgr.shape
    flat_img = img_bgr.reshape(-1, 3).astype(np.float32)
    
    # KDTree for fast nearest-neighbor matching
    tree = cKDTree(jet_colors.astype(np.float32))
    _, indices = tree.query(flat_img, k=1)
    
    # Reconstruct the decoded grayscale image
    gray_img = indices.reshape(H, W).astype(np.uint8)
    return gray_img

class VTGraFDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = None
        for path in Path(root_dir).rglob('Fissure 1'):
            self.root_dir = path.parent
            break
            
        if self.root_dir is None:
            raise FileNotFoundError("VT-GraF dataset structure (Fissure 1 directory) not found.")
            
        self.fissure_dirs = sorted([d for d in self.root_dir.glob('Fissure *') if d.is_dir()])
        print(f"Dataset VT-GraF loaded with {len(self.fissure_dirs)} fissures: {[d.name for d in self.fissure_dirs]}")

    def __len__(self):
        return len(self.fissure_dirs)

    def __getitem__(self, idx):
        fissure_dir = self.fissure_dirs[idx]
        fissure_name = fissure_dir.name
        num = fissure_name.split(' ')[-1]
        
        # Fissure 2 actually contains files prefixed with fissure6
        if num == '2':
            prefix = 'fissure6'
        else:
            prefix = f"fissure{num}"
        
        path_vis = fissure_dir / f"{prefix}_visible.png"
        path_ir = fissure_dir / f"{prefix}_thermique.png"
        path_gt = fissure_dir / f"{prefix}_verite_terrain.png"
        
        if not path_vis.exists():
            raise FileNotFoundError(f"Visible image not found: {path_vis}")
        if not path_ir.exists():
            raise FileNotFoundError(f"Thermal image not found: {path_ir}")
        if not path_gt.exists():
            raise FileNotFoundError(f"Ground truth image not found: {path_gt}")
            
        # 1. Load visible grayscale
        img_vis = cv2.imread(str(path_vis), cv2.IMREAD_COLOR)
        if img_vis is not None: 
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY)
        else: 
            raise FileNotFoundError(f"Failed to read visible image {path_vis}")
            
        # 2. Load thermal JET and decode it properly
        img_ir_color = cv2.imread(str(path_ir), cv2.IMREAD_COLOR)
        if img_ir_color is not None:
            # Decode JET colormap to get proper physical linear temperature values
            img_ir = decode_jet_to_grayscale(img_ir_color)
            # Invert so that the hot cracks appear dark (matching the visible polarity)
            img_ir = 255 - img_ir
            img_ir = cv2.resize(img_ir, (img_vis.shape[1], img_vis.shape[0]), interpolation=cv2.INTER_LINEAR)
        else: 
            raise FileNotFoundError(f"Failed to read thermal image {path_ir}")
            
        # 3. Load ground truth (binarizing using the alpha channel if present)
        img_gt = cv2.imread(str(path_gt), cv2.IMREAD_UNCHANGED)
        if img_gt is not None:
            if img_gt.shape[-1] == 4:
                alpha_channel = img_gt[:, :, 3]
                gt_clean = (alpha_channel > 0).astype(np.float32)
            else:
                gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                gt_clean = (gray_gt < 127).astype(np.float32)
            gt_clean = cv2.resize(gt_clean, (img_vis.shape[1], img_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            raise FileNotFoundError(f"Failed to read ground truth image {path_gt}")
            
        vis_t = torch.from_numpy(img_vis).float() / 255.0
        ir_t  = torch.from_numpy(img_ir).float() / 255.0
        gt_t = torch.from_numpy(gt_clean)
        
        return {'id': fissure_name, 'visible': vis_t, 'infrared': ir_t, 'gt': gt_t}

