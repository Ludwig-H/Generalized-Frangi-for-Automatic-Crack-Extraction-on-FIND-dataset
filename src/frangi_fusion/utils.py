
import os, re, random, numpy as np
from glob import glob
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

_CMAP_TREE_CACHE = {}

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

def recover_scalar_from_cmap(img_rgb: np.ndarray, cmap_name: str = 'jet', invert: bool = False) -> np.ndarray:
    """
    Reconstruct scalar values (0.0 to 1.0) from an image encoded with a colormap (e.g. JET).
    """
    global _CMAP_TREE_CACHE
    
    # Ensure image is float 0..1
    if img_rgb.dtype == np.uint8:
        pixel_data = img_rgb.astype(np.float32) / 255.0
    else:
        pixel_data = img_rgb.astype(np.float32)
        if pixel_data.max() > 1.0:
            pixel_data /= pixel_data.max()

    h, w = pixel_data.shape[:2]
    # Keep only RGB if RGBA
    if pixel_data.shape[2] > 3:
        pixel_data = pixel_data[..., :3]
    
    flat_pixels = pixel_data.reshape(-1, 3)
    
    # Initialize Cache for this cmap if needed
    if cmap_name not in _CMAP_TREE_CACHE:
        N_SAMPLES = 2048
        scalar_values = np.linspace(0, 1, N_SAMPLES)
        cmap = plt.get_cmap(cmap_name)
        # cmap returns rgba, take rgb
        palette_rgb = cmap(scalar_values)[:, :3]
        tree = KDTree(palette_rgb)
        _CMAP_TREE_CACHE[cmap_name] = (tree, scalar_values)
    
    tree, scalar_refs = _CMAP_TREE_CACHE[cmap_name]
    
    # Query nearest color
    # distance, indices
    _, indices = tree.query(flat_pixels)
    
    recovered_flat = scalar_refs[indices]
    recovered_map = recovered_flat.reshape(h, w)
    
    if invert:
        recovered_map = 1.0 - recovered_map
        
    return recovered_map

def _is_image_file(p: str) -> bool:
    return p.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff",".bmp"))

def _read_image(path: str) -> np.ndarray:
    # Try PIL
    try:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.array(im)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[..., :3]
            return arr
    except Exception:
        pass
    # Try imageio
    try:
        import imageio.v2 as iio
        arr = iio.imread(path)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        return arr
    except Exception:
        pass
    # Try skimage (tifffile backend)
    try:
        from skimage.io import imread
        arr = imread(path)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        return arr
    except Exception as e:
        raise e

def to_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img.astype(np.float32)
    elif img.ndim == 3:
        c = img.shape[2]; arr = img.astype(np.float32)
        if c >= 3:
            w = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
            g = arr[..., :3].dot(w)
        elif c == 2:
            g = arr.mean(axis=2)
        else:
            g = arr[..., 0]
    else:
        raise ValueError("Unsupported image shape")
    g -= g.min()
    if g.max() > 0: g /= g.max()
    return (g * 255).clip(0, 255).astype(np.uint8)

def auto_discover_find_structure(root: str):
    """
    Discover FIND-like structure:
      - labels in 'lbl/' (or 'lbs', 'labels/', 'gt', etc.)
      - modalities in 'img/' (intensity / range / fused)
    Returns dict with keys: 'intensity', 'range', 'fused', 'label'.
    """
    all_imgs = [
        p for p in glob(os.path.join(root, '**', '*.*'), recursive=True)
        if _is_image_file(p)
    ]

    buckets = {
        'intensity': [],
        'range': [],
        'fused': [],
        'label': [],
        'filtered': []
    }

    for p in all_imgs:
        low = p.lower().replace('\\', '/')
        parts = low.split('/')

        # 1) Labels: dossiers lbl / labels / gt…
        if any(part in ('lbl', 'lbs', 'label', 'labels', 'gt', 'groundtruth', 'ground_truth') for part in parts):
            buckets['label'].append(p)
            continue

        # 2) Tout le reste : on considère que c'est du côté img/
        #    On raffine par mots-clés dans le chemin
        if any(k in low for k in ['fused', 'fusion']):
            buckets['fused'].append(p)
        elif any(k in low for k in ['range', 'depth']):
            buckets['range'].append(p)
        elif any(k in low for k in ['filtered']):
            buckets['filtered'].append(p)
        else:
            # Par défaut, on met en intensity
            buckets['intensity'].append(p)

    for k in buckets:
        buckets[k] = sorted(buckets[k])

    print("Found:",
          f"{len(buckets['intensity'])} intensity,",
          f"{len(buckets['range'])} range,",
          f"{len(buckets['fused'])} fused,",
          f"{len(buckets['filtered'])} filtered,",
          f"{len(buckets['label'])} labels.")
    return buckets
import re

def _extract_key(p: str) -> str:
    """
    Extract a numeric key from the filename, robust to zero-padding.
    Example:
      'im00215.bmp' -> '215'
      'crack_0215_range.png' -> '215'
    If no digits are found, fallback to lowercase basename.
    """
    base = os.path.basename(p)
    m = re.findall(r'\d+', base)
    if not m:
        return base.lower()
    # on convertit en int pour casser le padding, puis on revient à str
    num = int(m[-1])
    return str(num)


def load_modalities_and_gt_by_index(struct, index: int):
    """
    Load intensity / range / fused / filtered (and optional label) for a given index.

    - Base list: 'label' if available (pour évaluer uniquement là où il y a GT),
                 sinon 'intensity'.
    - Key = numeric part of filename, normalisée (ex: 'im00215.bmp' et 'im215.png' -> '215').
    - Pour chaque modalité ['intensity','range','fused','filtered','label'],
      on prend le premier fichier dont la clé matche.
    - Les images sont converties en uint8 [0,255] via to_gray_uint8.
    - Le label est binarisé (seuil 127) en {0,255}.
    - Pour range/filtered, si l'image est RGB, on décode la palette JET.
    """
    base_list = struct['label'] if struct.get('label') else struct.get('intensity', [])
    if not base_list:
        raise RuntimeError('No images found in FIND root (no label nor intensity).')

    index = index % len(base_list)
    key = _extract_key(base_list[index])

    out = {'paths': {}, 'arrays': {}}

    # On inclut bien 'filtered'
    for k in ['intensity', 'range', 'fused', 'filtered', 'label']:
        files = struct.get(k, [])
        if not files:
            continue

        # chercher fichiers de cette modalité qui partagent la même clé normalisée
        cand = [p for p in files if _extract_key(p) == key]
        if not cand:
            continue

        pth = cand[0]
        try:
            arr = _read_image(pth)
            out['paths'][k] = pth

            if k == 'label':
                g = to_gray_uint8(arr)
                out['arrays'][k] = ((g > 127).astype(np.uint8) * 255)
            elif k in ['range', 'filtered'] and arr.ndim == 3 and arr.shape[2] >= 3:
                # Decodage palette JET (supposée)
                # invert=False => Valeurs basses (Bleu/Noir) = Creux/Fissure
                scalar = recover_scalar_from_cmap(arr, cmap_name='jet', invert=False)
                out['arrays'][k] = (scalar * 255).clip(0, 255).astype(np.uint8)
            else:
                out['arrays'][k] = to_gray_uint8(arr)

        except Exception as e:
            print(f"Failed to read {pth} as {k}: {e}")
            continue

    return out
