
import os, re, random, numpy as np
from glob import glob

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

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
    all_imgs = [p for p in glob(os.path.join(root, '**', '*.*'), recursive=True) if _is_image_file(p)]
    buckets = {'intensity': [], 'range': [], 'fused': [], 'label': []}
    for p in all_imgs:
        low = p.lower().replace('\\','/')
        if any(k in low for k in ['label','labels','gt','groundtruth','ground_truth','mask']):
            buckets['label'].append(p)
        elif any(k in low for k in ['fused','fusion']):
            buckets['fused'].append(p)
        elif any(k in low for k in ['range','depth']):
            buckets['range'].append(p)
        elif any(k in low for k in ['intensity','gray','grayscale']):
            buckets['intensity'].append(p)
        else:
            buckets['intensity'].append(p)
    for k in buckets:
        buckets[k] = sorted(buckets[k])
    return buckets

def _extract_key(p: str) -> str:
    import os, re
    base = os.path.basename(p)
    m = re.findall(r'\\d+', base)
    return m[-1] if m else base

def load_modalities_and_gt_by_index(struct, index: int):
    base_list = struct['label'] if struct['label'] else struct['intensity']
    if not base_list: raise RuntimeError('No images found in FIND root.')
    index = index % len(base_list)
    key = _extract_key(base_list[index])
    out = {'paths':{}, 'arrays':{}}
    for k in ['intensity','range','fused','label']:
        cand = [p for p in struct.get(k,[]) if _extract_key(p)==key]
        if not cand: continue
        pth = cand[0]
        try:
            arr = _read_image(pth)
            out['paths'][k] = pth
            if k=='label':
                g = to_gray_uint8(arr)
                out['arrays'][k] = (g > 127).astype(np.uint8)*255
            else:
                out['arrays'][k] = to_gray_uint8(arr)
        except Exception:
            continue
    return out
