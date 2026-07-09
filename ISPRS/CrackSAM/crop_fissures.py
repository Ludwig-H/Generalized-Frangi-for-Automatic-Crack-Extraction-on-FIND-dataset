import cv2
import numpy as np
import os

def crop_subplots(image_path, output_dir, name):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: could not read {image_path}")
        return
    
    # Background is white (255, 255, 255). Let's find non-white areas.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    non_white = (gray < 250).astype(np.uint8) * 255
    
    # We project horizontally and vertically to find rows and columns
    row_proj = np.sum(non_white, axis=1)
    col_proj = np.sum(non_white, axis=0)
    
    # Find active row ranges
    row_indices = np.where(row_proj > 0)[0]
    col_indices = np.where(col_proj > 0)[0]
    
    # Find contiguous blocks
    def get_blocks(indices, min_size=50):
        blocks = []
        if len(indices) == 0:
            return blocks
        start = indices[0]
        for i in range(1, len(indices)):
            if indices[i] > indices[i-1] + 5:  # gap of more than 5 pixels
                end = indices[i-1]
                if end - start >= min_size:
                    blocks.append((start, end))
                start = indices[i]
        end = indices[-1]
        if end - start >= min_size:
            blocks.append((start, end))
        return blocks

    row_blocks = get_blocks(row_indices, min_size=80)
    col_blocks = get_blocks(col_indices, min_size=80)
    
    print(f"Detected {len(row_blocks)} rows and {len(col_blocks)} columns for {name}")
    for idx, r in enumerate(row_blocks):
        print(f"  Row {idx}: {r[0]} to {r[1]} (height: {r[1]-r[0]})")
    for idx, c in enumerate(col_blocks):
        print(f"  Col {idx}: {c[0]} to {c[1]} (width: {c[1]-c[0]})")
        
    # We want:
    # Row 0, Col 0: RGB (visible)
    # Row 0, Col 1: Thermal (infrared)
    # Row 1, Col 0: Frangi similarity
    # Row 1, Col 2: Overlay
    
    if len(row_blocks) >= 2 and len(col_blocks) >= 3:
        # Add some padding to include titles
        pad_top = 22
        pad_bottom = 5
        pad_left = 5
        pad_right = 5
        
        # Row 0
        r0_start = max(0, row_blocks[0][0] - pad_top)
        r0_end = min(img.shape[0], row_blocks[0][1] + pad_bottom)
        
        # Row 1
        r1_start = max(0, row_blocks[1][0] - pad_top)
        r1_end = min(img.shape[0], row_blocks[1][1] + pad_bottom)
        
        # Columns
        c0_start = max(0, col_blocks[0][0] - pad_left)
        c0_end = min(img.shape[1], col_blocks[0][1] + pad_right)
        
        c1_start = max(0, col_blocks[1][0] - pad_left)
        c1_end = min(img.shape[1], col_blocks[1][1] + pad_right)
        
        c2_start = max(0, col_blocks[2][0] - pad_left)
        c2_end = min(img.shape[1], col_blocks[2][1] + pad_right)
        
        # Crop and save
        rgb_img = img[r0_start:r0_end, c0_start:c0_end]
        thermal_img = img[r0_start:r0_end, c1_start:c1_end]
        frangi_img = img[r1_start:r1_end, c0_start:c0_end]
        overlay_img = img[r1_start:r1_end, c2_start:c2_end]
        
        cv2.imwrite(os.path.join(output_dir, f"{name}_rgb.png"), rgb_img)
        cv2.imwrite(os.path.join(output_dir, f"{name}_thermal.png"), thermal_img)
        cv2.imwrite(os.path.join(output_dir, f"{name}_frangi.png"), frangi_img)
        cv2.imwrite(os.path.join(output_dir, f"{name}_overlay.png"), overlay_img)
        print(f"Successfully saved cropped subplots for {name}")
    else:
        print(f"Error: Not enough rows/cols detected for {name}")

output_dir = "/workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/CrackSAM/results_images_cracksam"
for i in range(1, 6):
    name = f"Fissure_{i}"
    image_path = os.path.join(output_dir, f"{name}_comparison.png")
    crop_subplots(image_path, output_dir, name)
