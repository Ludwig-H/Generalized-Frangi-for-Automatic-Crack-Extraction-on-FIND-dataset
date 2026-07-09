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
    
    # Project horizontally to find rows
    row_proj = np.sum(non_white, axis=1)
    row_indices = np.where(row_proj > 0)[0]
    
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
    
    if len(row_blocks) < 2:
        print(f"Error: Not enough rows detected for {name}. Found: {len(row_blocks)}")
        return

    # Use fixed columns from the grid layout
    # Col 0: 10 to 589
    # Col 1: 605 to 1184
    # Col 2: 1200 to 1779
    pad_top = 22
    pad_bottom = 5
    pad_left = 5
    pad_right = 5

    c0_start = max(0, 10 - pad_left)
    c0_end = min(img.shape[1], 589 + pad_right)

    c1_start = max(0, 605 - pad_left)
    c1_end = min(img.shape[1], 1184 + pad_right)

    c2_start = max(0, 1200 - pad_left)
    c2_end = min(img.shape[1], 1779 + pad_right)

    # Row 0
    r0_start = max(0, row_blocks[0][0] - pad_top)
    r0_end = min(img.shape[0], row_blocks[0][1] + pad_bottom)
    
    # Row 1
    r1_start = max(0, row_blocks[1][0] - pad_top)
    r1_end = min(img.shape[0], row_blocks[1][1] + pad_bottom)
    
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

output_dir = "/workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/CrackSAM/results_images_cracksam"
for i in range(1, 6):
    name = f"Fissure_{i}"
    image_path = os.path.join(output_dir, f"{name}_comparison.png")
    crop_subplots(image_path, output_dir, name)
