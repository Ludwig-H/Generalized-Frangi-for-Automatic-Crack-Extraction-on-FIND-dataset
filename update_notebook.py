
import json
import os

notebook_path = "FIND_Frangi_Fusion_Avignon_Colab.ipynb"

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Read the code blocks
with open("batch_code.txt", "r") as f:
    batch_code = f.readlines() # Keep newlines

with open("noise_code.txt", "r") as f:
    noise_code = f.readlines() # Keep newlines

# --- 1. Modify Batch Processing Cell ---
batch_cell_found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "# --- Batch Processing 500 Images (USE_COMBO=True) ---" in source:
            batch_cell_found = True
            cell['source'] = batch_code
            print("Batch cell updated.")

# --- 2. Consolidate Noise Benchmark Cells ---
indices_to_remove = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Identify the noise-related cells
        if "def make_noisy_arrays" in source:
            indices_to_remove.append(i)
        elif "def frangi_predict_mask_from_arrays" in source:
            indices_to_remove.append(i)
        elif "def process_image_noise" in source:
            indices_to_remove.append(i)
        elif "df_speckle = run_noise_benchmark" in source:
            indices_to_remove.append(i)

# Remove old noise cells
for i in sorted(indices_to_remove, reverse=True):
    del nb['cells'][i]
print(f"Removed {len(indices_to_remove)} old noise benchmark cells.")

# Add new consolidated noise cell
new_noise_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": noise_code
}
nb['cells'].append(new_noise_cell)

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully updated via file injection.")
