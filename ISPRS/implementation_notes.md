# ISPRS Multi-Modal Crack Detection - Implementation Notes & Conventions

This guide serves as a technical reference for creating other folders, notebooks, or scripts within the `ISPRS` directory. It documents the dataset conventions, data processing pipelines, and algorithmic implementations.

---

## 1. Raphael-Dataset Conventions

### Directory & File Naming Anomaly
The dataset is structured into folders from `Fissure 1` to `Fissure 5`. However, there is a filename prefix anomaly for **Fissure 2** which must be handled programmatically:
*   `Fissure 1` $\rightarrow$ Files: `fissure1_visible.png`, `fissure1_thermique.png`, `fissure1_verite_terrain.png`
*   `Fissure 2` $\rightarrow$ **Files are prefixed with `fissure6`**: `fissure6_visible.png`, `fissure6_thermique.png`, `fissure6_verite_terrain.png`
*   `Fissure 3` $\rightarrow$ Files: `fissure3_visible.png`, `fissure3_thermique.png`, `fissure3_verite_terrain.png`
*   `Fissure 4` $\rightarrow$ Files: `fissure4_visible.png`, `fissure4_thermique.png`, `fissure4_verite_terrain.png`
*   `Fissure 5` $\rightarrow$ Files: `fissure5_visible.png`, `fissure5_thermique.png`, `fissure5_verite_terrain.png`

---

## 2. Thermal Modal Decoding (JET Colormap)

### The Grayscale Trap
Using standard grayscale conversion `cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)` on JET-colored images is mathematically and physically incorrect:
*   JET maps low values to Blue, intermediate values to Green, and high values to Red.
*   Standard grayscale conversion weights Green at $58.7\%$ and Red at only $29.9\%$.
*   As a result, mid-range temperatures (green) look twice as bright in standard grayscale as the actual maximum temperatures (red). This distorts the Hessian eigenvalues.

### Correct KDTree Decoding
To retrieve the true physical linear temperature:
1.  Generate the standard 256 BGR colors of the matplotlib `'jet'` colormap.
2.  Use a fast KDTree (`scipy.spatial.cKDTree`) to map each pixel of the input BGR image back to its closest index (0 to 255) in the JET palette.
3.  This yields a linear temperature grayscale image.

### Polarity Alignment (Crucial)
*   **Visible Modality**: Cracks appear as **dark lines on a bright background** (low values near 0).
*   **Thermal Modality**: Cracks appear as **hot regions** (bright values near 255).
*   **Inversion Step**: To align polarities for tensor-level fusion and default standard filters (which expect `black_ridges=True`), the decoded thermal image must be inverted:
    $$\text{img\_ir\_inverted} = 255 - \text{img\_ir\_decoded}$$
    This represents the hot crack as a dark curvilinear structure.

---

## 3. Ground Truth Processing

Ground truth PNG files (`*_verite_terrain.png`) are transparent images where the crack is drawn in black:
*   **Alpha Channel Decoding**: The alpha channel (4th channel) holds the crack mask. If `img.shape[-1] == 4`, decode using:
    $$\text{gt\_mask} = (\text{alpha\_channel} > 0)$$
*   **Grayscale Fallback**: If the alpha channel is missing, decode using:
    $$\text{gt\_mask} = (\text{grayscale} < 127)$$
*   **Interpolation**: When resizing the ground truth mask to match the visible image size, **always use nearest-neighbor interpolation** (`cv2.INTER_NEAREST`) to avoid blurring binary edges.

---

## 4. Algorithmic Parameters (Generalized Frangi Graph)

For evaluations on the Raphael dataset, the optimal parameters established in the latest notebook commits are:
*   **Fusion Weights**: `visible: 1/3` (~0.33) and `infrared: 2/3` (~0.67) to prioritize the thermal physical signature.
*   **Scales ($\Sigma$)**: Multi-scale set `Σ = [20, 30, 40]` to handle large and variable crack widths.
*   **Search Radius ($R$)**: $R = 3$ (local neighborhood of $7 \times 7$ pixels) to construct graph edges.
*   **Graph Clique ($K$)**: $K = 2$ (simplicial complexes of dual triangle cliques) to enforce 2D topological continuity.
*   **Centrality Threshold**: $\tau_{\text{centrality}} = 0.025$ to prune the pruned graph components.
*   **Evaluation Skeleton Thickness**: **5 pixels** (`pixels=5` in `thicken()`) for both the prediction and the ground truth skeletons (to increase evaluation robustness on high-resolution images).

---

## 5. Baseline (SAM 2) Implementation Details

When running comparisons against Segment Anything 2 (SAM 2):
1.  **Thermal Filter**: Apply `skimage.filters.frangi` to the inverted decoded thermal image using the same scales `sigmas=[20, 30, 40]`.
2.  **Point Prompts**: Threshold the top $0.5\%$ pixels of the Frangi response, skeletonize it, and sample $N=12$ points regularly along the centerline.
3.  **SAM 2 Model**: Use Meta's official `SAM2ImagePredictor` with the `sam2_hiera_large.pt` checkpoint.
4.  **RGB Conversion**: SAM 2 expects a 3-channel image. Convert the visible grayscale image using `cv2.cvtColor(img_vis, cv2.COLOR_GRAY2RGB)` before calling `predictor.set_image()`.
