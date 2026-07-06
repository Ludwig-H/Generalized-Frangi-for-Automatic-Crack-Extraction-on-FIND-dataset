# VT-GraF Benchmark (Visible--Thermal Granular Faults under severe clutter)

This benchmark evaluates crack extraction algorithms under severe surface texture clutter (asphalt road gravel aggregates) using aligned visible (optical) and thermal infrared (IR) modalities from the **VT-GraF-Dataset**.

---

## 1. The Challenge of Asphalt Granular Clutter
In pavement inspection, the visible optical modality suffers from extreme high-frequency noise because the borders of individual gravel stones and the gap spaces between them look identical to thin asphalt cracks. Traditional edge filters (and even zero-shot deep foundation models like SAM/SAM 2) applied directly to visible images produce massive false positives by trying to segment every stone.

The thermal infrared modality offers a clean physical contrast of the crack (which appears as a bright, warm anomaly), but at a lower resolution and with local thermal spot noise (caused by different stone heat capacities or moisture).

---

## 2. Compared Methods

### Method A: Ours (Generalized Frangi Graph)
* **Concept**: Unsupervised, training-free multimodal graph-based approach.
* **Mechanism**: 
  1. Computes scale-wise Hessian tensors for both visible and thermal modalities.
  2. Normalizes and combines them constructively at the Hessian level (**tensor-level fusion**).
  3. Builds a sparse graph on PyTorch GPU (supporting $K=1$ pixel-wise or $K=2$ simplicial dual triangle cliques).
  4. Extracts the Minimum Spanning Tree (MST) on the largest connected components.
  5. Computes a fast $O(N)$ Weighted Betweenness Centrality on the rooted tree to prune the graph top-down.
* **Key Advantages**: Solves crack discontinuities via global tree connectivity and completely filters out random textural noise using topological centrality.

### Method B: Baseline (Standard Frangi on Thermal + SAM 2 on Visible)
* **Concept**: Zero-shot foundation model guided by a classical physical edge detector.
* **Mechanism**:
  1. Applies a standard Frangi filter (multiscale Hessian eigenvalue analysis) to the **thermal image** (properly decoded from its JET colormap to a linear temperature scale).
  2. Thresholds the response strictly (retaining only the top 0.5% pixels) to isolate the main crack path.
  3. Skeletonizes the thresholded mask and samples $N=12$ coordinate points.
  4. Feeds the high-resolution **visible image** and these 12 points as **Point Prompts** to Meta's official **SAM 2** (Segment Anything Model 2, `sam2_hiera_large` version).
* **Limitation**: Standard Frangi has no global topological constraints. It generates fragmented prompts when the thermal signal is weak, and generates false prompts in areas containing local thermal anomalies, which leads SAM 2 to segment incorrect road textures.

---

## 3. How to Run the Benchmark

The folder contains a ready-to-run Jupyter notebook:
👉 **[VT_GraF_Benchmark_Colab.ipynb](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/VT-GraF%20Benchmark/VT_GraF_Benchmark_Colab.ipynb)**

### Colab Pro Execution Steps:
1. Upload this folder (`ISPRS/`) or mount your Google Drive in Google Colab Pro.
2. Ensure you select a **GPU runtime** (T4, V100, or A100).
3. Open the notebook and run all cells.
4. The notebook will automatically:
   * Download the VT-GraF-Dataset (5 dual-modality fissures).
   * Install SAM 2 from Meta's source repository.
   * Run both **Ours (Generalized Frangi Graph)** and the **Baseline (Frangi Thermal + SAM 2)**.
   * Calculate Jaccard index (IoU), Tversky index, and Wasserstein distance against the ground truth skeletons.
   * Plot comparative subplots showing the visible, thermal, ground truth, and both prediction masks side-by-side.
