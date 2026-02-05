# Universal and Robust Multi-Modal Crack Extraction via Generalized Frangi Graphs

**EUSIPCO 2026**

This repository contains the official implementation of our paper: **"Multi-Modal, Training-Free Crack Extraction *via* Generalized Frangi Graph"**.

We propose a "universal", **training-free approach** that robustly extracts crack networks across varying data distributions (from civil infrastructure to geological faults). Our method generalizes the classical **Frangi vesselness filter** to the multi-modal setting, fusing photometric (intensity) and geometric (range/depth) data at the Hessian level. Instead of pixel-wise classification, we construct a sparse graph driven by a pairwise Frangi similarity metric and extract a topological skeleton using **Weighted Betweenness Centrality**.

## 🚀 Key Features

*   **Training-Free & Universal:** No neural network training required. Robust to domain shifts (Zero-Shot transfer).
*   **Hessian-Level Fusion:** Fuses Intensity and Range/Depth data to reinforce geometric signals and suppress noise.
*   **Generalized Frangi Graph:** Goes beyond pixel-wise filtering by encoding local tubular geometry (elongation, contrast, alignment) into a graph structure.
*   **Topological Extraction:** Uses Minimum Spanning Tree (MST) and Betweenness Centrality to extract precise, continuous crack skeletons.

## 📂 Repository Structure

*   `FIND_Frangi_Fusion_Avignon_Colab.ipynb`: The main notebook reproducing all experiments (Avignon case study + FIND Benchmark + Noise Robustness).
*   `src/`: Source code for the Frangi-Fusion python package.
*   `scripts/`: Utility scripts for batch processing.

## 💻 Reproducibility

### Quick Start (Colab)

The easiest way to reproduce our results is to use the provided Jupyter Notebook: `FIND_Frangi_Fusion_Avignon_Colab.ipynb`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ludwig-H/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/blob/main/FIND_Frangi_Fusion_Avignon_Colab.ipynb)

### Google Drive Setup (Important!)

To fully reproduce the comparison with state-of-the-art methods (specifically **CrackSegDiff**) and the noise robustness benchmark, the notebook requires a specific Google Drive structure to load pre-computed masks (since we do not include the heavy Deep Learning model checkpoints here).

1.  **Mount your Drive** in the notebook.
2.  **Create the following folder structure**:
    ```text
    /content/drive/MyDrive/Datasets/FIND/Results/CrackSegDiff/20000_1000/test_output_fused/
    ```
3.  **Place CrackSegDiff results** in this folder. Filenames must match `imXXXXX_output_ens.png`.
4.  *(Optional)* For the noise benchmark, results should be in:
    ```text
    /content/drive/MyDrive/Datasets/FIND/Results/CrackSegDiff_noise/
    ```

**Note:** If these files are missing, the notebook will simply skip the comparison metrics and display results for our method only.

## 📄 Citation

If you use this code for your research, please cite our paper:

```bibtex
@misc{HauseuxEUSIPCO2026,
  title={Multi-Modal, Training-Free Crack Extraction via Generalized Frangi Graph},
  author={Hauseux, Louis and Antoine, Raphaël and Foucher, Philippe and Charbonnier, Pierre and Zerubia, Josiane},
  note={submitted to EUSIPCO 2026},
  year={2026}
}
```