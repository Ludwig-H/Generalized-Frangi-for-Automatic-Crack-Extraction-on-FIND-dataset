# Generalized Frangi Fusion for FIND

**Generalized Frangi with Multi-modal Fusion** is a Python package designed for automatic crack extraction, specifically benchmarked on the [FIND dataset](https://zenodo.org/records/6383044).

This toolkit implements an advanced pipeline that combines:
- **Hessian-based filtering** with absolute eigenvalue sorting and normalization.
- **Multi-modal fusion** to combine intensity and range/depth data.
- **Frangi similarity graphs** for robust connectivity analysis.
- **Sparse HDBSCAN clustering** for efficient grouping of crack pixels.
- **Minimum Spanning Tree (MST) & k-centers** for precise skeletonization.

## 📦 Installation

To install the package and its dependencies, clone the repository and install it in editable mode:

```bash
git clone https://github.com/Ludwig-H/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset.git
cd Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset
pip install -e .
```

### Requirements
The project requires Python 3.9+ and the following libraries:
- `numpy`, `scipy`, `scikit-image`, `matplotlib`
- `joblib`, `tqdm`, `tqdm-joblib`
- `hdbscan`, `networkx`, `pandas`
- `gdown`, `tifffile`, `imageio`, `Pillow`, `pot`

## 🚀 Usage

### Command Line Interface

The package provides a CLI tool `frangi-find-batch` to run the evaluation pipeline on the FIND dataset.

```bash
frangi-find-batch --find-root /path/to/FIND_dataset [OPTIONS]
```

**Arguments:**

- `--find-root` (Required): Path to the root folder of the unzipped FIND dataset.
- `--cracksegdiff-dir`: Optional directory containing CrackSegDiff outputs for comparison.
- `--num-images`: Number of images to process (default: 500).
- `--radius`: Radius for the Frangi similarity graph (default: 5).
- `--K`: Filtration parameter, 1 or 2 (default: 1).
- `--expz`: Exponent for the HDBSCAN similarity (default: 2.0).
- `--out-csv`: Output path for the results CSV (default: `results/find_batch_metrics.csv`).

**Example:**

```bash
frangi-find-batch --find-root ./data/FIND --num-images 10 --out-csv ./results/my_test.csv
```

### Python Notebooks

For interactive exploration and visualization of the pipeline steps:
- Open `notebooks/FIND_Frangi_Fusion_Colab.ipynb`.
- This notebook includes code to automatically download the dataset, visualize Hessian scales, graph construction, and final skeletons.

## 🛠️ Pipeline Details

1.  **Hessian Computation**: Multiscale Hessian matrices are computed. Eigenvalues are sorted $|\lambda_1| \le |\lambda_2|$ and normalized per scale.
2.  **Fusion**: Hessians from different modalities (e.g., intensity, range) are fused.
3.  **Graph Construction**: A similarity graph is built where nodes are pixels and edges represent Frangi-based similarity within a radius $R$.
4.  **Clustering**: A custom sparse implementation of HDBSCAN clusters the graph nodes to identify crack regions.
5.  **Skeletonization**: The core crack structure is extracted using MST $k$-centers and path finding.

## 📂 Project Structure

```
.
├── src/frangi_fusion/      # Core source code
│   ├── cli.py              # CLI entry point
│   ├── hessian.py          # Hessian filters & fusion
│   ├── frangi_graph.py     # Graph construction
│   ├── clustering_sparse.py # Sparse HDBSCAN
│   └── ...
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts
├── setup.cfg               # Package configuration
└── README.md               # Original documentation
```

## 📄 Citation

If you use this code, please refer to the `CITATION.cff` file in the root directory.
