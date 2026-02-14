# Gauthier_Cerema: Steger GPU Implementation

This directory contains a reconstruction of the crack extraction pipeline, adapted to use **Steger's Filter (1998)** and **GPU Acceleration** (via PyTorch).

## Key Features

1.  **GPU Acceleration**: Core image processing (Hessian, Eigenvalues, Convolution) is implemented using PyTorch tensors, allowing for significant speedups on CUDA-enabled devices (and efficient execution on CPU).
2.  **Steger Filter**: Instead of the probabilistic "Frangi Vesselness", we use Steger's differential geometry approach to locate the precise *center* of curvilinear structures with sub-pixel accuracy.
    *   **Criterion**: The center is defined where the first directional derivative along the normal vanishes ($t \in [-0.5, 0.5]$).

## Files

*   `steger_gpu.py`: Contains the `StegerHessian` class. This is the core engine. It precomputes Gaussian kernels and performs the detection logic.
*   `run_steger_analysis.py`: A script to run the filter on an input image (default: `data_avignon/Ortho_new_extrait.tif`) and save the results in `results/`.
*   `results/`: Directory containing generated outputs (Steger response map and visualizations).

## Usage

To run the analysis:

```bash
python3 run_steger_analysis.py
```

## Next Steps for Integration

To fully replicate the "Generalized Graph" approach:
1.  Use the `valid_mask` and `nx, ny` (orientation) from `steger_gpu.py` to build the sparse graph.
2.  Nodes are pixels where `valid_mask` is True.
3.  Edges connect neighboring valid pixels.
4.  Weighting can be simplified since Steger already filters for "tubular" geometry.
