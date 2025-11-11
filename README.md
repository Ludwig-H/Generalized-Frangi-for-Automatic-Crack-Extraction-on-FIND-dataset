# Generalized Frangi with Multi‑modal Fusion on FIND

This repository implements a *generalized Frangi* pipeline with **multi‑modal fusion** and an evaluation on the **FIND** benchmark.
It follows Gretsi ideas and adds a **Frangi similarity graph** on pixels, **HDBSCAN** on a **sparse distance graph**, an **MST** inside each cluster,
and a compact **fault network** obtained by **k‑centers** plus **MST paths**.

Main features
- Robust TIFF loading, grayscale conversion
- Hessian with reflective boundaries; eigenvalues are **sorted by absolute value** so that \(|\lambda_1| \le |\lambda_2|\)
- Eigenvalue *normalization per-scale* by dividing by \(\max_{x,y} |\lambda_2(x,y)|\) so that \(\lambda_1,\lambda_2\in[-1,1]\)
- Per-scale fusion of raw Hessians across modalities
- Frangi similarity graph within radius \(R\); optional **triangle‑connectivity** (Rips filtration) for \(K=2\)
- Sparse **HDBSCAN-like** implementation (mutual reachability → MST → condensed tree → EOM selection) operating **directly on CSR**
- MST + **k‑centers** and **exact MST paths** between centers to produce a crack skeleton
- Rich **Colab notebook** with formulas, figures, and many inline illustrations

Quick start
1. Open the notebook `notebooks/FIND_Frangi_Fusion_Colab.ipynb`.
2. It downloads FIND (`data.zip`) with `gdown`, unzips, then runs the full pipeline.
3. For batch evaluation and comparison to CrackSegDiff, use `scripts/run_batch_find.py`.

License: MIT
