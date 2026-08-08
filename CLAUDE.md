# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A research monorepo around one algorithm: **training-free crack/fault extraction by a Generalized Frangi
Graph** — multi-modal Hessians fused at the operator level, a sparse pairwise-similarity graph, then a
minimum spanning tree pruned by weighted betweenness centrality. It contains the published EUVIP 2026
artifact, GPU re-implementations for several benchmarks (FIND, CrackForest, IRT-Crack, VT-GraF), and an
ISPRS follow-up that asks whether this geometry can help SAM 2.

There is no single application to run. Work happens in one of the tracks below, each with its own
conventions.

## Commands

```bash
# Reference CPU package (editable install provides the `frangi-find-batch` entry point)
python -m pip install -e .
frangi-find-batch --find-root /path/to/FIND --K 2 --out-csv results/find_batch_metrics.csv

# CrackSAM 2 test suite (~215 tests, CPU-only, no checkpoints needed)
python -m pytest ISPRS/CrackSAM/tests

# A single CrackSAM 2 test file — cd first (see gotcha below)
cd ISPRS/CrackSAM && python -m pytest tests/test_model.py -q

# GCP script contracts (bash -n + static assertions, no cloud calls)
python -m pytest gcp-migration/tests

# Anti-shadow filter study (self-contained, no SAM 2)
python -m pytest "ISPRS/CrackSAM/results/2026-08-08_guidage_geometrique_anti_ombre/test_anti_shadow_filters.py"

# Regenerate Colab notebooks (run from the repo root: output paths are relative)
python build_vt_graf_notebook.py               # -> VT-GraF/Frangi_VT_GraF_GPU.ipynb
python CrackForest/build_notebook.py           # -> CrackForest/Frangi_CrackForest_GPU.ipynb
python "IRT-Crack Segmentation/build_notebook.py"

# EUVIP paper (both variants share LaTeX/main.tex)
cd EUVIP/LaTeX && pdflatex camera-ready.tex && bibtex camera-ready && pdflatex camera-ready.tex && pdflatex camera-ready.tex
```

**Never run bare `pytest` from the repo root.** Three root-level files match `test_*` but are not tests:
`test_k2_clean.py` and `test_k2_script.py` are Colab exports containing `!pip`/`!git` magics (SyntaxError
at collection), and `test_fast_filter.py` is a 50-million-element benchmark that executes on import.
Always pass an explicit test directory.

Test files under `ISPRS/CrackSAM/tests/` mostly insert `ISPRS/CrackSAM` into `sys.path` themselves, but
`test_model.py` and `test_frangigraph_gate_workflow.py` do not — they only import `cracksam2` when a
sibling module ran first, or when pytest is invoked from `ISPRS/CrackSAM`.

## Architecture

### One algorithm, four code paths

Changes to the method usually need to be reflected in more than one place. These are deliberate copies,
not accidental duplication:

| Path | Runtime | Role |
|---|---|---|
| `src/frangi_fusion/` | NumPy/SciPy, CPU | Reference implementation and installable package (`frangi-fusion`, CLI `frangi-find-batch`) |
| `ISPRS/src/` | PyTorch, GPU | Maintained GPU implementation (`FrangiHessianGPU`, `extract_frangi_graph_gpu`); what the ISPRS notebooks and CrackSAM 2 actually call |
| `ISPRS/CrackSAM/cracksam2/frangi.py` | wrapper | Fixes CrackSAM 2 defaults over the GPU extractor and adds the SAM 2 image/prompt conversion boundary |
| `EUVIP/code/` | frozen | Standalone published artifact (mirrored to `Ayana-Inria/Frangi-EUVIP`); its `src/frangi_fusion/` is a snapshot tied to the paper and must not silently drift |

`test_k2_clean.py` / `test_k2_script.py` at the root are the historical Colab exports the GPU code came
from; `cracksam2/frangi.py` documents that lineage explicitly.

### Pipeline stages (shared across implementations)

1. **Per-modality, per-scale Hessian** with `sigma**2` normalization, eigenvalues ordered by `|λ|`,
   normalized by `max|λ2|` so `λ ∈ [-1,1]` (`hessian.py`, `frangi_hessian.py`).
2. **Tensor-level fusion**: modalities are combined as a weighted sum of *raw Hessians* per scale, and
   eigenvalues are recomputed from the fused operator. Responses are never averaged after the fact — this
   is the paper's central claim, so do not "simplify" it into response fusion.
3. **Candidate nodes** from a quantile of the cross-scale max `|λ2|`.
4. **Pairwise Frangi similarity** within radius `R`: elongation × curvature strength × angular alignment
   (`_sim_elong`, `_sim_strength`, `_sim_angle`), max over scales, penalized by pixel distance, then
   thresholded on edges (`τ_E`) and nodes (`τ_V`).
5. **Connectivity**: `K=1` is the plain pixel graph; `K=2` builds triangle (simplicial) cliques for 2D
   topological continuity (`triangle_connectivity_graph`).
6. **Component selection**: largest connected component for FIND (single structure of interest), otherwise
   components above a relative size.
7. **Graph → mask**, by one of two routes that coexist: the batch CLI clusters with a hand-written
   HDBSCAN over the sparse distance matrix (`clustering_sparse.py`), while the notebooks and the paper use
   MST + `extract_backbone_centrality` — an O(N) weighted betweenness computed by rooting the tree,
   accumulating subtree masses bottom-up, re-rooting at the centrality maximum, and pruning top-down.
8. **Evaluation**: `skeletonize_lee()` + `thicken()` on both prediction and ground truth, then Jaccard,
   Tversky, and Wasserstein (`ot.emd2`) — `metrics.py`.

Parameter sets are dataset-specific and are recorded, with their provenance, in
`ISPRS/implementation_notes.md` §4 and `EUVIP/README.md` (the paper's FIND values differ from the CrackSAM
adaptation's, and that difference is load-bearing for the negative results below).

### Data conventions

`ISPRS/implementation_notes.md` is the authoritative reference; the traps that break results silently:

- **Range/thermal images are JET-colormapped.** Decode them by nearest-neighbour lookup against the
  matplotlib `jet` palette (`recover_scalar_from_cmap`, `decode_jet_to_grayscale`). A standard grayscale
  conversion makes mid-range green brighter than maximum red and corrupts the Hessian.
- **Polarity must be aligned before fusion**: visible cracks are dark ridges, thermal cracks are hot, so
  decoded thermal is inverted (`255 - x`).
- **FIND pairing** matches modalities and labels by the *last numeric run* in the filename with zero
  padding stripped (`im00215.bmp` ↔ `im215.png`), via `auto_discover_find_structure` /
  `load_modalities_and_gt_by_index`.
- **VT-GraF**: the `Fissure 2` folder contains files prefixed `fissure6`. Ground truth lives in the alpha
  channel, with a `< 127` grayscale fallback, and resizing uses nearest-neighbour only.
- Datasets, checkpoints, prompt caches, and run artifacts are gitignored and expected to be local.

### `ISPRS/CrackSAM/` — the SAM 2 track

Self-contained experiment with its own contracts and its own documentation set. Read
`ISPRS/CrackSAM/README.md` and `docs/01_EXPERIMENTAL_QUESTION.md` before changing anything here.

- `cracksam2/` is the active core: `model.py` (SAM 2 Hiera-L with q/v-only LoRA, re-implementing the
  predictor path because the public one is wrapped in `torch.no_grad`), `residual.py` (the active
  `verified_local_v1` adapter, plus `legacy_raster_v1` kept for reproducibility), `evidence_selection.py`,
  `gating.py` (NumPy-only logistic gate, deliberately pickle-free), `graph_cache.py` / `graph_types.py`
  (schema-v2 seven-channel raster cache), `oof.py` (fold 4 is reserved for gate calibration and must never
  be trained on).
- Five CLIs stay at `ISPRS/CrackSAM/` root to preserve import paths and experimental contracts:
  `prepare_cracksam2_data.py`, `precompute_frangi_prompts.py`, `train_sam2.py`, `evaluate_sam2.py`,
  `compute_exact_wasserstein.py`. `workflows/*.sh` orchestrate them and resolve paths from the CrackSAM
  root regardless of cwd.
- Configured through environment variables: `CRACKSAM2_DATA_ROOT`, `CRACKSAM2_PROMPT_ROOT`,
  `CRACKSAM2_ARTIFACT_ROOT`, `SAM2_CHECKPOINT`, `BASELINE_CHECKPOINT`, `FRANGIGRAPH_RUN_ROOT`.
  Install `requirements-sam2.txt` with `SAM2_BUILD_CUDA=0` and `--no-build-isolation`.
- Runs are resumable across Spot preemption: append-only journals, atomic rebuilds, and a
  `workflow_contract.json` that freezes selector parameters. The active contract is `schema_version=3`;
  a schema-2 run root cannot be resumed under it.
- The headline result is **negative and must stay stated as such**: the historical dense
  `mask_input` prompt lost IoU, and the map that was injected was `node_sim_max` — MST, components, and
  centrality were never used. The current track (FrangiGraph-SelectiveResidual) is a prototype whose
  shadow-robustness is an explicit hypothesis, not a demonstrated result.
- `results/<date>_<slug>/` directories are dated, immutable records. Absolute VM paths inside their JSON
  are provenance traces — do not rewrite them, and do not retro-edit published numbers.

### `gcp-migration/` — Spot VM control

`agents.md` (French) states the operating rules and takes precedence: always arm a guest shutdown
(`blackwell_preflight.sh --arm-shutdown 240`), always finish with `./gcp-migration/stop_and_verify.sh` and
confirm `TERMINATED`, and make training resumable from checkpoints because Spot VMs are preempted without
warning. `check_capacity.sh` and `check_quotas.sh` are read-only; `deploy.sh`, `start_and_verify.sh`, and
`stop_and_verify.sh` mutate cloud state and each require explicit confirmation. Only one G4 may run at a
time across this project and E-HGP.

## Conventions

- **Language follows the file.** The reference package, root `README.md`, and `ISPRS/implementation_notes.md`
  are English; `agents.md`, `EUVIP/README.md`, `gcp-migration/README.md`, and everything under
  `ISPRS/CrackSAM/docs|workflows|results` is French, including inline comments in the older algorithm code.
  Match what surrounds you.
- **Notebooks are generated, not hand-edited.** `build_notebook.py` / `build_vt_graf_notebook.py` emit the
  `.ipynb` from Python string cells; edit the builder and regenerate.
- **Documentation distinguishes executed results from hypotheses**, names variants canonically
  (`baseline_sam2_lora`, `frangi_dense_prompt_sam2_lora`, `frangi_graph_residual_sam2`), and reports
  confidence intervals for paired deltas. Preserve that discipline in any doc you touch.
- Root-level `main.tex`, `Manuscrit_de_these_LouisHauseux.pdf`, and the `EUVIP/`, `ISPRS/LaTeX/` sources are
  paper and thesis material; LaTeX build products (`*.aux`, `*.bbl`, `*.pdf`) are not committed.
