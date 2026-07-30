# Frangi-EUVIP

Code accompanying the accepted EUVIP 2026 paper:

> Louis Hauseux, Raphaël Antoine, Philippe Foucher, Pierre Charbonnier, and
> Josiane Zerubia, “Multi-Modal, Training-Free Crack Extraction via Generalized
> Frangi Graph,” European Workshop on Visual Information Processing (EUVIP),
> 2026.

The method builds a sparse graph from multi-scale Hessian features, fuses
aligned modalities at the operator level, and extracts a crack skeleton through
a minimum spanning tree and similarity-weighted tree centrality.

## Contents

| Paper experiment | Code |
| --- | --- |
| FIND clean benchmark, modality ablation, synthetic noise, CrackSegDiff comparison, and Palais des Papes | `notebooks/FIND_Palais_Noise_Experiments.ipynb` |
| Vaches Noires and U-Net transfer-learning comparison | `notebooks/Vaches_Noires_Experiments.ipynb` |
| Unchanged historical Colab export for the Vaches Noires results | `legacy/Vaches_Noires_original_colab_export.py` |
| Reusable Hessian, graph, MST, centrality, visualization, and metric code | `src/frangi_fusion/` |
| Optional command-line batch helpers | `scripts/` |

The converted Vaches Noires notebook is provided for convenience. The file in
`legacy/` preserves the original experimental artifact byte for byte.

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/Ayana-Inria/Frangi-EUVIP.git
cd Frangi-EUVIP
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the closest reproduction of the paper, open the notebooks in Google Colab
and run the cells in order. The main notebook can clone and install this
repository automatically.

## Data

- FIND is public: [Fused Image dataset for convolutional neural Network-based
  crack Detection (FIND)](https://doi.org/10.5281/zenodo.6383044).
- The non-public geological inputs and the precomputed baseline outputs used in
  the paper are available in the
  [companion data folder](https://drive.google.com/drive/folders/1iC7QrdB2vZmaunjjPrIcu9r5wMeBPMlk).

The FIND/Palais notebook documents the Google Drive locations expected for the
CrackSegDiff masks. The Vaches Noires notebook expects the historical
`MyDrive/crack_detection/` layout. The baseline training repositories and model
checkpoints are not redistributed here; the supplied evaluation code consumes
their precomputed predictions.

## Citation

Please cite the paper when using this code:

```bibtex
@inproceedings{HauseuxEUVIP2026,
  title     = {Multi-Modal, Training-Free Crack Extraction via Generalized Frangi Graph},
  author    = {Hauseux, Louis and Antoine, Raphaël and Foucher, Philippe and Charbonnier, Pierre and Zerubia, Josiane},
  booktitle = {European Workshop on Visual Information Processing (EUVIP)},
  year      = {2026},
  note      = {Accepted}
}
```

The entry can be completed with pages and DOI after publication.

## License

This code is released under the GNU General Public License v3.0 only
(`GPL-3.0-only`). See `LICENSE`. The datasets and third-party predictions retain
their own terms.
