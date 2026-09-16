# EUVIP 2026 oral presentation

[**Presentation PDF**](Presentation_EUVIP_2026_Hauseux.pdf) · [Beamer source](main.tex)

[![Slide overview](apercu.png)](Presentation_EUVIP_2026_Hauseux.pdf)

English, 16:9, **10 content slides**, plus the title, three section dividers and two bibliography pages: **16 PDF pages in total**. Shortened version, with 10 pt body text. Prepared for the 10-minute oral presentation of paper 81, with 2 minutes of questions, on 29 September 2026.

## Content

1. Crack networks and the training-free extraction problem.
2. Hessian geometry and the classical Frangi filter.
3. Pairwise orientation alignment: the Frangi graph.
4. Multimodal fusion of normalized Hessians.
5. Graph reduction, minimum spanning trees and weighted centrality.
6. Clean FIND results.
7. FIND under controlled synthetic noise.
8. Geological transfer: Vaches Noires and Palais des Papes.
9. Granular backgrounds and higher-order connectivity, K=2.
10. Conclusion & perspective: hierarchy-guided SAM/CrackSAM.

The scientific baseline is [the EUVIP camera-ready source](../LaTeX/main.tex). The clean-FIND table reports the best published graph results (63% IoU, 71% Tversky, 11 px Wasserstein; Table III), together with the published CrackSegDiff results (Table II). The separate modality-ablation panel is removed. Experimental figures are unchanged.

Citations use bracketed labels in the slide body or image captions and the complete reference at the bottom left. [references.tex](references.tex) supplies both the page notes and the bibliography. GRETSI 25, EUVIP 26 and ANS 26 are red. Page notes contain references only; image credits stay in the captions. The final slide uses blue for “Main steps” and the graph/hierarchy boxes, red for “Next steps” and the foundation-model box.

## Template and assets

The `theme/` folder contains the Inria 2024 Beamer theme. The title slide uses the **official Cerema horizontal logo in its original colors, on white**, downloaded from [Cerema's website](https://www.cerema.fr/themes/custom/uas_base/images/LogoCerema_horizontal.svg). `assets/cerema-official.svg` preserves the original file unchanged; `assets/cerema-official.pdf` is its vector conversion with an explicit white background. CairoSVG preserves the embedded CSS colors. The source URL and SVG SHA-256 are recorded in `assets/cerema-official.source.txt`. The former white recoloring and dark cartouche have been removed from this presentation. The poster is unchanged.

Experimental images are copied unchanged from `EUVIP/LaTeX/`; the VT-GraF illustration is credited to Cerema. `bootstrap_assets.py` can restore missing assets from its pinned source commit, checking their Git blob hashes, and restore the official Cerema artwork if missing. The build workflow commits those assets locally alongside the PDF, so the completed folder is self-contained. No font files are distributed; standard Latin Modern fonts are used.

## Build locally

Requires LuaLaTeX, latexmk, Beamer, TikZ, babel (English), lmodern and the LaTeX extra packages. Debian/Ubuntu packages:

```bash
sudo apt-get install latexmk texlive-luatex texlive-latex-extra texlive-fonts-recommended lmodern
cd EUVIP/presentation
make
```

The output is `Presentation_EUVIP_2026_Hauseux.pdf`. Auxiliary files stay in ignored `build/`. No shell escape is enabled. Restoring a missing Cerema vector PDF additionally requires CairoSVG (`python3-cairosvg`); the committed PDF needs no conversion or network access.

Optional checks, page renders, overview and a complete source archive:

```bash
sudo apt-get install python3-fitz python3-pil
make check PYTHON=/usr/bin/python3
```

`verify.py` checks the page/slide counts, aspect ratio, missing glyphs and unresolved references. It rejects box-overflow warnings and renders every page for visual inspection. The complete editable archive is written to `build/Presentation_EUVIP_2026_sources.zip`.

## GitHub build

[The dedicated workflow](../../.github/workflows/euvip-presentation.yml) builds changes to this presentation on `main` and can also be launched manually. It publishes only the generated presentation PDF, overview and restored presentation dependencies back to `main`, without force-pushing or creating a branch. If presentation sources changed during compilation, publication stops rather than replacing them with a stale build. The PDF, page renders, checks and source archive are also uploaded as a workflow artifact.

Paper/poster sources and research code are not modified by this workflow.
