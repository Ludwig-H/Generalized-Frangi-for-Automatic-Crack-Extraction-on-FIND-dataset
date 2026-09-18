# EUVIP 2026 oral presentation

[**Presentation PDF**](Presentation_EUVIP_2026_Hauseux.pdf) · [Beamer source](main.tex)

[![Slide overview](apercu.png)](Presentation_EUVIP_2026_Hauseux.pdf)

English, 16:9, **10 content slides**, plus the title, three section dividers and two bibliography pages: **16 PDF pages in total**. Shortened version, with 10 pt body text. Prepared for the 10-minute oral presentation of paper 81, with 2 minutes of questions, on 29 September 2026.

## Content

1. Crack networks and the training-free extraction problem.
2. Hessian geometry and the classical Frangi filter, reduced to shape and contrast.
3. Pairwise orientation alignment: our contribution, the Frangi graph.
4. Multimodal fusion of normalized Hessians.
5. Graph reduction, minimum spanning trees and weighted centrality, as a picture rather than as formulas.
6. Clean FIND results.
7. FIND under controlled synthetic noise.
8. Geological transfer: Vaches Noires and Palais des Papes.
9. Granular backgrounds and higher-order connectivity, K=2.
10. Conclusion & perspective: hierarchy-guided SAM/CrackSAM.

The scientific baseline is [the EUVIP camera-ready source](../LaTeX/main.tex). The clean-FIND table reports the best published graph results (63% IoU, 71% Tversky, 11 px Wasserstein; Table III), together with the published CrackSegDiff results (Table II), and a short sentence attributes the gap to supervision and possible overfitting. The separate modality-ablation panel is removed and the slides name only our best configuration. Experimental figures are unchanged.

Citations use bracketed labels in the slide body, in the table rows or in the image captions, and the reference itself at the bottom left, one reference per line. The bibliography slides print each address in full rather than hiding it behind a word. [references.tex](references.tex) supplies both the page notes and the bibliography: `\DeclareRef` holds the complete record used by the bibliography slides, and `\DeclareShort` the shortened form used by the page notes, so that every note fits on a single line. GRETSI 25, EUVIP 26 and ANS 26 are red.

Slide body text carries no terminal full stop, every caption, label and box starts with a capital, and the pipeline boxes all open with an imperative verb; the bottom-left notes and the bibliography keep ordinary punctuation. No slide carries a note in small print at the bottom: copyright credits and dataset citations sit in the image captions, side conditions belong to the equation or to the bullet that introduces them, and the bottom-left block holds references only. Where a slide needs a comment, as under the clean-FIND table, it is body text placed next to what it comments on. The page notes shorten a reference to keep it on one line, and mark any shortened title with an ellipsis. The final slide uses blue for “Main steps” and the graph/hierarchy boxes, red for “Next steps” and the foundation-model box.

## Template and assets

The title slide names the conference in full, European Workshop on Visual Information Processing, and keeps the five authors on one line; its text block is slightly wider than the theme default to allow it. The conclusion slide carries a QR code, drawn by the `qrcode` package as on the poster, pointing at [Ayana-Inria/Frangi-EUVIP](https://github.com/Ayana-Inria/Frangi-EUVIP), the repository already printed in the paper. The `theme/` folder contains the Inria 2024 Beamer theme. The title slide uses the **official Cerema horizontal logo in its original colors, on white**, downloaded from [Cerema's website](https://www.cerema.fr/themes/custom/uas_base/images/LogoCerema_horizontal.svg). `assets/cerema-official.svg` preserves the original file unchanged; `assets/cerema-official.pdf` is its vector conversion with an explicit white background. CairoSVG preserves the embedded CSS colors. The source URL and SVG SHA-256 are recorded in `assets/cerema-official.source.txt`. The former white recoloring and dark cartouche have been removed from this presentation. The poster is unchanged.

Experimental images are copied unchanged from `EUVIP/LaTeX/`; the VT-GraF illustration is credited to Cerema. The single exception is `assets/Resultat_1FIND_red.png`, a derivative of the unmodified `assets/Resultat_1FIND.png` in which the near-black skeleton overlay is recoloured to a legible red and widened by one pixel so that it survives the reduction to slide size; the extracted geometry is untouched, and deleting the derivative and pointing the two panels back at the original reverts it. [recolor_skeleton.py](recolor_skeleton.py) regenerates it and documents the transform. `bootstrap_assets.py` can restore missing assets from its pinned source commit, checking their Git blob hashes, and restore the official Cerema artwork if missing. The build workflow commits those assets locally alongside the PDF, so the completed folder is self-contained. No font files are distributed; standard Latin Modern fonts are used.

## Build locally

Requires LuaLaTeX, latexmk, Beamer, TikZ, `qrcode`, babel (English), lmodern and the LaTeX extra packages. Debian/Ubuntu packages:

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
