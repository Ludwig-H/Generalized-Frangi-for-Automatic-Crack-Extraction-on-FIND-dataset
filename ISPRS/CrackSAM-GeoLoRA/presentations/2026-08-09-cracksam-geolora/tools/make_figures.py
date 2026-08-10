#!/usr/bin/env python3
"""Prépare les images de la présentation. Aucune courbe, aucun graphique.

Trois sources, trois traitements :

* les planches par cas versionnées dans ``figures/generated/`` sont **découpées**
  en panneaux individuels, afin que les titres et les scores soient composés par
  Beamer et non par matplotlib — la typographie reste celle du gabarit Inria ;
* les cinq cas synthétiques qui valident la métrique tolérante sont **rendus
  comme images** (aucun axe, aucune légende), et leurs scores sont recalculés
  avec le code de mesure lui-même, puis imprimés pour être recopiés dans
  ``main.tex`` ;
* les figures du papier CrackSAM, archivées avec la présentation de juillet, sont
  **découpées de la même façon** pour les deux planches d'ouverture : leurs
  en-têtes de colonnes sont donc, eux aussi, recomposés par Beamer.

Le découpage ne suppose aucune géométrie fixe : les panneaux sont localisés par
les gouttières blanches de la planche, ce qui reste correct même si le titre
change de longueur d'une planche à l'autre.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PRESENTATION = HERE.parent
GEOLORA = PRESENTATION.parents[1]
SOURCE = GEOLORA / "figures" / "generated"
OUTPUT = PRESENTATION / "figures"

#: Figures du papier CrackSAM (Ge *et al.*, 2024), telles qu'extraites du PDF et
#: archivées avec la présentation Inria–Cerema du 10 juillet 2026.
PAPER = (
    GEOLORA.parent
    / "CrackSAM"
    / "reference"
    / "presentations"
    / "2026-07-10-inria-cerema"
    / "source"
    / "imgs"
)

#: Noms des huit panneaux d'une planche, dans l'ordre de lecture.
PANELS = ("image", "gt", "ref", "var", "frangi", "ofs", "ofa", "phase")

#: Planches retenues, celles que la présentation montre. Le préfixe dit la paire
#: comparée : ``b2t`` baseline vers tol3, ``t2g`` tol3 vers geo_tol3, ``b2g``
#: baseline vers geo (famille clDice).
CASES: dict[str, str] = {
    # baseline -> tol3 : ce que la dilatation gagne, et ce qu'elle coûte
    "b2t_gain_ct6774": "case_baseline_vs_tol3_reussite_04_cracktree200_6774.jpg",
    "b2t_loss_gaps552": "case_baseline_vs_tol3_echec_02_GAPS384_train_0552_1_641.jpg",
    "b2t_loss_crack171336": "case_baseline_vs_tol3_echec_00_CRACK500_20160405_171336_641_361.jpg",
    "b2t_loss_riss": "case_baseline_vs_tol3_echec_01_Rissbilder_for_Florian_9S6A2817_67_221_2558_2411.jpg",
    # tol3 -> geo_tol3 : les deux plus grands gains de la géométrie
    "t2g_gain_crack171336": "case_tol3_vs_geo_tol3_reussite_05_CRACK500_20160405_171336_641_361.jpg",
    "t2g_gain_riss": "case_tol3_vs_geo_tol3_reussite_04_Rissbilder_for_Florian_9S6A2817_67_221_2558_2411.jpg",
}

#: Vert = vrai positif, rouge = faux positif, bleu = manqué. Mêmes teintes que
#: ``scripts/04_figures.py``, pour que les vignettes synthétiques se lisent avec
#: le même code couleur que les planches réelles.
TRUE_POSITIVE = (0.15, 0.85, 0.25)
FALSE_POSITIVE = (0.90, 0.15, 0.15)
MISSED = (0.15, 0.45, 0.95)


def _load_metric():
    """Importe ``tolerant_scores`` depuis le script de mesure, tel quel."""

    path = GEOLORA / "scripts" / "05_tolerant_iou.py"
    spec = importlib.util.spec_from_file_location("tolerant_iou", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tolerant_iou"] = module
    spec.loader.exec_module(module)
    return module


def _bands(flags: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= minimum:
                out.append((start, index))
            start = None
    if start is not None and len(flags) - start >= minimum:
        out.append((start, len(flags)))
    return out


def split_panels(path: Path) -> list[Image.Image]:
    """Découpe une planche 2x4 en huit panneaux, par ses gouttières blanches."""

    image = Image.open(path).convert("RGB")
    content = np.asarray(image).astype(np.int16).min(axis=2) < 250
    rows = _bands(content.mean(axis=1) > 0.5, 60)
    if len(rows) != 2:
        raise RuntimeError(f"{path.name} : {len(rows)} bandes de lignes au lieu de 2")
    panels: list[Image.Image] = []
    # Retrait de quelques pixels sur chaque bord : il enlève le cadre des axes
    # et la ligne de titre de la rangée suivante, qui touche presque le bas des
    # panneaux du haut. Les vignettes reçoivent leur propre cadre dans Beamer.
    inset = 5
    for top, bottom in rows:
        columns = _bands(content[top:bottom].mean(axis=0) > 0.5, 60)
        if len(columns) != 4:
            raise RuntimeError(
                f"{path.name} : {len(columns)} bandes de colonnes au lieu de 4"
            )
        panels.extend(
            image.crop((left + inset, top + inset, right - inset, bottom - inset))
            for left, right in columns
        )
    return panels


#: Vignettes tirées des figures du papier CrackSAM, pour les deux planches
#: d'ouverture. Chaque entrée nomme la planche source, la rangée voulue et les
#: colonnes retenues — indices dans la grille détectée, en-tête et colonne des
#: étiquettes ``(a)``--``(h)`` exclues, qui tombent hors des gouttières.
#:
#: ``extracted_p10_0`` — Road420, en zéro-shot. Colonnes : ``Images``,
#: ``Ground Truth``, ``LoRA``, ``SegFormer``, ``MobileNet``, ``ResNet-50``,
#: ``Swin-B``, ``VGG-UNet``. On garde ``LoRA``, la variante dont notre baseline
#: descend, et trois concurrents ; les deux rangées retenues sont celles où la
#: chaussée est masquée par un reflet, puis par une ombre portée.
PAPER_COLUMNS = {
    "image": 0,
    "gt": 1,
    "sam": 2,
    "segformer": 3,
    "mobilenet": 4,
    "resnet": 5,
}

PAPER_ROWS: dict[str, tuple[str, int, dict[str, int]]] = {
    "cs_zs_reflet": ("extracted_p10_0.jpg", 2, PAPER_COLUMNS),
    "cs_zs_ombre": ("extracted_p10_0.jpg", 4, PAPER_COLUMNS),
}


def split_grid(path: Path) -> list[list[Image.Image]]:
    """Découpe une planche en grille, par ses gouttières blanches.

    Même principe que :func:`split_panels`, sans nombre de lignes ni de colonnes
    imposé : les en-têtes de colonnes et les étiquettes de rangées du papier sont
    du texte clairsemé, ils tombent donc hors des bandes détectées et ne sont pas
    repris — c'est voulu, Beamer les recompose.
    """

    image = Image.open(path).convert("RGB")
    content = np.asarray(image).astype(np.int16).min(axis=2) < 250
    rows = _bands(content.mean(axis=1) > 0.5, 60)
    columns = _bands(content.mean(axis=0) > 0.5, 60)
    if not rows or not columns:
        raise RuntimeError(f"{path.name} : aucune grille détectée")
    inset = 4
    return [
        [
            image.crop((left + inset, top + inset, right - inset, bottom - inset))
            for left, right in columns
        ]
        for top, bottom in rows
    ]


def trim(path: Path, margin: int = 6) -> Image.Image:
    """Retire les marges blanches d'une figure, en gardant ``margin`` pixels."""

    image = Image.open(path).convert("RGB")
    content = np.asarray(image).astype(np.int16).min(axis=2) < 250
    rows = np.flatnonzero(content.any(axis=1))
    columns = np.flatnonzero(content.any(axis=0))
    if not len(rows) or not len(columns):
        return image
    return image.crop(
        (
            max(int(columns[0]) - margin, 0),
            max(int(rows[0]) - margin, 0),
            min(int(columns[-1]) + 1 + margin, image.width),
            min(int(rows[-1]) + 1 + margin, image.height),
        )
    )


def agreement(prediction: np.ndarray, truth: np.ndarray) -> Image.Image:
    canvas = np.zeros((*truth.shape, 3), dtype=np.float32) + 0.10
    canvas[prediction & truth] = TRUE_POSITIVE
    canvas[prediction & ~truth] = FALSE_POSITIVE
    canvas[~prediction & truth] = MISSED
    array = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array).resize((256, 256), Image.NEAREST)


def synthetic_cases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Une ligne verticale de 3 px sur 48, et quatre façons de la rater.

    Les deux ruptures sont volontairement de longueurs différentes. La tolérance
    est une distance d'appariement : elle pardonne donc aussi une rupture, mais
    seulement tant que celle-ci est plus courte que ``2k``. Le dire est
    nécessaire, sinon la métrique promet une garantie topologique qu'elle n'a
    pas.
    """

    size = 64
    truth = np.zeros((size, size), dtype=bool)
    truth[8:56, 30:33] = True

    shifted = np.zeros_like(truth)
    shifted[8:56, 32:35] = True

    wide = np.zeros_like(truth)
    wide[8:56, 28:35] = True

    short_break = truth.copy()
    short_break[30:34, :] = False  # 4 px

    long_break = truth.copy()
    long_break[22:42, :] = False  # 20 px

    return {
        "synth_perfect": (truth, truth),
        "synth_shifted": (shifted, truth),
        "synth_wide": (wide, truth),
        "synth_break4": (short_break, truth),
        "synth_break20": (long_break, truth),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for slug, filename in CASES.items():
        panels = split_panels(SOURCE / filename)
        for name, panel in zip(PANELS, panels):
            panel.save(OUTPUT / f"{slug}_{name}.jpg", quality=92, subsampling=0)
        print(f"découpé {slug} ({filename})")

    grids: dict[str, list[list[Image.Image]]] = {}
    for slug, (filename, row, columns) in PAPER_ROWS.items():
        if filename not in grids:
            grids[filename] = split_grid(PAPER / filename)
        grid = grids[filename]
        for name, column in columns.items():
            grid[row][column].save(
                OUTPUT / f"{slug}_{name}.jpg", quality=92, subsampling=0
            )
        print(f"découpé {slug} ({filename}, rangée {row})")

    trim(PAPER / "extracted_p9_0.jpg").save(OUTPUT / "cs_flou_courbe.jpg", quality=94)
    print("détouré cs_flou_courbe (extracted_p9_0.jpg)")

    metric = _load_metric()
    radii = (0, 1, 2, 3, 5, 8)
    print("\ncas synthétiques — IoU tampon recalculée par scripts/05_tolerant_iou.py")
    print("cas            " + "".join(f"{f'k={k}':>9}" for k in radii))
    for slug, (prediction, truth) in synthetic_cases().items():
        agreement(prediction, truth).save(OUTPUT / f"{slug}.png")
        scores = [
            metric.tolerant_scores(prediction, truth, k)["iou_buffered"] for k in radii
        ]
        print(f"{slug:<15}" + "".join(f"{value:>9.3f}" for value in scores))

    try:
        import torch

        from geolora.losses import tolerant_loss  # noqa: E402
    except ImportError:
        print("\ntorch absent : perte tolérante non revérifiée")
        return 0

    print("\ncas synthétiques — perte tolérante (1 - F1 tolérant)")
    print("cas            " + "".join(f"{f'k={k}':>9}" for k in radii))
    for slug, (prediction, truth) in synthetic_cases().items():
        values = []
        for radius in radii:
            values.append(
                float(
                    tolerant_loss(
                        torch.from_numpy(prediction[None, None]).float(),
                        torch.from_numpy(truth[None, None]).float(),
                        radius,
                    )
                )
            )
        print(f"{slug:<15}" + "".join(f"{value:>9.4f}" for value in values))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(GEOLORA))
    raise SystemExit(main())
