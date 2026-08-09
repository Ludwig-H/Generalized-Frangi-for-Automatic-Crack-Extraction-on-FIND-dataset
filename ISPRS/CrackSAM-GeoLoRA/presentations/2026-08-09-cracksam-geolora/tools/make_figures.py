#!/usr/bin/env python3
"""Prépare les images de la présentation. Aucune courbe, aucun graphique.

Deux sources, deux traitements :

* les planches par cas versionnées dans ``figures/generated/`` sont **découpées**
  en panneaux individuels, afin que les titres et les scores soient composés par
  Beamer et non par matplotlib — la typographie reste celle du gabarit Inria ;
* les cinq cas synthétiques qui valident la métrique tolérante sont **rendus
  comme images** (aucun axe, aucune légende), et leurs scores sont recalculés
  avec le code de mesure lui-même, puis imprimés pour être recopiés dans
  ``main.tex``.

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

#: Noms des huit panneaux d'une planche, dans l'ordre de lecture.
PANELS = ("image", "gt", "ref", "var", "frangi", "ofs", "ofa", "phase")

#: Planches retenues, celles que la présentation montre. Le préfixe dit la paire
#: comparée : ``b2t`` baseline vers tol3, ``t2g`` tol3 vers geo_tol3, ``b2g``
#: baseline vers geo (famille clDice).
CASES: dict[str, str] = {
    # baseline -> tol3 : ce que la perte tolérante change
    "b2t_gain_ct6774": "case_baseline_vs_tol3_reussite_04_cracktree200_6774.jpg",
    "b2t_loss_crack171336": "case_baseline_vs_tol3_echec_00_CRACK500_20160405_171336_641_361.jpg",
    "b2t_loss_riss": "case_baseline_vs_tol3_echec_01_Rissbilder_for_Florian_9S6A2817_67_221_2558_2411.jpg",
    # tol3 -> geo_tol3 : ce que la géométrie change
    "t2g_gain_riss": "case_tol3_vs_geo_tol3_reussite_04_Rissbilder_for_Florian_9S6A2817_67_221_2558_2411.jpg",
    "t2g_gain_crack171336": "case_tol3_vs_geo_tol3_reussite_05_CRACK500_20160405_171336_641_361.jpg",
    "t2g_loss_gaps608": "case_tol3_vs_geo_tol3_echec_00_GAPS384_train_0608_1_1.jpg",
    # baseline -> geo : la famille clDice
    "b2g_gain_ct6774": "case_reussite_03_cracktree200_6774.jpg",
    "b2g_loss_cfd080": "case_echec_02_CFD_080.jpg",
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
