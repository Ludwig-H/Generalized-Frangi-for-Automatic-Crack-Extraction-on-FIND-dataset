#!/usr/bin/env python3
"""Prépare les crops légers utilisés par la présentation.

Les figures sources restent les rapports et la présentation de référence du
dépôt. Les bornes des crops sont volontairement explicites afin que le PDF soit
reproductible sans dépendre des jeux de données locaux non versionnés.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "figures"


def copy_asset(source: Path, target_name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, OUT / target_name)


def crop(source: Path, box: tuple[int, int, int, int], target_name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        panel = image.crop(box)
        panel.save(OUT / target_name, quality=95, optimize=True)


def crop_case_panels(source: Path, prefix: str) -> None:
    # Les rapports qualitatifs ont huit panneaux carrés sur un canevas 2160x360.
    # On conserve l'entrée, le GT, le prompt Frangi et les deux prédictions.
    bounds = {
        "input": (40, 100, 258, 318),
        "gt": (309, 100, 527, 318),
        "frangi": (844, 100, 1062, 318),
        "baseline": (1112, 100, 1330, 318),
        "guided": (1647, 100, 1865, 318),
    }
    for label, box in bounds.items():
        crop(source, box, f"{prefix}_{label}.jpg")


def main() -> None:
    reference = (
        ROOT
        / "ISPRS/CrackSAM/reference/presentations/2026-07-10-inria-cerema"
    )
    report = ROOT / "ISPRS/CrackSAM/results/frangi_milestone_report"

    # Illustrations de CrackSAM extraites du papier dans la présentation source.
    source_imgs = reference / "source/imgs"
    copy_asset(source_imgs / "sam_model_diagram.png", "sam_model_diagram.png")
    copy_asset(source_imgs / "extracted_p9_0.jpg", "cracksam_noise_curve.jpg")

    # Frangi-similarité et lecture graphe sur la fissure 2 de VT-GraF.
    comparison = reference / "figures/Fissure_2_comparison.png"
    crop(comparison, (5, 65, 585, 425), "find_visible.png")
    crop(comparison, (5, 505, 585, 870), "find_similarity.png")
    crop(comparison, (600, 505, 1185, 870), "find_centrality.png")
    crop(comparison, (5, 945, 585, 1310), "find_hessian_sigma30.png")

    # Cas appariés du rapport SAM 2. La sélection expose volontairement les
    # gains, les pertes et les cas stables/difficiles sur plusieurs domaines.
    cases = report / "figures/cases"
    crop_case_panels(
        cases / "road420/gain_frangi__2023_11_01_20_33_IMG_6353.jpg.jpg",
        "shadow_gain",
    )
    crop_case_panels(
        cases / "road420/gain_baseline__2023_10_30_16_44_IMG_6033.jpg.jpg",
        "good_detector_bad_prompt",
    )
    crop_case_panels(
        cases / "khanhha_original/gain_frangi__Sylvie_Chambon_319.jpg.jpg",
        "original_gain",
    )
    crop_case_panels(
        cases
        / "khanhha_noisy2/gain_frangi__Volker_DSC01646_226_19_1273_1645.jpg.jpg",
        "noisy2_gain",
    )
    crop_case_panels(
        cases
        / "khanhha_noisy1/gain_frangi__CRACK500_20160329_093924_1921_721.jpg.jpg",
        "noisy1_gain",
    )
    crop_case_panels(
        cases / "facade390/gain_frangi__DJ_Wall_66.JPG.jpg",
        "facade_gain",
    )
    crop_case_panels(
        cases
        / "khanhha_noisy2/gain_baseline__CRACK500_20160308_073532_1_361.jpg.jpg",
        "noisy2_loss",
    )
    crop_case_panels(
        cases / "facade390/gain_baseline__DJ_Wall_231.JPG.jpg",
        "facade_loss",
    )
    crop_case_panels(
        cases / "road420/both_good__2023_11_05_21_38_IMG_6516.jpg.jpg",
        "road_both_good",
    )
    crop_case_panels(
        cases / "facade390/both_weak__DJ_Wall_343.JPG.jpg",
        "facade_both_weak",
    )
    copy_asset(
        report / "figures/paired_delta_iou_distributions.png",
        "paired_delta_iou_distributions.png",
    )
    copy_asset(
        report / "figures/delta_iou_heatmap.png",
        "delta_iou_heatmap.png",
    )


if __name__ == "__main__":
    main()
