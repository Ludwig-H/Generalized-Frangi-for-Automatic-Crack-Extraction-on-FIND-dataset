#!/usr/bin/env python3
"""Audite le décodage thermique et le recalage, **avant** tout entraînement.

Trois sorties, toutes exigées par la spécification §12.2 :

``report.json``
    type de décodage retenu par image, chroma, erreur de palette, percentiles,
    et le désalignement estimé entre RGB et thermique.
``gallery.png``
    seize exemples : RGB, thermique brute, décodage correct, décodage naïf
    interdit, et la différence des deux. C'est la planche qui rend le piège JET
    visible plutôt que théorique.
``pairing_errors.csv``
    tout ce qui a échoué, ligne par ligne.

Le seuil d'exclusion est pré-enregistré : **désalignement médian > 3 px**.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_residual.data import load_mask  # noqa: E402
from thermal_residual.manifest import read_manifest  # noqa: E402
from thermal_residual.registration import (  # noqa: E402
    best_contrast_shift,
    summarize,
    verdict,
)
from thermal_residual.provenance import atomic_write_json  # noqa: E402
from thermal_residual.thermal_decode import (  # noqa: E402
    PALETTE_ERROR_WARNING,
    decode_thermal,
    load_rgb,
    naive_opencv_grayscale,
)

#: Seuil d'exclusion pré-enregistré, en pixels.
ALIGNMENT_THRESHOLD_PX = 3.0

#: Rayon de recherche du décalage. Toute estimation qui l'atteint est marquée
#: « saturée » : elle mesure la fenêtre, pas les données.
ALIGNMENT_RADIUS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--thermal-encoding", default="auto", choices=("auto", "grayscale", "jet"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gallery-size", type=int, default=16)
    parser.add_argument(
        "--alignment-samples",
        type=int,
        default=30,
        help="nombre de paires sur lesquelles estimer le recalage (0 pour désactiver)",
    )
    return parser.parse_args()


def build_gallery(records: list[dict], destination: Path, count: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = records[: max(1, count)]
    columns = 5
    figure, axes = plt.subplots(len(chosen), columns, figsize=(3.0 * columns, 2.3 * len(chosen)))
    axes = np.atleast_2d(axes)
    titles = (
        "RGB",
        "thermique (brute)",
        "décodage correct",
        "gris naïf — INTERDIT",
        "|correct − naïf|",
    )
    for row, record in enumerate(chosen):
        for column, (image, cmap) in enumerate(record["panels"]):
            axis = axes[row, column]
            axis.imshow(image, cmap=cmap, vmin=None if cmap is None else 0.0, vmax=None if cmap is None else 1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(titles[column], fontsize=9)
        axes[row, 0].set_ylabel(record["sample_id"], fontsize=7)
    figure.suptitle(
        "Décodage thermique — le panneau 4 est la conversion interdite, le 5 mesure l'écart",
        fontsize=11,
    )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=110)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    samples = read_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    errors: list[dict] = []
    gallery: list[dict] = []
    thermal_shifts: list = []
    control_shifts: list = []

    for index, sample in enumerate(samples):
        try:
            rgb = load_rgb(sample.rgb_path)
            thermal_rgb = load_rgb(sample.thermal_path)
            decoding = decode_thermal(thermal_rgb, encoding=args.thermal_encoding)
        except Exception as error:  # noqa: BLE001 — l'audit doit survivre à tout
            errors.append({"sample_id": sample.sample_id, "stage": "decode", "message": str(error)})
            continue

        naive = naive_opencv_grayscale(thermal_rgb)
        divergence = float(np.abs(decoding.scalar - naive).mean())

        row = {
            "sample_id": sample.sample_id,
            "height": sample.height,
            "width": sample.width,
            "naive_grayscale_divergence": divergence,
            **decoding.to_json(),
        }
        row.update({f"percentile_{k}": v for k, v in decoding.percentiles.items()})
        row.pop("percentiles", None)

        if args.alignment_samples and index < args.alignment_samples:
            mask = load_mask(sample.mask_path) > 0.5
            if mask.any():
                thermal_shift = best_contrast_shift(decoding.normalized, mask, ALIGNMENT_RADIUS)
                control_shift = best_contrast_shift(rgb.mean(axis=-1), mask, ALIGNMENT_RADIUS)
                thermal_shifts.append(thermal_shift)
                control_shifts.append(control_shift)
                row.update(
                    {
                        "shift_dy": thermal_shift.dy,
                        "shift_dx": thermal_shift.dx,
                        "shift_norm": thermal_shift.norm,
                        "shift_saturated": float(thermal_shift.saturated),
                        "control_shift_norm": control_shift.norm,
                    }
                )

        rows.append(row)
        if len(gallery) < args.gallery_size:
            gallery.append(
                {
                    "sample_id": sample.sample_id,
                    "panels": [
                        (rgb, None),
                        (thermal_rgb, None),
                        (decoding.normalized, "inferno"),
                        (naive, "inferno"),
                        (np.abs(decoding.scalar - naive), "magma"),
                    ],
                }
            )

    if gallery:
        build_gallery(gallery, args.output / "gallery.png", args.gallery_size)

    with open(args.output / "pairing_errors.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "stage", "message"])
        writer.writeheader()
        writer.writerows(errors)

    alignment = (
        verdict(
            summarize("thermique", thermal_shifts),
            summarize("rgb (contrôle)", control_shifts),
            threshold_px=ALIGNMENT_THRESHOLD_PX,
        )
        if thermal_shifts
        else {"decision": "inconnu", "control_is_clean": False}
    )
    encodings = {row["encoding"] for row in rows}
    palette_errors = [row["palette_error_mean"] for row in rows]
    report = {
        "manifest": str(args.manifest),
        "requested_encoding": args.thermal_encoding,
        "count": len(rows),
        "errors": len(errors),
        "encodings_seen": sorted(encodings),
        "palette_error_mean": float(np.mean(palette_errors)) if palette_errors else 0.0,
        "palette_error_max": float(np.max(palette_errors)) if palette_errors else 0.0,
        "palette_error_warning_threshold": PALETTE_ERROR_WARNING,
        "naive_grayscale_divergence_mean": float(
            np.mean([row["naive_grayscale_divergence"] for row in rows])
        )
        if rows
        else 0.0,
        "alignment": alignment,
        "rows": rows,
    }
    atomic_write_json(args.output / "report.json", report)

    print(f"{len(rows)} images auditées, {len(errors)} erreur(s)")
    print(f"décodages retenus : {sorted(encodings)}")
    print(f"erreur de palette moyenne : {report['palette_error_mean']:.4f} (alerte au-delà de {PALETTE_ERROR_WARNING})")
    print(
        f"écart au gris naïf : {report['naive_grayscale_divergence_mean']:.4f} "
        "— non nul signifie que la conversion standard aurait corrompu la Hessienne"
    )
    if thermal_shifts:
        control = alignment["control"]
        thermal = alignment["thermal"]
        print(
            f"contrôle RGB       : médiane {control['median_shift_px']:.1f} px, "
            f"{control['fraction_exact']:.0%} exactement à 0 px "
            f"→ estimateur {'fiable' if alignment['control_is_clean'] else 'NON FIABLE'}"
        )
        print(
            f"thermique          : médiane {thermal['median_shift_px']:.1f} px, "
            f"{thermal['fraction_within_2px']:.0%} à moins de 2 px, "
            f"saturation {thermal['saturation_rate']:.0%}"
        )
        print(f"verdict de recalage : {alignment['decision']} (seuil {ALIGNMENT_THRESHOLD_PX} px)")
    return 0 if alignment["decision"] != "REJETÉ" else 2


if __name__ == "__main__":
    raise SystemExit(main())
