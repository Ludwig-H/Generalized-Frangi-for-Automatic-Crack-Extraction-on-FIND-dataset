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

from thermal_residual.manifest import read_manifest  # noqa: E402
from thermal_residual.provenance import atomic_write_json  # noqa: E402
from thermal_residual.thermal_decode import (  # noqa: E402
    PALETTE_ERROR_WARNING,
    decode_thermal,
    load_rgb,
    naive_opencv_grayscale,
)

#: Seuil d'exclusion pré-enregistré, en pixels.
ALIGNMENT_THRESHOLD_PX = 3.0


def estimate_shift(reference: np.ndarray, moving: np.ndarray, radius: int = 8) -> tuple[float, float, float]:
    """Décalage entier maximisant la corrélation croisée normalisée.

    Recherche exhaustive dans ``[-radius, radius]²`` sur les gradients : le
    contraste absolu n'est pas comparable entre visible et thermique, mais les
    bords le sont. Retourne ``(dy, dx, corrélation)``.
    """

    from scipy import ndimage as ndi

    def edges(image: np.ndarray) -> np.ndarray:
        smoothed = ndi.gaussian_filter(image.astype(np.float32), 2.0)
        gy, gx = np.gradient(smoothed)
        magnitude = np.hypot(gy, gx)
        centred = magnitude - magnitude.mean()
        norm = float(np.linalg.norm(centred))
        return centred / norm if norm > 0 else centred

    left = edges(reference)
    right = edges(moving)
    best = (0.0, 0.0, -2.0)
    height, width = left.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0, y1 = max(0, dy), min(height, height + dy)
            x0, x1 = max(0, dx), min(width, width + dx)
            if y1 - y0 < height // 2 or x1 - x0 < width // 2:
                continue
            a = left[y0:y1, x0:x1]
            b = right[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
            score = float((a * b).sum())
            if score > best[2]:
                best = (float(dy), float(dx), score)
    return best


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
            dy, dx, score = estimate_shift(rgb.mean(axis=-1), decoding.normalized)
            row.update(
                {
                    "shift_dy": dy,
                    "shift_dx": dx,
                    "shift_norm": float(np.hypot(dy, dx)),
                    "shift_score": score,
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

    shifts = [row["shift_norm"] for row in rows if "shift_norm" in row]
    encodings = {row["encoding"] for row in rows}
    palette_errors = [row["palette_error_mean"] for row in rows]
    median_shift = float(np.median(shifts)) if shifts else float("nan")

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
        "alignment": {
            "sampled": len(shifts),
            "median_shift_px": median_shift,
            "p90_shift_px": float(np.percentile(shifts, 90)) if shifts else float("nan"),
            "threshold_px": ALIGNMENT_THRESHOLD_PX,
            "verdict": (
                "inconnu"
                if not shifts
                else ("accepté" if median_shift <= ALIGNMENT_THRESHOLD_PX else "REJETÉ")
            ),
        },
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
    if shifts:
        print(
            f"désalignement médian : {median_shift:.2f} px "
            f"→ {report['alignment']['verdict']} (seuil {ALIGNMENT_THRESHOLD_PX} px)"
        )
    return 0 if report["alignment"]["verdict"] != "REJETÉ" else 2


if __name__ == "__main__":
    raise SystemExit(main())
