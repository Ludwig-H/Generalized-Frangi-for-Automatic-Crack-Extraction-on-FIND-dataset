"""Recolour the extracted-skeleton overlays so that they survive the reduction to slide size.

Three result panels draw the extracted skeleton in a very dark red over a grey intensity
image. At the width the slides use, roughly two centimetres, the one-pixel line is both
sub-pixel and indistinguishable from the crack underneath. For each of them this script
writes a `_red` companion in which the pixels that already carry the overlay are remapped
to a legible red and widened by one pixel. The neutral background is untouched and no
geometry is added or removed: the overlay is located by its own redness, and the widening
is a single 3x3 maximum filter on the resulting coverage map. The originals are never
modified.

The threshold is derived from each image rather than fixed, because two of the three
backgrounds carry a warm cast of their own: a fixed floor tinted them pink instead of
picking out the line. The floor is the median redness of the image plus six median absolute
deviations, so it sits just above whatever cast the background has, and coverage reaches one
ten levels further up.

The Vaches Noires panels are deliberately left alone: they already overlay three saturated
colours, one per method, and their legend depends on those colours.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ASSETS = Path(__file__).resolve().parent / "assets"
SOURCES = ("Resultat_1FIND.png", "ResultatNoisy_1FIND.png", "PalaisDesPapes_Result.png")
RED = np.array([214.0, 21.0, 39.0])


def recolour(name: str) -> None:
    source = ASSETS / name
    target = ASSETS / f"{source.stem}_red.png"
    image = np.asarray(Image.open(source).convert("RGB")).astype(float)
    redness = image[..., 0] - (image[..., 1] + image[..., 2]) / 2.0
    median = float(np.median(redness))
    floor = median + max(3.0, 6.0 * float(np.median(np.abs(redness - median))))
    coverage = np.clip((redness - floor) / 10.0, 0.0, 1.0)
    widened = Image.fromarray((coverage * 255).astype("uint8")).filter(ImageFilter.MaxFilter(3))
    coverage = np.asarray(widened).astype(float)[..., None] / 255.0
    blended = image * (1.0 - coverage) + RED * coverage
    Image.fromarray(blended.round().clip(0, 255).astype("uint8")).save(target, optimize=True)
    print(f"{target.name}: {int((coverage > 0.5).sum())} overlay pixels, floor {floor:.1f}")


def main() -> None:
    for name in SOURCES:
        recolour(name)


if __name__ == "__main__":
    main()
