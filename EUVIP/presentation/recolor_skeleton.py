"""Recolour the FIND skeleton overlay so that it survives the reduction to slide size.

`assets/Resultat_1FIND.png` draws the extracted skeleton in a very dark red over a grey
intensity image. At the width the slides use, roughly two centimetres, the one-pixel line
is both sub-pixel and indistinguishable from the crack underneath. This script writes
`assets/Resultat_1FIND_red.png`, where the pixels that already carry the overlay are
remapped to a legible red and widened by one pixel. The neutral background is untouched
and no geometry is added or removed: the overlay is located by its own saturation, since
the underlying image is grey, and the widening is a single 3x3 maximum filter on the
resulting coverage map. The original asset is never modified.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ASSETS = Path(__file__).resolve().parent / "assets"
SOURCE = ASSETS / "Resultat_1FIND.png"
TARGET = ASSETS / "Resultat_1FIND_red.png"
RED = np.array([214.0, 21.0, 39.0])


def main() -> None:
    image = np.asarray(Image.open(SOURCE).convert("RGB")).astype(float)
    saturation = image.max(2) - image.min(2)
    coverage = np.clip((saturation - 6.0) / 14.0, 0.0, 1.0)
    widened = Image.fromarray((coverage * 255).astype("uint8")).filter(ImageFilter.MaxFilter(3))
    coverage = np.asarray(widened).astype(float)[..., None] / 255.0
    blended = image * (1.0 - coverage) + RED * coverage
    Image.fromarray(blended.round().clip(0, 255).astype("uint8")).save(TARGET, optimize=True)
    print(f"{TARGET.name}: {int((coverage > 0.5).sum())} overlay pixels")


if __name__ == "__main__":
    main()
