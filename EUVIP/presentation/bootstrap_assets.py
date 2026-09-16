"""Restore missing presentation assets.

Defense assets are pinned by Git blob hashes. The official Cerema SVG is
preserved unchanged and converted to a vector PDF on an explicit white ground.
The build workflow commits restored assets; subsequent builds need no network.
Existing assets are not overwritten.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent
BASE = (
    "https://raw.githubusercontent.com/Ludwig-H/Manuscrit-de-th-se/"
    "3593fbf25434c4715b37537eae1287bf49239fa7/Soutenance/soutenance/"
)
CEREMA_URL = "https://www.cerema.fr/themes/custom/uas_base/images/LogoCerema_horizontal.svg"
# destination: (source-relative path, expected Git blob SHA-1)
ASSETS = {
    "theme/beamerthemeinria.sty": ("theme/beamerthemeinria.sty", "12db1795ae22a1afcac17d4c025d796ced6e125c"),
    "theme/beamercolorthemeinria.sty": ("theme/beamercolorthemeinria.sty", "d381490d7ca094c353444939a6dc735165e3eee2"),
    "theme/beamerinnerthemeinria.sty": ("theme/beamerinnerthemeinria.sty", "e6bd3f334ef4b7f1ff5ca4072c2c76653520703a"),
    "theme/beamerouterthemeinria.sty": ("theme/beamerouterthemeinria.sty", "6a16d96c136cb3880f38eb81b9e373cd963f756b"),
    "theme/imgs/RF-INria_Bloc-marque.png": ("theme/imgs/RF-INria_Bloc-marque.png", "1228f98691b9c86c095a139f51e4f567da833141"),
    "theme/imgs/Inria-logo-rouge.png": ("theme/imgs/Inria-logo-rouge.png", "0873f916dbd1824cfab43a3d01d4f055c0754d4b"),
    "theme/imgs/Filet-7pt.png": ("theme/imgs/Filet-7pt.png", "7b31b83f65928e8901beb522f84aaa93da43ae9b"),
    "theme/imgs/angle.png": ("theme/imgs/angle.png", "25ca0ccdc035dfeb654c3aa64b9ee2866ed8e59f"),
    "assets/VTGraF_granularite.png": ("imgs/VTGraF_granularite.png", "61ebcc2a4e12211e55a72182202b4d75864c655b"),
}


def restore_cerema_logo() -> None:
    svg_path = ROOT / "assets/cerema-official.svg"
    pdf_path = ROOT / "assets/cerema-official.pdf"
    source_path = ROOT / "assets/cerema-official.source.txt"
    if svg_path.is_file() and pdf_path.is_file() and source_path.is_file():
        return
    if not svg_path.is_file():
        request = Request(CEREMA_URL, headers={"User-Agent": "Mozilla/5.0 EUVIP-presentation-build"})
        with urlopen(request, timeout=60) as response:
            data = response.read()
        root = ElementTree.fromstring(data)
        if root.tag != "{http://www.w3.org/2000/svg}svg":
            raise RuntimeError("The Cerema endpoint did not return an SVG logo")
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = svg_path.with_suffix(".svg.tmp")
        temporary.write_bytes(data)
        temporary.replace(svg_path)
    data = svg_path.read_bytes()
    if not pdf_path.is_file():
        # Only needed when the committed PDF is missing (Debian: python3-fitz).
        import fitz

        with fitz.open(stream=data, filetype="svg") as artwork:
            vector_data = artwork.convert_to_pdf()
        with fitz.open(stream=vector_data, filetype="pdf") as vector:
            with fitz.open() as output:
                rect = vector[0].rect
                page = output.new_page(width=rect.width, height=rect.height)
                page.draw_rect(page.rect, color=None, fill=(1, 1, 1))
                page.show_pdf_page(page.rect, vector, 0)
                output.save(str(pdf_path), garbage=4, deflate=True, no_new_id=True)
    if not source_path.is_file():
        source_path.write_text(
            "Official Cerema horizontal color logo\n"
            f"Source: {CEREMA_URL}\n"
            "Retrieved for this presentation: 2026-09-16\n"
            f"Original SVG SHA-256: {hashlib.sha256(data).hexdigest()}\n"
            "SVG preserved byte-for-byte, with original colors and paths.\n"
            "PDF: vector conversion with an explicit white background; no recoloring.\n",
            encoding="utf-8",
        )
    print(f"Official Cerema logo ready ({len(data)} SVG bytes)")


def main() -> None:
    for relative, (source, expected) in ASSETS.items():
        target = ROOT / relative
        if target.is_file():
            continue
        request = Request(BASE + source, headers={"User-Agent": "EUVIP-presentation-build"})
        with urlopen(request, timeout=60) as response:
            data = response.read()
        digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Hash mismatch for {relative}: {digest} != {expected}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
        print(f"Restored {relative} ({len(data)} bytes)")
    restore_cerema_logo()


if __name__ == "__main__":
    main()
