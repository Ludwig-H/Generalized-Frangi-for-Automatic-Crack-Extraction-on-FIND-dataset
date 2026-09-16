"""Check the compiled deck, render every page and package editable sources."""
from __future__ import annotations

import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
PDF = ROOT / "Presentation_EUVIP_2026_Hauseux.pdf"


def main() -> None:
    source = (ROOT / "main.tex").read_text(encoding="utf-8")
    options = re.findall(r"\\begin\{frame\}(?:\[([^\]]*)\])?", source)
    content_count = sum("noframenumbering" not in item for item in options)
    if content_count != 10:
        raise RuntimeError(f"Expected 10 content frames, found {content_count}")
    log = (ROOT / "build/main.log").read_text(errors="replace")
    for error in ("Missing character:", "There were undefined references", "Undefined control sequence", "Overfull \\hbox", "Overfull \\vbox"):
        if error in log:
            raise RuntimeError(f"LaTeX log contains: {error}")
    document = fitz.open(PDF)
    if len(document) != 16:
        raise RuntimeError(f"Expected 16 PDF pages, found {len(document)}")
    render = ROOT / "build/render"
    render.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1920, 1180), "white")
    draw = ImageDraw.Draw(sheet)
    for index, page in enumerate(document):
        if abs(page.rect.width / page.rect.height - 16 / 9) > 0.01:
            raise RuntimeError(f"Page {index + 1} is not 16:9")
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        filename = render / f"page-{index + 1:02d}.png"
        pixmap.save(filename)
        image = Image.open(filename).convert("RGB")
        image.thumbnail((468, 264))
        x, y = (index % 4) * 480, (index // 4) * 295
        sheet.paste(image, (x + 6, y + 5))
        draw.text((x + 8, y + 273), f"{index + 1:02d}", fill="black")
    sheet.save(ROOT / "apercu.png", optimize=True)
    warnings = [line for line in log.splitlines() if "Overfull" in line]
    report = {"pages": len(document), "content_slides": content_count,
              "aspect_ratio": "16:9", "overfull_warnings": warnings,
              "pdf_bytes": PDF.stat().st_size}
    (ROOT / "build/qa.json").write_text(json.dumps(report, indent=2) + "\n")
    archive = ROOT / "build/Presentation_EUVIP_2026_sources.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if not path.is_file() or "build" in relative.parts or "__pycache__" in relative.parts:
                continue
            if path.suffix.lower() in {".ttf", ".otf", ".woff", ".woff2"}:
                continue
            bundle.write(path, Path("presentation") / relative)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
