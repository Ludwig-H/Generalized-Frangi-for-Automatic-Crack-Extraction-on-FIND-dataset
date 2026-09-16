"""Restore missing defense assets from one pinned public commit.

The downloaded files are committed alongside the presentation by the build
workflow. Subsequent local builds need no network when these files are present.
Existing files are never overwritten, so deliberate local edits are preserved.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BASE = (
    "https://raw.githubusercontent.com/Ludwig-H/Manuscrit-de-th-se/"
    "3593fbf25434c4715b37537eae1287bf49239fa7/Soutenance/soutenance/"
)
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


if __name__ == "__main__":
    main()
