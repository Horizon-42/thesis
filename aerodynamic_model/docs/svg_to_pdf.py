#!/usr/bin/env python3
"""Convert every SVG in a folder to a vector PDF (for LaTeX \\includegraphics).

    python svg_to_pdf.py assets/geodetic_ecef assets/pdf

Rendering goes through headless Google Chrome, the same engine the diagrams were
tuned in (label fonts, the centred hats on n̂/q̂), so the PDFs match what the HTML
pages show. Each PDF page is exactly the SVG's viewBox size (1 SVG px = 1 CSS px =
1/96 in); text stays text (fonts embedded as subsets), lines stay vectors. The
page is white where the SVG is transparent.

Chrome is found at the macOS default path or on PATH; set CHROME=/path/to/chrome
to use another binary.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def find_chrome() -> str:
    explicit = os.environ.get("CHROME")
    if explicit:
        return explicit
    if Path(MAC_CHROME).exists():
        return MAC_CHROME
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    sys.exit("svg_to_pdf: Google Chrome not found; set CHROME=/path/to/chrome")


def svg_size(svg: Path) -> tuple[float, float]:
    """Width and height (px) from the root viewBox."""
    head = svg.read_text(encoding="utf-8")[:4000]
    m = re.search(r'<svg[^>]*\sviewBox="([-\d.]+)[ ,]+([-\d.]+)[ ,]+([\d.]+)[ ,]+([\d.]+)"', head)
    if not m:
        sys.exit(f"svg_to_pdf: {svg} has no root viewBox")
    return float(m.group(3)), float(m.group(4))


def convert(chrome: str, svg: Path, pdf: Path, work: Path) -> None:
    width, height = svg_size(svg)
    page = work / f"{svg.stem}.html"
    page.write_text(
        "<!doctype html><html><head><style>"
        f"@page {{ size: {width}px {height}px; margin: 0; }}"
        "html, body { margin: 0; padding: 0; }"
        f"img {{ display: block; width: {width}px; height: {height}px; }}"
        f'</style></head><body><img src="{svg.resolve().as_uri()}"></body></html>',
        encoding="utf-8",
    )
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            page.resolve().as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not pdf.exists() or pdf.stat().st_size == 0:
        sys.exit(f"svg_to_pdf: Chrome produced no PDF for {svg}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="folder containing .svg files (not searched recursively)")
    parser.add_argument("output_dir", type=Path, help="folder for the .pdf files (created if missing)")
    args = parser.parse_args()

    svgs = sorted(args.input_dir.glob("*.svg"))
    if not svgs:
        sys.exit(f"svg_to_pdf: no .svg files in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome()
    with tempfile.TemporaryDirectory() as tmp:
        for svg in svgs:
            pdf = args.output_dir / f"{svg.stem}.pdf"
            convert(chrome, svg, pdf, Path(tmp))
            print(f"{svg} -> {pdf}")
    print(f"converted {len(svgs)} file(s)")


if __name__ == "__main__":
    main()
