#!/usr/bin/env python3
"""
split_doc.py
============
Splits a bilingual Markdown document that uses <!-- LANG:ZH --> and <!-- LANG:EN -->
markers into two separate files.

Usage
-----
    python split_doc.py [INPUT] [--zh OUTPUT_ZH] [--en OUTPUT_EN]

Examples
--------
    # Default: reads DEV_GUIDE_BILINGUAL.md, writes DEV_GUIDE_ZH.md + DEV_GUIDE_EN.md
    python split_doc.py

    # Explicit paths
    python split_doc.py DEV_GUIDE_BILINGUAL.md --zh guide_zh.md --en guide_en.md

Rules
-----
- Lines containing exactly <!-- LANG:ZH --> switch the active language to Chinese.
- Lines containing exactly <!-- LANG:EN --> switch the active language to English.
- The marker lines themselves are NOT written to either output file.
- The optional preamble before the first marker is written to BOTH output files
  (useful for a shared YAML front-matter or top-level title).
- Consecutive blank lines are collapsed to at most one blank line in the output,
  so marker transitions don't leave extra whitespace.
"""

import argparse
import re
import sys
from pathlib import Path

# ── Marker patterns ─────────────────────────────────────────────────────────
MARKER_ZH = re.compile(r"<!--\s*LANG:ZH\s*-->", re.IGNORECASE)
MARKER_EN = re.compile(r"<!--\s*LANG:EN\s*-->", re.IGNORECASE)

# ── Header banners injected at the top of each output file ──────────────────
BANNER_ZH = """\
<!--
  此文件由 split_doc.py 从双语文档自动生成（中文版）
  原始双语文档：{source}
-->

"""

BANNER_EN = """\
<!--
  This file was automatically generated from the bilingual source by split_doc.py (English version).
  Source bilingual document: {source}
-->

"""


def collapse_blank_lines(lines: list[str]) -> list[str]:
    """Replace runs of more than one consecutive blank line with a single blank line."""
    result: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue  # skip extra blank
        result.append(line)
        prev_blank = is_blank
    # Remove leading/trailing blank lines
    while result and result[0].strip() == "":
        result.pop(0)
    while result and result[-1].strip() == "":
        result.pop()
    return result


def split_bilingual(
    source_path: Path,
    zh_path: Path,
    en_path: Path,
) -> tuple[int, int]:
    """
    Parse *source_path* and write Chinese content to *zh_path* and
    English content to *en_path*.

    Returns (zh_line_count, en_line_count) for a summary message.
    """
    source_text = source_path.read_text(encoding="utf-8")
    raw_lines = source_text.splitlines(keepends=True)

    # ── State machine ────────────────────────────────────────────────────────
    # lang = None  → preamble (written to both)
    # lang = 'zh'  → Chinese section
    # lang = 'en'  → English section
    lang: str | None = None

    preamble: list[str] = []
    zh_lines: list[str] = []
    en_lines: list[str] = []

    for raw_line in raw_lines:
        line_stripped = raw_line.strip()

        if MARKER_ZH.search(line_stripped):
            lang = "zh"
            continue  # discard marker line
        if MARKER_EN.search(line_stripped):
            lang = "en"
            continue  # discard marker line

        if lang is None:
            preamble.append(raw_line)
        elif lang == "zh":
            zh_lines.append(raw_line)
        else:
            en_lines.append(raw_line)

    # ── Post-process ─────────────────────────────────────────────────────────
    # Convert to plain string lists (strip keepends, re-add \n for clean output)
    def to_clean(lines: list[str]) -> list[str]:
        # keepends=True already, just normalise
        return [ln if ln.endswith("\n") else ln + "\n" for ln in lines]

    preamble_clean = to_clean(preamble)
    zh_clean = collapse_blank_lines(to_clean(zh_lines))
    en_clean = collapse_blank_lines(to_clean(en_lines))

    source_name = source_path.name

    # ── Write Chinese output ─────────────────────────────────────────────────
    with zh_path.open("w", encoding="utf-8") as fh:
        fh.write(BANNER_ZH.format(source=source_name))
        fh.writelines(preamble_clean)
        fh.write("\n")
        fh.writelines(zh_clean)
        fh.write("\n")

    # ── Write English output ─────────────────────────────────────────────────
    with en_path.open("w", encoding="utf-8") as fh:
        fh.write(BANNER_EN.format(source=source_name))
        fh.writelines(preamble_clean)
        fh.write("\n")
        fh.writelines(en_clean)
        fh.write("\n")

    return len(zh_clean), len(en_clean)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a bilingual Markdown document into Chinese and English files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="DEV_GUIDE_BILINGUAL.md",
        help="Path to the bilingual source document (default: DEV_GUIDE_BILINGUAL.md)",
    )
    parser.add_argument(
        "--zh",
        default=None,
        help="Output path for the Chinese-only file (default: <input_stem>_ZH.md)",
    )
    parser.add_argument(
        "--en",
        default=None,
        help="Output path for the English-only file (default: <input_stem>_EN.md)",
    )
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        print(f"ERROR: Input file not found: {source}", file=sys.stderr)
        sys.exit(1)

    stem = source.stem.replace("_BILINGUAL", "").replace("_bilingual", "")
    zh_out = Path(args.zh) if args.zh else source.parent / f"{stem}_ZH.md"
    en_out = Path(args.en) if args.en else source.parent / f"{stem}_EN.md"

    print(f"Reading  : {source}")
    print(f"Writing ZH → {zh_out}")
    print(f"Writing EN → {en_out}")

    zh_count, en_count = split_bilingual(source, zh_out, en_out)

    print(f"\nDone.")
    print(f"  Chinese  : {zh_count} lines  →  {zh_out}")
    print(f"  English  : {en_count} lines  →  {en_out}")


if __name__ == "__main__":
    main()
