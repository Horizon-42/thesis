"""A rough check of the text against the main ASD-STE100 writing rules. It is a heuristic, not the STE dictionary check.

    python scripts/two_tier_tutorial/ste_check.py [--show N]

It reads the prose of the page (paragraphs, list items, notes, and the captions of the demos) and counts:
  - sentences of more than 25 words (STE: 25 words at most in a descriptive sentence, 20 in a procedure);
  - passive voice (a form of "to be" + a past participle), found with a word list and an -ed/-en pattern;
  - contractions and "etc.", "e.g.", "i.e." (STE avoids them);
  - paragraphs of more than 6 sentences.
A sentence is "clean" when it has no flag. The 80 % target of the task is read on the share of clean sentences.
"""
from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"

BE = r"(?:is|are|was|were|be|been|being)"
# past participles that are not formed with -ed, plus a few -ed words that are adjectives in this text
IRREGULAR = "given|made|taken|chosen|written|known|shown|seen|done|read|set|held|kept|said|flown|drawn|built|run|put|left|found|got|lost|begun|spoken|broken|stood|sent|told|led|met|paid|bound|cut|hit|let"
ADJ_OK = {"based", "named", "called", "relative", "required", "limited", "stated", "fixed", "sealed", "unchanged", "assigned", "published", "coded", "defined", "observed", "recorded", "closed", "added", "needed", "used", "marked", "free"}
PASSIVE = re.compile(rf"\b{BE}\s+(?:not\s+|also\s+|never\s+|only\s+|then\s+|still\s+)?(\w+)\b", re.I)


def is_passive(sentence: str) -> bool:
    for m in PASSIVE.finditer(sentence):
        w = m.group(1).lower()
        if w in ADJ_OK:
            continue
        if re.fullmatch(IRREGULAR, w) or (w.endswith("ed") and len(w) > 4):
            return True
    return False


class Text(HTMLParser):
    """Collect the prose of p, li, td (if it ends in a full stop) and the notes."""

    KEEP = {"p", "li"}
    SKIP = {"script", "style", "code", "pre", "table", "svg", "nav", "kbd", "h1", "h2", "h3", "h4", "summary", "button", "figcaption"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth_skip = 0
        self.stack: list[str] = []
        self.buf: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "") or ""
        if tag in self.SKIP:
            self.depth_skip += 1
        if "pill" in cls or "src" in cls:
            self.depth_skip += 1; tag = "span-skip"
        self.stack.append(tag)
        if tag in self.KEEP and not self.depth_skip:
            self.buf = []

    def handle_endtag(self, tag):
        if not self.stack:
            return
        top = self.stack.pop()
        if top in self.KEEP and not self.depth_skip:
            text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
            if text:
                self.paragraphs.append(text)
            self.buf = []
        if top in self.SKIP or top == "span-skip":
            self.depth_skip -= 1

    def handle_data(self, data):
        if not self.depth_skip:
            self.buf.append(data)


def sentences(paragraph: str) -> list[str]:
    p = re.sub(r"\b(e\.g|i\.e|etc|vs|Fig|No)\.", r"\1", paragraph)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z“(\"\[])", p) if len(s.split()) >= 3]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--show", type=int, default=12); args = ap.parse_args()
    parser = Text()
    for f in sorted((SRC / "sections").glob("*.html")):
        parser.feed(f.read_text(encoding="utf-8"))
    paragraphs = list(parser.paragraphs)
    # the captions of the demos are in the JS sources: class: 'cap', html: '...'
    for f in sorted(SRC.glob("demos_*.js")):
        for m in re.finditer(r"class: 'cap', html: '((?:[^'\\]|\\.)*)'", f.read_text(encoding="utf-8")):
            sub = Text(); sub.feed("<p>" + m.group(1).replace("\\'", "'") + "</p>"); paragraphs += sub.paragraphs
    flagged: list[tuple[str, str]] = []; total = clean = 0; long_p = 0
    for para in paragraphs:
        sents = sentences(para)
        if len(sents) > 6:
            long_p += 1
        for s in sents:
            total += 1; words = len(re.findall(r"[A-Za-z0-9°%']+", s)); flags = []
            if words > 25: flags.append(f"long ({words} words)")
            if is_passive(s): flags.append("passive")
            if re.search(r"\b\w+n't\b|\b(?:it|that|there|we|they)'s\b|\betc\b|\be\.g\b|\bi\.e\b", s): flags.append("contraction or abbreviation")
            if flags: flagged.append((", ".join(flags), s))
            else: clean += 1
    pct = 100 * clean / total
    kinds = {}
    for f, _ in flagged:
        for k in f.split(", "):
            k = k.split(" (")[0]; kinds[k] = kinds.get(k, 0) + 1
    print(f"{total} sentences in {len(paragraphs)} paragraphs; clean {clean} ({pct:.1f} %); flags: {kinds}; paragraphs over 6 sentences: {long_p}")
    for f, s in flagged[: args.show]:
        print(f"  [{f}] {s[:190]}")
    return 0 if pct >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
