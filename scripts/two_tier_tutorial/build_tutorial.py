"""Build the tutorial: one self-contained HTML file (CSS, JS and the real data inline).

    python scripts/two_tier_tutorial/build_tutorial.py            # writes docs/two_tier/tutorial/index.html

Sources are in `src/`: `head.html`, `sections/*.html` (in file order), `foot.html`, `style.css`, the `*.js` files.
The figures (PNG and PDF) are made by `make_figures.py` and are linked by relative path, not inlined.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
OUT = HERE.parents[1] / "4dTrajectory" / "ts_transformer" / "docs" / "two_tier" / "tutorial"
JS_ORDER = ["core.js", "words.js", "exec.js", "figures.js", "demos_vocab.js", "demos_exec.js", "demos_prior.js", "demos_post.js", "boot.js"]


def main() -> None:
    data = json.loads((HERE / "data" / "tutorial_data.json").read_text(encoding="utf-8"))
    data["prior_params"] = json.loads((HERE / "data" / "prior_params.json").read_text(encoding="utf-8"))
    css = (SRC / "style.css").read_text(encoding="utf-8")
    js = "\n".join(f"/* ---- {name} ---- */\n" + (SRC / name).read_text(encoding="utf-8") for name in JS_ORDER)
    sections = "\n".join((p).read_text(encoding="utf-8") for p in sorted((SRC / "sections").glob("*.html")))
    html = (SRC / "head.html").read_text(encoding="utf-8").replace("{{CSS}}", css) + sections
    html += (SRC / "foot.html").read_text(encoding="utf-8").replace("{{DATA}}", json.dumps(data, separators=(",", ":"))).replace("{{JS}}", js)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {OUT / 'index.html'} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
