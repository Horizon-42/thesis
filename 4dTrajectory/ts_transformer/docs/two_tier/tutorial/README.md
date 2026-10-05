# Two-tier model: interactive tutorial

Open `index.html` in a browser. It is one self-contained file (no network needed). The figures are in `figures/png/`
(300 dpi) and `figures/pdf/` (vector); the page links them by relative path, so keep the folder together.

- Text: English, ASD-STE100 style (heuristic check: `scripts/two_tier_tutorial/ste_check.py`), Chinese notes on hard terms
  (button "中文注解" hides them).
- 26 interactive demos; 15 figures with suggested captions (the captions are on the page only, not in the images).
- Real data: the formal artefact `v12_20261005`, **train split only** (outline D85); no val or test day.

## Rebuild

The sources and tools are in `scripts/two_tier_tutorial/` (not under `docs/`, which holds no Python, `tests/test_architecture.py`).

```bash
conda activate aeroviz
python scripts/two_tier_tutorial/extract_real_data.py            # real flights, airports, counts (train split)
python scripts/two_tier_tutorial/prior_param_counts.py <checkout with prior/>   # parameter counts; the prior code is on dev-two-tier-v4-prior until stage B is merged
node   scripts/two_tier_tutorial/figure_data.js > scripts/two_tier_tutorial/data/figure_sim.json
python scripts/two_tier_tutorial/make_figures.py                 # 15 figures, PNG + PDF
python scripts/two_tier_tutorial/build_tutorial.py               # index.html
python scripts/two_tier_tutorial/tests/make_vectors.py && node scripts/two_tier_tutorial/tests/test_ports.js   # JS ports vs Python
```

`index.html#selftest` mounts every demo and lists errors at the top of the page.
