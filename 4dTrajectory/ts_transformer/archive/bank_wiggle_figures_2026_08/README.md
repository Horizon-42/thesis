# bank_wiggle_figures_2026_08 — the figure scripts of the bank-wiggle diagnosis

Archived 2026-09-26 from `docs/` (layout rule L20: `docs/` holds documents). Moved byte for byte; off the import path.

- `plot_control_wiggle_diagnosis.py` (2026-08-19, `3293cf7c`) and `plot_imitation_design.py` (2026-08-20, `7024ff92`)
  drew the figures of `docs/history/2026-08_control_path/2026-08-19_control_bank_wiggle_diagnosis.zh.md` from the published KSJC / KRDU batches
  into `docs/figures/`. The PNGs the document uses stay with it; `imitation_skill_band.png`, which only
  `plot_imitation_design.py` wrote and no document shows, moved here with the scripts.

They no longer run as they are: they expected to sit in `docs/` (`Path(__file__).parent / "figures"`), and
`plot_imitation_design.py` imports `score` from `score_control_arms.py` beside it, which is now the runner
`experiments/score_control_arms.py` (`python run_ts.py score_control_arms`).
