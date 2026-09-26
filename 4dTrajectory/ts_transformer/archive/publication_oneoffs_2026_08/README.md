# publication_oneoffs_2026_08 — two one-off fixes to already published frontend metadata

Archived 2026-09-26 from `docs/` (layout rule L20: `docs/` holds documents). Moved byte for byte; off the import
path; both have already done their job and are not meant to run again (their path arithmetic assumed `docs/`).

- `backfill_category_accuracy.py` (2026-08-24, `f5bd52d0`) — stamped the per-category `accuracy` block into
  `categories.json` entries published before `build_scenario_comparison_czml.py` wrote it itself.
- `relabel_published_categories.py` (2026-08-24, `8522ccc8`) — rewrote the legacy `ts_*` comparison-category labels to
  the `run_naming` grammar (labels only; no CZML, records, keys or directories). Publisher-managed categories are
  refreshed by `publish_ts_experiment_trajectories.py --refresh-labels-only`, which is the live path.
