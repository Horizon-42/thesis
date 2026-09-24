# heading_reading_2026_09 — the heading-reading comparison (vocabulary design §10.1)

`experiments/heading_reading_compare.py` (`run_ts.py heading_reading_compare`, runner reference R14) flew one train
sample under several heading readings with the real executor at fixed parameters: the holds reading of instruction-v2
(H1) and the per-step reading (H3) at several grids, leads and clearance rules. Its two measurements decided the
heading words of instruction-v3 (user, 2026-09-24): H3 on a 5° grid, labelled 4 s early, the clearance at the capture
turn's onset once the word in force converges.

- `4dTrajectory/outputs/POOLED/analyses/heading_reading_20260924/` (`d8e6ae35`): H1 against H3 5°/2°, leads 0–6 s;
- `4dTrajectory/outputs/POOLED/analyses/heading_clearance_20260924/` (`bd262763`): the three clearance rules.

**Archived 2026-09-24** when the holds reading and the interim spec fields (`heading_reading`, `heading_band_deg`,
`heading_clearance`) were removed from the live code: the runner compares readings the live labeller no longer has.
Moved byte for byte (`git mv`); its one test was cut verbatim out of `tests/test_autopilot.py` into
`tests/test_heading_reading_compare.py`. Off the import path; nothing here runs. To re-run a comparison, check out
`bd262763` (branch `dev-vocab-v3`).
