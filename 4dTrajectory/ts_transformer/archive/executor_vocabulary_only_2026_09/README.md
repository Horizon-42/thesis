# executor_vocabulary_only_2026_09 — the executor parameters that came from data or from flown checks

**Archived 2026-09-24**, the vocabulary-only plan's first stage: the user's rule that the executor takes no
information beyond the vocabulary (a parameter mined from data means it will not generalise). Moved byte for byte
(`git mv`); passages of live modules cut verbatim. Off the import path; nothing here runs. To re-run any of it, check
out `4cc2b948` (branch `dev-vocab-v3`), the last commit before the stage.

What replaced each part (executor design §9–§10): the turn rates and the bank limit are the vocabulary's
(`turn_rate_min_deg_s` / `turn_rate_max_deg_s`, `turn_bank_max_deg`), a word acts when said, τ_ψ is the heading lead
and p the bank limit over the lead (`autopilot/derive.py`, 32° / 4 s = 8°/s).

- `autopilot/observe.py` — method B: flew a seeded train sample of truth sentences with no delay, read each flown
  track back the way the data plane reads a radar track (the 15 s centred velocity fit, the 2 s grid, the labeller's
  signals), re-read its words and took the median lead per group as that group's delay. Under instruction-v3 it
  measured 0 s for both groups on every run (`v3_20260924`, spec `781d42c05ec4`, 200 train flights). Its three tests
  are in `tests/test_executor_method_b.py`.
- `autopilot/derive.py` — method A's flown checks for p: the executor's own largest turn (124.5°) overshooting by at
  most the heading tolerance, and turns said word by word (synthetic, read through a moving-average approximation
  of the velocity fit) flown inside every word's envelope, every 5 m/s from 60 to 140 m/s. The fifth review found
  its p unsettled (not monotone in p, moving 4.5–5.5°/s with the turn's phase against the rows, and circular where
  the observed aircraft rolled at the executor's own p). Its test is in `tests/test_executor_method_a_checks.py`.
- **Stage 2 (2026-09-24)**: the rest of `autopilot/measure.py` — the speed changes' pace (a_dec 0.28, a_acc 0.19,
  a_unspec 0.30 m/s² on train) and the landing aim (20.8 m, window 8.4–36.5 m) — is gone too: the pace is one of the
  vocabulary's speed steps over its shortest hold (0.25 m/s²), the crossing point the pointed runway's published TCH.
  The module had nothing left to measure and was removed; `autopilot/measure.py` here (the `4cc2b948` form) holds
  every line its stage-1 form had. Its two tests, as they stood, are in `tests/test_executor_measure.py`.
- `autopilot/measure.py` and `experiments/executor_spec.py` — the two live modules as they were at `4cc2b948`, whole
  (their live successors keep the stage-2 measurements): the measurement of r_turn (the median steady rate of ≥ 90°
  turns, 2.2°/s on train) and φ_cap (the median bank of the 115–140 m/s turns, 25°), and the runner's method A and B
  (`method_b`, the delay flags, `LEAD_WINDOW_S`).
