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
  measured 0 s for both groups on every run (`v3_20260924`, spec `781d42c05ec4`, 200 train flights). The runner's
  part, `method_b` of `experiments/executor_spec.py`, is in `experiments/executor_spec_method_b.py`; its three tests
  in `tests/test_executor_method_b.py`.
- `autopilot/derive.py` — method A's flown checks for p: the executor's own largest turn (124.5°) overshooting by at
  most the heading tolerance, and turns said word by word (synthetic, read through a moving-average approximation
  of the velocity fit) flown inside every word's envelope, every 5 m/s from 60 to 140 m/s. The fifth review found
  its p unsettled (not monotone in p, moving 4.5–5.5°/s with the turn's phase against the rows, and circular where
  the observed aircraft rolled at the executor's own p). Its test is in `tests/test_executor_method_a_checks.py`.
- `autopilot/measure_turns.py` — cut from `autopilot/measure.py`: r_turn (the median steady rate of ≥ 90° turns,
  2.2°/s on train) and φ_cap (the median bank of the 115–140 m/s turns, 25°).
