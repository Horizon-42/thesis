# executor_first_principles_2026_10 — the prototype and measurement scripts behind executor design drafts 3–5, D9 and D10

Archived 2026-10-12 on the user's order ("把实验相关脚本，挪到代码仓库中合适的地方"), when development moved from the Mac to
SP-AI. These are the one-off scripts of 2026-10-11/12 that led to the first-principles executor
(`docs/two_tier/design/executor_first_principles.zh.md`, §10.1, §15–§17), the leads 4/4/4 (D9) and the last-angle-word rule
instruction-v8 (D10). They ran on the Mac from the results directory itself; nothing in them was reviewed; they are not
package code. Off the import path on purpose (`tests/test_architecture.py`), and pytest does not collect them
(`pyproject.toml` `norecursedirs` holds `archive`) — the `test_*.py` files here are experiment scripts, not tests.

Read with the handover `docs/two_tier/readouts/2026-10-12_executor_handover.zh.md` §3.

## 1. Results

The outputs are NOT in git: `4dTrajectory/outputs/POOLED/spai_copy_20261011/plant_comparison/` on the Mac (one directory per
run, named by its `--out`; the README beside it, `spai_copy_20261011/README.md`, says what each holds). The numbers the
decisions rest on are in the design document (D9, D10) and the handover. The inputs that sat in the session's scratch area
were kept beside the results: `inputs/observed_1000/` (the observed evaluation grades of the 999-flight draw, read from SP-AI's
harvest evaluation reports by `obs_grades.py`), `inputs/observed_750/`, `inputs/` (key and track lists), `run_train100/`.

## 2. Paths (`scripts/archive_paths.py`)

On the Mac the scripts held absolute paths (the two-tier2 worktree, the results directory, the session's scratch area). Archived,
they read: `REPO` = the repository this archive sits in; `POOLED` = its `4dTrajectory/outputs/POOLED`; `RESULTS` = the results
directory above (it exists only where it was made or copied to); `HARVEST` = SP-AI's live harvest
`trajectory_data_process/outputs/harvest`. The Mac runs read a read-only COPY of that harvest, `spai_copy_20261011/harvest/`
(pulled 2026-10-11 by `pull.sh`, 1863 track files), so a rerun on SP-AI reads the same flights from their source. The instruction
artefact `instruction_language/v12_20261005` and the executor spec `executor/v17_20261005` are SP-AI's (the Mac held train-only
copies). Run from a checkout whose `4dTrajectory/outputs` holds them (a worktree has none unless linked).

The scripts change the repository's behaviour only by patching module attributes in their own process (`flights.arrival_manifest_path`,
`executor.Plant`, the labeller's `vertical_pieces` / `read_heights` / `final_path_angle_deg`); they never edit the packages.
Scripts written against older code (the drafts 3–4 prototypes) may not run against the code they now sit beside: they are the
record of what was run, not maintained tools.

## 3. Scripts

| script | what it did |
|---|---|
| `pull.sh` | read-only pull of the instruction artefact (train), the executor spec and the arrivals from SP-AI |
| `track_list.py`, `list_tracks_1000.py` | which track files a seeded draw needs (to pull only those) |
| `obs_grades.py` | the observed evaluation grades of a key list, from a harvest evaluation report on stdin |
| `compare_plants.py`, `summarize.py`, `sweep.sh` | the same stored closed-loop sentences flown on the point-mass and the first-order-lag plant, graded; τ sweep |
| `regenerate_closed_loop.py`, `summarize_regenerated.py` | the closed-loop correction words regenerated on the lag plant (three arms) |
| `identify_bank_response.py` | the roll response read from 1 Hz ADS-B positions, observed and per plant (τ_bank not identifiable) |
| `compare_compensation.py` | an actuator observer + dead-beat lead between executor and plant, with τ mismatch |
| `lag_executor.py`, `compare_new_executor.py`, `closed_loop_new_vs_current.py` | the first lag-aware executor prototype (observer, exact stop formula) |
| `compare_stopping_rate.py`, `test_roll_rate_limit.py` | the lateral law's stopping rate; the roll-rate limit p |
| `test_speed_target.py`, `test_speed_mid.py`, `test_speed_mid_sqrt.py`, `counterfactual_speed_margin.py`, `speed_diag.py` | the speed-gate failures ("unspecified" at the window's lower edge) |
| `diagnose_failures.py`, `counterfactual_deadband.py` | the baseline's 77 failures by gate; the corrector's lateral dead band |
| `final_phase_thresholds.py`, `phase_split_thresholds.py`, `phase_split_thresholds_p.py` | corrector thresholds by approach phase; 4 s words from 2 s words |
| `limits_and_rates.py` | which executor limits bind; how fast bank, thrust and load factor move |
| `open_loop_4s.py`, `probe_turns.py`, `vert_diag.py` | the 4 s open-loop words flown; the executor's turn offset; a vertical diagnosis |
| `pre_word.py`, `angle_corr_timing.py`, `angle_corr_err.py` | the observed sink before a descent word; where the angle corrections fall (evidence for D7) |
| `new_design_eval.py` | the draft-5 executor on 750 train flights, arms of the leads (shared setup: `spec_of`, `params_of`) |
| `diagnose_new_design.py`, `diagnose_2.py`, `diagnose_3.py` | its lateral and vertical failures; p = 5 vs 15; where the threshold height error is made |
| `lead_eval_1000.py`, `lead_eval_1000_combo.py`, `summarize_lead_eval_1000.py` | V12 on 999 train flights: each column's lead 0–8 s, and 4/4/4 (D9) |
| `last_angle_word.py` | where the last angle word is read, crossed with the vertical verdicts (Q4) |
| `lead_eval_tail.py`, `compare_tail.py`, `tail_check.py` | the two end-of-track rules A and B (D10); `tail_check.py` found the first version's downwind-leg error |
| `v8_refusals.py` | instruction-v8 on all 44,703 train signals: refusals and last-word class changes |
