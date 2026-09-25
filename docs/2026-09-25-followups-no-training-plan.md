# Fixing the code-health follow-ups that do not touch training (work plan, 2026-09-25)

The user: "把所有不会影响训练的问题都修复 — 在新 worktree 做，改完没问题后再合并；多个 package 的问题，要在各个 package 中做好文档更新".
Source list: the status table of `docs/code-health-followups.md` (column "Fix affects training / post-training?" = **no**).
This file is the work plan and its status; it is folded into the follow-ups table and `docs/CHANGELOG.md` and deleted
before the merge.

Branch `dev-followups-no-training`, worktree `.claude/worktrees/followups-no-training`, base `dev-two-tier` `d1030041`.

## Guards

1. **The training chain's code identity must not move.** `autopilot.spec.executor_source_sha256()` =
   `d3ff89290e65…` (the current executor spec `v7_20260925` records it) and
   `instructions.artefact.labeller_source_sha256()` = `55f6f0bcd6ee…` (the sentence artefact records it). Checked after
   every package with `scratchpad/training_hashes.sh <checkout>`. Hence these files are **frozen** (a byte of change
   refuses the spec): all of `autopilot/`; the labeller's `instructions/{spec,words,airport,signals,piecewise,envelope,
   measure}.py` and `instructions/labeller/*`; `aerodynamic_model/torch_dynamics.py`, `aircraft/reference_speeds.py`,
   `final_approach/crossing.py`, `geokit/__init__.py`, `trajectory_data_process/harvest/airports.py`,
   ts `config.py`, `data/dataset.py`, `geometry/flyability.py`, `outputs/constraints/speed_floor.py`,
   `outputs/dynamics/{context,rollout}.py`, `outputs/envelope.py`.
2. **No change to what `build_series` produces** (the executor rebuilds every flight and demands the stored signals).
3. **Do not touch the files of `dev-post-train`** (another session): ts `CLAUDE.md` (index lines only if unavoidable),
   `docs/reference/runners.md`, `experiments/prior_free_generation.py`, `prior_landing_reward.py`, `prior/generate.py`,
   `tests/test_architecture.py`.
4. No artefact on disk is rewritten, no evaluation / harvest / export is re-run (the user's rule).

## Excluded, with the reason (reported to the user)

| Entry | Why not now |
|---|---|
| §17 `require_matching_runway_data`; #15h `airports.py` elevation default; §20 flyability copy; §34, §37, §39; §38 (CONTROL_NAMES, speed-floor lag credit, cos γ); #13's `dataset.py` checks | frozen files (guard 1) |
| §20 `start_state._slope`; #10; #14 | would change `build_series` (guard 2) |
| #1–#4, #6 | change the harvest's output on the next download / re-roster |
| Training review 5, 6 | the replay overlay's schema changes (v3) — the published v2 overlays would be refused until re-exported: needs the user's OK |
| §21 Lookback | a record-contract field + republishing every prediction directory |
| #9, #11 | scenario JSON / evaluation records change shape: every stored scenario input / report regenerated |
| §15 KRDU 14 | a UX decision (fourth state vs surfaced reason) the gap document leaves to the user |
| ts `lead_landings` | "the owner's call": it changes the old control models' intent inputs |
| §23 optimizer velocity floor | a design choice (the floor's source) that changes optimizer solutions |
| `build_runway_config.py` | needs FAA NASR widths and plate minima as new sources |
| Performance index: the 60 t observed fallback for a substituted type | what mass an observed record of another airframe should carry is a judgement that moves the gate |

## Dismissed (row says so, entry deleted)

- §6 `write_arrival_records` clears `arrivals/` — deliberate since, documented as a root CLAUDE.md hazard; `--observed-only`
  is the non-destructive mode.
- §33 readers compare stored configs field by field — refusing is right under the no-compatibility rule (2026-09-19);
  stored CV results that lack fields are superseded, not read as defaults.

## To fix, by package (status: todo / done `<sha>`)

**trajectory_data_process**
- §1 one `_iso` (classify, store, merge) — todo
- §3 `source_event_availability` validation apart from the derivation — todo
- #5 CZML censored tail: start at the closest support sample, no silent 70 m/s — todo
- #7 `GROUND_START_AGL_M` comment: state what was measured, on which fleet — todo
- #15g import `ALTITUDE_SOURCE` — todo
- #15i `--rerender-czml` renders with the audit's policy — todo
- #15j landing-screen provenance records `max_crossing_height_m` — todo
- #17 a sweep for staging leftovers — todo

**flight_scenarios**
- #8 datum: docs say what runs; dead `_geoid_transformer` / `waypoints_to_msl` removed — todo
- #13 `build.py` refuses instead of the fallback that never binds — todo
- #15a `fitted_approach` `hae_minus_msl_m` required; #15b `procedure_final` elevation required — todo
- scene data plane (7), (8), (9), (12), (14) + the three test gaps; (11) documented — todo
- #12 population note (`docs/population_reference.md`) — todo

**evaluation**
- §11 `METHODOLOGY.event` says `final_time_s` is the last sample — todo
- §12 `_reference_aggregate` comment — todo
- `THRESHOLD_SPEED_GATE.md` §3.3: the unsourced "+5/−0 kt" marked or removed — todo

**4dTrajectory/optimization**
- §2 `summary_row`: required identity fields refuse, optional ones stated — todo
- #15c references keyed by `flight_key` — todo
- the numpy-2 `test_optimizer.py:361`; the `arr_airport` fixture — todo

**aircraft / aerodynamic_model**
- #21 OpenAP caches read with a schema check — todo
- the two readout scripts resolve with the default provider — todo
- T2: `chart_scale` required in the lag kernels; the transport-chart rollout entry points without a caller — todo

**ts_transformer** (not the frozen files)
- Training review 3 (a non-runner `instructions/training_files.py`), 4 (index write guarded, all-or-nothing), 7 (one
  `require_stored_sentence` for the exporters and the backend), 8 (shared helpers where not frozen), 9, 10 — todo
- the three ablation runners refuse an existing output directory — todo
- `predictability_report` + §36: the forecast's dynamics batch, public, used by the report — todo
- two dead loss helpers deleted — todo
- `experiments.pipeline` flags renamed to what it emits — todo
- the auto-batch probe passes a future when the model consumes one — todo
- `docs/` measurement code: the package runners stop importing `docs/` scripts; `sys.path` preambles — todo
- §27 one record-emitting helper in `cli/predict.py` — todo
- §28 / §29 barrier and trombone share the bank inversion; the barrier's coordination guarded — todo
- §35 the anchor gate calls the stall-speed function and the shared constants — todo
- §40 the two conventions named where they are read — todo
- `EXPERIMENTS_MAIN` into `repo_layout` — todo
- #15e `usable_series` exclusions recorded; #15f `channels.py` docstring; #16 the width study resolves the frozen root;
  #22 `git_state` imported — todo
- the stale closed-loop docstring (`outputs/control/strategy.py:176-178`) — todo

**aeroviz_backend / aeroviz-4d/python / repo root**
- backend: `training_set` through the shared module; shared helpers — todo
- §4 comparison-CZML fixture: say the version is a pass-through — todo
- `run_all_tests.sh` header and known failures; `docs/open-items.md` nominal-law narration; ts README pointer — todo

**Docs** (each package): its `CLAUDE.md` index line / reference doc where a contract changed; `docs/CHANGELOG.md`;
the follow-ups table's rows flipped with commits and the fixed / obsolete / dismissed entries deleted.
