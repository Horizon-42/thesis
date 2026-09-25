# Code-health follow-ups (deferred)

Findings noticed while working elsewhere, recorded rather than fixed on the spot so the
change that surfaced them stays reviewable. Nothing here is a live bug unless it says so.

Each entry states what was **verified** versus what is **judgement**, so a later reader can
tell how much re-checking it needs. **Keep the status table below current**: a new entry adds a row — its status and whether a fix would
touch the current training or post-training; when an entry is fixed or dismissed, its row says so (with the commit) and
the entry itself is deleted.

## Status of every entry (verified against the code 2026-09-25)

Checked entry by entry against `dev-two-tier` `1a7ac875` plus branch `dev-frontend-followups`: **41 open, 13
partly, 18 resolved, 5 obsolete** (the code it is about was archived). *open*: the problem is still
in the code; *partly*: some of it is fixed (the note says what is left); *resolved*: fixed (the note says by what);
*obsolete*: the code is gone. Rows follow the entries' order below; note the two sets of numbers (§19–§21 each appear
twice). Also found and fixed on `dev-frontend-followups`, never entered below: the terrain preload's tile counts and the
range ring radius re-rendering the app (`f73015df`, `1d8630c0`), the Pilot catalog failure hidden on entering
Trajectory (`fd9f205f`, `1d8630c0`), Observe's sample count planning loads per keystroke (`3a4aea81`, `d57f7fc7`).
Since (2026-09-26): the training-affecting ones fixed on branch `dev-training-followups` `dcbf6388` — not merged until the
rebuild after post-training stage 2 (user 2026-09-25) — say "fixed on a branch" in their rows and keep their entries until
it merges (24, the B77W preset, review #14 and #21, the performance index's B722 point); its review added three entries
(the last three rows).

| Entry | Status | What remains, or what resolved it | Fix affects training / post-training? |
|---|---|---|---|
| Training review 1 — every chart hover re-renders the whole app | resolved | cursor in its own context, 3D layer a leaf (`eac9c60a`); entry removed | — |
| Training review 2 — leaving Training throws its session away | resolved | panel kept mounted, session per airport (`bab2d5fe`, `2a368296`); entry removed | — |
| Training review 3 — `instruction_training_export.py` is runner and library | open | helpers still raise `SystemExit`; the backend still repeats `open_base_set`'s checks (`backend.py:73-95`) | no: the Training view's exporters and backend |
| Training review 4 — race on `training/index.json` | open | index written without re-reading (`:606-607`); the write loop is still not all-or-nothing | no: the Training view's exports |
| Training review 5 — replay draws no heading band for a dynamics failure | open | export still skips it (`executor_training_export.py:225-226`); no v3 overlay schema | no: the Training view's exports |
| Training review 6 — replay's off-word intercept count covers the whole flight | open | export and live backend can still show different counts | no: the Training view's exports |
| Training review 7 — three strictnesses of "the re-read sentence is the stored one" | open | no shared `require_stored_sentence` | no: the Training view's exports |
| Training review 8 — duplication across the exporters and the backend | open | all eight copies remain (a third `_git_state` at `instruction_spec.py:91`) | no: same results, fewer copies |
| Training review 9 — smaller exporter points | open | all four remain (`indent=2`, dead branch, unpinned params, two `captureBeforeThresholdM`) | no: the Training view's exports |
| Training review 10 — the prior's first predicted step has no exporter-side test | open | one case only; no prior-v3 overlay on disk | no: an exporter test |
| Training review 11 — `TRAINING_STRATA` not pinned | resolved | pinned by the backend's `MirrorTest` (2026-09-25); entry removed | — |
| Three ts ablation runners write into filter-less directories (09-24) | open | a rerun would still overwrite; the arm files refuse by name as recorded | no: old ablation runners |
| `build_runway_config.py` cannot rebuild `runway_thresholds.json` (09-20) | open | no NASR width / plate-minima source | no: unless the runway configuration is regenerated |
| Two suites fail at HEAD (09-20) | partly | the download test is fixed (`803605e0`); `test_optimizer.py:361` (numpy 2) and `test_write_reference_records_from_observed_tracks` still fail | no: tests only |
| ts: the closed-loop training's EXECUTOR side is not written (09-18) | obsolete | the trained executor it would retrain is archived; left: the dead hook `outputs/base.py:115` and a stale docstring (`strategy.py:176-178`) | — |
| ts: the manoeuvre runners repeat an argument skeleton (09-18) | obsolete | five of six runners archived (`79f871f1`) | — |
| ts: `predictability_report` builds its own dynamics rows (09-18) | partly | the `plan_z` crash cannot happen; the copy remains and never adds `cta_s` | no: a control-model readout |
| ts: `lead_landings` reads outer-test-hash landings (09-13) | open | still the owner's call | no: the old control models' intent input; the prior builds its own context and drops the sealed test days |
| ts: two dead loss helpers (09-07) | open | both remain with test-only callers (its correction on `DYNAMICS_CONDITION_NAMES` is resolved) | no: control-model training |
| ts: `train_only_diagnostics` is one unreferenced helper (09-07) | resolved | module deleted (`dddee9ae`) | — |
| ts: `experiments.pipeline`'s flags no longer match the ones it emits (09-07) | open | old flag names at `pipeline.py:1145/1169` | no: the control-model pipeline |
| 1. two byte-identical `_iso` | open | still two, plus a third inline at `harvest/merge.py:284` | no: same output |
| 2. `summary_row` writes explicit JSON nulls | open | unchanged | no: the optimizer's evaluation export |
| 3. `source_event_availability` re-validates its manifest | open | unchanged (judgement) | no: same output |
| 4. stale schema fixture in the comparison-CZML tests | open | fixture pins v2; the report is v9 | no: a test fixture |
| 5. `trajectory_data_process/tests` has 12 pre-existing failures | partly | all 12 fixed; the numpy `test_optimizer` failure remains | no: tests only |
| 6. `write_arrival_records` clears its output directory | open | still deletes the roster — now deliberate and a root Open Items hazard; `--observed-only` avoids it | no: the fix only protects the roster; nothing built changes |
| `run_all_tests.sh` has 13 pre-existing failures | partly | 11 fixed; the numpy test and the `arr_airport` test still fail; the exit code still says nothing; the script header is stale | no: tests only |
| 7. `roster_context_keys` leaks a raw FileNotFoundError | resolved | `records.py:298-300` | — |
| 8. `record_from_dict` does not enforce increasing `t` | resolved | `records.py:138-142` | — |
| 9. `arrival.py` restates `STATE_KEYS` | resolved | imported (`arrival.py:37`) | — |
| 10. empty `evaluate_batch` reports `mixed` | resolved | reports `empty` (`metrics.py:689-694`) | — |
| 11. last-sample kinematics under event names | partly | speed and heading now from the crossing; `final_time_s` still the last sample | no: a field's name, not a number |
| 12. `_reference_aggregate` unweighted mean of means | open | the comment is still missing | no: a comment |
| 13. eleven `test_ts_pipeline.py` reuse-guard failures | resolved | fixtures fixed (`ed708dea`, `803605e0`) | — |
| 14. A320-family speed windows exclude most crossings | resolved | superseded: the gate uses published VREF (`53ae8e80`); its residuals are obsolete | — |
| 2026-09-03 — 14 pre-existing failures in `trajectory_data_process/tests` | resolved | all 14 fixed; the suite is still not green for the two failures above | — |
| ts: reusable measurement code lives in `docs/` (09-07) | partly | conftest done; package runners still import `docs/` scripts; 18 `sys.path` preambles; hubs not migrated | no: readout scripts |
| ts: the auto-batch probe measures a smaller graph than a latent run | open | **worse than logged**: with `latent_dim > 0` and `--batch-size auto` the probe now raises (`latent.py:275`) | no: the control-model trainer; the prior trains with its own loop |
| scene data plane: review leftovers (09-07) | partly | (10) and (13) obsolete (archived); (7)-(9), (11), (12), (14) and the three test gaps open — `scene_context` has no live consumer | no: no consumer |
| ts: T2 leftovers (09-07) | partly | `needs_reference` resolved; `chart_scale=None` and the transport-chart entry points open; the open-items and README pointers half done | no: dead branches the executor never takes |
| 15. KRDU 14's arrivals render as "indeterminate" | open | needs the backend and a UX decision | no: the Observe view |
| 16. observed record without `landing_aero` cannot say why | resolved | removed with the v9 gate | — |
| 17. `require_matching_runway_data` has no production caller | open | still dead code | no: dead code |
| 18. `runway`-mode solves pile up at the window's upper edge | resolved | targets published V_ref (`88893126`); records solved before keep it until re-solved | — |
| 20. two copies of the point-mass inversion and the OLS slope | open | neither folded onto `aircraft/kinematics.py` | no: same formula; only a degenerate-fit slope could differ |
| 19. speed anchors calibrated on wind-contaminated ground speeds | resolved | the gate reads published speeds | — |
| 21. single-valued FAA approach-speed rows for multi-flap types | open, **blocked on a source** | no primary document publishes a reduced-flap V_REF at MLW for any of the five (searched 2026-09-25, `docs/literature/approach_speeds_reduced_flap/`); the entry's premise is doubtful — see the entry | **yes — executor**: the executor's approach speed for E75L, B737, A319, E170, E190 flights (25 % of the prior's train flights); the gate too |
| 22. four types publish no minimum operating mass | partly | LJ45 closed (`d5acb04f`); GLF5, C25A, C525 remain | gate only: the speed gate's mass range; the executor scales by √(m / MALW), not by the minimum |
| 23. the optimizer's velocity floor and V_ref target come from the stall model | partly | target fixed; the floor is still 1.10 × stall | no: the optimizer |
| 19. `control_basis_oracle --checkpoint` fingerprints without the roster | resolved | `e8df12f` | — |
| 20. `random_train_anchor=True` raises before the first epoch | resolved | `7e947bb5`, with a test | — |
| 21. the drawn "Lookback" is the whole pre-anchor track | open | no `lookbackSamples` record field yet | no: the comparison CZML |
| 24. OpenSky typecodes absent from Doc 8643 | fixed on a branch | `MARKETING_ALIASES` (H900, CL61, G450, G650, F2EX, F2LX; AS29 gliders left out) on branch `dev-training-followups` `dcbf6388`, merged with the rebuild after post-training stage 2 (user 2026-09-25) | **yes — data plane**: those flights would get a type, hence dynamics; the current sentence artefact refused for them until rebuilt |
| 25. 1,164 FAA airframes without an unambiguous typecode | partly | crosswalk `750daafb`: 2,737 → 914 unresolved; no third source; foreign rows unhandled | **yes — data plane**: as 24: types, hence dynamics, change |
| 26. `config.seq_len - 1` at sites that have `default_anchor` | resolved | `9d12b114`, `e2e1c84d` | — |
| 27. `cli/predict.py` builds records at near-identical sites | open | five copies remain | no: the control-model predict CLI |
| 28. bank inversion and load coordination in two hook modules | partly | coordination shared (`87f46cc6`); the inversion and its 0.5 lift floor still twice | no: command hooks; the executor runs none |
| 29. barrier's load coordination perturbs ungated rows | open | no guard at `barrier_filter.py:198` | no: command hooks; the executor runs none |
| 30. rolled-window table encoded once per window set | obsolete | plan head archived (`9dbb4921`) | — |
| 31. drop-stretch branch lays probes on the stretch | obsolete | archived with the plan head | — |
| 32. instruction-leg speed points keyed in the head's path-to-go | obsolete | archived with the plan head | — |
| 33. readers compare stored configs field by field | open | unchanged; its suggested fix (absent fields as defaults) now clashes with the no-compatibility rule | no: control-model config readers |
| 34. thrust-fraction speed floor inert by a rounding accident | open | the structural form not adopted for thrust-fraction | no: command hooks; the executor runs none |
| 35. the anchor-eligibility gate restates the stall speed | open | still inline with its own 1.225 | no: the control strategy's gate |
| 36. `batch_dynamics_tensors` duplicates the forecast's batch | open | a guard test exists; the copy remains | no: a control-model readout |
| 37. specific-force constants not in any checkpoint's identity | open | unguarded; its suggested fix also clashes with the no-compatibility rule | no: control-model checkpoints |
| 38. formulas left restated outside the contract rows | partly | `VerticalChannel` resolved; `CONTROL_NAMES` 3 → 2; guidance loop, lag credit ×5, cos γ ×3 remain | **yes — executor**: only its cos γ point: the transport-chart RHS the executor integrates (round-off in every flight); the rest no |
| 39. specific-force speed floor holds sin γ at the hold's start | open | unchanged | no: command hooks; the executor runs none |
| 40. two definitions of "the truth's end" | open | both conventions remain | no: ts readouts |
| ts: `EXPERIMENTS_MAIN` lives outside `repo_layout` (09-23) | open | not moved (the `HARVEST_ROOT` part is resolved) | no: a runner path |
| Review of trajectory_data_process + flight_scenarios (09-23), #1–#22 | open | 19 of 22 open; #17 partly (the named staging leftover is gone, nothing cleans such leftovers); #14 and #21 fixed on branch `dev-training-followups` `dcbf6388`, merged with the rebuild after post-training stage 2 (user 2026-09-25) | **yes — data plane**: #14 (the crossing scan without a fit) and possibly #10 change `build_series`; #19–#21 change types; #1–#4 re-harvest only; the rest no |
| `test_write_reference_records_from_observed_tracks` fails (09-23) | open | the fixture still lacks `arr_airport` | no: a test fixture |
| `THRESHOLD_SPEED_GATE.md` §3.3 quotes an unsourced +5/−0 kt margin (09-23) | open | neither sourced nor removed | no: a document |
| B77W preset `landing_mass` 19 % above its MALW (09-24) | fixed on a branch | every preset lands at its published MALW (B77W 251.3 t, A320 66.0 t, C172 1,111 kg) on branch `dev-training-followups` `dcbf6388`, merged with the rebuild after post-training stage 2 (user 2026-09-25) | **yes — executor**: B77W flights' mass and approach speed (√(m / MALW)) |
| Pilot frontend tests use a 145 / 135 / 155 kt fixture (09-24) | resolved | real catalog fixture + a range preset (`3d0fc579`, `d2323adf`); entry removed | — |
| Performance index: stage-2 review leftovers (09-24) | partly | B722's mass fixed (own types land at the published MALW, index schema v2) on branch `dev-training-followups` `dcbf6388`, merged with the rebuild after post-training stage 2 (user 2026-09-25); the observed mass label, the two readout scripts and the MD88 / LJ35 / GLF3 judgements unchanged | **yes — executor**: B722's mass, MD88 → B737, LJ35 → B737 and GLF3 → B763 change what those flights fly; the mass label, the 60 t observed fallback and the readout scripts: no |
| The native stall-margin range after the preset masses moved (09-26) | open | the index's decisions are still the 09-24 ones | **yes — executor**, if the rule is re-applied: B733, E550 and E545 would change decision |
| OpenAP-direct types land at OpenAP's MLW, not the published MALW (09-26) | open | unchanged | **yes — executor**: those types' mass and approach speed |
| An alias-resolved identity is not recorded as one (09-26) | open | unchanged | no: provenance only, the types are the same |

**Fix affects training / post-training?** — against what the two-tier chain runs today (the labeller's `instruction_signals`,
the executor `autopilot/` and its replay, `prior_train` / `prior_select` / `prior_free_generation`, the land-by-reward
post-training `prior_landing_reward`; traced from their imports, 2026-09-25): **no** — it touches nothing they run;
**yes — executor** — what the executor flies changes (its dynamics, speeds, or the RHS it integrates), so the replay gate
and the post-training's rewards change, and the sentence artefact, which records the aircraft tables, must be rebuilt;
**yes — data plane** — `build_series`' output changes, so the executor refuses the current sentence artefact (it rebuilds
each flight and demands the stored signals) until it is rebuilt, and the prior trains on the rebuilt one; **re-harvest
only** — the stored harvest changes only with a new download or rebuild (which changes every split: root CLAUDE.md);
**gate only** — the replay gate's evaluation numbers, not training; **—** — already fixed or obsolete.

## Training module review: what it found outside the module (2026-09-25)

The whole-module review of the Training view (frontend `aeroviz-4d/src/{data,hooks,components,scene}/training*`, the
live executor's backend `aeroviz_backend/autopilot_segment/`; branch `dev-autopilot-live`) fixed what was inside. These
are outside it — the app shell, the exporters, the prior — and are recorded, not changed.

Resolved since and removed (numbers kept, the items refer to each other): **1** a chart hover re-rendered the whole app,
**2** leaving Training threw its session away (both 2026-09-26, branch `dev-frontend-followups`: the cursor is its own
context, the 3D layer a leaf, the panel stays mounted), **11** `TRAINING_STRATA` unpinned (2026-09-25).

**3. `instruction_training_export.py` is a runner AND the Training files' library** — *verified*. It holds the schemas,
`KIND_*`, `SPLIT`, `stored_sentence` / `reread`, `open_base_set`, `read_overlays`, `write_overlay`; two runners and the
backend import it (the backend: `KIND_READBACK`, `SAMPLE_SCHEMA`, `SPLIT`), against "nothing imports a runner". Its
helpers raise `SystemExit`, which is not an `Exception`: from a server thread it would escape the handler and drop the
request unanswered — why the backend re-implements `open_base_set`'s checks instead of calling it. Fix: a torch-free,
non-runner `ts_transformer/instructions/training_files.py` raising `ValueError`; each `main()` turns those into
`parser.error`.

**4. A race on `training/index.json`** — *verified by reading*. `instruction_training_export.py:550-555` reads the index
at the start and writes `[*existing, entry]` after every airport is built, minutes later, without reading it again: a
set another export added meanwhile is dropped. `write_overlay` guards exactly this for `overlays.json` (`:508`). The
write loop is not all-or-nothing either, against its docstring (`:563`). Fix: one "append the entry, refuse if the file
changed" helper for both manifests.

**5. The replay draws no heading band for a dynamics failure, on a false premise** — *verified by reading*.
`executor_training_export.py:44-45` / `:225-226`, the TS rule `drawn = refused === null && outcome !== "dynamics_failure"`
(`trainingOverlays.ts`) and `test_training_overlays.py::test_a_dynamics_failure_keeps_its_verdicts_and_draws_no_band`
say "the judge read the failed state"; `judge.read_flown` reads to `end_row − 1` for a dynamics failure, and the
exported track holds those rows. The live executor draws the band since this review. Fix: a new overlay schema name
(`aeroviz-training-executor-v3`) on both sides that draws it, and a test with a real failure instead of a relabelled
outcome.

**6. The replay's "left to intercept the final on its own" counts the whole flight** — *verified by reading,
judgement*. The export fails the heading word in force at the FIRST off-word cycle and reports the flight's total count;
the live executor (since this review) counts only the cycles its word was in force. The two agree on which word fails
and may differ on the number shown. Fix, if wanted: count per word in the export too.

**7. Three strictnesses of "the re-read sentence is the stored one"** — *verified*. The executor export compares the
words grid only (`:421-424`); `replay.draw` and the live executor's `open_flight` compare the words and the runway index;
the instruction export also compares the capture, clearance and "unspecified" rows. Fix: one shared
`require_stored_sentence`.

**8. Duplication across the three exporters and the backend** — *verified*. `_r` (three copies), `_git_state` (a verbatim
copy of `repo_layout.git_state`), the repo-relative path written three ways, the stored-sentence slice inlined twice
although `stored_sentence` exists, the geoid-undulation wrapper three times, the two overlay `main()`s one skeleton, and
"the words in force" computed three ways plus the TS copy (`flight_payload`, `prior.data.sentence_steps`,
`autopilot.sentence._filled`). The backend keeps its own `_check` / `end_state_row` / chart shift rather than import a
runner (item 3).

**9. Smaller exporter points** — *verified*. `sample.json` is written with `indent=2` (one number per line in a
multi-MB file; the overlays are compact for that reason; no schema change needed). The `no_check_reason` "cleared at
entry" branch (`executor_training_export.py:108-110`) cannot be reached for truth sentences (a step-0 clearance is
written as kind "clear"; checked on all five `instruction_v3_day_split` samples). Unpinned: the executor `params`
values the TS reader accepts only as numbers or strings; `captureBeforeThresholdM` written from two sources (the
labeller's check and the display corridor) and never compared.

**10. The prior's first predicted step has no exporter-side test** — *verified*. The frontend now reads the truth there
as the word in force (`trainingOverlays.priorTruthAt`) and checks `truthP = changeP × wordsP` and `changeP = 1` on every
file; `test_training_overlays.py:188` covers one word said inside the observation window with the value written by
hand. Missing: a word said exactly at `N_LOOK`, two words of one column inside the window (the later must win), and
agreement with the sample's `words.inForce`. No `aeroviz-training-prior-v3` overlay is on disk yet, so the frontend's
new check has met only the fixture.

## Three ts ablation runners write into fixed directories that carry no aircraft filter (2026-09-24)

**Verified** (code review of the A320-fallback removal): `experiments/kinematic_ablation.py:514`,
`experiments/history_ablation.py:509` and `experiments/overfit_diagnostic.py` name their output
directory without the aircraft filter and create it with `mkdir(exist_ok=True)`. A rerun today
(default `all-flights`) would write into the directory the retired `all` run left, e.g.
`4dTrajectory/outputs/POOLED/ts_itransformer_kinematic_weight_ablation`. The pipeline's
directories carry the filter tag (C31); these three do not. The overwrite-on-rerun itself is
older than the change.

**Judgement**: give them the filter tag (or refuse an existing directory, as
`coordinate_ablation` does through `refuse_repeated_test`) the next time one of them is run.
Also: four arm declarations under `4dTrajectory/ts_transformer/docs/experiments/`
(`airport_frame_arms.json`, `airport_frame_seed2024_arms.json`, `final_constraint_arms.json`,
`state_v2_anchor_relative_arms.json`) set `"aircraft_filter": "all"` and `closure_p1c_arms.json`
sets `"aircraft_type"`; they are records of finished campaigns and now fail by name at
`arm_config` if re-run.

## `build_runway_config.py` can no longer rebuild `runway_thresholds.json` (2026-09-20)

**Verified** — the generator writes `name` / `length_ft` / `surface` / `thresholds` per runway.
The configuration on disk also carries `width_ft` and `runway_width_effective_date` (which
`load_airport` REQUIRES, and which come from FAA NASR, not from the OurAirports CSVs the
generator reads) and, since TD20, `published_minima`. Regenerating would have dropped the first
two silently long before this change; it would now drop three. The generator therefore refuses
outright when its output already exists, rather than writing a file the loader will reject.

**Judgement**: the fix is to give the generator the two sources it is missing — the NASR runway
widths and the plate minima (`extract_approach_minima.py` already produces the second) — so that
one command rebuilds the whole file. Until then the file is maintained by
`extract_approach_minima.py` for the minima and by hand for the widths, which is exactly the
"fix the GENERATOR, never the JSON" rule (TD8) being broken by necessity. Not urgent: the airport
set has not changed since the widths were added.

## Two suites fail at HEAD, outside the ts tree (2026-09-20)

**Verified** — `./run_all_tests.sh` on `dev-two-tier-feasibility` at `e74d5644`, with the working
tree touching only `4dTrajectory/ts_transformer/`, so neither failure comes from the instruction
vocabulary. Three modeling+backend tests fail, one of which the runner's own header already
documents as known and unrelated (the numpy scalar-conversion deprecation in
`test_fixed_time_objective_weights_control_effort_at_one`). The other two are NOT documented:

- `4dTrajectory/optimization/tests/test_scenario_optimization.py::test_write_reference_records_from_observed_tracks`
  — `ValueError: …_reference_eval.json: source.arr_airport must be a non-empty string`, raised by
  `evaluation/records.py:108`. The evaluation record contract gained a required `source.arr_airport`
  (report v7–v9 line, commits bb643d58 / 8bf9f5cc / 53ae8e80); the optimizer's reference-record
  writer does not fill it, so the test's fixture cannot be written any more.
- `trajectory_data_process/tests/test_download_landings.py::test_download_reuses_interrupted_checkpoint_start_for_cache_keys`
  — the interrupted-checkpoint start reads back as `None` instead of the stored timestamp.

**Judgement**: both look like a contract that moved without its writer, not flakes — the first one
names the missing field outright. Neither is on the ts_transformer path (nothing there writes an
evaluation record), which is why they are recorded here rather than fixed. Whoever owns the
report-v9 record contract should decide whether the optimizer's reference writer must carry
`arr_airport` or the contract should not require it on a reference row.

## ts_transformer: the closed-loop training's EXECUTOR side is not written (2026-09-18)

**Judgement** (manoeuvre-token plan §2.7 step 2): retraining the executor on its own flown legs
needs the training window set to serve a ROLLED window (the flown history) with TIME-INDEXED
truth targets (the truth's rows at the flown anchor's absolute times) — today
`WindowContext.override` returns a substitute window with ZERO targets (the plan path's v5.2
contract), so the control objective would have nothing to track. The change is in
`data/dataset.py` (`TrajectoryWindows.batch` / `override`) and is a spine change; the plan runs
this step only if gate E's first reading moves, so it waits for that reading. The prior side
(`manoeuvre_prior --rolled`) is done.

## ts_transformer: the manoeuvre runners repeat an argument skeleton (2026-09-18)

**Judgement**: `manoeuvre_readout` / `_prior` / `_prior_readout` / `_lockstep` / `_code_atlas` /
`_gates` each declare `--out` (refused if it exists), `--device`, `--limit` and the
absolute-or-REPO_ROOT path resolution by hand; the cohort rebuild is one function now
(`support.rebuild_cohort`), the flags could be one `add_manoeuvre_arguments(parser, …)` +
`resolve_out(...)`. Cosmetic; do it when a seventh runner arrives.

## ts_transformer: `experiments/predictability_report` builds its own dynamics rows (2026-09-18)

**Verified** (opus review of 9002cc0): it is the only model-forward path that builds dynamics
rows outside `ControlContext._build_row` / `_dynamics_batch`, so on a `manoeuvre-code`
checkpoint it dies with `plan_z`'s "a batch carries exactly one z source" — loud, and its
docstring now says so. Routing it through `_dynamics_batch` would make it work; nobody has
asked for that report on a manoeuvre arm.

## ts_transformer: `lead_landings` reads outer-test-hash landings as context (2026-09-13)

**Verified** (code read, runway-intent R0 review): `data/intent_conditioning.py::lead_landings`
takes EVERY `assigned` row of the tracks roster — outer-test-hash flights included — as the
candidate leaders of a requested (train / val) flight, so a test flight's landing time and runway
can become a development flight's `lead_landing`. Nothing trains on it by default (only
`intent_conditioning=truth-…` arms read it, and those read the future by design), and no
runway-intent R0 rule touches it — R0 builds its own pool without test-hash flights
(`data/runway_context.build_airport_context`).

**Judgement**: small, but it is a crack in "the outer-test split is never read"; the fix is to
drop test-hash rows (the checkpoint's `split_name_for_dataset_id`) from the leader pool — which
changes the lead of every flight whose true leader was a test flight, i.e. the stored truth-intent
arms' inputs, so it is the owner's call.

## ts_transformer: two dead loss helpers (2026-09-07)

**Verified** by AST walk over the module, on `dev-t3`: `objective.masked_mse` and
`objective.position_velocity_consistency_loss` have no live caller — only
`tests/test_state_objective.py` and `tests/test_windows.py` (one file, `test_ts_transformer.py`, until 2026-09-10). They moved with the rest of the objective in T3-15 rather
than being deleted there, because T3 is a structural pass and a deletion is a T1-shaped
change with its own evidence to state (the `kinematic` component is weighted zero on every
path today, which is why they went quiet). Deleting both would also remove four tests that
currently test nothing else.

**Judgement**: safe to delete, and it should be decided together with whether the
kinematic-consistency term is ever coming back.

**Corrected 2026-09-07 (T3 review)**: this entry also listed `dataset`'s unused
`DYNAMICS_CONDITION_NAMES` import as safe to delete. It was NOT — three tests read the name
through `dataset`, so removing the import alone cost seven failures. That is now fixed
properly (the tests import from `control.conditioning`, `dataset` no longer re-exports it),
and it is the reason this file states verified-vs-judgement: "unused inside this module" is
not "unreferenced", and only the second one licenses a deletion.

## ts_transformer: train_only_diagnostics is one unreferenced helper (2026-09-07)

**Verified**: `select_outer_train_series` was deleted in the T3 review — its only caller went
to `archive/oracle_teacher_2026_08/` in T2. What is left in the module is
`rank_outer_train_candidates`, which has no caller either, live or archived.

**Judgement**: the module should go with the runner cleanup (T4-27), not before — deleting a
file is cheap, but the split-discipline it documents ("open exactly one outer-train flight,
never val/test") is worth keeping in view while the capacity diagnostics are being reworked.

## ts_transformer: experiments.pipeline's flags no longer match the ones it emits (2026-09-07)

**Verified**: T3-19 renamed fifteen `ts_transformer` flags to the `TSConfig` field each sets.
`run_ts.py pipeline` emits two of them (`--control-state-supervision-clock`,
`--control-rollout-integrator-dt-s`) but keeps its own older names
(`--control-state-clock`, `--control-rollout-dt`) on its own CLI, with a comment at the
emission site saying so.

**Judgement**: renaming the pipeline's surface too would make one setting have one name
everywhere, at the cost of breaking a habit and a running campaign's command lines. Out of
T3's scope ("update every runner that SPELLS one of the renamed flags"); worth doing when no
campaign is in flight.

Opened 2026-08-17, during the `final_approach` / `evaluation` design pass.

---

## 1. `harvest/classify.py` and `harvest/store.py` define a byte-identical `_iso`

**Verified.** Both are:

```python
def _iso(time_s: float) -> str:
    return datetime.fromtimestamp(time_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
```

`classify.py:268` and `store.py:287`, same package. `store.py` additionally has
`_iso_precise` (millisecond `isoformat`). Timestamp formatting is identity-bearing in this
project — `landing_time_utc` is part of `flight_key` — so two copies of the format string
is exactly the kind of duplication that produced the `INBOUND_TOLERANCE_M` drift risk.

**Suggested:** one `_iso` / `_iso_precise` pair in `store.py`, imported by `classify.py`.
Low risk, no behaviour change. Check first whether the second-precision truncation is
deliberate in one of the two call sites before unifying.

## 2. `evaluation_export.summary_row` writes explicit JSON nulls

**Verified** (`4dTrajectory/optimization/evaluation_export.py:152` and the surrounding
block). Every field is built as `source.get(name)`, so a scenario that does not carry
`target_source`, `callsign`, `icao24`, … produces a roster row with the key **present and
null** rather than absent.

Why it is worth recording: `evaluation.arrival` was just fixed for precisely this Python
hazard — `dict.get(key, DEFAULT)` returns `None`, not `DEFAULT`, for a key present with a
null value, so an explicit null silently took a "this record aims elsewhere" branch. The
roster row is a different payload and no consumer currently keys a decision off its
`target_source`, so this is **latent, not live**. But the two shapes travel together, and
the next consumer to read the roster inherits the trap.

**Suggested:** decide per field whether it is required (index it) or optional (omit the key
when absent). Audited on the shipped artifacts: all 144,764 `*_eval.json` records carry
`target_source == "runway_threshold"` explicitly, so nothing on disk is affected today.

## 3. `harvest/observed.py::source_event_availability` re-validates its own manifest

**Judgement, not a defect.** The function interleaves deriving the availability block with
per-row `isinstance` validation of `tracks/manifest.json` — record shape, outcome
vocabulary, `event_status` presence, and every `source_integrity.excluded` entry.

This is the same shape as `evaluation.metrics._validated_observed_availability`, which was
deleted in the 2026-08-17 pass. The difference that matters: this one reads a **file**, so
validating is legitimate — a manifest can be stale or hand-edited. What is worth revisiting
is that the validation is spread through a derivation rather than done once at the read
boundary, so the manifest contract has no single place a reader can find it.

**Suggested:** if `tracks/manifest.json` gains a typed loader, move these checks into it and
leave this function doing arithmetic only. Not worth doing on its own.

## 4. Stale schema fixture in the comparison-CZML tests

**Verified.** `aeroviz-4d/python/tests/test_build_scenario_comparison_czml.py:539` builds a
report fixture pinned to `terminal-approach-evaluation-v2` — two schema generations behind
the current `v5`. It does not fail because `build_scenario_comparison_czml` copies
`schema_version` through into the comparison index without validating it, which is the
correct behaviour for a pass-through.

So this is a **documentation defect**: the fixture asserts a contract no producer has
written since v2, and a reader checking "what does the builder expect?" gets a wrong answer.
The same class of staleness is what let the v4 → v5 bump ship green past the frontend suite.

**Suggested:** either regenerate the fixture from the current producer, or add a one-line
comment saying the version is deliberately arbitrary because the builder does not read it.

## 5. `trajectory_data_process/tests` has 12 pre-existing failures

**Verified pre-existing** (reproduced with the working tree stashed, twice, on 2026-08-17).
Not a code-health item — recorded so nobody re-diagnoses it as fallout from an unrelated
change.

- `test_ts_pipeline.py` — 11 failures, all downstream of the same cause:
  `TrainingPlan.cv_reuse_error()` returns `"one or more lateral-pass eligibility rosters are
  missing"` where the tests expect `None`. The fixtures do not create the eligibility
  rosters that `experiments.pipeline` now requires.
- `test_download_landings.py::test_download_reuses_interrupted_checkpoint_start_for_cache_keys` — 1.

Separately, `4dTrajectory/optimization/collocation/tests/test_optimizer.py::
test_fixed_time_objective_weights_control_effort_at_one` tracks the numpy 2.x scalar-
conversion change and is already documented in `CLAUDE.md`.

**Suggested:** fix the fixtures (build the roster, or let the plan treat a missing roster as
"no eligibility filter") so the suite is green and a real regression is visible again. A
suite with 13 known-red tests cannot report a 14th.

**Checked 2026-08-17:** rebuilding the five real `lateral_pass_eligibility.json` rosters did
NOT fix these — the tests build their own `tmp_path` trees, so the cause is in the fixtures,
not in missing production artifacts. Still 12 failed / 30 passed afterwards.

## 6. `write_arrival_records` clears its output directory, taking co-located artifacts

**Verified.** `harvest/arrivals.py` starts with `_clear(root)` on `arrivals/`, and
`lateral_pass_eligibility.json` lives in that same directory. So any `--evaluate-only` or
`--reclassify-existing` run silently deletes the eligibility roster, which the ts pipeline
then reports as "one or more lateral-pass eligibility rosters are missing" — a message that
points at the roster rather than at what removed it.

Observed directly: a KRDU `--evaluate-only` run left KRDU as the only airport of five
without a roster.

**Suggested:** either write the roster somewhere the arrivals rebuild does not own, or have
the clear step preserve known derived siblings, or have the harvest rebuild the roster after
writing arrivals. Low risk either way; the roster is cheap to regenerate
(`ensure_lateral_pass_roster`).

## `run_all_tests.sh` has 13 pre-existing failures, and they hide new ones

**Verified** (2026-08-20, clean `git worktree` at `84725d6`, i.e. the commit this session
started from — so none of them come from the `simple-v3` work):

- `trajectory_data_process/tests/test_ts_pipeline.py` — **11 failures**. The runner's printed
  summary no longer matches what the tests assert: they expect lines like
  `loss      : final_time=1, kinematic=3, terminal=0.02` and a `config    : TSConfig defaults`
  block, while the runner now prints a different banner. Also affected: the CV default grid,
  fixed/random anchor artifact paths, and five `--skip-train` checkpoint-rejection cases.
  Most likely stale since `dd6191d` / `13d14a6` changed the config and CLI surface — the
  tests were not updated with them.
- `trajectory_data_process/tests/test_download_landings.py::test_download_reuses_interrupted_checkpoint_start_for_cache_keys`
  — 1 failure, not investigated.
- `4dTrajectory/optimization/collocation/tests/test_optimizer.py::test_fixed_time_objective_weights_control_effort_at_one`
  — 1 failure, already documented in `4dTrajectory/CLAUDE.md` as a numpy 2.x regression.

**Re-verified 2026-09-08** (main tree at `1db22dd`, after the `dev-observed-load-factor-metar`
merge and the `cli/freeze.py` rename that restored ts_transformer test collection): **15
failures**, the 13 above plus two that are new since 2026-08-20 and both predate the merge
(their causing commits are ancestors of `787e9a2`):

- `trajectory_data_process/tests/test_ts_pipeline.py::test_prediction_labels_distinguish_coordinate_frames`
  — the twelfth pipeline drift: asserts `'ENU' in plan.label`, the label now reads
  `state · iTransformer · kinematic · state-v1 · pooled cohort`. Same family as the eleven.
- `4dTrajectory/optimization/tests/test_scenario_optimization.py::test_write_reference_records_from_observed_tracks`
  — `evaluation.records` now refuses a reference record whose `source.arr_airport` is empty
  (`records.py:108`, from the speed-gate work `bb643d5`..`53ae8e8`), and the optimizer's
  `write_reference_records` fixture never set it. Either the writer must carry `arr_airport`
  (the record contract says it should) or the fixture is stale — not investigated which.

Every ts_transformer test passes in the same run (they had been uncollectable, not failing,
while `cli/freeze_test.py` matched the `*_test.py` pattern); `aeroviz-4d/python` 155 passed.

**Re-verified 2026-09-13** (worktree off `dev-leg-ctrl` at `d654872`, experiment-picker change;
neither failing test imports a changed module, and both fail when run alone): **3 failures,
1872 passed** — the numpy one, `test_write_reference_records_from_observed_tracks` (same
`source.arr_airport` refusal) and `test_download_reuses_interrupted_checkpoint_start_for_cache_keys`.
The twelve `test_ts_pipeline.py` failures no longer appear; `aeroviz-4d/python` 157 passed.

**Judgement**: the eleven are assertion drift rather than broken behaviour — the pipeline
itself runs — but that is inferred from the assertion text, not from exercising the runner.

**Why this matters more than the count suggests**: `run_all_tests.sh` exits 1 either way, so
its exit code currently carries no information, and a *new* failure in the modeling suite is
invisible unless someone diffs the failure list by hand. The per-subsystem suites are still
clean (`ts_transformer` 366 passed, `aeroviz-4d/python` 154 passed), so the working practice
is to run those directly and treat the aggregate script as advisory until this is fixed.

---

Opened 2026-08-23, during the `evaluation` review that accompanied the threshold speed
gate (schema v6).

## 7. `evaluation.records.roster_context_keys` leaks a raw FileNotFoundError

**RESOLVED 2026-09-07** (`dev-evaluation-review-fixes`): a missing manifest returns `None` and
`record_files` raises the crafted message; a roster ROW lacking `arr_airport`/`runway` now
raises instead of silently falling back to materializing the batch.

**Verified** (reproduced): `python -m evaluation --input <dir-without-summary.json>`
raises `FileNotFoundError: .../summary.json` with a bare traceback. The crafted message
for exactly this case lives in `record_files` ("has no summary.json manifest; pass a
record file for a loose record"), but the CLI calls `contexts_from_roster` →
`roster_context_keys` FIRST, and that function reads `p / "summary.json"` without an
existence check (`records.py:227`); `_load_json` only converts `ValueError`.

**Suggested:** in `roster_context_keys`, return `None` (the documented "roster cannot
name them" outcome) when `summary.json` does not exist — the CLI then falls through to
`record_files`, which raises the intended message. Error-path quality only; no verdict
can change.

## 8. `record_from_dict` does not enforce strictly-increasing `t`

**RESOLVED 2026-09-07**: checked in the `_state` loop (`evaluation/tests/test_records.py`).

**Verified**: a record whose `states` carry `t = [0, 10, 5]` and `final_time_s = 5.0`
passes `record_from_dict`. The contract says `t` is "the one field with hard contracts —
`final_time_s == states[-1]['t']` to 1e-6, and strictly increasing offsets"
(`evaluation/CLAUDE.md`, `evaluation_export.py`), but only the first half is validated
at the boundary; producers enforce ordering on the write side only. A non-monotonic
record would corrupt `flight_time_delta_s` and the interpolated crossing's `t` silently.

**Judgement:** add `t[i] > t[i-1]` to the `_state` loop in `record_from_dict` (one pass,
already iterating). Cheap, and it makes the documented invariant structural on the read
side too.

## 9. `evaluation/arrival.py` restates `STATE_KEYS` as a literal tuple

**RESOLVED 2026-09-07**: `_TIMED_STATE_KEYS = ("t", *STATE_KEYS)` imported from `records`.

**Verified**: the crossing-interpolation dict comprehension iterates
`("t", "lat", "lon", "alt", "V", "psi", "gamma", "m")` (`arrival.py`,
`_computed_arrival`) — a mirror of `records.STATE_KEYS` restated without a mirror
comment. The project rule is "a schema literal in a consumer is a mirror — import it"
(`CLAUDE.md`). Adding a state key would silently skip it in interpolated crossings.

**Suggested:** `("t", *STATE_KEYS)` imported from `evaluation.records`.

## 10. `evaluate_batch` on an EMPTY iterable reports `subject: "mixed"`

**RESOLVED 2026-09-07**: an empty batch reports `subject: "empty"`.

**Verified**: with zero records, `sorted(subjects)[0] if len(subjects) == 1 else
"mixed"` takes the else branch, so an empty batch serializes `subject: "mixed"`,
`solve_rate: 0.0`. Not reachable through the CLI (`record_files` raises on an empty
roster), only via direct library calls.

**Judgement:** label it `"empty"` or raise; cosmetic until someone streams a filtered
generator that comes up empty and reads "mixed" as two subjects having been present.

## 11. Observed rows publish last-sample kinematics under event-flavoured names

**Judgement** (mechanism verified, impact assessed): for observed records the
deviation's `speed_ms`, `heading_rad`, and `final_time_s` are LAST-SAMPLE quantities
(median 325 m before the threshold), while `cross_track_m`/`vertical_m` in the same row
are event-based estimates AT the threshold. `_row` publishes all five flat with no
distinction, so a consumer can read `speed_ms`/`final_time_s` as crossing quantities.
The v6 speed gate sidesteps this (observed `crossing_speed_ms` is `None`), but the
pre-existing three fields keep the ambiguity.

**Suggested:** either rename on the row (`final_sample_*`) for observed subjects or
document the split in `methodology.event`; renaming touches report consumers, so it
should ride the next schema bump.

## 12. `_reference_aggregate` reports an unweighted mean of per-flight means

**Judgement**: `path_lateral_m.mean` in the batch reference aggregate is
`fmean(per-flight means)` — every flight weighs equally regardless of its resample
count (fixed at 101, so today the two definitions coincide; the p95 is dropped
entirely at the aggregate level). Worth a one-line comment stating the weighting so a
future variable-N resample does not silently change the metric's meaning.

## 13. Eleven `test_ts_pipeline.py` reuse-guard tests fail at HEAD: fixtures predate the roster requirement

**Verified** (2026-08-24, while running the fleet reclassify). `TrainingPlan.cv_reuse_error`
and `checkpoint_reuse_error` in `run_ts.py pipeline` require every airport's
`lateral_pass_eligibility.json` beside its arrival manifest, but the hermetic tests
(`monkeypatch HARVEST_ROOT` → tmp dir) build their harvest fixture with `_manifest(...)`
only — no roster — so 11 tests fail with "one or more lateral-pass eligibility rosters
are missing" on a clean tree, independent of disk state. (Three OTHER tests in the same
file read the REAL harvest root and were failing for the opposite reason — the rosters
really were missing after the 2026-08-21 re-roster; rebuilding them via
`ensure_lateral_pass_roster` fixed exactly those three, confirming the split.) A few of
the 11 fail on assertion text that has drifted for other reasons (candidate counts, loss
line format), so this is fixture rot, not one missing file.

**Suggested:** give `_manifest` a companion that also writes a minimal roster (or have
the fixture call `ensure_lateral_pass_roster` against a stub approach report), and
re-derive the drifted assertion strings. Owned by whoever is actively working the ts
pipeline; not fixed here to avoid colliding with in-flight changes.

Same family, found 2026-08-24:
`test_download_landings.py::test_download_reuses_interrupted_checkpoint_start_for_cache_keys`
writes a checkpoint fixture with `"version": 2` (correct when `3008baf` bumped it on
07-23) but `runner.CHECKPOINT_VERSION` is now 4, so `checkpoint_start` rejects the
fixture and the test fails at HEAD. The fixture should import the constant instead of
restating it — the version-pinned-in-a-test trap, again.

## 14. A320-family speed-gate windows exclude most of the real fleet's crossings

**Verified** (2026-08-24, first fleet baseline speed-gate measurement —
`evaluation/docs/BASELINE_SPEED_GATE_RESULTS.md` §5). Speed-fail structure clusters by
manufacturer, which weather cannot produce: A21N 91.9 % too-slow (0 fast) of 881
graded, A20N 66.4 %, A321 58.2 %, A319 53.2 %, A320 44.7 % — while every 737-family
type passes 75–84 % with a mild FAST skew (B738/B38M ≈18 %), same days, same
airports. A window whose floor excludes 92 % of a type's real crossings measures the
window, not the fleet: the `aircraft` package's landing `Cl_max` and/or OpenAP
landing mass for the A320 family place 1.23·Vs1g above real crossing speeds
(conversely B738/B38M/C56X anchors may sit slightly low).

**RESOLVED 2026-08-24** (`6e31f2d` + republished results in
`BASELINE_SPEED_GATE_RESULTS.md` §8): A320-family landing Cl_max calibrated to 3.0
(from Airbus's VLS = 1.23·Vs1g + published VLS figures, pinned by
`aircraft/tests/test_aero_anchors.py`); C56X airframe facts restored from the C550
surrogate's to certificated values. Post-fix the manufacturer cluster is gone
(A319/A320/A321 pass 78.9–85.4 %; C56X fast-fail 53.5 → 20.7 %). REMAINING
residuals, each needing its own evidence: **A21N 66 % slow / A20N 27 % slow** (the
neo subfleet lands far below the MLW the window anchors on — needs an operational
landing-mass model, not a Cl tweak), **B763** 34 % slow (heavy-bucket Cl 2.4
uncertain, n = 239), **E75L** 22 % slow. Per-TYPE quoting remains the rule for the
residual types.

## 2026-09-03 — 14 pre-existing failures in `trajectory_data_process/tests`

**Verified** (reproduced at commit `abfb8b9`, before the airport-frame work, in a throwaway
worktree; identical list on the current tree). `test_ts_pipeline.py` (13): the tests assert
the pre-2026-08-24 label grammar (`'ENU' in label`, `'loss      : final_time=1, …'`,
`'position/local-velocity'`), a `(27 candidates)` CV grid that is now 45, and
`checkpoint_reuse_error()` / `cv_reuse_error()` on fixtures that carry no
`lateral_pass_eligibility.json` (the roster check was added later and fires first).
`test_download_landings.py::test_download_reuses_interrupted_checkpoint_start_for_cache_keys`
(1): expects a checkpoint start time that comes back `None`. None of these is exercised by
`run_ts.py frame_ablation`, which calls `__main__.py` directly.

**Suggested:** rewrite the label assertions against `run_naming.run_display_name`, give the
reuse-check fixtures a roster (or assert the roster message first), and re-check the
download checkpoint contract. `run_all_tests.sh` currently exits 1 on these plus the known
numpy-2.x optimizer failure, so a green run needs them addressed or quarantined.

## ts_transformer: reusable measurement code lives in `docs/` (verified, 2026-09-07)

`docs/` holds ~12 runnable `.py` files, several of them libraries other scripts import
(`compare_frame_arms`, `phase0_intent_diagnostics`, `p1_closure_oracle`, `closure_*` helpers
are already in the package but the readouts are not). Consequences seen: no tests, no `conftest.py` at the time (25 of 32 test files still
carry their own `sys.path` preamble; the seven newest do not), and package modules that
cite a `docs/` script as their data producer (`closure_output.py`, `config.py`,
`__main__.py` all name `docs/p1_closure_oracle.py labels`).

Done so far: `strata_masks` + `STRAIGHT_TORTUOSITY` moved into `data/approach_difficulty.py`
(one source, `compare_frame_arms` imports them); `tests/conftest.py` added; the L0 basis
fit went in as `control/basis_fit.py` (`control/oracle/basis.py` until T2) + `run_ts.py control_basis_oracle` with tests.

Remaining: the migration table in `4dTrajectory/ts_transformer/docs/2026-09-07_package_audit_plan.zh.md`
§七 (hubs first: `p1_closure_oracle`, `compare_frame_arms`, `score_control_arms`), and the
25 per-file `sys.path` preambles in `tests/`.

## ts_transformer: the auto-batch probe measures a smaller graph than a latent run trains (review finding, 2026-09-07)

`batching._heterogeneous_control_probe_prediction` downcasts to a plain `ControlPrediction`, so
with `latent_dim > 0` the batch-size probe never builds the posterior encoder or the KL graph;
the resolved batch size is measured against less memory than training uses. Judgement: small
(the latent adds two linear layers), but a probe that lies is a probe; fix = let the probe run
the model's real forward with a synthetic future when `consumes_future`.

## scene data plane: review leftovers (opus, 2026-09-07; the HIGH/MEDIUM items are fixed)

Verified by the review, deferred here because the scene encoder is not being built (L4 gate
failed on the measurement the plane was built for): (7) `scene_context` mirrors the on-final
membership constants instead of importing them upward from `final_approach_geometry`
(pinned by a test today); (8) `ego_alt_hae_m` is a required argument nothing consumes — a
datum trap; (9) the ego's ETA uses the caller's ground speed while each neighbour's uses a
two-sample finite difference, and straight-line `distance/|v|` gives an outbound aircraft
a finite ETA — the observable-lead-ETA proxy the L4 diagnostic found uninformative (corr
0.11 with the true landing) would need an along-path estimator anyway; (10) the
`since_last_landing` / `lead` sentinels collide with real values (3.0 % of KRDU anchors clip
at 3600 s, 8 of them mean "no landing yet") — a validity bit per column; (11) `hour_utc` /
`weekday` are UTC, which conflates time zones under merged-airport training; (12)
`max(dt, 1e-3)` turns a duplicate timestamp into a 1000× speed; (13) no per-window scene
cache — 15.7 ms per scene, ~9 track reads each, would be rebuilt every epoch and every
DataLoader worker holds its own reader cache; precompute `SceneArrays` per (flight_key,
anchor) before any training uses them; (14) unused imports and two one-line duplicates
(`parse_utc_s` ≡ `intent_conditioning.parse_utc`, `OUTCOME_ASSIGNED`). Test gaps: a
neighbour that landed just before t₀ with samples still in the window; an airborne
neighbour on the parallel runway's final; an ego already established.


---

## ts_transformer: T2 leftovers (package audit, 2026-09-07)

Noticed while archiving the oracle teacher and the nominal-law hook; none of them belongs to
that change.

- **No live `CommandHook` declares `needs_reference = True` any more (verified).** The
  nominal law was the only one, and it is archived. The protocol field survives in
  `outputs/dynamics/hooks.py`, and so does the machinery it drives — `track_reference` in
  `aerodynamic_model/torch_piecewise_rollout.py` (an extra unhooked endpoint rollout per
  segment) and its pass-through in `torch_lag_dynamics.py` and
  `outputs/dynamics/backends.py:FirstOrderLagBackend._hooked_schedule`. It is not dead
  (the barrier reads `state.actuators`, and the combined lateral/vertical hook that
  `OPEN_ITEMS` still lists would need the reference back), but nothing exercises the
  `True` branch outside `aerodynamic_model`'s own tests. Decide before T3-20 folds the
  backend classes into a table.
- **`aerodynamic_model`'s lag kernels still accept `chart_scale=None` (verified).** T2
  deleted the only caller that passed it (the `(first-order-lag, transport-chart-velocity)`
  registry entry); `FirstOrderLagBackend.chart_scale` is no longer optional on this side,
  but `lag_state_scale` / `rollout_piecewise_constant*` in
  `aerodynamic_model/torch_lag_dynamics.py` keep the branch. Cross-package, so left alone.
- **`aerodynamic_model/torch_transport_chart_dynamics.py`'s two rollout entry points have no
  non-test caller repo-wide (verified in the T2 review).** `rollout_piecewise_constant` and
  `rollout_piecewise_constant_at_times` were reached only through the deleted
  `TransportChartVelocityBackend`; the chart's other exports
  (`transport_chart_state_to_channels` / `_to_geodetic`) are still read by
  `outputs/dynamics/backends.py`, and the lag and scaled kernels have their own rollouts. The
  module's own tests still exercise both, so the suite does not notice. Same cross-package
  call as the `chart_scale=None` entry above.
- **`docs/open-items.md` line 79 and `README.md`'s "Known gaps" still narrate the
  nominal-law hook as a live option (judgement).** Both sit under banners that mark their
  section historical, and neither names a module path, so T2 did not touch them; a reader
  skimming the root status file could still take "the vertical complement" as something
  that can be run today.
- **`docs/control_parameter_prediction.zh.md` is now mostly a map of archived code
  (judgement).** Its §4/§5/§8 call graph, module-status table and script index describe the
  teacher chain; T2 banners it rather than rewriting it, but `README.md` and `CLAUDE.md`
  still point at it as the authority for "which `control/` modules are live". T3/T4 should
  either re-derive that table or demote the pointer.
---

Opened 2026-09-07, during the `evaluation` review (`evaluation/docs/2026-09-07_review_fix_plan.zh.md`).

## 15. KRDU 14's 13 arrivals still render as "indeterminate" in the Observe view

**Judgement.** `evaluation/docs/UNJUDGED_RUNWAY_VERDICT_GAP.md`: the evaluation half is fixed
(KRDU 32 / KSMF 35R now carry an LNAV/VNAV vertical path), so the "never judged" bucket is
down to KRDU 14, which has no RNAV procedure at all. The frontend still paints those flights
the same grey as a judged-indeterminate one; the fourth UI state / surfaced skip reason is a
product decision the gap document lays out. 13 flights, so it is cosmetic until another
procedure-less runway enters the fleet.

## 16. An observed record without `landing_aero` cannot say WHY — RESOLVED BY REMOVAL (2026-09-07)

The v9 gate keys on `source.aircraft_type` and no longer reads `landing_aero`; an observed
record without a type reads "airframe could not be resolved", which is now the only way the
key goes missing (the harvest writes it whenever the icao24 resolves).

**Judgement (original).** `harvest/observed.py` omits the key when the icao24 does not resolve, and a
record written before 2026-08-24 lacks it for every flight. Evaluation reports both as
"airframe could not be resolved" — for a stale batch that reason is wrong, and unlike the
`crossing_span` case there is no cure-naming raise. Fix on the producer side: write
`landing_aero: null` plus an explicit `airframe_resolution` outcome, and let evaluation treat
an ABSENT key as a stale artifact. Deferred because every current observed batch would need
`--evaluate-only` (which the root `CLAUDE.md` warns rewrites the v5 arrivals roster).

## 17. `harvest.airports.require_matching_runway_data` has no production caller

**Verified** (2026-09-07, by the M1 review): `runway_data_fingerprint` is written into the
reclassify / freshness-rebuild manifests as provenance and never checked; the checker
`require_matching_runway_data` is dead code. Pre-existing. Either wire it (the reclassify
reader is the natural place) or delete it; note the digest hashes every `Runway` field,
so it moved for all 26 runways when `tch_source`/`baro_vnav_minima` were added
(`test_approach_verticals.py` pins that as expected).

## 18. `THRESHOLD_SPEED_GATE.md` §7: the upper edge is where `runway`-mode solves pile up

**RESOLVED for new solves (2026-09-23)**: the target is the type's published approach speed
(`flight_scenarios/docs/population_reference.md` FS5), not a class constant. The records on disk
were solved against the old 145 kt target and keep this edge sensitivity until re-solved.

**Verified** (2026-09-07 measurement). With the A320 family's calibrated `Cl_max` the 64.5 t
window is 126.7–146.7 kt and the class-default target of 145 kt sits 1.7 kt under the top, so
the ~15 % "fast" fails in `KMSY/runway` are decided by sub-knot replay drift. The real fix is
per-type `reference_speed_kt` derived from `1.23·Vs1g(landing_mass)` instead of a class
constant (already followup-listed under the E75L case); until then quote fast-fail rates with
the margin.

## 20. Two copies of the point-mass inversion and of the OLS slope, deferred under a campaign

**Verified** (2026-09-07, opus review of `dev-observed-load-factor-metar`).
`4dTrajectory/ts_transformer/geometry/flyability.py:259-262` computes `n` from `psi_dot`/`gamma_dot`
with the same two lines `aircraft/kinematics.load_factor_from_rates` now owns (it also
needs `mu`, so the shared function should return the two components), and
`flight_scenarios/start_state._slope` is `aircraft.kinematics.linear_slope` returning 0.0
instead of raising on a degenerate abscissa. Both live on the ts dataset/eval path that a
formal campaign was running from, so they were left untouched; fold them onto the shared
functions between campaigns (behaviour-identical for validated records).

## 19. Per-type speed-window anchors were calibrated on wind-contaminated ground speeds — RESOLVED (2026-09-07, v9)

The anchors are no longer calibrated at all: the gate reads the type's PUBLISHED approach
speed (`aircraft/reference_speeds.json`, provenance in `docs/reference_speeds/README.md`);
`evaluation/docs/THRESHOLD_SPEED_GATE.md` §3.1 and `BASELINE_SPEED_GATE_RESULTS.md` §10.

**Verified (original)** (2026-09-07, `evaluation/docs/BASELINE_SPEED_GATE_RESULTS.md` §9). The A320
family's landing Cl_max 3.0 (§8) was derived from proxy ground speeds that the METAR
correction now shows were reading 5–8 kt slow; with the wind in, the Airbus family sits
mid-window and the 737 bucket (Cl_max 2.7) crosses at a median 1.37–1.41 × Vs1g(MLW), on
the +20 kt edge, 40 % "fast". Re-derive the per-type anchors on the corrected airspeed
(and the A21N landing mass) before quoting any observed speed-fail rate; owner item O4.

## 21. Single-valued FAA approach-speed rows for multi-flap types (E75L, B737, A319, E170, E190)

**Verified** (2026-09-07, `evaluation/docs/BASELINE_SPEED_GATE_RESULTS.md` §10.3). The FAA
Aircraft Characteristics Database gives ONE approach speed (the maximum-flap value) for these
types where the dual-value rows (B738 140/144, CRJ9 132/141) carry the reduced-flap speed too.
Their observed rows produce the v9 gate's residual "fast" clusters (E75L 302 of 3,401, B737
221 of 5,537), 2–7 kt over an upper edge built from the other flap setting; Eurocontrol's Vat
for the same types is 6–7 kt higher. Rule A keeps the FAA value. The fix is a PUBLISHED
reduced-flap V_ref at MALW for each (manufacturer FCOM/QRH excerpt or an FSB report) added as
`approach_speed_max_kt` with its source in `aircraft/reference_speeds.json` +
`docs/reference_speeds/README.md` — a data change, no code. Do not widen the window instead.

**Searched 2026-09-25 — no source** (`docs/literature/approach_speeds_reduced_flap/`, 22 documents: Airbus/Boeing/Embraer
airport-planning documents, FAA/EASA/Transport Canada evaluation reports, TCDS, NTSB dockets, ANP): none publishes a
reduced-flap V_REF at MLW for any of the five; the Airbus AC gives the A319's CONF FULL value only (126 kt at 62,500 kg),
Boeing a 737-700 132 kt with no flap stated. **The premise is doubtful** (*judgement*, from verified text): the FAA ACD's
data dictionary (note 2) CALCULATES the second speed when a report gives two categories and one speed, 14 CFR 97.3 bounds
category C below 141 kt, and every Boeing dual row in `reference_speeds.json` (B37M, B38M, B39M, B738, B739, B788, B789)
has a lower value of exactly 140 kt — the category edge, not a flap setting. Before any speed window is built on those
dual rows, re-check their "flap reading" notes; for the five types here, nothing to add until a primary source appears.

## 22. Four types publish no minimum operating mass (GLF5, C25A, LJ45, C525)

**Verified** (2026-09-07). 404 observed rows grade speed-indeterminate with the type named
(`speed_indeterminate_reasons`). The manufacturers' public pages returned 403/404 at retrieval
time (`docs/reference_speeds/README.md` "Types with no published minimum mass"); a TCDS does
not state OEW. A published BOW/OEW (spec sheet, APM) for each closes it as a cited JSON row.

## 23. The optimizer's velocity floor and V_ref target still come from the stall model

**Half resolved (2026-09-23)**: the V_ref target and the floor's cap now read
`aircraft.reference_speeds` (FS5, `optimizer_reference.md` K10). The floor itself is still
`1.10 × stall_speed_ms` on the stall model, so the gaps below still apply to the floor.

**Verified** (opus review, 2026-09-07). Since v9 the evaluation gate anchors on the published
approach speed while `4dTrajectory/optimization/scenario_optimization.py` floors velocity at
`1.10 × stall_speed_ms` on `aircraft.aero_params` and targets the class-default 145 kt. Measured
gap 1.23·Vs1g(MALW) − published, over the 39 OpenAP-resolvable types: LJ45 −73 kt, GLF5 −37,
GLF6 −33, C525/C25A −31, GL5T −27, A306 −16, C550 −16, C56X −15, CRJ2 −13, B762 −9, A320 −8.5,
B788 −8; +8 B735, +6 B734 — the bizjet rows fly fallback A320 dynamics, so a floor-riding
solve of one fails v9 by 30–70 kt. Feed the floor's reference and the target V_ref from
`aircraft.reference_speeds` (the same table the gate reads) when the optimizer is next
re-solved; do not widen the gate (`THRESHOLD_SPEED_GATE.md` §7).

## 19. ~~`run_ts.py control_basis_oracle --checkpoint` fingerprints the data without the eligibility roster~~ FIXED (`e8df12f`)

**Verified** (2026-09-07, hit while building `run_ts.py anytime_curve`): the teacher-table mode
called `require_matching_data_provenance(payload, arrival_data_provenance(manifests))` with no
`eligibility_rosters`, while every current checkpoint's `data_provenance` is
`ts-arrival-data-v3-eligibility-bound` and carries the pre-split lateral-pass roster. The
fingerprint taken without the roster lists all 14 435 KRDU arrival candidates where the
checkpoint carries the 14 378 eligible ones, so the strict comparison failed with "checkpoint
training data does not match the current arrival manifests" — on a checkpoint and a manifest
that are both correct. Reproduced against `l1_lowdim_20260907/L1_native32`,
`l2_warm_posterior_20260907/L2d_warm_beta0p01` and `closure_p1c_20260905/C_pred`; it was a live
blocker for **L5.a**, whose documented fit died at the check before fitting anything.

**Fixed 2026-09-07 by `e8df12f`**: the rule has one owner,
`data_provenance.checkpoint_data_provenance(payload, manifests)` — the roster is read iff the
checkpoint recorded one, so a checkpoint trained before the sidecar existed is not handed one.
The fitter uses it for both the check and the table's stamp; `run_ts.py anytime_curve` calls the
same helper (its own copy of the rule was deleted when `dev-l2` merged). Kept here as the record
of why the helper exists: **a new replaying runner that calls `arrival_data_provenance` directly
reintroduces this**, and a test that patches that function is what hid it the first time.

## 20. `random_train_anchor=True` training raises before the first epoch

**Verified** (2026-09-07, hit while building the A0 arm-label test): `train.fit_model` builds
its training window set as

```python
training_dataset_class = {False: FixedAnchorTrajectoryWindows,
                          True: RandomAnchorTrajectoryWindows}[config.random_train_anchor]
train_set = training_dataset_class(..., fitted_teacher=fitted_teacher)
```

but `RandomAnchorTrajectoryWindows.__init__` takes no `fitted_teacher`, so any
`random_train_anchor=True` run dies with `TypeError:
RandomAnchorTrajectoryWindows.__init__() got an unexpected keyword argument 'fitted_teacher'`
before a single epoch. Introduced by `e9e3639` (L5.a: the fitted teacher became a
training-time input); no current recipe uses random anchors, which is why the suite is green.

It is a **prerequisite for the A0-random arm** (anytime design §2.2): that arm is exactly
`random_train_anchor=True` + the L1.b teacherless supervision, and §六 3 says the fixed arm's
curve must not be published alone. The fitted teacher is refused with random anchors anyway
(`TSConfig`), so the fix is to pass it only to the fixed-anchor class, or to accept and
refuse it in the random one — one line either way, plus a test that constructs the random
window set through `fit_model`.

## 21. The drawn "Lookback" is the whole track before the anchor, not the model's input window

**Verified** (2026-09-07, found reviewing the anytime record publication):
`build_scenario_comparison_czml._lookback_states` returns every observed sample with `t <= 0`
and the entity is named `Lookback`, coloured `LOOKBACK_COLOR` and labelled *"Predictor input"*
in the frontend legend. That is exact at the L−1 anchor — the record's series starts at the
arrival window and the anchor is `seq_len − 1`, so "everything before the anchor" IS the
`seq_len`-sample input window — and it is what every batch published before 2026-09-07 was.

A re-anchored anytime record (`run_ts.py anytime_curve --write-records`) breaks the
coincidence: at the 12 km bin the anchor is sample ~207–216 of a 60-sample lookback, so the
faded line draws ~430 s of approach under a name that claims the model was conditioned on all
of it. Nothing is mis-PLACED (the anchor, the forecast and the reference are all on the right
clock — that is `source.anchorTimeS` and is tested); only the faded segment's *meaning* is
overstated.

The builder cannot narrow it on its own: the states record carries `anchorIndex` and
`anchorTimeS` but no `seq_len`. The fix is a record-contract field —
`build_prediction_record` stamping `source.lookbackSamples` (= `config.seq_len`, which every
call site has) and the builder slicing the last N samples ending at the anchor — plus a
republish of anything already written, since existing records lack the field. Deferred because
it changes the contract and would restate every published prediction directory; documented at
both ends (`LOOKBACK_COLOR`, `_lookback_states`) so nobody reads the line as the input window
in the meantime.
## 24. OpenSky typecodes absent from the Doc 8643 snapshot (H900, CL61, G450, F2EX, F2LX, G650, AS29)

**Verified** (2026-09-08, 38 observed rows). The identity resolver rejects an OpenSky typecode
the ICAO Doc 8643 snapshot does not list, and OpenSky uses a few marketing codes for which
the ICAO designator is known (H900 → H25B, CL61 → CL60, G450 → GLF4, F2EX/F2LX → F2TH,
G650 → GLF6). An alias table in `aircraft/icao_type_designators.py` (`normalize_typecode`),
each entry with its source, would resolve them; do not accept unknown codes generally.

**Fixed on branch `dev-training-followups` `dcbf6388`, merged with the rebuild after post-training stage 2 (user 2026-09-25)**: each alias names the snapshot row it stands for (the designator is read from it); an alias the
snapshot lists itself is refused. Delete this entry when the branch merges.

## 25. 1,164 observed rows: FAA-registered airframes whose model has no unambiguous ICAO typecode

**Verified** (2026-09-08). The FAA registry names the model but the model→ICAO crosswalk in
`build_aircraft_identity_database.py` leaves it ambiguous (policy: never guess), and OpenSky
has no typecode for the address either. A third identity source with per-address types
(the tar1090 / adsb.lol basic aircraft database, or OpenSky's current CSV — the snapshot in
`data/AIRCRAFT/aircraftDatabase.csv` is from 2026-07) would settle most of them; add it to
the builder as a lower-authority source validated against Doc 8643, the way OpenSky is.
Plus 283 rows unmatched anywhere (mostly foreign registrations: Mexico, Ireland, Canada).

## 26. `config.seq_len - 1` still spelled out at ~12 sites that already have `default_anchor`

**Verified** (2026-09-08, A2b review). `default_anchor(config)` moved from `inference/forecast.py` to
`config.py` so `dataset` could reserve a share of its random draws for L−1 without an import
cycle. It is now importable everywhere `config` already is, but the literal `config.seq_len - 1`
is still the spelled-out L−1 anchor in `anchor_grid.py:154`, `approach_clustering/cli.py:71`,
`run_ts.py anytime_curve:743`, `run_ts.py predictability_report` (×4),
`run_ts.py clock_attribution:115` and `run_ts.py control_capacity_ceiling:233` — every one of
which already imports `config`. A pure rename, no behaviour, at THOSE sites.

**Corrected 2026-09-09 (package review A-2)**: the `training/train.py`, `training/fixed_anchor_validation.py` and
`cli/predict.py` sites this entry listed were NOT pure renames — under a common anchor floor
(`minimum_anchor_index`, the history ablation) they had to read the floor, and renaming them to
`default_anchor` would have preserved the defect. They now read `dataset.fixed_anchor_index` /
`FixedAnchorTrajectoryWindows.anchor`. The runner-side sites above are replay paths with no
floor and stay a rename for the §4.5 runner move.

## 27. `cli/predict.py` builds prediction records at six near-identical sites

**Verified** (2026-09-08, L3.e review). The main path, the quantile-fan leaves, the latent
mode / random / shuffled diagnostics, the posterior arm and the closure-from-labels arm each
end in the same three lines — `zip(batch_series, forecasts, strict=True)`, then
`build_prediction_record(...)` and `observed_series_metrics(...)` into their own list. Adding
`--truncate-at-threshold` had to be threaded through every one of them (`_cut_at_threshold`),
and the next per-record post-step will have to be threaded through them again; a flag applied
to five of the six would silently make one output directory incomparable with the rest. One
`_emit(records, metrics, batch_series, forecasts)` helper would collapse them. Pure
refactor, no behaviour — left out of the L3.e change rather than folded into it, because the
change was already touching the forecast contract.

## 28. The lag-compensated bank inversion and the load coordination live in two hook modules

**Verified** (2026-09-08, L3.e review). `outputs/constraints/barrier_filter.py` and
`outputs/constraints/trombone.py` both spell out `tau_eff = tau(1 - e^{-dt/tau})`,
`committed = tan(mu_actuator)*tau_eff`, `lift = (n cos mu).clamp(min=0.5)`,
`scale = V_h/(g*lift)`, `atan((scale*dpsi - committed)/(dt - tau_eff))`, and
`n' = n cos mu / cos mu'` clamped to the envelope — about 25 lines carrying three physical
assumptions (the 0.5 lift floor, the lag credit, the coordination law). A
`outputs/constraints/turning.py` with `lag_effective_s`, `turn_geometry`,
`bank_for_heading_change` and `coordinate_load` would give them one home. Left out of the
L3.e change deliberately: the barrier is the ADOPTED delivery layer with published numbers,
and the L3.e equivalence run rests on it being byte-identical, so the extraction wants its
own commit with its own equivalence check.

## 29. `barrier_filter`'s load coordination perturbs the load of rows it never gated

**Verified** (2026-09-08, L3.e review). `coordinated = load * cos(bank) / cos(filtered)` is
evaluated unconditionally, and on a gate-closed row `filtered == bank`, so it computes
`(load*c)/c` — which is NOT the identity in IEEE (measured: an ULP apart for ~40 % of random
operand pairs, in both float32 and float64). The barrier therefore moves the load factor of
every ungated row by up to an ULP. The trombone was given
`torch.where(filtered == bank, load, ...)` for exactly this reason; the barrier was left
alone because changing it moves an adopted, measured layer's output and would have to be
re-measured against the published `control_hooks_v2_20260906` numbers rather than folded
into L3.e.

## 30. The plan path's rolled-window table is encoded once per window set — six copies on a pooled run (2026-09-12)

**Verified** (review of the step-5 tooling, `dev-plan-pool`): `outputs/plan/strategy.py`'s
`PlanContext` runs `normalizer.encode(np.asarray(rolled.windows, float64)).astype(float32)` over
the WHOLE rolled table when it binds a window set, and `training/train.py::prepare_session`
builds one train set plus one validation set PER AIRPORT (`validation_datasets`), so a pooled
five-airport run encodes the table six times. KRDU's table is 97,633 samples (70 MB as
`[S, 30, 6]` float32); the pooled cohort is 3.2× the flights, so ~225 MB per copy, ~1.35 GB
resident across the six sets plus a ~450 MB float64 transient per construction, during CUDA
training. Pre-existing code; the pooled run (`step5_pool_fan4_head`) is what first pays it.

**Judgement**: cache the encoded array on the `RolledWindowTable` keyed by the normalizer (one
copy per normalizer, shared by every set built under it). Not folded into the step-5 change
because it is a training-plane refactor with its own measurement (memory, not numbers).

## 31. The time closure's "drop the stretch" branch lays its probes on top of the stretch it is dropping

**Verified** (2026-09-14, R3.2 review, reproduced: `outputs/plan/forecast.py`, `fly_lockstep`'s closure
block). When a flight with a stretch in force reads late, the branch re-lays the route without it,
`lay(-state.stretch_m)`, and re-closes with `close_time(route_0, ..., lay=lay)` — but `lay` adds
`state.stretch_m` (still in force at that point) to every request, so the inner path lever's probes are
laid on top of the old stretch, all read too long, and all are refused. Reproduced: 30 km in force, a
flight 120 s early at the floor → the branch returns X = +120 s with nothing laid. Fix: pass
`lay=lambda extra, _s=state.stretch_m: lay(extra - _s)` there. Not folded in: R3.2 adopted no closure
change (plan §18.2), and this moves flown results (rarely: a stretch in force and a late read).

## 32. An instruction leg's speed points are keyed in the head's path to go and read at the route's

**Verified** (2026-09-14, R3.2). `forecast.leg_route` gives an instruction leg the speed points
`((anchor.remaining_m, anchor speed), (instruction.remaining_m, instruction speed))` — the HEAD's
path-to-go coordinates — while the controller (`guidance/controller.py`, `v_ref`) and the time closure
(`route_time_s`) read the schedule at the ROUTE's own path to go, which the `LEG_EXTENSION_M`
placeholder past the fix (8 km, then back onto the join) inflates. So the speed flown to the fix depends
on the placeholder's length (synthetic leg, heading away from the join: 6.8 s of leg time between 8 and
16 km of placeholder) and the fix's speed point is often never reached on the leg. **Judgement**: key the
leg's points at the route's own coordinates (the anchor at `route.remaining_m(0)`, the fix at its first
pass). Changes the flown speed on every instruction leg → a re-measure of the plan-path steps; not done.

## 33. Three more readers compare a stored config field by field with `!=` (2026-09-14)

**Verified** (M1 review of the specific-force axis):
- **The readers:** `experiments/pipeline.cv_reuse_error` compares the full `TSConfig(...).to_dict()` with a stored
  `cv_results.json`'s `base_config`. `experiments/coordinate_ablation` (one stored CV result against another) and
  `inference/build_multiflight_capacity_report._load_results` have the same shape.
- **The problem:** a field added after the artifact was written reads as a difference. Both stored
  `cv_results.json` already lacked 45–50 fields at `775b59e`, so neither was reusable before the change either.
- **The fix is ready:** the campaign runner's resume check got exactly this rule, `config.absent_field_defaults`,
  on 2026-09-14.

**Judgement**: adopt the same helper in the three readers. It turns "never reusable" into "reusable when only
absent fields differ", so it changes which stored CV results a `pipeline --skip-train` run reuses — the owner's
call.

## 34. The thrust-fraction speed floor's soft form is inert by a rounding accident (2026-09-14)

**Verified** (while building the specific-force floor):
- **The design:** `soft_max(x, b, s) = b + s·softplus((x − b)/s)` equals `x` exactly only where softplus takes
  its linear branch, input > 20.
- **The accident:** `_INERT_DEMAND` parks the demand exactly 20 softnesses below the box. At the box floor that
  ratio is `(−0.2 + 0.6)/0.02 = 20.000000000000004`, so the branch is taken, and `b + s·((x − b)/s)` happens to
  round back to `x`.
- **Evidence:** `test_the_soft_speed_floor_is_inert_where_it_demands_nothing` passes on those exact values. The
  specific-force twin, at 20 softnesses of 0.00717 g, did NOT round back. Its first fix, returning the command
  past the switch, still sat ON the switch at the box floor (M2 review: 25–29 % of random boxes miss). It is now
  parked ONE softness further, so every in-box command is strictly past the switch.

**Judgement**: harmless today, and pinned by the test. But any change to `MIN_THRUST_FRACTION`, the softness or
the multiplier can break it silently in the soft path (the test would catch the box-floor case only). Adopting
the specific-force form (park 21 softnesses below and `torch.where(x − b ≥ 20·s, x, soft_max(...))`) would
make it structural. That could
move thrust-fraction outputs by an ulp where they are currently not exact, so it is the owner's call.

**Addendum (2026-09-15, N4 review):** `control_condition_features` is one more field every stored
`cv_results.json` lacks. Nothing new breaks, since both stored files were already unusable for reuse.

## 35. The control path's anchor-eligibility gate restates the stall speed (2026-09-15)

**Verified** (N4 review):
- `outputs/control/strategy.airborne_control_candidates` computes `√(2 m g / (ρ₀ S Cl_max))` inline.
- It uses its own `SEA_LEVEL_DENSITY_KG_M3 = 1.225` and `flyability`'s `G`.
- The repository's one definition is `aircraft.aero_params.stall_speed_ms`. The optimizer's floor, evaluation's
  threshold speed gate and (since N4) the `ratios` condition channel all call it.

**Judgement**: the values agree today (both 1.225 and 9.81). Calling `stall_speed_ms` would make that
structural. The gate is spelled into the stored `airborne-1.10-stall-margin-v1` output-eligibility policy, so
the change must be shown to select the same anchors before it lands.

## 36. `predictability_report.batch_dynamics_tensors` duplicates the forecast's context batch (2026-09-15)

**Verified** (N4 review):
- It is `outputs/control/forecast._dynamics_batch` without the CTA branch.
- So every argument added to `dynamics_arrays` has to be added in both places. Both
  `control_thrust_parameterization` and `control_condition_features` were.
- A test now pins that its condition rows equal the forecast's.

**Judgement**: make `_dynamics_batch` public and call it from the report. The report is validation-only and
never runs a CTA arm.

## 37. The specific-force contract's constants are not in any checkpoint's identity (2026-09-15)

**Verified** (N6 review S2):
- The specific-force box `[−0.20, 0.23]` g and its neutral −0.05 are constants of `outputs/envelope.py`.
  They are not config fields, and the control target contract does not spell them.
- So a stored specific-force checkpoint would load and fly under moved constants with the same name. Today
  that is the two `sf_n3` checkpoints.
- The speed-command and path-angle contracts spell their constants into the target contract
  (`ControlContract.identity_suffix`, since 2026-09-16; before, `envelope.speed_command_identity()`).

**Judgement**: do the same for specific-force. Adding it would refuse the two stored `sf_n3` checkpoints,
so it needs a reading of the absent spelling as today's values (like `absent_field_defaults`), or a re-save.
It is the owner's call.

## 38. Formulas the control-contract refactor left restated outside the contract rows (2026-09-16)

Found by the sf-n7 architecture review before the merge; out of the refactor's scope
(`4dTrajectory/ts_transformer/docs/2026-09-16_two_tier_transformer_feasibility.zh.md` §11.2), recorded so they
are not rediscovered.

**Verified** (code read, 2026-09-16):
- The plan guidance restates the path loop: `outputs/plan/guidance/controller.py` computes
  `γ̇ = (γ_cmd − γ)/hold`, `n = cos γ + V·γ̇/g`, `/cos φ`, clamp — the thrust-fraction-pinned plan path's own
  loop, predating `torch_lag_dynamics.path_angle_load_factor`.
- The lag-credit inversion `(mean·hold − flying·τ_eff)/(hold − τ_eff)` has five copies: the speed floor's
  two inversions, `PlanGuidance`, the barrier's bank and the trombone's bank.
- `cos γ` has three spellings: `√(1−sin²γ)` (the path loop), `horizontal/speed` (the chart RHS), `cos(γ)`
  (the ENU twin and the numpy inverse).
- `CONTROL_NAMES` exists three times with two spellings: `aerodynamic_model.torch_dynamics` and
  `outputs/control/heads.py` (`thrust_N`, …) and `outputs/envelope.py` (`thrust_fraction`, …).
- (Resolved 2026-09-16: the barrier, the trombone and the speed floor read the load through
  `outputs/constraints/vertical.VerticalChannel` and are admitted under `specific-force+path-angle`.)

**Judgement**: the first two are the ones worth doing — one lag-credit helper, and the guidance asking
`path_angle_load_factor` — but each moves a stored plan-path or hook artifact at round-off, so each needs its
own golden check.


## 39. The specific-force speed floor holds sin γ at the hold's start (2026-09-16)

**Verified** (T1.5 re-verification, RK4 simulation of one hold): `outputs/constraints/speed_floor.py`
`_specific_force_floor` inverts `V' = g(n_x − sin γ)` with `sin γ` read at the segment's START. When the path is
still rising at the start of the hold (under the path-angle contract: the loop still pulling up, γ* 3° above γ) and
the command holds the path, the hold ends 0.24 m/s under the floor with the thrust ceiling not binding; the
specific-force contract's own twin shows the same effect (−0.85 m/s). Not introduced by the path-angle hooks.

**Judgement**: small in practice (a realistic loop lag at a segment start is ~10 % of that 3°). A fix would use
the hold-mean `sin γ` from the two-lag closed form, and moves every stored hooked specific-force artifact, so it
needs its own golden check.

## 40. Two definitions of "the truth's end" across the ts readouts (2026-09-17)

**Verified** (the L1 campaign's first `short_horizon_readout` died on it; fixed by counting, `4066345`): the
anchor sets (`anchor_grid.anchors_for_bin`, `window_anchors`, `truth_duration_s`) admit a flight on its
SUPERVISION rows — the observed track closed to the threshold — while `inference/receding.py`
(`mean_displacement_to`, `displacement_at`) and `lead_time_error` read the OBSERVED rows, which stop a median
6 s / 380 m short at KRDU. So a flight can be admitted with 60 s of truth after its anchor and have 45 s of
readable truth. Both readouts now state the gap (`truth_shorter_than_horizon`, held / absent counts) rather
than raise, and the segment-plan readout admits its fixed set on the observed rows.

**Judgement**: one convention would be better — either the readouts read the supervision rows (the package's
stated truth) or the anchor sets admit on the observed ones — but changing `displacement_at`'s truth moves
every S1 number and the `lead_time_error` accounting it mirrors, so it is a decision, not a fix.

## ts_transformer: `EXPERIMENTS_MAIN` lives outside `repo_layout` (2026-09-23)

**Verified**: `support.EXPERIMENTS_MAIN` is a repository path defined in `experiments/support.py`.
Layout rule L3 makes `repo_layout.py` the one definition of repository paths — the confusion of two
such paths (`TS_SCRIPT` imported as `RUN_TS`) is what broke the archived stage B′ queue. Import it
instead; no behaviour change. (The four runners that restated `HARVEST_ROOT` import it since the
data-merge change of 2026-09-23.)

## Review of trajectory_data_process + flight_scenarios against evaluation / ts (2026-09-23)

Three read-only reviewers plus a cross-pipeline check, before merging the 2026-08-22..09-22
download. FIXED in that change (not listed here): KSMF 35R's threshold and every runway's course
now from the CIFP (TD21), the merge's dropped `source_integrity` and `--jobs` (TD24), a plain
download clearing a merged root (TD24), `download_landings.py`'s drifted CIFP cycle, the arrival
slice ending before its measured crossing and the 0–2 s early `entry_time_utc` (TD22), and the
delete-then-build arrivals / observed writers (TD17). What is left, one item each:

1. **Two duplicate-row policies in one store** — *verified in code, size unmeasured*.
   `history_store.py:99-110` (direct download) keeps the LAST of conflicting `(icao24, time)`
   state rows (`INSERT OR REPLACE`); the sidecar path (`adsb_metadata.py:125-131`, the old root's
   freshness rebuild) returns None for them. Such rows are common (`nonunique_time_icao_rows`:
   KRDU 62,118, KSJC 318,722, KSTL 111,918), so the merged live root mixes the two rules.
2. **Short tracks dropped uncounted on a direct download** — *verified in code*. `tracks.py:239-240`
   drops tracks with < 10 samples after the freshness filter and crop, and the runner writes no
   population audit (why a fresh download's manifest has no `source_integrity`), against
   `store.py`'s "every fetched track is the denominator".
3. **A cross-window duplicate with a different key is stored twice** — *judgement, not triggered
   on 2026-09-23 (zero overlap)*. `merge.py:224-240` and `store.py:170-176` compare the whole
   `flight_key`; one physical flight truncated at a window edge (other outcome, other end time)
   or with a different first callsign gets another key. `classify.py:10` defines identity as
   `(icao24, landing time)`; the merge could check that too.
4. **Rows without geoaltitude are dropped before the ground-run split** — *judgement, low impact*.
   `tracks.py:225-228`, 337: a turnaround or touch-and-go under 900 s can glue the arrival to the
   departure, and `source_timed_final_block` keeps only the departure block.
5. **CZML censored tail** — *verified, viewer only*. `czml.py:275-286` starts the extrapolated tail
   at `source_sample_range[1]` while `extrapolation_distance_m` is measured from
   `diagnostics.closest_support_sample_index`: the drawn crossing precedes the last supporting
   sample in 1,871/7,827 censored KRDU flights (new download). Line 280 has a silent
   `speed_ms = 70.0` fallback, and the tail's timing differs from `crossing_span`'s trapezoid.
6. **`runway_targets` lists only runways with an included arrival** — *verified*. `arrivals.py`
   `setdefault` per included flight; `ts data/runway_context.py:274-301` and
   `experiments/runway_intent_r0.py:180` / `r1.py:275` treat it as the airport's runway set, so
   the entry-sector centroid and the candidate set move with the data window (KSTL 06, KMSY 20
   absent from the new download; KRDU 32 / KSMF 35R added: 349 m / 539 m centroid shifts).
7. **`GROUND_START_AGL_M`'s "empty band" no longer holds** — *verified*. KRDU 32 brings helicopter
   traffic (ZEUS*): 7 `takeoff_in_segment` exclusions starting 32–70 m up 18–20 km out, and one
   rostered arrival, `ZEUS11_32_aeaaaf_20260910T191932Z`, at 116.0 m (TD15). Re-measure on v7.
8. **The vertical-datum docs describe code that no longer runs** — *verified*. `flight_scenarios/
   CLAUDE.md` and `datum.py:1-35` say EGM96 via pyproj; `flight_to_msl` subtracts the runway's CIFP
   `hae_minus_msl_m` (KRDU 05L −32.0 m vs EGM96 −33.53 m). `_geoid_transformer`,
   `geoid_undulation_m`, `waypoints_to_msl` (`datum.py:57-116`) and `vertical_datum.msl_to_hae` have
   no callers but are exported — a new consumer following the docs is 1.5 m off.
   `observed.py:114-117` hand-rolls the same subtraction instead of calling `flight_to_msl`.
9. **ts `openap-direct` flies a different A320 than the optimizer** — *verified*.
   `ts data/dataset.py:574-584` passes `aircraft_provider="openap"`; flight_scenarios / the
   optimizer use `"auto"` (presets win, `build.py:44`). FFT2440: ts 66,000 kg / 124 m², optimizer
   66,300 kg / 122.6 m². Any ts-vs-optimizer thrust or flyability comparison mixes two models.
   (`FlightScenario.from_dict`, `scenario.py:314-322`, would also reload an openap-built scenario
   with "auto" — latent, no writer does.)
10. **Three velocity estimates for one first sample** — *verified*. `scenario.initial` fits a
    forward 15 s window (`start_state.py:70`), ts rows and optimizer reference records a centred
    window clipped at the start (`:121-131`), the observed record the whole track incl. pre-ring
    samples. RPA4668: V 87.30 vs 82.80 m/s, ψ 7.1° apart; `evaluation/reference.py` claims they agree.
11. **`flight_time_s` has a different origin per subject** — *verified*. Observed records start at
    first reception (FFT2440's arrival starts at t = 33.9 s), optimizer records at ring entry, ts
    records at the anchor (and keep the ring-entry `entry_time_utc`, `export.py:148`); the batch
    "flight times" are not comparable across subjects.
12. **ts and optimizer populations differ and nothing joins them** — *verified*. ts keeps the
    flights flight_scenarios drops as `UnusableFittedApproach` (9 of 9 tested built; 3 end 35–55 m
    short with no crossing row), and applies the lateral roster and `openap-direct`; the optimizer
    applies neither, only the per-runway cap. Per-airport rates are over different flights.
13. **Fallbacks that can never bind on manifest input** — *verified*. `build.py:88-115` silently
    falls back to `target_source="track_end"` when `threshold_target_state` is None (non-manifest
    path), so `ts dataset.py:585-588` can never fire; the TCH/glidepath skips at `dataset.py:522-529`
    are dead because `arrivals._validate_runway_target` requires those fields.
14. **ts observed-crossing scan has no on-final check without a fit** — *judgement*. `dataset.py:680`:
    with `fitted is None` the scan starts at row 1 with the plane test alone (contra C3). No
    spurious cut seen on 9 tested flights. **Fixed on branch `dev-training-followups` `dcbf6388`, merged with the rebuild after post-training stage 2 (user 2026-09-25)**: 3 of the 44 unfitted flights were cut at a
    pass off the final; the scan now reads `threshold_crossing_mask`.
15. **Small nits** — *verified*: `fitted_approach.py:62-63` defaults `hae_minus_msl_m=0.0` (a
    forgotten offset is 32 m low; make it required); `procedure_final.py:260` `elevationFt or 0.0`;
    `scenario_optimization.py:932-958` looks references up by `(id, icao24, landing_time)`, not
    `flight_key`; observed `source.id` is the raw callsign (10 fleet tracks differ, e.g. `'0  YP'`);
    `train.usable_series` exclusions are printed only; `channels.py`'s docstring calls `u` height
    above the THRESHOLD (it is threshold + TCH); `arrivals.py` restates
    `"opensky_history_geoaltitude_m"`; `airports.py:354` `entry.get("elevation_m", 0.0)`;
    `altitude_outliers --rerender-czml` renders with `DEFAULT_POLICY` whatever the audit's policy;
    `runner.py:267-271` omits `max_crossing_height_m` from the landing-screen provenance.
16. **Two replay paths still read the live root by path** — *verified in code, low impact*.
    `ts experiments/control_basis_oracle.build_cohort_series` (the width study) rebuilds a
    reference prediction's cohort from the live manifest and records the digest without comparing
    it; after the 2026-09-23 merge a v5-era reference is rebuilt from v7 slices. Resolve it by the
    reference checkpoint's digest (`repo_layout.checkpoint_arrival_manifest`) if the study is rerun.
17. **Staging leftovers are never cleaned** — *verified*. A SIGKILL mid-write leaves
    `approach/.records-staging-*` / `.records-previous-*`, or a harvest root's
    `.<ICAO>-reclassify-*` / `-merge-*` (the v5 root holds a 170 MB `.KSJC-reclassify-akrwpor_`
    from 2026-08-24). Harmless to readers (they follow rosters); disk only.
18. **`test_write_reference_records_from_observed_tracks` fails on a clean tree** — *verified*
    (2026-09-23, on HEAD a27dd441 in a scratch worktree). Its `_scenario()` fixture gives
    `source={"id", "runway"}` with no `arr_airport`, and `evaluation/records.py` has required a
    non-empty `source.arr_airport` since bb643d58 (report v7). `run_all_tests.sh` documents ONE
    known failure; this is a second one, so the "anything beyond that is a regression" rule no longer
    holds until the fixture is fixed.
19. **The OpenSky registration crosswalk counts a previous airframe's registration** — *verified*.
    `build_aircraft_identity_database` joins OpenSky rows to FAA models by registration only. A US
    N-number (and its Mode S address) is reissued, so an old airframe's row votes for the new one:
    BOEING 737-86N lost its B738 vote (8 of 9, below the 95 % share) because N452AC was a G200
    before; 11 KRDU arrivals lost their type with the 2026-09-22 snapshot. Joining on
    (registration, serial number) — both files carry the serial — would drop such stale votes, but
    it moves every crosswalk-derived model, so it needs a measured before/after.
20. **The identity is the registry snapshot's, not the flight date's** — *verified*. The resolver
    reads one FAA snapshot (2026-09-22) for flights from 2026-06 onward: 617 of 70,053 FAA-typed
    observed flights (0.9 %) are on a registration whose certificate was issued AFTER the flight
    (36 of them among the 506 flights whose type changed with the snapshot update), and 2 airframes
    flown in the summer are gone from the snapshot (deregistered). FAA's DEREG file plus CERT ISSUE DATE would let the resolver
    pick the registration valid on the flight date; that changes `resolve()`'s signature (a flight
    date), which touches flight_scenarios and the ts dataset.
21. **The OpenAP caches are read without a schema check** — *verified by the 2026-09-23 identity
    review*. `aircraft/query_aircraft_parameters.py:115-116` loads `aircraft_id_lookup.json` and
    `openap_aircraft_parameters.json` without checking `schema_version`, and
    `build_openap_aircraft_database.py` still writes the parameters file's version as a literal `1`
    (the lookup's now comes from `identity.OPENSKY_LOOKUP_SCHEMA`). **Fixed on the same branch** (`load_json` refuses
    another schema by name; the literal is `OPENAP_PARAMETERS_SCHEMA`).
22. **Two runners keep their own copy of the git-state helper** — *verified* (2026-09-24).
    `experiments/instruction_spec.py` and `experiments/instruction_training_export.py` each define a
    private `_git_state()` byte-identical to `repo_layout.git_state()` (added for `executor_spec`);
    replace both with the import.

## `test_write_reference_records_from_observed_tracks` fails on dev-two-tier (2026-09-23)

**Verified** (found while testing the published-approach-speed change; reproduced on unchanged
`dev-two-tier` at `3b2b0bcf` in the main checkout). `4dTrajectory/optimization/tests/test_scenario_optimization.py
::test_write_reference_records_from_observed_tracks` fails with
`ValueError: …_reference_eval.json: source.arr_airport must be a non-empty string`
(`evaluation/records.py:108`, raised while the test reads back the reference record it wrote).
Cause not investigated. Not caused by the approach-speed change; `run_all_tests.sh`'s header still
lists only the numpy-scalar failure as known.


## `THRESHOLD_SPEED_GATE.md` §3.3 quotes a "+5/−0 kt" margin that AC 91-79B does not contain (2026-09-23)

**Verified** against the saved `docs/regulation/FAA_AC_91-79B_2023.pdf` (dated 8/28/23, text
extracted with `pdftotext -layout`): no "+5", "5/", "minus" or "5 kt" anywhere. The only threshold
speed rule is §5.2.2 "Excess Airspeed": "The recommended approach reference landing airspeed (Vref),
plus wind and gust additives, should be maintained until 50 ft over the runway threshold." The
"+5/−0 kt" and the "Boeing/Airbus FCTM … typically +5 kt" sentence therefore have no primary text on
disk (perhaps an earlier AC 91-79A; not checked). The gate's window itself rests on FSF ALAR 7.1 and
is unaffected; the sentence needs a source or removal.

## B77W preset: `landing_mass` is 19 % above its published MALW (2026-09-24)

**Verified** (review of the published-approach-speed change). The B77W preset carries no
`max_landing_kg`, so `landing_mass` = 0.85 × 351,530 = 298.8 t against the FAA MALW of 251.3 t; the
sqrt(m / MALW) law then extrapolates its 149 kt to 162.5 kt at that mass (176.2 kt at the Pilot
panel's MTOW). **Fixed on branch `dev-training-followups` `dcbf6388`, merged with the rebuild after post-training stage 2 (user 2026-09-25)** — delete this entry when it merges. The A320 preset (66.3 t vs 66.0 t) and C172 (983 kg vs 1,157 kg) are closer. Giving the
presets `max_landing_kg` from their published row would fix it, but it moves the preset masses, the
analysis's native stall-margin range (B77W is its lower end) and the 2 B77W arrivals' scenarios.

## Performance index: points the stage-2 review left for later (2026-09-24)

**Verified** (opus review of the performance-index change).
- `trajectory_data_process/harvest/observed.py:106` labels every modelled landing mass
  `openap_landing_mass`, also for presets and the index's own-parameter types; and a substituted type now
  gets no mass, so its observed record falls back to the nominal 60 t (C25A/C525/PC24 6.8 t → 60 t). The gate
  never reads that mass; the label should say where the mass came from. Changes observed records only when
  they are regenerated.
- `docs/literature/control_normalization/measurements/per_class_readout.py:51` and
  `4dTrajectory/ts_transformer/docs/specific_force_teacher_distribution.py:49` resolve `dynamics_typecode`
  with `provider="openap"`; on a record of an index own-parameter type (only produced under `auto`) they raise.
- Index data worth a look (judgement): B722's Poll–Schumann landing mass is 5.2 % above its FAA MALW, so its
  target is 136.4 kt against a published 133 kt (**fixed on the training follow-ups branch**: own types land at the
  published MALW); MD88 flies as B737 because a substitute must be a native
  airframe and its PS synonym MD82 is an own-parameter row; the similarity distance has no mass term, so LJ35
  (6.5 t) flies as B737 and GLF3 as B763 (dynamically similar, by the method's definition).

## The native stall-margin range after the preset masses moved (2026-09-26)

**Verified** (numbers; opus review of `dev-training-followups`), **judgement** (what to do). The substitution analysis
(`docs/aircraft_performance/`, 2026-09-24) accepts an OpenAP synonym or a surrogate only when its stall margin (FAA
approach speed ÷ the model's 1 g stall speed at the landing mass) lies inside the NATIVE range of the modelled types.
With the presets landing at their published MALW (the training follow-ups branch) that range moves from 1.134 (B77W) –
1.623 to 1.182 (B734) – 1.623, and three decisions fall below the new lower edge: B733 kept on its B734 synonym (1.148),
E550 (1.153) and E545 (1.167), both own-parameter rows. `final_mapping.py:154` also still computes the Poll–Schumann
types' margins at Poll–Schumann's own landing mass. At the rebuild: either re-apply the rule (those three change decision)
or state that the range is frozen at the 09-24 analysis. The analysis document carries a dated note saying so.

## OpenAP-direct types land at OpenAP's MLW, not the published MALW (2026-09-26)

**Verified** (opus review of `dev-training-followups`). The branch makes every preset and own-parameter index type land
at the published MALW its approach speed is scaled from; an OpenAP-direct type still lands at OpenAP's `mlw_kg`, which
differs from the FAA MALW by C550 +11.1 %, E145 +3.2 %, B752 +2.7 %, A319 +2.5 % (above) and B737 −11.3 %, B37M −9.2 %,
B744 −8.9 %, B763 −6.3 % (below). If "landing mass = the MALW the speed is scaled from" is to hold for every modelled
type, decide it before the rebuild (it moves those types' mass and approach speed).

## An alias-resolved identity is not recorded as one (2026-09-26)

**Judgement** (opus review of `dev-training-followups`). An OpenSky code resolved through `MARKETING_ALIASES` is labelled
like a direct designator (`direct_designator` / `opensky_icao24_validated`); nothing in the identity says an alias was
applied. Recording it changes the identity payload, so it needs its own schema name.
