# Code-health follow-ups (deferred)

Findings noticed while working elsewhere, recorded rather than fixed on the spot so the
change that surfaced them stays reviewable. Nothing here is a live bug unless it says so.

Each entry states what was **verified** versus what is **judgement**, so a later reader can
tell how much re-checking it needs. **Keep the status table below current**: a new entry adds a row — its status and whether a fix would
touch the current training or post-training; when an entry is fixed or dismissed, its row says so (with the commit) and
the entry itself is deleted.

## Status of every entry (verified against the code 2026-09-25)

Checked entry by entry against `dev-two-tier` `1a7ac875` plus branch `dev-frontend-followups`, then updated after branch
`dev-followups-no-training` (2026-09-25) fixed every entry whose fix touches neither the training nor the post-training
(the right-hand column): **17 open, 8 partly, 3 fixed on a branch, 50 resolved or dismissed, 5 obsolete**. *open*: the problem is still in the
code; *partly*: some of it is fixed (the note says what is left); *resolved*: fixed (the note says by what);
*dismissed*: not a defect (the note says why); *obsolete*: the code is gone. A resolved, dismissed or obsolete entry's
text is removed below (its row stays); rows follow the entries' order; note the two sets of numbers (§19–§21 each appear
twice). Also found and fixed on `dev-frontend-followups`, never entered below: the terrain preload's tile counts and the
range ring radius re-rendering the app (`f73015df`, `1d8630c0`), the Pilot catalog failure hidden on entering
Trajectory (`fd9f205f`, `1d8630c0`), Observe's sample count planning loads per keystroke (`3a4aea81`, `d57f7fc7`).
Since (2026-09-26): the training-affecting ones fixed on branch `dev-training-followups` `dcbf6388` — not merged until the
rebuild after post-training stage 2 (user 2026-09-25) — say "fixed on a branch" in their rows and keep their entries until
it merges (24, the B77W preset, review #14 and #21, the performance index's B722 point); its review added three entries
(the rows after the performance index's).

| Entry | Status | What remains, or what resolved it | Fix affects training / post-training? |
|---|---|---|---|
| Training review 1 — every chart hover re-renders the whole app | resolved | cursor in its own context, 3D layer a leaf (`eac9c60a`); entry removed | — |
| Training review 2 — leaving Training throws its session away | resolved | panel kept mounted, session per airport (`bab2d5fe`, `2a368296`); entry removed | — |
| Training review 3 — `instruction_training_export.py` is runner and library | resolved | `instructions/training_files.py`, torch-free, raises `ValueError`; the backend uses its checks (`9ce5fda2`, `3c9930d1`) | — |
| Training review 4 — race on `training/index.json` | resolved | a manifest is rewritten only while it is what the run read; every airport checked before any is written (`9ce5fda2`, `3c9930d1`) | — |
| Training review 5 — replay draws no heading band for a dynamics failure | open | export still skips it; needs an overlay v3 schema and a re-export (the user's OK) | no: the Training view's exports |
| Training review 6 — replay's off-word intercept count covers the whole flight | open | export and live backend can still show different counts; changes the overlay (re-export) | no: the Training view's exports |
| Training review 7 — three strictnesses of "the re-read sentence is the stored one" | resolved | one `training_files.require_stored_sentence` for both exporters and the backend (`9ce5fda2`); `autopilot/replay.draw` keeps its own (frozen by the executor's code identity) | — |
| Training review 8 — duplication across the exporters and the backend | partly | rounding, band payload, `git_state`, repo-relative paths, the stored-sentence slice and the geoid wrapper shared (`9ce5fda2`); left: the words in force in `prior.data` and `autopilot.sentence` (training / frozen), the backend's `end_state_row` and chart shift, the three `export()` skeletons | no: same results, fewer copies |
| Training review 9 — smaller exporter points | resolved | compact `sample.json`, the unreachable branch gone, `params` pinned, one `captureBeforeThresholdM` (`9ce5fda2`, `3c9930d1`) | — |
| Training review 10 — the prior's first predicted step has no exporter-side test | resolved | a word at `N_LOOK` and the later of two words in the window, against `words_in_force` (`9ce5fda2`) | — |
| Training review 11 — `TRAINING_STRATA` not pinned | resolved | pinned by the backend's `MirrorTest` (2026-09-25); entry removed | — |
| Three ts ablation runners write into filter-less directories (09-24) | resolved | they refuse an existing output directory (`59408197`); entry removed | — |
| `build_runway_config.py` cannot rebuild `runway_thresholds.json` (09-20) | open | no NASR width / plate-minima source | no: unless the runway configuration is regenerated |
| Two suites fail at HEAD (09-20) | resolved | the numpy-2 test and the `arr_airport` fixture fixed (`eec35792`); entry removed | — |
| ts: the closed-loop training's EXECUTOR side is not written (09-18) | obsolete | the stale docstring fixed (`59408197`); the hook `WindowContext.override` stays: frozen `data/dataset.py` asks it; entry removed | — |
| ts: the manoeuvre runners repeat an argument skeleton (09-18) | obsolete | five of six runners archived (`79f871f1`); entry removed | — |
| ts: `predictability_report` builds its own dynamics rows (09-18) | resolved | it runs the model on `forecast.dynamics_batch` (`b10b1d68`); entry removed | — |
| ts: `lead_landings` reads outer-test-hash landings (09-13) | open | still the owner's call | no: the old control models' intent input; the prior builds its own context and drops the sealed test days |
| ts: two dead loss helpers (09-07) | resolved | deleted with their own tests (`59408197`); entry removed | — |
| ts: `train_only_diagnostics` is one unreferenced helper (09-07) | resolved | module deleted (`dddee9ae`); entry removed | — |
| ts: `experiments.pipeline`'s flags no longer match the ones it emits (09-07) | resolved | renamed to the `TSConfig` fields (`59408197`); entry removed | — |
| 1. two byte-identical `_iso` | resolved | `harvest/utc.py` (`3a91b07f`); entry removed | — |
| 2. `summary_row` writes explicit JSON nulls | resolved | identity fields required, `callsign` optional (`eec35792`); entry removed | — |
| 3. `source_event_availability` re-validates its manifest | resolved | validation apart from the derivation (`3a91b07f`); entry removed | — |
| 4. stale schema fixture in the comparison-CZML tests | resolved | a placeholder version, the pass-through pinned (`06801fe7`); entry removed | — |
| 5. `trajectory_data_process/tests` has 12 pre-existing failures | resolved | the last one, the numpy test, fixed (`eec35792`); entry removed | — |
| 6. `write_arrival_records` clears its output directory | dismissed | deliberate, a root Open Items hazard; `--observed-only` is the non-destructive mode; entry removed | — |
| `run_all_tests.sh` has 13 pre-existing failures | resolved | none left; the header expects exit 0 (`eec35792`, `06801fe7`); entry removed | — |
| 7. `roster_context_keys` leaks a raw FileNotFoundError | resolved | `records.py:298-300`; entry removed | — |
| 8. `record_from_dict` does not enforce increasing `t` | resolved | `records.py:138-142`; entry removed | — |
| 9. `arrival.py` restates `STATE_KEYS` | resolved | imported (`arrival.py:37`); entry removed | — |
| 10. empty `evaluate_batch` reports `mixed` | resolved | reports `empty` (`metrics.py:689-694`); entry removed | — |
| 11. last-sample kinematics under event names | resolved | speed and heading from the crossing; `methodology.event.final_time_s` says the observed one is the last sample (`5f5aeea9`); entry removed | — |
| 12. `_reference_aggregate` unweighted mean of means | resolved | stated in its docstring and EV4 (`5f5aeea9`); entry removed | — |
| 13. eleven `test_ts_pipeline.py` reuse-guard failures | resolved | fixtures fixed (`ed708dea`, `803605e0`); entry removed | — |
| 14. A320-family speed windows exclude most crossings | resolved | superseded: the gate uses published VREF (`53ae8e80`); entry removed | — |
| 2026-09-03 — 14 pre-existing failures in `trajectory_data_process/tests` | resolved | all fixed; entry removed | — |
| ts: reusable measurement code lives in `docs/` (09-07) | partly | runners import `inference/arm_readout.py`, 15 test preambles removed (`3f38f8af`); left: the `docs/*.py` readouts themselves and the audit plan's other hubs | no: readout scripts |
| ts: the auto-batch probe measures a smaller graph than a latent run | resolved | the probe hands a latent model the future, as training does (`b10b1d68`); entry removed | — |
| scene data plane: review leftovers (09-07) | resolved | (7)–(9), (12), (14) and the three test gaps fixed, (11) documented, a landed-before-t₀ neighbour no longer a lead (`96d299d1`, `50412e71`); (10), (13) obsolete; entry removed | — |
| ts: T2 leftovers (09-07) | resolved | `chart_scale` required (`6c363ef9`); the transport-chart rollouts kept as the scaled chart's test reference; the pointers demoted (`06801fe7`); entry removed | — |
| 15. KRDU 14's arrivals render as "indeterminate" | open | needs the backend and a UX decision | no: the Observe view |
| 16. observed record without `landing_aero` cannot say why | resolved | removed with the v9 gate; entry removed | — |
| 17. `require_matching_runway_data` has no production caller | open | still dead code; `harvest/airports.py` is frozen by the executor's code identity | no: dead code (but its file is frozen) |
| 18. `runway`-mode solves pile up at the window's upper edge | resolved | targets published V_ref (`88893126`); entry removed | — |
| 20. two copies of the point-mass inversion and the OLS slope | open | the flyability copy is in a frozen file; `start_state._slope` feeds `build_series` | no: same formula; only a degenerate-fit slope could differ |
| 19. speed anchors calibrated on wind-contaminated ground speeds | resolved | the gate reads published speeds; entry removed | — |
| 21. single-valued FAA approach-speed rows for multi-flap types | open, **blocked on a source** | no primary document publishes a reduced-flap V_REF at MLW for any of the five (searched 2026-09-25, `docs/literature/approach_speeds_reduced_flap/`); the entry's premise is doubtful — see the entry | **yes — executor**: the executor's approach speed for E75L, B737, A319, E170, E190 flights (25 % of the prior's train flights); the gate too |
| 22. four types publish no minimum operating mass | partly | LJ45 closed (`d5acb04f`); GLF5, C25A, C525 remain | gate only: the speed gate's mass range; the executor scales by √(m / MALW), not by the minimum |
| 23. the optimizer's velocity floor and V_ref target come from the stall model | partly | target fixed; the floor is still 1.10 × stall (a design decision) | no: the optimizer |
| 19. `control_basis_oracle --checkpoint` fingerprints without the roster | resolved | `e8df12f`; entry removed | — |
| 20. `random_train_anchor=True` raises before the first epoch | resolved | `7e947bb5`, with a test; entry removed | — |
| 21. the drawn "Lookback" is the whole pre-anchor track | open | no `lookbackSamples` record field yet (a record-contract change + republishing) | no: the comparison CZML |
| 24. OpenSky typecodes absent from Doc 8643 | fixed on a branch | `MARKETING_ALIASES` (H900, CL61, G450, G650, F2EX, F2LX; AS29 gliders left out) on branch `dev-training-followups` `dcbf6388`, merged with the rebuild after post-training stage 2 (user 2026-09-25) | **yes — data plane**: those flights would get a type, hence dynamics; the current sentence artefact refused for them until rebuilt |
| 25. 1,164 FAA airframes without an unambiguous typecode | partly | crosswalk `750daafb`: 2,737 → 914 unresolved; no third source; foreign rows unhandled | **yes — data plane**: as 24: types, hence dynamics, change |
| 26. `config.seq_len - 1` at sites that have `default_anchor` | resolved | `9d12b114`, `e2e1c84d`; entry removed | — |
| 27. `cli/predict.py` builds records at near-identical sites | resolved | one `_emit` (`b10b1d68`); entry removed | — |
| 28. bank inversion and load coordination in two hook modules | resolved | `outputs/constraints/turning.py`, the trombone bit-identical (`8ac40809`); entry removed | — |
| 29. barrier's load coordination perturbs ungated rows | resolved | `turning.coordinated_load`: an unmoved row's load untouched — the barrier's output moves by ≤ 1 ULP there, published barrier runs not re-run (`8ac40809`); entry removed | — |
| 30. rolled-window table encoded once per window set | obsolete | plan head archived (`9dbb4921`); entry removed | — |
| 31. drop-stretch branch lays probes on the stretch | obsolete | archived with the plan head; entry removed | — |
| 32. instruction-leg speed points keyed in the head's path-to-go | obsolete | archived with the plan head; entry removed | — |
| 33. readers compare stored configs field by field | dismissed | refusing a stored config that lacks a field is right under the no-compatibility rule (2026-09-19); entry removed | — |
| 34. thrust-fraction speed floor inert by a rounding accident | open | the structural form not adopted; `speed_floor.py` is frozen by the executor's code identity | no: command hooks; the executor runs none (but its file is frozen) |
| 35. the anchor-eligibility gate restates the stall speed | resolved | calls `stall_speed_ms`, bit-identical on 200,000 draws (`ddc16476`); entry removed | — |
| 36. `batch_dynamics_tensors` duplicates the forecast's batch | resolved | deleted; the report calls `dynamics_batch` (`b10b1d68`); entry removed | — |
| 37. specific-force constants not in any checkpoint's identity | open | unguarded; its suggested fix clashes with the no-compatibility rule; `outputs/envelope.py` is frozen | no: control-model checkpoints |
| 38. formulas left restated outside the contract rows | partly | `VerticalChannel` resolved; `CONTROL_NAMES` 3 → 2; guidance loop, lag credit ×5, cos γ ×3 remain, in frozen files | **yes — executor**: only its cos γ point: the transport-chart RHS the executor integrates (round-off in every flight); the rest no |
| 39. specific-force speed floor holds sin γ at the hold's start | open | unchanged; `speed_floor.py` is frozen | no: command hooks; the executor runs none (but its file is frozen) |
| 40. two definitions of "the truth's end" | resolved | each named where it is read (`57f1e386`); making them one is a decision that moves S1 numbers; entry removed | — |
| ts: `EXPERIMENTS_MAIN` lives outside `repo_layout` (09-23) | resolved | moved (`59408197`); entry removed | — |
| Review of trajectory_data_process + flight_scenarios (09-23), #1–#22 | partly | fixed #5, #7 (the comment), #8, #12 (documented), #13, #15a–d, f, g, i, j, #16, #17, #18, #22 (`3a91b07f`, `130a25cb`, `96d299d1`, `eec35792`, `57f1e386`, `f1c38875`); #14 and #21 fixed on branch `dev-training-followups` `dcbf6388` (merged with the rebuild); open #1–#4, #6, #9–#11, #15e, #15h, #19, #20 | **yes — data plane**: #14 (the crossing scan without a fit) and possibly #10 change `build_series`; #19, #20 change types; #1–#4, #6 re-harvest only; #15e what training runs write; the rest no |
| `test_write_reference_records_from_observed_tracks` fails (09-23) | resolved | the fixture carries a real scenario source (`eec35792`); entry removed | — |
| `THRESHOLD_SPEED_GATE.md` §3.3 quotes an unsourced +5/−0 kt margin (09-23) | resolved | removed, AC 91-79B §5.2.2 quoted (`5f5aeea9`); entry removed | — |
| B77W preset `landing_mass` 19 % above its MALW (09-24) | fixed on a branch | every preset lands at its published MALW (B77W 251.3 t, A320 66.0 t, C172 1,111 kg) on branch `dev-training-followups` `dcbf6388`, merged with the rebuild after post-training stage 2 (user 2026-09-25) | **yes — executor**: B77W flights' mass and approach speed (√(m / MALW)) |
| Pilot frontend tests use a 145 / 135 / 155 kt fixture (09-24) | resolved | real catalog fixture + a range preset (`3d0fc579`, `d2323adf`); entry removed | — |
| Performance index: stage-2 review leftovers (09-24) | partly | the observed mass label and the readout scripts fixed (`6c363ef9`, `3c4a92a4`); B722's mass fixed on branch `dev-training-followups` `dcbf6388` (merged with the rebuild); the 60 t observed fallback and the MD88 / LJ35 / GLF3 judgements remain | **yes — executor**: B722's mass, MD88 → B737, LJ35 → B737 and GLF3 → B763 change what those flights fly; the 60 t fallback: no |
| The native stall-margin range after the preset masses moved (09-26) | fixed on a branch | re-judged by the rule (user 2026-09-26) on branch `dev-training-followups` `fde40395`: E545, E550 → C550, B733 → exclude | **yes — executor**: E545 / E550 flights fly as C550 |
| OpenAP-direct types land at OpenAP's MLW, not the published MALW (09-26) | open | sources compared (entry); awaits the user's choice, and the FAA's B737 cell is wrong either way | **yes — executor**: those types' mass and approach speed |
| An alias-resolved identity is not recorded as one (09-26) | open | unchanged | no: provenance only, the types are the same |
| The Training view draws executor and observed tracks with EGM96, not the runway's offset (09-25) | open | new; see the entry | no: the Training view's exports (a re-export) |
| `READABLE_REPORT_SCHEMA_VERSIONS` reads four report versions (09-25) | open | new; see the entry | **yes — data plane**: ts `lateral_eligibility` reads reports through it |
| ts `docs/reference/runners.md` still names `instruction_training_export` as the Training helpers' home (09-25) | open | new; dev-post-train's file, left untouched | no: a document |

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

Resolved since and removed (numbers kept, the items refer to each other): **1**, **2** (branch `dev-frontend-followups`),
**3**, **4**, **7**, **9**, **10** (branch `dev-followups-no-training`: `instructions/training_files.py`), **11**. Item 8
is partly fixed.

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

**8. Duplication across the exporters and the backend — what is left** (the rest shared since 2026-09-25,
`instructions/training_files.py`, `repo_layout.git_state` / `repo_relative`, `geoid_undulation_m` as an array). "The words
in force" is still computed by `prior.data.sentence_steps` (the prior's training code) and `autopilot.sentence._filled`
(frozen by the executor's code identity) beside `training_files.words_in_force` and the TypeScript copy; the backend's
`end_state_row` and chart shift restate the executor exporter's (both need `autopilot`, so they cannot live in the
torch-free `training_files`); the three exporters' `export()` bodies share one skeleton.

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

## ts_transformer: reusable measurement code lives in `docs/` (verified, 2026-09-07)

What is left (2026-09-25): the twelve `docs/*.py` readouts stay command-line scripts over the package — nothing in the
package imports them since `compare_frame_arms`' library half became `inference/arm_readout.py` (L20) — and the audit
plan's other hubs (`score_control_arms` first; `4dTrajectory/ts_transformer/docs/2026-09-07_package_audit_plan.zh.md`
§七) are not migrated. Five test files keep a `sys.path` line: three add `4dTrajectory/optimization`, which
`tests/conftest.py` does not, and `test_architecture.py` / `test_prior_procedure.py` belong to branch `dev-post-train`.

## 15. KRDU 14's 13 arrivals still render as "indeterminate" in the Observe view

**Judgement.** `evaluation/docs/UNJUDGED_RUNWAY_VERDICT_GAP.md`: the evaluation half is fixed
(KRDU 32 / KSMF 35R now carry an LNAV/VNAV vertical path), so the "never judged" bucket is
down to KRDU 14, which has no RNAV procedure at all. The frontend still paints those flights
the same grey as a judged-indeterminate one; the fourth UI state / surfaced skip reason is a
product decision the gap document lays out. 13 flights, so it is cosmetic until another
procedure-less runway enters the fleet.

## 17. `harvest.airports.require_matching_runway_data` has no production caller

**Verified** (2026-09-07, by the M1 review): `runway_data_fingerprint` is written into the
reclassify / freshness-rebuild manifests as provenance and never checked; the checker
`require_matching_runway_data` is dead code. Pre-existing. Either wire it (the reclassify
reader is the natural place) or delete it; note the digest hashes every `Runway` field,
so it moved for all 26 runways when `tch_source`/`baro_vnav_minima` were added
(`test_approach_verticals.py` pins that as expected).

## 20. Two copies of the point-mass inversion and of the OLS slope, deferred under a campaign

**Verified** (2026-09-07, opus review of `dev-observed-load-factor-metar`).
`4dTrajectory/ts_transformer/geometry/flyability.py:259-262` computes `n` from `psi_dot`/`gamma_dot`
with the same two lines `aircraft/kinematics.load_factor_from_rates` now owns (it also
needs `mu`, so the shared function should return the two components), and
`flight_scenarios/start_state._slope` is `aircraft.kinematics.linear_slope` returning 0.0
instead of raising on a degenerate abscissa. Both live on the ts dataset/eval path that a
formal campaign was running from, so they were left untouched; fold them onto the shared
functions between campaigns (behaviour-identical for validated records).

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

## Review of trajectory_data_process + flight_scenarios against evaluation / ts (2026-09-23)

Three read-only reviewers plus a cross-pipeline check, before merging the 2026-08-22..09-22
download. FIXED in that change (not listed here): KSMF 35R's threshold and every runway's course
now from the CIFP (TD21), the merge's dropped `source_integrity` and `--jobs` (TD24), a plain
download clearing a merged root (TD24), `download_landings.py`'s drifted CIFP cycle, the arrival
slice ending before its measured crossing and the 0–2 s early `entry_time_utc` (TD22), and the
delete-then-build arrivals / observed writers (TD17). Fixed since by branch `dev-followups-no-training`
(2026-09-25): #5, #7 (the comment re-measured), #8, #12 (documented, FS8), #13, #15a–d, f, g, i, j, #16, #17,
#18, #22. What is left, one item each (numbers kept; #14 and #21 are fixed on branch `dev-training-followups`):

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
6. **`runway_targets` lists only runways with an included arrival** — *verified*. `arrivals.py`
   `setdefault` per included flight; `ts data/runway_context.py:274-301` and
   `experiments/runway_intent_r0.py:180` / `r1.py:275` treat it as the airport's runway set, so
   the entry-sector centroid and the candidate set move with the data window (KSTL 06, KMSY 20
   absent from the new download; KRDU 32 / KSMF 35R added: 349 m / 539 m centroid shifts).
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
14. **ts observed-crossing scan has no on-final check without a fit** — *judgement*. `dataset.py:680`:
    with `fitted is None` the scan starts at row 1 with the plane test alone (contra C3). No
    spurious cut seen on 9 tested flights. **Fixed on branch `dev-training-followups` `dcbf6388`, merged with the rebuild after post-training stage 2 (user 2026-09-25)**: 3 of the 44 unfitted flights were cut at a
    pass off the final; the scan now reads `threshold_crossing_mask`.
15. **Small nits — two left** (the rest fixed 2026-09-25): `train.usable_series` exclusions are printed only (#15e:
    recording them changes what every training run writes into its checkpoint metadata — a decision); `airports.py:354`
    `entry.get("elevation_m", 0.0)` (#15h: `harvest/airports.py` is frozen by the executor's code identity).
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

## B77W preset: `landing_mass` is 19 % above its published MALW (2026-09-24)

**Verified** (review of the published-approach-speed change). The B77W preset carries no
`max_landing_kg`, so `landing_mass` = 0.85 × 351,530 = 298.8 t against the FAA MALW of 251.3 t; the
sqrt(m / MALW) law then extrapolates its 149 kt to 162.5 kt at that mass (176.2 kt at the Pilot
panel's MTOW). **Fixed on branch `dev-training-followups` `dcbf6388`, merged with the rebuild after post-training stage 2 (user 2026-09-25)** — delete this entry when it merges. The A320 preset (66.3 t vs 66.0 t) and C172 (983 kg vs 1,157 kg) are closer. Giving the
presets `max_landing_kg` from their published row would fix it, but it moves the preset masses, the
analysis's native stall-margin range (B77W is its lower end) and the 2 B77W arrivals' scenarios.

## Performance index: points the stage-2 review left for later (2026-09-24)

**Verified** (opus review of the performance-index change). Fixed 2026-09-25: the observed record's mass label says
which model the mass came from (`dynamics_source`), and the two readout scripts resolve a record with its own provider.
Left:
- A substituted type gets no mass, so its observed record falls back to the nominal 60 t (C25A/C525/PC24 6.8 t → 60 t).
  The gate never reads that mass; what mass an observed record of another airframe should carry is a judgement.
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

**Fixed on branch `dev-training-followups` `fde40395`** (user 2026-09-26: re-apply the rule): E545 and E550 own →
substitute C550, B733 → exclude; the same three change whether or not the OpenAP-direct types also move to the MALW.
`scripts/rebuild_inputs.py` reproduces all 199 stored verdicts in its 2026-09-24 mode. Delete this entry when it merges.

## OpenAP-direct types land at OpenAP's MLW, not the published MALW (2026-09-26)

**Verified** (opus review of `dev-training-followups`). The branch makes every preset and own-parameter index type land
at the published MALW its approach speed is scaled from; an OpenAP-direct type still lands at OpenAP's `mlw_kg`, which
differs from the FAA MALW by C550 +11.1 %, E145 +3.2 %, B752 +2.7 %, A319 +2.5 % (above) and B737 −11.3 %, B37M −9.2 %,
B744 −8.9 %, B763 −6.3 % (below). If "landing mass = the MALW the speed is scaled from" is to hold for every modelled
type, decide it before the rebuild (it moves those types' mass and approach speed).

**Sources compared** (2026-09-26, manufacturer excerpts in `docs/reference_speeds/excerpts/`). The published figure is
the FAA Aircraft Characteristics Database 2024-10 `MALW_lb`, one row per ICAO designator; its `Approach_Speed_knot` is
defined AT that MALW (data dictionary row 14), so the pair belongs together. OpenAP 2.4's `mlw` is one model per
designator (the file names it: "Boeing 737-700", "Boeing 767-300"…) with no per-value citation. Three kinds of gap:
- one designator, two variants or weight options, both published: B763 (Boeing 767-300 136,077 kg — OpenAP; -300ER
  145,149 — FAA), A319 (Airbus WV 61,000 — FAA; 62,500 — OpenAP); B744 and E145 look the same but no manufacturer
  document is on disk;
- OpenAP's figure is not the type's: C550 6,804 kg (FAA TCDS A22CE Rev 74 p.6: Model 550 lands at 12,700 / 13,500 lb =
  5,761 / 6,123 kg; OpenAP's is ≈ its own MTOW), B38M 66,300 (Boeing 737-8: 68,174 / 69,308; 66,300 is a 737-800
  figure), B37M 60,000 (Boeing 737 MAX ACAP 2.1.1, 737-7: 63,911 / 66,043), A20N 66,000 (Airbus A320neo 67,400; 66,000 is the A320ceo's),
  B752 92,200 (Boeing 757-200 89,811 / 95,254; 92,254 is the 757-200PF freighter's);
- the FAA's figure is wrong: B737 — the row is "Boeing 737-700" but its MALW 145,600 lb (66,043 kg) is the 737-7
  (MAX 7)'s; Boeing's 737-700 lands at 58,059 / 58,604 kg (737NG ACAP 2.1.2), and OpenAP's 58,600 is right. The
  provenance README already calls this row "doubly suspect"; B739's FAA cell was replaced by Boeing's for the same
  reason. Replacing B737's MALW moves its approach-speed anchor (130 kt then sits at 58.6 t) and so the speed gate.

## An alias-resolved identity is not recorded as one (2026-09-26)

**Judgement** (opus review of `dev-training-followups`). An OpenSky code resolved through `MARKETING_ALIASES` is labelled
like a direct designator (`direct_designator` / `opensky_icao24_validated`); nothing in the identity says an alias was
applied. Recording it changes the identity payload, so it needs its own schema name.

## The Training view draws executor and observed tracks with EGM96, not the runway's offset (2026-09-25)

**Verified by reading** (found fixing #8). `flight_scenarios.datum.flight_to_msl` converts a track to MSL with its
runway's CIFP offset (`hae_minus_msl_m`: KRDU 05L −32.0 m), and the comparison CZML adds that same offset back. The
Training view's exports go the other way with EGM96 (`geoid_undulation_m`: −33.53 m there): the observed signals
(`instruction_training_export.Globe.undulation_m`), the executor replay's track (`executor_training_export.track_payload`)
and the live executor's (`aeroviz_backend/autopilot_segment/payload.track_payload`) are drawn ~1.5 m below the height
they were measured at. Consistent within the view, 1.5 m off the observed layer and the terrain. Fix: carry the runway's
offset into the exports (the sample has `candidates[].elevationM`; the offset is the artefact's per-runway datum) and
drop `geoid_undulation_m`; it changes every exported sample and overlay (a re-export).

## `evaluation.metrics.READABLE_REPORT_SCHEMA_VERSIONS` reads four report versions (2026-09-25)

**Verified by reading.** `READABLE_REPORT_SCHEMA_VERSIONS` accepts v6, v7, v8 and v9 for a consumer that reads only the
fields they share; its one consumer is ts `data/lateral_eligibility.py`, which builds the lateral-pass roster the ts
datasets filter on. A version list read by a consumer is the schema-version branch the no-compatibility rule (2026-09-19)
forbids; refusing everything but v9 would refuse the rosters built from older reports — the user's call, and a change
on the training data plane.

## ts `docs/reference/runners.md` still names `instruction_training_export` as the Training helpers' home (2026-09-25)

**Verified.** `runners.md:260` says "The shared helpers live in `instruction_training_export`"; since 2026-09-25 they
are `instructions/training_files.py` (layout L30's note). The file belongs to branch `dev-post-train`, so it was not
edited; one line to change when that branch has merged.

