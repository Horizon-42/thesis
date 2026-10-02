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
(the right-hand column): **27 open, 7 partly, 67 resolved or dismissed, 5 obsolete** (recounted from the table 2026-09-28; one open row is blocked on a source). *open*: the problem is still in the
code; *partly*: some of it is fixed (the note says what is left); *resolved*: fixed (the note says by what);
*dismissed*: not a defect (the note says why); *obsolete*: the code is gone. A resolved, dismissed or obsolete entry's
text is removed below (its row stays); rows follow the entries' order; note the two sets of numbers (§19–§21 each appear
twice). Also found and fixed on `dev-frontend-followups`, never entered below: the terrain preload's tile counts and the
range ring radius re-rendering the app (`f73015df`, `1d8630c0`), the Pilot catalog failure hidden on entering
Trajectory (`fd9f205f`, `1d8630c0`), Observe's sample count planning loads per keystroke (`3a4aea81`, `d57f7fc7`).
Since (2026-09-26): the training-affecting ones fixed on branch `dev-training-followups` (`dcbf6388`, `fde40395`,
`d3d83716`: 24, the B77W preset, review #14 and #21, the performance index's B722 point, the native stall-margin range,
the OpenAP-direct landing mass) are merged into `dev-two-tier` (`acb93b55`, user 2026-09-26) ahead of the rebuild; its reviews
added three entries (the rows after the performance index's).

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
| ts: reusable measurement code lives in `docs/` (09-07) | resolved | runners import `inference/arm_readout.py`, 15 test preambles removed (`3f38f8af`); the thirteen `docs/*.py` moved on branch `docs-reorg` (2026-09-26): seven became runners, six went to three archives, `test_docs_holds_no_python` guards it (L20); entry removed; the tests' `sys.path` lines are the next row | — |
| ts: five test files still touch `sys.path` (09-26) | open | new; see the entry | no: tests |
| ts: ten dated records stay in the `docs/` root because hashed code cites them by path (09-26) | resolved | moved into `docs/history/` and `archive/*/docs/`, every path rewritten (branch `dev-executor-v11`, v11 milestone 1); entry removed | — |
| ts: the auto-batch probe measures a smaller graph than a latent run | resolved | the probe hands a latent model the future, as training does (`b10b1d68`); entry removed | — |
| scene data plane: review leftovers (09-07) | resolved | (7)–(9), (12), (14) and the three test gaps fixed, (11) documented, a landed-before-t₀ neighbour no longer a lead (`96d299d1`, `50412e71`); (10), (13) obsolete; entry removed | — |
| ts: T2 leftovers (09-07) | resolved | `chart_scale` required (`6c363ef9`); the transport-chart rollouts kept as the scaled chart's test reference; the pointers demoted (`06801fe7`); entry removed | — |
| 15. KRDU 14's arrivals render as "indeterminate" | open | needs the backend and a UX decision | no: the Evaluation view |
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
| 24. OpenSky typecodes absent from Doc 8643 | resolved | `MARKETING_ALIASES` (H900, CL61, G450, G650, F2EX, F2LX; AS29 gliders left out; `dcbf6388`, merged `acb93b55`); entry removed | — |
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
| Review of trajectory_data_process + flight_scenarios (09-23), #1–#22 | partly | fixed #5, #7 (the comment), #8, #12 (documented), #13, #15a–d, f, g, i, j, #16, #17, #18, #22 (`3a91b07f`, `130a25cb`, `96d299d1`, `eec35792`, `57f1e386`, `f1c38875`); #14 and #21 (`dcbf6388`, merged `acb93b55`); open #1–#4, #6, #9–#11, #15e, #15h, #19, #20 | **yes — data plane**: possibly #10 changes `build_series`; #19, #20 change types; #1–#4, #6 re-harvest only; #15e what training runs write; the rest no |
| `test_write_reference_records_from_observed_tracks` fails (09-23) | resolved | the fixture carries a real scenario source (`eec35792`); entry removed | — |
| `THRESHOLD_SPEED_GATE.md` §3.3 quotes an unsourced +5/−0 kt margin (09-23) | resolved | removed, AC 91-79B §5.2.2 quoted (`5f5aeea9`); entry removed | — |
| B77W preset `landing_mass` 19 % above its MALW (09-24) | resolved | every preset lands at its published MALW (B77W 251.3 t, A320 66.0 t, C172 1,111 kg; `dcbf6388`, merged `acb93b55`); entry removed | — |
| Pilot frontend tests use a 145 / 135 / 155 kt fixture (09-24) | resolved | real catalog fixture + a range preset (`3d0fc579`, `d2323adf`); entry removed | — |
| Performance index: stage-2 review leftovers (09-24) | partly | the observed mass label and the readout scripts fixed (`6c363ef9`, `3c4a92a4`); B722's mass fixed (`dcbf6388`, merged `acb93b55`); the 60 t observed fallback and the MD88 / LJ35 / GLF3 judgements remain | **yes — executor**: MD88 → B737, LJ35 → B737 and GLF3 → B763 change what those flights fly; the 60 t fallback: no |
| The native stall-margin range after the preset masses moved (09-26) | resolved | re-judged by the rule (user 2026-09-26): E545, E550 → C550, B733 → exclude (`fde40395`, merged `acb93b55`); entry removed | — |
| OpenAP-direct types land at OpenAP's MLW, not the published MALW (09-26) | resolved | every modelled airframe lands at its published MALW, B737's MALW from Boeing (user 2026-09-26; `d3d83716`, merged `acb93b55`); entry removed | — |
| An alias-resolved identity is not recorded as one (09-26) | open | unchanged | no: provenance only, the types are the same |
| A ts checkpoint does not record where its landing masses came from (09-26) | open | needs the user's decision (a payload change) | no for the two-tier chain (its artefacts record the aircraft tables); yes for any control-path checkpoint replayed after the rebuild |
| The Training view draws executor and observed tracks with EGM96, not the runway's offset (09-25) | resolved | every writer adds the flight's runway offset — the exporters (`0b1fe50a`), the live executor (`c708a497`, `2a64d3f4`); `geoid_undulation_m` deleted; `instruction_v3_day_split` and its four overlays re-exported (`check-publication` 0 errors); the older sets kept as published with EGM96 heights (user 2026-09-26; `aeroviz-4d/docs/36-…` §2.3); entry removed | — |
| `READABLE_REPORT_SCHEMA_VERSIONS` reads four report versions (09-25) | open | new; see the entry | **yes — data plane**: ts `lateral_eligibility` reads reports through it |
| ts `docs/reference/runners.md` still names `instruction_training_export` as the Training helpers' home (09-25) | resolved | the line names `instructions/training_files.py` (branch `docs-reorg`); entry removed | — |
| `aeroviz-4d/python/requirements.txt` still lists `pyproj` (09-26) | open | new; see the entry | no: a requirements list |
| `autopilot/__init__.py` names the executor design by its old path (09-26) | resolved | points at `docs/two_tier/executor_design.zh.md` (branch `dev-executor-v11`, v11 milestone 1); entry removed | — |
| The land law leaves a shallow final class's tube near the threshold on some profiles (09-26) | open | new; see the entry | **yes — executor** (a law change is a new spec) |
| No mode reports when the executor's glidepath floor binds (09-26) | resolved | mode `glidepath_floor`, counted per vertical word; the replay summary files a word it pushed out apart (branch `dev-executor-v11`, v11 milestone 2); entry removed | — |
| The Training export and the live model flight fly a model without the procedure's masks (09-26) | resolved | fixed on `dev-training-rounds` (2026-09-27): the export speaks under the model's own masks and reads its sentences with the readout's `said_rows` (glidepath stop included), the overlay records the sets with their data digests, the backend rebuilds them, checks the digests and cuts the re-flight at the same step (`aeroviz-autopilot-segment-v5`); entry removed | — |
| The judge folds a wrong-parallel-runway approach into timeout / crossed off runway; its landed lateral limit is 1,000 m where there is no parallel (09-26) | resolved | outcome `crossed_other_runway` (another runway's threshold crossed over that runway, lined up, any height; a line-up that never reaches it stays a timeout); landed within the runway's FAS course half-width, 106.7 m (branch `dev-executor-v11`, v11 milestone 2); entry removed | — |
| `outputs/envelope.py` names `docs/specific_force_teacher_distribution.py` (09-26) | resolved | names the runner `experiments/specific_force_teacher_distribution.py` (branch `dev-executor-v11`, v11 milestone 1); entry removed | — |
| `instructions/grammar.py` is outside the labeller sha (09-26) | open | new; see the entry | **yes — every sentence artefact and prior**: adding it to the hash changes the labeller sha they record; do it with the next vocabulary spec |
| The judge calls a captured, on-centreline crossing that is too high "crossed off runway" (09-27) | resolved | outcome `crossed_too_high` (branch `dev-executor-v11`, v11 milestone 2); entry removed | — |
| The terrain downloader overwrites the shared `download_manifest.csv` (09-27) | open | new; see the entry | no: data tooling |
| The executor spec binds the source's BYTES, comments included (09-27) | resolved | the hash is over each file's logic, `spec.logic` (branch `dev-executor-v11`, v11 milestone 1); entry removed | — |
| `ExecutorParams.check` lets a NaN rate or factor through (09-27) | resolved | every value finite and positive (branch `dev-executor-v11`, v11 milestone 1); entry removed | — |
| A free sentence is judged against the runway pointed at the executor's stop, not at its end (09-27) | open | new; see the entry | **yes — post-training**: the landing reward reads the outcome |
| `faa_separation(visual_parallels=True)` is a second, different "visual" (09-27) | open | new; see the entry | no: only the archived runway-intent runner and one test call it |
| The labeller sha is over bytes and checked against the running code: one edit deadlocks the chain (09-28) | open | new; see the entry | **yes — every artefact's and prior's identity record** (no retrain); the fix edits `autopilot/replay.py`, inside the executor hash — batch with the executor-identity entry |
| The executor spec refuses any other code, over 28 modules including all of `config.py` and `data/dataset.py` (09-28) | open | new; see the entry; the rule is the user's (2026-09-27) | **yes — executor identity** (no behaviour change) |
| Two-tier data are bound to each airport's whole arrival manifest by its bytes (09-28) | open | new; see the entry | **yes — executor identity**: `autopilot/flights.py` is hashed, so the fix moves the source hash — batch with the executor-identity entry |
| A Training overlay is bound to its set's file sha256 and writing time (09-28) | resolved | bound by what it shares with the set — spec, candidates, frame, and flight by flight its keys, words / steps and every drawn height's HAE − MSL; overlays v2, executor / prior / generation v4, augmented generation v2 (`2f557772`; every published overlay re-exported 2026-09-28, the user's OK); entry removed | — |
| The frontend pins the vocabulary spec's sha (09-28) | open | new; see the entry — low value while the superseded `instruction_v3` set stays listed | no: a frontend constant |
| The single-flight executor's pin refuses at run time, not in the tests (09-28) | open | new; see the entry | no: the backend |
| A glidepath stop overrides a judged event that came before it (09-30) | open | new; see the entry | **yes — post-training**: stage 2's reward reads `said_rows` |
| The read-back altitude chart's coloured labels are grey (09-30) | open | new; see the entry | no: the Training view |
| A low go-around is stored as the landing: the best-aligned crossing, not the last (10-01) | open | new; see the entry | **yes**: flight identity, arrival slices and every sentence of the 15 flights; a harvest reclassification moves every downstream artefact |
| Stored tracks carry other aircraft's samples; no read-time position repair (10-01) | open | new; see the entry | **yes**: the arrival slices every model reads (if repaired in the harvest view) |
| A wake shortfall's end rounds `counted` a step short in the window loop (10-01) | open | new; see the entry | **yes**: the steps M4-in-windows (R37) trains a wake-shortfall follower on |
| The labeller fingerprint hashes content code, not only the vocabulary's format (10-02) | open | new; see the entry | no: a check, not a training input — but the fix changes what the stored specs record |
| The edge and executor fingerprints hash row-placement and data-plane code (10-02) | open | new; see the entry | no: a check — but the fix changes what the stored checkpoints and passed records name |

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

## 15. KRDU 14's 13 arrivals still render as "indeterminate" in the Evaluation view

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
(`4dTrajectory/ts_transformer/archive/two_tier_v2_2026_09/docs/2026-09-16_two_tier_transformer_feasibility.zh.md` §11.2), recorded so they
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
#18, #22; and by `dev-training-followups` (`dcbf6388`, merged `acb93b55`): #14, #21. What is left, one item each (numbers kept):

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

## Performance index: points the stage-2 review left for later (2026-09-24)

**Verified** (opus review of the performance-index change). Fixed 2026-09-25: the observed record's mass label says
which model the mass came from (`dynamics_source`), and the two readout scripts resolve a record with its own provider.
Left:
- A substituted type gets no mass, so its observed record falls back to the nominal 60 t (C25A/C525/PC24 6.8 t → 60 t).
  The gate never reads that mass; what mass an observed record of another airframe should carry is a judgement.
- Index data worth a look (judgement): B722's Poll–Schumann landing mass is 5.2 % above its FAA MALW, so its
  target is 136.4 kt against a published 133 kt (fixed, `acb93b55`: own types land at the published MALW); MD88 flies as B737 because a substitute must be a native
  airframe and its PS synonym MD82 is an own-parameter row; the similarity distance has no mass term, so LJ35
  (6.5 t) flies as B737 and GLF3 as B763 (dynamically similar, by the method's definition).


## A ts checkpoint does not record where its landing masses came from (2026-09-26)

**Verified** (opus review of `dev-training-followups`), the fix is **judgement**. Under `aircraft_filter=openap-direct`
`load_checkpoint` skips the performance-index check (`train.py:1417-1430`), and no ts identity records
`reference_speeds.json` (`reference_speeds_identity` is used only by `evaluation/metrics.py:177`). So a control-path
checkpoint trained before the landing masses moved to the published MALW would load silently and be rolled out at the new
masses (17 OpenAP-direct types move by ≥ 1 %); under `modelled` / `all-flights` it is refused only because the index's basis
texts happen to change. Recording a reference-speeds / landing-mass identity in the checkpoint and checking it under every
filter that uses dynamics is a payload change — the user's decision. The two-tier chain is not exposed: its sentence
artefact records the aircraft tables and is rebuilt.

## An alias-resolved identity is not recorded as one (2026-09-26)

**Judgement** (opus review of `dev-training-followups`). An OpenSky code resolved through `MARKETING_ALIASES` is labelled
like a direct designator (`direct_designator` / `opensky_icao24_validated`); nothing in the identity says an alias was
applied. Recording it changes the identity payload, so it needs its own schema name.

## `evaluation.metrics.READABLE_REPORT_SCHEMA_VERSIONS` reads four report versions (2026-09-25)

**Verified by reading.** `READABLE_REPORT_SCHEMA_VERSIONS` accepts v6, v7, v8 and v9 for a consumer that reads only the
fields they share; its one consumer is ts `data/lateral_eligibility.py`, which builds the lateral-pass roster the ts
datasets filter on. A version list read by a consumer is the schema-version branch the no-compatibility rule (2026-09-19)
forbids; refusing everything but v9 would refuse the rosters built from older reports — the user's call, and a change
on the training data plane.

## `aeroviz-4d/python/requirements.txt` still lists `pyproj` (2026-09-26)

**Verified** (opus review of `c708a497`). Its last importer, `flight_scenarios.datum.geoid_undulation_m` (the EGM96
grid), was deleted 2026-09-26; no live `.py` imports `pyproj` (the archived exporter under
`ts_transformer/archive/instruction_vocabulary_2026_09/` still names the deleted function and is not live). The file is a
loose list, not the environment spec (`docs/environment.md`): drop the line, or say what it is for.

## The land law leaves a shallow final class's tube near the threshold on some profiles (2026-09-26)

**Verified** (opus review of `9557315d`, reproduced on `dev-two-tier` before it). A synthetic straight-in said as three
angle words shallowing 4.3° → 3.2° → 2.3° keeps "descend to land" inside its tube when the last (2.3°) leg is 15 rows,
but leaves it for 3 of 131 rows at 20 rows (and at 30 and 50), crossing at the admitted upper edge (TCH + 12.5 m): the
same under the law before and after the glidepath floor, so it is older than it. `tests/test_autopilot.py`'s shallowing
case uses 15 rows. Judgement: the reach test (`in_reach`, from the tube's lower edge at the steepest class) and the
crossing point's clamp to the admitted heights meet at an edge; look at it with the next executor change. Executor v11 (2026-09-27) changed only the law below the published glidepath
(join from below) and did not address or re-measure this profile; still open for the executor after v11.

## ts: five test files still touch `sys.path` (2026-09-26)

**Verified** (split off the resolved `docs/` scripts entry). `test_architecture`, `test_final_approach_geometry`,
`test_guidance_skeleton_mirrors`, `test_import_boundaries` and `test_prior_procedure` add to `sys.path`; three of them add
`4dTrajectory/optimization`, which `tests/conftest.py` does not. **Judgement**: move that one path into `conftest.py` and
drop the per-file lines, or say in each why it stays.

## `instructions/grammar.py` is outside the labeller sha (2026-09-26)

**Verified** (`instructions/artefact.py` `LABELLER_MODULES`, `dev-two-tier` `f4a63bca`). The vocabulary's compatibility
rules the prior's speaker masks by (`grammar.step_allowed`, prior design §5.1) live in `instructions/grammar.py`, which is
not among the files the labeller sha covers; a change to it alone would change what every model may say with no identity
moving. Unchanged since it was written (`0bbe6abf`, 2026-09-25). Adding it to `LABELLER_MODULES` changes the labeller sha
every sentence artefact and every prior's `config.json` records, so it waits for the next vocabulary spec (judgement).

## The terrain downloader overwrites the shared `download_manifest.csv` (2026-09-27)

**Verified** (`tnm_elevation_downloader/download_tnm_elevation.py` `write_manifest` opens the manifest with `"w"`;
`preprocess_aeroviz_airport.sh` step 7 passes no `--manifest`). The default manifest
`data/usgs_tnm_elevation/download_manifest.csv` records every airport's tiles, but one airport's download
rewrites it with that airport's rows only, so the one-click script run for a NEW airport erases the other
airports' provenance. Avoided for KAUS by downloading first with
`--manifest data/usgs_tnm_elevation/download_manifest.KAUS.csv` (the script then skips its download step because
the tiles exist). Fix (judgement): a per-airport default manifest `<out>/<ICAO>/download_manifest.csv`, or merge
rows by airport group instead of rewriting.

## A free sentence is judged against the runway pointed at the executor's stop, not at its end (2026-09-27)

**Plausible** (opus review of executor v11 milestone 2, `prior_free_generation.flight_rows`). A free sentence's outcome is
`judge.outcome_of` against the runway pointer of the LAST step the executor flew (`runway[-1]`). The executor flies on after
three outcomes the judge reads earlier (crossing without the capture, another runway's threshold, the stall cut-off), and a
model may re-point the runway in those extra steps, so the outcome can be read against a runway pointed after the flight
ended — in principle turning another runway's crossing into a crossing of the "pointed" one. Not observed. Fix (judgement):
read the pointer in force at the judge's end row (a second `outcome_of` against it, or stop the executor at those outcomes);
it moves free-generation outcomes and the landing reward, so it waits for a stage that re-reads them.

## `faa_separation(visual_parallels=True)` is a second, different "visual" (2026-09-27)

**Verified** (opus review of the two separation readings, `11239312`). `runway_schedule.faa_separation(visual_parallels=True)`
turns every parallel relation into `INDEPENDENT` and drops the diagonals. That is a second definition of "visual", and it is
not the one `inference.separation.VISUAL` applies: combined with `losses(..., IFR)` it still judges a turn-on beside a
parallel final by radar or vertical. Only the archived `runway_intent_r3` and `test_runway_schedule.py`'s
`visual_parallels` test call it. Since 2026-09-27 it also contradicts the decided visual reading: it frees close pairs
because "pilots maintain visual separation", which the multi-aircraft reading rules out (design §9 item 16). Fix
(judgement): delete the flag and its test once nothing live needs the runway-intent reading, or name it after what it
does (parallels as independent).

## The labeller sha is over bytes and checked against the running code: one edit deadlocks the chain (2026-09-28)

From the audit of the two-tier chain's identity checks (2026-09-28, `dev-two-tier` `97c579bd`), like the next five
entries. **Verified.** `instructions/artefact.py` `labeller_source_sha256` hashes the BYTES of `LABELLER_MODULES` (14
files) and `final_approach/crossing.py`, so a comment moves it (the executor's hash stopped doing that on 2026-09-27).
`autopilot/replay.py` `open_executor` refuses an artefact unless its recorded labeller is the running code
(`require_current_labeller`) and the spec's labeller is too (`replay.py:79-84`); `experiments/prior_train.py`
`load_prior` refuses a prior whose recorded labeller is not the artefact's (`:107`). Together: after any edit to one of
those 15 files the executor refuses artefact v5, and an artefact rebuilt by the new code refuses base, landing and
augmented — retrain, or revert. This is why those files are on the frozen list and the training-affecting fixes waited
on `dev-training-followups`. What the check guards is already checked by result: every exporter and the backend re-read
each flight they show or fly and demand the stored sentence (`training_files.require_stored_sentence`, backend
`fly.py:137`). Fix (judgement): hash the logic (`autopilot.spec.logic`); keep the running-code refusal only where one
artefact is written (`instruction_labels`); record the labeller elsewhere as provenance; bind a prior to an artefact by
a digest of the arrays it trained on (`sentences_*` and `signals_*` of train / select / val) instead of the code that
wrote them — C26's rule (identity is the data, not its producer). Existing priors need the digest recorded once beside
them (writes next to published models: the user's call). Makes the `grammar.py` entry moot.

## The executor spec refuses any other code, over 28 modules including all of `config.py` and `data/dataset.py` (2026-09-28)

**Verified.** `autopilot/spec.py` `executor_source_files` covers `autopilot/`'s 15 files and the 13 repository modules
they import directly, among them `ts_transformer.config` (2,989 lines, the whole ts line's), `ts_transformer.data.dataset`
(1,691 lines), `geokit` and `trajectory_data_process.harvest.airports`; `require_current_executor` refuses a spec written
by other code — in the replay, every exporter and the backend's spec choice. `outputs/POOLED/executor/v5`–`v10` are six
specs with the same params sha `0d6a68a92c6f`, differing only in the source hash; v8 was refused because its hash covered
the archived CAT-K code. A full spec with its replays is ~2 GB, and each new one forced the val re-reads
(`prior/v3_reread_v9`, `_v10`, `_v11`) and an overlay re-export. It did catch a real change once: with `dataset.py` in
the hash, a change moved 711 of 1,000 select flights' mass or approach speed (CHANGELOG 2026-09-26). The rule is the
user's (2026-09-27: a spec and every readout flown with it describe the code that runs). Fix (judgement, the user's
decision): decide by result — re-fly a fixed sample (the stored train replay's flights, some per airport) and compare
row by row, as `aeroviz_backend/autopilot_segment/check_single.py` does for the single-flight executor; equal → the spec
stands for the new code (the new code identity recorded beside it), different → a new spec; and extend
`autopilot/flights.require_same_flight` to what the executor reads from a series (type, mass, approach speed), so
`config.py` and `dataset.py` can leave the hash.

## Two-tier data are bound to each airport's whole arrival manifest by its bytes (2026-09-28)

**Verified.** `autopilot/flights.py` `rebuild_series` (`:55`) and `instructions/training_files.py`
`runway_hae_minus_msl_m` (`:251`) refuse unless each airport's `arrivals/manifest.json` has the sha256 the artefact
recorded. The file holds every record of the airport (KRDU 24,202), and `--evaluate-only` / `--merge-source` rewrite it
(root Open Items): any such rewrite refuses the replay, the exports, the live flights and the traffic loop, even when
not one flight they use changed — though `require_same_flight`, right after, compares every rebuilt flight with its
stored signals (1e-6). The same class as C26 (2026-09-07: a byte-bound roster identity refused every checkpoint). Fix
(judgement): drop the file comparison, keep the per-flight one, and compare the one runway offset a flight uses.
`flights.py` is inside the executor hash, so under today's rule the fix itself needs a new spec: do it with the previous
entry.

## The frontend pins the vocabulary spec's sha (2026-09-28)

**Verified.** `aeroviz-4d/src/data/trainingSample.ts` `TRAINING_SPEC_SHA256` (`:53`) is compared at `:620`, `:641` and
`trainingOverlays.ts:736`, although the sample carries the vocabulary's tables and the reader checks them. Its cost is a
one-line edit when the spec is re-measured, never a re-export. It is also the only thing that hides the superseded
flight-split set `instruction_v3` (sample v7, spec `0b4ea75be36d`), still listed in every airport's index (older sets
kept as they are, the user 2026-09-26). Judgement: low value; remove it only once that set leaves the index.

## The single-flight executor's pin refuses at run time, not in the tests (2026-09-28)

**Verified.** `aeroviz_backend/autopilot_segment/single.py` `require_mirrored_source` (`:118`) runs when the backend
picks its spec (`backend.py:148`): a logic change in any of the 32 files it mirrors makes the backend refuse every live
flight until `single.py` is ported and `MIRRORED_SOURCE_SHA256` moved. `tests/test_single_executor.py` compares the two
executors through every mode and limit, and `check_single` over a fleet. Judgement: the pin belongs in the suite (a
change not ported fails a test); at run time it only reports the same thing later, to the user.

## A glidepath stop overrides a judged event that came before it (2026-09-30)

**Verified by reading** (opus review of `prior_generation_records`, dev-generation-records). `prior_free_generation
.flight_rows` (`:352-353`) makes a sentence's outcome `below_glidepath` and its end `(stop + 1) × step_s` whenever
`glidepath_stops` found a stop, whatever `judge.outcome_of` read — even when the judge's event came at an EARLIER row.
`glidepath_stops` scans the step boundaries up to the executor's `done_cycle`, and the executor flies on after three events
the judge reads earlier (a crossing without the capture, another runway's threshold, the stall — the stall counts as
`dynamics_failure` but does not end the flight). So such a sentence's outcome is `below_glidepath` although it ended
before, and its end runs past the judged event — `prior_generation_records`' record then keeps states after it (for a stall,
states of the failure, against `executor_replay.executor_forecast`'s own rule). Not counted how often. Judgement: a stop
counts only before the judge's end row (`stops.step` boundary ≤ `outcome.end_row`). It moves free-generation outcomes and
stage 2's reward (`prior_augmented_reward` reads `said_rows`), so it waits for a stage that re-reads them. Same family as
"A free sentence is judged against the runway pointed at the executor's stop".

## The read-back altitude chart's coloured labels are grey (2026-09-30)

**Verified by reading** (opus review of `dev-training-roles`). `components/training/ReadbackAltitude.tsx:66-67` and `:78` draw
`<text className="training-readback-tick" fill={…}>` — the threshold label in `TRAINING_DESIGNATED_COLOR`, the angle words'
labels in their column's colour — and `index.css`'s `.training-readback-tick { fill: … }` overrides the attribute (an SVG
presentation attribute loses to any stylesheet rule, AV40), so they render in the tick grey. Fix as `dev-training-roles` did
for the window strip's ✕ and the sentence bar's model end time: `style={{ fill }}`. Not checked in the browser.

## A low go-around is stored as the landing: the best-aligned crossing, not the last (2026-10-01)

**Verified** (opus review of R40 `go_around_census`, and its formal run `outputs/POOLED/traffic/go_arounds_20261001/`).
`trajectory_data_process/harvest/threshold_event.py:113-119` takes as the landing the threshold crossing under 100 m with the
smallest |cross|, not the last one. When a low go-around crossed better aligned than the landing that followed, it becomes
`landing_sample_index` — so the flight's `landing_time_utc` (part of its `flight_key`), its arrival slice and its sentence
end at a go-around. On the training days 15 flights hold a level 150 m above their stored landing and end back on a final
(`landings_not_the_last` → "landed later"; 9 labelled: KRDU UAL2485, RPA9931, KSTL DAL1371, SWA3286, BOE57A, UAL214, N320TS,
KSMF SWA1521, KMSY GJS4412); the real landing comes a median 669 s later. 73 more land then leave (touch-and-go circuits,
38 labelled) — their stored landing is a touch-and-go, which the landing rule reads as a landing anyway. Judgement: take the
LAST qualifying crossing (or the last before the sustained ground run). It changes identities and slices of those flights and
needs `--reclassify-existing` and a rebuild of everything downstream — the user's call.

## Stored tracks carry other aircraft's samples; no read-time position repair (2026-10-01)

**Verified** (R40 `go_around_census`). Stored tracks hold the odd run of another aircraft's samples — a position 5–40 km off
with that aircraft's altitude (KRDU RPA5593 sample 570: 15 km away and 400 m up between two samples on its final; MXY1067
samples near 6, 36, 39; N48CL 36–38; ASA2625 633). `harvest/altitude_filter.py` repairs altitude needles only, and a run of
three is the median of its own five-sample window, so the view passes them through. R40 sets them aside locally (over 1 km
from the median position of the 5 samples either side): 206 samples over the 45,075 training-day landing tracks. Not counted
how many fall inside arrival slices, nor what they do to the labeller or the evaluation's observed baseline. Judgement: a
position repair beside the altitude one in the derived view (`store.read_track_view`) — it would change arrival slices every
model reads, so it waits for a rebuild.

## A wake shortfall's end rounds `counted` a step short in the window loop (2026-10-01)

**Verified** (the 7.7 review, `dev-window-rewind`). `WindowLoop.results` (`experiments/traffic_window.py`) counts a judged
aircraft's steps as `round((end["t_s"] − first_s) / step_s)`. A loss in an episode ends on the grid, so that is the step the
judge ended it at. A wake shortfall at a landing (`Judging.landing`) stamps the landing's time instead — off the grid,
inside `(t − step, t]` of the step whose landing check ended it — and the aircraft had spoken at that step. When the landing
falls in the first half of the step, `round` gives one step fewer. **Judgement** (not measured): `counted` is what R34
reports as the steps counted and what R37 trains on, so a wake-shortfall follower trains on one step fewer than it said
before its end — at most one step of one sentence a shortfall, a small and one-sided bias. R43 reads that step as the step
whose time reaches the end (`traffic_window_rewind.own_step_at`). The fix in the loop (the same `ceil`) changes R37's
training rows: it waits for the next window training.

## The labeller fingerprint hashes content code, not only the vocabulary's format (2026-10-02)

**Verified** (branch `dev-scene-time-grid`). `instructions/artefact.py` `labeller_source_sha256` hashes the BYTES of
`LABELLER_MODULES`, `signals.py` among them — the projection of a built flight into the airport frame, which decides the
rows' content, not what a word means. The user's rule (2026-10-02): a fingerprint binds format only; code that only changes
the data's content, or whose behaviour a test can pin, needs no hash (the data are iid draws of one process; C33 replaced
the executor's source hash by a conformance check the same way). Because the hash is over bytes, the rows' re-cut onto the
UTC steps had to live outside `signals.py` (`data.dataset.on_utc_steps`; artefact v6 keeps v5's spec), and two comments in
`signals.py` are stale for v6 and cannot be fixed without changing the hash: `FlightSignals.entry_time_utc` says "the
arrival slice's entry, its first kept sample" — since v6 it is row 0, the first even UTC second at or after that sample —
and the module docstring's "2 s resampling" does not say where. **Judgement**: hash only the modules that define the
vocabulary and the words' meaning (`spec.py`, `words.py`, the reading rules), and pin the rest by a behaviour check on
fixed flights (golden sentences), as C33 does; fix the two comments in the same change. Every stored spec records the
current hash, so the change needs a rule for the specs already measured (the user's decision, case by case). The same
change can name the rows' grid in the signals' format (`SIGNALS_SCHEMA` lives in `signals.py`, so it could not be bumped
for v6): today a v5 artefact is told from a v6 one by its data alone — `prior.scene.presence` refuses rows off the steps,
by name (the 2026-10-02 review's point 6).

## The edge and executor fingerprints hash row-placement and data-plane code (2026-10-02)

**Verified** (branch `dev-scene-time-grid`). `experiments/traffic_scene_data.py` `EDGE_SOURCES` hashes `prior/scene.py`,
`prior/scene_data.py`, `experiments/traffic_scene_data.py` and `experiments/traffic_speaking.py` beside the feature
definitions (`inference/scene_edges.py`, `runway_schedule.py`, `instructions/airport.py`); the re-cut onto the UTC steps
changed the first group and every stored scene / traffic prior now refuses (`prior_train.load_prior`) — wanted this time,
since the traffic model is retrained on v6, but the refusal comes from where rows are placed, not from what an edge
feature means. The executor's `spec.executor_source_files` takes every module `autopilot/*.py` imports directly, so
`data/dataset.py` (through `autopilot/flights.py`) is in it: a data-plane change asks for a new conformance pass although
the conformance check already compares each flight's inputs by digest. **Judgement**: hash the feature definitions only and
pin the placement by tests; leave the data plane out of the executor's code and let the input digests carry it.

