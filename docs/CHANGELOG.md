# AeroViz-4D Development Changelog

Dated log of significant changes, root causes, and decisions, referenced from `CLAUDE.md`. This file is deliberately NOT loaded into every session — read it when investigating history: why a design is the way it is, when/why a default changed, what a past bug or postmortem looked like, or which outputs a change made stale. Append new entries at the top (`### YYYY-MM-DD — title`); when a change produces a durable fact (gotcha, default, contract), also update the corresponding section in `CLAUDE.md`.

Entries verified via full test suites + tsc + vite build at the time; "verified in-browser" noted only where done. Merged same-day, same-topic entries.

### 2026-07-19 — flyability check, instance-norm re-ablation on real data, predictions in the frontend

Three things asked for together: the post-hoc flyability check (route 1 of the README's four),
re-ablating `--instance-norm` on real data, and getting prediction results to render in the
frontend alongside the optimizer's.

**Post-hoc flyability (`4dTrajectory/ts_transformer/flyability.py`, 16 tests).** The
load-factor point-mass model inverts in closed form — `A = psi_dot V cos(gamma)/g = n sin(mu)`,
`B = gamma_dot V/g + cos(gamma) = n cos(mu)`, so `n = hypot(A,B)`, `mu = atan2(A,B)`; `n` fixes
`Cl`, `Cl` fixes drag, and `T = m(V_dot + g sin(gamma)) + D` closes it. One pass, no solver, no
casadi, so it lives in the torch env. Earth-frame transport terms are subtracted first.
`predict` writes `flyability_report.json` and prints a summary line.

- **The calibration matters more than the check.** First run against REAL flown tracks scored
  **0/149 fully flyable** — those are trajectories real aircraft flew, so the check was wrong.
  Cause: `thrust_negative`. Median required thrust on a real arrival is **0.43 kN** (idle), and
  a negative requirement means the aircraft needed more drag than a clean airframe has —
  speedbrake, flaps, gear. Every approach does this; one clean-configuration polar cannot
  represent it. Reclassified as SOFT (reported, not counted unflyable), and the report leads
  with the delta against the observed tracks measured by identical code, because both sides
  carry the same polar bias. **The observed baseline is the floor, not 100%.**
- `Cl_max` from `aero_params_for_aircraft` (2.7 for an A320), NOT `LoadFactorSimulator`'s
  hardcoded 1.5 — they disagree by 80% and `aero_params.py` is the documented source of truth.

**Instance-norm re-ablation on real KRDU data — the synthetic OFF default holds.** All 8 cells
(2 models × 2 horizon modes × on/off), same hyperparameters, each graded on its own
checkpoint's test split; artifacts in `4dTrajectory/outputs/KRDU/_ablation_norm/`, roll-up in
`ablation_results.json`.

- **OFF wins 19 of 20 accuracy comparisons, one tie**: every cell on val loss, FDE, ADE p95 and
  lateral p95; 3 of 4 on mean ADE with PatchTST window a dead heat (2580 vs 2571 m — the cell
  where OFF wins the other four metrics). A sweep that lopsided is not run-to-run variance,
  which was the open question: individual gaps here are much smaller than on synthetic data
  (1.2–2.7× vs 2.4–6.5×), and an earlier partial pass on ONE metric had shown an apparent
  reversal in that same PatchTST window cell. It did not survive consistent scoring.
- **The signature is lateral p95**: all four instance-norm-ON cells land at 14.28–14.50 km —
  near-constant across both architectures and both horizon modes. That is a model that cannot
  place the endpoint at all; strip the absolute level and the prediction ends at a distance set
  by the frame, not by the flight. OFF spans 2.6–8.5 km, i.e. it varies with the flight.
- **Flyability moves the OPPOSITE way** (ON better in 3 of 4 cells; PatchTST window 89.3% vs
  29.6%, i.e. the configuration 2.2× worse at the threshold looks three times more flyable).
  Instance norm is not flying better, it is predicting blander paths — easy to fly, far from
  the truth. A straight line is perfectly flyable and completely wrong. Recorded in the README
  because it is the standing argument for never reading flyability alone. (iTransformer full is
  the exception at 46.1% off vs 34.9% on.)

**Headline KRDU runs retrained, and two published conclusions did not survive it.** The 4
published checkpoints carried a split that `hash(flight_key)` reproduces for only 552/995
flights — they predate the per-flight-hash split fix, exactly the "retrain before comparing"
note in the entry below. The partition itself is clean (train/val/test verified disjoint), so
the published numbers were not wrong, just on a split no current run can reproduce and not
comparable with the ablation. Retrained on the current keying (702/141/152):

- **"The compounding cost lands in the tail, not the mean" — withdrawn.** It rested on lateral
  mean 1.5× vs p95 2.2×; on the new split the two ratios are equal to within noise
  (iTransformer 1.55× vs 1.61×). A tail effect survives in FDE (PatchTST p95 1.78× against a
  1.36× mean) but that is a different claim than the one written.
- **"Zero gate passes in all four runs" — withdrawn.** iTransformer now passes 3/152 (full)
  and 1/152 (chained). The substance holds (2% is not a usable predictor) but "zero, always"
  was a property of that split.
- **Survived:** one-pass full beats chained window on whole-approach lateral error for both
  models on both splits (1.5–1.6× mean, 1.5–2.1× p95), and the short-lead/long-lead crossover
  between PatchTST and iTransformer (now cleanest within the full-mode pair: 184 vs 571 m at
  10 s, reversing by 300 s).

Standing lesson recorded in the README and CLAUDE.md: **treat any single-split margin under
~1.5× as provisional** — that is the size of effect a split change moved here.

**Flyability shipped with a wrong assumption, caught by its own guard.** The check graded a
whole batch against ONE envelope, documented as safe because "every harvested arrival is type
UNK and resolves to the single `--aircraft-type` fallback". A boundary assertion added to state
that assumption fired on the first real batch: `_resolve_aircraft` falls through to an
**`icao24` → OpenAP lookup** that recovers the real airframe — 20 distinct types across 400 KRDU
arrivals, A320 only 224 of them. So ~44% of every batch was being judged by an A320's `Cl_max`
and max thrust, invisibly, because the report never named the airframe it used.
`report_for_records` now takes one `Aircraft` per flight, builds one envelope per distinct type,
and the roll-up carries `fleet` + `envelopes`. All flyability numbers moved: the observed-track
baseline went 58.4% → **63.2%**. The CLAUDE.md/README claims about UNK were corrected.

**`summary.json` now carries an `accuracy` block** (+ per-row `ade_m`/`fde_m`/`overlap_steps`).
A batch's error against the observed tracks is its headline result and it existed only as a
printed mean — comparing eight ablation cells meant scraping stdout. Mean AND p95, because
chained window-mode error compounds into the tail. `overlap` is a required argument to
`write_batch`, not optional: an optional metric is one that silently goes missing.

**Predictions render in the frontend.** `build_scenario_comparison_czml.py` detects the record
schema (`optimizer_states`/`simulator_states` vs `predicted_states`) and emits `pred-` entities
for a prediction batch; the entity-id prefix is already what the frontend keys `kind` off, so
`predicted` gets its own purple colour and legend checkbox. Two follow-on fixes:

- **Off-target recolouring restricted to the optimizer schema.** A forecast essentially always
  misses the 106.75 m gate, so it marked 27/27 groups off-target and repainted every prediction
  yellow — the kind colour was never once visible and a marker that fires on everything carries
  no information. `properties.status` stays accurate; deviation is reported by the evaluation
  report and comparison index.
- **The frontend repaint skip was keyed on the wrong thing.** It skipped legend repaint whenever
  `status === "offTarget"`, but what it exists to preserve is a baked VERDICT colour (the
  yellow). Predictions never get that bake yet are always `offTarget`, so they rendered from the
  CZML — matching their legend swatch only because `PREDICTION_COLOR` and the TS legend entry
  happen to hold the same RGB. Now keyed on whether a verdict colour was actually baked.

**Not verified in-browser** — the Chrome extension was not connected for this session. Backed by
tsc clean, 451 frontend tests, `npm run build`, 52 ts_transformer + 25 CZML-builder tests, and a
structural check of the published artifacts against every contract point the frontend reads
(categories.json fields incl. `constrained`, entity-id → kind mapping, status, position samples).

### 2026-07-19 — code review fixes: ts_transformer contracts, env resolution, identity single-sourcing

Applied the 15 findings of a full-diff review (ts_transformer + the uncommitted env-script changes):

- **train.py**: a non-finite train/val loss now RAISES ("training diverged…") instead of sailing through the best-val bookkeeping — previously a run that went NaN before any finite improvement wrote the last (NaN) weights as a "successful" checkpoint and emitted literal `NaN`/`Infinity` into history.json. `load_checkpoint` now uses `weights_only=True` (payload is tensors + primitives) and refuses a checkpoint whose serialized channel order differs from `channels.CHANNELS` (a same-length reorder loaded cleanly and silently mis-mapped every channel). Flights too short to yield one training window (window mode: `seq_len + pred_len`; build only requires `seq_len + 1`) are excluded — counted, not silent — before the split, so they no longer occupy split slots and produce the misleading "empty window set" abort.
- **Aircraft type is part of the checkpoint**: `TSConfig.aircraft_type` (default A320, moved to config.py) is serialised with everything else; predict defaults to the train-time value instead of its own CLI default, and warns when explicitly overridden — the type sets target Vref/TCH, i.e. the ENU frame and the gate target the normalizer stats were fit under.
- **predict --device**: the compute device is a runtime property, no longer read from the checkpoint (an explicit-`cuda` checkpoint was unusable on a CPU-only machine with no override).
- **Horizon cap is stated, never silent**: window-mode chaining still stops at 300 steps (600 s), but a forecast that ends short of the threshold is flagged `horizonCapped` (record source), `horizon_capped` (summary row), and predict prints a per-batch WARNING — previously those ~2% of flights were graded as huge gate failures indistinguishable from model error.
- **Split stability**: train/val/test assignment is now a per-flight sha256 of `(seed, flight_id)` instead of a positional permutation — adding/removing one flight to the harvest reshuffled the entire split (old test flights silently entered training on a retrain). Fractions are now approximate; empty train/val raises with the real cause (tiny datasets previously rounded val/test to zero and died later blaming window sizes). NOTE: this reshuffles the split once relative to the existing KRDU checkpoints — retrain before comparing new runs against them.
- **synthetic.py**: icao24 derived via `zlib.crc32(runway)` instead of the process-salted builtin `hash()` — identically-seeded synthetic data now has identical flight identities across interpreters (previously `predict --split test` on regenerated data intersected on zero flights).
- **Identity + filename single-sourcing**: `flight_key` moved to `flight_scenarios/identity.py`; `scenario_optimization._scenario_filename` and ts dataset/export both import it (the two copies had already drifted: `flight{i}` vs `scenario{i}` fallback — unified on `flight{i}`). Record-filename suffixes + `REFERENCES_DIR` hoisted into `optimization/evaluation_export.py`, imported by both writers; the "MUST match" mirror comments are gone.
- **dataset.py**: falling through to `*_landings.json` (a download-only dir with no `*_arrivals.json`) now prints an explicit UNTRUNCATED-tracks warning — a different task/duration distribution than the windows were sized for.
- **Env resolution deduplicated + made content-aware**: new `scripts/activate_aeroviz_env.sh`, sourced by `run_all_tests.sh` (warn-and-continue) and `start_aeroviz_fullstack.sh` (abort). Candidates are probed with `import casadi` instead of trusted by name (on this machine `aviation` exists but is another project's env — both scripts previously accepted it when active and fell back to it when `aeroviz` was missing or `AEROVIZ_CONDA_ENV` was mistyped; the launcher then crash-looped the backend on `import casadi`). An explicit `AEROVIZ_CONDA_ENV` is now the ONLY candidate. The launcher ACTIVATES the env rather than direct-exec'ing `envs/<env>/bin/python`, so activate.d hooks (the libstdc++ `LD_LIBRARY_PATH` fix) apply to the backend subtree; missing conda now reports "conda is not on PATH" instead of probing `/envs/...`.
- **Docs**: CLAUDE.md's Key Defaults ts bullet updated to the real 60/300 window sizes and the measured-distribution rationale (it still carried the superseded 30/150 + "~3.5–5 min arrival" story that config.py refutes); requirements.txt pin rationales corrected (the torch comment claimed `weights_only=True` while the code passed `False` — now the code matches the claim; the numpy `copy=None` rationale matched nothing).

Follow-up pass over the review's below-the-cap cleanup findings (same day):

- **forecast.py**: the HORIZON_FULL branch of `forecast_approach` was a copy of `recursive_forecast`'s single-pass body that also skipped its anchor validation — both modes now run through `recursive_forecast` (full mode's `pred_len` covers `max_steps`, so it is one pass; an explicit `anchor < seq_len - 1` now gets the clear ValueError instead of a cryptic vendor shape error). `_forward` forces `model.eval()` (dropout noise from a train-mode model would compound through the chain). `default_anchor` dropped its never-read `series` parameter.
- **models.py**: PatchTST now receives `act=config.activation` (the vendored Model takes activation as a bare kwarg, so `TSConfig.activation` previously applied to iTransformer only while the checkpoint recorded it for both).
- **metrics.py**: `displacement_metrics` folded into `trajectory_metrics`; `error_components` computed once per call (was twice, with per-step displacement recomputed twice more) via a `displacement_grid` key; `_positions` uses `channels.POSITION_IDX` instead of re-deriving it. `_spread` stays vectorised — delegating to the stdlib-only `evaluation/stats.signed_spread` would sort millions of boxed floats in `evaluate_split` — but a new seam test pins the two equal on the same input, replacing the mirror comment with a checked property.
- **export.py**: `PredictionRecord.source`/`final_time_s` became read-through properties (three stored copies of `final_time_s` per record could disagree after a future edit; `evaluation` rejects `final_time_s != states[-1].t`); the whole observed track is converted to states ONCE and the anchor sample + reference span are slices of it (the per-sample GeodeticState loop previously ran twice over the tail); summary rows go through the new `evaluation_export.summary_row` (shared with `scenario_optimization._summary_record`, which now wraps it); a redundant `min()` removed.
- **train.py**: `_predict_split` stores decoded windows as float32 (float64 doubled the live peak — hundreds of MB at full-mode scale on the 16 GB swap-bound machine — for precision metre-scale metrics cannot use); the duplicated val `evaluate_split` call in both ternary arms collapsed.
- **__main__.py predict**: the split filter now runs on the RAW flight dicts (keyed identically to `build_series` via `flight_key`) BEFORE the expensive series build — a default `--split test` predict previously built and discarded ~85% of the work. Verified live: "built 1/1 series" for a 14-flight file.
- **dataset.py**: the track-span check moved BEFORE `build_scenario` (too-short flights no longer pay aircraft resolution + per-sample least-squares fits before being skipped; the raw-waypoint span is the identical value); dead code removed (`Normalizer.decode_torch`, `FlightSeries.duration_s`/`.source`).
- **channels.py**: dead `VELOCITY_IDX` + `Frame.to_dict`/`from_dict` removed; new `Frame.latlon_from_en` single-sources the inverse projection, used by `states_from_channels` AND `synthetic.py`'s waypoint generator (which previously hand-rolled a fifth copy of the frame formula; it now builds a `channels.Frame` and dropped its direct geokit constant imports).
- **package.json**: `npm run backend` resolves its interpreter through `scripts/activate_aeroviz_env.sh` instead of bare `python` (it was the third, uncoordinated env-selection seam).
- Deliberately NOT applied: batching the predict loop (would complicate the single-flight inference seam for a non-bottleneck), early-stopping the window-mode chain at the threshold (changes the measured object — `truncate_at_threshold` takes a global argmin), the vendored PatchTST `pv()` NameError on non-default positional encodings (`layers.py` is documented "copied whole, unmodified"; fixing it breaks the byte-identical vendoring contract — landmine noted here instead), and a src-layout re-packaging of ts_transformer's flat modules (the sys.path-shadowing hazard is real but latent; disproportionate to fix now).

### 2026-07-19 — ts_transformer: first REAL-data run (KRDU, 995 arrivals) — both models × both horizon modes

Harvested ADS-B landed (3747 arrivals, 5 airports, 815 MB). Full matrix trained and graded on KRDU: 995 arrivals over 6 runways, split by flight 697/149/149, 96k training windows, 120-epoch cap / patience 15 / `lr=5e-4`, RTX 4060. All four prediction batches graded by `python -m evaluation`. Artifacts under `4dTrajectory/outputs/KRDU/ts_{model}_{mode}/` and `ts_pred_{model}_{mode}/`.

**Displacement error at matched lead times** (the only cross-mode-comparable axis — headline ADE/FDE average over different horizon-length distributions and must not be compared):

| model | mode | 10 s | 30 s | 60 s | 120 s | 300 s | 600 s |
|---|---|---:|---:|---:|---:|---:|---:|
| iTransformer | window | 259 | 390 | 687 | — | — | — |
| iTransformer | full | 587 | 611 | 840 | 1502 | 3227 | 6135 |
| PatchTST | window | 158 | 409 | 995 | — | — | — |
| PatchTST | full | 253 | 399 | 830 | 1914 | 3983 | 7142 |

**Whole-approach prediction graded at the threshold** (149 test flights, directly comparable):

| model | mode | lateral mean | lateral p95 | path deviation |
|---|---|---:|---:|---:|
| iTransformer | full | **1070 m** | **3136 m** | **1895 m** |
| iTransformer | window (chained ×10) | 1594 m | 6898 m | 1918 m |
| PatchTST | full | 1804 m | 4666 m | 2488 m |
| PatchTST | window (chained ×10) | 2815 m | 7354 m | 3152 m |

Findings: (1) **the cost of chaining lands in the tail, not the mean** — chained-window vs one-pass-full is 1.5× on lateral mean but 2.2× on p95, because once one chained pass goes wrong the next nine extrapolate from a wrong history; training directly for the long horizon beats chaining a short one. (2) **iTransformer beats PatchTST at long lead for a structural reason** — PatchTST is channel-independent (`TSTiEncoder`), so it cannot represent the east/north coupling of a turning aircraft, while iTransformer attends *across* variates; note the reversal at 10 s lead (PatchTST 158 vs 259 m), where the aircraft is nearly straight and independence costs nothing. (3) **0/149 gate passes in all four runs is the honest result** — 106.75 m is FAA containment for a planned/flown approach, not a forecast-accuracy target; the number measures the gap between a statistical prediction and a certifiable trajectory. (4) Real is much harder than synthetic (423 vs 286 m ADE): synthetic approaches are straight-in, real ones are vectored, and *when* the turn onto final happens is a controller's decision a single-aircraft model structurally cannot see.

**Three real bugs the first real-data run exposed**, none of which a loss curve would have shown:

- **Silent training-data contamination.** A harvest directory holds five overlapping views of the same flights — `*_arrivals.json` (truncated, the training input), `*_landings.json` (SAME flights untruncated), `*_combined_czml_input.json` (all runways merged), plus `*_heading_rejected.json` / `*_local_rejected.json` (tracks the harvester explicitly THREW OUT). `glob("*.json")` loaded every flight three times over plus the known-bad ones. `dataset.select_flight_files` now takes the first matching pattern only, never mixes, always excludes `*_rejected*`, and prints what it skipped.
- **Aircraft type is `"UNK"` for all 3747 flights** and `flight_scenarios._resolve_aircraft` raises rather than guessing — the batch died on flight #1. Added `--aircraft-type` (default `A320`, printed every run). Not cosmetic: it sets the target state's Vref and threshold-crossing height, i.e. what the gates measure the final state against.
- **Sizing was wrong twice** (see the environment entry below for the corrected defaults).

### 2026-07-19 — Environment consolidated into `aeroviz`; the `aviation` name collision documented

`aeroviz` (py3.12) is now the single thesis environment — `torch` installed alongside the existing acquisition (`traffic`, `pyopensky`), CIFP (`cifparse`, `arinc424`), `casadi`/IPOPT, `openap` and conda-forge geospatial stacks. `run_all_tests.sh` needs no change: its `4dTrajectory` entry now covers the ts_transformer suite too.

**The name collision.** CLAUDE.md said "Python env: conda `aviation`", carried over from another machine — `run_all_tests.sh`'s own comment records "Env names differ per machine (`aeroviz` here, `aviation` elsewhere)". On THIS machine `aviation` is the editable env of `/home/supercomputing/studys/AivationTransformer`, an unrelated project (pure-pip, py3.11, 254 packages, 8.6 GB). Acting on the stale line nearly deleted the real thesis env; the ts_transformer package was also initially targeted at `aviation` for the same reason. Both CLAUDE.md lines are corrected and the collision is now documented under Environment.

**Consolidating the other way is BLOCKED — measured, not assumed.** `cifparse` >= 2.0.4 (aeroviz has 2.0.9) uses PEP 701 f-strings (nested same-type quotes), which is Python 3.12+ syntax. Every version from 2.0.4 up fails `compileall` on py3.11; only 2.0.0 and earlier import there — a 9-patch regression in the ARINC 424 parser that feeds `approach_constraints`. Upstream's PyPI metadata claims `requires_python >=3.10` and is simply wrong. So a py3.11 env cannot host the thesis at current package versions. `casadi`/`openap`/`arinc424` install fine on 3.11; `cifparse` alone is the blocker.

**One real interaction between torch and the existing stack, found and fixed.** `import torch` then `import traffic` raised `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'CXXABI_1.3.15' not found`; the reverse order worked. Cause: pip's manylinux torch wheel has no RPATH into the env, so it resolves `libstdc++.so.6` from the system (CXXABI ≤ 1.3.13), and once that SONAME is loaded conda-forge matplotlib's `_c_internal_utils.so` — which needs 1.3.15 — is answered by the already-loaded old one. `run_all_tests.sh` runs every suite in ONE pytest process with `4dTrajectory` (torch) listed before `trajectory_data_process` (traffic), i.e. precisely the failing order. Fixed with `$CONDA_PREFIX/etc/conda/activate.d/zz-libstdcxx.sh` prepending `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` (plus a matching `deactivate.d`). Note it only takes effect under `conda activate` — calling `envs/aeroviz/bin/python` directly still reproduces the failure. Full suite after the fix: **495 passed, 1 failed** (the pre-existing `test_fixed_time_objective_weights_control_effort_at_one` numpy scalar-conversion `TypeError`); the count rose from 464 because the 32 ts_transformer tests now run in the same invocation.

Anything installed into `aviation` during the attempt (casadi, openap, arinc424, the broken cifparse, plus a tqdm/wcwidth bump and an editable geokit) was rolled back; that env is verified back at torch 2.9.1+cu128 / tqdm 4.67.1 / wcwidth 0.3.0 with no thesis packages. Env spec backups for `aeroviz` live in `.env-backup/` (pip freeze, conda explicit, environment.yml).

**ts_transformer defaults re-sized against the real data** (see the entry below for the original synthetic sizing). The 3747 harvested arrivals have durations p5 235 s / p50 328 s / p90 607 s / p99 920 s — much longer than the "25 km at 120 m/s ⇒ 3.5–5 min" straight-line estimate, because real arrivals are vectored (downwind legs, base turns, holds), so the flown path far exceeds the straight-line distance to the entry ring. `seq_len` 30 → **60** (120 s), `DEFAULT_PRED_LEN_FULL` 150 → **300** (600 s): full mode now covers the complete remaining approach for **97.8%** of flights, where 150 covered **57.6%**.

### 2026-07-19 — `4dTrajectory/ts_transformer`: iTransformer + PatchTST integrated as a learned-prediction sibling to the optimizer

New package answering *what trajectory WILL this aircraft fly* (learned, no dynamics model) alongside `optimization`'s *what SHOULD it fly*. Both emit the same evaluation records, so `python -m evaluation --input <dir>` grades either against the identical regulatory gates.

**Vendoring.** `vendor/itransformer/` (MIT, thuml @ `c2426e6`) and `vendor/patchtst/` (Apache-2.0, yuqinie98 @ `204c21e`) copied byte-identical with only import paths rewritten, each with its `LICENSE` + a `PROVENANCE.md` recording what was copied/dropped and why. Vendored rather than installed because neither upstream is a packaged library and both resolve internal imports through a top-level `layers/` — installed side by side they collide (both ship a different `layers/Embed.py` and `layers/SelfAttention_Family.py`). Dropped from iTransformer: the Reformer/Flowformer/Flashformer/Informer attention variants, and with them the `reformer_pytorch` + `einops` deps. One shared `TSConfig` drives both (upstream's own argparse-namespace contract); `models.py` adapts iTransformer's 4-arg call and PatchTST's 1-arg call to one `model(x)`.

**Env.** Installed editable `geokit` into conda `aviation` (torch 2.9.1+cu128); the package is casadi-free by design, so it lives in the torch env while the optimizer stays in `aeroviz`. `requirements.txt` written. Verified both models forward CPU+CUDA on the RTX 4060.

**Three findings that changed the design, each measured not assumed:**

1. **Sizing.** An arrival truncated at the 25 km ring is only **~3.5–5 min** (~110–150 samples at 2 s), not the 8–12 min first assumed. The initial defaults (L=60 @ 4 s = 4 min lookback) exceeded a whole approach and skipped **5 of every 6** flights as "shorter than one window". Now `dt=2 s`, `L=30`, `H=30` (window) / `150` (full) — all 120 synthetic flights build, ~65 anchors each.
2. **Instance normalisation must be OFF.** iTransformer's `use_norm` / PatchTST's `revin` are ON upstream and strip each window's absolute level as nuisance; in a threshold-anchored ENU frame absolute position *is* the signal. Off wins in **all four** model×mode cells by 2.4–6.5× on ADE (iTransformer window 771→286 m, full 1972→303 m; PatchTST window 910→672 m, full 2268→701 m) — and on converges *sooner* (49–56 vs 74–90 epochs), i.e. to a worse optimum, not undertrained. Defaults flipped; `--instance-norm` re-enables for re-ablation on real data.
3. **Two alignment bugs caught by running the real thing** (both would have produced plausible-looking wrong numbers): (a) `FlightSeries.flight_id` was the callsign, which repeats daily and across runway files — `predict --split test` returned 48 flights for an 18-flight split, and the train/val/test split leaked. Identity is now `dataset.flight_key` = `id_runway_icao24_landingTime`, the same function that produces the record filename. (b) The reference record covered the whole track while the prediction covered anchor→threshold; `evaluation.reference` resamples both at 101 fractions of *their own* arc length, so it reported 4349 m of "path deviation" that was pure span mismatch — **833 m** once span-matched. Records now anchor `t=0` at the anchor sample with `initial_state` the observed state there, and `states[0] == initial_state` as in an optimizer record.

**Contracts.** Channels `(e, n, u, ve, vn, vu)` in a threshold-anchored ENU frame; `psi`/`gamma` never regressed directly (±π wrap averages 179° and −179° to 0°, pointing the aircraft backwards exactly at the turn onto final) but derived as `atan2(vn, ve)`, which *is* the math-ENU convention, so the compass/ENU substitution has no place left to happen. `m` carried, never predicted (unobservable from ADS-B). Records are reference-shaped (`controls == []`) and built via `optimization/evaluation_export.py` — importable here because it is casadi-free — rather than hand-rolling a second copy of the record JSON. `final_time_s` always from `states[-1]["t"]`, never `pred_len × dt` (threshold truncation makes them differ; `evaluation.records` rejects >1e-6).

**Status.** 30 tests pass; verified end-to-end through the real CLI (train on GPU → predict → `python -m evaluation`). **Never trained on real data** — none existed in the tree (`trajectory_data_process/outputs/` absent, no `credentials.json`), so `synthetic.py` generates straight-in approaches for smoke-testing and fixtures. Every number above is synthetic and is plumbing evidence, not a result. Pre-existing unrelated failure noted: `collocation/tests/test_optimizer.py::test_fixed_time_objective_weights_control_effort_at_one` (numpy scalar-conversion `TypeError`).

### 2026-07-07 — "Profile" → "Approach view" rename (systematic)

The 2D approach page/toggle is now **Approach view** throughout (UI text, identifiers, CSS, files; `git mv` preserved history): `useApproachView.ts`, `ApproachViewPanel.tsx`, `ApproachViewToggle.tsx`, `approachViewAnalysis.ts`, `approachViewSources.ts`; identifiers like `isApproachViewOpen`/`setApproachViewOpen`, `approachViewMode`, `ApproachViewTrack/Sample/Input`, `buildApproachViewTracks`, `planApproachViewSources`; CSS `.approach-view-*`; toggle labels "View"/"Hide view". Deliberately NOT renamed — the geometric "profile" (altitude cross-section) keeps the term: `runwayProfileGeometry`/`RunwayProfilePoint`, `procedureProfileProjection`, `procedureVerticalProfileOverlay`, `ProfilePlot`/`ProfileOverlay`, the "Vertical profile" side-view mode label. Not re-checked in-browser.

### 2026-07-07 — Approach view: whole-track plotting, review fixes, perf cache (three passes, same day)

- **Whole track**: `sampleEntityTrack` walks outward in both directions from the current time (`TRACK_SAMPLE_STEP_SECONDS = 5`, `MAX_TRACK_SAMPLES_PER_DIRECTION = 600` backstop) returning the whole time-ordered track; all points kept tagged by containment tier; an aircraft is plotted iff some sample reaches PRIMARY (`trackEngagesProcedure` — full approach for flights that fly the procedure, unrelated traffic excluded). `splitTrackByContainment` splits ordered samples into contiguous runs sharing boundary points (gapless lines); PRIMARY/SECONDARY drawn solid (primary brighter), OUTSIDE dashed `6 5` + dimmed.
- **Review fixes**: walks also stop on `samePosition` (the HOLD-tail gotcha — otherwise never terminates and pads 600 duplicate threshold points per entity per tick); landed/parked entities dropped via liveness check (sample one step back == current → past the real track end → null); not-yet-airborne (null current) dropped; plot domain grows to include plotted track samples (corner-cut stretches no longer clipped) and the selected-flight label gets the plot clipPath; `ApproachViewInput.current` tightened to non-null; engages-gate = `trail.some(PRIMARY)`.
- **Perf**: per-entity cache of the classified whole track (`trackCacheRef` keyed by flight id), rebuilt only when geometry deps change (loaded procedure/frame, active routes, source set) — never on clock ticks; each ~120 ms tick then only re-checks liveness (1–2 `getValue`) + classifies the current marker (O(aircraft × track length) → O(aircraft) per tick). `sampleEntityTrack` split into `currentIfLive` + `sampleWholeTrack`; analysis module exposes `classifyProfileSample` / `classifyTrackSamples` / `trackEngagesProcedure` / `sortProfileTracksBySelection` (`buildApproachViewTracks` kept as their batch composition for tests).
- DEFERRED: interior-gap `break` in the walk is latent (current CZMLs are single-interval); the pre-existing `useCzmlLoader` clock write is still ungated for the Observe+comparison two-writer case. Not re-checked in-browser.

### 2026-07-07 — Approach view mirrors the active tab

- New pure `src/data/approachViewSources.ts` `planApproachViewSources(...) → {observed, optimized}` — the single source for "which trajectory sources are the current tab's globe content": `observed` reuses `planObservedTracks(...).visible`; `optimized` = `mode === "optimize" && hasOptimizedSource`. Used by BOTH the hook (what it samples) and the panel's "CZML linked" badge (`profile.sourceLinked`). Observe plots observed; Optimize plots only the optimized playback; Fly/Compare plot neither.
- **Root-cause follow-up**: `planObservedTracks` no longer keeps the observed CZML loaded behind an open view outside Observe (`relevant = mode === "observe"`). `useCzmlLoader` drives the shared `viewer.clock` from the observed CZML's hours-long span, so loading it in Optimize hijacked the clock and made the optimized playback aircraft vanish; releasing it also saves a 100+ MB load per non-Observe tab. Runtime clock behavior not re-checked in-browser.
- KNOWN GAP: the Observe 3-colour comparison overlay is a separate datasource not yet fed to the view — Observe-with-comparison plots neither source; wiring it in is a follow-up.

### 2026-07-07 — Pipeline exposes the control mesh; mesh defaults single-sourced

- `run_scenario_pipeline.py --n-segments` (unconstrained) / `--n-seg-per-phase` (constrained), either/or by mode, defaulting to CollocationOptimizer's own 8/3; validated ≥2/≥1 at both CLIs.
- The constrained batch path never passed `n_seg_per_phase` before (stuck at the default with no override) — now threaded main → `optimize_scenarios_constrained_iaf` → `_optimize_one_scenario_iaf` → both IAF selectors → `_solve_iaf` → `CollocationOptimizer`.
- Defaults single-sourced in `collocation/optimizer.py` (`DEFAULT_N_SEGMENTS`/`DEFAULT_N_SEG_PER_PHASE`); backend + batch import them; frontend "Control segs/leg" default aligned 2→3 (defaults had silently diverged: frontend 2, backend fallback 4, optimizer/batch 3). The frontend→backend HTTP path had always wired `nSegPerPhase` correctly — the gap was batch-only. Verified: nsp 2→4 changes the constrained plan 85→125 nodes.

### 2026-07-07 — Optimize tab: solve-time readout; HSL hook benchmarked, kept dormant

- `optimization_backend._optimize` puts `result["timings"] = {buildS, solveS, playbackS, totalS}` in the response (was log-only); frontend `OptimizationTimings` + `parseTimings` (absent → null) + a "Solve time" row. Plan readouts restyled to a single column of full-width rows (3-col grid truncated values).
- HSL: `collocation/components.py` reads `AEROVIZ_IPOPT_LINSOL`/`AEROVIZ_IPOPT_HSLLIB` (casadi's IPOPT has the `hsllib` loader — no rebuild; setup in `docs/hsl-linear-solver-setup.md`). Measured: free Coin-HSL Archive has only MA27, 3–27× slower than MUMPS here (small NLPs; OpenBLAS/OpenMP clash) → dormant, kept for a future MA57 academic license.

### 2026-07-06 — Review pass: toggle false-open, ownership seams, manifest-only eval reads, required guards

- **[BUG]** `runwayMatchesSelection(null, X)` is match-ALL, so with the top bar on "All runways" every toggle showed "Hide profile" and clicks closed the invisible page; `open` now also requires `selectedRunway !== null`.
- `RunwayProfileToggle` prop `borrowSelection` (PilotPanel, unconstrained only): opening saves the pre-open `{selectedRunway, isRunwayProfileOpen}` (only when opening actually changes the selection) and restores on close/unmount. Constrained Optimize doesn't borrow — the forced-display hook owns the runway there.
- `useForcedProcedureDisplay`: `SavedDisplay` now carries the profile-open state, restored together with the runway (ownsRunway only); `forceRunway: null` never touches either. `ready` flip only fires when a drive is pending; savedRef bakes `ownsRunway` (what makes a non-null→null `forceRunway` flip restore correctly).
- **Evaluation read side manifest-ONLY** (user decision, no glob fallback): `evaluation.records.load_records` reads a batch dir via its `summary.json` roster; manifest-less dir / listed-but-missing / empty roster raise; `--pattern` removed from both CLIs. (Globbing counted orphans — the KRDU 1023-vs-996 class.)
- **`min_altitude_m` REQUIRED** on `rollout_controls`/`simulate_controls` — the 0.0 default silently validated diverged replays km below elevated fields; target-less replays now pass an explicit 0.0.
- **Constrained-ness is an explicit manifest field**: `_upsert_category` stamps `"constrained": bool`; builder `--constrained`; frontend `ComparisonCategory` requires it, `_cons`-suffix detection deleted; all 5 airports' `categories.json` migrated in place.
- Cleanups: record-filename suffix constants; shared `_fake_optimizer` test factory.

### 2026-07-06 — Rollout guard margin (residual low-success root cause)

97% of KRDU/asdb gate-failures were rollouts truncated by the zero-margin ground guard at exactly the floor: min-time plans deliberately ride the floor, and cm-scale integration noise (measured: 3.9 cm dip for 1.5 s on a replay that lands 0.7 m out) tripped a guard meant for km-scale divergence. Fix: `ROLLOUT_GUARD_MARGIN_M = 5.0`, guard = floor − 5, both rollout call sites. (Diagnostic lesson: the eval record's `final_time_s` IS the truncated end — compare against the states-file's plan T to detect truncation.) Same day:

- `_clear_stale_records` deletes stale top-level records at batch start (27 orphan evals inflated a report).
- `--fitting-type {hs,trapezoidal,rk4}` end-to-end (`FITTING_SCHEMES`; default hs). rk4 verdict: fine on smooth constrained solves; basin-fragile on aggressive unconstrained min-time (auto M 9.1 km off; only M=64 recovers HS's optimum at ~4× cost) — HS stays default.
- `--state-substeps M` end-to-end through both solve paths + frontend "State substeps" input → backend `state_substeps` on both branches (cache keys include it). Measurements in Key Defaults.
- `ipopt.max_iter` wired ("backend never finishes" postmortem): constrained M=64 ≈ 640 subintervals × per-node inequality rows, no iter cap, solves serialized behind the worker lock. `DEFAULT_MAX_ITERATIONS = 3000`, `CollocationOptimizer(max_iterations=…)` on all three solver constructions; backend `maxIterations` reaches both branches (make_optimizer had accepted-and-ignored it).
- Stale-artifact audit closed the `write_reference_records` and CZML-builder (`clear_stale_outputs`) accumulation gaps; single-file overwrites audited clean.
- NOTE: `runway_cons` off-target populations likely contain the wrongly-truncated family — re-examine after re-run. All categories need re-running: `python run_scenario_pipeline.py --jobs 6`.

### 2026-07-06 — Observe constrained auto-open + shared Profile toggle + forced-display hook

- Extracted `src/hooks/useForcedProcedureDisplay.ts` from PilotPanel. Contract `{active, forceRunway}`: non-null `forceRunway` → the hook OWNS `selectedRunway` (save-once/force/restore-on-inactive + dependency-free restore-on-unmount); `forceRunway: null` (Observe) → never reads/writes `selectedRunway`, drives only the panel + `procedures` layer.
- Observe trigger in ControlPanel: active = trajectories layer ON ∧ comparison overlay ON ∧ constrained category ∧ runway selected.
- Shared toggle component (one parameterised toggle driving the global open state; opening focuses `selectedRunway` on the governed runway) used in ControlPanel, PilotPanel's Target-State header, and ProcedurePanel.
- **Dock-handoff race**: two forcing docks switching share one React passive flush — the incoming hook read the outgoing dock's still-forced display and saved a polluted baseline. Fixed with a one-render `ready` gate (no-op on first commit; the sibling's restore is batched with the `ready` flip in React 18). Integration regression test with the real AppProvider. Verified in-browser (KRDU 05L).

### 2026-07-06 — Evaluation window symlog label pile-up

Under `symlog(v) = sign·log10(1+|v|)` every |v|<1 collapses to ~0, stacking the ±0.01/±0.1 decade labels; `EvaluationReportWindow.tsx` now skips sub-1 decades (keeps 0, ±1, ±10, …).

### 2026-07-06 — All-modes sweep + unconstrained batch's trapezoidal fitting (low-success root cause)

- Omitting `--target-type` runs all three modes (`asdb`/`runway`/`runway_cons`) per airport; `--with-constraint` without it is rejected.
- Unconstrained success 4–14% vs constrained 76–97%: the batch's `optimize_scenario` still used trapezoidal after the 07-05 HS flip (batch-edition seam class). Min-time solves ride the floor exactly where 2nd-order trapezoidal is dynamically unfaithful: plan on-target, rollout 5–15 km off (full-T, not truncated). A/B (DAL1407): trap 5950 m vs HS 3.4 m, and HS found a better optimum at ~3× solve time. Fixed to HS.

### 2026-07-06 — Arrival-segment truncation: 25 km entry ring; locals excluded

Landing tracks were validated by their END only, so depart-and-return flights started ON the field. New `trajectory_data_process/arrival_segment.py`: walk backward from touchdown; the arrival starts after the LAST run of ≥3 consecutive samples outside the 25 km ring (hysteresis: one jittery fix can't cut); plain arrivals also cropped to the ring so every arrival shares one entry boundary (the on-ring state distribution = interaction-study boundary condition). Never-outside tracks: `local` if start ≤5 km (takeoff→circuit, written to `<ICAO>_local_rejected.json`, never silently dropped), else coverage-limited arrival kept whole. `truncate_flights` rebases times to 0 and annotates `arrival_truncated`/`cut_samples`/`arrival_duration_s`/`entry_time_utc`. Integrated in `build_arrivals.py` (renamed from `landings_to_czml.py`): raw `*_landings.json` untouched; derived `*_arrivals.json` feed CZMLs + czml-input; airport centre from `config/runway_thresholds.json`. `FlightScenario.source` carries `entry_time_utc` (co-temporal placement key). All 5 airports regenerated: KMSY 400, KRDU 996, KSJC 319, KSMF 714, KSTL 1054 arrivals. All pre-existing batch outputs predate this (and the floor fixes) — stale.

### 2026-07-05 — Evaluation detail window: legible profiles + one colour language

`DeviationProfile` ranked-dot charts replace the illegible bar walls (lateral: log axis; vertical: signed symlog); each chart draws AND labels its own gate in-plot, dots coloured by that gate, legend carries the outside count. One colour language: red/green = per-flight gate verdict only (summary cards made neutral; scatter legend explains its colours). Values are the backend report's rows verbatim. Verified in-browser (KRDU runway_cons, 1001 trajectories).

### 2026-07-05 — Off-target marking moved onto the RESULT path

"Successful flights ending mid-air" = off-target flights with guard-truncated rollouts (correctly classified; they merely LOOKED successful because only the reference was marked). Now the simulator/result path bakes `OFF_TARGET_COLOR` yellow + "(off target)" name; reference drops to dark-amber `OFF_TARGET_REF_COLOR`; plan keeps legend orange. `useComparisonTrajectoryLayer` skips its repaint for `properties.status == "offTarget"`. NOTE: KRDU runway 32's 77/200 off-target (vs 6–7 on 23L/R) is a real quality signal — likely the replay-divergence family.

### 2026-07-05 — Below-ground trajectories: altitude floors + rollout ground guard

Three root causes for optimized trajectories below field elevation (per-leg step-down floors verified working — the dives lived where no floor existed):
1. The global altitude bound (`components.altitude_floor_m` = target − 300 m, documented as a never-binding box) BINDS — min-time solves dive to it. Fixed: margin 300 → 5 m, a real operational floor; `min(initial, target)` deliberately NOT used (a start below the floor is bad data and fails loudly).
2. The start→first-fix transition phase had no floor above the global one (FFT2071 dove to −173 m). Fixed: transition altitude bound = min(start alt, `_first_leg_entry_floor_m`) − margin (min() IS needed here — a start below the first fix's altitude is legitimate climb-to-join geometry).
3. The batch rollout had no ground envelope (`CasadiSimulator.step` has no checks): `_GroundCheckedSimulator` + `rollout_controls(min_altitude_m=…)`.
Also: evaluation gates judge the FINAL state only, so mid-flight dives never failed a gate — pre-fix success rates were inflated. All batch outputs stale.

### 2026-07-05 — In-app evaluation report window ("Details")

The comparison builder PUBLISHES the evaluation report verbatim (`publish_evaluation_report` → `comparison/<category>/evaluation_report.json`); the frontend fetches that copy and only formats/sorts/plots — no metric recomputed client-side (new metrics go in `evaluation/metrics.py`; contract documented in `src/data/evaluationReport.ts`). `EvaluationReportWindow.tsx`: draggable floating window (portal, same shell as Dynamics-Comparison) with summary cards, 8260.58D gates note (values from `report.thresholds`), aggregates table, three SVG charts, full verdict table. `OptimizationSummary` Details button; report fetched lazily, cached per (airport, category); missing report shows a helpful message. Verified in-browser (KSMF runway_cons).

### 2026-07-05 — Comparison CZML: off-target status + evaluation metrics into the frontend

Builder `--evaluation-report`: verdicts keyed by eval filename, joined to summary rows via `eval_file`; solved-but-failed-gates → status `offTarget` with `lateralErrM`/`verticalErrM` on the index record; `optimization_stats(summary, report)` builds the index `optimization` block (successful/successRate/avgStateErrorM/avgTimeS — nothing recomputed). No report → byte-compatible plain behavior. Runner: the (cheap) evaluation report now always runs before the tails; the CZML step passes it; reusing a pre-evaluation optimization skips evaluation with a loud note. Frontend: `ComparisonGroup.status = "solved"|"offTarget"|"failed"`; flight list flags off-target yellow (`.flight-table-offtarget`, #ffcd28); `OptimizationSummary` metrics were already wired.

### 2026-07-05 — Panel "final horiz err" ≠ playbackDrift, round 2 (verified in-browser)

Two stacked causes behind Δ25 m vs playbackDrift 0.6 m: (1) the LOOP_STOP wrap heuristic never fired (Cesium preserves overshoot on wrap) — replaced all elapsed-based heuristics with `clock.onStop`; `makeReadoutEmitter` returns `{tick, stop}` (stop = throttle-bypassed exact-`stopS` emit, deduped), both playback hooks subscribe to onTick + onStop; Reset/backward scrubs never raise onStop. (2) `czml_common.document_packet` truncated the clock interval end to whole seconds — now `iso_ms` (74.7 m/s × 0.338 s ≈ the phantom 25 m). Backend audited clean otherwise (`playbackDriftM` and the CZML doc end read the same terminal sample).

### 2026-07-05 — Constrained-IAF batch 0% solve rate (two stacked bugs)

1. `scenario_optimization._solve_iaf` still unpacked `segments, _spans = build_constraint_segments(...)` after the 07-03 change to a plain list return — every constrained batch since 07-03 was broken (backend HTTP path updated, batch caller missed). Seam regression test added.
2. The batch's threshold target (from `config/runway_thresholds.json`) sat up to 390 m from the procedure's CIFP threshold → frame-anchor guard fired. `_snap_target_to_procedure` snaps the solve target horizontally onto the procedure's last waypoint (keeps altitude/Vref/pavement heading/glidepath); `_iaf_result` writes the eval record against the SNAPPED target; reference records keep the scenario target. (Diagnostic note: the batch log truncates errors to exception TYPE names per IAF — reproduce one scenario for a real traceback.)

### 2026-07-05 — Evaluation review fixes (degenerate records + HTML escaping)

- One 1-sample "solved" record (rollout truncated at its first step) aborted the whole batch report. Fixed at both ends: producer `_require_usable_rollout` (<2 samples → recorded as a FAILED scenario); evaluation guards via shared `reference.horizontal_arc_length_m` — comparison skipped with a row note, visualize drops undrawable polylines.
- HTML: embedded JSON escapes `</` as `<\/`; `esc()` on every data-derived string reaching innerHTML/Plotly.
- Contract: `final_time_s == states[-1].t` required on solved records; `resample_by_arc_length` rejects n<2; chart labels use unique record file basenames.

### 2026-07-05 — One pipeline runner: `run_scenario_pipeline.py`

The comparison + evaluation runners MERGED (both old scripts deleted). They duplicated the expensive steps and wrote divergent opt_dir contents (the comparison runner silently overwrote `reference_file` pointers away). Now optimization always runs with `--reference-tracks`; tails selectable via `--outputs czml,eval` (default both); `--skip-optimize` reuses an existing summary.json. Category keys `asdb`/`runway`/`runway_cons` unchanged.

### 2026-07-05 — `evaluation` package (regulation-derived gates)

New root package `evaluation/` — the file-based seam at the end of the pipeline (geokit + stdlib only, never imports the optimizer). Record contract + gates: see Key Defaults. Salient points:
- `evaluation_export.py` maps the true-dynamics rollout onto the contract — `rollout_piecewise_constant`'s samples already carry the active control per sample (nothing re-derived). Both batch modes roll out once and write `*_eval.json` next to every `*_states.json`, INCLUDING failed scenarios (empty record = how solve rate is computed from files alone); summary rows carry `eval_file`. Old `*_states.json` can't be converted offline (no controls) — re-run for eval records.
- Reference records: observed track in the same contract (`flight_scenarios.state_samples_from_track`, times rebased, same target; `controls == []`). `write_reference_records` looks flights up by full identity `(id, icao24, landing_time_utc)` (missing ⇒ raise) → `references/<identity>_reference_eval.json`; `reference_file` stamped on every eval record, failed included. `compare_to_reference` = flight-time delta + path-shape deviation (arc-length resampling — time-matching would conflate speed profiles with geometry).
- CLIs: `python -m evaluation --input <dir> --output report.json` (+ threshold flags); `python -m evaluation.visualize` renders one self-contained HTML (Plotly CDN + embedded DATA; English body, abbreviations expanded on use; 8260.58D citations in the gate-sources note; `--max-tracks` cap stated on the page). Verified in-browser.

### 2026-07-05 — ψ corridor kills the looping/crawling pathology; drift guard; HS default

- Postmortem (nsp=2 loops): the solve CONVERGED onto a local optimum with ±2π winding (feasible — the join ψ box pins one node; excursions cancel), and the node-pinned terminal masked a 4.5–4.9 km true-dynamics rollout drift (trapezoidal on a winding path is node-feasible but dynamically meaningless).
- **ψ corridor** (structural fix): constrained heading variable bounds tightened from ±3π to the route heading hull ± 90° — winding optima cease to exist. The whole 07-04 family of crawls/Max_Iterations/basin-twitchiness died: 2×2 (HEAVE/custom × trap/HS) 4/4 clean, nsp 2/3/4/6 all converge.
- **Drift guard**: `playback_terminal_drift_m` → `playbackDriftM` on every response + stderr WARNING > 50 m. Never touches the NLP.
- Fitting verdict (doglegged H05LZ): trapezoidal 226–296 m drift vs HS 0.6–0.9 m at every nsp; HS ≈ 2–2.5× solve time. **Constrained default flipped to HS** (`_DEFAULT_CONSTRAINED_SCHEME = hermiteSimpsonNormalizedFullTransport`); trapezoidal stays selectable.
- Readout artifact (wall-throttled ~12 Hz emitter missing the terminal sample by up to throttle×multiplier sim-seconds) fixed via `makeReadoutEmitter`, shared by the optimized-playback and Compare hooks — later superseded by the onStop rework above.

### 2026-07-04 — Join constraints: pre-FAF fix passage + flexible FAC intercept + alignment tiers

(Current semantics in Key Defaults.) Highlights and lessons:
- ⑪ Fix passage evolved to the PRE-FAF fix only (all-fixes and entry-fix variants superseded, per user); enforced as a smooth squared form rescaled by `1/(2·tol)` so violations read in metres (no |·| kink); tolerance = the leg's k·RNP halfwidth, procedure-sourced.
- ⑫ Flexible FAC intercept replaced the exact-FAF pin: linear cross-track equality + upstream-only window ≥ L_final/5 before the FAF; `fac_distance_to_ltp` is THE one distance measure (glidepath d reuses it). Vertical semantics stay published-geography-keyed (`LpvFinalSpec.d_faf_m`/`prefaf_floor_m`; both None → gate off, byte-identical).
- Two 2π-branch bugs found by evaluating `nlp.g` at failed iterates and ranking violated rows: the terminal ψ pinned the wrong branch on double-90° routes (fix: `_route_unwrapped_target_psi` walks chained leg courses), and the join intercept `cos(ψ−course) ≥ cos30°` was 2π-periodic (fix: branch-aware linear box). Plus the duration-split regularizer for the flat time-split direction.
- Constraint families refactored into explicit per-family row functions behind a dispatcher (uniform `list[(expr, lb, ub)]`); `components.unwrap_angle()` collapses four unwrap copies. Two-tier FAC heading alignment added (positions-only corridors let large heading errors fit between nodes).
- `log_optimizer_config` writes one stderr line before any solving (optimizer/scheme/fitting/dynamics/transport/constrained).
- Frontend follow-up: custom starts KEEP the RNAV IF selection (the selector names the procedure; the start is independent) so constrained solves can start off-fix; transition threshold unified with the passage tolerance (`_first_fix_join_tolerance_m`) — the old 1–2 km dead zone (no transition, no disc) removed. Verified end-to-end on H05LZ HEAVE with a custom start 9.2 km out.
- KNOWN DATA GAP: per-leg RNP is not extracted from CIFP — RNP-AR procedures (H05LZ) get the default RNP 1.0 disc (926 m at k=0.5) instead of ~278 m (RNP 0.3).

### 2026-07-03 — CIFP thresholds everywhere; displaced-threshold root cause

- Root cause of up-to-970 m target gaps (KSJC 30R): runway.geojson `runway_surface` edges are PAVEMENT ends while CIFP's threshold is the DISPLACED landing threshold, plus a real `build_runway_ring` bug (declared length re-centred on the endpoint midpoint with asymmetric displaced offsets → rigid shift by `(he_disp − le_disp)/2`). OurAirports coordinates themselves verified accurate.
- Constrained target anchored on CIFP: `procedureThresholdAnchor(constraint, document)` — position = the constraint's last waypoint, altitude = CIFP threshold elevation, ψ = final course in the simulator convention; `PilotPanel.computeTrajectory` overrides the request target whenever a `procedureConstraint` is attached. runway.geojson still drives the unconstrained target and rendering.
- `procedure-details/index.json` hoists each runway's CIFP `threshold` `{lon, lat, elevationFt}` (null when uncoded); unconstrained targets prefer it too (`buildRunwayThresholdTargets(collection, index?)`; heading stays pavement-derived). CIFP is also more complete than OA (e.g. KSTL 30L's displacement missing from OA).
- `build_runway_ring` rewritten: `runway_surface` corners ON the OA endpoints; `landing_zone` ends moved inward by the displaced distance. All 5 airports regenerated; landing_zone matches CIFP ≤ 24 m except pure OA data gaps.

### 2026-07-03 — approach_constraints/collocation review fixes

- **One ψ convention**: `course_bearing` now returns the model convention (`atan2(Δn, Δe)`); the optimizer consumes it instead of re-deriving course math (guiding rule: constraint/course math has ONE source — `approach_constraints`).
- `ConstraintReport` unit-aware: metre and radian violations separated (`max_violation()` / `max_angular_violation()`, `is_feasible(tol_m, tol_rad)`) — 1 rad had counted as "1 m", a real false-feasible.
- Transition phase actually built: `_phase_plan` prepends the unconstrained start→first-fix phase (start > 2 km from the first leg's start fix); an approach whose first leg IS the final gets the FAF intercept on the transition phase. `build_constraint_segments` returns a plain list.
- Frame-anchor contract validated loudly; terminal-bank 1-node-phase latent bug fixed (`phase_starts`); per-phase auto state substeps (~3 s target; M-selection deliberately ignores `fixed_duration` so fixed- and free-time NLPs share one decision layout, the fixed solve seeding the free one); glidepath d measured on the same GARP→LTP axis as the lateral corridor; normalized position box 1e7→2e6 m; dead machinery removed (`partition_node_indices`, `LpvFinalSpec.da_hat_m`); README de-staled; mass-frozen approximation stated; HS docstring corrected (4th-order, O(h⁵) local).

### (undated, ~2026-06-28) — Backend SIGABRT root cause + solver isolation

The service "shut itself down": casadi's thread-unsafe symbolic construction under `ThreadingHTTPServer` (three concurrent NLP builds) corrupted the heap → SIGABRT; the old launcher then co-killed the frontend.
- `isolated_backend.py` `IsolatedRunner`: casadi-heavy endpoints (`/optimization/run`, `/dynamics-comparison/run`) run in a worker subprocess (`ProcessPoolExecutor(max_workers=1, mp_context="spawn")`); a native abort → `BrokenProcessPool` → clean `SolverCrashError` 500, next request spawns a fresh worker (self-heal). Decorator backends keep the same method interface (tests inject in-process fakes).
- Memory-aware worker lifecycle: `POST /optimization/session/{open,close}` (+ dynamics-comparison twins) ref-count ONE resident warm worker tied to the frontend tab (spawn costs ~1.2 s/call otherwise); no session → ephemeral worker per call; idle watchdog (`AEROVIZ_WORKER_IDLE_TIMEOUT_S`, 600 s) reclaims stranded workers. Frontend `workerSessionClient.ts` opens/closes per Pilot sub-mode + `navigator.sendBeacon` close on `pagehide`; all best-effort, the watchdog is the backstop.
- `casadi_lock.CASADI_LOCK` (RLock) serializes every in-process casadi entry point incl. `SimulationBackend.reset/step` (previously unlocked).
- `start_aeroviz_fullstack.sh` rewritten as the supervisor (see Gotchas). Spawn-picklable probes live in `isolation_probes.py`.

### 2026-06-28 — Scenario→optimization→CZML pipeline (scaffolds, since filled) + `flight_scenarios` seam

- `scenario_optimization.py` writes one `*_states.json` per scenario = `{source, final_time_s, optimizer_states[], simulator_states[]}` (states `{t,lat,lon,alt,V,psi,gamma,m}`); `optimizer_states` = NLP node states, `simulator_states` = controls rolled through `CasadiSimulator` (lives in the `aerodynamic_model` layer — never imports the backend above it). `build_scenario_comparison_czml.py` renders reference/optimizer/simulator as three coloured time-dynamic paths.
- `flight_scenarios` package: `scenario.py` (record + JSON round-trip + `aircraft_for_code`), `start_state.py` (track → initial state via two-sample finite difference), `build.py`, CLI `python -m flight_scenarios`. `final_state_from_track` populates `FlightScenario.target`.

### 2026-06-28 / 2026-06-27 — `geokit`: one geodesy/units source

- `geokit.constants`: WGS84 (`WGS84_A`/`_E2`/`_B`), `SPHERE_RADIUS_M` (default WGS84 a; switchable `EARTH_RADIUS_MEAN_M`), `NM_M`/`FT_M`/`KT_MS`/`METRES_PER_DEG_LAT`/`DEG2RAD`. `geokit.geodesy`: haversine, equirectangular, bearing, flat-distance, metres-per-degree, bounds. `geokit.units`: exact speed (`kt_to_ms` = 1852/3600, ft/min, km/h, mph) + length (`nm_to_m`, `ft_to_m`) conversions — replaced the truncated `0.51444` and 4 divergent Earth radii project-wide; `aircraft_sets.py` SI-mirror fields derive from it.
- Frontend: `geokit/scripts/export_constants_json.py` → `src/generated/geoConstants.json`, re-exported by `procedureGeoMath.ts` (~16 files migrated off local constants); drift-guard test fails if the JSON drifts. Fixed a real bug: two TS modules had used different haversine radii.
- Aero layer imports constants only (symbolic functions untouched). The 30 km study was regenerated for the ~0.1% WGS84-a shift (conclusions unchanged). Pedagogical flat-Earth `runway_bearing_rad` stays local by design.

### 2026-06-27 — Full (exact) geodetic transport as an explicit option

The geodetic RHS's ψ transport had silently dropped a cross term `V·sinγ·sinψ·cosψ·(1/(R_N+h) − 1/(R_M+h))` (~3–4 orders below the main meridian-convergence term; γ transport was already exact). Now `transport ∈ {"none","approx","full"}` on `make_geodetic_dynamics_model`/`make_geodetic_step_integrator`; `"approx"` = historical default (byte-identical). New `*FullTransport` (+ Normalized) schemes end-to-end (backend names `casadiDirectCollocationNormalizedFullTransport(+Trapezoidal/+Rk4)` etc.; frontend Dynamics options). Compare mode gains opt-in system F (full transport). `transport_term_comparison.py` + zh doc: divergence ~mm over 120 s, dt-independent (vector-field, not truncation). Decision: default stays approx; no silent approximations going forward.

### 2026-06-25 / 2026-06-24 — RNAV(GPS) tutorials + canonical ProcedureConstraint + CIFP block-altitude fix

- Tutorials: `aeroviz-4d/docs/34-how-to-read-rnav-gps-approach.{zh,en}.html` — self-contained, interactive (auto-wrapped glossary tooltips, SVG fix/segment hotspots, slide-in glossary panel), worked on KRDU RNAV (GPS) Y RWY 5L. Verified in-browser.
- **Canonical `ProcedureConstraint`** (front↔back): `src/data/procedureConstraint.ts` + Python mirror `aeroviz_backend/procedure_constraint.py` — one JSON shape (ordered waypoints with altitude windows + final course + glidepath + nominal speed); `buildProcedureConstraint(document, {branchId})`. CIFP→`AltitudeConstraint` conversion unified into one `altitudeConstraintFromCifp` (the two copies had diverged — one dropped block upper bounds).
- **CIFP block-altitude fix**: ARINC 424 "B" descriptor is a WINDOW (at-or-below Alt1, at-or-above Alt2); the parser had dropped Alt2. `ProcedureLeg.altitude_ft_2` added; KSTL regenerated. Chart-cross-referenced golden test (`test_krdu_r05ly_matches_published_rnav_gps_chart`) guards the parser against published-chart values. Documented gap: leg speed restrictions not extracted (cifparse exposes no speed field; 0 coded in the dataset) — `speedMaxKt` is ready when a source appears.

### 2026-06-24 / 2026-06-23 — Normalized geodetic scheme (conditioning fix)

- Root cause of Max_Iterations on H05LZ N=10 free-time solves: conditioning, not the seed — radian lat/lon (~1e-6 rad/s derivatives) next to metre/m-s states; the `1/(R_M+h)` factor makes position defect rows ~6–7 orders smaller than altitude rows.
- Fix: `*Normalized` schemes reparameterise the decision state to metres from the target anchor (`n=(lat−lat_t)·R`, `e=(lon−lon_t)·R·cos(lat_t)`) — an EXACT affine change of variables (unlike localEnu's flat-tangent approximation); same geodetic RHS inside the defect. Robust across N and arrival windows; identical trajectories on benign problems. The localEnu cold-start hybrid (an earlier workaround) was removed as superseded. Tutorial: `4dTrajectory/docs/geodetic_state_normalization.zh.md`. Compare mode gains system N (normalized) which overlays C — live proof the reparameterisation changes nothing.

### 2026-06-23 — Dynamics Compare mode (Pilot panel) + follow-ups

Third Pilot-panel mode: flies the start state under one constant control as the study's systems — A fixed-tangent ENU (anchored at the START in Compare mode; the 30 km study anchors at the target), B per-step re-anchored (reference), C geodetic RHS +transport, D no-transport, opt-in N (normalized) and F (full transport) — replayed as coloured, hideable CZML paths on Cesium's clock with deviation charts (horiz/alt/head/speed/fpa vs B) and a final-value table.
- Core extracted to `dynamics_comparison.py` `compare_dynamics(...)` (30 km study now calls it, output byte-identical). Endpoint `POST /dynamics-comparison/run` (`dynamics_comparison_backend.py`); shared `czml_common.py` (epoch/iso/document-packet) + `responseValidators.ts` back both backends/clients.
- Frontend: `dynamicsComparisonClient.ts`, `useDynamicsComparisonPlayback.ts` (loads CZML, drives the clock, hides systems, camera-follow), `DynamicsComparisonCharts.tsx` (draggable portal window — see the backdrop-filter gotcha). Per-system tinted aircraft models (`colorBlendMode: MIX`) oriented via `VelocityOrientationProperty`, wrapped in `makeStableVelocityOrientation` (a CallbackProperty returning the last valid orientation when HOLD extrapolation zeroes the velocity — otherwise the parked model snaps to a default attitude).
- Live State panel shows B's state (dense backend `samples` in the trajectory-play shape) + colored per-system delta chips interpolated from the chart (`interpolateComparisonDeltas`); `fpa` tracked end-to-end as an error metric. Trajectory Play reuses the same chip strip for a live Δ-vs-target readout (replaced the old Lat/Lon/Alt Error rows).
- Custom start state in Compare (RNAV fixes + runway select); run history persisted per run (`dynamics_comparison_history.py`, git-ignored dir) with backend-averaged history endpoints (common distance grid, shortest run's range) + frontend Average/Clear buttons.
- Review fixes: rollout never records sub-surface/non-finite samples (stops + truncation note via `requestedDurationS`); endpoint-inclusive `even_sample_indices`; double-checked-locked integrator cache; chart memoization; START preview hidden during comparison playback.

### 2026-06-23 — Optimizer = dynamics × fitting; shared stall model

- `_DEFECT_SCHEMES` entries are `(make_dynamics, make_defect)`: localEnu is a CONTINUOUS dynamics (fixed ENU tangent frame) collocatable with any fitting (defect converts geodetic nodes into the target-anchored ENU frame via `geodetic_state_to_enu_expr`); only `reanchoredEnu` stays shooting-only (per-step re-anchoring is discrete).
- Shared stall model `aero_params_for_aircraft(aircraft)` (mass-based Cl_max, A320 ≈ 2.7) used by optimizer AND playback — they had diverged (2.7 vs 1.5), replaying optimized trajectories ~1.6 km off.
- Cold-start hybrid (`cold_start_scheme`: fixed-time seed solved with a cheaper dynamics, free-time refines) was added with a whole-flow timing stderr line (`log_optimization_timing`: build/coldStart/freeTime/solve/playback/total) — the hybrid was later removed (superseded by normalization); the timing log remains.

### 2026-06-22 — Pluggable defect schemes; solver backend verdict; CZML playback; dense state

- **Defect schemes**: trapezoidal (order 2) / Hermite-Simpson (order 4, default) / RK4 (order 4, shooting) on the continuous geodetic RHS + `reanchoredEnu` (the playback integrator as a shooting defect). HS and RK4 are the same order — they differ in construction (implicit collocation vs explicit shooting). A stepper can be a shooting defect but NOT a polynomial collocation defect. `collocation_scheme_comparison.py` accuracy ladder: trapezoidal ~5 m vs HS/RK4 sub-metre. Frontend split the optimizer choice into Dynamics × Fitting dropdowns (`optimizerToParts`/`partsToOptimizer`/`validFittingsForDynamics`); legacy optimizer names remain valid on the backend. Tutorial `4dTrajectory/docs/collocation_schemes.zh.html` (interactive convergence demos).
- **`localEnu` scheme + 30 km study** (`dynamics_comparison_30km.py` + zh doc): fixed local-ENU @ target ≈ 335 m horiz error over 30 km, RHS-no-transport ≈ 145 m, full RHS +transport ≈ 0.03 m (validates RHS ≡ re-anchored). `make_local_enu_step_integrator(ref_geo)` reduces exactly to the re-anchored stepper when ref = current point.
- **Solver backend**: `solver_backend` switch (ipopt/sqpmethod) exists but sqpmethod is NOT usable — cold it bails instantly from linear-interp guesses; warm-started (needs duals, exact Hessian) it's still slower than cold IPOPT because CasADi's sqpmethod uses a dense active-set QP (~300× per-iteration cost, no banded OCP structure). IPOPT stays the only exposed backend; a real warm-start payoff would need acados/HPIPM. `solver_backend_benchmark.py` documents this.
- **Playback**: the optimized trajectory plays as backend-built CZML on Cesium's clock (`trajectory_playback.build_optimized_trajectory_playback` — rolls the N piecewise-constant controls once through the SAME geodetic integrator as the live sim, sub-mm match). Trail = one short polyline per sample interval with ms-precision availability (grows behind the aircraft), coloured by control segment (blue→red). The aircraft packet carries NO orientation — the frontend sets it from the sampled state with the live-Pilot convention (`headingPitchRollQuaternion`, heading −ψ, pitch γ+α, roll −μ). `useOptimizedTrajectoryPlayback.ts` drives the clock + throttled ~12 Hz readout sampling (`sampleTrajectoryAt`). Rollout truncates (not raises) on envelope exit — it's a viz aid. Manual Pilot mode unchanged (interactive).
- **Dense-state collocation**: control on N segments, state collocated on N·M sub-intervals (`sub_steps`; auto ~3 s, cap 16) — fixed the km-scale optimizer→playback mismatch (coarse discrete operator ≠ fine playback RK4). The multiple-shooting polish machinery was REMOVED (dense-state raw solutions are playback-consistent). Raising nSegments instead would refine control too → the "wrinkle" convergence pathology.
- **CIFP transition-altitude misparse**: initial fixes with no published crossing altitude were placed at the procedure-wide Transition Altitude (18000 ft) → infeasible starts. Parser no longer falls back to `trans_alt`; qualifier from `alt_desc`; an IF with no own altitude derives one by interpolating from the nearest published fix on the branch (`derivedInitialFixAltitudeFt`) rather than being dropped. Postmortem: `aeroviz-4d/docs/33-cifp-transition-altitude-misparse-postmortem.md`. (The ready-made cifparse/arinc424 packages had the same fallback bug.)

### 2026-06-21 — Geodetic continuous dynamics for direct collocation

Replaced the fixed-ENU transcription with one continuous geodetic RHS shared by optimizer and playback: `make_geodetic_dynamics_model` (point-mass RHS in `(lat, lon, h, V, psi, gamma)` radians; position kinematics via WGS84 `R_M`/`R_N`; transport terms on ψ̇/γ̇) + `make_geodetic_step_integrator` (RK4, degrees externally). Validation (`geodetic_vs_reanchored_error.py`): geodetic+transport tracks the re-anchored RK4 playback to ~0.3 mm over 5 km; without transport ~2.9 m drift. Interactive doc `geodetic_dynamics_transport.zh.html`.

### 2026-04-20 — OCS geometry + FAA DOF obstacle layer

- `ocsGeometry.ts` `buildFinalApproachOCS` implemented (primary trapezoid + two 7:1-slope secondary panels); `useOcsLayer` renders three semi-transparent polygons per route (FAF→threshold pairs from `procedures.geojson`; primary half-width from the route's tunnel descriptor, 150 m fallback); `ocsSurfaces` layer toggle. Altitudes read from LineString z-values (CIFP geometry alt); switching to MCA is a one-function change documented in `docs/03-ocs-geometry.zh.md` §5.6.
- `preprocess_obstacles.py` parses fixed-width DOF `.Dat` (haversine radius filter, default 20 km) → `obstacles.geojson`; `useObstacleLayer` renders type-coloured cylinders (`RELATIVE_TO_GROUND`) with AGL labels; `obstacles` layer toggle.

### 2026-04-19 — DSM terrain hook

`useDsmTerrainLayer` rewritten onto the preprocessed heightmap pipeline (`terrain/dsmHeightmapTerrain.ts`); returns `{status, metadata, provider, error}`; wired in `CesiumViewer` + demo page; `dsmTerrain` layer toggle.
