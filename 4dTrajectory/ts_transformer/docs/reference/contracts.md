# ts_transformer reference — contracts

Full text behind the index lines of `4dTrajectory/ts_transformer/CLAUDE.md`, moved here
VERBATIM on 2026-09-16 when that file had grown to 118 KB (1,202 lines) and was loaded into
every ts session. Only the `### <ID>` headings are new. Every heading has exactly one
index line in CLAUDE.md that ends in its ID; find one with `grep -n '^### C7 ·' docs/reference/*.md`.

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.** Measurements and campaign evidence still
belong in `docs/reference/ENGINEERING_NOTES.md` / the design documents, status in `docs/history/OPEN_ITEMS_2026-09-18.md`.

**2026-09-18:** the main line's 2026-09-14…09-17 additions to that CLAUDE.md (the control
contracts, the two-tier axes and traps) were placed here the same way when this branch was
rebased — same rule, new IDs.

## Contracts (violating these breaks loading, or silently scores the wrong thing)

(C25 and C26 stood under Layout in the old file; they are contracts.)

### C1 · channel names and order

- **Channels = `(e, n, u, edot, ndot, udot)`, and names AND order are load-bearing.** The tuple
  indexes tensors, normalizer stats and checkpoints; `load_checkpoint` refuses a mismatch. The
  velocity channels are the EXACT chart derivatives of the position channels (full-transport
  Jacobian, `geokit.wgs84_curvature_radii`), not raw physical ENU components — see the velocity
  seam in `flight_scenarios/CLAUDE.md`.

### C2 · the chart origin is `target_chart`

- **The chart origin is NOT "the threshold"; `FlightSeries.target_chart` is.** `enu` /
  `runway-aligned` anchor at the assigned threshold (target_chart ≡ 0), `airport-enu` at the
  airport reference point. Every consumer that judges distance-to-go must measure from
  `target_chart`; **a new one that reads `hypot(e, n)` is silently wrong under the airport frame
  and only there.**

### C3 · "it crossed the threshold" = `d ≤ 0` AND on the final

- **"It crossed the threshold" is `d ≤ 0` AND on the final — never the plane alone**
  (`final_approach_geometry.threshold_crossing_index` / `threshold_crossing_mask`, the one rule every consumer of
  an arrival point reads — the forecast cut, the trombone and, since 2026-09-26, the dataset's scan of the OBSERVED
  rows of a flight without a fitted crossing (`dataset._observed_threshold_crossing`: 3 of the 44 unfitted flights
  had been cut at a plane pass off the final; a fitted flight is scanned after its fit, on the final); the
  closest-approach / in-segment refinement is the caller's). A vectored flight's
  DOWNWIND is parallel to the course and opposite it, several km abeam, and passes `d = 0` out
  there, so the plane-only rule fired ABEAM (2026-09-09: 96.5 % of the vectored
  `--truncate-at-threshold` cuts > 1 km from the threshold, median |xt| 8.7 km at a median
  1.75 km above it; the trombone's `tromboneRefNoCrossing` 31.6 % and ~12 km of reference path
  on a 25 km approach). Straight-in records are unaffected — there the two rules answer the same
  row. A trajectory that never satisfies it has NO crossing, and the caller must say so rather
  than cut somewhere — so `tromboneRefNoCrossing` now means "never got ONTO the final" and reads
  HIGHER, not lower, than the 31.6 % above, and under `--truncate-at-threshold` the crossing rule
  OWNS `truncatedAtThreshold`: a record with no crossing is whole with the flag CLEARED, even
  where the fixed-time postprocessor's closest-approach rule had set it (`horizonCapped` marks
  the records that rule never reached, so whether it cut one stays recoverable). Two things a reader must know are true of the answer, not the
  rule: the caller's alignment is a central difference over POSITIONS, so the flying just after a
  row is in its direction; and `MEMBERSHIP_FLOOR_M` (500 m) is what decides whether a record is
  cut at all.

### C27 · the control head's FOUR contracts (`control_thrust_parameterization`)

- **The control head has FOUR contracts — `control_thrust_parameterization` — and each is ONE row**
  (2026-09-14 → 09-16; `docs/history/2026-09_specific_force/2026-09-14_specific_force_control_design.md`; the one-row structure:
  `archive/two_tier_v2_2026_09/docs/2026-09-16_two_tier_transformer_feasibility.zh.md` §11):
  - **`thrust-fraction`** (the default, every stored run, pinned by every named recipe) is the
    box below. **`specific-force`** makes the first column `n_x = (T − D)/W` (box `[−0.20, 0.23]` g,
    neutral −0.05; the lag RHS re-solves `T = clamp(W·a_x + D, −0.2·T_max, T_max)` at EVERY RK4
    stage, so drag cancels and `V̇ = g(a_x − sin γ)`). **`speed-command`** (§12, FAILED, kept
    loadable) makes it Δv relative to the anchor airspeed, flown by an 8 s speed loop.
    **`specific-force+path-angle`** (§14–15) keeps n_x and makes the THIRD column a path-angle
    target γ\*, flown by a 3 s path loop through the load the RHS resolves.
  - **Where a contract lives — never branch on the value anywhere else:**
    `outputs/envelope.py` `ControlContract` row = names, units, box, neutral, the lag `law`
    (`aerodynamic_model.torch_lag_dynamics`: `resolve_load` / `resolve_thrust` per RK4 stage, and
    `geodetic_load` / `geodetic_controls` off a geodetic state for the record and the heading-rate
    loss), the inverse-dynamics `teacher`, `longitudinal` (the speed floor's inversion),
    `saturation_labels`, `record_command_columns`, `identity_suffix` (spelled into the target
    contract; empty for thrust-fraction and specific-force and MUST stay empty, or every stored
    checkpoint of those is refused), `relative_to_anchor_speed`. `config.CONTROL_PARAMETERIZATION_SCOPES`
    row = where it may be used (slug, point-mass rows, fitted teacher, hook modules); the vocabulary is
    derived from it and the envelope registry asserts the keys agree. A new contract is one row on each
    side plus its law class; `tests/test_control_contracts.py` pins the structure.
  - **Rules:** `dynamics_arrays` / `anchor_controls` / `actual_controls` / `commanded_controls`
    take the parameterisation as a REQUIRED argument; the plan path pins thrust-fraction.
    `dynamics_arrays` and `condition_vector` also take the condition feature set
    (`control_condition_features`) as a REQUIRED argument — the sets share one width, so a
    forgotten one is a silent mislabel. A control `Forecast` carries `commands` (the flown schedule
    in contract units) and `control_parameterization` exactly when it carries `controls`; anything
    that cuts or joins controls cuts or joins commands.
  - **Reproducibility across the 2026-09-16 refactor:** on CPU every stored checkpoint of every
    contract predicts, exports and scores BIT-IDENTICALLY (golden over N4_twin / N3 / N6 / N7);
    gradients are bit-identical under thrust-fraction and on the point-mass rows, and differ by
    round-off (float64 relative ≤ 5e-14) under the three drag-resolving laws, which now share one
    drag tensor per stage. **On CUDA the compiled kernels changed**: deterministic per code, but
    ≤ 2e-13 relative from the old code for EVERY contract — so a GPU re-prediction of a stored run
    matches its records to round-off, and a GPU retrain of any stored run at this code is not
    bit-identical (as after a torch upgrade; same-code twins stay the rule). The path-angle record's
    resolved load moved by ≤ 1e-16 relative (cos γ is now √(1−sin²γ) there, as in the RHS).
  - **The same thrust RANGE is not the same dynamics:** with the drag cancelled the speed has
    NO drag feedback, so a biased n_x drifts where a biased δ settles. That is why its neutral
    is a descent speed hold, not level trim.

### C4 · controls are dimensionless; the two load-factor floors

- **Controls are DIMENSIONLESS in this package** (`outputs/envelope.py` is the single source):
  `(thrust_fraction ∈ [-0.2, 1.0], bank_rad ∈ ±π/4, load_factor ∈ [0.2, 2.0])`, same box on every
  airframe. Newtons appear in exactly two places — `physical_controls()` into the dynamics, and
  `inference/forecast.py` out to the evaluation record. **The thrust floor is negative on purpose** (an
  approach needs net-negative force this clean polar does not model). This is NOT the optimizer's
  envelope, which is a flyability claim; this one is a learned head's search space. **The two
  load-factor floors differ on purpose and neither moves (decided 2026-09-09, review C-12)**:
  the head's box floor is 0.2, `flyability`'s hard floor is 0.5, so a control-path segment at
  n ∈ [0.2, 0.5) is unflyable by construction on the published metric — a fact about the old
  path, not a bug to fix in either number. A guidance layer that COMMANDS the load factor (the
  plan-and-guidance design) reads `flyability`'s envelope, never this box.

### C5 · `config.py` single source; `TSConfig` typed views

- **`config.py` is the single source** and everything in it is serialised into every checkpoint.
  `config.input_channels` is what the model sees, `config.channels` what it predicts. `TSConfig`
  is FLAT on purpose (the vendored networks, `run_naming`, the CLI and every stored checkpoint
  read the flat dict); what OWNS each field and validates it are the typed views built in
  `__post_init__` — `config.cohort` / `.backbone` / `.training` / `.output` (`StateOutput`,
  `ClosureOutput` or `ControlOutput` with its `duration` / `dynamics` / `objective` / `hook` /
  `latent` axes). **A field owned by another output's view is refused off its default**
  (`_validate_ownership`: `control_command_hook='barrier'` on a state run is "belongs to the
  control output"), and `from_dict` normalises such fields to their defaults on load. Rules
  that read two views live on `_validate_cross`. Adding a field: put it on exactly one view
  (`_check_view_partition` fails the import otherwise).

### C6 · `dt_s`, `seq_len`, `pred_len` and the measured horizon

- `dt_s = 2.0`, `seq_len = 60` (120 s), `pred_len` = 30 (window, 60 s) / **300** (full, 600 s).
  The horizon was sized from the MEASURED duration distribution (p50 328 s / p95 651 s), covering
  **97.8 %** of flights — the "an arrival is ~3.5–5 min" straight-line estimate was WRONG (real
  arrivals are vectored), **do not resize from it**. The ~2 % over the horizon are cut at H and
  flagged `horizonCapped`; their gate verdicts are cap artifacts, not model error.

### C7 · a new loss term goes in `loss_component_names`

- **A new loss term must be added to the path strategy's `loss_component_names`**, not only to the
  objective's `extras` — otherwise `KeyError` on the first batch, *after* the slow dataset
  build.

### C8 · a latent run's KL: where it is spent

- **A latent run's total KL says nothing about WHERE it is spent, and that is the whole
  failure mode.** Three 180-epoch arms were read as "the information is just small" while
  every posterior mean sat ON the prior mean (displacement 0.05–0.2 prior σ against a gate
  of one) and the budget bought only a narrower posterior. `history.json`'s `latent` block
  therefore carries `component_kl_mean_term_nats` / `component_kl_variance_term_nats` (the
  split; the variance term is the REMAINDER of `per_dimension_kl`, never a second closed
  form), `mean_displacement_sigma`, `component_kl_per_dim` and `active_units_0p05` — the
  FIXED 0.05-nat ruler, because `active_units` moves with the free-bits budget and made that
  gate unreadable across arms. **The three `component_*` numbers sum to
  `component_kl_nats_per_flight`, NOT to the charged `kl_nats_per_flight`** (free bits and
  the mixture estimator separate the two). Read them with `run_ts.py latent_readout
  --history <run>/history.json` (which needs no `--arm`: this is the epoch-1 reading), or off
  a checkpoint with `run_ts.py latent_probe`; the gate sentence is
  `outputs.control.latent.displacement_verdict` and its ruler `DEAD_MEAN_DISPLACEMENT_SIGMA`, so no
  surface restates either. A scalar auxiliary target concentrates the information in ONE
  dimension, so on such an arm the median displacement can miss what `component_kl_per_dim`
  shows — and that target is an INPUT of the posterior encoder, so a small `latent_aux`
  proves nothing on its own.

### C9 · turn-rate supervision reads the RHS

- **A supervision term on the turn rate reads the RHS, never a restated identity.** The
  shared point-mass RHS integrates `ψ̇ = g·n_realized·sin φ / (V·cos γ)` with the
  STALL-LIMITED load factor; the textbook `g·tan φ / V` equals it only on a coordinated
  level turn. `aerodynamic_model.torch_dynamics.heading_rate_rad_s` reads the ψ row out of
  `enu_rhs` itself for exactly this reason, and it is evaluated on
  `EndpointControlRollout.actual_controls` — the commands under the point-mass model, the
  ACTUATOR STATES under the lag. Pricing the command under the lag charges a turn the
  rollout never flew. Its target's uniform `(k+1)·T/N` endpoints coincide with the
  rollout's `cumsum(segment_durations)` ONLY through `true-time-position ⇒ uniform
  durations` (the config refuses anything else); `reference_control_supervision`'s midpoints
  ride on the same chain.

### C10 · the anchor state is `batch_contract.anchor_state`

- **The anchor state is `batch_contract.anchor_state(x, C)`, never `x[:, -1]`** — a history
  is `[B, L, C + K]` with `K` input-only conditioning columns (`target_conditioning`,
  `intent_conditioning`), and the control loss refuses a `[B, C + K]` anchor on the first
  batch. Two call sites had the raw slice until 2026-09-05; any control run with conditioning
  died there.

### C11 · `cta=self-q` vs `cta=given`

- **`cta=self-q` and `cta=given` are two ARMS, and only the first is quotable.** `given`
  hands the decoder the truth duration; `self-q` (`predict --cta-from-quantiles`) hands it
  the model's OWN `q_τ` and reads no future. The B3 arm TRAINS as `given` — the truth CTA
  drives the rollout while the quantile head trains as a target beside it — so the run
  directory honestly wears `cta=given` and only the prediction directory says `self-q`.
  The fan is `quantiles/qNN/` (plus `a20lo` / `a20hi` when a conformal table exists), the
  top-1 records ARE the q50 decode, and every record carries `source.ctaFromQuantiles` /
  `ctaQuantile`. Refused with `--cta-offset-s` and with `--z-from-posterior`.

### C12 · every quantile record says whether it is calibrated

- **A quantile record says whether it is CALIBRATED, always.** `source.calibrated` is
  written on every quantile record — `false` is the claim that this checkpoint has no
  conformal table, not a missing key — and `durationIntervalS` / `durationIntervalStratum` /
  `durationIntervalCohort` (the table's split, airports and smoke flag) appear only with one.
  The table is bound to `checkpoint_sha256` AND to `DURATION_QUANTILES`: one left behind by
  other weights, or written under other levels, RAISES rather than widening this run's
  intervals by someone else's δ.

### C13 · calibrated coverage is measured; the gate reads the deployed block

- **The calibrated coverage is MEASURED, never guaranteed, and the gate reads the DEPLOYED
  block.** Two separate traps. (i) The calibration split is also the split the checkpoint was
  SELECTED on, so the finite-sample split-conformal guarantee does not hold as constructed;
  every surface says "empirically measured cross-half coverage" and the guarantee-bearing
  read is pre-registered as one test-split measurement at `freeze-test`. (ii) A per-stratum δ
  is measured on its own stratum's members, but DEPLOYMENT falls through
  `INTERVAL_STRATUM_PRECEDENCE` — a refused stratum's flights take the pooled δ, and the
  pooled row says nothing about them (synthetic check: pooled row 0.794, the fallen-through
  group 0.167, deployed pooled 0.756). `calibration.calibrate` therefore publishes a
  `deployed` block that scores every held-out flight under the δ it would really get, and
  **design gate 3.4-2 reads that number**.

### C14 · `intent_conditioning=truth-…` reads the future (FROZEN)

- **`intent_conditioning=truth-…` checkpoints read the FUTURE** — FROZEN 2026-09-09
  (`INTENT_CONDITIONINGS_AVAILABLE`; the stored Phase 0 arms load, the scene encoder is
  archived under `archive/scene_encoder_2026_09/`) — (the truth join point, the
  lead's true landing time) — the Phase 0 upper-bound instrument of the scene design, never a
  result to quote as a predictor; the run name carries `intent=truth-…` so it cannot pass as
  one. Its lead and remaining-time channels are measured at the window's anchor, so it
  refuses random train anchors; `FlightSeries.lead_landing is None` means "roster never
  consulted" and raises, `LeadLanding(None)` means "no earlier landing" and reads as a
  clear runway. A `truth-join-duration` arm's `final_time_error_s` is an identity check
  (its input IS the duration target), never a duration result.

### C15 · `flight_metrics` is required on `write_batch`

- **`flight_metrics` is a REQUIRED arg to `write_batch`** — an optional metric is one that silently
  goes missing. Its accuracy block is built BEFORE any record file is written, so a non-finite
  ADE refuses the batch instead of leaving a record directory with no `summary.json` (review B-3).

### C16 · τ past RK4's stability limit

- **τ past RK4's stability limit produces NaN, not a worse answer** (explicit RK4 on
  `y' = -y/τ` is unstable above `h/τ = 2.785`, `config.RK4_REAL_AXIS_STABILITY_LIMIT`);
  `TSConfig` refuses exactly that bound at construction — until 2026-09-09 it refused `τ < h`,
  2.8× stricter than the instability it cited (review C-13).

### C17 · the fixed anchor is one definition

- **The fixed anchor is ONE definition: `dataset.fixed_anchor_index(config, minimum_anchor_index)`,
  read off a window set as `FixedAnchorTrajectoryWindows.anchor` / `.anchor_indices`.** It is
  `L-1` unless an experiment supplies a common floor (`run_ts.py history_ablation` trains every
  candidate `seq_len` at `max(L) - 1`). Every fixed-anchor consumer — the common-grid truth, the
  cohort floor, the terminal-velocity weights, the report metrics — takes the anchor as a REQUIRED
  argument from the window set; none may restate `seq_len - 1`. Until 2026-09-09 three of them
  did, so under the ablation the selection metric scored predictions against a truth taken
  `(max L − L)·dt` earlier than their anchor (review A-2; every number that runner published
  before then is stale).

### C18 · `probe_dynamics` carries every real batch key

- **`probe_dynamics(batch_size, device, config)` carries every key a real training batch carries**
  — the supervision targets are added under the same conditions `ControlContext._build_row` adds them,
  and `tests/test_supervision_terms.py` pins the two key sets equal. A key added to the real
  batch without the probe is a bare `KeyError` under `--batch-size auto`, after the dataset
  build (review B-1).

### C19 · instance normalisation stays off

- **Instance normalisation is OFF and must stay off** (iTransformer `use_norm`, PatchTST
  `revin`). In a threshold-anchored frame absolute position IS the signal. Signature of ON:
  lateral p95 pins at 14.3–14.5 km in all cells — a model that cannot place the endpoint.

### C20 · prediction records are anchored at `t=0`

- **Prediction records are anchored at `t=0` = the anchor sample**; `initial_state` is the
  observed state THERE, and the reference record must cover the SAME span (a whole-track
  reference against an anchor→threshold prediction reports kilometres of pure span mismatch).
  **That rebase does not survive into a shared clock — the CZML builder must add
  `source.anchorTimeS` back.** `observed_states` is REQUIRED in the schema and is the only source
  for the `look-` lookback entity.

### C21 · prediction is the anchor's own past only

- **Prediction is the anchor's own past only**: the anchor's control state is inverted from the
  observed lookback (`outputs.control.supervision.anchor_controls`), never from the first command.

### C22 · the teacher inverse inverts the configured forward model

- **The teacher inverse must be the inverse OF THE CONFIGURED FORWARD MODEL.** A schedule solved
  against the wrong equations is finite, bounded, the right shape, and its own optimizer reports
  a falling loss — it simply reproduces nothing. `outputs/dynamics/inverse.py` registers each
  inverse under the SAME config key as its forward model; a model added without one fails at
  registry lookup. The transport term (ω×v) is UNCONDITIONAL and there is no correct "off".

### C23 · a fitted teacher table belongs to one width, anchor and cohort

- **A fitted teacher table belongs to ONE width, anchor and cohort** — `control_imitation_target=
  "fitted"` reads per-flight schedules that reproduce the truth only at the N, the anchor, the
  uniform partition and the total duration they were fitted under. Six things are checked, and
  the run is refused on any of them: the SCHEMA and the `uniform` duration mode at load, then the
  cohort's AIRPORTS (flight keys are unique within an airport only, so a foreign table would
  otherwise read as "covers 0 of N"), N, the anchors the dataset actually samples, coverage (with
  the count — there is no partial mode) and each flight's `truth_duration_s` to 1e-6 s.

### C24 · a path's training-time input is opened once by `fit_model`

- **A path's TRAINING-TIME INPUT is opened by `fit_model` once (`OutputStrategy.training_input`),
  never by a dataset** — the control path's fitted teacher table, the plan path's rolled-window
  table (v5.2). `fit_model` hands it to the train and validation window sets (`training_input=`,
  renamed from `fitted_teacher=` 2026-09-11); every replay path (`evaluate-fit`, `predict
  --z-from-posterior`, the approach-cohort comparison, any `forecast`) builds its own window set
  WITHOUT it and must keep working when the file is gone. Its provenance rides in
  `checkpoint_metadata.json` / the payload under the object's own `metadata_key`
  (`fitted_teacher`, `plan_rolled_windows`), never in `data_provenance` — that object is compared
  for equality by `evaluate-fit` and `freeze-test`.

### C25 · replay runners fingerprint through `checkpoint_data_provenance`

**A replay runner fingerprints through `data_provenance.checkpoint_data_provenance(payload,
manifests)`, never `arrival_data_provenance(manifests)`** — the helper reads the pre-split
lateral-pass roster iff the checkpoint recorded one. Without the roster the v5 cohort
fingerprints as 14 435 KRDU arrivals against the checkpoint's 14 378 and
`require_matching_data_provenance` reports "the manifest changed" on a correct pair. The A0
runner hit it first; the L5.a fitter had the same plain call and died at startup on every v5
checkpoint until `e8df12f` (`docs/code-health-followups.md` §19).

### C26 · the ts data identity is the eligible SET, never roster bytes

**The ts identity of an eligibility roster is its eligible SET, never the roster file's
bytes** (`data_provenance`, schema `ts-arrival-data-v4-eligible-set`, 2026-09-08). The roster
embeds UPSTREAM provenance — `sources.evaluation_report_sha256` — so regenerating the observed
evaluation moves its bytes over an eligible set that did not change by one flight. That is not
hypothetical: on 2026-09-07 the five observed reports went v6 → v9 with byte-identical eligible
sets (KRDU 14 378 keys), and the byte-bound v3 identity then refused EVERY checkpoint trained
before it — `predict`, `evaluate-fit`, `run_ts_*` replay runners and `publish_ts_experiment_
trajectories.py` (which held a second copy of the comparison). So:
- the compared eligibility entry is `{schema_version, policy, eligible_set_sha256}`, hashed by
  `eligible_set_digest(keys)` — THE one definition, which `splits.data_selection_audit` also
  hashes its split rosters with, and which the publisher and `experiments.pipeline` import rather
  than re-deriving. **The roster's `counts` are NOT in it**: three of the five are reject
  tallies read off the observed evaluation (`excluded_lateral_indeterminate`,
  `evaluation_only`), so a re-graded flight that never was eligible would move them and refuse
  the checkpoint — the same defect, one level down. The other two are already compared as
  `arrival_candidate_count` and `source_records`;
- the byte facts and the counts stay AUDITABLE and out of every comparison:
  `data_selection.pre_split_eligibility[*].roster_sources` (from `eligibility_sources`) keeps
  the roster path, its digest, its counts and the evaluation report it was joined against;
- a checkpoint carrying the retired `ts-arrival-data-v3-eligibility-bound` fingerprint stays
  usable EXACTLY: its eligible set is re-derivable from the checkpoint alone — **`source_records`
  IS the eligible set** (the roster's keys are validated to exist in the manifest, so the records
  that survive the filter are exactly the eligible ones) — so `require_matching_data_provenance`
  compares it to today's per airport, refuses by name with both set digests, and then reads the
  v3 entry as today's. It deliberately reads NOTHING else from the payload: an earlier version
  rebuilt the checkpoint's `TSConfig` to recompute `data_selection`'s split digests, which
  re-imposed the model-recipe contract on a reader that needs none of it and raised `TypeError`
  on three real pooled checkpoints. Never by a registry of old roster bytes;
- `checkpoint_metadata.json` / `history.json` name the map `eligible_sets`; artifacts written
  before 2026-09-08 name `eligibility_rosters` (roster bytes) and are read through the
  checkpoint payload instead — a CV `cv_results.json` of that generation can only be checked by
  bytes, so `experiments.pipeline` re-runs CV when they moved (and prints which reason it was);
- **an IN-FLIGHT `cross_validation/cv_candidate_progress.json` from before 2026-09-08 is refused
  on resume**: the run contract it is bound to now names `eligible_sets` where it named the
  roster files, so `_load_candidate_progress` raises — naming the file and saying to delete it —
  rather than silently re-running finished candidates. Deleting that one file restarts the
  search from candidate 0; a finished `cv_results.json` is read, not refused.

### C28 · a codebook is a hashed artefact and every executor / prior checkpoint binds to its sha

**ARCHIVED 2026-09-20** — code under `archive/manoeuvre_codes_2026_09/` (its README says what moved and why; plan v3 §10's audit). The text below describes the archived code and is kept as its record. The codebook (`manoeuvre/tokenizer.py` `write_codebook` / `load_codebook`, plan §2.4; 2026-09-18) is a directory: `codebook.pt` holds the tokenizer's kind, FSQ levels, row count, segment length, the SCALE constants the rows were encoded under (`SEGMENT_ROW_SCALE`, `STATE_ROW_SCALE` — part of what decides a code, so part of the artefact), the fitted cohort's identity (C26's eligible-set digests; an empty identity is refused at write), the source checkpoint (path, sha256, run name) and the weights, ALL under one `sha256`; `codebook.json` mirrors everything but the weights and carries that sha. Loading verifies the sha, that the manifest equals the hashed payload, and that this build's scale constants are the artefact's — else it refuses by name. A loaded tokenizer is FROZEN: no gradient, and its mode stays `eval` under the parent's `train()`. **An executor trained AGAINST a codebook (`manoeuvre_codebook`) stamps `codebook_sha256` into `checkpoint_metadata.json`, and `load_checkpoint` refuses it when the directory's tokenizer weights no longer equal the ones inside the checkpoint** (`ControlStrategy.verify_checkpoint_payload`, the strategy hook `load_checkpoint` calls) — the prior's codes and the executor's z must be ONE vocabulary. A jointly trained executor carries its tokenizer inside the checkpoint and has no sha until exported; exporting a codebook from an executor that was trained against one is refused (that directory is its codebook). Writing never overwrites: an existing directory refuses.

### C29 · a replay finds a checkpoint's arrival manifests by the digest it recorded

2026-09-23 (the 2026-08-22..09-22 download merged into the live harvest). A checkpoint's
`data_provenance` pins each airport's `arrival_manifest_sha256` and every source track's
SHA-256, so it can only replay against the GENERATION of the harvest it trained on — not
against whatever `outputs/harvest` holds today. The superseded live root is frozen whole
(`trajectory_data_process.harvest.generations`, TD23 in `trajectory_data_process/docs/
06-harvest-reference.md`: moved aside, `FROZEN.json` with its manifest digests, registered in
`FROZEN_GENERATIONS`), and every replay path resolves the manifests through
`repo_layout.checkpoint_arrival_manifest(s)(payload, harvest_root)`: among the live root and the
frozen roots, the file whose bytes hash to the recorded digest; none raises by airport and
digest. The arrival loader reads a non-current schema only when the bytes are a frozen
generation's, so a frozen roster is read AS WRITTEN (the checkpoint's data is not converted).
Users: `experiments/support.checkpoint_manifests`, `anytime_curve.load_arm` (and through it
`latent_probe`, `chain_sensitivity`), `control_capacity_ceiling`, `clock_attribution`,
`predictability_report`, `control_basis_oracle` (teacher fit), `runway_hypotheses`.
`predict` / `evaluate-fit` take `--data` explicitly — pass the frozen root's manifests there.
New training always reads the live root (`repo_layout.HARVEST_ROOT`, the one definition; the
four runners that restated it import it).

### C30 · the instruction sentence artefact: one spec sha, written once, rows aligned with the signals

2026-09-23 (the instruction labeller); **v4 since 2026-10-03** (`docs/two_tier/two_tier_design.md` §4, §9.2, §14.2 A1–A3,
A8). An artefact directory under `4dTrajectory/outputs/POOLED/instruction_language/<name>/` is written by the runners in
order and never overwritten (`instructions.artefact._fresh` refuses an existing file):
`signals_{train,select,val}.npz` + `signals.json` (the flights, from the live harvest's eligible arrivals, split BY
OPERATING DAY (C32) — a test day's tracks are never opened, and `write_signals` / `load_signals` refuse a test-day flight
or one filed under another split; each flight carries `entry_time_utc` (row 0) and `landing_time_utc` — built by
`build_series` + `usable_series` under the default `TSConfig`, so the population is the models'), `candidates.json`
(each airport's candidate runways: the arrival manifest's `runway_targets`, the FAA CIFP runway geometry the modeling
target is built from; position, elevation, true course only — never the published glidepath or TCH — and every runway
end the harvest builds, which the landing rule reads), `spec.json` + `measurements.json`,
`sentences_{train,select,val}.npz` + `labels.json` + `readout.{json,md}`, `conformance/` (below) and, once an executor spec
flies them, `closed_loop/` (C38). Every file code reads back carries its format's name (`SIGNALS_SCHEMA`,
`CANDIDATES_SCHEMA`, `SPEC_SCHEMA` `ts-instruction-spec-v5`, `SENTENCES_SCHEMA` `ts-instruction-sentences-v3`) and is
refused under any other; **a name changes with its file's shape**, in the same change (2026-09-24, the user's rule). The
spec's sha covers every word, grid, class, tolerance and the reading rule (`instructions.spec.READING_RULE`,
`instruction-v4`); `load_sentences` refuses a sentences file read with another sha, `VocabularySpec.from_dict` refuses a
missing or extra key and another reading rule — no compatibility.

**The labeller is identified by what it reads, never by its source** (design D21, 2026-10-03; until v3 `spec.json`
recorded `labeller_source_sha256` and every runner refused other code): `instruction_labels` writes `conformance/` — a
fixed reference sample (`instructions.conformance`: train, seed 1337, 50 labelled and 10 refused flights an airport, their
signals, outcomes and word grids; since A15, `ts-instruction-conformance-reference-v2`, each labelled sentence's word
grid also on its UTC Δ grid at 4 and 8 s, `labeller.interval.on_utc_grid`, or why that grid refuses it, D49) — and a
`passed-<code12>.json` is written for each labeller code that reads it again the same, on the 2 s rows and every Δ grid (`check`, runner `instruction_conformance`, from a clean checkout; the code is named by the LOGIC of
`LABELLER_MODULES`, `io_utils.logic_sha256`). `require_conforming_labeller` asks for it before labelling, measuring a spec
or replaying; the sentence files record which code wrote them, as information.

A sentence's words line up row for row with the FIRST `len(words)` rows of its flight's signals (`signal_index` names
the flight): the sentence ends before its landing — the harvest's and the evaluator's condition (`read.landing_passages`:
the threshold plane crossed, interpolated with `final_approach.crossing.bracket_fraction`, within `LandingScreen`'s
1000 m of the centreline capped at half the spacing to a parallel runway — `airport.landing_cross_limit_m`, a MIRROR of
the harvest's `_runway_bracket_cross_limit` — and within its 100 m above the threshold). The landing is the passage the
flight never comes back from; a passage it comes back from must lie in a go-around's low pass (`read.flight_go_arounds`)
or the flight is refused — `read.admit` is the one gate the labeller, the measurements and the figures share. A flight
with go-arounds is read APPROACH BY APPROACH (D26): each go-around row is the row of its climb word, the descent that
reaches it says "no level-off", each approach has its own capture row, "unspecified" and speed words, row 0 says the
first low pass's runway and the runway word that ends a go-around the next approach's; the sentence file keeps each
go-around's row and the row its runway is said again (`go_around_offsets`). A change of speed says its STEPS (D43,
`labeller.speed.run_steps`): each 5 m/s grid value from the word in force to the run's target, at the first row where the
smoothed speed is nearer to it than to the value before, the target at the latest at the run's last row; the executor
flies a speed word at a_max (`speed_accel_max_mps2`) and "unspecified" at its own pace a_U (`autopilot.speed`). A flight's `typecode` is its OWN ICAO type
(`source["resolved_typecode"]`) or null; the default `TSConfig` keeps flights without aircraft dynamics (C31). **Since
artefact v6 (2026-10-02, the user) every row is on the UTC clock's even seconds** (`data.dataset.on_utc_steps`); a
sentence is put on a coarser row interval Δ (2, 4, 8 s, D25) by `labeller.interval.on_interval` on the UTC multiples of Δ,
never stored: each 2 s word on the NEAREST Δ row, a tie on the later (D45), a heading word moved across a runway word in
the frame where it is heard (D46: the class nearest its absolute track under the new course; no refusal). A spec is the vocabulary's format, not the data's: `instruction_spec --spec-from <artefact>`
(`artefact.keep_spec`) keeps another artefact's spec byte for byte, and `spec_from.json` says so.

### C31 · the aircraft filter: drop a flight only where dynamics are used

2026-09-24 (user decisions: "去掉 all 的 A320 回退", then "ts 用2 按需丢弃"). The retired `all` filter
flew every type without dynamics as an A320 (`TSConfig.aircraft_type`, default `A320`); both are gone
and a stored config carrying them is refused by name (`RETIRED_AIRCRAFT_FILTER`,
`RETIRED_FALLBACK_FIELD`; the field is dropped only under `openap-direct`, which never read it).
`config.AIRCRAFT_FILTERS`:

| filter | keeps | for |
|---|---|---|
| `all-flights` | every flight; one whose type has no dynamics (the performance index excludes it, or nothing models it, or its identity is unresolved) is kept as a scenario WITHOUT dynamics — no aircraft, mass NaN — and counted in `BuildReport.without_dynamics` | state output, the instruction labeller (kinematics only) |
| `modelled` | the flights the aircraft model can fly (preset, performance index own/substitute, OpenAP-direct); the rest rejected by name (`rejected_aircraft`) | control output |
| `openap-direct` | ICAO types OpenAP models under their own designator | unchanged |

**The default is by need**: `aircraft_filter=None` resolves at construction to
`config.default_aircraft_filter(prediction_output)` — `all-flights` for state, `modelled` for control —
so a stored config always names a value, and `aircraft_filter` is a REQUIRED serialized field (the
default moved; a config without it would read today's). `_validate_cross` refuses control output
under `all-flights` (its inverse-dynamics labels need mass, wing area and thrust). Everything that
needs dynamics asks `scenario.dynamics(purpose)`, which raises `NoAircraftDynamics` naming the purpose:
the control rollout's context, supervision and anchor gate, the lockstep verdict, the flyability
report. `predict` skips a flight without dynamics (its evaluation record's states carry a mass and are
written with `allow_nan=False`) and states the count in `summary["skipped"]["no_aircraft_dynamics"]`;
training's validation metrics still score every flight. `data_selection.json` (schema
`ts-data-selection-v3-performance-index`) records the performance index's sha256, and so does every
checkpoint (`payload["performance_index"]`): `load_checkpoint` refuses one trained under another index
unless its filter is `openap-direct` (which bypasses the index), as `FlightScenario.from_dict` refuses a
saved scenario — a rebuilt cohort would otherwise be flown as other aircraft. `BuildReport` counts
`selected_typecodes` and `without_dynamics` on BUILT series only, and `instruction_signals` over the
usable ones. Where a new config is derived from a stored one of another output: the control-basis
oracle reads a state seed's cohort under `modelled` (the flights the seed's predict scored); the
frame-ablation resume check compares the filter an arm trains under today
(`frame_ablation.resume_declaration`), so an arm trained under `all` is refused at resume. The executor
(`autopilot/`) builds its flights under the default (`all-flights`, the signals' own build) and flies only
those with dynamics: `replay.group_of` counts a flight without them ("no aircraft dynamics"), and its
`STAND_IN` group is now the performance index's substitutes only (`rollout_context` asks
`scenario.dynamics`); its plant's control config names `modelled`.

Every filter tags its pipeline directory and category (`_all_flights`, `_modelled`, `_openap_direct`):
the bare name belonged to `all`, whose outputs a new run must never overwrite. Run names read the
by-need default, so neither default shows a `fleet` item.

Size (the 2026-09-24 census of the eligible roster, before the build's own skips): train 50,693 flights,
of which 46,260 (91.3 %) have dynamics and 36,294 (71.6 %) are OpenAP-direct; 4,433 have none (3,821
excluded by the index, 612 unresolved identity). Val 10,635 / 9,668 / 7,574. Test sealed.

### C32 · the two-tier line splits by operating day, dealt once and committed; test days are sealed

2026-09-24 (`docs/two_tier/prior_design.zh.md` §3.3). `data/day_split.py`: an operating day is the UTC date
9 h earlier (`OPERATIONAL_DAY_SHIFT`, the overnight traffic minimum); a flight's day is its LANDING day (the one time
both the arrivals and the tracks roster carry; 20 of 72,247 eligible arrivals enter and land on different days).
The days are dealt by COUNT over sha256(seed:day): round-half-up 15 % test, 15 % val, 1/7 of the rest `select`
(the internal model-selection set), the remainder train — 14 / 14 / 9 / 53 of the 90 days, seed 1337. A count deal
depends on the whole list (7 more days move a development day into test, 16 a test day out), so it is made ONCE and
committed: `data/day_split_20260924.json` (`pinned_day_split`); `instruction_signals` refuses a harvest whose days
differ — extending the split is a decision that keeps these test days sealed, never a re-deal. `DaySplit.from_dict`
re-deals the recorded list and refuses a mismatch; `split_of` refuses a day not in the list; `development_split`
refuses a test day by name (`SealedDay`). Sealed means: no test-day flight's track is opened, labelled, counted in the
landing context (`prior.scene.context_landings`) or put in a scene. The per-flight split (`data.splits`) stays the
other ts models'; a cross-model comparison uses the flights both hold out (1,458: on a test day AND in the flight
split's test).

### C33 · the executor spec: written once, opened for executor code that flies its reference tracks within the bounds

2026-09-24 (`autopilot/spec.py`, `autopilot/replay.py`), **changed 2026-10-01 (the user): the executor is checked by what
it flies, not by its source**; v4 (`ts-executor-spec-v7`, design §5, §9.2 #3, §14.2 A4–A6). An executor spec is a directory
written once (`spec.json` + `measurements.json`, an existing file refuses): the parameters under their own sha
(`ExecutorParams`: cycle, roll rate, τ_γ, γ̇_max factor, timeout factor, word clock — every value from the vocabulary, the
procedure standards or a fixed choice, nothing from data; the decision-altitude check has none, D38), the vocabulary spec
sha it was measured against, and `source`: the logic hash of the executor code that measured it (`executor_source_sha256`
over `executor_source_files`: every `autopilot/` module but `spec.py` plus the repository modules they import directly, by
module name — `evaluation.thresholds` among them; each file's LOGIC, docstrings and comments free). Beside it,
`conformance/` (`autopilot/conformance.py`, runner `executor_conformance`, R42): the REFERENCE TRACKS — 50 labelled train
flights an airport (the replay's draw, seed 1337, own dynamics) flown on their labelled words by the code that measured
the spec, every cycle and every verdict, with each flight's input digest — and one `passed-<code12>.json` for each
executor code that flew them again within the bounds in EVERY way the executor flies (single-aircraft batch,
multi-aircraft batch with staggered starts, the single-flight executor): states ≤ 1e-6 m horizontally and vertically,
other floats ≤ 1e-6, every limit, mode, done cycle, outcome and word verdict equal. `replay.open_executor` refuses a spec of
another schema or vocabulary, and opens it only for executor code with a passed record against its reference
(`spec.require_conforming_executor`) and labeller code with one against the artefact's (C30); `replay.open_spec` opens it
without, for the check itself. **A code change whose tracks stay within the bounds needs one check (~30 s), and no spec,
training or readout is redone; one that leaves them needs a new spec.** `executor_spec` writes a new spec's reference and
passed record with it, from a clean checkout. Every `autopilot/` module counts, so a change to `closed_loop.py` or
`replay.py` asks for the check too.

### C34 · a prior checkpoint belongs to one sentence artefact

**Archived 2026-10-03 with the instruction-v3 prior** (`archive/two_tier_v3_2026_10/prior/`, design §14.2 A0): this
entry is the record of that code; the v4 prior is stage B (design §14.3).

2026-09-24 (`experiments/prior_train.py` `load_prior`). A prior is `checkpoint.pt` + `config.json` under
`ts-prior-checkpoint-v3`, written by `prior_train`, `prior_landing_reward` and `prior_augmented_reward` alike. It is refused
unless its vocabulary spec sha, the labeller source sha and the day split recorded in `config.json` are those of the
sentence artefact it is opened with, its candidate-runway table equals the artefact's, and its state loads whole
(`strict=True`). A new artefact under the same spec and labeller (v4 → v5 on 2026-09-26) opens every stored prior.

### C35 · a prior speaks under the procedure's masks it was trained under, recorded beside its checkpoint

**Archived 2026-10-03 with the instruction-v3 prior** (`archive/two_tier_v3_2026_10/prior/`, design §14.2 A0): this
entry is the record of that code; the v4 prior is stage B (design §14.3).

2026-09-26 (`prior/masks.py`; prior design §5.1). Two kinds of mask take words away when the prior speaks, kept apart: the
vocabulary's rules (the grammar, the runway not said again, the listener's lock) are the speaker's own, always on, bound
through the spec sha (C34); the procedure's masks belong to the post-training stage and are NAMED SETS with a version
(`masks.SETS`: `procedure-altitudes-v2`, post-training design §3.4; `procedure-altitudes-v1`, the edge alone, retired at
`181295fc`). Every runner that writes a prior's `checkpoint.pt` writes `procedure_masks.json` beside it
(`ts-prior-procedure-masks-v1`: the sets, each with the sha256 of the data it reads — for the altitudes every
`RunwayProcedure` field — and the checkpoint's sha256): `prior_train` and `prior_landing_reward` none,
`prior_augmented_reward` `procedure-altitudes-v2` (each runner's `STAGE_PROCEDURE_MASKS`). `load_prior` returns the model's
own built for the artefact's airports (`LoadedPrior.procedure_masks`) and refuses a directory without the record, another
schema, another checkpoint beside it, a set this code does not implement, or procedure data other than the record's.
`Speaker` / `ClosedLoop` / `speak_and_fly` take `procedure_masks` with no default; `prior_free_generation` uses the model's
own unless `--procedure-masks none|<sets>` says otherwise. A change to what a set allows is a new name, the old one deleted
— a model trained under a set the code no longer has is refused, never spoken under the new rules.

### C36 · loss of separation is judged by `inference/separation.py` under two readings; the closed loop's checks and reward use `VISUAL`

2026-09-27 (multi-aircraft design §3.2, §9 items 15–16; readout `two_tier/readouts/2026-09-27_parallel_runway_separation.md`).
`losses(traffic, separation, reading)` judges every pair at one instant; `next_behind` / `wake_at_threshold` (TBL 5-5-2 at
the leader's threshold) are the same under both readings and take none. `IFR` is JO 7110.65BB's instrument rules as
written. `VISUAL` (user 2026-09-27) assumes visual approach clearances and NEVER visual separation (7-2-1: the vocabulary
has no traffic-in-sight / maintain-visual-separation words). It differs from `IFR` only in two places, and never judges
a pair `IFR` would not (a random-scene test): dependent and independent parallels are free once both aircraft are
TURNED IN (`_turned_in`: track within `runway_schedule.FAA_VISUAL_INTERCEPT_MAX_DEG` = 30° of the course, 7-4-4 c2 a /
c3 a, AND on its own side of the midline between the two centrelines — the midline is our reading of "will intercept";
reading it as "heading toward its centreline" made on-centreline drift a loss); established finals of other directions
are not judged (3-10-4 not modelled). A close pair (< 2,500 ft) stays one runway under both (7-4-4 c1 b, N JO 7110.805,
needs visual separation). `Traffic` holds its contract: along, signed track − course (in [−180, 180]) and signed
distance right of the centreline are finite exactly when a runway is in force; an established aircraft has a runway.
`Separation.right_nm` (from `parallel_relations`) carries each parallel centreline's signed offset, both orders. The
package does not reach `instructions` (architecture test): the caller measures capture, clock position and the angles.

### C37 · a traffic prior is a single-aircraft prior with a traffic attention that starts at zero (`ts-prior-checkpoint-v5`)

**Archived 2026-10-03 with the instruction-v3 prior** (`archive/two_tier_v3_2026_10/prior/`, design §14.2 A0): this
entry is the record of that code; the v4 prior is stage B (design §14.3).

2026-09-28 (multi-aircraft design §2.5, §6.2, §9 items 22–23). `prior.model.with_traffic(model, EDGE_FEATURES)` keeps every
weight of a single-aircraft prior (augmented, for the multi-aircraft post-training) and adds, in every layer after the
aircraft attention, a `TrafficAttention`: each aircraft reads only the OTHER aircraft present at its step, the edge
features into a bias per head and into the value read, its output layer without a bias and its weight at zero. The model's own
aircraft attention then reads only the aircraft itself, as it did alone. So the traffic attention adds exactly 0 until it
learns, and the model answers as the single prior does for each aircraft alone — to rounding only (a batch holding
several aircraft sums in another order than one holding one: 1e-15 in double, 1.2e-5 on augmented's logits in float32;
the single prior itself differs as much between the two layouts). Zeroing the edge layers of a scene prior's attention
would not do this — the softmax over all present aircraft gives another aircraft weight whatever its edges. An aircraft
with no other present at a step reads 0 there, however the layer has learned — the output layer has no bias (the
first review found a bias there learning from the first step) — and no softmax row is fully masked (no NaN forward or
backward). Only a single-aircraft prior gains one (`with_traffic` and `Prior` refuse other edge features). At zero only
the output layer has a gradient; the layers inside follow once it moves. `edges` holds the traffic features, whose first
columns are the model's own edge features (`SINGLE_EDGE_FEATURES`). The checkpoint `ts-prior-checkpoint-v5` is v3's payload
plus `edge_features`, `traffic_features`, `edge_source_sha256` and `start` (the single prior's directory and checkpoint
sha256); `load_prior` reads v3, v4 and v5 by name and checks the edge code of v4 and v5; the single-aircraft `Speaker`
refuses both.

### C38 · the closed-loop sentences: flown by an executor spec, corrected toward the observed path, checked by what they read

2026-10-04 (`autopilot/closed_loop.py`, `instructions/artefact.py` `write_closed_loop` / `load_closed_loop`, runner
`instruction_closed_loop` R50; design §4.9, D32–D34, D42, D44–D46). `<artefact>/closed_loop/` is written once, from a clean checkout,
with an executor spec (C33): for each split and each row interval given (D25), `<split>_<Δ>s.npz`
(`ts-instruction-closed-loop-v4`, refused unless every field is there and its spec sha is the artefact's): each flown
flight's words from its first predicted step (Δ row 16 s / Δ; row 0 says every column), which words the reading added
(`correction`), its states on the data's 2 s rows from its first row to its last said row, its Δ rows marked
(`on_interval`, D51, since A15; v3 stored the Δ rows only), observed before the first predicted step, flown from it — a
flown row between two Δ rows the executor's state at the end of its cycle there (airport-frame e/n, MSL height, track,
ground speed, vertical rate); the replay check compares every 2 s row, its errors against the observed path (`lateral_m` right positive,
`vertical_m`, NaN past the end of the observed path), the rows where §4.9 makes no heading / angle correction
(`uncorrectable`, D34), the last 2 s row of the open-loop reading whose words each row has said (`observed_row`) and the
matched point's observed time there (`matched_row`, 2 s rows), whether each flight was done at its time limit (`timed_out`) and the
executor parameters' sha it was flown with; `summary.json` counts the flights without a sentence by reason (not flown by `replay.group_of`,
refused on the row interval, refused by the closed loop — the grammar read at the flown height), the correction words per
column, the D34 readings (since A15 also `outside_the_tolerance`: per column the correctable rows, those outside Y / H —
the share is D34's third reading — and of those the rows after which no correction TOWARD the path is in force
(`without_a_correction_toward_the_path`, `closed_loop.outside_rows`, read from the stored sentence and its open-loop
reading: the heading word in force on the path's side of the observed word's track, the angle class steeper when too
high): the rule of D50 that §14.6 checks. §4.9 ends a correction when the error changes sign and starts the other one a
row later, so an overshoot row counts there), the rows past the end and the lateness of the observed heading words. The reading says each
observed word at the PLACE where the observed aircraft heard it, not at its time (D42): the words of the 2 s open-loop
reading (not the Δ grid), each said at the first Δ row whose matched observed time is less than Δ/2 before the word's
2 s time (D45: the nearest row; of several, each column's last word), so the words wait while the flown aircraft is
behind (a tie goes to the later row, as on the Δ grid, A14); the first predicted step says the words in force before Δ/2
after its observed time. A heading word is said in the frame
where it is heard (D46), when the observed word in force or the correction changes and its track differs from the one the
executor holds; a change of runway alone says none. Past the end of the observed path (D44) e_y is measured against the
last segment's line, there is no e_h, no correction is said (one in force ends: Claude's reading) and the rows count as
rows without correction. The flight runs until
the executor is done or reaches the replay's time limit (`replay.time_limits_s`: the remaining observed time from the
sentence's first row × 1.5, plus 900 s a go-around), so a sentence has the flown rows and its replay the same limit. It adds one-class heading
corrections beyond 30 m and one-class angle corrections beyond 15 m under a descent class (the spec's
`closed_loop_lateral_m` / `closed_loop_vertical_m`), ending under half of it, at a sign change or at a new observed word;
a level hold is the executor's capture of the level in force (Claude's reading). `conformance/` holds a reference sample
(train, seed 1337, 10 flights an airport, every row interval written) and a `passed-<code12>.json` per code (the logic of
the executor's files and the labeller's) that reads it again with the same words, flags and refusals and states within
1e-6 m; `require_conforming_closed_loop` asks for it before `executor_replay --closed-loop`, which refuses sentences of
other executor parameters, flies each from its first predicted step on the time clock and requires each flight to fly its
stored states again.
