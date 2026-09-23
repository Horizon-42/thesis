# ts_transformer reference — contracts

Full text behind the index lines of `4dTrajectory/ts_transformer/CLAUDE.md`, moved here
VERBATIM on 2026-09-16 when that file had grown to 118 KB (1,202 lines) and was loaded into
every ts session. Only the `### <ID>` headings are new. Every heading has exactly one
index line in CLAUDE.md that ends in its ID; find one with `grep -n '^### C7 ·' docs/reference/*.md`.

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.** Measurements and campaign evidence still
belong in `docs/ENGINEERING_NOTES.md` / the design documents, status in `docs/OPEN_ITEMS.md`.

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
  (`final_approach_geometry.threshold_crossing_index`, the one rule both consumers of an arrival
  point read; the closest-approach / in-segment refinement is the caller's). A vectored flight's
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
  (2026-09-14 → 09-16; `docs/2026-09-14_specific_force_control_design.md`; the one-row structure:
  `docs/2026-09-16_two_tier_transformer_feasibility.zh.md` §11):
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
