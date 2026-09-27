# ts_transformer reference — experiment runners (anytime / calibrated ETA / latent lines)

Full text behind the index lines of `4dTrajectory/ts_transformer/CLAUDE.md`, moved here
VERBATIM on 2026-09-16 when that file had grown to 118 KB (1,202 lines) and was loaded into
every ts session. Only the `### <ID>` headings are new (R1 and R5 also carry the one-line group headings that introduced them). Every heading has exactly one
index line in CLAUDE.md that ends in its ID; find one with `grep -n '^### C7 ·' docs/reference/*.md`.

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.** Measurements and campaign evidence still
belong in `docs/reference/ENGINEERING_NOTES.md` / the design documents, status in `docs/history/OPEN_ITEMS_2026-09-18.md`.

**2026-09-18:** the main line's 2026-09-14…09-17 additions to that CLAUDE.md (the control
contracts, the two-tier axes and traps) were placed here the same way when this branch was
rebased — same rule, new IDs.

### R1 · `run_ts.py anytime_curve` — A0

**Runners for the anytime / calibrated-ETA line** (2026-09-07,
`docs/history/2026-09_latent_anytime/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`):

- `run_ts.py anytime_curve` — **A0**: replays `--checkpoint LABEL=PATH` (repeatable) from the
  `anchor_grid` REMAINING-PATH grid (which it imports, and which the `anchor-grid-common-grid-ade`
  selection metric selects four bins of) and reports, per bin per stratum, ADE (mean / p50 / p95) and
  FDE beside the time-free chamfer and Fréchet, |Δt| p50/p80 and the predicted duration p50.
  Five things it exists to keep right: the strata are computed once at **L−1 and fixed for
  every bin** (relabel per bin and a flight leaves the vectored stratum exactly when it rolls
  out on final — a survivor curve reads as an improving one); the bin coordinate is
  `approach_difficulty.remaining_path_profile_m`, which the covariate itself reads, so a
  consumer binning on distance-to-go never restates the arc length; **bins hold different
  flights**, so the monotonicity verdict is PAIRED over the flights present in both adjacent
  bins and prints that n (and reads the ADE **median**, the package's convention, not the
  mean); each checkpoint's arm is named from its OWN `random_train_anchor` (`A0-fixed` /
  `A0-random`) because one run may hold both and their difference IS the out-of-distribution
  cost; and every per-flight row stays in the artifact so another paired reading needs no
  re-run. Refused: `cta_conditioning=given` and `intent_conditioning=truth-…` (both read the
  future, the latter afresh at every anchor). `--command-hook` / `--hook-saturation` mean what
  they mean in `predict`. **`--min-future-s 60` empties the 2 km bin and most of the 4 km one**
  (≈27 s / 53 s of truth left at approach speed) — stated as `n=0 / partial`, but s_freeze can
  then only be read at ≥ 6 km, and the **~125 s duration-head floor** makes |Δt| p80 RISE
  toward the runway anyway (L1_native32 vectored: 64 s at 12 km, 141 s at 4 km).
  **`--write-records` makes a bin PUBLISHABLE**: the same forecasts the cells were scored on
  (ONE forward pass — the record summary's per-flight `ade_m` IS the curve's) also go through
  `export.write_batch` into `<out>/records/<label>/<bin>km/`, the shape `predict` writes, so
  `python -m evaluation` and the comparison-CZML publisher take it unchanged. Each
  `summary.json` gains an `anytime` block (schema `ts-anytime-records-v1`: campaign, arm, bin,
  the anchor rule, split/limit, `measured_flights`, `records`, coverage) — a bin is a
  re-anchored SUBSET of the split, and `publish_ts_experiment_trajectories.py` reads that block
  to give it its own category key (`…_a12km_val`), its own picker id (`<run>@12km`, because the
  picker dedupes by experiment id), the record campaign as its picker group, and a label that
  states the bin and BOTH denominators (`244 of 300 flights, --limit 300 of 1404 in the split`).
  Under `--write-records` the whole artifact is built inside the `.partial-*` staging directory
  and renamed on success, so a crash publishes no half-written record set — and the flag is
  STRICTER than the curve alone (the record contract refuses a non-finite metric that the curve
  would have printed as a NaN cell). **A bin of a POOLED checkpoint cannot be published per
  airport**: the runner replays the whole cohort its provenance names and offers no airport
  narrowing, so the directory holds every airport's flights; the publisher refuses it rather
  than filing all of them under each airport's category.

### R2 · `run_ts.py eta_calibration` — B2

- `run_ts.py eta_calibration` — **B2**: split-conformal (CQR) calibration of a
  quantile-bearing checkpoint's interval (`duration_head` ∈ `quantile`, `two-head`). Reads the DURATION HEAD ALONE
  (`outputs.control.forecast.duration_quantile_predictions` — one forward per flight, no rollout, no CTA),
  which is why it is seconds of CPU and why it may load a `cta=given` checkpoint
  (`load_arm(..., refuse_cta_given=False)`, the only instrument that may; the
  `intent_conditioning` refusal still applies — that oracle is IN the history). The VAL
  split is halved by the checkpoint's own `split_seed`: **half A fits the DEPLOYED δ and
  half B measures what it covered** — the mirror is a stability check, never averaged in.
  `--split test` / `train` are refused with the reason; a `--limit` SMOKE table is refused
  at the sidecar unless `--allow-smoke-table`. **The cut itself is probeable, READ-ONLY**:
  `--half-seed N` re-cuts the halves through the same `calibration_halves` and is refused
  without `--readout-only` (which writes the readout to `--out` and touches no sidecar), the
  table then carrying `half_seed` / `deployed_half_rule` and a `PROBE HALF RULE` banner that
  `write_conformal_table` refuses with no escape hatch — a deployed δ comes from the
  documented rule alone. **Read a single cut's deployed coverage with its CUT noise**: over
  five seeds on KRDU val the two coverages are strongly anti-correlated (a δ fitted on an
  easy half under-covers the other and over-covers when mirrored), so the deployed−mirror
  gap has sd ≈ 0.05 against a binomial SE of 0.015 and FLIPS SIGN; B1_quantile α=0.2 reads
  0.745 at the deployed seed 1337 and 0.791–0.818 at four others (mean 0.790), i.e. gate
  3.4-2's verdict on that arm is cut-dependent (2026-09-08, `docs/CHANGELOG.md`). Per stratum
  (`approach_difficulty.strata_masks`), and only the three
  `calibration.INTERVAL_STRATUM_PRECEDENCE` can deploy are fitted at all; below
  `calibration.MIN_CALIBRATION_FLIGHTS` = 30 in a half the stratum is REFUSED and a flight
  in it falls through to the pooled δ. The table lands in the checkpoint's
  `checkpoint_metadata.json` under `conformal` — a SIDECAR, never in `data_provenance`
  (which `evaluate-fit` / `freeze-test` compare for equality) — and carries its own cohort
  (`airports`, `limit`, `smoke_test`) plus the half rule.

### R3 · `run_ts.py quantile_fan_readout` — B3

- `run_ts.py quantile_fan_readout` — **B3**: `--arm <pred_dir>` of a `--cta-from-quantiles`
  run; it reads the five `qNN` leaves only (the calibrated endpoints are `predict
  --interval-endpoints`, off by default and not part of any gate). Per stratum: the share of
  flights whose truth duration falls in `[q10, q90]` and in the calibrated interval, the
  median widths (the veto reads the vectored one against 120 s), and the truth path's
  chamfer to the NEAREST of the five decodes against its chamfer to q50 — gate 3.4-3, read
  on the in-fan subset with the whole cohort beside it. **That geometric column is a
  readout, not a coverage guarantee** (§六 6): five trajectories are not a distribution over
  trajectories. **Its `cal.hit` column is IN-SAMPLE on a val arm** — the flights it scores
  are the ones the δ was fitted on — and is marked `cal.hit*`; the gate's coverage is the
  calibration readout's DEPLOYED block.

### R4 · `run_ts.py eta_error_readout` — B0

- `run_ts.py eta_error_readout` — **B0**: |`final_time_error_s`| p50/p80/p90 and the SIGNED
  p10/p50/p90 (same for `fde_m`) per stratum, straight out of existing `summary.json` files.
  A row is used only if it carries every metric AND every `STRATA_COVARIATES` field — a
  present-but-null `established_at_anchor` would otherwise read as False and change stratum.
  Measured 2026-09-07 on KRDU val: vectored |Δt| p80 **65.8–72.5 s** against straight-in
  **12.0–20.3 s**, so one pooled ETA interval cannot serve both strata.

### R5 · `run_ts.py latent_probe` — L2.f

**The latent line's own runner** (`docs/history/2026-09_latent_anytime/2026-09-07_latent_intent_design.zh.md` §六 L2.f):

- `run_ts.py latent_probe` — **L2.f**: the training-side densities of one or more latent
  checkpoints on a split (`--checkpoint LABEL=PATH`, repeatable), through the SAME cohort
  rebuild as A0 (`load_arm` / `cohort_series`, so the roster rule has one owner): prior and
  posterior per-dimension spread, the posterior mean's displacement in prior sigmas, the
  per-dimension KL and its mean/variance split, the prior's total std against N(0, I).
  Refuses a non-latent checkpoint (no posterior) and, through the shared loader, a
  `cta=given` / `intent=truth-…` one (that loader takes an `instrument` name so each runner
  refuses in its own voice). `--limit N` is a PREFIX of the split, not a sample — the
  displacement median moved 25 % between 100 and 200 KRDU flights, so a limited table is a
  smoke test and the artifact says so. **The posterior reads the future: never a prediction
  result.**

### R6 · `run_ts.py latent_fan_readout` — 4(a)

- `run_ts.py latent_fan_readout` — **4(a)**: the sample fan read by the B line's gate 3.4-3
  protocol, so the latent fan and the quantile fan are one deliverable measured one way. The
  chamfer-to-nearest-leaf logic has ONE implementation (`experiments.quantile_fan_readout.leaf_rows`
  / `leaf_geometry` / `geometry_cell`, imported); this runner supplies the leaves. `--arm
  <pred_dir>` of a `predict --latent-samples K --latent-random K` run, refused without its
  `modes/` leaves. Per stratum: the truth's chamfer to the top-1 decode, to the nearest of the
  `modes/` leaves and to the nearest of the `random/` leaves, each with the share of flights
  the nearest leaf beats top-1 on; minADE_K and top-1 ADE off the same records (the SAME
  definition as `run_ts.py latent_readout`, so the two artifacts cross-check); and the fan's
  lateral spread (p50 of the widest pairwise endpoint gap). **The random fan is the reading,
  not a footnote** — a fan always contains something nearer the truth than its own mean, so a
  nearest-leaf number alone measures nothing; the prior fan is informative only where it beats
  the N(0, I) control. KRDU val, 1404 flights, both L2.g seeds and L2.z: prior 124 m at
  0.92–0.93 against random 166–201 m at 0.34 (2026-09-08).

### R7 · the manoeuvre-token chain (`run_ts.py manoeuvre_*`, 2026-09-18)

**ARCHIVED 2026-09-20** — code under `archive/manoeuvre_codes_2026_09/` (its README says what moved and why; plan v3 §10's audit). The text below describes the archived code and is kept as its record. Six runners, all refusing an existing output directory (an artefact is never overwritten) and all rebuilding the cohort from the EXECUTOR checkpoint's own split with its provenance verified:

- `manoeuvre_codebook --checkpoint <joint arm>/checkpoint.pt --out 4dTrajectory/outputs/codebooks/<name>` — the frozen codebook (C28); refused for an executor trained AGAINST a codebook (that directory is its codebook).
- `manoeuvre_readout --campaign <dir> --segment-s S --out <dir> [--write-records]` — gate T over one segment length's trained arms: every val flight at the fixed anchor flown for Δ under protocol C, ADE[0, Δ] on the 1 s grid, each token arm paired flight by flight with the SAME seed's no-token twin (gain = twin − arm, positive when the code helps; never a p value), code usage (T(iii)), the K rule; arms not yet trained are listed as pending; `--write-records` writes the Δ-long forecasts for the picker with a `manoeuvre_readout` summary block. A `manoeuvre_readout` / `manoeuvre_lockstep` records directory is published ONLY with `--category-variant` (the publisher's `VARIANT_RECORD_BLOCKS` refuses it bare); the registry's variant key `<run>@<slug>` is read with the slug lower-cased, so `@lockstep-A` in `intents.json` meets the `lockstep-a` category key (2026-09-18).
- `manoeuvre_prior --codebook <dir> --executor <ckpt> --out <dir> [--continuous]` — the prior on the executor's cohort (its split: the prior never sees a val flight; the operating-day split is P4's), selected on the val next term, the bigram baseline written beside it (T(ii)); `--campaign-id/--experiment-id` open the experiment manifest.
- `manoeuvre_prior_readout --prior <dir>/prior.pt --codebook <dir> --out <dir>` — val NLL / accuracy vs the bigram, the landed decision's accuracy, the flip rate between adjacent asks on the truth history.
- `manoeuvre_lockstep --executor <ckpt> --codebook <dir> --protocol C|A|A-truth|none [--prior <prior.pt>] --out <dir> [--write-records]` — one round = the segment; `none` (2026-09-18) flies a NO-TOKEN executor (`plan_conditioning = off`, the campaign's twin arm) with no code under the same rounds and budget, the codebook being the reference labeller only (the truth's and the flown legs' code columns; it must share the executor's segment) — the control every coded protocol is read against, added when the command-vocabulary executor's closed-loop lead turned out to be code-blindness; the three artefacts must be ONE vocabulary (a joint executor's codebook is the one exported from it, a frozen-codebook executor carries the sha, the prior carries the sha it trained on); the stratum table (ADE mean / p50, FDE, flyable, established, the displacement at 60 / 120 / 180 / 300 s, e_track and e_plan by round, how flights ended) and the per-flight rows; the budget is T₀ + max(30 s, 0.1·T₀) of the truth's duration (a cap; under A the prior's landed decides).
- `manoeuvre_gates --gate T|X|P-open-loop|P-discrete-vs-continuous|E|S …` — every gate over written artefacts, keyed by seed, refusing one seed; X needs `--guidance-established` (the rule guidance's share along the truth segments, measured).
- `manoeuvre_code_atlas --executor <ckpt> --codebook <dir> --out <dir> [--flights 8]` — plan §2.4's "码怎么看": every code flown from a few typical val states (half straight-in, half vectored), the K paths and end points in each state's start frame beside the truth's segment and code; the end-point spread over codes is what says whether the executor is conditioned on z at all.

### R8 · two-tier v3 stage A (`run_ts.py plan_cohort --arms`, `frame_ablation --only`, `manoeuvre_lockstep` (protocol `none`), `executor_failure_modes`, `executor_grid_gate`, `two_tier_grid_queue`; 2026-09-18)

The no-token executor's own axes — lookback L × segment Δ, `docs/experiments/two_tier_v3_grid_arms.json` (40 arms, `L<L>_D<Δ>_s<seed>`, plan `docs/history/2026-09_two_tier_v3/2026-09-18_two_tier_plan_v3.zh.md` §5–§8, decisions D1–D11; the development notes `docs/history/2026-09_two_tier_v3/2026-09-18_two_tier_v3_stage_a_notes.md`). Nothing here launches before the user's sign-off (M-A0′).

- **One development cohort per cell** (D1): every arm names its own `"development_cohort"` (an arm-level path wins over the file's; arms of one cell share it), and `plan_cohort --arms <declaration> --airport KRDU --name <campaign> --output-dir <campaign>` writes all of them from ONE data load (the build under the shortest lookback keeps every flight any cell keeps), plus `cohorts.json` (per-cell counts) and `data_selection.json`. A cell's usable set = the flights with one window at L−1 and Δ of truth after it on BOTH sides of the split (`train.usable_series`), minus the train flights the random-anchor window set (the control path's airborne rule) gives no anchor — the flight the trainer would otherwise refuse the run over. The split is the per-flight hash on the pinned `split_seed` (1337), so it is the same set-independent split in every cell and both seeds; `--arms` refuses a declaration whose split seed is not pinned, or whose arms share a cohort path but differ beyond `seed`. KRDU, written 2026-09-18: train 6798–6857 / val 1392–1405 per cell (3956 of the locked split's 12218 are the aircraft filter's, cell-independent).
- **`frame_ablation --only KEY …`** runs a subset of the arms (the queue trains one cell, reads it, moves on); resume is unchanged (an arm skips on its `history.json`) — and an arm that names a `development_cohort` resumes only when its training recorded that cohort with the same train / val flights (`data_selection.development_cohort`); `plan_cohort` rewrites the file in place, so the path alone proves nothing (2026-09-23).
- **The first prediction is at L−1** (D2): the declaration pins `anchor_floor_index 0` and `random_train_anchor_l1_share 0` (D7), `tests/test_two_tier_v3_grid.py` pins every decision the file carries (a silently changed lookback is what the 09-18 campaign was built on).
- **`manoeuvre_lockstep` (protocol `none`) takes NO codebook** (before 2026-09-18 evening a same-segment codebook labelled the rows; the stage A arms have none): the row carries no code columns, the payload no codebook; schema `ts-manoeuvre-lockstep-v2` adds `first_prediction` and per-row `first_prediction_row` and names the round counts `predictions` / `held_predictions` and the per-round record `rounds` (v1: `asks`, `held_asks`, `asks_e`). **`--anchor-remaining-km X`** (X in `anchor_strata.DEFAULT_ANCHOR_GRID_KM`; the plan's 12 / 8 / 6) is §3.1's second reading: `lockstep.from_remaining_path` cuts each flight (`dataset.series_from_row`: the rows before are dropped, the clock kept, the supervision cut alike) so that the row `anchor_grid.bin_anchor` places at the bin — the sample nearest X km of path left, with a complete lookback before it and Δ of truth after — becomes the executor's fixed anchor; a flight with no admissible row is counted (`first_prediction.flights_without_a_row`), never flown from elsewhere. The strata are re-read at that row (at 12 km the vectored group is thin: report n per group).
- **`manoeuvre_lockstep … --first-prediction-row N`** (reading (c), added 2026-09-19 after M-A1): every flight cut so that row N of the whole flight becomes the executor's fixed anchor (`lockstep.from_row`, the same cut as the bin reading; N at or after the executor's own first row, a flight without the executor's horizon of truth after N is counted, not flown; `first_prediction.common_row`). Reading (a) confounds lookback with starting point (an L = 120 s executor starts 120 s in and flies a shorter, later segment than an L = 60 s one); at a common row every cell flies the SAME segment and differs only in what it saw. The queue names it `row<N>` (`--readings row59`), the gate reads it with `--reading row59`.
- **`executor_relative_gate --baseline SEED=DIR SEED=DIR --candidate SEED=DIR SEED=DIR (--seed-line-from <grid_gate.json> | --seed-line EST_ALL EST_VEC ADE_M) --out <dir>`** (§3.3 rows A3 and B, D45, 2026-09-19): a candidate closed-loop reading (a changed executor, or a token protocol) against its protocol-none baseline on the SAME flights — per seed the two payloads' rows are intersected and both readings recomputed there (`stratum_table` + `gates.cell_reading`), the two sides must start their closed loops by the same rule, and the four readings must share the lookback (`anchor`), the seconds flown per round (`executed_s`) and the split, with no `--limit` prefix (2026-09-23; the SEGMENT may differ — A3-a flew a 60 s forecast 20 s at a time against a 20 s baseline), the seed line is the grid gate's p75 lines or three numbers named on the command line (the output states the source). `gates.gate_relative`: improvement = candidate − baseline for the established shares, baseline − candidate for the vectored ADE; row A3 = on a primary both seeds improve and one beyond the line, the other primary not worse, fully flyable ≥ floor; row B = every metric not worse on both seeds, one metric beyond the line on both seeds, fully flyable ≥ floor. Writes `relative_gate.json/.txt`; never overwrites.
- **`executor_failure_modes --lockstep <dir> --out <dir> [--stratum vectored] [--per-mode 3]`** (A2, §5.2; NOT a gate): over a lockstep directory written with `--write-records`, every non-established flight of the stratum is read in the runway's course frame (`manoeuvre/failure_modes.py`: to-go along the final approach course, cross-track positive right, heading error, the package's established rule per row, the onsets of heading alignment; the turn delay = the flown path's last onset minus the truth's) and given one of six modes in order — `established-short`, `overshoot` (crossed the extended centreline AHEAD of the threshold, never established), `no-turn` (no alignment onset after the first row, unaligned at the end), `passed-abeam`, `parallel-offset`, `other`; the row also carries each side's first heading-aligned time and the to-go at the first prediction (a downwind abeam is already past the threshold plane). Two meanings of "established" meet here: the lockstep row's `reference.established` (crossed the threshold on the final) selects the flights; the mode names use the centreline rule. `failure_modes.json/.txt` hold the rows and the per-mode p50s (with `ever_established_share` and `to_go_start_p50_m`), and `records_<mode>/` the first N flights of each mode as a record directory (`inference/export.copy_record_subset`: the three files per flight copied, the roster filtered, the subset's OWN `accuracy` block recomputed off the retained rows — `metrics_from_row`, which needs the `cross_track_p95_m` / `altitude_p95_m` row columns `write_batch` writes since 2026-09-18 — and `subset` naming the source) with a `failure_modes` summary block, registered in the publisher's `VARIANT_RECORD_BLOCKS` so it publishes only under `--category-variant`.
- **`executor_grid_gate --campaign <dir> --arms <declaration> --out <dir> [--reading L-1]`** (A1's verdict, §3.3): reads `<campaign>/lockstep/<arm>/<reading>/manoeuvre_lockstep.json` for every arm, tabulates every cell per seed (`gates.cell_reading`: n, established all / vectored / straight-in, fully flyable, vectored ADE mean, FDE p50), and — once every cell has both seeds — `gates.gate_grid` (exactly two seeds shared by every cell): the seed line is the grid's own p75 of |seed a − seed b| over the cells (p50 recorded beside it; D9), a cell is eligible when fully flyable ≥ 0.95 on both seeds, a "leader" on a primary (established over all flights; over the vectored group) is within the seed line of EACH seed's best, the winners are the leaders on both primaries or — when the two disagree — on all flights, and the tie goes to the shorter segment then the shorter lookback; `decisive` is false only when every eligible cell is a leader on BOTH primaries (L and Δ are not the deciding variables in this range; the shortest cell is taken for A2). A missing reading lists as pending and leaves the verdict absent, so the queue can call it after every cell; a reading of another schema, another split, a `--limit` prefix or a segment that is not the cell's is refused by name (`reading_problem`, 2026-09-23 — so the 2026-09-18 campaign's v2 payloads are not re-judged by this code). The plan's verdict is the `L-1` reading's.
- **`two_tier_grid_queue --arms … --campaign … --airport KRDU [--readings L-1 12km 8km 6km | row<N>] [--cells …] [--dry-run]`** (§8): per cell in declaration order (Δ ascending, then L ascending) — `frame_ablation --only` its two arms, then `manoeuvre_lockstep` (protocol `none`) per arm × reading into `<campaign>/lockstep/<arm>/<reading>/` (skipped when its artefact exists; no records — 40 × 4 would be ~6 GB; the winner's readings are re-flown with `--write-records` after M-A1), then the grid gate into `<campaign>/gate/after_<cell>/`; timestamped lines, `CELL <cell> complete` for a watcher, the first failed step stops the chain with its exit code (D11; a step that died after creating its output directory leaves one the runners refuse to overwrite — move it aside as `<dir>.aborted-<UTC>`, rerun), ≥ 3 GB free checked per cell, the PID in `<campaign>/two_tier_grid_queue.pid` for the run's duration (removed at exit; a second queue refuses to start while it is alive; a file that does not hold one positive PID is refused by name — an empty one used to read as PID 0, which `os.kill` treats as the process group). The cells are the gate's (`grid_cells` off each arm's config), so `--cells` names what the table prints. Run detached from the runs worktree at the committed SHA.

### R9 · two-tier v3 stage B (`manoeuvre_lockstep --cohort`, the held token, `executor_relative_gate` gate B1, `two_tier_b_queue`; 2026-09-19)

**ARCHIVED 2026-09-20** — code under `archive/manoeuvre_codes_2026_09/` (its README says what moved and why; plan v3 §10's audit). The text below describes the archived code and is kept as its record. **Live now**: `manoeuvre_lockstep` flies the no-token closed loop only — no `--codebook`, `--protocol`, `--prior` or `--prior-landing-ends-flight` — and its payload schema is `ts-manoeuvre-lockstep-v4` (the v3 token / prior keys and `e_plan` gone). `executor_relative_gate`, `executor_failure_modes`, `executor_grid_gate` and `two_tier_grid_queue` read that payload unchanged; the relative gate's row B1 stays as the upper-bound row. The second layer's relative gate on the stage A executor L60_D20 — `archive/manoeuvre_codes_2026_09/docs/experiments/two_tier_v3_b_arms.json` (12 arms `<configuration>_<tokenizer>_s<seed>`: S20 / S60held / S60h60 × K16 / cv × 1337 / 2024; plan `docs/history/2026-09_two_tier_v3/2026-09-18_two_tier_plan_v3.zh.md` §5.2, decisions D30–D47; the development notes' §7). Every arm shares ONE cohort, the grid's L60_D60 `development_cohort.json` (D47: records ≥ 120 s, train 6853 / val 1404). Nothing launches before the user's "跑" (M-B0′).

- **`manoeuvre_lockstep … --cohort <development_cohort.json>`** (B0, D33): flies only that cohort's roster of `--split`, in the checkpoint's order (`cohort_keys`; a cohort flight the checkpoint's split lacks refuses), the payload's `cohort` block being the package's `development_cohort_audit` plus the path and the flown count. **The baselines are the queue's own step 0** (`baseline_steps`, the declaration's `stage_b.baselines`, only those the selected groups are judged against): the grid's L60_D20 re-read with records on the B cohort (`<campaign>/baseline/L60_D20_s<seed>/L-1/`; the baseline of S20 and S60-held) and L60_D60 flown 20 s at a time (`--execute-s 20`, A3-a re-flown by this code; `baseline/L60_D60_exec20_s<seed>/L-1/`; S60-h60's), each with `executor_failure_modes` into `failure_modes/<name>_s<seed>_none/`. A gate never reads another campaign's payload: `executor_relative_gate` and `executor_failure_modes` refuse a payload whose schema is not this code's `LOCKSTEP_SCHEMA` by name, and `gates.cell_reading` reads `executed_s` strictly (no compatibility — the repo rule of 2026-09-19; the stage A readings on disk are v2, their gates and failure-mode tables stand as written and are not re-run).
- **The held token in `lockstep.fly`** (D31 / D38 in `defaults.md`): a coded protocol's round flies `round_step_s` = the config's `token_step_s` (`--execute-s` stays a protocol-none option), one token for `token_hold` rounds; payload schema `ts-manoeuvre-lockstep-v3` (`token_span_s` / `token_step_s` / `token_hold` / `prior_landing_ends_flight`; per row `token_refreshes`, `prior_landed_at_s`, `prior_landed_error_s`; per round `token_index`, `phase`; strata `token_refreshes_p50`, `prior_landed_share`, `prior_landed_error_p50_s`). `--prior-landing-ends-flight` (A / A-truth only) restores the 09-18 rule; off (D37) the landing is recorded.
- **`executor_relative_gate`** writes a third verdict row, **gate B1** (§5.2.2: the truth-token upper bound — beyond the seed line on ≥ 1 of the three metrics on both seeds, fully flyable ≥ the floor, WITHOUT row B's not-worse clause: a truth token that buys one thing at another's price still carries information the prior is worth training for; row B, read on the prior's token, is where the price counts).
- **`two_tier_b_queue --arms … --campaign … --airport KRDU [--groups …] [--device auto] [--dry-run]`** (§8.2): step 0 the baselines (above); then one group = one configuration × vocabulary, both seeds (`groups_of`); per group, in declaration order (K16: S20, S60-h60, S60-held; then cv): `frame_ablation --only` its two arms → per arm `manoeuvre_codebook` (into the declaration's `stage_b.codebook_dir`, `4dTrajectory/outputs/codebooks/two_tier_v3_b_20260919_<arm>/`), `manoeuvre_lockstep --protocol C --write-records` (`<campaign>/lockstep/<arm>/C/`), `executor_failure_modes` (`failure_modes/<arm>_C/`), `manoeuvre_code_atlas` (`atlas/<arm>/`; under a held token the K paths are the horizon long and the truth's segment the span long, both named) → `executor_relative_gate` against the configuration's baseline (`stage_b.configurations[<c>].baseline` names a `stage_b.baselines` entry; the seed line from `stage_b.seed_line_from`, the grid's `gate/after_L120_D120/grid_gate.json` — a verdict, not a reading) into `gate/b1_<group>/` → ONLY when `verdicts.b1.pass` (read at run time): per arm `manoeuvre_prior --seed <the arm's>` (`priors/<arm>/`), lockstep A (records) and A-truth, failure modes on A → the relative gate on the A readings into `gate/b_<group>/`. A step whose artefact exists is skipped; the first failure stops the chain (D42); `GROUP <g> complete` / `GATE B1 PASS|FAIL` / `STOP:` for a watcher; the free disk (≥ 3 GB) checked per group; PID `<campaign>/two_tier_b_queue.pid`; the baseline checkpoints, the cohort file and a seed line carrying every metric's p75 must exist before the queue starts (checked at plan time); `--groups` narrows the arm groups and step 0 to the baselines they need.


### R10 · the instruction labeller: `instruction_signals` → `instruction_spec` → `instruction_labels` → `instruction_figures`

2026-09-23 (`docs/two_tier/instruction_vocabulary_design.zh.md` §3, §7–§8; artefact contract C30).
`instruction_signals --out <new dir> [--airports …] [--workers N] [--limit N]` deals the eligible arrivals by the
committed day split (C32; refused when the harvest's days are not its days), reads the train, select and val flights
(process pool, 500 keys per chunk, spawn) — a test day's flight is only counted from the roster, with how many of
them the per-flight split also holds out — and writes the signals (with the day split) and `candidates.json`;
`--limit` is a SMOKE option recorded in `signals.json`. `instruction_spec --dir`
measures on TRAIN only, and only on the flights and rows the labeller admits (`read.admit`; the
refusals are counted in `measurements.json` `not_admitted`) — pass A with `measure.provisional_spec()`
(the mean turn rate of turns ≥ `turn_rate_min_from_deg`, the rate and bank of their rows, speed transition accelerations, the course error on
the last 1.5 km flown, the move-piece angles for the descent classes, and the track / level wander
for the bands the two CHOSEN tolerances are read against: `sensitivity`), pass B with the measured
course tolerance (the aligned final's offsets by distance, the heading grid comparison on the rows
before it) — and writes `spec.json` (`measure.SUGGESTED` + `MeasuredValues`, the labeller source
hash, the git state; the rules in `measurements.json`). `instruction_labels --dir` reads train, select and
val with that spec — refusing a spec measured by other code — and writes the sentences, `labels.json`
and the readout. `instruction_figures --dir
[--count 24] [--seed 1337]` draws a seeded half straight-in / half vectored sample of VAL flights into
`figures/` with an `index.csv` for a verdict column. Every step refuses to write over an existing file.

### R11 · `run_ts.py instruction_training_export` — the frontend's Training sets

2026-09-23 (`aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`; the frontend side: `aeroviz-4d/docs/35-viewer-reference.md`
AV19–AV23; rewritten 2026-09-24 for `instruction-v3`, sample `aeroviz-training-sample-v7`). `instruction_training_export
--dir <artefact> --airports-root <…/public/data/airports> --airport ICAO
[--airport …] [--per-stratum 20] [--seed 1337] [--set-id <READING_RULE with _>] [--title …]` writes, per airport,
`<root>/<ICAO>/training/<set-id>/sample.json` (schema `SAMPLE_SCHEMA`) and adds the set to that airport's
`training/index.json` (schema `aeroviz-training-index-v1`, kept; every other set kept as it is). Everything is refused
before anything is written: the set's directory existing, or the index already listing the id. It draws only VAL flights:
the airport's labelled flights in a permutation seeded by `--seed`, read in order until both strata
(`readout.flight_record`) hold `--per-stratum`; the pool, the count read and the rule are written into the file and the
index. Every flight read is RE-READ with `read_flight` and must equal its stored sentence (the words grid, runway,
capture, clearance and "unspecified" rows) or the export stops naming the flight and the first differing cell;
`require_current_labeller` is deliberately NOT called, so the frozen artefact stays exportable after unrelated code
changes. All envelope geometry comes from `instructions/display.py` (outside the labeller hash, built only from
`envelope.py` / `labeller/*`): a heading word is its band over the rows it is judged on (`envelope.heading_word_rows`: its
row plus `heading_lead_s` to the next word's, never past the clearance) with each row's verdict (`display.rows_inside`:
`envelope.heading_words_inside` asked one row at a time), refused unless the count is `Reading.checks["heading"]`'s; the
capture turn is its rows (clearance → capture) and `checks["capture_turn"]`; the stratum is `readout.flight_record`'s
(from `turning_deg`). The runner adds only the geodesy (airport-frame metres → lat/lon; MSL → HAE for the track
and the tube walls: plus the flight's runway's offset, `training_files.runway_hae_minus_msl_m`, read from the live
arrival manifest and refused unless it is the one the artefact recorded — so after a re-roster no artefact re-exports,
as the overlay exporters already could not through `rebuild_series`) and refuses a word kind outside `WORD_KINDS` (the
frontend's `TRAINING_WORD_KINDS`). Torch-free; ~2 s for five airports. Tests: `tests/test_instruction_training_export.py`
(every write into `tmp_path`; the schema / rule / columns / kinds mirrors checked against `trainingSample.ts`).

### R12 · the executor: `executor_spec` → `executor_sensitivity` → `executor_replay`

2026-09-24 (`docs/two_tier/executor_design.zh.md` §9–§11; layout L31). `executor_spec --instructions <artefact>
--dir <new dir> --word-clock {time,distance,track}` (`ts-executor-spec-v6` since v11, 2026-09-27: the executor takes no
information beyond the vocabulary, the pointed runway's published threshold crossing height and glidepath, and the
procedure standards' roll rate) refuses a dirty tree, an existing directory and a labeller other than the artefact's;
takes τ_ψ (the heading lead) by method A from the vocabulary and p = `ROLL_RATE_DEG_S` (5°/s, FAA Order 8260.3G App. E
§4 ¶6.a, ICAO Doc 8168 Vol II; up to v10 it was the bank limit over the lead, 8°/s). Nothing is measured from data: the turn rates, the bank
limit, the speed changes' pace (a speed step over the shortest speed hold, 0.25 m/s²) and the altitude tolerance are
the vocabulary's, read at run time; the landing crosses each candidate's published TCH, read at replay
(`autopilot/runway_data.py`, the harvest's runway data at the evaluation CLI's default configuration and CIFP; the draw
records the heights); a word acts when said. The design's fixed choices (Δt, τ_γ, the γ̇ factor, the timeout factor)
are module constants and are written into `measurements.json` beside `spec.json`, with where every other value comes
from. Seconds.
`executor_sensitivity --instructions --executor <spec dir> [--per-airport 400] [--seed 1337] [--out]` flies one seeded
TRAIN sample per variant (the spec, then τ_ψ over 2–6 s, p over 4/6/10°/s, the γ̇ factor over 1–3, one at a time);
writes `sensitivity.json` into a new directory (default beside the spec).
`executor_replay --instructions --executor --split {train,val} --out <new dir> [--per-airport 0 = every flight]
[--seed] [--chunk 500]` is the §11 readout: own-dynamics flights gated, a stand-in's (the performance index's
substitute) reported, a flight without aircraft dynamics counted (C31); each airport flown in
chunks, judged, written as control-path prediction records (`records/<ICAO>/`, the plant contract's law resolving
the newtons) that `python -m evaluation` grades; the drawn flights' observed records are linked read-only and graded
by the same evaluation code (the harvest's own reports predate the speed gate's current methodology), and paired by
`flight_key`; `replay.json` holds every flight's row and the gate table (per
group, airport and stratum: landed, words inside per word judged, evaluation where the observed passes, ≥ 0.95; beside
the words, the heading words the clock told two at a time and the share without them).
**The VAL replay is stage 4 and runs only on the user's go-ahead**; development uses train. Every write refuses an
existing directory. Tests: `tests/test_autopilot.py` (every write into `tmp_path`).

### R13 · publishing the executor and the prior: `executor_training_export`, `prior_training_export`, `publish_ts_experiment_trajectories.py --executor-replay`

2026-09-24 (`aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §2.6, §4.6; the frontend side: `aeroviz-4d/docs/35-viewer-reference.md`
AV24–AV25). Two runners write OVERLAYS beside a Training set (R11), never into it: a file of their own schema under
`<root>/<ICAO>/training/<overlay-id>/`, listed in the airport's `training/overlays.json` (`OVERLAYS_SCHEMA`
`aeroviz-training-overlays-v1`, one entry per overlay: its kind, the set it is drawn over, that set's sample sha256, the
file). The index (`aeroviz-training-index-v1`) is not touched. The shared helpers live in `instructions/training_files.py`
(`open_base_set`: the set must be a read-back of this reading rule and spec under this `SAMPLE_SCHEMA`, drawn from val;
`base_flights`: each of its flights found in the artefact, its stored sentence equal to the set's events;
`read_overlays` / `write_overlay`: an id listed or a directory existing is refused; the payload is written compact).
Every airport is built before any is written.

`executor_training_export --executor <spec dir> --replay <spec dir>/replay-val --instructions <artefact>
--airports-root <…/public/data/airports> --set instruction_v3 --airport ICAO [--airport …] [--overlay-id
executor_<spec dir name>] [--device cpu]` (schema `aeroviz-training-executor-v3` since executor v11, 2026-09-27: a crossing names the runway crossed, and the outcomes `crossed_too_high` and `crossed_other_runway`; v2 since 2026-09-24, `instruction-v3`;
the replay `ts-executor-replay-v3`): opens the spec with
`replay.open_executor` (refused unless this executor code measured it), rebuilds the set's flights, flies those the
replay flies (own and stand-in dynamics) in one batch per airport and judges them. `replay.json` keeps each flight's
word verdicts without the words, so `word_verdicts` maps the judge's results back to the sentence (`said_words` uses
the judge's own bookkeeping, `judge.words_said` and `judge.read_flown`) and rebuilds the
`replay.word_results` list; that list, the outcome, the flown-as-said flag and the counts of words not judged / not
reached / superseded must equal the formal replay row exactly, the crossing within `CROSSING_TOLERANCE` (1e-9: another
batch composition reassociates float sums by an ulp), or the export stops naming the flight. One status per word
(`STATUSES`); a heading word is inside / outside by the judge's `verdict.words["heading"]` result for it (its rows from the
flown step it was told plus the lead), not judged when that result has no row (the lead reaches the clearance it was told
or its capture, or the next heading word was told on the same flown step); the heading word left to intercept the final on
its own is failed with that check. Each judged heading word carries its band on the flown rows (`display.heading_band`,
refused unless it gives back the judge's count) and each flight judged the flown track as the judge's gate read it
(`judgedTrackDeg`, its step k the exported track's point k), both on the observed track's branch (`chart_shift_deg`) —
except a dynamics failure, whose judge read the failed state the exported track leaves out: its words keep their statuses
and checks with no band and no judged track. The flown track goes out every 2 s step to
its outcome's row (MSL and HAE, the heading on the observed branch), with the formal row's evaluation verdicts, alignment
and limits, and the replay's gate table for the airport and all airports. **Run it from a worktree**: the spec's source hash counts `geokit` only when it resolves inside the
repository — outside from a worktree (as when the spec was measured), inside from the main checkout, where the spec is
refused (`docs/code-health-followups.md`, 2026-09-24). ~7 s per airport of 40 flights on CPU.

`prior_generation_training_export --prior <prior dir, or one round of a post-training run> --instructions <artefact>
--executor <spec dir> [--readout <its val free generation>] --airports-root … --set <a read-back set> --airport ICAO
[--airport …] [--samples 4] [--temperature 1.0] [--seed 1337] [--overlay-id generation_<name>[_r<NN>]_<sha256, 8 digits>]`
(2026-09-26, schema `aeroviz-training-generation-v3` since executor v11, 2026-09-27 — a crossing names the runway crossed, two more outcomes; v2 since the models were named on 2026-09-26 — v1 carried a free `--label`;
kind `prior-generation`; the frontend side AV31, AV32): the model's OWN sentences over the set's
flights — `prior_free_generation.speak_and_fly` on the flights its val readout flies (own dynamics; the rest listed with their
group), `--samples` each, one CPU generator seeded once and drawn in the order the airports are named (so a re-run is
identical); per sample the words as events on the flight's own rows (from `N_LOOK`), `flight_rows`' outcome and bookkeeping
(the outcome read again by `judge.outcome_of` must agree), the crossing, the forbidden mass per masked column, and the flown
track every 2 s on the flight's clock to the outcome's row (a dynamics failure's to the row before) in MSL and HAE. No per-word
verdicts (they would judge the executor, not the model). The words run to where the EXECUTOR stopped (`flight_rows`' count),
which follows `endS` for a crossing without the capture and the stall cut-off (`Executor.finished` stops at neither): written
as the formal readout counts them, shaded by the frontend. `--readout` copies the model's formal val free generation landed
shares (all / straight-in / vectored, null where the draw had none) at the payload's airport (`here`) and pooled (`all`),
refused unless it is this prior (path from `4dTrajectory/outputs/` on), this executor spec (content sha), artefact, val,
`N_LOOK`, samples, temperature and the procedure's altitudes exactly when the model's own masks hold them, and its draw's
per-airport count is a number or `EVERY_FLIGHT` (the draw's phrase, pinned against `replay.py`). **The prior speaks as it was
trained to**: under the vocabulary's rules and its OWN procedure's masks (`open_trained_prior` returns them from
`load_prior`), its sentences read by the formal readout's own `prior_free_generation.said_rows` — under the procedure's
altitudes a sentence stops at the first flown step below the glidepath lower edge (`BELOW_GLIDEPATH`; `sample_payload`'s
``stop``: no crossing, the track to that step's end state); `generation.procedureMasks` names each set with the digest of
the data it read, which the live backend checks before flying a sample again (AV26). The
checkpoint: `prior_training_export.open_trained_prior` (shared with `prior_training_export`). **The model is named, not
labelled** (`MODEL_NAMES` = base / landing / augmented, the post-training design's table; mirrored by the frontend's
`TRAINING_MODEL_NAMES`): `model_identity` reads the config — no `fine_tuning` is `base`, else the method that post-trained it
(`fine_tuning.schema` less its `-v<N>`: `METHOD_MODELS`, from `LANDING_REWARD_SCHEMA` / `AUGMENTED_REWARD_SCHEMA`; every version
of a method is the same model; an unnamed method is refused — a new stage's name is agreed with the user first) and its
round; `model_block` writes `{name, round, run (the directory holding the rounds; base: its own), checkpointSha256, variant,
trainedAt, fineTuning: {schema, from, fromName, fromRound} | null}`, the start model named from its own `config.json`, read in
the round's own outputs tree (`in_tree_of`: the rounds ran from worktrees). The frontend groups the overlays by name and run
and switches between the rounds (AV32). The executor spec must be one this code opens
(2026-09-26 evening: `v10_20260926`; the code refuses every earlier one, C33). ~2 min a
model for five airports of 40 flights on CPU, 1.2–1.9 MB an airport; a re-run is identical. Tests:
`tests/test_prior_generation_training_export.py` (the pieces, and `main` end to end on a synthetic artefact with stand-in
series). **`--augment-seed N`** (2026-09-27, the Training module §2.8): every flight flies from an AUGMENTED start instead —
stage 2's own code (`prior_free_generation.augmented_starts` / `augmented_inputs`, `prior.augment`: ±15°, ±150 m, ±5 %,
plausible within the train split's 1st–99th percentile start altitude, ≤ `AUGMENT_TRIES` draws), one move a flyable flight
drawn from `np.random.default_rng(N)` afresh at each airport in the set's order (apart from the samples' torch generator, so
every model exported with one seed flies each flight from the same moved start), time limit × `augment.TIMEOUT_FACTOR`;
written as kind `prior-generation-augmented`, schema `aeroviz-training-augmented-generation-v1` (the v3 kind untouched): no
`readout` (refused with `--readout`), `generation.augment` = the draw's rule, per flight `augmentDraws` (null: not drawn, not
on its own dynamics), `augmentation` (null: no plausible draw — not flown) and `observed` (the moved rows 0 … `N_LOOK` − 1);
default id `generation_augstart_<name>[_r<NN>]_<sha8>`.

`prior_training_export --prior <prior dir> --instructions <artefact> --airports-root … --set instruction_v3 --airport
ICAO [--airport …] [--overlay-id prior_<prior dir name>]` (schema `aeroviz-training-prior-v3` since the prior's third
version: the per-step arrays cover the predicted steps only, from `firstPredictedRow` = `N_LOOK`; null change metrics for a
column that never changes after the first step; the first-step runway beside the airport frequency and the rules — older
names are refused): the checkpoint is refused unless `prior_train.load_prior` opens it on the artefact (spec, labeller,
day split, candidate table, a whole state), it is not a smoke run and — for a variant that reads the landing context —
today's tracks rosters are the ones it trained with, by sha256 (`prior_train.roster_digests`; never the path, which is
the checkout's: a prior trained in a worktree is exported from the main checkout); inference is teacher-forced over the stored sentence (`prior.data.flight_record`), as it was trained
and read out. Per flight, column and predicted step: `changeP` (1 − P(unchanged)), the `k` most likely words given
a word is said (`TOP_K` 3, fewer where the column has fewer values: an airport's candidate runways) and their
probabilities, `truthP`; each flight's NLL per step in all and per column, computed before rounding
(`PROBABILITY_DIGITS` 4). The val `readout.json` travels unchanged. The inference path reproduces `readout.json`'s val NLL
per step to 1e-9 (checked 2026-09-24 over all 10,540 val flights). ~2 s per airport on CPU.

`publish_ts_experiment_trajectories.py --executor-replay <spec dir>/replay-val --executor-campaign CAMPAIGN [--airport …]
[--dry-run]` puts the replay's records under Experiments, one category per airport (`experiment_executor_<run>_<spec
sha12>_<split>`): `ExecutorReplay` / `ExecutorPublicationPlan`, not the checkpoint plan (the records came from an executor
spec, not a `TSConfig`). The category is named from the spec (its sha, its parameters as rows, run = the spec directory's
name, horizon `sentence` — a value the frontend's `EXPERIMENT_HORIZON_MODES` mirror lists after `config.HORIZON_MODES`),
filed under CAMPAIGN's `intents.json` entry (blocked without one), with the replay's own evaluation report (never re-run)
and the CZML split every `EXECUTOR_GROUPS_PER_CZML` (500) flights; the preflight requires the records' summary to be this
spec's and split's, of this airport only, and as many as `replay.json`'s recorded flights; an existing category is
refused. Its manifest (`ts-executor-publication-v1`, under `<output-root>/executor/<run>/<ICAO>/<split>/`) is passed by
the checkpoint refresh and the publication index. Checkpoint flags are refused in this mode. **Order**: publish only once
the frontend the users run carries `sentence` in `EXPERIMENT_HORIZON_MODES` — a category of it under an older mirror empties
that airport's whole picker (AV6).

Tests: `tests/test_training_overlays.py` (the verdict mapping on synthetic flights against `word_results`, the prior's
predictions on a small untrained network, the overlay helpers' refusals, the frontend mirrors, and the prior's `main()` end
to end on a synthetic artefact and checkpoint — every write into `tmp_path`) and
`tests/test_publish_ts_experiment_trajectories.py` (the executor mode, the builder stubbed, every root in `tmp_path`).
**Generations**: both runners open only artefacts this code reads. The 2026-09-24 publication (executor spec
`2674ab8c71a9`, prior `v1_20260924`, over the v5 `instruction_v2` sets) was made at `a320d1bd` on
`dev-publish-executor-prior`, where tests also ran both `main()` on those artefacts; since 9fb1b137 (signals v2, sample v6,
no A320 stand-in) neither the artefact nor the spec opens, so the next publication needs the next generation — since
2026-09-24 an `instruction-v3` artefact, Training v7 sets (R11), an executor spec (then `ts-executor-spec-v4`, now `-v5`) and replay
(`ts-executor-replay-v3`) measured by this code, a prior trained on the new sentences; and the frontend's
`TRAINING_SPEC_SHA256` moved to the v3 spec's sha, or every v7 set is refused.

### R14 · `run_ts.py heading_reading_compare` — the heading readings flown by the real executor (vocabulary design §10.1)

**ARCHIVED 2026-09-24** (`archive/heading_reading_2026_09/`, README there) with the holds reading it compared; to re-run,
check out `bd262763`.

2026-09-24. One TRAIN sample (`replay.draw_flights`: `--per-airport` flights of each airport on their own type's dynamics,
a seeded permutation) read under every variant (`VARIANTS`: `H1` = the holds reading; `H3-<step>-L<lead>` = the per-step
reading at a 5° or 2° step, merged with a half-step band, labelled `lead` seconds early), each flown by the executor at the
formal executor spec's parameters, HELD FIXED (its source hash is not required: the refactor of `judge` / `replay` moved it).
The vocabulary values are `--base-spec`'s (the previous vocabulary's spec file, schema `ts-instruction-spec-v3` and sha
checked), except the heading reading's and the heading tolerance (half a step + `HEADING_WANDER_ALLOWANCE_DEG`). Per flight:
`judge.outcome_of` (no word is judged), the evaluation verdict paired with the OBSERVED flight's graded here by the same
evaluation code (`observed_verdicts`: the harvest's record files linked read-only under `--out`, a cut `summary.json`; the
harvest's stored report may predate a change of the evaluation's methodology), the time-aligned mean and largest horizontal
distance to the observed flight, the heading words, the clearance and "unspecified" rows (both move with the reading). Rows
find their flight by dataset id (`tag_rows`: `fly_variant` returns them grouped by airport). Groups: all, the H1 strata,
"onto final" (an H1 inserted intercept ≥ 90°). Also one line per variant over every drawn flight, a refusal counted as a
failure. Writes `compare.json` (`ts-heading-reading-compare-v1`), `compare.md`, `records/<variant>/<ICAO>/`,
`observed/<ICAO>/` into a NEW directory. The measurement of 2026-09-24: `outputs/POOLED/analyses/heading_reading_20260924/`.

### R15 · the prior: `prior_train` → `prior_select` (prior design §7–§9, step 1)

2026-09-24, the third version. `prior_train --instructions <artefact split by day> --variant full|no-context|unordered
--out <campaign>/<variant>_s<seed> [--seed N] [--device cuda] [--limit N] [TrainConfig fields]` trains ONE variant
(`prior.data.VARIANTS`: the landing context in the candidates' vectors, the ordered heads) on single-aircraft scenes of the
artefact's train split, early-stops on val (per-epoch NLL per predicted step only) and reads the SELECT split (its own
operating days): `selection_readout.json` — per column the NLL per predicted step (all; after the first), the first
predicted step's top-1, after it P(change) / top-1 / top-5 where a word is said (null for a column that never changes
there, the runway) and false changes; the first-step runway's top-1 / top-2, by direction and side, established on the
final at `N_LOOK` or not, per airport (`prior.readout.runway_breakdown`), beside each airport's own runway frequency and the
causal rules B0 / B1 / B3 (`prior.readout.runway_rules`: the tracks roster's landings less the sealed test days, the
artefact's entry sectors, B0's majority from train-day landings); the two baselines counted from its own training
flights. Also `checkpoint.pt` (`ts-prior-checkpoint-v3`), `config.json` (the artefact's spec, labeller and day split, the
tracks rosters' sha256, `n_look`, the git state), `procedure_masks.json` (none, C35), `history.json`. A clean tree unless `--limit` (SMOKE: the first N flights
of each split). `prior_select --campaign <dir> [--seed 1337] [--replicate-seed 2024]` reads every run (one comparison:
artefact, rosters, training settings but the seed, smoke limit, commit — and this code that commit), applies the rule
(readouts doc §4): the leader at the seed, the seed line = |full at the seed − full at the replicate|, the first of
`PREFERENCE` (no-context, full, unordered) within the line; reads val ONCE on the chosen run and only then writes
`choice.json` (`ts-prior-choice-v2`) and the chosen run's `readout.json` — neither is ever overwritten. Tests:
`tests/test_prior.py` (both runners end to end on a synthetic artefact and tracks roster, every write in `tmp_path`).

### R16 · `run_ts.py prior_scene_census` — the scene prior's step 0 (prior design §9)

2026-09-24. `prior_scene_census --instructions <artefact split by day> --out <new dir>` reads the TRAIN days only:
the split sizes from `signals.json` (days, flights, test-day flights and how many the flight split also holds out);
sentence lengths, how many have no predicted step (≤ `prior.scene.N_LOOK` rows), the share captured at row `N_LOOK`
(`capture_row` ≤ `N_LOOK`) and cleared before it (`join_row` < `N_LOOK`); per predicted step (rows `N_LOOK`…end): the other aircraft in the
scene with a sentence and without one (background: refused by the labeller), whether a LEADER is there (design §8: lands
earlier on the same observed runway and is at most `LEADER_RANGE_M` = 10 km closer to its threshold, straight-line; the
nearest one's gap is recorded), whether the airport had a landing on a candidate runway in the previous `CONTEXT_WINDOW_S`
(30 min; the tracks roster's assigned landings minus the sealed test days, `prior.scene.context_landings`) and how
often that window reaches back into a test day; and the segments the flights chain into by overlapping time. Only
train-day flights are in the scene index (a step near 09Z misses the adjacent day's neighbours); departures and
overflights are in no scene. Records each tracks roster's sha256. Writes `census.json`.

### R17 · `run_ts.py prior_free_generation` — the prior speaks, the executor flies (prior design §9.1)

2026-09-25. `prior_free_generation --prior <chosen run> --instructions <artefact> --executor <executor spec dir> --split
select|val [--per-airport N] [--samples 4] [--temperature 1] [--seed 1337] [--chunk 64] [--device cuda] --out <new dir>`
draws the split's flights on their own dynamics (`replay.draw`, seeded), and flies each from the observed state at the
prior's first predicted step (`flight_inputs(..., anchor=N_LOOK)`): `--samples` times with the prior speaking —
`prior.generate.Speaker` samples a step's six words in column order (temperature `--temperature`) from the positions so far
(observed to `N_LOOK`, then the executor's), the landings before the step and its own words, with the vocabulary's
compatibility rules as a mask (`instructions.grammar.step_allowed`: the labeller's own check), and the executor
(`autopilot.executor.Executor`, stepped, `autopilot.sentence.Spoken`) flies the step; the runway column is masked by the
executor's own rule where it would refuse a change (`Executor.runway_locked`: cleared since the last go-around, or captured
— stricter than the vocabulary's), a flight the executor is done with says nothing and its row is frozen (a failed state
may be non-finite), and the readout cuts each flight's words at its end — and once with the labelled words
(the words in force at `N_LOOK`, then the sentence, on the spec's clock over the observed rows from there). Outcomes by
`judge.outcome_of` against the runway pointed at the end; time limit = observed remaining time × the spec's timeout
factor. Readout per group (all, strata, airport × stratum): outcome shares, first runway = observed, landed on the
observed runway, landing time − observed, words said per column after the first step, runway changes, go-arounds,
cleared at the end (and among the timeouts), the probability the prior put on what the grammar forbade (approach, angle).
Writes `generation.json` (every flight row) and `sentences.npz`; val only from a clean
tree. The executor spec must be this executor code's (`replay.open_executor`). ~36 s for 50 flights × (1 + 2) loops.
Since `Prior.extend` (2026-09-25) the speaker encodes row by row: the probabilities differ from a whole re-encode by
~1e-6, so a re-run of the readouts recorded before it (`v3_freegen_20260925/{select,val}_400x4`) may flip a draw — the
same model re-read at this code is not bit for bit the recorded one. Since 2026-09-25 (`dev-prior-fast`) the speaker builds
a step's rows for all the flights of one airport geometry together (`data.rows_inputs`, element by element; `row_inputs`
is its one-flight case) and writes them to the model's inputs in one copy: bit for bit the per-flight result (old and
new code dumped on the same seeds — free generation and the select split's inputs — 21/21 arrays identical;
numpy's element-wise functions on this i7-14700, AVX2 without AVX-512, give the same bits whatever the array's length),
and a 16-flight × 8-sentence chunk takes 10.0 s instead of 26.6 s. Splitting a round across processes would change the
random draws each chunk gets — not done.
`--procedure-masks` (2026-09-26, post-training design §3; was `--glidepath-mask`, the edge alone, 2026-09-25): the
procedure's masks the prior speaks under — `own` (the default: the ones its directory records, C35), `none`, or a
comma-separated list of sets; the log says whether they are the model's own. Under `procedure-altitudes-v2` each flight's
candidates' finals (`prior.procedure`) mask the speaker's altitude and descent-angle columns
(`Speaker(procedure_masks=...)` → `procedure.AltitudeMasks`, for the runway and approach just sampled): inside the FAF and the LPV cone a level below the
glidepath less 60 m less half a step and "descend to land" from below it; before the join (the first row inside that
region) a level below the published DA less half a step; once a row before the join was under the entry height (the
glidepath at the FAF) less half a step and no go-around is in force, a level more than half a step above the aircraft and
the climb class; "unchanged" (altitude) on a word that no longer holds. The speaker keeps each flight's join and dip per
candidate a row at a time (`AltitudeMasks.joined` / `dipped`, the running form of `procedure.pre_join`); the masked
probability is recorded per column (`forbidden_mass`). Every sentence, the labelled reference's too, ends at the first
flown step whose end state is more than the track tolerance (half a step + the tube's margin) below the glidepath lower
edge (`glidepath_stops`, read off the flown states at the step boundaries after the flight — where an in-loop check would
have stopped it), outcome `below_glidepath` (`BELOW_GLIDEPATH`, not the judge's); the rules before the join have no stop.
Each prior sentence and each flight's observed track are read before the join (`procedure.pre_join_readout`, rows from
`N_LOOK`: the most under the DA, the most climbed after the dip, the most under the FAA MVA where not cleared —
`prior.mva`, FUS3 charts under `repo_layout.MVA_ROOT`), counted in each summary's `pre_join` past the track tolerance
(DA, MVA) and an altitude step (climb), said beside observed; `generation.json`'s `procedure_masks` says whether the
altitudes were on. Without them the run is draw for draw the one before it (40 select sentences checked). `generation.json` is `ts-prior-free-generation-v5` since executor v11, 2026-09-27 (the judge's two new outcomes and a crossing's `runway_index`; v4 since 2026-09-26: `procedure_masks`, the
time-limit factors per start kind, the MVA chart read, the prior rows' `pre_join` / `pre_join_observed` — the climb read
only inside a stretch of barred rows from `N_LOOK`, restarting after a go-around); v2 added `below_glidepath` and
`glidepath_mask`. `--augment-seed S` (post-training design §4): every drawn flight is
flown from an augmented start (`prior.augment`: rotated about the airport ±15°, raised ±150 m, sped up ±5 %, one draw a
flight with seed S until plausible — the start's altitude inside its airport's train-day 1–99 % range at the first
predicted step, its airspeed above the executor's stall floor; `AUGMENT_TRIES` = 10, a flight none fits is left out and
counted); the labelled words are not flown then (they belong to the source's start); an augmented start's time limit is
`augment.TIMEOUT_FACTOR` (2.0, design §4.5) × the source's observed remaining time, a real start's the spec's (1.5). v3
recorded `augment_seed`, `augmented_left_out` and each flight's augmentation.


### R20 · `run_ts.py prior_procedure_check` — the procedure's altitude masks on labelled data (post-training design §3.6)

2026-09-25 (the edge), 2026-09-26 (the rules before the join). `prior_procedure_check --instructions <artefact> --executor
<executor spec dir> [--split train] [--replay-per-airport 400] [--seed 1337] [--chunk 64] --out <new dir>` reads every
labelled flight of the split (`labelled_rows`: a sentence's rows are its signals' first rows, contract C30; its runway
throughout) and counts, for the candidates' finals (`procedure.published_procedures`): the labelled altitude and
descent-angle words the masks would forbid where the prior would say them (the word in force at `N_LOOK`, then every one
said), rule by rule — the edge (levels and "descend to land" apart), the DA, no climbing back (levels and the climb class
apart) — the later steps whose altitude word in force they would make the prior replace, the tracks with a row more than
the track tolerance below the edge (with the depth at each one's deepest row), and the observed tracks' readouts before
the join (`procedure.pre_join_readout`); then flies `--replay-per-airport` flights per airport (own dynamics, `replay.draw`) on
their labelled words from `N_LOOK` as free generation's reference does, and counts the flights `glidepath_stops` stops
(at the stop: distance to go, depth, the altitude word heard; the outcome is the judge's). Pass (`passes`, written
before the run, user 2026-09-25): ≤ 1 % of the altitude and descent-angle words forbidden (every rule) and ≤ 3 % of the
replays stopped. Writes `check.json` (`ts-prior-procedure-check-v2`; v1 counted the edge alone); from a clean tree, never
over an existing directory.

### R21 · `run_ts.py prior_augmented_reward` — post-training stage 2 (post-training design §3–§5)

2026-09-26. `prior_augmented_reward --prior <the first stage's kept round> --base <the data-only step-1 run>
--instructions <artefact> --executor <spec> --out <new dir> [--rounds 8] [--real-per-airport 200]
[--augmented-per-airport 200] [--samples 8] [--select-per-airport 200] [--select-samples 2] [--seed 1337] [--chunk 32]
[--learning-rate 1e-5] [--kl-weight 0.04] … [--smoke]`. Each round draws a pool of 1.25 × (real + augmented) train-day
flights per airport (own dynamics); each airport's first `--real-per-airport` fly from their own start (`real_starts`), and
of the rest each is moved by a fresh augmentation until plausible (`prior_free_generation.augmented_starts`) and each
airport's first `--augmented-per-airport` are kept (`round_starts`; the augmentations' own random stream, seeded with
(seed, round), apart from the pool's draw; `sentences.json` records per airport the starts, the mean draws a start and the
share redrawn, and the sources given up — the readout of design §4.3 — and the reward by start kind); flies each start
`--samples` times under the stage's procedure's masks (`STAGE_PROCEDURE_MASKS`: `procedure-altitudes-v2`, whatever the
start model's own; every round's `procedure_masks.json` records them, C35) and the edge's stop (`speak_starts`; a real start's time limit the spec's, an
augmented one's `augment.TIMEOUT_FACTOR`); rewards as the first stage (a stopped sentence earns 0); one pass of `RewardTuner` with the FIRST STAGE'S RECIPE (design §5): the reward term and
`--kl-weight` (0.04) × the pull to the BASE model, both scored under the masks each sentence was said under
(`Speaker.allowed` → `train.allowed_tensors`), and `--data-weight` (1, refused at 0) × the teacher-forced NLL of a batch
of train-day flights (their ADS-B rows and labelled words) with every update. Without the data term the pull alone
either let the model leave the data (a fixed 0.04) or, driven by a KL budget, swung between that and erasing the first
stage (four runs, readouts §9). Every round records its model's distance to the base on its fresh sentences before the
pass — on the sentences it trains on (the starts with a contrast), the real starts' and the augmented ones' apart
(`RewardTuner.distance`, `distance_at_start.{real,augmented}`, `distance_sentences` their counts) — and the pass's
per-batch distance (`kl_trace`). Select readouts under the masks: the real starts (same
flights and seed every round) and one fixed augmented start per flight (seed + 7919; a flight none fits is left out and
counted, never replaced; the augmentations are in `config.json`), each with its readouts before the join beside the
observed tracks', plus the teacher-forced NLL (recorded only). `choice.json`: among the rounds within round 0's guards
(real landed ≥ − 0.01, landed on the observed runway ≥ − 0.02, and on the real starts and on the augmented ones alike, in
each of approach / heading / altitude / angle / speed the words a flight says after its first step no farther from the
LABELLED words — the select flights', an augmented start's its source's; |ln(said / labelled)|, the labelled counted from
each sentence's second predicted step to its end, `labelled_words` — than round 0 plus ln 1.2), the highest augmented
landed share, the earliest within 0.015. Writes like R19 (`ts-prior-augmented-reward-v5`, the clipped ratio as R19; v4 — the
same without it — is the stopped restart of readouts §15; v3 — augmented starts only, the
edge alone, the guard on the real starts — is readouts §10's run; v1 — no data term, a fixed pull, the NLL and heading
guards — and v2 — no data term, a KL budget — are the stopped runs of readouts §9; v1–v3 trained under
`procedure-altitudes-v1`, which the code no longer has); val is read afterwards with `prior_free_generation` (real
starts, and `--augment-seed` for augmented ones) under the stage's masks for every model compared: the kept round's own,
and `--procedure-masks procedure-altitudes-v2` NAMED for the start model beside it (landing's own are none — by default it
would be read under other masks than the round; so too a round 0 kept).

### R22 · `run_ts.py prior_glidepath_diagnosis` — why the labelled replays sink below the glidepath lower edge (prior readouts §12)

2026-09-26. `prior_glidepath_diagnosis --instructions <artefact> --executor <executor spec dir> [--split train]
[--per-airport 400] [--seed 1337] [--chunk 64] --out <new dir>` flies the stage-0 check's kind of sample (R20: own
dynamics, `replay.draw`, the labelled words from `N_LOOK` on the spec's clock) and compares the executor with the observed
aircraft cycle by cycle — the observed aircraft read at the row the word clock matched to the executor (`TrackClock`), both
against the published glidepath of the runway in force. Per stopped replay: executor and observed against the glidepath at
the stop, the observed binned (on the glidepath ≤ 30 m below / above the lower edge / above the stop line / beyond it);
over the stopped replays, the height the executor gave up to the observed aircraft summed by the words in force (altitude
kind × angle class × lateral capture); per descent class, on each replay's longest ≥ 20 s run of "descend to land" after
the capture, both mean path angles (height lost ÷ distance to go covered, each ≥ 500 m), the observed aircraft against the
glidepath at the run's start and the height given up per minute; at the first captured cycle inside the FAF, executor −
observed, apart for the replays that flew "descend to land" with the shallowest class before it. What-ifs (`WHAT_IFS`,
`--what-ifs`) fly the same sample with a line or two of `Vertical.rate` replaced in-process (`law_changed`; refused unless
each line is there exactly once) — the aim inside the word's tube after the capture, against the v11 law (level below the
published glidepath, toward the crossing point on or above it): `toward_below_glidepath` (the law up to v10: toward the
crossing point below the glidepath too), `class_centre_in_tube` (the class's nominal angle, never steeper than the line to
the crossing point, below the glidepath too), `join_from_below_centre` (the law below, the class centre on or above it), and
the two centre laws with the landing's reach read from the aircraft's own height instead of the tube's lower edge
(`*_own_reach`); each is read in full as the law is,
and on the replay gate's flights from row 0 (`replay_words`: landed, words inside, failures by check); the executor's
source and spec are untouched. Writes `diagnosis.json` (`ts-prior-glidepath-diagnosis-v3` since executor v11's third
milestone, 2026-09-27: "the law" is the v11 law; v2 the same day measured the what-ifs against the v10 law —
`analyses/shallow_class_law_20260927/`; v1 read one what-if, stops and outcomes only); from a clean tree, never over an existing
directory; development splits only.

### R18 · `run_ts.py prior_closed_loop` — archived 2026-09-25 → `archive/closed_loop_sft_2026_09/` (`docs/reference/entries.md` there)

### R19 · `run_ts.py prior_landing_reward` — the landing reward (prior design §9.3)

2026-09-25. `prior_landing_reward --prior <the step-1 run> --instructions <artefact> --executor
<executor spec dir> --out <new dir> [--rounds 8] [--per-airport 400] [--samples 8] [--select-per-airport 200]
[--select-samples 2] [--learning-rate 1e-5] [--warmup-steps 20] [--weight-decay 0.01] [--clip-norm 1]
[--tokens-per-batch 16384] [--kl-weight 0.04] [--data-weight 1] [--seed 1337] [--chunk 32] [--device cuda] [--smoke]`.
Each round draws `--per-airport` train-day flights (own dynamics, seed + round; repeats counted) and flies each
`--samples` times with the prior speaking as in free generation (`speak_and_fly`, temperature 1). A sentence's reward
(`prior.landing_reward`) is 1 when the executor's judge lands it on a runway in the airport's landing direction at the
first predicted step — the candidates landed on in the 30 min before (the input's landing pool, the flight's own left out,
no test day) and those within 90° of them; any runway with no landing in the window — else 0; its advantage is the reward
less its flight's mean (not divided by the spread). Only flights whose sentences differ are trained on. One pass of
`train.RewardTuner` over this round's sentences only: the clipped surrogate over the sentence's own words
(`train.flight_surrogate`: per word −min(r·A, clip(r, 1 − ε, 1 + ε)·A), r against the model frozen at the pass's start —
the one that said them — ε = `--clip-ratio` 0.2, per step, dropout off, the unmasked distribution; since 2026-09-26,
readouts §15: without it a pass's stale sentences pushed "unchanged" down until the model ran from the base) +
`--kl-weight` × the sample estimate of the KL to the frozen reference + `--data-weight` × a teacher-forced batch of the
train split per update (a round with no flight to train on is refused by name); each pass records the share of words
whose ratio left the interval (`clipped_share`, per batch `clipped_trace`: PPO's clip fraction). The pass's
`reward_mean` is the surrogate's loss (≈ −words a step × the mean advantage plus the words' movement), not comparable
with the advantage-weighted NLL of the runs before the clipped ratio. The select readout (`select_readout`) is free generation on the select days in batches of 64 flights (`SELECT_CHUNK`), the same flights and seed every round, the teacher-forced NLL, and the share landed against the landing direction. `choice.json`: among
the rounds within the guards of round 0 (landed on the observed runway ≥ round 0's − 0.02, heading words per flight ≤
1.2 × round 0's; a round that landed nothing is excluded), the highest select landed share, the earliest within 0.015. Writes `config.json`,
`round_00/readout.json`, `round_<k>/{sentences.npz, sentences.json, checkpoint.pt, config.json, procedure_masks.json,
readout.json}` (the stage's masks: none, C35; a round directory is a prior run `prior_free_generation` reads — the val
readout, once), `history.json`, `choice.json`; from a
clean tree unless `--smoke`. Val and the sealed test days are never read. `ts-prior-landing-reward-v3` since the clipped
ratio (v2: advantage × the NLL, no ratio — the adopted landing model's run).

### R23 · `run_ts.py heading_lead_ablation` — the heading lead, the bank limit and the roll rate moved over one train sample (two-tier progress report 2026-09-27, P0 item 1)

2026-09-27. `heading_lead_ablation --instructions <formal artefact> --executor <formal executor spec> --reference-replay
<its replay-train/replay.json> [--per-airport 400] [--seed 1337] [--leads 2 4 6 8] [--bank-limits 25 32] [--bank-rates
derived 3 5] [--keep-records] --out <new dir> [--resume]`. A cell is (lead L, bank limit φ, roll rate p): the formal
vocabulary with `heading_lead_s` = L and `turn_bank_max_deg` = φ and nothing else, the formal executor parameters with
τ_ψ = L (method A, `autopilot/derive.py`) and p = φ ÷ L (`derived`: the executor's p up to spec v10) or a given p (under
tan φ ÷ 2L, `derive.stopping_roll_rate_deg_s`, the executor's own turns can outrun their stopping rate). Every cell relabels the formal train replay gate's sample (`replay.draw`, own and stand-in
dynamics) under its own vocabulary and flies it through `executor_replay.fly_airport` (judged, written as records, graded
by evaluation, paired with the observed verdicts graded once). The reference cell (the formal values) is flown first and
must reproduce the formal replay in every field the replay stores but its two verdicts, or nothing else is flown. Per
dynamics group × stratum: landed; words inside, judged under the cell's own vocabulary (comparable neither across L nor
across φ — a longer lead judges a heading word later, the capture turn is judged against φ); evaluation; heading words
per flight; over the landed flights the time-aligned mean and largest distance to the observed track and the time-free
Hausdorff distance (the observed rows closed to the threshold, the flown line ended at its interpolated crossing, both
read as lines at `DENSIFY_M` = 5 m; a non-finite value counted apart); the landing time; the cycles the bank limit and p
bound; the airport × stratum cells clearing the three gates; refusals; paired changes against the reference. No code the
executor's or the labeller's source hash covers changes, so the formal executor v10 and sentence artefact v5 stay current.
Writes `plan.json` (commit and library versions; `--resume` continues only the same plan), `observed/<ICAO>/`,
`cells/<nn>_<cell>/{cell.json, evaluation/<ICAO>.json}` (the flown records dropped once graded unless `--keep-records`),
`ablation.json` (`ts-heading-lead-ablation-v1`) and `ablation.md`; train only, CPU, from a clean tree at the plan's commit,
checked before a cell is flown and before its result is written.
