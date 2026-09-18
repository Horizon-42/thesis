# ts_transformer reference — experiment runners (anytime / calibrated ETA / latent lines)

Full text behind the index lines of `4dTrajectory/ts_transformer/CLAUDE.md`, moved here
VERBATIM on 2026-09-16 when that file had grown to 118 KB (1,202 lines) and was loaded into
every ts session. Only the `### <ID>` headings are new (R1 and R5 also carry the one-line group headings that introduced them). Every heading has exactly one
index line in CLAUDE.md that ends in its ID; find one with `grep -n '^### C7 ·' docs/reference/*.md`.

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.** Measurements and campaign evidence still
belong in `docs/ENGINEERING_NOTES.md` / the design documents, status in `docs/OPEN_ITEMS.md`.

**2026-09-18:** the main line's 2026-09-14…09-17 additions to that CLAUDE.md (the control
contracts, the two-tier axes and traps) were placed here the same way when this branch was
rebased — same rule, new IDs.

### R1 · `run_ts.py anytime_curve` — A0

**Runners for the anytime / calibrated-ETA line** (2026-09-07,
`docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`):

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

**The latent line's own runner** (`docs/2026-09-07_latent_intent_design.zh.md` §六 L2.f):

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

Six runners, all refusing an existing output directory (an artefact is never overwritten) and all rebuilding the cohort from the EXECUTOR checkpoint's own split with its provenance verified:

- `manoeuvre_codebook --checkpoint <joint arm>/checkpoint.pt --out 4dTrajectory/outputs/codebooks/<name>` — the frozen codebook (C28); refused for an executor trained AGAINST a codebook (that directory is its codebook).
- `manoeuvre_readout --campaign <dir> --segment-s S --out <dir> [--write-records]` — gate T over one segment length's trained arms: every val flight at the fixed anchor flown for Δ under protocol C, ADE[0, Δ] on the 1 s grid, each token arm paired flight by flight with the SAME seed's no-token twin (gain = twin − arm, positive when the code helps; never a p value), code usage (T(iii)), the K rule; arms not yet trained are listed as pending; `--write-records` writes the Δ-long forecasts for the picker with a `manoeuvre_readout` summary block. A `manoeuvre_readout` / `manoeuvre_lockstep` records directory is published ONLY with `--category-variant` (the publisher's `VARIANT_RECORD_BLOCKS` refuses it bare); the registry's variant key `<run>@<slug>` is read with the slug lower-cased, so `@lockstep-A` in `intents.json` meets the `lockstep-a` category key (2026-09-18).
- `manoeuvre_prior --codebook <dir> --executor <ckpt> --out <dir> [--continuous]` — the prior on the executor's cohort (its split: the prior never sees a val flight; the operating-day split is P4's), selected on the val next term, the bigram baseline written beside it (T(ii)); `--campaign-id/--experiment-id` open the experiment manifest.
- `manoeuvre_prior_readout --prior <dir>/prior.pt --codebook <dir> --out <dir>` — val NLL / accuracy vs the bigram, the landed decision's accuracy, the flip rate between adjacent asks on the truth history.
- `manoeuvre_lockstep --executor <ckpt> --codebook <dir> --protocol C|A|A-truth|none [--prior <prior.pt>] --out <dir> [--write-records]` — one round = the segment; `none` (2026-09-18) flies a NO-TOKEN executor (`plan_conditioning = off`, the campaign's twin arm) with no code under the same rounds and budget, the codebook being the reference labeller only (the truth's and the flown legs' code columns; it must share the executor's segment) — the control every coded protocol is read against, added when the command-vocabulary executor's closed-loop lead turned out to be code-blindness; the three artefacts must be ONE vocabulary (a joint executor's codebook is the one exported from it, a frozen-codebook executor carries the sha, the prior carries the sha it trained on); the stratum table (ADE mean / p50, FDE, flyable, established, the displacement at 60 / 120 / 180 / 300 s, e_track and e_plan by round, how flights ended) and the per-flight rows; the budget is T₀ + max(30 s, 0.1·T₀) of the truth's duration (a cap; under A the prior's landed decides).
- `manoeuvre_gates --gate T|X|P-open-loop|P-discrete-vs-continuous|E|S …` — every gate over written artefacts, keyed by seed, refusing one seed; X needs `--guidance-established` (the rule guidance's share along the truth segments, measured).
- `manoeuvre_code_atlas --executor <ckpt> --codebook <dir> --out <dir> [--flights 8]` — plan §2.4's "码怎么看": every code flown from a few typical val states (half straight-in, half vectored), the K paths and end points in each state's start frame beside the truth's segment and code; the end-point spread over codes is what says whether the executor is conditioned on z at all.

### R8 · two-tier v3 stage A (`run_ts.py plan_cohort --arms`, `frame_ablation --only`, `manoeuvre_lockstep --protocol none`, `executor_failure_modes`, `executor_grid_gate`, `two_tier_grid_queue`; 2026-09-18)

The no-token executor's own axes — lookback L × segment Δ, `docs/experiments/two_tier_v3_grid_arms.json` (40 arms, `L<L>_D<Δ>_s<seed>`, plan `docs/2026-09-18_two_tier_plan_v3.zh.md` §5–§8, decisions D1–D11; the development notes `docs/2026-09-18_two_tier_v3_stage_a_notes.md`). Nothing here launches before the user's sign-off (M-A0′).

- **One development cohort per cell** (D1): every arm names its own `"development_cohort"` (an arm-level path wins over the file's; arms of one cell share it), and `plan_cohort --arms <declaration> --airport KRDU --name <campaign> --output-dir <campaign>` writes all of them from ONE data load (the build under the shortest lookback keeps every flight any cell keeps), plus `cohorts.json` (per-cell counts) and `data_selection.json`. A cell's usable set = the flights with one window at L−1 and Δ of truth after it on BOTH sides of the split (`train.usable_series`), minus the train flights the random-anchor window set (the control path's airborne rule) gives no anchor — the flight the trainer would otherwise refuse the run over. The split is the per-flight hash on the pinned `split_seed` (1337), so it is the same set-independent split in every cell and both seeds; `--arms` refuses a declaration whose split seed is not pinned, or whose arms share a cohort path but differ beyond `seed`. KRDU, written 2026-09-18: train 6798–6857 / val 1392–1405 per cell (3956 of the locked split's 12218 are the aircraft filter's, cell-independent).
- **`frame_ablation --only KEY …`** runs a subset of the arms (the queue trains one cell, reads it, moves on); resume is unchanged (an arm skips on its `history.json`).
- **The first ask is L−1** (D2): the declaration pins `anchor_floor_index 0` and `random_train_anchor_l1_share 0` (D7), `tests/test_two_tier_v3_grid.py` pins every decision the file carries (a silently changed lookback is what the 09-18 campaign was built on).
- **`manoeuvre_lockstep --protocol none` takes NO codebook** (before 2026-09-18 evening a same-segment codebook labelled the rows; the stage A arms have none): the row carries no code columns, the payload no codebook; schema `ts-manoeuvre-lockstep-v2` adds `first_ask` and per-row `first_ask_row`. **`--anchor-remaining-km X`** (X in `anchor_strata.DEFAULT_ANCHOR_GRID_KM`; the plan's 12 / 8 / 6) is §3.1's second reading: `lockstep.from_remaining_path` cuts each flight (`dataset.series_from_row`: the rows before are dropped, the clock kept, the supervision cut alike) so that the row `anchor_grid.bin_anchor` places at the bin — the sample nearest X km of path left, with a complete lookback before it and Δ of truth after — becomes the executor's fixed anchor; a flight with no admissible row is counted (`first_ask.flights_without_a_row`), never flown from elsewhere. The strata are re-read at that row (at 12 km the vectored group is thin: report n per group).
- **`executor_failure_modes --lockstep <dir> --out <dir> [--stratum vectored] [--per-mode 3]`** (A2, §5.2; NOT a gate): over a lockstep directory written with `--write-records`, every non-established flight of the stratum is read in the runway's course frame (`manoeuvre/failure_modes.py`: to-go along the final approach course, cross-track positive right, heading error, the package's established rule per row, the onsets of heading alignment; the turn delay = the flown path's last onset minus the truth's) and given one of six modes in order — `established-short`, `overshoot` (crossed the extended centreline AHEAD of the threshold, never established), `no-turn` (no alignment onset after the first row, unaligned at the end), `passed-abeam`, `parallel-offset`, `other`; the row also carries each side's first heading-aligned time and the to-go at the first ask (a downwind abeam is already past the threshold plane). Two meanings of "established" meet here: the lockstep row's `reference.established` (crossed the threshold on the final) selects the flights; the mode names use the centreline rule. `failure_modes.json/.txt` hold the rows and the per-mode p50s (with `ever_established_share` and `to_go_start_p50_m`), and `records_<mode>/` the first N flights of each mode as a record directory (`inference/export.copy_record_subset`: the three files per flight copied, the roster filtered, the subset's OWN `accuracy` block recomputed off the retained rows — `metrics_from_row`, which needs the `cross_track_p95_m` / `altitude_p95_m` row columns `write_batch` writes since 2026-09-18 — and `subset` naming the source) with a `failure_modes` summary block, registered in the publisher's `VARIANT_RECORD_BLOCKS` so it publishes only under `--category-variant`.
- **`executor_grid_gate --campaign <dir> --arms <declaration> --out <dir> [--reading L-1]`** (A1's verdict, §3.3): reads `<campaign>/lockstep/<arm>/<reading>/manoeuvre_lockstep.json` for every arm, tabulates every cell per seed (`gates.cell_reading`: n, established all / vectored / straight-in, fully flyable, vectored ADE mean, FDE p50), and — once every cell has both seeds — `gates.gate_grid` (exactly two seeds shared by every cell): the seed line is the grid's own p75 of |seed a − seed b| over the cells (p50 recorded beside it; D9), a cell is eligible when fully flyable ≥ 0.95 on both seeds, a "leader" on a primary (established over all flights; over the vectored group) is within the seed line of EACH seed's best, the winners are the leaders on both primaries or — when the two disagree — on all flights, and the tie goes to the shorter segment then the shorter lookback; `decisive` is false only when every eligible cell is a leader on BOTH primaries (L and Δ are not the deciding variables in this range; the shortest cell is taken for A2). A missing reading lists as pending and leaves the verdict absent, so the queue can call it after every cell. The plan's verdict is the `L-1` reading's.
- **`two_tier_grid_queue --arms … --campaign … --airport KRDU [--readings L-1 12km 8km 6km] [--cells …] [--dry-run]`** (§8): per cell in declaration order (Δ ascending, then L ascending) — `frame_ablation --only` its two arms, then `manoeuvre_lockstep --protocol none` per arm × reading into `<campaign>/lockstep/<arm>/<reading>/` (skipped when its artefact exists; no records — 40 × 4 would be ~6 GB; the winner's readings are re-flown with `--write-records` after M-A1), then the grid gate into `<campaign>/gate/after_<cell>/`; timestamped lines, `CELL <cell> complete` for a watcher, the first failed step stops the chain with its exit code (D11; a step that died after creating its output directory leaves one the runners refuse to overwrite — move it aside as `<dir>.aborted-<UTC>`, rerun), ≥ 3 GB free checked per cell, the PID in `<campaign>/two_tier_grid_queue.pid` for the run's duration (removed at exit; a second queue refuses to start while it is alive). The cells are the gate's (`grid_cells` off each arm's config), so `--cells` names what the table prints. Run detached from the runs worktree at the committed SHA.

