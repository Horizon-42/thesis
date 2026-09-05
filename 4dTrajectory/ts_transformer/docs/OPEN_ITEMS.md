# ts_transformer — open items

Status log for the package: what each campaign settled, what it adopted, and what is left.
Moved out of `CLAUDE.md` so it is read when planning work, not injected into every session.
Newest campaigns first, long-standing scope limits last.

---


- **Control command-hook campaigns DONE 2026-09-06 (`control_hooks_20260906` v1 at KRDU,
  `control_hooks_v2_20260906` at KRDU + KSJC; report
  `docs/2026-09-06_control_hooks_results.zh.md`).** Adopted: the v2 soft barrier as a
  predict-time safety layer; not adopted: any hook inside the training loop (six arms, none
  beat its predict-time counterpart), the hard gate. Open: the combined lateral-barrier +
  vertical-nominal hook at predict time; the baseline ending 157 / 162 m below the glidepath
  — traced 2026-09-07 to the last minute of the rollout (on the final it sits within ±13 m
  of the glidepath; at the truth's landing time it is 540–680 m short and 140 m low, path
  angle −4…−5° vs −3°), NOT to data, coordinates or the fitted tail (2–6 s), and read as
  four objective-design faults — the isotropic 10 km metre-scale position loss (both
  paths) prices a 150 m height error at 2e-4 per endpoint (a mean over 64), the 47×
  open-loop imitation teacher never speaks to the rollout's own drift, the threshold
  anchors present (the 1.25-weight fitted terminal row, `state_endpoint_loss_weight`)
  share the 10 km scale and nothing stops at the ground, and the path loss carries no
  gradient to the time grid (training rescales durations to the truth; the overrun is
  inference-only):
  `docs/2026-09-07_control_training_review.zh.md` (P0: per-channel position scale, the
  vertical-only procedure term, a threshold-plane crossing loss; P1: a closed-loop
  DAgger-style teacher from the guidance laws). **The km-level error is elsewhere**: vectored
  flights carry 76 % / 60 % of pooled ADE and their error is the ATC join decision, which the
  ego-only input cannot see and a single-output head can only average — design for traffic
  context + join-anchor multimodal output (scene encoder, K join-distance anchors as decoder
  queries, WTA training, top-1 stays on the existing record contract, Phase 0 = oracle
  upper bound before any architecture work): `docs/2026-09-07_scene_join_anchor_design.zh.md`; a "committed to the final" gate for the vectored flights the v1 / hard
  barrier hurt (gate opening at d < 8 km or ≥ 16 km; every bin is net positive under v2 soft); a second seed at KSJC (its −66 m FDE gain is the smaller
  effect); PatchTST and the other three airports.
- **Final-approach constraint campaign DONE 2026-09-04/05 (`final_constraint_20260904`, KRDU +
  KSJC, 3 predict-only + 5 trained arms per airport; report
  `docs/2026-09-05_final_constraint_results.zh.md`, readout `docs/compare_constraint_arms.py`).**
  Bounded output adopted as candidate default (see the config entry above); penalty vetoed;
  projection kept as deployment fallback. Not done: making `corridor-bounded` THE default
  (decide together with the state-v3 continuity term, which addresses the start of the path
  the corridor does not), PatchTST, control output.
- **Procedure constraints in the learned model (2026-09-04 design + measurement):**
  measured on every 3rd rostered arrival (`docs/measure_procedure_adherence.py`) that **0.0 %**
  of observed KRDU/KSJC flights pass an off-axis IAF of their runway's RNAV(GPS) procedure,
  that 85–97 % (KRDU) / 38–83 % (KSJC) are established in the k=0.5 LPV cone by the FAF,
  and that once established 87–99 % of samples sit inside the cone and the −60/+120 m
  glidepath window (the ±22 m gate is met over the whole final by only 14–69 %). So the
  only data-consistent procedure constraint is the final segment (corridor + glidepath,
  gated by each flight's own join distance, never `d_faf`); IAF legs / pre-FAF join
  window / fix discs are normative and must not enter a loss. Design + measurements:
  `docs/2026-09-04_procedure_constraints_design.zh.md`; the method survey (penalty,
  bounded reparametrization, projection layers, primal-dual, sampling, two-stage with the
  optimizer) with reading list and the P0–P3 order:
  `docs/2026-09-04_constraint_methods_survey.zh.md`.
- **Index of the 2026-09-03 frame / runway / state-output experiments (four docs, one
  narrative, Chinese): `docs/2026-09-03_runway_frame_experiments_index.zh.md`; the
  runway-assignment reading list is `docs/literature/runway_assignment/README.md`.**
- **Runway-hypothesis expansion DONE 2026-09-03 (`run_ts_runway_hypotheses.py`, no training):**
  one threshold-anchored forecast per candidate runway, scored in the true runway's chart.
  The assigned label reproduces the baseline bit-for-bit (the chain check). What the label is
  worth: at KRDU a causal "active configuration" rule (most-used runway among development
  landings in the 30 min before entry) recovers the DIRECTION (majority runways 80–83 %) but
  guesses the majority sibling for the minority runway (05R/23L 29–31 %), costing +19 %
  pooled FDE (+30 % straight-in), i.e. ~500–800 m on those flights; at KSJC the same rule is
  93.8 % right and costs nothing (30L/30R are 230 m apart). An oracle over the real sibling
  gains 79 m of median FDE at KRDU against 32 m for a mirror-image fake sibling at the same
  separation, so about half of a K=2 sibling oracle is picking the luckier forecast, not
  runway knowledge; at KSJC the fake sibling gains MORE than the real one. The forecast's own
  closest approach to its hypothesised runway is useless as a selector (37–45 %). Left/right
  between parallels is the genuine unresolved mode; direction is not. →
  `docs/2026-09-03_runway_hypothesis_expansion.md`
- **The state model's KRDU endpoints sit ~250 m NW of every runway, and it is the model,
  not the data** (`docs/2026-09-03_krdu_nw_endpoint_bias.md`): a world-fixed translation of
  the whole predicted path present from the FIRST predicted step (240–350 m off the
  aircraft's actual position, path then parallel to the truth within 1.3°), on straight-in
  flights (established +204 m lateral miss, vectored +24 m), reproduced by noise-free
  synthetic straight-in histories, both seeds. Sign = KRDU's population-mean lateral drift
  (63 % of anchors SE of the centreline; observed +60 s drift median 0, mean +192 m NW);
  KSJC's drift is SE-ward and shows no bias. The objective cannot see it: 300 m on a
  straight-in is ~9e-4 per point against a ~0.08 pooled loss dominated by vectored
  kilometres, and the state output has no continuity to the anchor and no cross-track
  term. Read every arm-A per-runway cross-track number with this translation in mind.
  The anchor-relative output (state-v2 candidate, same doc set) fixes the start of the
  path and the straight-in stratum but loses ~350 m of vectored FDE at KRDU on both
  seeds — vetoed by its own pre-registered rule; a continuity term on an absolute output
  (keeping the endpoint prior) is the open next candidate.
- **Airport-frame ablation DONE 2026-09-03 (14 runs, KRDU + KSJC, two seeds; keep `enu`).**
  Removing the threshold anchor makes the model average across each parallel pair (KRDU:
  endpoints nearer the sibling 1.5 % → 12–15 %, minority runway pulled ~600 m, its FDE
  +30–45 %); target coordinates as input channels change none of that; the vectored-stratum
  gain (H2) flipped sign on the second seed. Seed floor on this axis: the threshold arm moves
  5–22 m pooled ADE across seeds, the airport arms up to 107 m — read every margin against
  that. Runner `run_ts_frame_ablation.py` (state arms, val split, resumable, no CV, no CZML;
  `--experiment-id` runs refuse a dirty worktree at EVERY arm start), readout
  `docs/compare_frame_arms.py`, results
  `docs/2026-09-03_airport_frame_ablation_results.md`. Not done: PatchTST A/B, control
  output, a KSJC cohort with enough 30R/12L flights to test the parallel pair there.
- **The KRDU run is DONE (three generations; current = 2026-07-20 B3)** — artifacts in
  `4dTrajectory/outputs/KRDU/ts_{model}_{mode}/` + `ts_pred_*` (B3 transport-consistent channels
  + physical-velocity fit; the previous generation is parked in `outputs/KRDU/_pre_b3_transport/`,
  the first is not reproducible — **quote ONLY current-artifact numbers**), tables in the package
  README.
  Robust across all generations: one-pass `full` beats chained `window` on whole-approach lateral
  error for both models (1.5–1.7× mean); PatchTST leads at short lead while iTransformer leads at
  600 s on the raw-tensor accounting (5438 vs 7384 m, n=893 — direction held in all three
  generations, margins 1.16–1.36×; channel-independence can't represent a turn's east/north
  coupling). NOT stable across retrains: gate-pass counts (0–4 of 152 — only "forecast ≠
  certifiable approach" survives) and the tail-vs-mean story (architecture-dependent).
  Hence: **treat any margin under ~1.5× as provisional** — both a split change and a ≤0.3 % data
  rescale moved effects of that size. The two lead-time accountings (record vs raw-tensor) are
  NOT comparable — the README states both with their n.
  Remaining: only KRDU trained (4 other airports harvested; the per-threshold ENU frame makes
  pooling a real design question, not a bigger `--data` glob).
- **The gate-pass conclusion needs re-deriving, not re-quoting.** The recorded "gate-pass counts
  0–4 of 152" scored a ±3 m vertical window against data offset ~33 m by the datum bug. Accuracy
  metrics (ADE/FDE, deviation vs a reference in the same frame) should be nearly unchanged; the
  gate verdicts were not measuring what they claimed.
- Follow-ups: single-aircraft only (no traffic interaction / ATC intent) and deterministic (no
  multimodality) — both are the survey's named open problems. Flyability is MEASURED but not
  FIXED (nothing projects a prediction back inside the envelope — README routes 2–4), and its
  polar is clean-configuration only, which is why it is read as a delta.
