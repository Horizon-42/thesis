# Airport-center frame ablation — development plan (2026-09-03)

## Question

The model's only knowledge of the target runway is geometric: the chart is anchored at the
assigned threshold, so the destination is always the origin and the position channels ARE
distance-to-go. This ablation separates that into two testable claims:

- **H1 (target conditioning)**: threshold anchoring is a strong prior — removing it degrades
  FDE, worst at parallel-runway airports (KSJC 12L/12R separation 228.4 m, deterministic
  model must average between modes).
- **H2 (route stability)**: threshold anchoring *costs* accuracy on vectored flights, because
  the same physical airspace (downwind, STARs) appears at different chart coordinates
  depending on the anchoring runway. An airport-fixed frame makes route structure
  location-stable and should help exactly the strata where the model is worst
  (vectored: 2000–3900 m ADE vs ~500 m straight-in).
- **H3 (symbolic conditioning)**: an airport-center frame PLUS the target's chart position
  fed as explicit input recovers the threshold-anchored baseline — i.e. the model can
  consume coordinates as data, not only as geometry.

## Arms

| Arm | Frame origin | Target info in input | Config |
|-----|--------------|----------------------|--------|
| A (baseline) | assigned runway threshold | implicit (origin) | `coordinate_frame="enu"` (current default) |
| B | airport reference point | none | `coordinate_frame="airport-enu"` |
| C | airport reference point | explicit input-only channels | `coordinate_frame="airport-enu"` + `target_conditioning="channels"` |

**Scope**: `state` output only (purely kinematic — cheapest, no control/rollout coupling),
iTransformer, `full` horizon mode (600 s; endpoint effects are the point). PatchTST is
**structurally unable to use arm C** (channel-independent backbone, no cross-channel
coupling — see `models.py` / vendor PROVENANCE), so it is excluded from C and optional for
A/B. Control-output arms are a follow-up, not this ablation.

## Phase 0 — prerequisites (blocking)

1. **v5 re-roster** — every arrival manifest on disk is stale
   (`harvest-arrivals-v5-takeoff-excluded`; loaders raise on v4):
   `python -m trajectory_data_process.harvest --airport <ICAO> --evaluate-only`
   for KRDU and KSJC (the two airports this ablation uses). No download, no reassignment.
2. **Freeze the comparison protocol**: one seed, same cohort, same recipe across arms. The
   train/val split is built inside `train` from the seeded config and persisted in the
   checkpoint — same seed + same manifest ⇒ same split; verify by comparing the persisted
   split of arm A and arm B checkpoints before quoting any A-vs-B number.
3. **Fresh arm-A baseline**: existing KRDU checkpoints predate the v5 roster (trained on
   cohorts containing the removed flights). Retrain arm A on the re-rostered manifest so all
   three arms share one cohort. Do NOT compare against the archived B3 numbers.

## Phase 1 — the origin-⇔-threshold audit (do this BEFORE the new frame)

The risky part is not the new frame class — it is every consumer that silently equates
"chart origin" with "runway threshold". Introduce ONE value that names the distinction:

- Add `target_chart` — the scenario's target state projected into the frame
  (`(e_tgt, n_tgt, u_tgt)` + `runway_heading_rad`, computed once at `dataset.py:871`
  alongside the frame, carried on `FlightSeries`). Under threshold frames it is
  identically `(0, 0, 0)`, so refactoring consumers onto it is **behavior-preserving for
  arm A** and can be landed + tested before any frame change.

Known consumers to move onto `target_chart` (grep-verified 2026-09-03; re-grep for
`horizontal_distance_m`, `to_world_horizontal`, and bare `IDX["u"]`/altitude-vs-zero
comparisons before starting — this list is a floor, not a roster):

1. `dataset._observed_threshold_crossing` (dataset.py:930) — crosses the threshold plane at
   `along == 0`, i.e. the plane passes through the FRAME ORIGIN. Must become
   `along_from_target == 0` with `along` measured from `target_chart`. Under an airport
   frame the current code would detect crossings of a plane through the airport center —
   silently wrong labels, no error.
2. `channels.horizontal_distance_m` (channels.py:218) — "distance from the frame origin".
   Used by `forecast.py:456`; under an airport frame it becomes distance-to-airport-center,
   wrong by up to the runway–ARP offset (km-scale). Either give it a target argument or
   compute distance-to-target at the call sites.
3. `terminal_state_loss` / arc-length / duration supervision — anything that scores the
   endpoint "at the threshold" must reference `target_chart`, not the origin.
4. `anchor_eligibility` / `lateral_eligibility` / `approach_difficulty` — the CLAUDE.md
   says covariates are computed in world EN from the observed track (frame-independent);
   verify per module and leave alone where true.
5. `fixed_anchor_validation`, `fixed_dt_supervision` fitted-tail extrapolation — the tail
   extrapolates toward the threshold; confirm it is expressed target-relative.

Deliverable: refactor + a test asserting `target_chart == 0` under `"enu"` /
`"runway-aligned"`, and that dataset outputs (labels, crossing times, supervision weights)
are BIT-IDENTICAL before/after the refactor on a real KRDU manifest slice. Land this as its
own commit.

## Phase 2 — arm B: `airport-enu` frame

1. **Frame mode** (`coordinate_frames.py`): the projection math is `ENUFrame`'s unchanged —
   only the anchor differs. Add `"airport-enu"` to `COORDINATE_FRAMES` (config.py:30) and
   extend the resolver: `frame_for_state(state, coordinate_frame, airport_ref=None)` where
   the new mode REQUIRES `airport_ref` and raises without it (validate once at the boundary,
   per repo conventions). Existing modes ignore the argument.
2. **Airport reference source**: `trajectory_data_process.harvest.airports.Airport`
   (`code, lat, lon, elevation_msl_m` — the same single source the optimizer and evaluator
   already read; do NOT use the frontend's `airport.json` copy). Resolve once per manifest —
   the manifest declares its airport (validated at dataset.py:125) and the optimizer's
   import path proves `harvest.airports` is importable from modeling code. Anchor:
   `lat0=lat, lon0=lon, alt0=elevation_msl_m` (MSL — the record contract at this seam,
   consistent with `scenario.target`).
3. **What already just works** (verify, don't rebuild):
   - `frame_params` `[lat0, lon0, alt0, heading]` flows to the control rollout for
     chart↔geodetic conversion — follows the new anchor automatically (state-output runs
     don't exercise it, but keep it correct).
   - `runway_heading_rad` is already carried SEPARATELY from the frame rotation
     (dataset.py:553–556) precisely so ENU rollouts decompose along the actual runway.
   - `states_from_channels` inverts back to geodetic, so evaluation records, gates and
     ADE/FDE stay defined in WORLD space — metrics are directly comparable across arms with
     zero evaluation changes.
   - Checkpoint safety: `coordinate_frame` is serialized and validated exactly;
     `load_checkpoint` refuses old checkpoints under the new mode. `run_naming` surfaces the
     non-default frame in the run label (meta = deviations from defaults) — confirm the
     label renders before the first long run.
   - CLI: `--coordinate-frame` already exists (`__main__.py:461`); add the new choice only.
4. **Normalizer sanity**: position-channel spread grows by the runway–ARP offset (~1–3 km
   against a 25 km ring) — no action expected, but eyeball the persisted per-channel stats
   of arm B vs A once.
5. **Tests**: parameterize the existing channel round-trip tests over the new mode; add one
   test that a flight built under `"enu"` and under `"airport-enu"` differs in chart values
   by exactly the constant anchor offset (positions) and not at all in velocities; add one
   that `_observed_threshold_crossing` finds the SAME crossing time under both frames.

## Phase 3 — arm C: explicit target conditioning

1. **Config**: `target_conditioning: "none" | "channels"` (default `"none"`), serialized
   into the checkpoint like everything else in `TSConfig`; refuse
   `target_conditioning="channels"` combined with PatchTST at config validation (it is
   structurally inert there — an arm that cannot work should not be runnable).
2. **Do NOT extend `CHANNELS`** — it is a bidirectional contract (`states_from_channels`
   would try to PREDICT the conditioning, and the tuple indexes normalizer stats and
   checkpoints). Instead append **input-only** constant channels at the window/batch seam,
   after normalization: `(e_tgt, n_tgt, u_tgt, cos ψ_rwy, sin ψ_rwy)` from `target_chart`,
   broadcast over the 60 history steps. Normalize e/n/u_tgt with the POSITION channels'
   stats (commensurate scale); cos/sin are already unit.
3. **Model**: iTransformer treats channels as variate tokens — 5 extra tokens through the
   shared encoder; the flatten-in-channel-order state projector grows its input width
   `(6+5)·d → pred_len·6` while the OUTPUT stays 6 channels. Serialize the augmented input
   channel tuple next to `CHANNELS` in the checkpoint so a mismatch refuses to load (follow
   the existing pattern — that is what locked out the `ve/vn/vu` generation).
4. **Under arm A this is a free control**: `target_chart = 0` makes the extra channels
   constant zeros + `cos/sin ψ` — running A+conditioning as a cheap 4th cell tests that the
   mechanism itself is harmless (expected: no change; if it moves metrics, the plumbing is
   wrong, not the science).

## Phase 4 — runs

Cells (one seed, then replicate the headline comparisons once with a second seed to bound
seed noise on THIS axis — known elsewhere to be ADE ~1.7 m, verify here):

| Cell | Airport | Arm | Why |
|------|---------|-----|-----|
| 1, 2 | KRDU | A, B | calibrated testbed, has baselines and route-mix spread |
| 3 | KRDU | C | H3 on the better testbed |
| 4, 5 | KSJC | A, B | the parallel-runway stress case — H1's predicted failure lives here |
| 6 | KSJC | C | does explicit target restore the parallel endpoint? |

Each cell: `train → predict → eval` with the existing chain (`__main__.py` /
`run_ts_pipeline.py` machinery), `--coordinate-frame airport-enu` (+ the conditioning flag)
being the ONLY deltas. New output dirs via `run_slug()` — never rename or overwrite the
existing generation trees. Check free disk before the sweep (the ts experiment trees are
what moves it).

## Phase 5 — readout (decide these BEFORE looking)

Stratified, never pooled — per-row covariates (`route_tortuosity`, `remaining_path_m`,
`established_at_anchor`, …) already exist in `summary.json`:

1. **H1**: FDE by arm, stratified. Prediction: B ≫ A at KSJC; smaller gap at KRDU.
   Smoking gun: KSJC endpoint CROSS-TRACK distribution vs the assigned runway — bimodal or
   shifted ~114–228 m toward the sibling runway under arm B.
2. **H2**: ADE on the vectored stratum (high tortuosity, not established at anchor).
   Prediction if H2 is real: B < A there, even while B loses on FDE.
3. **H3**: C vs A overall. C ≈ A ⇒ symbolic coordinates work; A ≻ C ⇒ geometric anchoring
   beats symbolic conditioning. Either is a defensible thesis finding — write both
   interpretations down now.
4. **Read magnitudes, not p-values** (n≈1400 gives p=3e-16 for pure seed noise elsewhere in
   this package), and treat any margin under ~1.5× as provisional — margins that size have
   not survived retrains here.
5. Confirm `horizonCapped` counts are unchanged across arms (a shifted rate means the
   crossing/label fix changed the cohort, not the model).

## Risks / traps

- **The Phase 1 audit is the whole risk.** A missed origin-as-threshold consumer produces
  silently wrong LABELS under arm B (e.g. crossing detection at the airport center), which
  reads as "airport frame is terrible" — a confound, not a result. Hence: audit first,
  behavior-preserving, bit-identical-output test, own commit.
- Arm B's duration/full-loop pathology may interact: flights anchored near the threshold
  lose the "target at origin" cue entirely. Note affected flights (anchor_range small)
  rather than letting them dominate a stratum.
- Do not inherit any conclusion from pre-v5 or pre-refactor artifacts; quote
  current-generation numbers only (standing package rule).
- Multi-airport pooling is NOT solved by this ablation (each airport still gets its own
  frame) — do not oversell arm B as the pooling answer; it only makes the within-airport
  frame runway-independent, which is a prerequisite, not the solution.

## Commit sequence

1. `target_chart` refactor + bit-identity test (no behavior change).
2. `airport-enu` frame mode + resolver + airport-ref plumbing + tests.
3. `target_conditioning` config + input-only channels + checkpoint contract + tests.
4. Runs (each cell's artifacts under its own `run_slug()` dir).
5. Readout doc `docs/2026-09-0X_airport_frame_ablation_results.md`; update package
   `CLAUDE.md`/README only with the durable conclusions; changelog entry.
