# Plan — Apply evaluation to observed ADS-B tracks, and publish it to the frontend

Status: **not started** (unblocked as of 2026-07-21). The three data-plane bugs it was
waiting on — HAE/MSL datum, displaced thresholds, parallel-runway assignment — are fixed
and merged. This is the original request: compute the **observed baseline** the optimizer
and the learned predictor are implicitly measured against, and surface it in the UI.

Everything below is grounded in symbols that exist today; file:line anchors are given so a
later session can navigate straight to them. Read this alongside:
- `evaluation/` (the judging package — unchanged by the bug fixes)
- `4dTrajectory/optimization/evaluation_export.py` (the record/roster contract, casadi-free)
- `4dTrajectory/ts_transformer/export.py` (the precedent for a control-free eval batch)
- `aeroviz-4d/src/data/evaluationReport.ts` + `EvaluationReportWindow.tsx` (the UI that already renders the report)

---

## 1. Why — the missing baseline

Every quality number the project reports (optimizer 69.7 % gate pass on KRDU runway,
the ts_transformer gate counts) is scored against the FAA 8260.58D gates in
`evaluation/thresholds.py`. Nobody has ever computed the number those gates give the **real
flown approaches**. Without it, "69.7 %" has no reference frame. This is the
`notes_7_20.md` question "which baseline?" made concrete.

Naively pointing `python -m evaluation` at the `references/` directory gives **1.8 % pass
(18/996 KRDU)** — safe, completed airline landings graded as failures. That number is
wrong, and understanding *why* is the whole design.

---

## 2. The core design problem — an observed track's `states[-1]` is NOT its arrival

`evaluation/metrics.py::final_state_deviation` (metrics.py:54) measures `record.states[-1]`
vs `record.target_state`. That is correct for a **solve**, which terminates at its target by
construction. It is wrong for an **observation**, which is a truncated recording of a flight
that keeps going:

- **966 / 996** KRDU tracks end a **median 325 m short** of the threshold, still airborne
  (ADS-B coverage stops before the flare). So `states[-1]` measures *where coverage stopped*,
  not approach accuracy.
- The honest observed metric fits each flight's **own established final-approach line** and
  extrapolates it to the threshold (along-track = 0). This also self-validates: the fitted
  glidepath comes out **3.02–3.13°** at all five airports — textbook — proving the
  extrapolation recovers real approaches rather than inventing them.

After the datum + displaced-threshold fixes, measured on that extrapolated crossing:

| airport | established rate | lateral median / gate | vertical median / gate |
|---|---|---|---|
| KRDU | 580/996 | 4.8 m / 100 % | +4.3 m / 44 % |
| KMSY | 259/400 | 3.3 m / 100 % | +5.7 m / 50 % |
| KSTL | 401/1054 | 5.2 m / 72 %* | +4.9 m / 31 % |
| KSJC | 184/319 | 7.9 m / 85 %* | +0.3 m / 51 % |
| KSMF | 399/714 | 9.8 m / 100 % | +2.7 m / 65 % |

\* KSTL/KSJC lateral < 100 % is residual parallel-runway contamination — will clear once
those two airports are re-harvested with the bug-③ fix.

**Design consequence:** evaluation needs an explicit *arrival event* abstraction. The two
artifact kinds have genuinely different arrival semantics; do not paper over that with a
special case buried in `final_state_deviation`.

### The `asdb`-category tautology trap

In the `asdb` category the reference target is `target_source: "track_end"` — the target IS
the last observed sample. Grading an observed track against that is 0 m lateral, 0 m
vertical, 100 % pass for every flight — meaningless. **Observed evaluation must use a
runway-threshold target** (the `runway` / `runway_cons` categories,
`target_source: "runway_threshold"`), OR the extrapolated-crossing metric which is
independent of the record's stored target. Prefer the latter — it is what makes the number
comparable to a solve.

---

## 3. Architecture — one subject-kind, not a special case

The record contract (`evaluation/records.py:1-36`) is
`{source, initial_state, target_state, final_time_s, states[], controls[]}`. Three flavours
today, distinguished only by list emptiness:
- **solved**: `states` non-empty, `controls` 1:1 aligned
- **reference / observed**: `states` non-empty, `controls == []`
- **unsolved**: both empty, `final_time_s: null`, plus `reason`

Observed tracks already ARE the reference-shaped record (`reference_evaluation_record`,
evaluation_export.py:70). So "evaluating ADS-B" is mostly about **removing the assumption
that a control-free record isn't a first-class subject**, not adding a parallel pipeline.

**Ambiguity to resolve first (blocking design decision):** `controls == []` is shared by
BOTH observed references AND ts predictions (ts `export.py` writes reference-shaped records
too). So emptiness cannot distinguish "observed" from "predicted". Introduce an explicit
`subject` on the record's `source` (or on the summary roster): `observed | optimized |
predicted`. This is the single honest schema addition, and it is what lets the UI stop
reporting a meaningless `solve_rate` of 1.0 for observed data (every observation trivially
"has states", so the current `TrajectoryRecord.solved = bool(states)` at records.py:71 is
1.0 by construction and must be relabelled, not reported).

---

## 4. Implementation steps

### 4.1 `evaluation/arrival.py` — make the arrival event explicit  (Python, additive)

New module. One function that returns the **arrival deviation** a record should be gated on,
dispatching on subject:

- `optimized` / `predicted` → `states[-1]` vs `target_state` (**existing numbers do not
  move** — this is the current `final_state_deviation`, wrapped).
- `observed` → extrapolated threshold crossing, gated on an **established-on-final**
  precondition. Not established → a first-class outcome `not_established` (analogous to
  `unsolved`), never silently extrapolated.

Established-on-final precondition (validated in the prototype; tune constants once against
the re-harvested data):
- fit position & cross-track vs along-track over the window `along ∈ [-3000, -300] m`
  (or `[-6000, 0]`), require ≥ 8 samples spanning ≥ 500 m
- max |cross-track| ≤ 400 m (on the final-approach corridor)
- track aligned with runway course within ≤ 20°
- fitted glidepath ∈ [2.0°, 4.5°]
- vertical fit residual ≤ 15 m (a clean straight descent)
- arrival = the fitted lines evaluated at along-track = 0

Use `geokit.METRES_PER_DEG_LAT` / `metres_per_deg_lon` for the ENU projection — do **not**
hand-roll `111_320.0` (see the geokit gotcha in CLAUDE.md). Reuse `flight_scenarios` course
convention (math-ENU, ψ = 0 at East).

Keep `evaluation/` dependency-light (geokit + stdlib only; it deliberately never imports the
optimizer — `__init__.py:1-12`). The extrapolation is pure geometry, so this holds.

Wire `arrival.py` into `metrics.evaluate_record` so the gate check
(`thresholds.DeviationThresholds`, thresholds.py:40) runs on the arrival deviation. Solve /
prediction paths must be byte-identical to today — pin that with a regression test on an
existing `*_eval.json`.

### 4.2 Observed-batch writer  (Python, follows the ts precedent)

Observed records + a `summary.json` roster in one directory, so `evaluation.load_records`
(records.py:126, manifest-only — no glob) can read them. `ts_transformer/export.py::write_batch`
is the exact precedent: it already writes reference-shaped `*_eval.json` + a roster via the
shared `evaluation_export.summary_row` (evaluation_export.py:115). Mirror it for observed
tracks. Reuse `reference_evaluation_record`; stamp `subject: "observed"`.

Source of the observed tracks: the same `*_combined_czml_input.json` the scenarios come from,
loaded through `flight_scenarios.load_observed_flights` (so altitudes are MSL — do NOT read
the file directly). Target = the runway threshold via `flight_scenarios.threshold_target_state`.

### 4.3 Publish as a comparison category  (reuse the existing seam)

`aeroviz-4d/python/build_scenario_comparison_czml.py` already publishes
`evaluation_report.json` verbatim per category and upserts `comparison/categories.json`
(`_upsert_category`, with the explicit `constrained: bool`). Add an `observed` category the
same way. Then **the frontend needs no new plumbing** — see §5.

The implemented split now publishes the observed category from
`prepare_scenario_inputs.py`; `run_scenario_optimization.py` separately sweeps
`("fitted-adsb",False),("runway",False),("runway",True)`.

Note on the comparison CZML for observed: the observed track already renders as the white
reference line (deep-copied from `trajectories.czml`, ellipsoidal). The observed *evaluation*
is a report, not a new CZML path — you likely only need the `evaluation_report.json`, not a
new set of `*.czml` entities. Decide whether the observed category ships CZML at all or is
report-only (report-only is simpler and sufficient for the baseline number).

### 4.4 Frontend — subject-aware labelling  (TypeScript, small)

The report already fetches and renders end to end; the only real change is honesty about the
subject:

- `src/data/evaluationReport.ts`: add optional `subject` to the report type; add
  `established` / `not_established` counts alongside `solved` / `unsolved`.
- `src/components/EvaluationReportWindow.tsx`: when `subject === "observed"`, relabel the
  "solve rate" card as **"established rate"** (an observed track is not "solved") and the
  gate columns keep their meaning. Do not show a 1.0 solve rate.
- `src/data/airportData.ts::isComparisonCategory` requires `key/label/dir/constrained` — the
  observed category's manifest entry must supply all four (`constrained: false`).

---

## 5. What the frontend already provides (no new plumbing)

Confirmed present today:
- URL builder `airportEvaluationReportUrl(airport, categoryDir)` — `src/data/airportData.ts:99`
- Types + guard `EvaluationReport` / `isEvaluationReport` — `src/data/evaluationReport.ts`
- Lazy fetch + cache on the Details click — `src/components/OptimizationSummary.tsx:60`
  (handles a missing report distinctly via `isMissingJsonAsset`)
- Renderer (summary cards, aggregate table, deviation charts, per-flight verdict table) —
  `src/components/EvaluationReportWindow.tsx`
- Category discovery `useComparisonCategories` (`idle|loading|ready|empty|error`) —
  `src/hooks/useComparisonCategories.ts:37`

So once an `observed` category with an `evaluation_report.json` is published, the existing
Observe-dock `OptimizationSummary` → `EvaluationReportWindow` flow displays it. The §4.4
change is only relabelling, not new UI.

---

## 6. Data prerequisites (do these first)

The bug fixes made all observed-derived artifacts stale (see the CHANGELOG entry
"three vertical/lateral reference bugs"). Before the numbers mean anything:

1. **Re-harvest KSJC and KSTL** (bug ③ changed runway assignment). Credentials verified
   working from the Mac (`~/Library/Application Support/pyopensky/settings.conf`); offline
   de-dup is NOT enough (would cost KSJC 42 % of its flights). KRDU/KMSY/KSMF need no
   re-harvest.
2. **Regenerate scenarios → references** through the fixed code path
   (`python -m flight_scenarios … --target-from-threshold`, then `write_reference_records`).
   `PROJ_NETWORK=ON` (or a pre-synced EGM96 grid `us_nga_egm96_15.tif`) is required or
   `flight_scenarios/datum.py` raises loudly.
3. `clean_pipeline_data.py` resets the stale artifacts safely (git-tracked-safe; keeps raw
   downloads and `_`-parked dirs by default).

---

## 7. Open decisions to make before coding

1. **Subject marker location** — on each record's `source.subject`, or only on the
   `summary.json` roster? Roster-only is less invasive but means a lone `*_eval.json` can't
   self-describe. Recommend: on `source`, mirrored into the roster row.
2. **Arrival metric** — extrapolated crossing (recommended; comparable to a solve, and
   independent of the stored target) vs the record's stored-target deviation (only valid in
   `runway`/`runway_cons`, tautological in `asdb`).
3. **`not_established` handling** — report as its own bucket (recommended) vs drop silently.
   Never drop silently (project rule: bounded coverage must be stated in output).
4. **Observed category ships CZML?** — report-only is simplest (the white reference line
   already exists in every comparison category). Decide before wiring the builder.
5. **Threshold-crossing height (TCH) per airframe** — the small residual +0.3..+5.7 m
   positive bias is partly real (aircraft cross at/above TCH) and partly per-runway TCH ≠ the
   assumed 15 m. Decide whether to gate observed on the same fixed window or a per-runway TCH.

---

## 8. Acceptance criteria

- `python -m evaluation` (or the observed-batch entry) on the observed records reports an
  **established rate** and a gate pass rate in the 30–65 % range (not 1.8 %, not a
  tautological 100 %).
- Solve and prediction `*_eval.json` numbers are **unchanged** (regression-pinned).
- The `observed` category appears in `categories.json` and its `evaluation_report.json`
  renders in `EvaluationReportWindow` with the "established rate" label, not a 1.0 solve rate.
- `not_established` flights are counted and shown, never dropped.
- Full suite green (`./run_all_tests.sh`, both groups exit 0) + `npx vitest run` + `tsc` +
  `npm run build`.

---

## 9. Prototype artifacts (reference only — not committed to the tree)

The established-on-final metric and the before/after numbers above were produced by
throwaway probes during investigation. If a later session wants to reproduce them, the
method is fully specified in §2 and §4.1; the numbers are pinned in the CHANGELOG entry.
Do not treat the probe scripts as the implementation — `arrival.py` is the real deliverable.
