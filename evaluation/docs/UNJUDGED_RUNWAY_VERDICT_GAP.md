# "Never judged" and "judged indeterminate" render identically

Status: open finding, not yet fixed — needs a product decision, see "Directions" below.

## Summary

A runway with no CIFP Path Point record (no published vertically guided
procedure — currently KRDU 14/32) never receives a terminal-approach verdict
at all: `trajectory_data_process/harvest/observed.py` skips those tracks
before they ever reach `evaluation.metrics.evaluate_batch`. But the frontend
paints every one of those flights the same neutral grey as a flight that
*was* evaluated and came back genuinely `indeterminate`. A user looking at
the map cannot tell "this runway has no benchmark to judge against" from
"this specific approach's uncertainty interval straddled a gate boundary" —
both look like 100% indeterminate. This was discovered while explaining why
KRDU RW32 renders as fully indeterminate in the Observe view.

Measured impact on the current KRDU harvest: 1617 of 16 056 assigned
landings (1604 on RW32, 13 on RW14, ≈10%) never get a verdict, and are then
displayed as if they did.

## Root-cause chain

1. **No Path Point → no TCH.** `trajectory_data_process/harvest/cifp.py`
   decodes ARINC 424 Path Point records; a runway without one gets
   `threshold_crossing_height_m=None` (`harvest/airports.py:342`). Verified
   on the current CIFP cycle: KRDU 05L/05R/23L/23R all publish TCH
   (15.27–18.11 m); 14 and 32 do not.

2. **Harvest skips the whole track, loudly.**
   `trajectory_data_process/harvest/observed.py:167-172`, inside
   `write_observed_records()`:

   ```python
   runway = airport.runway(row["runway"])
   if runway.threshold_crossing_height_m is None:
       skipped.append(
           SkippedTrack(row["flight_key"], f"runway {runway.ident} publishes no LPV TCH")
       )
       continue
   record = observed_record(track, runway, mass_kg=mass_kg)
   ```

   No `TrajectoryRecord` is ever built for these flights — lateral is not
   computed either. They are recorded in `approach/summary.json["skipped"]`
   with a reason, and are simply absent from `trajectories[]` in
   `evaluation_report.json`. This part is intentional and already documented
   in the module docstring ("a bounded coverage that is stated in the output
   rather than silently shrinking the batch") and in `CLAUDE.md` under
   "Per-runway TCH is published in the CIFP".

   The `evaluation` package itself has no equivalent skip — it is a generic
   judge (`evaluation/metrics.py::evaluate_record`) that will happily accept
   a record for a non-LPV runway and return a real verdict
   (`"indeterminate"` for the vertical component, since
   `AssessmentContext.limits()` — `evaluation/thresholds.py:200-224` — can
   never resolve a vertical bound without a TCH). The skip lives entirely on
   the harvest producer side, not in the judge.

3. **The backend re-introduces these flights and defaults them to
   `"undecided"`.** `aeroviz_backend/observed_trajectories.py` builds the
   CZML flight list from `tracks/manifest.json`'s `outcome == "assigned"`
   rows (`:97-102`) — independent of TCH availability, so RW32's 1604
   flights are back in the list. It then looks up each flight's verdict:

   ```python
   # :108-113 — batch counts
   published = evaluation.by_flight_key.get(key)
   if published is not None:
       matched += 1
   counts[published or "undecided"] += 1

   # :161-166 — per-flight verdict sent to the frontend
   "byFlightId": {
       str(row["flight_key"]): evaluation.by_flight_key.get(
           str(row["flight_key"]), "undecided"
       )
       for row in selected
   },
   ```

   `evaluation.by_flight_key` (`:196-216`) is built only from
   `evaluation_report.json["trajectories"]`, which never contains a RW32/14
   flight key (step 2). The `.get(key, "undecided")` fallback therefore
   fires for every one of them, and produces the exact same string
   (`"undecided"`) that a genuinely-evaluated indeterminate flight gets at
   `:213-214` (`if raw_verdict == "indeterminate": by_flight_key[...] =
   "undecided"`).

4. **The frontend cannot distinguish the two.**
   `aeroviz-4d/src/utils/observedVerdictColors.ts` maps `ObservedVerdict =
   "pass" | "fail" | "undecided"` to colour — there is no fourth state. Its
   own docstring explains the *intended* scope of the grey bucket: "An
   unavailable observed threshold estimate is folded into the same neutral
   colour. It has no defensible crossing to judge and is not itself a flying
   failure." That rationale was written for a flight whose *own fit*
   couldn't produce a threshold-crossing estimate (a property of that one
   approach). It does not obviously cover "this runway has no published
   procedure at all" (a property of the runway, true for every flight that
   ever lands there) — those are different claims, and the current code
   conflates them without that being a documented decision.

## The two consumers of `evaluation_report.json` disagree quietly

`evaluation_report.json["verdict_counts"]` (read directly by
`EvaluationSummary.tsx` via `airportEvaluationReportUrl`) is computed only
over the 14 439 flights that were actually evaluated — RW32/14 are outside
its denominator entirely, so the summary panel's pass/fail/indeterminate
percentages never reflect them.

The same file, read by `aeroviz_backend/observed_trajectories.py` to colour
the map, folds RW32/14 into the `"undecided"` bucket alongside genuinely
judged flights. So the summary panel and the map disagree about what
"indeterminate" means for the same underlying data — one excludes unjudged
runways, the other silently includes them as if judged.

## Directions (not decided here)

- **Add a fourth UI state** (e.g. `"unjudged"` / "no procedure") distinct
  from `"undecided"`, threaded through `ObservedVerdict`,
  `observed_trajectories.py`'s fallback, and
  `observedVerdictColors.ts`/labels/hints. Most faithful to what actually
  happened, but touches the verdict type used across
  `EvaluationSummary.tsx`, `EvaluationReportWindow.tsx`, and their tests.
- **Surface the skip reason instead of defaulting silently.** The reason
  string already exists server-side (`SkippedTrack.reason`,
  `approach/summary.json["skipped"]`) but is never joined into the
  `byFlightId`/CZML response — only the `undecided` fallback survives to the
  frontend. Even without a new colour, exposing *why* (tooltip: "runway 32
  publishes no LPV TCH") instead of implying "evaluated as indeterminate"
  would remove the misleading claim.
- **Leave the colour as-is, but document the merge explicitly** in
  `observedVerdictColors.ts`'s docstring and `FINAL_APPROACH_VERDICT_STANDARD.md`,
  and correct `EvaluationSummary.tsx`'s denominator story so the summary
  panel and the map are at least consistent with each other about what they
  count.

Any of these needs a decision from whoever owns the Observe-view UX before
implementation; this document only records the mechanism and where the two
consumers currently disagree.
