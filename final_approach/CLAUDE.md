# final_approach — the single final-approach geometry

Runway frame, segment fit, arg-min runway assignment, threshold-event contract.
Pure `geokit` + stdlib: **no I/O, no regulation constants**. Imported by BOTH
`trajectory_data_process/harvest` and `evaluation/arrival.py`.

The split of responsibility is deliberate: **assignment asks *which* runway (relative),
`evaluation/arrival.py` asks *how good* (absolute).**

## Gotchas (recurring, verified)

- **A landing must be assigned to ONE runway, and assignment must be an arg-min over ALL
  thresholds at once, not a per-threshold test plus a guard.** The predecessor classified per threshold and undid double-assignment with a
  pairwise parallel-runway guard; the guard's logic was right and the shipped artifacts *still*
  had **232 of KSJC's 319 unique landings (72.7 %) in two runways' files** (169 in 30L+30R,
  63 in 12L+12R; 32 at KSTL). `final_approach.assign_runway` fits once per candidate and takes
  the arg-min, so two runways are unrepresentable rather than guarded against.
  What discriminates: **median absolute cross-track** separates parallels (13.9 m correct vs
  230.5 m wrong at KSJC, whose separation is 228.4 m — the wrong file's median offset IS the
  separation); distance to threshold does NOT (763 vs 791 m) and is worst at displaced
  thresholds; **direction of travel** is the only thing separating the two ends of one runway
  (they share a centreline).
  Background on the predecessor: `classify_landing_flights` ran per threshold,
  `RUNWAY_THRESHOLD_RADIUS_M` was 1000 m and parallels sit 250–400 m apart on an identical
  heading, so geometry AND heading accepted either — 169 of KSJC 30L's 200 flights were also in
  30R's file (12L∩12R 63, KSTL 30L∩30R 32); KRDU/KSMF/KMSY were unaffected (parallels exceed the
  radius). `sibling_thresholds` then arbitrated, restricted to same-direction runways — the
  opposite end of the same runway must be excluded, because a full rollout stops on top of it.
  Downstream signature of the bug: an observed lateral error whose MEDIAN *is* the parallel
  separation.
- **Fit only the FINAL INBOUND RUN, never an along-track range.** A real arrival occupies the
  same along-track band more than once — downwind leg, vectoring, a go-around, or a track
  exported against the wrong runway end so it holds the approach AND the landing roll. One
  shipped KSJC track ranged over −23.5 km to +18.7 km yet ended at +2.6 km; a range filter mixed
  downwind samples in and produced a **median cross-track of 8.7 km**, which then decided a
  runway assignment. `_final_inbound_run` walks backward from the window's inner edge and stops
  on a reversal; this also subsumes direction (an outbound track yields no fit at all).
- **Assignment must never filter on approach quality.** If the harvest rejected tracks on the
  criterion `evaluation` later reports, every survivor would pass by construction and the
  established rate would be 1.0 — manufactured by the selection, not measured. Hence
  `final_approach` exposes facts only (no `established` flag, no gate constant). Same reason the
  harvest quota counts **assigned landings, not established ones**.
- **The threshold-event schema has exactly one validator: `final_approach.event_contract.validate_event`.**
  The producer (`harvest.threshold_event.require_current_threshold_event`) and the evaluator
  (`evaluation.arrival._observed_arrival`) used to hand-roll ~65 lines each over the same
  payload. Both now call the shared one and keep only the binding they alone can make — producer
  against a `Runway` (frame fingerprint + snapshot), evaluator against an `AssessmentContext`.
  Order in both is **identity first, payload second**, so a stale artifact reports as stale
  ("run --reclassify-existing") instead of complaining about a field that was fine for the data
  it was measured against.
- **A validator that compares two optional fields to each other passes when BOTH are missing.**
  `validate_event` checked `ESTIMATED_OBSERVABILITY_BY_METHOD.get(method) != observability`;
  with neither field present that is `None != None` → False, so an `estimated` event carrying no
  method and no observability validated clean and fell through to the censored branch — graded
  as a real threshold crossing on the strength of two absent fields. Each single-field case
  already failed, which is why it survived review. Fixed by requiring the lookup to RESOLVE
  (`expected is None` → reject). Same family, same pass:
  `record.source.get("target_source", DEFAULT)` returns `None`, not `DEFAULT`, for a key present
  with a null value — an explicit null therefore bypassed the authoritative-threshold
  cross-check that its own docstring promised could not be bypassed. **Use `get(key) or DEFAULT`
  when a null must be read as "unspecified".**
