# final_approach

Policy-free runway-frame geometry and the one robust straight-segment fitter used by
the project. The package has no I/O, airport configuration, verdict, or aviation gate.

## Production boundaries

```text
harvest source-valid threshold bracket
  -> direct 3-D interpolation (no fit)

harvest right-censored final pass
  -> assign_runway() -> one winning SegmentFit
  -> runway-threshold-event-v1 reuses that fit
  -> evaluation consumes the stored event (no fit)

arrival modeling
  -> fit_flight_final_approach() uses the same geometry for its model target/tail
```

The last line is an active modeling consumer, not evaluation. Within the observed
harvest/evaluation flow, a censored event is fitted once during runway assignment and
never refitted downstream.

## Modules

- `frame.py` defines the runway-aligned projection. `TrackPoint.alt_m` and
  `RunwayFrame.elevation_m` must use the same vertical datum.
- `fit.py` selects one contiguous, straight inbound suffix and fits signed cross-track
  and height against along-track distance.
- `assign.py` compares candidate runway fits and returns exactly one of `assigned`,
  `ambiguous`, `unassignable`, or `not_landing`.
- `event_contract.py` contains only the shared schema/method discriminators consumed by
  the harvest producer and evaluator.

## Fitter contract

For a runway frame, along-track distance is negative before the landing threshold and
increases toward zero. The default fit window is `[-5000, -300] m`.

The fitter:

1. walks backward to isolate the last contiguous inbound pass;
2. keeps the nearest laterally straight suffix, excluding a base-to-final turn;
3. uses bounded Theil-Sen seeds and MAD residual gates to remove gross source outliers;
4. fits one OLS line for cross-track and one for height over the same retained samples;
5. returns geometry and auditable residual diagnostics only.

It does not publish a calibrated numeric uncertainty. Observed events therefore carry
`uncertainty: {status: uncalibrated}`, and evaluation uses the point estimate against
its explicit component bounds.

## Load-bearing separation

Runway assignment asks a relative question: which runway best matches the final pass?
Evaluation asks an absolute question: does the stored threshold event satisfy the
selected terminal bounds? Putting those bounds into assignment would pre-filter the
population and manufacture a high pass rate.

A direct threshold bracket bypasses fitting because it already observes the requested
physical plane. A plausible bracket that fails source-integrity checks is unavailable;
it is not silently replaced by a fitted crossing.

## Datum and ordering invariants

- Input points are time ordered. The sign of along-track progress distinguishes the two
  ends of one physical runway.
- `course_deg` is a compass bearing: north is 0 degrees and values increase clockwise.
- The harvest uses HAE points and HAE runway elevation. Modeling may use MSL, but it must
  supply an MSL frame as well. This package performs no datum conversion.
- `SegmentFit.first_sample_index`, `last_sample_index`, and rejected indices refer to the
  original input sequence.

The current resolver design and migration evidence are recorded in
[`docs/threshold-event-simplified-implementation.zh.md`](../docs/threshold-event-simplified-implementation.zh.md).
The older [`FIT_MODEL_OPTIMIZATION.md`](FIT_MODEL_OPTIMIZATION.md) is retained strictly
as a superseded design archive; it is not a runtime mode.

## Tests

```bash
conda run -n aeroviz python -m pytest final_approach/tests -q
```
