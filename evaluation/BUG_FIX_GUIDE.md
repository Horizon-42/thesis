# Evaluation review findings: implemented bug-fix guide

Status: implemented and regression-tested

This guide records the failure mechanism, implemented correction, regeneration
impact, and acceptance test for each review finding. Derived records/reports use
the new contracts directly; there is no dual-read compatibility path.

## 1. Non-finite values could pass positional gates (P1)

### Failure mechanism

Python comparisons with `NaN` are false. A final latitude, longitude, altitude,
or derived deviation containing `NaN` therefore produced no violation and could
be marked successful. `json.dumps()` then emitted non-standard `NaN`, which a
browser JSON parser rejects.

### Implemented correction

- `evaluation.records.record_from_dict()` validates every numeric initial,
  target, state, time, and control field with a path-aware finite check.
- File parsing rejects the JSON tokens `NaN`, `Infinity`, and `-Infinity`.
- `AssessmentContext` validates courses, widths, FSDs, and cycles.
- `evaluation.metrics` validates every derived deviation and uncertainty again
  before classification.
- Report, visualization, observed-record, and harvested-track writers use
  strict JSON (`allow_nan=False`).

### Acceptance test

`evaluation/tests/test_evaluation.py` covers non-finite values in every record
region and literal non-standard JSON. `test_terminal_standard.py` covers the
original final-altitude case.

## 2. Reference paths used different physical spans (P1)

### Failure mechanism

Each complete path was normalized over its own arc length. If an observed ADS-B
tail stopped 325 m before a computed target, equal arc fractions represented
different physical locations. Even coincident centreline paths then reported
roughly 162 m mean and 325 m maximum separation, and their time delta described
different journeys.

### Implemented correction

`reference_span()` measures both start and end gaps. Path and flight-time metrics
are computed only when both are at most 1 m. Otherwise the row records:

- `comparison_status: "skipped"`;
- start/end gaps;
- the 1 m tolerance; and
- a reason.

No cropping or extrapolation was introduced because either would add a second
policy decision and could manufacture an endpoint not present in one source.

### Acceptance test

The regression constructs coincident paths ending about 325 m apart, verifies a
direct comparison raises, and verifies batch evaluation publishes no numerical
path/time comparison.

## 3. Evaluation subject was guessed (P2)

### Failure mechanism

Missing `source.subject` defaulted to `optimized`. TS prediction outputs without
the field were therefore mislabeled, and an observed record could select the
wrong arrival-event semantics.

### Implemented correction

- Subject is mandatory at the record boundary.
- Optimization success/failure producers stamp `optimized`.
- TS outputs stamp `predicted`.
- Observed references and harvest records stamp `observed`.
- Export helpers require a keyword-only subject, preventing silent omission.

### Acceptance test

Missing subject raises. Optimizer and predictor seam tests assert the stamped
value and validate the written batch through the real evaluation reader.

## 4. Verdict-changing methodology was not preserved (P2)

### Failure mechanism

The former observed evaluator refitted the record using caller-overridable fit
windows and established-approach criteria, but reports serialized only the
positional gates. A published result could not be reproduced.

### Implemented correction

Evaluation no longer fits an observed trajectory. Runway assignment serializes
the single policy-free event, including fit window, sample range, residuals,
uncertainty, and extrapolation. Reports preserve that event plus:

- assessment context and resolved bounds;
- authoritative source/effective cycles;
- event method and threshold-plane tolerance;
- 95% interval classification rule and multiplier;
- explicitly unmodelled uncertainty sources; and
- reference endpoint policy.

There are no hidden CLI fit/gate overrides. The report is the complete statement
of the method that changed a verdict.

### Acceptance test

Batch tests assert the schema, cycles, resolved context, methodology, and strict
serialization. Observed rows retain the complete source event.

## 5. Overlay selector discarded stable identity (P2)

### Failure mechanism

`source.id` is a callsign and can repeat. The HTML payload exposed only that
value, so two different flights produced indistinguishable selector entries and
chart titles.

### Implemented correction

Every overlay carries `id`, `flight_key`, record `file`, and a display `label`
combining the callsign with the stable key (or filename if needed). Verdict-to-
CZML joining continues to use `flight_key` only.

### Acceptance test

Two records with the same callsign and different flight keys produce two unique
labels and preserve both keys.

## 6. Non-positive overlay caps embedded every track (P2)

### Failure mechanism

The old sampler returned all items when `count <= 0`. `--max-tracks 0` or a
negative value therefore bypassed the cap and could create an unusably large
HTML file.

### Implemented correction

- argparse accepts only a positive integer;
- `build_payload()` rejects non-positive programmatic values; and
- `_sample_evenly()` rejects them as a final invariant.

### Acceptance test

Zero and negative values raise before track serialization.

## Regeneration and verification

The observed track schema now includes `observed_threshold_event`; evaluation
records require explicit subject; reports use
`terminal-approach-evaluation-v2`. These are derived artifacts and must be
regenerated. A legacy observed track is rejected with a local-reclassification
instruction; it is never silently refitted.

Use local reclassification for legacy or stale derived track records:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KRDU --reclassify-existing
```

This command reuses the stored HAE samples and performs no OpenSky download.

Focused verification:

```bash
conda run -n aeroviz python -m pytest evaluation/tests/ -q
conda run -n aeroviz python -m pytest trajectory_data_process/harvest/tests/ -q
conda run -n aeroviz python -m pytest \
  4dTrajectory/optimization/tests/test_scenario_optimization.py -q

cd aeroviz-4d
npx vitest run \
  src/components/__tests__/EvaluationReportWindow.test.tsx \
  src/components/__tests__/EvaluationSummary.test.tsx \
  src/utils/__tests__/observedVerdictColors.test.ts \
  src/hooks/__tests__/useObservedVerdictColors.test.ts
npm run build
```

## 7. Threshold events could cross runway-data cycles (P1)

### Failure mechanism

An estimated event previously named its runway but did not identify the exact
runway frame used by assignment. `--evaluate-only` could therefore combine an
old cross-track estimate and landing index with a newer threshold position,
course, elevation/datum, runway width, TCH, glidepath, or procedure cycle.

### Implemented correction

- Every estimated event carries a canonical SHA-256 fingerprint of all runway
  facts used to derive or interpret it, including the FAA runway and CIFP
  cycles.
- Arrival and observed-evaluation preparation require that fingerprint to
  match the active `Runway` before consuming the event or landing index.
- Missing and mismatched fingerprints fail with an instruction to run
  `--reclassify-existing`; no legacy fallback or downstream refit exists.
- Reclassification reads only stored HAE samples, stages the entire result,
  and atomically swaps `tracks/` after all records succeed.

### Acceptance test

Tests change only the procedure cycle and verify that the former event is
rejected, then verify local reclassification preserves samples and stamps the
current fingerprint without invoking acquisition.

## 8. Event availability had a selected denominator (P1)

### Failure mechanism

Only assigned, fitted tracks became evaluation records. Computing availability
from that selected set made it 100% by construction and erased ambiguous and
unassignable arrival candidates—the cases for which an event was unavailable.

### Implemented correction

The source manifest now records `event_status` for every classification outcome.
Availability is calculated before evaluation filtering over assigned,
ambiguous, and unassignable tracks. Known `not_landing` tracks are outside the
arrival-candidate population and are reported as `excluded_not_landing`.
Reports serialize the denominator name and all counts, and reject inconsistent
caller-supplied aggregates.

### Acceptance test

A manifest with one assigned, one ambiguous, one unassignable, and one known
non-landing track reports `1/3`, not `1/1`; the excluded count is one.

## 9. LPV vertical unavailability hid lateral failures (P2)

### Failure mechanism

LPV vertical is currently indeterminate. Its missing-scale explanation was
attached even when the lateral component failed, and the UI preferred that
explanation over the actual lateral violation.

### Implemented correction

A composite `fail` now retains its component violations and has no competing
indeterminate-only reason. Explanatory reasons are assembled only for an
overall `indeterminate` verdict, including every indeterminate component.

### Acceptance test

An LPV event outside the runway edge has overall `fail`, a `lateral`
violation, and no vertical-unavailability reason masking it.

## 10. Charts painted indeterminate as failure (P2)

### Failure mechanism

The report row carried a three-way verdict, but the time and WebGL scatter
models reduced it to the `success` boolean. Both `fail` and `indeterminate`
therefore became red.

### Implemented correction

Both chart data models carry `pass | fail | indeterminate` directly. Pass is
green, failure red, and indeterminate neutral gray; legends use the same three
categories.

### Acceptance test

Component tests inspect the SVG time point and the exported WebGL color mapping
and verify that indeterminate differs from both pass and failure.

## 11. Referenced TS state payload allowed non-standard JSON (P2)

### Failure mechanism

Strict evaluation/reference files contained empty inline state arrays and
pointed to a separate state payload. That referenced file was still serialized
with Python's permissive default, allowing `NaN` or infinity into an otherwise
strict batch.

### Implemented correction

The state payload writer now also uses `allow_nan=False`, so the batch write
fails before publishing a non-standard referenced file.

### Acceptance test

A non-finite predicted state causes `write_batch()` to raise instead of writing
a batch that the strict evaluation loader would later reject.
