# Evaluation review findings and bug-fix guide

This guide turns the six review findings against the `evaluation` package into
an implementation and verification plan. It describes the current failure
modes, the safest fixes, the affected producers and consumers, regression
tests, artifact-regeneration requirements, and acceptance criteria.

The findings cover two correctness bugs (P1) and four report-contract or UI
bugs (P2):

| Priority | Finding | Primary risk |
|---|---|---|
| P1 | Non-finite deviations pass the gates | False successes and invalid JSON |
| P1 | Reference paths are compared over different physical spans | Systematically false path and time comparisons |
| P2 | `source.subject` is optional and guessed | Prediction reports are mislabeled as optimized |
| P2 | Established-approach criteria are not serialized | Observed results cannot be reproduced |
| P2 | Overlay entries expose only the callsign | Repeated callsigns are indistinguishable |
| P2 | Non-positive `--max-tracks` values bypass the cap | Unexpectedly huge HTML reports |

## Recommended implementation order

Apply the fixes in this order because the earlier items change the record and
report contracts used by later tests:

1. Make `source.subject` explicit and stamp every producer.
2. Reject non-finite record values, configuration values, and deviations.
3. Make reference comparisons require a common physical span.
4. Serialize established-approach criteria in every report.
5. Preserve stable flight identities in overlay payloads and labels.
6. Reject non-positive overlay caps.
7. Regenerate derived evaluation artifacts and run the focused suites.

Do not add compatibility fallbacks for old evaluation records or reports. They
are derived, regenerable artifacts, and the repository policy requires their
regeneration instead of dual-read behavior.

## 1. P1: reject non-finite values before applying gates

### Current behavior

`evaluation.metrics.evaluate_record()` applies ordinary comparisons:

```python
if deviation.lateral_m > thresholds.lateral_max_m:
    ...
if deviation.vertical_m < -thresholds.vertical_below_max_m:
    ...
if deviation.vertical_m > thresholds.vertical_above_max_m:
    ...
```

IEEE `NaN` is unordered, so every comparison above is false. A record whose
final latitude, longitude, altitude, or derived deviation is `NaN` therefore
gets no violations and is marked successful. The same value then enters the
batch spreads. Python's `json.dumps()` emits `NaN` by default, even though it is
not valid JSON, so a generated report can also fail in a browser or another
strict JSON consumer.

The same failure exists when a CLI gate is `NaN` or infinite. For example,
`--lateral-max-m nan` makes every lateral comparison false.

### Required behavior

- A record containing a non-finite required numeric value is invalid and must
  be rejected with a path-aware `ValueError`; it must never be converted into a
  successful or unsuccessful scientific result.
- Gate and established-approach configuration must be finite and internally
  valid before evaluation starts.
- A non-finite derived `ArrivalDeviation` must be rejected before any gate,
  marginality, or aggregate calculation.
- JSON and HTML serialization must use strict JSON and fail if a non-finite
  value somehow escapes the earlier checks.

### Recommended changes

#### `evaluation/records.py`: enforce the numeric record contract

Add a small finite-number validator based on `math.isfinite(float(value))` and
use it from `record_from_dict()`. Include the field path in failures, for
example:

```text
/path/to/flight_eval.json: states[18].alt must be finite, got nan
```

Validate all of the following, not just the final coordinate that exposed the
bug:

- every `STATE_KEYS` value in `initial_state`;
- every `STATE_KEYS` value in a non-null `target_state`;
- `t` and every `STATE_KEYS` value in every state sample, including interior
  samples;
- non-null `final_time_s`;
- numeric values present in aligned control rows.

Continue enforcing `final_time_s == states[-1].t` after the finite checks. Do
not rely only on subtraction for this invariant because a `NaN` subtraction
also makes the current `> 1e-6` comparison false.

For file input, make `load_record()` reject JSON constants such as `NaN`,
`Infinity`, and `-Infinity` in both the evaluation file and a `states_ref`
source file. A `parse_constant` callback on `json.loads()` gives the earliest
and clearest file-boundary failure. Keep the explicit `record_from_dict()`
checks as well because tests and producers also construct records in memory.

This change can also replace the current boundary-only key check with an
all-sample check while traversing the states. A missing interior key should
raise at the record boundary instead of failing later in a metric.

#### `evaluation/thresholds.py`: validate positional gates

Add `DeviationThresholds.__post_init__()` and require:

- `lateral_max_m`, `vertical_below_max_m`, and `vertical_above_max_m` are
  finite;
- all three magnitudes are non-negative.

Zero may remain legal if the caller intentionally requests an exact gate. A
negative gate has no meaningful interpretation and should raise.

#### `evaluation/arrival.py`: validate established criteria

Add `EstablishedCriteria.__post_init__()` and require:

- both `window_m` values are finite and ordered outer-to-inner;
- `max_cross_track_m` and `max_vertical_rms_m` are finite and non-negative;
- both `glidepath_range_deg` values are finite and ordered low-to-high.

If the intended contract requires the fit window to remain before the
threshold, also require both window values to be negative. Keep this condition
explicit and tested rather than assuming it from the defaults.

#### `evaluation/metrics.py`: add a defense-in-depth deviation check

Immediately after `arrival_deviation()` returns a deviation, verify every
populated numeric field of `ArrivalDeviation` is finite:

- required: `lateral_m`, `vertical_m`, `speed_ms`, `heading_rad`, and
  `flight_time_s`;
- optional when present: both sigma values, `glidepath_deg`, and
  `extrapolation_m`.

Also require non-negative lateral magnitude, sigma values, and extrapolation
distance. This check protects the gates if a future arrival implementation
creates an invalid value from otherwise finite inputs.

Reject the record rather than adding a normal gate violation. A malformed
numeric artifact is not evidence that a trajectory scientifically failed a
gate.

#### Strict serialization

Set `allow_nan=False` on `json.dumps()` in:

- `evaluation/__main__.py` for `evaluation_report.json`;
- `evaluation/visualize.py` for the embedded `const DATA` payload.

The evaluation-record producers should use the same strict setting when
writing new derived records. This is a final safety net, not a substitute for
field-specific validation and error messages.

### Regression tests

Add focused tests in `evaluation/tests/test_evaluation.py` for:

1. `NaN` final latitude, longitude, altitude, time, and target values;
2. an infinite value in an interior state sample;
3. a non-finite control value;
4. a file containing the literal JSON token `NaN`;
5. `DeviationThresholds` constructed with `NaN`, infinity, or a negative
   magnitude;
6. a non-finite `EstablishedCriteria` field;
7. a non-finite derived deviation, using a direct in-memory record or a
   monkeypatched arrival result;
8. strict report and HTML serialization.

Each case must raise; no case may produce `success=True`, a `NaN` aggregate, or
the text `NaN`/`Infinity` in a report.

### Acceptance criteria

- Non-finite input cannot reach a gate comparison.
- Non-finite thresholds and criteria cannot be constructed from CLI flags.
- All successful, failed, and aggregate numeric report values are finite.
- Both JSON output paths are strict-JSON serializable.

## 2. P1: compare references only over a common physical span

### Current behavior

`evaluation.reference.compare_to_reference()` independently resamples the full
record and the full reference at 101 fractions of each path's own horizontal
arc length. That is only a valid shape comparison when both paths describe the
same physical start and end support.

Runway-target and fitted-threshold scenarios violate that assumption. The
computed trajectory normally ends at the target, while the raw ADS-B reference
often ends before it. The repository documents a median missing tail of about
325 m for KRDU. Two coincident straight centerline paths, one ending 325 m
earlier, are therefore paired like this:

```text
fraction       0.00     0.50     1.00
computed       start    midpoint  target
reference      start    earlier   325 m short
reported gap   0 m      ~162 m    ~325 m
```

The mean and maximum are artifacts of comparing different locations, not path
deviation. `record.final_time_s - reference.final_time_s` is likewise not a
journey-time delta when the endpoints differ.

The TS exporter already crops the *start* of a reference to its prediction
anchor, but that does not solve the missing reference tail at the end.

### Required behavior

- Path metrics and flight-time delta must only be emitted when the two records
  cover a common physical span.
- A skipped comparison must retain the raw observed baseline duration and state
  why the comparison was skipped.
- Skipped rows must not enter batch reference aggregates or HTML comparison
  charts.
- The implementation must not silently extrapolate measured ADS-B data or
  invent a timing model.

### Recommended immediate policy: endpoint-aligned or skipped

Use a conservative eligibility check before `compare_to_reference()`:

1. Calculate horizontal start gap between the two first samples.
2. Calculate horizontal end gap between the two last samples.
3. Compare only when both gaps are within a named, documented endpoint
   tolerance.
4. Otherwise retain `reference.file` and `reference.flight_time_s`, omit
   `flight_time_delta_s` and both path-deviation blocks, and add a structured
   skip reason containing the measured gaps.

A small tolerance such as 1 m is appropriate for records that are designed to
copy a shared anchor or target while allowing serialization noise. Define it as
a named constant and test its boundary. Do not use a tolerance large enough to
hide the normal 325 m ADS-B truncation.

Example skipped row block:

```json
{
  "file": "references/flight_reference_eval.json",
  "flight_time_s": 372.4,
  "comparison_status": "skipped",
  "note": "physical-span endpoint mismatch: start 0.0 m, end 324.8 m"
}
```

For an eligible row, use `"comparison_status": "compared"` and include the
existing delta and path blocks. A status is preferable to making consumers
infer meaning from missing keys, but the existing `note` convention may be
kept for the minimum patch if every consumer is updated consistently.

Apply the eligibility check in one shared reference helper, not separately in
`metrics.py` and `visualize.py`. Both report rows and overlay construction need
the same definition of comparable support.

### Why skipping is the safe first fix

There are three possible long-term policies:

| Policy | Benefit | Methodological cost |
|---|---|---|
| Skip endpoint-mismatched metrics | Never fabricates data; fixes false metrics immediately | Reference comparison coverage falls, possibly sharply |
| Crop both paths to a common physical boundary | Retains measured-only comparisons | Requires a defensible spatial-progress coordinate and time interpolation |
| Extrapolate the observed reference to the target | Restores start-to-target metrics | Requires a published geometry, altitude, velocity, and timing extrapolation model |

The first policy is recommended for the bug fix because the other two change
the research methodology. If the thesis later needs comparison coverage for
truncated tracks, select and document one of those policies explicitly and
serialize its parameters in the report. Do not restore the current metrics by
matching equal arc fractions after merely trimming equal *distances*; equal
distance traveled is not necessarily the same physical location when paths
diverge.

The HTML track overlay may still draw the complete record and reference for
visual inspection. It should label them as having different endpoints and must
not present the suppressed numerical comparison as valid.

### Regression tests

Add tests in `evaluation/tests/test_evaluation.py` that cover:

1. the exact counterexample: coincident centerline paths with a shared start
   and endpoints 325 m apart;
2. both path blocks and `flight_time_delta_s` are absent for that row;
3. the raw reference `flight_time_s` and measured endpoint gaps remain visible;
4. the skipped row does not increment `report["reference"]["compared"]` or
   feed aggregates;
5. paths with aligned endpoints retain the existing offset and time-delta
   results;
6. start mismatch is also rejected;
7. gaps exactly inside and outside the named tolerance behave predictably;
8. zero-horizontal-extent paths retain their existing skip behavior.

Add a visualization test ensuring a skipped comparison does not appear in
`refRows`, the delta histogram, or the path-deviation chart.

### Acceptance criteria

- The 325 m shortened-centerline counterexample produces no path or time-delta
  metric, not approximately 162/325 m.
- Every emitted time delta refers to aligned physical endpoints.
- Batch reference aggregates contain only eligible comparisons.
- The report exposes why and by how much an ineligible span differed.

## 3. P2: require an explicit evaluation subject

### Current behavior

`evaluation.arrival.subject_of()` currently uses:

```python
record.source.get("subject", DEFAULT_SUBJECT)
```

The TS prediction exporter in `4dTrajectory/ts_transformer/export.py` adds
predictor metadata but does not add `source.subject`. Predictions are therefore
guessed to be `optimized`. The arrival calculation happens to be the same for
those two subjects today, but the top-level report and HTML metadata are wrong,
and `_row()` omits the row subject when it equals the guessed default.

An empty `controls` list cannot identify a subject: both observed references and
state-output predictions use it.

### Required contract

Every evaluation record must contain exactly one of:

```json
"source": { "subject": "optimized" }
"source": { "subject": "predicted" }
"source": { "subject": "observed" }
```

Missing, non-string, or unknown subjects must raise a path-aware `ValueError`
at `record_from_dict()`. `subject_of()` should validate and return the explicit
value; it must not infer a default.

### Producer updates

Stamp the subject where the producer knows the artifact's meaning:

- optimization solved and failed records from
  `4dTrajectory/optimization/scenario_optimization.py`: `optimized`;
- TS state-output and control-output prediction records from
  `4dTrajectory/ts_transformer/export.py`: `predicted`;
- observed comparison/reference records created through
  `reference_evaluation_record()`: `observed`;
- harvest observed records in
  `trajectory_data_process/harvest/observed.py`: already `observed`; retain and
  test it.

`4dTrajectory/optimization/evaluation_export.py` is shared by optimizer and TS
writers, so its generic `evaluation_record()` must preserve or require the
caller-supplied subject rather than guessing `optimized`. Its
`reference_evaluation_record()` can require an explicitly observed source or
stamp `observed` because that function's contract is specifically an observed
track. Whichever interface is chosen, add assertions at the writer boundary so
a future producer cannot omit the field.

Do not mutate reusable scenario-source dictionaries in place. Copy the source,
stamp the artifact subject on the copy, and write that copy to the record and
states payload.

### Report updates

- Keep the existing batch subject calculation, including `mixed`.
- Serialize `subject` on every trajectory row, including `optimized`; do not
  omit the default-looking value.
- Update `evaluation/README.md`, `records.py`, and `arrival.py` documentation to
  remove all statements about a default.

### Regression tests

- Replace `test_subject_defaults_to_optimized` with a missing-subject rejection
  test.
- Assert all three accepted values dispatch correctly.
- Assert every optimization record writer, including failed-record paths,
  writes `optimized`.
- Assert TS state and control prediction records write `predicted` to both the
  evaluation record and its canonical states payload.
- Assert reference and harvest records write `observed`.
- Assert prediction reports have top-level and row-level `predicted` labels.
- Assert mixed batches remain `mixed` while every row remains explicit.

### Regeneration requirement

All existing `*_eval.json` and `*_reference_eval.json` derived artifacts that
lack `source.subject` must be regenerated by their owning pipeline. Do not add
a read-time migration or filename/control-shape heuristic.

## 4. P2: persist established-approach criteria in reports

### Current behavior

`evaluate_batch()` serializes `DeviationThresholds` but not the
`EstablishedCriteria` passed to the same evaluation. These CLI flags materially
change observed established verdicts:

- `--fit-window-m`;
- `--max-cross-track-m`;
- `--glidepath-range-deg`;
- `--max-vertical-rms-m`.

After a JSON or HTML report is detached from its command line, its established
rate cannot be reproduced or audited.

### Recommended report shape

Add a top-level block beside `thresholds`:

```json
"established_criteria": {
  "window_m": [-5000.0, -300.0],
  "max_cross_track_m": 400.0,
  "glidepath_range_deg": [2.0, 4.5],
  "max_vertical_rms_m": 6.0
}
```

Build it from the exact `criteria` instance used by `evaluate_batch()`. Convert
tuples to JSON arrays so the in-memory report is already a JSON-native schema,
not only serializable by accident.

Because `build_payload()` embeds the `evaluate_batch()` report, the HTML data
will inherit the block automatically. Also render the criteria in the HTML
aggregate/methodology section; embedding invisible data alone is insufficient
for a published report meant to be read without developer tools.

The JSON CLI and HTML CLI already use the same argument constructors in
`evaluation/cli.py`; preserve that single source.

### Regression tests

- Default criteria appear exactly in `evaluate_batch()` output.
- Custom values passed to `evaluate_batch()` are serialized exactly.
- CLI overrides appear in `evaluation_report.json`.
- HTML contains the embedded values and visible methodology labels.
- Criteria validation from finding 1 rejects non-finite or inverted ranges.

### Acceptance criteria

A saved JSON or HTML report contains every value needed to reproduce both the
arrival gates and the established/not-established decisions.

## 5. P2: preserve unique flight identities in overlay payloads

### Current behavior

`evaluation.visualize._track_entry()` writes only:

```json
{ "id": "AFR074", ... }
```

`source.id` is a callsign and is not unique across an observed batch. Although
evaluation report rows already preserve `flight_key` and `file`, overlay
entries discard both. The selector text and plan-view title use `t.id`, making
two flights with the same callsign indistinguishable.

### Recommended payload contract

Carry all three roles explicitly:

```json
{
  "id": "AFR074",
  "flight_key": "AFR074_05L_abc123_20260811T101500Z",
  "file": "AFR074_05L_abc123_20260811T101500Z_eval.json",
  "label": "AFR074_05L_abc123_20260811T101500Z",
  ...
}
```

- `id` remains the human callsign.
- `flight_key` is the preferred stable identity.
- `file` is the record-level fallback for older in-memory fixtures or other
  producers with no flight key.
- `label` is computed once in Python as `flight_key`, otherwise the evaluation
  filename without `_eval.json`, otherwise `id` only when unique.

Use `label` in the selector, plan-view title, and any overlay-specific hover
text. The callsign may be included as a shorter prefix, but the unique suffix
must remain visible. If neither `flight_key` nor `file` exists and duplicate
IDs occur, raise a clear payload-construction error rather than silently
creating ambiguous options.

Keep HTML escaping at every sink as the current code does.

### Regression tests

- Build a payload from two records with the same callsign and different
  `flight_key` values; assert distinct overlay labels.
- Repeat with no flight keys but distinct record filenames; assert the filename
  fallback is distinct.
- Assert selector text and chart-title JavaScript use `label`, not `id`.
- Keep the hostile-string HTML-escaping regression.

### Acceptance criteria

Every selector option maps visibly and deterministically to one evaluation
record, even when callsigns repeat.

## 6. P2: enforce a positive overlay cap

### Current behavior

`_sample_evenly()` currently returns all items when `count <= 0`:

```python
if count <= 0 or len(items) <= count:
    return list(items)
```

Consequently `--max-tracks 0` and negative values disable the cap and embed
every drawable trajectory. On a large observed batch this can create a very
large, unusable HTML file.

### Recommended behavior

Define `max_tracks` as a strictly positive integer:

- make the CLI parser reject zero and negative values with a message such as
  `--max-tracks must be >= 1`;
- make `_sample_evenly()` or `build_payload()` independently raise `ValueError`
  for `count <= 0`, so direct Python callers cannot bypass the invariant;
- retain the existing behavior when the positive cap exceeds the number of
  drawable records.

Rejecting non-positive values is clearer than assigning special zero semantics
because the option is documented as a maximum number of overlays to embed. If
the product later needs a no-overlay mode, add an explicit flag rather than
overloading a broken cap value.

### Regression tests

- `_sample_evenly(items, 0)` and `_sample_evenly(items, -1)` raise.
- CLI invocations with `--max-tracks 0` and `-1` exit through `argparse` with a
  useful error.
- Caps of 1, a middle value, exactly the item count, and more than the item
  count retain their current documented behavior.

### Acceptance criteria

No non-positive cap can cause all trajectories to be embedded.

## Cross-cutting report and documentation updates

After implementation, update the existing contract documentation rather than
leaving this guide as the only source:

- `evaluation/records.py`: explicit subject and finite numeric contract;
- `evaluation/arrival.py`: no subject fallback and validated criteria;
- `evaluation/reference.py`: common-span eligibility and skip semantics;
- `evaluation/metrics.py`: `established_criteria`, row subject, and reference
  status schema;
- `evaluation/visualize.py`: stable overlay labels and positive cap;
- `evaluation/README.md`: regenerate old artifacts, report criteria, and explain
  that reference metrics cover only endpoint-aligned spans.

Also remove or revise stale statements elsewhere that claim all references
already cover the same start-to-target journey. In particular, search for:

```bash
rg -n 'defaulting to|same start.*target|their own.*arc length|path-shape|source.subject' \
  evaluation 4dTrajectory trajectory_data_process docs CLAUDE.md
```

Do not edit unrelated documentation merely because it is found by the search;
only update statements made false by these fixes.

## Derived-artifact regeneration

Subject enforcement intentionally invalidates old derived records that omit
the field. Report-schema changes also make old JSON and HTML reports incomplete.
Regenerate them from preserved source inputs:

```bash
# Observed records and their evaluation/CZML from the existing harvest manifest
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport <AIRPORT> --evaluate-only

# Optimizer records/references and reports (use the original experiment flags)
conda run -n aeroviz python run_scenario_optimization.py \
  --airport <AIRPORT> --target-type runway --outputs eval

# TS prediction records must be re-exported with their original checkpoint,
# split, horizon, and experiment flags. Preserve outer-test isolation rules.
```

For TS data, do not open or rerun outer-test results merely to validate these
development fixes. Use train/validation fixtures and `--split development`.
Only regenerate a final test release when the experiment is frozen and the user
explicitly requests it under the repository's release protocol.

Record the original evaluation CLI flags when regenerating. Once finding 4 is
fixed, those exact criteria will be embedded in new reports.

## Verification commands

Use the repository's `aeroviz` environment for every Python command:

```bash
conda run -n aeroviz python -m pytest evaluation/tests -v

conda run -n aeroviz python -m pytest \
  4dTrajectory/optimization/tests/test_scenario_optimization.py -v

conda run -n aeroviz python -m pytest \
  4dTrajectory/ts_transformer/tests/test_ts_transformer.py -v

conda run -n aeroviz python -m pytest \
  trajectory_data_process/harvest/tests -v
```

Then perform two small end-to-end checks using development fixtures:

```bash
conda run -n aeroviz python -m evaluation \
  --input <development-batch> --output /tmp/evaluation_report.json

conda run -n aeroviz python -m evaluation.visualize \
  --input <development-batch> --output /tmp/evaluation_report.html \
  --max-tracks 5
```

Validate the JSON with a strict parser, inspect the visible criteria in the
HTML, verify repeated callsigns have unique selector labels, and confirm
endpoint-mismatched references show a skip reason without numerical comparison
metrics.

## Final completion checklist

- [ ] Missing or unknown `source.subject` is rejected.
- [ ] Optimized, predicted, observed, failed, and reference producers stamp the
      correct subject.
- [ ] Every trajectory report row carries its subject.
- [ ] Record, criteria, threshold, and deviation numeric values must be finite.
- [ ] JSON and embedded HTML data use strict serialization.
- [ ] Reference metrics are emitted only for common physical spans.
- [ ] Reference skips retain the raw baseline and an auditable reason.
- [ ] `established_criteria` is embedded and visibly rendered.
- [ ] Overlay selectors and titles display a stable unique identity.
- [ ] Non-positive `--max-tracks` values are rejected by CLI and direct APIs.
- [ ] Derived records and reports are regenerated without backward-compatibility
      fallbacks.
- [ ] Focused evaluation, optimization-export, TS-export, and harvest tests pass.

