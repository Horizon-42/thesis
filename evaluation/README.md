# Terminal approach evaluation

This package evaluates one runway-threshold event. It does not test the whole
LPV cone and it does not certify that an aircraft actually flew the selected
procedure.

The implemented data path is deliberately narrow:

- U.S. airports backed by the current FAA NASR and CIFP cycles;
- LPV as the primary benchmark;
- RNP APCH LNAV/VNAV with explicitly approved Baro-VNAV as the sole fallback;
- `pass`, `fail`, and `indeterminate` component/composite results.

The normative rationale and source audit are in
[FINAL_APPROACH_VERDICT_STANDARD.md](FINAL_APPROACH_VERDICT_STANDARD.md).

## Event flow

```text
raw ADS-B samples
  -> runway assignment + one final-segment fit
  -> policy-free observed_threshold_event
  -> explicit HAE/MSL conversion
  -> runway/procedure-specific terminal verdict
```

Observed evaluation never calls `fit_final_segment()`. The assignment stage
serializes the crossing estimate, uncertainty, source sample range, fit window,
residual diagnostics, and extrapolation distance. Arrival preparation and CZML
reuse the stored landing anchor/event as well. A legacy derived track without
those fields must be reclassified from its stored samples.

Optimized and predicted records use their terminal state and must be at the
threshold plane. Along-track and cross-track values are reported separately;
great-circle endpoint distance is not labelled lateral deviation.

## Bounds

For LPV:

```text
lateral = min(0.5 × published LPV lateral FSD, 0.5 × runway width)
vertical = 0.5 × validated DO-229 vertical FSD
```

The repository has no licensed, validated DO-229 vertical-scaling
implementation. Therefore LPV vertical is `indeterminate`, and LPV overall is
`indeterminate` unless another required component fails. No WCH, TCH, alert
limit, or Baro-VNAV tolerance is substituted.

For the explicit LNAV/VNAV Baro-VNAV fallback:

```text
lateral = min(0.15 NM, 0.5 × runway width)
vertical = ±22 m
```

The fallback is not selected silently. `baro_vnav_approved` must be true in the
evaluation context.

Observed estimates are classified using their 95% fit interval:

- `pass`: the complete interval is inside the allowed interval;
- `fail`: the two intervals do not overlap;
- `indeterminate`: the intervals overlap a boundary or a required bound/event
  is unavailable.

## Inputs and identity

Every record requires `source.subject` equal to `observed`, `optimized`, or
`predicted`. Producers stamp it explicitly; the evaluator does not guess.

The stable cross-artifact identity is `source.flight_key`. Callsign (`source.id`)
is display text and can repeat. Reports and HTML overlays preserve both the
flight key and record filename.

All required numeric record values must be finite. JSON input rejects `NaN` and
infinity, and every JSON output uses `allow_nan=False`.

## Assessment context

Verdict policy does not live in raw tracks, arrival manifests, scenarios, or
model outputs. The CLI resolves an evaluation-owned `AssessmentContext` from:

- `trajectory_data_process/config/runway_thresholds.json` (FAA NASR runway
  width and effective cycle); and
- `data/CIFP/CIFP_260806/FAACIFP18` (LPV Path Point facts and cycle).

Each report embeds the context and its resolved limits. The current standalone
CLI rejects non-U.S. airports because no authoritative non-FAA data adapter is
implemented.

## CLI

```bash
conda run -n aeroviz python -m evaluation \
  --input <record.json-or-batch-directory> \
  --output evaluation_report.json

conda run -n aeroviz python -m evaluation.visualize \
  --input <record.json-or-batch-directory> \
  --output evaluation_report.html \
  --max-tracks 30
```

A batch directory is read only through `summary.json` and its
`results[].eval_file` roster. `--max-tracks` must be positive.

The harvest pipeline supplies its already loaded airport context directly:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KRDU --evaluate-only
```

`--evaluate-only` never changes assignment output. It rejects an observed event
whose runway-frame fingerprint does not match the active FAA runway/CIFP data.
To rebuild assignment, the threshold estimate, arrivals, evaluation, CZML, and
publication from the already downloaded HAE samples, use:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KRDU --reclassify-existing
```

This mode does not query or download from OpenSky. It writes the complete new
classification into a staging directory and replaces `tracks/` only after every
stored record succeeds. Use `--full-redownload` only when the stored source
samples themselves must be replaced.

## Reference comparisons

Path and flight-time comparisons are descriptive and do not affect the terminal
verdict. They are computed only when both physical start and end positions agree
within 1 m. A mismatched ADS-B tail is marked `skipped`; the package does not
normalize two different physical spans and report the result as path error.

## Report essentials

`terminal-approach-evaluation-v2` contains:

- complete assessment contexts and resolved bounds;
- event/uncertainty/reference-comparison methodology;
- three-way verdict counts;
- per-flight signed along-track, cross-track, and vertical deviations;
- component and composite results;
- stable identity and event audit data; and
- reference comparison status and endpoint gaps.

For observed-only reports, `observed.event_estimated_rate` is explicitly
`event_estimated / arrival_candidates_excluding_not_landing`. Assigned,
ambiguous, and unassignable tracks are in that denominator; known
`not_landing` tracks are reported separately as excluded. This population is
computed from `tracks/manifest.json` before evaluation-record filtering.

`success` remains a convenience boolean equal to `verdict == "pass"`; consumers
must use `verdict` when distinguishing failure from indeterminate.

## Tests

```bash
conda run -n aeroviz python -m pytest evaluation/tests/ -q
conda run -n aeroviz python -m pytest trajectory_data_process/harvest/tests/ -q
```
