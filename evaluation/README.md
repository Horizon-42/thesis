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
  -> runway assignment + producer-side threshold-event estimation
  -> policy-free observed_threshold_event v2
  -> explicit HAE/MSL conversion
  -> runway/procedure-specific terminal verdict
```

Observed evaluation never imports or calls `fit_final_segment()`. The harvest stage
directly interpolates a valid measured threshold bracket; if none exists, it uses a
3/4/5 km fit-window ensemble and serializes the crossing estimate and effective
uncertainty. Arrival preparation and CZML reuse the stored landing anchor/event as
well. Version-1 derived events must be reclassified from their stored samples. See
[the fit-model optimization design](../final_approach/FIT_MODEL_OPTIMIZATION.md).

Optimized and predicted records use their terminal state and must be at the
threshold plane. Along-track and cross-track values are reported separately;
great-circle endpoint distance is not labelled lateral deviation.

## Bounds

For LPV:

```text
lateral = min(0.5 × published LPV lateral FSD, 0.5 × runway width)
vertical error = threshold-event altitude - (LTP elevation + published TCH)
vertical bound = ±7.5 m
```

The approved design uses the DO-229 angular LPV scale with its `15 m` minimum
linear full-scale deflection, followed by ICAO's required one-half-FSD
normal-operation fraction (`0.5 × 15 m = 7.5 m`). The exact source chapter and
section indices, the scale derivation, and the landing-safety claim boundary
are documented in
[FINAL_APPROACH_VERDICT_STANDARD.md](FINAL_APPROACH_VERDICT_STANDARD.md#42-vertical-rule).

The evaluator resolves this LPV scale itself; trajectories and harvested
threshold events do not carry approach policy. No WCH/TCH range, SBAS alert
limit, or Baro-VNAV tolerance is substituted for the LPV bound.

For the explicit LNAV/VNAV Baro-VNAV fallback:

```text
lateral = min(0.15 NM, 0.5 × runway width)
vertical = ±22 m
```

The fallback is not selected silently. `baro_vnav_approved` must be true in the
evaluation context. The ±22 m gate additionally requires an authoritative
Baro-VNAV threshold-path altitude. The configured non-LPV runway fallback does
not currently publish that reference, so evaluation still reports its lateral
component but marks the vertical component and composite verdict
`indeterminate`; it does not infer a target altitude from the trajectory.

All subjects use the same point-estimate rule:

- `pass`: the signed threshold-event estimate is inside the inclusive bound;
- `fail`: the estimate is outside the bound;
- `indeterminate`: the event or an applicable bound is unavailable.

Observed event-v2 uncertainty and its 95% interval remain in the report as
estimator-quality diagnostics. They do not shrink the aviation bound and do
not change pass/fail into `indeterminate`.

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

`terminal-approach-evaluation-v3` contains:

- complete assessment contexts and resolved bounds;
- the published-TCH vertical reference, DO-229 minimum-clamped scale model,
  `15 m` one-sided FSD, ICAO `0.5` fraction, and resolved `7.5 m` bound;
- event/point-verdict/diagnostic-uncertainty/reference-comparison methodology;
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
