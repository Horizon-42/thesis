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
[FINAL_APPROACH_VERDICT_STANDARD.md](docs/FINAL_APPROACH_VERDICT_STANDARD.md).

## Event flow

```text
raw ADS-B samples
  -> runway assignment + producer-side threshold-event estimation
  -> policy-free runway-threshold-event-v1
  -> explicit HAE/MSL conversion
  -> runway/procedure-specific terminal verdict
```

Observed evaluation never imports or calls `fit_final_segment()`. The producer has two
mutually exclusive physical cases. A source-valid pair that brackets the threshold
plane is interpolated directly in 3D with one fraction. If the selected inbound pass
ends before the plane, the event reuses the single `[-5000, -300] m` robust fit that
already won runway assignment; it does not fit again. A plausible bracket that fails
source integrity is unavailable and cannot silently fall back to extrapolation.
Arrival preparation and CZML reuse the stored landing anchor/event. See the
[first-principles archive](../docs/threshold-event-first-principles-development.zh.md)
and the [implemented simplified design](../docs/threshold-event-simplified-implementation.zh.md).

Optimized and predicted records use their terminal state and must be at the
threshold plane. Along-track and cross-track values are reported separately;
great-circle endpoint distance is not labelled lateral deviation.

## Bounds

Lateral, under BOTH benchmarks:

```text
lateral = 0.5 × published runway width       # runway_half_width_at_threshold
```

That is the whole lateral rule. It is a LANDING-GEOMETRY claim — did the crossing
lie over the pavement — not a navigation-containment one. The procedure's own
containment (LPV course width 106.75 m, or the RNP APCH LNAV 0.15 NM = 277.8 m
allowance) is 2.3×–18× wider than every runway in this fleet, so the former
`min(guidance, runway/2)` rule never once selected the guidance term; it is now
carried as procedure provenance (`lpv_course_width_m`) and bounds nothing. See
[FINAL_APPROACH_VERDICT_STANDARD.md](FINAL_APPROACH_VERDICT_STANDARD.md#33-lateral-rule).

Vertical, for LPV and approved LNAV/VNAV:

```text
vertical error = threshold-event altitude - (LTP elevation + published TCH)
vertical bound = ±22 m
```

The single project-wide vertical value comes from ICAO Doc 9613, Fifth Edition,
Volume II, Part C, Chapter 5, Section A, §5.3.4.4.7: `+22 m/-22 m` for RNP APCH
Baro-VNAV final-approach deviation. This project uses it consistently as an
RNAV terminal-geometry acceptance bound for LPV and approved LNAV/VNAV. It is
not presented as a universal legal landing-certification limit. The exact
claim boundary, rejected alternatives, and evidence are documented in
[FINAL_APPROACH_VERDICT_STANDARD.md](FINAL_APPROACH_VERDICT_STANDARD.md#1-decision-and-claim-boundary).

The retired `±7.5 m` value was half of the minimum close-in LPV display FSD. It
is a guidance-tracking scale, not a universal threshold-crossing or landing
outcome standard, and is no longer computed or serialized. Trajectories and
harvested threshold events do not carry evaluation policy.

Speed, for every subject (v9):

```text
V_ref,lo(m) = V_min × sqrt(m / MALW)      # the type's PUBLISHED approach speed at MALW
V_ref,hi(m) = V_max × sqrt(m / MALW)      #   (FAA Aircraft Characteristics Database, Oct 2024)
computed : [V_ref,lo(m) × sqrt(max(n, 1)),     V_ref,hi(m) + 20 kt]     # at the crossing mass
observed : [V_ref,lo(m_min) × sqrt(max(n, 1)), V_ref,hi(MALW) + 20 kt]  # over the type's mass range
```

`V_min`/`V_max` are the FSB approach speed at Maximum Allowable Landing Weight (the
dual flap-configuration values where the FAA gives them), `MALW` the weight it is
quoted at, `m_min` the type's lowest published operating mass — all from
`aircraft/reference_speeds.json`, each number with a source id, the documents downloaded
under `data/reference_speeds/` and indexed in `docs/reference_speeds/README.md`. The
+20 kt is the FSF ALAR Briefing Note 7.1 stabilized-approach speed element, an energy
criterion that stays at 1 g. The record's type is `source.dynamics_typecode` (computed)
or `source.aircraft_type` (observed); the mass is the crossing state's own on computed
records (an observed flight's mass is not measured, hence the type-range window); the
load factor is the last control's (`controls[-1].load_factor`) on records with
controls, measured from the flight's own ADS-B kinematics over the final 20 s on
observed baselines, and a declared 1 g on state-output predictions. Computed subjects
are judged on the crossing model airspeed. Observed baselines are judged on the event's
fitted crossing GROUND speed corrected by the field's METAR headwind component
(`data/metar/<ICAO>/`, fetched by `trajectory_data_process/metar/fetch_iem_asos.py`;
`--metar-root`) under its own criterion id — or, when no report within 30 min is
usable, on the raw ground speed as a stated proxy, with the reason on the row. The
verdict is pass/fail against that determinate window; `indeterminate` names its cause
(`speed_indeterminate_reasons` on the batch). Full rationale, worked numbers, the
measured load-factor distribution and trackable sources:
[THRESHOLD_SPEED_GATE.md](docs/THRESHOLD_SPEED_GATE.md).

The LNAV/VNAV Baro-VNAV fallback differs only in the vertical component; its
lateral bound is the same runway half-width. It is selected when the runway has
no LPV Path Point, and its ±22 m gate applies when the runway's own RNAV (GPS)
approach publishes an LNAV/VNAV line of minima together with a runway-leg
vertical path (angle + crossing altitude, decoded from the same CIFP file —
`baro_vnav_approved` in the context is derived from those facts, never from a
caller flag). KRDU 32 (3.50°, 45 ft) and KSMF 35R (3.00°, 64 ft) resolve this way.
A runway with no RNAV procedure at all (KRDU 14) still reports its lateral
component but marks the vertical component and composite verdict
`indeterminate`; evaluation never infers a target altitude from the trajectory.

All subjects use the same point-estimate rule:

- `pass`: the signed threshold-event estimate is inside the inclusive bound;
- `fail`: the estimate is outside the bound;
- `indeterminate`: the event or an applicable bound is unavailable.

Observed event uncertainty is currently marked `uncalibrated`; no numeric 95% interval
is manufactured. This does not shrink or expand the aviation bound and does not change
the point-estimate pass/fail rule.

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
It also rejects a tracks manifest that predates source-timing cleanup. To migrate a
legacy downloaded dataset
without modifying it, use distinct legacy and staging roots:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KRDU \
  --rebuild-fresh-from /path/to/legacy-harvest \
  --output /path/to/new-source-timed-staging
```

Fresh samples use OpenSky `lastposupdate` as their horizontal clock; held state rows and
position-update gaps over 15 seconds do not enter the track. A direct event is
interpolated from its two source samples, while a censored event reuses assignment's
one fit. `geoaltitude` remains HAE. Because OpenSky provides no separate geometric
altitude update timestamp, height changes under a held horizontal position are audited
in the track's `source_integrity` block and are not emitted as additional 3D samples.
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

`terminal-approach-evaluation-v6` contains:

- complete assessment contexts and resolved bounds, tagged with the lateral
  criterion id `runway_half_width_at_threshold`;
- the published-TCH vertical reference, common `±22 m` RNAV terminal bound,
  exact ICAO source location, and non-certification claim boundary;
- the crossing-speed window per record (v9: anchored on the type's published approach
  speed, criterion id `published_vref_at_crossing_mass_and_n_to_vref_plus_20kt`; v6–v8
  carried the stall-anchored `vref_1p23_vs…` ids), its `speed_result`, and the
  `methodology.terminal_speed` source audit;
- event/point-verdict/uncalibrated-uncertainty/reference-comparison methodology;
- three-way verdict counts;
- per-flight signed along-track, cross-track, and vertical deviations;
- component and composite results;
- stable identity and event audit data; and
- reference comparison status and endpoint gaps.

For observed-only reports, `observed.event_estimated_rate` is explicitly
`event_estimated / arrival_candidates_excluding_not_landing`. Assigned,
ambiguous, and unassignable tracks are in that denominator; known
`not_landing` tracks are reported separately as excluded. Source arrival candidates
excluded because no two-point fresh final block remains are counted as unavailable in
`source_integrity_excluded_candidates`. This population is
computed from `tracks/manifest.json` before evaluation-record filtering.

`success` remains a convenience boolean equal to `verdict == "pass"`; consumers
must use `verdict` when distinguishing failure from indeterminate.

## Tests

```bash
conda run -n aeroviz python -m pytest evaluation/tests/ -q
conda run -n aeroviz python -m pytest trajectory_data_process/harvest/tests/ -q
```


## TODO
evaluation/arrival.py 中 依然试图在 final segment 上做拟合；这个需要优化，在final_approach 中直接做插值即可。