# Observed threshold-event fitting redesign

Status: implemented as `observed-threshold-event-v4` and regenerated across the
five production airports on 2026-08-14. Version 4 replaces version 3's
state-row-time speed screen with the ADS-B position-clock integrity check below.

Scope: the policy-free observed ADS-B threshold-event producer in
`trajectory_data_process.harvest`. Evaluation consumes the serialized event and
must never fit or refit observed samples.

## 1. Decision

The observed event must be estimated by component:

```text
selected final inbound pass
        |
        +-- lateral position at LTP plane
        |      valid measured position bracket -> direct interpolation
        |      otherwise                       -> final-segment fit
        |
        +-- vertical height at LTP plane
               always                          -> final-segment fit
```

The vertical event fit remains an ordinary straight-line least-squares fit over
the preferred runway-frame window `[-3000, -300] m`, with `[-4000, -300] m`
and `[-5000, -300] m` as availability fallbacks. The runway-assignment fit is
unchanged.

This is intentionally not a more complicated regression. Small experiments on
the newly completed metadata reject Huber, time-parametric, quadratic, freshness-
weighted, and velocity-screened alternatives. The new evidence instead exposes
a data-alignment error in the version-2 design: a threshold position bracket is
not proof that the `geoaltitude` value in those same state-vector rows was updated
at that position.

The direct bracket therefore remains the best available lateral observation but
is no longer used as the observed vertical crossing height.

## 2. Required boundaries

- Raw track samples remain raw HAE observations.
- Runway assignment and threshold-event estimation remain producer-side derived
  processing.
- The event contains geometry, estimator provenance, and quality diagnostics;
  it contains no LPV/LNAV-VNAV policy and no verdict.
- Evaluation validates and applies the selected terminal standard to the stored
  point. It does not select samples, interpolate, or call `fit_final_segment()`.
- Optimized and predicted trajectories are unaffected: their threshold state is
  evaluated directly.
- Changing the evaluation standard never causes ADS-B refitting.
- The backfilled metadata remains a source sidecar. It is not copied into the raw
  coordinate array. No-download reclassification reads the sidecar by exact
  `(icao24, state-row time)` key; a new harvest requests the same source fields
  directly.

## 3. What the new metadata establishes

The completed sidecars under
`trajectory_data_process/outputs/adsb-metadata/<ICAO>/` contain state-vector
`velocity`, `lastposupdate`, and `lastcontact`, plus operational-status GVA where
available.

### 3.1 Alignment and freshness probe

A cross-airport probe of 348 tracks and 14,258 final-fit samples found:

| Check | Result |
|---|---:|
| exact state-row matches | `100%` |
| ambiguous matches | `0` |
| `time - lastposupdate` median | `0.273 s` |
| `time - lastposupdate` p95 | `0.842 s` |
| maximum observed position age | `7.199 s` |
| reported/geometric speed absolute difference median | `3.336 m/s` |
| reported/geometric speed absolute difference p95 | `12.567 m/s` |
| GVA `45 m` | `347 / 348 tracks` |
| GVA `150 m` | `1 / 348 tracks` |

`lastposupdate` supplies horizontal-position freshness. The downloaded source has
no corresponding per-row geometric-altitude update time, so it cannot synchronize
the threshold position and `geoaltitude` after the fact. GVA is a source-integrity
diagnostic, not a correction vector, and the nearly constant `45 m` value cannot
choose among candidate regressors or recover a threshold height.

### 3.2 Regression probe

Using the current directly bracketed crossing as a comparison proxy, not as
ground truth, the 348-track probe produced:

| Candidate vertical estimator | MAE vs direct proxy | p95 absolute difference |
|---|---:|---:|
| straight OLS, `[-3000, -300] m` | `5.390 m` | `13.891 m` |
| position-deduplicated OLS | worse | `15.246 m` |
| Huber robust line | worse | `16.186 m` |
| time-parametric line | worse | `15.281 m` |
| quadratic in along-track | worse | `16.997 m` |

Discarding samples when reported and geometric speed differed by more than
`20 m/s` also worsened the p95 result. Screening whole tracks at `15 m/s`
improved the proxy p95 to `12.360 m` but retained only `47.1%` of tracks. That
is unacceptable availability loss and is not a fitting method.

These results reject added estimator complexity and metadata weighting.

### 3.5 Full threshold-bracket position-jump audit

Version 3 divided the geodesic distance between bracket rows by the difference
between their state-row `time` values. That clock is not the position clock:
OpenSky can publish a new state row while repeating an older position. The full
five-airport audit compared 21,873 accepted direct brackets and 7,721 brackets
rejected by that old screen against `lastposupdate` and reported ground
`velocity`:

| Quantity | Accepted direct brackets | Old speed-screen rejects |
|---|---:|---:|
| state-row gap, median | `1.000 s` | `1.000 s` |
| real position-update gap, median | `1.066 s` | `5.181 s` |
| position-derived speed, median | `68.961 m/s` | `72.259 m/s` |
| reported ground speed, median | `68.943 m/s` | `72.363 m/s` |
| absolute speed disagreement, median | `3.057 m/s` | `1.032 m/s` |
| absolute speed disagreement, p99 | `22.622 m/s` | `21.254 m/s` |

Thus the old rejection is structurally caused by the wrong clock, not by high
aircraft speed. But the rejected set also contains genuine jumps: 40 pairs still
exceed `200 m/s` after division by the real position-update interval, and the
largest disagrees with reported speed by more than `2,100 m/s`.

Version 4 therefore does not remove validation. It requires:

```text
0 < delta(lastposupdate) <= 30 s
0 < each reported ground speed <= 200 m/s
abs(position-derived speed - mean reported speed) <= 25 m/s
0.5 <= position-derived speed / mean reported speed <= 1.5
```

The `25 m/s` limit is the accepted-control p99 (`22.622 m/s`) rounded up;
the `30 s` freshness limit rounds up that control set's maximum (`27.25 s`).
The independent ratio bound prevents a very low reported speed from passing on
the absolute allowance alone. These are empirical source-integrity gates, not
LPV performance limits. Applied to the audit corpus, the combined rule retains
21,702/21,873 old direct brackets and recovers 7,588/7,721 old rejects, while
continuing to reject 304 inconsistent or excessively stale brackets across both
groups.

### 3.3 Threshold-altitude dynamics probe

A second stratified sample contained 904 LPV tracks from KMSY, KRDU, KSJC,
KSMF, and KSTL. Every selected record had a published TCH, a valid direct LTP
position bracket, and a lateral point inside the existing lateral gate.

| Observation | Result |
|---|---:|
| same altitude on both bracket rows | `426 / 904` |
| next altitude change found within 10 samples | `863 / 904` |
| time to next altitude change, median | `2.0 s` |
| first-change absolute rate, p95 | `7.6 m/s` |
| drop of at least `15 m` within `3 s` | `49 / 904` |

The last group is decisive evidence against coupling the direct position and
vertical estimates. Its median direct height error relative to published TCH was
`+8.939 m`; the direct point passed the `±7.5 m` vertical gate only `32.65%` of
the time. The 3 km line fit passed `73.47%` of the time. Closeness to TCH is not
used to choose the estimator; that would bias the measurement toward the
standard. The selection is based on the physically implausible altitude steps.

Representative records show the failure directly:

| Flight | Height at position bracket | First post-bracket height | Time |
|---|---:|---:|---:|
| KMSY UAL1493 | `40.4 m` | `10.0 m` | `1.2 s` after crossing |
| KMSY AAL2208 | `40.4 m` | `25.2 m` | `1.2 s` after crossing |
| KMSY SWA1232 | `48.1 m` | `17.6 m` | `1.8 s` after crossing |

For UAL1493, `lastposupdate` makes the horizontal motion coherent with the
reported speed, while `geoaltitude` drops `30.4 m` in the next state row. This
is an altitude/position update-alignment problem, not a trajectory capable of a
physical `30.4 m/s` descent at the runway threshold.

### 3.4 Window probe

On a separate 894-track cross-airport window cohort, shorter windows increasingly reproduce the
direct proxy because they approach the bracket, but they do not behave more
consistently relative to the independent published TCH reference:

| Window | MAE vs direct proxy | p95 vs direct | MAE vs TCH | p95 vs TCH |
|---|---:|---:|---:|---:|
| `[-3000, -300] m` | `4.849` | `12.588` | `4.532` | `11.895` |
| `[-2000, -200] m` | `4.414` | `11.958` | `4.596` | `11.702` |
| `[-1500, -100] m` | `3.901` | `10.689` | `4.671` | `12.055` |
| `[-1000, -100] m` | `3.693` | `10.054` | `4.946` | `12.961` |
| `[-1000, 0] m` | `2.918` | `7.968` | `4.934` | `12.723` |

The 3 km window has the best overall TCH MAE and avoids fitting the threshold
altitude staircase itself. The 2 km candidate has a slightly smaller overall TCH
p95 but degrades one airport materially and is not consistently better across
airports. There is no evidence for replacing the transparent 3 km primary.

## 4. Version-4 estimator

### 4.1 Final-pass selection

Runway assignment continues to select the final inbound pass and supplies its
winning fit. Threshold processing searches only after that fit. An earlier
overflight, go-around, or reciprocal pass cannot supply the event.

### 4.2 Lateral component

For consecutive runway-frame positions `a` and `b` satisfying
`a.along <= 0 <= b.along`, compute:

```text
fraction = -a.along / (b.along - a.along)
cross    = a.cross + fraction * (b.cross - a.cross)
```

The pair must be strictly inbound and state-row time must increase. Position
integrity is then checked against the real `lastposupdate` interval and the two
ADS-B reported ground-speed values using §3.5. State-row time is never used to
derive motion speed. If metadata is absent, ambiguous, stale, or inconsistent,
the bracket is rejected and the primary final-segment fit supplies the lateral
intercept. The lateral verdict limits themselves are unchanged.

### 4.3 Vertical component

Fit height against runway-frame along-track distance over these windows:

1. `[-3000, -300] m` preferred;
2. `[-4000, -300] m` if the preferred window cannot meet sample/span rules;
3. `[-5000, -300] m` as the final availability fallback.

For the primary fit:

```text
threshold_crossing_altitude_hae
    = runway_threshold_elevation_hae
    + primary_fit.height_at_threshold_m
```

The version-2 direct bracket altitude is retained only as a non-gating
`direct_vertical_proxy` diagnostic when available. It is not the event point.

No TCH value, `±7.5 m` bound, median correction, airport correction, GVA value,
or verdict result enters the fit. This prevents circularly fitting observations
toward the standard that later evaluates them.

### 4.4 Diagnostics and uncertainty

The producer serializes:

- the primary fit and every available candidate fit;
- statistical intercept diagnostics;
- maximum candidate-window sensitivity;
- direct-versus-fit vertical proxy disagreement when a bracket exists;
- the direct bracket's lateral provenance and rejected-pair audit; and
- unmodelled ADS-B source integrity and altitude update alignment.

The former event-v2 vertical “empirical error floor” is retired. It was calibrated
against direct bracket altitude as though that value were truth; the new metadata
and altitude-dynamics probe invalidate that interpretation. Direct-versus-fit
distribution values may be reported as disagreement diagnostics, not as accuracy
bounds.

The event's diagnostic vertical margin is:

```text
largest candidate statistical 95% intercept margin
    + maximum candidate-window sensitivity
```

It is not a complete ADS-B accuracy claim and does not control the aviation
verdict. Evaluation continues to classify the point estimate against the
applicable bound and displays uncertainty separately.

## 5. Event contract v4

Every estimated event carries:

```text
schema_version: observed-threshold-event-v4
method_version: 4
method: direct_lateral_fitted_vertical | final_segment_window_ensemble
component_methods:
  lateral: threshold_plane_interpolation | final_segment_window_ensemble
  vertical: final_segment_window_ensemble
component_source_sample_ranges:
  lateral: inclusive source indices
  vertical: inclusive source indices
runway and exact runway-data fingerprint
threshold crossing latitude, longitude, and HAE altitude
signed cross-track offset
primary vertical fit window and candidate fits
lateral and vertical diagnostic sigmas
component extrapolation distances
fit/interpolation diagnostics, including both clocks, reported speed,
position-derived speed, disagreement, ratio, and applied integrity limits
unmodelled uncertainty sources
```

Earlier events are obsolete derived artifacts. Consumers must reject them and
request `--reclassify-existing`. Reclassification uses the stored raw HAE samples
plus the protected ADS-B metadata sidecar; it does not redownload history. Exact
key matching is mandatory, and conflicting duplicate state rows are treated as
metadata unavailable rather than guessed. The reclassification provenance stores
the sidecar schema, airport, manifest path, and manifest SHA-256.

## 6. Evaluation and rendering boundary

Evaluation may validate the v4 event, convert HAE to MSL using the authoritative
runway datum offset, and apply the selected final-approach standard. It must not
import the fitter or replace either component estimate.

CZML may draw an inferred tail only when lateral position is also fit because the
measured trajectory never reached the threshold. A record with a direct lateral
bracket already contains measured geometry on both sides of the plane; drawing an
extra fit tail would duplicate and mislabel it.

## 7. Acceptance and production verification

1. A synthetic track whose bracket altitude is deliberately wrong still gets its
   lateral position from the bracket and its vertical point from the 3 km fit.
2. A track without a valid bracket gets both components from the fit ensemble.
3. The serialized source ranges identify lateral and vertical inputs separately.
4. Earlier event artifacts are rejected with a reclassification instruction.
5. Evaluation tests prove observed processing consumes the event and never refits.
6. Existing optimized/predicted terminal-state evaluation is unchanged.
7. Focused final-approach, harvest, evaluation, and pipeline-integration tests pass.

All seven criteria are satisfied. The related Python suite reports `157 passed`.
The existing no-download `--reclassify-existing` mode then rebuilt tracks,
arrivals, evaluation, CZML, and frontend publication for every production airport:

| Airport | Assigned v4 events | Direct lateral | Evaluated | Pass | Fail | Indeterminate |
|---|---:|---:|---:|---:|---:|---:|
| KMSY | 4,299 | 4,224 | 4,299 | 3,183 | 1,116 | 0 |
| KRDU | 16,530 | 1,215 | 14,967 | 10,702 | 4,265 | 0 |
| KSJC | 10,945 | 10,901 | 10,945 | 8,608 | 2,337 | 0 |
| KSMF | 4,683 | 4,578 | 4,412 | 3,814 | 598 | 0 |
| KSTL | 9,328 | 8,378 | 9,328 | 7,634 | 1,694 | 0 |
| **Total** | **45,785** | **29,296** | **43,951** | **33,941** | **10,010** | **0** |

The combined point-verdict pass rate is `77.2246%`. The difference between
assigned and evaluated counts is the existing published-TCH eligibility rule,
not an estimator failure. A post-run audit parsed every assigned track and found
only event schema v4, method version 4, an estimated status, and a fitted vertical
component. It also checked all 29,296 direct events against their serialized
integrity limits and found zero violations. The remaining 16,489 events use
fitted lateral: 16,163 have no bracket, 88 exceed the 30-second real position
gap, and 239 bracket attempts disagree with reported speed. One event rejected
an early pair and accepted a later valid pair, so rejection reasons sum to one
more than fitted-lateral events. Local and frontend-published reports are
byte-equivalent as parsed JSON. No OpenSky query was made during regeneration.

## 8. Method references

These publications support the runway-frame, selected-final-pass, and explicit
missing-event treatment. They do not define this project's numerical threshold
height estimator, so no stronger claim is made.

- [NASA/TM-20220019263](../docs/literature_review/threshold_event_estimation/NASA_TM_20220019263.pdf),
  pp. 8–9: resolve landing runway/distance to threshold before calculating
  localizer and glideslope deviations over contiguous final-approach intervals.
- [Olive et al. (2020)](../docs/literature_review/threshold_event_estimation/Olive_et_al_2020_Trajectory_Event_Detection.pdf),
  §§3.3 and 3.5: runway-context landing/final-approach event detection and the
  treatment of successive alignments, go-arounds, circle-to-land cases, and
  unreliable ground flags.
- [Waltert and Figuet (2024)](../docs/literature_review/threshold_event_estimation/Waltert_Figuet_2024_ADSB_Missing_Landing_Time.pdf),
  §2.3: treat low-altitude coverage as missing-event estimation and validate on
  complete landings. Their target is remaining landing time, not threshold
  altitude; only the validation pattern transfers here.

The normative LPV/LNAV-VNAV verdict sources and section-level citations remain in
[`evaluation/FINAL_APPROACH_VERDICT_STANDARD.md`](../evaluation/FINAL_APPROACH_VERDICT_STANDARD.md).
