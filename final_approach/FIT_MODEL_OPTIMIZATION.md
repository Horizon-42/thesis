# Observed threshold-event estimator optimization

Status: implemented and verdict coupling corrected on 2026-08-13

Scope: the policy-free observed threshold-event producer in
`trajectory_data_process.harvest`; evaluation remains a consumer and never fits
or refits an observed trajectory.

## 1. Objective

Estimate the aircraft navigation reference point at the Landing Threshold Point
(LTP) plane (`along_track = 0`) for an assigned final inbound trajectory.

The event contains geometry and estimator-quality diagnostics only. It must not
contain LPV limits, an approach verdict, or evaluation policy. Evaluation
separately compares the serialized event point estimate with the published-TCH
path.

The estimator must distinguish two physically different cases:

1. **Observed crossing:** the final inbound ADS-B samples bracket the LTP plane.
2. **Unobserved crossing:** reception ends before the LTP plane and the crossing
   must be extrapolated.

The former implementation always used case 2, even when case 1 was available.

The purpose of fitting is therefore narrow: estimate the one physical crossing
event when reception ends before it. Fitting is not an additional aviation
gate, and its residual or population error distribution must not redefine the
LPV/LNAV-VNAV limits later applied by evaluation.

### 1.1 Method pattern checked against published work

The implementation follows the established trajectory-processing pattern,
without claiming that any cited paper defines this project's exact altitude
estimator:

- NASA/TM-20220019263, pp. 8–9, first resolves the landing runway and distance
  to its threshold, then calculates localizer/glideslope deviations over
  contiguous final-approach intervals. This supports runway-frame event
  extraction before performance assessment.
- Olive et al. (2020), §§3.3 and 3.5, identify landing/final-approach events
  using runway context and explicitly document unreliable ground flags,
  successive alignments, go-arounds, and circle-to-land corner cases. This
  supports the selected-final-pass rule rather than an unconstrained search of
  the whole track.
- Waltert and Figuet (2024), §2.3, treat low-altitude ADS-B coverage as a
  missing-event estimation problem. They train only on fully covered landings
  and split complete trajectories into train, validation, and test sets. Their
  target is remaining landing time, not crossing altitude; the transferable
  lesson is the validation design, not their XGBoost model.

Local copies read for this design are:

- [NASA/TM-20220019263](../docs/literature_review/threshold_event_estimation/NASA_TM_20220019263.pdf)
- [Olive et al., 2020](../docs/literature_review/threshold_event_estimation/Olive_et_al_2020_Trajectory_Event_Detection.pdf)
- [Waltert and Figuet, 2024](../docs/literature_review/threshold_event_estimation/Waltert_Figuet_2024_ADSB_Missing_Landing_Time.pdf)

## 2. Evidence motivating the change

The merged five-airport observed set contains 43,951 LPV-evaluable assigned
tracks. Of those, 29,319 have samples on both sides of the LTP plane. The old
`[-5000, -300] m` straight-line fit nevertheless discarded those samples and
extrapolated every crossing.

For the 29,319 bracketed records, direct linear interpolation and the former
fit differed vertically by:

| Statistic | Direct crossing minus former fit |
|---|---:|
| median signed | `+3.69 m` |
| median absolute | `4.81 m` |
| 95th-percentile absolute | `19.03 m` |

The positive displacement is consistent with the physical trajectory beginning
to round out relative to the extension of the earlier glidepath. Consequently,
the former value represented the intersection of an earlier fitted line with the
threshold plane, not necessarily the aircraft's physical crossing.

Some apparent brackets are position jumps. A structural validity screen using
`sample gap <= 5 s` and implied horizontal speed `<= 200 m/s` retains 21,599
crossings. These bounds are data-quality limits, not approach-performance
criteria. They are deliberately above normal transport-category approach speeds
and reject only a temporally sparse or physically implausible interpolation.

## 3. Estimator decision flow

```text
assigned runway + winning final-inbound fit + raw HAE samples
                         |
                         v
search after the winning fit for the first valid LTP bracket
              / yes                       \ no
             v                             v
threshold-plane interpolation      multi-window extrapolation
             |                             |
             +------ policy-free observed_threshold_event ------+
                                                           |
                                                           v
                                            evaluation consumes event
                                            (no fit and no refit)
```

Only samples after the winning assignment fit are searched. This preserves the
selected final inbound pass and prevents an earlier overflight or go-around from
supplying the crossing.

## 4. Direct threshold-plane interpolation

For consecutive runway-frame samples `a` and `b` satisfying
`a.along <= 0 <= b.along`, compute:

```text
fraction = -a.along / (b.along - a.along)
cross    = a.cross  + fraction * (b.cross  - a.cross)
height   = a.height + fraction * (b.height - a.height)
```

The pair is usable only when:

- time is strictly increasing;
- the gap is no more than `5 s`;
- along-track movement is toward and through the threshold; and
- implied horizontal speed is no more than `200 m/s`.

A pair that crosses the threshold plane but fails one of these checks is recorded
in `interpolation_rejections` and skipped. The search continues through later
consecutive pairs on the same selected final pass; only the absence of any valid
later bracket selects extrapolation. This prevents one position spike from hiding
a later physical threshold crossing while keeping the rejected pair auditable.

The event method is `threshold_plane_interpolation`, method version 2, and
`extrapolation_m` is zero. Its source range is exactly the two inclusive source
sample indices.

OpenSky geometric altitude in this data is quantized to 25 ft (`7.62 m`). The
direct vertical 95% half-width is therefore at least half a quantum (`3.81 m`).
The producer also retains the larger uncertainty indicated by the final-segment
fit. This prevents interpolation from manufacturing sub-quantum certainty.

For lateral uncertainty, the producer retains the final-fit statistical margin
and the direct-versus-fit disagreement, whichever is larger. The latter makes a
late lateral change visible in the event uncertainty instead of silently treating
the earlier centreline extension as the measured crossing.

The report must continue to list ADS-B integrity and systematic position error as
unmodelled unless the source supplies suitable integrity metadata.

## 5. Extrapolated crossing

When no valid bracket exists, extrapolation remains necessary. It is produced in
the harvest/final-approach stage, never in evaluation.

### 5.1 Preferred and fallback windows

The preferred fit uses `[-3000, -300] m`. If it does not meet the fit's minimum
sample-count and along-track-span requirements, the producer selects the first
valid wider window in this order: `[-4000, -300] m`, then `[-5000, -300] m`.
Against all 21,599 valid direct crossings, these candidate windows performed as
follows:

| Window | Median signed error | Mean absolute error | 95th absolute error |
|---|---:|---:|---:|
| `[-5000, -300] m` | `-2.53 m` | `5.67 m` | `14.39 m` |
| `[-4000, -300] m` | `-2.37 m` | `5.44 m` | `13.43 m` |
| `[-3000, -300] m` | `-2.32 m` | `5.10 m` | `12.65 m` |
| `[-5000, -500] m` | `-2.66 m` | `6.21 m` | `15.62 m` |

The 3 km window has the lowest measured error while retaining a much longer
baseline than the extrapolation distance. This is a data-driven estimator choice,
not an aviation verdict threshold.

A second check expanded to all `21,873` assigned tracks with a valid direct
crossing, including assignment-only non-LPV runways, and left out one whole
airport at a time. With a median offset fitted only on the other four airports,
the held-out vertical results were:

| Window | Leave-one-airport-out MAE | Leave-one-airport-out 95th absolute error |
|---|---:|---:|
| `[-3000, -300] m` | `4.71 m` | `11.33 m` |
| `[-4000, -300] m` | `5.08 m` | `12.12 m` |
| `[-5000, -300] m` | `5.27 m` | `12.95 m` |

The 3 km candidate was best for every held-out airport. The implementation does
not apply the learned median correction: its residual bias still varies by
airport, and adding a correction without an independent airport/cycle cohort
would overstate generality. The validation is used to choose the window, while
the stored point remains the transparent line intercept.

### 5.2 Window-sensitivity ensemble

The producer also fits every available wider candidate. These candidates do not
vote on the point estimate: the 3 km fit is primary when available, otherwise the
first valid wider window is primary. Their maximum departure from that primary
intercept is serialized as window/model sensitivity.

For each component, the serialized diagnostic margin is:

```text
max(
    largest candidate statistical 95% margin + window sensitivity,
    empirical extrapolation error floor
)
```

The vertical empirical floor is selected for the actual primary estimator and
rounded upward to the next `0.5 m`: `13.0 m` for the 3 km window (`12.65 m`
measured p95), `13.5 m` for 4 km (`13.43 m`), or `14.5 m` for 5 km (`14.39 m`).
The lateral floor is `10.5 m`, rounding upward the observed 95th-percentile
direct-versus-former-fit difference (`10.29 m`). The selected primary window,
measured vertical p95, rounding quantum, applied margins, and calibration
population are serialized in the event. They are estimator-quality parameters,
not LPV limits and not per-flight Gaussian confidence claims.

This cohort was used to set the empirical floors. The leave-one-airport-out
check validates the window ranking, but a later airport/cycle cohort is still
required before the diagnostic margins are described as generally calibrated
beyond the current five-airport data.

Most importantly, these diagnostics do not decide conformance. A nominal
zero-deviation estimate can pass the `±7.5 m` point gate even when its empirical
error diagnostic is wider than `7.5 m`. The report exposes both facts instead
of silently replacing the standard with a tighter data-confidence test.

## 6. Event contract version 2

Every current event carries:

```text
schema_version: observed-threshold-event-v2
status: estimated | unavailable
method: threshold_plane_interpolation | final_segment_window_ensemble
method_version: 2
runway and runway-data fingerprint
threshold crossing latitude, longitude, and HAE altitude
signed cross-track offset
cross-track and altitude sigma retained as evaluation-report diagnostics
explicit 95% uncertainty half-widths and their components
source sample range
fit diagnostics and winning assignment-fit range
interpolation diagnostics, rejected-bracket audit, or candidate-window diagnostics
extrapolation distance
unmodelled uncertainty sources
```

Version 1 events are obsolete derived artifacts. Consumers reject them and direct
the operator to `--reclassify-existing`. Reclassification reads the stored HAE
samples and performs no OpenSky download.

## 7. Evaluation boundary

Evaluation may:

- validate the event, datum, runway fingerprint, and finite values;
- convert the serialized HAE crossing altitude to MSL;
- apply the selected LPV or LNAV/VNAV limits; and
- classify the supplied point estimate against the selected component bounds;
- report the supplied uncertainty interval as a non-gating diagnostic.

Evaluation must not:

- import or call `fit_final_segment()`;
- select ADS-B samples;
- interpolate the threshold crossing;
- alter the producer's uncertainty; or
- replace an unavailable event with a downstream estimate.

A regression test monkeypatches the final-approach fitter to raise and verifies
that evaluation of a stored event still succeeds.

## 8. Acceptance criteria

1. A valid final-inbound bracket produces the interpolated physical crossing,
   not the earlier fit intercept.
2. A rejected threshold bracket remains auditable and does not prevent a later
   valid bracket from being used.
3. A path without a valid bracket uses the preferred 3 km fit when available;
   otherwise it uses the 4 km or 5 km fallback with that window's own calibrated
   vertical floor. It serializes all available candidates, sensitivity,
   calibration floors, and uncertainty composition.
4. Direct vertical uncertainty is never smaller than half the 25 ft altitude
   quantum at 95% confidence.
5. Version 1 events are rejected as obsolete derived artifacts.
6. Arrival/CZML consumers do not draw an inferred tail for a directly observed
   crossing.
7. Evaluation tests prove no observed refitting occurs.
8. Existing raw samples, arrival/model schemas, trajectory identities, LPV
   limits, and lateral verdict formula remain unchanged.
9. A zero-deviation extrapolated event passes the point gate regardless of a
   wider diagnostic interval; a point outside the gate fails even if that
   interval overlaps it.

## 9. Reference audit

All three PDFs above were downloaded from their publishers and read in full,
not inferred from search-result outlines. They are publication artifacts rather
than versioned operational standards, so “newest edition” does not apply. The
normative LPV/LNAV-VNAV sources and their edition/status checks remain in
[`evaluation/FINAL_APPROACH_VERDICT_STANDARD.md`](../evaluation/FINAL_APPROACH_VERDICT_STANDARD.md#11-official-source-audit).

```text
f07a0e9d00e2453e785af0628be85c7e1e83d85ea2cd40ebda4e6dfaf63dac4c  NASA_TM_20220019263.pdf
9f5c2cc44713ad9b1ad390ce9b849febd00011ec03b590189fa3769ee75038c7  Olive_et_al_2020_Trajectory_Event_Detection.pdf
2ac53bcecb009ae8c6386e2f9bcbfac1f0c7b9e0cb8add1dab92533ce88a4f70  Waltert_Figuet_2024_ADSB_Missing_Landing_Time.pdf
```
