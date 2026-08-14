# Observed final-approach fit design

Status: **implemented as `observed-threshold-event-v6`; focused tests pass and a
read-only five-airport impact audit is recorded below. Production artifacts still
require `--reclassify-existing`.**

This document describes the one maintained observed-event algorithm. Rejected
experiments are retained here as evidence only; they are not executable modes,
compatibility branches, or evaluation fallbacks.

## 1. Question and boundary

For an observed arrival, the producer estimates one policy-free point:

> the trajectory position at the Landing Threshold Point (LTP) plane.

The point contains a signed cross-track offset and a height above the runway threshold.
It contains no LPV/LNAV-VNAV limit and no pass/fail result.

The processing boundary is:

```text
raw, time-ordered ADS-B track
    -> airport-level landing screen
    -> runway and physical-pass selection
    -> one robust 3D final-segment fit
    -> observed-threshold-event-v6
    -> evaluation applies the selected standard without refitting
```

Optimized and predicted trajectories do not use this ADS-B fitter. Evaluation reads
their terminal state or their final 3D threshold bracket, as documented in
[`evaluation/FINAL_APPROACH_VERDICT_STANDARD.md`](../evaluation/FINAL_APPROACH_VERDICT_STANDARD.md).

## 2. Root defects

The earlier producer had four independent faults.

### 2.1 A plane crossing was mistaken for a landing crossing

The old bracket predicate only required a pair to cross the infinite threshold plane.
It did not require the crossing to remain near the runway in either the lateral or
vertical direction. A later overflight could therefore be combined with an earlier
fit.

### 2.2 The inbound tolerance accumulated without limit

The old backward walk compared adjacent samples. A 100 m tolerance was therefore
renewed at every sample: a gradual reversal smaller than 100 m per sample could travel
kilometres and remain in one supposed inbound pass.

The corrected walk compares each earlier sample with the most negative along-track
coordinate already accepted in the suffix. The 100 m allowance is now a bound on total
retreat, not a renewable per-sample allowance.

### 2.3 Ordinary least squares was dominated by isolated altitude corruption

The fixed cohort contains isolated geometric-altitude values near 983 m and 10,828 m
inside otherwise normal descents. Raw OLS gives these values unbounded leverage and can
move the threshold intercept by hundreds of metres.

### 2.4 The bracket bounded the pass end but not its beginning

The v5 producer truncated fitting after the selected bracket, but an unanchored fit
still searched the entire earlier prefix for its last sample at or inside the `-300 m`
window edge. If reception on the selected later pass resumed at `-100 m`, that pass had
no eligible fit-window sample, so the search silently fell back to a fittable earlier
approach. The event then combined the later bracket and landing identity with the
earlier pass's crossing estimate.

The v6 fitter takes the selected bracket's pre-threshold sample as an explicit pass
anchor. It first walks backward to the latest real along-track reversal, establishing
both ends of that physical inbound run, and searches for fit-window samples only
inside it. A selected pass with insufficient pre-threshold coverage is now reported
`unavailable`; it never borrows another pass.

These faults explain the previously high `indeterminate` rate. They are pass-selection
and source-robustness faults, not evidence that a large fraction of successful
landings violated the final-approach standard.

## 3. Current algorithm

### 3.1 Airport-level landing screen

The existing broad airport screen runs before runway selection. It requires at least
two samples, a closest approach within 1,000 m of a threshold, height within 1,500 m,
and at least 300 m of observed descent. These are deliberately generous structural
checks, not evaluation limits.

### 3.2 Structural threshold brackets

Every consecutive sample pair is projected against every airport runway. A bracket is
eligible only when:

```text
before.along < 0 <= after.along
after.along > before.along
sample time increases
abs(interpolated cross-track) <= 1000 m
abs(interpolated height above threshold) <= 100 m
```

The last two values define a broad finite landing structure. They do not classify
approach quality. The 1,000 m bound is the existing airport landing radius and is much
wider than the runway-edge verdict. The 100 m bound is roughly five times the highest
published threshold-crossing height in the current airport set; a crossing farther
away vertically is not a landing at that threshold.

An eligible pair must also pass the source-integrity checks:

```text
0 < delta(lastposupdate) <= 30 s
0 < each reported ground speed <= 200 m/s
abs(position-derived speed - mean reported speed) <= 25 m/s
0.5 <= position-derived speed / mean reported speed <= 1.5
```

Distance is divided by `delta(lastposupdate)`, not state-row time. OpenSky may repeat an
older position in a newer state row, so using row time creates false speed jumps. The
25 m/s limit rounds the accepted-control p99 of 22.62 m/s upward; the 30 s bound rounds
the control-set maximum of 27.25 s upward. These are data-integrity checks, not aviation
performance limits.

For each runway, the eligible bracket with the smallest absolute cross-track offset is
retained. Recency is only an exact-score tie-break. Across runways, the smallest offset
wins; a margin below 50 m is reported as ambiguous. Selecting the latest airport-wide
bracket is explicitly forbidden because parallel runway thresholds are longitudinally
staggered.

The bracket chooses the runway and ends the selected physical pass. It is not the
published threshold-event estimate.

### 3.3 Bracket-anchored fit

When a bracket wins, its immediately preceding sample is the fit's explicit pass
anchor. The fitter walks backward only through the contiguous inbound run leading to
that anchor, and cannot search either the later tail or an earlier physical pass.

When no source-valid bracket exists, all runway frames use the same robust final-segment
fit and the normal relative assignment rule. Fits outside the same broad landing
structure are unavailable; they are never silently published.

The preferred event window is `[-3000, -300] m`. `[-4000, -300] m` and
`[-5000, -300] m` are availability and sensitivity candidates. The event point comes
from one primary fit:

```text
cross_at_threshold  = fitted cross-track intercept at along = 0
height_at_threshold = fitted height intercept at along = 0
```

Both coordinates use the same retained samples and the same estimator. There is no
`direct_lateral_fitted_vertical` hybrid.

### 3.4 Robust sample selection

Within a continuous inbound window:

1. compute a deterministic Theil-Sen vertical line seed from at most 64 evenly spaced,
   endpoint-preserving samples;
2. calculate vertical residuals;
3. estimate scale with `1.4826 * MAD`, with a minimum scale of
   `7.62 / sqrt(12) m` for the documented 25 ft altitude quantum;
4. reject residuals beyond the fixed 3.03-scale gross-outlier cut;
5. require the original minimum of 8 retained samples and 500 m span; and
6. fit ordinary least-squares cross-track and height lines to the same retained rows.

The published fit remains an auditable OLS line. The robust seed only prevents isolated
source corruption from choosing that line. Rejected original sample indices are
serialized in fit diagnostics. Bounding only the seed prevents quadratic runtime on
dense aggregated receiver rows; the residual check and final OLS still use every
eligible row.

Lag-1 residual autocorrelation continues to reduce the effective sample count used by
the intercept uncertainty. Window sensitivity is also serialized. Evaluation treats
these as diagnostics and does not move or shrink the aviation gate.

## 4. Why direct 3D interpolation was rejected

The sidecar data were sufficient to test, but not approve, a direct vertical bracket.
The candidate propagated the 25 ft altitude intervals with independently reported
vertical rates under both state-row `time` and `lastposupdate` clocks. It used no TCH,
LPV bound, fitted answer, or airport-specific correction.

On 500 fixed direct-bracket records, the source-only acceptance rate was:

| Airport | Accepted |
|---|---:|
| KMSY | 38/100 |
| KRDU | 20/100 |
| KSJC | 90/100 |
| KSMF | 81/100 |
| KSTL | 42/100 |
| **All** | **271/500 (54.2%)** |

The two clock interpretations agreed on eligibility for only 337/500 records. This
failed the pre-registered cross-airport transfer and clock-equivalence criteria. Raw
bracket altitude therefore remains an audit diagnostic; it is not mixed into the event.

## 5. Fixed-cohort validation of the corrected structure

The 2026-08-14 cohort used seed `coherent-3d-threshold-20260814`: 100 previously direct
and 20 fallback records per airport, 600 total. No ML outer-test prediction or result
was opened.

Structural bracket selection found a candidate for 490/500 previously direct records:

- 483 selected the same runway and exact source pair;
- 7 made plausible runway corrections;
- 10 had no structurally valid bracket;
- all 7 corrected pairs passed the independent position/speed checks;
- 173 records had multiple runway candidates, with a minimum winning margin of
  132.5 m, so none crossed the 50 m ambiguity rule; and
- an airport-wide “latest bracket” rule would have changed 88/500 selections, mainly
  because parallel thresholds are offset along track, so that rule was rejected.

The cumulative inbound guard and gross-altitude rejection fixed the isolated failures
without changing the 100/100 fallback control fits in the pre-implementation probe.

Before the selected-pass lower-bound correction, the production `classify_track()` and
v5 producer were
run read-only over the same 600 frozen keys with the real sidecar lookup. They produced:

| Airport | Estimated event | Not landing | Assigned but event unavailable |
|---|---:|---:|---:|
| KMSY | 112/120 | 7 | 1 |
| KRDU | 120/120 | 0 | 0 |
| KSJC | 118/120 | 0 | 2 |
| KSMF | 119/120 | 0 | 1 |
| KSTL | 117/120 | 2 | 1 |
| **All** | **586/600 (97.7%)** | **9** | **5** |

The production run selected a structural bracket for 499/600 records, including some
previously labelled fallback records, and assigned 591/600. This is an availability
result, not a tuned pass-rate target. The discrepancy from the prototype's projected
588/600 is recorded rather than hidden: two additional production records correctly
failed the complete fit-structure or minimum-fit-availability checks.

### 5.1 v6 selected-pass-boundary impact audit

After the v6 correction, every existing estimated bracket event was read from the
stored track and its 5 km assignment fit was recomputed with the bracket-before sample
as the pass anchor. The audit made no writes and performed no network access.

| Airport | Existing estimated bracket events | Fit changed | Becomes unavailable |
|---|---:|---:|---:|
| KMSY | 4,216 | 0 | 0 |
| KRDU | 1,225 | 0 | 0 |
| KSJC | 11,188 | 1 | 1 |
| KSMF | 4,600 | 0 | 0 |
| KSTL | 8,589 | 0 | 0 |
| **All** | **29,818** | **1** | **1** |

The affected record is
`N968RC_30L_ad7b58_20260502T041134Z`: its selected bracket is source pair
`[791, 792]`, while v5 fitted `[604, 684]` from the earlier approach. V6 reports the
event unavailable because the selected later pass has no fittable pre-threshold
window. This audit establishes that the defect is real and that current v5 artifacts
must be regenerated; it is not a claim that future datasets can contain only one such
record.

The fixed input identity is preserved by these manifest hashes:

| Airport | Track manifest SHA-256 | ADS-B metadata manifest SHA-256 |
|---|---|---|
| KMSY | `d4447c1a3f5a7c7a1997d017b30a0dbdca4f7d077cf8d827326f0f5a6c64de11` | `9bb21bf8c91c60d4da39e4881ed109b5ecfe58351cbe85cd0f9d01ea04a393b9` |
| KRDU | `defb565df5db29f9b0f515510017ec1e09aaa8cf2d3a878dd3f985e208171f5f` | `64a4abaea72b92c05305b819a88e75d61cabda249443a60d427499be1ae7515a` |
| KSJC | `832a62f5eba52cc817eb97de2e2b820063e32193a8269161cefc590df0e7e66e` | `35d3a2b5d5228f6e86ae2ee9f1b67f5baef375eb93ad4097a1e98c39d617cd02` |
| KSMF | `571f770a64ff1c73aa0a87b6e3e09a90bbb31c56e1fc6684e7ba17684d2983b9` | `3c022aa953cc7e55295922011e0d4ce318a39594e53bfd0b299f85454fe91004` |
| KSTL | `1d86220f79f27068b485341b917b03aa0471b161cb7b35053ae6d4fcdca57c58` | `7e9c658ec336f63df24068f556a487ad968022740a01eb4853580131b6c135ec` |

## 6. Event contract and consumers

`observed-threshold-event-v6` has one estimator method:

```text
status: estimated | unavailable
method: final_segment_robust_fit
method_version: 6
runway and runway-data fingerprint
source_sample_range
threshold-crossing latitude, longitude, and HAE altitude
signed cross-track offset
fit and uncertainty diagnostics
rejected source-sample indices
optional threshold_bracket audit block
bracket rejection audit
```

The optional bracket block is explicitly labelled
`runway_and_pass_anchor_not_event_estimator`. It records the source pair and integrity
evidence without pretending that its raw altitude is the fitted event.

Evaluation accepts only v6, converts HAE to MSL with the authoritative runway datum,
and applies the standard. Arrival preparation and CZML also consume the stored event;
none of these stages imports or calls the fitter. Older derived events require
`--reclassify-existing`; raw tracks and downloaded ADS-B sidecars are not downloaded or
modified by reclassification.

## 7. Rejected alternatives

The following remain rejected and must not return as maintained modes:

- direct vertical interpolation or a lateral-direct/vertical-fit hybrid;
- generic fit windows allowed to search after the selected bracket;
- adding a raw post-threshold point to OLS;
- Kalman innovation whose acceptance widens with an already-corrupt prior;
- fitting physical-jump fragments that collapse availability;
- longest consistency paths that retain only about 43% of aggregated samples;
- terminal medians that allow a coherent wrong tail to survive; and
- choosing the latest threshold bracket across runways.

## 8. Sources

- OpenSky Network, *Trino — Historical Data*, API 1.4.0, state-vector fields and units:
  <https://openskynetwork.github.io/opensky-api/trino.html>. Checked 2026-08-14.
- Schäfer et al., *Bringing up OpenSky: A Large-scale ADS-B Sensor Network for
  Research*, IPSN 2014, pp. 5 and 7. Local verified copy:
  [`OpenSky_IPSN_2014.pdf`](../docs/literature_review/threshold_event_estimation/OpenSky_IPSN_2014.pdf),
  SHA-256 `bee1f78524a05094e0203c7b45b4fd89c4b9745fcf1254a194830c559c0781f5`.
- Olive et al., *Filtering Techniques for ADS-B Trajectory Preprocessing*, §§2.2–2.5,
  §3.1, §3.4, and §3.5. Local publisher copy:
  [`Olive_et_al_2025_ADSB_Filtering.pdf`](../docs/literature_review/threshold_event_estimation/Olive_et_al_2025_ADSB_Filtering.pdf),
  SHA-256 `4d0412aba68622ef919f12a20dea528200368fcb1d8f27c7d2c862c406a112d0`.
- Olive et al. (2020), §§3.3 and 3.5, runway-context event detection and successive
  passes. Local copy:
  [`Olive_et_al_2020_Trajectory_Event_Detection.pdf`](../docs/literature_review/threshold_event_estimation/Olive_et_al_2020_Trajectory_Event_Detection.pdf).
- NASA/TM-20220019263, pp. 8–9, resolve runway context before calculating localizer
  and glideslope deviations. Local copy:
  [`NASA_TM_20220019263.pdf`](../docs/literature_review/threshold_event_estimation/NASA_TM_20220019263.pdf).
