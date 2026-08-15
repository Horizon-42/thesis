# Terminal final-approach verdict standard

Status: implemented for the stated U.S. data path. The evaluator applies the
corrected `±7.5 m` LPV threshold rule to the threshold-event point estimate and
uses report schema v3; Section 13 records the acceptance criteria used to
verify the implementation.

Standards checked: 2026-08-13

Applies to: `evaluation/` for observed, optimized, and predicted trajectories,
plus the policy-free observed threshold-event interface

## 1. Decision

The evaluation grades one runway-threshold arrival event. It does not verify
the whole LPV final-approach segment.

The first implementation is for U.S. airports. It uses current FAA runway and
FAS data while applying the ICAO operating limits described below. It does not
add non-U.S. data-source adapters or unrelated approach types.

Two benchmarks are supported:

1. **LPV**, the primary benchmark.
2. **RNP APCH to LNAV/VNAV minima using approved Baro-VNAV**, the only
   fallback.

Pure LNAV is not the fallback because it has no approved vertical guidance
path and therefore no vertical full-scale deflection (FSD). LNAV still has
published altitude restrictions and a minimum descent altitude (MDA), but
those are altitude floors, not a symmetric vertical-deviation gate at the
runway threshold.

The verdict evaluates:

- signed lateral offset from the runway centreline;
- signed vertical offset from the configured desired path;
- whether a valid threshold event and applicable bounds exist.

Estimator uncertainty is retained as diagnostic metadata. It does not tighten
an operational bound or replace a geometric pass/fail result with
`indeterminate`.

For observed ADS-B, the verdict does not fit the trajectory. The data flow is:

```text
raw ADS-B samples
    -> runway assignment and final-segment fitting
    -> policy-free derived observed threshold event
    -> datum conversion and LPV or LNAV/VNAV evaluation
```

`classify_track()` already calls `assign_runway()`, which retains the winning
`SegmentFit` as `Assignment.fit`. The threshold-event producer uses that final
inbound pass to interpolate a measured crossing when available, or performs
the one producer-side extrapolation otherwise. Downstream stages consume the
serialized result and must not call `fit_final_segment()` again.

The current fixed thresholds are withdrawn:

- `106.75 m` is a one-sided FAA LPV lateral FSD floor at the landing threshold
  point (LTP), not the normal tracking limit. ICAO uses one-half FSD.
- `-3.05/+6.10 m` comes from FAA wheel crossing height (WCH) procedure-design
  allowances. WCH is not a vertical tracking tolerance.

The replacement criteria are:

| Benchmark | Effective lateral bound at threshold | Vertical bound |
|---|---:|---:|
| LPV | `min(0.5 × LPV lateral FSD, 0.5 × runway width)` | `±7.5 m` from the published-TCH path, using the DO-229 angular LPV scale with its `15 m` minimum linear FSD |
| LNAV/VNAV with Baro-VNAV | `min(0.15 NM, 0.5 × runway width)` | `±22 m from the Baro-VNAV path` |

No one-third factor is used. One-half LPV FSD comes from current ICAO Doc
9613, Volume II, Part C, Chapter 5, Section B, §5.3.3.1.1.1(b). The close-in
LPV full-scale magnitude is `15 m`; therefore the normal bound is
`0.5 × 15 m = 7.5 m`. One-half runway width is exact
centreline-to-edge geometry.

The lateral bound formula remains unchanged. Both lateral and vertical
components use the same point-estimate classification rule.

## 2. Claim boundary

### 2.1 What the result means

The result is a **standards-informed terminal geometric verdict** for an
observed, optimized, or predicted trajectory.

It answers:

> At the configured runway-threshold event, is the trajectory aligned with
> the selected LPV or LNAV/VNAV benchmark and the runway?

For observed ADS-B, selecting an LPV benchmark does not prove that the crew
was cleared for or flew LPV. It means only that the observed geometry is being
compared with that reference.

### 2.2 What the result does not mean

It does not prove:

- conformance over the entire final approach segment (FAS);
- avionics integrity or alert performance;
- obstacle-surface containment;
- compliance with an operator stabilized-approach policy;
- the approach mode actually selected in the aircraft;
- regulatory compliance after decision altitude (DA); or
- touchdown or landing-gear containment.

At the threshold, the aircraft is normally below DA and continuing visually.
Extending the instrument path to the threshold is useful for this trajectory
study, but it is not a regulatory certification of the visual landing phase.

## 3. Evaluated event

### 3.1 Runway coordinate frame

Transform the terminal state into a runway-aligned local frame:

- `s`: signed along-track displacement from the threshold plane;
- `x`: signed cross-track displacement from the runway centreline;
- `z`: signed vertical displacement from the desired path; and
- `delta_track`: wrapped track-angle difference from runway course.

Use one documented sign convention, for example `s < 0` before the threshold.

Do not call final-to-target great-circle distance “lateral deviation.” That
distance mixes along-track and cross-track error. Report `s` and `x`
separately.

### 3.2 Optimized and predicted trajectories

For a trajectory whose target is the threshold-arrival state:

1. normally evaluate `states[-1]`;
2. interpolate to `s = 0` only when the last segment brackets the threshold
   and the overshoot is due to output discretization;
3. never search earlier states for a more favourable result; and
4. return `event_status = "not_reached"` if the trajectory ends materially
   before the threshold.

Do not extrapolate a computed trajectory to manufacture a pass.

### 3.3 Observed ADS-B trajectories

The last ADS-B point often represents receiver loss rather than the threshold.
The harvest stage therefore selects the runway and physical inbound pass, then
produces the threshold event before evaluation. The production resolver has exactly
two mutually exclusive physical cases.

For adjacent source-timed samples with runway-frame along coordinates
$a_i<0\le a_{i+1}$, define

$$
\alpha=\frac{-a_i}{a_{i+1}-a_i}.
$$

The direct event uses this same $\alpha$ for time, cross-track and HAE height. It
does not run `fit_final_segment()`. If the winning inbound pass ends strictly before
the threshold, the censored event reuses the one robust `[-5000,-300] m` fit that
already won runway assignment. The event producer never runs a second fit or a
multi-window ensemble. A plausible geometric bracket that fails source integrity is
`invalid_support`; it cannot silently fall back to an extrapolated fit.

The bracket displacement check uses the stored track's ADS-B reported ground speed.
The track has already rewritten sample time to `lastposupdate`; state-row `time` is not
used as a second position clock.

The harvested track record serializes `runway-threshold-event-v1`. It contains only
physical measurement/estimation facts:

```text
status                         estimated or unavailable
observability                  within_observed_support, right_censored,
                               invalid_support, or unavailable
method                         direct_linear_bracket, censored_robust_line, or none
runway
threshold_frame_snapshot       physical LTP frame and source cycles only
threshold_frame_fingerprint    canonical physical-frame binding
threshold_crossing_lat and threshold_crossing_lon
threshold_crossing_altitude_m
altitude_datum                 HAE for the current harvest
signed_cross_track_m           right-positive
event_time_s                   interpolated for direct; null for censored
source_sample_range            bracket pair or inclusive winning-fit indices
interpolation_fraction         alpha for direct; null for censored
extrapolation_distance_m       zero for direct; positive for censored
uncertainty                    {status: uncalibrated}
source_integrity and diagnostics
unavailable_reason             only when no event was produced
```

For a direct event:

```text
signed_cross_track_m = (1-alpha) * cross_i + alpha * cross_(i+1)

threshold_crossing_altitude_hae
    = (1-alpha) * altitude_i + alpha * altitude_(i+1)
```

For a censored event, both coordinates are the intercepts of the same already selected
assignment fit. The crossing latitude/longitude is the inverse projection of
`(along=0, signed_cross, height)` in the stored physical frame. Rendering and
evaluation reuse this point without reconstruction. The strict mathematical design,
source-only experiment and implementation limits are in
[`docs/threshold-event-simplified-implementation.zh.md`](../docs/threshold-event-simplified-implementation.zh.md).

The event's physical-frame fingerprint covers threshold position, course, HAE/MSL
elevations, datum offset, physical source identifiers and effective source cycles. It
deliberately excludes runway width, TCH, glidepath and LPV course width: those are
evaluation policy. The report separately hashes the complete evaluation context.
Consumers reject a missing or mismatched physical fingerprint. The operator then runs
`--reclassify-existing`, which recomputes derived assignment/event data from the
stored HAE samples without downloading ADS-B again.

The event must not contain an approach type, FAS profile, LPV/LNAV-VNAV
limits, or any verdict. The raw `samples` array remains raw. `flight_key`
remains the stable record identity and is not redefined inside the event.

Evaluation consumes this event. It returns `indeterminate` when its status is
not `estimated`, required provenance is invalid, or an applicable component
bound is unavailable. It does not select samples, fit lines, or replace the
stored estimate. Numeric estimator uncertainty is not yet calibrated, so the event
does not publish a fabricated confidence interval. A valid point estimate is still
classified against the applicable bound.

Arrival preparation uses the already stored landing-sample index. CZML uses
the stored event and its source range to render any explicitly labelled
inferred threshold tail. Neither stage refits the trajectory.

### 3.4 Optional terminal-window diagnostic

A short pre-threshold diagnostic may later report maximum error over a fixed
physical interval such as `[-D, 0] m`. It must not use “last N samples,” because
sample rates differ.

This diagnostic is out of the first implementation and is non-gating until its
distance and criteria are separately validated.

## 4. LPV benchmark

### 4.1 Lateral rule

Let:

- `F_lat` be the one-sided LPV lateral FSD at the threshold from authoritative
  FAS data;
- `W` be current runway width;
- `B_guidance = 0.5 × F_lat`;
- `B_runway = 0.5 × W`; and
- `B_lat = min(B_guidance, B_runway)`.

For signed cross-track deviation `x`:

```text
pass if abs(x) <= B_lat
fail otherwise
```

The report retains all three bounds so it can explain which one controlled.

FAA Order 8260.58D Formula 3-1-1 sets a minimum threshold course width of
350 ft, converted and rounded to 106.75 m. When that value applies:

```text
ICAO normal LPV bound = 0.5 × 106.75 = 53.375 m
```

That is an example, not a universal fixed value. The actual FAS data controls.

For the current claim, this lateral rule needs no correction. The runway-width
term is a project geometric guard that keeps the evaluated navigation reference
point within the runway edges. It does not prove landing-gear, wing, or whole-
aircraft containment; making that stronger claim would require aircraft
footprint/gear geometry and a separately justified margin.

### 4.2 Vertical rule

LPV has approved angular vertical guidance. The evaluated event is the
trajectory's navigation reference point crossing the LTP runway-threshold
plane. Its signed error is:

```text
z = trajectory altitude at the LTP plane
    - (LTP elevation + published FAS TCH)
```

TCH defines the nominal desired-path altitude. It is not itself an error
tolerance.

The standards chain for the threshold bound is:

1. **RTCA vertical scale.** DO-229D §2.2.4.4.4 defines final-approach
   vertical scaling and the angular full-scale relation
   `alpha_vert,FS = ±0.25 × FAS glidepath angle`. EASA CM-AS-002 Issue 01
   Revision 01 §6.3.2 and its Comment Response Document, page 3, comment 13,
   publicly reproduce that relation.
2. **LPV lower scale limit.** RTCA DO-229D §2.2.5.4.4 applies the LPV
   minimum linear vertical deviation (MLVD), expressed as a one-sided
   full-scale magnitude, of `15 m`. Garmin's current certified-aircraft guide,
   Chapter 2 **Flight Instruments**, page 2-15, **Glidepath - GPS Source**,
   confirms that GPS glidepath FSD is angular with upper and lower limits and
   gives the LPV lower full-scale limit as `±49 ft (15 m)`.
3. **Normal operating fraction.** ICAO Doc 9613, Fifth Edition, Volume II,
   Part C, Chapter 5, Section B, §5.3.3.1.1.1(b), says acceptable LPV flight
   technical error is maintained within one-half of vertical FSD.

The scale is not treated as two optional receiver branches. It remains angular
where the angular value lies between the linear limits; close to the LTP, that
value reaches the LPV `15 m` lower limit. The evaluated event is at the LTP
plane, so the applicable one-sided FSD is `15 m`:

```text
F_vert,threshold = 15 m
B_vert            = 0.5 × 15 m = 7.5 m

pass if -7.5 m <= z <= +7.5 m
fail otherwise
```

The `0.5` factor is not tuned to the dataset. It is the fraction specified by
ICAO. The `15 m` value is not an SBAS alert limit; it is the DO-229 LPV
minimum linear full-scale magnitude.

This rule classifies **LPV threshold-path conformance**. It does not certify a
safe landing. Current EU Air Operations rule CAT.OP.MPA.310 requires an
operator-defined safe threshold-crossing margin but does not supply a universal
metre interval. FAA AC 91-79B §5.2.3 explains that excess threshold height
consumes additional landing distance, so a landing-safety upper limit depends
on aircraft performance and runway available.

### 4.3 Required LPV inputs

An LPV verdict requires:

- authoritative FAS data and effective cycle;
- threshold coordinates and compatible vertical datum;
- runway true course and current width;
- published FAS TCH for the selected runway/procedure;
- the evaluation methodology identifier for the DO-229 angular LPV scale with
  its `15 m` minimum linear limit;
- compatible aircraft/trajectory altitude reference.

The rule is source-independent. Observed, optimized, and predicted trajectories
use the same threshold altitude, datum conversion, signed error, and `±7.5 m`
bound. Available source uncertainty is reported separately and does not alter
that bound or the point-estimate verdict.

Do not substitute:

- WCH or TCH ranges;
- SBAS vertical alert limits such as 35 m or 50 m;
- obstacle-clearance surfaces;
- Baro-VNAV's ±22 m rule; or
- an angular value below the DO-229 `15 m` LPV lower limit; or
- a manufacturer-specific display value presented as a universal landing-
  safety tolerance.

### 4.4 LPV composite result

```text
if computed/predicted event_status == not_reached:
    verdict = fail
elif observed event cannot be estimated:
    verdict = indeterminate
elif lateral == fail or vertical == fail:
    verdict = fail
elif lateral == pass and vertical == pass:
    verdict = pass
else:
    verdict = indeterminate
```

## 5. LNAV/VNAV fallback benchmark

### 5.1 Why this is the fallback

RNP APCH to LNAV/VNAV minima supplies approved vertical guidance. When
Baro-VNAV is used, ICAO Doc 9613 gives a clear vertical-deviation limit.

Pure LNAV is excluded from the composite 3D verdict because it provides no
approved vertical glidepath. Its published altitude restrictions and MDA can
be evaluated separately, but they do not define terminal vertical FSD.

### 5.2 Lateral rule

ICAO Doc 9613 §5.3.4.4.6 specifies normal final-segment cross-track deviation
within ±0.15 NM:

```text
B_guidance = 0.15 NM = 277.8 m
B_runway = 0.5 × runway width
B_lat = min(B_guidance, B_runway)
```

Apply the same signed cross-track point classification used for LPV. The runway
bound will normally control at the threshold.

### 5.3 Vertical rule

ICAO Doc 9613 §5.3.4.4.7 states that, when Baro-VNAV provides vertical-path
guidance during the FAS, deviations above and below the path must not exceed
22 m:

```text
B_vert = 22 m
pass if abs(z) <= 22 m
fail otherwise
```

The evaluation context must explicitly declare approved Baro-VNAV and supply
an authoritative desired-path altitude at the threshold. Do not infer either
fact from trajectory shape, the model target, or the presence of a charted
vertical descent angle.

The current configured non-LPV `Runway` is an assignment frame: it carries
threshold elevation but no published TCH or other authoritative Baro-VNAV
threshold-path altitude. In that case, preserve the fallback without inventing
a vertical reference:

```text
lateral = evaluate normally
vertical = indeterminate
overall = fail if lateral fails, otherwise indeterminate
```

Supplying a validated Baro-VNAV path reference later enables the ±22 m gate;
it must not require re-fitting the trajectory.

### 5.4 Fallback composite result

The event and composite logic are the same as LPV. Both lateral and vertical
components are required.

The report label is:

```text
RNP APCH LNAV/VNAV (Baro-VNAV) terminal geometric verdict
```

## 6. Estimator quality and invalid values

### 6.1 Point verdict and uncertainty status

For signed event estimate `e` and inclusive component bounds `[L, U]`:

```text
pass          if L <= e <= U
fail          otherwise
indeterminate only when e or an applicable bound is unavailable
```

The current observed-event producer has no validated numeric uncertainty model. It
therefore serializes `uncertainty.status = uncalibrated`, and the report emits no
numeric interval. A later producer may publish `e ± 1.96 sigma` only after a separate
calibration establishes what `sigma` means. Such an interval would describe estimator
quality; it would not be a second aviation limit and would not participate in verdicts.

This separation is essential. The official `±7.5 m` LPV value is already the
normal-operation deviation bound derived from one-half FSD. Requiring a noisy
data source's complete confidence interval to fit inside it silently shrinks
the standard and makes a zero-error estimate impossible to pass whenever its
uncertainty exceeds `7.5 m`. Conversely, an out-of-bound point must not become
`indeterminate` merely because a wide interval overlaps the gate.

Observed diagnostics preserve source range, interpolation/extrapolation status, timing,
speed consistency, and censored-fit residual facts. They support future calibration;
they do not currently claim a confidence level or avionics integrity.

### 6.2 Finite validation

All coordinates, times, altitudes, widths, FSD values, bounds, uncertainty
values, and deviations used in a verdict must be finite. Required widths, FSD
values, and uncertainty scales must also have valid signs and ranges.

`NaN` or infinity must never pass through false comparisons or be written as
non-standard JSON. Reject the record or return an explicit invalid result, and
serialize JSON with non-finite output disabled.

## 7. Evaluation-owned context

Keep the physical event and evaluation policy separate:

- the harvested record owns the policy-free observed threshold event described
  in Section 3.3;
- evaluation owns the selected approach benchmark and all verdict limits; and
- raw samples, arrival manifests, flight scenarios, optimizer outputs, and
  prediction contracts do not acquire approach-profile fields.

The evaluator receives an explicit assessment context containing:

- benchmark: `lpv` or `rnp_apch_lnav_vnav_baro`;
- airport/runway identity, true course, width, source, and effective cycle;
- procedure/FAS source and effective cycle;
- LPV lateral FSD when published;
- published LPV TCH and its altitude datum;
- the fixed LPV vertical-scale policy identifier and resolved `15 m` minimum
  FSD / `7.5 m` half-FSD values; and
- explicit approved Baro-VNAV applicability for the fallback; and
- an authoritative Baro-VNAV desired-path altitude when the ±22 m fallback
  gate is available.

The resolved context is copied into the derived evaluation report. The
observed event and its producer provenance are also copied into the report so
the estimate is auditable without refitting. Evaluation first converts the
observed HAE crossing altitude and the benchmark desired-path altitude to one
explicit common datum, then subtracts them:

```text
vertical deviation
    = observed threshold-crossing altitude
    - benchmark desired-path altitude
```

Never subtract HAE ADS-B altitude directly from an MSL/orthometric benchmark.

Subject must be explicit at evaluation time. Do not default an omitted subject
to `optimized`.

## 8. Report requirements

Each report must preserve:

- schema version;
- subject and benchmark;
- runway and procedure identities;
- source documents and effective cycles;
- event method and status;
- the observed event's observability, source sample range,
  interpolation/extrapolation distance, diagnostics, uncertainty status, altitude
  datum, physical-frame snapshot, and schema version;
- signed `s`, `x`, and `z` deviations;
- guidance, runway, effective lateral, and vertical bounds;
- LPV vertical scale model (`do229_lpv_angular_min_clamped`), one-sided minimum
  FSD (`15 m`), ICAO fraction (`0.5`), and resolved bound (`7.5 m`);
- calibrated diagnostic intervals when available, otherwise the explicit
  `uncalibrated` status, always marked as non-verdict metadata;
- lateral, vertical, and composite results; and
- every evaluation parameter that can change a verdict.

An observed event-availability rate must be computed from the source
classification population before assigned-track filtering. Its denominator is
assigned + ambiguous + unassignable arrival candidates; tracks classified as
`not_landing` are outside that population and must be counted separately. The
report must name this denominator and serialize estimated, unavailable, and
excluded counts.

Use `pass`, `fail`, and `indeterminate` for required components. Use
`not_applicable` only for a genuinely non-applicable descriptive component,
not to make an incomplete composite verdict pass.

Reuse the existing `flight_key` and record filename in rows and overlays.
Callsign alone is not a stable identity and may repeat within a batch.

## 9. Separate review fixes

The following correctness fixes are implemented and do not change the aviation
standard:

1. Reject non-finite inputs and JSON output.
2. Compare optimized and observed reference paths over a common physical span.
3. Require an explicit evaluation subject.
4. Serialize all verdict-changing methodology.
5. Preserve existing stable flight identity in overlay selectors.
6. Reject non-positive `--max-tracks`, or define zero as producing no overlays.
7. Bind observed events to the exact runway-data frame and cycle.
8. Compute event availability over source arrival candidates, before filtering.
9. Do not let an indeterminate-component explanation mask a concrete failure.
10. Preserve the three-way verdict in every chart color model.
11. Apply strict JSON-number output to referenced state payloads too.

Reference-path comparison is a descriptive metric, not part of the terminal
verdict. Independently normalizing two paths with different endpoints compares
different physical locations and is invalid.

## 10. Worked examples

### 10.1 LPV lateral example

For a procedure with `F_lat = 106.75 m` and a 45.72 m runway:

```text
guidance half-FSD = 53.375 m
runway half-width = 22.86 m
effective bound   = 22.86 m
```

A terminal cross-track offset of 40 m fails, even though it is inside LPV
half-FSD, because it is outside the runway edge.

### 10.2 LPV vertical threshold example

For a trajectory whose navigation reference point crosses the LTP plane `6 m`
above the published-TCH path:

```text
LPV close-in FSD = 15 m
ICAO fraction     = 0.5
effective bound   = 7.5 m
abs(+6 m) <= 7.5 m → pass
```

An otherwise identical `+8 m` crossing fails. A diagnostic estimator interval
may accompany either result, but it does not widen or shrink `7.5 m`.

For a common `50 ft` published TCH, the geometric pass interval for the
navigation reference point is approximately `25.4 ft` through `74.6 ft` above
the LTP elevation. This conversion is illustrative; evaluation uses metres and
the published TCH/datum for the selected procedure.

### 10.3 LNAV/VNAV vertical example

For Baro-VNAV vertical deviation `z = +10 m`:

```text
abs(10 m) <= 22 m → pass
```

This replaces the invalid conclusion produced by the old `+6.10 m` WCH-based
limit.

Do not silently change an LPV benchmark to LNAV/VNAV. The fallback must be
explicitly selected and supported by applicable procedure data.

## 11. Official source audit

### 11.1 Normative basis

| Source | Currency and passages read | Use | Local copy |
|---|---|---|---|
| ICAO Doc 9613, *Performance-based Navigation (PBN) Manual* | Fifth Edition, 2023. ICAO's 2026 catalogue and Store still identify this edition; ICAO's roadmap plans the next edition for 2027. Volume II, Part C, Ch. 5, Section A §§5.3.4.4.6–8 and Section B §§5.3.3.1.1–5.3.3.3.1.1 were read. The key LPV fraction is in Section B §5.3.3.1.1.1(b). | Authoritative international one-half-FSD rule; RNP APCH lateral rule; Baro-VNAV ±22 m rule; angular LPV scaling basis. | [ICAO Doc 9613](../docs/regulation/ICAO_Doc_9613_5th_Ed_2023.pdf) |
| RTCA DO-229D, *MOPS for GPS/SBAS Airborne Equipment* | §2.2.4.4.4 defines final-approach vertical deviation and `±0.25 × GPA` angular FSD; §2.2.5.4.4 sets the LPV MLVD to `15 m`. | Normative source of the angular scale and its close-in `15 m` one-sided minimum. | No official full-text copy is bundled; RTCA distribution is licence-controlled. |
| RTCA DO-229F | Issued 2020-06-11 and still the newest DO-229 revision in RTCA's store on 2026-08-13. RTCA's official change description says the revision primarily adds requirement tags, edits `shall` wording, and makes identified editorial/clarification updates; it does not identify the LPV vertical-scale geometry as changed. | Currency check against the newest RTCA revision. It is not the certification basis named by current C146e. | No official full-text copy is bundled; see the official product link in §11.2. |
| EASA CM-AS-002 Issue 01 Revision 01, *Clarifications to AMC 20-27* | Issued 2012-10-25 and now superseded by ED Decision 2019/011/R. §6.3.2 was read. | Official public regulator reproduction of DO-229D's `±0.25 × GPA` angular FSD relation. Used as trace evidence, not as a current standalone requirement. | [EASA CM-AS-002](../docs/regulation/EASA_CM-AS-002_Issue-01_Revision-01.pdf) |
| EASA Proposed CM-AS-002 Issue 01 Comment Response Document | Page 3, comment 13 was read. EASA accepted the correction from `±0.25°` to `±0.25 × FAS glidepath angle` and recorded DO-229D §2.2.4.4.4. | Auditable explanation of the angular formula and why `0.25°` is wrong. | [EASA CRD](../docs/regulation/EASA_CM-AS-002_Issue-01_CRD.pdf) |
| Garmin, *AXIS Pilot's Guide for Certified Aircraft*, 190-03123-01 Rev B | Current guide dated July 2026. Chapter 2, **Flight Instruments**, page 2-15, **Glidepath - GPS Source** was read. | Current certified-avionics corroboration that LPV FSD is angular with a lower limit of `±49 ft (15 m)` and upper limit of `±492 ft (150 m)`. It does not supply the ICAO half-FSD fraction and is not presented as a universal landing rule. | [Garmin AXIS guide](../docs/regulation/Garmin_AXIS_Pilots_Guide_190-03123-01_Rev_B_2026.pdf) |
| FAA Order 8260.58D, *United States Standard for PBN Instrument Procedure Design* | Active; issued 2025-01-15. §1-3-1.f(2)(b), §3-1-5.c(3), Figure 3-1-7, and Formula 3-1-1 were read. | Explains WCH/TCH and the FAA 106.75 m lateral FSD floor. It is not the universal operating rule. | [FAA Order 8260.58D](../docs/regulation/Order_8260.58D.pdf) |
| FAA AC 20-138D Change 2, *Airworthiness Approval of Positioning and Navigation Systems* | FAA lists it active. Change 2 dated 2016-04-07. Ch. 4 and §§15-7 through 15-7.8 were read. | Confirms the certified SBAS installation/source chain and that LPV deviation details come from RTCA DO-229. It predates TSO-C146e and is not used alone for the scaling formula. | [FAA AC 20-138D Change 2](../docs/regulation/FAA_AC_20-138D_Change_2.pdf) |
| EASA ETSO-C146e A1 | Applicable from 2020-07-25 and current in EASA's live ETSO register on 2026-08-13. §§1–5 and the relevant appendix introduction were read. | Public official confirmation that current C146e equipment requirements use RTCA DO-229E Section 2. | [EASA ETSO-C146e A1](../docs/regulation/EASA_ETSO-C146e_A1.pdf) |
| Commission Regulation (EU) No 965/2012, consolidated 2026-02-22 | CAT.OP.MPA.310, PDF page 169, was read. | Current commercial-operations cross-check: the operator must ensure a safe threshold-crossing margin, but the rule does not prescribe a universal metre tolerance. | [EU Air Operations](../docs/regulation/EU_Regulation_965-2012_consolidated_2026-02-22.pdf) |
| FAA AC 91-79B, *Aircraft Landing Performance and Runway Excursion Mitigation* | Active; issued 2023-08-28. §5.2.3 was read. | Shows why excess threshold height is a landing-distance problem: each 10 ft above the standard 50 ft TCH adds about 200 ft in the AC's rule of thumb. It does not define an LPV path-error bound. | [FAA AC 91-79B](../docs/regulation/FAA_AC_91-79B_2023.pdf) |
| FAA CIFP Readme, Volume 2608 | Cycle effective 2026-08-06 to 2026-09-03. All seven pages were read. | Confirms Path Point records and current FAA CIFP status. | [FAA CIFP Readme 2608](../data/CIFP/CIFP_260806/CIFP%20Readme%202608.pdf) |
| FAA NASR APT layout, readme, and data archive | Current 2026-08-06 distribution checked. Runway width, true alignment, effective-date, and datum fields were read; `APT_RWY.csv` supplied the configured runway widths. | Authoritative runway source for the implemented U.S. aerodromes. Other States require their authoritative AIP/aerodrome source. | [APT layout](../docs/regulation/FAA_NASR_APT_DATA_LAYOUT_2025-10-23.pdf), [CSV readme](../docs/regulation/FAA_NASR_CSV_README_2026-08-06.pdf), [complete official archive](../docs/regulation/FAA_NASR_APT_CSV_2026-08-06.zip) |

RTCA's official electronic products are sold under licence; RTCA does not
expose a free official full-text download. The repository therefore does not
claim to contain DO-229. The design records the exact RTCA section indices,
uses the official EASA material as the public regulator trace for the angular
formula, and uses a current official avionics manual as independent evidence
for the `15 m` LPV lower scale. Reports must name the angular, minimum-clamped
scale model rather than presenting `15 m` as an independent landing tolerance.

### 11.2 Official status pages

- [ICAO Store: Doc 9613 Fifth Edition](https://store.icao.int/en/performance-based-navigation-pbn-manual-doc-9613)
- [ICAO roadmap: next PBN Manual edition planned for 2027](https://www.icao.int/air-navigation-bureau/gnss-rfi/roadmap/medium-term-actions)
- [FAA Order 8260.58D status](https://www.faa.gov/regulations_policies/orders_notices/index.cfm/go/document.information/documentID/1043458)
- [FAA AC 20-138D status](https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentID/1023966)
- [EASA current ETSO register](https://www.easa.europa.eu/en/domains/aircraft-products/etso/list-of-all-etso)
- [EASA CM-AS-002 archive and supersession status](https://www.easa.europa.eu/en/document-library/product-certification-consultations/easa-cm-002)
- [EU Air Operations consolidated 2026-02-22](https://eur-lex.europa.eu/eli/reg/2012/965/2026-02-22/eng)
- [FAA AC 91-79B active status](https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentID/1042093)
- [Garmin AXIS guide: LPV vertical scale](https://www8.garmin.com/manuals/webhelp/GUID-7317DC85-4516-4684-BBAF-FD7BE84D5E86/EN-US/GUID-72F88211-BD89-4C39-878B-F90C00A32ABB.html)
- [FAA current CIFP downloads](https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/cifp/download/)
- [RTCA DO-229E official electronic product](https://my.rtca.org/productdetails?id=a1B3600000211rIEAQ)
- [RTCA DO-229F official electronic product](https://my.rtca.org/productdetails?id=a1B1R0000092uanUAA)
- [RTCA standards purchase and licensing information](https://www.rtca.org/standards/)

### 11.3 File hashes

```text
d06e3fdd7cc2c24adcd174f997aace1b860eea6c64fbd5410be4ab2ccddae8e4  ICAO_Doc_9613_5th_Ed_2023.pdf
ead3c8e089a88dbde5de0a35a47ad72b9d484a54d408f1b0db15828581f31638  Order_8260.58D.pdf
4b6684234ca4fae293fc128115221f8caa6040e528e00fe92faab818219fa26e  FAA_AC_20-138D_Change_2.pdf
94e02dfc4527b2bf0572f24b462d52e89d493f213f63785805bfc9686ffa4b6d  EASA_ETSO-C146e_A1.pdf
6895d2846b4980f55f8d82a8d1de9f528aaeabe88c3398d661c4325c20051885  EASA_CM-AS-002_Issue-01_Revision-01.pdf
4860f35970c479908dd7e3ddb52283b1a9c4ae94b4ed0e62d6602231e92dee0d  EASA_CM-AS-002_Issue-01_CRD.pdf
09f597acb6095406b5dac5310301979858b295f0077f3780920948a4c96c7c46  Garmin_AXIS_Pilots_Guide_190-03123-01_Rev_B_2026.pdf
43f8848f03bcba832e7afff3ee1ccf996beab3487f8be78556859a4dd869f563  EU_Regulation_965-2012_consolidated_2026-02-22.pdf
ad2d975209916548d091ef81ea541bc7d8fc42aaab3acfbbdfda6f8cc509b0a6  FAA_AC_91-79B_2023.pdf
bb9d8698beae04c2834e7fdfb9e1574a09efb867f222d94e2dcf52c9ab7f05c3  FAA_NASR_APT_DATA_LAYOUT_2025-10-23.pdf
6338a19182d38294cdca37cf4838e643c55ab5a720f10b46b552cae89844bd24  FAA_NASR_CSV_README_2026-08-06.pdf
47023fc1f557594435aaf06cfa0e056abe37c48b3c869a5a316e66d7cf54ba0f  CIFP Readme 2608.pdf
a6ad75ba834fcc423fbc7f7aebb3e9d075ae169da3e9fce5693cd21f2355b6ca  FAACIFP18 (CIFP 260806)
dd9768780197ba3e14d447be0be9cf95e1e55e7c56c8ec4dfecf5dc4f4a10ef1  FAA_NASR_APT_CSV_2026-08-06.zip
```

## 12. Implementation boundary

Implementation of this vertical correction may change:

- `evaluation/` assessment context, authoritative target validation, limits,
  report methodology/schema, and focused tests;
- the frontend evaluation-report type/parser and its focused tests for the new
  derived report schema; and
- regenerated evaluation JSON/HTML artifacts.

Implementation must not change:

- raw ADS-B samples or the in-memory raw trajectory model;
- arrival-manifest, flight-scenario, optimizer, or prediction contracts;
- stable identity generation;
- the current lateral bound formula or its inputs;
- unrelated approach types.

The separately approved threshold-estimator optimization may change only the
policy-free derived observed event and its producer-side fitting. It must not
move fitting into evaluation or add approach policy to trajectory/model data.

The current runway model already carries published CIFP TCH. A pure evaluation-policy
change therefore requires neither re-harvesting nor event reclassification when the
physical-frame fingerprint is current. An event-estimator schema change does require
local `--reclassify-existing`, but never a new OpenSky download. Derived events and
reports have no dual-read compatibility path.

## 13. Acceptance criteria

The corrected implementation is accepted when:

1. LPV resolves a one-sided minimum-clamped FSD of `15 m`, applies ICAO's
   one-half fraction, and classifies threshold error against `[-7.5, +7.5] m`;
2. the LNAV/VNAV fallback requires explicit approved Baro-VNAV context and an
   authoritative threshold-path reference for its ±22 m gate; without the
   latter, lateral remains evaluable while vertical and overall are
   indeterminate;
3. along-track and cross-track error are distinct;
4. a valid bracket produces one direct 3D event without fitting, while a
   right-censored pass reuses the winning assignment fit without a second fit;
5. evaluation, arrival preparation, and CZML do not call
   `fit_final_segment()` for an assigned stored track;
6. non-finite values cannot pass or enter JSON;
7. each observed event is fingerprint-bound to the exact runway frame and
   source cycles, with stale events rejected;
8. all methodology and source cycles are serialized;
9. reference comparisons use a common physical span;
10. existing stable identity appears in overlays;
11. observed availability uses the pre-filter arrival-candidate denominator;
12. focused evaluation tests pass in the `aeroviz` conda environment;
13. observed, optimized, and predicted pipeline entry points produce valid
    reports; and
14. raw trajectories, arrival manifests, scenarios, optimizer outputs, and
    prediction outputs contain no approach profile, limit, or verdict policy;
15. each report records the DO-229 angular/minimum-clamped scale model, `15 m`
    FSD, ICAO `0.5` fraction, resolved `7.5 m` bound, and the source section
    indices from Section 11; and
16. boundary tests cover exact `±7.5 m`, values just inside/outside, non-finite
    values, and prove that missing calibration does not change an otherwise valid
    point verdict; and
17. observed events remain explicitly distinguishable as direct, right-censored,
    invalid support, or unavailable.
