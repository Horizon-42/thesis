# Terminal final-approach verdict standard

Status: implemented for the stated U.S. data path. LPV vertical is
intentionally indeterminate pending a licensed and validated RTCA
deviation-scaling implementation.

Standards checked: 2026-08-12

Applies to: the observed threshold-event interface and `evaluation/`

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
- whether a valid threshold event exists; and
- uncertainty in the measured or derived values.

For observed ADS-B, the verdict does not fit the trajectory. The data flow is:

```text
raw ADS-B samples
    -> runway assignment and final-segment fitting
    -> policy-free derived observed threshold event
    -> datum conversion and LPV or LNAV/VNAV evaluation
```

`classify_track()` already calls `assign_runway()`, which retains the winning
`SegmentFit` as `Assignment.fit`. That fit is the single source of the observed
threshold estimate. Downstream stages must consume the serialized estimate and
must not call `fit_final_segment()` again.

The current fixed thresholds are withdrawn:

- `106.75 m` is a one-sided FAA LPV lateral FSD floor at the landing threshold
  point (LTP), not the normal tracking limit. ICAO uses one-half FSD.
- `-3.05/+6.10 m` comes from FAA wheel crossing height (WCH) procedure-design
  allowances. WCH is not a vertical tracking tolerance.

The replacement criteria are:

| Benchmark | Effective lateral bound at threshold | Vertical bound |
|---|---:|---:|
| LPV | `min(0.5 × LPV lateral FSD, 0.5 × runway width)` | `0.5 × DO-229E-derived LPV vertical FSD` |
| LNAV/VNAV with Baro-VNAV | `min(0.15 NM, 0.5 × runway width)` | `±22 m from the Baro-VNAV path` |

No one-third factor is used. One-half LPV FSD comes from current ICAO Doc
9613. One-half runway width is exact centreline-to-edge geometry.

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
The runway-assignment stage already solves this by fitting several
final-approach samples and evaluating the winning fit at `s = 0`.

The harvested track record must serialize a small
`observed_threshold_event` derived directly from `Assignment.fit`. The event
contains measurement and fitting facts only:

```text
status                         estimated or unavailable
method and method version
runway
runway_data_fingerprint         exact runway/frame/cycle binding
threshold_crossing_lat and threshold_crossing_lon
threshold_crossing_altitude_m
altitude_datum                 HAE for the current harvest
signed_cross_track_m           right-positive
cross_track_sigma_m            fit standard error at s = 0
altitude_sigma_m               fit standard error at s = 0
source_sample_range            inclusive original sample indices
fit_window_m
sample_count and along-track span
cross-track and altitude residual diagnostics
extrapolation_m
unavailable_reason             only when no event was produced
```

The altitude is physical crossing altitude in HAE:

```text
threshold_crossing_altitude_hae
    = runway threshold elevation HAE
    + Assignment.fit.height_at_threshold_m
```

The crossing latitude/longitude is the same `s = 0`, signed-cross-track point
resolved in the exact runway frame used by assignment. Storing that point lets
rendering reuse the estimate without reconstructing it from a later runway-data
cycle.

The event also carries an audit snapshot and canonical fingerprint of every
runway fact that can affect assignment or interpretation: threshold position,
course, HAE/MSL elevations and datum offset, runway width, TCH, glidepath,
LPV course width, source identifiers, and effective FAA runway/CIFP cycles.
Consumers reject a missing or mismatched fingerprint. The operator then runs
`--reclassify-existing`, which recomputes derived assignment/event data from the
stored HAE samples without downloading ADS-B again.

The event must not contain an approach type, FAS profile, LPV/LNAV-VNAV
limits, or any verdict. The raw `samples` array remains raw. `flight_key`
remains the stable record identity and is not redefined inside the event.

Evaluation consumes this event. It returns `indeterminate` when its status is
not `estimated` or required uncertainty/provenance is invalid. It does not
select samples, fit lines, or replace the stored estimate.

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

### 4.2 Vertical rule

LPV has approved vertical guidance and a vertical FSD. The FSD is not a simple
metre value stored in the FAA CIFP Path Point record.

The authoritative data path is:

```text
State-published FAS data
    (LTP/FTP, FPAP, glidepath angle, TCH, length offset, etc.)
        +
RTCA DO-229E-conformant SBAS deviation/scaling model
        ↓
vertical FSD at the evaluated position
        ↓
ICAO normal bound = 0.5 × vertical FSD
```

Let `F_vert(p)` be the one-sided vertical FSD produced by that validated model
at terminal position `p`, and let `z` be signed deviation from the desired
vertical path:

```text
B_vert = 0.5 × F_vert(p)
pass if abs(z) <= B_vert
fail otherwise
```

The desired path altitude at the threshold is defined using the authoritative
FAS geometry, including TCH. TCH defines the nominal path; it is not an error
tolerance.

### 4.3 Required LPV inputs

An LPV verdict requires:

- authoritative FAS data and effective cycle;
- threshold coordinates and compatible vertical datum;
- runway true course and current width;
- a read and implemented DO-229E deviation/scaling specification, or recorded
  normalized avionics deviation data;
- compatible aircraft/trajectory altitude reference; and
- quantified uncertainty or an explicit list of missing uncertainty sources.

Ordinary ADS-B does not contain the aircraft's LPV vertical deviation
indication. The practical project path is therefore a validated DO-229E
reference implementation using authoritative FAS data.

Until that implementation is available, LPV vertical is `indeterminate`.
This does not mean LPV lacks vertical guidance. It means the evaluator is
missing the certified scale needed to apply the ICAO half-FSD rule.

Do not substitute:

- WCH or TCH ranges;
- SBAS vertical alert limits such as 35 m or 50 m;
- obstacle-clearance surfaces;
- Baro-VNAV's ±22 m rule; or
- manufacturer-specific display values presented as universal.

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

Apply the same signed cross-track and uncertainty classification used for LPV.
The runway bound will normally control at the threshold.

### 5.3 Vertical rule

ICAO Doc 9613 §5.3.4.4.7 states that, when Baro-VNAV provides vertical-path
guidance during the FAS, deviations above and below the path must not exceed
22 m:

```text
B_vert = 22 m
pass if abs(z) <= 22 m
fail otherwise
```

The evaluation context must explicitly declare approved Baro-VNAV. Do not
infer it from trajectory shape or from the presence of a charted vertical
descent angle.

### 5.4 Fallback composite result

The event and composite logic are the same as LPV. Both lateral and vertical
components are required.

The report label is:

```text
RNP APCH LNAV/VNAV (Baro-VNAV) terminal geometric verdict
```

## 6. Uncertainty and invalid values

### 6.1 Interval classification

For signed estimate `e`, allowed magnitude `B`, and total 95% uncertainty
half-width `U95`:

```text
measurement interval = [e - U95, e + U95]
allowed interval     = [-B, B]

pass          if the measurement interval is wholly inside the allowed interval
fail          if the intervals do not overlap
indeterminate otherwise
```

Do not widen an official bound to hide uncertainty.

Observed uncertainty should include all material sources that can be
quantified: ADS-B position and altitude, fit regression, extrapolation, timing,
datum conversion, and runway/FAS reference data. List unmodelled sources.

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
- LPV vertical FSD only after the RTCA implementation is validated; and
- explicit approved Baro-VNAV applicability for the fallback.

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
- the observed event's source sample range, fit window, extrapolation,
  diagnostics, uncertainty, altitude datum, and producer version;
- signed `s`, `x`, and `z` deviations;
- guidance, runway, effective lateral, and vertical bounds;
- uncertainty intervals and missing uncertainty sources;
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

### 10.2 LNAV/VNAV vertical example

For Baro-VNAV vertical deviation `z = +10 m`:

```text
abs(10 m) <= 22 m → pass
```

This replaces the invalid conclusion produced by the old `+6.10 m` WCH-based
limit.

### 10.3 Missing LPV scale

If FAS path geometry is present but the DO-229E vertical scaling model is not:

```text
LPV lateral  = assessable
LPV vertical = indeterminate
LPV overall  = indeterminate unless another required component fails
```

Do not silently change the benchmark to LNAV/VNAV. The fallback must be
explicitly selected and supported by applicable procedure data.

## 11. Official source audit

### 11.1 Normative basis

| Source | Currency and passages read | Use | Local copy |
|---|---|---|---|
| ICAO Doc 9613, *Performance-based Navigation (PBN) Manual* | Fifth Edition, 2023. ICAO's 2026 catalogue and Store still identify this edition; ICAO's roadmap plans the next edition for 2027. Volume II, Part C, Ch. 5, Section A §§5.3.4.4.6–8 and Section B §§5.3.3.1.1–5.3.3.3.1.1 were read. | Universal LPV half-FSD rule; RNP APCH lateral rule; Baro-VNAV ±22 m rule; angular LPV scaling basis. | [ICAO Doc 9613](../docs/regulation/ICAO_Doc_9613_5th_Ed_2023.pdf) |
| FAA Order 8260.58D, *United States Standard for PBN Instrument Procedure Design* | Active; issued 2025-01-15. §1-3-1.f(2)(b), §3-1-5.c(3), Figure 3-1-7, and Formula 3-1-1 were read. | Explains WCH/TCH and the FAA 106.75 m lateral FSD floor. It is not the universal operating rule. | [FAA Order 8260.58D](../docs/regulation/Order_8260.58D.pdf) |
| FAA AC 20-138D Change 2, *Airworthiness Approval of Positioning and Navigation Systems* | FAA lists it active. Change 2 dated 2016-04-07. Ch. 4 and §§15-7 through 15-7.8 were read. | Confirms the certified SBAS installation/source chain and that LPV deviation details come from RTCA DO-229. It predates TSO-C146e and is not used alone for the scaling formula. | [FAA AC 20-138D Change 2](../docs/regulation/FAA_AC_20-138D_Change_2.pdf) |
| EASA ETSO-C146e A1 | Applicable from 2020-07-25 and current in EASA's live ETSO register on 2026-08-12. §§1–5 and the relevant appendix introduction were read. | Public official confirmation that current C146e equipment requirements use RTCA DO-229E Section 2. | [EASA ETSO-C146e A1](../docs/regulation/EASA_ETSO-C146e_A1.pdf) |
| FAA CIFP Readme, Volume 2608 | Cycle effective 2026-08-06 to 2026-09-03. All seven pages were read. | Confirms Path Point records and current FAA CIFP status. | [FAA CIFP Readme 2608](../data/CIFP/CIFP_260806/CIFP%20Readme%202608.pdf) |
| FAA NASR APT layout, readme, and data archive | Current 2026-08-06 distribution checked. Runway width, true alignment, effective-date, and datum fields were read; `APT_RWY.csv` supplied the configured runway widths. | Authoritative runway source for the implemented U.S. aerodromes. Other States require their authoritative AIP/aerodrome source. | [APT layout](../docs/regulation/FAA_NASR_APT_DATA_LAYOUT_2025-10-23.pdf), [CSV readme](../docs/regulation/FAA_NASR_CSV_README_2026-08-06.pdf), [complete official archive](../docs/regulation/FAA_NASR_APT_CSV_2026-08-06.zip) |

RTCA availability needs two separate statements:

- DO-229F, issued 2020-06-11, is the newest RTCA DO-229 revision found in the
  official store as of 2026-08-12.
- FAA TSO-C145e/C146e and EASA ETSO-C146e incorporate DO-229E. The official
  RTCA DO-229E product page explicitly confirms that relationship.

Both official electronic products are sold by RTCA for USD 475 each; RTCA does
not expose a free official full-text download. Neither licensed document is in
this repository. The implementation must obtain and read DO-229E, then check
the relevant DO-229F changes before choosing and recording the implemented
scaling revision. No formula may be reconstructed from store descriptions or
unofficial copies.

### 11.2 Official status pages

- [ICAO Store: Doc 9613 Fifth Edition](https://store.icao.int/en/performance-based-navigation-pbn-manual-doc-9613)
- [ICAO roadmap: next PBN Manual edition planned for 2027](https://www.icao.int/air-navigation-bureau/gnss-rfi/roadmap/medium-term-actions)
- [FAA Order 8260.58D status](https://www.faa.gov/regulations_policies/orders_notices/index.cfm/go/document.information/documentID/1043458)
- [FAA AC 20-138D status](https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentID/1023966)
- [EASA current ETSO register](https://www.easa.europa.eu/en/domains/aircraft-products/etso/list-of-all-etso)
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
bb9d8698beae04c2834e7fdfb9e1574a09efb867f222d94e2dcf52c9ab7f05c3  FAA_NASR_APT_DATA_LAYOUT_2025-10-23.pdf
6338a19182d38294cdca37cf4838e643c55ab5a720f10b46b552cae89844bd24  FAA_NASR_CSV_README_2026-08-06.pdf
47023fc1f557594435aaf06cfa0e056abe37c48b3c869a5a316e66d7cf54ba0f  CIFP Readme 2608.pdf
a6ad75ba834fcc423fbc7f7aebb3e9d075ae169da3e9fce5693cd21f2355b6ca  FAACIFP18 (CIFP 260806)
dd9768780197ba3e14d447be0be9cf95e1e55e7c56c8ec4dfecf5dc4f4a10ef1  FAA_NASR_APT_CSV_2026-08-06.zip
```

## 12. Implementation boundary

Implementation may change:

- the smallest policy-free harvest serialization code needed to copy the
  winning `Assignment.fit` into `observed_threshold_event`;
- observed arrival preparation and CZML code only to remove their duplicate
  fits and consume existing stored indices/event data;
- `evaluation/` context, coordinate math, event consumption, gates,
  aggregation, report schema, visualization payload, CLI, and tests; and
- regenerated evaluation JSON/HTML artifacts.

Implementation must not change:

- raw ADS-B samples or the in-memory raw trajectory model;
- the final-segment fitting or runway-assignment algorithms merely to support
  evaluation;
- arrival-manifest, flight-scenario, optimizer, or prediction contracts;
- stable identity generation;
- `trajectory_data_process` to carry approach profiles, limits, or verdicts;
- unrelated approach types.

The derived harvested-track record schema changes only by adding the
policy-free event. Old track records, arrival views, CZML, and evaluation
reports are regenerable and must be regenerated. Do not add a dual-read
compatibility path that silently refits old records.

## 13. Acceptance criteria

The current implementation is accepted when:

1. LPV vertical and composite results remain `indeterminate` until a licensed
   DO-229E scaling implementation is read, currency-checked against DO-229F,
   and validated against authoritative examples;
2. the LNAV/VNAV fallback requires explicit approved Baro-VNAV context;
3. along-track and cross-track error are distinct;
4. the winning runway-assignment fit produces one serialized, policy-free
   observed threshold event with uncertainty;
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
    prediction outputs contain no approach profile, limit, or verdict policy.
