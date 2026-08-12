# Terminal final-approach verdict standard

Status: design specification; implementation is pending

Standards and publication status checked: 2026-08-12

Applies to: `evaluation/`

Does not change: trajectory records, arrival manifests, flight scenarios,
optimizer contracts, prediction contracts, or stable flight identities

## Executive decision

The evaluation answers one narrow question:

> At the runway-threshold arrival event, is the trajectory sufficiently aligned
> laterally and vertically with the configured final-approach benchmark?

It does **not** verify the entire LPV final-approach segment or “LPV cone.” It
does not certify a procedure, prove which procedure an observed flight actually
flew, determine whether an approach was stabilized under an operator SOP, or
prove that touchdown and ground roll remained on the runway.

The evaluated object is one terminal threshold-arrival state:

- for an optimized or predicted trajectory, normally the final computed state
  associated with the threshold target;
- for an observed ADS-B trajectory whose receiver coverage ends before the
  threshold, one fitted threshold-crossing estimate derived from several usable
  final-approach samples;
- optionally, a short physical-distance terminal window may be reported as a
  separate diagnostic, but its maximum error does not silently replace the
  threshold verdict.

LPV is the primary benchmark. RNP APCH to LNAV minima is the only fallback
benchmark supported by this design. The benchmark is evaluation context; it is
not an intrinsic property added to a trajectory.

The current fixed thresholds are withdrawn:

- `106.75 m` is the FAA minimum one-sided LPV lateral full-scale deflection
  (FSD) at the landing threshold point, not the normal operating deviation
  limit. ICAO Doc 9613 specifies one-half FSD for normal LPV flight technical
  error.
- `-3.05/+6.10 m` is derived from FAA wheel crossing height (WCH) procedure-
  design allowances. WCH is not a vertical tracking-error tolerance.

For the terminal LPV verdict, use:

```text
lateral guidance bound = 0.5 * actual lateral FSD at the threshold
runway-edge bound       = 0.5 * actual runway width
effective lateral bound = min(lateral guidance bound, runway-edge bound)

vertical bound          = 0.5 * actual vertical FSD at the threshold
```

The two lateral source bounds are retained in the report for audit, but they
produce one terminal lateral result. This directly encodes both facts that
matter at the threshold: the state must satisfy the LPV tracking criterion and
its reference point must not be outside the runway edges.

No one-third factor is used. One-half FSD comes directly from current ICAO PBN
criteria; one-half runway width follows exactly from centreline-to-edge
geometry. There is no official basis for replacing either with one-third.

If the actual LPV vertical FSD cannot be obtained or validly reconstructed for
the configured procedure, the LPV vertical verdict is `indeterminate`. The
implementation must not invent a value from WCH, threshold crossing height,
obstacle surfaces, integrity alert limits, or a convenient pass rate.

## 1. Scope and claim boundary

### 1.1 What this verdict evaluates

The verdict evaluates the terminal relationship among:

1. an evaluated threshold-arrival state;
2. the configured runway threshold, centreline, width, elevation, and course;
3. the configured LPV benchmark, or the single RNAV-LNAV fallback; and
4. the uncertainty of the state and reference geometry.

The report may be produced for any of the three existing trajectory subjects:

| Subject | Terminal state used |
|---|---|
| `optimized` | Computed terminal state, normally `states[-1]` |
| `predicted` | Predicted terminal state, normally `states[-1]` |
| `observed` | Threshold-crossing state fitted from final ADS-B samples |

“LPV benchmark” in an observed report means geometric conformance to the LPV
reference selected for the evaluation. ADS-B alone does not prove that the
crew selected or was cleared for LPV.

### 1.2 What this verdict does not evaluate

The main verdict does not evaluate:

- every sample along the LPV final approach segment;
- an obstacle-clearance or procedure-protection volume;
- navigation-system integrity or alert performance;
- a complete stabilized-approach SOP;
- aircraft configuration, thrust, crew action, weather minima, or clearance;
- touchdown, landing-gear footprint, or ground-roll containment;
- approach types other than LPV and the RNAV-LNAV fallback; or
- prediction accuracy against an observed path over mismatched physical spans.

A later research question may define a separate full-final-segment conformance
metric. It must not be presented as this terminal verdict.

### 1.3 Why the verdict is terminal rather than a whole-volume test

The project currently asks whether observed, optimized, and predicted arrivals
reach the runway target plausibly. The records do not contain avionics
deviation indications or confirmed flown-procedure data, and the optimized
terminal state is explicitly targeted at the runway threshold. A whole-FAS
test would therefore answer a different question, require more procedure data,
and treat each sample over a distance-varying angular scale.

Using several observed samples to estimate one threshold crossing does not turn
the result into a whole-segment verdict. Those samples support estimation of
the terminal event; they are not all independently gated as LPV-conformance
samples.

## 2. Why the current thresholds are invalid

### 2.1 The current lateral number is full scale, not the normal limit

FAA Order 8260.58D §3-1-5.c(3), Figure 3-1-7, and Formula 3-1-1 define the LPV
course width at the landing threshold point as the greater of:

```text
350 ft
tan(1.5 degrees) * distance from GARP
```

The order converts and rounds that value to 0.25 m. The 350 ft floor therefore
becomes 106.75 m. This is a one-sided FSD value used in the FAA procedure-
design/coding model.

ICAO Doc 9613 Fifth Edition, Volume II, Part C, Chapter 5, Section B
§5.3.3.1.1.1(b) states that LPV flight technical error is acceptable when the
aircraft is maintained within one-half FSD laterally and vertically. Therefore,
when the actual threshold lateral FSD is 106.75 m:

```text
normal lateral guidance bound = 106.75 / 2 = 53.375 m
```

That correction still does not guarantee that the aircraft reference point is
over pavement. A 45.72 m (150 ft) runway has a half-width of only 22.86 m.

### 2.2 The current vertical numbers are the wrong kind of quantity

FAA Order 8260.58D uses WCH/TCH when designing how the vertical path crosses
the runway threshold for aircraft geometry. The permitted WCH relationship is
not a statement that flown vertical deviation may be only 10 ft low or 20 ft
high.

LPV vertical tracking must instead use one-half of the actual vertical FSD
applicable at the evaluated point. ICAO also describes LPV lateral and vertical
display scaling as angular and derived from the final approach segment (FAS)
data and avionics requirements. Consequently, WCH, TCH, vertical FSD, and
vertical tracking error must remain distinct fields.

### 2.3 Integrity limits and obstacle surfaces are also not substitutes

ICAO LPV navigation-system error alerting values are integrity-monitoring
limits, not flight technical error gates. PANS-OPS/TERPS obstacle surfaces are
procedure-design protection geometry, not normal tracking tolerances. Neither
may be used to fill a missing vertical FSD.

### 2.4 Why one-third is rejected

No controlling source inspected for this design specifies one-third of LPV
course width as the normal terminal tolerance. Choosing one-third would be an
empirical thesis rule with a regulatory appearance.

The factors used here have separate, traceable derivations:

- `0.5 * FSD`: the ICAO normal LPV flight-technical-error criterion;
- `0.5 * runway width`: the exact distance from runway centreline to an edge;
- `min(...)`: the mathematical intersection of both required lateral
  conditions at the same terminal point.

## 3. Terminal event definition

### 3.1 Runway-aligned coordinates

Convert terminal geometry into a local runway frame:

- `s`: signed along-track displacement from the threshold plane;
- `x`: signed cross-track displacement from the runway centreline;
- `z`: signed vertical displacement from the desired path at the event;
- `delta_track`: wrapped track-angle difference from runway course.

Use one documented sign convention, for example `s < 0` before the threshold
and `s > 0` beyond it. The verdict gates `x` and `z`. It reports `s` and
`delta_track` separately.

The existing computed-trajectory metric uses great-circle final-to-target
distance as “lateral” error. That conflates along-track and cross-track error:
a state 30 m short but exactly on the centreline appears to have a 30 m lateral
miss. The replacement must never make that conflation.

### 3.2 Optimized and predicted trajectories

For a record whose target is the configured threshold-arrival state:

1. use the final computed state when it represents the terminal target event;
2. transform it into the runway frame;
3. report `s`, signed `x`, signed `z`, speed difference, track difference, and
   event time;
4. do not search backwards for a more favourable point.

If the trajectory crosses the threshold plane between its last two states and
the endpoint is beyond the plane only because of output discretization, linear
interpolation to `s = 0` is allowed and must be reported as interpolation.

If the trajectory terminates materially before the threshold and has no valid
threshold event, it cannot pass the threshold verdict. Use
`event_status = "not_reached"`; do not extrapolate an optimized or predicted
trajectory to manufacture an arrival.

Any numerical plane tolerance exists only to absorb coordinate precision. It
must be derived from record precision or solver tolerance, serialized, and
must not become an operational along-track allowance.

### 3.3 Observed ADS-B trajectories

Receiver coverage often ends before the runway threshold. Evaluating the last
raw ADS-B point would mostly measure where reception stopped. Instead:

1. select the final samples within a declared physical along-track fit window;
2. fit signed cross-track position and height relative to the desired path as
   functions of along-track position;
3. evaluate both fits at `s = 0` to obtain one threshold-crossing estimate;
4. propagate fit and extrapolation uncertainty into the verdict; and
5. reject the estimate when the fit is invalid or extrapolation exceeds the
   declared method limit.

The fit-selection criteria are data-quality/methodology criteria. They must be
serialized under names such as `fit_window_m`, `max_fit_cross_track_m`,
`fit_glidepath_range_deg`, and `max_vertical_fit_rms_m`. Passing them means the
threshold event can be estimated; it is not itself an approach pass.

An observed result describes conformance to the configured benchmark. Do not
label the flight as having flown LPV solely because LPV was the benchmark.

### 3.4 Optional terminal-window diagnostic

A target-constrained optimizer can trivially make its last state exact while
approaching that state poorly. A separate diagnostic may therefore measure a
short terminal interval before the threshold.

If enabled, it must:

- use a physical along-track interval, such as `[-D, 0] m`, not “last N
  samples”;
- resample consistently by along-track distance;
- report maximum absolute cross-track and vertical-path error, plus coverage;
- have its own name, criteria, and status; and
- remain non-gating until the thesis explicitly adopts and validates it.

The present standard deliberately does not select `D` or create a hard bound.
Doing so would add a second scientific question and requires a separate design
decision.

## 4. LPV terminal verdict

### 4.1 Required evaluation inputs

An LPV terminal evaluation requires:

- threshold latitude, longitude, and elevation;
- runway true course and current runway width;
- the desired vertical-path altitude at the threshold-arrival event;
- actual LPV lateral FSD at the threshold;
- actual LPV vertical FSD at the threshold;
- source, cycle/effective date, units, and datum for each value; and
- uncertainty sufficient for the selected classification rule.

The desired vertical altitude is normally the applicable final-path crossing
altitude, not bare runway elevation. Its aircraft reference point and vertical
datum must match the trajectory altitude. A mismatch between antenna/aircraft
reference, ellipsoidal height, orthometric height, runway elevation, and path
crossing altitude can dominate the result.

### 4.2 Lateral criterion

Let:

- `F_lat` be the one-sided lateral FSD at the threshold;
- `W` be current physical runway width;
- `B_guidance = 0.5 * F_lat`;
- `B_runway = 0.5 * W`; and
- `B_lat = min(B_guidance, B_runway)`.

For signed terminal cross-track error `x`, the deterministic rule is:

```text
lateral pass if abs(x) <= B_lat
lateral fail otherwise
```

The report must retain `B_guidance`, `B_runway`, and `B_lat`. The user receives
one terminal lateral verdict, while the report remains able to explain which
bound controlled.

This is a reference-point containment proxy at the threshold. It does not prove
landing-gear containment at touchdown. A gear-footprint or touchdown study is
out of scope.

### 4.3 Vertical criterion

Let `F_vert` be the one-sided actual LPV vertical FSD at the threshold and
`z` the signed terminal deviation from the desired vertical path:

```text
B_vert = 0.5 * F_vert
vertical pass if abs(z) <= B_vert
vertical fail otherwise
```

If `F_vert` is absent, not finite, not positive, from an incompatible datum, or
not traceable to the evaluated procedure/validated model:

```text
vertical status = indeterminate
LPV overall status = indeterminate, unless another required component fails
```

The evaluator must not fall back to the old WCH window.

### 4.4 LPV composite logic

```text
if event_status is not valid:
    verdict = fail or indeterminate according to the explicit event state
elif lateral_status == fail or vertical_status == fail:
    verdict = fail
elif lateral_status == pass and vertical_status == pass:
    verdict = pass
else:
    verdict = indeterminate
```

Use `fail` when a computed/predicted trajectory was required to reach the
threshold but did not. Use `indeterminate` when an observed threshold event
cannot be estimated from available surveillance data. This distinction keeps
model failure separate from measurement insufficiency.

## 5. RNAV-LNAV fallback verdict

### 5.1 When the fallback applies

Use the fallback only when the evaluation run explicitly selects
`rnav_lnav`, for example because validated LPV scale data are unavailable and
an RNP APCH to LNAV minima is the intended research benchmark.

Do not infer the fallback from trajectory shape. Do not add LP, LNAV/VNAV,
Baro-VNAV, RNP AR, ILS, GLS, visual, or other modes to this implementation.

### 5.2 Lateral criterion

ICAO Doc 9613 Volume II, Part C, Chapter 5, Section A §5.3.4.4.6 specifies
normal lateral deviation within one-half the RNP value in the final approach
segment to LNAV or LNAV/VNAV minima. For the standard RNP APCH final value of
RNP 0.3:

```text
B_guidance = 0.5 * 0.3 NM = 0.15 NM = 277.8 m
B_runway = 0.5 * W
B_lat = min(B_guidance, B_runway)
```

The same terminal cross-track test and uncertainty rule then apply. The runway
bound will normally control near the threshold; the 0.15 NM value must never be
misrepresented as pavement containment.

### 5.3 Vertical status and claim wording

LNAV does not provide approved vertical guidance. Therefore:

- `vertical_status = not_applicable` for the RNAV-LNAV benchmark;
- vertical target error remains a descriptive metric;
- the result is labelled `RNAV-LNAV terminal lateral verdict`; and
- it must not be described as vertically guided approach conformance.

The fallback passes when it has a valid terminal event and the effective
lateral criterion passes. Missing vertical guidance is not silently replaced
by a thesis-created vertical gate.

## 6. Uncertainty and non-finite values

### 6.1 Interval classification

Do not widen an official bound to compensate for measurement uncertainty. For
a signed estimate `e`, allowed magnitude `B`, and total 95% uncertainty
half-width `U95`:

```text
measurement interval = [e - U95, e + U95]
allowed interval     = [-B, B]

pass          if measurement interval is wholly inside allowed interval
fail          if measurement interval is wholly outside allowed interval
indeterminate otherwise
```

For deterministic optimized outputs, measurement uncertainty may be absent,
but coordinate/model precision and authoritative runway/path uncertainty must
not be claimed as zero without justification. Predicted-state uncertainty may
be reported when the model supplies a validated interval.

For fitted ADS-B arrivals, include all quantified material sources available:
fit regression, extrapolation, surveillance position/altitude, datum
conversion, timing/interpolation, and runway/path reference uncertainty. The
report must list missing uncertainty sources rather than implying the model is
complete.

### 6.2 Mandatory finite validation

Every state coordinate, altitude, time, threshold, FSD, runway width,
uncertainty, and derived deviation used by a verdict must be finite. Required
positive quantities must also be strictly positive.

`NaN`, positive infinity, and negative infinity can make ordinary Python
comparisons evaluate in unsafe ways and can produce non-standard JSON. A record
containing a non-finite verdict input must not pass. The evaluator must reject
it or return an explicit invalid/indeterminate row, and JSON output must be
written with non-finite values disallowed.

## 7. Evaluation-owned context and report contract

### 7.1 Do not put approach profiles into trajectory data

Trajectory records describe the evaluated motion and retain their current
schema. They must not acquire LPV scale tables, runway widths, assessment
profiles, or verdict methodology.

The evaluator instead receives an `AssessmentContext` from its CLI, pipeline
caller, or authoritative procedure/runway resolver. Conceptually:

```json
{
  "subject": "observed",
  "benchmark": "lpv",
  "runway": {
    "airport": "CYYC",
    "ident": "35R",
    "threshold_lat_deg": 0.0,
    "threshold_lon_deg": 0.0,
    "course_true_deg": 0.0,
    "width_m": 60.0,
    "vertical_path_at_threshold_m": 0.0,
    "source": "authoritative source and effective date"
  },
  "lpv": {
    "procedure_id": "authoritative identifier",
    "lateral_fsd_at_threshold_m": 106.75,
    "vertical_fsd_at_threshold_m": null,
    "source": "authoritative procedure/FAS source and cycle"
  },
  "method": {
    "terminal_event": "subject_specific_v1",
    "fit_window_m": [-5000.0, -300.0],
    "max_extrapolation_m": null,
    "confidence": 0.95
  }
}
```

The numeric values above illustrate serialization, not new standards-derived
defaults. `max_extrapolation_m: null` deliberately records that the method limit
still requires validation; it must be resolved before an observed fit can earn
a pass. The actual context is resolved once per compatible evaluation batch and
copied into the derived evaluation report for reproducibility.

The subject must be explicit at evaluation time. It may come from an existing
record field or a required pipeline/CLI argument, but the evaluator must not
guess `optimized` merely because a producer omitted metadata.

### 7.2 Reuse existing stable identity

The repository already has stable identity information such as `flight_key`
and record filenames. Evaluation rows and overlays must preserve and display
those existing values. Do not create another flight identity field and do not
use callsign alone as a selector key, because callsigns can repeat in a batch.

### 7.3 Derived report sketch

The evaluation report, not the trajectory, owns the resolved standard:

```json
{
  "schema_version": 2,
  "subject": "observed",
  "assessment": {
    "benchmark": "lpv",
    "standard": "ICAO Doc 9613, Fifth Edition, 2023",
    "terminal_only": true,
    "runway": {
      "airport": "CYYC",
      "ident": "35R",
      "width_m": 60.0,
      "source": "source and effective date"
    },
    "guidance": {
      "procedure_id": "identifier",
      "lateral_fsd_at_threshold_m": 106.75,
      "vertical_fsd_at_threshold_m": 30.0,
      "source": "source and effective date"
    },
    "criteria": {
      "lateral_fsd_fraction": 0.5,
      "runway_width_fraction": 0.5,
      "vertical_fsd_fraction": 0.5,
      "fit_window_m": [-5000.0, -300.0],
      "max_fit_cross_track_m": 400.0,
      "fit_glidepath_range_deg": [2.0, 4.5],
      "max_vertical_fit_rms_m": 6.0,
      "max_extrapolation_m": null,
      "confidence": 0.95
    }
  },
  "trajectories": [
    {
      "flight_key": "existing stable key",
      "file": "existing-record-name.json",
      "event": {
        "status": "estimated",
        "method": "observed_final_fit",
        "extrapolation_m": 325.0
      },
      "deviation": {
        "along_m": 0.0,
        "cross_signed_m": 12.0,
        "vertical_signed_m": 4.0
      },
      "bounds": {
        "lateral_guidance_m": 53.375,
        "runway_half_width_m": 30.0,
        "lateral_effective_m": 30.0,
        "vertical_half_fsd_m": 15.0
      },
      "components": {
        "lateral": "pass",
        "vertical": "pass"
      },
      "verdict": "pass"
    }
  ]
}
```

Again, the numbers illustrate the shape only. All material criteria and source
revisions must be serialized, including the observed-fit overrides noted in
the review. Regenerable old reports should be regenerated into the new schema;
the evaluator should not grow speculative dual-read compatibility.

## 8. Reference-comparison boundary

The optimized-versus-observed reference metrics are separate from the terminal
verdict. They must compare a common physical span.

Observed ADS-B tails commonly stop before the optimized target. Resampling each
entire path over its own arc length pairs different physical locations and can
report a large difference for coincident paths with different endpoints.

The implementation must choose and serialize one defensible method:

1. crop both paths to their overlapping runway-aligned span and compare there;
2. extend the observed fitted reference to the threshold with uncertainty and
   compare both over that common span; or
3. mark the path and time comparisons unavailable when endpoints/spans are not
   compatible.

It must not independently normalize mismatched full path lengths. Flight-time
delta is meaningful only when both durations refer to the same start and end
events.

## 9. Worked examples

### 9.1 Minimum FAA LPV scale and a 150 ft runway

```text
FAA minimum lateral FSD at threshold = 106.75 m
ICAO normal half-FSD bound            = 53.375 m
150 ft runway width                   = 45.72 m
runway half-width                     = 22.86 m
effective lateral bound               = min(53.375, 22.86) = 22.86 m
```

For a terminal state 40 m right of centreline:

- current 106.75 m gate: pass;
- LPV half-FSD alone: pass;
- new effective terminal lateral gate: fail.

This corrects the practical problem: a trajectory cannot pass terminal lateral
alignment while its reference point is outside the runway edge.

### 9.2 Vertical error of +10 m

The old `+6.10 m` rule fails a +10 m state, but that conclusion has no valid
LPV basis.

- if actual vertical FSD is 30 m, half FSD is 15 m and +10 m passes;
- if actual vertical FSD is 16 m, half FSD is 8 m and +10 m fails;
- if actual vertical FSD is unavailable, the result is indeterminate.

The evaluator cannot choose among these outcomes from WCH.

### 9.3 Observed tail ending 325 m before the threshold

The final raw ADS-B point is not the threshold event. A qualified fit may
estimate `x(0)` and `z(0)` from the final approach samples. The report records
325 m extrapolation, uncertainty, fit criteria, and event status. If the fit
cannot support that extrapolation, the observed result is indeterminate rather
than a false failure at the receiver-loss point.

### 9.4 RNAV-LNAV fallback on the same runway

```text
LNAV final guidance bound = 0.15 NM = 277.8 m
runway half-width          = 22.86 m
effective lateral bound    = 22.86 m
```

The terminal state still has to align with the runway. The report labels
vertical guidance `not_applicable`, so the fallback makes no unsupported LPV
or Baro-VNAV claim.

## 10. Official source and edition audit

### 10.1 Source hierarchy

Use sources in this order:

1. current ICAO PBN navigation specifications for the internationally
   applicable operational baseline;
2. the actual State AIP/FAS data and current runway data for the evaluated
   procedure;
3. applicable State, aircraft, avionics, or operator data when required to
   resolve an installation-specific value; and
4. advisory material only for a clearly labelled research proxy.

FAA Order 8260.58D explains the existing project's 106.75 m constant and can
provide procedure-design data for FAA procedures. It is not elevated into a
universal operating rule. The universal half-FSD criterion comes from ICAO;
the actual procedure data supplies the FSD.

### 10.2 Downloaded documents actually read

“Read” below means the operative passages and surrounding definitions were
inspected in the PDF, rather than relying on a search-result outline.

| Source | Version/status verification | Passages read and use | Local copy |
|---|---|---|---|
| ICAO Doc 9613, *Performance-based Navigation (PBN) Manual* | Fifth Edition, 2023. ICAO's 2026 catalogue and Store identify this edition; ICAO's roadmap says the next edition is planned for 2027. Checked 2026-08-12. | Volume II, Part C, Ch. 5, Section A §§5.3.4.4.1–8 and Section B §§5.3.3.1.1–5.3.3.3.1.1, with FAS/VNAV context. Primary universal basis for half-FSD LPV and the LNAV fallback. | [ICAO Doc 9613](../docs/regulation/ICAO_Doc_9613_5th_Ed_2023.pdf) |
| ICAO *Products and Services Catalogue* | Official 2026 catalogue; its Doc 9613 entry confirms Fifth Edition, 398 pages, ISBN 978-92-9275-221-7, and order number. | Doc 9613 entry on printed p. 60; edition identity only. | [ICAO 2026 catalogue](../docs/regulation/ICAO_Products_and_Services_Catalogue_2026.pdf) |
| FAA Order 8260.58D, *United States Standard for PBN Instrument Procedure Design* | Active; issued 2025-01-15. The saved PDF matched the current FAA-served file when checked 2026-08-12. | §1-3-1.f(2)(b) for WCH/TCH context; §3-1-5.c(3), Figure 3-1-7, and Formula 3-1-1 for LPV threshold lateral course width. | [FAA Order 8260.58D](../docs/regulation/Order_8260.58D.pdf) |
| FAA NASR APT CSV Data Layout | Layout dated 2025-10-23, distributed with the 2026-08-06 NASR cycle checked 2026-08-12. | `APT_RWY.RWY_WIDTH`, `APT_RWY_END.TRUE_ALIGNMENT`, effective date, units, and definitions. Used to validate one available authoritative runway-geometry source for US data. | [FAA NASR APT layout](../docs/regulation/FAA_NASR_APT_DATA_LAYOUT_2025-10-23.pdf) |
| FAA NASR CSV Readme | Dated 2026-08-06 and supplied by the official current subscription page. | Status/datum text and the APT file-family description. | [FAA NASR CSV readme](../docs/regulation/FAA_NASR_CSV_README_2026-08-06.pdf) |
| FAA CIFP Readme, Volume 2608 | Official page lists cycle 260806 as effective 2026-08-06 through 2026-09-03; checked 2026-08-12. | All seven pages: effective dates, ARINC 424-18 status, included records, coding qualifications, exclusions, and CRC/error terms. | [FAA CIFP Readme 2608](../data/CIFP/CIFP_260806/CIFP%20Readme%202608.pdf) |

The public Doc 9613 PDF was obtained from EUROCONTROL's PBN Portal library.
Its publisher, cover, edition, page count, order number, and ISBN match the
official ICAO catalogue and Store record. That validates the publication
identity, though it is not a byte comparison with ICAO's paid eLibrary file.

### 10.3 Official online status pages

- [ICAO Store: Doc 9613 Fifth Edition, 2023](https://store.icao.int/en/performance-based-navigation-pbn-manual-doc-9613)
- [ICAO GNSS-RFI roadmap: next PBN Manual edition planned for 2027](https://www.icao.int/air-navigation-bureau/gnss-rfi/roadmap/medium-term-actions)
- [FAA: Order 8260.58D active document page](https://www.faa.gov/regulations_policies/orders_notices/index.cfm/go/document.information/documentID/1043458)
- [FAA: 6 August 2026 NASR subscription](https://www.faa.gov/air_traffic/flight_info/aeronav/Aero_Data/NASR_Subscription/2026-08-06/)
- [FAA: current CIFP download page](https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/cifp/download/)

The local Canadian document previously considered is not used as the universal
basis. Older local TERPS, PANS-OPS, or aerodrome documents are also not used to
create the new numeric gates when their currency cannot be established.

### 10.4 File integrity

```text
d06e3fdd7cc2c24adcd174f997aace1b860eea6c64fbd5410be4ab2ccddae8e4  ICAO_Doc_9613_5th_Ed_2023.pdf
7f92832a5778358e1810f634a80586ad19a0317a4f77ebe0ea83fd123fc3f1f2  ICAO_Products_and_Services_Catalogue_2026.pdf
ead3c8e089a88dbde5de0a35a47ad72b9d484a54d408f1b0db15828581f31638  Order_8260.58D.pdf
bb9d8698beae04c2834e7fdfb9e1574a09efb867f222d94e2dcf52c9ab7f05c3  FAA_NASR_APT_DATA_LAYOUT_2025-10-23.pdf
6338a19182d38294cdca37cf4838e643c55ab5a720f10b46b552cae89844bd24  FAA_NASR_CSV_README_2026-08-06.pdf
31baa086aee802f3673abac0b994db534c2409b22e1dbda4d2a6d5bfb4529c7c  CIFP_260806.zip
47023fc1f557594435aaf06cfa0e056abe37c48b3c869a5a316e66d7cf54ba0f  CIFP Readme 2608.pdf
a6ad75ba834fcc423fbc7f7aebb3e9d075ae169da3e9fce5693cd21f2355b6ca  FAACIFP18 (CIFP 260806)
```

## 11. Implementation boundaries and acceptance criteria

### 11.1 Permitted change scope

Implementation should be confined to:

- `evaluation/` domain types, terminal-event extraction, coordinate math,
  gates, aggregation, visualization payload, CLI, and tests;
- the smallest existing pipeline call sites required to pass evaluation-owned
  context explicitly; and
- regeneration of derived evaluation JSON/HTML artifacts after the schema is
  finalized.

It must not modify `trajectory_data_process` data contracts, arrival manifests,
flight-scenario schemas, optimizer inputs, prediction records, or stable
identity generation merely to carry evaluation policy.

### 11.2 Required regression coverage

Tests must establish at least:

1. non-finite coordinates, deviations, bounds, and uncertainty cannot pass or
   be emitted as non-standard JSON;
2. computed along-track and cross-track deviations are distinct;
3. final computed state selection does not search for a favourable earlier
   sample;
4. observed multi-sample fitting produces exactly one terminal event;
5. excessive or invalid observed extrapolation is indeterminate;
6. the LPV lateral limit is `min(half FSD, half runway width)`;
7. LPV vertical uses half actual vertical FSD and never WCH;
8. missing LPV vertical FSD yields indeterminate, not a fallback constant;
9. RNAV-LNAV is the only enabled fallback and has no vertical guidance gate;
10. uncertainty intervals produce pass/fail/indeterminate correctly;
11. evaluation criteria and source revisions are serialized;
12. subject is explicit at evaluation time;
13. existing `flight_key`/filename identity reaches overlay selectors;
14. zero or negative `--max-tracks` is rejected, or zero consistently produces
    no overlays under the documented CLI contract; and
15. reference comparisons never resample mismatched entire physical spans.

### 11.3 Pipeline acceptance

Before publishing regenerated reports:

- run the focused `evaluation` tests in the `aeroviz` conda environment;
- run the observed evaluate-only path for at least one existing airport;
- run one optimized and one predicted evaluation through their existing
  pipeline entry points;
- validate report JSON with non-finite output disabled;
- open the generated HTML and verify repeated callsigns remain distinguishable;
- confirm no trajectory, arrival-manifest, or flight-scenario schema changed;
- compare counts so invalid or indeterminate records are retained rather than
  silently dropped; and
- record the exact procedure/runway source cycle in each report.

## 12. Unresolved prerequisite

The implementation cannot claim a standards-backed LPV vertical pass until an
authoritative, procedure-compatible source or validated reconstruction for the
vertical FSD at the threshold is identified and tested. Current local runway
and CIFP material is sufficient to investigate the data path, but this
document does not assume that the existing parser already exposes the required
vertical scale.

Until that prerequisite is resolved, the honest outcomes are:

- LPV lateral can be evaluated when lateral FSD and runway width are valid;
- LPV vertical is `indeterminate`;
- LPV overall is therefore `indeterminate` unless lateral or event validity
  already fails; or
- an evaluation run may explicitly select the RNAV-LNAV fallback and clearly
  report only its terminal lateral claim.

That limitation is preferable to another precise-looking but unsupported
vertical threshold.
