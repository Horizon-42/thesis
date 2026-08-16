# RNAV terminal-approach verdict standard

Status: implemented by `evaluation`, report schema
`terminal-approach-evaluation-v4`.

This document defines the small, project-wide terminal-geometry check used for
observed, optimized, and predicted trajectories. It is deliberately not an
ADS-B landing detector, a reconstruction of the whole LPV protection volume,
or a certification claim that an aircraft landed safely.

## 1. Decision and claim boundary

The evaluator answers one question:

> At the runway threshold plane, is the trajectory reference point reasonably
> aligned with the selected runway and the published threshold-crossing path?

It evaluates one threshold event only. It does not evaluate touchdown point,
flare, rollout, obstacle clearance, aircraft extremities, or whether the crew
actually flew the benchmark procedure.

The implemented vertical rule is:

```text
pass iff -22 m <= vertical error <= +22 m
```

This is a **project RNAV terminal-geometry acceptance bound**. Its numerical
basis is the internationally published `+22 m/-22 m` RNP APCH Baro-VNAV final
approach deviation limit in ICAO Doc 9613, Fifth Edition (2023), Volume II,
Part C, Chapter 5, Section A, §5.3.4.4.7. It is applied consistently to the
vertically guided RNAV benchmarks supported by this project, including LPV.

This use is intentionally narrower than “landing success” and broader than the
source paragraph's Baro-VNAV equipment context. There is no official universal
LPV threshold-crossing pass/fail number that directly classifies every
commercial landing. The report therefore records both the source and this
claim boundary instead of presenting `22 m` as a legal landing-certification
limit.

## 2. Why `±7.5 m` was removed

The previous rule combined two real but differently scoped facts:

1. the LPV vertical display scale is angular and is minimum-clamped to a
   one-sided `15 m` full-scale value close to the runway; and
2. ICAO Doc 9613, Volume II, Part C, Chapter 5, Section B,
   §5.3.3.1.1.1(b) expects flight technical error to remain within one-half
   full-scale deflection during normal operation.

That value is useful for assessing close-in LPV guidance tracking. It is not an
official universal classifier for the altitude of a trajectory reference point
at the threshold, nor for whether an aircraft landed. ICAO Doc 9613
§§5.3.4.5.7–5.3.4.5.8 also places excessive LPV deviation in an operational
context: the approach is discontinued unless the required visual references
are available. Once visual, flare, and touchdown phases are involved, display
full-scale deflection is not a complete landing-outcome rule.

The repository therefore no longer publishes or computes a `7.5 m` verdict,
secondary grade, legacy fallback, or LPV vertical-FSD field. Derived reports
must be regenerated as schema v4.

## 3. Mathematical definition

### 3.1 Threshold frame

For runway threshold/LTP geodetic position `(phi_0, lambda_0)`, true inbound
runway course `theta`, and a trajectory point `(phi, lambda, h)`, project the
horizontal displacement into a local east/north frame `(E, N)`. Define:

```text
x =  E sin(theta) + N cos(theta)      along-track coordinate
y =  E cos(theta) - N sin(theta)      signed cross-track coordinate
```

The threshold plane is `x = 0`; the inbound side is `x < 0`.

Let:

```text
h_LTP = authoritative runway-threshold elevation
TCH   = published threshold-crossing height for the selected FAS
h_ref = h_LTP + TCH
z     = h_event - h_ref
```

All terms in the subtraction must use the same vertical datum. Observed event
altitude is stored as HAE and explicitly converted before comparison; computed
records use the evaluation record's MSL altitude and must agree with the
authoritative target context.

### 3.2 Event selection

For optimized and predicted trajectories, evaluation uses the terminal state
when it lies within `1 m` of `x = 0`. If the last segment alone brackets the
plane, it interpolates that segment. It never searches or fits an earlier
portion of the path.

For observed trajectories, evaluation consumes the policy-free
`runway-threshold-event-v1` emitted by runway assignment and final-approach
processing:

- a source-valid bracketing pair is interpolated in three dimensions with one
  common interpolation fraction; or
- when reception is right-censored before the threshold, the single robust fit
  that won runway assignment supplies the estimate.

Evaluation does not import or call `fit_final_segment()` and does not refit
ADS-B data. The event contains physical geometry and provenance, never TCH,
acceptance limits, or a verdict.

### 3.3 Lateral rule

The existing lateral design is retained.

For LPV:

```text
B_guidance = 0.5 * published LPV lateral FSD at the LTP
B_runway   = 0.5 * published runway width
B_lateral  = min(B_guidance, B_runway)
lateral pass iff abs(y) <= B_lateral
```

For the explicit RNP APCH LNAV/VNAV fallback:

```text
B_guidance = 0.15 NM = 277.8 m
B_lateral  = min(B_guidance, 0.5 * runway width)
```

The runway term is a project geometric guard: the evaluated reference point
must not be outside the runway edge at the threshold. It does not claim that
all aircraft extremities are inside the pavement.

No `1/3` scale is used. ICAO supplies the one-half-FSD normal tracking factor;
the runway half-width is an independent physical cap.

### 3.4 Vertical rule

For every vertically evaluable LPV or approved LNAV/VNAV context:

```text
B_vertical = 22 m
vertical pass iff -B_vertical <= z <= +B_vertical
```

The boundary is inclusive. `22.0001 m` fails; exactly `22 m` passes.

The LNAV/VNAV fallback is selected only when `baro_vnav_approved` is true and
an authoritative threshold-path reference is available. Otherwise its lateral
component may be evaluated, while vertical and composite results are
`indeterminate`.

### 3.5 Composite verdict

Each component is `pass`, `fail`, or `indeterminate`:

```text
if lateral == fail or vertical == fail: overall = fail
else if lateral == pass and vertical == pass: overall = pass
else: overall = indeterminate
```

For observed data, an unavailable threshold event produces `indeterminate`.
For a solved optimized/predicted trajectory that does not reach or bracket the
threshold, the result is `fail` because reaching the requested terminal state
is part of the solution contract.

Uncalibrated estimator uncertainty is diagnostic only. It neither widens nor
shrinks the bound and cannot silently change a point verdict.

## 4. Why other official values are not used as the gate

### 4.1 FAA Order 8260.58D TCH/WCH

FAA Order 8260.58D, issued 2025-01-15, §1-3-1.f(2)(b) and Table 1-3-1 define
procedure-design threshold-crossing-height/ wheel-crossing-height constraints.
Those values define the nominal path and aircraft-height-group design, not an
allowed positive/negative tracking error for an individual flight. The
evaluator correctly uses published TCH to construct `h_ref`, but does not turn
the TCH range into a deviation tolerance.

### 4.2 FAA landing-distance material

FAA AC 91-79B, §5.2.3 explains that crossing above the planned threshold height
increases landing distance (the AC gives an approximate distance penalty per
additional height). It supports the operational importance of threshold
height, but it does not publish a universal symmetric pass/fail error bound.

FAA AC 20-191, §5.2.10.1.6 contains touchdown-location criteria for certain
CAT II/III ILS/GBAS low-visibility systems. Those criteria concern a different
operation and touchdown region, so they are not substituted for an RNAV/LPV
threshold event.

### 4.3 EASA air-operations rule

The current consolidated EASA Air Operations rule, CAT.OP.MPA.310, requires
operators to establish procedures ensuring a safe threshold-crossing margin.
It does not supply one universal numeric trajectory-error threshold. That
operator-specific obligation cannot be reconstructed from the project records.

### 4.4 Alert limits and obstacle surfaces

SBAS alert limits, integrity containment, OCS geometry, obstacle clearance,
and avionics display scales answer different safety or guidance questions.
They are not interchangeable with this terminal reference-point geometry
check.

## 5. Data audit behind the change

The KMSY investigation separated the standard from the event estimator and
runway assignment:

- `4,097` KMSY records had direct threshold brackets, so their vertical value
  did not come from final-segment fitting;
- their median signed vertical error under the published-TCH reference was
  about `+9.7 m`;
- their median absolute cross-track error was about `1.68 m`, with only six
  outside a `22.86 m` runway half-width; and
- `3,859` later reached a runway-contained sample no more than `10 m` above
  runway elevation.

Thus the very high failure rate under `±7.5 m` was not primarily caused by the
fit model or runway assignment. Directly observed threshold brackets were
already being rejected by a guidance-scale-derived classifier whose claim was
too strong.

A read-only recomputation of the existing five-airport reports with the same
events and lateral limits, changing only the vertical boundary to `±22 m`,
gave:

| Airport | Total | Fail | Fail rate |
|---|---:|---:|---:|
| KMSY | 4,150 | 257 | 6.19% |
| KRDU | 14,439 | 270 | 1.87% |
| KSJC | 11,157 | 7 | 0.06% |
| KSMF | 4,231 | 8 | 0.19% |
| KSTL | 8,769 | 280 | 3.19% |
| Total | 42,746 | 822 | 1.92% |

This is a diagnostic before regeneration, not a target pass rate. Remaining
failures must still be attributed to lateral deviation, vertical event
geometry, source integrity, runway assignment, or terminal-state completion;
the evaluator must not tune the bound airport by airport.

## 6. Report and reproducibility contract

Schema `terminal-approach-evaluation-v4` serializes:

- the threshold event method and provenance;
- runway and procedure source cycles and fingerprints;
- the published TCH reference and resolved lateral/vertical bounds;
- standard id `icao_doc_9613_rnp_apch_fas_22m`;
- exact ICAO source location and the non-certification claim boundary;
- signed along-track, cross-track, and vertical deviations;
- component and composite three-way verdicts; and
- stable `flight_key` plus display callsign and record filename.

Every required numeric input and output must be finite. JSON containing NaN or
infinity is rejected. A physical threshold event from a different runway frame
or data cycle cannot be reused. Changing only evaluation policy permits reuse
of the policy-free physical event, but requires regeneration of the report.

## 7. Authoritative references read

| Source | Current/version check | Key sections used | Local copy |
|---|---|---|---|
| ICAO Doc 9613, *Performance-based Navigation (PBN) Manual* | Fifth Edition, 2023; the edition identified in the current ICAO catalogue during the 2026-08 audit | Vol. II, Part C, Ch. 5, Section A §§5.3.4.4.7–5.3.4.4.8; Section B §§5.3.3.1.1.1(b), 5.3.4.5.7–5.3.4.5.8 | [PDF](../docs/regulation/ICAO_Doc_9613_5th_Ed_2023.pdf) |
| FAA Order 8260.58D, *United States Standard for PBN Instrument Procedure Design* | Active; issued 2025-01-15 | §1-3-1.f(2)(b), Table 1-3-1; final-segment path/TCH material | [PDF](../docs/regulation/Order_8260.58D.pdf) |
| FAA AC 91-79B, *Aircraft Landing Performance and Runway Excursion Mitigation* | Active; issued 2023-08-28 | §5.2.3 | [PDF](../docs/regulation/FAA_AC_91-79B_2023.pdf) |
| FAA AC 20-191, *Airworthiness Approval of Airborne Systems for CAT II/III Operations* | Current FAA copy; issued 2026-05-20 | §5.2.10.1.6; reviewed and rejected as the common RNAV threshold gate | [PDF](../docs/regulation/FAA_AC_20-191_2026.pdf) |
| Commission Regulation (EU) No 965/2012, consolidated Air Operations rules | EASA Easy Access Rules Revision 24, March 2026 | CAT.OP.MPA.310 | [PDF](../docs/regulation/EU_Regulation_965-2012_consolidated_2026-02-22.pdf) |

The RTCA DO-229 full text is licence-controlled and is not redistributed here.
Its LPV scale was reviewed only to explain the retired `7.5 m` derivation; it is
not a dependency of the v4 verdict.

## 8. Acceptance tests

Implementation acceptance requires:

1. exact `±22 m` boundaries pass and the first representable outside values
   fail;
2. LPV and approved LNAV/VNAV contexts resolve the same vertical bound;
3. lateral logic remains unchanged;
4. observed evaluation consumes the serialized event and never refits;
5. unavailable observed events remain `indeterminate`;
6. non-finite values are rejected;
7. reports use schema v4 and contain the source/claim-boundary metadata; and
8. the frontend preserves `pass`, `fail`, and `indeterminate` distinctly.
