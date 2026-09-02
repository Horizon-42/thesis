# Threshold-crossing speed gate — design and sources

**Status:** implemented (report schema `terminal-approach-evaluation-v6`).
**Code:** `evaluation/speed_gate.py` (policy), `evaluation/metrics.py` (composition),
`aircraft/aero_params.stall_speed_ms` (the stall model, single source),
`flight_scenarios/build.py` (the producer-written `source.landing_aero` block).
**Measured results:** `BASELINE_SPEED_GATE_RESULTS.md` (2026-08-24 fleet baseline —
read its §5 before quoting any speed-fail rate: the per-type window anchor, not
weather, dominates the fail structure).

## 1. What the gate claims — and what it does not

The lateral and vertical gates ask **where** the threshold crossing was. This gate asks
whether the aircraft carried a **plausible amount of energy** across it: a trajectory
that reaches the right point at 220 kt — or at 5 kt above stall — is not a flyable
approach, and before v6 it graded `pass`.

The claim is deliberately narrow:

> At the runway-threshold event, the crossing speed must lie inside the stabilized-
> approach window anchored on the **project's own stall model** at the **record's own
> crossing mass**.

It is a *model-consistency / energy* claim about the terminal state of a trajectory. It
is **not** an operational speed check (no wind additives, no gust logic, no company SOP),
and **not** a certification statement about any real aircraft. The report says this in
`methodology.terminal_speed.claim_boundary`.

## 2. The rule

For a record crossing the threshold at mass `m` (kg), with the aircraft's wing area `S`
(m²) and landing-configuration maximum lift coefficient `Cl_max`:

## Stall? lift coefficient is larger then the maximum lift coefficient, then the aircraft is in stall, and the speed gate will fail.

```text
V_s1g  = sqrt(2 m g / (rho0 · S · Cl_max))      # 1-g level stall speed, TAS (m/s)
# Real stall speed; thrust, bank angle;

# validate for 3 airplanes, look up stall speeds is match to the real aircraft; Cessna 53 flaps up, flaps down 48;
# flight envolope; 

V_ref  = 1.23 × V_s1g                            # reference landing speed
gate   : V_ref ≤ V_crossing ≤ V_ref + 20 kt      # inclusive at both edges
```

with `g = 9.81 m/s²`, `rho0 = 1.225 kg/m³` (ISA sea level), `20 kt = 10.289 m/s`.

Every symbol is per-record: `m` and `V_crossing` come from the interpolated crossing
state; `S` and `Cl_max` come from the record's `source.landing_aero` block, written by
the same seam that gave the optimizer its aerodynamics — so the gate judges each record
against the aircraft **the model actually flew**, not against a fleet-wide constant.

## 3. Why each step, with sources

### 3.1 The anchor: a 1-g stall speed (why not a fixed per-type table)

The project already owns exactly one stall model: `AeroParams.Cl_max` +
`V = sqrt(2mg/(ρS·Cl_max))`, used by the casadi dynamics (its stall branch), and by the
optimizer's velocity floor (`scenario_optimization`, `1.10 × V_s`). The gate reuses that
function (`aircraft.aero_params.stall_speed_ms` — moved there in this change so the
optimizer and evaluation import the *same* line of code). Consequences:

- **A solve the optimizer admits and the gate judges share one stall model by
  construction.** A published-vs-implemented mismatch here would manufacture failures
  (or passes) out of a constant, which is precisely the class of bug the lateral
  criterion's history warns about (`evaluation/CLAUDE.md`, the inert-bound postmortem).
- A fixed per-type V_ref table was rejected: the fleet resolves to ~20 distinct
  typecodes per airport via OpenAP (`flight_scenarios/CLAUDE.md`, "Aircraft
  resolution"), there is no authoritative per-type V_ref source covering all of them at
  arbitrary landing mass, and a table would drift from the model's own stall floor.
  Landing V_ref is mass-dependent in reality and in this model; the formula gives that
  for free.

### 3.2 The multiplier: 1.23 × the 1-g stall speed

- **14 CFR §25.125(b)(2)(i)** (transport-category landing rule): "A stabilized
  approach, with a calibrated airspeed of not less than V_REF, must be maintained down
  to the 50 ft height", where "In non-icing conditions, V_REF may not be less than
  **1.23 V_SR0**" (V_SR0 = reference stall speed in the landing configuration).
  https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-25/subpart-B/subject-group-ECFR14f0e2fcc647a42/section-25.125
- **14 CFR §25.103** defines V_SR relative to the 1-g stall speed — which is exactly
  what the model's `V_s1g` is (the speed where `L = W` at `Cl_max`). So `1.23 × V_s1g`
  is the direct model analogue of the regulatory V_REF floor.
  https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-25/subpart-B/section-25.103
- EASA CS-25.125 states the same 1.23 V_SR0 floor, so the anchor is not FAA-specific.
- Historical note: pre-1998 certifications used 1.3 × V_S0 with a minimum-speed
  (0-g-break) stall speed; 1.23 × V_S1g is the modern restatement of the *same
  physical speed* (V_S1g ≈ V_S0/0.94, and 1.3 × 0.94 ≈ 1.22). Either convention lands
  within ~1 kt here; the current regulation's form is used.

### 3.3 The window: [V_REF, V_REF + 20 kt]

- **FSF ALAR Briefing Note 7.1 — "Stabilized Approach"** (Flight Safety Digest,
  Aug–Nov 2000; ALAR Task Force criteria, V1.1 Nov 2000), Table 1 "Recommended
  Elements of a Stabilized Approach", element 3:
  > "The aircraft speed is not more than V_REF + 20 knots indicated airspeed and not
  > less than V_REF."
  https://flightsafety.org/wp-content/uploads/2016/09/alar_bn7-1stablizedappr.pdf
  These criteria apply from the stabilization height (1,000 ft IMC / 500 ft VMC)
  **down to landing** — an approach that leaves the window below that height requires
  a go-around — so the window is valid *at* the threshold, not only at 1,000 ft.
- **FAA AC 91-79B** ("Aircraft Landing Performance and Runway Excursion Mitigation")
  treats excess threshold-crossing speed as a primary overrun factor and quotes a
  threshold-crossing airspeed margin of **+5/−0 kt** around the target for the
  performance data to be valid; Boeing/Airbus FCTM guidance targets V_REF + additives
  (typically +5 kt) at the threshold.
  https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_91-79B_FAA.pdf
  The gate deliberately uses the *wider* ALAR window, not the FCTM ±5 kt target:
  the model flies no wind and no gust additives, so its legitimate crossing speeds
  span [V_REF, V_REF + real-world additive range], and a ±5 kt gate would grade the
  absence of a wind model rather than the trajectory. +20 kt is the widest bound any
  of the cited operational sources call acceptable.
- The window is **inclusive** at both edges, matching the point-estimate rule the
  other two components use (`methodology.uncertainty.verdict_rule`).

### 3.4 Which speed is judged

The record's `V` is the dynamics model's airspeed-equivalent TAS (the model flies in
still air, so TAS = ground speed = inertial speed). The regulatory speeds are CAS.
The gate compares them directly, and states the approximation:

- At this fleet's threshold elevations (1–187 m MSL across KRDU, KSJC, KSMF, KMSY,
  KSTL) the TAS/CAS split is under 1 % (≈ 1.4 kt at worst) — one order below the
  window's 20 kt width. Should a high-elevation airport ever enter the fleet, revisit
  (at 5,000 ft the split is ~8 %). This is also why `rho0` is sea-level ISA: it keeps
  the gate bit-consistent with the optimizer's floor, which uses the same constant.

## 4. Data contract

| Input | Source | Owner |
|---|---|---|
| `V_crossing`, `m` | the interpolated crossing state (`evaluation/arrival.py`, `ArrivalDeviation.crossing_speed_ms` / `crossing_mass_kg`) | evaluation |
| `S`, `Cl_max` | `source.landing_aero = {wing_area_m2, cl_max_landing}` — written by `flight_scenarios.build_scenario` from the same `AeroParams` the optimizer/replay fly | producer |
| 1.23, +20 kt, the formula | `evaluation/speed_gate.py` + `aircraft.aero_params.stall_speed_ms` | evaluation policy / shared model |

The producer-supplies-facts / evaluation-owns-policy split mirrors `hae_minus_msl_m`.
Absent-vs-invalid follows the observed-event pattern:

- **Absent (or explicit null) `landing_aero`** → the speed component is
  `indeterminate` with a named reason, and (for computed subjects) the composite
  verdict is `indeterminate`. Deliberately loud: a gate that silently skips records
  never binds, and *"a bound that cannot change an answer is worse than no bound"*
  (`CLAUDE.md`). Records produced before this change grade indeterminate until
  regenerated — which matches the repo state (no optimizer batch on disk; ts
  checkpoints already stale for other reasons).
- **Present but malformed** (missing key, non-positive, non-finite, not an object) →
  `ValueError`. A producer that wrote *something* wrong is a contract violation, not a
  data gap.

## 5. Subjects and scope

| Subject | Speed gate | Judged quantity |
|---|---|---|
| `optimized` | composed into the verdict | crossing model airspeed (state V at the event) |
| `predicted` | composed into the verdict | same record contract, same crossing interpolation |
| `observed` | **composed into the verdict (2026-08-24)** | fitted crossing **ground speed**, a stated proxy — see below |

**History.** The original v6 design excluded observed subjects entirely (no crossing
speed existed, and ground speed is not airspeed). The owner overrode the exclusion on
2026-08-24 — the whole point of the baseline is to run the SAME three gates the
models run — after the prerequisites were built: the harvest now serializes a fitted
crossing ground speed on every estimated event, and observed records carry their
resolved airframe's `landing_aero` + landing mass (the same identity→OpenAP chain
`build_scenario` uses), so baseline and modeled twins share one set of stall
assumptions.

**The proxy, stated rather than hidden** (`speed_gate.OBSERVED_SPEED_POLICY`,
`OBSERVED_SPEED_CRITERION_ID = …_ground_speed_proxy`, and
`methodology.terminal_speed.observed_proxy_caveat`):

1. The judged value is GROUND speed — wind is unmodelled, and a 10 kt headwind is
   half the 20 kt window, so an observed speed fail can reflect the day's wind
   rather than the flight. Quote observed speed rates with that caveat, and never
   compare them to computed speed rates as if they measured the same quantity —
   the distinct criterion id on every row is what keeps that honest.
2. The value is the censored fit's extrapolation (or the direct bracket's
   interpolation) of ADS-B reported ground speed — a measured-derived estimate,
   not a sample.
3. `crossing_speed_ms` (airspeed) stays `None` on observed rows — no crossing
   airspeed was ever measured; the proxy lives in its own field
   (`crossing_ground_speed_ms`), so the two quantities can never be silently mixed.

An observed record whose airframe cannot be resolved from its icao24 has no stall
window and grades speed-`indeterminate` (loudly, reason named), as does one whose
event fitted no crossing speed; either composes the verdict to indeterminate.

## 6. Worked numbers (landing mass, the model's Cl_max classes)

| Class example | m (kg) | S (m²) | Cl_max | V_s1g | V_ref = 1.23·V_s1g | window |
|---|---|---|---|---|---|---|
| A320-class (MTOW 30–100 t) | 66,300 | 122.6 | 2.7 | 56.6 m/s | 69.7 m/s = 135.4 kt | 135.4 – 155.4 kt |
| A320-class at 60 t | 60,000 | 122.6 | 2.7 | 53.9 m/s | 66.3 m/s = 128.8 kt | 128.8 – 148.8 kt |
| E75L-class at 34 t | 34,000 | 83.5 | 2.7 | 49.1 m/s | 60.5 m/s = 117.5 kt | 117.5 – 137.5 kt |
| B77W-class (MTOW > 100 t) | 251,290 | 436.8 | 2.4 | 62.0 m/s | 76.2 m/s = 148.1 kt | 148.1 – 168.1 kt |

Sanity anchors: real-world A320 V_REF (full flaps, typical landing weight) is
~130–140 kt and B777-300ER V_REF ~140–150 kt — the model windows bracket the
operational numbers, which is what a model-consistency gate needs.

## 7. Known interactions (read before interpreting a batch)

- **The optimizer's velocity floor is *below* the gate's lower bound by design.**
  The floor is `min(1.10 × V_s, V_ref_aircraft)` so that observed touchdown-speed
  targets stay admissible; the gate's lower bound is `1.23 × V_s`. A min-time solve
  that rides its floor near the threshold **can and should fail** the gate — that is
  the gate detecting an unflyably slow (or target-chasing) terminal state, not a
  contradiction. Conversely `fitted_adsb_crossing` / `track_end` targets carry the
  *observed* crossing speed, which is a ground speed; on a headwind day it can sit
  below V_ref, and a solve that faithfully hits it will fail the speed gate. Quote
  speed-gate rates per `target_source`, never pooled.
- **The category-default target V_ref can itself fail the gate — that is a finding,
  not a bug.** OpenAP-resolved aircraft get an approach group by MTOW class
  (`query_aircraft_parameters._default_approach`): everything 5.7–150 t targets
  145 kt. For an E75L-class aircraft the stall-anchored window tops out at
  ~137.5 kt, so a `runway` solve that reaches its commanded 145 kt target will fail
  the speed gate — correctly flagging that a one-size 145 kt V_ref is unrealistically
  fast for light narrow-bodies. The right fix is per-type approach data (e.g. derive
  `reference_speed_kt` as `1.23 × V_s1g(landing_mass)` instead of a class constant),
  recorded as a follow-up in `docs/code-health-followups.md`; absorbing it by widening
  the gate would hide exactly what the gate exists to show.
- The gate judges the **rollout's** crossing state (same state the other gates use),
  so plan-vs-replay drift shows up here too.

## 8. Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Fixed per-type V_ref table (e.g. B738 = 141 kt) | no authoritative source spanning the resolved fleet at arbitrary mass; drifts from the model's own stall physics; mass-independence is wrong in-model |
| FCTM-style V_REF ± 5 kt target | grades the absence of a wind/additive model, not the trajectory |
| Gate observed subjects with a widened window | still measures wind + a non-crossing sample; a wider bound that "usually passes" is an inert bound |
| Density at threshold elevation instead of ρ₀ | breaks bit-consistency with the optimizer floor for < 1 % effect at this fleet's elevations; revisit only with a high-elevation airport |
| Composite-only reporting (no per-component result) | consumers (ts lateral-eligibility precedent) need per-component access; `speed_result` is serialized like `lateral_result` |

## 9. Report surface (v6)

- Per row: `speed_result`, `bounds.speed_criterion` (`vref_1p23_vs1g_to_vref_plus_20kt`),
  `bounds.stall_speed_ms`, `bounds.speed_lower_ms`, `bounds.speed_upper_ms`,
  `deviation.crossing_speed_ms`, `deviation.crossing_mass_kg`; `"speed"` joins
  `violations` on a fail.
- Per batch: `speed_result_counts`, `crossing_speed_ms` spread,
  `methodology.terminal_speed` (criterion, formula, both sources, subject scope,
  claim boundary — self-describing years later, like the vertical block).
- Schema version bumped v5 → v6 in **all four homes** (producer, ts seam import,
  frontend mirror `EVALUATION_REPORT_SCHEMA_VERSION`, fixtures via the constant).
  Every existing on-disk v5 report is stale and must be regenerated before the ts
  pipeline or the frontend will accept it — consistent with the standing "re-roster
  everything" open item.

## 10. References

1. 14 CFR §25.125 "Landing" — V_REF ≥ 1.23 V_SR0; stabilized approach at CAS ≥ V_REF
   to 50 ft. https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-25/subpart-B/subject-group-ECFR14f0e2fcc647a42/section-25.125
2. 14 CFR §25.103 "Stall speed" — V_SR defined from the 1-g stall.
   https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-25/subpart-B/section-25.103
3. FSF ALAR Briefing Note 7.1 "Stabilized Approach", Table 1 element 3 — speed in
   [V_REF, V_REF + 20 kt]; stabilization heights 1,000 ft IMC / 500 ft VMC; go-around
   if unstabilized below. Flight Safety Digest, Aug–Nov 2000.
   https://flightsafety.org/wp-content/uploads/2016/09/alar_bn7-1stablizedappr.pdf
4. FAA AC 91-79B "Aircraft Landing Performance and Runway Excursion Mitigation" —
   threshold-crossing speed margin +5/−0 kt; excess speed/TCH as overrun factors.
   https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_91-79B_FAA.pdf
5. EASA CS-25 (CS-25.125) — the same 1.23 V_SR0 landing reference-speed floor.
   https://www.easa.europa.eu/en/document-library/certification-specifications/cs-25-amendment-28
