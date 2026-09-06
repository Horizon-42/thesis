# Threshold-crossing speed gate — design and sources

**Status:** implemented (report schema `terminal-approach-evaluation-v7`, 2026-09-07: the
lower bound is anchored on the stall speed at the crossing LOAD FACTOR, §3.5; v6 anchored
both bounds on the 1-g stall speed).
**Code:** `evaluation/speed_gate.py` (policy), `evaluation/metrics.py` (composition),
`evaluation/arrival.py` (the crossing state, mass and load factor),
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

For a record crossing the threshold at mass `m` (kg) and load factor `n`, with the
aircraft's wing area `S` (m²) and landing-configuration maximum lift coefficient `Cl_max`:

```text
V_s1g   = sqrt(2 m g / (rho0 · S · Cl_max))      # 1-g level stall speed, TAS (m/s)
V_s(n)  = V_s1g · sqrt(max(n, 1))                # stall speed under the lift n·m·g the
                                                 # manoeuvre demands (Cl_required = Cl_max)
lower   = 1.23 × V_s(n)                          # V_ref at the crossing load factor
upper   = 1.23 × V_s1g + 20 kt                   # V_ref (1 g) + the ALAR additive
gate    : lower ≤ V_crossing ≤ upper             # inclusive at both edges
```

with `g = 9.81 m/s²`, `rho0 = 1.225 kg/m³` (ISA sea level), `20 kt = 10.289 m/s`.

Every symbol is per-record: `m` and `V_crossing` come from the interpolated crossing
state; `n` from the control active over the final rollout step (§3.5); `S` and `Cl_max`
from the record's `source.landing_aero` block, written by the same seam that gave the
optimizer its aerodynamics — so the gate judges each record against the aircraft **the
model actually flew**, not against a fleet-wide constant. The aircraft is stalled when
`Cl_required = n m g / (½ ρ V² S)` exceeds `Cl_max`; the lower bound is that condition
with the 1.23 margin, which is why it must carry `n`.

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

### 3.5 The load factor: why the lower bound moves with `n` and the upper does not

The model's own dynamics (`aerodynamic_model/casadi_simulator.py`) fly
`γ̇ = g (n cos μ − cos γ) / V` and `ψ̇ = g n sin μ / (V cos γ)`, and its stall drag
branch already computes `Cl_required` from `n·m·g`. Until v6 the gate anchored on the
1-g stall speed regardless: a solve crossing in a 25° bank or a pull-up had its lower
bound computed as if it were flying straight and level. Physically the lift the
manoeuvre demands is `n·m·g`, so the speed at which the wing reaches `Cl_max` is
`V_s1g·√n`; a crossing at `1.23·V_s1g` with `n = 2` is *below* its actual stall speed
(`1.41·V_s1g`) and used to pass.

**Measured on the optimizer batches on disk** (2026-09-07; `controls[-1].load_factor`,
stall facts rebuilt from `dynamics_typecode` because those records predate
`landing_aero`):

| batch | graded | n median | n p95 | n max | flips, both bounds ×√n | flips, lower bound only, n ≥ 1 | passing yet V < V_s1g·√n |
|---|---|---|---|---|---|---|---|
| KMSY/runway | 3,439 | 1.010 | 1.015 | 1.231 | 227 | 1 | 0 |
| KMSY/fitted_adsb | 3,308 | 1.010 | 1.015 | 1.128 | 158 | 20 | 0 |
| KMSY/runway_cons | 2,853 | 0.997 | 1.005 | 1.160 | 4 | 0 | 0 |
| KSJC/runway | 3,383 | 1.011 | 1.015 | 1.372 | 192 | 0 | 0 |
| KSTL/runway_cons | 4,485 | 0.996 | 1.040 | 1.142 | 73 | 6 | 0 |

The bulk crosses at `n ≈ 1`; the tail reaches 1.37, i.e. a 17 % higher stall speed —
the whole 20 kt window. Two policy choices follow from the physics and the table:

- **Only the lower bound scales.** `V_REF` is defined by 14 CFR 25.125 against the 1-g
  reference stall speed, and the `+20 kt` upper edge is an energy / overrun criterion
  (FSF ALAR, AC 91-79B), not a stall margin. Scaling both edges with `√n` moved the
  upper edge by a fraction of a knot and flipped 4–227 verdicts per batch, all of them
  records piled up against that edge (§7) — noise, not information.
- **`n` is clamped at 1 for the bound.** A push-over (`n < 1`) at the threshold does not
  make a slow crossing flyable: the flare that follows needs `n ≥ 1`, at which point the
  1-g floor applies again. The row still reports the measured `n`.

**Where `n` comes from** (`ArrivalDeviation.crossing_load_factor` +
`crossing_load_factor_source`):

- Records with `controls` (optimizer solves, control-output ts predictions):
  `controls[-1].load_factor`, source `controls_last_step`. The rollout writes on every
  sample the control that PRODUCED it (`aerodynamic_model/rollout.py`), so the last
  control is the one active over the final step — the segment both the `terminal_state`
  and the `interpolated_threshold` crossings lie on. A record whose controls lack the
  column is malformed and raises.
- Records without `controls` (observed baselines, state-output ts predictions): `n = 1`,
  source `assumed_1g`, declared on every row. A censored observed crossing is a
  straight-line fit (`γ̇ = ψ̇ = 0` by construction) and a measured ADS-B bracket's `γ̇`
  is 25 ft-quantisation noise (`evaluation/CLAUDE.md`), so inverting the kinematics
  would add noise, not a measurement.

`aircraft.aero_params.stall_speed_ms` gained the `load_factor` keyword (default 1.0);
the optimizer's velocity floor still calls it at 1 g, so the "one stall model" property
of §3.1 is unchanged.

## 4. Data contract

| Input | Source | Owner |
|---|---|---|
| `V_crossing`, `m` | the interpolated crossing state (`evaluation/arrival.py`, `ArrivalDeviation.crossing_speed_ms` / `crossing_mass_kg`) | evaluation |
| `n` | `controls[-1].load_factor` when the record carries controls, else 1 g declared (`crossing_load_factor` / `crossing_load_factor_source`, §3.5) | evaluation reads, producer writes the controls |
| `S`, `Cl_max` | `source.landing_aero = {wing_area_m2, cl_max_landing}` — written by `flight_scenarios.build_scenario` from the same `AeroParams` the optimizer/replay fly | producer |
| 1.23, +20 kt, the formula, the `n ≥ 1` clamp | `evaluation/speed_gate.py` + `aircraft.aero_params.stall_speed_ms` | evaluation policy / shared model |

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

| Subject | Speed gate | Judged quantity | `n` for the lower bound (§3.5) |
|---|---|---|---|
| `optimized` | composed into the verdict | crossing model airspeed (state V at the event) | `controls[-1].load_factor` |
| `predicted` | composed into the verdict | same record contract, same crossing interpolation | `controls[-1].load_factor` for control-output models; 1 g declared for state-output ones (`controls == []`) |
| `observed` | **composed into the verdict (2026-08-24)** | fitted crossing **ground speed**, a stated proxy — see below | 1 g declared (`assumed_1g`) |

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

| Class example | m (kg) | S (m²) | Cl_max | V_s1g | V_ref = 1.23·V_s1g | window at n = 1 |
|---|---|---|---|---|---|---|
| A320 family (calibrated, `BASELINE_SPEED_GATE_RESULTS.md` §8) at MLW | 64,500 | 122.6 | 3.0 | 53.0 m/s | 65.2 m/s = 126.7 kt | 126.7 – 146.7 kt |
| 737-class bucket (MTOW 30–100 t) at 60 t | 60,000 | 122.6 | 2.7 | 53.9 m/s | 66.3 m/s = 128.8 kt | 128.8 – 148.8 kt |
| E75L-class at 34 t | 34,000 | 83.5 | 2.7 | 49.1 m/s | 60.5 m/s = 117.5 kt | 117.5 – 137.5 kt |
| B77W-class (MTOW > 100 t) | 251,290 | 436.8 | 2.4 | 62.0 m/s | 76.2 m/s = 148.1 kt | 148.1 – 168.1 kt |

The load factor lifts the lower edge by `√n` (§3.5): +2.5 % at n = 1.05 (a 17° bank),
+4.9 % at 1.10 (25°), +10.9 % at 1.23 (the KMSY maximum), +17.1 % at 1.37 (the KSJC
maximum) — for the A320 row that last case is a 148 kt lower edge, above the 1-g upper
edge, i.e. no speed passes: an aircraft pulling 1.37 g across the threshold has no
stabilized-approach window. Sanity anchors: real-world A320 V_REF (full flaps, typical
landing weight) is ~130–140 kt and B777-300ER V_REF ~140–150 kt — the model windows
bracket the operational numbers, which is what a model-consistency gate needs.

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
- **The upper edge is where `runway`-mode solves pile up.** With the A320 family's
  calibrated `Cl_max` the 64.5 t window is 126.7–146.7 kt and the class-default target
  of 145 kt sits 1.7 kt under the top, so the 15 % "fast" fails in KMSY/runway are
  decided by sub-knot replay drift. Quote fast-fail rates with that margin, and read a
  change in them after any constant tweak as edge sensitivity before reading it as
  flight behaviour.
- **The optimizer batches on disk (70,267 records, 15 batches) predate
  `source.landing_aero` and grade speed-indeterminate to the last record** — their
  three-gate pass count is 0 until the block is backfilled
  (`4dTrajectory/optimization/backfill_landing_aero.py`, same typecode → `AeroParams`
  chain the producer uses) and the reports regenerated.

## 8. Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Fixed per-type V_ref table (e.g. B738 = 141 kt) | no authoritative source spanning the resolved fleet at arbitrary mass; drifts from the model's own stall physics; mass-independence is wrong in-model |
| FCTM-style V_REF ± 5 kt target | grades the absence of a wind/additive model, not the trajectory |
| Gate observed subjects with a widened window | still measures wind + a non-crossing sample; a wider bound that "usually passes" is an inert bound |
| Density at threshold elevation instead of ρ₀ | breaks bit-consistency with the optimizer floor for < 1 % effect at this fleet's elevations; revisit only with a high-elevation airport |
| Composite-only reporting (no per-component result) | consumers (ts lateral-eligibility precedent) need per-component access; `speed_result` is serialized like `lateral_result` |
| Scale both bounds with `√n` | the upper edge is an energy criterion defined at 1 g; measured, it only flips records piled against that edge (§3.5) |
| Relax the lower bound for `n < 1` | a push-over at the threshold precedes a flare that needs `n ≥ 1`; the 1-g floor is the binding one |
| Invert ADS-B kinematics for the observed `n` | 25 ft altitude quantisation makes `γ̇` noise; a declared 1 g is honest, a fitted `n` would be fiction |

## 9. Report surface (v7)

- Per row: `speed_result`, `bounds.speed_criterion`
  (`vref_1p23_vs_at_n_to_vref_1g_plus_20kt`; observed rows carry
  `vref_1p23_vs1g_to_vref_1g_plus_20kt_ground_speed_proxy` — their window IS the 1-g one,
  the id says so),
  `bounds.stall_speed_ms` (1 g), `bounds.stall_speed_at_n_ms`, `bounds.speed_lower_ms`,
  `bounds.speed_upper_ms`, `deviation.crossing_speed_ms`, `deviation.crossing_mass_kg`,
  `deviation.crossing_load_factor`, `deviation.crossing_load_factor_source`; `"speed"`
  joins `violations` on a fail.
- Per batch: `speed_result_counts`, `crossing_speed_ms` / `crossing_ground_speed_ms`
  spreads, `crossing_load_factor` (`mean/min/p95/max`, `below_1g` = rows whose lower
  bound was clamped to the 1-g floor, `assumed_1g` = rows judged at a declared 1 g),
  `methodology.terminal_speed` (criterion, formula, the load-factor rule and its sources,
  subject scope, claim boundary — self-describing years later, like the vertical block).
- An EMPTY window (`speed_lower_ms > speed_upper_ms`, n above ~1.25–1.40 on this fleet)
  is a verdict, not an error: every speed fails and both bounds on the row show why
  (`SpeedGateBounds.empty`). Measured n on the optimizer batches tops out at 1.37, so it
  is reachable; it has not yet occurred.
- Schema version v6 → v7 in **all four homes** (producer, ts seam, frontend mirror
  `EVALUATION_REPORT_SCHEMA_VERSION`, fixtures via the constant). The ts seam
  (`lateral_eligibility`) reads only `lateral_result`, which v6 and v7 share, so it
  imports `READABLE_REPORT_SCHEMA_VERSIONS = (v6, v7)` and keeps accepting the v6
  reports on disk; the frontend displays v6 with a "graded at 1 g" note and v5 as
  pre-speed-gate. v5 → v6 (2026-08-24) had the same four-home discipline.

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
