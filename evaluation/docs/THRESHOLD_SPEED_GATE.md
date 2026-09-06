# Threshold-crossing speed gate — design and sources

**Status:** implemented (report schema `terminal-approach-evaluation-v9`, 2026-09-07: the
window is anchored on the type's PUBLISHED approach speed, §3.1; v6–v8 anchored on the
project's own stall model — v7 added the crossing load factor, §3.5; v8 measured the
observed load factor and added the METAR headwind correction, §3.6).
**Code:** `evaluation/speed_gate.py` (policy), `evaluation/metrics.py` (composition),
`evaluation/arrival.py` (the crossing state, mass and load factor),
`aircraft/reference_speeds.py` + `aircraft/reference_speeds.json` (the published table),
`docs/reference_speeds/README.md` (provenance of every number in it),
`flight_scenarios/build.py` / `trajectory_data_process/harvest/observed.py` (the producers
that name the record's aircraft type).
**Measured results:** `BASELINE_SPEED_GATE_RESULTS.md` (§1–§9 the v6–v8 stall-anchored
measurements and why they were the window's error; §10 the published-window result).

## 1. What the gate claims — and what it does not

The lateral and vertical gates ask **where** the threshold crossing was. This gate asks
whether the aircraft carried a **plausible amount of energy** across it: a trajectory
that reaches the right point at 220 kt — or at 5 kt above stall — is not a flyable
approach, and before v6 it graded `pass`.

The claim is deliberately narrow:

> At the runway-threshold event, the crossing speed must lie inside the stabilized-
> approach window anchored on the **approach speed the aircraft's type publishes**,
> scaled to the **record's own crossing mass** (or, for an observed flight whose mass is
> not measured, spanning the type's published mass range).

It is a claim about the terminal state of a trajectory against a documented landing
speed. It is **not** an operational speed check (no wind additives, no gust logic, no
company SOP), and **not** a certification statement about any real aircraft. The report
says this in `methodology.terminal_speed.claim_boundary`. The verdict is **pass or
fail** against a determinate window; `indeterminate` means only that nothing could be
judged (no type on the record, no published entry, no published minimum mass for an
observed row, no crossing speed) and the reason is on the row.

## 2. The rule

For a record of ICAO type `T` crossing the threshold at load factor `n`, with the
type's published approach speed at its Maximum Allowable Landing Weight — `V_min` (the
lowest landing flap-configuration value), `V_max` (the highest; equal when the type
publishes one value) — and the masses `MALW` and `m_min` (the type's lowest published
operating mass):

```text
V_ref,lo(m) = V_min · sqrt(m / MALW)             # published speed scaled to mass m
V_ref,hi(m) = V_max · sqrt(m / MALW)

computed record, crossing mass m known:
    lower = V_ref,lo(m)     · sqrt(max(n, 1))     # V_ref under the lift n·m·g
    upper = V_ref,hi(m)     + 20 kt               # V_ref (1 g) + the ALAR additive
observed record, mass unmeasured (the type's published mass range):
    lower = V_ref,lo(m_min) · sqrt(max(n, 1))
    upper = V_ref,hi(MALW)  + 20 kt
gate    : lower ≤ V_crossing ≤ upper             # inclusive at both edges
```

with `20 kt = 10.289 m/s`. The square-root law is the one physical step: at a constant
lift coefficient `V² ∝ m`, so a speed quoted at one weight gives the speed at any
other — the same law `aircraft.aero_params.stall_speed_ms` embodies, applied to a
published anchor instead of a modelled `Cl_max`.

Every symbol is per-record: `m` and `V_crossing` come from the interpolated crossing
state; `n` from the control active over the final rollout step, or from the observed
flight's own kinematics (§3.5); `T` from `source.dynamics_typecode` (the type the model
flew, written by `flight_scenarios.build`) or `source.aircraft_type` (the resolved
airframe, written by the harvest's observed writer); `V_min`, `V_max`, `MALW`, `m_min`
from `aircraft/reference_speeds.json`, every entry of which cites the document it was
read from (§3.1).

## 3. Why each step, with sources

### 3.1 The anchor: the type's published approach speed (why not the stall model)

Versions v6–v8 anchored the window on the project's own stall model — OpenAP's wing
area with a landing `Cl_max` assigned by MTOW bucket (2.7 for 30–100 t, 3.0 for the
A320 family after a calibration) — on the argument that the optimizer's velocity floor
used the same function, so admitted solves and judged solves shared one stall model.
That argument was sound for the model twins and wrong for the ground truth: the
observed fleet all landed, and under the wind-corrected v8 gate 26 % of it still
"failed" speed, clustered by airframe family (`BASELINE_SPEED_GATE_RESULTS.md` §9).
The 737 bucket's `1.23·Vs1g(MLW)` came out at 134 kt for a 737-800 whose published
approach speed is 140–147 kt; the failures were the anchor's, not the flights'.

The anchor is therefore the approach speed each type **publishes**, kept in
`aircraft/reference_speeds.json` with a source id on every number and the documents
themselves downloaded and indexed in `docs/reference_speeds/README.md` (URL, retrieval
date, SHA-256, page). Per type:

- **`approach_speed_kt`** — the FAA Office of Airports *Aircraft Characteristics
  Database* ("Aircraft Characteristics (October 2024)") column `Approach_Speed_knot`:
  "Approach Speed to the runway at Maximum Allowable Landing Weight (MALW)", the highest
  Flight Standardization Board / manufacturer value over the landing flap
  configurations, with `Approach_Speed_minimum/maximum_knot` where the FSB gives dual
  flap-configuration values (AC 150/5300-13B definitions). It covers every ICAO type in
  this fleet, bizjets and the C172 included, and is the airport-design authority's own
  per-type landing speed.
- **`malw_kg`** — the `MALW_lb` the speed is quoted at (one FAA cell, B739, holds the
  kilogram figure and is replaced by the Boeing ACAP value, noted in the table).
- **`min_mass_kg`** — the type's lowest PUBLISHED operating mass, for the observed
  window's lower edge: the manufacturer's minimum flight weight where it publishes one
  (CRJ900 APM), else its OEW/BOW from the airport planning document (Boeing ACAP 2.1,
  Embraer APM Table 2.1), else OpenAP 2.4's OEW, each cited; a type with none reads
  `null` and its observed rows grade indeterminate with that reason.
- Corroboration, recorded but not used as the anchor: the manufacturers' own statements
  (Airbus AC 3-5-0 "Final Approach Speed" at MLW; the CRJ900 APM's V_ref-vs-weight
  chart) and the Eurocontrol Aircraft Performance Database `Vat`, which agree with the
  FAA values within the flap-configuration spread.

Why the published speed and not `1.23 × V_SR0` from certification: the certified
`V_SR0` is not published per type, while the FSB approach speed is — and it *is*
`V_REF` at MALW in the landing configuration, which is the quantity 14 CFR 25.125
defines and FSF ALAR bounds. The stall model stays what it was for the optimizer (its
velocity floor, `scenario_optimization._stall_speed_ms`) and is no longer read by the
gate; `source.landing_aero`, which carried its inputs onto records, is no longer read
by evaluation either.

### 3.2 What the published speed is (and the two flap values)

- **FAA AC 150/5300-13B** defines the Aircraft Approach Category from "the highest
  reported Approach Speed/Stall Speed in Flight Standardization Board (FSB) or aircraft
  manufacturer's documentation" at MALW; the Aircraft Characteristics Database carries
  that speed per ICAO type and, for aircraft with dual values, the minimum (optimum
  landing flap) and maximum (a reduced landing flap) approach speeds. Both matter: an
  operator's flap choice is not on the record, so the window's lower edge uses the
  lower value and its upper edge the higher — e.g. B738 140/144 kt, CRJ9 132/141 kt.
  **Stated approximation:** both values are scaled from the row's one MALW; where the
  FSB quotes the lower value at a lower landing weight (A321: 140 kt at 75,500 kg and
  142 kt at 77,800 kg) the lower edge comes out ~1.5 % low — permissive, noted in the
  row and in `methodology.terminal_speed.reference_speeds.stated_approximation`.
- **14 CFR §25.125(b)(2)(i)** is the definition behind the published value: V_REF may
  not be less than 1.23 V_SR0, and the FSB approach speed is the certified V_REF at
  MALW in the landing configuration. The gate no longer computes the 1.23 — it reads
  the speed the regulation produced.
  https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-25/subpart-B/subject-group-ECFR14f0e2fcc647a42/section-25.125
- The manufacturers state the same quantity: Airbus AC 3-5-0 "the indicated airspeed
  at threshold in the landing configuration, at the certificated maximum flap setting
  and Maximum Landing Weight" (A320-200 136 kt at 66,000 kg — the FAA row reads 136 kt
  at 145,505 lb); Bombardier's CRJ900 APM 00-03-03 charts "Landing Speed – VREF
  (KIAS), flaps 45" against gross weight, and its curve follows `sqrt(m)`.

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
  (at 5,000 ft the split is ~8 %). The published speeds are indicated/calibrated
  airspeeds at sea level, so no density term enters the window.

### 3.5 The load factor: why the lower bound moves with `n` and the upper does not

The model's own dynamics (`aerodynamic_model/casadi_simulator.py`) fly
`γ̇ = g (n cos μ − cos γ) / V` and `ψ̇ = g n sin μ / (V cos γ)`, and its stall drag
branch already computes `Cl_required` from `n·m·g`. Until v6 the gate anchored on the
1-g speed regardless: a solve crossing in a 25° bank or a pull-up had its lower
bound computed as if it were flying straight and level. Physically the lift the
manoeuvre demands is `n·m·g`, so the speed at which the wing reaches a given lift
coefficient is the 1-g speed times `√n`; a crossing at `V_ref` with `n = 2` is *below*
its actual stall speed (`1.41·V_s1g > 1.23·V_s1g`) and used to pass. The same law
applies to the published `V_ref` (it is `1.23 V_SR0` at 1 g): the lower edge is
`V_ref,lo · √max(n, 1)`.

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
  upper edge by a fraction of a knot and flipped 4–227 verdicts per batch (measured
  under the v7 stall anchor), all of them records piled up against that edge — noise,
  not information.
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
- Observed baselines: `n` is MEASURED from the flight's own kinematics
  (`arrival._observed_load_factor`, source `adsb_kinematics`): ψ and γ are fitted
  linearly over the final 20 s of measured track before the crossing (the samples' own
  V/ψ/γ are already windowed least-squares velocities), and
  `aircraft.kinematics.load_factor_from_rates` applies the point-mass inversion. On a
  3° final at 70 m/s the 25 ft altitude quantum costs ~0.002 in `n` over such a window
  and the heading fit ~0.005 — a measurement, where a per-sample rate would be noise.
  The window's length, sample count, fitted rates and how far before the threshold the
  measured track ended (`0` for a measured bracket, the fit's extrapolation for a
  censored track, whose appended crossing row is a straight line with zero rates by
  construction) are on the row (`crossing_load_factor_window`). Fewer than four samples
  in the window ⇒ `assumed_1g` with the reason on the row. The rotating-frame transport
  terms (~1e-5 rad/s, 7e-5 in `n`) are not applied.
- State-output ts predictions carry no controls: `n = 1`, source `assumed_1g`, declared
  on every row.

(`aircraft.aero_params.stall_speed_ms` carries the same `load_factor` keyword for the
optimizer's own use; the gate applies `√n` to the published speed directly.)

**Measured on the observed fleet (v8, 2026-09-07; `crossing_load_factor_window` on
every row):**

| airport | measured rows | n measured / declared | n p50 | p95 | p99 | max | n > 1.01 | n > 1.05 | window p50 | ends before threshold p50 | verdicts moved vs 1 g |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KRDU | 14,438 | 14,423 / 15 | 1.0022 | 1.0158 | 1.0219 | 1.122 | 2,382 | 11 | 18 samples | 298 m | 84 |
| KSJC | 11,151 | 11,150 / 1 | 1.0035 | 1.0116 | 1.0172 | 1.084 | 963 | 3 | 20 | 0 m | 59 |
| KSTL | 8,765 | 8,762 / 3 | 1.0048 | 1.0189 | 1.0241 | 1.067 | 2,098 | 1 | 17 | 0 m | 37 |
| KSMF | 4,229 | 4,229 / 0 | 1.0025 | 1.0095 | 1.0133 | 1.049 | 168 | 0 | 20 | 0 m | 17 |
| KMSY | 4,149 | 4,146 / 3 | 1.0037 | 1.0156 | 1.0217 | 1.041 | 698 | 0 | 17 | 0 m | 14 |

Real approaches cross at 1 g to within a percent — the assumption v7 made is now a
measurement, and it holds. (The 211 verdicts that moved under the v8 stall anchor were
all pass → fail with `n` between 1.003 and 1.017 lifting a floor those flights sat
within a knot of; under the published window, §10 of the results document, the floor
is nowhere near them.)

### 3.6 Wind: the METAR headwind correction for observed baselines

The observed crossing speed is a GROUND speed and the window is an airspeed window;
an ordinary 10 kt headwind is half the window, so a proxy-judged fail can be the
day's weather. OpenSky state vectors carry neither airspeed nor true heading (no wind
triangle), so the wind comes from the field itself: the ASOS/METAR reports the IEM
archive republishes (routine hourly plus specials, direction degrees TRUE, speed and
gust in knots, UTC — `trajectory_data_process/metar/fetch_iem_asos.py` fetches them
into `data/metar/<ICAO>/`, with provenance beside each file). `evaluation.wind` joins
each flight to the report nearest its landing time:

```text
headwind          = W · cos(direction_from − runway course)     # both degrees true
airspeed estimate = crossing ground speed + headwind
```

judged in the same window under `…_metar_airspeed_estimate`. The correction is a
deterministic number with stated limits (the tower's 10 m wind is not the threshold
wind; hourly sampling; gusts not applied) — it is not an uncertainty model, and the
verdict stays pass/fail (v8 carried a declared ±5 kt and a `speed_marginal` count; v9
dropped them: a three-valued verdict was judged the wrong answer to a wrong window).
Every speed-graded row carries `speed_margin_ms`, the judged value's signed distance to
the nearest bound (positive inside the window), so a residual fail can be read off the
row; the batch splits `speed_result_counts` by criterion so estimate-judged and
proxy-judged rows are never pooled. A report older than 30 min or a
variable-direction wind yields no estimate: the row is judged on the ground-speed
proxy under its own id and `wind.status` says why (on this fleet the dominant cause is
a variable wind, not report age); the batch reports `wind_counts`. Nothing on disk is
rewritten — the join is at read time, like the altitude repair and the arrival-window
slice. The harvest's own `--evaluate-only` evaluation and every evaluation CLI take
`--metar-root` (default `data/metar`).

## 4. Data contract

| Input | Source | Owner |
|---|---|---|
| `V_crossing`, `m` | the interpolated crossing state (`evaluation/arrival.py`, `ArrivalDeviation.crossing_speed_ms` / `crossing_mass_kg`) | evaluation |
| `n` | `controls[-1].load_factor` when the record carries controls; the ADS-B kinematic inversion over the final 20 s on observed baselines; else 1 g declared (`crossing_load_factor` / `_source` / `_window`, §3.5) | evaluation reads, producer writes the controls / the track |
| wind | `data/metar/<ICAO>/*.csv` (IEM ASOS archive, `fetch_iem_asos`), joined by `source.landing_time_utc` and the context's runway course (§3.6) | evaluation reads at evaluation time |
| `T` (ICAO type) | `source.dynamics_typecode` — the type the model flew, written by `flight_scenarios.build_scenario` and copied by both computed producers; `source.aircraft_type` — the resolved airframe, written by `harvest/observed.py` (`speed_gate.TYPECODE_KEYS`) | producer |
| `V_min`, `V_max`, `MALW`, `m_min` | `aircraft/reference_speeds.json` (`aircraft.reference_speeds`), one source id per number; documents under `data/reference_speeds/`, index `docs/reference_speeds/README.md` | evaluation policy (the table is curated, not fitted) |
| +20 kt, the `√(m/MALW)` law, the `n ≥ 1` clamp, the two mass bases | `evaluation/speed_gate.py` + `aircraft.reference_speeds.ReferenceSpeed.vref_kt` | evaluation policy |

The producer-supplies-facts / evaluation-owns-policy split mirrors `hae_minus_msl_m`:

- **No type on the record** → speed `indeterminate` with `NO_TYPECODE_REASON`
  (computed) or `OBSERVED_UNRESOLVED_AIRFRAME_REASON` (observed); the composite is
  indeterminate. Deliberately loud: a gate that silently skips records never binds.
- **A type with no entry in the table**, or an observed row whose type publishes no
  minimum mass → `indeterminate`, the type NAMED in the reason, and the batch's
  `speed_indeterminate_reasons` counts it — coverage of the table is a stated fact,
  never a silent skip. Adding a type means adding a cited row to the JSON and its
  document to the README, nothing in code.
- **A malformed table row** (a speed that is not min ≤ main ≤ max, a minimum mass at or
  above MALW, an unlisted source id) raises at load: a curated fact that cannot be
  trusted is a contract violation, not a data gap.
- `source.landing_aero` (v6–v8's stall inputs) is neither read nor validated any more;
  `flight_scenarios.build` still writes it as provenance of the model's own velocity
  floor, the harvest's observed writer no longer does.

## 5. Subjects and scope

| Subject | Speed gate | Judged quantity | `n` for the lower bound (§3.5) |
|---|---|---|---|
| `optimized` | composed into the verdict | crossing model airspeed (state V at the event) | `controls[-1].load_factor` |
| `predicted` | composed into the verdict | same record contract, same crossing interpolation | `controls[-1].load_factor` for control-output models; 1 g declared for state-output ones (`controls == []`) |
| `observed` | **composed into the verdict (2026-08-24)** | fitted crossing ground speed **+ METAR headwind** = airspeed estimate when a report is usable, else the raw **ground speed** as a stated proxy — see below and §3.6; window over the type's published mass range (`mass_basis = type_mass_range`) | measured from the final 20 s of ADS-B kinematics (`adsb_kinematics`); `assumed_1g` only when the window is too short |

**History.** The original v6 design excluded observed subjects entirely (no crossing
speed existed, and ground speed is not airspeed). The owner overrode the exclusion on
2026-08-24 — the whole point of the baseline is to run the SAME three gates the
models run — after the prerequisites were built: the harvest now serializes a fitted
crossing ground speed on every estimated event, and observed records carry their
resolved airframe's ICAO type (`aircraft_type`, the same identity chain
`build_scenario` uses), which is what the published table is keyed on.

**Why the observed window spans the type's mass range.** An ADS-B track carries no
mass; the harvest's record mass is an assumed airframe landing mass, not a
measurement. A window framed at an assumed mass would fail real flights for being
lighter or heavier than the assumption. The determinate statement that holds for every
correctly flown landing of the type is: not below `V_ref` at the lightest mass the type
can fly at, not above `V_ref` at its maximum landing weight plus the ALAR additive. For
a 737-800 that is 116–167 kt CAS (`m_min` 41,412 kg OEW, MALW 66,350 kg, 140/144 kt);
wide, because the unknown is wide, and every crossing outside it is a genuine anomaly
or a data error, explainable row by row. Computed records know their mass and get the
20 kt (plus flap-spread) window at it.

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

An observed record whose airframe cannot be resolved from its icao24 has no window
and grades speed-`indeterminate` (loudly, reason named), as does one whose type has no
published entry or minimum mass, or whose event fitted no crossing speed; each
composes the verdict to indeterminate and is counted in
`speed_indeterminate_reasons`.

## 6. Worked numbers (the published table)

| Type | published V (kt, lo / hi) | MALW (kg) | m_min (kg, kind) | computed window at 60 t, n = 1 | observed window (type mass range), n = 1 |
|---|---|---|---|---|---|
| A320 | 136 / 136 | 66,000 (145,505 lb) | 42,600 (OEW, OpenAP 2.4) | 129.7 – 149.7 kt = 66.7 – 77.0 m/s | 109.3 – 156.0 kt |
| B738 | 140 / 144 | 66,350 (146,275 lb) | 41,412 (OEW, Boeing ACAP 2.1.3) | 133.1 – 156.9 kt | 110.6 – 164.0 kt |
| E75L | 126 / 126 | 34,000 (74,957 lb) | 21,500 (BOW, Embraer APM 2.1) | at 30 t: 118.4 – 138.4 kt | 100.2 – 146.0 kt |
| CRJ9 | 132 / 141 | 33,340 (73,500 lb) | 20,412 (MFW, CRJ900 APM 00-02-01) | at 30 t: 125.2 – 153.8 kt | 103.3 – 161.0 kt |

(`V(m) = V · sqrt(m / MALW)`; the observed lower edge is the lo value at `m_min`, the
observed upper edge the hi value at MALW plus 20 kt.) The load factor lifts the lower
edge by `√n` (§3.5): +2.5 % at n = 1.05 (a 17° bank), +4.9 % at 1.10 (25°), +10.9 %
at 1.23, +17.1 % at 1.37 — for the A320 computed row that last case is a 151.9 kt
lower edge, above the 149.7 kt upper edge, i.e. no speed passes: an aircraft pulling
1.37 g across the threshold has no stabilized-approach window. Under the v8 stall
anchor the same A320 row read 128.8–148.8 kt and the B738 row 128.8–148.8 kt too (one
bucket for both), which is the 737 family's "fast" cluster in one line.

## 7. Known interactions (read before interpreting a batch)

- **The optimizer's velocity floor is *below* the gate's lower bound by design.**
  The floor is `min(1.10 × V_s, V_ref_aircraft)` on the project's stall model so that
  observed touchdown-speed targets stay admissible; the gate's lower bound is the
  published `V_ref` at the crossing mass, above that floor. A min-time solve
  that rides its floor near the threshold **can and should fail** the gate — that is
  the gate detecting an unflyably slow (or target-chasing) terminal state, not a
  contradiction. Conversely `fitted_adsb_crossing` / `track_end` targets carry the
  *observed* crossing speed, which is a ground speed; on a headwind day it can sit
  below V_ref, and a solve that faithfully hits it will fail the speed gate. Quote
  speed-gate rates per `target_source`, never pooled.
- **The category-default target V_ref can itself fail the gate — that is a finding,
  not a bug.** OpenAP-resolved aircraft get an approach group by MTOW class
  (`query_aircraft_parameters._default_approach`): everything 5.7–150 t targets
  145 kt. For an E75L at 30 t the published window tops out at 138.4 kt, so a
  `runway` solve that reaches its commanded 145 kt target will fail the speed gate —
  correctly flagging that a one-size 145 kt V_ref is unrealistically fast for light
  narrow-bodies. The right fix is to feed the optimizer's target the same published
  table (`aircraft.reference_speeds`), recorded as a follow-up in
  `docs/code-health-followups.md`; absorbing it by widening the gate would hide
  exactly what the gate exists to show.
- The gate judges the **rollout's** crossing state (same state the other gates use),
  so plan-vs-replay drift shows up here too.
- **The upper edge is where `runway`-mode solves pile up.** The class-default target
  of 145 kt sits within a few knots of the published upper edge for the A320 family at
  its typical scenario mass (149.7 kt at 60 t, 154.6 kt at MLW), so "fast" fails in
  `runway` batches can be decided by sub-knot replay drift. Quote fast-fail rates with
  that margin, and read a change in them after any table edit as edge sensitivity
  before reading it as flight behaviour.
- **The optimizer batches on disk (70,267 records, 15 batches) carry
  `source.dynamics_typecode`**, which is all the v9 gate needs — their v6 reports
  (speed-indeterminate to the last record, for want of `landing_aero`) become gradable
  by regenerating the reports (`run_all_evaluations.py`), no backfill and no re-solve.

## 8. Rejected alternatives

| Alternative | Why rejected |
|---|---|
| The project's own stall model as the anchor (v6–v8: OpenAP wing area × a bucketed landing Cl_max) | not a published number for any type; on the ground truth it failed 26 % of flights that all landed, by airframe family (`BASELINE_SPEED_GATE_RESULTS.md` §9) — the anchor's error, not the flights' |
| A mass-independent per-type V_ref (e.g. B738 = 141 kt flat) | V_ref is mass-dependent in reality and in the model; the published pair (speed, MALW) plus `√(m/MALW)` keeps the dependence at no cost |
| A per-type anchor CALIBRATED on the observed corrected airspeeds ("plan B") | circular as a ground-truth test; kept only as the fallback for a type with no published entry, which the current fleet does not need (all 40 types are in the FAA table) |
| A three-valued verdict (pass / marginal / fail) with a declared ±5 kt on the estimate | the evaluation is the foundation every model result is judged on; a probabilistic verdict defers the question instead of answering it — the fix was the window, not the verdict (owner decision 2026-09-07) |
| An observed window framed at the harvest's assumed airframe mass | the flight's mass is not measured; framing at an assumption fails real flights for being lighter or heavier than assumed — the type's published mass range is the determinate statement |
| FCTM-style V_REF ± 5 kt target | grades the absence of a wind/additive model, not the trajectory |
| Gate observed subjects with a widened window | still measures wind + a non-crossing sample; a wider bound that "usually passes" is an inert bound |
| Composite-only reporting (no per-component result) | consumers (ts lateral-eligibility precedent) need per-component access; `speed_result` is serialized like `lateral_result` |
| Scale both bounds with `√n` | the upper edge is an energy criterion defined at 1 g; measured, it only flips records piled against that edge (§3.5) |
| Relax the lower bound for `n < 1` | a push-over at the threshold precedes a flare that needs `n ≥ 1`; the 1-g floor is the binding one |
| Invert ADS-B kinematics for the observed `n` | 25 ft altitude quantisation makes `γ̇` noise; a declared 1 g is honest, a fitted `n` would be fiction |

## 9. Report surface (v9)

- Per row: `speed_result`, `bounds.speed_criterion`
  (`published_vref_at_crossing_mass_and_n_to_vref_plus_20kt` on computed rows;
  `published_vref_over_type_mass_range_and_n_to_vref_plus_20kt` plus
  `_metar_airspeed_estimate` or `_ground_speed_proxy` on observed rows),
  `bounds.reference_typecode`, `bounds.reference_sources` (the table's source ids for
  the speed, the MALW and the minimum mass — each row traces to its documents),
  `bounds.mass_basis` (`crossing_mass` / `type_mass_range`), `bounds.vref_low_ms`,
  `bounds.vref_high_ms` (the published speeds at the framing masses),
  `bounds.speed_lower_ms` (the low one after `√n`), `bounds.speed_upper_ms`,
  `speed_margin_ms`, `speed_reason` (why the speed component is indeterminate, on the
  row itself — readable when the composite is a lateral/vertical fail),
  `deviation.crossing_speed_ms`,
  `deviation.crossing_mass_kg`, `deviation.crossing_load_factor` (+ `_source`,
  `_window`), `crossing_airspeed_estimate_ms` + `wind` on observed rows; `"speed"` joins
  `violations` on a fail; an indeterminate speed names its reason in `reason`.
- Per batch: `speed_result_counts`, `speed_result_counts_by_criterion`,
  `speed_indeterminate_reasons` (count per cause over the SAME rows as the counts —
  unmeasured crossings included — so the reasons sum to the indeterminate tally),
  `wind_counts`,
  `crossing_speed_ms` / `crossing_ground_speed_ms` / `crossing_airspeed_estimate_ms`
  spreads, `crossing_load_factor` (`mean/min/p95/max`, `below_1g`, `adsb_kinematics`,
  `assumed_1g`), `methodology.terminal_speed` (criterion, formula, the table's
  definition and provenance, the two mass bases, the load-factor rule, sources, subject
  scope, claim boundary — self-describing years later, like the vertical block).
- Dropped from v8: `speed_uncertainty_ms`, `speed_marginal`,
  `speed_uncertainty_unknown`, `bounds.stall_speed_ms`, `bounds.stall_speed_at_n_ms`.
- An EMPTY window (`speed_lower_ms > speed_upper_ms`, n above ~1.25–1.40 on this fleet)
  is a verdict, not an error: every speed fails and both bounds on the row show why
  (`SpeedGateBounds.empty`).
- Schema version v8 → v9 in **all four homes** (producer, ts seam, frontend mirror
  `EVALUATION_REPORT_SCHEMA_VERSION`, fixtures via the constant). The ts seam
  (`lateral_eligibility`) reads only `lateral_result`, which every version since v6
  shares, so it imports `READABLE_REPORT_SCHEMA_VERSIONS = (v6, v7, v8, v9)`; the
  frontend displays v6–v8 with a "stall-anchored" note and v5 as pre-speed-gate.

## 10. References

0. FAA Office of Airports, Airports Planning and Environmental Division — *Aircraft
   Characteristics Database*, "Aircraft Characteristics (October 2024)" (xlsx; approach
   speed at MALW per ICAO type, FSB-validated; definitions sheet). Downloaded and
   hashed: `docs/reference_speeds/README.md`.
   https://www.faa.gov/airports/engineering/aircraft_char_database
0a. FAA AC 150/5300-13B "Airport Design" — Aircraft Approach Category and approach
   speed definitions the database follows.
   https://www.faa.gov/airports/resources/advisory_circulars/index.cfm/go/document.current/documentnumber/150_5300-13
0b. Manufacturer airport planning documents (Airbus A319/A320/A321 AC 3-5-0 "Final
   Approach Speed"; Boeing 737NG/737 MAX/757/767/777/787 ACAP §2.1 weights; Embraer 175
   APM Table 2.1; Bombardier CRJ900 APM CSP C-020 00-02-01 and 00-03-03) and the
   Eurocontrol Aircraft Performance Database — corroboration and minimum masses; each
   listed with URL, retrieval date and SHA-256 in `docs/reference_speeds/README.md`.
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
