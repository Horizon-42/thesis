# flight_scenarios — who gets into a dataset (reference)

Full text behind the index lines of `flight_scenarios/CLAUDE.md`, moved here VERBATIM on 2026-09-16 so that
file stays a short index (it is injected into every session that touches the tree). Only the
`### <ID>` headings and the dated **Correction / Note** paragraphs are new; every heading has
exactly one index line in that CLAUDE.md ending in its ID (`grep -n '^### # ·' flight_scenarios/docs/population_reference.md`).

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.**

## Population: who gets into a dataset

### FS1 · `build_scenarios_from_arrivals` is strict; `dataset.build_scenario_dataset` is the batch layer

- **`build_scenarios_from_arrivals` is strict; `dataset.build_scenario_dataset` is the batch
  layer.** The strict builder raises on any flight it cannot build, which is right for a
  library and wrong for a batch: **35 flights of the 42,725 rostered arrivals (0.08 %) have
  no usable fitted final approach** (KMSY 1, KRDU 1, KSJC 25, KSMF 8, KSTL 0) and aborted the
  fitted-ADS-B dataset for 4 of the 5 airports. **That count is against the 2026-08-19 roster
  generation; the roster is now 42,650 and the 35 has NOT been re-measured** — the numerator
  needs an uncapped rebuild, so quote it as a ~0.08 % rate, not as a current tally. What IS
  current: over the capped selections on disk, 20 flights are dropped (KMSY 1, KRDU 0, KSJC 13,
  KSMF 6, KSTL 0), same shape. They now raise `UnusableFittedApproach` (its own type, so a
  broad `except ValueError` cannot swallow a real contract violation next to it) and the
  batch layer drops and NAMES them.

### FS2 · `--max-per-runway` caps the population, and the cap is written down

- **`--max-per-runway` caps the population, and the cap is written down.** Default 2000 via
  `prepare_scenario_inputs.py`; at that value the fleet is 23,429 flights / 70,287 solves
  instead of 42,650 / 127,950 (both re-read off the `.selection.json` files on disk
  2026-09-06; this used to say 23,453 / 70,359 against 42,725 / 128,175). Selection is evenly
  spaced over landing time within each runway, so a capped runway still spans the whole
  harvest window. It is derived from the arrival ROSTER alone — no source track is opened
  for a discarded flight, which is also why
  a capped KRDU build costs 0.9 GB / 7 s instead of 2.4 GB / 24 s — and therefore does not
  depend on the target type: **both prepared datasets select the same flights**, which is
  what keeps the per-flight comparison between `fitted_adsb` and `runway` paired. Every
  decision lands in `<scenarios>.selection.json` (`flight-scenarios-selection-v1`): a bounded
  population that is not stated reads as a full one, and every rate computed from it is then
  quietly wrong.

### FS3 · `source["dynamics_typecode"]` is what the v9 speed gate keys on

- **`source["dynamics_typecode"]` is what `evaluation`'s v9 threshold speed gate keys
  on** — the ICAO type the model FLEW (the fallback type when the identity's type has
  no dynamics; `resolved_typecode` keeps the identity), looked up in the published
  approach-speed table `aircraft/reference_speeds.json`; a computed record without it
  grades speed-indeterminate (loud, by design). Both record producers (optimizer batch
  and ts_transformer export) copy `scenario.source`, so this one write covers both.
  `source["landing_aero"]` (`{wing_area_m2, cl_max_landing}`, the same `AeroParams` the
  optimizer/replay fly) is still written as provenance of the model's own velocity
  floor but is no longer read by evaluation (v6–v8 anchored on it).
  → `evaluation/docs/THRESHOLD_SPEED_GATE.md`

### FS5 · The threshold target's speed is the airframe's published approach speed at the target mass

- `threshold_target_state` sets `V = aircraft.approach.reference_speed_ms(mass_kg)`. Since
  2026-09-24 (user decision, option A of `docs/aircraft_performance/2026-09-23_missing_performance_substitution.zh.md` §9)
  `Approach.speeds` is the airframe's row in `aircraft/reference_speeds.json` (FAA Aircraft
  Characteristics Database `Approach_Speed_knot` at MALW, its dual flap-configuration values, that
  MALW) and `reference_speed_ms(m)` = `ReferenceSpeed.vref_kt(m, edge="high")` = the published speed ×
  sqrt(m / MALW). That is the threshold speed gate's own law and its upper reference at the crossing
  mass (`evaluation.speed_gate.speed_gate_bounds(...).vref_high_ms`), so a solve that hits its target
  crosses inside the gate's window [V_ref,lo(m)·sqrt(n), V_ref,hi(m) + 20 kt]; for a type that publishes
  one value it sits on the (inclusive) lower edge. Basis: FAA AC 91-79B §5.2.2, "Vref, plus wind and
  gust additives, should be maintained until 50 ft over the runway threshold" — the model flies no
  wind, so no additive.
- Before (to 2026-09-23) every OpenAP type of 5.7–150 t targeted 145 kt and every heavier one 155 kt;
  the first cut of this change (2026-09-23, never merged) used the published speed unscaled, which put
  13 types — the A320 among them — just under the gate's lower edge by construction (review finding).
- **Which row.** The row belongs to the airframe whose MASS the `Aircraft` carries
  (`query_aircraft_parameters._mass_airframe`): a direct OpenAP type its own; an OpenAP synonym its
  surrogate's (LJ45 is cached with GLF6's 37.9 t, so it lands at GLF6's speed — its own 123 kt at that
  mass would read 257 kt); C56X its own, because its airframe-identity correction restored its mass.
  The presets use their own rows (A320 136 kt at 66 t, B77W 149, C172 62). **Gate caveat for
  synonyms:** the speed gate keys on the record's `dynamics_typecode` — the synonym's own code — and
  scales the synonym's own row by the record's (surrogate) mass, so for 12 of the 21 synonyms a target
  at the surrogate's speed sits outside the synonym's window (worst LJ45: window [256.6, 276.6] kt).
  The cause predates this change (the gate keyed on the synonym before too). **Resolved by FS6:** synonyms
  are no longer flown under their own code at all; the performance index flies each as an explicit
  substitute (code = the surrogate's) or with its own parameters (C56X).
- Types covered: the 172 rows of the 2026-09-07 pack + 21 rows added 2026-09-23/24 for the OpenAP and
  own-parameter types it did not list (`docs/reference_speeds/README.md`). **B3XM** is not in the FAA table:
  `get_aircraft_parameters("B3XM")` refuses it and `openap_support_kind("B3XM")` is None, so it is
  neither buildable nor in `openap_direct_typecodes()`. No harvested flight is a B3XM.
- **Saved scenarios are checked.** `FlightScenario.from_dict` rebuilds the aircraft from its code (today's
  envelope) but restores the saved target; a `runway_threshold` target whose V is not
  `reference_speed_ms(target.m)` is refused by name ("predates the published approach speeds —
  regenerate it with prepare_scenario_inputs.py"). Every `flight_scenarios/outputs/*_threshold_scenarios.json`
  prepared before 2026-09-24 is in that state.
- Stated approximation (in the `Approach` docstring): the published speed is an indicated airspeed,
  flown as a true airspeed (< 1 % at these threshold elevations).
- Consumers: the optimizer's `runway`-mode terminal pin, the interactive optimizers' default floor and
  `--resume`'s target check (`optimizer_reference.md` K10); evaluation's reported `speed_ms`
  deviation (`state V − target V`); the backend catalog's `terminalSpeedKt` and the Pilot panel's target
  range (the gate window at the landing mass). The ts datasets read only the target's position, ψ and γ,
  so no ts cohort or checkpoint moves. Optimizer records already on disk keep the target they were solved
  against until re-solved.

### FS6 · Types without a native model are decided by the performance index; none flies as an A320 by default

- **Order** (`scenario.aircraft_for_code`, provider `auto`): a hand-tuned preset; else the row of
  `aircraft/performance_index.json` (schema `aircraft-performance-index-v1`, loader
  `aircraft/performance_index.py`); else the type's own OpenAP model if OpenAP models it DIRECTLY and it
  has a published approach speed. OpenAP synonyms are no longer flown with their surrogate's data under
  their own code (`get_aircraft_parameters` refuses them); B3XM (no FAA speed) is refused. Provider
  `openap` (ts `openap-direct`) bypasses presets and the index, as before.
- **Rows** (2026-09-24, 209 types = the 199 types of the train/val census without a native model, decided by
  `docs/aircraft_performance/2026-09-23_missing_performance_substitution.zh.md` (user decisions
  2026-09-23/24), + the 10 OpenAP synonyms no observed flight carries, decided by the analysis's synonym
  rule): `own` (31 types: 21 from type-certificate data sheets + FAA ACD masses, 10 from the Poll–Schumann
  file) builds the type's own `Aircraft` (own landing mass, wing area, installed thrust, each citing a source
  id listed in the file; its own published speed; the MTOW-class procedure); `substitute` (53) returns the
  substitute airframe UNDER ITS OWN CODE, so mass, speed and speed gate stay one airframe (`resolved_typecode`
  keeps the identity); `exclude` (125: propeller aircraft by user decision, rotorcraft/military, no FAA
  approach speed, no same-class airframe) raises `KeyError` naming the reason. The loader refuses a row that
  would shadow a native airframe, a substitute that is not native, an own row without a published speed or a
  cited source, and a file that leaves any OpenAP synonym undecided.
- **Which index.** `performance_index_identity()` (sha256 + row count) is stamped into every scenario's
  `source["performance_index_sha256"]` and every `<scenarios>.selection.json`; `FlightScenario.from_dict`
  refuses a scenario built under another index or none (its code would be re-resolved into another airframe
  under its stored aero), so EVERY scenario file prepared before 2026-09-24 — threshold and fitted-ADS-B —
  must be regenerated before a re-solve.
- **No A320, ever.** `_resolve_aircraft` raises `NoAircraftDynamics(typecode, reason)` when nothing
  flies the type; `dataset.build_scenario_dataset` drops such flights and names them in
  `<scenarios>.selection.json` (`excluded_no_dynamics`, schema `flight-scenarios-selection-v2`); the CLI has
  no `--aircraft-type` any more, and `build_scenario` no fallback parameter (retired 2026-09-24 with the ts
  `all` filter, the last caller; user: "去掉 all 的 A320 回退").
- **Scenarios without dynamics, for trajectory-only consumers** (user decision 2026-09-24, "按需丢弃": drop
  a flight only where dynamics are used). `build_scenario(..., require_dynamics=False)` returns, for a flight
  without dynamics, a scenario with `aircraft = aero = None`, masses and target V `UNKNOWN_WITHOUT_DYNAMICS`
  (NaN) and `source["no_dynamics_reason"]`; its target position, heading and glidepath come from the
  published runway target only (a synthetic flight without an airframe has none and raises). A consumer that
  needs dynamics calls `scenario.dynamics(purpose)`, which raises `NoAircraftDynamics` naming the purpose —
  never computes on NaN. The ts data plane is the one caller (`aircraft_filter = all-flights`, the state
  output and the instruction labeller); the optimizer and the scenario files keep requiring dynamics.
- **Audit**: `source["performance_index_decision"]` (own / substitute / exclude / None);
  `dynamics_source` is `aircraft-performance-index-v1` for an own-parameter type.
- **Observed records** (`resolve_airframe`): the mass is returned only when the model flies the type as
  itself (preset, OpenAP-direct, index own); a substituted or excluded type keeps its type and gets no mass.
  Observed records change only when regenerated (with the user's permission).

### FS4 · `source["flight_key"]` is populated here

- **`source["flight_key"]` is populated here.** Optimizer evaluation rows used to report
  `flight_key: null` while observed rows carried it, because `build_scenario` never copied
  it. It does now — but only when the flight has an `id`, since `flight_key`'s fallback is
  the caller's list index and this function does not have one; a key built on the wrong index
  would disagree with the record filename `_scenario_filename` derives.
