# flight_scenarios

Build **neutral modeling inputs** from observed trajectory data. Given a flight's
observed track (the CZML-input format) and an aircraft identity, it produces a
serializable **`FlightScenario`** that feeds *both* the trajectory optimizer and a future
data-driven model — without depending on either.

## 1. Where this sits (architecture)

The project has two planes that are otherwise decoupled:

- **Data plane** — `trajectory_data_process` turns OpenSky history into **CZML-input**
  JSON (`[{id, callsign, waypoints: [[t, lon, lat, alt], …]}]`). It has *no* dependency
  on aircraft/dynamics code.
- **Modeling plane** — `aircraft` (`AircraftSpec`/`AeroParams`), `aerodynamic_model`
  (`GeodeticState`), `geokit`, the optimizer (`4dTrajectory/optimization`), and a future
  data-driven model.

`flight_scenarios` is the **seam** between them — exactly analogous to how CZML-input is
the seam between the data plane and the visualization. It reads the neutral CZML-input
*format* and produces a neutral modeling-input *record*:

```
CZML-input JSON ─► flight_scenarios ─► 4dTrajectory/optimization   (a problem instance)
                        │            └► future data-driven model    (a training example)
                        ▼
          aircraft · aerodynamic_model · geokit
```

It depends **downward** on the modeling primitives and is imported **upward** by the two
consumers — so neither consumer depends on the other, and there are no cycles. (That is
why it is a top-level package, not under `4dTrajectory/optimization`: putting it there
would make the data-driven model transitively depend on the solver.)

## 2. The `FlightScenario` record

```python
FlightScenario(
    initial = GeodeticState(lat, lon, alt, V, psi, gamma, m),  # the start state
    aircraft = AircraftSpec,                                    # e.g. A320
    aero     = AeroParams,                                      # aero_params_for_aircraft(spec)
    source   = {"id", "callsign", "icao24", "runway", "n_samples", …},
    target   = GeodeticState | None,                            # optional (e.g. the threshold)
)
```

It round-trips through JSON (`to_dict`/`from_dict`, `save_scenarios`/`load_scenarios`),
so a CLI run produces a small dataset the optimizer can replay and the model can train on.

## 3. The one piece of physics: the start state

The point-mass state is `(lat, lon, alt, V, psi, gamma, m)`. Three of these come for free
and one is supplied:

- `lat, lon, alt` — read straight off the first track sample.
- `m` — the aircraft mass (the track says nothing about mass).

The remaining three, **`V` (speed), `psi` (heading), `gamma` (flight-path angle)**, are
*not* stored in the track — they are **estimated by finite-differencing** two samples a
short window apart at the start of the track. Take the anchor sample `p0` and a sample
`p1` about `window_s` later, and let

- `horizontal_m` = great-circle ground distance `p0 → p1`  (use `geokit.haversine_m`)
- `vertical_m`   = `alt1 − alt0`
- `dt`           = `t1 − t0`

Then:

```
        ┌ p1
        │  ╱│
   path │ ╱ │ vertical_m         V     = √(horizontal_m² + vertical_m²) / dt
        │╱  │                    gamma = atan2(vertical_m, horizontal_m)   (+ = climb)
     p0 └───┘                    psi   = bearing(p0 → p1)                  (0 = N, CW)
        horizontal_m
```

- **`V`** is the speed *along the flight path*: the hypotenuse of the horizontal run and
  the vertical rise, divided by the time. (Pure ground speed `horizontal_m/dt` ignores the
  climb; for a 3° approach the difference is tiny, but the along-path speed is what the
  point-mass `V` means.)
- **`gamma`** is the angle of that hypotenuse above the horizontal — `atan2(rise, run)`.
- **`psi`** is the compass heading of the horizontal step — a great-circle bearing.

### → The TODO

`start_state.initial_state_from_track` has the geometry (`horizontal_m`, `vertical_m`,
`dt`) computed for you and the three formulas in a `TODO ①` comment. Fill in `V`, `psi`,
`gamma`, build the `GeodeticState`, and delete the `raise NotImplementedError`. The two
`xfail` tests in `tests/test_start_state.py` (a level due-east track and a climbing one)
become green when it is correct — run:

```bash
python -m pytest flight_scenarios/tests -q
```

## 4. Usage

```bash
# one scenario per flight; aircraft auto-resolved per flight from its icao24:
python -m flight_scenarios \
  --input trajectory_data_process/outputs/landings/KRDU/KRDU_05L_landings.json \
  --output scenarios_krdu_05l.json
# --aircraft-type A320  (optional) is the fallback for flights whose icao24 can't be resolved
```

```python
from flight_scenarios import build_scenarios_from_czml_input, load_scenarios

scenarios = build_scenarios_from_czml_input("…_landings.json")   # per-flight icao24 -> Aircraft
state = scenarios[0].initial          # a GeodeticState ready for the optimizer
aero  = scenarios[0].aero             # AeroParams for the same run
```

## 5. Notes / extension points

- **Aircraft identity is resolved from the flight's `icao24`** (CZML-input's `type` is
  `"UNK"`). `aircraft_for_code` checks the hand-tuned `AIRCRAFT_PRESETS` (A320/B77W/C172)
  first, then resolves any OpenAP-supported typecode via
  `aircraft/query_aircraft_parameters.py` (geometry/mass/engine/drag + an MTOW-bucketed
  approach default). `build_scenario` prefers the `icao24`; the CLI `--aircraft-type` is the
  fallback when an `icao24` isn't in the OpenAP lookup. Serialization stores `aircraft.code`,
  so round-trips stay unchanged.
- **`target`** is optional. The same kinematics at the *end* of the track (or a runway
  threshold) gives a target state; wire it into `build_scenario` when needed.
- Read **CZML-input**, not CZML — CZML is the rendered presentation format (quaternions,
  styling); CZML-input is the neutral `[[t, lon, lat, alt], …]` track this package wants.
