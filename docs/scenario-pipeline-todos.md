# Scenario → optimization → CZML pipeline: what's left to implement

The pipeline turns an observed flight into a 3-way comparison (observed vs optimizer plan
vs simulator rollout). The **architecture is wired**; this lists the `TODO` blocks you fill
in, in dependency order, with the exact formula/hint and how to verify each.

```
flight_scenarios  ──►  4dTrajectory/optimization/scenario_optimization.py  ──►  *_states.json
 (initial+target+aircraft)        (optimizer states + simulator states)            │
                                                                                   ▼
                              aeroviz-4d/python/build_scenario_comparison_czml.py ──► comparison.czml
                                  (reference + optimizer + simulator, 3 colours)
```

## Checklist

| # | File | Function | What to implement | Verify (turns the xfail green) |
|---|------|----------|-------------------|--------------------------------|
| 0 | `flight_scenarios/scenario.py` | `aircraft_for_code` | *(your WIP)* finish the OpenAP→`AircraftSpec` mapping, or keep the preset lookup | `pytest flight_scenarios/tests` |
| 1 | `4dTrajectory/optimization/scenario_optimization.py` | `optimize_scenario` (TODO ①) | run the optimizer, assemble the result | (CLI; runs the solver) |
| 2 | `4dTrajectory/optimization/scenario_optimization.py` | `simulate_controls` (TODO ②) | roll the controls through `CasadiSimulator` | `pytest …/test_scenario_optimization.py::test_simulate_controls_rolls_forward` |
| 3 | `aeroviz-4d/python/build_scenario_comparison_czml.py` | `_states_to_waypoints` (TODO ①) | state dicts → `(t, lon, lat, alt)` | `pytest …/test_build_scenario_comparison_czml.py::test_states_to_waypoints_order` |
| 4 | `aeroviz-4d/python/build_scenario_comparison_czml.py` | `_reference_entity_from_adsb` (TODO ②) | copy + recolour the matching ADS-B flight | `…::test_reference_entity_copies_and_recolors` |

> `flight_scenarios/start_state.py` (the start/target kinematics) is **already done** — no TODO there.

---

## 0. `flight_scenarios/scenario.py` — `aircraft_for_code` *(your in-progress edit)*

You changed this to build an `AircraftSpec` from the OpenAP database
(`get_aircraft_parameters`). The field names don't match the real dataclasses in
`aircraft/query_aircraft_parameters.py`:

| Your code | Actual field |
|---|---|
| `mass.max_takeoff_mass_kg` | `mass.mtow_kg` |
| `engine.max_thrust_n` | `engine.max_thrust_n_each` × `engine.number` |
| `engine.approach_thrust_guess_n` | *(not in OpenAP)* |
| `drag.terminal_speed_kt` / `…_min_kt` / `…_max_kt` | *(not in OpenAP — `AircraftDrag` has only `cd0, k, e, landing_gear_drag_increment`)* |
| `drag.final_approach_*` / `drag.threshold_crossing_height_m` | *(not in OpenAP)* |

**The gap:** OpenAP gives geometry / mass / drag-coefficients / engine, but **not** the
approach speeds or final-approach geometry that `AircraftSpec` needs. So a spec can't be
built purely from OpenAP — overlay the OpenAP-available fields on a preset's approach
defaults, or supply those defaults explicitly. (Until then, the preset lookup on `HEAD`
works for A320/B77W/C172.)

---

## 1 & 2. `4dTrajectory/optimization/scenario_optimization.py`

### TODO ① — `optimize_scenario`: run the optimizer

```python
optimizer = CasadiDirectCollocationOptimizer(n_segments, dt, max_duration, aircraft)
final_time, node_control, node_state = optimizer.optimize_free_time(initial, target, max_duration)
# node_state rows  = [lat, lon, alt, V, psi, gamma]   (the plan)
# node_control rows = [thrust_N, bank_rad, load_factor]
optimizer_states = _node_states_to_samples(node_state, final_time, initial.m)
simulator_states = simulate_controls(initial, node_control, final_time, aircraft, dt=rollout_dt_s)
return ScenarioOptimization(scenario.source, float(final_time), optimizer_states, simulator_states)
```

### TODO ② — `simulate_controls`: roll the controls through the real simulator

```python
controls = list(node_control)
segment_duration = final_time / len(controls)
sim = CasadiSimulator(aircraft, dt)
state, t = initial_state, 0.0
samples = [StateSample.from_state(0.0, state)]
for segment_index, row in enumerate(controls):
    control = LoadFactorControl(thrust=float(row[0]), bank_rad=float(row[1]), load_factor=float(row[2]))
    segment_end = (segment_index + 1) * segment_duration
    while t < segment_end - 1e-9:
        step_dt = min(dt, segment_end - t)
        try:
            state = sim.step(state, control, step_dt)
        except ValueError:
            return samples          # left the flight envelope — truncate, don't crash
        t += step_dt
        samples.append(StateSample.from_state(t, state))
return samples
```

Run it: `python 4dTrajectory/optimization/scenario_optimization.py --scenarios scen.json --output-dir states/`

---

## 3 & 4. `aeroviz-4d/python/build_scenario_comparison_czml.py`

### TODO ① — `_states_to_waypoints`: state seq → CZML geometry

```python
return [(s["t"], s["lon"], s["lat"], s["alt"]) for s in states]   # note: lon BEFORE lat
```

### TODO ② — `_reference_entity_from_adsb`: copy the matching ADS-B flight

```python
for packet in adsb_czml:
    if packet.get("id") == flight_id:
        entity = copy.deepcopy(packet)               # don't mutate the source
        entity["id"] = "scenario-reference"
        entity["name"] = f"Reference {flight_id}"
        entity["path"]["material"]["solidColor"]["color"]["rgba"] = list(color_rgba)
        return entity
return None                                          # no matching flight
```

Run it: `python aeroviz-4d/python/build_scenario_comparison_czml.py --state-file states/AFR074_05L_states.json --airport KRDU --output comparison.czml`

---

## End-to-end, once the TODOs are filled

```bash
# 1. observed flights -> scenarios (needs aircraft_for_code working)
python -m flight_scenarios --input trajectory_data_process/outputs/landings/KRDU/KRDU_05L_landings.json \
    --aircraft A320 --output scen.json
# 2. scenarios -> optimizer + simulator state files
python 4dTrajectory/optimization/scenario_optimization.py --scenarios scen.json --output-dir states/
# 3. state files -> comparison CZML (one per scenario)
python aeroviz-4d/python/build_scenario_comparison_czml.py --state-file states/AFR074_05L_states.json \
    --airport KRDU --output comparison.czml
```
