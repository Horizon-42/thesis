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
| 1 | `4dTrajectory/optimization/scenario_optimization.py` | `optimize_scenario` (TODO ①) | run the optimizer, assemble the result | (CLI; runs the solver) |
| 2 | `4dTrajectory/optimization/scenario_optimization.py` | `simulate_controls` (TODO ②) | roll the controls through `CasadiSimulator` | `pytest …/test_scenario_optimization.py::test_simulate_controls_rolls_forward` |
| 3 | `aeroviz-4d/python/build_scenario_comparison_czml.py` | `_states_to_waypoints` (TODO ①) | state dicts → `(t, lon, lat, alt)` | `pytest …/test_build_scenario_comparison_czml.py::test_states_to_waypoints_order` |
| 4 | `aeroviz-4d/python/build_scenario_comparison_czml.py` | `_reference_entity_from_adsb` (TODO ②) | copy + recolour the matching ADS-B flight | `…::test_reference_entity_copies_and_recolors` |

> Two prerequisites are **already done** — no TODO in either:
> - `flight_scenarios/start_state.py` — the start/target kinematics.
> - `flight_scenarios/scenario.py` `aircraft_for_code` — aircraft resolution (see below).

---

## ✅ Done — `flight_scenarios/scenario.py` `aircraft_for_code` (aircraft resolution)

This was a WIP that tried to build the old flat `AircraftSpec` from OpenAP with mismatched
field names. It's now **resolved** by the `AircraftSpec → Aircraft` refactor (2026-06-28),
which unified the flat `AircraftSpec` and the OpenAP `AircraftParameters` into one nested
[`Aircraft`](../aircraft/aircraft_sets.py) (`geometry` / `mass` / `engine` / `approach` /
`drag`):

- `get_aircraft_parameters(code)` now returns an **`Aircraft`** directly — OpenAP supplies
  `geometry`/`mass`/`engine`/`drag`; OpenAP has **no approach envelope**, so a
  **category-default `approach`** (narrow_body / wide_body / general_aviation, mirroring the
  presets) fills that group. That closes the gap that blocked the old mapping.
- `aircraft_for_code(code)` resolves **presets first** (A320/B77W/C172 — they carry a
  calibrated approach envelope), then falls back to `get_aircraft_parameters` for any other
  OpenAP-supported type. Unknown codes raise `KeyError`.

`FlightScenario.aircraft` is now typed `Aircraft`; serialization is unchanged (stored by
`aircraft.code`, rebuilt via `aircraft_for_code`). Verify: `pytest flight_scenarios/tests`
(11 pass).

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
# 1. observed flights -> scenarios (aircraft resolution already works)
python -m flight_scenarios --input trajectory_data_process/outputs/landings/KRDU/KRDU_05L_landings.json \
    --output scen.json   # aircraft auto-resolved per flight from icao24; --aircraft-type A320 is the fallback
# 2. scenarios -> optimizer + simulator state files
python 4dTrajectory/optimization/scenario_optimization.py --scenarios scen.json --output-dir states/
# 3. state files -> comparison CZML (one per scenario)
python aeroviz-4d/python/build_scenario_comparison_czml.py --state-file states/AFR074_05L_states.json \
    --airport KRDU --output comparison.czml
```
