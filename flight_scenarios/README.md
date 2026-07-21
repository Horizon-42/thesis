# flight_scenarios

`flight_scenarios` is the data-to-modeling seam. It reads model-ready arrivals through
`outputs/harvest/<ICAO>/arrivals/manifest.json` and produces serializable
`FlightScenario` records for optimization and evaluation.

## Architecture

```text
harvest tracks (HAE)
  → arrivals/manifest.json (assigned, published CIFP path, final-entry crop)
  → load_model_arrivals() (HAE → MSL)
  → FlightScenario
       ├─ initial: observed state at terminal entry
       ├─ target: observed end or published runway Path Point
       ├─ aircraft: per-flight type resolved from type/icao24/fallback
       └─ aero: matching aerodynamic parameters
  → optimizer / reference evaluation
```

The package does not glob JSON files and does not accept the removed per-runway landing
arrays. The arrival manifest is already the all-runway, de-duplicated roster.

## Scenario record

```python
FlightScenario(
    initial=GeodeticState(lat, lon, alt, V, psi, gamma, m),
    target=GeodeticState(...),
    aircraft=Aircraft,
    aero=AeroParams,
    source={
        "id": ...,
        "icao24": ...,
        "runway": ...,
        "landing_time_utc": ...,
        "entry_time_utc": ...,
        "target_source": ...,
        "altitude_source": ...,
    },
)
```

Position comes from the track. Velocity, math-ENU heading, and flight-path angle are fit
from a short sample window. Aircraft resolution tries the declared type, then `icao24`
through OpenAP, then the explicit `--aircraft-type` fallback.

## Target modes

- Default: target the final observed sample.
- `--target-from-threshold`: target the arrival record's `runway_target`, whose position,
  threshold crossing height, course, and glidepath came from the published CIFP Path Point.

Canonical harvest arrivals always carry that target. The static
`runway_thresholds.json` lookup remains only for synthetic/in-memory scenarios.

## Vertical datum

Harvest geometry is HAE because Cesium needs ellipsoidal height. Scenario state, runway
elevations, CIFP altitudes, and evaluation gates are MSL. `load_model_arrivals()` converts
HAE → MSL exactly once and tags the result, while `build_scenario()` also protects direct
in-memory callers. Unknown altitude sources fail instead of being guessed.

## CLI

One airport:

```bash
conda run -n aviation python -m flight_scenarios \
  --airport KRDU --target-from-threshold \
  --output-dir flight_scenarios/outputs
```

Explicit manifest and output:

```bash
conda run -n aviation python -m flight_scenarios \
  --input trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json \
  --target-from-threshold \
  --output flight_scenarios/outputs/KRDU_arrivals_threshold_scenarios.json
```

With neither `--airport` nor `--input`, the CLI processes every
`*/arrivals/manifest.json` under `--harvest-root` and writes one scenario file per airport.

Python API:

```python
from flight_scenarios import build_scenarios_from_arrivals, load_scenarios

scenarios = build_scenarios_from_arrivals(
    "trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json",
    aircraft_type="A320",
    airport="KRDU",
    target_from_threshold=True,
)
```

## Tests

```bash
conda run -n aviation pytest -o "pythonpath=. geokit/src" flight_scenarios/tests -q
```
