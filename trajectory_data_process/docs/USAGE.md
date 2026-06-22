# Usage guide

This package downloads airport trajectories from the OpenSky **history database** and
exports them either as CZML input (for visualization) or as a partitioned training
dataset. Every point carries **geometric altitude**, which is the value used for
visualization.

## 1. Prerequisites

- `pip install traffic pandas`
- **Historical-data access must be granted to your OpenSky account.** Authenticating
  is not enough: the Trino historical database is a separate entitlement. If the
  account lacks it, queries fail with `PERMISSION_DENIED: Access Denied: Cannot
  execute query` (the downloader turns this into a one-line message). Request access
  via the OpenSky Network ([DB guide](https://traffic-viz.github.io/data_sources/opensky_db.html)).
- **Configure credentials** in `~/.config/pyopensky/settings.conf`:

  ```ini
  [opensky]
  username = your_opensky_username
  password = your_opensky_password
  ```

  or via environment variables `OPENSKY_USERNAME` / `OPENSKY_PASSWORD`. Note the
  variable is `OPENSKY_USERNAME`, **not** `OPENSKY_USER`; with no recognized
  credentials, `traffic` falls back to interactive browser auth on every query.
- **Do not query "today".** The historical DB lags real time, so the most recent
  hours return nothing. Pass `--start` (or `--begin`) at least a day or two back.
- Reference data lives under `aeroviz-4d/public/data/common/`:
  - `airports.csv` — airport centers and elevations.
  - `runways.csv` — runway-threshold coordinates (for `--runway`).

## 2. Pipeline overview

```
OpenSky history DB ──► download_trajectories.py ──► outputs/<airport>/*_czml_input_*.json
 (traffic, geo alt)         │                              │
                            │                              └─► generate_czml.py ─► trajectories.czml
                            └─► (training mode) partitioned JSONL dataset + manifest
```

The flow is built on one model: `trajectory.Trajectory`, a time-sorted list of
`TrajectoryPoint`s, each with `geo_altitude_m`, `baro_altitude_m`, `heading_deg`,
`on_ground`. History rows are grouped per aircraft and split into separate
trajectories on a time gap (`--segment-gap-sec`) or a change of departure/arrival
airport.

## 3. CZML mode (default)

```bash
python trajectory_data_process/download_trajectories.py \
  --airport KRDU \
  --begin 2026-04-19T10:00:00Z --end 2026-04-19T10:30:00Z \
  --runway 23R --max-trajectories 20
```

Writes `outputs/krdu/krdu_czml_input_<UTC>.json`: a list of flights, each

```json
{ "id": "EDV5269", "callsign": "EDV5269", "type": "UNK",
  "icao24": "a1b2c3", "dep_airport": "KJFK", "arr_airport": "KRDU", "runway": "23R",
  "altitude_source": "opensky_history_geoaltitude_m",
  "waypoints": [[0, -78.75, 36.83, 1500.0], [12, -78.76, 36.81, 1100.0]] }
```

Each waypoint is `[offset_seconds, longitude, latitude, geometric_altitude_metres]`,
exactly what `aeroviz-4d/python/generate_czml.py` consumes.

### Selecting by runway threshold (`--runway`)

Without `--runway`, a trajectory is kept when its closest point to the airport is
within `--max-end-distance-km` (default 2.5 km). With `--runway 23R`, the threshold
coordinates are read from `runways.csv` and a trajectory is kept only when its final
approach point lands within `--runway-threshold-radius-m` (default 600 m) of **that
threshold** — i.e. you select the exact runway end the aircraft arrives at, not just
the airport.

### Key CZML flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--runway` | none | keep only arrivals at this runway threshold, e.g. `23R` |
| `--runway-threshold-radius-m` | 600 | proximity to the threshold to count as an arrival |
| `--match-radius-km` | 35 | trajectory must approach within this distance of the airport |
| `--max-end-distance-km` | 2.5 | arrival-anchor distance limit (ignored when `--runway` is set) |
| `--approach-window-min` | 20 | keep this many minutes before the arrival anchor |
| `--exclude-ground` | off | drop on-ground points from the export |
| `--max-trajectories` | 80 | cap exported flights |

## 4. Training mode

```bash
python trajectory_data_process/download_trajectories.py \
  --airport KRDU --begin 2026-04-19T00:00:00Z --end 2026-04-19T06:00:00Z \
  --dataset-mode training --fetch-profile terminal_all
```

Writes a partitioned `outputs/<dataset>/v3/airport=KRDU/year=…/month=…/day=…/[hour=…]/`
tree:

- `raw_tracks/` — one JSONL record per trajectory (full point list, both altitudes).
- `airport_events/` — entry/exit episodes labelled `landing` / `depart` / `pass` /
  `ambiguous`, each with attached `training_points`.
- `quarantine/` — events dropped because a required geometric altitude was missing.
- `manifests/` — one JSON manifest per run.

Training flags: `--airport-event-radius-nm` (default 5), `--low-altitude-agl-m`
(default 600), `--segment-gap-sec` (default 900).

## 5. Fetch profiles & time window

- `--fetch-profile airport_ops` (default): query rows whose estimated departure or
  arrival airport is the target — smallest result, arrivals/departures only.
- `--fetch-profile terminal_all`: also query a bounding box around the airport
  (`--bbox-lat-pad`, `--bbox-lon-pad`) to include pass-through traffic.
- `--begin` / `--end` accept ISO (`2026-04-19T10:00:00Z`) or Unix seconds.
- Long windows are split into `--chunk-hours` queries (default 1 h).

## 6. One-command pipeline

```bash
python run_asd-b_fetch_and_generate.py \
  --airport KRDU --begin 2026-04-19T10:00:00Z --end 2026-04-19T10:30:00Z --runway 23R
```

Runs the download stage, then `generate_czml.py`, writing
`aeroviz-4d/public/data/airports/KRDU/trajectories.czml`. Unknown flags
(`--begin`, `--end`, `--runway`, `--fetch-profile`, …) are forwarded to the download
stage. Add `--generate-procedures` to also rebuild RNAV/RNP procedure assets, or
`--input-json <czml_input.json>` to render an existing file without re-downloading.

## 7. Landings per runway threshold (bulk download)

To collect a fixed number of historical **landings at every runway threshold** of the
project's main airports, use the dedicated entry point. It is driven by a maintained
mapping file and keeps the caller trivial — all logic lives in `landings.py`.

### Runway-threshold mapping

`config/runway_thresholds.json` lists each airport's open runways and their two
thresholds (ident, lat/lon, elevation, heading). It is reusable by any other code
that needs threshold geometry. Regenerate it from the OurAirports CSVs whenever the
airport set or source data changes:

```bash
python trajectory_data_process/build_runway_config.py            # default 5 airports
python trajectory_data_process/build_runway_config.py --airports KRDU KSJC
```

### Download

```bash
# 20 landings for every threshold of every airport in the config:
python trajectory_data_process/download_landings.py --count 20

# Narrow to specific airports and scan further back:
python trajectory_data_process/download_landings.py --count 30 --airports KRDU KSJC \
  --max-lookback-days 60
```

For each airport it issues **one history query per time chunk** — a bounding box of
`--bbox-radius-km` (default 30 km) around the airport — and reuses the trajectories for
all of that airport's thresholds, scanning backward from `--start` (default: now) in
`--chunk-hours` steps until each threshold has `--count` landings or
`--max-lookback-days` is reached. The bbox query downloads only terminal-area state
vectors (≈80% fewer rows than the full-track airport join) and, because it is purely
geometric, also catches arrivals whose `estarrival` metadata is missing or wrong.
Landing flight records therefore have `arr_airport`/`dep_airport` set to `null`. A trajectory counts as a landing at a threshold when
its closest point passes within `--runway-threshold-radius-m` of the threshold while
**tracking the runway heading** and having **descended** from earlier in the track.
The heading test picks the correct runway end and the descent test excludes climbing
departures — this works even though real ADS-B coverage rarely reaches the ground near
a runway (tracks typically end a few hundred feet up on short final).

Output (one CZML-input file per threshold, plus a summary):

```text
outputs/landings/<AIRPORT>/<AIRPORT>_<RUNWAY>_landings.json   # e.g. KRDU_23R_landings.json
outputs/landings/summary_<UTC>.json                          # collected count per threshold
```

Each flight additionally carries `landing_time_utc` (the absolute time it reached the
threshold). The files are CZML **input**, not CZML — the frontend loads one
`trajectories.czml` per airport, so convert before using them in the app.

### Render to the frontend (per-runway + combined)

`landings_to_czml.py` renders an airport's landing files into the frontend folder,
**one CZML per runway plus a combined CZML**, with a manifest:

```bash
# all runways of KRDU
python trajectory_data_process/landings_to_czml.py --airport KRDU

# only specific runway ends
python trajectory_data_process/landings_to_czml.py --airport KRDU --runway 23R 23L
```

Output under `aeroviz-4d/public/data/airports/<ICAO>/`:

```text
landings/<ICAO>_<RWY>.czml   one CZML per runway (e.g. landings/KRDU_23R.czml)
landings/index.json          manifest: { airport, combined, runways:[{runway,file,count}] }
trajectories.czml            all runways combined (the default the frontend loads)
```

It de-duplicates landings by `(icao24, landing_time_utc)`, re-uniques ids, and runs
`generate_czml.py` per runway and once for the combined file.

### Loading in the frontend

The app loads **per airport** by default (`trajectories.czml` = all runways). When a
`landings/index.json` manifest is present, the ControlPanel shows a **Landing Runway**
selector (`All runways` + one entry per runway with its count); picking a runway loads
just that runway's `landings/<ICAO>_<RWY>.czml`. The selection resets to `All` when the
airport is switched.

### Resume & caching (re-running)

There are two layers, and they behave differently:

- **Output resume (default).** On each run the existing `*_landings.json` files are
  loaded; thresholds already at `--count` are **not** re-collected, and an airport
  whose thresholds are all satisfied is skipped entirely (no query). New landings are
  merged and de-duplicated by `(icao24, landing_time_utc)`. Pass `--overwrite` to
  ignore existing files and refetch from scratch.
- **Trino query cache (`traffic`).** Each history query result is cached to
  `~/Library/Caches/opensky` keyed by the exact `(airport, time window, columns)`.
  Re-querying the **same window** never re-hits Trino. **Caveat:** `--start` defaults
  to *now*, so each run uses different windows and misses the cache — pass a **fixed
  `--start`** (and the same `--chunk-hours`) to reuse it.

Note: a runway end that is idle in the requested period legitimately yields few or no
landings, so its airport keeps scanning back to `--max-lookback-days` on every run.
Bound the search with `--max-lookback-days` rather than chasing those ends.

## 8. Notes

- Geometric altitude is mandatory: points without `geoaltitude` are dropped, and a
  history result lacking the `geoaltitude` column raises an error rather than
  silently producing baro-only data.
- There is no live/REST/OAuth path and no barometric bias correction — both were
  removed with the move to the history DB.
