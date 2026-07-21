# trajectory_data_process

Downloads real airport trajectories from the **OpenSky history database** and turns
them into the JSON that `aeroviz-4d/python/generate_czml.py` renders. It can also
emit a partitioned training dataset.

Single data source, single trajectory model, geometric altitude everywhere.

## Why this design

- **One source — the OpenSky history DB** (`traffic.data.opensky.history`). No live
  REST polling, no anonymous fallback, no OAuth track scraping.
- **Geometric altitude is required.** History rows carry `baroaltitude` and
  `geoaltitude` in the same record; the exported altitude is geometric, referenced
  to the ellipsoid, so no barometric bias correction is needed.
- **One trajectory model** (`trajectory.Trajectory`) is parsed once and reused by
  both the CZML export and the training-dataset builder.

## Layout

```
trajectory.py                  Trajectory / TrajectoryPoint model + builder from history rows
geo.py                         great-circle distance helpers
acquisition/
  opensky_history.py           history-DB fetch (requires geoaltitude)
  airports.py                  airport center + elevation from common/airports.csv
  runways.py                   runway-threshold coordinates from common/runways.csv
processing/
  czml_export.py               Trajectory -> CZML-input flight (geometric altitude)
  trajectory_events.py         airport entry/exit episode extraction + classification
datasets/
  dataset_store.py             partition layout + JSONL helpers
  training_dataset.py          raw-track / training-event / quarantine assembly
landings.py                    scan-back engine: N landings per runway threshold
config/runway_thresholds.json  maintained airport/runway/threshold mapping
build_runway_config.py         regenerate runway_thresholds.json from OurAirports CSVs
download_trajectories.py       CLI entry point (czml | training)
download_landings.py           CLI entry point (bulk landings per threshold)
```

## Prerequisites

- `pip install traffic pandas`
- **Historical-data access must be granted to your OpenSky account.** Authenticating
  is not enough: the Trino historical database is a separate entitlement. If the
  account lacks it, queries fail with `PERMISSION_DENIED: Access Denied: Cannot
  execute query` (the downloader turns this into a one-line message). Request access
  via the OpenSky Network ([DB guide](https://traffic-viz.github.io/data_sources/opensky_db.html)).
- **Configure credentials** in `~/.config/pyopensky/settings.conf` (Linux) or
  `~/Library/Application Support/pyopensky/settings.conf` (macOS — pyopensky
  resolves its config dir via platformdirs, so `~/.config` is never consulted there):

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

## Pipeline overview

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

For the **structure of each JSON shape** — the raw Trino rows, the CZML-input, and
the final CZML — with worked examples, see
[docs/06-data-formats-trino-to-czml.md](docs/06-data-formats-trino-to-czml.md).

## CZML mode (default)

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

## Training mode

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

## Fetch profiles & time window

- `--fetch-profile airport_ops` (default): query rows whose estimated departure or
  arrival airport is the target — smallest result, arrivals/departures only.
- `--fetch-profile terminal_all`: also query a bounding box around the airport
  (`--bbox-lat-pad`, `--bbox-lon-pad`) to include pass-through traffic.
- `--begin` / `--end` accept ISO (`2026-04-19T10:00:00Z`) or Unix seconds.
- Long windows are split into `--chunk-hours` queries (default 1 h).

## One-command pipeline

```bash
python run_asd-b_fetch_and_generate.py \
  --airport KRDU --begin 2026-04-19T10:00:00Z --end 2026-04-19T10:30:00Z --runway 23R
# -> aeroviz-4d/public/data/airports/KRDU/trajectories.czml
```

Runs the download stage, then `generate_czml.py`, writing
`aeroviz-4d/public/data/airports/KRDU/trajectories.czml`. Unknown flags
(`--begin`, `--end`, `--runway`, `--fetch-profile`, …) are forwarded to the download
stage. Add `--generate-procedures` to also rebuild RNAV/RNP procedure assets, or
`--input-json <czml_input.json>` to render an existing file without re-downloading.

## Landings per runway threshold (bulk download)

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
`--chunk-hours` steps (default 6 h) until each threshold has `--count` landings or
`--max-lookback-days` is reached. A threshold is abandoned once the scan has gone
`--dry-give-up-days` (default 4) past its last new landing — a **fixed duration,
independent of `--chunk-hours`** — so an idle runway end no longer drags the whole
airport back to the full lookback. The bbox query downloads only terminal-area state
vectors (≈80% fewer rows than the full-track airport join) and, because it is purely
geometric, also catches arrivals whose `estarrival` metadata is missing or wrong.
Landing flight records therefore have `arr_airport`/`dep_airport` set to `null`. A trajectory counts as a landing at a threshold when
its closest point passes within `--runway-threshold-radius-m` of the threshold while
**tracking the runway heading** and having **descended** from earlier in the track.
The heading test picks the correct runway end and the descent test excludes climbing
departures — this works even though real ADS-B coverage rarely reaches the ground near
a runway (tracks typically end a few hundred feet up on short final).

#### What is selected and where each track starts/ends

- **Selection (a landing at runway R):** the trajectory's closest point to R's threshold
  is (a) within `--runway-threshold-radius-m` (default 1000 m) laterally, (b) **below a
  generous AGL ceiling** (1500 m above the threshold) so high overflights that merely
  clip the threshold are rejected, (c) tracking R's runway heading (±35°), and (d) the
  aircraft has descended ≥300 m from earlier in the track. Together these exclude the
  opposite runway end, climbing departures, and cruise/descent overflights.
- **End (anchor):** each saved track ends at the **anchor** = the point closest to the
  threshold. Points *after* it (rollout/taxi) are dropped. In practice the anchor sits a
  few hundred feet up because low-altitude ADS-B coverage is sparse near runways.
- **Start (truncation):** only the **last 25 minutes before the anchor** are kept
  (the approach window, fixed at 25 min in this flow). At approach speed that is far more
  than the 30 km box spans, so in practice the track starts wherever the data starts.
- **Why starts are not all at the same distance:** `--bbox-radius-km` is a **square**
  (±30 km in lat/lon), so entry distance ranges from ~30 km (edge) to ~42 km (corner).
  A track that starts much closer (e.g. 13 km) simply was **not received earlier** — a
  coverage gap, or a >`--segment-gap-sec` (900 s) gap that split the track into a
  separate segment. The pipeline keeps the real samples; it never back-fills the missing
  earlier portion.

Output (one CZML-input file per threshold, plus a summary):

```text
outputs/landings/<AIRPORT>/<AIRPORT>_<RUNWAY>_landings.json   # e.g. KRDU_23R_landings.json
outputs/landings/summary_<UTC>.json                          # collected count per threshold
```

Each flight additionally carries `landing_time_utc` (the absolute time it reached the
threshold). The files are CZML **input**, not CZML — the frontend loads one
`trajectories.czml` per airport, so convert before using them in the app.

### Build the arrivals dataset + frontend CZMLs (per-runway + combined)

`build_arrivals.py` (renamed from `landings_to_czml.py`) first cuts every raw
landing track to its ARRIVAL SEGMENT — the final entry into the terminal ring
(`--entry-radius-km`, default 25 km; see `arrival_segment.py`) down to
touchdown — excluding pure local circuits into `<ICAO>_local_rejected.json`;
raw `*_landings.json` stay untouched, the derived `*_arrivals.json` feed
everything downstream. It then renders **one CZML per runway plus a combined
CZML**, with a manifest. With no `--airport` it processes **every downloaded
airport** under `outputs/landings/`:

```bash
# all downloaded airports (default)
python trajectory_data_process/build_arrivals.py

# one airport, or a subset of its runways
python trajectory_data_process/build_arrivals.py --airport KRDU
python trajectory_data_process/build_arrivals.py --airport KRDU --runway 23R 23L

# custom terminal-entry ring (km; must sit inside the 30 km harvest crop)
python trajectory_data_process/build_arrivals.py --airport KRDU --entry-radius-km 20
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
landings. It is abandoned after `--dry-give-up-days` of dry scanning rather than going
all the way to `--max-lookback-days`, which is the main lever on run time. `--chunk-hours`
only controls query size / responsiveness, not how far back a dry runway is tried.

## Tests

```bash
python -m pytest trajectory_data_process/tests -q
```

## Notes

- Geometric altitude is mandatory: points without `geoaltitude` are dropped, and a
  history result lacking the `geoaltitude` column raises an error rather than
  silently producing baro-only data.
- There is no live/REST/OAuth path and no barometric bias correction — both were
  removed with the move to the history DB.
- The OpenSky Trino account has a small query quota (typically 2 running + 2 queued).
  The download CLIs install a Ctrl-C / SIGTERM handler that **cancels the in-flight
  query** before exiting, so an interrupted run does not leave queries occupying the
  quota. If you ever do exhaust it (e.g. a hard kill), clear stuck queries at
  <https://trino.opensky-network.org/ui/> (filter by your user, then Kill).
- The design notes under `docs/01`–`docs/05` predate this refactor and describe the
  former REST/live training pipeline; they are kept only as history. This README and
  `docs/06-data-formats-trino-to-czml.md` are authoritative.
```
