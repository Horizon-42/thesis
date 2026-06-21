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
download_trajectories.py       CLI entry point (czml | training)
docs/USAGE.md                  full usage guide
```

## Prerequisites

- `pip install traffic pandas`
- Configure OpenSky database access for `traffic`:
  https://traffic-viz.github.io/data_sources/opensky_db.html

## Quick start

CZML input for KRDU, landings at runway threshold 23R:

```bash
python trajectory_data_process/download_trajectories.py \
  --airport KRDU \
  --begin 2026-04-19T10:00:00Z --end 2026-04-19T10:30:00Z \
  --runway 23R \
  --max-trajectories 20
# -> trajectory_data_process/outputs/krdu/krdu_czml_input_<UTC>.json
```

One command, fetch + render to the AeroViz frontend asset:

```bash
python run_asd-b_fetch_and_generate.py \
  --airport KRDU --begin 2026-04-19T10:00:00Z --end 2026-04-19T10:30:00Z --runway 23R
# -> aeroviz-4d/public/data/airports/KRDU/trajectories.czml
```

Training dataset:

```bash
python trajectory_data_process/download_trajectories.py \
  --airport KRDU --begin 2026-04-19T00:00:00Z --end 2026-04-19T06:00:00Z \
  --dataset-mode training --fetch-profile terminal_all
```

See **[docs/USAGE.md](docs/USAGE.md)** for every flag and the output schemas.

## Tests

```bash
python -m pytest trajectory_data_process/tests -q
```

> The design notes under `docs/01`–`docs/05` predate this refactor and describe the
> former REST/live training pipeline; they are kept only as history. `docs/USAGE.md`
> and this README are authoritative.
