# Project Context

This repository is a thesis research workspace for 4D aircraft trajectory
prediction, airport approach validation, and aviation visualization. The active
engineering system is AeroViz-4D: a React/TypeScript/CesiumJS visualization app
plus Python data pipelines for ADS-B trajectories, airport geometry, RNAV
procedures, obstacles, charts, and terrain.

The project is research and visualization software only. It is not certified
navigation software, not an operational flight tool, and not a replacement for
official FAA, ICAO, Nav Canada, or local procedure publications.

## Research Goal

The thesis topic is multi-aircraft airport approach 4D trajectory prediction
using ADS-B data. A 4D trajectory is longitude, latitude, altitude, and time.

Main research objectives:

- Build high-quality approach-phase trajectory datasets from ADS-B/OpenSky data.
- Analyze trajectory patterns and multi-aircraft interactions near airports.
- Train or validate sequence/generative trajectory prediction models.
- Visualize predicted or replayed trajectories against runway, RNAV procedure,
  terrain, obstacle, and timing context.

The current concrete app focus is validation and explanation: show where an
aircraft is, when it is there, how it moves relative to procedure geometry, and
whether the trajectory looks plausible in the runway/approach context.

## Repository Map

- `aeroviz-4d/`: main AeroViz-4D app.
  - `src/`: React UI, Cesium hooks, geometry utilities, data adapters, tests.
  - `public/data/`: browser-served generated data. Common CSV files are tracked;
    most airport/generated data is local working data.
  - `python/`: preprocessing, CIFP parsing, chart linking, obstacle parsing, and
    CZML generation scripts.
  - `docs/`: implementation plans, pipeline notes, dev logs, and aviation data
    design notes.
- `trajectory_data_process/`: standalone ADS-B acquisition, normalization, and
  training dataset helpers. Kept outside `aeroviz-4d` so data acquisition is not
  coupled to the frontend.
- `trajectory_data_process.harvest`: canonical observed-track download, derivation,
  evaluation, and CZML publication entry point.
- `prepare_scenario_inputs.py` / `run_scenario_optimization.py`: prepared modeling
  data and optimizer/publication entry points.
- `preprocess_aeroviz_airport.sh`: full airport data preprocessing pipeline for
  AeroViz browser assets.
- `generate_aeroviz_airport_procedure_data.sh`: RNAV/RNP procedure asset
  generation shortcut.
- `bc_lidar_downloader/`, `usgs_lidar_downloader/`, `opentopography_downloader/`:
  standalone terrain/DSM/DEM acquisition helpers.
- `aerodrome_model/`, `aerodynamic_model/`, `runway_schedule/`,
  `flight_procedure_design/`, `4dTrajectory/`, `models/`: research notes,
  literature, model explorations, and supporting aviation material.
- `data/`: local external datasets such as CIFP, DOF, RNAV charts, DEM/DSM, and
  downloaded artifacts. This directory is ignored by git.

## Core Data Flow

Dynamic trajectory path:

```text
OpenSky history DB (traffic)
  -> trajectory_data_process/download_trajectories.py
  -> Trajectory model (geometric altitude) -> czml export and/or training records
  -> *_czml_input_*.json
  -> aeroviz-4d/python/generate_czml.py
  -> aeroviz-4d/public/data/airports/<ICAO>/trajectories.czml
  -> aeroviz-4d/src/hooks/useCzmlLoader.ts
  -> CesiumJS playback
```

Static airport/procedure path:

```text
OurAirports CSV
  -> aeroviz-4d/python/preprocess_airports.py
  -> airport.json and runway.geojson

FAA CIFP / FAACIFP18
  -> aeroviz-4d/python/preprocess_waypoints.py
  -> waypoints.geojson
  -> aeroviz-4d/python/preprocess_procedures.py
  -> procedures.geojson and procedure-details/*.json

FAA DOF
  -> aeroviz-4d/python/preprocess_obstacles.py
  -> obstacles.geojson

FAA RNAV chart PDFs
  -> aeroviz-4d/python/download_faa_rnav_charts.py
  -> data/RNAV_CHARTS/<ICAO>/
  -> procedure preprocessing publishes chart manifests/assets
```

Terrain path:

```text
OpenTopography DEM or LiDAR-derived DSM/DEM
  -> data/opentopography, data/usgs_lidar, or data/bc_lidar
  -> aeroviz-4d/scripts/build_dsm_heightmap_terrain.mjs
  -> public/data/airports/<ICAO>/dsm/heightmap-terrain/**
  -> aeroviz-4d/src/hooks/useDsmTerrainLayer.ts
```

## AeroViz-4D Frontend Architecture

The frontend is a Vite app using React 18, TypeScript, CesiumJS, Vitest, and
Testing Library.

Global state lives in `aeroviz-4d/src/context/AppContext.tsx` using React
context plus `useState`; there is no Redux/Zustand. Important state includes:

- `viewer`: CesiumJS Viewer instance.
- `airports` and `activeAirportCode`: loaded from
  `public/data/airports/index.json`; default airport is currently `KRDU`.
- `selectedFlightId`: selected/tracked aircraft.
- `trajectoryDataSource`: loaded CZML data source.
- `layers`: visibility flags for terrain, DSM terrain, runways, waypoints,
  OCS surfaces, trajectories, obstacles, obstacle labels, and procedures.
- Procedure UI state: branch visibility, annotations, width measurement, display
  level, selected annotation.
- Runway profile UI state.

`aeroviz-4d/src/App.tsx` routes between:

- main flight view,
- `/dsm-terrain-demo` or `#dsm-terrain-demo`,
- `/procedure-details` or `#procedure-details`.

Cesium integration is mostly in hooks:

- `useCesiumViewer`: initialize viewer, load airport config, set camera.
- `useCzmlLoader`: load trajectory CZML and synchronize clock/data source.
- `useRunwayLayer`, `useWaypointLayer`, `useObstacleLayer`, `useTerrainLayer`,
  `useDsmTerrainLayer`, `useOcsLayer`, `useProcedureSegmentLayer`: layer
  lifecycle and Cesium entity/data source management.

Pure geometry and assessment logic lives under `aeroviz-4d/src/utils/` and
should stay testable without Cesium where possible.

## Data Contracts

CZML input JSON consumed by `generate_czml.py` is a list of flight records:

```json
[
  {
    "id": "UAL123",
    "callsign": "United 123",
    "type": "B738",
    "waypoints": [[0, -119.38, 49.95, 4500]]
  }
]
```

Each waypoint is:

```text
[offset_seconds, longitude_degrees, latitude_degrees, altitude_metres]
```

Generated CZML uses a document packet first, then one entity per flight.
Positions use Cesium `cartographicDegrees` with repeated groups of:

```text
[seconds_offset, lon, lat, altitude_metres]
```

Airport browser assets are expected under:

```text
aeroviz-4d/public/data/airports/<ICAO>/
  airport.json
  runway.geojson
  waypoints.geojson
  procedures.geojson
  procedure-details/index.json
  procedure-details/*.json
  charts/index.json
  obstacles.geojson
  trajectories.czml
  dsm/heightmap-terrain/**
```

Not every airport currently has every asset. The app is expected to warn or
degrade rather than crash when optional generated files are missing.

Current airport manifest:

- `KRDU`: default.
- `KSJC`
- `CYLW`
- `CYYC`
- `CYVR`

## Common Commands

Frontend:

```bash
cd aeroviz-4d
npm install
npm run dev
npm run build
npm test -- --run
npm run test:coverage
npm run build:dsm-heightmap-terrain -- --airport KRDU
```

Python environment note from `aeroviz-4d/agent.md`: use the conda `aviation`
environment for project Python scripts and tests. Avoid system Python,
Homebrew Python, and unqualified `python` when running important project
commands.

Examples:

```bash
conda run -n aviation pytest aeroviz-4d/python/tests
conda run -n aviation pytest trajectory_data_process/tests
```

Preferred script interpreter:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python
```

Pipeline examples:

```bash
./preprocess_aeroviz_airport.sh KRDU

./generate_aeroviz_airport_procedure_data.sh KRDU

/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  -m trajectory_data_process.harvest --airport CYYC

/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  aeroviz-4d/python/generate_czml.py --airport CYYC \
  --input path/to/flight_array.json \
  --output aeroviz-4d/public/data/airports/CYYC/trajectories.czml
```

FAA RNAV charts:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  aeroviz-4d/python/download_faa_rnav_charts.py KRDU
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  aeroviz-4d/python/download_faa_rnav_charts.py KRDU --cycle 2604 --dry-run
```

OpenSky training data smoke command:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  trajectory_data_process/download_trajectories.py \
  --dataset-mode training \
  --airport KRDU \
  --begin 2026-04-19T10:00:00Z \
  --end 2026-04-19T10:15:00Z \
  --fetch-profile terminal_all \
  --max-trajectories 10
```

## Important Environment and Data Notes

- The Vite frontend requires `VITE_CESIUM_ION_TOKEN` in `.env` for Cesium Ion.
- `data/` is ignored by git and contains large/local datasets.
- `trajectory_data_process/credentials.json` and `trajectory_data_process/outputs`
  are ignored. Do not commit credentials or fetched raw outputs.
- Root `.gitignore` ignores PDFs, `.DS_Store`, `node_modules`, `data`, Python
  `__pycache__`, `.env`, and most generated browser data.
- `aeroviz-4d/public/data/common/airports.csv` and `runways.csv` are tracked
  common source data.
- `trajectory_data_process/download_trajectories.py` downloads airport trajectories
  from the OpenSky history DB through `traffic` (geometric altitude required); it has
  no live/REST/OAuth path.

## Domain Glossary

- ADS-B: aircraft broadcast surveillance data.
- TMA: Terminal Maneuvering Area, controlled airspace around airports.
- 4D trajectory: longitude, latitude, altitude, and time.
- CTA: Controlled Time of Arrival.
- RNAV/RNP: area navigation / required navigation performance procedure family.
- CIFP / ARINC 424: aviation procedure data source/record format.
- IAF, IF, FAF: Initial, Intermediate, and Final Approach Fix.
- OCS: Obstacle Clearance Surface.
- OCA/OCH: Obstacle Clearance Altitude/Height.
- DOF: FAA Digital Obstacle File.
- CZML: Cesium time-dynamic JSON format.
- GeoJSON: static geometry format used for runways, waypoints, procedures, and
  obstacles.
- DSM/DEM: digital surface/elevation model used for terrain context.

## Known Boundaries and Assumptions

- Procedure rendering is research-grade. Several path terminators and protected
  surface constructions are simplified, estimated, or flagged as future work.
- The CIFP parser intentionally covers a subset first and should preserve
  unsupported or unresolved legs as warnings instead of silently dropping them.
- Procedure details and UI should surface provenance, warnings, simplifications,
  and chart/data notes.
- Some vertical guidance and missed approach surfaces are debug/estimated
  geometry, not certified TERPS/PANS-OPS construction.
- OpenSky track quality varies. Altitude handling has multiple modes:
  `raw`, `touchdown-bias`, `approach-bias`, and `auto-bias`. Bias modes apply
  uniform shifts to preserve trajectory shape.
- If changing generated data paths, update both Python `data_layout.py` usage
  and frontend URL helpers in `src/data/airportData.ts`.
- Keep acquisition/network code separate from pure normalization and geometry
  utilities so tests remain deterministic.

## Useful Existing Docs

- `README.md`: root AeroViz RNAV chart and parser notes.
- `CLAUDE.md`: current development commands and architecture summary.
- `Description.md`: bilingual thesis/project description.
- `DEV_GUIDE_EN.md` / `DEV_GUIDE_ZH.md`: broader AeroViz development guide.
- `trajectory_data_process/README.md`: ADS-B acquisition and normalization
  workflow.
- `aeroviz-4d/readme.md`: app purpose, features, validation scope, and quick
  start.
- `aeroviz-4d/docs/10-rnav-procedure-parsing-visualization-pipeline.md`: RNAV
  procedure pipeline and warning policy.
- `aeroviz-4d/docs/11-rnav-procedure-intermediate-data-layer.md`: procedure
  intermediate data design.
- `aeroviz-4d/docs/18-refactor-vis-branch-dev-log.md`: current procedure
  visualization migration notes and known blockers.
- `data/DOF/README.md`: FAA obstacle data notes and AeroViz integration.
