# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AeroViz-4D: Airport 4D trajectory and terrain digital-twin visualization system for thesis research. Combines a React/TypeScript/CesiumJS frontend with Python data pipeline tools to visualize aircraft trajectories (position + time) in 3D terminal airspace.

## Repository Layout

- **aeroviz-4d/** — Main visualization app (React + CesiumJS frontend, Python CZML generator)
- **trajectory_data_process/** — Trajectory acquisition, processing, and dataset helpers
- **bc_lidar_downloader/** — BC LiDAR terrain data downloader
- **run_asd-b_fetch_and_generate.py** — Orchestrator: fetch -> normalize -> generate CZML pipeline

## Build & Dev Commands

### Frontend (aeroviz-4d/)

```bash
cd aeroviz-4d
npm install
npm run dev                          # Vite dev server with HMR
npm run build                        # tsc + vite production build
npm test                             # Vitest (watch mode)
npx vitest run                       # Single run, no watch
npx vitest run src/utils/__tests__/ocsGeometry.test.ts  # Single test file
npm run test:coverage                # Coverage report
npm run build:dsm-tiles              # Build 3D Tiles from GeoTIFF
npm run build:dsm-heightmap-terrain  # Generate heightmap terrain tiles
```

### Python (aeroviz-4d/python/)

```bash
pip install -r aeroviz-4d/python/requirements.txt
python -m pytest aeroviz-4d/python/tests/test_generate_czml.py -v
python -m pytest aeroviz-4d/python/tests/ --cov=. --cov-report=html
```

### Data Pipeline (end-to-end)

```bash
# Full pipeline: fetch live data → normalize → generate CZML
python run_asd-b_fetch_and_generate.py --airport CYYC --mode live

# Reuse existing raw JSON (skip fetch, run normalization + generation)
python run_asd-b_fetch_and_generate.py --input-json trajectory_data_process/outputs/cyyc_raw_*.json

# Skip straight to CZML generation from already-normalized data
python run_asd-b_fetch_and_generate.py --input-json trajectory_data_process/outputs/cyyc_czml_input_*.json
```

## Architecture

### Frontend State & Component Structure

Global state lives in `AppContext` (context + useState, no Redux). Key state: `viewer` (CesiumJS Viewer instance), `airport` config, `selectedFlightId`, `layers` visibility toggles, `playbackSpeed`.

Components read context via `useApp()` hook. CesiumJS logic is encapsulated in custom hooks:
- `useCesiumViewer` — initializes Viewer, loads airport.json, sets camera
- `useCzmlLoader` — loads CZML data source, syncs Cesium clock
- `useRunwayLayer` / `useTerrainLayer` — data layer management
- `useDsmTerrainLayer` — loads preprocessed `.f32` heightmap tiles via `terrain/dsmHeightmapTerrain.ts`; returns `{ status, metadata, provider, error }` so consumers can display terrain info or manage toggling; controlled by `layers.dsmTerrain` toggle in the main app

UI components (ControlPanel, HUD, FlightTable) overlay on the Cesium canvas via CSS grid with `pointer-events: none`.

### Data Flow

```
OpenSky API → fetch_cylw_opensky.py → *_raw_*.json
    → trajectory_normalization.py (altitude bias correction, filtering) → *_czml_input_*.json
    → generate_czml.py (bearing, velocity, orientation) → trajectories.czml
    → useCzmlLoader hook → CesiumJS rendering
```

Static data follows a similar pattern: OurAirports CSV → `preprocess_airports.py` → `runway.geojson`; ARINC 424 CIFP → `preprocess_waypoints.py` → `waypoints.geojson`.

### Key Data Formats

- **CZML**: JSON array where first element is a "document" packet (clock config), subsequent elements are entity packets with time-sampled positions via `cartographicDegrees: [secondsOffset, lon, lat, altMetres, ...]`
- **GeoJSON**: Used for static layers (runways, waypoints, OCS surfaces)
- Airport config: `public/data/airport.json` — `{code, lon, lat, height}`

### Utility Modules

- `ocsGeometry.ts` — Pure math for PANS-OPS obstacle clearance surface computation (no side effects)
- `czmlBuilder.ts` — Pure CZML packet construction helpers

## Environment

- Requires `VITE_CESIUM_ION_TOKEN` in `.env` (Cesium Ion access token)
- Vite config uses `vite-plugin-cesium` which handles Cesium asset copying and `CESIUM_BASE_URL` setup
- TypeScript strict mode is enabled (strict null checks, noUnusedLocals, noUnusedParameters)
- Test environment: jsdom with vitest globals enabled

## Domain Context

This is a thesis research project. Key aviation concepts in the code:
- **TMA** (Terminal Maneuvering Area) — controlled airspace around airports
- **OCS** (Obstacle Clearance Surface) — PANS-OPS geometry ensuring terrain clearance on approach
- **4D Trajectory** — aircraft position (lon, lat, alt) + time; the "4th dimension" is the scheduled arrival time
- **CTA** (Controlled Time of Arrival) — ATC-assigned time slot at a fix point

The project serves dual purposes: thesis visualization/validation, and reusable research component library.

## Changelog

### 2026-04-20 — Finish OCS geometry and add final-approach OCS layer

Completed the PANS-OPS final-approach Obstacle Clearance Surface (OCS) pipeline: filled in the TODO in `src/utils/ocsGeometry.ts`, wrote full unit-test assertions, and added a new `useOcsLayer` hook that derives FAF→threshold pairs from `procedures.geojson` and renders three semi-transparent Cesium polygons per route (red primary + two orange 7:1-slope secondary panels, `perPositionHeight: true` for the slope to show).

- `src/utils/ocsGeometry.ts` — implemented `buildFinalApproachOCS` (bearing → perpendiculars → primary trapezoid → secondary outer edges with `faf.altM − secondaryWidthM/7` drop at FAF and `threshold.altM` at the runway end). 13/13 unit tests pass.
- `src/hooks/useOcsLayer.ts` (new) — dual-useEffect pattern matching `useObstacleLayer`; primary half-width pulled from the route's tunnel descriptor (`tunnel.lateralHalfWidthNm × 1852`), falls back to 150 m.
- `src/components/CesiumViewer.tsx` — activated `useOcsLayer()`.
- `src/components/ControlPanel.tsx` — added `ocsSurfaces` to `ACTIVE_LAYER_KEYS` so the toggle renders.
- `docs/03-ocs-geometry.zh.md` (new) — Chinese tutorial with the flat-earth math derivation, a worked KRDU R05LY example, the altitude-provenance section (geometry altitude vs MCA and how to switch), and a concepts clarifier for OCS vs OCH vs MCA.

Altitudes are currently read from the LineString z-values (i.e. CIFP `geometryAltitudeFt × 0.3048`). Switching to MCA (`altitudeFt`) is a one-function change documented in `docs/03-ocs-geometry.zh.md §5.6`.

### 2026-04-20 — Add FAA DOF obstacle visualization layer

Added end-to-end pipeline for rendering FAA Digital Obstacle File (DOF) obstacles as 3D cylinders in CesiumJS. Obstacles are color-coded by type (TOWER=red, BLDG=steelblue, WINDMILL=green, etc.) and positioned with `HeightReference.RELATIVE_TO_GROUND` so they sit on terrain.

- `python/preprocess_obstacles.py` — parses fixed-width DOF `.Dat` files, filters by haversine radius (default 20 km / ~10.8 NM to cover the approach corridor), outputs `obstacles.geojson`
- `useObstacleLayer` hook — loads GeoJSON, creates cylinder entities with AGL-height labels; follows `useWaypointLayer` dual-useEffect pattern
- Added `"obstacles"` to `LayerKey` with toggle in `ControlPanel`
- DOF data documentation at `data/DOF/README.md`

Usage: `python preprocess_obstacles.py --input <DOF .Dat> --airport`

### 2026-04-19 — Refactor DSM terrain into reusable hook

Rewrote `useDsmTerrainLayer` to use the preprocessed heightmap pipeline (`terrain/dsmHeightmapTerrain.ts`) instead of decoding raw GeoTIFF in the browser. The hook now returns `{ status, metadata, provider, error }` and can be dropped into any page.

- `DsmTerrainDemoPage` delegates terrain loading to the hook (keeps its own overlay/camera logic)
- `CesiumViewer` wires the hook so DSM terrain is available in the main flight view
- Added `dsmTerrain` to `LayerKey` with a toggle in `ControlPanel`
