# AeroViz-4D

AeroViz-4D is a research visualization prototype for 4D aircraft trajectory
prediction, procedure-aware visualization, and validation. It combines a
Cesium-based 3D scene with 2D analytical views so predicted or replayed aircraft
trajectories can be checked against runway geometry, RNAV procedure paths,
obstacle surfaces, terrain context, and time-varying motion.

The main aim is to support thesis work on 4D trajectory prediction
visualization and validation: the app makes it easier to inspect where an
aircraft is, when it is there, how it moves relative to published procedure
geometry, and whether the trajectory behavior is plausible in the runway and
approach context.

## Main Functions

- 3D airport visualization with Cesium, including runway surfaces, waypoints,
  obstacles, terrain layers, procedure routes, and aircraft trajectories.
- 4D trajectory playback from CZML, with simulation time controlling aircraft
  positions and trail/history behavior.
- RNAV procedure visualization from parsed CIFP data, including runway-grouped
  procedure controls, branches, fixes, route visibility, and approximate
  procedure tunnels.
- Runway-scoped 2D trajectory profile panel, with vertical `x-z` and plan
  `x-y` views for validating aircraft motion relative to the selected runway
  threshold, centerline, and RNAV horizontal plate.
- Procedure Details page for inspecting an individual RNAV procedure, including
  plan view, vertical profile, branch/fix focus, procedure metadata, local chart
  links, data notes, and aviation term explanations.
- Data preprocessing scripts for browser-ready airport, runway, obstacle,
  waypoint, procedure, and chart assets.

## Validation Purpose

AeroViz-4D is intended to help validate trajectory prediction results visually
and analytically:

- compare predicted or replayed aircraft paths with RNAV procedure geometry;
- inspect altitude and lateral behavior in runway-centered 2D views;
- check whether aircraft remain inside a procedure's horizontal plate;
- review procedure fixes, altitude constraints, and branch structure;
- use the 3D scene to cross-check spatial context against terrain, obstacles,
  runways, and approach paths.

This project is for research and visualization only. It is not certified
navigation software, not an operational flight tool, and not a replacement for
official FAA or local published procedure charts.

## Tech Stack

- React 18 and TypeScript
- Vite
- CesiumJS
- Vitest and Testing Library
- Python preprocessing scripts for aviation datasets

## Project Structure

```text
aeroviz-4d/
  src/                 React, Cesium hooks, UI components, geometry utilities
  public/data/         Browser-served generated datasets
  python/              Dataset preprocessing and validation scripts
  scripts/             Supporting build scripts
  docs/                Development plans, pipeline notes, and dev logs
```

## Quick Start

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Build the frontend:

```bash
npm run build
```

Run tests:

```bash
npm test -- --run
```

## Important Data Notes

Generated browser data under `public/data/` is part of the local working
dataset. Some procedure-detail and chart assets may need to be regenerated for
fresh checkouts or new airports.

The procedure pipeline currently focuses on research-grade RNAV visualization.
Some ARINC/CIFP path terminators, official vertical profile records, and
chart-specific details are simplified or reserved for future work. Known
simplifications are surfaced in the UI where available.

## Related Documentation

- `docs/10-rnav-procedure-parsing-visualization-pipeline.md`
- `docs/12-runway-rnav-trajectory-profile-view.md`
- `docs/14-procedure-details-page-plan.md`
- `docs/15-procedure-details-page-dev-log.md`
