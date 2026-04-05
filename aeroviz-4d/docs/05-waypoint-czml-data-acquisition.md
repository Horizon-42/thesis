# Tutorial 05 - Waypoint and CZML Data Acquisition

Covers:
- `python/preprocess_waypoints.py`
- `python/generate_czml.py`
- runtime files under `public/data/`

## Goal

Provide a repeatable pipeline for:
1. collecting waypoint/navigation data,
2. normalizing it into frontend-friendly GeoJSON,
3. generating time-based trajectory CZML for Cesium playback.

## Data Sources

### OpenNav (waypoints)
- Example URL: `https://opennav.com/waypoint/CA`
- Provides waypoint rows with:
  - ident,
  - latitude (mostly DMS string),
  - longitude (mostly DMS string).

### HorizonPilot downloads (bulk reference datasets)
- Example page: `https://sites.google.com/view/horizonpilot/pre-flight-planning/data-base-downloads`
- Provides links to larger files (Excel/CSV/KML/TXT/PDF), including:
  - waypoints,
  - navaids,
  - aerodromes,
  - duplicate-waypoint notes.

Use OpenNav for fast public extraction and HorizonPilot as a cross-check or bulk fallback.

## Pipeline Overview

1. Fetch source rows (OpenNav HTML table).
2. Parse and normalize coordinates to decimal degrees.
3. Filter by airport-centered region (default CYLW radius).
4. Build `waypoints.geojson` for map rendering.
5. Build `trajectories.czml` for 4D playback.
6. Validate output with tests and frontend build.

## Coordinate Normalization Rules

`preprocess_waypoints.py` supports these coordinate forms:
- Standard DMS: `49° 7' 0.00" N`
- DMS with HTML entities
- Compact DMS variants (where present)
- Decimal fallback

Normalization checks:
- latitude must be in `[-90, 90]`
- longitude must be in `[-180, 180]`
- invalid rows are skipped

## Waypoint Role Assignment

OpenNav table entries do not encode procedure role directly. For visualization,
this project assigns role buckets by far-to-near ordering relative to the
reference airport:
- early segment: `IAF`
- middle segment: `IF`
- late segment: `FAF`
- end segment: `MAPt`

This is a display-oriented approximation, not a certified procedure database.

## Outputs

### `public/data/waypoints.geojson`
Feature schema:
- geometry: `Point [lon, lat, altM]`
- properties:
  - `name`
  - `type` (`IAF|IF|FAF|MAPt`)
  - `sequence`
  - `procedure`
  - `source`
  - `distance_km`

### `public/data/trajectories.czml`
- First packet is `document` with simulation clock.
- Remaining packets are aircraft entities with:
  - sampled position (`cartographicDegrees`),
  - model,
  - velocity-based orientation,
  - path trail,
  - label.

## Reproducible Commands (aviation environment)

From project root:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python python/preprocess_waypoints.py
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python python/generate_czml.py
```

Run validation:

```bash
cd python
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python -m pytest tests/test_generate_czml.py

cd ../
npm test -- --run src/utils/__tests__/czmlBuilder.test.ts
npm run build
```

## Quality and Risk Notes

1. Waypoint duplicates can exist across regions with same ident.
2. Public web table formats may change; parser should be kept simple and tested.
3. OpenNav rows may contain mixed precision and formatting styles.
4. Generated role labels are heuristic unless backed by formal procedure data.
5. CZML timing quality depends on input waypoint time offsets.

## Recommended Refresh Strategy

- Regenerate waypoints when source table changes or airport scope changes.
- Regenerate CZML whenever scheduling output changes.
- Keep generated files in `public/data/` for direct frontend loading.
- Re-run tests/build after every data regeneration step.
