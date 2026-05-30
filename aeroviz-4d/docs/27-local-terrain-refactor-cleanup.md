# Local Terrain Refactor Cleanup

## Current Contract

Airport-local terrain is no longer selected by DEM/DSM category. The active
package is selected by `precision.horizontalResolutionM`, where smaller values
mean higher precision.

Generated packages must include:

```json
"precision": {
  "horizontalResolutionM": 2.0,
  "verticalAccuracyM": null,
  "source": "terrain-source-metadata"
}
```

Old `metadata.json` files without this field are intentionally rejected by the
frontend so the app does not silently choose stale or lower-precision terrain.

## Regeneration Reminder

Regenerate active airport-local terrain packages before using the local terrain
toggle:

```bash
PYTHONPATH=python python python/preprocess_usgs_tnm_terrain.py --airport KRDU
npm run build:local-terrain -- --airport KSJC
npm run build:local-terrain -- --airport CYVR
```

For USGS TNM sources, the Python preprocessor writes `terrain-source.json` beside
each staged source, then publishes the highest precision source by default.

## Legacy Data To Review

Old generated data may still exist under the pre-refactor `dsm/` package name.
Do not delete these blindly; this is the review list for manual cleanup after
regeneration or migration succeeds.

- `public/data/airports/CYVR/dsm/3dtiles/`
  - Old 3D Tiles terrain path. The current app uses heightmap `.f32` terrain.
- `public/data/airports/<ICAO>/dsm/heightmap-terrain/`
  - Old package directory name. Migrate to
    `public/data/airports/<ICAO>/local-terrain/heightmap/`.
- `public/data/airports/<ICAO>/dsm/source/`
  - Old staged source parent. Migrate to
    `public/data/airports/<ICAO>/local-terrain/sources/`.

Source staging directories such as
`public/data/airports/KRDU/local-terrain/sources/usgs-tnm-dem/` and
`public/data/airports/KRDU/local-terrain/sources/usgs-tnm-dsm/` are not deletion candidates
until their regenerated `terrain-source.json` files and active heightmap package
have been verified.
