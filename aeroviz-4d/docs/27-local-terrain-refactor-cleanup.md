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

## Candidate Old Data To Review

Do not delete these blindly; this is the review list for manual cleanup after
regeneration succeeds.

- `public/data/airports/CYVR/dsm/3dtiles/`
  - Old 3D Tiles terrain path. The current app uses heightmap `.f32` terrain.
- `public/data/airports/CYVR/dsm/heightmap-terrain/`
  - Existing generated package lacks `precision.horizontalResolutionM`; regenerate
    before keeping or deleting.
- `public/data/airports/KRDU/dsm/heightmap-terrain/`
  - Existing generated package currently points at `usgs-tnm-dem` and lacks
    precision metadata; regenerate so DEM/DSM priority is resolution-based.
- `public/data/airports/KSJC/dsm/heightmap-terrain/`
  - Existing generated package lacks `precision.horizontalResolutionM`; regenerate
    before keeping or deleting.

Source staging directories such as
`public/data/airports/KRDU/dsm/source/usgs-tnm-dem/` and
`public/data/airports/KRDU/dsm/source/usgs-tnm-dsm/` are not deletion candidates
until their regenerated `terrain-source.json` files and active heightmap package
have been verified.
