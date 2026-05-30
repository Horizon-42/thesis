# Airport Local Heightmap Terrain

This document describes the local high-resolution terrain path used for airport
analysis. It is the implementation of the "local airport terrain" approach:
use Cesium World Terrain as broad context, but use airport-scoped local
heightmap terrain where the project has DSM/DEM data.

## Data Contract

Each airport can provide its own terrain package under:

```text
public/data/airports/<ICAO>/dsm/heightmap-terrain/
```

Required files:

```text
metadata.json
tiles/<level>/<x>/<y>.f32
```

`metadata.json` must include precision metadata:

```json
"precision": {
  "horizontalResolutionM": 2.0,
  "verticalAccuracyM": null,
  "source": "terrain-source-metadata"
}
```

`horizontalResolutionM` is the ground sample distance used to choose among
candidate local sources. Smaller values win; DEM/DSM names are not priority
signals.

The application does not special-case CYVR, KSJC, or any other airport. The
active airport code determines the metadata URL:

```text
/data/airports/<ICAO>/dsm/heightmap-terrain/metadata.json
```

If the metadata file exists and has precision metadata, the app can use that
airport's local terrain. If it does not exist, the local terrain status is
`Missing`, a front-end dialog is shown, and the scene falls back to the current
terrain provider. If the package is old and lacks `precision.horizontalResolutionM`,
the status is `Error` and the dialog asks for regeneration.

## Build Command

The generator is airport-parametric:

```bash
npm run build:local-terrain -- --airport CYVR
npm run build:local-terrain -- --airport KSJC
npm run build:dsm-heightmap-terrain -- --airport CYVR
```

By convention, source discovery is handled by
`scripts/build_dsm_heightmap_terrain.mjs`. When no `--input-dir` is passed it
scores all known candidate GeoTIFF source directories by
`precision.horizontalResolutionM` from `terrain-source.json`, falling back to the
GeoTIFF raster transform when possible. Outputs always land in the airport's own
`public/data/airports/<ICAO>/dsm/heightmap-terrain/` folder.

## Runtime Behavior

The main viewer uses `useAirportLocalTerrainLayer` as the airport-local terrain manager.
When the `Airport Local Terrain` layer is enabled:

1. The hook resolves metadata from the active airport code.
2. The heightmap terrain provider is cached by metadata URL.
3. A focused set of airport-center tiles is preloaded first. This includes all
   available ancestors plus a highest-resolution tile neighborhood around the
   local terrain footprint center.
4. After focused preload completes, the provider is installed as
   `viewer.scene.terrainProvider`.
5. The remaining tiles listed in `metadata.levels` are warmed in the background
   with bounded concurrency.
6. Globe cache/refinement settings are tuned so the loaded local tile set stays
   resident:
   - `maximumScreenSpaceError = 0.5`
   - `tileCacheSize >= metadata.tileCount + 32`
   - `preloadAncestors = true`
   - `preloadSiblings = true`
   - `loadingDescendantLimit = 1000`

This keeps airport-center switching responsive without abandoning cache warming.
The user sees `Preload X/Y` for the blocking focused set first; once it switches
to `Active X/Y`, the provider is already installed and the rest of the airport
package is warming in the background.

When satellite imagery is hidden, the local heightmap uses a gray height-based
material with 5 m and 25 m contour bands. The shader intentionally avoids
`materialInput.slope` and `normalEC`: Cesium's `CustomHeightmapTerrainProvider`
reports `hasVertexNormals=false`, so a material that requires normals is skipped
by Cesium and falls back to a flat globe color.

## Multi-Airport Switching

Provider cache keys are metadata URLs, not hard-coded airport names. Switching
from CYVR to KSJC loads or reuses:

```text
/data/airports/CYVR/dsm/heightmap-terrain/metadata.json
/data/airports/KSJC/dsm/heightmap-terrain/metadata.json
```

Switching back to an airport reuses the loaded provider and its cached tile
promises for the lifetime of the viewer.

## Provider Priority

Only one provider can own `viewer.scene.terrainProvider` at a time. The priority
is:

1. Active airport-local terrain
2. Cesium World Terrain
3. Ellipsoid terrain

`useTerrainLayer` avoids overwriting the scene when airport-local terrain is
active, which prevents a late World Terrain promise from replacing the local
provider.

## HUD Status

The HUD exposes two separate terrain signals:

- `Load`: Cesium's live terrain tile queue from `tileLoadProgressEvent`
- `Local`: airport-local terrain state

`Local` can be:

- `Off`: layer disabled
- `Missing`: no local package exists for the active airport
- `Loading`: metadata/provider is being created
- `Preload X/Y`: local tiles are being loaded into cache
- `Active X/Y`: local terrain provider is installed and the remaining local
  tiles are warming in the background
- `Active`: local terrain provider is installed and the local tile cache is warm
- `Error`: metadata or tile loading failed; old packages without precision
  metadata also land here and show a regeneration dialog

## Limits

This approach gives deterministic local terrain loading for airports that have
heightmap packages. It does not create local terrain for airports with no DSM/DEM
source data. To add another airport, generate that airport's package with the
same path contract; no frontend code changes should be required.
