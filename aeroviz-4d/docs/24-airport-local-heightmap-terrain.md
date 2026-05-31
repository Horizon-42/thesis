# Airport Local Heightmap Terrain

This document describes the local high-resolution terrain path used for airport
analysis. It is the implementation of the "local airport terrain" approach:
use Cesium World Terrain as broad context, but use airport-scoped local
heightmap terrain where the project has DSM/DEM data.

## Naming Contract

`local-terrain` means the airport-scoped package that the frontend can load.
It is deliberately neutral: the active package may have been generated from a
bare-earth DEM, a surface DSM, or another future elevation source.

Source-specific names live only under `local-terrain/sources/`:

```text
public/data/airports/<ICAO>/local-terrain/sources/usgs-tnm-dem/
public/data/airports/<ICAO>/local-terrain/sources/usgs-tnm-dsm/
```

The source folder name describes the source type. It is not the runtime package
name and it is not used as a priority signal.

## Data Contract

Each airport can provide its own terrain package under:

```text
public/data/airports/<ICAO>/local-terrain/heightmap/
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

`metadata.bounds` is the renderable terrain footprint derived from valid
elevation samples. The full GeoTIFF extent remains in `sourceBounds` and, for the
raw inspection image, `originalTifHeatmap.bounds`. This distinction matters for
DSM products with large no-data margins: using the full raster rectangle would
make low-detail Cesium parent tiles appear to pull a flat local-terrain apron
outward during zoom.

The application does not special-case CYVR, KSJC, or any other airport. The
active airport code determines the metadata URL:

```text
/data/airports/<ICAO>/local-terrain/heightmap/metadata.json
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
npm run build:local-terrain -- --airport KRDU
```

To refresh only the generated inspection overlays, including hillshade and
height tint, without rewriting the `.f32` terrain tiles:

```bash
npm run build:local-terrain:visual-assets -- --airport KRDU
```

By convention, source discovery is handled by
`scripts/build_local_terrain_heightmap.mjs`. When no `--input-dir` is passed it
scores all known candidate GeoTIFF source directories by
`precision.horizontalResolutionM` from `terrain-source.json`, falling back to the
GeoTIFF raster transform when possible. Outputs always land in the airport's own
`public/data/airports/<ICAO>/local-terrain/heightmap/` folder.

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
5. By default the remaining tiles are streamed by Cesium on demand. Callers can
   opt into full-package background warming with `backgroundPreload: true`, but
   the main viewer leaves it off to avoid loading thousands of height tiles during
   normal airport switching.
6. Globe cache/refinement settings are tuned so the loaded local tile set stays
   resident:
   - `maximumScreenSpaceError = 1.5`
   - `tileCacheSize` bounded between the local minimum and maximum cache limits
   - `preloadAncestors = true`
   - `preloadSiblings = false`
   - `loadingDescendantLimit = 64`

This keeps airport-center switching responsive without forcing a full cache warm.
The user sees `Preload X/Y` for the blocking focused set first; once it switches
to `Active`, the provider is already installed and non-focused tiles stream only
when camera movement asks for them.

Two optional imagery-only visual aids can be layered above the terrain without
changing heights:

- `Terrain Hillshade`: a multi-direction relief cue with tuned
  brightness/contrast/gamma for low-relief airports
- `Terrain Height Tint`: the generated local height-color overlay for analysis
  coloring

Terrain `.f32` tiles use `0 m` for no-data samples. The generator and frontend
only repair fallback edge samples at the highest terrain level. Coarser parent
tiles keep no-data as flat fallback so Cesium's LOD refinement cannot promote
GeoTIFF no-data margins into raised terrain while zooming; close-up tiles still
get a small edge fill to reduce visible cliffs at the real data boundary.

When satellite imagery is hidden, the local heightmap uses a gray height-based
material with derivative-based relief. It does not draw contour bands. The
shader intentionally avoids `materialInput.slope` and `normalEC`: Cesium's
`CustomHeightmapTerrainProvider` reports `hasVertexNormals=false`, so a material
that requires normals is skipped by Cesium and falls back to a flat globe color.

## Multi-Airport Switching

Provider cache keys are metadata URLs, not hard-coded airport names. Switching
from CYVR to KSJC loads or reuses:

```text
/data/airports/CYVR/local-terrain/heightmap/metadata.json
/data/airports/KSJC/local-terrain/heightmap/metadata.json
```

Switching back to an airport reuses the loaded provider. Height tile promises are
kept in a bounded LRU cache so repeated inspection of the same area stays fast
without retaining every generated tile for every visited airport.

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
- `Active X/Y`: local terrain provider is installed and optional background
  warming is still running
- `Active`: local terrain provider is installed and the local tile cache is warm
- `Error`: metadata or tile loading failed; old packages without precision
  metadata also land here and show a regeneration dialog

## Limits

This approach gives deterministic local terrain loading for airports that have
heightmap packages. It does not create local terrain for airports with no DSM/DEM
source data. To add another airport, generate that airport's package with the
same path contract; no frontend code changes should be required.
