# Airport Local Hillshade Plan

## Goal

Add an airport-scoped hillshade overlay for local terrain so DEM/DSM relief is
readable with satellite imagery enabled, without changing the current terrain
height exaggeration workflow.

The first implementation uses one bounded PNG overlay per airport terrain
package. That keeps runtime cost small and avoids creating tens of thousands of
extra imagery tiles for large local terrain packages. If a future airport needs
close-range cartographic sharpness, the same metadata contract can be extended
to XYZ tiles.

## Scope

- Generate a multi-direction hillshade image beside each local heightmap
  terrain package.
- Expose the generated image through `metadata.json`.
- Load the image as a bounded Cesium imagery layer.
- Add a UI layer toggle named `Terrain Hillshade`.
- Use the valid-sample terrain footprint, not the full GeoTIFF bounding box, so
  no-data raster margins do not become visible local terrain.
- Fill local terrain no-data/fallback edge samples only at the highest detail
  level so close-up edges avoid tall 0 m cliffs without letting coarse parent
  tiles grow a raised apron during zoom.
- Do not set or change `scene.verticalExaggeration`; the HUD slider remains the
  only user-facing exaggeration control.
- Do not enable real terrain shadow maps by default.

## Data Contract

`public/data/airports/<ICAO>/local-terrain/heightmap/metadata.json` may include:

```json
"hillshade": {
  "url": "/data/airports/<ICAO>/local-terrain/heightmap/local_terrain_hillshade.png",
  "width": 1024,
  "height": 830,
  "alpha": 0.62,
  "note": "Multi-direction transparent hillshade overlay generated from local terrain source elevations."
}
```

The image is an RGBA overlay covering `metadata.bounds`. Shadow pixels use black
with variable alpha; highlight pixels use low-opacity white. This avoids the
flat gray wash that a semitransparent grayscale image can create over satellite
imagery.

`fallbackHeightM` stays at `0`. The generator writes `metadata.bounds` from the
valid elevation footprint and keeps the original GeoTIFF bounds on
`originalTifHeatmap.bounds` for source inspection. The frontend repairs fallback
samples only on `metadata.maxLevel`, so old low-detail parent tiles do not turn
no-data margins into raised terrain while the camera is zooming.

## Build Command

Full local terrain package generation writes hillshade automatically:

```bash
npm run build:local-terrain -- --airport KRDU
```

For existing packages, regenerate only visual inspection assets and metadata:

```bash
npm run build:local-terrain:visual-assets -- --airport KRDU
```

## Runtime Behavior

`useTerrainHillshadeLayer` reads the active airport-local terrain metadata. If a
`hillshade` entry exists and the layer is enabled, it adds a
`SingleTileImageryProvider` constrained to `metadata.bounds`.

The layer is independent of the local terrain provider. It can show over Cesium
World Terrain or airport-local heightmap terrain, but it disappears when the
active airport has no generated local terrain metadata or no hillshade asset.

## Performance Notes

- Single image overlay: one texture request per airport switch, bounded to the
  local terrain rectangle.
- No additional terrain mesh, no shadow map, and no terrain provider swap.
- Default alpha should stay low enough to preserve satellite imagery and
  procedure colors.
- If the image becomes visibly blurry during close-up inspection, add a tiled
  `hillshade.tiles` contract later with explicit min/max levels instead of
  replacing this lightweight default.

## Verification

- TypeScript build passes.
- Existing layer toggle tests are updated for the new layer key.
- The local terrain build script writes `local_terrain_hillshade.png` and the
  matching `metadata.hillshade` object.
- The Cesium hook removes the imagery layer on airport changes, layer disable,
  missing metadata, or unmount.
