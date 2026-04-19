# DSM 3D Tiles Generation

This document explains the algorithm used by
[`scripts/build_dsm_3dtiles.mjs`](../scripts/build_dsm_3dtiles.mjs).

The script converts one DSM GeoTIFF into a small Cesium-ready 3D Tiles dataset.
It also writes debug assets that make it easier to verify whether the generated
mesh is positioned and oriented correctly on the globe.

## Inputs

The script looks for the DSM GeoTIFF in this order:

1. `../data/DSM/CYVR/bc_092g015_3_3_3_xli1m_utm10_20240217_20250425.tif`
2. `public/data/DSM/CYVR/bc_092g015_3_3_3_xli1m_utm10_20240217_20250425.tif`

The current file is treated as:

- projected coordinates in EPSG:3157, NAD83(CSRS) / UTM zone 10N
- one DSM height band
- one metre raster spacing
- `-32767` as no-data, when reported by the GeoTIFF

The script currently hard-codes `UTM_ZONE = 10`, so it is specific to this CYVR
tile unless that constant and the input path are generalized.

## Outputs

The script writes files to `public/data/DSM/CYVR/3dtiles/`:

- `dsm.glb`: the DSM surface as a glTF binary mesh
- `dsm.b3dm`: the same glTF wrapped as a 3D Tiles batched model
- `tileset.json`: the Cesium 3D Tiles entry point
- `dsm_overlay.png`: a colored 2D height image
- `dsm_overlay_exact.glb`: the 2D height image placed on an exact projected
  plane using the same coordinate transform as the DSM mesh
- `metadata.json`: useful bounds, CRS, statistics, and debug information

Run it with:

```bash
npm run build:dsm-tiles
```

## Coordinate Pipeline

The key idea is that the glTF mesh is not authored directly in longitude and
latitude. It is authored in a local East-North-Up coordinate frame, then the
3D Tiles root transform places that local frame on the Earth.

The pipeline for each DSM mesh vertex is:

```text
raster grid coordinate
  -> GeoTIFF projected easting/northing
  -> UTM zone 10 lon/lat
  -> Cesium ECEF Cartesian
  -> local ENU coordinate around the tile center
  -> glTF POSITION attribute
```

Cesium then applies the `tileset.json` root transform:

```text
local ENU glTF coordinate -> ECEF world coordinate -> rendered globe position
```

This is why the generated model can stay numerically small. The mesh vertices
are local metre offsets around the DSM tile center instead of large Earth-fixed
coordinates.

## Algorithm Steps

### 1. Read GeoTIFF Metadata

The script uses `geotiff` to read:

- raster width and height
- projected origin
- projected pixel resolution
- projected bounding box
- no-data value
- the DSM height raster

The bounding box gives the projected UTM extents:

```text
west easting, south northing, east easting, north northing
```

The origin and resolution are used later to map arbitrary sampled raster
positions back to projected coordinates.

### 2. Compute Raster Statistics

The script scans the height band and ignores no-data values. It computes:

- minimum DSM height
- maximum DSM height
- mean DSM height

The mesh does not use absolute DSM heights directly. Instead, it subtracts the
minimum height and applies a vertical exaggeration:

```text
rendered height = (sampled DSM height - min DSM height) * 80
```

This makes the small DSM height variation easier to see in the demo. It also
means the generated 3D model is a visualized DSM surface, not a survey-grade
absolute-height terrain replacement.

### 3. Resample Into a Martini Grid

The script builds a fixed `513 x 513` height grid:

```js
const GRID_SIZE = 513;
```

Martini requires a grid size of `2^n + 1`, and `513` is `512 + 1`.

For each grid cell, the script maps from Martini grid coordinates back into
the source raster:

```text
sourceX = gridX / (GRID_SIZE - 1) * (width - 1)
sourceY = gridY / (GRID_SIZE - 1) * (height - 1)
```

Then it samples the DSM using bilinear interpolation. If a sampled pixel is
missing or invalid, the raster minimum is used as a fallback.

### 4. Simplify the Surface With Martini

The script passes the `513 x 513` height grid to `@mapbox/martini`.

Martini builds a triangulated irregular network from the raster. The script
requests a mesh with:

```js
const MAX_ERROR_M = 0.08;
```

That error is applied to the unexaggerated sampled height grid. Lower values
keep more triangles. Higher values simplify more aggressively.

Martini returns:

- vertex grid coordinates
- triangle indices

Those are still raster-space values at this point. They are not yet globe
coordinates.

### 5. Convert Raster Vertices to Projected Coordinates

For each Martini vertex, the script maps from source raster coordinates to
projected UTM coordinates.

It uses pixel-center logic:

```text
easting  = originX + (sourceX + 0.5) * resolutionX
northing = originY + (sourceY + 0.5) * resolutionY
```

The `+ 0.5` matters: it places vertices at raster pixel centers instead of
pixel corners.

For this GeoTIFF, `resolutionY` is negative because raster rows increase
downward while northing increases upward.

### 6. Convert UTM to Longitude and Latitude

The script includes a local `utmToLonLat()` implementation for WGS84-style UTM
math. It converts the EPSG:3157 projected easting/northing into lon/lat degrees.

For this tile, the central meridian comes from UTM zone 10:

```text
central meridian = (zone - 1) * 6 - 180 + 3
```

Important assumption: this is close enough for visualization in Cesium. EPSG:3157
uses NAD83(CSRS), while Cesium's `Cartesian3.fromDegrees()` uses the WGS84
ellipsoid. For this demo-sized tile, that datum difference is small compared to
the visualization goal, but a production geospatial pipeline should use a full
projection library such as PROJ.

### 7. Convert Lon/Lat/Height to Local ENU

The script chooses the center of the GeoTIFF bounding box as the local origin:

```text
center easting  = (west easting + east easting) / 2
center northing = (north northing + south northing) / 2
```

That center is converted to lon/lat, then Cesium creates an East-North-Up
transform at that point:

```js
Cesium.Transforms.eastNorthUpToFixedFrame(
  Cesium.Cartesian3.fromDegrees(centerLon, centerLat, 0)
)
```

For every mesh vertex:

1. Convert projected easting/northing to lon/lat.
2. Convert lon/lat/rendered height to ECEF with `Cartesian3.fromDegrees()`.
3. Multiply by the inverse ENU transform.

The resulting local coordinates use:

- `x`: east
- `y`: north
- `z`: up

Those values become the glTF `POSITION` attribute.

### 8. Fix Triangle Winding and Normals

The script computes triangle normals from the local ENU coordinates.

If a triangle normal points downward, detected by `nz < 0`, the triangle winding
is flipped. Then the normal is accumulated into each triangle vertex and
normalized.

This produces upward-facing normals for lighting and avoids the mesh appearing
inside-out.

### 9. Write the DSM glTF

The script uses `@gltf-transform/core` to create a binary glTF:

- `POSITION`: local ENU vertex positions
- `NORMAL`: computed vertex normals
- indices: triangle index buffer
- material: green, rough, double-sided DSM surface

The generated file is:

```text
public/data/DSM/CYVR/3dtiles/dsm.glb
```

The glTF is authored as Z-up local geometry. When Cesium loads it directly,
the viewer must use:

```ts
upAxis: Cesium.Axis.Z,
forwardAxis: Cesium.Axis.X,
```

The `forwardAxis` value is important. Cesium's direct model loader defaults to
`forwardAxis: Z`, which applies an unwanted axis correction for this mesh. 3D
Tiles model content defaults to X forward, so the demo sets both paths
explicitly to keep them consistent.

### 10. Wrap the glTF as b3dm

Cesium 3D Tiles can reference different tile content formats. This script wraps
the GLB into a simple `b3dm` container.

The b3dm contains:

- a 28-byte b3dm header
- a feature table JSON with `BATCH_LENGTH: 0`
- no feature table binary
- no batch table
- the padded GLB payload

The generated file is:

```text
public/data/DSM/CYVR/3dtiles/dsm.b3dm
```

### 11. Write tileset.json

The script writes a single-root 3D Tiles tileset:

```text
tileset root
  -> transform: local ENU to ECEF
  -> boundingVolume.box: local mesh bounds
  -> content.uri: dsm.b3dm
```

The bounding volume is computed from the actual local mesh positions:

```text
box center = (local min + local max) / 2
half axes  = (local max - local min) / 2
```

The root transform is the Cesium ENU transform at the tile center. Together,
the local box and root transform tell Cesium where the tile lives on the globe
and when it should be rendered.

The tileset marks the glTF as Z-up:

```json
{
  "asset": {
    "version": "1.0",
    "gltfUpAxis": "Z"
  }
}
```

### 12. Generate the Height Overlay PNG

The script also creates `dsm_overlay.png`, a colored 2D visualization of the
DSM heights.

It rescales the DSM to a maximum width of `1024` pixels and colors each pixel
with a simple height ramp:

```text
low    -> blue
middle -> teal/yellow
high   -> red
```

No-data pixels are transparent.

This PNG by itself is only an image. It should not be treated as spatially
correct until it is placed with the same projected transform as the mesh.

### 13. Generate the Exact Overlay glTF

To verify orientation and placement, the script creates `dsm_overlay_exact.glb`.

This is a flat textured plane using the four projected GeoTIFF corners:

```text
southWest -> southEast -> northEast -> northWest
```

Each corner goes through the same coordinate pipeline as the DSM mesh:

```text
projected UTM corner -> lon/lat -> ECEF -> local ENU
```

The plane is placed above the DSM surface:

```text
overlay height = exaggerated model height + 25 metres
```

It floats above the mesh so you can visually compare footprint and orientation
without the DSM surface hiding it.

This exact overlay replaced the older rectangular lon/lat image overlay. A
`SingleTileImageryProvider` over a lon/lat rectangle is not a reliable verifier
for this file because the source raster is projected in UTM, not authored as a
simple geographic rectangle.

## Important Assumptions and Limitations

- The input CRS is assumed to be UTM zone 10N.
- The script performs its own UTM conversion instead of using PROJ.
- Vertical values are displayed relative to the raster minimum, not as absolute
  terrain heights.
- Vertical exaggeration is hard-coded to `80`.
- The tileset contains one b3dm tile, not a multiresolution tile pyramid.
- The generated DSM surface is suitable for visualization and debugging, not
  as a replacement for a proper terrain provider.

## How to Verify the Result

1. Run:

   ```bash
   npm run build:dsm-tiles
   ```

2. Open the DSM demo page.

3. Check three layers:

   - the green DSM mesh
   - the floating exact overlay image
   - the red GeoTIFF footprint polyline

If the green DSM mesh, exact overlay, and red footprint agree, then the raster
orientation and projected placement are consistent.

If only a rectangular imagery overlay disagrees, the problem is probably the
overlay method, not the DSM mesh. The exact overlay is the more trustworthy
debug reference because it uses the same transform as the generated model.
