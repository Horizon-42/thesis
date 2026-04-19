# DSM Heightmap Terrain Processing Review

This document explains the current heightmap terrain pipeline in
[`scripts/build_dsm_heightmap_terrain.mjs`](../scripts/build_dsm_heightmap_terrain.mjs)
and lists likely reasons the generated terrain can look noisy.

This is a review note only. It does not propose a final fix yet.

## Current Input

The script now reads all GeoTIFF files from:

```text
../data/bc_lidar/CYVR/dsm
```

The current generated metadata reports:

- source GeoTIFF count: `54`
- generated terrain tiles: `3702`
- mosaic source size: `12785 x 12567` projected metres
- generated terrain format: little-endian `Float32Array`
- Cesium terrain tile size: `129 x 129` samples
- generated Cesium levels: `0-16`
- height range: `-9.02 m` to `243.82 m`
- valid source sample count: `112,280,108`

The source CRS is treated as:

- horizontal: `EPSG:3157 NAD83(CSRS) / UTM zone 10N`
- vertical: `CGVD2013(CGG2013)`, used directly as metres

## Output Files

The script writes into:

```text
public/data/DSM/CYVR/heightmap-terrain
```

Important outputs:

- `metadata.json`: bounds, source file list, tile ranges, stats
- `tiles/{level}/{x}/{y}.f32`: Cesium terrain height tiles
- `dsm_height_overlay.png`: height tint sampled through the same lon/lat pipeline as the terrain
- `dsm_original_tif_heatmap.png`: projected mosaic heatmap sampled from the source GeoTIFF values

The metadata key is still named `originalTifHeatmap`, but after the multi-tile
refactor it is no longer a literal view of one raw TIFF pixel grid. It is a
projected mosaic heatmap sampled from the original TIFF values in UTM space.

## Generation Algorithm

### 1. Discover Source Tiles

The script lists all `.tif` and `.tiff` files in the input directory and sorts
them by filename.

For each file it reads:

- raster width and height
- projected origin
- projected resolution
- projected bounding box
- no-data value
- DSM height band
- per-file min, max, mean, and valid sample count

### 2. Build One Projected Mosaic Footprint

The script does not stitch a full in-memory raster. Instead, it computes the
union of all source projected bounds:

```text
west  = min(source west)
south = min(source south)
east  = max(source east)
north = max(source north)
```

That union is the terrain patch footprint. It is a rectangle, not an exact
coverage mask.

### 3. Build A Source Tile Index

To avoid scanning all 54 GeoTIFFs for every terrain sample, the script builds a
simple projected-space index:

```js
const SOURCE_TILE_INDEX_CELL_SIZE_M = 512;
```

Each source GeoTIFF is assigned to every 512 m index cell touched by its UTM
bounding box.

When the terrain builder needs a DSM value at one UTM coordinate, it only checks
the source tiles registered in that nearby index cell.

### 4. Choose Cesium Terrain Tiles

The script uses Cesium's `GeographicTilingScheme`.

For every level from `0` to `16`, it finds all Cesium geographic terrain tiles
that intersect the mosaic lon/lat rectangle.

Every generated tile contains:

```text
129 x 129 float32 height samples
```

These are written as raw little-endian `.f32` files.

### 5. Sample Each Terrain Height

For every sample in each Cesium terrain tile:

```text
Cesium tile row/column
  -> lon/lat in the Cesium geographic tile rectangle
  -> EPSG:3157 UTM easting/northing
  -> source GeoTIFF candidates from the 512 m index
  -> bilinear sample in each candidate source raster
  -> average valid candidate values
  -> fallback to 0 if no valid source value exists
```

The source-raster sampling uses pixel-center logic:

```text
x = (easting - originX) / resolutionX - 0.5
y = (northing - originY) / resolutionY - 0.5
```

The `-0.5` is the inverse of mapping a pixel center to projected coordinates.
It keeps the sample point aligned to source pixel centers rather than treating
the GeoTIFF origin as the center of the first pixel.

No-data values are ignored during bilinear interpolation. If all nearby source
pixels are no-data, the sample is considered missing.

### 6. Handle Overlapping Source Tiles

Some source GeoTIFFs can overlap.

The current rule is:

```text
if multiple source tiles provide valid values:
  use their arithmetic average
```

This is deterministic, but it does not prefer one survey date, one file, or one
quality level over another.

### 7. Runtime Loading In Cesium

The browser loads the output through
[`src/terrain/dsmHeightmapTerrain.ts`](../src/terrain/dsmHeightmapTerrain.ts).

It creates:

```ts
Cesium.CustomHeightmapTerrainProvider
```

The provider fetches `.f32` files on demand and returns `Float32Array` height
tiles to Cesium.

If Cesium asks for a tile outside the generated range, the runtime returns a
flat tile at:

```js
fallbackHeightM = 0
```

The demo page then sets:

```js
viewer.scene.verticalExaggeration = 25;
```

So a real `10 m` height difference is drawn as `250 m` of visible relief.

## Why The Result Can Look Noisy

### 1. DSM Is Not Bare-Earth Terrain

The input folder is `dsm`, not `dem`.

A DSM is a digital surface model. It includes surfaces such as:

- buildings
- aircraft and airport equipment, depending on acquisition timing
- vegetation and tree canopy
- bridges
- other above-ground structures

That means high-frequency spikes are expected. If the goal is a smooth ground
terrain surface, the `dem` folder is usually the better input.

### 2. The Current Demo Exaggerates Height By 25x

The current metadata has a max height of about:

```text
243.82 m
```

With `25x` visual exaggeration, that can be drawn as roughly:

```text
6095 m
```

Even normal DSM variation can become visually extreme. The screenshot's
needle-like terrain may be a display exaggeration problem before it is a
coordinate problem.

Review check:

1. Set terrain exaggeration to `1x`.
2. Compare the same area again.
3. If the scene becomes plausible, the main issue is visualization scale.

### 3. The Mosaic Footprint Is A Rectangle, Not An Exact Mask

The script generates a rectangular union of all source bounds.

If there are holes between source TIFFs, water areas, clipped source areas, or
large no-data regions inside that rectangle, terrain samples in those places
fall back to `0 m`.

Where valid DSM meets fallback `0 m`, Cesium will draw abrupt cliffs.

Review check:

1. Turn off `DSM terrain surface`.
2. Leave `Original TIFF heatmap` on.
3. Look for transparent or sharp-edged missing-data areas.

### 4. Source Tiles May Come From Different Dates

The source filenames include different acquisition date ranges, such as:

```text
20240217
20250425
20250826
```

If overlapping tiles were produced on different dates, buildings, parked
aircraft, vegetation, or water surfaces may not match exactly.

The current overlap rule averages values. Averaging can soften small
differences, but it can also create visible seams if two source surfaces are
very different.

### 5. The Pipeline Does No Smoothing Or Outlier Filtering

The builder currently uses the source DSM values directly.

It does not:

- smooth the raster
- remove isolated spikes
- clamp values to percentiles
- filter vegetation/buildings
- classify bare earth
- reject suspicious outliers

This is intentional for inspection, but it means source noise becomes terrain
noise.

### 6. Downsampling Can Alias Fine DSM Detail

The source DSM is roughly 1 m resolution. Cesium terrain tiles are generated as
`129 x 129` height grids inside geographic tiles.

At high levels this is fairly detailed, but it is still a resampling process.
At lower levels, many source pixels are represented by far fewer terrain
samples. High-frequency DSM features can alias into jagged terrain while Cesium
is switching terrain LODs.

### 7. Fallback Height Is 0 m

The builder uses:

```js
const FALLBACK_HEIGHT_M = 0;
```

This is safe for "missing data is flat" behavior, but it can be visually wrong
near valid surfaces whose real heights are much higher or lower.

In the current full mosaic, many valid areas have means tens of metres above
zero. A sudden fallback to `0 m`, then multiplied by `25x`, produces strong
walls and pits.

### 8. The Vertical Datum Is Used Directly

The source vertical CRS is `CGVD2013(CGG2013)`, while Cesium's globe is based on
an ellipsoid.

This should mostly affect absolute vertical offset, not random noise. It is
unlikely to explain the needle-like artifacts by itself, but it matters if we
later need survey-grade alignment with other terrain or aviation surfaces.

## How To Review Before Changing Processing

Use the demo toggles in `/dsm-terrain-demo`:

1. Turn off `DSM terrain surface`.
2. Turn on `Original TIFF heatmap`.
3. Turn on `Satellite image`.
4. Check whether the heatmap itself has noisy/spiky patterns.

Then:

1. Turn `DSM terrain surface` back on.
2. Set terrain exaggeration to `1x` in the HUD.
3. Compare the same area.

Interpretation:

- Heatmap noisy, terrain noisy: source DSM or no-data pattern is probably the main issue.
- Heatmap smooth, terrain noisy: terrain sampling, fallback values, or Cesium LOD is probably the main issue.
- Terrain only noisy at high exaggeration: viewer exaggeration is probably the main issue.
- Noise concentrated at tile edges: overlap handling, no-data gaps, or rectangular footprint is probably involved.
- Airport itself looks reasonable but surrounding city is spiky: DSM is behaving like a surface model, not bare-earth terrain.

## Likely Next Processing Choices

After review, the likely options are:

1. Use `data/bc_lidar/CYVR/dem` for actual Cesium terrain, and keep DSM as a separate obstacle/surface layer.
2. Keep DSM terrain but reduce exaggeration to `1x-3x`.
3. Add no-data masking so missing source areas do not become hard `0 m` cliffs.
4. Add an outlier report before generating terrain, especially for values above a chosen threshold.
5. Add optional smoothing or percentile clamping for visualization-only terrain.
6. Generate a separate exact source coverage mask so the terrain patch is not just the rectangular union.

For now, the processing should be reviewed before changing the algorithm again.
