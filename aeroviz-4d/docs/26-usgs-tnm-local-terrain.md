# USGS TNM Local Terrain Preprocessing

This note documents how to convert the USGS TNM elevation data under
`data/usgs_tnm_elevation` into the local terrain format that AeroViz-4D already
loads with `useAirportLocalTerrainLayer`.

## Inspected KRDU Inputs

The KRDU folder currently contains two terrain source shapes:

- `KRDU/dem/USGS_13_n36w079_20250507.tif`
  - Float32 GeoTIFF, 10812 x 10812
  - NAD83 geographic degrees
  - elevation values in metres
  - no-data value `-999999`
- `KRDU/dsm/source_laz/*.laz`
  - 10 USGS LPC LAZ files
  - no embedded SRS in the LAS headers
  - XY coordinates match NAD83 / North Carolina StatePlane ftUS (`EPSG:2264`)
  - Z values are feet and must be scaled by `0.3048`

The active local terrain package is selected by source precision, not by the
DEM/DSM label. The DEM remains useful for bare-earth clearance analysis; the
LAZ-derived DSM is useful when the desired surface includes buildings,
vegetation, and other above-ground returns.

## Output Contract

Both source kinds are normalized to GeoTIFF staging data, then converted to the
existing browser terrain package:

```text
public/data/airports/<ICAO>/dsm/heightmap-terrain/
  metadata.json
  tiles/<level>/<x>/<y>.f32
```

The frontend does not need changes for this package. The local terrain layer
already resolves:

```text
/data/airports/<ICAO>/dsm/heightmap-terrain/metadata.json
```

## Commands

Auto-stage available sources and publish the highest precision package:

```bash
PYTHONPATH=python python python/preprocess_usgs_tnm_terrain.py --airport KRDU
```

Bare-earth DEM only:

```bash
PYTHONPATH=python python python/preprocess_usgs_tnm_terrain.py --airport KRDU --source dem
```

LAZ-derived DSM:

```bash
PYTHONPATH=python python python/preprocess_usgs_tnm_terrain.py --airport KRDU --source dsm
```

Stage both normalized GeoTIFFs without publishing final `.f32` tiles:

```bash
PYTHONPATH=python python python/preprocess_usgs_tnm_terrain.py --airport KRDU --source both --stage-only
```

When `--source both` is used without `--stage-only`, `--publish-source dem` or
`--publish-source dsm` chooses which staged source becomes the active local
terrain package. The default `--publish-source auto` chooses the smallest
`precision.horizontalResolutionM`.

## Processing Path

DEM path:

1. Read the USGS DEM GeoTIFF.
2. Crop it to the airport footprint. For KRDU, the module uses the union of
   `product_bbox` values in `download_manifest.csv`.
3. Write a compact staged GeoTIFF under
   `public/data/airports/KRDU/dsm/source/usgs-tnm-dem/`.
4. Write `terrain-source.json` beside the staged GeoTIFF with
   `precision.horizontalResolutionM`.
5. Run `scripts/build_dsm_heightmap_terrain.mjs` against that staged source.

DSM path:

1. Read the LAZ files with PDAL using `override_srs=EPSG:2264` for KRDU.
2. Merge the point views.
3. Reproject XY to NAD83 UTM zone 17N (`EPSG:26917`).
4. Scale Z from feet to metres.
5. Rasterize with `writers.gdal` using `output_type=max` at 2 m resolution.
6. Write `terrain-source.json` beside the staged GeoTIFF with
   `precision.horizontalResolutionM`.
7. Run `scripts/build_dsm_heightmap_terrain.mjs` against the staged DSM GeoTIFF.

The final heightmap tiles use the same
`float32-little-endian-heightmap` contract documented in
`docs/24-airport-local-heightmap-terrain.md`.
