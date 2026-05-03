# USGS LiDAR DSM Downloader

This subpackage downloads USGS 3DEP LiDAR point cloud source around US airports
and derives DSM GeoTIFFs with PDAL. DEM rasters for the full AeroViz airport
pipeline are handled by `opentopography_downloader`.

## Sources

- TNM Access API products endpoint:
  <https://tnmaccess.nationalmap.gov/api/v1/products>
- TNM dataset list:
  <https://tnmaccess.nationalmap.gov/api/v1/datasets>
- DSM source for CONUS: `Lidar Point Cloud (LPC)` LAZ tiles
- Direct DSM source for Alaska where available: `Ifsar Digital Surface Model (DSM)`

Important USGS distinction: USGS does not usually publish CONUS DSM as ready-made
GeoTIFF tiles through TNM; the available high-resolution surface source is LPC
point cloud data. The script uses PDAL by default to rasterize those LAZ files
into DSM GeoTIFFs.

## Quick Start

Dry-run around an airport:

```bash
python usgs_lidar_downloader/download_usgs_lidar.py KSFO --dry-run
```

Download DSM source LAZ files and derived DSM GeoTIFFs:

```bash
python usgs_lidar_downloader/download_usgs_lidar.py KSFO
```

Multiple airports:

```bash
python usgs_lidar_downloader/download_usgs_lidar.py KSFO KLAX KJFK
```

By default the script downloads DSM inputs only. DSM for CONUS is downloaded as
LPC `.laz` point clouds and immediately rasterized with PDAL into `dsm/*.tif`
using 1 m cells and `max` aggregation. The derived DSM rasters are reprojected
to the airport's NAD83 UTM zone, matching the assumption in the current
heightmap builder.

The command-line interface is intentionally small. If a one-off run needs a
different radius, source dataset, PDAL executable, raster resolution, or DSM
aggregation method, edit the `DEFAULT_*` constants near the top of
`download_usgs_lidar.py`.

## Output Layout

Airport downloads use the airport code as the group folder:

```text
data/usgs_lidar/
  KSFO/
    download_manifest.csv
    dsm/
      *.tif              # created by the default PDAL DSM derivation
      source_laz/
        *.laz            # DSM source point clouds from USGS LPC
```

This mirrors the shape used by the existing BC data:

```text
data/bc_lidar/
  CYVR/
    dem/
      *.tif
    dsm/
      *.tif
```

The manifest keeps the common BC columns first:

```text
status,message,group,product,scale,maptile,filename,year,projection,spacing,url,target
```

Additional USGS fields follow those columns, including `dataset`, `source_id`,
`publication_date`, `bbox`, and `derived_target`.

## Frontend Terrain Build

After deriving DSM GeoTIFFs, run the existing heightmap builder:

```bash
cd aeroviz-4d
npm run build:dsm-heightmap-terrain -- --airport KSFO
```

The builder now checks `../data/usgs_lidar/<AIRPORT>/dsm` after
`../data/bc_lidar/<AIRPORT>/dsm`. You can also point it explicitly:

```bash
npm run build:dsm-heightmap-terrain -- \
  --airport KSFO \
  --input-dir ../data/usgs_lidar/KSFO/dsm
```

## Notes

- The default search radius is 5 km around each airport coordinate.
- Latest-per-tile filtering is enabled by default, which helps near airports
  where multiple 3DEP projects overlap the same 10 km DEM tile.
- `DEFAULT_DSM_SOURCE = "ifsar"` is mainly for Alaska. For most CONUS airports,
  keep the default LPC source.
- PDAL is required for the default DSM GeoTIFF generation path. Install it with
  `brew install pdal`, or set `DEFAULT_DERIVE_DSM = False` if you only want to
  download source files.
- Existing files are skipped by default. Use `--overwrite` to replace them.
