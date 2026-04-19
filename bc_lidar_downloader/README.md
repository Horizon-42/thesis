# BC LiDAR DEM/DSM Downloader

This folder contains a standalone downloader for the public LidarBC portal.
It queries the portal's ArcGIS FeatureServer index and downloads the direct
GeoTIFF links from the returned `s3Url` fields.

## Source

- Portal: <https://lidar.gov.bc.ca/pages/download-discovery>
- FeatureServer: <https://services1.arcgis.com/xeMpV7tU1t4KD3Ei/arcgis/rest/services/LidarBC_Open_LIDAR/FeatureServer>

Relevant layers:

- DSM 1:2,500 index: layer `1`
- DSM 1:10,000 index: layer `2`
- DSM 1:20,000 index: layer `3`
- DEM 1:2,500 index: layer `5`
- DEM 1:20,000 index: layer `6`

## Quick Start

Dry-run around an airport first. Airport coordinates are read from
`aeroviz-4d/public/data/airports.csv`.

```bash
python bc_lidar_downloader/download_bc_lidar.py \
  --airports CYVR \
  --airport-radius-km 5 \
  --product both \
  --scale 2500 \
  --latest-per-tile \
  --out data/bc_lidar \
  --dry-run
```

Download them:

```bash
python bc_lidar_downloader/download_bc_lidar.py \
  --airports CYVR \
  --airport-radius-km 5 \
  --product both \
  --scale 2500 \
  --latest-per-tile \
  --out data/bc_lidar
```

You can also pass multiple airports:

```bash
python bc_lidar_downloader/download_bc_lidar.py \
  --airports CYVR CYYC CYLW \
  --airport-radius-km 5 \
  --product both \
  --scale 2500 \
  --latest-per-tile \
  --out data/bc_lidar
```

Use a tile list file:

```bash
python bc_lidar_downloader/download_bc_lidar.py \
  --tile-file bc_lidar_downloader/tiles.txt \
  --product dsm \
  --scale 2500 \
  --out data/bc_lidar
```

The script writes `download_manifest.csv` under the output directory unless
`--manifest` is provided.

Airport downloads are grouped by airport, so multiple airports do not mix files
in the same folder:

```text
data/bc_lidar/
  CYVR_Vancouver_International_Airport/
    dem/
      *.tif
    dsm/
      *.tif
  CYYC_Calgary_International_Airport/
    dem/
      *.tif
    dsm/
      *.tif
```

## Notes

- `--scale 2500` is recommended when you need both DEM and DSM around airports.
  For example, CYVR has DEM coverage in the 1:20,000 index but DSM coverage is
  available from the 1:2,500 index.
- Use `--airports` with `--airport-radius-km` for airport-centred downloads.
  Airport code matching checks `ident`, `icao_code`, `gps_code`, `local_code`,
  and `iata_code` in `aeroviz-4d/public/data/airports.csv`.
- Some tiles have more than one record from different acquisition years. Use
  `--latest-per-tile` to keep only the newest record for each product/tile, or
  `--year 2023` to request a specific year.
- DSM has 1:2,500, 1:10,000, and 1:20,000 index layers.
- DEM has 1:2,500 and 1:20,000 index layers.
- Existing files are skipped by default. Use `--overwrite` to replace them.
- Use `--workers` to control concurrent downloads. The default is `4`.
