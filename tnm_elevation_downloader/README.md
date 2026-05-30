# TNM Elevation Downloader

Airport-centered downloader for high-resolution USGS The National Map elevation
products. It supports:

- ICAO/airport identifiers from `aeroviz-4d/public/data/common/airports.csv`
- direct `--lat` / `--lon` point queries
- a default 5 km search radius around each airport or point
- DEM downloads from TNM 3DEP DEM products
- DSM downloads where raster DSM exists, or lidar point cloud source for DSM
  derivation in most CONUS areas

## Sources Read

- The National Map Downloader: <https://apps.nationalmap.gov/downloader/>
- TNMAccess API docs: <https://tnmaccess.nationalmap.gov/api/v1/docs>
- TNM dataset list: <https://tnmaccess.nationalmap.gov/api/v1/datasets>
- USGS FAQ on TNMAccess: <https://www.usgs.gov/faqs/there-api-accessing-national-map-data>
- USGS FAQ on elevation products/formats:
  <https://www.usgs.gov/faqs/what-types-elevation-datasets-are-available-what-formats-do-they-come-and-where-can-i-download>

The Downloader web app is backed by TNMAccess. This package calls the
`/products` endpoint with a WGS84 `bbox`, `datasets`, `prodFormats`, `max`, and
`offset`, then writes a manifest before downloading files.

## Quick Start

Inspect DEM products around an airport without downloading:

```bash
python -m tnm_elevation_downloader.download_tnm_elevation KRDU --dry-run
```

Download DEM products around an airport using the default 5 km radius:

```bash
python -m tnm_elevation_downloader.download_tnm_elevation KRDU --product dem
```

Download DSM source data around a latitude/longitude point:

```bash
python -m tnm_elevation_downloader.download_tnm_elevation \
  --lat 35.87764 --lon -78.78747 --label KRDU_point --product dsm
```

Download both DEM and DSM source products around multiple airports:

```bash
python -m tnm_elevation_downloader.download_tnm_elevation \
  KRDU KDEN KSFO --product both --radius-km 8 --dry-run
```

When downloads run, the script prints an aggregate progress bar with completed
file count, known bytes, and transfer speed. Use `--no-progress` if you are
capturing logs and only want the final summary plus manifest.

## Dataset Keys

DEM choices:

- `dem_1m` - 3DEP 1 meter DEM tiles (`GeoTIFF,IMG`)
- `dem_s1m` - seamless 1 meter DEM, limited availability (`GeoTIFF`)
- `dem_opr` - original product resolution DEM source products
- `dem_13` - 1/3 arc-second DEM, about 10 meters
- `dem_1` - 1 arc-second DEM, about 30 meters

DSM choices:

- `dsm_lpc` - lidar point cloud (`LAS,LAZ`), the usual high-resolution source
  for deriving DSM rasters in CONUS
- `dsm_ifsar` - IfSAR DSM raster (`TIFF`), mainly Alaska where available

By default, DEM queries start with `dem_1m` and fall back to broader DEM
coverage if no 1 meter products intersect the query box. Use
`--no-dem-fallback` to require the selected DEM dataset exactly.

The downloader also keeps only the latest returned product per dataset
footprint by default, which avoids downloading multiple historical versions of
the same DEM tile. Use `--include-historical` when you need every version TNM
returns.

Downloads are concurrent by default with `--workers 4`. That is intentionally
conservative for TNM/USGS public endpoints: it improves throughput without
opening dozens of simultaneous connections. If a run is throttled or unreliable,
drop to `--workers 1` or `--workers 2`; for a stable connection, modest values
such as `--workers 6` can be tried.

## Output Layout

```text
data/usgs_tnm_elevation/
  KRDU/
    dem/
      *.tif or *.img
    dsm/
      source_laz/
        *.laz
  download_manifest.csv
```

The manifest records status, product family, dataset key, TNM title/source ID,
query bbox, product bbox, download URL, target path, and product notes.

## DSM Note

For many CONUS airports, TNM does not expose ready-made DSM GeoTIFF tiles.
Use `--product dsm --dsm-source dsm_lpc` to download LAZ point cloud source
tiles, then rasterize them with PDAL or the existing
`usgs_lidar_downloader/download_usgs_lidar.py` workflow if you need a DSM
GeoTIFF for Cesium terrain preprocessing.
