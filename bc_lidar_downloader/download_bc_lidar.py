#!/usr/bin/env python3
"""
Download BC LiDAR DEM/DSM tiles from the public LidarBC ArcGIS index.

How this downloader works
-------------------------
The LidarBC web page is an ArcGIS Hub site. The clickable map in that site is
backed by an ArcGIS FeatureServer. Each FeatureServer layer is just a searchable
table of tile polygons. The important fields in that table are:

* maptile: the BCGS tile identifier
* filename: the GeoTIFF file name
* year: acquisition/publication year
* s3Url: the direct object-store URL that the browser downloads

So this script does the same work a user would do manually:

1. Resolve input, for example airport code CYVR -> latitude/longitude from
   aeroviz-4d/public/data/airports.csv.
2. Query the ArcGIS layer for DEM and/or DSM records near that coordinate.
3. Write a CSV manifest so you can inspect exactly what matched.
4. Download each returned s3Url concurrently.

No ArcGIS Python package is required. The ArcGIS REST API is regular HTTP with
form parameters, and the downloads are normal HTTPS file downloads.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


SERVICE_ROOT = (
    "https://services1.arcgis.com/xeMpV7tU1t4KD3Ei/arcgis/rest/services/"
    "LidarBC_Open_LIDAR/FeatureServer"
)

# Product/layer mapping from the LidarBC FeatureServer.
LAYER_IDS: dict[tuple[str, str], int] = {
    ("dsm", "2500"): 1,
    ("dsm", "10000"): 2,
    ("dsm", "20000"): 3,
    ("dem", "2500"): 5,
    ("dem", "20000"): 6,
}

OUT_FIELDS = [
    "OBJECTID",
    "filename",
    "maptile",
    "path",
    "grid_scale",
    "year",
    "projection",
    "spacing",
    "contract",
    "oper_name",
    "op_number",
    "s3Url",
]

TILE_ID_RE = re.compile(r"^[0-9A-Za-z]+$")


@dataclass(frozen=True)
class IndexedTile:
    # One row returned by a LidarBC index layer.
    #
    # `group` is used only for organizing output. For airport-based queries it
    # becomes something like "CYVR_Vancouver_International_Airport", which keeps
    # files for multiple airports from being mixed in one directory.
    group: str | None
    product: str
    scale: str
    attributes: dict[str, Any]

    @property
    def url(self) -> str:
        return str(self.attributes.get("s3Url") or "").strip()

    @property
    def filename(self) -> str:
        filename = str(self.attributes.get("filename") or "").strip()
        parsed = urlparse(self.url)
        return filename or Path(parsed.path).name

    @property
    def maptile(self) -> str:
        return str(self.attributes.get("maptile") or "").strip()


@dataclass
class DownloadResult:
    tile: IndexedTile
    target: Path
    status: str
    message: str = ""


@dataclass(frozen=True)
class Airport:
    # Minimal airport record needed for LiDAR lookup. The source CSV contains
    # many more columns, but only code/name/coordinate are needed here.
    code: str
    name: str
    latitude: float
    longitude: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query and download multiple BC LiDAR DEM/DSM tiles from the "
            "public LidarBC ArcGIS index."
        )
    )
    parser.add_argument(
        "--product",
        choices=["dem", "dsm", "both"],
        default="both",
        help="Product type to download. Default: both.",
    )
    parser.add_argument(
        "--scale",
        choices=["20000", "10000", "2500"],
        default="20000",
        help=(
            "BCGS index grid scale. DEM supports 20000 and 2500; DSM supports "
            "20000, 10000, and 2500. Default: 20000."
        ),
    )
    parser.add_argument(
        "--tiles",
        nargs="*",
        default=[],
        help="BCGS tile IDs, separated by spaces or commas, for example 092c058 092b041.",
    )
    parser.add_argument(
        "--tile-file",
        type=Path,
        help="Text file with one BCGS tile ID per line. Commas are also accepted.",
    )
    parser.add_argument(
        "--airports",
        nargs="*",
        default=[],
        help=(
            "Airport identifiers from airports.csv, separated by spaces or commas. "
            "Matches ident, ICAO, GPS, local, or IATA code. Example: CYVR"
        ),
    )
    parser.add_argument(
        "--airport-csv",
        type=Path,
        default=Path("aeroviz-4d/public/data/airports.csv"),
        help="Airport CSV path. Default: aeroviz-4d/public/data/airports.csv.",
    )
    parser.add_argument(
        "--airport-radius-km",
        type=float,
        default=5.0,
        help="Search radius around each airport coordinate in kilometres. Default: 5.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Query tiles intersecting a WGS84 longitude/latitude bounding box.",
    )
    parser.add_argument(
        "--where",
        help=(
            "Extra ArcGIS SQL where clause. Example: \"maptile LIKE '092c%%'\". "
            "Combined with tile/year filters using AND."
        ),
    )
    parser.add_argument("--year", type=int, help="Filter acquisition year.")
    parser.add_argument(
        "--latest-per-tile",
        action="store_true",
        help=(
            "If multiple records match the same product/maptile, keep only the "
            "record with the highest acquisition year."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/bc_lidar"),
        help="Output directory. Default: data/bc_lidar.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest CSV path. Default: <out>/download_manifest.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only query and write the manifest; do not download files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of skipping them.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent downloads. Default: 4.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Network timeout in seconds. Default: 120.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Download retry attempts per file. Default: 3.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum records per product to query. Useful for testing.",
    )
    return parser.parse_args()


def normalize_codes(raw_codes: Iterable[str]) -> list[str]:
    # argparse gives a list, but users may type either:
    #   --airports CYVR CYYC
    # or:
    #   --airports CYVR,CYYC
    # Normalize both forms into unique uppercase airport identifiers.
    codes: list[str] = []
    seen: set[str] = set()
    for item in raw_codes:
        for part in item.split(","):
            code = part.strip().upper()
            if code and code not in seen:
                codes.append(code)
                seen.add(code)
    return codes


def normalize_tiles(raw_tiles: Iterable[str], tile_file: Path | None) -> list[str]:
    # Same idea as airport code parsing, but for BCGS tile IDs. We keep tile IDs
    # lowercase because the LidarBC service stores examples like "092g014".
    pieces: list[str] = []
    for item in raw_tiles:
        pieces.extend(part.strip() for part in item.split(","))

    if tile_file:
        text = tile_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            pieces.extend(part.strip() for part in line.split(","))

    tiles: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        if not piece:
            continue
        tile = piece.lower()
        if not TILE_ID_RE.match(tile):
            raise ValueError(f"Invalid BCGS tile ID: {piece!r}")
        if tile not in seen:
            tiles.append(tile)
            seen.add(tile)
    return tiles


def load_airports(csv_path: Path, codes: list[str]) -> list[Airport]:
    # Load only the airport rows requested by the user. The airport file follows
    # the OurAirports-style schema, where the same airport may be referred to by
    # ident, ICAO, GPS, local, or IATA code. For CYVR, ident/ICAO/GPS are CYVR
    # and IATA is YVR, so either CYVR or YVR can resolve to the same row.
    if not codes:
        return []
    if not csv_path.exists():
        raise ValueError(f"Airport CSV not found: {csv_path}")

    wanted = set(codes)
    matched: dict[str, Airport] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aliases = {
                (row.get("ident") or "").upper(),
                (row.get("icao_code") or "").upper(),
                (row.get("gps_code") or "").upper(),
                (row.get("local_code") or "").upper(),
                (row.get("iata_code") or "").upper(),
            }
            hits = wanted.intersection(alias for alias in aliases if alias)
            if not hits:
                continue

            lat = row.get("latitude_deg")
            lon = row.get("longitude_deg")
            if not lat or not lon:
                continue

            airport = Airport(
                code=(row.get("ident") or next(iter(hits))).upper(),
                name=row.get("name") or "",
                latitude=float(lat),
                longitude=float(lon),
            )
            for hit in hits:
                matched[hit] = airport

    missing = [code for code in codes if code not in matched]
    if missing:
        raise ValueError(f"Airport code(s) not found in {csv_path}: {', '.join(missing)}")

    airports: list[Airport] = []
    seen_airports: set[str] = set()
    for code in codes:
        airport = matched[code]
        key = f"{airport.code}:{airport.latitude}:{airport.longitude}"
        if key not in seen_airports:
            airports.append(airport)
            seen_airports.add(key)
    return airports


def airport_bbox(airport: Airport, radius_km: float) -> list[float]:
    # ArcGIS spatial queries can intersect a point, but an airport is an area and
    # we usually want surrounding terrain too. The script therefore creates a
    # small WGS84 envelope around the airport coordinate. This is an internal
    # implementation detail; the user-facing input is still the airport code.
    if radius_km <= 0:
        raise ValueError("--airport-radius-km must be greater than zero")

    lat_delta = radius_km / 111.32
    cos_lat = math.cos(math.radians(airport.latitude))
    if abs(cos_lat) < 0.000001:
        lon_delta = 180.0
    else:
        lon_delta = radius_km / (111.32 * cos_lat)

    return [
        airport.longitude - lon_delta,
        airport.latitude - lat_delta,
        airport.longitude + lon_delta,
        airport.latitude + lat_delta,
    ]


def safe_path_component(value: str) -> str:
    # Keep generated directory names portable and shell-friendly.
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def airport_output_group(airport: Airport) -> str:
    return safe_path_component(f"{airport.code}_{airport.name}")


def selected_products(product: str, scale: str) -> list[str]:
    # Not every product exists at every index scale. This mirrors the layer list
    # in the public FeatureServer:
    #   DEM: 1:20,000 and 1:2,500
    #   DSM: 1:20,000, 1:10,000, and 1:2,500
    products = ["dem", "dsm"] if product == "both" else [product]
    invalid = [p for p in products if (p, scale) not in LAYER_IDS]
    if invalid:
        supported = ", ".join(
            f"{p}:{s}" for p, s in sorted(LAYER_IDS.keys()) if p in invalid
        )
        raise ValueError(
            f"Product/scale combination is not available: {', '.join(invalid)} "
            f"at {scale}. Supported: {supported}"
        )
    return products


def arcgis_post_json(url: str, params: dict[str, Any], timeout: int) -> dict[str, Any]:
    # ArcGIS REST endpoints accept normal form-encoded POST bodies. POST is used
    # instead of GET so long where clauses and geometry parameters do not run
    # into URL-length limits.
    payload = urlencode({k: v for k, v in params.items() if v is not None}).encode("utf-8")
    req = Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "bc-lidar-downloader/1.0")

    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ArcGIS HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"ArcGIS network error: {exc}") from exc

    data = json.loads(body)
    if "error" in data:
        raise RuntimeError(f"ArcGIS error: {data['error']}")
    return data


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_where(tile_chunk: list[str] | None, year: int | None, extra_where: str | None) -> str:
    # ArcGIS FeatureServer queries use SQL-like `where` strings. We build the
    # common filters here and let users append a custom clause with --where.
    clauses: list[str] = []
    if tile_chunk:
        quoted = ", ".join(f"'{tile}'" for tile in tile_chunk)
        clauses.append(f"maptile IN ({quoted})")
    if year is not None:
        clauses.append(f"year = {year}")
    if extra_where:
        clauses.append(f"({extra_where})")
    return " AND ".join(clauses) if clauses else "1=1"


def query_layer(
    *,
    group: str | None,
    product: str,
    scale: str,
    tiles: list[str],
    bbox: list[float] | None,
    year: int | None,
    extra_where: str | None,
    timeout: int,
    limit: int | None,
) -> list[IndexedTile]:
    # Query one FeatureServer layer, handling ArcGIS pagination.
    #
    # A single layer can contain more than the service's max record count. ArcGIS
    # returns `exceededTransferLimit` when more rows are available, so we keep
    # requesting pages with resultOffset until all matching features are read.
    layer_id = LAYER_IDS[(product, scale)]
    query_url = f"{SERVICE_ROOT}/{layer_id}/query"
    record_count = 1000
    results: list[IndexedTile] = []
    tile_chunks = list(chunked(tiles, 200)) if tiles else [None]

    for tile_chunk in tile_chunks:
        offset = 0
        where = build_where(tile_chunk, year, extra_where)
        while True:
            params: dict[str, Any] = {
                "f": "json",
                "where": where,
                "outFields": ",".join(OUT_FIELDS),
                "returnGeometry": "false",
                "orderByFields": "maptile ASC, filename ASC",
                "resultOffset": offset,
                "resultRecordCount": min(record_count, limit - len(results))
                if limit is not None
                else record_count,
            }
            if bbox:
                params.update(
                    {
                        "geometry": ",".join(str(v) for v in bbox),
                        "geometryType": "esriGeometryEnvelope",
                        "inSR": "4326",
                        "spatialRel": "esriSpatialRelIntersects",
                    }
                )

            data = arcgis_post_json(query_url, params, timeout)
            features = data.get("features") or []
            for feature in features:
                attrs = feature.get("attributes") or {}
                if attrs.get("s3Url"):
                    results.append(
                        IndexedTile(
                            group=group,
                            product=product,
                            scale=scale,
                            attributes=attrs,
                        )
                    )

            if limit is not None and len(results) >= limit:
                return results[:limit]
            if not data.get("exceededTransferLimit") and len(features) < record_count:
                break
            if not features:
                break
            offset += len(features)

    return results


def output_path(base_dir: Path, tile: IndexedTile) -> Path:
    # Airport queries write:
    #   <out>/<airport_name>/<dem|dsm>/<filename>
    #
    # Manual tile/bbox queries keep the old simpler shape:
    #   <out>/<dem|dsm>/<filename>
    filename = Path(tile.filename).name
    if not filename:
        filename = f"{tile.product}_{tile.maptile}.tif"
    if tile.group:
        return base_dir / tile.group / tile.product / filename
    return base_dir / tile.product / filename


def keep_latest_per_tile(tiles: list[IndexedTile]) -> list[IndexedTile]:
    # Some tile IDs have older and newer records. When --latest-per-tile is set,
    # keep only the highest year for each airport/product/maptile combination.
    latest: dict[tuple[str, str, str], IndexedTile] = {}
    for tile in tiles:
        key = (tile.group or "", tile.product, tile.maptile or tile.filename)
        current = latest.get(key)
        if current is None:
            latest[key] = tile
            continue

        tile_year = int(tile.attributes.get("year") or -1)
        current_year = int(current.attributes.get("year") or -1)
        if (tile_year, tile.filename) > (current_year, current.filename):
            latest[key] = tile

    return sorted(latest.values(), key=lambda t: (t.product, t.maptile, t.filename))


def dedupe_tiles(tiles: list[IndexedTile]) -> list[IndexedTile]:
    # A tile can be reached more than once if query areas overlap. For airport
    # downloads, the airport group is part of the key, so the same GeoTIFF can be
    # written into each relevant airport folder instead of being dropped globally.
    deduped: dict[tuple[str, str, str], IndexedTile] = {}
    for tile in tiles:
        key = (tile.group or "", tile.product, tile.url or f"{tile.maptile}:{tile.filename}")
        deduped[key] = tile
    return sorted(deduped.values(), key=lambda t: (t.product, t.maptile, t.filename))


def download_one(
    tile: IndexedTile,
    target: Path,
    *,
    overwrite: bool,
    timeout: int,
    retries: int,
) -> DownloadResult:
    # Download through a .part file first. If a network error interrupts a large
    # GeoTIFF, the final target path is not left looking like a valid complete
    # file. The .part file is renamed only after a successful download.
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        return DownloadResult(tile=tile, target=target, status="skipped", message="exists")

    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_name(target.name + ".part")
    last_error = ""

    for attempt in range(1, retries + 1):
        try:
            req = Request(tile.url, method="GET")
            req.add_header("User-Agent", "bc-lidar-downloader/1.0")
            with urlopen(req, timeout=timeout) as resp, part_path.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            part_path.replace(target)
            return DownloadResult(tile=tile, target=target, status="downloaded")
        except (HTTPError, URLError, OSError) as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(min(2**attempt, 10))

    return DownloadResult(tile=tile, target=target, status="failed", message=last_error)


def write_manifest(path: Path, rows: list[DownloadResult]) -> None:
    # The manifest is intentionally useful even for --dry-run. It is the best
    # place to inspect coverage, years, exact URLs, and final output paths before
    # spending time and disk space on large downloads.
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "message",
        "group",
        "product",
        "scale",
        "maptile",
        "filename",
        "year",
        "projection",
        "spacing",
        "url",
        "target",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            attrs = row.tile.attributes
            writer.writerow(
                {
                    "status": row.status,
                    "message": row.message,
                    "group": row.tile.group or "",
                    "product": row.tile.product,
                    "scale": row.tile.scale,
                    "maptile": row.tile.maptile,
                    "filename": row.tile.filename,
                    "year": attrs.get("year", ""),
                    "projection": attrs.get("projection", ""),
                    "spacing": attrs.get("spacing", ""),
                    "url": row.tile.url,
                    "target": str(row.target),
                }
            )


def main() -> int:
    args = parse_args()

    try:
        products = selected_products(args.product, args.scale)
        tiles = normalize_tiles(args.tiles, args.tile_file)
        airport_codes = normalize_codes(args.airports)
        airports = load_airports(args.airport_csv, airport_codes)
        airport_queries = [(airport, airport_bbox(airport, args.airport_radius_km)) for airport in airports]
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not tiles and not airport_queries and not args.bbox and not args.where:
        print(
            "error: provide --airports, --tiles, --tile-file, --bbox, or --where so the query is bounded",
            file=sys.stderr,
        )
        return 2

    out_dir = args.out
    manifest = args.manifest or out_dir / "download_manifest.csv"

    all_tiles: list[IndexedTile] = []
    for product in products:
        print(f"Querying {product.upper()} layer at BCGS 1:{args.scale} ...")
        product_matches: list[IndexedTile] = []

        query_bboxes: list[tuple[str | None, list[float] | None]]
        if airport_queries:
            query_bboxes = []
            for airport, bbox in airport_queries:
                group = airport_output_group(airport)
                query_bboxes.append((group, bbox))
                print(
                    f"  airport {airport.code}: {airport.name} "
                    f"({airport.latitude:.6f}, {airport.longitude:.6f}), "
                    f"radius={args.airport_radius_km:g} km, output={group}"
                )
        else:
            query_bboxes = [(None, args.bbox)]

        for group, bbox in query_bboxes:
            try:
                matches = query_layer(
                    group=group,
                    product=product,
                    scale=args.scale,
                    tiles=tiles,
                    bbox=bbox,
                    year=args.year,
                    extra_where=args.where,
                    timeout=args.timeout,
                    limit=args.limit,
                )
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            product_matches.extend(matches)

        product_matches = dedupe_tiles(product_matches)
        print(f"  found {len(product_matches)} file(s)")
        all_tiles.extend(product_matches)

    if args.latest_per_tile:
        original_count = len(all_tiles)
        all_tiles = keep_latest_per_tile(all_tiles)
        if len(all_tiles) != original_count:
            print(f"Keeping latest record per product/tile: {len(all_tiles)} of {original_count}")

    planned = [
        DownloadResult(tile=tile, target=output_path(out_dir, tile), status="planned")
        for tile in all_tiles
    ]

    if args.dry_run:
        write_manifest(manifest, planned)
        print(f"Dry run complete. Manifest: {manifest}")
        return 0

    if not planned:
        write_manifest(manifest, planned)
        print(f"No files matched. Manifest: {manifest}")
        return 0

    workers = max(1, args.workers)
    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                download_one,
                row.tile,
                row.target,
                overwrite=args.overwrite,
                timeout=args.timeout,
                retries=args.retries,
            )
            for row in planned
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            label = f"{result.tile.product.upper()} {result.tile.maptile} {result.tile.filename}"
            if result.status == "failed":
                print(f"  failed: {label} ({result.message})")
            else:
                print(f"  {result.status}: {label}")

    results.sort(key=lambda r: (r.tile.product, r.tile.maptile, r.tile.filename))
    write_manifest(manifest, results)

    failed = sum(1 for row in results if row.status == "failed")
    downloaded = sum(1 for row in results if row.status == "downloaded")
    skipped = sum(1 for row in results if row.status == "skipped")
    print(
        f"Done. downloaded={downloaded}, skipped={skipped}, failed={failed}. "
        f"Manifest: {manifest}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
