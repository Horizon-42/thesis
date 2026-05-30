#!/usr/bin/env python3
"""
Download USGS The National Map elevation products around airports or points.

The script uses the TNMAccess products endpoint, which is the REST API behind
The National Map Downloader. It is intentionally manifest-first: run with
--dry-run to inspect every matched product URL before downloading large DEM,
DSM, or lidar source files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"

DEFAULT_AIRPORT_CSV = Path("aeroviz-4d/public/data/common/airports.csv")
DEFAULT_OUT_DIR = Path("data/usgs_tnm_elevation")
DEFAULT_RADIUS_KM = 5.0
DEFAULT_PAGE_SIZE = 100
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_RETRIES = 3
DEFAULT_WORKERS = 4
DEFAULT_PROGRESS_INTERVAL_SECONDS = 0.5

DEGREES_LAT_KM = 111.32

SAFE_COMPONENT_RE = re.compile(r"[^0-9A-Za-z._-]+")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    product: str
    dataset: str
    prod_formats: str
    url_keys: tuple[str, ...]
    output_subdir: str
    spacing: str
    notes: str


@dataclass(frozen=True)
class Airport:
    code: str
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class QueryContext:
    group: str
    label: str
    bbox: list[float]
    latitude: float
    longitude: float


@dataclass(frozen=True)
class TnmProduct:
    context: QueryContext
    spec: DatasetSpec
    item: dict[str, Any]

    @property
    def product(self) -> str:
        return self.spec.product

    @property
    def title(self) -> str:
        return str(self.item.get("title") or "")

    @property
    def source_id(self) -> str:
        return str(self.item.get("sourceId") or "")

    @property
    def url(self) -> str:
        urls = self.item.get("urls") or {}
        if isinstance(urls, dict):
            for key in self.spec.url_keys:
                value = str(urls.get(key) or "").strip()
                if value:
                    return value

        for key in ("downloadURL", "downloadURLRaster", "downloadLazURL"):
            value = str(self.item.get(key) or "").strip()
            if value:
                return value
        return ""

    @property
    def filename(self) -> str:
        parsed = urlparse(self.url)
        filename = Path(parsed.path).name
        if filename:
            return filename

        title = safe_path_component(self.title or self.source_id or self.spec.key)
        suffix = ".laz" if self.spec.key == "dsm_lpc" else ".tif"
        return f"{title}{suffix}"

    @property
    def format(self) -> str:
        return str(self.item.get("format") or "")

    @property
    def size_in_bytes(self) -> int | None:
        value = self.item.get("sizeInBytes")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @property
    def product_bbox(self) -> str:
        bbox = self.item.get("boundingBox")
        if not isinstance(bbox, dict):
            return ""
        keys = ("minX", "minY", "maxX", "maxY")
        try:
            return ",".join(f"{float(bbox[key]):.8f}" for key in keys)
        except (KeyError, TypeError, ValueError):
            return ""


@dataclass
class DownloadResult:
    product: TnmProduct
    target: Path
    status: str
    message: str = ""
    size_in_bytes: int | None = None


ProgressCallback = Callable[[Path, int], None]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "dem_1m": DatasetSpec(
        key="dem_1m",
        product="dem",
        dataset="Digital Elevation Model (DEM) 1 meter",
        prod_formats="GeoTIFF,IMG",
        url_keys=("GeoTIFF", "TIFF", "IMG"),
        output_subdir="dem",
        spacing="1 meter",
        notes="3DEP bare-earth DEM tiles where 1 m coverage is available.",
    ),
    "dem_s1m": DatasetSpec(
        key="dem_s1m",
        product="dem",
        dataset="Seamless 1-m DEM (S1M)",
        prod_formats="GeoTIFF",
        url_keys=("GeoTIFF", "TIFF"),
        output_subdir="dem",
        spacing="1 meter",
        notes="Seamless 1 m DEM, limited availability.",
    ),
    "dem_opr": DatasetSpec(
        key="dem_opr",
        product="dem",
        dataset="Original Product Resolution (OPR) Digital Elevation Model (DEM)",
        prod_formats="GeoTIFF,IMG",
        url_keys=("GeoTIFF", "TIFF", "IMG"),
        output_subdir="dem",
        spacing="original product resolution",
        notes="Project-level original-resolution DEM source products.",
    ),
    "dem_13": DatasetSpec(
        key="dem_13",
        product="dem",
        dataset="National Elevation Dataset (NED) 1/3 arc-second",
        prod_formats="GeoTIFF",
        url_keys=("GeoTIFF", "TIFF"),
        output_subdir="dem",
        spacing="1/3 arc-second, about 10 meters",
        notes="Fallback DEM coverage when 1 m products are unavailable.",
    ),
    "dem_1": DatasetSpec(
        key="dem_1",
        product="dem",
        dataset="National Elevation Dataset (NED) 1 arc-second",
        prod_formats="GeoTIFF",
        url_keys=("GeoTIFF", "TIFF"),
        output_subdir="dem",
        spacing="1 arc-second, about 30 meters",
        notes="Broad fallback DEM coverage.",
    ),
    "dsm_lpc": DatasetSpec(
        key="dsm_lpc",
        product="dsm",
        dataset="Lidar Point Cloud (LPC)",
        prod_formats="LAS,LAZ",
        url_keys=("LAZ", "LAS"),
        output_subdir="dsm/source_laz",
        spacing="source point cloud",
        notes=(
            "High-resolution lidar source used to derive DSM rasters in most "
            "CONUS areas; TNM usually does not publish ready-made CONUS DSM GeoTIFFs."
        ),
    ),
    "dsm_ifsar": DatasetSpec(
        key="dsm_ifsar",
        product="dsm",
        dataset="Ifsar Digital Surface Model (DSM)",
        prod_formats="TIFF",
        url_keys=("TIFF", "GeoTIFF"),
        output_subdir="dsm",
        spacing="5 meters",
        notes="Ready-made DSM raster source where Alaska IfSAR products exist.",
    ),
}

DEM_FALLBACK_ORDER = ["dem_1m", "dem_s1m", "dem_opr", "dem_13", "dem_1"]
DEM_DATASET_KEYS = tuple(key for key, spec in DATASET_SPECS.items() if spec.product == "dem")
DSM_DATASET_KEYS = tuple(key for key, spec in DATASET_SPECS.items() if spec.product == "dsm")


def normalize_codes(raw_codes: Iterable[str]) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for item in raw_codes:
        for part in item.split(","):
            code = part.strip().upper()
            if code and code not in seen:
                codes.append(code)
                seen.add(code)
    return codes


def safe_path_component(value: str) -> str:
    cleaned = SAFE_COMPONENT_RE.sub("_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def bbox_from_point(latitude: float, longitude: float, radius_km: float) -> list[float]:
    if radius_km < DEFAULT_RADIUS_KM:
        raise ValueError(f"radius must be at least {DEFAULT_RADIUS_KM:g} km")

    lat_delta = radius_km / DEGREES_LAT_KM
    cos_lat = math.cos(math.radians(latitude))
    lon_delta = 180.0 if abs(cos_lat) < 0.000001 else radius_km / (DEGREES_LAT_KM * cos_lat)

    return [
        max(-180.0, longitude - lon_delta),
        max(-90.0, latitude - lat_delta),
        min(180.0, longitude + lon_delta),
        min(90.0, latitude + lat_delta),
    ]


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)


def bbox_text(bbox: list[float]) -> str:
    return ",".join(f"{value:.8f}" for value in bbox)


def load_airports(csv_path: Path, codes: list[str]) -> list[Airport]:
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
                code=(row.get("ident") or sorted(hits)[0]).upper(),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download TNM DEM rasters or DSM source/raster products around an "
            "airport or latitude/longitude point."
        )
    )
    parser.add_argument(
        "airport_codes",
        nargs="*",
        metavar="AIRPORT",
        help="Airport identifiers, for example KRDU KDEN. Commas are accepted.",
    )
    parser.add_argument(
        "--airports",
        nargs="*",
        default=[],
        help="Airport identifiers from airports.csv. Alias of the positional arguments.",
    )
    parser.add_argument(
        "--icaos",
        nargs="*",
        default=[],
        help="ICAO identifiers. Alias of --airports.",
    )
    parser.add_argument("--lat", type=float, help="Latitude for a point query.")
    parser.add_argument("--lon", type=float, help="Longitude for a point query.")
    parser.add_argument(
        "--label",
        help="Output group label for --lat/--lon. Default: point_<lat>_<lon>.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Direct WGS84 longitude/latitude bounding box query.",
    )
    parser.add_argument(
        "--radius-km",
        type=float,
        default=DEFAULT_RADIUS_KM,
        help="Search radius around airport or point in kilometres. Minimum/default: 5.",
    )
    parser.add_argument(
        "--product",
        choices=["dem", "dsm", "both"],
        default="dem",
        help="Product family to download. Default: dem.",
    )
    parser.add_argument(
        "--dem-dataset",
        choices=DEM_DATASET_KEYS,
        default="dem_1m",
        help="DEM dataset key. Default: dem_1m.",
    )
    parser.add_argument(
        "--no-dem-fallback",
        action="store_true",
        help="Disable DEM fallback from 1 m products to broader DEM coverage.",
    )
    parser.add_argument(
        "--include-historical",
        action="store_true",
        help="Keep all historical product versions. Default keeps the latest per footprint.",
    )
    parser.add_argument(
        "--dsm-source",
        choices=DSM_DATASET_KEYS,
        default="dsm_lpc",
        help="DSM dataset/source key. Default: dsm_lpc.",
    )
    parser.add_argument(
        "--airport-csv",
        type=Path,
        default=DEFAULT_AIRPORT_CSV,
        help="Airport CSV path. Default: aeroviz-4d/public/data/common/airports.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory. Default: data/usgs_tnm_elevation.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest CSV path. Default: <out>/download_manifest.csv.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write manifest only.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--limit", type=int, help="Maximum products per selected dataset/context.")
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="TNM page size. Default: 100.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent downloads. Default: 4.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the aggregate download progress bar.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds. Default: 120.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Retry attempts per request. Default: 3.",
    )
    args = parser.parse_args(argv)
    args.airports = [*args.airport_codes, *args.airports, *args.icaos]
    del args.airport_codes
    del args.icaos
    return args


def validate_args(args: argparse.Namespace) -> None:
    has_airports = bool(normalize_codes(args.airports))
    has_point = args.lat is not None or args.lon is not None
    if has_point and (args.lat is None or args.lon is None):
        raise ValueError("provide both --lat and --lon for a point query")
    if has_point and not (-90 <= args.lat <= 90 and -180 <= args.lon <= 180):
        raise ValueError("--lat/--lon must be valid WGS84 decimal degrees")
    if args.radius_km < DEFAULT_RADIUS_KM:
        raise ValueError(f"--radius-km must be at least {DEFAULT_RADIUS_KM:g}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if args.page_size <= 0:
        raise ValueError("--page-size must be greater than zero")
    if args.workers <= 0:
        raise ValueError("--workers must be greater than zero")
    if not (has_airports or has_point or args.bbox):
        raise ValueError("provide at least one airport, --lat/--lon, or --bbox")


def build_query_contexts(args: argparse.Namespace) -> list[QueryContext]:
    contexts: list[QueryContext] = []
    airport_codes = normalize_codes(args.airports)
    for airport in load_airports(args.airport_csv, airport_codes):
        contexts.append(
            QueryContext(
                group=safe_path_component(airport.code),
                label=f"{airport.code}: {airport.name}",
                bbox=bbox_from_point(airport.latitude, airport.longitude, args.radius_km),
                latitude=airport.latitude,
                longitude=airport.longitude,
            )
        )

    if args.lat is not None and args.lon is not None:
        label = args.label or f"point_{args.lat:.5f}_{args.lon:.5f}"
        contexts.append(
            QueryContext(
                group=safe_path_component(label),
                label=label,
                bbox=bbox_from_point(args.lat, args.lon, args.radius_km),
                latitude=args.lat,
                longitude=args.lon,
            )
        )

    if args.bbox:
        lat, lon = bbox_center(list(args.bbox))
        contexts.append(
            QueryContext(
                group=safe_path_component(args.label or "bbox"),
                label=args.label or "bbox",
                bbox=list(args.bbox),
                latitude=lat,
                longitude=lon,
            )
        )

    return contexts


def dem_candidate_keys(selected_key: str, fallback: bool) -> list[str]:
    if not fallback:
        return [selected_key]
    try:
        start = DEM_FALLBACK_ORDER.index(selected_key)
    except ValueError:
        return [selected_key]
    return DEM_FALLBACK_ORDER[start:]


def target_path(out_dir: Path, product: TnmProduct) -> Path:
    return out_dir / product.context.group / product.spec.output_subdir / product.filename


def format_bytes(byte_count: int | float) -> str:
    value = float(max(0, byte_count))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def progress_bar(fraction: float, width: int = 24) -> str:
    bounded = max(0.0, min(1.0, fraction))
    filled = int(round(width * bounded))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class DownloadProgress:
    def __init__(
        self,
        products: list[TnmProduct],
        out_dir: Path,
        *,
        enabled: bool,
        stream: Any = sys.stderr,
    ) -> None:
        self.enabled = enabled
        self.stream = stream
        self.is_tty = bool(getattr(stream, "isatty", lambda: False)())
        self.interval_seconds = DEFAULT_PROGRESS_INTERVAL_SECONDS if self.is_tty else 10.0
        self.lock = threading.Lock()
        self.started_at = time.monotonic()
        self.last_rendered_at = 0.0
        self.last_line_length = 0
        self.total_files = len(products)
        self.completed_files = 0
        self.expected_by_target = {
            str(target_path(out_dir, product)): product.size_in_bytes or 0 for product in products
        }
        self.current_by_target = {target: 0 for target in self.expected_by_target}

    @property
    def total_known_bytes(self) -> int:
        return sum(size for size in self.expected_by_target.values() if size > 0)

    def update(self, target: Path, bytes_written: int) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.current_by_target[str(target)] = max(0, bytes_written)

    def complete(self, result: DownloadResult) -> None:
        if not self.enabled:
            return
        target = str(result.target)
        with self.lock:
            self.completed_files += 1
            if result.status in {"downloaded", "skipped"}:
                final_size = result.size_in_bytes or self.expected_by_target.get(target, 0)
                self.expected_by_target[target] = final_size
                self.current_by_target[target] = final_size

    def _line(self) -> str:
        with self.lock:
            completed_files = self.completed_files
            known_total = self.total_known_bytes
            known_done = sum(
                min(self.current_by_target.get(target, 0), expected)
                for target, expected in self.expected_by_target.items()
                if expected > 0
            )
            unknown_done = sum(
                self.current_by_target.get(target, 0)
                for target, expected in self.expected_by_target.items()
                if expected <= 0
            )

        elapsed = max(0.001, time.monotonic() - self.started_at)
        total_done = known_done + unknown_done
        speed = total_done / elapsed
        if known_total > 0:
            fraction = known_done / known_total
            byte_text = f"{format_bytes(known_done)}/{format_bytes(known_total)}"
            if unknown_done:
                byte_text += f" + {format_bytes(unknown_done)}"
            return (
                f"  {progress_bar(fraction)} {fraction * 100:5.1f}% "
                f"{completed_files}/{self.total_files} files "
                f"{byte_text} {format_bytes(speed)}/s"
            )
        return (
            f"  {completed_files}/{self.total_files} files "
            f"{format_bytes(total_done)} {format_bytes(speed)}/s"
        )

    def render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_rendered_at < self.interval_seconds:
            return

        line = self._line()
        if self.is_tty:
            padded = line.ljust(self.last_line_length)
            self.stream.write("\r" + padded)
            self.last_line_length = len(padded)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
        self.last_rendered_at = now

    def close(self) -> None:
        if self.enabled and self.is_tty:
            self.stream.write("\n")
            self.stream.flush()


def tnm_payload_error(data: dict[str, Any]) -> str:
    if data.get("error"):
        return str(data["error"])
    if data.get("errors"):
        errors = data.get("errors")
        if isinstance(errors, list) and not errors:
            return ""
        return str(errors)
    if data.get("message") and "items" not in data and "total" not in data:
        return str(data["message"])
    return ""


def http_json(
    url: str,
    params: dict[str, Any],
    *,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    query = urlencode({key: value for key, value in params.items() if value is not None})
    req = Request(f"{url}?{query}", method="GET")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "aeroviz-tnm-elevation-downloader/1.0")

    last_error = ""
    for attempt in range(1, max(1, retries) + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            data = json.loads(body)
            payload_error = tnm_payload_error(data)
            if payload_error:
                last_error = f"TNM error: {payload_error}"
                raise RuntimeError(last_error)
            return data
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"TNM HTTP {exc.code}: {detail}"
            if exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(last_error) from exc
        except (URLError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = str(exc)
        if attempt < max(1, retries):
            time.sleep(min(2**attempt, 10))

    raise RuntimeError(last_error or "TNM request failed")


def query_products_for_spec(
    context: QueryContext,
    spec: DatasetSpec,
    *,
    timeout: int,
    retries: int,
    page_size: int,
    limit: int | None,
) -> list[TnmProduct]:
    results: list[TnmProduct] = []
    page_size = max(1, min(page_size, 1000))
    if page_size % 5 != 0:
        page_size += 5 - (page_size % 5)

    offset = 0
    while True:
        request_size = page_size
        if limit is not None:
            request_size = min(request_size, limit - len(results))
            if request_size <= 0:
                break

        data = http_json(
            TNM_PRODUCTS_URL,
            {
                "datasets": spec.dataset,
                "bbox": bbox_text(context.bbox),
                "prodFormats": spec.prod_formats,
                "outputFormat": "JSON",
                "max": request_size,
                "offset": offset,
            },
            timeout=timeout,
            retries=retries,
        )
        items = data.get("items") or []
        if not isinstance(items, list):
            raise RuntimeError(f"TNM returned unexpected items payload for {spec.dataset}")

        for item in items:
            if isinstance(item, dict):
                results.append(TnmProduct(context=context, spec=spec, item=item))

        total = int(data.get("total") or len(results))
        if len(items) < request_size or len(results) >= total:
            break
        offset += len(items)

    return results


def product_date_key(product: TnmProduct) -> str:
    for key in ("publicationDate", "lastUpdated", "dateCreated", "modificationInfo"):
        value = str(product.item.get(key) or "")
        if value:
            return value
    return ""


def latest_per_footprint(products: list[TnmProduct]) -> list[TnmProduct]:
    selected: dict[tuple[str, str, str], TnmProduct] = {}
    for product in products:
        footprint = product.product_bbox or product.filename or product.title
        key = (product.context.group, product.spec.key, footprint)
        current = selected.get(key)
        if current is None or product_date_key(product) > product_date_key(current):
            selected[key] = product
    return sorted(selected.values(), key=lambda product: product.title)


def query_products(
    contexts: list[QueryContext],
    args: argparse.Namespace,
) -> tuple[list[TnmProduct], list[str]]:
    products: list[TnmProduct] = []
    notes: list[str] = []

    for context in contexts:
        if args.product in {"dem", "both"}:
            dem_found = False
            for key in dem_candidate_keys(args.dem_dataset, not args.no_dem_fallback):
                spec = DATASET_SPECS[key]
                found = query_products_for_spec(
                    context,
                    spec,
                    timeout=args.timeout,
                    retries=args.retries,
                    page_size=args.page_size,
                    limit=args.limit,
                )
                if found and not args.include_historical:
                    found = latest_per_footprint(found)
                if found:
                    products.extend(found)
                    dem_found = True
                    if key != args.dem_dataset:
                        notes.append(
                            f"{context.label}: no {args.dem_dataset} DEM products; using {key}."
                        )
                    break
            if not dem_found:
                notes.append(f"{context.label}: no DEM products found.")

        if args.product in {"dsm", "both"}:
            spec = DATASET_SPECS[args.dsm_source]
            found = query_products_for_spec(
                context,
                spec,
                timeout=args.timeout,
                retries=args.retries,
                page_size=args.page_size,
                limit=args.limit,
            )
            if found and not args.include_historical:
                found = latest_per_footprint(found)
            if found:
                products.extend(found)
            else:
                notes.append(f"{context.label}: no {args.dsm_source} DSM products found.")

    return products, notes


def write_manifest(path: Path, rows: list[DownloadResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "message",
        "group",
        "label",
        "product",
        "dataset_key",
        "dataset",
        "spacing",
        "format",
        "title",
        "source_id",
        "publication_date",
        "last_updated",
        "query_bbox",
        "product_bbox",
        "url",
        "target",
        "size_in_bytes",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            product = row.product
            writer.writerow(
                {
                    "status": row.status,
                    "message": row.message,
                    "group": product.context.group,
                    "label": product.context.label,
                    "product": product.product,
                    "dataset_key": product.spec.key,
                    "dataset": product.spec.dataset,
                    "spacing": product.spec.spacing,
                    "format": product.format,
                    "title": product.title,
                    "source_id": product.source_id,
                    "publication_date": product.item.get("publicationDate") or "",
                    "last_updated": product.item.get("lastUpdated") or "",
                    "query_bbox": bbox_text(product.context.bbox),
                    "product_bbox": product.product_bbox,
                    "url": product.url,
                    "target": str(row.target),
                    "size_in_bytes": row.size_in_bytes or product.size_in_bytes or "",
                    "notes": product.spec.notes,
                }
            )


def planned_results(products: list[TnmProduct], out_dir: Path) -> list[DownloadResult]:
    return [
        DownloadResult(product=product, target=target_path(out_dir, product), status="planned")
        for product in products
    ]


def download_one(
    product: TnmProduct,
    *,
    out_dir: Path,
    overwrite: bool,
    timeout: int,
    retries: int,
    progress_callback: ProgressCallback | None = None,
) -> DownloadResult:
    target = target_path(out_dir, product)
    url = product.url
    if not url:
        return DownloadResult(product=product, target=target, status="failed", message="missing url")

    if target.exists() and target.stat().st_size > 0 and not overwrite:
        return DownloadResult(
            product=product,
            target=target,
            status="skipped",
            message="exists",
            size_in_bytes=target.stat().st_size,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_name(target.name + ".part")
    last_error = ""

    for attempt in range(1, max(1, retries) + 1):
        try:
            req = Request(url, method="GET")
            req.add_header("User-Agent", "aeroviz-tnm-elevation-downloader/1.0")
            with urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", None) or 200
                if status >= 400:
                    return DownloadResult(
                        product=product,
                        target=target,
                        status="failed",
                        message=f"HTTP {status}",
                    )
                bytes_written = 0
                with part_path.open("wb") as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        bytes_written += len(chunk)
                        if progress_callback:
                            progress_callback(target, bytes_written)
            part_path.replace(target)
            return DownloadResult(
                product=product,
                target=target,
                status="downloaded",
                size_in_bytes=target.stat().st_size,
            )
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {exc.code}: {detail}"
        except (URLError, OSError) as exc:
            last_error = str(exc)

        try:
            part_path.unlink()
        except OSError:
            pass
        if progress_callback:
            progress_callback(target, 0)
        if attempt < max(1, retries):
            time.sleep(min(2**attempt, 10))

    return DownloadResult(product=product, target=target, status="failed", message=last_error)


def download_all(products: list[TnmProduct], args: argparse.Namespace) -> list[DownloadResult]:
    if not products:
        return []

    results: list[DownloadResult] = []
    workers = min(max(1, args.workers), len(products))
    progress = DownloadProgress(products, args.out, enabled=not args.no_progress)
    print(f"Downloading {len(products)} product(s) with {workers} worker(s).", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_one,
                product,
                out_dir=args.out,
                overwrite=args.overwrite,
                timeout=args.timeout,
                retries=args.retries,
                progress_callback=progress.update,
            ): product
            for product in products
        }
        pending = set(futures)
        progress.render(force=True)
        while pending:
            done, pending = wait(pending, timeout=DEFAULT_PROGRESS_INTERVAL_SECONDS, return_when=FIRST_COMPLETED)
            if not done:
                progress.render()
                continue

            for future in done:
                result = future.result()
                results.append(result)
                progress.complete(result)
            progress.render(force=True)

    progress.close()
    return results


def summarize_results(rows: list[DownloadResult]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    if not counts:
        return "no products"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        validate_args(args)
        contexts = build_query_contexts(args)
        products, notes = query_products(contexts, args)
        manifest = args.manifest or args.out / "download_manifest.csv"

        for note in notes:
            print(f"note: {note}", file=sys.stderr)

        if args.dry_run:
            results = planned_results(products, args.out)
            write_manifest(manifest, results)
            print(f"Dry run: {summarize_results(results)}. Manifest: {manifest}")
            return 0

        results = download_all(products, args)
        write_manifest(manifest, results)
        failed = sum(1 for row in results if row.status == "failed")
        print(f"Done: {summarize_results(results)}. Manifest: {manifest}")
        return 1 if failed else 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
