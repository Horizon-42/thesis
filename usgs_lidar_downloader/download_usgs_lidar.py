#!/usr/bin/env python3
"""
Download USGS 3DEP LPC DSM source data around US airports.

The output layout intentionally mirrors the BC LiDAR downloader:

    <out>/<AIRPORT>/dsm/*.tif
    <out>/<AIRPORT>/dsm/source_laz/*.laz
    <out>/<AIRPORT>/download_manifest.csv

The full AeroViz preprocessing pipeline downloads DEM rasters through
opentopography_downloader. For DSM in the conterminous United States, USGS
normally publishes LiDAR point cloud tiles rather than ready-made DSM rasters.
This script therefore downloads LPC LAZ files as the DSM source and can
optionally derive DSM GeoTIFFs with PDAL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"

DEM_DATASETS: dict[str, dict[str, str]] = {
    "1m": {
        "dataset": "Digital Elevation Model (DEM) 1 meter",
        "formats": "GeoTIFF,IMG",
        "scale": "1m",
        "spacing": "1 meter",
    },
    "s1m": {
        "dataset": "Seamless 1-m DEM (S1M)",
        "formats": "GeoTIFF",
        "scale": "s1m",
        "spacing": "1 meter",
    },
}

DSM_SOURCES: dict[str, dict[str, str]] = {
    "lpc": {
        "dataset": "Lidar Point Cloud (LPC)",
        "formats": "LAZ",
        "scale": "lpc",
        "spacing": "",
    },
    "ifsar": {
        "dataset": "Ifsar Digital Surface Model (DSM)",
        "formats": "TIFF",
        "scale": "ifsar",
        "spacing": "5 meter",
    },
}

# Default workflow configuration. The command line is intentionally small.
# DEM is handled by opentopography_downloader in the full airport pipeline, so
# this USGS downloader stays focused on LPC DSM source and DSM GeoTIFF derivation.
DEFAULT_PRODUCT = "dsm"
DEFAULT_DEM_DATASET = "1m"
DEFAULT_DSM_SOURCE = "lpc"
DEFAULT_DERIVE_DSM = True
DEFAULT_LATEST_PER_TILE = True
DEFAULT_AIRPORT_CSV = Path("aeroviz-4d/public/data/common/airports.csv")
DEFAULT_AIRPORT_RADIUS_KM = 5.0
DEFAULT_OUT_DIR = Path("data/usgs_lidar")
DEFAULT_LIMIT: int | None = None
DEFAULT_PAGE_SIZE = 20
DEFAULT_QUERY_RETRIES = 3
DEFAULT_MAX_QUERY_SPLIT_DEPTH = 2
DEFAULT_WORKERS = 4
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_RETRIES = 3

DEFAULT_DSM_RESOLUTION_M = 1.0
DEFAULT_DSM_OUTPUT_TYPE = "max"
DEFAULT_DSM_NODATA = -999999.0
DEFAULT_DSM_EXPRESSION: str | None = None
DEFAULT_DSM_INPUT_CRS: str | None = None
DEFAULT_DSM_TARGET_CRS = "auto-utm"
DEFAULT_PDAL = "pdal"

DATE_RE = re.compile(r"(\d{4})")
SAFE_COMPONENT_RE = re.compile(r"[^0-9A-Za-z._-]+")
STANDARD_DEM_TILE_RE = re.compile(r"\b(?P<zone>\d{1,2})\s+(?P<tile>x\d+y\d+)\b", re.I)
S1M_TILE_RE = re.compile(r"\b(?P<tile>[ns]\d{4}[ew]\d{4})\b", re.I)
USGS_PREFIX_RE = re.compile(r"^(USGS_(?:1M|LPC)_|USGS_)", re.I)


@dataclass(frozen=True)
class Airport:
    code: str
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class QueryContext:
    group: str | None
    label: str
    bbox: list[float]
    latitude: float
    longitude: float


@dataclass(frozen=True)
class IndexedProduct:
    group: str | None
    product: str
    scale: str
    dataset: str
    source_kind: str
    item: dict[str, Any]
    query_bbox: list[float] | None
    query_latitude: float | None
    query_longitude: float | None

    @property
    def url(self) -> str:
        urls = self.item.get("urls") or {}
        for key in ("TIFF", "GeoTIFF", "LAZ", "LAS", "IMG"):
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

        title = safe_path_component(str(self.item.get("title") or "usgs_product"))
        suffix = ".laz" if self.source_kind == "lpc" else ".tif"
        return f"{title}{suffix}"

    @property
    def maptile(self) -> str:
        title = str(self.item.get("title") or "")
        filename = Path(self.filename).stem

        s1m_match = S1M_TILE_RE.search(title) or S1M_TILE_RE.search(filename)
        if s1m_match:
            return s1m_match.group("tile").lower()

        dem_match = STANDARD_DEM_TILE_RE.search(title) or STANDARD_DEM_TILE_RE.search(
            filename.replace("_", " ")
        )
        if dem_match:
            return f"{int(dem_match.group('zone'))}_{dem_match.group('tile').lower()}"

        cleaned = USGS_PREFIX_RE.sub("", filename)
        cleaned = re.sub(r"[^0-9A-Za-z]+", "_", cleaned).strip("_")
        return cleaned.lower() or str(self.item.get("sourceId") or filename).lower()

    @property
    def year(self) -> str:
        for key in ("publicationDate", "lastUpdated", "dateCreated", "modificationInfo"):
            value = str(self.item.get(key) or "")
            match = DATE_RE.search(value)
            if match:
                return match.group(1)
        return ""

    @property
    def projection(self) -> str:
        title = str(self.item.get("title") or "")
        filename = Path(self.filename).stem
        if self.scale == "s1m":
            return "epsg6350"

        dem_match = STANDARD_DEM_TILE_RE.search(title) or STANDARD_DEM_TILE_RE.search(
            filename.replace("_", " ")
        )
        if dem_match:
            return f"utm{int(dem_match.group('zone'))}"
        return ""

    @property
    def size_in_bytes(self) -> int | None:
        value = self.item.get("sizeInBytes")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


@dataclass
class DownloadResult:
    product: IndexedProduct
    target: Path
    status: str
    message: str = ""
    derived_target: Path | None = None
    derived_status: str = ""
    derived_message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download USGS 3DEP LPC source and derive DSM GeoTIFFs for airports."
        )
    )
    parser.add_argument(
        "airport_codes",
        nargs="*",
        metavar="AIRPORT",
        help="Airport code(s), for example KSFO KLAX KJFK. Commas are also accepted.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory. Default: data/usgs_lidar.",
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
        "--airports",
        nargs="*",
        default=[],
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    args.airports = [*args.airport_codes, *args.airports]
    if not args.airports:
        parser.error("provide at least one airport code, for example KSFO")
    del args.airport_codes

    args.product = DEFAULT_PRODUCT
    args.dem_dataset = DEFAULT_DEM_DATASET
    args.dsm_source = DEFAULT_DSM_SOURCE
    args.derive_dsm = DEFAULT_DERIVE_DSM
    args.latest_per_tile = DEFAULT_LATEST_PER_TILE
    args.airport_csv = DEFAULT_AIRPORT_CSV
    args.airport_radius_km = DEFAULT_AIRPORT_RADIUS_KM
    args.bbox = None
    args.limit = DEFAULT_LIMIT
    args.manifest = None
    args.page_size = DEFAULT_PAGE_SIZE
    args.query_retries = DEFAULT_QUERY_RETRIES
    args.workers = DEFAULT_WORKERS
    args.timeout = DEFAULT_TIMEOUT_SECONDS
    args.retries = DEFAULT_RETRIES
    args.dsm_resolution_m = DEFAULT_DSM_RESOLUTION_M
    args.dsm_output_type = DEFAULT_DSM_OUTPUT_TYPE
    args.dsm_nodata = DEFAULT_DSM_NODATA
    args.dsm_expression = DEFAULT_DSM_EXPRESSION
    args.dsm_input_crs = DEFAULT_DSM_INPUT_CRS
    args.dsm_target_crs = DEFAULT_DSM_TARGET_CRS
    args.pdal = DEFAULT_PDAL
    return args


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
    if radius_km <= 0:
        raise ValueError("--airport-radius-km must be greater than zero")

    lat_delta = radius_km / 111.32
    cos_lat = math.cos(math.radians(airport.latitude))
    lon_delta = 180.0 if abs(cos_lat) < 0.000001 else radius_km / (111.32 * cos_lat)
    return [
        airport.longitude - lon_delta,
        airport.latitude - lat_delta,
        airport.longitude + lon_delta,
        airport.latitude + lat_delta,
    ]


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)


def safe_path_component(value: str) -> str:
    cleaned = SAFE_COMPONENT_RE.sub("_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def build_query_contexts(args: argparse.Namespace) -> list[QueryContext]:
    airport_codes = normalize_codes(args.airports)
    airports = load_airports(args.airport_csv, airport_codes)
    contexts: list[QueryContext] = []

    for airport in airports:
        contexts.append(
            QueryContext(
                group=safe_path_component(airport.code),
                label=f"{airport.code}: {airport.name}",
                bbox=airport_bbox(airport, args.airport_radius_km),
                latitude=airport.latitude,
                longitude=airport.longitude,
            )
        )

    if args.bbox:
        lat, lon = bbox_center(args.bbox)
        contexts.append(
            QueryContext(
                group=None,
                label="bbox",
                bbox=list(args.bbox),
                latitude=lat,
                longitude=lon,
            )
        )

    return contexts


def selected_specs(args: argparse.Namespace) -> list[tuple[str, str, dict[str, str]]]:
    specs: list[tuple[str, str, dict[str, str]]] = []
    if args.product in ("dem", "both"):
        specs.append(("dem", args.dem_dataset, DEM_DATASETS[args.dem_dataset]))
    if args.product in ("dsm", "both"):
        specs.append(("dsm", args.dsm_source, DSM_SOURCES[args.dsm_source]))
    return specs


def tnm_payload_error(data: dict[str, Any]) -> str:
    if data.get("error"):
        return str(data["error"])
    if data.get("errors"):
        return str(data["errors"])
    if data.get("message") and "items" not in data and "total" not in data:
        return str(data["message"])
    return ""


def split_bbox(bbox: list[float]) -> list[list[float]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lon = (min_lon + max_lon) / 2
    mid_lat = (min_lat + max_lat) / 2
    return [
        [min_lon, min_lat, mid_lon, mid_lat],
        [mid_lon, min_lat, max_lon, mid_lat],
        [min_lon, mid_lat, mid_lon, max_lat],
        [mid_lon, mid_lat, max_lon, max_lat],
    ]


def http_json(
    url: str,
    params: dict[str, Any],
    timeout: int,
    retries: int = DEFAULT_QUERY_RETRIES,
) -> dict[str, Any]:
    query = urlencode({k: v for k, v in params.items() if v is not None})
    req = Request(f"{url}?{query}", method="GET")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "usgs-lidar-downloader/1.0")

    last_error = ""
    data: dict[str, Any] | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            try:
                data = json.loads(body)
            except json.JSONDecodeError as exc:
                detail = body[:500].replace("\n", " ")
                last_error = f"TNM returned invalid JSON: {detail}"
                if attempt >= retries:
                    raise RuntimeError(last_error) from exc
            else:
                payload_error = tnm_payload_error(data)
                if payload_error:
                    last_error = f"TNM error: {payload_error}"
                    data = None
                    if attempt >= retries:
                        raise RuntimeError(last_error)
                else:
                    break
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"TNM HTTP {exc.code}: {detail}"
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(last_error) from exc
        except URLError as exc:
            last_error = f"TNM network error: {exc}"
            if attempt >= retries:
                raise RuntimeError(last_error) from exc
        time.sleep(min(2**attempt, 10))
    else:
        raise RuntimeError(last_error or "TNM request failed")

    if data is None:
        raise RuntimeError(last_error or "TNM request failed")

    return data


def query_tnm_products_once(
    *,
    context: QueryContext,
    product: str,
    source_kind: str,
    spec: dict[str, str],
    timeout: int,
    page_size: int,
    query_retries: int,
    limit: int | None,
) -> list[IndexedProduct]:
    results: list[IndexedProduct] = []
    offset = 0
    max_page_size = max(1, min(page_size, 1000))

    while True:
        request_size = max_page_size
        if limit is not None:
            request_size = min(request_size, limit - len(results))
            if request_size <= 0:
                return results

        data = http_json(
            TNM_PRODUCTS_URL,
            {
                "datasets": spec["dataset"],
                "bbox": ",".join(str(value) for value in context.bbox),
                "prodFormats": spec["formats"],
                "outputFormat": "JSON",
                "max": request_size,
                "offset": offset,
            },
            timeout,
            retries=query_retries,
        )
        items = data.get("items") or []
        for item in items:
            candidate = IndexedProduct(
                group=context.group,
                product=product,
                scale=spec["scale"],
                dataset=spec["dataset"],
                source_kind=source_kind,
                item=item,
                query_bbox=context.bbox,
                query_latitude=context.latitude,
                query_longitude=context.longitude,
            )
            if candidate.url:
                results.append(candidate)

        total = int(data.get("total") or 0)
        if limit is not None and len(results) >= limit:
            return results[:limit]
        if not items or offset + len(items) >= total:
            return results
        offset += len(items)


def query_tnm_products(
    *,
    context: QueryContext,
    product: str,
    source_kind: str,
    spec: dict[str, str],
    timeout: int,
    page_size: int,
    query_retries: int,
    limit: int | None,
    split_depth: int = DEFAULT_MAX_QUERY_SPLIT_DEPTH,
) -> list[IndexedProduct]:
    try:
        return query_tnm_products_once(
            context=context,
            product=product,
            source_kind=source_kind,
            spec=spec,
            timeout=timeout,
            page_size=page_size,
            query_retries=query_retries,
            limit=limit,
        )
    except RuntimeError as exc:
        if split_depth <= 0:
            raise

        print(
            f"  TNM query retrying with split bbox for {product.upper()} "
            f"({str(exc)[:160]})",
            file=sys.stderr,
        )
        split_results: list[IndexedProduct] = []
        for index, child_bbox in enumerate(split_bbox(context.bbox), start=1):
            child_context = QueryContext(
                group=context.group,
                label=f"{context.label}/q{index}",
                bbox=child_bbox,
                latitude=context.latitude,
                longitude=context.longitude,
            )
            split_results.extend(
                query_tnm_products(
                    context=child_context,
                    product=product,
                    source_kind=source_kind,
                    spec=spec,
                    timeout=timeout,
                    page_size=page_size,
                    query_retries=query_retries,
                    limit=limit,
                    split_depth=split_depth - 1,
                )
            )
            if limit is not None and len(split_results) >= limit:
                return dedupe_products(split_results)[:limit]
        return dedupe_products(split_results)


def dedupe_products(products: list[IndexedProduct]) -> list[IndexedProduct]:
    deduped: dict[tuple[str, str, str], IndexedProduct] = {}
    for product in products:
        key = (
            product.group or "",
            product.product,
            product.url or f"{product.maptile}:{product.filename}",
        )
        deduped[key] = product
    return sorted(deduped.values(), key=lambda item: (item.product, item.maptile, item.filename))


def product_date_key(product: IndexedProduct) -> tuple[str, str, str]:
    item = product.item
    publication = str(item.get("publicationDate") or "")
    last_updated = str(item.get("lastUpdated") or "")
    created = str(item.get("dateCreated") or "")
    return (publication, last_updated, created or product.filename)


def keep_latest_per_tile(products: list[IndexedProduct]) -> list[IndexedProduct]:
    latest: dict[tuple[str, str, str], IndexedProduct] = {}
    for product in products:
        key = (product.group or "", product.product, product.maptile)
        current = latest.get(key)
        if current is None or product_date_key(product) > product_date_key(current):
            latest[key] = product
    return sorted(latest.values(), key=lambda item: (item.product, item.maptile, item.filename))


def output_path(base_dir: Path, product: IndexedProduct) -> Path:
    filename = Path(product.filename).name
    if product.group:
        product_root = base_dir / product.group / product.product
    else:
        product_root = base_dir / product.product

    if product.product == "dsm" and product.source_kind == "lpc":
        return product_root / "source_laz" / filename
    return product_root / filename


def one_airport_manifest(base_dir: Path, contexts: list[QueryContext]) -> Path | None:
    airport_groups = [context.group for context in contexts if context.group]
    if len(airport_groups) == 1 and len(contexts) == 1:
        return base_dir / airport_groups[0] / "download_manifest.csv"
    return None


def resolve_manifest(args: argparse.Namespace, contexts: list[QueryContext]) -> Path:
    if args.manifest:
        return args.manifest
    return one_airport_manifest(args.out, contexts) or args.out / "download_manifest.csv"


def download_one(
    product: IndexedProduct,
    target: Path,
    *,
    overwrite: bool,
    timeout: int,
    retries: int,
) -> DownloadResult:
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        return DownloadResult(product=product, target=target, status="skipped", message="exists")

    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_name(target.name + ".part")
    last_error = ""

    for attempt in range(1, retries + 1):
        try:
            req = Request(product.url, method="GET")
            req.add_header("User-Agent", "usgs-lidar-downloader/1.0")
            with urlopen(req, timeout=timeout) as resp, part_path.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            part_path.replace(target)
            return DownloadResult(product=product, target=target, status="downloaded")
        except (HTTPError, URLError, OSError) as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(min(2**attempt, 10))

    return DownloadResult(product=product, target=target, status="failed", message=last_error)


def utm_zone(longitude: float) -> int:
    return max(1, min(60, int((longitude + 180) // 6) + 1))


def auto_utm_epsg(latitude: float, longitude: float) -> str:
    zone = utm_zone(longitude)
    return f"EPSG:{32600 + zone if latitude < 0 else 26900 + zone}"


def dsm_tif_path(laz_path: Path) -> Path:
    if laz_path.parent.name == "source_laz":
        return laz_path.parent.parent / f"{laz_path.stem}.tif"
    return laz_path.with_suffix(".tif")


def resolve_dsm_target_crs(result: DownloadResult, target_crs: str) -> str | None:
    value = target_crs.strip()
    if not value or value.lower() == "source":
        return None
    if value.lower() != "auto-utm":
        return value

    lat = result.product.query_latitude
    lon = result.product.query_longitude
    if lat is None or lon is None:
        return None
    return auto_utm_epsg(lat, lon)


def resolve_executable(executable: str) -> str | None:
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    if Path(executable).exists():
        return executable
    return None


def pdal_pipeline(
    *,
    input_path: Path,
    output_path: Path,
    input_crs: str | None,
    target_crs: str | None,
    expression: str | None,
    resolution: float,
    output_type: str,
    nodata: float,
) -> list[dict[str, Any]]:
    reader: dict[str, Any] = {"type": "readers.las", "filename": str(input_path)}
    if input_crs:
        reader["override_srs"] = input_crs

    stages: list[dict[str, Any]] = [reader]
    if target_crs:
        stages.append({"type": "filters.reprojection", "out_srs": target_crs})
    if expression:
        stages.append({"type": "filters.expression", "expression": expression})

    stages.append(
        {
            "type": "writers.gdal",
            "filename": str(output_path),
            "resolution": resolution,
            "output_type": output_type,
            "data_type": "float32",
            "nodata": nodata,
        }
    )
    return stages


def derive_dsm_one(
    result: DownloadResult,
    *,
    pdal_exe: str,
    overwrite: bool,
    input_crs: str | None,
    target_crs: str,
    expression: str | None,
    resolution: float,
    output_type: str,
    nodata: float,
) -> DownloadResult:
    laz_path = result.target
    tif_path = dsm_tif_path(laz_path)
    result.derived_target = tif_path

    if result.status == "failed":
        result.derived_status = "skipped"
        result.derived_message = "download failed"
        return result
    if not laz_path.exists():
        result.derived_status = "failed"
        result.derived_message = f"source LAZ not found: {laz_path}"
        return result
    if tif_path.exists() and tif_path.stat().st_size > 0 and not overwrite:
        result.derived_status = "skipped"
        result.derived_message = "exists"
        return result

    tif_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_target_crs = resolve_dsm_target_crs(result, target_crs)
    pipeline = pdal_pipeline(
        input_path=laz_path,
        output_path=tif_path,
        input_crs=input_crs,
        target_crs=resolved_target_crs,
        expression=expression,
        resolution=resolution,
        output_type=output_type,
        nodata=nodata,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as f:
        json.dump({"pipeline": pipeline}, f, indent=2)
        pipeline_path = Path(f.name)

    try:
        completed = subprocess.run(
            [pdal_exe, "pipeline", str(pipeline_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        result.derived_status = "failed"
        result.derived_message = str(exc)[:500]
        return result
    finally:
        try:
            pipeline_path.unlink()
        except OSError:
            pass

    if completed.returncode == 0:
        result.derived_status = "derived"
        result.derived_message = f"target_crs={resolved_target_crs or 'source'}"
    else:
        result.derived_status = "failed"
        stderr = completed.stderr.strip() or completed.stdout.strip()
        result.derived_message = stderr[:500]
    return result


def derive_dsm_geotiffs(args: argparse.Namespace, results: list[DownloadResult]) -> list[DownloadResult]:
    lpc_results = [
        result
        for result in results
        if result.product.product == "dsm" and result.product.source_kind == "lpc"
    ]
    if not lpc_results:
        return results

    pdal_path = resolve_executable(args.pdal)
    if not pdal_path:
        for result in lpc_results:
            result.derived_target = dsm_tif_path(result.target)
            result.derived_status = "failed"
            result.derived_message = "PDAL executable not found"
        return results

    for index, result in enumerate(lpc_results, start=1):
        derive_dsm_one(
            result,
            pdal_exe=pdal_path,
            overwrite=args.overwrite,
            input_crs=args.dsm_input_crs,
            target_crs=args.dsm_target_crs,
            expression=args.dsm_expression,
            resolution=args.dsm_resolution_m,
            output_type=args.dsm_output_type,
            nodata=args.dsm_nodata,
        )
        label = f"{result.target.name} -> {result.derived_target.name if result.derived_target else 'DSM'}"
        if result.derived_status == "failed":
            print(f"  DSM derive failed {index}/{len(lpc_results)}: {label} ({result.derived_message})")
        else:
            print(f"  DSM {result.derived_status} {index}/{len(lpc_results)}: {label}")

    return results


def iso_bbox(item: dict[str, Any]) -> str:
    bbox = item.get("boundingBox") or {}
    values = [bbox.get("minX"), bbox.get("minY"), bbox.get("maxX"), bbox.get("maxY")]
    if any(value is None for value in values):
        return ""
    return ",".join(str(value) for value in values)


def write_manifest(path: Path, rows: list[DownloadResult]) -> None:
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
        "derived_status",
        "derived_message",
        "derived_target",
        "source_kind",
        "dataset",
        "format",
        "title",
        "source_id",
        "size_in_bytes",
        "publication_date",
        "last_updated",
        "bbox",
        "meta_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = row.product.item
            writer.writerow(
                {
                    "status": row.status,
                    "message": row.message,
                    "group": row.product.group or "",
                    "product": row.product.product,
                    "scale": row.product.scale,
                    "maptile": row.product.maptile,
                    "filename": row.product.filename,
                    "year": row.product.year,
                    "projection": row.product.projection,
                    "spacing": DEM_DATASETS.get(row.product.scale, {}).get("spacing", "")
                    or DSM_SOURCES.get(row.product.scale, {}).get("spacing", "")
                    or ("1 meter" if row.derived_status == "derived" else ""),
                    "url": row.product.url,
                    "target": str(row.target),
                    "derived_status": row.derived_status,
                    "derived_message": row.derived_message,
                    "derived_target": str(row.derived_target or ""),
                    "source_kind": row.product.source_kind,
                    "dataset": row.product.dataset,
                    "format": item.get("format", ""),
                    "title": item.get("title", ""),
                    "source_id": item.get("sourceId", ""),
                    "size_in_bytes": row.product.size_in_bytes or "",
                    "publication_date": item.get("publicationDate", ""),
                    "last_updated": item.get("lastUpdated", ""),
                    "bbox": iso_bbox(item),
                    "meta_url": item.get("metaUrl", ""),
                }
            )


def validate_args(args: argparse.Namespace) -> None:
    if args.dsm_resolution_m <= 0:
        raise ValueError("--dsm-resolution-m must be greater than zero")
    if args.page_size <= 0:
        raise ValueError("--page-size must be greater than zero")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if args.product in ("dsm", "both") and args.dsm_source != "lpc" and args.derive_dsm:
        raise ValueError("--derive-dsm only applies to --dsm-source lpc")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        contexts = build_query_contexts(args)
        specs = selected_specs(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not contexts:
        print("error: provide --airports and/or --bbox so the query is bounded", file=sys.stderr)
        return 2

    all_products: list[IndexedProduct] = []
    for product, source_kind, spec in specs:
        print(f"Querying {product.upper()} from USGS dataset: {spec['dataset']}")
        product_matches: list[IndexedProduct] = []

        for context in contexts:
            print(
                f"  {context.label}: bbox="
                f"{context.bbox[0]:.6f},{context.bbox[1]:.6f},"
                f"{context.bbox[2]:.6f},{context.bbox[3]:.6f}"
            )
            try:
                matches = query_tnm_products(
                    context=context,
                    product=product,
                    source_kind=source_kind,
                    spec=spec,
                    timeout=args.timeout,
                    page_size=args.page_size,
                    query_retries=args.query_retries,
                    limit=args.limit,
                )
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            product_matches.extend(matches)

        product_matches = dedupe_products(product_matches)
        print(f"  found {len(product_matches)} file(s)")
        all_products.extend(product_matches)

    if args.latest_per_tile:
        original_count = len(all_products)
        all_products = keep_latest_per_tile(all_products)
        if len(all_products) != original_count:
            print(f"Keeping latest record per product/tile: {len(all_products)} of {original_count}")

    has_lpc_dsm = any(
        product.product == "dsm" and product.source_kind == "lpc" for product in all_products
    )
    if args.derive_dsm and has_lpc_dsm and not args.dry_run and not resolve_executable(args.pdal):
        print(
            f"error: PDAL executable not found: {args.pdal!r}. Install PDAL "
            "or edit DEFAULT_PDAL / DEFAULT_DERIVE_DSM in this script.",
            file=sys.stderr,
        )
        return 2

    manifest = resolve_manifest(args, contexts)
    planned = [
        DownloadResult(product=product, target=output_path(args.out, product), status="planned")
        for product in all_products
    ]
    if args.derive_dsm:
        for row in planned:
            if row.product.product == "dsm" and row.product.source_kind == "lpc":
                row.derived_target = dsm_tif_path(row.target)
                row.derived_status = "planned"

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
    started = datetime.now()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                download_one,
                row.product,
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
            label = f"{result.product.product.upper()} {result.product.maptile} {result.product.filename}"
            if result.status == "failed":
                print(f"  failed: {label} ({result.message})")
            else:
                print(f"  {result.status}: {label}")

    results.sort(key=lambda row: (row.product.product, row.product.maptile, row.product.filename))
    if args.derive_dsm:
        results = derive_dsm_geotiffs(args, results)

    write_manifest(manifest, results)

    failed = sum(1 for row in results if row.status == "failed")
    downloaded = sum(1 for row in results if row.status == "downloaded")
    skipped = sum(1 for row in results if row.status == "skipped")
    derived_failed = sum(1 for row in results if row.derived_status == "failed")
    derived = sum(1 for row in results if row.derived_status == "derived")
    elapsed = datetime.now() - started
    print(
        f"Done in {elapsed}. downloaded={downloaded}, skipped={skipped}, "
        f"failed={failed}, derived={derived}, derived_failed={derived_failed}. "
        f"Manifest: {manifest}"
    )
    return 1 if failed or derived_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
