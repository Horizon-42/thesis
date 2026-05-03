#!/usr/bin/env python3
"""
Download airport-centered DEM rasters from the OpenTopography API.

The script asks OpenTopography to crop one raster per airport/bbox, so it avoids
the USGS tile-index workflow and does not need local mosaicking. Set an API key
with OT_API_KEY in .env, in the shell environment, or pass --api-key.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USGS_DEM_URL = "https://portal.opentopography.org/API/usgsdem"
GLOBAL_DEM_URL = "https://portal.opentopography.org/API/globaldem"

DEFAULT_AIRPORT_CSV = Path("aeroviz-4d/public/data/common/airports.csv")
DEFAULT_OUT_DIR = Path("data/opentopography")
DEFAULT_AIRPORT_RADIUS_KM = 5.0
DEFAULT_DATASET = "USGS10m"
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_RETRIES = 3
DEFAULT_WORKERS = 4
DEFAULT_ENV_FILE = Path(".env")

SAFE_COMPONENT_RE = re.compile(r"[^0-9A-Za-z._-]+")
USGS_DEM_FALLBACK_ORDER = ["USGS1m", "USGS10m", "USGS30m"]
FALLBACKABLE_ERROR_MARKERS = (
    "no data",
    "no dem",
    "not available",
    "not found",
    "empty response",
)

USGS_DATASETS: dict[str, dict[str, str]] = {
    "USGS1m": {
        "endpoint": USGS_DEM_URL,
        "param": "datasetName",
        "product": "dem",
        "spacing": "1 meter",
        "notes": "restricted by OpenTopography to academic/authorized users",
    },
    "USGS10m": {
        "endpoint": USGS_DEM_URL,
        "param": "datasetName",
        "product": "dem",
        "spacing": "1/3 arc-second, about 10 meter",
        "notes": "",
    },
    "USGS30m": {
        "endpoint": USGS_DEM_URL,
        "param": "datasetName",
        "product": "dem",
        "spacing": "1 arc-second, about 30 meter",
        "notes": "",
    },
}

GLOBAL_DATASETS: dict[str, dict[str, str]] = {
    "SRTMGL3": {"product": "dem", "spacing": "90 meter", "notes": ""},
    "SRTMGL1": {"product": "dem", "spacing": "30 meter", "notes": ""},
    "SRTMGL1_E": {"product": "dem", "spacing": "30 meter ellipsoidal", "notes": ""},
    "AW3D30": {"product": "dem", "spacing": "30 meter", "notes": ""},
    "AW3D30_E": {"product": "dem", "spacing": "30 meter ellipsoidal", "notes": ""},
    "SRTM15Plus": {"product": "dem", "spacing": "500 meter", "notes": ""},
    "NASADEM": {"product": "dem", "spacing": "30 meter", "notes": ""},
    "COP30": {"product": "dsm", "spacing": "30 meter", "notes": "Copernicus global DSM"},
    "COP90": {"product": "dsm", "spacing": "90 meter", "notes": "Copernicus global DSM"},
    "EU_DTM": {"product": "dem", "spacing": "30 meter", "notes": ""},
    "GEDI_L3": {"product": "dtm", "spacing": "1000 meter", "notes": ""},
    "GEBCOIceTopo": {"product": "dem", "spacing": "500 meter", "notes": ""},
    "GEBCOSubIceTopo": {"product": "dem", "spacing": "500 meter", "notes": ""},
    "CA_MRDEM_DTM": {"product": "dem", "spacing": "30 meter", "notes": "Canada DTM"},
    "CA_MRDEM_DSM": {"product": "dsm", "spacing": "30 meter", "notes": "Canada DSM"},
}

ALL_DATASETS = {**USGS_DATASETS, **GLOBAL_DATASETS}


@dataclass(frozen=True)
class Airport:
    code: str
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class DownloadJob:
    group: str
    label: str
    bbox: list[float]
    dataset: str
    requested_dataset: str
    output_format: str
    out_dir: Path
    target: Path

    @property
    def product(self) -> str:
        return dataset_info(self.dataset)["product"]

    @property
    def spacing(self) -> str:
        return dataset_info(self.dataset)["spacing"]

    @property
    def notes(self) -> str:
        return dataset_info(self.dataset)["notes"]


@dataclass
class DownloadResult:
    job: DownloadJob
    status: str
    message: str = ""
    size_in_bytes: int | None = None


def parse_dotenv_value(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    quote = value[0]
    if quote in {"'", '"'}:
        end_index = value.find(quote, 1)
        if end_index >= 0:
            return value[1:end_index]
        return value[1:]
    return value.split("#", 1)[0].strip()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key or (key in os.environ and os.environ[key]):
            continue
        os.environ[key] = parse_dotenv_value(raw_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OpenTopography DEM/DSM rasters for airport-centered bboxes."
    )
    parser.add_argument(
        "airport_codes",
        nargs="*",
        metavar="AIRPORT",
        help="Airport code(s), for example KRDU KSJC. Commas are also accepted.",
    )
    parser.add_argument(
        "--airports",
        nargs="*",
        default=[],
        help="Airport ICAO/ident codes. Multiple values or comma-separated values are accepted.",
    )
    parser.add_argument(
        "--icaos",
        nargs="*",
        default=[],
        help="Alias for --airports. Example: --icaos KRDU KSJC or --icaos KRDU,KSJC.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Download one raster for a WGS84 longitude/latitude bounding box.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(ALL_DATASETS),
        default=DEFAULT_DATASET,
        help="OpenTopography dataset. Default: USGS10m.",
    )
    parser.add_argument(
        "--output-format",
        choices=["GTiff", "AAIGrid", "HFA"],
        default="GTiff",
        help="Output format. Default: GTiff.",
    )
    parser.add_argument(
        "--airport-csv",
        type=Path,
        default=DEFAULT_AIRPORT_CSV,
        help="Airport CSV path. Default: aeroviz-4d/public/data/common/airports.csv.",
    )
    parser.add_argument(
        "--airport-radius-km",
        type=float,
        default=DEFAULT_AIRPORT_RADIUS_KM,
        help="Crop radius around each airport coordinate in kilometres. Default: 5.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory. Default: data/opentopography.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest CSV path. Default: <out>/download_manifest.csv.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Read OT_API_KEY from this .env file if present. Default: .env.",
    )
    parser.add_argument(
        "--api-key",
        help="OpenTopography API key. Defaults to OT_API_KEY or OPENTOPOGRAPHY_API_KEY.",
    )
    parser.add_argument(
        "--no-dem-fallback",
        action="store_true",
        help="Disable USGS DEM fallback, for example USGS1m -> USGS10m -> USGS30m.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest and print planned requests without downloading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of skipping them.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent downloads. Default: 4.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds. Default: 300.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Download retries per request. Default: 3.",
    )

    args = parser.parse_args()
    load_env_file(args.env_file)
    if not args.api_key:
        args.api_key = os.environ.get("OT_API_KEY") or os.environ.get("OPENTOPOGRAPHY_API_KEY")
    args.airports = [*args.airport_codes, *args.airports, *args.icaos]
    del args.airport_codes
    del args.icaos
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


def safe_path_component(value: str) -> str:
    cleaned = SAFE_COMPONENT_RE.sub("_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


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


def dataset_info(dataset: str) -> dict[str, str]:
    info = ALL_DATASETS.get(dataset)
    if info is None:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if dataset in GLOBAL_DATASETS:
        return {
            "endpoint": GLOBAL_DEM_URL,
            "param": "demtype",
            **info,
        }
    return info


def output_suffix(output_format: str) -> str:
    return {
        "GTiff": ".tif",
        "AAIGrid": ".asc",
        "HFA": ".img",
    }[output_format]


def output_path(base_dir: Path, group: str, dataset: str, product: str, output_format: str) -> Path:
    filename = f"{safe_path_component(group)}_{dataset}{output_suffix(output_format)}"
    return base_dir / safe_path_component(group) / product / filename


def fallback_datasets(dataset: str, enabled: bool) -> list[str]:
    if not enabled or dataset not in USGS_DEM_FALLBACK_ORDER:
        return [dataset]
    index = USGS_DEM_FALLBACK_ORDER.index(dataset)
    return USGS_DEM_FALLBACK_ORDER[index:]


def candidate_job(job: DownloadJob, dataset: str) -> DownloadJob:
    info = dataset_info(dataset)
    return replace(
        job,
        dataset=dataset,
        target=output_path(job.out_dir, job.group, dataset, info["product"], job.output_format),
    )


def candidate_text(job: DownloadJob, enabled: bool) -> str:
    return " -> ".join(fallback_datasets(job.dataset, enabled))


def build_jobs(args: argparse.Namespace) -> list[DownloadJob]:
    airport_codes = normalize_codes(args.airports)
    airports = load_airports(args.airport_csv, airport_codes)
    jobs: list[DownloadJob] = []
    info = dataset_info(args.dataset)

    for airport in airports:
        group = safe_path_component(airport.code)
        bbox = airport_bbox(airport, args.airport_radius_km)
        jobs.append(
            DownloadJob(
                group=group,
                label=f"{airport.code}: {airport.name}",
                bbox=bbox,
                dataset=args.dataset,
                requested_dataset=args.dataset,
                output_format=args.output_format,
                out_dir=args.out,
                target=output_path(args.out, group, args.dataset, info["product"], args.output_format),
            )
        )

    if args.bbox:
        group = "bbox"
        jobs.append(
            DownloadJob(
                group=group,
                label="bbox",
                bbox=list(args.bbox),
                dataset=args.dataset,
                requested_dataset=args.dataset,
                output_format=args.output_format,
                out_dir=args.out,
                target=output_path(args.out, group, args.dataset, info["product"], args.output_format),
            )
        )

    return jobs


def api_url(job: DownloadJob, api_key: str | None, *, include_key: bool) -> str:
    info = dataset_info(job.dataset)
    min_lon, min_lat, max_lon, max_lat = job.bbox
    params: dict[str, str | float] = {
        info["param"]: job.dataset,
        "south": min_lat,
        "north": max_lat,
        "west": min_lon,
        "east": max_lon,
        "outputFormat": job.output_format,
    }
    if include_key and api_key:
        params["API_Key"] = api_key
    return f"{info['endpoint']}?{urlencode(params)}"


def bbox_text(bbox: list[float]) -> str:
    return ",".join(f"{value:.6f}" for value in bbox)


def write_manifest(path: Path, rows: list[DownloadResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "message",
        "group",
        "label",
        "product",
        "requested_dataset",
        "dataset",
        "spacing",
        "output_format",
        "bbox",
        "url_without_api_key",
        "target",
        "size_in_bytes",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            job = row.job
            writer.writerow(
                {
                    "status": row.status,
                    "message": row.message,
                    "group": job.group,
                    "label": job.label,
                    "product": job.product,
                    "requested_dataset": job.requested_dataset,
                    "dataset": job.dataset,
                    "spacing": job.spacing,
                    "output_format": job.output_format,
                    "bbox": bbox_text(job.bbox),
                    "url_without_api_key": api_url(job, None, include_key=False),
                    "target": str(job.target),
                    "size_in_bytes": row.size_in_bytes or "",
                    "notes": job.notes,
                }
            )


def read_error_detail(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:500].replace("\n", " ")
    except OSError:
        return ""


def download_single(
    job: DownloadJob,
    *,
    api_key: str,
    overwrite: bool,
    timeout: int,
    retries: int,
) -> DownloadResult:
    if job.target.exists() and job.target.stat().st_size > 0 and not overwrite:
        return DownloadResult(
            job=job,
            status="skipped",
            message="exists",
            size_in_bytes=job.target.stat().st_size,
        )

    job.target.parent.mkdir(parents=True, exist_ok=True)
    part_path = job.target.with_name(job.target.name + ".part")
    last_error = ""

    for attempt in range(1, max(1, retries) + 1):
        try:
            req = Request(api_url(job, api_key, include_key=True), method="GET")
            req.add_header("User-Agent", "opentopography-dem-downloader/1.0")

            with urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if status == 204:
                    return DownloadResult(job=job, status="failed", message="no data")

                content_type = (resp.headers.get("Content-Type") or "").lower()
                first_chunk = resp.read(1024 * 1024)
                if not first_chunk:
                    return DownloadResult(job=job, status="failed", message="empty response")

                if any(kind in content_type for kind in ("json", "text", "html", "xml")):
                    detail = first_chunk.decode("utf-8", errors="replace")[:500].replace("\n", " ")
                    return DownloadResult(
                        job=job,
                        status="failed",
                        message=f"unexpected {content_type or 'text'} response: {detail}",
                    )

                with part_path.open("wb") as out:
                    out.write(first_chunk)
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)

            part_path.replace(job.target)
            return DownloadResult(
                job=job,
                status="downloaded",
                size_in_bytes=job.target.stat().st_size,
            )
        except HTTPError as exc:
            detail = read_error_detail(exc)
            last_error = f"HTTP {exc.code}: {detail}".strip()
        except (URLError, OSError) as exc:
            last_error = str(exc)

        try:
            part_path.unlink()
        except OSError:
            pass
        if attempt < max(1, retries):
            time.sleep(min(2**attempt, 10))

    return DownloadResult(job=job, status="failed", message=last_error)


def should_try_fallback(result: DownloadResult) -> bool:
    if result.status != "failed":
        return False
    message = result.message.lower()
    return any(marker in message for marker in FALLBACKABLE_ERROR_MARKERS)


def download_one(
    job: DownloadJob,
    *,
    api_key: str,
    overwrite: bool,
    timeout: int,
    retries: int,
    dem_fallback: bool,
) -> DownloadResult:
    attempted: list[str] = []
    candidates = fallback_datasets(job.dataset, dem_fallback)
    last_result: DownloadResult | None = None

    for dataset in candidates:
        current_job = candidate_job(job, dataset)
        result = download_single(
            current_job,
            api_key=api_key,
            overwrite=overwrite,
            timeout=timeout,
            retries=retries,
        )

        if result.status in {"downloaded", "skipped"}:
            if dataset != job.requested_dataset:
                attempted_text = "; ".join(attempted)
                detail = f"fallback from {job.requested_dataset}"
                if attempted_text:
                    detail = f"{detail}; previous attempts: {attempted_text}"
                result.message = detail if not result.message else f"{result.message}; {detail}"
            return result

        attempted.append(f"{dataset}: {result.message}")
        last_result = result
        if not should_try_fallback(result):
            return result

    if last_result:
        last_result.message = "; ".join(attempted)
        return last_result
    return DownloadResult(job=job, status="failed", message="no datasets attempted")


def validate_args(args: argparse.Namespace) -> None:
    if args.workers <= 0:
        raise ValueError("--workers must be greater than zero")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if args.retries <= 0:
        raise ValueError("--retries must be greater than zero")
    if args.bbox:
        min_lon, min_lat, max_lon, max_lat = args.bbox
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("--bbox must be MIN_LON MIN_LAT MAX_LON MAX_LAT")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        jobs = build_jobs(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not jobs:
        print("error: provide at least one airport code or --bbox", file=sys.stderr)
        return 2

    manifest = args.manifest or args.out / "download_manifest.csv"
    planned = [DownloadResult(job=job, status="planned") for job in jobs]
    dem_fallback = not args.no_dem_fallback

    print(f"Dataset: {args.dataset} ({dataset_info(args.dataset)['spacing']})")
    if len(fallback_datasets(args.dataset, dem_fallback)) > 1:
        print(f"DEM fallback order: {' -> '.join(fallback_datasets(args.dataset, dem_fallback))}")
    for job in jobs:
        print(
            f"  {job.label}: bbox={bbox_text(job.bbox)} "
            f"candidates={candidate_text(job, dem_fallback)} -> {job.target}"
        )

    if args.dry_run:
        write_manifest(manifest, planned)
        print(f"Dry run complete. Manifest: {manifest}")
        return 0

    if not args.api_key:
        print(
            "error: OpenTopography API key required. Set OT_API_KEY in .env or pass --api-key.",
            file=sys.stderr,
        )
        return 2

    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                download_one,
                job,
                api_key=args.api_key,
                overwrite=args.overwrite,
                timeout=args.timeout,
                retries=args.retries,
                dem_fallback=dem_fallback,
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            label = f"{result.job.group} {result.job.dataset}"
            if result.status == "failed":
                print(f"  failed: {label} ({result.message})")
            else:
                print(f"  {result.status}: {label} -> {result.job.target}")

    results.sort(key=lambda row: (row.job.group, row.job.dataset))
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
