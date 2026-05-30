"""
preprocess_usgs_tnm_terrain.py
==============================
Convert USGS TNM elevation downloads into AeroViz local terrain packages.

The frontend already consumes airport-local terrain as:

  public/data/airports/<ICAO>/local-terrain/heightmap/metadata.json
  public/data/airports/<ICAO>/local-terrain/heightmap/tiles/<level>/<x>/<y>.f32

This module normalizes the two USGS source shapes into GeoTIFF staging data,
then reuses scripts/build_local_terrain_heightmap.mjs to write that browser-ready
heightmap package.

Supported source kinds:
- DEM GeoTIFF: cropped to the airport footprint, values used directly as metres.
- DSM LAZ: rasterized with PDAL to a DSM GeoTIFF, XY reprojected to UTM metres,
  Z scaled from feet to metres for the inspected KRDU USGS LPC source.

Usage:
  # Bare-earth terrain.
  python python/preprocess_usgs_tnm_terrain.py --airport KRDU --source dem

  # Surface model from LAZ point cloud.
  python python/preprocess_usgs_tnm_terrain.py --airport KRDU --source dsm

  # Stage both normalized GeoTIFFs and publish the highest precision source.
  python python/preprocess_usgs_tnm_terrain.py --airport KRDU --source both
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from data_layout import (
    AEROVIZ_ROOT,
    airport_data_path,
    airport_local_terrain_sources_dir,
    normalize_airport_code,
)


SourceKind = Literal["dem", "dsm"]

DEFAULT_AIRPORT_CODE = "KRDU"
DEFAULT_USGS_TNM_ROOT = AEROVIZ_ROOT.parent / "data" / "usgs_tnm_elevation"
DEFAULT_FALLBACK_RADIUS_KM = 20.0
DEFAULT_DSM_RESOLUTION_M = 2.0
DEFAULT_LAZ_VERTICAL_SCALE = 0.3048
DEFAULT_NODATA = -999999.0
PROGRESS_BAR_WIDTH = 28
COMMAND_HEARTBEAT_SECONDS = 10.0
TERRAIN_SOURCE_METADATA_FILE = "terrain-source.json"

# The inspected KRDU LPC files have no embedded SRS. Their XY values match
# NAD83 / North Carolina StatePlane ftUS (EPSG:2264), while Z values are feet.
DEFAULT_LAZ_SOURCE_SRS_BY_AIRPORT = {
    "KRDU": "EPSG:2264",
}


@dataclass(frozen=True)
class GeoBBox:
    west: float
    south: float
    east: float
    north: float

    def as_gdal_te(self) -> list[str]:
        return [
            f"{self.west:.10f}",
            f"{self.south:.10f}",
            f"{self.east:.10f}",
            f"{self.north:.10f}",
        ]


@dataclass(frozen=True)
class StagedTerrainSource:
    source_kind: SourceKind
    source_dir: Path
    output_tif: Path
    horizontal_resolution_m: float | None = None


def parse_bbox(value: str) -> GeoBBox:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be west,south,east,north in decimal degrees")

    west, south, east, north = (float(part) for part in parts)
    if west >= east or south >= north:
        raise ValueError("--bbox must satisfy west < east and south < north")

    return GeoBBox(west=west, south=south, east=east, north=north)


def union_bboxes(bboxes: list[GeoBBox]) -> GeoBBox:
    if not bboxes:
        raise ValueError("Cannot union an empty bbox list")

    return GeoBBox(
        west=min(bbox.west for bbox in bboxes),
        south=min(bbox.south for bbox in bboxes),
        east=max(bbox.east for bbox in bboxes),
        north=max(bbox.north for bbox in bboxes),
    )


def bbox_from_center_radius(lon: float, lat: float, radius_km: float) -> GeoBBox:
    lat_delta = radius_km / 111.32
    lon_delta = radius_km / (111.32 * max(0.01, math.cos(math.radians(lat))))
    return GeoBBox(
        west=lon - lon_delta,
        south=lat - lat_delta,
        east=lon + lon_delta,
        north=lat + lat_delta,
    )


def read_manifest_bboxes(
    manifest_path: Path,
    airport_code: str,
    *,
    bbox_column: str = "product_bbox",
) -> list[GeoBBox]:
    if not manifest_path.exists():
        return []

    normalized_airport = normalize_airport_code(airport_code)
    bboxes: list[GeoBBox] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if normalize_airport_code(row.get("group") or "") != normalized_airport:
                continue

            raw_bbox = (row.get(bbox_column) or row.get("query_bbox") or "").strip()
            if not raw_bbox:
                continue
            bboxes.append(parse_bbox(raw_bbox))

    return bboxes


def load_airport_center(airport_code: str) -> tuple[float, float]:
    airport_json_path = airport_data_path(airport_code, "airport.json")
    data = json.loads(airport_json_path.read_text(encoding="utf-8"))
    return float(data["lon"]), float(data["lat"])


def resolve_processing_bbox(
    *,
    airport_code: str,
    explicit_bbox: GeoBBox | None,
    usgs_root: Path,
    fallback_radius_km: float,
) -> GeoBBox:
    if explicit_bbox is not None:
        return explicit_bbox

    manifest_bbox = union_or_none(
        read_manifest_bboxes(usgs_root / "download_manifest.csv", airport_code)
    )
    if manifest_bbox is not None:
        return manifest_bbox

    center_lon, center_lat = load_airport_center(airport_code)
    return bbox_from_center_radius(center_lon, center_lat, fallback_radius_km)


def union_or_none(bboxes: list[GeoBBox]) -> GeoBBox | None:
    return union_bboxes(bboxes) if bboxes else None


def default_target_srs_for_bbox(bbox: GeoBBox) -> str:
    center_lon = (bbox.west + bbox.east) / 2
    center_lat = (bbox.south + bbox.north) / 2
    zone = math.floor((center_lon + 180) / 6) + 1

    if center_lat >= 0:
        # NAD83 UTM is a good match for USGS TNM CONUS data.
        return f"EPSG:{26900 + zone}"

    return f"EPSG:{32700 + zone}"


def source_root_for_airport(usgs_root: Path, airport_code: str) -> Path:
    return usgs_root / normalize_airport_code(airport_code)


def staging_dir_for_source(airport_code: str, source_kind: SourceKind) -> Path:
    return airport_local_terrain_sources_dir(airport_code) / f"usgs-tnm-{source_kind}"


def list_source_files(input_dir: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not input_dir.exists():
        return []

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def require_tool(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise RuntimeError(f"Required command not found on PATH: {tool_name}")


def format_elapsed(seconds: float) -> str:
    minutes, remaining_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {remaining_seconds}s"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def progress_bar(completed: int, total: int, *, width: int = PROGRESS_BAR_WIDTH) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"

    clamped_completed = max(0, min(completed, total))
    filled = round((clamped_completed / total) * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_stage_progress(completed: int, total: int, label: str) -> None:
    percent = round((completed / total) * 100) if total > 0 else 0
    print(f"{progress_bar(completed, total)} {completed}/{total} {percent:3d}% {label}", flush=True)


def run_command(
    command: list[str],
    *,
    dry_run: bool,
    label: str,
    heartbeat_seconds: float = COMMAND_HEARTBEAT_SECONDS,
) -> None:
    print(f"[command] {label}: {shlex.join(command)}", flush=True)
    if dry_run:
        print(f"[skip] {label}: dry run", flush=True)
        return

    start = time.monotonic()
    process = subprocess.Popen(command)
    next_heartbeat = start + heartbeat_seconds

    while True:
        return_code = process.poll()
        now = time.monotonic()
        if return_code is not None:
            break

        if now >= next_heartbeat:
            print(f"[progress] {label}: still running after {format_elapsed(now - start)}", flush=True)
            next_heartbeat = now + heartbeat_seconds

        time.sleep(0.5)

    elapsed = format_elapsed(time.monotonic() - start)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    print(f"[done] {label}: finished in {elapsed}", flush=True)


def metres_per_degree_at_latitude(lat_deg: float) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    metres_per_degree_lat = (
        111132.92
        - 559.82 * math.cos(2 * lat_rad)
        + 1.175 * math.cos(4 * lat_rad)
        - 0.0023 * math.cos(6 * lat_rad)
    )
    metres_per_degree_lon = (
        111412.84 * math.cos(lat_rad)
        - 93.5 * math.cos(3 * lat_rad)
        + 0.118 * math.cos(5 * lat_rad)
    )
    return metres_per_degree_lon, metres_per_degree_lat


def horizontal_resolution_from_gdalinfo(geotiff_path: Path) -> float:
    require_tool("gdalinfo")
    result = subprocess.run(
        ["gdalinfo", "-json", str(geotiff_path)],
        text=True,
        capture_output=True,
        check=True,
    )
    metadata = json.loads(result.stdout)
    geo_transform = metadata.get("geoTransform")
    if not isinstance(geo_transform, list) or len(geo_transform) < 6:
        raise RuntimeError(f"Cannot read GeoTIFF transform from {geotiff_path}")

    pixel_width = abs(float(geo_transform[1]))
    pixel_height = abs(float(geo_transform[5]))
    corners = metadata.get("cornerCoordinates") or {}
    center = corners.get("center")
    if pixel_width < 1 and pixel_height < 1 and isinstance(center, list) and len(center) >= 2:
        metres_per_degree_lon, metres_per_degree_lat = metres_per_degree_at_latitude(float(center[1]))
        return max(pixel_width * metres_per_degree_lon, pixel_height * metres_per_degree_lat)

    return max(pixel_width, pixel_height)


def write_terrain_source_metadata(
    *,
    source: StagedTerrainSource,
    horizontal_resolution_m: float,
    note: str,
    dry_run: bool,
) -> StagedTerrainSource:
    metadata = {
        "schemaVersion": 1,
        "source": {
            "kind": source.source_kind,
            "label": f"USGS TNM {source.source_kind.upper()}",
            "sourceDir": str(source.source_dir.relative_to(AEROVIZ_ROOT)),
        },
        "precision": {
            "horizontalResolutionM": horizontal_resolution_m,
            "verticalAccuracyM": None,
            "notes": [note],
        },
    }
    metadata_path = source.source_dir / TERRAIN_SOURCE_METADATA_FILE
    if dry_run:
        print(f"[skip] Would write {metadata_path}: {json.dumps(metadata, indent=2)}", flush=True)
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return StagedTerrainSource(
        source_kind=source.source_kind,
        source_dir=source.source_dir,
        output_tif=source.output_tif,
        horizontal_resolution_m=horizontal_resolution_m,
    )


def select_highest_precision_source(
    staged: dict[SourceKind, StagedTerrainSource],
) -> SourceKind:
    missing_precision = [
        source_kind
        for source_kind, source in staged.items()
        if source.horizontal_resolution_m is None
    ]
    if missing_precision:
        missing = ", ".join(missing_precision)
        raise RuntimeError(
            f"Cannot choose terrain source by precision because {missing} lacks precision metadata. "
            "Regenerate staging data so terrain-source.json is written."
        )

    return min(
        staged,
        key=lambda source_kind: (
            staged[source_kind].horizontal_resolution_m or math.inf,
            str(staged[source_kind].source_dir),
        ),
    )


def stage_dem_geotiff(
    *,
    airport_code: str,
    usgs_root: Path,
    bbox: GeoBBox,
    dry_run: bool,
) -> StagedTerrainSource:
    require_tool("gdalwarp")

    source_dir = source_root_for_airport(usgs_root, airport_code) / "dem"
    source_paths = list_source_files(source_dir, (".tif", ".tiff"))
    if not source_paths:
        raise FileNotFoundError(f"No DEM GeoTIFF files found in {source_dir}")

    output_dir = staging_dir_for_source(airport_code, "dem")
    output_tif = output_dir / "usgs_tnm_dem_wgs84_elevation_m.tif"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "gdalwarp",
        "-overwrite",
        "-multi",
        "-wo",
        "NUM_THREADS=ALL_CPUS",
        "-t_srs",
        "EPSG:4326",
        "-te_srs",
        "EPSG:4326",
        "-te",
        *bbox.as_gdal_te(),
        "-r",
        "bilinear",
        "-dstnodata",
        str(DEFAULT_NODATA),
        "-of",
        "GTiff",
        "-co",
        "COMPRESS=LZW",
        "-co",
        "TILED=YES",
        "-co",
        "BIGTIFF=IF_SAFER",
        *[str(path) for path in source_paths],
        str(output_tif),
    ]
    run_command(command, dry_run=dry_run, label="Crop DEM GeoTIFF")
    precision_input = output_tif if output_tif.exists() else source_paths[0]
    horizontal_resolution_m = horizontal_resolution_from_gdalinfo(precision_input)
    return write_terrain_source_metadata(
        source=StagedTerrainSource(source_kind="dem", source_dir=output_dir, output_tif=output_tif),
        horizontal_resolution_m=horizontal_resolution_m,
        note="Computed from GeoTIFF pixel size after DEM normalization.",
        dry_run=dry_run,
    )


def numeric_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def transformation_matrix_for_z_scale(scale: float) -> str:
    return f"1 0 0 0 0 1 0 0 0 0 {scale:g} 0 0 0 0 1"


def pdal_bounds_string(bounds: tuple[float, float, float, float]) -> str:
    min_x, min_y, max_x, max_y = bounds
    return f"([{min_x:.3f},{max_x:.3f}],[{min_y:.3f},{max_y:.3f}])"


def bbox_corner_lon_lats(bbox: GeoBBox) -> list[tuple[float, float]]:
    return [
        (bbox.west, bbox.south),
        (bbox.west, bbox.north),
        (bbox.east, bbox.south),
        (bbox.east, bbox.north),
    ]


def bounds_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def transform_bbox_with_gdaltransform(
    bbox: GeoBBox,
    target_srs: str,
) -> tuple[float, float, float, float]:
    if shutil.which("gdaltransform") is None:
        raise RuntimeError(
            "Cannot transform the LAZ crop bbox because Python GDAL bindings "
            "and the gdaltransform command are both unavailable."
        )

    input_text = "\n".join(
        f"{lon:.10f} {lat:.10f}" for lon, lat in bbox_corner_lon_lats(bbox)
    ) + "\n"
    result = subprocess.run(
        ["gdaltransform", "-s_srs", "EPSG:4326", "-t_srs", target_srs],
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    transformed_points: list[tuple[float, float]] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        transformed_points.append((float(parts[0]), float(parts[1])))

    if len(transformed_points) != 4:
        raise RuntimeError(f"gdaltransform returned {len(transformed_points)} transformed corners")

    return bounds_from_points(transformed_points)


def transform_bbox_to_srs(bbox: GeoBBox, target_srs: str) -> tuple[float, float, float, float]:
    if target_srs.upper() in {"EPSG:4326", "OGC:CRS84"}:
        return (bbox.west, bbox.south, bbox.east, bbox.north)

    try:
        from osgeo import osr
    except ImportError:
        return transform_bbox_with_gdaltransform(bbox, target_srs)

    osr.UseExceptions()
    source = osr.SpatialReference()
    source.SetFromUserInput("EPSG:4326")
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    target = osr.SpatialReference()
    target.SetFromUserInput(target_srs)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform = osr.CoordinateTransformation(source, target)
    transformed_points = [
        (point[0], point[1])
        for point in (
            transform.TransformPoint(lon, lat)
            for lon, lat in bbox_corner_lon_lats(bbox)
        )
    ]
    return bounds_from_points(transformed_points)


def build_laz_to_dsm_pipeline(
    *,
    laz_paths: list[Path],
    output_tif: Path,
    source_srs: str,
    target_srs: str,
    target_bounds: tuple[float, float, float, float],
    resolution_m: float,
    vertical_scale: float,
) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for laz_path in laz_paths:
        stages.append(
            {
                "type": "readers.las",
                "filename": str(laz_path),
                "override_srs": source_srs,
            }
        )

    if len(laz_paths) > 1:
        stages.append({"type": "filters.merge"})

    stages.extend(
        [
            {
                "type": "filters.reprojection",
                "in_srs": source_srs,
                "out_srs": target_srs,
            },
            {
                "type": "filters.transformation",
                "matrix": transformation_matrix_for_z_scale(vertical_scale),
            },
            {
                "type": "writers.gdal",
                "filename": str(output_tif),
                "resolution": resolution_m,
                "output_type": "max",
                "data_type": "float32",
                "nodata": DEFAULT_NODATA,
                "bounds": pdal_bounds_string(target_bounds),
                "gdalopts": "COMPRESS=LZW,TILED=YES,BIGTIFF=IF_SAFER",
            },
        ]
    )
    return stages


def stage_laz_dsm(
    *,
    airport_code: str,
    usgs_root: Path,
    bbox: GeoBBox,
    laz_source_srs: str | None,
    target_srs: str | None,
    dsm_resolution_m: float,
    laz_vertical_scale: float,
    dry_run: bool,
) -> StagedTerrainSource:
    require_tool("pdal")

    normalized_airport = normalize_airport_code(airport_code)
    source_dir = source_root_for_airport(usgs_root, airport_code) / "dsm" / "source_laz"
    laz_paths = list_source_files(source_dir, (".laz", ".las"))
    if not laz_paths:
        raise FileNotFoundError(f"No LAZ/LAS files found in {source_dir}")

    resolved_source_srs = laz_source_srs or DEFAULT_LAZ_SOURCE_SRS_BY_AIRPORT.get(normalized_airport)
    if resolved_source_srs is None:
        raise ValueError(
            "The LAZ files do not have a reliable embedded SRS for this workflow. "
            "Pass --laz-source-srs, e.g. --laz-source-srs EPSG:2264 for KRDU."
        )

    resolved_target_srs = target_srs or default_target_srs_for_bbox(bbox)
    target_bounds = transform_bbox_to_srs(bbox, resolved_target_srs)

    output_dir = staging_dir_for_source(airport_code, "dsm")
    grid_label = numeric_label(dsm_resolution_m)
    output_tif = output_dir / f"usgs_tnm_lpc_dsm_grid_{grid_label}m_elevation_m.tif"
    pipeline_path = output_dir / f"usgs_tnm_lpc_dsm_grid_{grid_label}m_pipeline.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_laz_to_dsm_pipeline(
        laz_paths=laz_paths,
        output_tif=output_tif,
        source_srs=resolved_source_srs,
        target_srs=resolved_target_srs,
        target_bounds=target_bounds,
        resolution_m=dsm_resolution_m,
        vertical_scale=laz_vertical_scale,
    )
    pipeline_json = json.dumps({"pipeline": pipeline}, indent=2) + "\n"
    if dry_run:
        print(pipeline_json, end="")
    else:
        pipeline_path.write_text(pipeline_json, encoding="utf-8")

    run_command(["pdal", "pipeline", str(pipeline_path)], dry_run=dry_run, label="Rasterize LAZ DSM")
    return write_terrain_source_metadata(
        source=StagedTerrainSource(source_kind="dsm", source_dir=output_dir, output_tif=output_tif),
        horizontal_resolution_m=dsm_resolution_m,
        note="Configured PDAL writers.gdal raster resolution.",
        dry_run=dry_run,
    )


def build_heightmap_terrain(
    *,
    airport_code: str,
    source_dir: Path,
    dry_run: bool,
) -> None:
    require_tool("node")
    script_path = AEROVIZ_ROOT / "scripts" / "build_local_terrain_heightmap.mjs"
    run_command(
        [
            "node",
            str(script_path),
            "--airport",
            normalize_airport_code(airport_code),
            "--input-dir",
            str(source_dir),
        ],
        dry_run=dry_run,
        label="Build Cesium heightmap tiles",
    )


def available_source_kinds(airport_code: str, usgs_root: Path) -> list[SourceKind]:
    source_root = source_root_for_airport(usgs_root, airport_code)
    available: list[SourceKind] = []
    if list_source_files(source_root / "dem", (".tif", ".tiff")):
        available.append("dem")
    if list_source_files(source_root / "dsm" / "source_laz", (".laz", ".las")):
        available.append("dsm")
    return available


def source_kinds_to_stage(
    source: str,
    *,
    airport_code: str | None = None,
    usgs_root: Path | None = None,
) -> list[SourceKind]:
    if source == "auto":
        if airport_code is None or usgs_root is None:
            raise ValueError("source='auto' requires airport_code and usgs_root")
        available = available_source_kinds(airport_code, usgs_root)
        if not available:
            raise FileNotFoundError(
                f"No DEM GeoTIFF or DSM LAZ/LAS source data found for {normalize_airport_code(airport_code)}"
            )
        return available
    if source == "both":
        return ["dem", "dsm"]
    if source in {"dem", "dsm"}:
        return [source]  # type: ignore[list-item]
    raise ValueError(f"Unsupported source kind: {source}")


def preprocess(args: argparse.Namespace) -> dict[SourceKind, StagedTerrainSource]:
    airport_code = normalize_airport_code(args.airport)
    usgs_root = Path(args.usgs_root).expanduser().resolve()
    explicit_bbox = parse_bbox(args.bbox) if args.bbox else None
    bbox = resolve_processing_bbox(
        airport_code=airport_code,
        explicit_bbox=explicit_bbox,
        usgs_root=usgs_root,
        fallback_radius_km=args.fallback_radius_km,
    )
    staged: dict[SourceKind, StagedTerrainSource] = {}

    print(
        "Processing bbox "
        f"{bbox.west:.6f},{bbox.south:.6f},{bbox.east:.6f},{bbox.north:.6f}"
    )
    source_kinds = source_kinds_to_stage(
        args.source,
        airport_code=airport_code,
        usgs_root=usgs_root,
    )
    total_steps = len(source_kinds) + (0 if args.stage_only else 1)
    completed_steps = 0

    for source_kind in source_kinds:
        print_stage_progress(completed_steps, total_steps, f"Starting {source_kind.upper()} staging")
        if source_kind == "dem":
            staged[source_kind] = stage_dem_geotiff(
                airport_code=airport_code,
                usgs_root=usgs_root,
                bbox=bbox,
                dry_run=args.dry_run,
            )
        else:
            staged[source_kind] = stage_laz_dsm(
                airport_code=airport_code,
                usgs_root=usgs_root,
                bbox=bbox,
                laz_source_srs=args.laz_source_srs,
                target_srs=args.target_srs,
                dsm_resolution_m=args.dsm_resolution_m,
                laz_vertical_scale=args.laz_vertical_scale,
                dry_run=args.dry_run,
            )
        completed_steps += 1
        print_stage_progress(completed_steps, total_steps, f"Finished {source_kind.upper()} staging")

    publish_source = source_kinds[0]
    if args.source in {"auto", "both"}:
        publish_source = (
            select_highest_precision_source(staged)
            if args.publish_source == "auto"
            else args.publish_source
        )
    if not args.stage_only:
        print_stage_progress(
            completed_steps,
            total_steps,
            (
                f"Starting {publish_source.upper()} heightmap package "
                f"({staged[publish_source].horizontal_resolution_m:.3f} m precision)"
            ),
        )
        build_heightmap_terrain(
            airport_code=airport_code,
            source_dir=staged[publish_source].source_dir,
            dry_run=args.dry_run,
        )
        completed_steps += 1
        print_stage_progress(
            completed_steps,
            total_steps,
            f"Finished {publish_source.upper()} heightmap package",
        )

    return staged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess USGS TNM DEM/LAZ data into AeroViz local heightmap terrain.",
    )
    parser.add_argument("--airport", default=DEFAULT_AIRPORT_CODE, help="Airport ICAO code")
    parser.add_argument(
        "--usgs-root",
        default=str(DEFAULT_USGS_TNM_ROOT),
        help="Root containing <ICAO>/dem and <ICAO>/dsm/source_laz",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "dem", "dsm", "both"],
        default="auto",
        help=(
            "Source type to stage. The default auto stages all available source kinds "
            "and publishes the highest precision package."
        ),
    )
    parser.add_argument(
        "--publish-source",
        choices=["auto", "dem", "dsm"],
        default="auto",
        help=(
            "When --source both is used, choose which staged source becomes the active app terrain. "
            "The default auto publishes the smallest horizontalResolutionM."
        ),
    )
    parser.add_argument(
        "--bbox",
        help="Optional crop bbox as west,south,east,north in EPSG:4326. Defaults to manifest product_bbox union.",
    )
    parser.add_argument(
        "--fallback-radius-km",
        type=float,
        default=DEFAULT_FALLBACK_RADIUS_KM,
        help="Airport-centered radius if no bbox and no download manifest are available.",
    )
    parser.add_argument(
        "--laz-source-srs",
        help="Source SRS for LAZ files without embedded SRS. KRDU defaults to EPSG:2264.",
    )
    parser.add_argument(
        "--target-srs",
        help="Target projected SRS for LAZ rasterization. Defaults to the bbox UTM zone.",
    )
    parser.add_argument(
        "--dsm-resolution-m",
        type=float,
        default=DEFAULT_DSM_RESOLUTION_M,
        help="PDAL DSM raster resolution in target projected metres.",
    )
    parser.add_argument(
        "--laz-vertical-scale",
        type=float,
        default=DEFAULT_LAZ_VERTICAL_SCALE,
        help="Scale applied to LAZ Z values before heightmap generation. KRDU source uses feet, so default is 0.3048.",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Only write normalized GeoTIFF staging data; skip final .f32 terrain tile generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print GDAL/PDAL/Node commands without running them.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()
