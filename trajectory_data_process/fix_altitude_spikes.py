#!/usr/bin/env python
"""Detect and optionally repair one-sample altitude spikes in harvest tracks.

Purpose
-------
Observed trajectory rendering can show needle-like vertical peaks when one sample has an
extreme altitude outlier between two nominal neighbors. This utility finds those patterns
and can repair them deterministically.

Safety
------
Default mode is DRY-RUN (no file writes). Use --apply to persist edits.

What it scans
-------------
- Harvest records listed by tracks/manifest.json
- By default only outcome=assigned (the rendered observed roster)

What it writes in --apply mode
------------------------------
1) Per-track JSON: updates samples[idx][3] for repaired points and adds an
   altitude_outlier_repair audit block.
2) tracks/manifest.json: adds provenance.altitude_outlier_repair summary.

Usage
-----
Dry-run one airport:
  python trajectory_data_process/fix_altitude_spikes.py --airport KRDU

Dry-run all airports under harvest root:
  python trajectory_data_process/fix_altitude_spikes.py --all-airports

Apply changes:
  python trajectory_data_process/fix_altitude_spikes.py --airport KRDU --apply
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "altitude-outlier-repair-v1"
DEFAULT_JUMP_M = 150.0
DEFAULT_MAX_DT_S = 2.5
DEFAULT_SYMMETRY_RATIO = 0.35
DEFAULT_OUTCOME = "assigned"


@dataclass(frozen=True)
class RepairPoint:
    sample_index: int
    time_offset_s: float
    original_alt_m: float
    repaired_alt_m: float
    left_alt_m: float
    right_alt_m: float
    left_dt_s: float
    right_dt_s: float


@dataclass(frozen=True)
class TrackResult:
    file_path: Path
    flight_key: str
    outcome: str
    runway: str | None
    repair_points: tuple[RepairPoint, ...]


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round1(value: float) -> float:
    return round(float(value), 1)


def _detect_spikes(
    samples: list[list[Any]],
    *,
    jump_m: float,
    max_dt_s: float,
    symmetry_ratio: float,
) -> list[RepairPoint]:
    """Find one-sample spikes and propose repaired altitudes.

    Spike pattern at i:
    - i has neighbors i-1 and i+1
    - both adjacent jumps exceed jump_m
    - jump directions are opposite (up then down, or down then up)
    - both adjacent time gaps are <= max_dt_s
    - jump magnitudes are roughly symmetric: |abs(up)-abs(down)| / max(...) <= symmetry_ratio
    """
    points: list[RepairPoint] = []
    if len(samples) < 3:
        return points

    for i in range(1, len(samples) - 1):
        left = samples[i - 1]
        mid = samples[i]
        right = samples[i + 1]
        if len(left) < 4 or len(mid) < 4 or len(right) < 4:
            continue

        t0, a0 = left[0], left[3]
        t1, a1 = mid[0], mid[3]
        t2, a2 = right[0], right[3]
        if not all(_finite_number(v) for v in (t0, t1, t2, a0, a1, a2)):
            continue

        dt_left = float(t1) - float(t0)
        dt_right = float(t2) - float(t1)
        if dt_left <= 0.0 or dt_right <= 0.0:
            continue
        if dt_left > max_dt_s or dt_right > max_dt_s:
            continue

        up = float(a1) - float(a0)
        down = float(a2) - float(a1)
        abs_up = abs(up)
        abs_down = abs(down)
        if abs_up < jump_m or abs_down < jump_m:
            continue
        if up * down >= 0.0:
            continue

        max_jump = max(abs_up, abs_down)
        asym = abs(abs_up - abs_down) / max_jump if max_jump > 0.0 else 0.0
        if asym > symmetry_ratio:
            continue

        repaired = _round1((float(a0) + float(a2)) / 2.0)
        points.append(
            RepairPoint(
                sample_index=i,
                time_offset_s=float(t1),
                original_alt_m=float(a1),
                repaired_alt_m=repaired,
                left_alt_m=float(a0),
                right_alt_m=float(a2),
                left_dt_s=dt_left,
                right_dt_s=dt_right,
            )
        )

    return points


def _apply_track_repairs(payload: dict[str, Any], repairs: list[RepairPoint]) -> dict[str, Any]:
    updated = dict(payload)
    samples = updated.get("samples")
    if not isinstance(samples, list):
        raise ValueError("track payload does not contain a samples array")

    for point in repairs:
        samples[point.sample_index][3] = point.repaired_alt_m

    updated["samples"] = samples
    updated["altitude_outlier_repair"] = {
        "schema_version": SCRIPT_VERSION,
        "applied_utc": _iso_now(),
        "point_count": len(repairs),
        "points": [
            {
                "sample_index": p.sample_index,
                "time_offset_s": p.time_offset_s,
                "original_alt_m": p.original_alt_m,
                "repaired_alt_m": p.repaired_alt_m,
                "left_alt_m": p.left_alt_m,
                "right_alt_m": p.right_alt_m,
                "left_dt_s": p.left_dt_s,
                "right_dt_s": p.right_dt_s,
            }
            for p in repairs
        ],
    }
    return updated


def _write_json(path: Path, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        text = json.dumps(payload, indent=1, allow_nan=False) + "\n"
    else:
        text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    path.write_text(text, encoding="utf-8")


def _track_result_to_dict(result: TrackResult) -> dict[str, Any]:
    return {
        "file": str(result.file_path),
        "flight_key": result.flight_key,
        "outcome": result.outcome,
        "runway": result.runway,
        "repair_count": len(result.repair_points),
        "repair_points": [
            {
                "sample_index": point.sample_index,
                "time_offset_s": point.time_offset_s,
                "original_alt_m": point.original_alt_m,
                "repaired_alt_m": point.repaired_alt_m,
                "left_alt_m": point.left_alt_m,
                "right_alt_m": point.right_alt_m,
                "left_dt_s": point.left_dt_s,
                "right_dt_s": point.right_dt_s,
            }
            for point in result.repair_points
        ],
    }


def _write_report_json(
    path: Path,
    *,
    dry_run: bool,
    harvest_root: Path,
    jump_m: float,
    max_dt_s: float,
    symmetry_ratio: float,
    include_non_assigned: bool,
    outcome: str,
    airports: list[Path],
    all_results: list[dict[str, Any]],
    total_scanned: int,
    total_tracks: int,
    total_points: int,
) -> None:
    payload = {
        "schema_version": "altitude-outlier-repair-report-v1",
        "generated_utc": _iso_now(),
        "mode": "dry_run" if dry_run else "apply",
        "config": {
            "harvest_root": str(harvest_root),
            "airports": [p.name for p in airports],
            "jump_m": jump_m,
            "max_dt_s": max_dt_s,
            "symmetry_ratio": symmetry_ratio,
            "include_non_assigned": include_non_assigned,
            "outcome": outcome,
        },
        "summary": {
            "airports": len(all_results),
            "tracks_scanned": total_scanned,
            "tracks_with_spikes": total_tracks,
            "points_repaired": total_points,
        },
        "airports": [
            {
                "airport": result["airport"],
                "tracks_scanned": result["tracks_scanned"],
                "tracks_with_spikes": result["tracks_with_spikes"],
                "points_repaired": result["points_repaired"],
                "tracks": [_track_result_to_dict(item) for item in result["results"]],
            }
            for result in all_results
        ],
    }
    _write_json(path, payload, pretty=True)


def _process_airport(
    airport_root: Path,
    *,
    dry_run: bool,
    jump_m: float,
    max_dt_s: float,
    symmetry_ratio: float,
    outcome: str,
    include_non_assigned: bool,
) -> dict[str, Any]:
    manifest_path = airport_root / "tracks" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"invalid manifest records in {manifest_path}")

    candidates = []
    for row in records:
        if not isinstance(row, dict):
            continue
        row_outcome = row.get("outcome")
        if not include_non_assigned and row_outcome != outcome:
            continue
        rel = row.get("file")
        if not isinstance(rel, str):
            continue
        candidates.append((row, (airport_root / "tracks" / rel)))

    track_results: list[TrackResult] = []
    points_total = 0

    for row, path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        samples = payload.get("samples")
        if not isinstance(samples, list):
            continue

        repairs = _detect_spikes(
            samples,
            jump_m=jump_m,
            max_dt_s=max_dt_s,
            symmetry_ratio=symmetry_ratio,
        )
        if not repairs:
            continue

        result = TrackResult(
            file_path=path,
            flight_key=str(payload.get("flight_key") or row.get("flight_key") or path.stem),
            outcome=str(payload.get("outcome") or row.get("outcome") or "unknown"),
            runway=(str(payload.get("runway")) if payload.get("runway") is not None else None),
            repair_points=tuple(repairs),
        )
        track_results.append(result)
        points_total += len(repairs)

        if not dry_run:
            updated = _apply_track_repairs(payload, repairs)
            _write_json(path, updated, pretty=False)

    if not dry_run:
        provenance = manifest.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
            manifest["provenance"] = provenance
        provenance["altitude_outlier_repair"] = {
            "schema_version": SCRIPT_VERSION,
            "applied_utc": _iso_now(),
            "dry_run": False,
            "jump_m": jump_m,
            "max_dt_s": max_dt_s,
            "symmetry_ratio": symmetry_ratio,
            "tracks_repaired": len(track_results),
            "points_repaired": points_total,
            "scope_outcome": "all" if include_non_assigned else outcome,
        }
        manifest["written_utc"] = _iso_now()
        _write_json(manifest_path, manifest, pretty=True)

    return {
        "airport": airport_root.name,
        "tracks_scanned": len(candidates),
        "tracks_with_spikes": len(track_results),
        "points_repaired": points_total,
        "results": track_results,
    }


def _select_airports(harvest_root: Path, airport_codes: list[str] | None, all_airports: bool) -> list[Path]:
    if airport_codes and all_airports:
        raise ValueError("choose --airport or --all-airports, not both")

    if all_airports:
        return sorted(p for p in harvest_root.iterdir() if p.is_dir())

    if airport_codes:
        selected = []
        for code in airport_codes:
            path = harvest_root / code.upper()
            if not path.is_dir():
                raise FileNotFoundError(f"airport directory not found: {path}")
            selected.append(path)
        return selected

    detected = sorted(p for p in harvest_root.iterdir() if p.is_dir())
    if not detected:
        raise FileNotFoundError(f"no airport directories under {harvest_root}")
    return detected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect/repair one-sample altitude spikes in harvest tracks")
    parser.add_argument(
        "--harvest-root",
        default="trajectory_data_process/outputs/harvest",
        help="harvest root directory (default: trajectory_data_process/outputs/harvest)",
    )
    parser.add_argument(
        "--airport",
        action="append",
        dest="airports",
        help="airport ICAO to process (repeatable)",
    )
    parser.add_argument(
        "--all-airports",
        action="store_true",
        help="process every airport under harvest root",
    )
    parser.add_argument(
        "--jump-m",
        type=float,
        default=DEFAULT_JUMP_M,
        help=f"minimum adjacent altitude jump to consider a spike (default: {DEFAULT_JUMP_M})",
    )
    parser.add_argument(
        "--max-dt-s",
        type=float,
        default=DEFAULT_MAX_DT_S,
        help=f"max adjacent sample gap for spike detection (default: {DEFAULT_MAX_DT_S})",
    )
    parser.add_argument(
        "--symmetry-ratio",
        type=float,
        default=DEFAULT_SYMMETRY_RATIO,
        help=(
            "max normalized asymmetry between the up/down jump magnitudes "
            f"(default: {DEFAULT_SYMMETRY_RATIO})"
        ),
    )
    parser.add_argument(
        "--include-non-assigned",
        action="store_true",
        help="scan all manifest outcomes (default scans assigned only)",
    )
    parser.add_argument(
        "--outcome",
        default=DEFAULT_OUTCOME,
        help=f"target outcome when --include-non-assigned is not set (default: {DEFAULT_OUTCOME})",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="optional path to write a structured JSON report (works in dry-run and apply)",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="detect only, do not write files (default mode)",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="write repaired samples and audit metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = not args.apply

    harvest_root = Path(args.harvest_root).resolve()
    if not harvest_root.is_dir():
        raise FileNotFoundError(f"harvest root not found: {harvest_root}")

    airports = _select_airports(harvest_root, args.airports, args.all_airports)

    print(
        "mode=DRY_RUN" if dry_run else "mode=APPLY",
        f"harvest_root={harvest_root}",
        f"airports={','.join(p.name for p in airports)}",
        f"jump_m={args.jump_m}",
        f"max_dt_s={args.max_dt_s}",
        f"symmetry_ratio={args.symmetry_ratio}",
    )

    all_results: list[dict[str, Any]] = []
    total_scanned = 0
    total_tracks = 0
    total_points = 0

    for airport_root in airports:
        result = _process_airport(
            airport_root,
            dry_run=dry_run,
            jump_m=float(args.jump_m),
            max_dt_s=float(args.max_dt_s),
            symmetry_ratio=float(args.symmetry_ratio),
            outcome=str(args.outcome),
            include_non_assigned=bool(args.include_non_assigned),
        )
        all_results.append(result)
        total_scanned += int(result["tracks_scanned"])
        total_tracks += int(result["tracks_with_spikes"])
        total_points += int(result["points_repaired"])

        print(
            f"[{result['airport']}] scanned={result['tracks_scanned']} "
            f"tracks_with_spikes={result['tracks_with_spikes']} points={result['points_repaired']}"
        )

        for item in result["results"][:5]:
            print(
                f"  - {item.flight_key} runway={item.runway} "
                f"repairs={len(item.repair_points)} file={item.file_path.name}"
            )
        if len(result["results"]) > 5:
            print(f"  ... +{len(result['results']) - 5} more tracks")

    print("SUMMARY")
    print(
        f"airports={len(all_results)} scanned={total_scanned} "
        f"tracks_with_spikes={total_tracks} points={total_points} dry_run={dry_run}"
    )

    if args.report_json:
        report_path = Path(args.report_json)
        if not report_path.is_absolute():
            report_path = Path.cwd() / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_report_json(
            report_path,
            dry_run=dry_run,
            harvest_root=harvest_root,
            jump_m=float(args.jump_m),
            max_dt_s=float(args.max_dt_s),
            symmetry_ratio=float(args.symmetry_ratio),
            include_non_assigned=bool(args.include_non_assigned),
            outcome=str(args.outcome),
            airports=airports,
            all_results=all_results,
            total_scanned=total_scanned,
            total_tracks=total_tracks,
            total_points=total_points,
        )
        print(f"report_json={report_path}")


if __name__ == "__main__":
    main()
