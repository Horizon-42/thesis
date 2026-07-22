"""CLI: harvest one airport, then judge its observed approaches.

    python -m trajectory_data_process.harvest --airport KRDU --count 200
    python -m trajectory_data_process.harvest --airport KRDU --evaluate-only

The measured ``tracks/`` roster is re-derived only by downloading. ``arrivals/`` and
``approach/`` are pure derived views, so ``--evaluate-only`` rebuilds both from the stored
track manifest after a crop, CIFP, fit, or criteria change.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from evaluation.metrics import evaluate_batch

from trajectory_data_process.acquisition.opensky_history import install_query_cancel_on_interrupt
from trajectory_data_process.arrival_segment import ENTRY_RADIUS_KM
from trajectory_data_process.harvest.airports import load_airport
from trajectory_data_process.harvest.arrivals import arrival_manifest_path, write_arrival_records
from trajectory_data_process.harvest.observed import (
    REPORT_NAME,
    load_observed_records,
    write_observed_records,
)
from trajectory_data_process.harvest.czml import render_observed_czml
from trajectory_data_process.harvest.publish import publish_observed_report
from trajectory_data_process.harvest.runner import (
    HarvestPlan,
    checkpoint_start,
    clear_harvest_checkpoint,
    harvest_airport,
)
from trajectory_data_process.harvest.store import HarvestPaths, read_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "trajectory_data_process/config/runway_thresholds.json"
DEFAULT_OUTPUT = REPO_ROOT / "trajectory_data_process/outputs/harvest"
DEFAULT_CIFP = REPO_ROOT / "data/CIFP/CIFP_260319/FAACIFP18"
DEFAULT_FRONTEND_DATA = REPO_ROOT / "aeroviz-4d/public/data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trajectory_data_process.harvest")
    parser.add_argument("--airport", required=True, help="ICAO code, e.g. KRDU")
    parser.add_argument("--count", type=int, default=200,
                        help="assigned landings wanted per runway (default: 200)")
    parser.add_argument("--start", help="ISO UTC instant to scan backward from (default: now)")
    parser.add_argument("--max-lookback-days", type=float, default=30.0)
    parser.add_argument("--chunk-hours", type=float, default=6.0)
    parser.add_argument("--radius-km", type=float, default=30.0)
    parser.add_argument("--entry-radius-km", type=float, default=ENTRY_RADIUS_KM,
                        help="terminal-entry radius for the model-ready arrival dataset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cifp", type=Path, default=DEFAULT_CIFP,
                        help="ARINC 424 CIFP file supplying per-runway published TCH")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="skip download; rebuild arrivals/, approach/, and publication")
    parser.add_argument("--no-cache", action="store_true", help="bypass the history query cache")
    parser.add_argument(
        "--full-redownload",
        action="store_true",
        help="ignore a stored download start and bypass the OpenSky query cache",
    )
    parser.add_argument("--frontend-data", type=Path, default=DEFAULT_FRONTEND_DATA,
                        help="publish the report into this public/data tree as the "
                             "'observed' comparison category")
    parser.add_argument("--no-publish", action="store_true",
                        help="skip publishing to the frontend")
    parser.add_argument("--no-czml", action="store_true",
                        help="skip rendering the observed CZML layer (report only)")
    parser.add_argument("--multiplier", type=int, default=None,
                        help="optional CZML clock multiplier")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.entry_radius_km < args.radius_km:
        raise SystemExit(
            f"--entry-radius-km must be in (0, --radius-km), got "
            f"{args.entry_radius_km:g} vs {args.radius_km:g}"
        )
    code = args.airport.upper()
    paths = HarvestPaths(root=args.output, code=code)
    if not args.evaluate_only and not args.full_redownload:
        completed = _completed_download_manifest(
            paths,
            expected_runways=_configured_runways(args.config, code),
            target_per_runway=args.count,
            radius_km=args.radius_km,
            requested_start=args.start,
        )
        if completed is not None:
            clear_harvest_checkpoint(paths)
            print(
                f"[harvest] {code}: completed tracks already exist; skipping "
                f"({completed['per_runway']}, given up "
                f"{completed['provenance'].get('given_up', [])}). "
                "Use --full-redownload to download it again."
            )
            return 0
    if args.full_redownload:
        clear_harvest_checkpoint(paths)

    airport = load_airport(code, config_file=args.config, cifp_file=args.cifp)

    if args.evaluate_only:
        manifest = read_manifest(paths)
        print(f"[harvest] reusing stored tracks: {manifest['counts']}")
    else:
        install_query_cancel_on_interrupt()
        download_start, cached = _resolve_download_options(
            requested_start=args.start,
            paths=paths,
            full_redownload=args.full_redownload,
            no_cache=args.no_cache,
        )
        result = harvest_airport(
            airport,
            paths,
            HarvestPlan(
                target_per_runway=args.count,
                start=download_start,
                chunk_hours=args.chunk_hours,
                max_lookback_days=args.max_lookback_days,
                radius_km=args.radius_km,
                cached=cached,
            ),
        )
        manifest = result.manifest
        print(f"[harvest] {args.airport}: {manifest['counts']} over "
              f"{result.chunks_fetched} chunks")

    arrivals = write_arrival_records(
        airport, paths, entry_radius_km=args.entry_radius_km
    )
    print(
        f"[harvest] model-ready arrivals: {arrivals['counts']} -> "
        f"{arrival_manifest_path(paths)}"
    )

    summary = write_observed_records(airport, paths)
    records = load_observed_records(paths)
    report = evaluate_batch(records)
    (paths.approach / REPORT_NAME).write_text(json.dumps(report, indent=1), encoding="utf-8")

    if not args.no_czml:
        rendered = render_observed_czml(
            paths, frontend_data_root=args.frontend_data, multiplier=args.multiplier
        )
        print(f"[harvest] observed CZML: {rendered.flights} flights over "
              f"{len(rendered.runway_czml)} runway(s) -> {rendered.combined_czml}")

    if not args.no_publish:
        published = publish_observed_report(
            report, frontend_data_root=args.frontend_data, airport=code
        )
        print(f"[harvest] published observed category -> {published}")

    _print_digest(code, manifest, summary, report)
    return 0


def _print_digest(code: str, manifest: dict, summary: dict, report: dict) -> None:
    counts = manifest["counts"]
    print(f"\n{code} harvest")
    print(f"  tracks      : {manifest['total']} total — "
          + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"  per runway  : {manifest['per_runway']}")
    if summary["skipped"]:
        print(f"  skipped     : {len(summary['skipped'])} (no published LPV TCH)")
    observed = report.get("observed", {})
    if observed:
        print(f"  established : {observed['established']}/{summary['total']} "
              f"({observed['established_rate']:.0%})  "
              f"not established {observed['not_established']}")
        print(f"  gates       : {report['successful']} pass of {report['measured']} measured "
              f"— {observed['marginal']} marginal (the data cannot decide)")
    if report.get("lateral_m"):
        print(f"  lateral  m  : {report['lateral_m']}")
        print(f"  vertical m  : {report['vertical_m']}")


def _parse_start(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _configured_runways(config_path: Path, code: str) -> set[str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        str(threshold["ident"])
        for runway in config["airports"][code]["runways"]
        for threshold in runway["thresholds"]
    }


def _completed_download_manifest(
    paths: HarvestPaths,
    *,
    expected_runways: set[str],
    target_per_runway: int,
    radius_km: float,
    requested_start: str | None,
) -> dict | None:
    """Return a compatible completed manifest, including deliberate give-ups."""
    try:
        manifest = read_manifest(paths)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    provenance = manifest.get("provenance")
    per_runway = manifest.get("per_runway")
    records = manifest.get("records")
    if not isinstance(provenance, dict) or not isinstance(per_runway, dict):
        return None
    if not isinstance(records, list) or manifest.get("total") != len(records):
        return None

    try:
        stored_radius = float(provenance["radius_km"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isclose(stored_radius, radius_km, rel_tol=0.0, abs_tol=1e-9):
        return None

    if requested_start is not None:
        value = provenance.get("start_utc") or provenance.get("scanned_to_utc")
        if not isinstance(value, str):
            return None
        try:
            if _parse_start(value) != _parse_start(requested_start):
                return None
        except ValueError:
            return None

    given_up_value = provenance.get("given_up", [])
    if not isinstance(given_up_value, list):
        return None
    given_up = {str(ident) for ident in given_up_value}

    def runway_complete(ident: str) -> bool:
        try:
            count = int(per_runway.get(ident, 0))
        except (TypeError, ValueError):
            return False
        return count >= target_per_runway or ident in given_up

    return manifest if expected_runways and all(map(runway_complete, expected_runways)) else None


def _resolve_download_options(
    *,
    requested_start: str | None,
    paths: HarvestPaths,
    full_redownload: bool,
    no_cache: bool,
    log=print,
) -> tuple[datetime | None, bool]:
    """Choose a stable scan anchor and whether pyopensky may use its query cache.

    An identical start gives the backward scanner identical chunk boundaries, which is
    required for pyopensky's per-query cache to match a previous harvest. Explicit CLI
    input always wins. A full redownload deliberately returns to the old ``now`` anchor
    and bypasses that cache.
    """
    explicit = _parse_start(requested_start)
    cached = not (no_cache or full_redownload)
    if explicit is not None:
        return explicit, cached
    if full_redownload:
        log("[harvest] full redownload: using the current time and bypassing OpenSky cache")
        return None, False

    stored = _stored_download_start(paths)
    if stored is not None:
        log(
            f"[harvest] reusing previous download start {stored.isoformat()} "
            "to match OpenSky cache entries"
        )
    return stored, cached


def _stored_download_start(paths: HarvestPaths) -> datetime | None:
    """Read an active checkpoint anchor, then fall back to the completed manifest."""
    active_start = checkpoint_start(paths)
    if active_start is not None:
        return active_start
    try:
        manifest = read_manifest(paths)
    except FileNotFoundError:
        return None
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        return None
    value = provenance.get("start_utc") or provenance.get("scanned_to_utc")
    if not isinstance(value, str) or not value:
        return None
    try:
        return _parse_start(value)
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
