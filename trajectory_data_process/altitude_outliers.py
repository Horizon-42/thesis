"""Audit ADS-B altitude outliers, and republish the observed CZML without them.

This tool NEVER writes to ``tracks/``. The repair itself lives in
``trajectory_data_process.harvest.altitude_filter`` and is applied where a stored track is
read into a derived view, so the observed CZML, the model-ready arrival slices and the
evaluation records are all filtered by construction. What is left for a CLI is the two
things that layer cannot do for itself: say what it is repairing, and rebuild the
artifacts that were published before it existed.

    # what the filter would replace, per airport (read-only, the default);
    # no --airport means every harvested airport
    python -m trajectory_data_process.altitude_outliers --airport KRDU
    python -m trajectory_data_process.altitude_outliers \
        --report-json outputs/altitude-outliers.json

    # republish public/data/<ICAO>/trajectories.czml from the unmodified tracks
    python -m trajectory_data_process.altitude_outliers --airport KRDU --rerender-czml

WHAT --rerender-czml DOES AND DOES NOT COVER
--------------------------------------------
It runs the pipeline's own renderer, so the entity ids, packet shape and clock cannot
drift from a full harvest. Batch comparison CZMLs resolve their white observed reference
by entity id inside this same canonical file, so they follow it without being rebuilt.

Training data needs NO rebuild either: ``load_arrival_flights`` filters on the way out, so
the next dataset build is already clean. What a rebuild buys is the roster's own count of
repaired samples plus refreshed evaluation records --
``python -m trajectory_data_process.harvest --airport <ICAO> --evaluate-only``.

The stored ``observed_threshold_event`` is the one thing still derived from raw samples:
it was fitted during assignment, before this filter existed. Outliers reported inside an
event's source range are listed by ``--report-json`` so the case can be judged; clearing
them needs ``--reclassify-existing``, which re-derives assignment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trajectory_data_process.harvest.altitude_filter import (
    DEFAULT_POLICY,
    FILTER_SCHEMA_VERSION,
    AltitudeOutlier,
    AltitudePolicy,
    filter_altitude_outliers,
)
from trajectory_data_process.harvest.arrivals import ARRIVALS_DIR, MANIFEST_NAME
from trajectory_data_process.harvest.czml import DEFAULT_FRONTEND_DATA, render_observed_czml
from trajectory_data_process.harvest.store import HarvestPaths, read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARVEST_ROOT = REPO_ROOT / "trajectory_data_process/outputs/harvest"
REPORT_SCHEMA_VERSION = "altitude-outlier-report-v1"


@dataclass(frozen=True)
class TrackAudit:
    """One track's outliers, and where they land in the views that consume it."""

    flight_key: str
    file: str
    outcome: str
    runway: str | None
    samples: int
    outliers: tuple[AltitudeOutlier, ...]
    in_arrival_slice: int
    in_threshold_event_fit: int
    longest_run: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "flight_key": self.flight_key,
            "file": self.file,
            "outcome": self.outcome,
            "runway": self.runway,
            "samples": self.samples,
            "outlier_count": len(self.outliers),
            "in_arrival_slice": self.in_arrival_slice,
            "in_threshold_event_fit": self.in_threshold_event_fit,
            "longest_run": self.longest_run,
            "outliers": [outlier.to_dict() for outlier in self.outliers],
        }


@dataclass(frozen=True)
class AirportAudit:
    code: str
    tracks_scanned: int
    samples_scanned: int
    tracks: tuple[TrackAudit, ...]

    @property
    def outliers(self) -> int:
        return sum(len(track.outliers) for track in self.tracks)

    @property
    def in_arrival_slice(self) -> int:
        return sum(track.in_arrival_slice for track in self.tracks)

    @property
    def in_threshold_event_fit(self) -> int:
        return sum(track.in_threshold_event_fit for track in self.tracks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "airport": self.code,
            "tracks_scanned": self.tracks_scanned,
            "samples_scanned": self.samples_scanned,
            "tracks_with_outliers": len(self.tracks),
            "outliers": self.outliers,
            "in_arrival_slice": self.in_arrival_slice,
            "in_threshold_event_fit": self.in_threshold_event_fit,
            "tracks": [track.to_dict() for track in self.tracks],
        }


def audit_airport(
    paths: HarvestPaths,
    *,
    outcomes: tuple[str, ...],
    policy: AltitudePolicy = DEFAULT_POLICY,
) -> AirportAudit:
    """Run detection over one airport's stored tracks. Reads only."""
    slices = _arrival_slices(paths)
    scanned = 0
    samples_scanned = 0
    audited: list[TrackAudit] = []

    for row in read_manifest(paths)["records"]:
        if row["outcome"] not in outcomes:
            continue
        track = json.loads((paths.tracks / row["file"]).read_text(encoding="utf-8"))
        samples = track["samples"]
        scanned += 1
        samples_scanned += len(samples)
        outliers = filter_altitude_outliers(samples, policy=policy).outliers
        if not outliers:
            continue
        indices = [outlier.index for outlier in outliers]
        first, last = slices.get(row["flight_key"], (None, None))
        fit_range = _event_fit_range(track)
        audited.append(
            TrackAudit(
                flight_key=row["flight_key"],
                file=row["file"],
                outcome=row["outcome"],
                runway=row["runway"],
                samples=len(samples),
                outliers=outliers,
                in_arrival_slice=(
                    0 if first is None else sum(first <= i <= last for i in indices)
                ),
                in_threshold_event_fit=(
                    0
                    if fit_range is None
                    else sum(fit_range[0] <= i <= fit_range[1] for i in indices)
                ),
                longest_run=_longest_run(indices),
            )
        )

    return AirportAudit(
        code=paths.code,
        tracks_scanned=scanned,
        samples_scanned=samples_scanned,
        tracks=tuple(audited),
    )


def _arrival_slices(paths: HarvestPaths) -> dict[str, tuple[int, int]]:
    """The model-input slice per flight, when an arrival roster has been built."""
    manifest = paths.airport / ARRIVALS_DIR / MANIFEST_NAME
    if not manifest.exists():
        return {}
    return {
        row["flight_key"]: (row["first_sample_index"], row["last_sample_index"])
        for row in json.loads(manifest.read_text(encoding="utf-8"))["records"]
    }


def _event_fit_range(track: dict[str, Any]) -> tuple[int, int] | None:
    """The sample range the stored threshold event was fitted from, when it has one."""
    event = track.get("observed_threshold_event")
    if not isinstance(event, dict):
        return None
    fit_range = event.get("source_sample_range")
    if not isinstance(fit_range, list) or len(fit_range) != 2:
        return None
    return int(fit_range[0]), int(fit_range[1])


def _longest_run(indices: list[int]) -> int:
    longest = 1
    run = 1
    for previous, current in zip(indices, indices[1:]):
        run = run + 1 if current == previous + 1 else 1
        longest = max(longest, run)
    return longest


def _select_airports(harvest_root: Path, codes: list[str] | None) -> list[str]:
    if codes:
        selected = []
        for code in codes:
            upper = code.upper()
            if not (harvest_root / upper).is_dir():
                raise SystemExit(f"no harvest for {upper} under {harvest_root}")
            selected.append(upper)
        return selected
    detected = sorted(p.name for p in harvest_root.iterdir() if p.is_dir())
    if not detected:
        raise SystemExit(f"no airport directories under {harvest_root}")
    return detected


def _report(audits: list[AirportAudit], *, policy: AltitudePolicy, outcomes: tuple[str, ...],
            harvest_root: Path) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "filter_schema_version": FILTER_SCHEMA_VERSION,
        "generated_utc": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "harvest_root": str(harvest_root),
        "policy": policy.to_dict(),
        "outcomes": list(outcomes),
        "summary": {
            "airports": len(audits),
            "tracks_scanned": sum(a.tracks_scanned for a in audits),
            "samples_scanned": sum(a.samples_scanned for a in audits),
            "tracks_with_outliers": sum(len(a.tracks) for a in audits),
            "outliers": sum(a.outliers for a in audits),
            "in_arrival_slice": sum(a.in_arrival_slice for a in audits),
            "in_threshold_event_fit": sum(a.in_threshold_event_fit for a in audits),
        },
        "airports": [audit.to_dict() for audit in audits],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trajectory_data_process.altitude_outliers",
        description="Audit altitude outliers; optionally republish the observed CZML.",
    )
    parser.add_argument("--airport", action="append", dest="airports",
                        help="ICAO code (repeatable); default is every harvested airport")
    parser.add_argument("--harvest-root", type=Path, default=DEFAULT_HARVEST_ROOT)
    parser.add_argument("--outcome", action="append", dest="outcomes",
                        help="track outcomes to scan (repeatable, default: assigned — the "
                             "only bucket any derived view is built from)")
    parser.add_argument("--min-deviation-m", type=float,
                        default=DEFAULT_POLICY.min_deviation_m)
    parser.add_argument("--max-vertical-rate-m-s", type=float,
                        default=DEFAULT_POLICY.max_vertical_rate_m_s)
    parser.add_argument("--half-window", type=int, default=DEFAULT_POLICY.half_window)
    parser.add_argument("--report-json", type=Path,
                        help="write the full per-track audit to this path")
    parser.add_argument("--rerender-czml", action="store_true",
                        help="republish public/data/<ICAO>/trajectories.czml through the "
                             "filter; stored tracks are still never modified")
    parser.add_argument("--frontend-data", type=Path, default=DEFAULT_FRONTEND_DATA)
    parser.add_argument("--multiplier", type=int, default=None,
                        help="optional CZML clock multiplier")
    parser.add_argument("--top", type=int, default=5,
                        help="worst tracks to print per airport (default: 5)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    harvest_root = args.harvest_root.resolve()
    if not harvest_root.is_dir():
        raise SystemExit(f"harvest root not found: {harvest_root}")

    policy = AltitudePolicy(
        half_window=args.half_window,
        min_deviation_m=args.min_deviation_m,
        max_vertical_rate_m_s=args.max_vertical_rate_m_s,
    )
    outcomes = tuple(args.outcomes) if args.outcomes else ("assigned",)
    codes = _select_airports(harvest_root, args.airports)

    print(f"harvest_root={harvest_root} airports={','.join(codes)} outcomes={','.join(outcomes)}")
    print(f"policy={policy.to_dict()}  (tracks/ is read-only in every mode)")

    audits: list[AirportAudit] = []
    for code in codes:
        paths = HarvestPaths(root=harvest_root, code=code)
        audit = audit_airport(paths, outcomes=outcomes, policy=policy)
        audits.append(audit)
        print(
            f"\n[{code}] scanned {audit.tracks_scanned} tracks / {audit.samples_scanned} "
            f"samples — {audit.outliers} outlier(s) in {len(audit.tracks)} track(s); "
            f"{audit.in_arrival_slice} inside a model arrival slice, "
            f"{audit.in_threshold_event_fit} inside a stored threshold-event fit"
        )
        for track in sorted(audit.tracks, key=lambda t: len(t.outliers), reverse=True)[: args.top]:
            worst = max(track.outliers, key=lambda o: abs(o.correction_m))
            print(
                f"   {track.flight_key} rwy={track.runway} "
                f"outliers={len(track.outliers)} longest_run={track.longest_run} "
                f"worst={worst.observed_alt_m:.0f} m -> {worst.replacement_alt_m:.0f} m "
                f"at t={worst.time_offset_s:.1f} s"
            )
        if len(audit.tracks) > args.top:
            print(f"   … +{len(audit.tracks) - args.top} more track(s)")

        if args.rerender_czml:
            rendered = render_observed_czml(
                paths, frontend_data_root=args.frontend_data, multiplier=args.multiplier
            )
            print(
                f"   rerendered {rendered.flights} flights -> {rendered.combined_czml} "
                f"({rendered.altitude_outliers} altitude(s) replaced in "
                f"{rendered.flights_with_altitude_outliers} flight(s))"
            )

    total = sum(a.outliers for a in audits)
    print(
        f"\nSUMMARY airports={len(audits)} "
        f"tracks={sum(a.tracks_scanned for a in audits)} "
        f"tracks_with_outliers={sum(len(a.tracks) for a in audits)} outliers={total}"
    )

    if args.report_json:
        path = args.report_json if args.report_json.is_absolute() else Path.cwd() / args.report_json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                _report(audits, policy=policy, outcomes=outcomes, harvest_root=harvest_root),
                indent=1,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        print(f"report_json={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
