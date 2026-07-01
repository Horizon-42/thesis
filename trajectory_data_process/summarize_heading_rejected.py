#!/usr/bin/env python3
"""Summarize the heading-rejected landings for false-kill review.

``download_landings.py`` keeps every landing whose approach direction disagrees with
the runway heading in a parallel ``{CODE}_{ident}_heading_rejected.json`` file (tagged
with the measured errors). This reads them all and prints, per metric:

* an ASCII histogram of the direction error (degrees),
* summary stats (count / median / mean / max, and how many are *borderline* — only
  just past the tolerance), and
* the key false-kill signal — landings whose **geometry** says they were aligned and
  only the (noisy) ADS-B track disagreed (``geometry_ok_track_bad``): those are the
  most likely to be real landings we discarded.

    python trajectory_data_process/summarize_heading_rejected.py
    python trajectory_data_process/summarize_heading_rejected.py --airports KRDU --metric course
    python trajectory_data_process/summarize_heading_rejected.py --bin-width 5

No plotting dependencies — the histogram is text, so it runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":  # pragma: no cover - direct execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TOLERANCE_DEG = 20.0
BORDERLINE_MARGIN_DEG = 10.0  # error within [tol, tol+margin) counts as "just past" the gate.
MAX_ERROR_DEG = 180.0
METRICS = ("course", "track", "max")


def find_rejected_files(output_root: Path, airports: set[str] | None = None) -> list[Path]:
    """Every ``*_heading_rejected.json`` under ``output_root`` (optionally per airport)."""
    files = sorted(output_root.glob("*/*_heading_rejected.json"))
    if airports:
        files = [p for p in files if p.parent.name.upper() in airports]
    return files


def load_rejected(files: list[Path]) -> list[dict[str, Any]]:
    """Load the flight records from the given files, tagging each with its source."""
    records: list[dict[str, Any]] = []
    for path in files:
        for flight in json.loads(path.read_text(encoding="utf-8")):
            flight["_airport"] = path.parent.name.upper()
            records.append(flight)
    return records


def metric_error(record: dict[str, Any], metric: str) -> float | None:
    """The chosen direction error for one record, or None when unavailable."""
    course = record.get("course_error_deg")
    track = record.get("track_error_deg")
    if metric == "course":
        return course
    if metric == "track":
        return track
    available = [e for e in (course, track) if e is not None]  # metric == "max"
    return max(available) if available else None


def histogram(values: list[float], bin_width: float, max_deg: float = MAX_ERROR_DEG) -> list[tuple[float, float, int]]:
    """Bin ``values`` into ``[lo, hi)`` buckets over ``[0, max_deg]``."""
    n_bins = max(1, int(round(max_deg / bin_width)))
    counts = [0] * n_bins
    for value in values:
        idx = min(int(value // bin_width), n_bins - 1)
        counts[max(idx, 0)] += 1
    return [(i * bin_width, min((i + 1) * bin_width, max_deg), counts[i]) for i in range(n_bins)]


def false_kill_signals(records: list[dict[str, Any]]) -> dict[str, int]:
    """Cross-tabulate why each rejection happened, per its own tolerance.

    ``geometry_ok_track_bad`` is the headline: geometry says the approach WAS aligned
    and only the ADS-B track disagreed — the likeliest real-landing false kills.
    """
    signals = {
        "geometry_ok_track_bad": 0,  # course within tol, track past it -> probably a real landing
        "track_ok_geometry_bad": 0,  # track within tol, course past it
        "both_bad": 0,               # both directions disagree
        "opposite_end": 0,           # course error > 150 deg -> likely the wrong runway end
    }
    for record in records:
        tol = record.get("heading_tolerance_deg") or DEFAULT_TOLERANCE_DEG
        course = record.get("course_error_deg")
        track = record.get("track_error_deg")
        course_bad = course is not None and course > tol
        track_bad = track is not None and track > tol
        if course is not None and not course_bad and track_bad:
            signals["geometry_ok_track_bad"] += 1
        elif track is not None and not track_bad and course_bad:
            signals["track_ok_geometry_bad"] += 1
        elif course_bad and track_bad:
            signals["both_bad"] += 1
        if course is not None and course > 150.0:
            signals["opposite_end"] += 1
    return signals


def summarize(records: list[dict[str, Any]], metric: str, bin_width: float) -> dict[str, Any]:
    """Build the full summary payload for ``format_report``."""
    errors = [e for r in records if (e := metric_error(r, metric)) is not None]
    tolerances = [r.get("heading_tolerance_deg") or DEFAULT_TOLERANCE_DEG for r in records]
    tol = statistics.median(tolerances) if tolerances else DEFAULT_TOLERANCE_DEG
    borderline = sum(1 for e in errors if tol <= e < tol + BORDERLINE_MARGIN_DEG)

    per_runway: dict[str, int] = {}
    for record in records:
        key = f"{record.get('_airport', '?')} {record.get('runway') or '?'}"
        per_runway[key] = per_runway.get(key, 0) + 1

    return {
        "metric": metric,
        "total": len(records),
        "with_error": len(errors),
        "tolerance_deg": tol,
        "borderline": borderline,
        "stats": {
            "min": min(errors) if errors else None,
            "median": statistics.median(errors) if errors else None,
            "mean": statistics.fmean(errors) if errors else None,
            "max": max(errors) if errors else None,
        },
        "histogram": histogram(errors, bin_width),
        "signals": false_kill_signals(records),
        "per_runway": dict(sorted(per_runway.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def format_report(summary: dict[str, Any], bar_width: int = 40) -> str:
    """Render the summary as a text report (ASCII histogram + tables)."""
    lines: list[str] = []
    total, with_error = summary["total"], summary["with_error"]
    lines.append(f"Heading-rejected landings: {total}  (metric = {summary['metric']}_error_deg, "
                 f"tolerance ≈ {summary['tolerance_deg']:.0f}°)")
    if total == 0:
        lines.append("  (no heading-rejected files found — nothing to summarize)")
        return "\n".join(lines)

    s = summary["stats"]
    lines.append(f"  error  min {s['min']:.1f}°  median {s['median']:.1f}°  "
                 f"mean {s['mean']:.1f}°  max {s['max']:.1f}°   "
                 f"(borderline within +{BORDERLINE_MARGIN_DEG:.0f}°: {summary['borderline']})")
    if with_error < total:
        lines.append(f"  ({total - with_error} record(s) had no {summary['metric']} error and were skipped)")

    lines.append("")
    lines.append(f"Error histogram ({summary['metric']}_error_deg):")
    peak = max((c for _lo, _hi, c in summary["histogram"]), default=0)
    for lo, hi, count in summary["histogram"]:
        if count == 0:
            continue
        bar = "█" * max(1, round(bar_width * count / peak)) if peak else ""
        lines.append(f"  [{lo:5.0f}, {hi:5.0f})°  {count:5d}  {bar}")

    sig = summary["signals"]
    lines.append("")
    lines.append("False-kill signals:")
    lines.append(f"  geometry_ok_track_bad : {sig['geometry_ok_track_bad']:5d}  "
                 "(geometry aligned, only ADS-B track disagreed — likely real landings)")
    lines.append(f"  track_ok_geometry_bad : {sig['track_ok_geometry_bad']:5d}  "
                 "(ADS-B track aligned, geometry disagreed)")
    lines.append(f"  both_bad              : {sig['both_bad']:5d}  (both directions disagree)")
    lines.append(f"  opposite_end          : {sig['opposite_end']:5d}  (course > 150° — wrong runway end)")

    lines.append("")
    lines.append("Per runway:")
    for key, count in summary["per_runway"].items():
        lines.append(f"  {key:<12} {count}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Summarize heading-rejected landings for false-kill review")
    p.add_argument("--output-root", default=str(here / "outputs" / "landings"),
                   help="Where the *_heading_rejected.json live")
    p.add_argument("--airports", nargs="+", default=None, help="Subset of airport codes (default: all found)")
    p.add_argument("--metric", choices=METRICS, default="course",
                   help="Which direction error to histogram (default: course = the robust geometric one)")
    p.add_argument("--bin-width", type=float, default=10.0, help="Histogram bin width in degrees")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    airports = {a.upper() for a in args.airports} if args.airports else None
    files = find_rejected_files(Path(args.output_root), airports)
    records = load_rejected(files)
    print(f"[rejected] scanned {len(files)} file(s) under {args.output_root}")
    print(format_report(summarize(records, args.metric, args.bin_width)))


if __name__ == "__main__":
    main()
