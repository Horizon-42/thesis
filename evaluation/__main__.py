"""CLI: evaluate trajectory record files → one JSON report.

    python -m evaluation --input <record.json | batch directory> \
        [--output evaluation_report.json] \
        [--lateral-max-m 106.75] [--vertical-below-max-m 3.05] [--vertical-above-max-m 6.10]

Input records follow the contract in ``records.py`` (the optimization batch writes
them as ``*_eval.json``, including empty records for unsolved configurations). A
directory input must be a batch directory: its ``summary.json`` manifest names the
run's records (no filename guessing — see ``load_records``). The report JSON
carries the batch metrics + one row per trajectory; a compact summary is printed
to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.metrics import evaluate_batch
from evaluation.records import load_records
from evaluation.thresholds import DeviationThresholds


def main(argv: list[str] | None = None) -> None:
    defaults = DeviationThresholds()
    parser = argparse.ArgumentParser(
        description="Evaluate optimized-trajectory records against their target states."
    )
    parser.add_argument("--input", required=True,
                        help="one record JSON, or a batch directory (read via its "
                             "summary.json manifest)")
    parser.add_argument("--output", default="evaluation_report.json",
                        help="where to write the report JSON")
    parser.add_argument("--lateral-max-m", type=float, default=defaults.lateral_max_m,
                        help="lateral gate (8260.58D Formula 3-1-1 course semiwidth floor)")
    parser.add_argument("--vertical-below-max-m", type=float,
                        default=defaults.vertical_below_max_m,
                        help="vertical gate below target (8260.58D WCH window)")
    parser.add_argument("--vertical-above-max-m", type=float,
                        default=defaults.vertical_above_max_m,
                        help="vertical gate above target (8260.58D WCH window)")
    args = parser.parse_args(argv)

    thresholds = DeviationThresholds(
        lateral_max_m=args.lateral_max_m,
        vertical_below_max_m=args.vertical_below_max_m,
        vertical_above_max_m=args.vertical_above_max_m,
    )
    records = load_records(args.input)
    report = evaluate_batch(records, thresholds)
    report["input"] = str(args.input)
    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_summary(report, out)


def _print_summary(report: dict[str, Any], out: Path) -> None:
    """Console digest of a ``metrics.evaluate_batch`` report dict (schema there)."""
    total, solved, ok = report["total"], report["solved"], report["successful"]
    print(f"evaluated {total} trajectories -> {out}")
    print(f"  solve rate    {solved}/{total} = {report['solve_rate']:.1%}")
    among = report["success_rate_among_solved"]
    suffix = f"   (among solved: {ok}/{solved} = {among:.1%})" if among is not None else ""
    print(f"  success rate  {ok}/{total} = {report['success_rate']:.1%}{suffix}")
    thresholds = report["thresholds"]
    lateral = report["lateral_m"]
    if lateral is not None:
        print(
            f"  lateral dev   mean {lateral['mean']:.1f} m   p95 {lateral['p95']:.1f} m   "
            f"max {lateral['max']:.1f} m   (limit {thresholds['lateral_max_m']:.2f} m)"
        )
    vertical = report["vertical_m"]
    if vertical is not None:
        print(
            f"  vertical dev  mean|.| {vertical['mean_abs']:.1f} m   "
            f"p95|.| {vertical['p95_abs']:.1f} m   max|.| {vertical['max_abs']:.1f} m   "
            f"(window -{thresholds['vertical_below_max_m']:.2f}/"
            f"+{thresholds['vertical_above_max_m']:.2f} m)"
        )
    times = report["final_time_s"]
    if times is not None:
        print(
            f"  flight time   mean {times['mean']:.1f} s   "
            f"min {times['min']:.1f} s   max {times['max']:.1f} s"
        )
    reference = report["reference"]
    if reference is not None:
        delta = reference["flight_time_delta_s"]
        path = reference["path_lateral_m"]
        print(
            f"  vs observed   Δtime mean {delta['mean']:+.1f} s "
            f"[{delta['min']:+.1f}, {delta['max']:+.1f}]   "
            f"path lateral mean {path['mean']:.0f} m   max {path['max']:.0f} m   "
            f"({reference['compared']} compared)"
        )
    failures = [row for row in report["trajectories"] if not row["success"]]
    if failures:
        print(f"  {len(failures)} not successful:")
        for row in failures[:10]:
            what = row.get("reason") or "; ".join(row["violations"])
            print(f"    {row['id']}: {what}")
        if len(failures) > 10:
            print(f"    … and {len(failures) - 10} more")


if __name__ == "__main__":
    main()
