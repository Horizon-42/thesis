"""CLI: evaluate trajectory records into a strict JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.cli import add_context_args, contexts_from_args
from evaluation.metrics import evaluate_batch
from evaluation.records import load_records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate U.S. runway-threshold trajectory events."
    )
    parser.add_argument("--input", required=True,
                        help="one record JSON or a batch directory with summary.json")
    parser.add_argument("--output", default="evaluation_report.json")
    add_context_args(parser)
    args = parser.parse_args(argv)

    records = load_records(args.input)
    report = evaluate_batch(records, contexts=contexts_from_args(records, args))
    report["input"] = str(args.input)
    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    _print_summary(report, out)


def _print_summary(report: dict[str, Any], out: Path) -> None:
    counts = report["verdict_counts"]
    print(f"evaluated {report['total']} trajectories -> {out}")
    print(
        f"  verdicts      pass {counts['pass']}  fail {counts['fail']}  "
        f"indeterminate {counts['indeterminate']}"
    )
    observed = report.get("observed")
    if observed:
        print(
            f"  events        estimated {observed['event_estimated']}/"
            f"{observed['event_estimated'] + observed['event_unavailable']} = "
            f"{observed['event_estimated_rate']:.1%}"
        )
    if report["lateral_m"]:
        print(f"  cross-track   {report['lateral_m']}")
    if report["vertical_m"]:
        print(f"  vertical      {report['vertical_m']}")
    if report["reference"]:
        print(f"  references    {report['reference']['compared']} comparable span(s)")


if __name__ == "__main__":
    main()
