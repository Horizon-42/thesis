#!/usr/bin/env python3
"""Evaluate every trajectory record batch on disk, in one sweep.

A thin orchestrator — discovery plus the two existing evaluation entry points,
nothing else. Three batch families, one report written INTO each batch directory:

  observed  (baseline)   trajectory_data_process/outputs/harvest/<A>/approach/
  optimized              4dTrajectory/outputs/<A>/{fitted_adsb,runway,runway_cons}/
  predicted              4dTrajectory/outputs/<A>/ts_pred_*/

Observed batches are evaluated the way the harvest itself does it (streamed
records, airport-loaded contexts, the producer's event-availability block) —
their roster does not name airports per row, so the generic CLI would
materialize the whole batch. Computed batches go through the standard
``python -m evaluation`` path, which streams via the summary.json roster.

Nothing is rebuilt or downloaded: this judges the records that exist. Bounded
coverage, stated: experiment trees (4dTrajectory/outputs/POOLED training runs,
experiments/, comparison publications) are NOT swept — evaluate one of those
directly with ``python -m evaluation --input <dir>``.

Notes:
  * Computed records written before 2026-08-23 carry no ``source.landing_aero``,
    so the v6 speed gate grades them indeterminate until their batch is re-run.
  * ``--html`` renders the overlay report next to each JSON (one extra streamed
    pass per batch).

Usage:
    python run_all_evaluations.py                      # everything found
    python run_all_evaluations.py --airport KRDU KSJC  # subset of airports
    python run_all_evaluations.py --kind observed      # one family only
    python run_all_evaluations.py --html               # also render HTML
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluation.__main__ as evaluation_cli  # noqa: E402
import evaluation.visualize as evaluation_visualize  # noqa: E402
from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG  # noqa: E402
from evaluation.context import contexts_for_airport  # noqa: E402
from evaluation.metrics import evaluate_batch  # noqa: E402
from trajectory_data_process.harvest.airports import load_airport  # noqa: E402
from trajectory_data_process.harvest.observed import (  # noqa: E402
    REPORT_NAME,
    SUMMARY_NAME,
    iter_observed_records,
)
from trajectory_data_process.harvest.store import HarvestPaths  # noqa: E402

HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
OPT_OUTPUTS_ROOT = REPO_ROOT / "4dTrajectory" / "outputs"
# MUST match run_scenario_optimization.ALL_MODES' category keys (that runner is
# deliberately not imported here — it drags the whole optimization stack in).
OPTIMIZED_CATEGORIES = ("fitted_adsb", "runway", "runway_cons")
PREDICTED_GLOB = "ts_pred_*"
HTML_NAME = "evaluation_report.html"
KINDS = ("observed", "optimized", "predicted")


class Batch(NamedTuple):
    kind: str       # one of KINDS
    airport: str
    label: str      # e.g. "KRDU observed", "KRDU runway_cons"
    path: Path


def discover(airports: list[str] | None, kinds: set[str]) -> list[Batch]:
    """Every batch directory (has a summary.json) in the requested slice."""
    if airports is None:
        found = {p.name for root in (HARVEST_ROOT, OPT_OUTPUTS_ROOT) if root.is_dir()
                 for p in root.iterdir() if p.is_dir()}
        airports = sorted(found)
    batches: list[Batch] = []
    for code in airports:
        if "observed" in kinds:
            approach = HarvestPaths(HARVEST_ROOT, code).approach
            if (approach / SUMMARY_NAME).is_file():
                batches.append(Batch("observed", code, f"{code} observed", approach))
        airport_root = OPT_OUTPUTS_ROOT / code
        if "optimized" in kinds:
            for category in OPTIMIZED_CATEGORIES:
                directory = airport_root / category
                if (directory / SUMMARY_NAME).is_file():
                    batches.append(
                        Batch("optimized", code, f"{code} {category}", directory)
                    )
        if "predicted" in kinds:
            for directory in sorted(airport_root.glob(PREDICTED_GLOB)):
                if (directory / SUMMARY_NAME).is_file():
                    batches.append(
                        Batch("predicted", code, f"{code} {directory.name}", directory)
                    )
    return batches


def evaluate_observed(batch: Batch, *, html: bool) -> None:
    """The harvest's own evaluation step, minus any rebuilding."""
    paths = HarvestPaths(HARVEST_ROOT, batch.airport)
    # The same authoritative data the generic CLI resolves, so both sweep paths
    # grade against identical contexts (evaluation.cli owns the defaults).
    airport = load_airport(
        batch.airport, config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP
    )
    contexts = contexts_for_airport(airport)
    summary = json.loads((paths.approach / SUMMARY_NAME).read_text(encoding="utf-8"))
    report = evaluate_batch(
        iter_observed_records(paths),
        contexts=contexts,
        observed_availability=summary["event_availability"],
    )
    report["input"] = str(paths.approach)
    out = paths.approach / REPORT_NAME
    out.write_text(json.dumps(report, indent=1, allow_nan=False), encoding="utf-8")
    counts = report["verdict_counts"]
    print(f"evaluated {report['total']} trajectories -> {out}")
    print(f"  verdicts      pass {counts['pass']}  fail {counts['fail']}  "
          f"indeterminate {counts['indeterminate']}")
    if html:
        payload = evaluation_visualize.build_payload_streamed(
            paths.approach, contexts=contexts
        )
        (paths.approach / HTML_NAME).write_text(
            evaluation_visualize.render_html(
                payload,
                title=f"{batch.airport} observed baseline",
                source_label=str(paths.approach),
            ),
            encoding="utf-8",
        )
        print(f"  html          -> {paths.approach / HTML_NAME}")


def evaluate_computed(batch: Batch, *, html: bool) -> None:
    """The documented per-batch pipeline steps, verbatim."""
    evaluation_cli.main([
        "--input", str(batch.path),
        "--output", str(batch.path / REPORT_NAME),
    ])
    if html:
        evaluation_visualize.main([
            "--input", str(batch.path),
            "--output", str(batch.path / HTML_NAME),
            "--title", f"{batch.label} evaluation",
        ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Not swept: POOLED/experiment trees — use python -m evaluation directly.",
    )
    parser.add_argument("--airport", nargs="+", metavar="ICAO",
                        help="airports to sweep (default: every one found on disk)")
    parser.add_argument("--kind", nargs="+", default=list(KINDS), choices=KINDS,
                        help="batch families to evaluate; 'observed' is the ADS-B "
                             "baseline the computed results are compared against")
    parser.add_argument("--html", action="store_true",
                        help="also render the HTML overlay report per batch")
    args = parser.parse_args(argv)

    kinds = set(args.kind)
    airports = [code.upper() for code in args.airport] if args.airport else None
    batches = discover(airports, kinds)
    if not batches:
        print("no evaluatable batches found for the requested slice")
        return 1

    failures: list[tuple[Batch, str]] = []
    for index, batch in enumerate(batches, 1):
        print(f"\n━━ [{index}/{len(batches)}] {batch.label}  ({batch.kind})")
        try:
            if batch.kind == "observed":
                evaluate_observed(batch, html=args.html)
            else:
                evaluate_computed(batch, html=args.html)
        except Exception as exc:  # noqa: BLE001 — sweep: report per batch, keep going
            failures.append((batch, f"{type(exc).__name__}: {exc}"))
            print(f"✗ {batch.label}: {failures[-1][1]}")

    print(f"\n{'━' * 60}")
    print(f"evaluated {len(batches) - len(failures)}/{len(batches)} batch(es)")
    for batch, error in failures:
        print(f"  ✗ {batch.label}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
