"""Backfill ``source.landing_aero`` on optimizer records written before the block existed.

The optimizer batches on disk (70,267 records across 15 batches, measured 2026-09-07)
were solved from scenarios prepared on 2026-08-23, one day before
``flight_scenarios.build_scenario`` began writing the stall facts the threshold speed
gate anchors on. Every one of them therefore grades speed-indeterminate and their
three-gate pass count is 0. The solve itself never read the block, so it can be added
without re-solving: the block is a pure function of the aircraft the model flew,
``source.dynamics_typecode`` -> ``aircraft_for_code`` -> ``aero_params_for_aircraft`` --
the SAME chain ``build_scenario`` runs (``flight_scenarios/build.py``).

Dry run by default; ``--apply`` rewrites the record files in place and prints the
counts. The evaluation reports are NOT regenerated here -- run
``python -m evaluation --input <batch> --output <batch>/evaluation_report.json`` (or
``run_all_evaluations.py``) afterwards. Only batches whose roster is ``optimized`` are
touched; prediction and reference records never carried the gap.

Usage::

    python 4dTrajectory/optimization/backfill_landing_aero.py --root 4dTrajectory/outputs
    python 4dTrajectory/optimization/backfill_landing_aero.py --root 4dTrajectory/outputs --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aircraft.aero_params import aero_params_for_aircraft  # noqa: E402
from evaluation.speed_gate import LANDING_AERO_KEY  # noqa: E402
from flight_scenarios.scenario import aircraft_for_code  # noqa: E402

SUMMARY_NAME = "summary.json"
# The producers' serialization (scenario_optimization writes records compact and
# NaN-free); the default separators would grow every record by ~8 % for nothing.
_JSON_KWARGS = {"separators": (",", ":"), "allow_nan": False}


@dataclass
class BackfillCounts:
    batches: int = 0
    skipped_batches: int = 0
    already: int = 0
    backfilled: int = 0

    def __str__(self) -> str:
        return (
            f"batches {self.batches} (skipped non-optimized {self.skipped_batches}) · "
            f"already carrying landing_aero {self.already} · backfilled {self.backfilled}"
        )


def landing_aero_for(source: dict, *, aircraft_provider: str = "auto") -> dict[str, float]:
    """The block ``build_scenario`` would have written for this record's aircraft.

    ``dynamics_typecode`` is the code of the aircraft the model flew (surrogates
    included -- ``AircraftSelection.audit_fields``), and ``aircraft_provider`` must be
    the batch's own (``build_scenario(aircraft_provider=...)``; the default ``auto``
    matches every batch on disk, verified on 326 flights against their observed twins).
    """
    code = source.get("dynamics_typecode")
    if not isinstance(code, str) or not code:
        raise ValueError(
            f"record {source.get('flight_key') or source.get('id')!r} names no "
            "dynamics_typecode; the aircraft the model flew is unknown"
        )
    aero = aero_params_for_aircraft(aircraft_for_code(code, provider=aircraft_provider))
    return {"wing_area_m2": aero.S, "cl_max_landing": aero.Cl_max}


def backfill_batch(
    batch: Path, *, apply: bool, counts: BackfillCounts, aircraft_provider: str = "auto"
) -> None:
    rows = json.loads((batch / SUMMARY_NAME).read_text(encoding="utf-8")).get("results")
    files = [batch / row["eval_file"] for row in rows or [] if row.get("eval_file")]
    if not files:
        counts.skipped_batches += 1
        return
    first = json.loads(files[0].read_text(encoding="utf-8"))
    if first.get("source", {}).get("subject") != "optimized":
        counts.skipped_batches += 1
        return
    counts.batches += 1
    for file in files:
        record = json.loads(file.read_text(encoding="utf-8"))
        source = record["source"]
        if source.get(LANDING_AERO_KEY) is not None:
            counts.already += 1
            continue
        source[LANDING_AERO_KEY] = landing_aero_for(
            source, aircraft_provider=aircraft_provider
        )
        counts.backfilled += 1
        if apply:
            file.write_text(json.dumps(record, **_JSON_KWARGS), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True,
                        help="directory whose batches (any depth) carry summary.json")
    parser.add_argument("--apply", action="store_true",
                        help="rewrite the records; without it, only count")
    parser.add_argument("--aircraft-provider", default="auto",
                        help="the provider the batch was built with (build_scenario's "
                             "aircraft_provider; default auto)")
    args = parser.parse_args(argv)
    counts = BackfillCounts()
    for manifest in sorted(args.root.rglob(SUMMARY_NAME)):
        backfill_batch(
            manifest.parent, apply=args.apply, counts=counts,
            aircraft_provider=args.aircraft_provider,
        )
    print(f"{'applied' if args.apply else 'dry run'}: {counts}")
    if not args.apply and counts.backfilled:
        print("re-run with --apply to write, then regenerate each batch's evaluation report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
