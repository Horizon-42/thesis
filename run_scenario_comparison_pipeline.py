#!/usr/bin/env python
"""Run the whole scenario → optimization → comparison-CZML pipeline in one shot.

Three steps, chained by shelling out to the existing CLIs (each already tested):

  1. flight_scenarios               landings/<ICAO>/<ICAO>_combined_czml_input.json
                                    ─► flight_scenarios/outputs/<ICAO>_…_scenarios.json
  2. scenario_optimization          scenarios ─► 4dTrajectory/outputs/<ICAO>/<category>/
                                                   {*_states.json, summary.json}
  3. build_scenario_comparison_czml summary   ─► aeroviz-4d/public/data/airports/<ICAO>/
                                                   comparison/<category>/{*.czml, index, categories.json}

Only THREE inputs; every file path is defaulted from the current examples. The
``category`` (the output sub-folder + the frontend's category key) is derived
from (target_type, with_constraint):

    target_type=adsb   · with_constraint=False ─► asdb         (ADS-B target)
    target_type=runway · with_constraint=False ─► runway       (Runway target)
    target_type=runway · with_constraint=True  ─► runway_cons  (Runway target, constrained)

so the three canonical runs land in three sibling folders under each airport —
4dTrajectory/outputs/<ICAO>/{asdb,runway,runway_cons}/ — the per-airport layer
this adds over the old flat 4dTrajectory/outputs/<category>/.

Usage:
    python run_scenario_comparison_pipeline.py --airport KRDU --target-type runway --with-constraint
    python run_scenario_comparison_pipeline.py --airport KSJC --target-type adsb
    python run_scenario_comparison_pipeline.py --airport KRDU --target-type runway --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ── Default I/O roots (per the current examples; not CLI inputs) ───────────────
LANDINGS_DIR = REPO_ROOT / "trajectory_data_process" / "outputs" / "landings"
SCENARIOS_DIR = REPO_ROOT / "flight_scenarios" / "outputs"
OPT_OUTPUTS_ROOT = REPO_ROOT / "4dTrajectory" / "outputs"
COMPARISON_AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
OPT_SCRIPT = REPO_ROOT / "4dTrajectory" / "optimization" / "scenario_optimization.py"
CZML_SCRIPT = REPO_ROOT / "aeroviz-4d" / "python" / "build_scenario_comparison_czml.py"

TARGET_TYPES = ("adsb", "runway")

# category key (= output sub-folder + frontend category key) and its display label.
_CATEGORY_LABELS = {
    "asdb": "ADS-B target",
    "asdb_cons": "ADS-B target (constrained)",
    "runway": "Runway target",
    "runway_cons": "Runway target (constrained)",
}


def category_key(target_type: str, with_constraint: bool) -> str:
    base = "asdb" if target_type == "adsb" else "runway"
    return f"{base}_cons" if with_constraint else base


class Plan:
    """The resolved paths + the three commands for one pipeline run (pure data, so
    it can be previewed with --dry-run or asserted in a test)."""

    def __init__(self, airport: str, target_type: str, with_constraint: bool) -> None:
        self.airport = airport.strip().upper()
        self.target_type = target_type
        self.with_constraint = with_constraint
        self.threshold = target_type == "runway"
        self.category = category_key(target_type, with_constraint)
        self.label = _CATEGORY_LABELS[self.category]

        tag = "_threshold" if self.threshold else ""
        self.czml_input = LANDINGS_DIR / self.airport / f"{self.airport}_combined_czml_input.json"
        self.scenarios = SCENARIOS_DIR / f"{self.airport}_combined_czml_input{tag}_scenarios.json"
        self.opt_dir = OPT_OUTPUTS_ROOT / self.airport / self.category
        self.summary = self.opt_dir / "summary.json"
        self.comparison_dir = (
            COMPARISON_AIRPORTS_ROOT / self.airport / "comparison" / self.category
        )

    def steps(self) -> list[tuple[str, list[str]]]:
        py = sys.executable
        scenarios_cmd = [
            py, "-m", "flight_scenarios",
            "--input", str(self.czml_input),
            "--combined",
            "--output", str(self.scenarios),
        ]
        if self.threshold:
            scenarios_cmd.append("--target-from-threshold")

        optimize_cmd = [
            py, str(OPT_SCRIPT),
            "--scenarios", str(self.scenarios),
            "--output-dir", str(self.opt_dir),
        ]
        if self.with_constraint:
            # Constrained-IAF: optimize via the runway's RNAV(GPS) procedure (one
            # trajectory per scenario, IAF chosen by shortest 3D path).
            optimize_cmd += ["--constrained-iaf", "--iaf-selection", "shortest",
                             "--airport", self.airport]

        comparison_cmd = [
            py, str(CZML_SCRIPT),
            "--summary", str(self.summary),
            "--output-dir", str(self.comparison_dir),
            "--airport", self.airport,
            "--category", self.category,
            "--category-label", self.label,
        ]
        return [
            ("1/3 scenarios", scenarios_cmd),
            ("2/3 optimization", optimize_cmd),
            ("3/3 comparison CZML", comparison_cmd),
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--airport", default="KRDU", help="airport ICAO (default: KRDU)")
    parser.add_argument(
        "--target-type", choices=TARGET_TYPES, default="adsb",
        help="target state: 'adsb' = end of the observed track (default); "
             "'runway' = the published runway threshold",
    )
    parser.add_argument(
        "--with-constraint", action="store_true",
        help="enforce the runway's RNAV(GPS) procedure (constrained-IAF optimization)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the resolved paths + the three commands without running them",
    )
    args = parser.parse_args()

    plan = Plan(args.airport, args.target_type, args.with_constraint)

    print(f"airport={plan.airport}  target_type={plan.target_type}  "
          f"with_constraint={plan.with_constraint}  ->  category={plan.category!r}")
    print(f"  input     : {plan.czml_input}")
    print(f"  scenarios : {plan.scenarios}")
    print(f"  states    : {plan.opt_dir}")
    print(f"  comparison: {plan.comparison_dir}")

    if not plan.czml_input.exists():
        parser.error(
            f"missing CZML-input landings file: {plan.czml_input}\n"
            f"(expected trajectory_data_process/outputs/landings/{plan.airport}/"
            f"{plan.airport}_combined_czml_input.json)"
        )

    steps = plan.steps()
    if args.dry_run:
        print("\n--dry-run: would execute:")
        for label, cmd in steps:
            print(f"\n  [{label}]\n    {' '.join(cmd)}")
        return

    for label, cmd in steps:
        print(f"\n=== [{label}] ===\n{' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    print(f"\n✓ pipeline complete for {plan.airport} [{plan.category}]")
    print(f"  states     -> {plan.opt_dir}")
    print(f"  comparison -> {plan.comparison_dir}")
    print(f"  categories -> {plan.comparison_dir.parent / 'categories.json'}")


if __name__ == "__main__":
    main()
