#!/usr/bin/env python
"""Run the whole scenario → optimization → comparison-CZML pipeline in one shot.

Three steps, chained by shelling out to the existing CLIs (each already tested):

  1. flight_scenarios               landings/<ICAO>/<ICAO>_combined_czml_input.json
                                    ─► flight_scenarios/outputs/<ICAO>_…_scenarios.json
  2. scenario_optimization          scenarios ─► 4dTrajectory/outputs/<ICAO>/<category>/
                                                   {*_states.json, summary.json}
  3. build_scenario_comparison_czml summary   ─► aeroviz-4d/public/data/airports/<ICAO>/
                                                   comparison/<category>/{*.czml, index, categories.json}

Every file path is defaulted from the current examples. The ``category`` (the
output sub-folder + the frontend's category key) is derived from
(target_type, with_constraint):

    target_type=adsb   · with_constraint=False ─► asdb         (ADS-B target)
    target_type=runway · with_constraint=False ─► runway       (Runway target)
    target_type=runway · with_constraint=True  ─► runway_cons  (Runway target, constrained)

so the three canonical runs land in three sibling folders under each airport —
4dTrajectory/outputs/<ICAO>/{asdb,runway,runway_cons}/ — the per-airport layer
this adds over the old flat 4dTrajectory/outputs/<category>/.

Airport selection:
  * --airport <ICAO>  runs that one airport.
  * (omitted)         runs EVERY K-prefixed airport that has landings data.

--skip-optimize reuses an already-computed optimization: if this airport+category
already has a 4dTrajectory/outputs/<ICAO>/<category>/summary.json, steps 1–2 are
skipped and only the comparison CZML (step 3) is rebuilt; if it does not exist,
the full pipeline runs from scratch.

Usage:
    # one airport, one category:
    python run_scenario_comparison_pipeline.py --airport KRDU --target-type runway --with-constraint
    # every K-airport, ADS-B target:
    python run_scenario_comparison_pipeline.py --target-type adsb
    # rebuild only the CZML for every airport that already has the optimization:
    python run_scenario_comparison_pipeline.py --target-type runway --skip-optimize
    # preview without running:
    python run_scenario_comparison_pipeline.py --dry-run
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


def czml_input_name(airport: str) -> str:
    return f"{airport}_combined_czml_input.json"


def discover_k_airports() -> list[str]:
    """Every K-prefixed airport under the landings dir that has a combined
    CZML-input file (the input the pipeline needs). US ICAO codes start with 'K'."""
    if not LANDINGS_DIR.exists():
        return []
    airports: list[str] = []
    for child in sorted(LANDINGS_DIR.iterdir()):
        code = child.name.upper()
        if child.is_dir() and code.startswith("K") \
                and (child / czml_input_name(code)).exists():
            airports.append(code)
    return airports


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
        self.czml_input = LANDINGS_DIR / self.airport / czml_input_name(self.airport)
        self.scenarios = SCENARIOS_DIR / f"{self.airport}_combined_czml_input{tag}_scenarios.json"
        self.opt_dir = OPT_OUTPUTS_ROOT / self.airport / self.category
        self.summary = self.opt_dir / "summary.json"
        self.comparison_dir = (
            COMPARISON_AIRPORTS_ROOT / self.airport / "comparison" / self.category
        )

    def optimization_exists(self) -> bool:
        """Whether this airport+category already has an optimization result to reuse."""
        return self.summary.exists()

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


def run_for_airport(
    airport: str,
    target_type: str,
    with_constraint: bool,
    *,
    dry_run: bool,
    skip_optimize: bool,
) -> bool:
    """Run (or preview) the pipeline for one airport. Returns True if it ran /
    would run, False if it was skipped (missing input and nothing to reuse)."""
    plan = Plan(airport, target_type, with_constraint)
    reuse = skip_optimize and plan.optimization_exists()

    mode = "reuse optimization → CZML only" if reuse else "full pipeline"
    print(f"\n━━ {plan.airport}  [{plan.category}]  ·  {mode}")
    print(f"   scenarios : {plan.scenarios}")
    print(f"   states    : {plan.opt_dir}")
    print(f"   comparison: {plan.comparison_dir}")

    all_steps = plan.steps()
    if reuse:
        steps = all_steps[-1:]  # comparison CZML only
    else:
        if not plan.czml_input.exists():
            print(f"   ⚠ skip: missing input {plan.czml_input}")
            return False
        if skip_optimize:
            print("   (no existing optimization found → running from scratch)")
        steps = all_steps

    if dry_run:
        for label, cmd in steps:
            print(f"   [{label}] {' '.join(cmd)}")
        return True

    for label, cmd in steps:
        print(f"\n=== [{plan.airport} · {label}] ===\n{' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    print(f"✓ {plan.airport} [{plan.category}] done → {plan.comparison_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--airport", default=None,
        help="airport ICAO; OMIT to run every K-prefixed airport that has landings data",
    )
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
        "--skip-optimize", action="store_true",
        help="if this airport+category already has an optimization result "
             "(summary.json), skip steps 1–2 and only (re)build the comparison CZML; "
             "otherwise run the full pipeline from scratch",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the resolved paths + the commands without running them",
    )
    args = parser.parse_args()

    if args.airport:
        airports = [args.airport.strip().upper()]
    else:
        airports = discover_k_airports()
        if not airports:
            parser.error(f"no K-prefixed airports with landings data under {LANDINGS_DIR}")
        print(f"no --airport given → running {len(airports)} K-airport(s): "
              f"{', '.join(airports)}")

    ran = 0
    for airport in airports:
        if run_for_airport(
            airport, args.target_type, args.with_constraint,
            dry_run=args.dry_run, skip_optimize=args.skip_optimize,
        ):
            ran += 1

    verb = "previewed" if args.dry_run else "completed"
    print(f"\n✓ {verb} {ran}/{len(airports)} airport(s)  "
          f"[target-type={args.target_type}, with-constraint={args.with_constraint}, "
          f"skip-optimize={args.skip_optimize}]")


if __name__ == "__main__":
    main()
