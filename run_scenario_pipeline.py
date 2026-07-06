#!/usr/bin/env python
"""Run the scenario → optimization → {comparison CZML, evaluation} pipeline in one shot.

The merge of the former run_scenario_comparison_pipeline.py and
run_scenario_evaluation_pipeline.py: both shared steps 1–2 (scenarios +
optimization) and only differed in the tail, so one script now runs the
expensive optimization ONCE per airport/category and derives every output from
it — the two tails are selectable with ``--outputs``.

Steps, chained by shelling out to the existing CLIs (each already tested):

  1. flight_scenarios               landings/<ICAO>/<ICAO>_combined_czml_input.json
                                    ─► flight_scenarios/outputs/<ICAO>_…_scenarios.json
  2. scenario_optimization          scenarios + observed tracks ─► 4dTrajectory/outputs/<ICAO>/<category>/
       (--reference-tracks, always)    references/*_reference_eval.json   (written FIRST —
                                        the observed tracks in the evaluation contract)
                                      {*_states.json, *_eval.json, summary.json}
                                        (every eval record points at its reference via
                                         reference_file; failed ones included)
  3. python -m evaluation           eval records ─► <opt_dir>/evaluation_report.json
       (always — the CZML tail consumes its per-flight verdicts + batch metrics)
  4. [eval] python -m evaluation.visualize
                                    eval records ─► <opt_dir>/evaluation_report.html
  5. [czml] build_scenario_comparison_czml (--evaluation-report)
                                    summary + report ─► aeroviz-4d/public/data/airports/<ICAO>/
                                                 comparison/<category>/{*.czml, index, categories.json}
                                      (solved-but-off-target flights yellow; the index's
                                       optimization block carries the evaluation metrics)

The ``category`` (output sub-folder + frontend category key) is derived from
(target_type, with_constraint):

    target_type=adsb   · with_constraint=False ─► asdb         (ADS-B target)
    target_type=runway · with_constraint=False ─► runway       (Runway target)
    target_type=runway · with_constraint=True  ─► runway_cons  (Runway target, constrained)

Omitting --target-type runs ALL THREE modes (asdb, runway, runway_cons) per
airport — the full category sweep the frontend's comparison picker offers.

Airport selection:
  * --airport <ICAO>  runs that one airport.
  * (omitted)         runs EVERY K-prefixed airport that has landings data.

--skip-optimize reuses an already-computed optimization: if this airport+category
already has a summary.json, steps 1–2 are skipped and only the selected tails
(3–5) run; if it does not exist, the full pipeline runs from scratch.

Usage:
    # one airport, both outputs (frontend CZML + evaluation report/HTML):
    python run_scenario_pipeline.py --airport KRDU --target-type runway --with-constraint
    # one airport, ALL THREE modes (asdb + runway + runway_cons):
    python run_scenario_pipeline.py --airport KRDU
    # trapezoidal-fitting comparison run (default is Hermite-Simpson):
    python run_scenario_pipeline.py --airport KRDU --target-type runway --fitting-type trapezoidal
    # evaluation only:
    python run_scenario_pipeline.py --airport KRDU --outputs eval
    # rebuild only the comparison CZML from an existing optimization:
    python run_scenario_pipeline.py --airport KRDU --outputs czml --skip-optimize
    # every K-airport, ADS-B target, preview without running:
    python run_scenario_pipeline.py --target-type adsb --dry-run
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
OUTPUT_KINDS = ("czml", "eval")

# The full category sweep, run when --target-type is omitted: every mode the
# frontend's comparison picker offers, as (target_type, with_constraint).
ALL_MODES = (("adsb", False), ("runway", False), ("runway", True))

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
    """The resolved paths + commands for one airport/category run (pure data, so
    it can be previewed with --dry-run or asserted in a test)."""

    def __init__(self, airport: str, target_type: str, with_constraint: bool,
                 outputs: tuple[str, ...], jobs: int = 0,
                 fitting: str = "hs") -> None:
        self.airport = airport.strip().upper()
        self.target_type = target_type
        self.with_constraint = with_constraint
        self.outputs = outputs
        # Parallel optimizer worker processes (passed through to scenario_optimization's
        # --jobs; 0 = auto = half the CPU cores, 1 = serial).
        self.jobs = jobs
        # Transcription fitting for the solves (scenario_optimization --fitting):
        # "hs" = Hermite-Simpson (default) or "trapezoidal" (comparison runs). Both
        # fittings write into the SAME category dir — an experiment overwrites the
        # previous batch there (the optimizer's stale-record cleanup keeps it consistent).
        self.fitting = fitting
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
        self.report = self.opt_dir / "evaluation_report.json"
        self.report_html = self.opt_dir / "evaluation_report.html"

    def optimization_exists(self) -> bool:
        """Whether this airport+category already has an optimization result to reuse."""
        return self.summary.exists()

    def steps(self, *, reuse: bool = False) -> list[tuple[str, list[str]]]:
        """The commands to run; ``reuse`` drops steps 1–2 and keeps the selected tails."""
        py = sys.executable
        named: list[tuple[str, list[str]]] = []

        if not reuse:
            scenarios_cmd = [
                py, "-m", "flight_scenarios",
                "--input", str(self.czml_input),
                "--combined",
                "--output", str(self.scenarios),
            ]
            if self.threshold:
                scenarios_cmd.append("--target-from-threshold")
            named.append(("scenarios", scenarios_cmd))

            # --reference-tracks makes the optimization CLI write the reference eval
            # records FIRST (observed tracks -> <opt_dir>/references/) and point every
            # eval record at its reference. Always on: it is what the evaluation tail
            # consumes and it is harmless to the CZML tail — and it keeps the
            # opt_dir contents identical no matter which outputs are selected.
            optimize_cmd = [
                py, str(OPT_SCRIPT),
                "--scenarios", str(self.scenarios),
                "--output-dir", str(self.opt_dir),
                "--reference-tracks", str(self.czml_input),
                "--jobs", str(self.jobs),
                "--fitting", self.fitting,
            ]
            if self.with_constraint:
                # Constrained-IAF: optimize via the runway's RNAV(GPS) procedure (one
                # trajectory per scenario, IAF chosen by shortest 3D path).
                optimize_cmd += ["--constrained-iaf", "--iaf-selection", "shortest",
                                 "--airport", self.airport]
            named.append(("references + optimization", optimize_cmd))

        # The evaluation report always runs (it is cheap): the eval tail renders it and
        # the CZML tail consumes its verdicts (off-target yellow) + batch metrics. The one
        # exception: reusing an optimization from BEFORE the evaluation package exists no
        # eval records to judge — skip the evaluation steps loudly and build a plain CZML
        # (re-run without --skip-optimize to get verdicts/metrics).
        evaluable = not reuse or any(self.opt_dir.glob("*_eval.json"))
        if not evaluable:
            print("   ⚠ reused optimization has no *_eval.json records (pre-evaluation run) "
                  "— skipping evaluation; comparison CZML gets no verdicts/metrics")
        if evaluable:
            named.append(("evaluation report", [
                py, "-m", "evaluation",
                "--input", str(self.opt_dir),
                "--output", str(self.report),
            ]))

            if "eval" in self.outputs:
                named.append(("evaluation HTML", [
                    py, "-m", "evaluation.visualize",
                    "--input", str(self.opt_dir),
                    "--output", str(self.report_html),
                ]))

        if "czml" in self.outputs:
            comparison_cmd = [
                py, str(CZML_SCRIPT),
                "--summary", str(self.summary),
                "--output-dir", str(self.comparison_dir),
                "--airport", self.airport,
                "--category", self.category,
                "--category-label", self.label,
            ]
            if evaluable:
                comparison_cmd += ["--evaluation-report", str(self.report)]
            # Feed the scenario initial states so the index carries V + mass for EVERY
            # flight — including failed optimizations (which have no states file). On a
            # full run step 1 writes the file before this step; on reuse it may be absent.
            if not reuse or self.scenarios.exists():
                comparison_cmd += ["--scenarios", str(self.scenarios)]
            named.append(("comparison CZML", comparison_cmd))

        total = len(named)
        return [(f"{i}/{total} {name}", cmd) for i, (name, cmd) in enumerate(named, 1)]


def run_for_airport(
    airport: str,
    target_type: str,
    with_constraint: bool,
    outputs: tuple[str, ...],
    *,
    dry_run: bool,
    skip_optimize: bool,
    jobs: int = 0,
    fitting: str = "hs",
) -> bool:
    """Run (or preview) the pipeline for one airport. Returns True if it ran /
    would run, False if it was skipped (missing input and nothing to reuse)."""
    plan = Plan(airport, target_type, with_constraint, outputs, jobs=jobs, fitting=fitting)
    reuse = skip_optimize and plan.optimization_exists()

    mode = "reuse optimization" if reuse else "full pipeline"
    fit = "" if reuse else f"  ·  fitting: {plan.fitting}"
    print(f"\n━━ {plan.airport}  [{plan.category}]  ·  {mode}{fit}  ·  outputs: {', '.join(outputs)}")
    print(f"   scenarios : {plan.scenarios}")
    print(f"   states    : {plan.opt_dir}")
    if "czml" in outputs:
        print(f"   comparison: {plan.comparison_dir}")
    if "eval" in outputs:
        print(f"   report    : {plan.report}")

    if not reuse:
        if not plan.czml_input.exists():
            print(f"   ⚠ skip: missing input {plan.czml_input}")
            return False
        if skip_optimize:
            print("   (no existing optimization found → running from scratch)")
    steps = plan.steps(reuse=reuse)

    if dry_run:
        for label, cmd in steps:
            print(f"   [{label}] {' '.join(cmd)}")
        return True

    for label, cmd in steps:
        print(f"\n=== [{plan.airport} · {label}] ===\n{' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    print(f"✓ {plan.airport} [{plan.category}] done  ·  outputs: {', '.join(outputs)}")
    return True


def _parse_outputs(raw: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in raw.split(",") if token.strip())
    unknown = [t for t in tokens if t not in OUTPUT_KINDS]
    if unknown or not tokens:
        raise argparse.ArgumentTypeError(
            f"--outputs takes a comma list from {OUTPUT_KINDS}, got {raw!r}"
        )
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--airport", default=None,
        help="airport ICAO; OMIT to run every K-prefixed airport that has landings data",
    )
    parser.add_argument(
        "--target-type", choices=TARGET_TYPES, default=None,
        help="target state: 'runway' = the published runway threshold (the evaluation "
             "gates are threshold-referenced); 'adsb' = end of the observed track. "
             "OMIT to run all three modes (asdb, runway, runway_cons) per airport",
    )
    parser.add_argument(
        "--with-constraint", action="store_true",
        help="enforce the runway's RNAV(GPS) procedure (constrained-IAF optimization); "
             "requires --target-type (the all-modes sweep already includes runway_cons)",
    )
    parser.add_argument(
        "--outputs", type=_parse_outputs, default=OUTPUT_KINDS, metavar="czml,eval",
        help="which tails to produce from the optimization: 'czml' (frontend comparison "
             "CZML), 'eval' (evaluation report JSON + HTML); default: both",
    )
    parser.add_argument(
        "--jobs", type=int, default=0,
        help="parallel optimizer worker processes (passed to scenario_optimization --jobs; "
             "0 = auto = half the CPU cores, 1 = serial)",
    )
    parser.add_argument(
        "--fitting-type", choices=("hs", "trapezoidal"), default="hs",
        help="transcription fitting for the solves (scenario_optimization --fitting): "
             "'hs' = Hermite-Simpson (4th order, default), 'trapezoidal' = 2nd order "
             "(comparison runs; replays drift km-scale on aggressive min-time solves). "
             "Both write into the same category dir — a run overwrites the previous batch",
    )
    parser.add_argument(
        "--skip-optimize", action="store_true",
        help="if this airport+category already has an optimization result "
             "(summary.json), skip steps 1–2 and only (re)build the selected outputs; "
             "otherwise run the full pipeline from scratch",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the resolved paths + the commands without running them",
    )
    args = parser.parse_args()

    if args.target_type is None:
        if args.with_constraint:
            parser.error("--with-constraint requires --target-type "
                         "(the all-modes sweep already includes runway_cons)")
        modes = ALL_MODES
        print(f"no --target-type given → running all {len(modes)} modes per airport: "
              + ", ".join(category_key(t, c) for t, c in modes))
    else:
        modes = ((args.target_type, args.with_constraint),)

    if args.airport:
        airports = [args.airport.strip().upper()]
    else:
        airports = discover_k_airports()
        if not airports:
            parser.error(f"no K-prefixed airports with landings data under {LANDINGS_DIR}")
        print(f"no --airport given → running {len(airports)} K-airport(s): "
              f"{', '.join(airports)}")

    ran = 0
    runs = [(airport, mode) for airport in airports for mode in modes]
    for airport, (target_type, with_constraint) in runs:
        if run_for_airport(
            airport, target_type, with_constraint, tuple(args.outputs),
            dry_run=args.dry_run, skip_optimize=args.skip_optimize, jobs=args.jobs,
            fitting=args.fitting_type,
        ):
            ran += 1

    verb = "previewed" if args.dry_run else "completed"
    print(f"\n✓ {verb} {ran}/{len(runs)} run(s) "
          f"({len(airports)} airport(s) × {len(modes)} mode(s))  "
          f"[modes={','.join(category_key(t, c) for t, c in modes)}, "
          f"outputs={','.join(args.outputs)}, skip-optimize={args.skip_optimize}]")


if __name__ == "__main__":
    main()
