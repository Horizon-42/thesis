#!/usr/bin/env python
"""Run the scenario → optimization → {comparison CZML, evaluation} pipeline in one shot.

The merge of the former run_scenario_comparison_pipeline.py and
run_scenario_evaluation_pipeline.py: both shared steps 1–2 (scenarios +
optimization) and only differed in the tail, so one script now runs the
expensive optimization ONCE per airport/category and derives every output from
it — the two tails are selectable with ``--outputs``.

Steps, chained by shelling out to the existing CLIs (each already tested):

  0b. trajectory_data_process.harvest --evaluate-only   (unless --skip-observed)
                                    tracks/manifest.json
                                    ─► arrivals/manifest.json (assigned, LPV-targeted,
                                        final terminal entry → threshold)
                                    ─► observed CZML + evaluation report
  1. flight_scenarios               arrivals/manifest.json
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

    target_type=fitted-adsb · with_constraint=False ─► fitted_adsb (Fitted ADS-B crossing)
    target_type=runway      · with_constraint=False ─► runway      (Runway target)
    target_type=runway      · with_constraint=True  ─► runway_cons (Runway target, constrained)

Omitting --target-type runs ALL THREE modes (fitted_adsb, runway, runway_cons) per
airport — the full category sweep the frontend's comparison picker offers.

Airport selection:
  * --airport <ICAO>  runs that one airport.
  * (omitted)         runs every K-prefixed airport with a harvest track or arrival
                      manifest. The observed stage promotes tracks into model-ready arrivals.

--skip-optimize reuses an already-computed optimization: if this airport+category
already has a summary.json, steps 1–2 are skipped and only the selected tails
(3–5) run; if it does not exist, the full pipeline runs from scratch.

Usage:
    # one airport, both outputs (frontend CZML + evaluation report/HTML):
    python run_scenario_pipeline.py --airport KRDU --target-type runway --with-constraint
    # one airport, ALL THREE modes (fitted_adsb + runway + runway_cons):
    python run_scenario_pipeline.py --airport KRDU
    # trapezoidal-fitting comparison run (default is Hermite-Simpson):
    python run_scenario_pipeline.py --airport KRDU --target-type runway --fitting-type trapezoidal
    # custom control mesh (n-segments = unconstrained; n-seg-per-phase = constrained):
    python run_scenario_pipeline.py --airport KRDU --n-segments 12 --n-seg-per-phase 4
    # evaluation only:
    python run_scenario_pipeline.py --airport KRDU --outputs eval
    # rebuild only the comparison CZML from an existing optimization:
    python run_scenario_pipeline.py --airport KRDU --outputs czml --skip-optimize
    # every K-airport, fitted ADS-B crossing target, preview without running:
    python run_scenario_pipeline.py --target-type fitted-adsb --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ── Default I/O roots (per the current examples; not CLI inputs) ───────────────
SCENARIOS_DIR = REPO_ROOT / "flight_scenarios" / "outputs"
OPT_OUTPUTS_ROOT = REPO_ROOT / "4dTrajectory" / "outputs"
COMPARISON_AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
OPT_SCRIPT = REPO_ROOT / "4dTrajectory" / "optimization" / "scenario_optimization.py"
CZML_SCRIPT = REPO_ROOT / "aeroviz-4d" / "python" / "build_scenario_comparison_czml.py"
HARVEST_TRACKS_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"

TARGET_TYPES = ("fitted-adsb", "runway")
OUTPUT_KINDS = ("czml", "eval")

# Control-mesh defaults — MUST mirror CollocationOptimizer's (collocation/optimizer.py:
# DEFAULT_N_SEGMENTS / DEFAULT_N_SEG_PER_PHASE). The pipeline shells out (import-light: no
# casadi), so it cannot import them; kept in sync by this comment.
DEFAULT_N_SEGMENTS = 8         # unconstrained: control segments over the whole trajectory
DEFAULT_N_SEG_PER_PHASE = 3    # constrained: control segments PER procedure leg

# The full category sweep, run when --target-type is omitted: every mode the
# frontend's comparison picker offers, as (target_type, with_constraint).
ALL_MODES = (("fitted-adsb", False), ("runway", False), ("runway", True))

# category key (= output sub-folder + frontend category key) and its display label.
_CATEGORY_LABELS = {
    "fitted_adsb": "Fitted ADS-B crossing",
    "runway": "Runway target",
    "runway_cons": "Runway target (constrained)",
}


def category_key(target_type: str, with_constraint: bool) -> str:
    if target_type not in TARGET_TYPES:
        raise ValueError(f"unknown target type {target_type!r}; expected one of {TARGET_TYPES}")
    if target_type == "fitted-adsb":
        if with_constraint:
            raise ValueError("fitted-adsb is incompatible with procedure constraints")
        return "fitted_adsb"
    base = "runway"
    return f"{base}_cons" if with_constraint else base


def arrival_manifest_path(airport: str) -> Path:
    return HARVEST_TRACKS_ROOT / airport / "arrivals" / "manifest.json"


def discover_k_airports() -> list[str]:
    """Every K-airport the pipeline can run now or promote via evaluate-only."""
    if not HARVEST_TRACKS_ROOT.exists():
        return []
    airports = []
    for child in sorted(HARVEST_TRACKS_ROOT.iterdir()):
        code = child.name.upper()
        tracks = child / "tracks" / "manifest.json"
        if (
            child.is_dir()
            and code.startswith("K")
            and (arrival_manifest_path(code).exists() or tracks.exists())
        ):
            airports.append(code)
    return airports


class Plan:
    """The resolved paths + commands for one airport/category run (pure data, so
    it can be previewed with --dry-run or asserted in a test)."""

    def __init__(self, airport: str, target_type: str, with_constraint: bool,
                 outputs: tuple[str, ...], jobs: int = 0,
                 fitting: str = "hs", state_substeps: int | None = None,
                 n_segments: int = DEFAULT_N_SEGMENTS,
                 n_seg_per_phase: int = DEFAULT_N_SEG_PER_PHASE) -> None:
        self.airport = airport.strip().upper()
        if target_type not in TARGET_TYPES:
            raise ValueError(f"unknown target type {target_type!r}; expected one of {TARGET_TYPES}")
        if target_type == "fitted-adsb" and with_constraint:
            raise ValueError("fitted-adsb is incompatible with --with-constraint")
        self.target_type = target_type
        self.with_constraint = with_constraint
        self.outputs = outputs
        # Parallel optimizer worker processes (passed through to scenario_optimization's
        # --jobs; 0 = auto = half the CPU cores, 1 = serial).
        self.jobs = jobs
        # Control-mesh sizes: n_segments for the unconstrained solve (whole-trajectory
        # control count), n_seg_per_phase for the constrained multiphase solve (control
        # segments PER procedure leg). Each mode uses only its own knob (see steps()).
        self.n_segments = n_segments
        self.n_seg_per_phase = n_seg_per_phase
        # Transcription fitting for the solves (scenario_optimization --fitting):
        # "hs" = Hermite-Simpson (default) or "trapezoidal" (comparison runs). Both
        # fittings write into the SAME category dir — an experiment overwrites the
        # previous batch there (the optimizer's stale-record cleanup keeps it consistent).
        self.fitting = fitting
        # State-collocation density M per control segment (scenario_optimization
        # --state-substeps); None = the optimizer's auto (~3 s state step, cap 16).
        self.state_substeps = state_substeps
        self.threshold = target_type == "runway"
        self.fitted_adsb = target_type == "fitted-adsb"
        self.category = category_key(target_type, with_constraint)
        self.label = _CATEGORY_LABELS[self.category]

        tag = "_threshold" if self.threshold else "_fitted_adsb"
        self.arrivals_manifest = arrival_manifest_path(self.airport)
        self.scenarios = SCENARIOS_DIR / f"{self.airport}_arrivals{tag}_scenarios.json"
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
                "--input", str(self.arrivals_manifest),
                "--output", str(self.scenarios),
            ]
            if self.threshold:
                scenarios_cmd.append("--target-from-threshold")
            elif self.fitted_adsb:
                scenarios_cmd.append("--target-from-fitted-adsb")
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
                "--reference-tracks", str(self.arrivals_manifest),
                "--jobs", str(self.jobs),
                "--fitting", self.fitting,
            ]
            if self.state_substeps is not None:
                optimize_cmd += ["--state-substeps", str(self.state_substeps)]
            if self.with_constraint:
                # Constrained-IAF: optimize via the runway's RNAV(GPS) procedure (one
                # trajectory per scenario, IAF chosen by shortest 3D path). The multiphase
                # mesh is set PER LEG (n_seg_per_phase); n_segments does not apply here.
                optimize_cmd += ["--constrained-iaf", "--iaf-selection", "shortest",
                                 "--airport", self.airport,
                                 "--n-seg-per-phase", str(self.n_seg_per_phase)]
            else:
                # Unconstrained: one phase, control over the whole trajectory (n_segments).
                optimize_cmd += ["--n-segments", str(self.n_segments)]
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
            if self.with_constraint:
                comparison_cmd.append("--constrained")
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


def run_observed(airport: str, *, dry_run: bool) -> bool:
    """The OBSERVED baseline — per airport, independent of any optimization.

    Re-judges the harvested tracks, renders them as the viewer's observed layer, and
    publishes the report as the ``observed`` comparison category. Runs with
    ``--evaluate-only``, so it never downloads: it recomputes from ``tracks/``, which is
    what makes it cheap enough to run every time. Any change to the fit window, the
    established criteria or the CIFP cycle invalidates the previous answer, and the
    verdict colours painted on the observed layer come from exactly this report — so the
    tracks and their verdicts stay one harvest.

    Skipped (not an error) when the airport has no harvest yet.
    """
    manifest = HARVEST_TRACKS_ROOT / airport / "tracks" / "manifest.json"
    if not manifest.exists():
        print(f"   ⚠ {airport}: no harvest at {manifest.parent} — skipping the observed "
              f"stage (run: python -m trajectory_data_process.harvest --airport {airport})")
        return False
    cmd = [sys.executable, "-m", "trajectory_data_process.harvest",
           "--airport", airport, "--evaluate-only"]
    if dry_run:
        print(f"   [observed] {' '.join(cmd)}")
        return True
    print(f"\n=== [{airport} · observed baseline] ===\n{' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return True


def run_for_airport(
    airport: str,
    target_type: str,
    with_constraint: bool,
    outputs: tuple[str, ...],
    *,
    dry_run: bool,
    skip_optimize: bool,
    input_will_exist: bool = False,
    jobs: int = 0,
    fitting: str = "hs",
    state_substeps: int | None = None,
    n_segments: int = DEFAULT_N_SEGMENTS,
    n_seg_per_phase: int = DEFAULT_N_SEG_PER_PHASE,
) -> bool:
    """Run (or preview) the pipeline for one airport. Returns True if it ran /
    would run, False if it was skipped (missing input and nothing to reuse).

    The arrival manifest is the only data-plane input; no legacy landing-file fallback."""
    plan = Plan(airport, target_type, with_constraint, outputs, jobs=jobs, fitting=fitting,
                state_substeps=state_substeps, n_segments=n_segments,
                n_seg_per_phase=n_seg_per_phase)
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
        # The manifest is the data-plane contract. Missing means the harvest/evaluation
        # stage did not produce a model-ready roster; there is no glob fallback.
        if not plan.arrivals_manifest.exists() and not (dry_run and input_will_exist):
            print(f"   ⚠ skip: missing input {plan.arrivals_manifest}")
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
        help="airport ICAO; omit to run every K-prefixed airport with a harvest manifest",
    )
    parser.add_argument(
        "--target-type", choices=TARGET_TYPES, default=None,
        help="target state: 'runway' = the published runway threshold; 'fitted-adsb' = "
             "the final_approach OLS threshold crossing (fitted position + measured "
             "terminal kinematics). OMIT to run all three modes "
             "(fitted_adsb, runway, runway_cons) per airport",
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
        "--state-substeps", type=int, default=None, metavar="M",
        help="state-collocation subintervals per control segment (scenario_optimization "
             "--state-substeps); omit for the optimizer's auto density (~3 s state step, "
             "capped at 16). Higher M = denser plan states = slower solves",
    )
    parser.add_argument(
        "--n-segments", type=int, default=DEFAULT_N_SEGMENTS,
        help=f"UNCONSTRAINED control segments over the whole trajectory (fitted_adsb/runway modes; "
             f"default {DEFAULT_N_SEGMENTS}, = CollocationOptimizer's). Ignored by constrained runs",
    )
    parser.add_argument(
        "--n-seg-per-phase", type=int, default=DEFAULT_N_SEG_PER_PHASE,
        help=f"CONSTRAINED control segments PER procedure leg (runway_cons mode; default "
             f"{DEFAULT_N_SEG_PER_PHASE}, = CollocationOptimizer's). Ignored by unconstrained runs",
    )
    parser.add_argument(
        "--fitting-type", choices=("hs", "trapezoidal", "rk4"), default="hs",
        help="transcription fitting for the solves (scenario_optimization --fitting): "
             "'hs' = Hermite-Simpson (4th order, default), 'trapezoidal' = 2nd order "
             "(comparison runs; replays drift km-scale on aggressive min-time solves), "
             "'rk4' = 4th-order explicit shooting defect (playback-consistent). "
             "All write into the same category dir — a run overwrites the previous batch",
    )
    parser.add_argument(
        "--skip-observed", action="store_true",
        help="skip the observed-baseline stage (re-judge harvested tracks, render the "
             "observed CZML layer, publish the 'observed' comparison category). That "
             "stage never downloads — it recomputes from the stored harvest",
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

    if args.state_substeps is not None and args.state_substeps < 1:
        parser.error(f"--state-substeps must be >= 1, got {args.state_substeps}")
    if args.n_segments < 2:
        parser.error(f"--n-segments must be >= 2, got {args.n_segments}")
    if args.n_seg_per_phase < 1:
        parser.error(f"--n-seg-per-phase must be >= 1, got {args.n_seg_per_phase}")

    if args.target_type is None:
        if args.with_constraint:
            parser.error("--with-constraint requires --target-type "
                         "(the all-modes sweep already includes runway_cons)")
        modes = ALL_MODES
        print(f"no --target-type given → running all {len(modes)} modes per airport: "
              + ", ".join(category_key(t, c) for t, c in modes))
    else:
        if args.target_type == "fitted-adsb" and args.with_constraint:
            parser.error("--target-type fitted-adsb cannot be combined with --with-constraint")
        modes = ((args.target_type, args.with_constraint),)

    if args.airport:
        airports = [args.airport.strip().upper()]
    else:
        airports = discover_k_airports()
        if not airports:
            parser.error(
                f"no K-prefixed airports with tracks/manifest.json or "
                f"arrivals/manifest.json under {HARVEST_TRACKS_ROOT} — run "
                "trajectory_data_process/download_landings.py first"
            )
        print(f"no --airport given → running {len(airports)} K-airport(s): "
              f"{', '.join(airports)}")

    ran = 0
    runs = [(airport, mode) for airport in airports for mode in modes]
    for airport in airports:
        # The observed baseline is per AIRPORT and independent of the optimization —
        # it is the measured reference every modelled category is read against, and it
        # is what colours the observed layer. Runs before the categories so the frontend
        # has a baseline even if a solve later fails.
        observed_scheduled = False
        if not args.skip_observed:
            observed_scheduled = run_observed(airport, dry_run=args.dry_run)
        for target_type, with_constraint in modes:
            if run_for_airport(
                airport, target_type, with_constraint, tuple(args.outputs),
                dry_run=args.dry_run, skip_optimize=args.skip_optimize,
                input_will_exist=args.dry_run and observed_scheduled,
                jobs=args.jobs,
                fitting=args.fitting_type,
                state_substeps=args.state_substeps,
                n_segments=args.n_segments,
                n_seg_per_phase=args.n_seg_per_phase,
            ):
                ran += 1

    verb = "previewed" if args.dry_run else "completed"
    print(f"\n✓ {verb} {ran}/{len(runs)} run(s) "
          f"({len(airports)} airport(s) × {len(modes)} mode(s))  "
          f"[modes={','.join(category_key(t, c) for t, c in modes)}, "
          f"outputs={','.join(args.outputs)}, skip-optimize={args.skip_optimize}]")


if __name__ == "__main__":
    main()
