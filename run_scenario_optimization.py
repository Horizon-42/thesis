#!/usr/bin/env python
"""Optimize prepared flight scenarios, then publish evaluation and comparison outputs.

Run ``prepare_scenario_inputs.py`` first.  This script deliberately does not harvest
tracks or build scenario JSON: its boundary is the prepared scenario dataset.

Steps, chained by shelling out to the existing CLIs:

  1. scenario_optimization          scenarios + observed tracks ─► 4dTrajectory/outputs/<ICAO>/<category>/
       (--reference-tracks, always)    ../shared_references/<target>/*_reference_eval.json
                                        (one canonical set per prepared target dataset)
                                      {*_states.json, *_eval.json, summary.json}
                                        (every eval record points at its reference via
                                         reference_file; failed ones included)
  2. python -m evaluation           eval records ─► <opt_dir>/evaluation_report.json
       (always — the CZML tail consumes its per-flight verdicts + batch metrics)
  3. [eval] python -m evaluation.visualize
                                    eval records ─► <opt_dir>/evaluation_report.html
  4. [czml] build_scenario_comparison_czml (--evaluation-report)
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
  * (omitted)         runs every K-prefixed airport with a prepared scenario JSON.

--skip-optimize reuses an already-computed optimization: if this airport+category
already has a summary.json, step 1 is skipped and only the selected publication
steps run; if it does not exist, optimization runs from scratch.

Usage:
    # first prepare arrivals, observed outputs, and scenario JSON:
    python prepare_scenario_inputs.py --airport KRDU
    # one airport, both outputs (frontend CZML + evaluation report/HTML):
    python run_scenario_optimization.py --airport KRDU --target-type runway --with-constraint
    # one airport, ALL THREE modes (fitted_adsb + runway + runway_cons):
    python run_scenario_optimization.py --airport KRDU
    # trapezoidal-fitting comparison run (default is Hermite-Simpson):
    python run_scenario_optimization.py --airport KRDU --target-type runway --fitting-type trapezoidal
    # custom control mesh (n-segments = unconstrained; n-seg-per-phase = constrained):
    python run_scenario_optimization.py --airport KRDU --n-segments 12 --n-seg-per-phase 4
    # evaluation only:
    python run_scenario_optimization.py --airport KRDU --outputs eval
    # rebuild only the comparison CZML from an existing optimization:
    python run_scenario_optimization.py --airport KRDU --outputs czml --skip-optimize
    # every prepared K-airport, fitted ADS-B crossing target, preview without running:
    python run_scenario_optimization.py --target-type fitted-adsb --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from optimization_run_config import (
    DEFAULT_MAX_DURATION_S,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_ROLLOUT_DT_S,
    build_optimization_config,
)

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

# Reference-artifact naming + cache contract, IMPORTED from the module that writes them:
# this validator's only job is to agree with that writer. Unlike the control-mesh defaults
# below — which live in `collocation.optimizer` and would drag casadi in — `evaluation_export`
# is deliberately casadi-free (numpy only), so the mirror rule does not apply and an import
# is available. REFERENCE_CACHE_SCHEMA is the one true mirror here; a test pins it.
sys.path.insert(0, str(REPO_ROOT / "4dTrajectory" / "optimization"))
from evaluation_export import (  # noqa: E402
    OBSERVED_TRACKS_DIR,
    OBSERVED_TRACK_SUFFIX,
    REFERENCE_EVAL_SUFFIX,
)
REFERENCE_CACHE_SCHEMA = "optimization-references-v3-shared-tracks"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# Control-mesh defaults — MUST mirror CollocationOptimizer's (collocation/optimizer.py:
# DEFAULT_N_SEGMENTS / DEFAULT_N_SEG_PER_PHASE). The pipeline shells out (import-light: no
# casadi), so it cannot import them; kept in sync by this comment.
DEFAULT_N_SEGMENTS = 8         # unconstrained: control segments over the whole trajectory
DEFAULT_N_SEG_PER_PHASE = 3    # constrained: control segments PER procedure leg

# Measured artifact footprint per flight per category on this pipeline (120 KRDU arrivals,
# HS, rollout_dt 0.5 s): records 186 KB (states 108 + eval 42 + reports 36/3) + comparison
# CZML 52 KB, plus 67 KB of observed reference record per prepared TARGET dataset. Used only
# for the pre-flight estimate below — a batch that fills the disk at 90% loses everything it
# has not yet committed, and the run is long enough that finding out at the end is expensive.
_BYTES_PER_FLIGHT_PER_CATEGORY = 238 * 1024
_BYTES_PER_FLIGHT_PER_TARGET = 67 * 1024
_FREE_SPACE_HEADROOM = 1.15


def default_jobs() -> int:
    """Worker processes for a batch that owns the machine.

    ``scenario_optimization``'s own auto is half the cores — right for a library call that
    should leave the box usable, wrong for the pipeline driver, whose whole job is this
    batch. Four threads are still left for the parent's IO and the desktop.
    """
    return max(1, (os.cpu_count() or 4) - 4)

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
    """Every K-airport with at least one prepared scenario dataset."""
    if not SCENARIOS_DIR.exists():
        return []
    return sorted({
        path.name.split("_", 1)[0].upper()
        for path in SCENARIOS_DIR.glob("K*_arrivals*_scenarios.json")
    })


class Plan:
    """The resolved paths + commands for one airport/category run (pure data, so
    it can be previewed with --dry-run or asserted in a test)."""

    def __init__(self, airport: str, target_type: str, with_constraint: bool,
                 outputs: tuple[str, ...], jobs: int = 0,
                 fitting: str = "hs", state_substeps: int | None = None,
                 n_segments: int = DEFAULT_N_SEGMENTS,
                 n_seg_per_phase: int = DEFAULT_N_SEG_PER_PHASE,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS,
                 rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
                 resume: bool = False,
                 max_groups_per_czml: int | None = None) -> None:
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
        # IPOPT iteration cap (see scenario_optimization.DEFAULT_MAX_ITERATIONS): the
        # dominant cost lever, since an unconvergeable scenario pays it in full.
        self.max_iterations = max_iterations
        # Rollout sample step. The simulator array is ~75% of every *_states.json, so this
        # is the disk lever; it is also the resolution the evaluation gates see.
        self.rollout_dt_s = rollout_dt_s
        self.resume = resume
        self.max_groups_per_czml = max_groups_per_czml
        self.optimization_config = build_optimization_config(
            constrained_iaf=with_constraint,
            fitting=fitting,
            n_segments=n_segments,
            n_seg_per_phase=n_seg_per_phase,
            state_substeps=state_substeps,
            max_duration_s=DEFAULT_MAX_DURATION_S,
            rollout_dt_s=rollout_dt_s,
            max_iterations=max_iterations,
            iaf_selection="shortest",
        )
        self.threshold = target_type == "runway"
        self.fitted_adsb = target_type == "fitted-adsb"
        self.category = category_key(target_type, with_constraint)
        self.label = _CATEGORY_LABELS[self.category]

        tag = "_threshold" if self.threshold else "_fitted_adsb"
        self.arrivals_manifest = arrival_manifest_path(self.airport)
        self.scenarios = SCENARIOS_DIR / f"{self.airport}_arrivals{tag}_scenarios.json"
        self.opt_dir = OPT_OUTPUTS_ROOT / self.airport / self.category
        # runway and runway_cons consume the same prepared threshold scenarios, so their
        # observed references are byte-for-byte identical. Keep one sibling anchor.
        reference_target = "runway" if self.threshold else "fitted_adsb"
        self.references_dir = f"../shared_references/{reference_target}"
        self.summary = self.opt_dir / "summary.json"
        self.comparison_dir = (
            COMPARISON_AIRPORTS_ROOT / self.airport / "comparison" / self.category
        )
        self.report = self.opt_dir / "evaluation_report.json"
        self.report_html = self.opt_dir / "evaluation_report.html"

    def optimization_exists(self) -> bool:
        """Whether this airport+category has one complete, internally consistent batch."""
        return self.optimization_reuse_error() is None

    def optimization_reuse_error(self) -> str | None:
        """Explain why ``--skip-optimize`` cannot safely reuse the current batch.

        ``summary.json`` is a roster, not a completion marker. Reuse is allowed only when
        its counts, every eval record, every solved states file, every ``states_ref`` and
        every observed reference pointer agree. This check intentionally avoids loading the
        large state/reference arrays; the evaluation command performs their deep schema
        validation immediately after reuse.
        """
        if not self.summary.is_file():
            return f"missing summary {self.summary}"
        try:
            summary = json.loads(self.summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"unreadable summary: {exc}"
        if not isinstance(summary, dict):
            return "summary is not an object"
        if summary.get("optimization_config") != self.optimization_config:
            return "summary optimization configuration does not match this run"
        rows = summary.get("results")
        if not isinstance(rows, list):
            return "summary has no results roster"
        counts = {key: summary.get(key) for key in ("total", "solved", "failed")}
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        ):
            return f"summary has invalid counts {counts}"
        if counts["total"] == 0 or len(rows) != counts["total"]:
            return (
                f"summary roster length {len(rows)} disagrees with total "
                f"{counts['total']}"
            )

        status_counts = {"solved": 0, "failed": 0}
        seen_eval: set[str] = set()
        root = self.opt_dir.resolve()
        reference_manifests: dict[Path, dict[str, Any]] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                return f"summary result {index} is not an object"
            status = row.get("status")
            if status not in status_counts:
                return f"summary result {index} has invalid status {status!r}"
            status_counts[status] += 1

            eval_name = row.get("eval_file")
            if not isinstance(eval_name, str) or not eval_name:
                return f"summary result {index} lacks eval_file"
            if eval_name in seen_eval:
                return f"summary lists duplicate eval_file {eval_name!r}"
            seen_eval.add(eval_name)
            eval_path = (root / eval_name).resolve()
            if not eval_path.is_relative_to(root) or not eval_path.is_file():
                return f"missing or unsafe eval_file {eval_name!r}"
            try:
                evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return f"unreadable eval_file {eval_name!r}: {exc}"
            if not isinstance(evaluation, dict):
                return f"eval_file {eval_name!r} is not an object"

            source = evaluation.get("source")
            if not isinstance(source, dict):
                return f"eval_file {eval_name!r} lacks source identity"
            for key in ("id", "runway", "icao24", "landing_time_utc"):
                if row.get(key) != source.get(key):
                    return f"eval_file {eval_name!r} disagrees on source {key}"

            reference_name = evaluation.get("reference_file")
            if not isinstance(reference_name, str) or not reference_name:
                return f"eval_file {eval_name!r} lacks reference_file"
            reference_path = (eval_path.parent / reference_name).resolve()
            if not reference_path.is_file():
                return f"missing reference_file {reference_name!r}"
            cache_path = reference_path.parent / "manifest.json"
            cache = reference_manifests.get(cache_path)
            if cache is None:
                try:
                    cache = json.loads(cache_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    return f"missing or unreadable reference cache manifest {cache_path}: {exc}"
                if (
                    not isinstance(cache, dict)
                    or cache.get("schema_version") != REFERENCE_CACHE_SCHEMA
                    or not isinstance(cache.get("records"), list)
                ):
                    return f"reference cache manifest {cache_path} lacks SHA-256 contract"
                signature = cache.get("source_signature")
                if not isinstance(signature, dict):
                    return f"reference cache manifest {cache_path} lacks source_signature"
                if self.scenarios.is_file() and (
                    signature.get("scenarios_sha256") != _file_sha256(self.scenarios)
                ):
                    return f"reference cache {cache_path} does not match prepared scenarios"
                if self.arrivals_manifest.is_file() and (
                    signature.get("arrivals_manifest_sha256")
                    != _file_sha256(self.arrivals_manifest)
                ):
                    return f"reference cache {cache_path} does not match arrival manifest"
                reference_manifests[cache_path] = cache
            cached_row = next(
                (
                    cached
                    for cached in cache["records"]
                    if isinstance(cached, dict)
                    and cached.get("file") == reference_path.name
                ),
                None,
            )
            if cached_row is None:
                return f"reference cache does not roster {reference_path.name!r}"
            cached_identity = cached_row.get("identity")
            if not isinstance(cached_identity, dict) or any(
                cached_identity.get(key) != source.get(key)
                for key in ("id", "runway", "icao24", "landing_time_utc")
            ):
                return f"reference cache identity disagrees for {reference_path.name!r}"
            expected_hash = cached_row.get("sha256")
            if (
                not isinstance(expected_hash, str)
                or _file_sha256(reference_path) != expected_hash
            ):
                return f"reference_file {reference_name!r} failed SHA-256 validation"
            # The record quotes its observed states from the shared track store, so the
            # store is part of what "this reference is intact" means.
            track_hash = cached_row.get("track_sha256")
            track_path = (
                reference_path.parent.parent
                / OBSERVED_TRACKS_DIR
                / (reference_path.name.removesuffix(REFERENCE_EVAL_SUFFIX)
                   + OBSERVED_TRACK_SUFFIX)
            )
            if (
                not isinstance(track_hash, str)
                or not track_path.is_file()
                or _file_sha256(track_path) != track_hash
            ):
                return f"observed track for {reference_name!r} failed SHA-256 validation"

            states_name = row.get("states_file")
            states_ref = evaluation.get("states_ref")
            if status == "solved":
                if not isinstance(states_name, str) or not states_name:
                    return f"solved result {index} lacks states_file"
                states_path = (root / states_name).resolve()
                if not states_path.is_relative_to(root) or not states_path.is_file():
                    return f"missing or unsafe states_file {states_name!r}"
                if not isinstance(states_ref, dict):
                    return f"eval_file {eval_name!r} lacks states_ref"
                ref_name = states_ref.get("file")
                if (
                    not isinstance(ref_name, str)
                    or (eval_path.parent / ref_name).resolve() != states_path
                    or states_ref.get("key") != "simulator_states"
                ):
                    return f"eval_file {eval_name!r} has inconsistent states_ref"
                if evaluation.get("states") != [] or evaluation.get("final_time_s") is None:
                    return f"eval_file {eval_name!r} has invalid solved-state metadata"
            elif (
                states_name is not None
                or states_ref is not None
                or evaluation.get("states") != []
                or evaluation.get("final_time_s") is not None
            ):
                return f"eval_file {eval_name!r} has invalid failed-state metadata"

        if (
            status_counts["solved"] != counts["solved"]
            or status_counts["failed"] != counts["failed"]
            or counts["solved"] + counts["failed"] != counts["total"]
        ):
            return f"summary status counts disagree: {status_counts} vs {counts}"
        return None

    def steps(self, *, reuse: bool = False) -> list[tuple[str, list[str]]]:
        """The commands to run; ``reuse`` drops optimization and keeps publication."""
        py = sys.executable
        named: list[tuple[str, list[str]]] = []

        if not reuse:
            # --reference-tracks makes the optimization CLI write the reference eval
            # records FIRST in the target dataset's shared sibling directory and point
            # every eval record at its reference. runway/runway_cons therefore reuse the
            # same byte-for-byte reference set.
            optimize_cmd = [
                py, str(OPT_SCRIPT),
                "--scenarios", str(self.scenarios),
                "--output-dir", str(self.opt_dir),
                "--reference-tracks", str(self.arrivals_manifest),
                "--references-dir", self.references_dir,
                "--jobs", str(self.jobs),
                "--fitting", self.fitting,
                "--max-iterations", str(self.max_iterations),
                "--rollout-dt", str(self.rollout_dt_s),
            ]
            if self.resume:
                optimize_cmd.append("--resume")
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

        # The report is part of the committed comparison generation, so evaluation always
        # runs before either output tail. A reused batch has already passed
        # optimization_reuse_error(), including its complete eval roster.
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
            if self.max_groups_per_czml is not None:
                comparison_cmd += [
                    "--max-groups-per-czml", str(self.max_groups_per_czml)
                ]
            comparison_cmd += ["--evaluation-report", str(self.report)]
            # Feed the scenario initial states so the index carries V + mass for EVERY
            # flight — including failed optimizations (which have no states file). A
            # reused archived optimization may no longer have its prepared scenario file.
            if not reuse or self.scenarios.exists():
                comparison_cmd += ["--scenarios", str(self.scenarios)]
            named.append(("comparison CZML", comparison_cmd))

        total = len(named)
        return [(f"{i}/{total} {name}", cmd) for i, (name, cmd) in enumerate(named, 1)]


def scenario_count(path: Path) -> int:
    """How many scenarios a prepared dataset holds (0 when it is not there yet)."""
    if not path.is_file():
        return 0
    try:
        return len(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError):
        return 0


def estimate_footprint_bytes(plans: list["Plan"]) -> int:
    """Bytes the selected runs will add, from the measured per-flight artifact sizes."""
    per_target: dict[Path, int] = {}
    total = 0
    for plan in plans:
        flights = scenario_count(plan.scenarios)
        total += flights * _BYTES_PER_FLIGHT_PER_CATEGORY
        per_target[plan.scenarios] = flights
    total += sum(per_target.values()) * _BYTES_PER_FLIGHT_PER_TARGET
    return total


def check_free_space(plans: list["Plan"]) -> tuple[bool, str]:
    """``(ok, message)`` — will the selected runs fit, with headroom?

    Checked before the first solve because the failure mode is otherwise the worst one
    available: tens of hours of compute, then a partially written record when the
    filesystem fills, in a directory whose ``summary.json`` never gets written.
    """
    needed = int(estimate_footprint_bytes(plans) * _FREE_SPACE_HEADROOM)
    free = shutil.disk_usage(REPO_ROOT).free
    gib = 1024 ** 3
    message = (
        f"estimated new artifacts {needed / gib:.1f} GiB "
        f"(incl. {(_FREE_SPACE_HEADROOM - 1) * 100:.0f}% headroom) "
        f"vs {free / gib:.1f} GiB free on {REPO_ROOT}"
    )
    return needed <= free, message


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
    state_substeps: int | None = None,
    n_segments: int = DEFAULT_N_SEGMENTS,
    n_seg_per_phase: int = DEFAULT_N_SEG_PER_PHASE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
    resume: bool = False,
    max_groups_per_czml: int | None = None,
) -> bool:
    """Run (or preview) the pipeline for one airport. Returns True if it ran /
    would run, False if it was skipped (missing input and nothing to reuse).

    Both the scenario JSON and arrival manifest are required prepared inputs."""
    plan = Plan(airport, target_type, with_constraint, outputs, jobs=jobs, fitting=fitting,
                state_substeps=state_substeps, n_segments=n_segments,
                n_seg_per_phase=n_seg_per_phase, max_iterations=max_iterations,
                rollout_dt_s=rollout_dt_s, resume=resume,
                max_groups_per_czml=max_groups_per_czml)
    reuse_error = plan.optimization_reuse_error() if skip_optimize else None
    reuse = skip_optimize and reuse_error is None

    mode = "reuse optimization" if reuse else "optimize prepared scenarios"
    fit = "" if reuse else f"  ·  fitting: {plan.fitting}"
    print(f"\n━━ {plan.airport}  [{plan.category}]  ·  {mode}{fit}  ·  outputs: {', '.join(outputs)}")
    print(f"   scenarios : {plan.scenarios}")
    print(f"   states    : {plan.opt_dir}")
    if "czml" in outputs:
        print(f"   comparison: {plan.comparison_dir}")
    if "eval" in outputs:
        print(f"   report    : {plan.report}")

    if not reuse:
        missing = [
            path for path in (plan.scenarios, plan.arrivals_manifest) if not path.exists()
        ]
        if missing:
            print("   ⚠ skip: missing prepared input(s): "
                  + ", ".join(str(path) for path in missing))
            print(f"   run: {sys.executable} prepare_scenario_inputs.py "
                  f"--airport {plan.airport}")
            return False
        if skip_optimize:
            print(f"   (existing optimization is not reusable: {reuse_error} "
                  "→ running from scratch)")
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
        help="airport ICAO; omit to run every K-prefixed airport with prepared scenarios",
    )
    parser.add_argument(
        "--target-type", choices=TARGET_TYPES, default=None,
        help="target state: 'runway' = the published runway threshold; 'fitted-adsb' = "
             "the final_approach OLS threshold crossing (fitted position + fitted "
             "approach kinematics). OMIT to run all three modes "
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
        help=f"parallel optimizer worker processes (passed to scenario_optimization --jobs; "
             f"0 = auto = CPU cores - 4 = {default_jobs()} here, 1 = serial). The library's "
             f"own auto is half the cores; this driver owns the machine for the batch",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"IPOPT iteration cap per solve (default {DEFAULT_MAX_ITERATIONS}). The single "
             "biggest cost lever: a scenario that will not converge pays the cap in full "
             "(measured ~13x a successful solve, ~48%% of an unconstrained batch's CPU for "
             "6.7%% of its flights). Lowering it trades slow successes for failures, so it "
             "is recorded in summary.json and --skip-optimize will not reuse across a change",
    )
    parser.add_argument(
        "--rollout-dt", type=float, default=DEFAULT_ROLLOUT_DT_S, metavar="SECONDS",
        help=f"true-dynamics rollout sample step (default {DEFAULT_ROLLOUT_DT_S}). The "
             "simulator array is ~75%% of every *_states.json, so doubling this roughly "
             "halves the on-disk footprint — and coarsens the states the gates measure",
    )
    parser.add_argument(
        "--max-groups-per-czml", type=int, default=None, metavar="N",
        help="split each runway's comparison CZML into files of at most N flight groups "
             "(omit for one file per runway). The frontend loads CZMLs named by the index's "
             "per-group 'czml' field, so splitting is transparent to it — and a 2000-flight "
             "runway is otherwise a single ~100 MB file no browser will parse",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse per-flight records already on disk for scenarios in the current roster "
             "and solve only the rest (passed to scenario_optimization --resume). For a "
             "multi-hour batch this is what makes an interrupted run cheap to restart",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="keep going when one airport/category fails, and report the failures at the "
             "end (exit 1). Without it the first failure aborts the whole sweep",
    )
    parser.add_argument(
        "--skip-space-check", action="store_true",
        help="run even when the estimated artifact footprint exceeds the free disk space",
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
        "--skip-optimize", action="store_true",
        help="if this airport+category already has an optimization result "
             "(summary.json), skip optimization and only (re)build the selected outputs; "
             "otherwise optimize the prepared scenario input",
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
    if args.max_iterations < 1:
        parser.error(f"--max-iterations must be >= 1, got {args.max_iterations}")
    if args.rollout_dt <= 0.0:
        parser.error(f"--rollout-dt must be > 0, got {args.rollout_dt}")
    if args.max_groups_per_czml is not None and args.max_groups_per_czml < 1:
        parser.error(
            f"--max-groups-per-czml must be >= 1, got {args.max_groups_per_czml}"
        )
    jobs = args.jobs if args.jobs else default_jobs()

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
                f"no K-prefixed airports with prepared scenario JSON under "
                f"{SCENARIOS_DIR} — run prepare_scenario_inputs.py first"
            )
        print(f"no --airport given → running {len(airports)} K-airport(s): "
              f"{', '.join(airports)}")

    runs = [(airport, mode) for airport in airports for mode in modes]
    settings = dict(
        jobs=jobs, fitting=args.fitting_type, state_substeps=args.state_substeps,
        n_segments=args.n_segments, n_seg_per_phase=args.n_seg_per_phase,
        max_iterations=args.max_iterations, rollout_dt_s=args.rollout_dt,
        resume=args.resume, max_groups_per_czml=args.max_groups_per_czml,
    )
    plans = [
        Plan(airport, target_type, with_constraint, tuple(args.outputs), **settings)
        for airport in airports
        for target_type, with_constraint in modes
    ]
    fits, space = check_free_space(plans)
    print(f"\ndisk: {space}")
    if not fits and not args.dry_run:
        if not args.skip_space_check:
            parser.error(
                f"not enough free disk space — {space}. Lower the population "
                "(prepare_scenario_inputs.py --max-per-runway), raise --rollout-dt, drop "
                "an output with --outputs, free space, or pass --skip-space-check"
            )
        print("⚠ continuing past the space check (--skip-space-check)")
    print(f"jobs: {jobs} worker process(es)  ·  max-iterations: {args.max_iterations}"
          f"  ·  rollout-dt: {args.rollout_dt}s")

    ran = 0
    failures: list[tuple[str, str]] = []
    for airport in airports:
        for target_type, with_constraint in modes:
            cell = f"{airport} [{category_key(target_type, with_constraint)}]"
            try:
                if run_for_airport(
                    airport, target_type, with_constraint, tuple(args.outputs),
                    dry_run=args.dry_run, skip_optimize=args.skip_optimize,
                    **settings,
                ):
                    ran += 1
            except subprocess.CalledProcessError as exc:
                # One cell failing used to abort the sweep and discard every airport after
                # it. Each cell is an independent batch, so the default is still to stop
                # (a broken recipe should not burn hours proving it), but --continue-on-error
                # keeps the rest and names the casualties at the end.
                failed_step = " ".join(exc.cmd) if isinstance(exc.cmd, list) else str(exc.cmd)
                print(f"\n✗ {cell} failed (exit {exc.returncode}): {failed_step}")
                failures.append((cell, f"exit {exc.returncode}"))
                if not args.continue_on_error:
                    raise

    verb = "previewed" if args.dry_run else "completed"
    print(f"\n✓ {verb} {ran}/{len(runs)} run(s) "
          f"({len(airports)} airport(s) × {len(modes)} mode(s))  "
          f"[modes={','.join(category_key(t, c) for t, c in modes)}, "
          f"outputs={','.join(args.outputs)}, skip-optimize={args.skip_optimize}]")
    if failures:
        print(f"✗ {len(failures)} run(s) failed:")
        for cell, why in failures:
            print(f"    {cell}: {why}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
