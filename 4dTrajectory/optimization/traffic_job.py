"""One interactive multi-aircraft job (design §10.3, step T11): fly one arrival in its recorded traffic (M1) or one
block of arrivals (M2), then evaluate and build the comparison files the viewer reads.

    python 4dTrajectory/optimization/traffic_job.py --spec <spec.json> --out <job dir>

``spec.json`` is ``{"mode": "m1", "airport", "manifest", "flightKey"}`` or ``{"mode": "m2", "airport",
"manifest", "blockStartUtc", "blockS"}``; ``manifest`` is the airport's ``arrivals/manifest.json``. The run is the
batch's own (``traffic_optimization``: the same scenarios, workers, writers, configuration and defaults, IM2), with
two differences: it reads only the tracks that can share the flight's window or the block's (IM5, the roster
filter of ``traffic_roster``), and it writes one directory (``traffic_job_files``) and registers no category.
``progress.json`` follows each aircraft and says which phase the job is in; ``state.json`` is the last write: ``done`` with
the readout of ``traffic.readout`` (the outcome counts; for M2 the delays and the losses left after the block's final check),
``perAircraft`` (what changed for each aircraft, and the time spent on it) and ``timing`` (the job's wall time by stage), all in
``traffic_job_files``, or ``failed`` with the reason.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_OPT_DIR = Path(__file__).resolve().parent
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))

import traffic_roster as roster  # noqa: E402
from evaluation_export import EVAL_SUFFIX, STATES_SUFFIX  # noqa: E402
from flight_scenarios.build import build_scenario  # noqa: E402
from flight_scenarios.dataset import load_model_arrivals_subset  # noqa: E402
from flight_scenarios.identity import flight_key  # noqa: E402
from optimization_run_config import DEFAULT_MAX_DURATION_S, DEFAULT_ROLLOUT_DT_S  # noqa: E402
from scenario_batch import run_batch, sidecar_filename  # noqa: E402
from scenario_optimization import DEFAULT_MAX_ITERATIONS, DEFAULT_PROCEDURE_ROOT  # noqa: E402
from traffic import M1_MODE, M2_MODE  # noqa: E402
from traffic.loop import LoopSettings  # noqa: E402
from traffic.readout import blocks_readout, readout  # noqa: E402
from traffic.scene import Traffic, traffic_from_arrivals  # noqa: E402
from traffic_job_files import (  # noqa: E402
    BLOCK_LENGTHS_S,
    COMPARISON_DIR,
    MODE_M1,
    MODE_M2,
    PHASE_EVALUATION,
    PHASE_READING,
    PHASE_SCENE,
    PROGRESS_FILE,
    RECORDS_DIR,
    REPORT_FILE,
    STATE_DONE,
    STATE_FAILED,
    STATE_FILE,
    phase_flying,
    phase_stage,
    write_json_atomic,
)
from traffic_optimization import (  # noqa: E402
    TRAFFIC_SUFFIX,
    _fly_one_block,   # the batch's own workers, not copies of them
    _fly_one_window,
    block_scenarios,
    m1_traffic_config,
    m2_traffic_config,
    run_optimization_config,
    window_scenario,
    worker_params,
    write_block,
    write_m2_summary,
)
from trajectory_data_process.harvest.utc import parse_iso_utc_s  # noqa: E402

CZML_SCRIPT = _REPO_ROOT / "aeroviz-4d" / "python" / "build_scenario_comparison_czml.py"
BLOCK_LABEL = "block_00"


def run_step(command: list[str]) -> None:
    """One subprocess of the job (evaluation, comparison builder), run from the repo root; its failure fails the job."""
    subprocess.run(command, cwd=_REPO_ROOT, check=True)


class Flown(NamedTuple):
    """What a flown job hands back: its readout, the types of the aircraft it controlled, and — M2 — the arrivals of the block it
    could not control (no aircraft dynamics model), which flew their records (``stayedRecords`` of ``state.json``)."""

    readout: dict[str, Any]
    types: dict[str, str | None]
    stayed: dict[str, dict[str, str | None]]


class Progress:
    """``progress.json`` of one job: the aircraft settled (``total`` is ``None`` until the job has chosen them), the one
    last finished (its ``flight_key``, ``None`` before the first) and the phase. Every change is written at once.

    It also keeps the job's clock (``clock``, monotonic seconds): the wall time of each stage (``phase_stage``); ``timing``
    closes the books. (The time spent on each AIRCRAFT is the solver's: the solves its sidecar lists, ``per_aircraft``.)"""

    def __init__(self, out: Path, clock: Callable[[], float] = time.monotonic) -> None:
        self._out = out
        self._clock = clock
        self._started = clock()
        self._fields: dict[str, Any] = {"done": 0, "total": None, "current": None, "phase": PHASE_READING}
        self._stage, self._stage_since = phase_stage(PHASE_READING), self._started
        self._stage_s: dict[str, float] = defaultdict(float)
        self._write()

    def _write(self) -> None:
        write_json_atomic(self._out / PROGRESS_FILE, self._fields)

    def phase(self, phase: str) -> None:
        now = self._clock()
        stage = phase_stage(phase)
        if stage != self._stage:                      # "optimizing 1 of 2" then "optimizing 2 of 2" is one stage
            self._stage_s[self._stage] += now - self._stage_since
            self._stage, self._stage_since = stage, now
        self._fields["phase"] = phase
        self._write()

    def finished(self, done: int, total: int, current: str | None) -> None:
        self._fields.update(done=done, total=total, current=current)
        self._write()

    def timing(self) -> dict[str, Any]:
        """``{totalS, phases}`` of ``traffic_job_files``: call it when the job is done (it closes the stage in progress)."""
        now = self._clock()
        phases = {**self._stage_s, self._stage: self._stage_s[self._stage] + now - self._stage_since}
        return {"totalS": round(now - self._started, 1), "phases": {name: round(s, 1) for name, s in phases.items()}}


def airport_traffic(manifest: Path, rows: list[roster.RosterRow], flights: list[dict[str, Any]]) -> Traffic:
    """The recorded traffic of the loaded ``flights``, carrying the runway targets of the WHOLE airport as the batch's
    does (``rules.separation``, the runway frames and the sidecar's ``runways_without_faf`` read all of them): for a
    runway none of the loaded flights lands on, the target of one arrival of the roster, read for that alone. The
    targets come in roster order, as the batch's."""
    traffic = traffic_from_arrivals(flights)
    absent: dict[str, str] = {}
    for r in rows:
        if r.runway not in traffic.runway_targets:
            absent.setdefault(r.runway, r.flight_key)
    if not absent:
        return traffic
    others = {f["runway"]: f["runway_target"] for f in load_model_arrivals_subset(manifest, list(absent.values()))}
    targets = {**others, **traffic.runway_targets}
    ordered = {runway: targets[runway] for runway in dict.fromkeys(r.runway for r in rows)}
    return Traffic(traffic.airport, traffic.flights, ordered)


def _fly_m1(spec: dict[str, Any], rows: list[roster.RosterRow], progress: Progress, records: Path, config: dict[str, Any],
            params: dict[str, Any], settings: LoopSettings) -> Flown:
    manifest, airport = Path(spec["manifest"]), spec["airport"]
    own = next((r for r in rows if r.flight_key == spec["flightKey"]), None)
    if own is None:
        raise ValueError(f"flight {spec['flightKey']!r} is not in {manifest}")
    near = roster.m1_near(rows, own, DEFAULT_MAX_DURATION_S)
    flights = load_model_arrivals_subset(manifest, [r.flight_key for r in near])
    traffic = airport_traffic(manifest, rows, flights)
    flight = next(f for f in flights if flight_key(f, 0) == own.flight_key)
    scenario = window_scenario(build_scenario(flight, airport=airport, target_from_threshold=True), traffic,
                               DEFAULT_MAX_DURATION_S)
    config["traffic"] = m1_traffic_config(settings, {
        "manifest": str(manifest), "population": len(rows), "sample": 1, "flight_key": own.flight_key,
        "traffic_tracks_read": len(flights)})
    progress.finished(0, 1, None)
    progress.phase(phase_flying(1, 1))

    def worker(payload):
        found = _fly_one_window(payload)
        progress.finished(1, 1, found[1])
        return found

    run_batch([scenario], output_dir=records, worker=worker, params=params, optimization_config=config,
              mode=M1_MODE, progress=" [traffic job]", jobs=1, scenarios_label=f"{airport} {own.flight_key}",
              references_dir=None, resume=False, sidecar_suffix=TRAFFIC_SUFFIX)
    return Flown(readout(records), {own.flight_key: traffic.flight(own.flight_key).typecode}, {})


def _fly_m2(spec: dict[str, Any], rows: list[roster.RosterRow], progress: Progress, records: Path, config: dict[str, Any],
            params: dict[str, Any], settings: LoopSettings) -> Flown:
    manifest, airport = Path(spec["manifest"]), spec["airport"]
    block_s = spec["blockS"]
    if block_s not in BLOCK_LENGTHS_S:
        raise ValueError(f"block length {block_s!r} s: one of {BLOCK_LENGTHS_S}")
    start = parse_iso_utc_s(spec["blockStartUtc"])
    block = roster.landing_in(rows, start, start + block_s)
    if not block:
        raise ValueError(f"no arrival lands in {block_s} s from {spec['blockStartUtc']}")
    near = roster.m2_near(rows, block, DEFAULT_MAX_DURATION_S)
    flights = load_model_arrivals_subset(manifest, [r.flight_key for r in near])
    traffic = airport_traffic(manifest, rows, flights)
    scenarios, block_traffic, no_dynamics = block_scenarios(flights, traffic, airport, start, start + block_s,
                                                            DEFAULT_MAX_DURATION_S)
    if not scenarios:
        raise ValueError(f"none of the {len(block)} arrival(s) in the block has aircraft dynamics")
    config["traffic"] = m2_traffic_config(settings, manifest, spec["blockStartUtc"], block_s, 1)
    progress.finished(0, len(scenarios), None)
    label, flown, summary = _fly_one_block(
        (BLOCK_LABEL, scenarios, block_traffic, params),
        on_progress=progress.finished, on_phase=progress.phase)
    if "error" in summary:
        raise RuntimeError(f"the block failed: {summary['error']}")
    rows_by_label = {label: write_block(records, flown, config)}
    write_m2_summary(records, config, rows_by_label, {label: {"label": label, **summary,
                                                              "skipped_no_dynamics": no_dynamics}})
    controlled = {s.source["flight_key"] for s in scenarios}
    stayed = {r.flight_key: {"callsign": r.callsign, "type": traffic.flight(r.flight_key).typecode}
              for r in block if r.flight_key not in controlled}
    return Flown(blocks_readout(records), {key: traffic.flight(key).typecode for key in controlled}, stayed)


def _sidecar_of(records: Path, key: str) -> dict[str, Any]:
    """An aircraft's traffic sidecar: its record's, or — for a scenario without a record — its failed sidecar (named from its
    flight key like the record it stands beside)."""
    return json.loads((records / sidecar_filename(key + STATES_SUFFIX, TRAFFIC_SUFFIX)).read_text(encoding="utf-8"))


def per_aircraft(records: Path, rows: list[roster.RosterRow], types: dict[str, str | None], reading: str,
                 ) -> dict[str, dict[str, Any]]:
    """``state.json``'s ``perAircraft`` (``traffic_job_files``): per aircraft the job was to control, its type, the loss
    instants it answers for in its first solve and in the solve its loop KEPT as the record (MD14: the one with the fewest counted
    loss instants — the loop's own count; shown as the census counts: ``answered_loss_instants`` of ``rounds[0]`` and of
    ``rounds[kept_round]``, MD10 not applied), its record's landing against the recorded flight's (the record's own flight time:
    the kept solve), its slot's delay, its losses in the block's final check, and the solves spent on it — their wall and CPU
    time, their count, the failed ones — summed from the solver's own timing: the ``solves`` of its sidecar, or of its failed
    sidecar when it has no record (every aircraft has one or the other; a failed sidecar that lost its solves says so with a
    null, never 0 s). Read from ``summary.json`` and the records' sidecars; ``types`` are the recorded traffic's and ``reading``
    the one the job's loop judges by (``LoopSettings.reading``: ``block_final`` is keyed by it)."""
    summary = json.loads((records / "summary.json").read_text(encoding="utf-8"))
    is_block = summary["mode"] == M2_MODE
    delays = {slot["flight_key"]: slot["delay_s"] for block in summary["blocks"] for slot in block["slots"]} if is_block else {}
    recorded_s = {r.flight_key: r.landing_utc_s - r.entry_utc_s for r in rows}
    found: dict[str, dict[str, Any]] = {}
    for row in summary["results"]:
        key = row["eval_file"].removesuffix(EVAL_SUFFIX)       # the record's name is its flight key's (every row has one)
        side = _sidecar_of(records, key)
        solves = side["solves"]
        entry = {"type": types[key], "firstSolveLosses": None, "finalLosses": None, "landingVsRecordS": None,
                 "delayS": delays[key] if key in delays else None, "blockCheckLosses": None,
                 "optimizeS": None, "optimizeCpuS": None, "solves": None, "failedSolves": None}
        if solves is not None:
            entry.update(optimizeS=round(sum(s["wallS"] for s in solves), 1), optimizeCpuS=round(sum(s["cpuS"] for s in solves), 1),
                         solves=len(solves), failedSolves=sum(not s["ok"] for s in solves))
        if row["status"] == "solved":
            rounds = side["rounds"]
            entry.update(firstSolveLosses=rounds[0]["answered_loss_instants"],
                         finalLosses=rounds[side["kept_round"]]["answered_loss_instants"],
                         landingVsRecordS=round(row["final_time_s"] - recorded_s[key], 1))
            if is_block:
                entry.update(delayS=side["slot"]["delay_s"], blockCheckLosses=side["block_final"][reading]["answered"])
        found[key] = entry
    return found


def run_job(spec: dict[str, Any], out: Path, *, max_iterations: int = DEFAULT_MAX_ITERATIONS,
            clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Fly ``spec`` into ``out`` (records, evaluation report, comparison files) and return what a ``done`` ``state.json``
    holds beyond its state: ``summary`` (the readout), ``perAircraft``, ``timing`` and ``stayedRecords`` (``traffic_job_files``)."""
    mode = spec["mode"]
    if mode not in (MODE_M1, MODE_M2):
        raise ValueError(f"job mode {mode!r}: one of {MODE_M1!r}, {MODE_M2!r}")
    settings = LoopSettings()
    config = run_optimization_config(max_duration_s=DEFAULT_MAX_DURATION_S, rollout_dt_s=DEFAULT_ROLLOUT_DT_S,
                                     max_iterations=max_iterations)
    params = worker_params(procedure_root=DEFAULT_PROCEDURE_ROOT, settings=settings,
                           max_duration=DEFAULT_MAX_DURATION_S, rollout_dt_s=DEFAULT_ROLLOUT_DT_S,
                           max_iterations=max_iterations)
    records = out / RECORDS_DIR
    records.mkdir(parents=True)
    progress = Progress(out, clock)
    rows = roster.read_roster(spec["manifest"])
    fly = _fly_m1 if mode == MODE_M1 else _fly_m2
    flown = fly(spec, rows, progress, records, config, params, settings)
    progress.phase(PHASE_EVALUATION)
    report = records / REPORT_FILE
    run_step([sys.executable, "-m", "evaluation", "--input", str(records), "--output", str(report)])
    progress.phase(PHASE_SCENE)
    # No --category: the builder then registers nothing in a categories.json; the job directory is not a publication.
    run_step([sys.executable, str(CZML_SCRIPT), "--summary", str(records / "summary.json"),
              "--output-dir", str(out / COMPARISON_DIR), "--airport", spec["airport"],
              "--evaluation-report", str(report)])
    return {"summary": flown.readout, "perAircraft": per_aircraft(records, rows, flown.types, settings.reading),
            "timing": progress.timing(), "stayedRecords": flown.stayed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One interactive multi-aircraft job")
    parser.add_argument("--spec", required=True, help="the job's spec.json")
    parser.add_argument("--out", required=True, help="the job directory (empty: the job creates its contents)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        result = run_job(json.loads(Path(args.spec).read_text(encoding="utf-8")), out)
    except Exception as exc:  # noqa: BLE001 — the job boundary: whatever stopped it is the reason in state.json
        traceback.print_exc()
        write_json_atomic(out / STATE_FILE, {
            "state": STATE_FAILED, "error": f"{type(exc).__name__}: {str(exc).partition(chr(10))[0][:300]}"})
        return 1
    write_json_atomic(out / STATE_FILE, {"state": STATE_DONE, "error": None, **result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
