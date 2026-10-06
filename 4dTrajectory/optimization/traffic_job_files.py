"""The layout of one interactive traffic job's directory (design §10.3), shared by the job process
(``traffic_job.py``) and the backend that starts and serves it (``aeroviz_backend/traffic_jobs.py``).
Standard library only: the backend imports it.

    <job>/spec.json          what to fly (written by the backend)
    <job>/process.json       {pgid, startTime, bootId} (written by the backend): the job's process group (its leader's
                             pid), the leader's start (/proc/<pid>/stat field 22) and the boot id — a group number is
                             reused, so a group is the job's only while its leader has this start and this boot
    <job>/job.log            the job process's output (the backend redirects it here)
    <job>/progress.json      {done, total, current, phase}: how many aircraft are settled (``total`` null until the job has
                             chosen them), the aircraft last finished, and what the job is doing (``PHASE_*`` below)
    <job>/records/           the batch records, summary.json, evaluation_report.json
    <job>/comparison/        the comparison builder's files: comparison_index.json, the CZML, the report
    <job>/state.json         the job's last write, {state, error, [summary, perAircraft, timing, stayedRecords]}: done (with its
                             readout, ``perAircraft``, ``timing`` and ``stayedRecords``) or failed (with the reason); the backend writes cancelled

``perAircraft`` (a ``done`` state's, written by ``traffic_job.per_aircraft`` from the records' sidecars) is a map
``{flight_key: {type, firstSolveLosses, finalLosses, landingVsRecordS, delayS, blockCheckLosses, optimizeS, optimizeCpuS,
solves, failedSolves}}``, one
entry per aircraft the job was to control (an M1 job: its one; an M2 job: every arrival of the block that has a dynamics
model, flown or not):

- ``type``: its ICAO type designator as the recorded traffic carries it (the census lists the same); null when untyped;
- ``firstSolveLosses`` / ``finalLosses``: the loss instants (distinct check instants, as the census counts them: step
  ``LoopSettings.step_s``, the VISUAL reading, MD10 NOT applied — the same kind of count as the catalog's ``lossInstants`` of
  the aircraft's record) the aircraft answers for in the traffic loop's first solve (``answered_loss_instants`` of ``rounds[0]``
  of its sidecar) and in the solve the loop KEPT as the record (MD14: the one with the fewest COUNTED loss instants — the loop's
  own count, ``counted_loss_instants``, MD10 applied, which is not shown — the earliest on a tie: the round its sidecar's
  ``kept_round`` names, ``answered_loss_instants`` of it); null when it was not flown (no record: its baseline or slot solve
  failed);
- ``landingVsRecordS``: its record's flight time (``final_time_s`` of the record: the kept solve) less the recorded flight's
  (roster landing less roster entry), in seconds — plus: it lands later; null when not flown;
- ``delayS``: M2 only — the slot's delay (the schedule's CTA less the aircraft's ETA, s) for an aircraft the schedule
  gave a slot, flown or not; null for an M1 job and for an aircraft that has no slot (its ETA solve failed);
- ``blockCheckLosses``: M2 only — the conflicts the aircraft answers for in the block's final check against every other
  aircraft (``sidecar["block_final"][reading]["answered"]``, ``reading`` the job's ``LoopSettings.reading``, VISUAL: a count
  of conflicts, not of instants; the aircraft with one are those ``traffic.readout.blocks_readout`` counts); null for an M1
  job and when not flown;
- ``optimizeS`` / ``optimizeCpuS`` / ``solves`` / ``failedSolves``: what the job's solver spent on the aircraft, summed from the
  solver's own timing — the ``solves`` its traffic sidecar lists (``scenario_optimization.SolveTime``, one per ``solve_iaf``
  call: kind, IAF, ok, wall and process CPU time), or, for an aircraft without a record, the ``solves`` of its failed sidecar
  (``optimization-traffic-failed-v1``): wall seconds, CPU seconds, how many solves, how many of them failed (``ok`` false). Every
  aircraft has them, an M2 aircraft's including its earliest-arrival solves, which come first, and every IAF the search tried —
  EXCEPT an aircraft whose failed sidecar lost its solves (``solves`` null: a failure outside the solve, e.g. a casadi error in
  a re-solve, or a block that raised): its four are null, "not recorded", never 0 s;

``stayedRecords`` (a ``done`` state's) is a map ``{flight_key: {callsign, type}}`` of the arrivals of an M2 job's block that it
could NOT control — they have no aircraft dynamics model, so they flew their records (``{}`` for an M1 job); ``callsign`` and
``type`` are null when the roster / the recorded traffic has none.

``timing`` (a ``done`` state's) is ``{totalS, phases: {stage: seconds}}``: the job's wall time from its first progress
write to the end of the comparison builder, and its split into the stages that happened, in order (``STAGE_*`` below: "reading
traffic", "earliest arrivals" (M2), "schedule" (M2), "optimizing", "evaluation", "building the scene"; every "optimizing k of n"
phase is the one stage "optimizing"). The batch's sidecars carry none of it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SPEC_FILE = "spec.json"
PROCESS_FILE = "process.json"
LOG_FILE = "job.log"
PROGRESS_FILE = "progress.json"
STATE_FILE = "state.json"
RECORDS_DIR = "records"
COMPARISON_DIR = "comparison"
#: The comparison builder's index (``build_scenario_comparison_czml.py``), the one file a reader starts from.
INDEX_FILE = "comparison_index.json"
REPORT_FILE = "evaluation_report.json"

STATE_RUNNING, STATE_DONE, STATE_FAILED, STATE_CANCELLED = "running", "done", "failed", "cancelled"

#: What the job is doing, in ``progress.json`` ``phase`` (plain words: the panel shows them as they are). The backend's
#: answer before the job has written anything is ``PHASE_STARTING``.
PHASE_STARTING = "starting"
PHASE_READING = "reading traffic"
PHASE_ETA = "earliest arrival of each aircraft"
PHASE_SCHEDULE = "schedule"
PHASE_EVALUATION = "evaluation"
PHASE_SCENE = "building the scene"

#: The stages the job's wall time is split into (``timing.phases`` of ``state.json``).
STAGE_READING, STAGE_ETA, STAGE_SCHEDULE, STAGE_FLYING, STAGE_EVALUATION, STAGE_SCENE = (
    "reading traffic", "earliest arrivals", "schedule", "optimizing", "evaluation", "building the scene")
_STAGE_OF_PHASE = {PHASE_READING: STAGE_READING, PHASE_ETA: STAGE_ETA, PHASE_SCHEDULE: STAGE_SCHEDULE,
                   PHASE_EVALUATION: STAGE_EVALUATION, PHASE_SCENE: STAGE_SCENE}


def phase_flying(k: int, n: int) -> str:
    """The aircraft being optimized: the ``k``-th of ``n`` (an M1 job one; an M2 job one per slot, in slot order). The
    function keeps its name (``traffic/block.py`` imports it); the user's word for the phase is "optimizing"."""
    return f"optimizing {k} of {n}"


def phase_stage(phase: str) -> str:
    """The stage a phase of ``progress.json`` belongs to: every "optimizing k of n" is the one stage ``STAGE_FLYING`` (named
    "optimizing"), the other phases one stage each. A phase that is none of them is an error."""
    return STAGE_FLYING if phase.startswith("optimizing ") else _STAGE_OF_PHASE[phase]

MODE_M1, MODE_M2 = "m1", "m2"
#: The block lengths a job takes (s): 15, 30 and 60 minutes (design IM3).
BLOCK_LENGTHS_S = (900, 1800, 3600)

#: The scenario list of an airport (design §10.6): written by ``traffic_scenarios.py``, served by the backend.
CATALOG_SCHEMA = "traffic-scenario-catalog-v2"
#: ``4dTrajectory/outputs``: the root the census writes its catalogs under and the backend serves them from.
OUTPUTS_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def catalog_path(outputs_root: Path, airport: str) -> Path:
    """Where an airport's scenario catalog lives: ``<4dTrajectory/outputs>/<ICAO>/traffic_scenarios/catalog.json``."""
    return Path(outputs_root) / airport / "traffic_scenarios" / "catalog.json"


def catalog_command(airport: str) -> str:
    """The command that makes an airport's scenario catalog (what the backend's 404 names)."""
    return f"python 4dTrajectory/optimization/traffic_scenarios.py --airport {airport}"


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write ``payload`` so that a reader polling ``path`` sees the old file or the new one, never half of it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)
