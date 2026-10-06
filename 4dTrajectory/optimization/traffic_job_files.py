"""The layout of one interactive traffic job's directory (design §10.3), shared by the job process
(``traffic_job.py``) and the backend that starts and serves it (``aeroviz_backend/traffic_jobs.py``).
Standard library only: the backend imports it.

    <job>/spec.json          what to fly (written by the backend)
    <job>/process.json       {pgid, startTime, bootId} (written by the backend): the job's process group (its leader's
                             pid), the leader's start (/proc/<pid>/stat field 22) and the boot id — a group number is
                             reused, so a group is the job's only while its leader has this start and this boot
    <job>/job.log            the job process's output (the backend redirects it here)
    <job>/progress.json      {done, total, current}, rewritten after each aircraft
    <job>/records/           the batch records, summary.json, evaluation_report.json
    <job>/comparison/        the comparison builder's files: comparison_index.json, the CZML, the report
    <job>/state.json         the job's last write, {state, error, [summary]}: done (with its readout) or failed
                             (with the reason); the backend writes cancelled
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

MODE_M1, MODE_M2 = "m1", "m2"
#: The block lengths a job takes (s): 15, 30 and 60 minutes (design IM3).
BLOCK_LENGTHS_S = (900, 1800, 3600)

#: The scenario list of an airport (design §10.6): written by ``traffic_scenarios.py``, served by the backend.
CATALOG_SCHEMA = "traffic-scenario-catalog-v1"
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
