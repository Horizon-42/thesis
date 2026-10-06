"""The interactive multi-aircraft jobs of the Optimize task (design ``4dTrajectory/docs/multi_aircraft_optimization/
design.md`` §10.3): list an airport's arrivals of a UTC day, serve its scenario catalog (§10.6), run ONE job at a
time, report on it, serve its files, cancel it.

A job is a subprocess of this backend's own interpreter (``4dTrajectory/optimization/traffic_job.py``), in its own
process group at ``nice 10`` and one solver thread: casadi is not thread-safe, a long job must not block the resident
single-aircraft optimizer, and the box also runs experiments (IM4). Everything the job writes goes to its own directory
under the jobs root (default ``~/.cache/aeroviz/traffic_jobs/<port>``, never ``public/data``: a job result is not a
publication, IM6); the newest ``KEEP_JOBS`` directories are kept. The viewer reads the job's comparison files through
``file`` — only the files the job's comparison index lists, the index itself and the evaluation report the index names.

The job directory records the job's process (``process.json``: the group, which the leader's pid names; the leader's start
time; the boot id): a job outlives its backend (a session of its own), and a job whose process is alive counts as running —
for a new start (409), for its status, for pruning — whichever backend started it. A process group number is reused once its
leader is gone, so a group is counted alive, and signalled, ONLY while its leader exists with the recorded start time and boot
id; a job whose process is gone without a ``state.json`` is made ``failed`` on disk at once, and its group number is never
consulted again. A job directory with a ``state.json`` is finished.

The backend does not hot-reload: a change here needs a restart.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from aeroviz_backend import paths
from aeroviz_backend.observed_trajectories import DEFAULT_HARVEST_ROOT, _normalize_airport
from scenario_batch import SOLVER_THREAD_ENV
from traffic_job_files import (
    BLOCK_LENGTHS_S,
    CATALOG_SCHEMA,
    COMPARISON_DIR,
    INDEX_FILE,
    LOG_FILE,
    MODE_M1,
    MODE_M2,
    OUTPUTS_ROOT,
    PHASE_STARTING,
    PROCESS_FILE,
    PROGRESS_FILE,
    SPEC_FILE,
    STATE_CANCELLED,
    STATE_DONE,
    STATE_FAILED,
    STATE_FILE,
    STATE_RUNNING,
    catalog_command,
    catalog_path,
    write_json_atomic,
)
from traffic_roster import landing_in, read_roster, utc_day_bounds_s
from trajectory_data_process.harvest.arrivals import arrival_manifest_path
from trajectory_data_process.harvest.store import HarvestPaths
from trajectory_data_process.harvest.utc import iso_utc, iso_utc_ms, parse_iso_utc_s

#: The jobs of one backend live in ``<root>/<port>``: two backends never share a directory.
DEFAULT_JOBS_ROOT = Path.home() / ".cache" / "aeroviz" / "traffic_jobs"
#: Job directories kept (the newest ones), the running job's included.
KEEP_JOBS = 5
#: What a ``done`` ``state.json`` holds beside its state (``traffic_job_files``): the readout, what changed for each aircraft, the timing.
DONE_FIELDS = ("summary", "perAircraft", "timing", "stayedRecords")
#: The job's niceness (IM4).
JOB_NICE = 10
#: How long a cancelled job's group has to end after SIGTERM before it is killed (s).
CANCEL_GRACE_S = 5.0
JOB_SCRIPT = paths.TRAJECTORY_OPTIMIZATION_DIR / "traffic_job.py"
# A job id is its start to the microsecond (the ids sort by start time; one job runs at a time) and a random suffix.
_JOB_ID = re.compile(r"^[0-9]{8}T[0-9]{12}Z-[0-9a-f]{8}$")
#: The fields of a request and their types (``blockS`` is an integer, the others text).
_M1_FIELDS = {"mode": str, "airport": str, "flightKey": str}
_M2_FIELDS = {"mode": str, "airport": str, "blockStartUtc": str, "blockS": int}


class JobBusy(RuntimeError):
    """A job is running: one at a time (409)."""


class BadJobRequest(ValueError):
    """The request names something the roster does not have, or something a job cannot take (400)."""


class JobNotFound(LookupError):
    """No such job, or no such file of it (404)."""


class JobStateInvalid(RuntimeError):
    """A job's ``state.json`` says ``done`` and lacks what a done state holds (500)."""


class CatalogUnreadable(RuntimeError):
    """An airport's scenario catalog is on disk but is not a catalog of this schema (500)."""


def traffic_jobs_root(port: int) -> Path:
    """The default jobs root of the backend that listens on ``port``."""
    return DEFAULT_JOBS_ROOT / str(port)


def job_command(spec: Path, out: Path) -> list[str]:
    """The job process, before ``nice``: this interpreter running ``traffic_job.py``."""
    return [sys.executable, str(JOB_SCRIPT), "--spec", str(spec), "--out", str(out)]


def _typecode_of_icao24(icao24: str) -> str | None:
    """The ICAO type of an aircraft by the identity resolver the scenarios use (the roster rows carry no type)."""
    from aircraft.identity import get_default_identity_resolver

    return get_default_identity_resolver().resolve(declared_type=None, icao24=icao24).typecode


def _signal_group(pgid: int, signum: int) -> None:
    """Send ``signum`` to every process of the group ``pgid`` (none left: nothing to do). The caller has established that
    the group is the job's."""
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")


@dataclass(frozen=True)
class ProcessIdentity:
    """What a pid is, so that a pid that has been given to another process is not mistaken for the one recorded."""

    boot_id: str
    #: Field 22 of ``/proc/<pid>/stat``: the process's start, in clock ticks after the boot.
    start_time: str
    #: Field 3: ``Z`` is a process that has exited and is waiting to be reaped.
    state: str


def process_identity(pid: int) -> ProcessIdentity | None:
    """The identity of the process ``pid`` now, or None when there is none (Linux: ``/proc``)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):     # gone before the open, or reaped between open and read (ESRCH)
        return None
    # the command name (field 2) is in parentheses and may hold spaces and parentheses: the fields follow the last ")"
    fields = stat[stat.rindex(")") + 2:].split()
    return ProcessIdentity(BOOT_ID_FILE.read_text(encoding="utf-8").strip(), start_time=fields[19], state=fields[0])


def _is_the_job(record: dict[str, Any]) -> bool:
    """Is the process ``process.json`` records still there, alive, and the same process (same start, same boot)?"""
    found = process_identity(record["pgid"])
    return (found is not None and found.state != "Z"
            and found.start_time == record["startTime"] and found.boot_id == record["bootId"])


def _peek_exit(child: subprocess.Popen[bytes]) -> bool:
    """Has the job's main process exited? It stays a zombie (not reaped): its pid cannot name another group meanwhile."""
    return os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _await(condition: Callable[[], bool], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)


class TrafficJobs:
    """Lists arrivals, runs one job at a time, serves its directory. Thread-safe (the server is threaded)."""

    def __init__(
        self,
        jobs_root: Path = DEFAULT_JOBS_ROOT,
        *,
        harvest_root: Path = DEFAULT_HARVEST_ROOT,
        outputs_root: Path = OUTPUTS_ROOT,
        command: Callable[[Path, Path], list[str]] = job_command,
        typecode_of: Callable[[str], str | None] = _typecode_of_icao24,
        keep: int = KEEP_JOBS,
    ) -> None:
        self.jobs_root = Path(jobs_root)
        self.harvest_root = Path(harvest_root)
        self.outputs_root = Path(outputs_root)
        self._command = command
        self._typecode_of = typecode_of
        self._keep = keep
        self._lock = threading.RLock()
        #: The jobs this backend started whose main process it has not yet seen exit (and reaped).
        self._children: dict[str, subprocess.Popen[bytes]] = {}

    # ── the roster ────────────────────────────────────────────────────────────

    def _manifest(self, airport: str) -> Path:
        path = arrival_manifest_path(HarvestPaths(self.harvest_root, _normalize_airport(airport)))
        if not path.is_file():
            raise FileNotFoundError(f"no arrivals roster for {airport} at {path}")
        return path

    def arrivals(self, airport: str, day: str) -> dict[str, Any]:
        """The arrivals that land on the UTC ``day`` (``YYYY-MM-DD``), by landing time. Reads the roster only."""
        try:
            parsed = date.fromisoformat(day)
        except ValueError as exc:
            raise ValueError(f"date must be YYYY-MM-DD, got {day!r}") from exc
        code = _normalize_airport(airport)
        rows = landing_in(read_roster(self._manifest(code)), *utc_day_bounds_s(parsed))
        return {"airport": code, "date": parsed.isoformat(), "arrivals": [
            {"flightKey": r.flight_key, "callsign": r.callsign, "runway": r.runway,
             "type": self._typecode_of(r.icao24), "entryUtc": iso_utc_ms(r.entry_utc_s),
             "landingUtc": iso_utc(r.landing_utc_s)} for r in rows]}

    def scenarios(self, airport: str) -> dict[str, Any]:
        """The airport's scenario catalog (design §10.6), as ``traffic_scenarios.py`` wrote it. Read only.
        :class:`FileNotFoundError` names the command that makes it; :class:`CatalogUnreadable` names the schema found."""
        code = _normalize_airport(airport)
        path = catalog_path(self.outputs_root, code)
        if not path.is_file():
            raise FileNotFoundError(f"no scenario catalog for {code} at {path}; make it with `{catalog_command(code)}`")
        try:
            catalog = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:       # both are ValueErrors: never the route's 400
            raise CatalogUnreadable(f"{path} is not UTF-8 JSON ({type(exc).__name__}): {exc}") from exc
        found = catalog.get("schema") if isinstance(catalog, dict) else None
        if found != CATALOG_SCHEMA:
            raise CatalogUnreadable(f"{path} has schema {found!r}, not {CATALOG_SCHEMA!r}; "
                                    f"make it again with `{catalog_command(code)}`")
        return catalog

    # ── the jobs on disk ──────────────────────────────────────────────────────

    def _job_ids(self) -> list[str]:
        """Every job directory, oldest first."""
        if not self.jobs_root.is_dir():
            return []
        return sorted(p.name for p in self.jobs_root.iterdir() if _JOB_ID.match(p.name))

    def _dir(self, job_id: str) -> Path:
        out = self.jobs_root / job_id
        if not _JOB_ID.match(job_id) or not out.is_dir():
            raise JobNotFound(f"no traffic job {job_id!r}")
        return out

    def _record(self, job_id: str) -> dict[str, Any] | None:
        path = self.jobs_root / job_id / PROCESS_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _finished(self, job_id: str) -> bool:
        return (self.jobs_root / job_id / STATE_FILE).is_file()

    def _fail(self, job_id: str, reason: str) -> None:
        write_json_atomic(self.jobs_root / job_id / STATE_FILE, {"state": STATE_FAILED, "error": reason})

    def _reap_children(self) -> None:
        """Every job this backend started whose main process has exited: what is left of its group (a child of the job: the
        evaluation, the builder) is killed while the zombie still holds its pid, the process is reaped and dropped, and a job
        that wrote no ``state.json`` is failed on disk with the exit code — so its group number is not consulted again."""
        for job_id, child in list(self._children.items()):
            if not _peek_exit(child):
                continue
            _signal_group(child.pid, signal.SIGKILL)
            code = child.wait()
            del self._children[job_id]
            if not self._finished(job_id):
                self._fail(job_id, f"the job process exited with code {code} before it wrote {STATE_FILE} (see {LOG_FILE})")

    def _settle(self, job_id: str) -> bool:
        """Is the job running? A job that has a ``state.json`` is not. A job whose process is gone without one is failed
        on disk now (its process.json is never consulted again); a job started by another backend is alive only while its
        leader is the process recorded."""
        self._reap_children()
        if self._finished(job_id):
            return False
        if job_id in self._children:
            return True
        record = self._record(job_id)
        if record is not None and _is_the_job(record):
            return True
        self._fail(job_id, f"the job process is gone and wrote no {STATE_FILE} (see {LOG_FILE})")
        return False

    def _running(self) -> str | None:
        """The job that holds the slot: unfinished and its process alive."""
        for job_id in reversed(self._job_ids()):
            if self._settle(job_id):
                return job_id
        return None

    def _prune(self) -> None:
        """Make room for one more job: keep the newest ``keep - 1`` directories, and never remove a job that is running
        (spared by name, however old)."""
        ids = self._job_ids()
        for job_id in ids[:max(0, len(ids) - (self._keep - 1))]:
            if not self._settle(job_id):
                shutil.rmtree(self.jobs_root / job_id)

    # ── a job ─────────────────────────────────────────────────────────────────

    def _checked_spec(self, request: dict[str, Any]) -> dict[str, Any]:
        """The job's spec file for ``request``, or :class:`BadJobRequest` naming what is wrong with it."""
        mode = request.get("mode")
        fields = {MODE_M1: _M1_FIELDS, MODE_M2: _M2_FIELDS}.get(mode) if isinstance(mode, str) else None
        if fields is None:
            raise BadJobRequest(f"mode must be {MODE_M1!r} or {MODE_M2!r}, got {mode!r}")
        absent, extra = sorted(set(fields) - set(request)), sorted(set(request) - set(fields))
        if absent or extra:
            raise BadJobRequest(f"a {mode} job takes {sorted(fields)}: missing {absent}, unexpected {extra}")
        for name, kind in fields.items():
            value = request[name]
            if not isinstance(value, kind) or isinstance(value, bool):
                raise BadJobRequest(f"{name} must be {'an integer' if kind is int else 'text'}, got {value!r}")
        try:
            airport = _normalize_airport(request["airport"])
            manifest = self._manifest(airport)
        except (ValueError, FileNotFoundError) as exc:
            raise BadJobRequest(str(exc)) from exc
        rows = read_roster(manifest)
        spec: dict[str, Any] = {"mode": mode, "airport": airport, "manifest": str(manifest)}
        if mode == MODE_M1:
            key = request["flightKey"]
            if not any(r.flight_key == key for r in rows):
                raise BadJobRequest(f"flight {key!r} is not in the {airport} arrivals roster")
            spec["flightKey"] = key
            return spec
        block_s = request["blockS"]
        if block_s not in BLOCK_LENGTHS_S:
            raise BadJobRequest(f"blockS must be one of {list(BLOCK_LENGTHS_S)} s, got {block_s!r}")
        try:
            start = parse_iso_utc_s(request["blockStartUtc"])
        except ValueError as exc:
            raise BadJobRequest(f"blockStartUtc is not a UTC time: {request['blockStartUtc']!r}") from exc
        if not landing_in(rows, start, start + block_s):
            raise BadJobRequest(f"no {airport} arrival lands in {block_s} s from {request['blockStartUtc']}")
        spec.update(blockStartUtc=request["blockStartUtc"], blockS=block_s)
        return spec

    def start(self, request: dict[str, Any]) -> dict[str, str]:
        """Start the job ``request`` names (``{mode: "m1", airport, flightKey}`` or ``{mode: "m2", airport,
        blockStartUtc, blockS}``). :class:`JobBusy` while another runs; :class:`BadJobRequest` for a bad request."""
        with self._lock:
            if self._running() is not None:
                raise JobBusy("a traffic job is running: cancel it or wait for it")
            spec = self._checked_spec(request)
            if not BOOT_ID_FILE.is_file():
                raise RuntimeError("traffic jobs identify their process by /proc, which this system does not have")
            self.jobs_root.mkdir(parents=True, exist_ok=True)
            self._prune()
            job_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}Z-{uuid.uuid4().hex[:8]}"
            out = self.jobs_root / job_id
            out.mkdir()
            write_json_atomic(out / SPEC_FILE, spec)
            # one solver thread: a solve beside the experiments of the box takes no more (scenario_batch's own set)
            env = {**os.environ, **{name: "1" for name in SOLVER_THREAD_ENV}}
            with open(out / LOG_FILE, "wb") as log:
                process = subprocess.Popen(
                    ["nice", "-n", str(JOB_NICE), *self._command(out / SPEC_FILE, out)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=paths.REPO_ROOT, env=env,
                    start_new_session=True)               # its own process group: cancel stops the job's children too
            identity = process_identity(process.pid)       # the process is ours and unreaped: its entry is there
            assert identity is not None
            write_json_atomic(out / PROCESS_FILE, {
                "pgid": process.pid, "startTime": identity.start_time, "bootId": identity.boot_id})
            self._children[job_id] = process
            return {"jobId": job_id}

    def status(self, job_id: str) -> dict[str, Any]:
        """``{state: running | done | failed | cancelled, progress: {done, total, current, phase}, error}`` of a job, and
        — once ``done`` — ``summary``, the readout of ``traffic.readout`` (outcome counts; for M2 the delays and the
        losses left after the block's final check), ``perAircraft`` (``traffic_job_files``: what changed for each
        aircraft, and the time spent on it), ``timing`` (the job's wall time by stage) and ``stayedRecords`` (the arrivals of an M2
        block it could not control). ``phase`` is what the job is doing (``PHASE_STARTING`` until it has written its first progress)."""
        with self._lock:
            out = self._dir(job_id)
            running = self._settle(job_id)
            progress_path = out / PROGRESS_FILE
            progress = (json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file()
                        else {"done": 0, "total": None, "current": None, "phase": PHASE_STARTING})
            if running:
                return {"state": STATE_RUNNING, "progress": progress, "error": None}
            state = json.loads((out / STATE_FILE).read_text(encoding="utf-8"))
            found = {"state": state["state"], "progress": progress, "error": state["error"]}
            if state["state"] == STATE_DONE:
                missing = [field for field in DONE_FIELDS if field not in state]
                if missing:
                    raise JobStateInvalid(f"job {job_id}: its state.json says done and has no {', '.join(missing)}")
                found.update({field: state[field] for field in DONE_FIELDS})
            return found

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Stop the job's process group (SIGTERM, then SIGKILL after ``CANCEL_GRACE_S``) and mark it cancelled. A job that
        is finished — a ``state.json`` written, ``done`` included — or whose process is gone is left as it is; a group is
        signalled only while its leader is the job's process."""
        with self._lock:
            out = self._dir(job_id)
            if self._settle(job_id):
                self._stop(job_id)
                if not self._finished(job_id):             # the job is dead: nothing writes state.json any more
                    write_json_atomic(out / STATE_FILE, {"state": STATE_CANCELLED, "error": None})
            return self.status(job_id)

    def _stop(self, job_id: str) -> None:
        """SIGTERM the job's group, SIGKILL what is left after ``CANCEL_GRACE_S``. A job this backend started is stopped
        through its unreaped process (its pid names no other group); another backend's, only while its recorded leader is
        still the process recorded, checked before each signal."""
        child = self._children.pop(job_id, None)
        if child is not None:
            _signal_group(child.pid, signal.SIGTERM)
            _await(lambda: _peek_exit(child), CANCEL_GRACE_S)
            _signal_group(child.pid, signal.SIGKILL)          # the job if it ignored SIGTERM, else what it left behind
            child.wait()
            return
        record = self._record(job_id)
        if record is None or not _is_the_job(record):
            return
        _signal_group(record["pgid"], signal.SIGTERM)
        _await(lambda: not _is_the_job(record), CANCEL_GRACE_S)
        if _is_the_job(record):
            _signal_group(record["pgid"], signal.SIGKILL)

    def shutdown(self) -> None:
        """The backend is stopping: a running job is cancelled with it (it is a session of its own, so the
        server's death would not reach it), and so is any process of a finished job still lingering."""
        with self._lock:
            running = self._running()
            if running is not None:
                self.cancel(running)
            for job_id in list(self._children):
                self._stop(job_id)

    def file(self, job_id: str, name: str) -> Path:
        """One file of the job's comparison directory, only if the job's comparison index lists it: the index, the
        CZML files of its groups, the evaluation report it names. :class:`JobNotFound` for any other name, and for a
        listed file that is not on disk."""
        with self._lock:
            comparison = self._dir(job_id) / COMPARISON_DIR
            index_path = comparison / INDEX_FILE
            if not index_path.is_file():
                raise JobNotFound(f"job {job_id} has no {INDEX_FILE} (yet)")
            index = json.loads(index_path.read_text(encoding="utf-8"))
            listed = {INDEX_FILE, index["evaluationReport"], *(group["czml"] for group in index["groups"])}
            if name not in listed:
                raise JobNotFound(f"{name!r} is not a file of job {job_id}'s comparison index")
            if not (comparison / name).is_file():
                raise JobNotFound(f"{name!r} is listed by job {job_id}'s comparison index but is not on disk")
            return comparison / name
