"""The traffic job manager on a FAKE job process (a small script that writes the files the real job writes):
one job at a time, the process group and niceness, cancel, the files served, the directories kept, the roster reads.
Every write goes under tmp_path; no solve, no track, no live root."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from aeroviz_backend import traffic_jobs as tj
from scenario_batch import SOLVER_THREAD_ENV
from traffic_job_files import (
    BLOCK_LENGTHS_S, CATALOG_SCHEMA, COMPARISON_DIR, INDEX_FILE, PROCESS_FILE, PROGRESS_FILE, SPEC_FILE, STATE_FILE,
    catalog_command, catalog_path, write_json_atomic)
from trajectory_data_process.harvest.utc import iso_utc, iso_utc_ms

T0 = 1_800_000_000.0

FAKE_JOB = textwrap.dedent('''
    import json, os, subprocess, sys, time
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    out, behaviour = args["--out"], args["--behaviour"]
    def write(name, payload):
        with open(os.path.join(out, name + ".tmp"), "w") as f:
            json.dump(payload, f)
        os.replace(os.path.join(out, name + ".tmp"), os.path.join(out, name))
    if behaviour == "late":
        time.sleep(60)                                         # nothing written for a long while
    write("progress.json", {"done": 0, "total": 2, "current": None, "phase": "optimizing 1 of 2"})
    if behaviour == "env":
        write("env.json", {name: os.environ.get(name) for name in sys.argv[sys.argv.index("--vars") + 1].split(",")})
    if behaviour == "orphan":
        child = subprocess.Popen(["sleep", "60"])              # a child of the job that outlives its main process
        write("pids.json", {"pid": os.getpid(), "child": child.pid})
        sys.exit(0)
    if behaviour == "done_then_hang":
        write("state.json", {"state": "done", "error": None, "summary": {"windows": 1}, "perAircraft": {},
                             "timing": {"totalS": 1.0, "phases": {"optimizing": 1.0}}, "stayedRecords": {}})
        time.sleep(60)
    if behaviour == "hang":
        child = subprocess.Popen(["sleep", "60"])              # a grandchild in the job's process group
        write("pids.json", {"pid": os.getpid(), "child": child.pid, "pgid": os.getpgid(0), "nice": os.nice(0)})
        time.sleep(60)
    if behaviour == "crash":
        sys.exit(3)
    if behaviour == "slow":
        time.sleep(0.3)
    write("progress.json", {"done": 2, "total": 2, "current": "X", "phase": "building the scene"})
    if behaviour == "fail":
        write("state.json", {"state": "failed", "error": "ValueError: no way"})
        sys.exit(1)
    write("state.json", {"state": "done", "error": None, "summary": {"windows": 1},
                         "perAircraft": {"X": {"firstSolveLosses": 3, "finalLosses": 0}},
                         "timing": {"totalS": 2.5, "phases": {"reading traffic": 0.5, "optimizing": 2.0}},
                         "stayedRecords": {"Y": {"callsign": "YYY1", "type": "C172"}}})
''')


def row(key, entry_s, runway="05L", icao24="a00001"):
    return {"flight_key": key, "callsign": key[:3], "icao24": icao24, "runway": runway,
            "entry_time_utc": iso_utc_ms(T0 + entry_s), "landing_time_utc": iso_utc(T0 + entry_s + 200.0),
            "arrival_duration_s": 200.0}


@pytest.fixture
def stack(tmp_path):
    """A harvest root with one KRDU roster, a fake job script, and a manager over both (all under tmp)."""
    manifest = tmp_path / "harvest" / "KRDU" / "arrivals" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"records": [row("AAA_05L", 0.0), row("BBB_05R", 300.0, "05R", icao24="b00002"),
                                                row("CCC_05L", 90000.0, icao24="c00003")]}))
    script = tmp_path / "fake_job.py"
    script.write_text(FAKE_JOB)
    behaviour = {"now": "finish"}
    asked = []

    def command(spec, out):
        return [sys.executable, str(script), "--spec", str(spec), "--out", str(out), "--behaviour", behaviour["now"],
                "--vars", ",".join(SOLVER_THREAD_ENV)]

    def typecode(icao24):
        asked.append(icao24)
        return "A320" if icao24 == "a00001" else None

    jobs = tj.TrafficJobs(tmp_path / "jobs", harvest_root=tmp_path / "harvest", outputs_root=tmp_path / "outputs",
                          command=command, typecode_of=typecode)
    yield jobs, behaviour, asked, tmp_path
    jobs.shutdown()


# ── the scenario catalog (design §10.6): read from disk, read only ─────────────────────────────────────────────

def _write_catalog(tmp_path, airport, payload):
    path = catalog_path(tmp_path / "outputs", airport)
    path.parent.mkdir(parents=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def test_the_catalog_of_an_airport_is_served_as_written(stack):
    jobs, _behaviour, _asked, tmp_path = stack
    catalog = {"schema": CATALOG_SCHEMA, "airport": "KRDU", "m1": [], "m2": {"900": [], "1800": [], "3600": []}}
    _write_catalog(tmp_path, "KRDU", catalog)
    assert jobs.scenarios("krdu") == catalog                          # the code is normalised as the other routes' are


def test_a_missing_catalog_is_a_file_not_found_that_names_the_command(stack):
    jobs, _behaviour, _asked, _tmp = stack
    with pytest.raises(FileNotFoundError) as caught:
        jobs.scenarios("KRDU")
    assert f"`{catalog_command('KRDU')}`" in str(caught.value)
    assert "no scenario catalog for KRDU" in str(caught.value)


def test_the_command_the_404_names_is_the_census_script_of_the_repository():
    script = catalog_command("KRDU").split()[1]
    assert (tj.paths.REPO_ROOT / script).is_file()
    assert catalog_command("KSEA").endswith("--airport KSEA")


@pytest.mark.parametrize("payload, found", [
    ({"schema": "traffic-scenario-catalog-v1", "m1": []}, "'traffic-scenario-catalog-v1'"),
    ({"m1": []}, "None"),
    ([1, 2], "None"),
])
def test_a_catalog_of_another_schema_is_refused_by_the_schema_it_found(stack, payload, found):
    jobs, _behaviour, _asked, tmp_path = stack
    _write_catalog(tmp_path, "KRDU", payload)
    with pytest.raises(tj.CatalogUnreadable) as caught:
        jobs.scenarios("KRDU")
    assert f"has schema {found}" in str(caught.value) and CATALOG_SCHEMA in str(caught.value)


def test_a_catalog_that_is_not_json_is_refused_by_name(stack):
    jobs, _behaviour, _asked, tmp_path = stack
    _write_catalog(tmp_path, "KRDU", "{not json")
    with pytest.raises(tj.CatalogUnreadable, match="is not UTF-8 JSON .JSONDecodeError"):
        jobs.scenarios("KRDU")


def test_a_catalog_that_is_not_utf8_is_refused_by_name_not_as_a_bad_request(stack):
    jobs, _behaviour, _asked, tmp_path = stack
    path = _write_catalog(tmp_path, "KRDU", "{}")
    path.write_bytes(b'{"schema": "\xff\xfe"}')                      # UnicodeDecodeError is a ValueError: the route maps those to 400
    with pytest.raises(tj.CatalogUnreadable, match="UnicodeDecodeError"):
        jobs.scenarios("KRDU")


def test_a_bad_airport_code_is_a_value_error_not_a_path(stack):
    jobs, _behaviour, _asked, _tmp = stack
    for bad in ("../x", "K R", ""):
        with pytest.raises(ValueError):
            jobs.scenarios(bad)


def group_alive(pgid):
    """Is anything of the group still there (the test's own probe: signal 0)?"""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for(condition, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        found = condition()
        if found:
            return found
        time.sleep(0.02)
    raise AssertionError("timed out")


def finished(jobs, job_id):
    return wait_for(lambda: jobs.status(job_id)["state"] != "running" and jobs.status(job_id))


M1 = {"mode": "m1", "airport": "KRDU", "flightKey": "AAA_05L"}
M2 = {"mode": "m2", "airport": "krdu", "blockStartUtc": iso_utc(T0 - 60.0), "blockS": 900}


# ── the roster ───────────────────────────────────────────────────────────────

def test_the_arrivals_of_a_utc_day_come_from_the_roster_alone(stack):
    jobs, _behaviour, asked, tmp_path = stack
    day = time.strftime("%Y-%m-%d", time.gmtime(T0))
    listed = jobs.arrivals("krdu", day)

    assert listed["airport"] == "KRDU" and listed["date"] == day
    assert [a["flightKey"] for a in listed["arrivals"]] == ["AAA_05L", "BBB_05R"]      # CCC lands the next day
    first = listed["arrivals"][0]
    assert first == {"flightKey": "AAA_05L", "callsign": "AAA", "runway": "05L", "type": "A320",
                     "entryUtc": iso_utc_ms(T0), "landingUtc": iso_utc(T0 + 200.0)}
    assert listed["arrivals"][1]["type"] is None                         # an aircraft the resolver cannot type
    assert asked == ["a00001", "b00002"]                                 # asked only for the day's rows
    assert not list((tmp_path / "harvest").rglob("tracks*"))             # and no track exists to be read


def test_a_flight_without_a_callsign_is_listed_with_a_null_one(stack):
    jobs, _behaviour, _asked, tmp_path = stack
    manifest = tmp_path / "harvest" / "KRDU" / "arrivals" / "manifest.json"
    records = json.loads(manifest.read_text())["records"]
    records[0]["callsign"] = None                           # the roster lacks the callsign of some flights
    manifest.write_text(json.dumps({"records": records}))
    day = time.strftime("%Y-%m-%d", time.gmtime(T0))
    listed = jobs.arrivals("KRDU", day)["arrivals"]
    assert [a["callsign"] for a in listed] == [None, "BBB"] and listed[0]["flightKey"] == "AAA_05L"


def test_a_bad_date_or_airport_is_refused_and_a_missing_roster_is_not_found(stack):
    jobs, *_ = stack
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        jobs.arrivals("KRDU", "21/05/2026")
    with pytest.raises(ValueError, match="airport"):
        jobs.arrivals("K", "2026-05-21")
    with pytest.raises(FileNotFoundError):
        jobs.arrivals("KSEA", "2026-05-21")


# ── requests ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("request_, message", [
    ({"mode": "m3", "airport": "KRDU"}, "mode must be"),
    ({**M1, "flightKey": "NOPE_05L"}, "not in the KRDU arrivals roster"),
    ({"mode": "m1", "airport": "KRDU"}, "missing \\['flightKey'\\]"),
    ({**M1, "blockS": 900}, "unexpected \\['blockS'\\]"),
    ({**M2, "blockS": 1200}, "blockS must be one of"),
    ({**M2, "blockS": True}, "blockS must be an integer"),
    ({**M2, "blockS": "900"}, "blockS must be an integer"),
    ({**M2, "blockS": 900.0}, "blockS must be an integer"),
    ({"mode": []}, "mode must be"),
    ({"mode": {}}, "mode must be"),
    ({**M1, "airport": 5}, "airport must be text"),
    ({**M1, "airport": ["KRDU"]}, "airport must be text"),
    ({**M1, "flightKey": ["AAA_05L"]}, "flightKey must be text"),
    ({**M2, "blockStartUtc": 5}, "blockStartUtc must be text"),
    ({**M2, "blockStartUtc": "yesterday"}, "not a UTC time"),
    ({**M2, "blockStartUtc": iso_utc(T0 + 5000.0)}, "no KRDU arrival lands"),
    ({**M1, "airport": "KSEA"}, "no arrivals roster"),
])
def test_a_request_the_roster_cannot_serve_is_a_bad_request_and_starts_nothing(stack, request_, message):
    jobs, _behaviour, _asked, tmp_path = stack
    with pytest.raises(tj.BadJobRequest, match=message):
        jobs.start(request_)
    assert not (tmp_path / "jobs").exists()


def test_the_default_jobs_root_is_the_backends_own_port(stack):
    assert tj.traffic_jobs_root(8765) == tj.DEFAULT_JOBS_ROOT / "8765"
    assert tj.traffic_jobs_root(8765) != tj.traffic_jobs_root(5174)


def test_the_block_lengths_are_15_30_and_60_minutes():
    assert BLOCK_LENGTHS_S == (900, 1800, 3600)


# ── a job ────────────────────────────────────────────────────────────────────

def test_a_job_runs_to_done_with_its_progress_and_its_readout(stack):
    jobs, behaviour, _asked, tmp_path = stack
    behaviour["now"] = "slow"
    job_id = jobs.start(M1)["jobId"]

    running = jobs.status(job_id)
    assert running["state"] == "running" and running["error"] is None
    done = finished(jobs, job_id)
    assert done == {"state": "done", "error": None, "summary": {"windows": 1},
                    "progress": {"done": 2, "total": 2, "current": "X", "phase": "building the scene"},
                    "perAircraft": {"X": {"firstSolveLosses": 3, "finalLosses": 0}},
                    "timing": {"totalS": 2.5, "phases": {"reading traffic": 0.5, "optimizing": 2.0}},
                    "stayedRecords": {"Y": {"callsign": "YYY1", "type": "C172"}}}      # all four of a done state, as written
    spec = json.loads((tmp_path / "jobs" / job_id / SPEC_FILE).read_text())
    assert spec == {"mode": "m1", "airport": "KRDU", "manifest": str(tmp_path / "harvest/KRDU/arrivals/manifest.json"),
                    "flightKey": "AAA_05L"}


def test_the_phase_the_job_wrote_is_in_the_status_while_it_runs(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "hang"
    job_id = jobs.start(M1)["jobId"]
    wait_for(lambda: (jobs.jobs_root / job_id / "pids.json").is_file())
    assert jobs.status(job_id)["progress"] == {"done": 0, "total": 2, "current": None, "phase": "optimizing 1 of 2"}
    jobs.cancel(job_id)


@pytest.mark.parametrize("lacking", [["perAircraft"], ["timing"], ["stayedRecords"],
                                     ["summary", "perAircraft", "timing", "stayedRecords"]])
def test_a_done_state_without_what_a_done_state_holds_is_refused_by_name(stack, lacking):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "slow"
    job_id = jobs.start(M1)["jobId"]
    finished(jobs, job_id)
    path = jobs.jobs_root / job_id / STATE_FILE
    state = json.loads(path.read_text())
    write_json_atomic(path, {k: v for k, v in state.items() if k not in lacking})
    with pytest.raises(tj.JobStateInvalid, match=f"has no {', '.join(lacking)}"):
        jobs.status(job_id)


def test_a_failed_job_has_no_per_aircraft(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "fail"
    job_id = jobs.start(M1)["jobId"]
    failed = finished(jobs, job_id)
    assert failed["state"] == "failed" and "perAircraft" not in failed and "summary" not in failed


def test_a_status_before_the_first_progress_has_no_total(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "late"                       # the job writes nothing for a minute: the status is the first one
    job_id = jobs.start(M1)["jobId"]
    assert not (jobs.jobs_root / job_id / PROGRESS_FILE).exists()
    assert jobs.status(job_id) == {"state": "running", "error": None,
                                   "progress": {"done": 0, "total": None, "current": None, "phase": "starting"}}
    jobs.cancel(job_id)


def test_an_m2_request_is_written_with_its_block(stack):
    jobs, _behaviour, _asked, tmp_path = stack
    job_id = jobs.start(M2)["jobId"]
    finished(jobs, job_id)
    spec = json.loads((tmp_path / "jobs" / job_id / SPEC_FILE).read_text())
    assert spec["mode"] == "m2" and spec["airport"] == "KRDU" and spec["blockS"] == 900
    assert spec["blockStartUtc"] == iso_utc(T0 - 60.0) and "flightKey" not in spec


def test_a_second_job_is_refused_while_one_runs_and_allowed_after(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "hang"
    first = jobs.start(M1)["jobId"]
    with pytest.raises(tj.JobBusy):
        jobs.start(M1)
    jobs.cancel(first)
    behaviour["now"] = "finish"
    assert finished(jobs, jobs.start(M1)["jobId"])["state"] == "done"


def test_a_failed_job_reports_its_reason_and_a_crashed_one_its_exit_code(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "fail"
    failed = finished(jobs, jobs.start(M1)["jobId"])
    assert failed["state"] == "failed" and failed["error"] == "ValueError: no way"
    behaviour["now"] = "crash"
    crashed = finished(jobs, jobs.start(M1)["jobId"])
    assert crashed["state"] == "failed" and "exited with code 3" in crashed["error"] and "state.json" in crashed["error"]


def test_a_job_runs_at_nice_10_in_its_own_process_group(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "hang"
    job_id = jobs.start(M1)["jobId"]
    pids = wait_for(lambda: json.loads(p.read_text()) if (p := jobs.jobs_root / job_id / "pids.json").is_file() else None)

    assert pids["pgid"] == pids["pid"] != os.getpgid(0)                        # a group of its own
    assert pids["nice"] == min(19, os.nice(0) + 10)                            # +10 over this process (19 is the cap)
    jobs.cancel(job_id)


def test_cancel_stops_the_job_and_everything_in_its_process_group(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "hang"
    job_id = jobs.start(M1)["jobId"]
    pids = wait_for(lambda: json.loads(p.read_text()) if (p := jobs.jobs_root / job_id / "pids.json").is_file() else None)

    cancelled = jobs.cancel(job_id)
    assert cancelled["state"] == "cancelled" and cancelled["error"] is None

    def gone(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False
    wait_for(lambda: gone(pids["pid"]) and gone(pids["child"]), timeout=5.0)        # the grandchild died with it
    assert json.loads((jobs.jobs_root / job_id / STATE_FILE).read_text())["state"] == "cancelled"
    assert jobs.status(job_id)["state"] == "cancelled"
    assert jobs.cancel(job_id)["state"] == "cancelled"                              # idempotent
    jobs.start(M1)                                                                  # and the next job may start


def test_cancelling_a_finished_job_leaves_it_as_it_is(stack):
    jobs, *_ = stack
    job_id = jobs.start(M1)["jobId"]
    assert finished(jobs, job_id)["state"] == "done"
    assert jobs.cancel(job_id)["state"] == "done"


def test_stopping_the_backend_cancels_the_running_job(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "hang"
    job_id = jobs.start(M1)["jobId"]
    wait_for(lambda: (jobs.jobs_root / job_id / "pids.json").is_file())
    jobs.shutdown()
    assert jobs.status(job_id)["state"] == "cancelled"


def test_only_the_newest_five_job_directories_are_kept(stack):
    jobs, *_ = stack
    ids = []
    for _ in range(7):
        ids.append(jobs.start(M1)["jobId"])
        finished(jobs, ids[-1])
        time.sleep(0.01)
    kept = sorted(p.name for p in jobs.jobs_root.iterdir())
    assert kept == sorted(ids)[-5:]
    with pytest.raises(tj.JobNotFound):
        jobs.status(ids[0])


@pytest.mark.parametrize("job_id", ["../x", "nope", "20261006T120000123456Z-zzzzzzzz", "20261006T120000123456Z-0123abcd"])
def test_an_unknown_or_malformed_job_id_is_not_found(stack, job_id):
    jobs, *_ = stack
    for call in (jobs.status, jobs.cancel):
        with pytest.raises(tj.JobNotFound):
            call(job_id)
    with pytest.raises(tj.JobNotFound):
        jobs.file(job_id, INDEX_FILE)


# ── files ────────────────────────────────────────────────────────────────────

def test_only_the_files_the_comparison_index_lists_are_served(stack):
    jobs, *_ = stack
    job_id = jobs.start(M1)["jobId"]
    finished(jobs, job_id)
    out = jobs.jobs_root / job_id
    with pytest.raises(tj.JobNotFound, match="no comparison_index.json"):
        jobs.file(job_id, INDEX_FILE)                                               # nothing built yet

    comparison = out / COMPARISON_DIR
    comparison.mkdir()
    write_json_atomic(comparison / INDEX_FILE, {"evaluationReport": "evaluation_report_g.json", "groups": [
        {"czml": "comparison_KRDU_05L_g.czml"}, {"czml": "comparison_KRDU_05R_g.czml"}]})
    for name in ("comparison_KRDU_05L_g.czml", "comparison_KRDU_05R_g.czml", "evaluation_report_g.json",
                 "comparison_KRDU_05L_old.czml", "secret.json"):
        (comparison / name).write_text("[]")

    for listed in (INDEX_FILE, "comparison_KRDU_05L_g.czml", "comparison_KRDU_05R_g.czml", "evaluation_report_g.json"):
        assert jobs.file(job_id, listed) == comparison / listed
    (comparison / "comparison_KRDU_05R_g.czml").unlink()                            # listed, but gone from the disk
    with pytest.raises(tj.JobNotFound, match="not on disk"):
        jobs.file(job_id, "comparison_KRDU_05R_g.czml")
    for unlisted in ("comparison_KRDU_05L_old.czml", "secret.json", "../state.json", "../records/summary.json",
                     "records", "", "/etc/passwd", f"../../{job_id}/state.json"):
        with pytest.raises(tj.JobNotFound):
            jobs.file(job_id, unlisted)


# ── the process model: threads, a job that outlives its backend, the slot, the directories ────────────

def test_the_job_runs_on_one_solver_thread_whatever_the_backends_environment(stack, monkeypatch):
    jobs, behaviour, *_ = stack
    for name in SOLVER_THREAD_ENV:
        monkeypatch.setenv(name, "8")
    behaviour["now"] = "env"
    job_id = jobs.start(M1)["jobId"]
    finished(jobs, job_id)
    seen = json.loads((jobs.jobs_root / job_id / "env.json").read_text())
    assert seen == {name: "1" for name in SOLVER_THREAD_ENV} and len(SOLVER_THREAD_ENV) == 5


def test_the_job_directory_records_the_jobs_process_its_group_its_start_and_the_boot(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "hang"
    job_id = jobs.start(M1)["jobId"]
    pids = wait_for(lambda: json.loads(p.read_text()) if (p := jobs.jobs_root / job_id / "pids.json").is_file() else None)
    record = json.loads((jobs.jobs_root / job_id / PROCESS_FILE).read_text())
    identity = tj.process_identity(pids["pid"])
    assert record == {"pgid": pids["pid"], "startTime": identity.start_time, "bootId": identity.boot_id}
    assert record["bootId"] == (tj.BOOT_ID_FILE.read_text().strip()) and record["startTime"].isdigit()
    jobs.cancel(job_id)


def test_the_identity_of_a_process_is_read_from_proc_whatever_its_name():
    me = tj.process_identity(os.getpid())
    assert me.state in {"R", "S"} and me.start_time == tj.process_identity(os.getpid()).start_time
    assert tj.process_identity(2 ** 22 + 12345) is None                       # no such process
    # a command name with spaces and parentheses must not shift the fields: field 22 is counted after the last ")"
    child = subprocess.Popen([sys.executable, "-c", "import os; os.execv('/bin/sleep', ['(a b) c)', '30'])"])
    try:
        wait_for(lambda: Path(f"/proc/{child.pid}/stat").read_text().startswith(f"{child.pid} (sleep)"))
        found = tj.process_identity(child.pid)
        assert found.start_time == str(int(found.start_time)) and found.state in {"R", "S"}
        assert found.start_time == tj.process_identity(child.pid).start_time
    finally:
        child.kill()
        child.wait()


def other_backend(jobs):
    """A second manager over the same jobs root: another backend, one that did not start the job. The first is gone:
    its job process is reaped by init in real life, here by a thread (a zombie would keep its group "alive")."""
    for child in jobs._children.values():
        threading.Thread(target=child.wait, daemon=True).start()
    jobs._children.clear()
    return tj.TrafficJobs(jobs.jobs_root, harvest_root=jobs.harvest_root, command=jobs._command,
                          typecode_of=jobs._typecode_of)


def test_a_job_that_outlives_its_backend_still_holds_the_slot_reports_running_and_can_be_cancelled(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "hang"
    job_id = jobs.start(M1)["jobId"]
    wait_for(lambda: (jobs.jobs_root / job_id / "pids.json").is_file())
    pids = json.loads((jobs.jobs_root / job_id / "pids.json").read_text())

    successor = other_backend(jobs)                       # the backend was restarted: it never saw this process
    assert successor.status(job_id)["state"] == "running"
    with pytest.raises(tj.JobBusy):
        successor.start(M1)
    assert successor.cancel(job_id)["state"] == "cancelled"
    wait_for(lambda: not group_alive(pids["pid"]), timeout=5.0)
    behaviour["now"] = "finish"
    finished(successor, successor.start(M1)["jobId"])     # the slot is free again


def test_a_job_that_exits_without_a_state_is_failed_on_disk_at_once_with_its_exit_code(stack, monkeypatch):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "crash"
    job_id = jobs.start(M1)["jobId"]
    status = finished(jobs, job_id)
    assert status["state"] == "failed" and "exited with code 3" in status["error"] and "state.json" in status["error"]
    on_disk = json.loads((jobs.jobs_root / job_id / STATE_FILE).read_text())
    assert on_disk["state"] == "failed" and "exited with code 3" in on_disk["error"]

    # from then on the job's process group number is never consulted again, by this backend or another
    def forbidden(*_a, **_k):
        raise AssertionError("the process group of a settled job was consulted")
    monkeypatch.setattr(tj.os, "killpg", forbidden)
    monkeypatch.setattr(tj, "process_identity", forbidden)
    for manager in (jobs, other_backend(jobs)):
        assert manager.status(job_id)["state"] == "failed"
        assert manager.cancel(job_id)["state"] == "failed"
        manager._running()


def test_a_foreign_job_whose_process_is_gone_without_a_state_is_failed_on_disk(stack):
    jobs, *_ = stack
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    identity = tj.process_identity(gone.pid)                      # the identity it had while it lived
    gone.wait()
    job_id = "20260101T000000000000Z-00000001"
    (jobs.jobs_root / job_id).mkdir(parents=True)
    write_json_atomic(jobs.jobs_root / job_id / PROCESS_FILE, {
        "pgid": gone.pid, "startTime": identity.start_time, "bootId": identity.boot_id})

    status = other_backend(jobs).status(job_id)
    assert status["state"] == "failed" and "process is gone" in status["error"]
    assert json.loads((jobs.jobs_root / job_id / STATE_FILE).read_text())["state"] == "failed"


def reused_group(jobs, **changed):
    """A job directory whose process.json names the pgid of a process that is NOT the job: the test's own process group
    leader (a long-lived sleeper), with the recorded identity altered by ``changed``."""
    sleeper = subprocess.Popen(["sleep", "30"], start_new_session=True)
    identity = tj.process_identity(sleeper.pid)
    job_id = "20260101T000000000000Z-00000002"
    (jobs.jobs_root / job_id).mkdir(parents=True)
    record = {"pgid": sleeper.pid, "startTime": identity.start_time, "bootId": identity.boot_id, **changed}
    write_json_atomic(jobs.jobs_root / job_id / PROCESS_FILE, record)
    return job_id, sleeper


@pytest.mark.parametrize("changed", [{"startTime": "1"}, {"bootId": "another-boot"}])
def test_a_pgid_that_now_belongs_to_another_process_is_not_the_job_and_is_never_signalled(stack, monkeypatch, changed):
    jobs, *_ = stack
    job_id, impostor = reused_group(jobs, **changed)
    sent = []
    monkeypatch.setattr(tj.os, "killpg", lambda *args: sent.append(args))
    try:
        manager = other_backend(jobs)
        assert manager._running() is None                           # it holds no slot
        status = manager.cancel(job_id)
        assert status["state"] == "failed" and "process is gone" in status["error"]
        assert sent == []                                           # not one signal to a group that is not the job's
        assert impostor.poll() is None                              # and the impostor is alive and unharmed
        manager.start(M1)                                           # no 409 either
    finally:
        impostor.kill()
        impostor.wait()


def test_the_same_process_with_the_recorded_identity_is_the_job(stack):
    jobs, *_ = stack
    job_id, sleeper = reused_group(jobs)                           # (the control: nothing altered)
    try:
        manager = other_backend(jobs)
        assert manager._running() == job_id
        with pytest.raises(tj.JobBusy):
            manager.start(M1)
        assert manager.cancel(job_id)["state"] == "cancelled"
        assert sleeper.wait(timeout=10) == -signal.SIGTERM
    finally:
        sleeper.kill()
        sleeper.wait()


def test_a_zombie_leader_is_not_a_running_job(stack):
    jobs, *_ = stack
    job_id, sleeper = reused_group(jobs)
    sleeper.kill()
    wait_for(lambda: tj.process_identity(sleeper.pid).state == "Z")   # exited, not yet reaped
    try:
        assert other_backend(jobs)._running() is None
    finally:
        sleeper.wait()


def test_no_exited_child_is_left_behind_a_finished_job(stack):
    jobs, *_ = stack
    job_id = jobs.start(M1)["jobId"]
    assert finished(jobs, job_id)["state"] == "done"
    assert jobs._children == {}                                     # polled and dropped, not a zombie for the backend's life
    pid = json.loads((jobs.jobs_root / job_id / PROCESS_FILE).read_text())["pgid"]
    assert tj.process_identity(pid) is None                         # reaped: no <defunct> entry


def test_a_running_job_stays_in_children_and_an_exited_one_leaves_at_the_next_status(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "hang"
    running = jobs.start(M1)["jobId"]
    wait_for(lambda: (jobs.jobs_root / running / "pids.json").is_file())
    assert jobs.status(running)["state"] == "running" and list(jobs._children) == [running]
    jobs.cancel(running)
    assert jobs._children == {}


def test_pruning_spares_a_running_job_by_name_whatever_its_age(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "hang"
    old = jobs.start(M1)["jobId"]                          # the oldest directory, alive
    wait_for(lambda: (jobs.jobs_root / old / "pids.json").is_file())
    newer = [f"{99990101 + i}T000000000000Z-{i:08x}" for i in range(6)]
    for job_id in newer:                                   # six newer finished jobs
        (jobs.jobs_root / job_id).mkdir()
        write_json_atomic(jobs.jobs_root / job_id / STATE_FILE, {"state": "done", "error": None})

    jobs._prune()
    kept = sorted(p.name for p in jobs.jobs_root.iterdir())
    assert old in kept                                     # by sort order it would have been the first to go
    assert kept == sorted([old, *newer[2:]])               # room for one more: the two oldest finished ones went
    jobs.cancel(old)


def test_the_oldest_directories_go_before_the_new_job_starts_not_after(stack, monkeypatch):
    jobs, *_ = stack
    for i in range(5):
        job_id = f"{20260101 + i}T000000000000Z-{i:08x}"
        (jobs.jobs_root / job_id).mkdir(parents=True, exist_ok=True)
        write_json_atomic(jobs.jobs_root / job_id / STATE_FILE, {"state": "done", "error": None})
    directories_at_launch = []
    real = subprocess.Popen

    def spy(*args, **kwargs):
        directories_at_launch.append(len(list(jobs.jobs_root.iterdir())))
        return real(*args, **kwargs)
    monkeypatch.setattr(tj.subprocess, "Popen", spy)

    finished(jobs, jobs.start(M1)["jobId"])
    assert directories_at_launch == [5]                    # four kept and the new one: never six, even for a moment
    assert len(list(jobs.jobs_root.iterdir())) == 5


def test_a_child_of_the_job_does_not_outlive_its_main_process(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "orphan"
    job_id = jobs.start(M1)["jobId"]
    pids = wait_for(lambda: json.loads(p.read_text()) if (p := jobs.jobs_root / job_id / "pids.json").is_file() else None)
    time.sleep(0.3)                                        # the main process exits at once, the child sleeps on

    assert jobs.status(job_id)["state"] == "failed"        # the slot is freed ...
    wait_for(lambda: not group_alive(pids["pid"]), timeout=5.0)     # ... only once the child is dead
    with pytest.raises(ProcessLookupError):
        os.kill(pids["child"], 0)


def test_a_cancel_never_overwrites_a_state_that_says_done(stack, monkeypatch):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "hang"
    job_id = jobs.start(M1)["jobId"]
    wait_for(lambda: (jobs.jobs_root / job_id / "pids.json").is_file())
    real_stop = jobs._stop

    def finish_while_stopping(job):
        # the job wrote its result in the instant before the signal reached it
        write_json_atomic(jobs.jobs_root / job / STATE_FILE, {"state": "done", "error": None, "summary": {"windows": 1},
                                                              "perAircraft": {}, "timing": {"totalS": 1.0, "phases": {}},
                                                              "stayedRecords": {}})
        real_stop(job)
    monkeypatch.setattr(jobs, "_stop", finish_while_stopping)

    answer = jobs.cancel(job_id)
    assert answer["state"] == "done" and answer["summary"] == {"windows": 1}
    assert json.loads((jobs.jobs_root / job_id / STATE_FILE).read_text())["state"] == "done"


def test_a_finished_jobs_lingering_process_does_not_hold_the_slot_or_get_cancelled(stack):
    jobs, behaviour, *_ = stack
    behaviour["now"] = "done_then_hang"
    job_id = jobs.start(M1)["jobId"]
    wait_for(lambda: (jobs.jobs_root / job_id / STATE_FILE).is_file())
    assert jobs.status(job_id)["state"] == "done"
    assert jobs.cancel(job_id)["state"] == "done"
    behaviour["now"] = "finish"
    assert finished(jobs, jobs.start(M1)["jobId"])["state"] == "done"
    pgid = json.loads((jobs.jobs_root / job_id / PROCESS_FILE).read_text())["pgid"]
    tj._signal_group(pgid, 9)                              # (the test's own cleanup)


# ── stopping the backend ─────────────────────────────────────────────────────

def test_the_sigterm_handler_cancels_the_running_job_then_terminates(stack):
    from aeroviz_backend import http_server
    jobs, behaviour, *_ = stack
    behaviour["now"] = "hang"
    job_id = jobs.start(M1)["jobId"]
    pids = wait_for(lambda: json.loads(p.read_text()) if (p := jobs.jobs_root / job_id / "pids.json").is_file() else None)
    app = http_server.AeroVizBackendApp(
        simulation_backend=object(), optimization_backend=object(), dynamics_comparison_backend=object(),
        observed_trajectory_backend=object(), traffic_jobs=jobs)
    terminated = []

    http_server.stop_traffic_jobs_then_terminate(app, terminate=terminated.append)(signal.SIGTERM, None)

    assert terminated == [signal.SIGTERM]                    # after the job was stopped, not before
    assert jobs.status(job_id)["state"] == "cancelled"
    wait_for(lambda: not group_alive(pids["pid"]), timeout=5.0)


BACKEND_STUB = textwrap.dedent('''
    import json, signal, sys
    root, harvest, fake, names = sys.argv[1:5]
    sys.path.insert(0, sys.argv[5])
    from aeroviz_backend import http_server, traffic_jobs as tj

    def command(spec, out):
        return [sys.executable, fake, "--spec", str(spec), "--out", str(out), "--behaviour", "hang", "--vars", names]

    jobs = tj.TrafficJobs(root, harvest_root=harvest, command=command, typecode_of=lambda icao24: None)
    app = http_server.AeroVizBackendApp(
        simulation_backend=object(), optimization_backend=object(), dynamics_comparison_backend=object(),
        observed_trajectory_backend=object(), traffic_jobs=jobs)
    signal.signal(signal.SIGTERM, http_server.stop_traffic_jobs_then_terminate(app))     # as main() installs it
    print(json.dumps(jobs.start({"mode": "m1", "airport": "KRDU", "flightKey": "AAA_05L"})), flush=True)
    signal.pause()
''')


def test_stopping_a_backend_with_sigterm_stops_its_job_and_the_backend_dies_of_the_signal(stack):
    jobs, _behaviour, _asked, tmp_path = stack
    backend = subprocess.Popen(
        [sys.executable, "-c", BACKEND_STUB, str(jobs.jobs_root), str(tmp_path / "harvest"), str(tmp_path / "fake_job.py"),
         ",".join(SOLVER_THREAD_ENV), str(tj.paths.REPO_ROOT)],
        stdout=subprocess.PIPE, text=True, cwd=tj.paths.REPO_ROOT)
    try:
        job_id = json.loads(backend.stdout.readline())["jobId"]
        pids = wait_for(lambda: json.loads(p.read_text()) if (p := jobs.jobs_root / job_id / "pids.json").is_file() else None)
        backend.send_signal(signal.SIGTERM)
        assert backend.wait(timeout=20) == -signal.SIGTERM        # the supervisor sees the signal, as without a handler
    finally:
        if backend.poll() is None:
            backend.kill()
    wait_for(lambda: not group_alive(pids["pid"]), timeout=5.0)
    assert json.loads((jobs.jobs_root / job_id / STATE_FILE).read_text())["state"] == "cancelled"


def test_a_process_reaped_between_open_and_read_has_no_identity(monkeypatch):
    """Linux answers ESRCH (ProcessLookupError) when the process goes between the open of /proc/<pid>/stat and
    the read: that is "no process", never an error out of status/start/cancel."""
    def gone(self, *a, **k):
        raise ProcessLookupError(3, "No such process")
    monkeypatch.setattr(tj.Path, "read_text", gone)
    assert tj.process_identity(12345) is None
