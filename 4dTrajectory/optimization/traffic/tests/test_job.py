"""The interactive traffic job end to end into a tmp directory, on fake fly functions (no NLP, no tracks, no data):
which tracks it reads, the records and summary it writes with the batch's writers, its progress file, the steps it
runs, and its final state."""

import json
from pathlib import Path

import numpy as np
import pytest

from aerodynamic_model.common import GeodeticState, LoadFactorControl
from aerodynamic_model.rollout import RolloutSample
from aircraft.aero_params import aero_params_for_aircraft
from aircraft.aircraft_sets import A320

import evaluation_export as ee
import traffic_job as tj
import traffic_optimization as to
from flight_scenarios import FlightScenario
from flight_scenarios.identity import flight_key
from scenario_replay import ScenarioOptimization
from traffic import M1_MODE, M2_MODE
from traffic.block import BLOCK_RECORD_SCHEMA
from traffic.loop import TRAFFIC_RECORD_SCHEMA
from traffic.scene import RecordedFlight, Traffic
from trajectory_data_process.harvest.arrivals import SCHEMA_VERSION
from trajectory_data_process.harvest.utc import iso_utc, iso_utc_ms
import traffic_job_files as files

T0 = 1_800_000_000.0
TARGET = GeodeticState(35.59, -78.49, 500.0, 80.0, 1.5, -0.05, A320.landing_mass)
CALLSIGNS = {"A": "AAA1", "B": "BBB2", "C": "CCC3", "D": "DDD4", "E": "EEE5", "F": "FFF6"}


def source(name, entry_s, duration_s=200.0, runway="05L"):
    """The flight dict as ``load_model_arrivals_subset`` returns it, as far as the job reads it."""
    src = {"id": CALLSIGNS[name], "icao24": "ad7f04", "runway": runway, "arr_airport": "KRDU",
           "landing_time_utc": iso_utc(T0 + entry_s + duration_s), "entry_time_utc": iso_utc_ms(T0 + entry_s),
           "target_source": "runway_threshold", "window_s": 15.0}
    src["flight_key"] = flight_key(src, 0)
    return {**src, "entry_s": entry_s, "duration_s": duration_s, "runway_target": {"runway": runway, "lat": 35.0}}


#: A is the commanded flight (M1) / in the block with B (M2). B enters 300 s after A, inside A's 2000 s window; C is
#: in the air at A's entry (it lands 50 s after it, before the block of the M2 tests); D and E are far away.
#: F is the only arrival of runway 05R, a day away: no window or block reads its track, only its runway target.
FLIGHTS = {"A": source("A", 0.0), "B": source("B", 300.0), "C": source("C", -100.0, 150.0),
           "D": source("D", -5000.0), "E": source("E", 9000.0), "F": source("F", 90000.0, runway="05R")}


def manifest(tmp_path):
    rows = [{"flight_key": f["flight_key"], "callsign": f["id"], "icao24": f["icao24"], "runway": f["runway"],
             "entry_time_utc": f["entry_time_utc"], "landing_time_utc": f["landing_time_utc"],
             "arrival_duration_s": f["duration_s"]} for f in FLIGHTS.values()]
    path = tmp_path / "arrivals" / "manifest.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"records": rows}))
    return path


def recorded(flight):
    t = T0 + flight["entry_s"] + np.array([0.0, flight["duration_s"]])
    return RecordedFlight(flight["flight_key"], flight["runway"], "A320", t, np.full(2, 35.6), np.full(2, -78.5), np.full(2, 900.0))


def scenario_of(flight):
    initial = GeodeticState(35.6, -78.5, 2000.0, 130.0, 1.5, -0.05, A320.mass.max_takeoff_kg)
    src = {k: v for k, v in flight.items() if k not in ("entry_s", "duration_s", "runway_target")}
    return FlightScenario(initial, A320, aero_params_for_aircraft(A320), src, TARGET)


def flown(scenario):
    """A solved record and its sidecar, as the batch workers return them."""
    control = LoadFactorControl(thrust=1.0e5, bank_rad=0.0, load_factor=1.0)
    rollout = [RolloutSample(0.0, scenario.initial, control, 0), RolloutSample(10.0, TARGET, control, 0)]
    result = ScenarioOptimization(scenario.source, 10.0, [], [], evaluation=ee.evaluation_record(
        scenario.initial, TARGET, rollout, scenario.source, subject="optimized"))
    record = result.to_dict()
    record["sidecar"] = {"outcome": "separated_at_baseline", "rounds": [
        {"final_time_s": 100.0, "losses": [], "background_losses": 0}], "ifr_losses": {"baseline": [], "final": []},
        "starts_in_loss": {}, "uncategorised_types": [], "runways_without_faf": [], "frame_east_error_max": 0.001}
    evaluation = dict(result.evaluation)
    evaluation["states"] = []
    return record, evaluation


@pytest.fixture
def stack(monkeypatch, tmp_path):
    """The job on fakes. ``seen`` records what it did: the keys it loaded, the steps it ran, the windows it flew."""
    seen = {"loaded": [], "steps": [], "windows": [], "progress": []}

    def load(manifest_path, keys):
        seen["loaded"].append(sorted(keys))
        return [f for f in FLIGHTS.values() if f["flight_key"] in keys]

    def run_step(command):
        seen["steps"].append(command)
        if command[2] == "evaluation":
            Path(command[command.index("--output") + 1]).write_text(json.dumps({"trajectories": []}))
        else:
            comparison = Path(command[command.index("--output-dir") + 1])
            comparison.mkdir()
            (comparison / files.INDEX_FILE).write_text(json.dumps({"groups": []}))

    monkeypatch.setattr(tj, "load_model_arrivals_subset", load)
    # as the real one: the runway targets are the loaded flights' (first of each runway)
    monkeypatch.setattr(tj, "traffic_from_arrivals", lambda flights: Traffic(
        "KRDU", tuple(recorded(f) for f in flights), {f["runway"]: f["runway_target"] for f in reversed(flights)}))
    monkeypatch.setattr(tj, "build_scenario", lambda flight, airport, target_from_threshold: scenario_of(flight))
    monkeypatch.setattr(to, "build_scenario", lambda flight, airport, target_from_threshold: scenario_of(flight))
    monkeypatch.setattr(tj, "run_step", run_step)
    return seen


def spec_m1(tmp_path, key="A"):
    return {"mode": "m1", "airport": "KRDU", "manifest": str(manifest(tmp_path)),
            "flightKey": FLIGHTS[key]["flight_key"]}


def spec_m2(tmp_path, block_s=900):
    return {"mode": "m2", "airport": "KRDU", "manifest": str(manifest(tmp_path)),
            "blockStartUtc": iso_utc(T0 + 60.0), "blockS": block_s}


def progress_log(monkeypatch, seen):
    real = tj.write_progress

    def spy(out, done, total, current):
        seen["progress"].append((done, total, current))
        real(out, done, total, current)
    monkeypatch.setattr(tj, "write_progress", spy)


def read(path):
    return json.loads(path.read_text())


# ── M1 ───────────────────────────────────────────────────────────────────────

def test_an_m1_job_reads_only_the_tracks_that_can_share_its_window(monkeypatch, tmp_path, stack):
    def fly(payload):
        index, scenario, params = payload
        stack["windows"].append(sorted(f.flight_key for f in scenario.traffic.flights))
        stack["params"] = params
        record, evaluation = flown(scenario)
        return index, scenario.source["flight_key"], record, evaluation, None

    monkeypatch.setattr(tj, "_fly_one_window", fly)
    tj.run_job(spec_m1(tmp_path), tmp_path / "job")

    own, b, c = (FLIGHTS[k]["flight_key"] for k in "ABC")
    # B enters inside A's window, C is in the air at A's entry; D landed long before and E enters long after the
    # window: their tracks are never read
    near, extra = stack["loaded"]
    assert near == sorted([own, b, c])
    assert extra == [FLIGHTS["F"]["flight_key"]]                   # and F alone, for the runway target of 05R
    # and the window the commanded aircraft flies in is what the batch's own window of the FULL roster would be
    full = Traffic("KRDU", tuple(recorded(f) for f in FLIGHTS.values()), {})
    assert stack["windows"] == [sorted(f.flight_key for f in [full.flight(own), *full.airborne(T0, T0 + 2000.0, exclude=own)])]
    assert stack["windows"] == [sorted([own, b, c])]
    assert stack["params"]["max_duration"] == 2000.0 and stack["params"]["rollout_dt_s"] == 0.5
    assert stack["params"]["solve_options"] == {"verbose": False, "max_iterations": 3000}      # the batch defaults (IM2)


def test_the_jobs_traffic_carries_the_runway_targets_of_the_whole_airport(monkeypatch, tmp_path, stack):
    # The near set holds runway 05L only; the roster also has 05R (F, a day away). The batch's Traffic carries both
    # targets (rules.separation, the runway frames and runways_without_faf read all of them), so the job's must.
    seen = {}

    def fly(payload):
        index, scenario, _params = payload
        seen["targets"] = scenario.traffic.runway_targets
        record, evaluation = flown(scenario)
        return index, scenario.source["flight_key"], record, evaluation, None

    monkeypatch.setattr(tj, "_fly_one_window", fly)
    tj.run_job(spec_m1(tmp_path), tmp_path / "job")

    assert list(seen["targets"]) == ["05L", "05R"]                           # in roster order, as the batch's
    assert seen["targets"]["05R"] == FLIGHTS["F"]["runway_target"]
    assert stack["loaded"][1] == [FLIGHTS["F"]["flight_key"]]               # one arrival of 05R, nothing else


def test_an_m2_jobs_traffic_carries_them_too(monkeypatch, tmp_path, stack):
    seen = {}
    real = fake_block(stack)

    def fly(payload, on_progress):
        seen["targets"] = payload[2].runway_targets
        return real(payload, on_progress)

    monkeypatch.setattr(tj, "_fly_one_block", fly)
    tj.run_job(spec_m2(tmp_path), tmp_path / "job")
    assert list(seen["targets"]) == ["05L", "05R"]


def test_no_extra_track_is_read_when_the_near_set_already_covers_every_runway(monkeypatch, tmp_path, stack):
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (payload[0], payload[1].source["flight_key"],
                                                                *flown(payload[1]), None))
    spec = spec_m1(tmp_path, key="F")                  # F itself is the 05R arrival; its window holds nobody else
    tj.run_job(spec, tmp_path / "job")
    assert len(stack["loaded"]) == 2                   # the near set, then 05L's target (05R is in the near set)
    assert stack["loaded"][0] == [FLIGHTS["F"]["flight_key"]]
    assert stack["loaded"][1] and FLIGHTS["F"]["flight_key"] not in stack["loaded"][1]


def test_an_m1_job_writes_the_batch_records_and_summary_and_runs_the_steps_without_a_category(
        monkeypatch, tmp_path, stack):
    progress_log(monkeypatch, stack)
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (payload[0], payload[1].source["flight_key"],
                                                                *flown(payload[1]), None))
    out = tmp_path / "job"
    summary = tj.run_job(spec_m1(tmp_path), out)

    records = out / files.RECORDS_DIR
    roster = read(records / "summary.json")
    assert roster["mode"] == M1_MODE and (roster["total"], roster["solved"]) == (1, 1)
    config = roster["optimization_config"]
    assert config["mode"] == "constrained_iaf" and config["max_duration_s"] == 2000.0 and config["max_iterations"] == 3000
    assert config["traffic"]["selection"]["manifest"] == str(tmp_path / "arrivals" / "manifest.json")
    assert config["traffic"]["selection"]["flight_key"] == FLIGHTS["A"]["flight_key"]
    assert config["traffic"]["selection"]["sample"] == 1 and config["traffic"]["selection"]["traffic_tracks_read"] == 3
    name = roster["results"][0]["states_file"]
    assert (records / name.replace("_states.json", "_traffic.json")).is_file()          # the sidecar beside the record
    assert (records / name.replace("_states.json", "_eval.json")).is_file()
    assert summary["outcomes"] == {"separated_at_baseline": 1}                          # traffic.readout's count

    evaluation, builder = stack["steps"]
    assert evaluation[1:3] == ["-m", "evaluation"] and evaluation[evaluation.index("--input") + 1] == str(records)
    assert evaluation[evaluation.index("--output") + 1] == str(records / files.REPORT_FILE)
    assert builder[1].endswith("build_scenario_comparison_czml.py")
    assert builder[builder.index("--summary") + 1] == str(records / "summary.json")
    assert builder[builder.index("--output-dir") + 1] == str(out / files.COMPARISON_DIR)
    assert builder[builder.index("--evaluation-report") + 1] == str(records / files.REPORT_FILE)
    assert "--category" not in builder and "--category-label" not in builder             # no categories.json entry
    assert stack["progress"] == [(0, 1, None), (1, 1, FLIGHTS["A"]["flight_key"])]


def test_a_finished_job_ends_with_state_json_and_the_readout(monkeypatch, tmp_path, stack):
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (payload[0], payload[1].source["flight_key"],
                                                                *flown(payload[1]), None))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_m1(tmp_path)))
    out = tmp_path / "job"

    assert tj.main(["--spec", str(spec_path), "--out", str(out)]) == 0
    state = read(out / files.STATE_FILE)
    assert state["state"] == "done" and state["error"] is None and state["summary"]["windows"] == 1
    assert read(out / files.PROGRESS_FILE) == {"done": 1, "total": 1, "current": FLIGHTS["A"]["flight_key"]}
    assert not list(out.glob("*.tmp"))                                                   # written atomically


def test_a_job_that_fails_ends_with_state_json_naming_the_reason(monkeypatch, tmp_path, stack):
    spec = spec_m1(tmp_path)
    spec["flightKey"] = "NOPE_05L_ad7f04_20260101T000000Z"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    out = tmp_path / "job"

    assert tj.main(["--spec", str(spec_path), "--out", str(out)]) == 1
    state = read(out / files.STATE_FILE)
    assert state["state"] == "failed" and "NOPE_05L" in state["error"] and state["error"].startswith("ValueError")
    assert stack["loaded"] == []                                                         # nothing read for an unknown key


def test_a_step_that_fails_fails_the_job(monkeypatch, tmp_path, stack):
    import subprocess
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (payload[0], payload[1].source["flight_key"],
                                                                *flown(payload[1]), None))

    def broken(command):
        raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(tj, "run_step", broken)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_m1(tmp_path)))
    out = tmp_path / "job"
    assert tj.main(["--spec", str(spec_path), "--out", str(out)]) == 1
    assert read(out / files.STATE_FILE)["state"] == "failed"


def test_an_aircraft_without_dynamics_fails_the_job_with_the_reason(monkeypatch, tmp_path, stack):
    from flight_scenarios.scenario import NoAircraftDynamics

    def none(flight, airport, target_from_threshold):
        raise NoAircraftDynamics("ZZZZ", "nothing models it")
    monkeypatch.setattr(tj, "build_scenario", none)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_m1(tmp_path)))
    out = tmp_path / "job"
    assert tj.main(["--spec", str(spec_path), "--out", str(out)]) == 1
    assert "no aircraft dynamics" in read(out / files.STATE_FILE)["error"]


# ── M2 ───────────────────────────────────────────────────────────────────────

def fake_block(stack, error=None):
    def fly(payload, on_progress):
        label, scenarios, traffic, params = payload
        stack["block"] = {"scenarios": sorted(s.source["flight_key"] for s in scenarios),
                          "traffic": sorted(f.flight_key for f in traffic.flights)}
        out = []
        for done, scenario in enumerate(scenarios, 1):
            record, evaluation = flown(scenario)
            record["sidecar"]["schema"] = "optimization-traffic-block-v2"
            out.append((scenario, record, evaluation, None))
            on_progress(done, len(scenarios), scenario.source["flight_key"])
        none = {"answered": 0, "not_answered": 0, "background": 0}
        summary = {"aircraft": len(scenarios), "scheduled": len(scenarios), "eta_failed": 0, "slot_failed": 0,
                   "schedule_speed_mps": 70.0,
                   "slots": [{"flight_key": s.source["flight_key"], "runway": "05L", "eta_utc_s": 0.0,
                              "cta_utc_s": 0.0, "delay_s": 0.0} for s in scenarios],
                   "outcomes": {s.source["flight_key"]: "separated_at_baseline" for s in scenarios},
                   "final_losses": {s.source["flight_key"]: {"visual": none, "ifr": none} for s in scenarios}}
        if error:
            summary["error"] = error
        return label, out, summary
    return fly


def test_an_m2_job_reads_the_tracks_of_its_block_and_what_flies_beside_it(monkeypatch, tmp_path, stack):
    progress_log(monkeypatch, stack)
    monkeypatch.setattr(tj, "_fly_one_block", fake_block(stack))
    out = tmp_path / "job"
    summary = tj.run_job(spec_m2(tmp_path), out)

    a, b, c = (FLIGHTS[k]["flight_key"] for k in "ABC")
    # the block [T0 + 60, T0 + 960) holds A (lands T0 + 200) and B (lands T0 + 500); C landed at T0 + 50, just before
    # it, and is in the air at A's entry; D landed 4800 s before it and E enters 9000 s in, long after B's entry plus
    # the horizon
    assert stack["loaded"] == [sorted([a, b, c]), [FLIGHTS["F"]["flight_key"]]]
    assert stack["block"]["scenarios"] == sorted([a, b])
    assert stack["block"]["traffic"] == sorted([a, b, c])
    assert stack["progress"] == [(0, 2, None), (1, 2, a), (2, 2, b)]

    roster = read(out / files.RECORDS_DIR / "summary.json")
    assert roster["mode"] == M2_MODE and (roster["total"], roster["solved"]) == (2, 2)
    assert [blk["label"] for blk in roster["blocks"]] == ["block_00"] and roster["blocks"][0]["skipped_no_dynamics"] == 0
    selection = roster["optimization_config"]["traffic"]["selection"]
    assert selection["blocks"] == {"start": iso_utc(T0 + 60.0), "block_s": 900, "count": 1}
    assert selection["manifest"] == str(tmp_path / "arrivals" / "manifest.json")
    assert summary["aircraft"] == 2 and summary["flown_aircraft_with_a_loss_left_after_the_block"]["visual"] == {
        "answered": 0, "not_answered": 0}
    assert all("--category" not in step for step in stack["steps"])


def test_an_m2_job_whose_block_fails_fails_with_its_reason(monkeypatch, tmp_path, stack):
    monkeypatch.setattr(tj, "_fly_one_block", fake_block(stack, error="RuntimeError: casadi"))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_m2(tmp_path)))
    out = tmp_path / "job"
    assert tj.main(["--spec", str(spec_path), "--out", str(out)]) == 1
    assert read(out / files.STATE_FILE)["error"] == "RuntimeError: the block failed: RuntimeError: casadi"
    assert not (out / files.COMPARISON_DIR).exists()


def test_an_m2_job_over_a_block_with_no_landing_or_a_wrong_length_fails(monkeypatch, tmp_path, stack):
    monkeypatch.setattr(tj, "_fly_one_block", fake_block(stack))
    empty = spec_m2(tmp_path)
    empty["blockStartUtc"] = iso_utc(T0 + 3000.0)                    # nothing lands in 3000 .. 3900 s
    with pytest.raises(ValueError, match="no arrival lands"):
        tj.run_job(empty, tmp_path / "job1")
    with pytest.raises(ValueError, match=r"block length 600"):
        tj.run_job(spec_m2(tmp_path, block_s=600), tmp_path / "job2")
    assert stack["loaded"] == []


def test_a_job_of_another_mode_is_refused(tmp_path, stack):
    with pytest.raises(ValueError, match="job mode 'm3'"):
        tj.run_job({"mode": "m3"}, tmp_path / "job")


# ── the real comparison builder over a job's directory ───────────────────────

def real_flown(scenario, *, block):
    """``flown`` with a real replay (the builder draws it) and the sidecar of the real loop / block."""
    record, evaluation = flown(scenario)
    key = scenario.source["flight_key"]
    record["simulator_states"] = [
        {"t": float(t), "lat": 35.60 - 0.001 * t, "lon": -78.5, "alt": 2000.0 - 10.0 * t, "V": 130.0,
         "psi": 1.5, "gamma": -0.05, "m": 70000.0} for t in range(0, 11)]
    record["optimizer_states"] = record["simulator_states"]
    record["source"] = {**record["source"], "hae_minus_msl_m": -32.0}      # what the comparison builder adds back
    record["sidecar"] = {**record["sidecar"], "flight_key": key,
                         "schema": BLOCK_RECORD_SCHEMA if block else TRAFFIC_RECORD_SCHEMA,
                         "recorded": [FLIGHTS["C"]["flight_key"]]}
    if block:
        record["sidecar"]["slot"] = {"eta_utc_s": 1.0, "cta_utc_s": 4.0, "delay_s": 3.0}
    return record, evaluation


def only_the_builder_runs_for_real(stack, monkeypatch, tmp_path):
    """The evaluation step is faked (its context data is not a test's business); the builder runs as the job runs it."""
    import subprocess

    def run_step(command):
        stack["steps"].append(command)
        if command[2] == "evaluation":
            Path(command[command.index("--output") + 1]).write_text(json.dumps({"trajectories": []}))
        else:
            subprocess.run(command, cwd=tj._REPO_ROOT, check=True)
    monkeypatch.setattr(tj, "run_step", run_step)
    path = tmp_path / "arrivals" / "manifest.json"
    manifest_rows = json.loads(path.read_text())["records"]
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "records": manifest_rows}))


def test_the_real_builder_reads_an_m1_jobs_directory_and_registers_no_category(monkeypatch, tmp_path, stack):
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (
        payload[0], payload[1].source["flight_key"], *real_flown(payload[1], block=False), None))
    spec = spec_m1(tmp_path)
    only_the_builder_runs_for_real(stack, monkeypatch, tmp_path)
    out = tmp_path / "job"
    tj.run_job(spec, out)

    comparison = out / files.COMPARISON_DIR
    index = read(comparison / files.INDEX_FILE)
    [window] = index["groups"]
    assert window["group"] == FLIGHTS["A"]["flight_key"] and window["traffic"]["outcome"] == "separated_at_baseline"
    assert window["traffic"]["recorded"] == [f"ref-{FLIGHTS['C']['flight_key']}"]
    assert "scene" not in index
    # every file the index lists is there, and nothing else registers the directory as a category
    assert (comparison / window["czml"]).is_file() and (comparison / index["evaluationReport"]).is_file()
    assert not list(tmp_path.rglob("categories.json"))


def test_the_real_builder_reads_an_m2_jobs_directory_as_one_scene(monkeypatch, tmp_path, stack):
    monkeypatch.setattr(tj, "_fly_one_block", fake_block(stack))
    real = tj._fly_one_block

    def with_real_records(payload, on_progress):
        label, flown_rows, summary = real(payload, on_progress)
        return label, [(sc, real_flown(sc, block=True)[0], ev, err) for sc, _rec, ev, err in flown_rows], summary
    monkeypatch.setattr(tj, "_fly_one_block", with_real_records)
    spec = spec_m2(tmp_path)
    only_the_builder_runs_for_real(stack, monkeypatch, tmp_path)
    out = tmp_path / "job"
    tj.run_job(spec, out)

    comparison = out / files.COMPARISON_DIR
    index = read(comparison / files.INDEX_FILE)
    assert {g["group"] for g in index["groups"]} == {FLIGHTS["A"]["flight_key"], FLIGHTS["B"]["flight_key"]}
    assert all(g["scene"]["delayS"] == 3.0 and g["scene"]["outcome"] == "separated_at_baseline" for g in index["groups"])
    assert index["scene"]["background"]["recorded"] == [f"ref-{FLIGHTS['C']['flight_key']}"]
    assert not list(tmp_path.rglob("categories.json"))
