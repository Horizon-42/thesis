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


#: The recorded traffic's type of a flight, when it is not an A320 (a test sets one with ``monkeypatch.setitem``).
TYPES = {}


def recorded(flight):
    t = T0 + flight["entry_s"] + np.array([0.0, flight["duration_s"]])
    return RecordedFlight(flight["flight_key"], flight["runway"], TYPES.get(flight["flight_key"], "A320"), t, np.full(2, 35.6), np.full(2, -78.5), np.full(2, 900.0))


def scenario_of(flight):
    initial = GeodeticState(35.6, -78.5, 2000.0, 130.0, 1.5, -0.05, A320.mass.max_takeoff_kg)
    src = {k: v for k, v in flight.items() if k not in ("entry_s", "duration_s", "runway_target")}
    return FlightScenario(initial, A320, aero_params_for_aircraft(A320), src, TARGET)


def solve(kind="baseline", ok=True, wall=5.0, cpu=4.5, iaf="ABCD", **extra):
    """One timed solve of a sidecar (``scenario_optimization.SolveTime.to_json``)."""
    return {"kind": kind, "iaf": iaf, "ok": ok, "wallS": wall, "cpuS": cpu, **extra}


def flown(scenario, rounds=None, solves=None, kept_round=0, final_time_s=100.0):
    """A solved record and its sidecar, as the batch workers return them (``rounds``: the sidecar's, default one clean, each
    with the loop's own ``counted_loss_instants`` and the census's ``answered_loss_instants``; ``kept_round``: the round whose solve is the record, MD14, and
    ``final_time_s`` that record's flight time; ``solves``: its timed solves, default one baseline solve)."""
    control = LoadFactorControl(thrust=1.0e5, bank_rad=0.0, load_factor=1.0)
    rollout = [RolloutSample(0.0, scenario.initial, control, 0), RolloutSample(final_time_s, TARGET, control, 0)]
    result = ScenarioOptimization(scenario.source, final_time_s, [], [], evaluation=ee.evaluation_record(
        scenario.initial, TARGET, rollout, scenario.source, subject="optimized"))
    record = result.to_dict()
    record["sidecar"] = {"outcome": "separated_at_baseline", "rounds": rounds or [
        {"final_time_s": 100.0, "counted_loss_instants": 0, "answered_loss_instants": 0, "losses": [], "background_losses": 0}],
        "kept_round": kept_round, "ifr_losses": {"baseline": [], "final": []},
        "starts_in_loss": {}, "uncategorised_types": [], "runways_without_faf": [], "frame_east_error_max": 0.001,
        "solves": solves or [solve()], "windowWallS": 6.0, "windowCpuS": 5.5}
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
    """Every write of progress.json, as the file holds it."""
    real = tj.Progress._write

    def spy(self):
        seen["progress"].append(dict(self._fields))
        real(self)
    monkeypatch.setattr(tj.Progress, "_write", spy)


def done_of(progress):
    """The (done, total, current) of each progress write, one per aircraft settled (the phase writes repeat it)."""
    return [(w["done"], w["total"], w["current"]) for i, w in enumerate(progress)
            if i == 0 or (w["done"], w["total"], w["current"]) != (progress[i - 1]["done"], progress[i - 1]["total"], progress[i - 1]["current"])]


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

    def fly(payload, on_progress, on_phase):
        seen["targets"] = payload[2].runway_targets
        return real(payload, on_progress, on_phase)

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
    summary = tj.run_job(spec_m1(tmp_path), out)["summary"]

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
    assert done_of(stack["progress"]) == [(0, None, None), (0, 1, None), (1, 1, FLIGHTS["A"]["flight_key"])]


def test_a_finished_job_ends_with_state_json_and_the_readout(monkeypatch, tmp_path, stack):
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (payload[0], payload[1].source["flight_key"],
                                                                *flown(payload[1]), None))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_m1(tmp_path)))
    out = tmp_path / "job"

    assert tj.main(["--spec", str(spec_path), "--out", str(out)]) == 0
    state = read(out / files.STATE_FILE)
    assert state["state"] == "done" and state["error"] is None and state["summary"]["windows"] == 1
    assert read(out / files.PROGRESS_FILE) == {"done": 1, "total": 1, "current": FLIGHTS["A"]["flight_key"],
                                                "phase": "building the scene"}               # the last thing the job did
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

class FakeClock:
    """The job's clock (``run_job(clock=...)``): the fakes advance it by what their real counterparts would take."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def loss(t, responsible=True, other="C"):
    """One sidecar loss: ``[t, other, kind, required, distance, vertical, responsible, rowed]``."""
    return [t, other, "in_trail", 5000.0, 4000.0, 0.0, responsible, responsible]


def failed_record(key, reason, solves):
    """What a scenario without a record carries to the batch writers: its failed sidecar (``traffic.loop.failed_sidecar``)."""
    return {"sidecar": {"schema": "optimization-traffic-failed-v1", "flight_key": key, "reason": reason, "solves": solves}}


def fake_block(stack, error=None, *, rounds=None, delays=None, block_check=None, slot_failed=(), eta_failed=(),
               clock=None, eta_s=0.0, solve_s=None, reading="visual", solves=None, kept=None, final_time_s=None):
    """``_fly_one_block`` on fakes, calling ``on_phase`` and ``on_progress`` in ``fly_block``'s order. ``rounds``/``delays``/
    ``block_check``/``solves``: per flight key (the sidecar's rounds, the slot's delay, the block's final-check ``answered``,
    the aircraft's timed solves — its ETA solves first); ``slot_failed``/``eta_failed``: keys whose solve failed (an ETA failure
    has no slot; both leave a failed sidecar with the solves they timed). ``clock``: the job's FakeClock, advanced by ``eta_s``
    for the ETA solves, a second for the schedule and ``solve_s[key]`` for each slot solve (the job's stage timing).
    ``kept``/``final_time_s``: per flight key the sidecar's ``kept_round`` (MD14) and the record's flight time (default round 0, 100 s).
    ``reading``: the reading the block's final check is keyed by (the job's ``LoopSettings.reading``; IFR is always beside it)."""
    rounds, delays, block_check, solve_s, solves = rounds or {}, delays or {}, block_check or {}, solve_s or {}, solves or {}
    kept, final_time_s = kept or {}, final_time_s or {}

    def tick(seconds):
        if clock is not None:
            clock.advance(seconds)

    def fly(payload, on_progress, on_phase):
        label, scenarios, traffic, params = payload
        stack["block"] = {"scenarios": sorted(s.source["flight_key"] for s in scenarios),
                          "traffic": sorted(f.flight_key for f in traffic.flights)}
        none = {"answered": 0, "not_answered": 0, "background": 0}
        keys = [s.source["flight_key"] for s in scenarios]
        done = 0
        on_phase("earliest arrival of each aircraft")
        tick(eta_s)
        for key in keys:
            if key in eta_failed:
                done += 1
                on_progress(done, len(keys), key)
        on_phase("schedule")
        tick(1.0)
        slotted = [k for k in keys if k not in eta_failed]
        for k, key in enumerate(slotted, 1):
            on_phase(f"optimizing {k} of {len(slotted)}")
            tick(solve_s.get(key, 0.0))
            done += 1
            on_progress(done, len(keys), key)
        out = []
        for scenario in scenarios:
            key = scenario.source["flight_key"]
            if key in eta_failed:
                reason = "BaselineFailed: ETA solve: y"
                out.append((scenario, failed_record(key, reason, solves.get(key, [solve("eta", ok=False)])), None, reason))
            elif key in slot_failed:
                reason = "BaselineFailed: slot solve (delay 9.0 s): x"
                out.append((scenario, failed_record(key, reason, solves.get(key, [solve("eta"), solve("slot", ok=False)])),
                            None, reason))
            else:
                record, evaluation = flown(scenario, rounds.get(key), solves.get(key, [solve("eta"), solve("slot")]),
                                           kept.get(key, 0), final_time_s.get(key, 100.0))
                record["sidecar"]["schema"] = "optimization-traffic-block-v3"
                record["sidecar"]["slot"] = {"eta_utc_s": 0.0, "cta_utc_s": 0.0, "delay_s": delays.get(key, 0.0)}
                record["sidecar"]["block_final"] = {"visual": none, "ifr": none,
                                                    reading: {**none, "answered": block_check.get(key, 0)}}
                out.append((scenario, record, evaluation, None))
        slots = [s for s in scenarios if s.source["flight_key"] not in eta_failed]
        summary = {"aircraft": len(scenarios), "scheduled": len(slots), "eta_failed": len(eta_failed), "slot_failed": 0,
                   "schedule_speed_mps": 70.0,
                   "slots": [{"flight_key": s.source["flight_key"], "runway": "05L", "eta_utc_s": 0.0,
                              "cta_utc_s": 0.0, "delay_s": delays.get(s.source["flight_key"], 0.0)} for s in slots],
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
    summary = tj.run_job(spec_m2(tmp_path), out)["summary"]

    a, b, c = (FLIGHTS[k]["flight_key"] for k in "ABC")
    # the block [T0 + 60, T0 + 960) holds A (lands T0 + 200) and B (lands T0 + 500); C landed at T0 + 50, just before
    # it, and is in the air at A's entry; D landed 4800 s before it and E enters 9000 s in, long after B's entry plus
    # the horizon
    assert stack["loaded"] == [sorted([a, b, c]), [FLIGHTS["F"]["flight_key"]]]
    assert stack["block"]["scenarios"] == sorted([a, b])
    assert stack["block"]["traffic"] == sorted([a, b, c])
    assert done_of(stack["progress"]) == [(0, None, None), (0, 2, None), (1, 2, a), (2, 2, b)]

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


# ── what changed for each aircraft (state.json perAircraft), the time it took, and what the job is doing ───────────

def test_an_m1_jobs_per_aircraft_shows_the_census_count_of_the_first_solve_and_of_the_solve_the_loop_kept(
        monkeypatch, tmp_path, stack):
    # MD14: the loop keeps the solve with the FEWEST counted loss instants (MD10 applied: its own count) — here round 1 (1), not
    # the last (3). What the job shows is the census's kind of count, `answered_loss_instants` (MD10 not applied: 4, 3, 5): the
    # first round's (4) and the kept round's (3) — never the loop's counted ones (2, 1), never rounds[-1] (5), never a recount of
    # the `losses` lists. The record IS the kept solve: its flight time is 107.44 s, the recorded flight took 200 s: -92.6.
    # Its solves, as the solver timed them: the baseline, a re-solve that failed, its retry, another re-solve: 4, one failed;
    # 5.0 + 3.0 + 4.0 + 2.0 = 14.0 s of wall time and 4.5 + 2.5 + 3.5 + 1.5 = 12.0 s of CPU
    rounds = [{"final_time_s": 100.0, "counted_loss_instants": 2, "answered_loss_instants": 4, "background_losses": 0,
               "losses": [loss(10.0), loss(10.0, other="D"), loss(11.0), loss(12.0)]},
              {"final_time_s": 107.44, "counted_loss_instants": 1, "answered_loss_instants": 3, "background_losses": 0,
               "losses": [loss(13.0), loss(14.0)]},
              {"final_time_s": 103.0, "counted_loss_instants": 3, "answered_loss_instants": 5, "background_losses": 0,
               "losses": [loss(13.0), loss(14.0)]}]
    timed = [solve("baseline", wall=5.0, cpu=4.5), solve("resolve", ok=False, wall=3.0, cpu=2.5, round=0),
             solve("retry", wall=4.0, cpu=3.5, round=0), solve("resolve", wall=2.0, cpu=1.5, round=1)]
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (
        payload[0], payload[1].source["flight_key"], *flown(payload[1], rounds, timed, kept_round=1, final_time_s=107.44), None))
    aircraft = tj.run_job(spec_m1(tmp_path), tmp_path / "job")["perAircraft"]
    assert aircraft == {FLIGHTS["A"]["flight_key"]: {
        "type": "A320", "firstSolveLosses": 4, "finalLosses": 3, "landingVsRecordS": -92.6, "delayS": None,
        "blockCheckLosses": None, "optimizeS": 14.0, "optimizeCpuS": 12.0, "solves": 4, "failedSolves": 1}}


def test_an_aircraft_that_needed_no_re_solve_has_one_solve_and_none_failed(monkeypatch, tmp_path, stack):
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (payload[0], payload[1].source["flight_key"],
                                                                *flown(payload[1]), None))
    aircraft = tj.run_job(spec_m1(tmp_path), tmp_path / "job")["perAircraft"][FLIGHTS["A"]["flight_key"]]
    assert (aircraft["solves"], aircraft["failedSolves"], aircraft["optimizeS"], aircraft["optimizeCpuS"]) == (1, 0, 5.0, 4.5)


def test_an_m1_aircraft_without_a_record_has_its_solves_from_its_failed_sidecar(monkeypatch, tmp_path, stack):
    key = FLIGHTS["A"]["flight_key"]
    reason = "BaselineFailed: all 2 IAF(s) infeasible"
    attempts = [solve("baseline", ok=False, wall=7.0, cpu=6.5, iaf="AAA"), solve("baseline", ok=False, wall=2.5, cpu=2.0, iaf="BBB")]
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (payload[0], key, failed_record(key, reason, attempts), None, reason))
    aircraft = tj.run_job(spec_m1(tmp_path), tmp_path / "job")["perAircraft"]
    assert aircraft == {key: {
        "type": "A320", "firstSolveLosses": None, "finalLosses": None, "landingVsRecordS": None, "delayS": None,
        "blockCheckLosses": None, "optimizeS": 9.5, "optimizeCpuS": 8.5, "solves": 2, "failedSolves": 2}}


def test_an_aircraft_whose_failed_sidecar_lost_its_solves_has_no_timing_never_zero_seconds(monkeypatch, tmp_path, stack):
    # a failure outside the solve (a casadi RuntimeError in a re-solve): the loop's failed sidecar carries `solves: null`
    key = FLIGHTS["A"]["flight_key"]
    reason = "RuntimeError: casadi blew up"
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (payload[0], key, failed_record(key, reason, None), None, reason))
    aircraft = tj.run_job(spec_m1(tmp_path), tmp_path / "job")["perAircraft"]
    assert aircraft == {key: {
        "type": "A320", "firstSolveLosses": None, "finalLosses": None, "landingVsRecordS": None, "delayS": None,
        "blockCheckLosses": None, "optimizeS": None, "optimizeCpuS": None, "solves": None, "failedSolves": None}}


def test_an_m2_aircraft_of_a_block_that_raised_has_its_slot_but_no_timing(monkeypatch, tmp_path, stack):
    a, b = (FLIGHTS[k]["flight_key"] for k in "AB")
    base = fake_block(stack, delays={a: 4.0})

    def fly(payload, on_progress, on_phase):
        label, out, summary = base(payload, on_progress, on_phase)
        reason = "block failed: RuntimeError: x"
        # B's solves were lost with the block; A flew
        return label, [(sc, failed_record(b, reason, None), None, reason) if sc.source["flight_key"] == b else (sc, rec, ev, err)
                       for sc, rec, ev, err in out], summary
    monkeypatch.setattr(tj, "_fly_one_block", fly)
    aircraft = tj.run_job(spec_m2(tmp_path), tmp_path / "job")["perAircraft"]
    assert (aircraft[b]["optimizeS"], aircraft[b]["optimizeCpuS"], aircraft[b]["solves"], aircraft[b]["failedSolves"]) == (None,) * 4
    assert aircraft[a]["solves"] == 2 and aircraft[a]["optimizeS"] == 10.0                 # the other's own, from its own sidecar


def test_an_m2_jobs_per_aircraft_adds_the_slots_delay_and_the_block_check_and_lists_the_aircraft_not_flown(
        monkeypatch, tmp_path, stack):
    a, b = (FLIGHTS[k]["flight_key"] for k in "AB")
    # A flies: its ETA solve (6 s), its slot solve and one re-solve, three rounds of 3, 1 and 2 counted loss instants (5, 2 and 4 in the
    # census's count), the loop keeping the second (MD14; its record flies 230 s); B's slot solve fails after its ETA solve: it has a slot
    # (a delay the schedule gave it) and no record, only a failed sidecar with the solves it timed
    monkeypatch.setattr(tj, "_fly_one_block", fake_block(
        stack,
        rounds={a: [{"final_time_s": 240.0, "counted_loss_instants": 3, "answered_loss_instants": 5, "losses": [loss(5.0)],
                     "background_losses": 0},
                    {"final_time_s": 230.0, "counted_loss_instants": 1, "answered_loss_instants": 2, "losses": [loss(5.0)],
                     "background_losses": 0},
                    {"final_time_s": 235.0, "counted_loss_instants": 2, "answered_loss_instants": 4, "losses": [loss(5.0)],
                     "background_losses": 0}]},
        kept={a: 1}, final_time_s={a: 230.0},
        solves={a: [solve("eta", wall=6.0, cpu=5.5), solve("slot", wall=20.0, cpu=19.0), solve("resolve", wall=4.0, cpu=3.5)],
                b: [solve("eta", wall=8.0, cpu=7.0), solve("slot", ok=False, wall=1.5, cpu=1.0)]},
        delays={a: 30.0, b: 12.5}, block_check={a: 4}, slot_failed={b}))
    aircraft = tj.run_job(spec_m2(tmp_path), tmp_path / "job")["perAircraft"]
    assert aircraft == {
        a: {"type": "A320", "firstSolveLosses": 5, "finalLosses": 2, "landingVsRecordS": 30.0,       # 230 s against the record's 200 s
            "delayS": 30.0, "blockCheckLosses": 4, "optimizeS": 30.0, "optimizeCpuS": 28.0, "solves": 3, "failedSolves": 0},
        b: {"type": "A320", "firstSolveLosses": None, "finalLosses": None, "landingVsRecordS": None,
            "delayS": 12.5, "blockCheckLosses": None, "optimizeS": 9.5, "optimizeCpuS": 8.0, "solves": 2, "failedSolves": 1}}


def test_an_aircraft_whose_eta_solve_failed_has_no_slot_no_delay_but_its_solves_all_the_same(monkeypatch, tmp_path, stack):
    a, b = (FLIGHTS[k]["flight_key"] for k in "AB")
    monkeypatch.setattr(tj, "_fly_one_block", fake_block(
        stack, eta_failed={a}, delays={b: 3.0},
        solves={a: [solve("eta", ok=False, wall=11.0, cpu=10.0, iaf="X"), solve("eta", ok=False, wall=9.0, cpu=8.5, iaf="Y")]}))
    aircraft = tj.run_job(spec_m2(tmp_path), tmp_path / "job")["perAircraft"]
    assert aircraft[a] == {"type": "A320", "firstSolveLosses": None, "finalLosses": None, "landingVsRecordS": None,
                           "delayS": None, "blockCheckLosses": None, "optimizeS": 20.0, "optimizeCpuS": 18.5, "solves": 2,
                           "failedSolves": 2}
    assert aircraft[b]["delayS"] == 3.0 and aircraft[b]["solves"] == 2                      # B's own: its ETA and its slot solve


def test_an_aircraft_whose_cta_was_beyond_the_horizon_has_a_failed_sidecar_with_the_solves_it_took_before(
        monkeypatch, tmp_path, stack):
    a = FLIGHTS["A"]["flight_key"]
    monkeypatch.setattr(tj, "_fly_one_block", fake_block(stack, slot_failed={a}, solves={a: [solve("eta", wall=4.0, cpu=3.5)]}))
    entry = tj.run_job(spec_m2(tmp_path), tmp_path / "job")["perAircraft"][a]
    assert (entry["optimizeS"], entry["solves"], entry["failedSolves"]) == (4.0, 1, 0)       # no slot solve was tried


def test_an_m2_job_names_the_arrivals_of_its_block_it_could_not_control_and_an_m1_job_has_none(monkeypatch, tmp_path, stack):
    from flight_scenarios.scenario import NoAircraftDynamics
    a, b = (FLIGHTS[k]["flight_key"] for k in "AB")

    def build(flight, airport, target_from_threshold):
        if flight["flight_key"] == b:
            raise NoAircraftDynamics("C172", "nothing models it")
        return scenario_of(flight)
    monkeypatch.setattr(to, "build_scenario", build)
    monkeypatch.setitem(TYPES, b, "C172")
    monkeypatch.setattr(tj, "_fly_one_block", fake_block(stack))
    result = tj.run_job(spec_m2(tmp_path), tmp_path / "job")
    # B lands in the block but has no dynamics model: it flew its record, named by the roster's callsign and the traffic's type
    assert result["stayedRecords"] == {b: {"callsign": CALLSIGNS["B"], "type": "C172"}}
    assert set(result["perAircraft"]) == {a} and stack["block"]["scenarios"] == [a]
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (payload[0], payload[1].source["flight_key"],
                                                                *flown(payload[1]), None))
    assert tj.run_job(spec_m1(tmp_path), tmp_path / "job1")["stayedRecords"] == {}


def test_every_aircraft_of_a_job_has_its_timing_flown_or_not(monkeypatch, tmp_path, stack):
    a, b = (FLIGHTS[k]["flight_key"] for k in "AB")
    monkeypatch.setattr(tj, "_fly_one_block", fake_block(stack, slot_failed={b}))
    aircraft = tj.run_job(spec_m2(tmp_path), tmp_path / "job")["perAircraft"]
    assert set(aircraft) == {a, b}
    for entry in aircraft.values():
        assert all(isinstance(entry[field], (int, float)) for field in ("optimizeS", "optimizeCpuS", "solves", "failedSolves"))


def test_the_block_check_is_read_under_the_reading_the_jobs_loop_judges_by_not_a_literal(monkeypatch, tmp_path, stack):
    a = FLIGHTS["A"]["flight_key"]
    real = tj.LoopSettings
    monkeypatch.setattr(tj, "LoopSettings", lambda: real(reading="ifr"))
    # the visual check says 0 for A, the ifr one 7: the job reads the one it judges by
    monkeypatch.setattr(tj, "_fly_one_block", fake_block(stack, reading="ifr", block_check={a: 7}))
    aircraft = tj.run_job(spec_m2(tmp_path), tmp_path / "job")["perAircraft"]
    assert aircraft[a]["blockCheckLosses"] == 7


def test_state_json_of_a_finished_job_carries_its_summary_per_aircraft_timing_and_stayed_records(monkeypatch, tmp_path, stack):
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (payload[0], payload[1].source["flight_key"],
                                                                *flown(payload[1]), None))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_m1(tmp_path)))
    out = tmp_path / "job"
    assert tj.main(["--spec", str(spec_path), "--out", str(out)]) == 0
    state = read(out / files.STATE_FILE)
    assert set(state) == {"state", "error", "summary", "perAircraft", "timing", "stayedRecords"}
    assert set(state["timing"]["phases"]) == {"reading traffic", "optimizing", "evaluation", "building the scene"}
    assert state["timing"]["totalS"] >= sum(state["timing"]["phases"].values()) - 0.5       # the books balance (rounded)


# ── how long it took ─────────────────────────────────────────────────────────

def clocked(stack, monkeypatch, clock, *, load_s, evaluation_s, scene_s):
    """The loaders and the two steps of the job take what the arguments say (on the job's FakeClock)."""
    real_load, real_step = tj.load_model_arrivals_subset, tj.run_step

    def load(manifest_path, keys):
        clock.advance(load_s)
        return real_load(manifest_path, keys)

    def step(command):
        clock.advance(evaluation_s if command[2] == "evaluation" else scene_s)
        real_step(command)
    monkeypatch.setattr(tj, "load_model_arrivals_subset", load)
    monkeypatch.setattr(tj, "run_step", step)


def test_an_m1_job_measures_the_wall_time_of_each_stage_and_in_total(monkeypatch, tmp_path, stack):
    clock = FakeClock()
    clocked(stack, monkeypatch, clock, load_s=20.0, evaluation_s=20.0, scene_s=3.0)       # two loads: the near set, a runway target

    def fly(payload):
        clock.advance(12.34)
        return payload[0], payload[1].source["flight_key"], *flown(payload[1]), None
    monkeypatch.setattr(tj, "_fly_one_window", fly)
    timing = tj.run_job(spec_m1(tmp_path), tmp_path / "job", clock=clock)["timing"]
    assert timing == {"totalS": 75.3, "phases": {"reading traffic": 40.0, "optimizing": 12.3, "evaluation": 20.0,
                                                 "building the scene": 3.0}}
    assert list(timing["phases"]) == ["reading traffic", "optimizing", "evaluation", "building the scene"]    # in the order it happened


def test_an_m2_job_splits_optimizing_from_the_earliest_arrivals_and_the_schedule(monkeypatch, tmp_path, stack):
    a, b = (FLIGHTS[k]["flight_key"] for k in "AB")
    clock = FakeClock()
    clocked(stack, monkeypatch, clock, load_s=20.0, evaluation_s=20.0, scene_s=3.0)
    monkeypatch.setattr(tj, "_fly_one_block", fake_block(stack, clock=clock, eta_s=30.0, solve_s={a: 40.0, b: 25.0}))
    result = tj.run_job(spec_m2(tmp_path), tmp_path / "job", clock=clock)
    # "optimizing 1 of 2" and "optimizing 2 of 2" are one stage; the schedule takes the second the fake gives it
    assert result["timing"] == {"totalS": 159.0, "phases": {
        "reading traffic": 40.0, "earliest arrivals": 30.0, "schedule": 1.0, "optimizing": 65.0, "evaluation": 20.0,
        "building the scene": 3.0}}


def test_a_phase_names_its_stage_and_an_unknown_phase_is_an_error():
    assert files.phase_stage("optimizing 3 of 9") == "optimizing"
    assert files.phase_stage(files.PHASE_ETA) == "earliest arrivals"
    assert [files.phase_stage(p) for p in (files.PHASE_READING, files.PHASE_SCHEDULE, files.PHASE_EVALUATION,
                                            files.PHASE_SCENE)] == ["reading traffic", "schedule", "evaluation", "building the scene"]
    with pytest.raises(KeyError):
        files.phase_stage("thinking")


def test_the_phase_follows_the_job_from_reading_the_traffic_to_building_the_scene(monkeypatch, tmp_path, stack):
    progress_log(monkeypatch, stack)
    monkeypatch.setattr(tj, "_fly_one_window", lambda payload: (payload[0], payload[1].source["flight_key"],
                                                                *flown(payload[1]), None))
    tj.run_job(spec_m1(tmp_path), tmp_path / "job")
    phases = [w["phase"] for w in stack["progress"]]
    assert list(dict.fromkeys(phases)) == ["reading traffic", "optimizing 1 of 1", "evaluation", "building the scene"]
    assert phases[0] == "reading traffic" and stack["progress"][0]["total"] is None       # written before anything is read


def test_an_m2_jobs_phases_are_the_blocks_between_reading_and_evaluation(monkeypatch, tmp_path, stack):
    progress_log(monkeypatch, stack)
    monkeypatch.setattr(tj, "_fly_one_block", fake_block(stack))
    tj.run_job(spec_m2(tmp_path), tmp_path / "job")
    assert list(dict.fromkeys(w["phase"] for w in stack["progress"])) == [
        "reading traffic", "earliest arrival of each aircraft", "schedule", "optimizing 1 of 2", "optimizing 2 of 2", "evaluation",
        "building the scene"]


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
        none = {"answered": 0, "not_answered": 0, "background": 0}
        record["sidecar"]["block_final"] = {"visual": none, "ifr": none}
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

    def with_real_records(payload, on_progress, on_phase):
        label, flown_rows, summary = real(payload, on_progress, on_phase)
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
