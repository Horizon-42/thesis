"""The traffic batch's record plumbing: the sidecar beside each solved record, resume, the sweep, the readout."""

import json

from aerodynamic_model.common import GeodeticState, LoadFactorControl
from aerodynamic_model.rollout import RolloutSample
from aircraft.aero_params import aero_params_for_aircraft
from aircraft.aircraft_sets import A320

import evaluation_export as ee
from flight_scenarios import FlightScenario
from scenario_batch import run_batch
from scenario_replay import ScenarioOptimization
from traffic import M1_MODE
from traffic.readout import readout

SUFFIX = "_traffic.json"
TARGET = GeodeticState(35.59, -78.49, 500.0, 80.0, 1.5, -0.05, A320.landing_mass)
SOLVES = []


def _scenario(flight_id):
    initial = GeodeticState(35.6, -78.5, 2000.0, 130.0, 1.5, -0.05, A320.mass.max_takeoff_kg)
    return FlightScenario(initial, A320, aero_params_for_aircraft(A320), {
        "id": flight_id, "icao24": "ad7f04", "landing_time_utc": "2026-06-18T21:37:36Z", "arr_airport": "KRDU",
        "runway": "05L", "target_source": "runway_threshold", "window_s": 15.0}, TARGET)


def _worker(payload):
    index, scenario, _params = payload
    SOLVES.append(scenario.source["id"])
    control = LoadFactorControl(thrust=1.0e5, bank_rad=0.0, load_factor=1.0)
    rollout = [RolloutSample(0.0, scenario.initial, control, 0), RolloutSample(10.0, TARGET, control, 0)]
    result = ScenarioOptimization(scenario.source, 10.0, [], [],
                                  evaluation=ee.evaluation_record(scenario.initial, TARGET, rollout, scenario.source,
                                                                  subject="optimized"))
    record = result.to_dict()
    record["sidecar"] = {"outcome": "separated", "rounds": [
        {"final_time_s": 100.0, "losses": [[1.0, "X", "in_trail", 5556.0, 4000.0, 0.0, True, True]],
         "background_losses": 1},
        {"final_time_s": 112.5, "losses": [], "background_losses": 0}],
        "ifr_losses": {"baseline": [[1.0, "X", "in_trail", 5556.0, 4000.0, 0.0, True, False]], "final": []},
        "starts_in_loss": {}, "uncategorised_types": ["ZZZZ"], "runways_without_faf": ["32"],
        "frame_east_error_max": 0.003}
    evaluation = dict(result.evaluation)
    evaluation["states"] = []
    return (index, scenario.source["id"], record, evaluation, None)


def _run(tmp_path, scenarios, resume):
    return run_batch(scenarios, output_dir=tmp_path, worker=_worker, params={}, optimization_config={"m": 1},
                     mode=M1_MODE, progress="", jobs=1, scenarios_label=None, references_dir=None,
                     resume=resume, sidecar_suffix=SUFFIX)


def test_the_sidecar_is_written_beside_the_record_and_resume_needs_it(tmp_path):
    SOLVES.clear()
    _run(tmp_path, [_scenario("AFR074")], resume=False)
    states = next(tmp_path.glob("*_states.json"))
    sidecar = tmp_path / states.name.replace("_states.json", SUFFIX)
    assert "sidecar" not in json.loads(states.read_text())
    assert json.loads(sidecar.read_text())["outcome"] == "separated"
    _run(tmp_path, [_scenario("AFR074")], resume=True)
    assert SOLVES == ["AFR074"]                                  # complete with its sidecar: resumed
    sidecar.unlink()
    _run(tmp_path, [_scenario("AFR074")], resume=True)
    assert SOLVES == ["AFR074", "AFR074"]                        # its sidecar gone: solved again


def test_the_sweep_removes_an_orphan_sidecar_and_the_readout_counts_the_roster(tmp_path):
    (tmp_path / f"OLD_05L{SUFFIX}").write_text("{}")
    _run(tmp_path, [_scenario("AFR074"), _scenario("BAW1")], resume=False)
    assert not (tmp_path / f"OLD_05L{SUFFIX}").exists()
    found = readout(tmp_path)
    assert found["windows"] == 2 and found["outcomes"] == {"separated": 2}
    assert found["re_solves"] == {1: 2}
    assert found["landing_time_change_s"]["median"] == 12.5
    assert found["windows_with_a_counted_visual_loss"] == {"baseline": 2, "final": 0}
    assert found["uncategorised_types"] == ["ZZZZ"] and found["background_losses_at_baseline"] == 2
    assert found["windows_with_an_answered_ifr_loss"] == {"baseline": 2, "final": 0}
    assert found["runways_without_a_coded_final"] == ["32"]


def test_the_m2_readout_counts_blocks(tmp_path):
    import json
    from traffic.readout import blocks_readout
    from traffic import M2_MODE
    none = {"answered": 0, "not_answered": 0, "background": 0}
    (tmp_path / "summary.json").write_text(json.dumps({"mode": M2_MODE, "blocks": [{
        "label": "block_00", "aircraft": 4, "scheduled": 3,
        "slots": [{"flight_key": "A", "delay_s": 0.0}, {"flight_key": "B", "delay_s": 80.0},
                  {"flight_key": "D", "delay_s": 0.0}],
        "outcomes": {"A": "separated_at_baseline", "B": "separated", "D": "solve_failed",
                     "C": "BaselineFailed: ETA solve: x"},
        "final_losses": {"A": {"visual": none, "ifr": dict(none, answered=3)},
                         "B": {"visual": dict(none, not_answered=1), "ifr": none}}},
        {"label": "block_01", "aircraft": 2, "scheduled": 0, "eta_failed": 0, "slot_failed": 0, "error": "x"}]}))
    found = blocks_readout(tmp_path)
    assert found["aircraft"] == 4 and found["scheduled"] == 3 and found["failed_blocks"] == ["block_01"]
    assert found["delay_s"] == {"median": 0.0, "max": 80.0, "delayed_over_60s": 1}
    assert found["outcomes"] == {"separated_at_baseline": 1, "separated": 1, "solve_failed_undelayed": 1,
                                 "eta_failed": 1}
    assert found["flown_aircraft_with_a_loss_left_after_the_block"] == {
        "visual": {"answered": 0, "not_answered": 1}, "ifr": {"answered": 1, "not_answered": 0}}


def _m2_args(tmp_path):
    from types import SimpleNamespace
    return SimpleNamespace(block_start="2026-06-18T21:00:00Z", block_s=3600.0, blocks=2, max_duration=2000.0,
                           airport="KRDU", procedure_root="root", rollout_dt=0.5, max_iterations=10,
                           output_dir=str(tmp_path), jobs=2)


def _m2_run(monkeypatch, tmp_path, outputs):
    """Drive ``_run_blocks`` with two blocks whose flown aircraft are ``outputs[label]`` (a fake block worker,
    threads instead of processes)."""
    from concurrent.futures import ThreadPoolExecutor
    import traffic_optimization as to
    from traffic.loop import LoopSettings
    scenarios = {"block_00": [_scenario("AAA1")], "block_01": [_scenario("BBB2"), _scenario("CCC3")]}
    first = to.parse_iso_utc_s("2026-06-18T21:00:00Z")
    monkeypatch.setattr(to, "load_model_arrivals", lambda manifest: [])
    monkeypatch.setattr(to, "traffic_from_arrivals", lambda flights: None)
    monkeypatch.setattr(to, "block_scenarios", lambda flights, traffic, airport, start, end, horizon: (
        scenarios[f"block_{int((start - first) // 3600):02d}"], None, 1))
    monkeypatch.setattr(to, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(to, "limit_solver_threads", lambda: None)
    monkeypatch.setattr(to, "_fly_one_block", lambda payload: outputs(payload[0], payload[1]))
    to._run_blocks(_m2_args(tmp_path), LoopSettings(), tmp_path / "manifest.json", {"m": 2})
    return json.loads((tmp_path / "summary.json").read_text())


def test_an_m2_run_writes_one_roster_with_every_block_and_a_failed_block(monkeypatch, tmp_path):
    from traffic import M2_MODE
    import threading
    (tmp_path / "OLD_05L_ad7f04_20260618T213736Z_eval.json").write_text("{}")      # a previous run's record
    block_01_done = threading.Event()

    def outputs(label, scenarios):
        if label == "block_01":                       # the block that raised: every aircraft a failed record
            block_01_done.set()
            return label, [(s, None, None, "block failed: RuntimeError: x") for s in scenarios], {
                "aircraft": 2, "scheduled": 0, "eta_failed": 0, "slot_failed": 0, "error": "RuntimeError: x"}
        assert block_01_done.wait(10.0)              # block_00 finishes last: the roster is sorted, not arrival order
        _i, _key, record, evaluation, _e = _worker((0, scenarios[0], {}))
        return label, [(scenarios[0], record, evaluation, None)], {
            "aircraft": 1, "scheduled": 1, "eta_failed": 0, "slot_failed": 0, "slots": [], "outcomes": {},
            "final_losses": {}}

    summary = _m2_run(monkeypatch, tmp_path, outputs)
    assert summary["mode"] == M2_MODE and (summary["total"], summary["solved"], summary["failed"]) == (3, 1, 2)
    assert [r["id"] for r in summary["results"]] == ["AAA1", "BBB2", "CCC3"]          # by block, then order
    assert [b["label"] for b in summary["blocks"]] == ["block_00", "block_01"]
    assert summary["blocks"][1]["error"] == "RuntimeError: x" and summary["blocks"][0]["skipped_no_dynamics"] == 1
    assert not (tmp_path / "OLD_05L_ad7f04_20260618T213736Z_eval.json").exists()      # swept
    files = {p.name for p in tmp_path.iterdir()}
    assert {r["eval_file"] for r in summary["results"]} <= files and len(list(tmp_path.glob("*" + SUFFIX))) == 1


def test_a_failing_block_gives_each_aircraft_a_failed_record(monkeypatch):
    import traffic_optimization as to

    def boom(*_a, **_k):
        raise RuntimeError("casadi")
    monkeypatch.setattr(to, "fly_block", boom)
    label, flown, summary = to._fly_one_block(("block_03", [_scenario("AAA1"), _scenario("BBB2")], None, {
        "procedure_root": "r", "settings": {}, "max_duration": 1.0, "rollout_dt_s": 0.5, "solve_options": {}}))
    assert [f[3] for f in flown] == ["block failed: RuntimeError: casadi"] * 2 and summary["error"] == "RuntimeError: casadi"


def test_an_m2_run_refuses_a_directory_of_the_per_block_layout_before_reading_the_roster(monkeypatch, tmp_path):
    import pytest
    import traffic_optimization as to
    (tmp_path / "block_00").mkdir()

    def no_read(manifest):
        raise AssertionError("the roster was read before the refusal")
    monkeypatch.setattr(to, "load_model_arrivals", no_read)
    with pytest.raises(SystemExit, match="per-block layout"):
        to._run_blocks(_m2_args(tmp_path), None, tmp_path / "manifest.json", {})


def test_the_readout_refuses_a_summary_of_another_mode(tmp_path):
    import pytest
    from traffic.readout import blocks_readout
    (tmp_path / "summary.json").write_text(json.dumps({"mode": M1_MODE}))
    with pytest.raises(ValueError, match="not an M2 run"):
        blocks_readout(tmp_path)
