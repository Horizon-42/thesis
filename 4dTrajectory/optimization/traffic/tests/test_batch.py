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
                     mode="traffic:m1", progress="", jobs=1, scenarios_label=None, references_dir=None,
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
