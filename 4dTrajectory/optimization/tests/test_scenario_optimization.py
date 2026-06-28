"""Tests for the scenario-optimization scaffold.

Plumbing (records, the node-state reshape, the target guard, filenames) passes already.
``test_simulate_controls_rolls_forward`` guards the forward rollout (TODO ②, now wired
through ``aerodynamic_model.rollout_piecewise_constant``). The full ``optimize_scenario``
path runs the solver, so it is exercised by the CLI, not the unit suite.
"""

import os
import sys
from pathlib import Path

import pytest

_OPT_DIR = Path(__file__).resolve().parents[1]
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scenario_optimization as so  # noqa: E402
from aerodynamic_model.common import GeodeticState  # noqa: E402
from aircraft.aero_params import aero_params_for_aircraft  # noqa: E402
from aircraft.aircraft_sets import A320  # noqa: E402
from flight_scenarios import FlightScenario  # noqa: E402


def _scenario(*, target: GeodeticState | None) -> FlightScenario:
    initial = GeodeticState(35.6, -78.5, 2000.0, 130.0, 1.5, -0.05, A320.mass.max_takeoff_kg)
    return FlightScenario(
        initial=initial,
        aircraft=A320,
        aero=aero_params_for_aircraft(A320),
        source={"id": "AFR074", "runway": "05L"},
        target=target,
    )


def test_node_states_to_samples_assigns_even_times():
    node_state = [
        [35.0, -78.0, 2000.0, 130.0, 1.0, -0.05],
        [35.1, -78.1, 1500.0, 120.0, 1.0, -0.05],
        [35.2, -78.2, 1000.0, 110.0, 1.0, -0.05],
    ]
    samples = so._node_states_to_samples(node_state, final_time=100.0, mass=78000.0)
    assert [s.t for s in samples] == pytest.approx([0.0, 50.0, 100.0])
    assert samples[0].lat == 35.0 and samples[0].m == 78000.0
    assert samples[-1].alt == 1000.0


def test_scenario_optimization_serialization():
    sample = so.StateSample(t=0.0, lat=35.0, lon=-78.0, alt=2000.0, V=130.0, psi=1.0, gamma=-0.05, m=78000.0)
    result = so.ScenarioOptimization(
        source={"id": "AFR074"},
        final_time_s=120.0,
        optimizer_states=[sample],
        simulator_states=[sample, sample],
    )
    d = result.to_dict()
    assert d["final_time_s"] == 120.0
    assert len(d["optimizer_states"]) == 1 and len(d["simulator_states"]) == 2
    assert d["optimizer_states"][0]["lat"] == 35.0


def test_scenario_filename():
    assert so._scenario_filename(_scenario(target=None), 0) == "AFR074_05L_states.json"


def test_optimize_scenario_requires_target():
    with pytest.raises(ValueError):
        so.optimize_scenario(_scenario(target=None))


def test_optimize_scenarios_skips_failures_and_continues(monkeypatch, tmp_path):
    # A real landings file mixes feasible and infeasible scenarios; one failure must not
    # abort the batch. Stub the per-scenario solve: first raises, second succeeds.
    target = GeodeticState(35.59, -78.49, 500.0, 80.0, 1.5, -0.05, A320.landing_mass)
    scenarios = [_scenario(target=target), _scenario(target=target)]
    attempts = {"n": 0}

    def fake_optimize_scenario(scenario, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("Direct collocation free-time optimization failed: Infeasible_Problem_Detected")
        return so.ScenarioOptimization(scenario.source, 12.0, [], [])

    monkeypatch.setattr(so, "optimize_scenario", fake_optimize_scenario)
    # jobs=1 (serial): the stub + shared counter live in this process, so the solve must
    # run here — spawned workers re-import the module fresh and would not see the
    # monkeypatch. The skip-and-continue orchestration under test is identical on both paths.
    written = so.optimize_scenarios(scenarios, output_dir=tmp_path, jobs=1)
    assert attempts["n"] == 2      # both attempted — did NOT abort on the first failure
    assert len(written) == 1       # the infeasible one is skipped, the feasible one written


def test_resolve_jobs_auto_and_explicit():
    # auto (0) leaves cores free: half the CPUs, capped at the task count, floored at 1
    assert so._resolve_jobs(0, 100) == max(1, (os.cpu_count() or 2) // 2)
    assert so._resolve_jobs(4, 100) == 4          # explicit count honoured
    assert so._resolve_jobs(8, 3) == 3            # never more workers than scenarios
    assert so._resolve_jobs(0, 0) == 1            # empty batch -> serial


def test_optimize_one_scenario_returns_record(monkeypatch):
    # The process-pool worker wraps optimize_scenario into a picklable record and
    # captures failures instead of raising (so one bad scenario never kills the pool).
    target = GeodeticState(35.59, -78.49, 500.0, 80.0, 1.5, -0.05, A320.landing_mass)
    scenario = _scenario(target=target)

    monkeypatch.setattr(so, "optimize_scenario",
                        lambda s, **k: so.ScenarioOptimization(s.source, 12.0, [], []))
    index, flight_id, result_dict, error = so._optimize_one_scenario((3, scenario, {}))
    assert index == 3 and flight_id == "AFR074" and error is None
    assert result_dict["final_time_s"] == 12.0

    def boom(scenario, **kwargs):
        raise ValueError("Infeasible_Problem_Detected")
    monkeypatch.setattr(so, "optimize_scenario", boom)
    index, flight_id, result_dict, error = so._optimize_one_scenario((4, scenario, {}))
    assert index == 4 and result_dict is None
    assert error.startswith("ValueError:")


def test_simulate_controls_rolls_forward():
    initial = GeodeticState(35.6, -78.5, 2000.0, 130.0, 1.5, -0.05, A320.mass.max_takeoff_kg)
    # two constant-control segments over a short 4 s horizon
    node_control = [[40000.0, 0.0, 1.0], [40000.0, 0.0, 1.0]]
    samples = so.simulate_controls(initial, node_control, final_time=4.0, aircraft=A320, dt=1.0)
    assert len(samples) >= 2
    assert samples[0].t == 0.0
    assert samples[0].lat == initial.latitude  # first sample is the initial state
    assert samples[-1].t == pytest.approx(4.0, abs=1.0)
