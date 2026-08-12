"""Tests for the scenario-optimization scaffold.

Plumbing (records, the node-state reshape, the target guard, filenames) passes already.
``test_simulate_controls_rolls_forward`` guards the forward rollout (TODO ②, now wired
through ``aerodynamic_model.rollout_piecewise_constant``). The full ``optimize_scenario``
path runs the solver, so it is exercised by the CLI, not the unit suite.
"""

import json
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

import evaluation_export as ee  # noqa: E402
import scenario_optimization as so  # noqa: E402
from aerodynamic_model.common import GeodeticState, LoadFactorControl  # noqa: E402
from aerodynamic_model.rollout import RolloutSample  # noqa: E402
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


def _fake_optimizer(dense_states_geo, final_time, controls, *, on_init=None, segment_durations_s=None):
    """A minimal CollocationOptimizer stand-in for seam tests — the one shared shape
    (``monkeypatch.setattr(so, "CollocationOptimizer", _fake_optimizer(...))``), so an
    interface change to the real optimizer is chased in one place."""

    class FakeOptimizer:
        def __init__(self, *args, **kwargs):
            if on_init is not None:
                on_init(*args, **kwargs)
            self.last_dense_states_geo = dense_states_geo
            if segment_durations_s is not None:
                self.segment_durations_s = segment_durations_s

        def optimize_free_time(self, initial, tgt, max_duration):
            return final_time, controls, None

    return FakeOptimizer


def _rollout_samples(initial: GeodeticState) -> list[RolloutSample]:
    """Three hand-built rollout samples spanning two control segments."""
    c0 = LoadFactorControl(thrust=1.0e5, bank_rad=0.1, load_factor=1.02)
    c1 = LoadFactorControl(thrust=8.0e4, bank_rad=-0.1, load_factor=0.98)
    mid = GeodeticState(35.7, -78.6, 1500.0, 120.0, 1.4, -0.05, initial.m)
    end = GeodeticState(35.8, -78.7, 1000.0, 110.0, 1.3, -0.05, initial.m)
    return [
        RolloutSample(0.0, initial, c0, 0),
        RolloutSample(5.0, mid, c0, 0),
        RolloutSample(10.0, end, c1, 1),
    ]


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
    # No icao24 / landing time in source -> falls back to the plain id_runway base.
    assert so._scenario_filename(_scenario(target=None), 0) == "AFR074_05L_states.json"


def test_scenario_filename_shares_the_ts_transformer_identity():
    # The stem is single-sourced in flight_scenarios.identity.flight_key — the same
    # function keys ts_transformer's split and record stems. If this seam breaks, learned
    # and optimized records for one flight stop sharing a filename stem.
    from flight_scenarios import flight_key

    src = {"id": "EJA969", "runway": "05R", "icao24": "ad7f04",
           "landing_time_utc": "2026-06-18T21:37:36Z"}
    assert flight_key(src, 0) == "EJA969_05R_ad7f04_20260618T213736Z"


def test_scenario_filename_disambiguates_by_icao24_and_time():
    # Same callsign + runway + aircraft, different landing -> distinct, collision-free names
    # (the old id_runway name silently overwrote one of these).
    def scn(icao24, landing):
        return FlightScenario(
            initial=GeodeticState(35.6, -78.5, 2000.0, 130.0, 1.5, -0.05, A320.mass.max_takeoff_kg),
            aircraft=A320, aero=aero_params_for_aircraft(A320),
            source={"id": "EJA969", "runway": "05R", "icao24": icao24, "landing_time_utc": landing},
            target=None,
        )
    a = so._scenario_filename(scn("ad7f04", "2026-06-18T21:37:36Z"), 0)
    b = so._scenario_filename(scn("ad7f04", "2026-06-23T18:45:21Z"), 1)
    assert a == "EJA969_05R_ad7f04_20260618T213736Z_states.json"
    assert a != b


def test_optimize_scenario_requires_target():
    with pytest.raises(ValueError):
        so.optimize_scenario(_scenario(target=None))


def test_optimize_scenarios_skips_failures_and_continues(monkeypatch, tmp_path):
    # A real landings file mixes feasible and infeasible scenarios; one failure must not
    # abort the batch. Stub the per-scenario solve: first raises, second succeeds.
    target = GeodeticState(35.59, -78.49, 500.0, 80.0, 1.5, -0.05, A320.landing_mass)
    # Distinct identities so the two scenarios write distinct files (as real batches do).
    scenarios = [_scenario(target=target), _scenario(target=target)]
    scenarios[0].source["id"] = "BAD001"
    attempts = {"n": 0}

    def fake_optimize_scenario(scenario, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("Direct collocation free-time optimization failed: Infeasible_Problem_Detected")
        return so.ScenarioOptimization(
            scenario.source, 12.0, [], [],
                evaluation=ee.evaluation_record(
                    scenario.initial, scenario.target, _rollout_samples(scenario.initial),
                    scenario.source, subject="optimized",
            ),
        )

    monkeypatch.setattr(so, "optimize_scenario", fake_optimize_scenario)
    # jobs=1 (serial): the stub + shared counter live in this process, so the solve must
    # run here — spawned workers re-import the module fresh and would not see the
    # monkeypatch. The skip-and-continue orchestration under test is identical on both paths.
    written = so.optimize_scenarios(
        scenarios,
        output_dir=tmp_path,
        jobs=1,
        n_segments=12,
        fitting="rk4",
        state_substeps=5,
        rollout_dt_s=0.25,
    )
    assert attempts["n"] == 2      # both attempted — did NOT abort on the first failure
    assert len(written) == 1       # the infeasible one is skipped, the feasible one written

    # BOTH scenarios got an evaluation record: the solved one points at the simulator
    # states in its canonical states file; the failed one has no state reference.
    eval_files = sorted(tmp_path.glob("*_eval.json"))
    assert len(eval_files) == 2
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in eval_files]
    solved = [p for p in payloads if p.get("states_ref")]
    failed = [p for p in payloads if not p.get("states_ref")]
    assert len(solved) == 1 and len(failed) == 1
    assert solved[0]["states"] == []
    assert solved[0]["states_ref"]["key"] == "simulator_states"
    assert failed[0]["controls"] == [] and failed[0]["final_time_s"] is None
    assert failed[0]["reason"].startswith("ValueError")
    assert failed[0]["target_state"]["lat"] == pytest.approx(35.59)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert all(row["eval_file"].endswith("_eval.json") for row in summary["results"])
    assert summary["optimization_config"] == {
        "mode": "unconstrained",
        "fitting": "rk4",
        "transcription_scheme": "rk4NormalizedFullTransport",
        "control_mesh": {"segments": 12},
        "state_substeps": 5,
        "max_duration_s": so.DEFAULT_MAX_DURATION_S,
        "rollout_dt_s": 0.25,
    }


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

    monkeypatch.setattr(
        so, "optimize_scenario",
        lambda s, **k: so.ScenarioOptimization(s.source, 12.0, [], [],
                                               evaluation={"states": [], "controls": []}),
    )
    index, flight_id, result_dict, eval_dict, error = so._optimize_one_scenario((3, scenario, {}))
    assert index == 3 and flight_id == "AFR074" and error is None
    assert result_dict["final_time_s"] == 12.0
    assert eval_dict == {"states": [], "controls": []}   # the evaluation record rides along

    def boom(scenario, **kwargs):
        raise ValueError("Infeasible_Problem_Detected")
    monkeypatch.setattr(so, "optimize_scenario", boom)
    index, flight_id, result_dict, eval_dict, error = so._optimize_one_scenario((4, scenario, {}))
    assert index == 4 and result_dict is None and eval_dict is None
    assert error.startswith("ValueError:")


def test_evaluation_record_aligns_controls_with_states():
    # The eval record maps the rollout samples 1:1: controls[i] is the ZOH control
    # active at states[i].t — no re-derivation of the control schedule.
    initial = GeodeticState(35.6, -78.5, 2000.0, 130.0, 1.5, -0.05, 60000.0)
    target = GeodeticState(35.9, -78.8, 130.0, 70.0, 0.8, -0.05, 60000.0)
    record = ee.evaluation_record(
        initial, target, _rollout_samples(initial), {"id": "AFR074"},
        subject="optimized",
    )
    assert len(record["states"]) == len(record["controls"]) == 3
    assert record["final_time_s"] == 10.0
    assert record["states"][0] == {"t": 0.0, "lat": 35.6, "lon": -78.5, "alt": 2000.0,
                                   "V": 130.0, "psi": 1.5, "gamma": -0.05, "m": 60000.0}
    # ZOH: the mid sample still flies segment 0's control; the last flies segment 1's.
    assert record["controls"][1]["thrust"] == pytest.approx(1.0e5)
    assert record["controls"][2]["thrust"] == pytest.approx(8.0e4)
    assert record["target_state"]["alt"] == 130.0
    assert record["source"] == {"id": "AFR074", "subject": "optimized"}


def test_failed_evaluation_record_keeps_boundary_conditions_with_empty_lists():
    initial = GeodeticState(35.6, -78.5, 2000.0, 130.0, 1.5, -0.05, 60000.0)
    record = ee.failed_evaluation_record(
        initial, None, {"id": "BAD"}, "ValueError: infeasible",
        subject="optimized",
    )
    assert record["states"] == [] and record["controls"] == []
    assert record["final_time_s"] is None and record["target_state"] is None
    assert record["initial_state"]["lat"] == 35.6
    assert record["reason"] == "ValueError: infeasible"
    assert record["source"]["subject"] == "optimized"


def test_eval_filename_mirrors_states_filename():
    assert so._eval_filename("EJA969_05R_ad7f04_20260618T213736Z_states.json") == \
        "EJA969_05R_ad7f04_20260618T213736Z_eval.json"
    assert so._reference_filename("EJA969_05R_states.json") == "EJA969_05R_reference_eval.json"


def test_write_reference_records_from_observed_tracks(tmp_path):
    # A due-north 100 m/s descending synthetic track matching the scenario's identity.
    import math
    # 500 m of latitude in degrees, through the tangent scale the velocity fit measures
    # against (R_M + h at the window anchor) — so the fitted V_north is exactly 100 m/s.
    from geokit import wgs84_curvature_radii
    lat_step = 500.0 / (math.pi / 180.0 * (wgs84_curvature_radii(35.6)[0] + 2000.0))
    waypoints = [[10.0 + 5.0 * k, -78.5, 35.6 + lat_step * k, 2000.0 - 50.0 * k]
                 for k in range(4)]
    # Tagged "synthetic" (= already MSL): these altitudes are constructed for an exact
    # velocity fit, and running them through the observed HAE->MSL conversion would perturb
    # the kinematics this test pins. The conversion itself is covered by
    # flight_scenarios/tests/test_datum.py, including a seam test seat-belting the fact that
    # write_reference_records reads through load_model_arrivals.
    flight = {"id": "AFR074", "icao24": "ad7f04", "landing_time_utc": "2026-06-18T21:37:36Z",
              "altitude_source": "synthetic", "waypoints": waypoints}
    target = GeodeticState(35.62, -78.5, 1850.0, 100.0, math.pi / 2, -0.1, 60000.0)
    scenario = _scenario(target=target)
    scenario.source.update({"icao24": "ad7f04", "landing_time_utc": "2026-06-18T21:37:36Z"})

    # A reference from an earlier run over a DIFFERENT flight set must not survive
    # (same stale-accumulation class as _clear_stale_records).
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    stale_ref = refs_dir / "OLD1_23L_dead00_20260101T000000Z_reference_eval.json"
    stale_ref.write_text("{}")

    written = so.write_reference_records([scenario], [flight], output_dir=tmp_path)
    assert not stale_ref.exists()
    assert written == [tmp_path / "references" /
                       "AFR074_05L_ad7f04_20260618T213736Z_reference_eval.json"]
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["controls"] == [] and len(record["states"]) == 4
    assert record["final_time_s"] == pytest.approx(15.0)          # rebased to 0
    assert record["states"][0]["t"] == 0.0 and record["states"][0]["lat"] == 35.6
    assert record["states"][1]["V"] == pytest.approx(math.hypot(100.0, 10.0), rel=1e-3)
    assert record["target_state"]["alt"] == 1850.0                # the scenario's target
    assert record["states"][0]["m"] == scenario.initial.m

    # A scenario whose flight is missing from the landings fails loudly.
    orphan = _scenario(target=target)
    orphan.source["id"] = "GHOST"
    with pytest.raises(ValueError, match="no observed flight"):
        so.write_reference_records([orphan], [flight], output_dir=tmp_path)


def test_reference_records_reuse_a_matching_canonical_source(monkeypatch, tmp_path):
    flight = {
        "id": "AFR074",
        "icao24": "ad7f04",
        "landing_time_utc": "2026-06-18T21:37:36Z",
        "altitude_source": "synthetic",
        "waypoints": [
            [0.0, -78.5, 35.6, 2000.0],
            [5.0, -78.49, 35.6, 1900.0],
        ],
    }
    target = GeodeticState(35.6, -78.48, 1800.0, 100.0, 0.0, -0.1, 60000.0)
    scenario = _scenario(target=target)
    scenario.source.update({"icao24": "ad7f04", "landing_time_utc": flight["landing_time_utc"]})
    signature = {"scenarios_sha256": "one", "arrivals_manifest_sha256": "two"}

    first = so.write_reference_records(
        [scenario],
        [flight],
        output_dir=tmp_path / "runway",
        references_dir="../shared_references/runway",
        source_signature=signature,
    )
    monkeypatch.setattr(
        so,
        "load_model_arrivals",
        lambda _path: (_ for _ in ()).throw(AssertionError("cache was not reused")),
    )
    second = so.write_reference_records(
        [scenario],
        [flight],
        output_dir=tmp_path / "runway_cons",
        references_dir="../shared_references/runway",
        source_signature=signature,
    )

    assert second == first
    assert first[0].parent == tmp_path / "shared_references" / "runway"


def test_reference_cache_rebuilds_when_a_record_changes(monkeypatch, tmp_path):
    flight = {
        "id": "AFR074",
        "icao24": "ad7f04",
        "landing_time_utc": "2026-06-18T21:37:36Z",
        "altitude_source": "synthetic",
        "waypoints": [
            [0.0, -78.5, 35.6, 2000.0],
            [5.0, -78.49, 35.6, 1900.0],
        ],
    }
    target = GeodeticState(35.6, -78.48, 1800.0, 100.0, 0.0, -0.1, 60000.0)
    scenario = _scenario(target=target)
    scenario.source.update({"icao24": "ad7f04", "landing_time_utc": flight["landing_time_utc"]})
    signature = {"scenarios_sha256": "one", "arrivals_manifest_sha256": "two"}

    [path] = so.write_reference_records(
        [scenario], [flight], output_dir=tmp_path, source_signature=signature,
    )
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["source"]["id"] = "DIFFERENT"
    path.write_text(json.dumps(changed), encoding="utf-8")

    calls = {"n": 0}

    def load_again(_tracks):
        calls["n"] += 1
        return [flight]

    monkeypatch.setattr(so, "load_model_arrivals", load_again)
    so.write_reference_records(
        [scenario], [flight], output_dir=tmp_path, source_signature=signature,
    )

    assert calls["n"] == 1
    assert json.loads(path.read_text(encoding="utf-8"))["source"]["id"] == "AFR074"


def test_batch_embeds_reference_pointers(monkeypatch, tmp_path):
    target = GeodeticState(35.59, -78.49, 500.0, 80.0, 1.5, -0.05, A320.landing_mass)
    scenarios = [_scenario(target=target), _scenario(target=target)]
    scenarios[0].source["id"] = "BAD001"

    def fake_optimize_scenario(scenario, **kwargs):
        if scenario.source["id"] == "BAD001":
            raise ValueError("Infeasible_Problem_Detected")
        return so.ScenarioOptimization(
            scenario.source, 12.0, [], [],
            evaluation=ee.evaluation_record(
                scenario.initial, scenario.target, _rollout_samples(scenario.initial),
                scenario.source, subject="optimized",
            ),
        )

    monkeypatch.setattr(so, "optimize_scenario", fake_optimize_scenario)
    so.optimize_scenarios(scenarios, output_dir=tmp_path, jobs=1, references_dir="references")

    solved = json.loads((tmp_path / "AFR074_05L_eval.json").read_text(encoding="utf-8"))
    failed = json.loads((tmp_path / "BAD001_05L_eval.json").read_text(encoding="utf-8"))
    assert solved["reference_file"] == "references/AFR074_05L_reference_eval.json"
    assert failed["reference_file"] == "references/BAD001_05L_reference_eval.json"


def test_require_usable_rollout_rejects_first_step_truncation():
    # An envelope exit at the very first integration step leaves a single-sample
    # rollout (zero horizontal extent) — that must FAIL the scenario (worker catches
    # it into a failed eval record) instead of exporting a degenerate "solved" one.
    initial = GeodeticState(35.6, -78.5, 2000.0, 130.0, 1.5, -0.05, 60000.0)
    control = LoadFactorControl(thrust=1e5, bank_rad=0.0, load_factor=1.0)
    with pytest.raises(ValueError, match="envelope"):
        so._require_usable_rollout([RolloutSample(0.0, initial, control, 0)])
    usable = [RolloutSample(0.0, initial, control, 0), RolloutSample(1.0, initial, control, 0)]
    assert so._require_usable_rollout(usable) is usable


def test_rollout_controls_carries_active_control_per_sample():
    initial = GeodeticState(35.6, -78.5, 2000.0, 130.0, 1.5, -0.05, A320.mass.max_takeoff_kg)
    node_control = [[40000.0, 0.0, 1.0], [50000.0, 0.1, 1.01]]
    samples = so.rollout_controls(initial, node_control, final_time=4.0, aircraft=A320, dt=1.0,
                                  min_altitude_m=0.0)
    assert samples[0].t == 0.0 and samples[0].control.thrust == 40000.0
    assert samples[-1].control.thrust == 50000.0 and samples[-1].segment_index == 1


def test_rollout_truncates_below_the_trajectory_floor():
    # The raw CasadiSimulator has NO envelope checks — a diverged replay used to record
    # kilometres below sea level. With min_altitude_m (the solve's guard altitude)
    # the rollout truncates at the guard instead.
    initial = GeodeticState(35.6, -78.5, 2000.0, 130.0, 1.5, -0.05, A320.mass.max_takeoff_kg)
    descending = [[0.0, 0.0, 0.98]]         # idle thrust, n slightly < 1 -> gentle descent
    full = so.rollout_controls(initial, descending, 60.0, A320, dt=0.5, min_altitude_m=0.0)
    capped = so.rollout_controls(initial, descending, 60.0, A320, dt=0.5,
                                 min_altitude_m=1900.0)
    assert full[-1].t == pytest.approx(60.0, abs=1.0)          # sea-level backstop far away
    assert capped[-1].t < full[-1].t                            # truncated at the floor
    assert all(s.state.altitude >= 1900.0 for s in capped)


def test_rollout_guard_sits_a_margin_below_the_nlp_floor():
    # Min-time plans RIDE the NLP's altitude floor, and a faithful replay oscillates
    # centimetres around it — a guard placed AT the floor truncated those replays on
    # integration noise (97% of an unconstrained batch failed on cm-scale dips). The
    # guard therefore sits ROLLOUT_GUARD_MARGIN_M below the floor: noise passes,
    # genuine divergence (tens of metres+) still truncates.
    from collocation.components import altitude_floor_m

    target_alt = 144.8
    guard = so.rollout_guard_altitude_m(target_alt)
    assert guard == pytest.approx(altitude_floor_m(target_alt) - so.ROLLOUT_GUARD_MARGIN_M)
    assert guard < altitude_floor_m(target_alt)


def test_optimize_scenario_passes_the_guard_altitude_to_the_rollout(monkeypatch):
    # White-box seam test: the rollout guard must be rollout_guard_altitude_m(target),
    # NOT the NLP floor itself (the zero-margin regression).
    target = GeodeticState(35.59, -78.49, 500.0, 80.0, 1.5, -0.05, A320.landing_mass)
    scenario = _scenario(target=target)
    captured: dict[str, float] = {}

    def fake_rollout(initial, node_control, final_time, aircraft, **kwargs):
        captured["min_altitude_m"] = kwargs["min_altitude_m"]
        return _rollout_samples(initial)

    monkeypatch.setattr(so, "CollocationOptimizer", _fake_optimizer(
        [[35.6, -78.5, 1000.0, 100.0, 1.5, -0.05],
         [35.59, -78.49, 500.0, 80.0, 1.5, -0.05]],
        10.0, [[40000.0, 0.0, 1.0]],
    ))
    monkeypatch.setattr(so, "rollout_controls", fake_rollout)
    so.optimize_scenario(scenario)
    assert captured["min_altitude_m"] == pytest.approx(
        so.rollout_guard_altitude_m(target.altitude)
    )


def test_scheme_for_fitting_maps_and_rejects():
    assert so._scheme_for_fitting("hs") == "hermiteSimpsonNormalizedFullTransport"
    assert so._scheme_for_fitting("trapezoidal") == "trapezoidalNormalizedFullTransport"
    assert so._scheme_for_fitting("rk4") == "rk4NormalizedFullTransport"
    with pytest.raises(ValueError, match="unknown fitting"):
        so._scheme_for_fitting("euler")


def test_optimize_scenario_fitting_selects_the_scheme(monkeypatch):
    # SEAM: the CLI's --fitting must reach CollocationOptimizer(scheme=...) — both
    # fittings compose with the normalized full-transport dynamics.
    target = GeodeticState(35.59, -78.49, 500.0, 80.0, 1.5, -0.05, A320.landing_mass)
    captured: dict[str, str] = {}

    monkeypatch.setattr(so, "CollocationOptimizer", _fake_optimizer(
        [[35.6, -78.5, 1000.0, 100.0, 1.5, -0.05],
         [35.59, -78.49, 500.0, 80.0, 1.5, -0.05]],
        10.0, [[40000.0, 0.0, 1.0]],
        on_init=lambda *a, **kw: captured.update(scheme=kw.get("scheme")),
    ))
    monkeypatch.setattr(so, "rollout_controls",
                        lambda initial, *a, **kw: _rollout_samples(initial))

    so.optimize_scenario(_scenario(target=target), fitting="trapezoidal")
    assert captured["scheme"] == "trapezoidalNormalizedFullTransport"
    so.optimize_scenario(_scenario(target=target))          # default stays HS
    assert captured["scheme"] == "hermiteSimpsonNormalizedFullTransport"


def test_optimize_scenario_passes_state_substeps(monkeypatch):
    # SEAM: --state-substeps must reach CollocationOptimizer(state_substeps=...);
    # None (the default) keeps the optimizer's auto per-phase density.
    target = GeodeticState(35.59, -78.49, 500.0, 80.0, 1.5, -0.05, A320.landing_mass)
    captured: dict[str, object] = {}

    monkeypatch.setattr(so, "CollocationOptimizer", _fake_optimizer(
        [[35.6, -78.5, 1000.0, 100.0, 1.5, -0.05],
         [35.59, -78.49, 500.0, 80.0, 1.5, -0.05]],
        10.0, [[40000.0, 0.0, 1.0]],
        on_init=lambda *a, **kw: captured.update(state_substeps=kw.get("state_substeps")),
    ))
    monkeypatch.setattr(so, "rollout_controls",
                        lambda initial, *a, **kw: _rollout_samples(initial))

    so.optimize_scenario(_scenario(target=target), state_substeps=8)
    assert captured["state_substeps"] == 8
    so.optimize_scenario(_scenario(target=target))
    assert captured["state_substeps"] is None               # default: auto density


def test_batch_clears_stale_records_from_a_previous_run(tmp_path):
    # Records from an earlier batch over a DIFFERENT scenario set survive by filename
    # and pollute every directory scan (python -m evaluation counted orphans into a
    # report). A fresh batch clears them; references/ is untouched (the CLI writes it
    # immediately before the batch).
    (tmp_path / "OLD1_23L_dead00_20260101T000000Z_states.json").write_text("{}")
    (tmp_path / "OLD1_23L_dead00_20260101T000000Z_eval.json").write_text("{}")
    refs = tmp_path / "references"
    refs.mkdir()
    keep = refs / "OLD1_23L_dead00_20260101T000000Z_reference_eval.json"
    keep.write_text("{}")

    so.optimize_scenarios([], output_dir=tmp_path, jobs=1)

    assert not list(tmp_path.glob("*_states.json"))
    assert not list(tmp_path.glob("*_eval.json"))
    assert keep.exists()
    assert (tmp_path / "summary.json").exists()


def test_simulate_controls_rolls_forward():
    initial = GeodeticState(35.6, -78.5, 2000.0, 130.0, 1.5, -0.05, A320.mass.max_takeoff_kg)
    # two constant-control segments over a short 4 s horizon
    node_control = [[40000.0, 0.0, 1.0], [40000.0, 0.0, 1.0]]
    samples = so.simulate_controls(initial, node_control, final_time=4.0, aircraft=A320, dt=1.0,
                                   min_altitude_m=0.0)
    assert len(samples) >= 2
    assert samples[0].t == 0.0
    assert samples[0].lat == initial.latitude  # first sample is the initial state
    assert samples[-1].t == pytest.approx(4.0, abs=1.0)


# ── Constrained min-time-IAF glue (synthetic; the real procedure data is gitignored) ──

def _pc(waypoints, *, branch_id="branch:X", nominal_kt=140.0):
    from aeroviz_backend.procedure_constraint import Glidepath, ProcedureConstraint
    return ProcedureConstraint(
        procedure_uid="UID", airport_icao="KRDU", runway_ident="RW05L",
        branch_id=branch_id, approach_course_deg=45.0,
        glidepath=Glidepath(3.0, 50.0), nominal_speed_kt=nominal_kt,
        waypoints=tuple(waypoints),
    )


def _wp(ident, lat, lon, *, alt_ft=None):
    from aeroviz_backend.procedure_constraint import ProcedureConstraintWaypoint
    return ProcedureConstraintWaypoint(
        fix_id=ident, ident=ident, role="IF", leg_type="TF",
        lon_deg=lon, lat_deg=lat, altitude=None, altitude_ref_ft=alt_ft,
        geometry_alt_ft=None, speed_max_kt=None, distance_from_start_m=0.0,
    )


def test_solve_iaf_feeds_the_optimizer_a_segment_list(monkeypatch):
    # SEAM regression: build_constraint_segments returns a plain LIST of SegmentSpec
    # (its old (segments, spans) tuple return is gone). _solve_iaf must pass that list
    # through to CollocationOptimizer(segments=...) verbatim — a stale 2-tuple unpack
    # here broke EVERY constrained batch (ValueError for != 2 legs, and for exactly
    # 2 legs a lone SegmentSpec reached the optimizer -> TypeError).
    from approach_constraints import SegmentSpec

    target = GeodeticState(35.88, -78.78, 130.0, 75.0, 0.8, -0.05, A320.landing_mass)
    # Three waypoints (two legs) ending AT the target -> the frame anchors at the runway.
    pc = _pc([
        _wp("CHWDR", 36.10, -78.70, alt_ft=5000),
        _wp("SCHOO", 36.00, -78.60, alt_ft=3000),
        _wp("RW05L", 35.88, -78.78),
    ])
    scenario = _scenario(target=target)

    captured = {}

    monkeypatch.setattr(so, "CollocationOptimizer", _fake_optimizer(
        [[35.9, -78.7, 500.0, 80.0, 0.8, -0.05]],
        100.0, [[1e5, 0.0, 1.0]],
        on_init=lambda *args, **kwargs: captured.update(
            segments=kwargs.get("segments"), scheme=kwargs.get("scheme"),
            state_substeps=kwargs.get("state_substeps"),
            n_seg_per_phase=kwargs.get("n_seg_per_phase")),
        segment_durations_s=[10.0],
    ))
    solve = so._solve_iaf(pc, scenario, target, A320, 60.0,
                          n_segments=8, dt=1.0, max_duration=600.0, verbose=False)
    assert solve.final_time == 100.0
    assert isinstance(captured["segments"], list) and len(captured["segments"]) >= 2
    assert all(isinstance(s, SegmentSpec) for s in captured["segments"])
    assert captured["scheme"] == "hermiteSimpsonNormalizedFullTransport"  # default fitting
    # n_seg_per_phase defaults to the optimizer's own default (single-sourced)
    assert captured["n_seg_per_phase"] == so.DEFAULT_N_SEG_PER_PHASE

    # --fitting / --state-substeps / --n-seg-per-phase all reach the CONSTRAINED path
    so._solve_iaf(pc, scenario, target, A320, 60.0,
                  n_segments=8, dt=1.0, max_duration=600.0, verbose=False,
                  fitting="trapezoidal", state_substeps=6, n_seg_per_phase=5)
    assert captured["scheme"] == "trapezoidalNormalizedFullTransport"
    assert captured["state_substeps"] == 6
    assert captured["n_seg_per_phase"] == 5


def test_snap_target_to_procedure_uses_the_cifp_threshold():
    # The config-derived threshold can sit hundreds of metres from the procedure's
    # CIFP threshold (displaced thresholds, e.g. KSJC 12L: 390 m) — the solve target
    # must snap horizontally onto the procedure's last waypoint, keeping the vertical/
    # kinematic components (threshold+TCH altitude, Vref, pavement heading, glidepath).
    config_target = GeodeticState(37.375, -121.94, 26.58, 70.0, -0.9, -0.052, 60000.0)
    pc = _pc([_wp("ROSTE", 37.30, -121.80), _wp("OMSEE", 37.34, -121.87),
              _wp("RW12L", 37.3712, -121.9365)])
    snapped = so._snap_target_to_procedure(config_target, [pc])
    assert (snapped.latitude, snapped.longitude) == (37.3712, -121.9365)
    assert snapped.altitude == 26.58 and snapped.V == 70.0
    assert snapped.psi == -0.9 and snapped.gamma == -0.052


def test_concat_to_runway_joins_transition_and_final():
    # transition CHWDR -> SCHOO  +  final SCHOO -> RW05L  =>  CHWDR -> SCHOO -> RW05L
    trans = _pc([_wp("CHWDR", 36.10, -78.70), _wp("SCHOO", 36.00, -78.60)], branch_id="branch:T")
    final = _pc([_wp("SCHOO", 36.00, -78.60), _wp("RW05L", 35.88, -78.78)], branch_id="branch:R")
    merged = so._concat_to_runway(trans, final)
    assert [w.ident for w in merged.waypoints] == ["CHWDR", "SCHOO", "RW05L"]
    assert merged.branch_id == "branch:T"                 # labelled by the transition (the IAF)
    dists = [w.distance_from_start_m for w in merged.waypoints]
    assert dists[0] == 0.0 and dists[1] < dists[2]        # cumulative, recomputed

    # a transition that does not end at the final's first fix does not feed it
    stray = _pc([_wp("CHWDR", 36.10, -78.70), _wp("ELSEW", 36.20, -78.90)], branch_id="branch:T")
    assert so._concat_to_runway(stray, final) is None


def test_resolve_procedure_path_picks_rnav_gps(tmp_path):
    details = tmp_path / "KRDU" / "procedure-details"
    details.mkdir(parents=True)
    (details / "index.json").write_text(json.dumps({"runways": [{
        "runwayIdent": "RW05L",
        "procedures": [
            {"procedureUid": "KRDU-H05LZ-RW05L", "procedureFamily": "RNAV_RNP"},
            {"procedureUid": "KRDU-R05LY-RW05L", "procedureFamily": "RNAV_GPS"},
        ],
    }]}), encoding="utf-8")
    path = so._resolve_procedure_path(tmp_path, "KRDU", "05L")
    assert path.name == "KRDU-R05LY-RW05L.json"           # RNAV(GPS), not the RNP
    with pytest.raises(ValueError):
        so._resolve_procedure_path(tmp_path, "KRDU", "99X")


def test_path_curve_length_ranks_shorter_path_lower():
    # The naive selector ranks IAFs by this 3D path length (no solve). A near, direct path
    # must score lower than a longer one that enters farther out.
    short = _pc([_wp("SCHOO", 36.00, -78.60, alt_ft=3000.0), _wp("RW05L", 35.88, -78.78, alt_ft=400.0)])
    long = _pc([
        _wp("OTTOS", 36.30, -78.40, alt_ft=6000.0), _wp("CHWDR", 36.15, -78.55, alt_ft=5000.0),
        _wp("SCHOO", 36.00, -78.60, alt_ft=3000.0), _wp("RW05L", 35.88, -78.78, alt_ft=400.0),
    ])
    assert so._path_curve_length_m(short) < so._path_curve_length_m(long)
    assert so._path_curve_length_m(_pc([_wp("X", 36.0, -78.6)])) == float("inf")  # single point
