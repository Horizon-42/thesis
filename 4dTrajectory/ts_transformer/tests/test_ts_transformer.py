"""The ts_transformer contracts: channel round-trip, windowing/masking, and the export seam.

What is deliberately NOT covered: whether the models predict *well*. Training quality needs
real harvested tracks and a GPU budget, and asserting a loss threshold against synthetic
straight-line approaches would only test that a transformer can extend a straight line.
The tests here pin the things that break silently instead — the heading convention, the
padding mask, the split, and whether an exported record actually satisfies the validator in
``evaluation.records``.

The end-to-end test trains for two epochs on synthetic data. That is a plumbing check
(does a checkpoint round-trip into a gradeable batch), not a quality check.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_TS_DIR = Path(__file__).resolve().parents[1]
if str(_TS_DIR) not in sys.path:
    sys.path.insert(0, str(_TS_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CLI_SPEC = importlib.util.spec_from_file_location(
    "ts_transformer_cli_test", _TS_DIR / "__main__.py"
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
ts_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(ts_cli)

import channels as ch  # noqa: E402
import control_models  # noqa: E402
import batching  # noqa: E402
import build_multiflight_capacity_report as capacity_report  # noqa: E402
import coordinate_frames as frames  # noqa: E402
import cross_validation as cv  # noqa: E402
import control_rollout as control_rollout_module  # noqa: E402
import dataset as dataset_module  # noqa: E402
import batch_benchmark as batch_probe  # noqa: E402
import evaluation_protocol  # noqa: E402
import experiment_index  # noqa: E402
import fixed_dt_control_loss as fixed_dt_loss_module  # noqa: E402
import oracle_teacher.pretraining as teacher_pretraining  # noqa: E402
import run_ts_history_ablation as history_ablation  # noqa: E402
import run_ts_pipeline as pipeline_module  # noqa: E402
import run_ts_predictability_report as predictability_report  # noqa: E402
import train as train_module  # noqa: E402
from arc_length_geometry import (  # noqa: E402
    arc_length_geometry_metrics,
    arc_length_state_loss_terms,
    arc_length_velocity_metrics,
    resample_horizontal_arc_length_numpy,
    resample_horizontal_arc_length_torch,
)
from anchor_eligibility import (  # noqa: E402
    CONTROL_ANCHOR_STALL_MARGIN, eligible_random_train_anchors,
)
from aerodynamic_model.common import GeodeticState  # noqa: E402
from batching import resolve_batch_size  # noqa: E402
from config import (  # noqa: E402
    AIRCRAFT_FILTER_OPENAP_DIRECT, HORIZON_FULL, HORIZON_NORMALIZED, HORIZON_WINDOW,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
    CHECKPOINT_SELECTION_METRICS,
    CONTROL_ARC_LOCAL_VELOCITY_TANGENT_SPEED,
    CONTROL_ARC_TERMINAL_RUNWAY_COMPONENTS,
    CONTROL_DYNAMICS_POINT_MASS,
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
    CONTROL_DURATION_FACTORIZED, CONTROL_DURATION_UNIFORM,
    CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED,
    CONTROL_GRADIENT_CLIP_GLOBAL,
    CONTROL_STATE_CLOCK_OBSERVED, CONTROL_STATE_CLOCK_PREDICTED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    CONTROL_TERMINAL_CLOCK_PREDICTED,
    CONTROL_TERMINAL_CLOCK_PREDICTED_DETACHED_TIME,
    CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION,
    CONTROL_RECIPE_SIMPLE_V1,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig, control_recipe, control_simple_v1_overrides,
)
from control_envelope import CONTROL_LOWER, CONTROL_UPPER  # noqa: E402
from control_loss_components import (  # noqa: E402
    ControlStateLossResult,
    control_tracking_loss_terms,
    last_reliable_terminal_velocity_target,
)
from control_training_curriculum import (  # noqa: E402
    ControlTrainingStage,
    build_control_training_stage_view,
    build_control_training_stages,
)
from control_training_diagnostics import (  # noqa: E402
    ControlTrainingDiagnosticsAccumulator,
    clip_gradients_by_policy,
    clip_gradients_by_global_norm,
    gradient_norms,
)
from control_regularization import control_regularization_signals  # noqa: E402
from dataset import (  # noqa: E402
    ARRIVAL_DATA_PROVENANCE_SCHEMA, FixedAnchorTrajectoryWindows, FlightEpochSampler,
    Normalizer, RandomAnchorTrajectoryWindows, arrival_data_provenance, build_series,
    cross_validation_folds, require_matching_data_provenance, split_by_flight,
    split_name_for_dataset_id, window_anchors,
)
from development_cohorts import DevelopmentCohort  # noqa: E402
from evaluation.metrics import evaluate_batch  # noqa: E402
from evaluation.records import load_records, record_from_dict  # noqa: E402
from evaluation.thresholds import AssessmentContext  # noqa: E402
from export import (  # noqa: E402
    accuracy_block, build_prediction_record, observed_series_metrics, record_stem, write_batch,
)
from fixed_dt_supervision import (  # noqa: E402
    FixedDTControlSupervision,
    build_fixed_dt_supervision,
)
from fixed_anchor_validation import (  # noqa: E402
    fixed_anchor_arc_length_geometry_metrics,
    fixed_anchor_common_grid_ade_metrics,
    fixed_anchor_common_grid_metrics,
    fixed_anchor_common_truth,
    fixed_anchor_common_weights_and_terminal_velocity,
)
from forecast import Forecast, forecast_approach, forecast_approaches  # noqa: E402
from metrics import (  # noqa: E402
    RAW_KINEMATIC_METRIC_KEYS, common_physical_time_flight_metrics,
    raw_kinematic_metrics, states_with_derived_velocity,
)
from models import build_model, parameter_count  # noqa: E402
from oracle_teacher.pretraining import CachedSchedulePretrainer  # noqa: E402
from prediction_outputs import (  # noqa: E402
    ControlBounds, ControlOutputHead, ControlPrediction, StatePrediction,
)
from terminal_state_loss import terminal_state_errors  # noqa: E402
from uniform_duration_control import UniformDurationControlHead  # noqa: E402
from aerodynamic_model.torch_dynamics import enu_rhs  # noqa: E402
from synthetic import synthetic_arrivals  # noqa: E402
from train import (  # noqa: E402
    CHECKPOINT_METADATA_SCHEMA, FIT_EVALUATION_NAME, FIT_EVALUATION_SCHEMA,
    evaluate_fit_splits,
    load_checkpoint, masked_mse, position_velocity_consistency_loss,
    prediction_loss, state_prediction_loss_components, train,
)

AIRPORT, RUNWAY = "KRDU", "05L"


def _terminal_contexts():
    context = AssessmentContext(
        benchmark="lpv", airport=AIRPORT, runway=RUNWAY,
        # The synthetic fixtures build their approaches on the runway_thresholds.json
        # point (flight_scenarios.runway_target.find_threshold), which sits 6.7 m from
        # the CIFP Path Point LTP a real KRDU context would carry. Pin the synthetic
        # one so this context describes the data it is grading.
        threshold_lat=35.8745003, threshold_lon=-78.802002,
        runway_course_deg=45.0, runway_width_m=45.72,
        runway_source="faa_nasr_apt_rwy", runway_source_cycle="2026-08-06",
        procedure_source="faa_cifp_path_point", procedure_source_cycle="2026-08-06",
        threshold_elevation_hae_m=141.86,
        threshold_elevation_msl_m=111.86,
        threshold_crossing_height_m=15.0,
        lpv_course_width_m=106.75,
    )
    return {(AIRPORT, RUNWAY): context}


def _fake_data_provenance(airport: str = AIRPORT):
    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": airport,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }


def test_test_release_is_checkpoint_bound_and_one_shot_per_flight(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    provenance = _fake_data_provenance()
    payload = {
        evaluation_protocol.TEST_RELEASE_PROTOCOL_FIELD:
            evaluation_protocol.TEST_RELEASE_SCHEMA,
        "data_provenance": provenance,
        "split": {"test": ["KRDU:flight-1", "KRDU:flight-2"]},
    }

    release = evaluation_protocol.create_test_release(
        checkpoint, payload, provenance
    )
    claim = evaluation_protocol.begin_test_evaluation(
        checkpoint,
        payload,
        provenance,
        ["KRDU:flight-1"],
        output_dir=tmp_path / "prediction",
    )
    evaluation_protocol.complete_test_evaluation(checkpoint, claim)

    recorded = json.loads(release.read_text(encoding="utf-8"))
    assert recorded["status"] == "partially_evaluated"
    assert recorded["claims"][0]["status"] == "complete"
    with pytest.raises(evaluation_protocol.TestReleaseError, match="already exposed"):
        evaluation_protocol.begin_test_evaluation(
            checkpoint,
            payload,
            provenance,
            ["KRDU:flight-1"],
            output_dir=tmp_path / "repeat",
        )

    second_claim = evaluation_protocol.begin_test_evaluation(
        checkpoint,
        payload,
        provenance,
        ["KRDU:flight-2"],
        output_dir=tmp_path / "second",
    )
    evaluation_protocol.complete_test_evaluation(checkpoint, second_claim)
    recorded = json.loads(release.read_text(encoding="utf-8"))
    assert recorded["status"] == "complete"

    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(evaluation_protocol.TestReleaseError, match="checkpoint"):
        evaluation_protocol.begin_test_evaluation(
            checkpoint,
            payload,
            provenance,
            ["KRDU:flight-2"],
            output_dir=tmp_path / "changed",
        )


def test_test_evaluation_requires_a_frozen_release(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    provenance = _fake_data_provenance()
    payload = {
        evaluation_protocol.TEST_RELEASE_PROTOCOL_FIELD:
            evaluation_protocol.TEST_RELEASE_SCHEMA,
        "data_provenance": provenance,
        "split": {"test": ["KRDU:flight-1"]},
    }

    with pytest.raises(evaluation_protocol.TestReleaseError, match="freeze-test"):
        evaluation_protocol.begin_test_evaluation(
            checkpoint,
            payload,
            provenance,
            ["KRDU:flight-1"],
            output_dir=tmp_path / "prediction",
        )


def test_legacy_checkpoint_cannot_be_released_as_a_fresh_blind_test(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"legacy checkpoint")
    provenance = _fake_data_provenance()
    payload = {
        "data_provenance": provenance,
        "split": {"test": ["KRDU:flight-1"]},
    }

    with pytest.raises(evaluation_protocol.TestReleaseError, match="predates"):
        evaluation_protocol.create_test_release(checkpoint, payload, provenance)


def test_development_cohort_checkpoint_cannot_be_frozen_for_test(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"development-only checkpoint")
    provenance = _fake_data_provenance()
    payload = {
        evaluation_protocol.TEST_RELEASE_PROTOCOL_FIELD:
            evaluation_protocol.TEST_RELEASE_SCHEMA,
        "data_provenance": provenance,
        "data_selection": {
            "development_cohort": {
                "schema_version": "ts-development-cohort-v1",
                "name": "KRDU-05L-cluster-0",
            },
        },
        "split": {"test": ["KRDU:flight-1"]},
    }

    with pytest.raises(
        evaluation_protocol.TestReleaseError, match="development-only cohort"
    ):
        evaluation_protocol.create_test_release(checkpoint, payload, provenance)


def _run_development_cohort_train_cli(
    monkeypatch, tmp_path, *, built_ids
):
    cohort = DevelopmentCohort(
        name="KRDU-05L-cluster-0",
        train_flight_ids=("KRDU:train",),
        val_flight_ids=("KRDU:val",),
        selection={"kind": "approach-cluster"},
    )
    outer_splits = {
        "train": ["KRDU:train"],
        "val": ["KRDU:val"],
        "test": ["KRDU:test"],
    }
    captured = {}
    monkeypatch.setattr(ts_cli, "load_development_cohort", lambda _path: cohort)
    monkeypatch.setattr(
        ts_cli, "arrival_data_provenance", lambda _data: _fake_data_provenance()
    )
    monkeypatch.setattr(
        ts_cli, "flight_keys_by_split", lambda _provenance, _config: outer_splits
    )
    monkeypatch.setattr(
        ts_cli, "load_flight_dicts", lambda _data, include_flight_keys: [{}]
    )
    monkeypatch.setattr(
        ts_cli,
        "_build_series_or_exit",
        lambda *_args: (
            [SimpleNamespace(dataset_id=dataset_id) for dataset_id in built_ids],
            SimpleNamespace(to_dict=lambda: {"built": len(built_ids)}),
        ),
    )
    monkeypatch.setattr(ts_cli, "data_selection_audit", lambda *_args: {})
    monkeypatch.setattr(ts_cli, "development_cohort_audit", lambda *_args: {})

    def capture_train(_series, _config, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(ts_cli, "train", capture_train)
    result = ts_cli.main([
        "train",
        "--data", str(tmp_path / "manifest.json"),
        "--output-dir", str(tmp_path / "run"),
        "--development-cohort", str(tmp_path / "cohort.json"),
    ])
    return result, captured


def test_development_cohort_checkpoint_has_no_outer_test_roster(
    monkeypatch, tmp_path
):
    result, captured = _run_development_cohort_train_cli(
        monkeypatch,
        tmp_path,
        built_ids=("KRDU:train", "KRDU:val"),
    )

    assert result == 0
    assert captured["reserved_test_keys"] == []


def test_development_cohort_rejects_incomplete_rebuild(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run_development_cohort_train_cli(
            monkeypatch,
            tmp_path,
            built_ids=("KRDU:train",),
        )


def test_invalid_teacher_arguments_do_not_begin_a_formal_run(monkeypatch, tmp_path):
    began_run = False
    outer_splits = {
        "train": ["KRDU:train"],
        "val": ["KRDU:val"],
        "test": ["KRDU:test"],
    }
    monkeypatch.setattr(
        ts_cli, "arrival_data_provenance", lambda _data: _fake_data_provenance()
    )
    monkeypatch.setattr(
        ts_cli, "flight_keys_by_split", lambda _provenance, _config: outer_splits
    )
    monkeypatch.setattr(ts_cli, "load_flight_dicts", lambda *_args, **_kwargs: [{}])
    monkeypatch.setattr(
        ts_cli,
        "_build_series_or_exit",
        lambda *_args: (
            [SimpleNamespace(dataset_id="KRDU:train")],
            SimpleNamespace(to_dict=lambda: {}),
        ),
    )
    monkeypatch.setattr(ts_cli, "data_selection_audit", lambda *_args: {})

    def record_begin(*_args, **_kwargs):
        nonlocal began_run
        began_run = True
        return tmp_path / "run" / experiment_index.RUN_MANIFEST_NAME

    monkeypatch.setattr(ts_cli, "begin_run", record_begin)
    monkeypatch.setattr(
        ts_cli, "train", lambda *_args, **_kwargs: pytest.fail("train must not start")
    )

    with pytest.raises(SystemExit):
        ts_cli.main([
            "train",
            "--data", str(tmp_path / "manifest.json"),
            "--output-dir", str(tmp_path / "run"),
            "--prediction-output", PREDICTION_CONTROL,
            "--control-teacher-schedules", str(tmp_path / "teacher_schedules.npz"),
            "--control-teacher-steps", "0",
            "--campaign-id", "teacher-contract",
            "--experiment-id", "invalid-steps",
        ])

    assert not began_run


def test_predict_cli_refuses_test_without_explicit_release(tmp_path):
    with pytest.raises(SystemExit):
        ts_cli.main([
            "predict",
            "--checkpoint", str(tmp_path / "checkpoint.pt"),
            "--data", str(tmp_path / "manifest.json"),
            "--output-dir", str(tmp_path / "prediction"),
            "--split", "test",
        ])


def _frame() -> frames.ENUFrame:
    return frames.ENUFrame(lat0=35.8745, lon0=-78.8020, alt0=133.0)


def _state(*, lat=35.90, lon=-78.85, alt=900.0, V=90.0, psi=0.5, gamma=-0.05, m=60_000.0):
    return GeodeticState(latitude=lat, longitude=lon, altitude=alt, V=V, psi=psi,
                         gamma=gamma, m=m)


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


def test_openap_direct_filter_rejects_synonym_before_scenario_fallback():
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=2, seed=3)
    flights[1] = {**flights[1], "type": "A306"}
    config = TSConfig(aircraft_filter=AIRCRAFT_FILTER_OPENAP_DIRECT)

    series, report = build_series(flights, config, airport=AIRPORT)

    assert [item.scenario.aircraft.code for item in series] == ["A320"]
    assert series[0].scenario.source["dynamics_source"].startswith("openap-")
    assert series[0].scenario.source["dynamics_surrogate_typecode"] == "A320"
    assert report.selected_typecodes == {"A320": 1}
    assert report.skipped["aircraft filter rejected"] == 1
    assert report.rejected_aircraft == {"OpenAP synonym model (A306)": 1}


def _identity_normalizer() -> Normalizer:
    return Normalizer(
        mean=np.zeros(len(ch.CHANNELS), dtype=np.float64),
        std=np.ones(len(ch.CHANNELS), dtype=np.float64),
    )


def _fitted_tail_flight():
    """A 100 m/s northbound final ending 500 m short of its published threshold."""
    from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

    lat0, lon0, elevation = 35.0, -78.0, 100.0
    cross_m, crossing_height_m = 25.0, 17.5
    waypoints = []
    for along_m in range(-5000, 0, 500):
        time_s = (along_m + 5000) / 100.0
        height = crossing_height_m - math.tan(math.radians(3.0)) * along_m
        waypoints.append([
            time_s,
            lon0 + cross_m / metres_per_deg_lon(lat0),
            lat0 + along_m / METRES_PER_DEG_LAT,
            elevation + height,
        ])
    return {
        "id": "FIT001", "callsign": "FIT001", "type": "A320", "icao24": "abc001",
        "arr_airport": "KFIT", "runway": "36",
        "landing_time_utc": "2026-01-01T00:00:00Z",
        "altitude_source": "opensky_history_geoaltitude_m",
        "runway_target": {
            "lat": lat0, "lon": lon0, "elevation_msl_m": elevation,
            "elevation_hae_m": elevation, "hae_minus_msl_m": 0.0, "course_deg": 0.0,
            "threshold_crossing_height_m": 15.0, "published_glidepath_deg": 3.0,
            "position_source": "faa_cifp_path_point",
            "vertical_source": "faa_cifp_path_point",
        },
        "waypoints": waypoints,
    }


def _post_threshold_flight():
    """The fitted-tail fixture continued 500 m beyond the threshold."""
    from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

    flight = _fitted_tail_flight()
    lat0, lon0, elevation = 35.0, -78.0, 100.0
    cross_m, crossing_height_m = 25.0, 17.5
    along_m = 500.0
    time_s = 55.0
    height = crossing_height_m - math.tan(math.radians(3.0)) * along_m
    flight["waypoints"].append([
        time_s,
        lon0 + cross_m / metres_per_deg_lon(lat0),
        lat0 + along_m / METRES_PER_DEG_LAT,
        elevation + height,
    ])
    return flight


# ── Channel contract ─────────────────────────────────────────────────────────

def test_channels_round_trip_reproduces_the_original_states():
    # The forward map goes through ground speed (V*cos gamma) and the inverse recomposes
    # V from three components; the two must compose to the identity or every exported
    # trajectory carries a systematic speed/angle bias that no metric would attribute here.
    frame = _frame()
    samples = [
        (0.0, _state(psi=0.5, gamma=-0.05)),
        (2.0, _state(lat=35.91, psi=-2.9, gamma=0.02, V=110.0)),
        (4.0, _state(lat=35.92, psi=3.0, gamma=-0.10, V=70.0)),
    ]
    times, values = ch.channels_from_states(samples, frame)
    recovered = ch.states_from_channels(times, values, frame, mass_kg=60_000.0)

    for (t_in, s_in), (t_out, s_out) in zip(samples, recovered):
        assert t_out == pytest.approx(t_in)
        assert s_out.latitude == pytest.approx(s_in.latitude, abs=1e-9)
        assert s_out.longitude == pytest.approx(s_in.longitude, abs=1e-9)
        assert s_out.altitude == pytest.approx(s_in.altitude, abs=1e-9)
        assert s_out.V == pytest.approx(s_in.V, rel=1e-12)
        assert s_out.psi == pytest.approx(s_in.psi, abs=1e-12)
        assert s_out.gamma == pytest.approx(s_in.gamma, abs=1e-12)


def test_heading_uses_math_enu_not_compass_bearing():
    # psi = 0 must mean due EAST (math-ENU), not due north. Getting this backwards is a
    # reflection about the 45-degree line: it still produces plausible-looking tracks and
    # plausible-looking metrics, and it silently reads an aligned aircraft as a 90-degree
    # intercept. Assert the velocity channels directly, both directions. The 0.5%
    # tolerance absorbs the transport factor (a chart derivative is the physical
    # component ± ~0.3%); a swapped convention is a 100-vs-0 error, not a 0.5% one.
    frame = _frame()
    eastbound = [(0.0, _state(psi=0.0, gamma=0.0, V=100.0))]
    _, values = ch.channels_from_states(eastbound, frame)
    assert values[0, ch.IDX["edot"]] == pytest.approx(100.0, rel=5e-3)
    assert values[0, ch.IDX["ndot"]] == pytest.approx(0.0, abs=1e-9)

    northbound = [(0.0, _state(psi=math.pi / 2, gamma=0.0, V=100.0))]
    _, values = ch.channels_from_states(northbound, frame)
    assert values[0, ch.IDX["edot"]] == pytest.approx(0.0, abs=1e-9)
    assert values[0, ch.IDX["ndot"]] == pytest.approx(100.0, rel=5e-3)

    # ...and the inverse agrees EXACTLY: pure-north velocity reads back as psi = +90 deg
    # (the transport factors cancel through the round trip).
    back = ch.states_from_channels(np.array([0.0]), values, frame, mass_kg=1.0)
    assert back[0][1].psi == pytest.approx(math.pi / 2)


def test_transport_factors_are_pinned_at_the_closed_form():
    # The full-transport Jacobian from a physical ENU velocity to this chart's
    # derivatives: f_n = A/(R_M+h), f_e = A·cos(lat0)/((R_N+h)·cos(lat)). Pinned
    # against an independent evaluation of the closed form at lat=35.90, h=900 m with
    # the _frame() anchor — a regression in either radius, either cosine, or the h
    # term moves these in the 4th–6th decimal.
    frame = _frame()
    f_e, f_n = frame.chart_velocity_factors(35.90, 900.0)
    assert f_e == pytest.approx(0.9990293554373, abs=1e-10)
    assert f_n == pytest.approx(1.0031236001934, abs=1e-10)

    # Wiring: the factors land on the right axes with the state's own position.
    northbound = [(0.0, _state(psi=math.pi / 2, gamma=0.0, V=100.0))]
    _, values = ch.channels_from_states(northbound, frame)
    assert values[0, ch.IDX["ndot"]] == pytest.approx(100.0 * f_n, rel=1e-12)


def test_integrating_the_velocity_channels_reproduces_the_position_channels():
    # THE transport-consistency property (2026-07-20 finding A7): for a state sequence
    # whose (V, psi, gamma) is the true physical velocity of its own positions, the
    # velocity channels are the exact time derivatives of the position channels.
    # Generate such a sequence with explicit-Euler steps of the geodetic position
    # kinematics (lat_dot = V_north/(R_M+h) etc. — the optimizer's full-transport RHS)
    # at constant physical velocity. The chart coordinates are linear in (lat, lon,
    # alt), so each Euler step's chart displacement is EXACTLY dt times the chart
    # derivative at the step start — a forward difference against edot/ndot/udot at the
    # step's left endpoint is an identity up to float rounding. Before the fix the
    # velocity channels held the raw physical components, off by the transport factors
    # (~0.3%, i.e. ~0.3 m/s here); the tolerance is ~3000x tighter than that regression.
    from geokit import wgs84_curvature_radii

    frame = _frame()
    V, psi, gamma = 100.0, 0.9, -0.05
    ground = V * math.cos(gamma)
    v_east, v_north, v_up = ground * math.cos(psi), ground * math.sin(psi), V * math.sin(gamma)

    lat, lon, alt = 35.95, -78.90, 1200.0
    dt = 0.05
    samples = []
    for k in range(3):
        samples.append((k * dt, _state(lat=lat, lon=lon, alt=alt, V=V, psi=psi, gamma=gamma)))
        r_m, r_n = wgs84_curvature_radii(lat)
        lat_rate = math.degrees(v_north / (r_m + alt))
        lon_rate = math.degrees(v_east / ((r_n + alt) * math.cos(math.radians(lat))))
        lat, lon, alt = lat + lat_rate * dt, lon + lon_rate * dt, alt + v_up * dt

    times, values = ch.channels_from_states(samples, frame)
    for step in (0, 1):
        for position, derivative in (("e", "edot"), ("n", "ndot"), ("u", "udot")):
            forward = (values[step + 1, ch.IDX[position]] - values[step, ch.IDX[position]]) / dt
            assert forward == pytest.approx(values[step, ch.IDX[derivative]], rel=1e-6), (
                f"d({position})/dt != {derivative} at step {step}"
            )


def test_channels_place_the_frame_origin_at_the_threshold():
    # u is height above the THRESHOLD, not above the ellipsoid — so a state sitting exactly
    # at the frame anchor has all-zero position channels.
    frame = _frame()
    at_origin = [(0.0, _state(lat=frame.lat0, lon=frame.lon0, alt=frame.alt0))]
    _, values = ch.channels_from_states(at_origin, frame)
    assert values[0, ch.IDX["e"]] == pytest.approx(0.0)
    assert values[0, ch.IDX["n"]] == pytest.approx(0.0)
    assert values[0, ch.IDX["u"]] == pytest.approx(0.0)


def test_runway_aligned_frame_rotates_and_round_trips_horizontal_channels():
    target = _state(psi=0.73)
    frame = frames.frame_for_state(target, "runway-aligned")
    assert type(frame) is frames.RunwayAlignedFrame
    times, values = ch.channels_from_states([(0.0, target)], frame)
    assert values[0, ch.IDX["edot"]] > 0.0
    assert abs(values[0, ch.IDX["ndot"]]) < values[0, ch.IDX["edot"]] * 0.01
    restored = ch.states_from_channels(times, values, frame, mass_kg=target.m)[0][1]
    assert restored.latitude == pytest.approx(target.latitude)
    assert restored.longitude == pytest.approx(target.longitude)
    assert restored.psi == pytest.approx(target.psi)


def test_coordinate_frame_setting_selects_a_concrete_implementation():
    target = _state(psi=0.73)

    assert type(frames.frame_for_state(target, "enu")) is frames.ENUFrame
    assert type(frames.frame_for_state(target, "runway-aligned")) is frames.RunwayAlignedFrame
    assert not hasattr(frames.frame_for_state(target, "enu"), "coordinate_mode")
    with pytest.raises(ValueError, match="unknown coordinate frame"):
        frames.frame_for_state(target, "other")

    enu_series, _ = _series(n_flights=1, coordinate_frame="enu")
    aligned_series, _ = _series(n_flights=1, coordinate_frame="runway-aligned")
    assert type(enu_series[0].frame) is frames.ENUFrame
    assert type(aligned_series[0].frame) is frames.RunwayAlignedFrame

    # Any anchor the pipeline builds has the whole lookback behind it; the anchor-state
    # control inversion differentiates that window, so it needs a real anchor, not 0.
    anchor = dataset_module.ANCHOR_CONTROL_SAMPLES
    enu_dynamics = dataset_module.dynamics_arrays(enu_series[0], anchor)
    aligned_dynamics = dataset_module.dynamics_arrays(aligned_series[0], anchor)
    runway_heading = enu_series[0].scenario.target.psi
    assert enu_dynamics["frame_params"][3] == pytest.approx(0.0)
    assert aligned_dynamics["frame_params"][3] == pytest.approx(runway_heading)
    assert enu_dynamics["runway_heading_rad"] == pytest.approx(runway_heading)
    assert aligned_dynamics["runway_heading_rad"] == pytest.approx(runway_heading)


def test_resample_lands_on_a_regular_grid_without_extrapolating():
    times = np.array([0.0, 1.0, 3.0, 7.5])
    values = np.tile(np.arange(len(times), dtype=float)[:, None], (1, len(ch.CHANNELS)))
    grid, resampled = ch.resample_uniform(times, values, 2.0)

    assert np.allclose(np.diff(grid), 2.0)
    assert grid[0] == 0.0
    # Never past the last real sample: 7.5s of track on a 2s grid stops at 6.0.
    assert grid[-1] == pytest.approx(6.0)
    assert len(resampled) == len(grid)


def test_resample_rejects_a_track_shorter_than_one_step():
    with pytest.raises(ValueError, match="too short"):
        ch.resample_uniform(np.array([0.0, 0.5]), np.zeros((2, len(ch.CHANNELS))), 4.0)


# ── Windowing + masking ──────────────────────────────────────────────────────

def test_normalized_windows_use_every_anchor_with_a_future_remainder():
    series, config = _series(n_flights=2, n_segments=16)
    s = series[0]
    anchors = window_anchors(s, config)
    assert anchors.start == config.seq_len - 1
    assert anchors.stop - 1 == min(s.n_samples - 1, s.n_supervision_samples - 2)

    dataset = RandomAnchorTrajectoryWindows([s], config, Normalizer.fit([s]))
    x, y, weights, final_time_s, _flight_weight = dataset[len(dataset) - 1]
    assert x.shape == (config.seq_len, len(config.channels))
    assert y.shape == (config.n_segments, len(config.channels))
    assert weights.shape == y.shape
    assert float(final_time_s) > 0.0


def test_normalized_windows_interpolate_the_endpoint_without_padding():
    series, config = _series(n_flights=2, n_segments=12)
    s = series[0]
    normalizer = Normalizer.fit([s])
    dataset = FixedAnchorTrajectoryWindows([s], config, normalizer)

    x, y, weights, final_time_s, _flight_weight = dataset[len(dataset) - 1]
    assert x.shape == (config.seq_len, len(config.channels))
    assert y.shape == (config.n_segments, len(config.channels))
    assert torch.all(weights > 0.0)
    expected_endpoint = normalizer.encode(s.supervision_values[-1:])[0]
    assert y[-1].numpy() == pytest.approx(expected_endpoint)
    assert float(final_time_s) == pytest.approx(
        s.supervision_times[-1] - s.times[dataset.index[-1][1]]
    )


def test_full_windows_use_physical_dt_and_mask_after_the_endpoint():
    config = TSConfig(
        seq_len=3,
        n_segments=4,
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=4,
        dt_s=2.0,
        random_train_anchor_min_future_s=0.0,
    )
    series, report = build_series([_fitted_tail_flight()], config, airport="KFIT")
    assert report.built == 1
    s = series[0]
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))

    _x, target, weights, final_time_s, _flight_weight = dataset[len(dataset) - 1]
    assert config.pred_len == 4
    assert target.shape == (4, len(config.channels))
    assert float(final_time_s) == pytest.approx(6.0)
    # Queries are +2, +4, +6 seconds, followed by one padded row.
    assert torch.all(weights[:3, :3].sum(dim=-1) > 0.0)
    assert torch.all(weights[3] == 0.0)
    endpoint = dataset.normalizer.decode(target[2:3].numpy())[0]
    assert endpoint[ch.IDX["e"]] == pytest.approx(25.0, abs=1e-4)


def test_window_mode_requires_a_complete_short_horizon():
    config = TSConfig(
        seq_len=3,
        horizon_mode=HORIZON_WINDOW,
        window_horizon_steps=4,
        dt_s=2.0,
        random_train_anchor_min_future_s=0.0,
    )
    series, report = build_series([_fitted_tail_flight()], config, airport="KFIT")
    assert report.built == 1

    anchors = window_anchors(series[0], config)
    assert anchors.stop - 1 == series[0].n_supervision_samples - config.pred_len - 1
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    _x, _target, weights, _final_time_s, _flight_weight = dataset[len(dataset) - 1]
    assert torch.all(weights.sum(dim=-1) > 0.0)


def test_window_anchor_requires_complete_physical_duration_for_fractional_endpoint():
    config = TSConfig(
        seq_len=3,
        horizon_mode=HORIZON_WINDOW,
        window_horizon_steps=4,
        dt_s=2.0,
    )
    [source], _ = _series(n_flights=1, seq_len=3)
    observed_times = np.arange(0.0, 12.0, 2.0)
    observed_values = np.zeros((len(observed_times), len(ch.CHANNELS)))
    series = dataset_module.FlightSeries(
        flight_id=source.flight_id,
        scenario=source.scenario,
        frame=source.frame,
        times=observed_times,
        values=observed_values,
        supervision_times=np.append(observed_times, 10.5),
        supervision_values=np.zeros((len(observed_times) + 1, len(ch.CHANNELS))),
        supervision_weights=np.ones(
            (len(observed_times) + 1, len(ch.CHANNELS))
        ) / len(ch.CHANNELS),
    )

    # Counting the four rows after index 2 would admit it, but 10.5 - 4.0 is only
    # 6.5 seconds. Recursive inference always advances 4 * 2 = 8 seconds per pass.
    assert list(window_anchors(series, config)) == []


def test_vectorized_interpolation_matches_the_scalar_reference():
    series, config = _series(n_flights=2, n_segments=17)
    s = series[0]
    normalizer = Normalizer.fit([s])
    dataset = RandomAnchorTrajectoryWindows([s], config, normalizer)
    sample_index = len(dataset) // 2
    _x, target, weights, final_time_s, _flight_weight = dataset[sample_index]
    _series_index, anchor = dataset.index[sample_index]
    query_times = s.times[anchor] + dataset.progress * float(final_time_s)
    encoded = normalizer.encode(s.supervision_values).astype(np.float32)

    expected_target = np.column_stack([
        np.interp(query_times, s.supervision_times, encoded[:, channel])
        for channel in range(len(config.channels))
    ]).astype(np.float32)
    expected_weights = np.column_stack([
        np.interp(query_times, s.supervision_times, s.supervision_weights[:, channel])
        for channel in range(len(config.channels))
    ]).astype(np.float32)

    assert target.numpy() == pytest.approx(expected_target)
    assert weights.numpy() == pytest.approx(expected_weights)


def test_masked_mse_ignores_padded_steps():
    # Two horizon steps, one of them padding. A huge error hidden in the padded step must
    # not move the loss at all — otherwise every short approach trains the model to
    # reproduce its own zero padding and forecast tails collapse toward the threshold.
    predicted = torch.tensor([[[1.0], [999.0]]])
    target = torch.tensor([[[0.0], [0.0]]])
    mask = torch.tensor([[[1.0], [0.0]]])
    assert float(masked_mse(predicted, target, mask)) == pytest.approx(1.0)

    all_valid = torch.tensor([[[1.0], [1.0]]])
    assert float(masked_mse(predicted, target, all_valid)) > 1.0


def test_fitted_tail_supervises_position_only_and_keeps_observed_inputs_separate():
    config = TSConfig(
        seq_len=3, n_segments=3, dt_s=2.0,
        random_train_anchor_min_future_s=0.0,
    )
    series, report = build_series([_fitted_tail_flight()], config, airport="KFIT")
    assert report.built == 1
    s = series[0]

    # Observations stop at t=44 (the raw t=45 endpoint is off-grid); fitted labels continue
    # at t=46/48/50, but series.values — the forecast input — remains measured-only.
    assert s.times[-1] == pytest.approx(44.0)
    assert s.supervision_times[-3:] == pytest.approx([46.0, 48.0, 50.0])
    assert s.n_supervision_samples == s.n_samples + 3

    tail_weights = s.supervision_weights[s.n_samples:]
    assert np.all(tail_weights[:, 3:] == 0.0)
    assert tail_weights.sum(axis=1) == pytest.approx([0.25, 0.25, 1.25])
    assert s.supervision_values[-1, ch.IDX["e"]] == pytest.approx(25.0, abs=1e-6)
    assert s.supervision_values[-1, ch.IDX["n"]] == pytest.approx(0.0, abs=1e-6)
    assert s.supervision_values[-1, ch.IDX["u"]] == pytest.approx(2.5, abs=1e-6)

    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    assert dataset.index[-1] == (0, s.n_samples - 1)
    x, _, weights, final_time_s, _flight_weight = dataset[len(dataset) - 1]
    assert x.shape == (config.seq_len, len(config.channels))
    assert weights.sum() == pytest.approx(1.75)
    assert torch.all(weights[:, 3:] == 0.0)
    assert float(final_time_s) == pytest.approx(6.0)


def test_normalized_interpolation_never_supervises_fitted_velocity_placeholders():
    config = TSConfig(
        seq_len=3, n_segments=4, dt_s=2.0,
        random_train_anchor_min_future_s=0.0,
    )
    series, report = build_series([_fitted_tail_flight()], config, airport="KFIT")
    assert report.built == 1
    s = series[0]
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))

    _, _, weights, final_time_s, _flight_weight = dataset[len(dataset) - 1]
    first_query_time = s.times[-1] + float(final_time_s) / config.n_segments
    assert s.times[-1] < first_query_time < s.supervision_times[s.n_samples]
    assert torch.all(weights[:, 3:] == 0.0)
    assert torch.all(weights[:, :3].sum(dim=-1) > 0.0)


def test_normalized_target_interpolates_and_stops_at_observed_threshold_crossing():
    config = TSConfig(
        seq_len=3, n_segments=4, dt_s=2.0,
        random_train_anchor_min_future_s=0.0,
    )
    series, report = build_series([_post_threshold_flight()], config, airport="KFIT")
    assert report.built == 1
    s = series[0]

    assert s.times[-1] == pytest.approx(48.0)
    assert s.supervision_times[-1] == pytest.approx(50.0)
    assert s.supervision_values[-1, ch.IDX["e"]] == pytest.approx(25.0, abs=1e-6)
    assert s.supervision_values[-1, ch.IDX["n"]] == pytest.approx(0.0, abs=1e-6)
    assert np.all(s.supervision_weights[-1] == pytest.approx(1.0 / len(ch.CHANNELS)))

    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    _, target, _, final_time_s, _flight_weight = dataset[len(dataset) - 1]
    assert float(final_time_s) == pytest.approx(2.0)
    endpoint = dataset.normalizer.decode(target[-1:].numpy())[0]
    assert endpoint[ch.IDX["n"]] == pytest.approx(0.0, abs=1e-4)


def test_observed_threshold_crossing_does_not_depend_on_a_fitted_tail(monkeypatch):
    monkeypatch.setattr(dataset_module, "fit_flight_final_approach", lambda _flight: None)
    config = TSConfig(seq_len=3, n_segments=4, dt_s=2.0)
    series, report = build_series([_post_threshold_flight()], config, airport="KFIT")

    assert report.built == 1
    assert series[0].times[-1] == pytest.approx(48.0)
    assert series[0].supervision_times[-1] == pytest.approx(50.0)
    assert series[0].supervision_values[-1, ch.IDX["n"]] == pytest.approx(0.0, abs=1e-6)


def test_channel_weighted_mse_ignores_fitted_velocity_placeholders():
    predicted = torch.zeros((1, 1, len(ch.CHANNELS)))
    target = torch.tensor([[[1.0, 1.0, 1.0, 999.0, 999.0, 999.0]]])
    weights = torch.tensor([[[1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0]]])
    assert float(masked_mse(predicted, target, weights)) == pytest.approx(1.0)


def test_prediction_loss_adds_scaled_final_time_error():
    config = TSConfig(
        final_time_scale_s=600.0,
        final_time_loss_weight=2.0,
        kinematic_consistency_loss_weight=0.0,
        terminal_loss_weight=0.0,
    )
    states = torch.zeros((1, 2, len(ch.CHANNELS)))
    prediction = StatePrediction(states=states, final_time_s=torch.tensor([900.0]))
    loss = prediction_loss(
        prediction,
        torch.zeros((1, len(ch.CHANNELS))),
        states,
        torch.ones_like(states),
        torch.tensor([600.0]),
        torch.ones(1),
        config,
        _identity_normalizer(),
    )
    assert float(loss) == pytest.approx(0.5)


def test_prediction_loss_applies_per_flight_airport_weights():
    config = TSConfig(
        n_segments=2,
        validation_common_grid_points=2,
        final_time_loss_weight=0.0,
        state_endpoint_loss_weight=0.0,
        kinematic_consistency_loss_weight=0.0,
        terminal_loss_weight=0.0,
    )
    target = torch.zeros((2, 2, len(ch.CHANNELS)))
    prediction = StatePrediction(
        states=torch.stack((torch.ones_like(target[0]), 3.0 * torch.ones_like(target[0]))),
        final_time_s=torch.full((2,), 10.0),
    )
    loss = prediction_loss(
        prediction,
        torch.zeros((2, len(ch.CHANNELS))),
        target,
        torch.ones_like(target),
        torch.full((2,), 10.0),
        torch.tensor([1.5, 0.5]),
        config,
        _identity_normalizer(),
    )

    expected_physical_position_mse = (3.0 * 1.5 + 27.0 * 0.5) / 2.0
    assert float(loss) == pytest.approx(expected_physical_position_mse / 10_000.0**2)


def test_default_checkpoint_selection_is_common_true_time_ade():
    assert TSConfig().checkpoint_selection_metric == CHECKPOINT_SELECTION_COMMON_GRID_ADE


def test_state_loss_is_isotropic_physical_position_plus_time_only():
    normalizer = Normalizer(
        mean=np.zeros(len(ch.CHANNELS)),
        std=np.array([100.0, 200.0, 10.0, 2.0, 3.0, 4.0]),
    )
    config = TSConfig(
        n_segments=2,
        validation_common_grid_points=2,
        final_time_loss_weight=1.0,
        state_endpoint_loss_weight=0.0,
        kinematic_consistency_loss_weight=99.0,
        terminal_loss_weight=99.0,
    )
    target = torch.zeros((1, 2, len(ch.CHANNELS)))
    predicted = target.clone()
    predicted[..., ch.IDX["e"]] = 1.0       # 100 m
    predicted[..., ch.IDX["u"]] = 10.0      # 100 m despite a different std
    predicted[..., list(ch.VELOCITY_IDX)] = 999.0
    components = state_prediction_loss_components(
        StatePrediction(states=predicted, final_time_s=torch.tensor([600.0])),
        torch.zeros((1, len(ch.CHANNELS))),
        target,
        torch.ones_like(target),
        torch.tensor([600.0]),
        torch.ones(1),
        config,
        normalizer,
    )

    assert float(components.state) == pytest.approx(
        (100.0**2 + 100.0**2) / 10_000.0**2
    )
    assert float(components.final_time) == pytest.approx(0.0)
    assert float(components.kinematic) == pytest.approx(0.0)
    assert float(components.terminal) == pytest.approx(0.0)


def test_state_position_loss_uses_true_time_not_equal_progress():
    config = TSConfig(
        n_segments=2,
        validation_common_grid_points=2,
        final_time_loss_weight=0.0,
    )
    target = torch.zeros((1, 2, len(ch.CHANNELS)))
    target[0, :, ch.IDX["e"]] = torch.tensor([50.0, 100.0])
    prediction = StatePrediction(
        states=target.clone(),
        final_time_s=torch.tensor([5.0]),
    )
    components = state_prediction_loss_components(
        prediction,
        torch.zeros((1, len(ch.CHANNELS))),
        target,
        torch.ones_like(target),
        torch.tensor([10.0]),
        torch.ones(1),
        config,
        _identity_normalizer(),
    )

    # At true t=5 s, the short prediction has already reached its 100 m endpoint while
    # truth is at 50 m.  Equal-progress loss would incorrectly be zero.
    assert float(components.state) == pytest.approx(
        (50.0**2 / 2.0) / 10_000.0**2
    )


def test_common_physical_time_metric_penalizes_an_early_ending_prediction():
    anchor = np.zeros(len(ch.CHANNELS))
    truth = np.zeros((2, len(ch.CHANNELS)))
    truth[:, ch.IDX["e"]] = [50.0, 100.0]
    predicted = np.zeros((1, len(ch.CHANNELS)))
    predicted[0, ch.IDX["e"]] = 50.0

    block = common_physical_time_flight_metrics(
        anchor_values=anchor,
        predicted_values=predicted,
        predicted_offsets_s=np.array([5.0]),
        predicted_final_time_s=5.0,
        truth_values=truth,
        truth_offsets_s=np.array([5.0, 10.0]),
        true_final_time_s=10.0,
        points=2,
    )

    # The old overlap-only metric saw only the exact 5 s point and returned zero.  The
    # common grid holds the predicted endpoint through the true 10 s horizon.
    assert block["ade_m"] == pytest.approx(25.0)
    assert block["fde_m"] == pytest.approx(50.0)
    assert block["final_time_error_s"] == pytest.approx(-5.0)
    assert block["coverage_ratio"] == pytest.approx(0.5)


def test_common_physical_time_metric_uses_the_truth_path_frame():
    anchor = np.zeros(len(ch.CHANNELS))
    truth = np.zeros((2, len(ch.CHANNELS)))
    truth[:, ch.IDX["e"]] = [100.0, 200.0]
    predicted = truth.copy()
    predicted[:, ch.IDX["n"]] = 50.0

    block = common_physical_time_flight_metrics(
        anchor_values=anchor,
        predicted_values=predicted,
        predicted_offsets_s=np.array([1.0, 2.0]),
        predicted_final_time_s=2.0,
        truth_values=truth,
        truth_offsets_s=np.array([1.0, 2.0]),
        true_final_time_s=2.0,
        points=2,
    )

    assert block["along_track_m"]["mean_abs"] == pytest.approx(0.0, abs=1e-9)
    assert block["cross_track_m"]["mean_signed"] == pytest.approx(50.0)
    assert block["vertical_m"]["mean_abs"] == pytest.approx(0.0, abs=1e-9)


def test_common_physical_time_metric_rejects_nonfinite_or_nonmonotonic_paths():
    anchor = np.zeros(len(ch.CHANNELS))
    path = np.zeros((2, len(ch.CHANNELS)))
    arguments = dict(
        anchor_values=anchor,
        predicted_values=path,
        predicted_final_time_s=2.0,
        truth_values=path,
        truth_offsets_s=np.array([1.0, 2.0]),
        true_final_time_s=2.0,
        points=2,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        common_physical_time_flight_metrics(
            **arguments, predicted_offsets_s=np.array([1.0, 1.0])
        )
    bad = path.copy()
    bad[0, ch.IDX["u"]] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        common_physical_time_flight_metrics(
            **{**arguments, "predicted_values": bad},
            predicted_offsets_s=np.array([1.0, 2.0]),
        )


def test_state_velocity_is_derived_from_the_same_piecewise_linear_position_curve():
    anchor = np.zeros(len(ch.CHANNELS))
    predicted = np.zeros((2, len(ch.CHANNELS)))
    predicted[:, ch.IDX["e"]] = [10.0, 40.0]
    predicted[:, ch.IDX["n"]] = [0.0, 8.0]
    predicted[:, list(ch.VELOCITY_IDX)] = 999.0

    derived = states_with_derived_velocity(
        anchor, predicted, np.array([2.0, 4.0])
    )

    assert derived[:, ch.IDX["edot"]].tolist() == pytest.approx([5.0, 7.5])
    assert derived[:, ch.IDX["ndot"]].tolist() == pytest.approx([0.0, 2.0])
    assert derived[:, ch.IDX["udot"]].tolist() == pytest.approx([0.0, 0.0])
    assert derived[:, list(ch.POSITION_IDX)] == pytest.approx(
        predicted[:, list(ch.POSITION_IDX)]
    )


def test_position_velocity_consistency_loss_is_zero_for_integrated_motion():
    anchor = torch.zeros((1, len(ch.CHANNELS)))
    anchor[0, ch.IDX["edot"]] = 1.0
    states = torch.zeros((1, 3, len(ch.CHANNELS)))
    states[0, :, ch.IDX["e"]] = torch.tensor([1.0, 2.0, 3.0])
    states[0, :, ch.IDX["edot"]] = 1.0

    loss = position_velocity_consistency_loss(
        anchor,
        states,
        torch.tensor([3.0]),
        _identity_normalizer(),
    )

    assert loss.tolist() == pytest.approx([0.0])


def test_position_velocity_consistency_loss_uses_physical_channel_scales():
    normalizer = Normalizer(
        mean=np.array([10.0, 20.0, 30.0, 5.0, 6.0, 7.0]),
        std=np.array([2.0, 3.0, 4.0, 8.0, 9.0, 10.0]),
    )
    states = torch.zeros((1, 3, len(ch.CHANNELS)))
    # One normalized e increment is 2 m. With dt=0.25 s it equals the decoded
    # edot=8 m/s represented by (8 - mean 5) / std 8.
    states[0, :, ch.IDX["e"]] = torch.tensor([0.0, 1.0, 2.0])
    states[0, :, ch.IDX["edot"]] = (8.0 - 5.0) / 8.0
    states[0, :, ch.IDX["ndot"]] = (0.0 - 6.0) / 9.0
    states[0, :, ch.IDX["udot"]] = (0.0 - 7.0) / 10.0
    anchor = states[:, 0].clone()
    anchor[0, ch.IDX["e"]] = -1.0

    loss = position_velocity_consistency_loss(
        anchor,
        states,
        torch.tensor([0.75]),
        normalizer,
    )

    assert loss.tolist() == pytest.approx([0.0], abs=1e-12)


def test_position_velocity_consistency_normalizes_displacement_by_position_scale():
    normalizer = Normalizer(
        mean=np.zeros(len(ch.CHANNELS)),
        std=np.array([100.0, 200.0, 300.0, 2.0, 4.0, 5.0]),
    )
    anchor = torch.zeros((1, len(ch.CHANNELS)))
    states = torch.zeros((1, 1, len(ch.CHANNELS)))
    states[0, 0, ch.IDX["e"]] = 0.1  # 10 m displacement, zero predicted velocity.

    loss = position_velocity_consistency_loss(
        anchor,
        states,
        torch.tensor([1.0]),
        normalizer,
    )

    assert loss.tolist() == pytest.approx([(10.0 / 100.0) ** 2 / 3.0])


def test_full_kinematic_loss_uses_dt_and_short_final_segment():
    config = TSConfig(
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=3,
        dt_s=2.0,
    )
    normalizer = _identity_normalizer()
    anchor = torch.zeros((1, len(ch.CHANNELS)))
    states = torch.zeros((1, 3, len(ch.CHANNELS)))
    states[0, :, ch.IDX["e"]] = torch.tensor([2.0, 3.0, 999.0])
    states[0, :2, ch.IDX["edot"]] = 1.0
    anchor[0, ch.IDX["edot"]] = 1.0
    weights = torch.zeros_like(states)
    weights[0, :2, list(ch.POSITION_IDX)] = 1.0

    loss = position_velocity_consistency_loss(
        anchor,
        states,
        torch.tensor([3.0]),
        normalizer,
        config=config,
        state_weights=weights,
    )

    assert loss.tolist() == pytest.approx([0.0], abs=1e-12)


def test_state_loss_has_one_explicit_physical_output_endpoint_task():
    config = TSConfig(
        final_time_loss_weight=0.0,
        state_endpoint_loss_weight=1.0,
        kinematic_consistency_loss_weight=0.0,
        terminal_loss_weight=99.0,
        validation_common_grid_points=2,
    )
    target = torch.zeros((1, 2, len(ch.CHANNELS)))
    predicted = target.clone()
    predicted[0, -1, ch.IDX["e"]] = 1.0

    loss = prediction_loss(
        StatePrediction(states=predicted, final_time_s=torch.tensor([2.0])),
        torch.zeros((1, len(ch.CHANNELS))),
        target,
        torch.ones_like(target),
        torch.tensor([2.0]),
        torch.ones(1),
        config,
        _identity_normalizer(),
    )

    assert float(loss) == pytest.approx((1.0 / 2.0 + 1.0) / 10_000.0**2)


def test_full_position_and_endpoint_loss_ignore_the_padded_suffix():
    config = TSConfig(
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=3,
        final_time_loss_weight=0.0,
        state_endpoint_loss_weight=1.0,
        kinematic_consistency_loss_weight=0.0,
        terminal_loss_weight=99.0,
    )
    target = torch.zeros((1, 3, len(ch.CHANNELS)))
    predicted = target.clone()
    predicted[0, 1, ch.IDX["e"]] = 1.0
    predicted[0, 2, ch.IDX["e"]] = 1000.0
    weights = torch.zeros_like(target)
    # The first two rows are valid. The 1000 m padded suffix must not enter the loss.
    weights[0, :2, ch.IDX["n"]] = 1.0

    loss = prediction_loss(
        StatePrediction(states=predicted, final_time_s=torch.tensor([3.0])),
        torch.zeros((1, len(ch.CHANNELS))),
        target,
        weights,
        torch.tensor([3.0]),
        torch.ones(1),
        config,
        _identity_normalizer(),
    )

    # Row 1 is the last supervised output endpoint, so it contributes once to the path
    # average and once to the explicit endpoint task. Row 2 is padding and contributes 0.
    assert float(loss) == pytest.approx((1.0 / 2.0 + 1.0) / 10_000.0**2)


def test_state_endpoint_rejects_noncontiguous_position_supervision():
    config = TSConfig(
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=3,
        final_time_loss_weight=0.0,
    )
    target = torch.zeros((1, 3, len(ch.CHANNELS)))
    weights = torch.zeros_like(target)
    weights[0, (0, 2), ch.IDX["e"]] = 1.0

    with pytest.raises(ValueError, match="contiguous prefix"):
        prediction_loss(
            StatePrediction(states=target.clone(), final_time_s=torch.tensor([3.0])),
            torch.zeros((1, len(ch.CHANNELS))),
            target,
            weights,
            torch.tensor([3.0]),
            torch.ones(1),
            config,
            _identity_normalizer(),
        )


def test_split_by_flight_is_disjoint_and_reproducible():
    series, config = _series(n_flights=10)
    train_a, val_a, test_a = split_by_flight(series, config)
    train_b, val_b, test_b = split_by_flight(series, config)

    ids = lambda group: [s.flight_id for s in group]  # noqa: E731
    assert ids(train_a) == ids(train_b) and ids(val_a) == ids(val_b)  # seeded -> stable

    everything = ids(train_a) + ids(val_a) + ids(test_a)
    assert len(everything) == len(set(everything)) == len(series)  # partition, no overlap
    for split_name, group in (("train", train_a), ("val", val_a), ("test", test_a)):
        assert all(split_name_for_dataset_id(item.dataset_id, config) == split_name
                   for item in group)


def test_split_seed_locks_outer_split_across_training_seeds():
    series, _config = _series(n_flights=100)
    first = TSConfig(seed=1337, split_seed=1337)
    repeated = TSConfig(seed=2027, split_seed=1337)

    def split_ids(config):
        return tuple(
            tuple(item.dataset_id for item in group)
            for group in split_by_flight(series, config)
        )

    assert split_ids(first) == split_ids(repeated)


def test_pooled_prediction_filters_checkpoint_split_to_current_airport_subset():
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{"airport": "KAAA"}],
    }
    assert ts_cli.split_keys_for_current_data(
        ["KAAA:flight-a", "KBBB:flight-b", "KAAA:flight-c"], provenance
    ) == ["KAAA:flight-a", "KAAA:flight-c"]


def test_windows_never_straddle_two_flights():
    # The index is (series, anchor) pairs, so a window cannot span flights. Guard it
    # explicitly: concatenating series into one array would be an easy "optimisation" that
    # silently teaches the model to fly from one aircraft's track into another's.
    series, config = _series(n_flights=3)
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    for s_idx, anchor in dataset.index:
        assert anchor - config.seq_len + 1 >= 0
        assert anchor < series[s_idx].n_samples


def _write_arrival_manifest(root: Path, ids: list[str], *, airport: str = "KRDU") -> Path:
    from dataset import flight_key

    arrivals = root / "arrivals"
    tracks = root / "tracks"
    records = tracks / "assigned" / "05L"
    records.mkdir(parents=True)
    roster = []
    source_roster = []
    runway_target = {
        "lat": 35.8, "lon": -78.8, "elevation_hae_m": 100.0,
        "elevation_msl_m": 133.5, "hae_minus_msl_m": -33.5,
        "course_deg": 50.0, "threshold_crossing_height_m": 15.0,
        "published_glidepath_deg": 3.0,
        "position_source": "faa_cifp_path_point",
        "vertical_source": "faa_cifp_path_point",
    }
    for index, ident in enumerate(ids):
        flight = {
            "flight_key": None,
            "callsign": ident,
            "icao24": f"abc{index:03d}",
            "runway": "05L",
            "landing_time_utc": f"2026-01-01T00:00:{index:02d}Z",
            "altitude_source": "opensky_history_geoaltitude_m",
            "samples": [[0.0, -78.8, 35.8, 500.0]],
        }
        key = flight_key(
            {
                "id": ident,
                "runway": "05L",
                "icao24": flight["icao24"],
                "landing_time_utc": flight["landing_time_utc"],
            },
            index,
        )
        flight["flight_key"] = key
        relative = f"assigned/05L/{key}.json"
        source_text = json.dumps(flight)
        (tracks / relative).write_text(source_text, encoding="utf-8")
        source_roster.append({"flight_key": key, "file": relative})
        roster.append({
            "flight_key": key,
            "source_file": relative,
            "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            "first_sample_index": 0,
            "last_sample_index": 0,
            "runway": "05L",
            "arrival_truncated": False,
            "cut_samples": 0,
            "arrival_duration_s": 0.0,
            "entry_time_utc": flight["landing_time_utc"],
        })
    (tracks / "manifest.json").write_text(
        json.dumps({"records": source_roster}), encoding="utf-8"
    )
    manifest = arrivals / "manifest.json"
    arrivals.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "harvest-arrivals-v4-source-timed-track-slices",
                "airport": airport,
                "source_manifest": "../tracks/manifest.json",
                "altitude_source": "opensky_history_geoaltitude_m",
                "runway_targets": {"05L": runway_target},
                "records": roster,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_ts_load_uses_only_the_arrival_manifest_roster(tmp_path):
    # An orphan beside the roster is deliberately ignored: no glob can leak rejected or
    # stale flights into the train/validation/test split.
    from dataset import load_flight_dicts

    _write_arrival_manifest(tmp_path, ["A", "B", "C"])
    (tmp_path / "tracks" / "assigned" / "05L" / "orphan.json").write_text(
        json.dumps({"id": "ORPHAN"}), encoding="utf-8"
    )

    assert [f["id"] for f in load_flight_dicts(tmp_path, verbose=False)] == ["A", "B", "C"]


def test_ts_load_aggregates_multiple_airport_manifests(tmp_path):
    from dataset import load_flight_dicts

    first = _write_arrival_manifest(tmp_path / "first", ["A"], airport="KAAA")
    second = _write_arrival_manifest(tmp_path / "second", ["B"], airport="KBBB")
    flights = load_flight_dicts([second, first], verbose=False)
    assert [(flight["arr_airport"], flight["id"]) for flight in flights] == [
        ("KAAA", "A"), ("KBBB", "B")
    ]


def test_ts_load_filters_qualified_keys_before_opening_source_tracks(tmp_path):
    manifest_path = _write_arrival_manifest(tmp_path, ["A", "B"], airport="KRDU")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_key = manifest["records"][0]["flight_key"]
    excluded = manifest["records"][1]
    (tmp_path / "tracks" / excluded["source_file"]).write_text(
        "excluded source track must stay closed", encoding="utf-8"
    )

    flights = dataset_module.load_flight_dicts(
        manifest_path,
        include_flight_keys={f"KRDU:{selected_key}"},
        verbose=False,
    )

    assert [flight["id"] for flight in flights] == ["A"]


def test_manifest_split_keys_are_resolved_without_loading_trajectory_values():
    config = TSConfig(seed=1337)
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": "KRDU",
            "source_records": [
                {"flight_key": f"flight-{index}", "source_sha256": f"{index:064x}"}
                for index in range(20)
            ],
        }],
    }

    resolved = dataset_module.flight_keys_by_split(provenance, config)

    assert set(resolved) == {"train", "val", "test"}
    assert sum(map(len, resolved.values())) == 20
    assert all(
        split_name_for_dataset_id(key, config) == split
        for split, keys in resolved.items()
        for key in keys
    )


def test_batch_probe_opens_outer_train_track_files_only(tmp_path):
    manifest_path = _write_arrival_manifest(
        tmp_path, [f"F{index:02d}" for index in range(40)]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = TSConfig(seed=1337)
    counts = {"train": 0, "val": 0, "test": 0}
    for row in manifest["records"]:
        split = split_name_for_dataset_id(f"KRDU:{row['flight_key']}", config)
        counts[split] += 1
        if split != "train":
            # A filtered loader would fail JSON/SHA validation if it opened this test/val file.
            source_path = tmp_path / "tracks" / row["source_file"]
            source_path.write_text("not opened by the batch probe", encoding="utf-8")

    flights, audit = batch_probe.load_outer_train_flights([manifest_path], config)
    assert len(flights) == counts["train"]
    assert audit["split_counts_from_manifest_rosters"] == counts
    assert audit["loaded_source_tracks"] == {
        "train": counts["train"], "validation": 0, "test": 0,
    }


def test_ts_load_rejects_duplicate_manifest_identity(tmp_path):
    from dataset import load_flight_dicts

    manifest_path = _write_arrival_manifest(tmp_path, ["A"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"].append(dict(manifest["records"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate flight_key"):
        load_flight_dicts(manifest_path, verbose=False)


def test_ts_load_rejects_legacy_json_input(tmp_path):
    from dataset import load_flight_dicts

    legacy = tmp_path / "KRDU_05L_landings.json"
    legacy.write_text(json.dumps([{"id": "OLD"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="arrival manifest"):
        load_flight_dicts(legacy, verbose=False)


def test_flight_identity_separates_the_same_callsign_on_different_runways():
    # Regression: FlightSeries.flight_id was flight["id"] (the callsign). Across a
    # multi-runway harvest that collides, and the collision is invisible until you notice
    # `predict --split test` returning 48 flights for an 18-flight split — because every
    # namesake on every runway matched. It also leaks the split: three copies of one id
    # land in train, val and test at once.
    from dataset import flight_key

    series, _ = _series(n_flights=6)
    ids = [s.flight_id for s in series]
    assert len(ids) == len(set(ids))

    same_callsign_other_runway = flight_key(
        {"id": "AAL1", "runway": "23R", "icao24": "a1b2c3",
         "landing_time_utc": "2026-06-18T21:37:36Z"}, 0)
    assert same_callsign_other_runway != flight_key(
        {"id": "AAL1", "runway": "05L", "icao24": "a1b2c3",
         "landing_time_utc": "2026-06-18T21:37:36Z"}, 0)


def test_split_membership_selects_exactly_the_flights_it_names():
    # The consumer of the leak above: predict --split test filters by flight_id, so the
    # filter must return exactly as many series as the split names.
    series, config = _series(n_flights=12)
    train_group, val_group, test_group = split_by_flight(series, config)
    wanted = {s.dataset_id for s in test_group}
    assert len([s for s in series if s.dataset_id in wanted]) == len(test_group)


def test_dataset_identity_qualifies_flight_key_with_airport():
    series, _ = _series(n_flights=1)
    assert series[0].dataset_id == f"{AIRPORT}:{series[0].flight_id}"


def test_cross_validation_folds_are_disjoint_and_cover_outer_train():
    series, config = _series(n_flights=30)
    outer_train, _outer_val, _outer_test = split_by_flight(series, config)
    folds = cross_validation_folds(outer_train, 3, seed=config.seed)
    identities = [item.dataset_id for fold in folds for item in fold]
    assert len(identities) == len(set(identities)) == len(outer_train)
    assert set(identities) == {item.dataset_id for item in outer_train}


def test_fixed_anchor_dataset_keeps_one_window_per_flight():
    series, config = _series(n_flights=4)
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    assert len(dataset) == len(series)
    assert all(anchor == config.seq_len - 1 for _series_index, anchor in dataset.index)


def test_vectorized_batch_matches_individual_random_anchor_samples():
    series, config = _series(n_flights=3, n_segments=16)
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    indices = np.array([0, len(dataset) // 2, len(dataset) - 1])

    batch = dataset.batch(indices)
    individual = [dataset[int(index)] for index in indices]
    for field, values in enumerate(batch):
        assert torch.equal(values, torch.stack([sample[field] for sample in individual]))


def test_common_anchor_is_independent_of_history_length():
    series, base = _series(n_flights=2)
    normalizer = Normalizer.fit(series)
    common_anchor = 89
    datasets = [
        FixedAnchorTrajectoryWindows(
            series,
            replace(base, seq_len=seq_len),
            normalizer,
            minimum_anchor_index=common_anchor,
        )
        for seq_len in (30, 60, 90)
    ]

    assert all(dataset.index[0] == (0, common_anchor) for dataset in datasets)
    samples = [dataset[0] for dataset in datasets]
    assert [sample[0].shape[0] for sample in samples] == [30, 60, 90]
    assert all(torch.equal(samples[0][1], sample[1]) for sample in samples[1:])
    assert all(float(samples[0][3]) == pytest.approx(float(sample[3])) for sample in samples[1:])


def test_history_ablation_runs_on_common_outer_train_anchors(tmp_path):
    series, config = _series(
        n_flights=20,
        device="cpu",
        seq_len=20,
        n_segments=8,
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        batch_size=32,
        epochs=1,
        patience=1,
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }

    result = history_ablation.run_history_ablation(
        series,
        config,
        seq_lens=(10, 20),
        data_provenance=provenance,
        output_dir=tmp_path,
        n_splits=2,
        epochs=1,
        patience=1,
        auto_batch_size=False,
        verbose=False,
    )

    assert result["selected_seq_len"] in (10, 20)
    assert result["common_anchor"]["index"] == 19
    assert result["leakage_guard"]["outer_test_used"] is False
    signatures = [
        history_ablation.anchor_signature(
            split_by_flight(series, replace(config, seq_len=20))[0],
            replace(config, seq_len=seq_len),
            19,
        )
        for seq_len in (10, 20)
    ]
    assert signatures[0] == signatures[1]
    for name in (
        "history_length_ablation.json",
        "best_history_length.json",
        "history_length_candidates.csv",
        "history_length_folds.csv",
        "plots/history_length_cv_loss.png",
        "plots/history_length_metrics.svg",
        "plots/index.md",
    ):
        assert (tmp_path / name).is_file(), name


def test_fixed_epoch_sampler_uses_every_flight_once_and_reshuffles():
    series, config = _series(n_flights=8)
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    first = list(FlightEpochSampler(dataset, seed=7))
    repeated = list(FlightEpochSampler(dataset, seed=7))
    reshuffled = list(FlightEpochSampler(dataset, seed=8))

    assert first == repeated
    assert first != reshuffled
    assert len(first) == len(series)
    assert {dataset.index[index][0] for index in first} == set(range(len(series)))


def test_random_epoch_sampler_selects_one_valid_anchor_per_flight():
    series, config = _series(n_flights=8)
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    first = list(FlightEpochSampler(dataset, seed=7))
    second = list(FlightEpochSampler(dataset, seed=8))

    for epoch in (first, second):
        assert len(epoch) == len(series)
        assert {dataset.index[index][0] for index in epoch} == set(range(len(series)))
    assert {dataset.index[index] for index in first} != {
        dataset.index[index] for index in second
    }


def test_random_anchor_requires_sixty_seconds_of_future_by_default():
    series, config = _series(n_flights=4)
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))

    assert dataset.minimum_future_s == pytest.approx(60.0)
    assert dataset.index
    for series_index, anchor in dataset.index:
        remaining = (
            series[series_index].supervision_times[-1]
            - series[series_index].times[anchor]
        )
        assert remaining >= 60.0 - 1e-9


def test_random_anchor_choice_is_stable_per_flight_when_roster_order_changes():
    series, config = _series(n_flights=8)

    def selected(group):
        dataset = RandomAnchorTrajectoryWindows(group, config, Normalizer.fit(group))
        return {
            dataset.series[dataset.index[index][0]].dataset_id: dataset.index[index][1]
            for index in dataset.epoch_indices(seed=17)
        }

    assert selected(series) == selected(list(reversed(series)))


def test_control_random_anchor_candidates_require_airborne_stall_margin():
    config = TSConfig(prediction_output=PREDICTION_CONTROL)
    stall_speed = math.sqrt(
        2.0 * 60_000.0 * 9.81 / (1.225 * 100.0 * 2.0)
    )
    values = np.zeros((4, len(ch.CHANNELS)))
    values[:, ch.IDX["edot"]] = [
        0.0, stall_speed, 1.11 * stall_speed, 2.0 * stall_speed,
    ]
    series = SimpleNamespace(
        values=values,
        scenario=SimpleNamespace(
            initial=SimpleNamespace(m=60_000.0),
            aero=SimpleNamespace(S=100.0, Cl_max=2.0),
        ),
    )

    assert CONTROL_ANCHOR_STALL_MARGIN == pytest.approx(1.10)
    assert eligible_random_train_anchors(series, range(4), config) == [2, 3]
    assert eligible_random_train_anchors(
        series, range(4), replace(config, prediction_output="state")
    ) == [0, 1, 2, 3]


def test_training_cohort_floor_filters_only_the_supplied_train_roster():
    config = TSConfig(seq_len=3, training_cohort_min_future_s=60.0)
    short = SimpleNamespace(
        dataset_id="KAAA:SHORT",
        times=np.array([0.0, 2.0, 4.0]),
        supervision_times=np.array([0.0, 2.0, 4.0, 54.0]),
    )
    eligible = SimpleNamespace(
        dataset_id="KAAA:ELIGIBLE",
        times=np.array([0.0, 2.0, 4.0]),
        supervision_times=np.array([0.0, 2.0, 4.0, 64.0]),
    )

    retained, audit = train_module.filter_training_cohort(
        [short, eligible], config, verbose=False
    )

    assert retained == [eligible]
    assert audit == {
        "scope": "train only after by-flight split",
        "anchor": "fixed L-1",
        "minimum_future_s": 60.0,
        "input_flights": 2,
        "retained_flights": 1,
        "excluded_flights": 1,
        "excluded": [{
            "dataset_id": "KAAA:SHORT",
            "fixed_anchor_remaining_s": 50.0,
        }],
    }


def test_common_grid_checkpoint_selection_still_validates_fixed_anchor():
    series, config = _series(
        n_flights=12,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_COMMON_GRID_ADE,
        epochs=1,
        patience=1,
        batch_size=32,
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        seq_len=20,
        n_segments=8,
        device="cpu",
    )
    train_series, val_series, _test_series = split_by_flight(series, config)
    fit = train_module.fit_model(
        train_series, val_series, config, verbose=False
    )

    assert fit.best_validation_selection > 0.0
    assert fit.history[0].validation_selection_metric == (
        CHECKPOINT_SELECTION_COMMON_GRID_ADE
    )
    assert fit.history[0].validation_selection_value == pytest.approx(
        fit.best_validation_selection
    )
    assert set(fit.history[0].validation_selection_by_airport) == {AIRPORT}
    timing = fit.history[0].timing
    assert timing["epoch_total_s"] > 0.0
    assert timing["train_forward_s"] > 0.0
    assert timing["train_rollout_loss_s"] > 0.0
    assert timing["train_backward_step_s"] > 0.0
    assert timing["val_objective_s"] > 0.0
    assert timing["val_checkpoint_selection_s"] > 0.0
    assert timing["optimizer_updates_per_s"] > 0.0
    profile = fit.history[0].validation_profile_by_airport[AIRPORT]
    assert profile["flights"] == len(val_series)
    assert profile["query_points"] > 0
    assert sum(profile["duration_bucket_flights"].values()) == len(val_series)


def test_common_grid_checkpoint_selection_reuses_one_truth_cache(monkeypatch):
    series, config = _series(
        n_flights=12,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_COMMON_GRID_ADE,
        validation_common_grid_points=7,
        epochs=2,
        patience=2,
        batch_size=32,
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        seq_len=20,
        n_segments=8,
        device="cpu",
    )
    train_series, val_series, _test_series = split_by_flight(series, config)
    cached_truth_ids: list[int] = []
    original = train_module.fixed_anchor_common_grid_ade_metrics

    def record_cache(*args, **kwargs):
        cached_truth = kwargs.get("common_truth")
        assert cached_truth is not None
        cached_truth_ids.append(id(cached_truth))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        train_module,
        "fixed_anchor_common_grid_ade_metrics",
        record_cache,
    )
    train_module.fit_model(train_series, val_series, config, verbose=False)

    assert len(cached_truth_ids) == config.epochs
    assert len(set(cached_truth_ids)) == 1


def test_shared_validation_forward_matches_two_pass_control_metrics(monkeypatch):
    series, config = _series(
        n_flights=4,
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        seq_len=8,
        n_segments=2,
        batch_size=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        device="cpu",
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    model = build_model(config).eval()
    legacy_components = train_module._dataset_loss_components(
        model, dataset, torch.device("cpu"), config.batch_size
    )
    legacy_common = train_module.evaluate_fixed_anchor_common_grid(
        model, dataset, normalizer, config, torch.device("cpu")
    )

    calls = 0
    original_forward = train_module.model_forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(train_module, "model_forward", counted_forward)
    plan = train_module.build_validation_batch_plan(dataset, config.batch_size)
    shared = train_module._evaluate_validation_airport(
        model,
        plan,
        torch.device("cpu"),
        None,
        include_deployable_replay=True,
    )
    shared_common = train_module.evaluate_fixed_anchor_common_grid(
        model,
        dataset,
        normalizer,
        config,
        torch.device("cpu"),
        replay=shared.replay,
    )

    assert calls == len(plan.batches)
    assert shared.components == pytest.approx(legacy_components, rel=1e-6, abs=1e-7)
    for key in (
        "ade_m",
        "fde_m",
        "final_time_mae_s",
        "terminal_velocity_error_mps",
        "arc_length_geometry_loss",
    ):
        assert shared_common[key] == pytest.approx(
            legacy_common[key], rel=1e-6, abs=1e-6
        )


def test_control_validation_replay_uses_dense_dynamics_queries():
    series, config = _series(
        n_flights=1,
        prediction_output=PREDICTION_CONTROL,
        seq_len=8,
        n_segments=2,
        validation_common_grid_points=5,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    raw_batch = next(dataset_module.iter_batches(dataset, 1, shuffle=False, seed=0))
    x, y, mask, final_time_s, _weights, dynamics, _dense = (
        train_module.unpack_batch(raw_batch)
    )
    assert dynamics is not None
    midpoint = 0.5 * (dynamics["control_lower"] + dynamics["control_upper"])
    prediction = ControlPrediction(
        controls=midpoint[:, None, :].expand(-1, 2, -1).contiguous(),
        segment_durations=torch.tensor([[0.75, 1.25]], dtype=torch.float32),
        final_time_s=torch.tensor([2.0], dtype=torch.float32),
    )

    replay = train_module._prediction_batch_replay(
        prediction, x, y, mask, final_time_s, dynamics, dataset
    )

    assert replay.predicted.shape == (1, 5, config.enc_in)
    assert replay.truth.shape == replay.predicted.shape
    assert replay.segment_durations_s.shape == (1, 5)
    np.testing.assert_allclose(replay.segment_durations_s, 0.4)
    assert replay.predicted_time_s.tolist() == pytest.approx([2.0])


def test_fixed_anchor_cache_is_bitwise_identical_to_uncached_builders():
    series, config = _series(
        n_flights=3,
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        seq_len=8,
        n_segments=2,
    )
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))

    for index in range(len(dataset)):
        cached = dataset._sample_arrays(index)
        uncached = dataset_module.TrajectoryWindows._sample_arrays(dataset, index)
        for cached_array, uncached_array in zip(cached, uncached):
            assert np.array_equal(cached_array, uncached_array)
        cached_dynamics = dataset._dynamics_arrays(index)
        uncached_dynamics = dataset_module.TrajectoryWindows._dynamics_arrays(
            dataset, index
        )
        for key in cached_dynamics:
            assert np.array_equal(cached_dynamics[key], uncached_dynamics[key])

    indices = np.arange(len(dataset), dtype=np.int64)
    cached_dense = dataset._fixed_dt_supervision(indices)
    uncached_dense = build_fixed_dt_supervision(
        dataset.series,
        dataset.encoded,
        dataset.index,
        dt_s=config.dt_s,
    )
    for field in ("query_offsets_s", "states", "weights", "valid"):
        assert torch.equal(getattr(cached_dense, field), getattr(uncached_dense, field))


def test_fit_evaluation_reuses_one_prediction_pass_per_split(monkeypatch):
    series, config = _series(
        n_flights=12,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        n_segments=8,
        device="cpu",
    )
    train_series, val_series, _test_series = split_by_flight(series, config)
    normalizer = Normalizer.fit(train_series)
    model = build_model(config)
    calls = 0
    original_forward = train_module.model_forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(train_module, "model_forward", counted_forward)
    evaluate_fit_splits(
        model, train_series, val_series, normalizer, config, torch.device("cpu")
    )

    assert calls == 2


def test_training_saves_checkpoint_before_derived_fit_replay(tmp_path, monkeypatch):
    series, config = _series(
        n_flights=12,
        epochs=1,
        patience=1,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        n_segments=8,
        device="cpu",
    )

    def fail_replay(*_args, **_kwargs):
        raise RuntimeError("derived report failed")

    monkeypatch.setattr(train_module, "evaluate_fit_splits", fail_replay)
    with pytest.raises(RuntimeError, match="derived report failed"):
        train(
            series,
            config,
            output_dir=tmp_path,
            data_provenance=_fake_data_provenance(),
            verbose=False,
        )

    assert (tmp_path / "checkpoint.pt").is_file()
    assert not (tmp_path / FIT_EVALUATION_NAME).exists()


def test_flight_loss_weights_give_every_airport_equal_epoch_weight():
    series, config = _series(n_flights=8)
    series[0].scenario.source["arr_airport"] = "KAAA"
    for item in series[1:]:
        item.scenario.source["arr_airport"] = "KBBB"
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))

    totals = {"KAAA": 0.0, "KBBB": 0.0}
    for index in FlightEpochSampler(dataset, seed=7):
        series_index, _anchor = dataset.index[index]
        totals[series[series_index].airport] += float(dataset[index][4])

    assert totals["KAAA"] == pytest.approx(totals["KBBB"])
    assert sum(totals.values()) == pytest.approx(len(series))


def test_normalizer_round_trips_and_survives_a_constant_channel():
    series, _ = _series(n_flights=4)
    normalizer = Normalizer.fit(series)
    values = series[0].values
    assert np.allclose(normalizer.decode(normalizer.encode(values)), values)

    # A channel with zero variance must not divide by zero.
    flat = np.zeros((10, len(ch.CHANNELS)))
    constant = Normalizer.fit([type(series[0])(
        flight_id="x", scenario=series[0].scenario, frame=series[0].frame,
        times=np.arange(10.0), values=flat)])
    assert np.all(np.isfinite(constant.encode(flat)))


def test_balanced_normalizer_weights_airports_then_flights_equally():
    series, _ = _series(n_flights=3)
    series[0].scenario.source["arr_airport"] = "KAAA"
    series[0].values[:] = 0.0
    for value, item in zip((10.0, 20.0), series[1:]):
        item.scenario.source["arr_airport"] = "KBBB"
        item.values[:] = value
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    assert np.allclose(normalizer.mean, 7.5)


# ── Models ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_both_models_return_structured_states_and_final_time(model_name):
    config = TSConfig(model=model_name, seq_len=32, n_segments=16, d_model=32, n_heads=4,
                      d_ff=64, e_layers=1)
    model = build_model(config)
    x = torch.randn(3, config.seq_len, config.enc_in)
    prediction = model(x)
    assert prediction.states.shape == (3, config.n_segments, config.enc_in)
    assert prediction.final_time_s.shape == (3,)
    assert torch.all(prediction.final_time_s > 0.0)


def test_control_head_bounds_controls_and_partitions_time_non_uniformly():
    bounds = ControlBounds(
        lower=(0.0, -math.pi / 4, 0.5),
        upper=(120_000.0, math.pi / 4, 2.0),
    )
    head = ControlOutputHead(input_dim=8, n_segments=5, bounds=bounds)
    result = head(torch.randn(3, 8), torch.tensor([100.0, 240.0, 60.0]))

    assert result.controls.shape == (3, 5, 3)
    assert result.segment_durations.shape == (3, 5)
    assert torch.all(result.segment_durations > 0.0)
    assert torch.allclose(result.segment_durations.sum(dim=-1), result.final_time_s)
    assert torch.all(result.controls >= head.lower)
    assert torch.all(result.controls <= head.upper)


def test_control_head_reserves_uniform_duration_mass_to_prevent_partition_collapse():
    head = ControlOutputHead(
        input_dim=1,
        n_segments=5,
        bounds=ControlBounds(
            lower=(0.0, -math.pi / 4, 0.5),
            upper=(120_000.0, math.pi / 4, 2.0),
        ),
        duration_uniform_floor=0.8,
    )
    with torch.no_grad():
        head.duration_projection.weight.zero_()
        head.duration_projection.bias.copy_(
            torch.tensor([100.0, -100.0, -100.0, -100.0, -100.0])
        )

    result = head(torch.zeros(1, 1), torch.tensor([100.0]))
    fractions = result.segment_durations[0] / result.final_time_s[0]

    # 80% of the duration is reserved uniformly. The remaining 20% stays learnable,
    # so even adversarial logits cannot recreate the historical ~95% single segment.
    fractions = fractions.detach()
    assert fractions.min().item() >= 0.8 / 5.0 - 1e-6
    assert fractions.max().item() <= 0.2 + 0.8 / 5.0 + 1e-6
    assert fractions.sum().item() == pytest.approx(1.0)


def test_uniform_duration_control_head_has_no_duration_parameters():
    head = UniformDurationControlHead(input_dim=8, n_segments=4)
    lower = torch.tensor([[0.0, -0.7, 0.5], [0.0, -0.5, 0.6]])
    upper = torch.tensor([[200_000.0, 0.7, 2.0], [150_000.0, 0.5, 1.8]])
    final_time = torch.tensor([80.0, 100.0], requires_grad=True)

    prediction = head(
        torch.randn(2, 8), final_time, lower=lower, upper=upper
    )

    assert not any("duration_projection" in name for name, _ in head.named_parameters())
    torch.testing.assert_close(
        prediction.segment_durations,
        torch.tensor([[20.0] * 4, [25.0] * 4]),
    )
    torch.testing.assert_close(
        prediction.segment_durations.sum(dim=1), prediction.final_time_s
    )
    assert torch.all(prediction.controls >= lower[:, None, :])
    assert torch.all(prediction.controls <= upper[:, None, :])


def test_control_simple_v1_is_a_frozen_serialized_recipe():
    config = TSConfig(
        control_recipe_name=CONTROL_RECIPE_SIMPLE_V1,
        **control_simple_v1_overrides(),
    )

    assert config.control_duration_parameterization == CONTROL_DURATION_UNIFORM
    assert config.control_state_objective == CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION
    assert config.checkpoint_selection_metric == CHECKPOINT_SELECTION_COMMON_GRID_ADE
    assert control_recipe(config)["name"] == CONTROL_RECIPE_SIMPLE_V1
    assert TSConfig.from_dict(config.to_dict()) == config
    assert "bounded-control-uniform-duration" in train_module.target_contract(config)
    with pytest.raises(ValueError, match="simple-v1 recipe fields are frozen"):
        replace(config, n_segments=32)


def test_control_simple_v1_cli_applies_defaults_and_rejects_conflicts():
    parser = argparse.ArgumentParser()
    ts_cli._add_data_args(parser)
    ts_cli._add_training_args(parser)
    args = parser.parse_args(
        [
            "--data", "unused.json",
            "--output-dir", "unused-output",
            "--control-recipe", CONTROL_RECIPE_SIMPLE_V1,
            "--seed", "2027",
        ]
    )

    config, batch_auto = ts_cli._config_from_args(args, parser)

    assert not batch_auto
    assert config.control_recipe_name == CONTROL_RECIPE_SIMPLE_V1
    assert config.seed == 2027
    assert config.n_segments == 64
    assert config.batch_size == 512
    assert config.control_duration_parameterization == CONTROL_DURATION_UNIFORM

    conflicting = parser.parse_args(
        [
            "--data", "unused.json",
            "--output-dir", "unused-output",
            "--control-recipe", CONTROL_RECIPE_SIMPLE_V1,
            "--n-segments", "32",
        ]
    )
    with pytest.raises(SystemExit):
        ts_cli._config_from_args(conflicting, parser)


def test_cached_teacher_pretrainer_accepts_native_control_batch(tmp_path):
    series, config = _series(
        n_flights=2,
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_duration_gradient=False,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    batch = dataset.batch(np.arange(len(dataset)))
    assert len(batch) == 6
    final_time = batch[3]
    dynamics = batch[5]
    controls = (
        dynamics["control_lower"][:, None, :]
        + dynamics["control_upper"][:, None, :]
    ).expand(-1, config.n_segments, -1) / 2.0
    durations = final_time[:, None].expand(-1, config.n_segments) / config.n_segments
    schedule_path = tmp_path / "teacher_schedules.npz"
    np.savez_compressed(
        schedule_path,
        dataset_ids=np.asarray([item.dataset_id for item in series]),
        controls=controls.numpy(),
        segment_durations_s=durations.numpy(),
    )

    audit = CachedSchedulePretrainer(
        schedule_path=schedule_path,
        steps=1,
        log_every=1,
    )(
        build_model(config),
        series,
        normalizer,
        config,
        torch.device("cpu"),
    )

    assert audit["dataset_ids"] == [item.dataset_id for item in series]
    assert audit["steps"] == 1


def _write_simple_v1_teacher_schedule(path, *, flights=32):
    np.savez_compressed(
        path,
        dataset_ids=np.asarray([f"KSJC:teacher-{index}" for index in range(flights)]),
        controls=np.zeros((flights, 64, 3), dtype=np.float32),
        segment_durations_s=np.ones((flights, 64), dtype=np.float32),
    )


def test_simple_v1_teacher_rejects_optimizer_contract_drift(tmp_path, monkeypatch):
    schedule_path = tmp_path / "teacher_schedules.npz"
    _write_simple_v1_teacher_schedule(schedule_path)
    monkeypatch.setattr(
        teacher_pretraining,
        "SIMPLE_V1_TEACHER_SCHEDULE_SHA256",
        hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="1000 steps"):
        CachedSchedulePretrainer(
            schedule_path=schedule_path,
            steps=1,
            recipe_name=CONTROL_RECIPE_SIMPLE_V1,
        )


def test_simple_v1_teacher_rejects_schedule_hash_drift(tmp_path):
    schedule_path = tmp_path / "teacher_schedules.npz"
    _write_simple_v1_teacher_schedule(schedule_path)

    with pytest.raises(ValueError, match="frozen schedule SHA-256"):
        CachedSchedulePretrainer(
            schedule_path=schedule_path,
            recipe_name=CONTROL_RECIPE_SIMPLE_V1,
        )


def test_simple_v1_teacher_rejects_wrong_cohort_size(tmp_path, monkeypatch):
    schedule_path = tmp_path / "teacher_schedules.npz"
    _write_simple_v1_teacher_schedule(schedule_path, flights=1)
    monkeypatch.setattr(
        teacher_pretraining,
        "SIMPLE_V1_TEACHER_SCHEDULE_SHA256",
        hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="exactly 32 schedules"):
        CachedSchedulePretrainer(
            schedule_path=schedule_path,
            recipe_name=CONTROL_RECIPE_SIMPLE_V1,
        )


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ((0.0,), (1.0,)),
        ((0.0, -1.0, 0.5), (1.0, 1.0)),
        ((0.0, -1.0, 0.5, 2.0), (1.0, 1.0, 2.0, 3.0)),
    ],
)
def test_control_bounds_require_one_bound_per_control(lower, upper):
    with pytest.raises(ValueError, match="exactly 3"):
        ControlBounds(lower=lower, upper=upper)


def test_control_output_is_parallel_and_requires_normalized_horizon():
    with pytest.raises(ValueError, match="requires horizon_mode='normalized'"):
        TSConfig(prediction_output=PREDICTION_CONTROL, horizon_mode=HORIZON_FULL)


@pytest.mark.parametrize(
    "backend",
    [
        CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
        CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    ],
)
def test_transport_chart_dynamics_is_an_explicit_control_only_contract(backend):
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_dynamics_backend=backend,
    )

    assert control_recipe(config)["dynamics_backend"] == backend
    assert f"+dynamics={backend}-v1" in train_module.target_contract(config)
    with pytest.raises(ValueError, match="requires a control prediction output"):
        TSConfig(control_dynamics_backend=backend)


def test_pipeline_carries_and_names_common_training_cohort():
    fixed = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="pooled",
        training_cohort_min_future_s=60.0,
    )
    random = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="pooled",
        random_train_anchor=True,
        training_cohort_min_future_s=60.0,
    )
    recipe = fixed._recipe_args()
    config, _source = fixed.resolved_train_config(use_best_config=False)
    prediction = pipeline_module.PredictionPlan(
        fixed, AIRPORT, ("eval",), split="val"
    )

    assert recipe[recipe.index("--training-cohort-min-future-s") + 1] == "60.0"
    assert config.training_cohort_min_future_s == pytest.approx(60.0)
    assert "cohort_min60" in fixed.train_dir.name
    assert "cohort_min60" in random.train_dir.name
    assert fixed.train_dir != random.train_dir
    assert "cohort_min60" in prediction.pred_dir.name
    assert "cohort_min60" in prediction.category


def test_control_state_supervision_clock_rejects_unknown_value():
    with pytest.raises(ValueError, match="control_state_supervision_clock"):
        TSConfig(control_state_supervision_clock="future")


def test_fixed_dt_control_state_loss_requires_observed_single_control_clock():
    with pytest.raises(ValueError, match="requires.*observed"):
        TSConfig(
            prediction_output=PREDICTION_CONTROL,
            control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        )
    with pytest.raises(ValueError, match="only by prediction_output='control'"):
        TSConfig(
            prediction_output=PREDICTION_STATE,
            control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
            control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        )


def test_uniform_control_durations_reject_non_control_outputs():
    with pytest.raises(ValueError, match="only by prediction_output='control'"):
        TSConfig(
            prediction_output=PREDICTION_STATE,
            control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        )


def test_legacy_control_config_without_duration_parameterization_is_rejected():
    serialized = TSConfig(prediction_output=PREDICTION_CONTROL).to_dict()
    serialized.pop("control_duration_parameterization")

    with pytest.raises(
        ValueError,
        match="missing control_duration_parameterization.*regenerate",
    ):
        TSConfig.from_dict(serialized)


def test_legacy_control_config_without_state_loss_grid_is_rejected():
    serialized = TSConfig(prediction_output=PREDICTION_CONTROL).to_dict()
    serialized.pop("control_state_loss_grid")

    with pytest.raises(ValueError, match="missing control_state_loss_grid.*regenerate"):
        TSConfig.from_dict(serialized)


@pytest.mark.parametrize(
    "field",
    [
        "control_state_objective",
        "control_state_duration_gradient",
        "control_horizon_curriculum_s",
        "control_horizon_curriculum_stage_epochs",
        "control_gradient_clip_norm",
        "control_gradient_clip_policy",
        "control_dynamics_backend",
        "control_dense_state_loss_weight",
        "control_geometry_loss_weight",
        "control_arc_horizontal_velocity_loss_weight",
        "control_arc_vertical_velocity_loss_weight",
        "control_arc_horizontal_velocity_scale_mps",
        "control_arc_vertical_velocity_scale_mps",
        "control_arc_local_velocity_parameterization",
        "control_arc_tangent_loss_weight",
        "control_arc_position_end_weight",
        "control_arc_terminal_parameterization",
        "control_arc_terminal_cross_track_emphasis",
        "control_arc_terminal_vertical_emphasis",
        "control_terminal_position_loss_weight",
        "control_terminal_velocity_loss_weight",
        "control_terminal_position_scale_m",
        "control_terminal_velocity_scale_mps",
        "control_terminal_supervision_clock",
        "control_duration_uniform_floor",
    ],
)
def test_legacy_control_config_without_physical_criteria_recipe_is_rejected(field):
    serialized = TSConfig(prediction_output=PREDICTION_CONTROL).to_dict()
    serialized.pop(field)

    with pytest.raises(ValueError, match=f"missing {field}.*regenerate"):
        TSConfig.from_dict(serialized)


def test_arc_length_geometry_objective_requires_aligned_selection_and_weights():
    common = {
        "prediction_output": PREDICTION_CONTROL,
        "control_state_supervision_clock": CONTROL_STATE_CLOCK_OBSERVED,
        "control_state_loss_grid": CONTROL_STATE_LOSS_GRID_FIXED_DT,
        "control_state_objective": CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
    }
    with pytest.raises(ValueError, match="fixed-anchor-arc-length-geometry"):
        TSConfig(**common)
    with pytest.raises(ValueError, match="velocity weight greater"):
        TSConfig(
            **common,
            checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
            control_geometry_loss_weight=1.0,
            control_terminal_position_loss_weight=2.0,
            control_terminal_velocity_loss_weight=1.0,
        )
    with pytest.raises(ValueError, match="local velocity weights"):
        TSConfig(
            **common,
            checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
            control_arc_horizontal_velocity_loss_weight=1.0,
        )
    with pytest.raises(
        ValueError, match="control_arc_vertical_velocity_scale_mps"
    ):
        TSConfig(
            **common,
            checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
            control_arc_vertical_velocity_scale_mps=0.0,
        )
    with pytest.raises(ValueError, match="arc-length-geometry requires n_segments >= 2"):
        TSConfig(
            **common,
            checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
            n_segments=1,
        )
    assert TSConfig(n_segments=1).n_segments == 1

    config = TSConfig(
        **common,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
    )

    assert config.control_terminal_position_loss_weight > config.control_geometry_loss_weight
    assert config.control_terminal_velocity_loss_weight > config.control_geometry_loss_weight
    assert control_recipe(config)["geometry_loss_weight"] == pytest.approx(0.75)
    assert control_recipe(config)[
        "arc_horizontal_velocity_loss_weight"
    ] == pytest.approx(0.25)
    assert control_recipe(config)["arc_vertical_velocity_scale_mps"] == pytest.approx(2.0)
    assert train_module.loss_component_names(config)[-3:] == (
        "terminal_velocity",
        "arc_horizontal_velocity",
        "arc_vertical_velocity",
    )
    assert "arc-length-geometry" in train_module.target_contract(config)


def test_predicted_terminal_clock_requires_dual_clock_physical_objective():
    common = {
        "prediction_output": PREDICTION_CONTROL,
        "control_terminal_supervision_clock": CONTROL_TERMINAL_CLOCK_PREDICTED,
    }
    with pytest.raises(ValueError, match="observed dense-state"):
        TSConfig(**common)
    with pytest.raises(ValueError, match="fixed-dt state loss"):
        TSConfig(
            **common,
            control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        )
    with pytest.raises(ValueError, match="requires the arc-length-geometry"):
        TSConfig(
            **common,
            control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
            control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        )
    config = TSConfig(
        **common,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
    )

    assert control_recipe(config)["terminal_supervision_clock"] == "predicted"
    assert "terminal-clock=predicted" in train_module.target_contract(config)


def test_arc_loss_ablation_components_share_one_objective_and_recipe():
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        control_arc_local_velocity_parameterization=(
            CONTROL_ARC_LOCAL_VELOCITY_TANGENT_SPEED
        ),
        control_arc_terminal_parameterization=CONTROL_ARC_TERMINAL_RUNWAY_COMPONENTS,
        control_arc_position_end_weight=4.0,
    )

    recipe = control_recipe(config)
    assert config.control_state_objective == CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY
    assert recipe["arc_local_velocity_parameterization"] == "tangent-speed"
    assert recipe["arc_terminal_parameterization"] == "runway-components"
    assert recipe["arc_position_end_weight"] == pytest.approx(4.0)
    assert train_module.loss_component_names(config)[-4:] == (
        "terminal_velocity",
        "arc_horizontal_tangent",
        "arc_horizontal_speed",
        "arc_vertical_velocity",
    )
    with pytest.raises(ValueError, match="control_arc_position_end_weight"):
        replace(config, control_arc_position_end_weight=0.5)


def test_control_horizon_curriculum_is_a_strict_physical_training_mode():
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        control_state_duration_gradient=False,
        control_horizon_curriculum_s=(60.0, 120.0, 240.0),
        control_horizon_curriculum_stage_epochs=10,
        epochs=31,
    )

    assert config.control_horizon_curriculum_s == (60.0, 120.0, 240.0)
    assert "horizon-curriculum=60,120,240s" in train_module.target_contract(config)
    assert "x10epochs" in train_module.target_contract(config)
    assert control_recipe(config)["horizon_curriculum_s"] == [60.0, 120.0, 240.0]

    with pytest.raises(ValueError, match="strictly increasing"):
        replace(config, control_horizon_curriculum_s=(120.0, 60.0))
    with pytest.raises(ValueError, match="align with the fixed-dt grid"):
        replace(config, control_horizon_curriculum_s=(61.0,))
    with pytest.raises(ValueError, match="leave at least one full-horizon epoch"):
        replace(config, epochs=30)
    with pytest.raises(ValueError, match="fixed train anchors"):
        replace(config, random_train_anchor=True)


def test_control_gradient_clip_is_explicit_and_control_only():
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_gradient_clip_norm=20.0,
    )

    assert control_recipe(config)["gradient_clip_norm"] == pytest.approx(20.0)
    assert control_recipe(config)["gradient_clip_policy"] == CONTROL_GRADIENT_CLIP_GLOBAL
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(config, control_gradient_clip_norm=-1.0)
    with pytest.raises(ValueError, match="only by prediction_output='control'"):
        TSConfig(control_gradient_clip_norm=20.0)
    with pytest.raises(ValueError, match="requires a positive clip norm"):
        TSConfig(
            prediction_output=PREDICTION_CONTROL,
            control_gradient_clip_policy=CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED,
        )


def test_final_time_decoupled_clip_requires_isolated_clock_gradients():
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_FACTORIZED,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_duration_gradient=False,
        control_gradient_clip_norm=20.0,
        control_gradient_clip_policy=CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED,
    )

    assert config.control_gradient_clip_policy == CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED
    with pytest.raises(ValueError, match="requires factorized durations"):
        replace(config, control_duration_parameterization=CONTROL_DURATION_UNIFORM)
    with pytest.raises(ValueError, match="requires observed state clock"):
        replace(
            config,
            control_state_supervision_clock=CONTROL_STATE_CLOCK_PREDICTED,
            control_state_duration_gradient=True,
        )
    with pytest.raises(ValueError, match="requires detached state-duration gradients"):
        replace(config, control_state_duration_gradient=True)


def test_control_gradient_clip_records_preclip_module_norms_and_caps_global_norm():
    class GradientGroups(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_encoder = torch.nn.Linear(1, 1, bias=False)
            self.control_head = torch.nn.Linear(1, 1, bias=False)
            self.final_time_head = torch.nn.Linear(1, 1, bias=False)

    model = GradientGroups()
    x = torch.ones(1, 1)
    loss = (
        3.0 * model.feature_encoder(x)
        + 4.0 * model.control_head(x)
        + 0.0 * model.final_time_head(x)
    ).sum()
    loss.backward()

    preclip, clipped = clip_gradients_by_global_norm(model, 1.0)
    postclip = gradient_norms(model)

    assert clipped is True
    assert preclip == pytest.approx(
        {
            "backbone": 3.0,
            "control_head": 4.0,
            "final_time_head": 0.0,
            "total": 5.0,
        }
    )
    assert postclip["total"] == pytest.approx(1.0)


def test_final_time_decoupled_clip_caps_control_backbone_only():
    class GradientGroups(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_encoder = torch.nn.Linear(1, 1, bias=False)
            self.control_head = torch.nn.Linear(1, 1, bias=False)
            self.final_time_head = torch.nn.Linear(1, 1, bias=False)

    model = GradientGroups()
    x = torch.ones(1, 1)
    loss = (
        3.0 * model.feature_encoder(x)
        + 4.0 * model.control_head(x)
        + 12.0 * model.final_time_head(x)
    ).sum()
    loss.backward()

    preclip, scopes = clip_gradients_by_policy(
        model,
        max_norm=1.0,
        policy=CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED,
    )
    postclip = gradient_norms(model)

    assert preclip["total"] == pytest.approx(13.0)
    assert scopes["control_backbone"]["pre_clip_norm"] == pytest.approx(5.0)
    assert scopes["control_backbone"]["coefficient"] == pytest.approx(0.2)
    assert scopes["control_backbone"]["triggered"] is True
    assert scopes["control_backbone"]["groups"] == ["backbone", "control_head"]
    assert scopes["control_backbone"]["capped"] is True
    assert scopes["final_time_head"]["pre_clip_norm"] == pytest.approx(12.0)
    assert scopes["final_time_head"]["triggered"] is False
    assert scopes["final_time_head"]["capped"] is False
    assert postclip["backbone"] == pytest.approx(0.6)
    assert postclip["control_head"] == pytest.approx(0.8)
    assert postclip["final_time_head"] == pytest.approx(12.0)


def test_control_horizon_curriculum_builds_exact_batched_prefix_views():
    controls = torch.zeros(2, 4, 3)
    prediction = ControlPrediction(
        controls=controls,
        segment_durations=torch.tensor(
            [[25.0, 25.0, 25.0, 25.0], [10.0, 10.0, 10.0, 10.0]]
        ),
        final_time_s=torch.tensor([100.0, 40.0]),
    )
    offsets = torch.arange(2.0, 102.0, 2.0).repeat(2, 1)
    valid = torch.zeros(2, 50, dtype=torch.bool)
    valid[0, :50] = True
    valid[1, :20] = True
    states = torch.zeros(2, 50, len(ch.CHANNELS))
    states[0, :, 0] = torch.arange(50)
    states[1, :, 0] = 100.0 + torch.arange(50)
    supervision = FixedDTControlSupervision(
        query_offsets_s=offsets,
        states=states,
        weights=torch.ones_like(states),
        valid=valid,
    )
    terminal = torch.stack((states[0, 49], states[1, 19]))
    stage = ControlTrainingStage("60s", 60.0, 1, 10)

    view = build_control_training_stage_view(
        prediction,
        supervision,
        terminal,
        prediction.final_time_s,
        stage,
    )

    torch.testing.assert_close(
        view.prediction.segment_durations,
        torch.tensor([[25.0, 25.0, 10.0, 0.0], [10.0, 10.0, 10.0, 10.0]]),
    )
    torch.testing.assert_close(
        view.segment_valid,
        torch.tensor([[True, True, True, False], [True, True, True, True]]),
    )
    assert view.supervision.query_offsets_s.shape == (2, 30)
    assert view.supervision.valid.sum(dim=1).tolist() == [30, 20]
    assert view.terminal_target[:, 0].tolist() == pytest.approx([29.0, 119.0])


def test_control_horizon_curriculum_keeps_exact_float64_query_boundary():
    fractions = torch.softmax(torch.linspace(-1.0, 1.0, 64), dim=0)
    prediction = ControlPrediction(
        controls=torch.zeros(1, 64, 3),
        segment_durations=(fractions * 328.0).unsqueeze(0),
        final_time_s=torch.tensor([328.0]),
    )
    offsets = torch.arange(2.0, 62.0, 2.0, dtype=torch.float64).unsqueeze(0)
    states = torch.zeros(1, 30, len(ch.CHANNELS))
    supervision = FixedDTControlSupervision(
        query_offsets_s=offsets,
        states=states,
        weights=torch.ones_like(states),
        valid=torch.ones(1, 30, dtype=torch.bool),
    )

    view = build_control_training_stage_view(
        prediction,
        supervision,
        states[:, -1],
        prediction.final_time_s,
        ControlTrainingStage("60s", 60.0, 1, 10),
    )

    assert view.prediction.segment_durations.dtype == torch.float64
    torch.testing.assert_close(
        view.prediction.segment_durations.sum(dim=1),
        torch.tensor([60.0], dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )
    assert view.supervision.query_offsets_s[0, -1] <= (
        view.prediction.segment_durations.sum(dim=1)[0]
    )


def test_control_horizon_curriculum_repairs_float32_duration_total_before_boundary():
    prediction = ControlPrediction(
        controls=torch.zeros(1, 64, 3),
        segment_durations=torch.full((1, 64), 0.93749994, dtype=torch.float32),
        final_time_s=torch.tensor([60.0], dtype=torch.float32),
    )
    offsets = torch.arange(2.0, 62.0, 2.0, dtype=torch.float64).unsqueeze(0)
    states = torch.zeros(1, 30, len(ch.CHANNELS))
    supervision = FixedDTControlSupervision(
        query_offsets_s=offsets,
        states=states,
        weights=torch.ones_like(states),
        valid=torch.ones(1, 30, dtype=torch.bool),
    )

    assert prediction.segment_durations.to(torch.float64).sum() < 60.0
    view = build_control_training_stage_view(
        prediction,
        supervision,
        states[:, -1],
        prediction.final_time_s,
        ControlTrainingStage("60s", 60.0, 1, 10),
    )

    # Dense rollout validates against the last cumulative segment boundary, so protect that
    # exact path rather than a separately reduced sum.
    total = view.prediction.segment_durations.cumsum(dim=1)[0, -1]
    torch.testing.assert_close(
        total, torch.tensor(60.0, dtype=torch.float64), rtol=0.0, atol=1e-12
    )
    assert view.supervision.query_offsets_s[0, -1] <= total


def test_full_control_stage_repairs_float32_duration_total_before_boundary():
    torch.manual_seed(0)
    durations = torch.softmax(torch.randn(1, 64), dim=1) * 580.0
    prediction = ControlPrediction(
        controls=torch.zeros(1, 64, 3),
        segment_durations=durations,
        final_time_s=torch.tensor([580.0]),
    )
    states = torch.zeros(1, 1, len(ch.CHANNELS))
    supervision = FixedDTControlSupervision(
        query_offsets_s=torch.tensor([[580.0]], dtype=torch.float64),
        states=states,
        weights=torch.ones_like(states),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )

    assert prediction.segment_durations.to(torch.float64).sum() < 580.0
    view = build_control_training_stage_view(
        prediction,
        supervision,
        states[:, -1],
        prediction.final_time_s,
        ControlTrainingStage("full", None, 1, None),
    )

    total = view.prediction.segment_durations.cumsum(dim=1)[0, -1]
    torch.testing.assert_close(
        total, torch.tensor(580.0, dtype=torch.float64), rtol=0.0, atol=1e-12
    )
    assert view.supervision.query_offsets_s[0, -1] <= total


def test_fixed_dt_rollout_closes_duration_clock_without_training_stage(monkeypatch):
    """Report replay must be safe even when it calls the rollout seam directly."""
    torch.manual_seed(0)
    durations = torch.softmax(torch.randn(1, 64), dim=1) * 580.0
    prediction = ControlPrediction(
        controls=torch.zeros(1, 64, 3),
        segment_durations=durations,
        final_time_s=torch.tensor([580.0]),
    )
    states = torch.zeros(1, 1, len(ch.CHANNELS))
    supervision = FixedDTControlSupervision(
        query_offsets_s=torch.tensor([[580.0]], dtype=torch.float64),
        states=states,
        weights=torch.ones_like(states),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )

    class CapturingBackend:
        def dense_rollout(
            self,
            inputs,
            query_offsets_s,
            query_valid,
            config,
            *,
            segment_valid,
        ):
            total = inputs.segment_durations_s.cumsum(dim=1)[0, -1]
            torch.testing.assert_close(
                total,
                torch.tensor(580.0, dtype=torch.float64),
                rtol=0.0,
                atol=1e-12,
            )
            assert query_offsets_s[0, -1] <= total
            return SimpleNamespace(
                query_channels=torch.zeros(1, 1, len(ch.CHANNELS)),
                segment_end_channels=torch.zeros(1, 64, len(ch.CHANNELS)),
            )

    monkeypatch.setattr(
        control_rollout_module,
        "control_dynamics_backend",
        lambda config: CapturingBackend(),
    )
    dynamics = {
        "initial_state": torch.zeros(1, 7),
        "initial_controls": torch.zeros(1, 3),
        "aero_params": torch.zeros(1, 1),
        "frame_params": torch.zeros(1, 1),
        "max_thrust_n": torch.ones(1),
    }

    _queries, _endpoints, closed_durations = (
        fixed_dt_loss_module.fixed_dt_rollout_channels(
        prediction,
        supervision,
        dynamics,
        TSConfig(),
        )
    )
    torch.testing.assert_close(
        closed_durations.sum(dim=1),
        prediction.final_time_s.to(torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("control_arc_tangent_loss_weight", 0.75),
        ("learning_rate", 5e-4),
        ("d_model", 128),
    ],
)
def test_capacity_report_recipe_detects_every_config_difference(
    tmp_path, field, different
):
    first = {"config": {"notes": {}, field: 0.25}}
    second = {"config": {"notes": {"comment": "display only"}, field: different}}
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    assert capacity_report._recipe(first) != capacity_report._recipe(second)
    with pytest.raises(ValueError, match=field):
        capacity_report._load_results([first_path, second_path])


def test_capacity_report_masks_unsupervised_reference_velocity_placeholders():
    diagnostics = {
        "channel_names": ["e", "n", "u", "edot", "ndot", "udot"],
        "anchor_state": [0.0, 0.0, 0.0, 10.0, 0.0, -1.0],
        "fixed_dt": {
            "offset_s": [2.0, 4.0, 6.0],
            "reference_state": [
                [20.0, 0.0, -2.0, 10.0, 0.0, -1.0],
                [40.0, 0.0, -4.0, 999.0, 999.0, 999.0],
                [60.0, 0.0, -6.0, 999.0, 999.0, 999.0],
            ],
            "predicted_state": [
                [20.0, 0.0, -2.0, 10.0, 0.0, -1.0],
                [40.0, 0.0, -4.0, 10.0, 0.0, -1.0],
                [60.0, 0.0, -6.0, 10.0, 0.0, -1.0],
            ],
            "reference_fully_measured": [True, False, False],
        },
    }

    charts = capacity_report._chart_diagnostics(diagnostics)

    assert charts["reference_horizontal_speed_mps"] == [10.0, None, None]
    assert charts["reference_vertical_speed_mps"] == [-1.0, None, None]
    assert charts["reference_consistency_mps"] == [0.0, None, None]
    assert charts["reference_acceleration_mps2"] == [0.0, None, None]


def test_control_horizon_curriculum_schedule_reserves_full_training():
    stages = build_control_training_stages(
        (60.0, 120.0, 240.0), epochs_per_stage=10, total_epochs=35
    )

    assert [(stage.label, stage.start_epoch, stage.end_epoch) for stage in stages] == [
        ("60s", 1, 10),
        ("120s", 11, 20),
        ("240s", 21, 30),
        ("full", 31, None),
    ]


def test_control_horizon_curriculum_selects_checkpoint_only_after_full_stage():
    series, config = _series(
        n_flights=2,
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        control_state_duration_gradient=False,
        control_horizon_curriculum_s=(2.0,),
        control_horizon_curriculum_stage_epochs=1,
        control_gradient_clip_norm=20.0,
        epochs=2,
        patience=1,
        seq_len=8,
        n_segments=2,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
        batch_size=2,
        control_rollout_integrator_dt_s=2.0,
        device="cpu",
    )

    fit = train_module.fit_model(series[:1], series[1:], config, verbose=False)

    assert [row.training_stage["label"] for row in fit.history] == ["2s", "full"]
    assert fit.history[0].validation_selection_metric == "curriculum-prefix-objective"
    assert (
        fit.history[1].validation_selection_metric
        == CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY
    )
    assert fit.best_validation_selection == pytest.approx(
        fit.history[1].validation_selection_value
    )
    for row in fit.history:
        diagnostics = row.control_training_diagnostics
        assert diagnostics["clip"]["max_norm"] == pytest.approx(20.0)
        assert diagnostics["clip"]["batches"] == 1
        assert 0.0 <= diagnostics["control_saturation"]["overall_rate"] <= 1.0


def test_state_config_without_duration_parameterization_remains_loadable():
    serialized = TSConfig().to_dict()
    serialized.pop("control_duration_parameterization")

    restored = TSConfig.from_dict(serialized)

    assert restored.control_duration_parameterization == CONTROL_DURATION_FACTORIZED


def test_observed_control_state_clock_preserves_partition_and_uses_true_total():
    prediction = ControlPrediction(
        controls=torch.randn(2, 2, 3),
        segment_durations=torch.tensor([[1.0, 3.0], [3.0, 2.0]]),
        final_time_s=torch.tensor([4.0, 5.0]),
    )
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
    )

    supervised = train_module.control_state_supervision_prediction(
        prediction, torch.tensor([8.0, 10.0]), config
    )

    assert supervised.controls is prediction.controls
    torch.testing.assert_close(
        supervised.segment_durations,
        torch.tensor([[2.0, 6.0], [6.0, 4.0]]),
    )
    torch.testing.assert_close(supervised.final_time_s, torch.tensor([8.0, 10.0]))
    torch.testing.assert_close(prediction.final_time_s, torch.tensor([4.0, 5.0]))
    assert train_module.target_contract(config) == (
        "bounded-control-nonuniform-duration-casadi-rollout-observed-clock-aligned-v3"
        "+duration-uniform-floor=0.8-v1"
    )


def test_predicted_control_state_clock_preserves_original_training_behavior():
    prediction = ControlPrediction(
        controls=torch.zeros(1, 2, 3),
        segment_durations=torch.tensor([[1.0, 3.0]]),
        final_time_s=torch.tensor([4.0]),
    )
    config = TSConfig(prediction_output=PREDICTION_CONTROL)

    assert train_module.control_state_supervision_prediction(
        prediction, torch.tensor([8.0]), config
    ) is prediction


@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
@pytest.mark.parametrize(
    "duration_parameterization",
    [CONTROL_DURATION_FACTORIZED, CONTROL_DURATION_UNIFORM],
)
def test_control_models_use_per_sample_bounds_and_aircraft_condition(
    model_name, duration_parameterization
):
    config = TSConfig(
        model=model_name,
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=duration_parameterization,
        seq_len=32,
        n_segments=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=1,
    )
    model = build_model(config)
    lower = torch.tensor([[0.0, -0.5, 0.5], [0.0, -0.7, 0.6]])
    upper = torch.tensor([[10_000.0, 0.5, 1.8], [250_000.0, 0.7, 2.0]])
    dynamics = {
        "condition": torch.rand(2, len(dataset_module.DYNAMICS_CONDITION_NAMES)),
        "control_lower": lower,
        "control_upper": upper,
    }
    result = model(torch.randn(2, config.seq_len, config.enc_in), dynamics)

    assert isinstance(result, ControlPrediction)
    assert result.controls.shape == (2, 4, 3)
    assert torch.all(result.controls >= lower[:, None, :])
    assert torch.all(result.controls <= upper[:, None, :])
    assert torch.allclose(result.segment_durations.sum(dim=-1), result.final_time_s)


@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_control_models_preserve_ordered_channel_identity(model_name):
    torch.manual_seed(7)
    config = TSConfig(
        model=model_name,
        prediction_output=PREDICTION_CONTROL,
        seq_len=32,
        n_segments=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=1,
        dropout=0.0,
        fc_dropout=0.0,
        head_dropout=0.0,
    )
    model = build_model(config).eval()
    history = torch.randn(1, config.seq_len, config.enc_in)
    mirrored = history.clone()
    mirrored[:, :, [0, 1]] = history[:, :, [1, 0]]
    mirrored[:, :, [3, 4]] = history[:, :, [4, 3]]
    original_features = model.feature_encoder.encode_features(history)
    swapped_features = model.feature_encoder.encode_features(mirrored)

    assert original_features.shape == (1, config.enc_in * config.d_model)
    assert torch.max(torch.abs(original_features - swapped_features)) > 1e-4


def test_control_loss_aligns_truth_to_predicted_cumulative_clock(monkeypatch):
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        seq_len=2,
        n_segments=2,
        d_model=8,
        n_heads=2,
        d_ff=16,
        e_layers=1,
        terminal_loss_weight=0.0,
        control_effort_loss_weight=0.0,
        control_smoothness_loss_weight=0.0,
    )
    channel_count = config.enc_in
    # Truth is linear in physical time and was sampled on the uniform true clock [5, 10].
    target = torch.tensor(
        [[[5.0] * channel_count, [10.0] * channel_count]], dtype=torch.float32
    )
    weights = torch.full_like(target, 1.0 / channel_count)
    # The learned partition asks for endpoints at [1, 10]. A clock-aligned target is [1, 10],
    # so this exact rollout must have zero state loss.
    physical_rollout = torch.tensor(
        [[[1.0] * channel_count, [10.0] * channel_count]], dtype=torch.float64
    )
    monkeypatch.setattr(
        control_rollout_module,
        "rollout_control_endpoints",
        lambda _controls, _durations, _dynamics, _config: SimpleNamespace(
            channels=physical_rollout,
            geodetic_states=torch.zeros(1, 2, 7, dtype=torch.float64),
        ),
    )
    prediction = ControlPrediction(
        controls=torch.zeros(1, 2, 3),
        segment_durations=torch.tensor([[1.0, 9.0]]),
        final_time_s=torch.tensor([10.0]),
    )
    normalizer = Normalizer(
        mean=np.zeros(channel_count, dtype=np.float64),
        std=np.ones(channel_count, dtype=np.float64),
    )
    dynamics = {
        "control_lower": torch.tensor([CONTROL_LOWER], dtype=torch.float32),
        "control_upper": torch.tensor([CONTROL_UPPER], dtype=torch.float32),
    }

    components = train_module.prediction_loss_components(
        prediction,
        torch.zeros(1, channel_count),
        target,
        weights,
        torch.tensor([10.0]),
        torch.ones(1),
        config,
        normalizer,
        dynamics,
    )

    assert components.state.item() == pytest.approx(0.0, abs=1e-12)


def test_true_time_control_loss_is_physical_position_endpoint_and_time_only(monkeypatch):
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_duration_gradient=False,
        n_segments=2,
        position_loss_scale_m=10_000.0,
        final_time_scale_s=600.0,
        state_endpoint_loss_weight=0.25,
        control_effort_loss_weight=0.0,
        control_smoothness_loss_weight=0.0,
    )
    normalizer = Normalizer(
        mean=np.zeros(config.enc_in),
        std=np.array([100.0, 200.0, 10.0, 2.0, 3.0, 4.0]),
    )
    # Physical position errors are [100, 200, 10] m and [0, 0, 20] m. Deliberately huge
    # future velocity values prove that the minimal objective does not supervise velocity.
    physical_rollout = torch.zeros(1, 2, config.enc_in, dtype=torch.float64)
    physical_rollout[0, 0, list(ch.POSITION_IDX)] = torch.tensor(
        [100.0, 200.0, 10.0], dtype=torch.float64
    )
    physical_rollout[0, 1, ch.IDX["u"]] = 20.0
    physical_rollout[..., list(ch.VELOCITY_IDX)] = 1_000_000.0
    monkeypatch.setattr(
        control_rollout_module,
        "rollout_control_endpoints",
        lambda _controls, _durations, _dynamics, _config: SimpleNamespace(
            channels=physical_rollout,
            geodetic_states=torch.zeros(1, 2, 7, dtype=torch.float64),
        ),
    )
    prediction = ControlPrediction(
        controls=torch.zeros(1, 2, 3),
        segment_durations=torch.tensor([[5.0, 5.0]]),
        final_time_s=torch.tensor([10.0]),
    )
    target = torch.zeros(1, 2, config.enc_in)
    weights = torch.full_like(target, 1.0 / config.enc_in)
    dynamics = {
        "control_lower": torch.tensor([CONTROL_LOWER], dtype=torch.float32),
        "control_upper": torch.tensor([CONTROL_UPPER], dtype=torch.float32),
    }

    components = train_module.control_prediction_loss_components(
        prediction,
        torch.zeros(1, config.enc_in),
        target,
        weights,
        torch.tensor([8.0]),
        torch.ones(1),
        config,
        normalizer,
        dynamics,
    )

    expected_path = ((100.0**2 + 200.0**2 + 10.0**2) + 20.0**2) / 2.0
    assert float(components.state) == pytest.approx(expected_path / 10_000.0**2)
    assert float(components.terminal) == pytest.approx(
        0.25 * 20.0**2 / 10_000.0**2
    )
    assert float(components.final_time) == pytest.approx((2.0 / 600.0) ** 2)
    assert float(components.kinematic) == pytest.approx(0.0)
    assert float(components.extras["control_effort"]) == pytest.approx(0.0)
    assert float(components.extras["control_smoothness"]) == pytest.approx(0.0)


def test_control_model_starts_from_neutral_uniform_rollout():
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        seq_len=8,
        n_segments=4,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        final_time_scale_s=600.0,
    )
    model = build_model(config).eval()
    history = torch.randn(2, config.seq_len, config.enc_in)
    lower = torch.tensor([CONTROL_LOWER, CONTROL_LOWER], dtype=torch.float32)
    upper = torch.tensor([CONTROL_UPPER, CONTROL_UPPER], dtype=torch.float32)
    dynamics = {
        "condition": torch.randn(2, len(dataset_module.DYNAMICS_CONDITION_NAMES)),
        "control_lower": lower,
        "control_upper": upper,
    }

    prediction = model(history, dynamics)
    # The untrained head emits the neutral physical control, not a fixed fraction of
    # whatever the bounds happen to be: 20% thrust, wings level, load factor one. It is
    # NEAR-neutral rather than exactly neutral on purpose — the projection carries a
    # deliberate NEUTRAL_LOGIT_PERTURBATION so gradient reaches the backbone at step 0
    # (see test_the_control_head_passes_gradient_to_the_backbone_at_initialisation).
    expected = torch.tensor(control_models.NEUTRAL_CONTROLS).view(1, 1, 3)
    span = torch.tensor(CONTROL_UPPER - CONTROL_LOWER, dtype=prediction.controls.dtype)
    expected_time = math.log(2.0) * config.final_time_scale_s

    assert torch.all(
        (prediction.controls - expected).abs() < 0.03 * span.view(1, 1, 3)
    )
    # Same story for the duration head: softplus(0)*scale, moved by at most
    # softplus'(0) * NEUTRAL_LOGIT_PERTURBATION * scale = 6 s (measured 4.6 s).
    torch.testing.assert_close(
        prediction.final_time_s,
        torch.full_like(prediction.final_time_s, expected_time),
        rtol=0.02,
        atol=0.0,
    )
    # This config takes the FACTORIZED duration head, whose softmax was exactly uniform
    # only because its logits were identically zero. Seeded logits make it uniform to
    # within ~1 % instead (the 0.8 uniform floor caps how far the learned part can move
    # it). The uniform-duration head, which has no duration projection at all, is still
    # exact — asserted in test_uniform_duration_control_head_has_no_duration_parameters.
    torch.testing.assert_close(
        prediction.segment_durations,
        prediction.final_time_s[:, None].expand(-1, config.n_segments)
        / config.n_segments,
        rtol=0.02,
        atol=0.0,
    )


@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_fixed_dt_objective_trains_both_backbones_without_duration_state_gradient(
    model_name,
):
    series, config = _series(
        n_flights=2,
        model=model_name,
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        control_state_duration_gradient=False,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    x, target, weights, final_time, flight_weights, dynamics, dense = dataset.batch(
        np.array([0, 1])
    )
    model = build_model(config)
    loss = prediction_loss(
        model(x, dynamics),
        x[:, -1],
        target,
        weights,
        final_time,
        flight_weights,
        config,
        normalizer,
        dynamics,
        dense,
        ControlTrainingStage("2s", 2.0, 1, 1),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.control_head.duration_projection.weight.grad is None
    assert model.control_head.control_projection.weight.grad is not None
    assert model.final_time_head.network[-1].bias.grad is not None


@pytest.mark.parametrize(
    "dynamics_backend",
    [
        CONTROL_DYNAMICS_REANCHORED_RK4,
        CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
        CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    ],
)
@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_control_dataset_and_rollout_loss_form_one_differentiable_training_step(
    dynamics_backend, model_name,
):
    series, config = _series(
        n_flights=2,
        model=model_name,
        prediction_output=PREDICTION_CONTROL,
        control_dynamics_backend=dynamics_backend,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    batch = dataset.batch(np.array([0, 1]))
    x, target, weights, final_time, flight_weights, dynamics = batch
    model = build_model(config)
    prediction = model(x, dynamics)
    loss = prediction_loss(
        prediction,
        x[:, -1],
        target,
        weights,
        final_time,
        flight_weights,
        config,
        normalizer,
        dynamics,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(dynamics) == {
        "condition", "initial_state", "initial_controls", "aero_params",
        "control_lower", "control_upper", "max_thrust_n", "frame_params",
        "runway_heading_rad",
    }
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.parameters()
    )


def test_control_simple_loss_forms_one_real_dynamics_training_step():
    series, config = _series(
        n_flights=2,
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_dynamics_backend=CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_duration_gradient=False,
        control_effort_loss_weight=0.0,
        control_smoothness_loss_weight=0.0,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    x, target, weights, final_time, flight_weights, dynamics = dataset.batch(
        np.array([0, 1])
    )
    model = build_model(config)

    prediction = model(x, dynamics)
    loss = prediction_loss(
        prediction,
        x[:, -1],
        target,
        weights,
        final_time,
        flight_weights,
        config,
        normalizer,
        dynamics,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert not any(
        "duration_projection" in name for name, _ in model.named_parameters()
    )
    torch.testing.assert_close(
        prediction.segment_durations,
        prediction.final_time_s.unsqueeze(1).expand_as(
            prediction.segment_durations
        ) / config.n_segments,
    )
    assert model.control_head.control_projection.weight.grad is not None
    assert model.final_time_head.network[-1].weight.grad is not None


def test_pipeline_carries_and_names_transport_chart_dynamics():
    plan = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="pooled",
        prediction_output=PREDICTION_CONTROL,
        control_dynamics_backend=CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
    )
    recipe = plan._recipe_args()
    config, _source = plan.resolved_train_config(use_best_config=False)
    prediction = pipeline_module.PredictionPlan(
        plan, AIRPORT, ("eval",), split="val"
    )

    assert recipe[recipe.index("--control-dynamics-backend") + 1] == (
        CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY
    )
    assert config.control_dynamics_backend == (
        CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY
    )
    assert plan.train_dir.name.endswith("_tcv")
    assert "transport_chart_velocity" not in plan.train_dir.name
    assert "transport_chart_velocity" in prediction.category
    assert "transport-chart-velocity dynamics" in prediction.label


def test_pipeline_carries_and_names_scaled_transport_chart_dynamics():
    plan = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="pooled",
        prediction_output=PREDICTION_CONTROL,
        control_dynamics_backend=(
            CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY
        ),
    )
    recipe = plan._recipe_args()
    config, _source = plan.resolved_train_config(use_best_config=False)
    prediction = pipeline_module.PredictionPlan(
        plan, AIRPORT, ("eval",), split="val"
    )

    assert recipe[recipe.index("--control-dynamics-backend") + 1] == (
        CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY
    )
    assert config.control_dynamics_backend == (
        CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY
    )
    assert plan.train_dir.name.endswith("_stcv")
    assert "scaled_transport_chart_velocity" in prediction.category
    assert "scaled-transport-chart-velocity dynamics" in prediction.label


def test_transport_chart_prediction_directory_stays_within_component_limit():
    plan = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="pooled",
        prediction_output=PREDICTION_CONTROL,
        control_dynamics_backend=CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
        control_state_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        control_state_duration_gradient=False,
        control_horizon_curriculum_s=(60.0, 120.0, 240.0),
        control_horizon_curriculum_stage_epochs=1,
        control_gradient_clip_norm=20.0,
        control_gradient_clip_policy=CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED,
        aircraft_filter=AIRCRAFT_FILTER_OPENAP_DIRECT,
        coordinate_frame="runway-aligned",
    )
    prediction = pipeline_module.PredictionPlan(
        plan, AIRPORT, ("eval",), split="val"
    )

    assert len(plan.train_dir.name.encode("utf-8")) <= 255
    assert len(prediction.pred_dir.name.encode("utf-8")) <= 255
    assert "transport_chart_velocity" in prediction.category
    assert "transport-chart-velocity dynamics" in prediction.label


def test_terminal_clock_artifact_keys_are_compact_and_collision_free():
    modes = (
        (CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION, ""),
        (CONTROL_TERMINAL_CLOCK_PREDICTED, "_tcp"),
        (CONTROL_TERMINAL_CLOCK_PREDICTED_DETACHED_TIME, "_tcpdt"),
    )
    train_dirs = []
    prediction_dirs = []
    categories = []

    for terminal_clock, filesystem_tag in modes:
        plan = pipeline_module.TrainingPlan(
            (AIRPORT,),
            "itransformer",
            training_mode="pooled",
            prediction_output=PREDICTION_CONTROL,
            control_state_clock=CONTROL_STATE_CLOCK_OBSERVED,
            control_terminal_clock=terminal_clock,
            control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
            control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
            control_state_duration_gradient=False,
        )
        prediction = pipeline_module.PredictionPlan(
            plan, AIRPORT, ("eval",), split="val"
        )

        assert len(plan.train_dir.name.encode("utf-8")) <= 255
        assert len(prediction.pred_dir.name.encode("utf-8")) <= 255
        if filesystem_tag:
            assert filesystem_tag in plan.train_dir.name
            assert filesystem_tag in prediction.pred_dir.name
        train_dirs.append(plan.train_dir)
        prediction_dirs.append(prediction.pred_dir)
        categories.append(prediction.category)

    assert len(set(train_dirs)) == len(modes)
    assert len(set(prediction_dirs)) == len(modes)
    assert len(set(categories)) == len(modes)
    assert "terminal_predicted_clock" in categories[1]
    assert "terminal_predicted_detached_time_clock" in categories[2]


def test_fixed_dt_control_targets_gather_existing_two_second_reference_rows():
    series, config = _series(
        n_flights=2,
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    batch = dataset.batch(np.array([0, 1]))

    assert len(batch) == 7
    dense = batch[-1]
    for row, item in enumerate(series):
        valid = dense.valid[row]
        offsets = dense.query_offsets_s[row, valid].numpy()
        np.testing.assert_allclose(
            offsets,
            np.arange(1, len(offsets) + 1, dtype=np.float64) * config.dt_s,
        )
        query_times = item.times[config.seq_len - 1] + offsets
        source = np.searchsorted(item.supervision_times, query_times)
        expected = normalizer.encode(item.supervision_values[source]).astype(np.float32)
        np.testing.assert_allclose(dense.states[row, valid].numpy(), expected)


def test_fixed_dt_control_targets_choose_nearest_row_across_float_ulp():
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    values = np.arange(10, dtype=np.float32).reshape(5, 2)
    series = SimpleNamespace(
        times=times,
        supervision_times=times,
        supervision_weights=np.ones_like(values),
    )

    dense = build_fixed_dt_supervision(
        [series], [values], [(0, 2)], dt_s=0.1
    )

    assert dense.query_offsets_s[0].tolist() == pytest.approx([0.1, 0.2])
    np.testing.assert_array_equal(dense.states[0].numpy(), values[[3, 4]])


def test_horizontal_arc_resampling_is_independent_of_node_spacing():
    sparse = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 3.0]], dtype=np.float64
    )
    uneven = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [3.0, 0.0, 3.0]],
        dtype=np.float64,
    )
    expected = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        resample_horizontal_arc_length_numpy(sparse, points=4), expected
    )
    torch.testing.assert_close(
        resample_horizontal_arc_length_torch(
            torch.from_numpy(uneven)[None],
            torch.ones(1, len(uneven), dtype=torch.bool),
            points=4,
        )[0],
        torch.from_numpy(expected),
    )
    metrics = arc_length_geometry_metrics(
        uneven, sparse, _identity_normalizer(), points=4
    )
    assert metrics["loss"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["distance_mean_m"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["path_length_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["path_length_log_error"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["horizontal_mean_m"] == pytest.approx(0.0, abs=1e-12)


def test_arc_length_geometry_detects_shape_error_with_matching_endpoints():
    straight = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64
    )
    dogleg = np.array(
        [[0.0, 0.0, 0.0], [1.5, 1.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    metrics = arc_length_geometry_metrics(
        straight, dogleg, _identity_normalizer(), points=9
    )

    assert metrics["horizontal_mean_m"] > 0.25
    assert metrics["terminal_position_m"] == pytest.approx(0.0)
    assert metrics["path_length_log_error"] > 0.0


def test_arc_position_progress_weight_emphasizes_late_geometry_error():
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    early_error = reference.copy()
    early_error[0, 2] = 2.0
    late_error = reference.copy()
    late_error[-1, 2] = 2.0

    early = arc_length_geometry_metrics(
        early_error,
        reference,
        _identity_normalizer(),
        points=3,
        position_end_weight=4.0,
    )
    late = arc_length_geometry_metrics(
        late_error,
        reference,
        _identity_normalizer(),
        points=3,
        position_end_weight=4.0,
    )

    assert early["unweighted_loss"] == pytest.approx(late["unweighted_loss"])
    assert late["loss"] > early["loss"]


def test_arc_length_geometry_loss_has_position_and_reliable_velocity_gradients():
    channels = len(ch.CHANNELS)
    anchor = torch.zeros(1, channels)
    anchor[0, list(ch.VELOCITY_IDX)] = torch.tensor([1.0, 0.0, -1.0])
    endpoints = torch.zeros(1, 3, channels)
    endpoints[0, :, ch.POSITION_IDX[0]] = torch.tensor([1.0, 2.0, 3.0])
    endpoints[0, :2, ch.POSITION_IDX[1]] = 0.25
    endpoints[0, :, list(ch.VELOCITY_IDX)] = torch.tensor([0.0, 1.0, 1.0])
    endpoints.requires_grad_()
    states = torch.zeros(1, 2, channels)
    states[0, :, ch.POSITION_IDX[0]] = torch.tensor([1.0, 2.0])
    states[0, 0, list(ch.VELOCITY_IDX)] = torch.tensor([1.0, 0.0, -1.0])
    states[0, 1, list(ch.VELOCITY_IDX)] = 999.0
    weights = torch.full_like(states, 1.0 / channels)
    weights[0, 1, list(ch.VELOCITY_IDX)] = 0.0
    supervision = FixedDTControlSupervision(
        query_offsets_s=torch.tensor([[2.0, 4.0]], dtype=torch.float64),
        states=states,
        weights=weights,
        valid=torch.ones(1, 2, dtype=torch.bool),
    )
    terminal = torch.zeros(1, channels)
    terminal[0, ch.POSITION_IDX[0]] = 3.0

    terms = arc_length_state_loss_terms(
        anchor,
        endpoints,
        terminal,
        supervision,
        _identity_normalizer(),
        points=8,
    )
    loss = (
        terms.position
        + terms.horizontal_velocity_mps
        + terms.vertical_velocity_mps
    ).mean()
    loss.backward()

    assert terms.position.item() > 0.0
    assert terms.horizontal_velocity_mps.item() > 0.0
    assert terms.horizontal_tangent.item() > 0.0
    assert terms.horizontal_speed_mps.item() > 0.0
    assert terms.vertical_velocity_mps.item() > 0.0
    assert 0 < terms.velocity_valid_points.item() < 8
    assert torch.count_nonzero(endpoints.grad[..., list(ch.POSITION_IDX)]).item() > 0
    assert torch.count_nonzero(
        endpoints.grad[..., list(ch.VELOCITY_IDX[:2])]
    ).item() > 0
    assert torch.count_nonzero(
        endpoints.grad[..., ch.VELOCITY_IDX[2]]
    ).item() > 0


def test_arc_length_geometry_compacts_sparse_position_supervision_rows():
    channels = len(ch.CHANNELS)
    anchor = torch.zeros(1, channels)
    endpoints = torch.zeros(1, 3, channels)
    endpoints[0, :, ch.POSITION_IDX[0]] = torch.tensor([1.0, 3.0, 4.0])
    states = torch.zeros(1, 4, channels)
    states[0, :, ch.POSITION_IDX[0]] = torch.tensor([1.0, 999.0, 999.0, 3.0])
    weights = torch.zeros_like(states)
    weights[0, 0] = 1.0 / channels
    weights[0, 3, list(ch.POSITION_IDX)] = 1.0 / channels
    supervision = FixedDTControlSupervision(
        query_offsets_s=torch.tensor([[2.0, 4.0, 6.0, 8.0]], dtype=torch.float64),
        states=states,
        weights=weights,
        valid=torch.ones(1, 4, dtype=torch.bool),
    )
    terminal = torch.zeros(1, channels)
    terminal[0, ch.POSITION_IDX[0]] = 4.0

    terms = arc_length_state_loss_terms(
        anchor,
        endpoints,
        terminal,
        supervision,
        _identity_normalizer(),
        points=4,
    )

    assert terms.position.item() == pytest.approx(0.0, abs=1e-12)


def test_fixed_anchor_arc_geometry_filters_the_same_sparse_reference_rows():
    channels = len(ch.CHANNELS)
    config = TSConfig(seq_len=1, n_segments=2)
    anchor = np.zeros((1, channels), dtype=np.float32)
    predicted = np.zeros((1, 2, channels), dtype=np.float32)
    predicted[0, :, ch.POSITION_IDX[0]] = [1.0, 4.0]
    reference = np.zeros((5, channels), dtype=np.float32)
    reference[:, ch.POSITION_IDX[0]] = [0.0, 1.0, 999.0, 999.0, 4.0]
    weights = np.zeros_like(reference)
    weights[:2] = 1.0 / channels
    weights[-1, list(ch.POSITION_IDX)] = 1.0 / channels
    item = SimpleNamespace(
        dataset_id="KAAA:SPARSE",
        scenario=SimpleNamespace(target=SimpleNamespace(psi=0.0)),
        times=np.array([0.0]),
        values=reference[:1],
        supervision_times=np.arange(5, dtype=np.float64),
        supervision_values=reference,
        supervision_weights=weights,
    )

    metrics = fixed_anchor_arc_length_geometry_metrics(
        [item],
        config,
        anchor,
        predicted,
        np.zeros((1, len(ch.VELOCITY_IDX)), dtype=np.float32),
        _identity_normalizer(),
        points=3,
    )

    assert metrics["arc_length_geometry_loss"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["arc_length_distance_mean_m"] == pytest.approx(0.0, abs=1e-12)


def test_arc_length_velocity_metrics_follow_position_alignment_and_mask_tail():
    predicted_positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    reference_positions = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    predicted_velocity = np.array(
        [[10.0, 2.0, -1.0], [10.0, 2.0, -1.0], [99.0, 99.0, 99.0]]
    )
    reference_velocity = np.array(
        [[10.0, 0.0, -2.0], [999.0, 999.0, 999.0]]
    )

    metrics = arc_length_velocity_metrics(
        predicted_positions,
        predicted_velocity,
        reference_positions,
        reference_velocity,
        np.array([True, False]),
        points=4,
    )

    assert metrics["velocity_valid_points"] == 1
    assert metrics["horizontal_velocity_mae_mps"] == pytest.approx(2.0)
    assert metrics["horizontal_tangent_mean"] == pytest.approx(
        1.0 - 10.0 / math.sqrt(104.0)
    )
    assert metrics["horizontal_speed_mae_mps"] == pytest.approx(
        math.sqrt(104.0) - 10.0
    )
    assert metrics["vertical_velocity_mae_mps"] == pytest.approx(1.0)


def test_terminal_state_runway_components_rotate_enu_and_apply_emphasis():
    channels = len(ch.CHANNELS)
    endpoints = torch.zeros(1, 1, channels)
    endpoints[0, 0, list(ch.POSITION_IDX)] = torch.tensor([10.0, 20.0, 3.0])
    endpoints[0, 0, list(ch.VELOCITY_IDX)] = torch.tensor([4.0, 5.0, 6.0])
    supervision = FixedDTControlSupervision(
        query_offsets_s=torch.tensor([[2.0]], dtype=torch.float64),
        states=torch.zeros(1, 1, channels),
        weights=torch.ones(1, 1, channels),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )

    errors = terminal_state_errors(
        endpoints,
        torch.zeros(1, channels),
        torch.zeros(1, channels),
        supervision,
        _identity_normalizer(),
        torch.tensor([math.pi / 2]),
        coordinate_frame="enu",
        cross_track_emphasis=3.0,
        vertical_emphasis=5.0,
    )

    assert errors.along_position_abs_m.item() == pytest.approx(20.0)
    assert errors.cross_position_abs_m.item() == pytest.approx(10.0)
    assert errors.vertical_position_abs_m.item() == pytest.approx(3.0)
    assert errors.position_runway_components_m.item() == pytest.approx(65.0)
    assert errors.along_velocity_abs_mps.item() == pytest.approx(5.0)
    assert errors.cross_velocity_abs_mps.item() == pytest.approx(4.0)
    assert errors.vertical_velocity_abs_mps.item() == pytest.approx(6.0)
    assert errors.velocity_runway_components_mps.item() == pytest.approx(47.0)


def test_terminal_velocity_target_falls_back_to_observed_anchor():
    states = torch.full((1, 2, len(ch.CHANNELS)), 999.0)
    weights = torch.zeros_like(states)
    supervision = FixedDTControlSupervision(
        query_offsets_s=torch.tensor([[2.0, 4.0]], dtype=torch.float64),
        states=states,
        weights=weights,
        valid=torch.ones(1, 2, dtype=torch.bool),
    )
    anchor = torch.zeros(1, len(ch.CHANNELS))
    anchor[0, list(ch.VELOCITY_IDX)] = torch.tensor([80.0, 2.0, -3.0])

    target = last_reliable_terminal_velocity_target(anchor, supervision)

    torch.testing.assert_close(target, torch.tensor([[80.0, 2.0, -3.0]]))


def test_validation_terminal_velocity_matches_fixed_dt_training_before_off_grid_crossing():
    config = TSConfig(seq_len=2, dt_s=2.0)
    supervision_values = np.zeros((4, len(ch.CHANNELS)), dtype=np.float32)
    supervision_values[2, list(ch.VELOCITY_IDX)] = [10.0, 20.0, 30.0]
    supervision_values[3, list(ch.VELOCITY_IDX)] = [100.0, 200.0, 300.0]
    item = SimpleNamespace(
        dataset_id="KAAA:OFFGRID",
        times=np.array([0.0, 2.0, 4.0]),
        values=supervision_values[:3],
        supervision_times=np.array([0.0, 2.0, 4.0, 5.0]),
        supervision_values=supervision_values,
        supervision_weights=np.full_like(
            supervision_values, 1.0 / len(ch.CHANNELS)
        ),
    )
    dense = build_fixed_dt_supervision(
        [item], [supervision_values], [(0, config.seq_len - 1)], dt_s=config.dt_s
    )
    training_target = last_reliable_terminal_velocity_target(
        torch.from_numpy(item.values[-1:]), dense
    )
    _weights, validation_target = fixed_anchor_common_weights_and_terminal_velocity(
        [item], config, np.array([1.0]), np.array([3.0])
    )

    np.testing.assert_allclose(
        validation_target, training_target.numpy(), rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(validation_target[0], [10.0, 20.0, 30.0])


def test_formal_common_grid_selector_returns_only_lean_position_time_metrics():
    series, config = _series(n_flights=1, seq_len=8, n_segments=2)
    item = series[0]
    anchor = config.seq_len - 1
    duration = float(item.supervision_times[-1] - item.times[anchor])
    offsets = np.array([duration / 2.0, duration])
    query_times = item.times[anchor] + offsets
    predicted = np.column_stack([
        np.interp(query_times, item.supervision_times, item.supervision_values[:, channel])
        for channel in range(len(ch.CHANNELS))
    ])[None, ...]

    metrics = fixed_anchor_common_grid_ade_metrics(
        series,
        config,
        item.values[anchor][None, :],
        predicted,
        np.array([duration]),
        np.diff(np.concatenate(([0.0], offsets)))[None, :],
        points=2,
    )

    assert metrics["ade_m"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["fde_m"] == pytest.approx(0.0, abs=1e-7)
    assert "dense_state_loss" not in metrics
    assert not any(key.startswith("arc_length_") for key in metrics)


def test_formal_common_grid_selector_reuses_identical_precomputed_truth():
    series, config = _series(n_flights=3, seq_len=8, n_segments=4)
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    model = build_model(config).eval()
    replay = train_module._predict_split(
        model,
        dataset,
        normalizer,
        torch.device("cpu"),
        config.batch_size,
    )
    arguments = (
        series,
        config,
        replay.anchors,
        replay.predicted,
        replay.predicted_time_s,
        replay.segment_durations_s,
    )
    baseline = fixed_anchor_common_grid_ade_metrics(*arguments, points=7)
    cached = fixed_anchor_common_grid_ade_metrics(
        *arguments,
        points=7,
        common_truth=fixed_anchor_common_truth(series, config, 7),
    )

    assert baseline.keys() == cached.keys()
    for key in baseline:
        if isinstance(baseline[key], np.ndarray):
            np.testing.assert_array_equal(cached[key], baseline[key])
        else:
            assert cached[key] == baseline[key]


@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
@pytest.mark.parametrize("integrator_dt_s", [2.0, 0.3])
def test_fixed_dt_control_loss_forms_one_differentiable_training_step(
    model_name, integrator_dt_s
):
    series, config = _series(
        n_flights=1,
        model=model_name,
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        patch_len=4,
        stride=2,
        final_time_scale_s=600.0,
        control_rollout_integrator_dt_s=integrator_dt_s,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    x, target, weights, final_time, flight_weights, dynamics, dense = dataset.batch(
        np.array([0])
    )
    model = build_model(config)
    loss = prediction_loss(
        model(x, dynamics),
        x[:, -1],
        target,
        weights,
        final_time,
        flight_weights,
        config,
        normalizer,
        dynamics,
        dense,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
@pytest.mark.parametrize(
    ("objective", "selection"),
    [
        (
            CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
            CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        ),
    ],
)
def test_terminal_tracking_losses_are_differentiable_with_transport_dynamics(
    model_name, objective, selection
):
    arc_ablation = (
        {
            "control_arc_local_velocity_parameterization": (
                CONTROL_ARC_LOCAL_VELOCITY_TANGENT_SPEED
            ),
            "control_arc_position_end_weight": 4.0,
            "control_arc_terminal_parameterization": (
                CONTROL_ARC_TERMINAL_RUNWAY_COMPONENTS
            ),
        }
        if objective == CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY
        else {}
    )
    series, config = _series(
        n_flights=1,
        model=model_name,
        prediction_output=PREDICTION_CONTROL,
        control_dynamics_backend=CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=objective,
        control_state_duration_gradient=False,
        checkpoint_selection_metric=selection,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        patch_len=4,
        stride=2,
        final_time_scale_s=600.0,
        control_rollout_integrator_dt_s=0.5,
        **arc_ablation,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    x, target, weights, final_time, flight_weights, dynamics, dense = dataset.batch(
        np.array([0])
    )
    model = build_model(config)
    components = train_module.prediction_loss_components(
        model(x, dynamics),
        x[:, -1],
        target,
        weights,
        final_time,
        flight_weights,
        config,
        normalizer,
        dynamics,
        dense,
    )
    components.total.backward()

    assert torch.isfinite(components.total)
    assert components.state.item() >= 0.0
    assert components.terminal.item() >= 0.0
    assert components.extras["terminal_velocity"].item() >= 0.0
    if objective == CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY:
        assert components.extras["arc_horizontal_tangent"].item() >= 0.0
        assert components.extras["arc_horizontal_speed"].item() >= 0.0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("objective", "selection"),
    [
        (
            CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
            CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        ),
    ],
)
def test_terminal_tracking_loss_cuda_transport_smoke(objective, selection):
    series, config = _series(
        n_flights=1,
        model="itransformer",
        prediction_output=PREDICTION_CONTROL,
        control_dynamics_backend=CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=objective,
        control_state_duration_gradient=False,
        checkpoint_selection_metric=selection,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        final_time_scale_s=600.0,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    x, target, weights, final_time, flight_weights, dynamics, dense = dataset.batch(
        np.array([0])
    )
    device = torch.device("cuda")
    model = build_model(config).to(device)
    dynamics = {name: value.to(device) for name, value in dynamics.items()}
    components = train_module.prediction_loss_components(
        model(x.to(device), dynamics),
        x[:, -1].to(device),
        target.to(device),
        weights.to(device),
        final_time.to(device),
        flight_weights.to(device),
        config,
        normalizer,
        dynamics,
        dense.to(device),
    )
    components.total.backward()

    assert torch.isfinite(components.total)
    assert torch.isfinite(components.extras["terminal_velocity"])


def test_config_rejects_a_head_count_that_does_not_divide_d_model():
    with pytest.raises(ValueError, match="n_heads"):
        TSConfig(d_model=100, n_heads=8)


def test_default_config_uses_selected_normalized_output_and_physics_losses():
    config = TSConfig()
    assert config.n_segments == 16
    assert TSConfig(model="patchtst").n_segments == 256
    assert config.epochs == 180
    assert config.patience == 20
    assert config.lr_plateau_patience == 3
    assert config.lr_plateau_factor == 0.5
    assert config.state_endpoint_loss_weight == 0.25
    assert config.kinematic_consistency_loss_weight == 3.0
    assert config.terminal_loss_weight == 0.02


@pytest.mark.parametrize(
    "field",
    [
        "state_endpoint_loss_weight",
        "kinematic_consistency_loss_weight",
        "terminal_loss_weight",
    ],
)
def test_negative_physics_loss_weights_are_rejected(field):
    with pytest.raises(ValueError, match=field):
        TSConfig(**{field: -0.1})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seq_len", 0),
        ("n_segments", 0),
        ("dt_s", 0.0),
        ("batch_size", 0),
        ("epochs", 0),
        ("learning_rate", 0.0),
        ("final_time_scale_s", 0.0),
        ("patience", 0),
        ("d_model", 0),
        ("n_heads", 0),
        ("e_layers", 0),
    ],
)
def test_non_positive_training_parameters_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        TSConfig(**{field: value})


def test_vendor_pred_len_contract_tracks_n_segments():
    assert TSConfig(n_segments=99).pred_len == 99


def test_arrival_data_provenance_binds_manifest_and_per_flight_sources(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "airport": "KRDU",
        "records": [
            {"flight_key": "B", "source_sha256": "b" * 64},
            {"flight_key": "A", "source_sha256": "a" * 64},
        ],
    }), encoding="utf-8")

    provenance = arrival_data_provenance(manifest)
    entry = provenance["manifests"][0]
    assert provenance["schema_version"] == ARRIVAL_DATA_PROVENANCE_SCHEMA
    assert entry["airport"] == "KRDU"
    assert entry["arrival_manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert entry["source_records"] == [
        {"flight_key": "A", "source_sha256": "a" * 64},
        {"flight_key": "B", "source_sha256": "b" * 64},
    ]
    require_matching_data_provenance({"data_provenance": provenance}, provenance)
    with pytest.raises(ValueError, match="does not match"):
        changed = json.loads(json.dumps(provenance))
        changed["manifests"][0]["arrival_manifest_sha256"] = "c" * 64
        require_matching_data_provenance(
            {"data_provenance": provenance},
            changed,
        )


def test_prediction_provenance_accepts_only_exact_training_airport_subset():
    stored = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [
            {"airport": "KAAA", "arrival_manifest_sha256": "a" * 64, "source_records": []},
            {"airport": "KBBB", "arrival_manifest_sha256": "b" * 64, "source_records": []},
        ],
    }
    subset = {**stored, "manifests": [stored["manifests"][1]]}
    require_matching_data_provenance(
        {"data_provenance": stored}, subset, allow_subset=True
    )
    changed = json.loads(json.dumps(subset))
    changed["manifests"][0]["arrival_manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="subset"):
        require_matching_data_provenance(
            {"data_provenance": stored}, changed, allow_subset=True
        )


def test_auto_batch_uses_config_default_without_cuda():
    config = TSConfig(batch_size=37)
    assert resolve_batch_size(
        config, torch.device("cpu"), auto=True, verbose=False
    ) == 37


def test_auto_batch_selects_2048_when_2048_probe_succeeds(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: None)
    monkeypatch.setattr(batching, "_probe_training_step", lambda *_args: None)

    assert batching.resolve_batch_size(
        TSConfig(), torch.device("cuda"), auto=True, verbose=False
    ) == 2048


def test_control_auto_batch_probe_uses_heterogeneous_duration_partitions():
    batch_size, n_segments = 8, 8
    final_time = torch.full((batch_size,), 80.0)
    prediction = ControlPrediction(
        controls=torch.zeros(batch_size, n_segments, 3),
        segment_durations=torch.full((batch_size, n_segments), 10.0),
        final_time_s=final_time,
    )

    probed = batching._heterogeneous_control_probe_prediction(prediction)
    fractions = probed.segment_durations / probed.final_time_s[:, None]
    baseline_steps = int(torch.ceil(
        prediction.segment_durations.max(dim=0).values / 0.5
    ).sum())
    heterogeneous_steps = int(torch.ceil(
        probed.segment_durations.max(dim=0).values / 0.5
    ).sum())

    assert torch.allclose(probed.segment_durations.sum(dim=-1), final_time)
    assert torch.unique(fractions, dim=0).shape[0] == batch_size
    assert heterogeneous_steps >= 2 * baseline_steps


def test_control_auto_batch_retains_one_power_of_two_safety_margin(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: None)
    monkeypatch.setattr(batching, "_probe_training_step", lambda *_args: None)

    assert batching.resolve_batch_size(
        TSConfig(prediction_output=PREDICTION_CONTROL),
        torch.device("cuda"),
        auto=True,
        verbose=False,
    ) == 1024


def test_control_auto_batch_training_probe_applies_heterogeneous_partition(monkeypatch):
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        device="cpu",
        seq_len=4,
        n_segments=2,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
    )
    original = batching._heterogeneous_control_probe_prediction
    calls: list[torch.Size] = []

    def tracked(prediction):
        calls.append(prediction.segment_durations.shape)
        return original(prediction)

    monkeypatch.setattr(batching, "_heterogeneous_control_probe_prediction", tracked)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    batching._probe_training_step(config, 2, torch.device("cpu"))

    assert calls == [torch.Size([2, 2])]


def test_control_auto_batch_training_probe_executes_clip_diagnostics(monkeypatch):
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        device="cpu",
        seq_len=4,
        n_segments=2,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
        control_gradient_clip_norm=20.0,
    )
    events: list[str] = []

    class TrackedDiagnostics(ControlTrainingDiagnosticsAccumulator):
        def record_prediction(self, prediction, dynamics):
            events.append("prediction")
            return super().record_prediction(prediction, dynamics)

        def record_gradients_and_clip(self, model):
            assert any(parameter.grad is not None for parameter in model.parameters())
            events.append("gradients")
            return super().record_gradients_and_clip(model)

    monkeypatch.setattr(
        batching,
        "ControlTrainingDiagnosticsAccumulator",
        TrackedDiagnostics,
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    batching._probe_training_step(config, 2, torch.device("cpu"))

    assert events == ["prediction", "gradients"]


def test_pipeline_carries_and_names_complete_control_recipe(tmp_path):
    plan = pipeline_module.TrainingPlan(
        ("KMSY", "KRDU"),
        "itransformer",
        training_mode="pooled",
        prediction_output=PREDICTION_CONTROL,
        n_segments=32,
        seed=2027,
        split_seed=1337,
        aircraft_type="A320",
        aircraft_filter=AIRCRAFT_FILTER_OPENAP_DIRECT,
        batch_size="16",
        control_effort_weight=1e-4,
        control_smoothness_weight=1e-2,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_duration_gradient=False,
        control_rollout_dt=0.5,
        output_dir=tmp_path,
    )
    recipe = plan._recipe_args()
    config, _source = plan.resolved_train_config(use_best_config=False)
    prediction = pipeline_module.PredictionPlan(plan, "KMSY", ("eval",), split="val")

    assert recipe[recipe.index("--prediction-output") + 1] == PREDICTION_CONTROL
    assert recipe[recipe.index("--split-seed") + 1] == "1337"
    assert recipe[recipe.index("--control-effort-weight") + 1] == "0.0001"
    assert recipe[recipe.index("--control-smoothness-weight") + 1] == "0.01"
    assert recipe[recipe.index("--control-duration-parameterization") + 1] == "uniform"
    assert recipe[recipe.index("--control-state-clock") + 1] == "observed"
    assert (
        recipe[recipe.index("--control-state-loss-grid") + 1]
        == "native-segment-endpoints"
    )
    assert (
        recipe[recipe.index("--control-state-objective") + 1] == "true-time-position"
    )
    assert "--no-control-state-duration-gradient" in recipe
    assert recipe[recipe.index("--control-rollout-dt") + 1] == "0.5"
    assert recipe[recipe.index("--aircraft-filter") + 1] == "openap-direct"
    assert config.prediction_output == PREDICTION_CONTROL
    assert config.aircraft_filter == AIRCRAFT_FILTER_OPENAP_DIRECT
    assert config.control_effort_loss_weight == pytest.approx(1e-4)
    assert config.control_duration_parameterization == CONTROL_DURATION_UNIFORM
    assert config.control_state_supervision_clock == CONTROL_STATE_CLOCK_OBSERVED
    assert config.control_state_loss_grid == CONTROL_STATE_LOSS_GRID_NATIVE
    assert (
        config.control_state_objective == CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION
    )
    assert config.control_state_duration_gradient is False
    assert "control" in prediction.pred_dir.name
    assert "uniform_duration" in prediction.pred_dir.name
    assert "observed_clock" in prediction.pred_dir.name
    assert "true_time_position" in prediction.pred_dir.name
    assert "detached_duration_gradient" in prediction.pred_dir.name
    assert "control" in prediction.category
    assert "openap_direct" in prediction.category
    assert "uniform durations" in prediction.label
    assert "true-time physical position criterion" in prediction.label
    assert "detached duration-state gradient" in prediction.label


def test_pipeline_carries_and_names_control_horizon_curriculum():
    plan = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="pooled",
        prediction_output=PREDICTION_CONTROL,
        epochs=4,
        control_state_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        control_state_duration_gradient=False,
        control_horizon_curriculum_s=(2.0, 4.0),
        control_horizon_curriculum_stage_epochs=1,
        control_gradient_clip_norm=20.0,
        control_gradient_clip_policy=CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED,
    )
    recipe = plan._recipe_args()
    config, _source = plan.resolved_train_config(use_best_config=False)
    prediction = pipeline_module.PredictionPlan(
        plan, AIRPORT, ("eval",), split="val"
    )

    assert recipe[recipe.index("--control-horizon-curriculum") + 1] == "2,4"
    assert recipe[recipe.index("--control-horizon-stage-epochs") + 1] == "1"
    assert config.control_horizon_curriculum_s == (2.0, 4.0)
    assert config.control_horizon_curriculum_stage_epochs == 1
    assert config.control_gradient_clip_norm == pytest.approx(20.0)
    assert (
        config.control_gradient_clip_policy
        == CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED
    )
    assert recipe[recipe.index("--control-gradient-clip-norm") + 1] == "20"
    assert (
        recipe[recipe.index("--control-gradient-clip-policy") + 1]
        == CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED
    )
    # The arc-length recipe suffix overruns the 255-byte path-component cap, so the
    # directory name is head + digest; the readable recipe survives in category/label.
    assert len(plan.train_dir.name.encode("utf-8")) <= (
        pipeline_module.MAX_PATH_COMPONENT_BYTES
    )
    assert "horizon_curriculum_2_4s_x1" in prediction.category
    assert "gradient_clip20_final_time_decoupled" in prediction.category
    assert "horizon curriculum 2→4 s × 1 epochs" in prediction.label
    assert "gradient clip 20 (final-time head decoupled)" in prediction.label


def test_pipeline_rejects_control_checkpoint_metadata_without_duration_recipe(
    tmp_path, monkeypatch
):
    plan = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="per-airport",
        prediction_output=PREDICTION_CONTROL,
        output_dir=tmp_path / "run",
    )
    plan.train_dir.mkdir(parents=True)
    plan.checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "arrivals.json"
    manifest.write_text("{}", encoding="utf-8")
    plan.data_manifests = (manifest,)
    monkeypatch.setattr(pipeline_module, "_manifest_digests", lambda _airports: [])
    roster = tmp_path / "lateral_pass_eligibility.json"
    roster.write_text("{}", encoding="utf-8")
    plan.eligibility_rosters = (roster,)
    eligibility_digest = hashlib.sha256(b"{}").hexdigest()
    monkeypatch.setattr(
        pipeline_module,
        "_eligibility_digests",
        lambda _airports: {AIRPORT: eligibility_digest},
    )

    config, _source = plan.resolved_train_config(use_best_config=False)
    legacy_recipe = control_recipe(config)
    legacy_recipe.pop("duration_parameterization")
    metadata = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA,
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "arrival_manifests": [],
        "eligibility_rosters": {AIRPORT: eligibility_digest},
        "random_train_anchor": plan.random_train_anchor,
        "training_cohort_min_future_s": plan.training_cohort_min_future_s,
        "random_train_anchor_min_future_s": plan.random_train_anchor_min_future_s,
        "checkpoint_selection_metric": plan.checkpoint_selection_metric,
        "validation_common_grid_points": plan.validation_common_grid_points,
        "prediction_output": config.prediction_output,
        "aircraft_filter": config.aircraft_filter,
        "horizon_mode": config.horizon_mode,
        "pred_len": config.pred_len,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": config.lr_plateau_factor,
            "patience": config.lr_plateau_patience,
        },
        "control_recipe": legacy_recipe,
    }
    plan.checkpoint_metadata.write_text(json.dumps(metadata), encoding="utf-8")

    assert plan.checkpoint_reuse_error() == (
        "checkpoint control recipe does not match the requested recipe"
    )


def test_experiment_index_keeps_legacy_incomplete_and_formal_runs_distinct(tmp_path):
    legacy_complete = tmp_path / "legacy" / "run_complete"
    legacy_complete.mkdir(parents=True)
    (legacy_complete / "checkpoint.pt").write_bytes(b"checkpoint")
    (legacy_complete / "history.json").write_text(
        json.dumps({"config": {"prediction_output": "state", "seed": 1337}}),
        encoding="utf-8",
    )
    legacy_incomplete = tmp_path / "legacy" / "run_aborted"
    legacy_incomplete.mkdir(parents=True)
    (legacy_incomplete / "experiment_notes.md").write_text("aborted", encoding="utf-8")
    formal = tmp_path / "openap_direct" / "control"
    formal.mkdir(parents=True)
    (formal / experiment_index.RUN_MANIFEST_NAME).write_text(
        json.dumps({
            "schema_version": experiment_index.RUN_MANIFEST_SCHEMA,
            "campaign_id": "openap-direct-20260729",
            "run_id": "control",
            "status": "completed",
            "config": {
                "prediction_output": "control",
                "aircraft_filter": "openap-direct",
                "seed": 1337,
            },
            "artifacts": {"checkpoint.pt": {}},
        }),
        encoding="utf-8",
    )

    document = experiment_index.rebuild_index(tmp_path)
    by_run = {entry["run_id"]: entry for entry in document["entries"]}

    assert by_run["run_complete"]["status"] == "completed"
    assert by_run["run_aborted"]["status"] == "incomplete"
    assert by_run["control"]["campaign_id"] == "openap-direct-20260729"
    assert by_run["control"]["aircraft_filter"] == "openap-direct"
    assert (tmp_path / "index.json").is_file()
    assert (tmp_path / "INDEX.md").is_file()


def test_common_grid_resampling_uses_explicit_nonuniform_control_clock():
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        seq_len=2,
        n_segments=2,
        d_model=8,
        n_heads=2,
        d_ff=16,
        e_layers=1,
    )
    anchor = np.zeros(config.enc_in)
    values = np.zeros((2, config.enc_in))
    values[:, 0] = [1.0, 3.0]

    sampled, capped = predictability_report.resample_prediction(
        anchor,
        values,
        3.0,
        config,
        np.array([1.0, 2.0, 3.0]),
        segment_durations_s=np.array([1.0, 2.0]),
    )

    assert not capped
    np.testing.assert_allclose(sampled[:, 0], [1.0, 2.0, 3.0])


def test_common_grid_report_executes_control_model_with_flight_dynamics():
    series, config = _series(
        n_flights=2,
        prediction_output=PREDICTION_CONTROL,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)
    run = predictability_report.LoadedRun(
        "control", Path("checkpoint.pt"), build_model(config), config, normalizer, {}
    )
    histories = torch.from_numpy(
        predictability_report.history_tensor(series, config, normalizer)
    )

    with torch.no_grad():
        values, final_time, durations, controls, control_durations = (
            predictability_report.predict_batch_nodes(
                run, histories, series, torch.device("cpu")
            )
        )

    assert values.shape == (2, config.validation_common_grid_points, config.enc_in)
    assert durations.shape == (2, config.validation_common_grid_points)
    assert controls is not None and controls.shape == (2, config.n_segments, 3)
    assert control_durations is not None
    assert control_durations.shape == (2, config.n_segments)
    np.testing.assert_allclose(durations.sum(axis=1), final_time, rtol=1e-6)
    np.testing.assert_allclose(control_durations.sum(axis=1), final_time, rtol=1e-6)


def test_control_distribution_statistics_reports_bounds_changes_and_duration_tails():
    controls = np.array([
        [[0.0, -0.5, 0.5], [50.0, 0.0, 1.0], [100.0, 0.5, 2.0]],
        [[0.0, -1.0, 0.5], [100.0, 0.0, 1.0], [200.0, 1.0, 2.0]],
    ])
    durations = np.array([[1.0, 2.0, 3.0], [0.5, 4.0, 8.0]])
    lower = np.array([[0.0, -0.5, 0.5], [0.0, -1.0, 0.5]])
    upper = np.array([[100.0, 0.5, 2.0], [200.0, 1.0, 2.0]])

    result = predictability_report.control_distribution_statistics(
        controls, durations, lower, upper
    )

    thrust = result["channels"]["thrust_N"]
    assert thrust["median"] == pytest.approx(75.0)
    assert thrust["near_lower_fraction"] == pytest.approx(2.0 / 6.0)
    assert thrust["near_upper_fraction"] == pytest.approx(2.0 / 6.0)
    assert thrust["adjacent_abs_change_median"] == pytest.approx(75.0)
    assert result["durations_s"]["min"] == pytest.approx(0.5)
    assert result["durations_s"]["max"] == pytest.approx(8.0)


def test_auto_batch_probe_executes_the_shared_physics_loss(monkeypatch):
    config = TSConfig(
        device="cpu",
        seq_len=4,
        n_segments=4,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
    )
    original_loss = train_module.prediction_loss
    calls: list[tuple[torch.Size, torch.Size]] = []

    def tracked_loss(prediction, anchor, target, *args):
        calls.append((anchor.shape, target.shape))
        return original_loss(prediction, anchor, target, *args)

    monkeypatch.setattr(train_module, "prediction_loss", tracked_loss)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    batching._probe_training_step(config, 2, torch.device("cpu"))

    assert calls == [
        (torch.Size([2, len(ch.CHANNELS)]), torch.Size([2, 4, len(ch.CHANNELS)]))
    ]


def test_cross_validation_never_passes_outer_val_or_test_to_fit(tmp_path, monkeypatch):
    series, config = _series(
        n_flights=36, device="cpu", epochs=1, patience=1,
        d_model=32, d_ff=64, n_heads=4, e_layers=1,
    )
    outer_train, outer_val, outer_test = split_by_flight(series, config)
    forbidden = {item.dataset_id for item in [*outer_val, *outer_test]}
    observed: list[set[str]] = []

    def fake_fit(train_series, val_series, fold_config, **_kwargs):
        identities = {item.dataset_id for item in [*train_series, *val_series]}
        observed.append(identities)
        score = float(fold_config.learning_rate + fold_config.d_model * 1e-8)
        row = SimpleNamespace(
            epoch=1, val_loss=score, val_by_airport={AIRPORT: score}
        )
        return SimpleNamespace(
            history=[row], best_val_loss=score, config=fold_config,
            model=object(), normalizer=Normalizer.fit(train_series),
            device=torch.device("cpu"),
        )

    monkeypatch.setattr(cv, "fit_model", fake_fit)
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }
    result = cv.cross_validate(
        series,
        config,
        output_dir=tmp_path,
        data_provenance=provenance,
        n_splits=3,
        cv_parameters=("learning_rate",),
        cv_epochs=1,
        cv_patience=1,
        verbose=False,
    )
    assert observed
    assert all(not (identities & forbidden) for identities in observed)
    assert set.union(*observed) == {item.dataset_id for item in outer_train}
    assert result["leakage_guard"]["outer_test_used"] is False
    assert result["schema_version"] == cv.RESULTS_SCHEMA
    assert result["selection_metric"] == (
        "mean outer-train-fold airport-macro fixed-anchor common physical-time ADE"
    )
    assert (tmp_path / "best_config.json").is_file()


def test_cross_validation_resumes_from_atomic_candidate_checkpoint(
    tmp_path, monkeypatch
):
    series, config = _series(
        n_flights=24,
        device="cpu",
        epochs=1,
        patience=1,
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        batch_size=32,
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }
    first_run_calls = 0

    def interrupted_fit(train_series, _val_series, fold_config, **_kwargs):
        nonlocal first_run_calls
        first_run_calls += 1
        if first_run_calls == 3:
            raise KeyboardInterrupt
        score = float(fold_config.weight_decay + fold_config.seed * 1e-9)
        row = SimpleNamespace(
            epoch=1,
            val_loss=score,
            val_by_airport={AIRPORT: score},
        )
        return SimpleNamespace(
            history=[row],
            best_val_loss=score,
            config=fold_config,
            model=object(),
            normalizer=Normalizer.fit(train_series),
            device=torch.device("cpu"),
        )

    monkeypatch.setattr(cv, "fit_model", interrupted_fit)
    with pytest.raises(KeyboardInterrupt):
        cv.cross_validate(
            series,
            config,
            output_dir=tmp_path,
            data_provenance=provenance,
            n_splits=2,
            cv_parameters=("weight_decay",),
            cv_epochs=1,
            cv_patience=1,
            verbose=False,
        )

    progress_path = tmp_path / cv.PROGRESS_NAME
    progress = json.loads(progress_path.read_text())
    assert progress["schema_version"] == cv.PROGRESS_SCHEMA
    assert progress["completed_candidates"] == 1
    assert [row["candidate"] for row in progress["candidates"]] == [0]
    assert not progress_path.with_suffix(progress_path.suffix + ".tmp").exists()

    resumed_calls = 0

    def resumed_fit(train_series, _val_series, fold_config, **_kwargs):
        nonlocal resumed_calls
        resumed_calls += 1
        score = float(fold_config.weight_decay + fold_config.seed * 1e-9)
        row = SimpleNamespace(
            epoch=1,
            val_loss=score,
            val_by_airport={AIRPORT: score},
        )
        return SimpleNamespace(
            history=[row],
            best_val_loss=score,
            config=fold_config,
            model=object(),
            normalizer=Normalizer.fit(train_series),
            device=torch.device("cpu"),
        )

    monkeypatch.setattr(cv, "fit_model", resumed_fit)
    result = cv.cross_validate(
        series,
        config,
        output_dir=tmp_path,
        data_provenance=provenance,
        n_splits=2,
        cv_parameters=("weight_decay",),
        cv_epochs=1,
        cv_patience=1,
        verbose=False,
    )

    assert resumed_calls == 2
    assert [row["candidate"] for row in result["candidates"]] == [0, 1]
    completed = json.loads(progress_path.read_text())
    assert completed["completed_candidates"] == 2
    assert completed["run_contract_sha256"] == result["run_contract_sha256"]


def test_cross_validation_rejects_candidate_checkpoint_from_another_contract(
    tmp_path, monkeypatch
):
    series, config = _series(
        n_flights=24,
        device="cpu",
        epochs=1,
        patience=1,
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        batch_size=32,
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }

    def fake_fit(train_series, _val_series, fold_config, **_kwargs):
        score = float(fold_config.weight_decay + fold_config.seed * 1e-9)
        row = SimpleNamespace(
            epoch=1,
            val_loss=score,
            val_by_airport={AIRPORT: score},
        )
        return SimpleNamespace(
            history=[row],
            best_val_loss=score,
            config=fold_config,
            model=object(),
            normalizer=Normalizer.fit(train_series),
            device=torch.device("cpu"),
        )

    monkeypatch.setattr(cv, "fit_model", fake_fit)
    cv.cross_validate(
        series,
        config,
        output_dir=tmp_path,
        data_provenance=provenance,
        n_splits=2,
        cv_parameters=("weight_decay",),
        cv_epochs=1,
        cv_patience=1,
        verbose=False,
    )
    progress_path = tmp_path / cv.PROGRESS_NAME
    progress = json.loads(progress_path.read_text())
    progress["run_contract_sha256"] = "0" * 64
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate checkpoint.*contract"):
        cv.cross_validate(
            series,
            config,
            output_dir=tmp_path,
            data_provenance=provenance,
            n_splits=2,
            cv_parameters=("weight_decay",),
            cv_epochs=1,
            cv_patience=1,
            verbose=False,
        )


def test_cross_validation_describes_every_checkpoint_selection_metric():
    assert set(cv.SELECTION_METRIC_DESCRIPTIONS) == set(CHECKPOINT_SELECTION_METRICS)


def test_cross_validation_runs_real_two_fold_search(tmp_path):
    series, config = _series(
        n_flights=20, device="cpu", epochs=1, patience=1,
        d_model=16, d_ff=32, n_heads=4, e_layers=1,
        seq_len=20, n_segments=10, batch_size=32,
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }
    result = cv.cross_validate(
        series,
        config,
        output_dir=tmp_path,
        data_provenance=provenance,
        n_splits=2,
        cv_parameters=("weight_decay",),
        cv_epochs=1,
        cv_patience=1,
        verbose=False,
    )
    assert len(result["candidates"]) == 2
    assert all(len(candidate["folds"]) == 2 for candidate in result["candidates"])
    assert all(
        "validation_selection_by_airport" in fold
        and "validation_metrics" not in fold
        for candidate in result["candidates"]
        for fold in candidate["folds"]
    )
    assert result["base_config"]["n_segments"] == config.n_segments
    assert "n_segments" not in result["best_overrides"]
    assert json.loads((tmp_path / "best_config.json").read_text()) == result["best_overrides"]


def test_cross_validation_exhausts_the_default_three_parameter_grid():
    config = TSConfig(n_segments=128)
    candidates = cv._candidate_overrides(config)

    assert len(candidates) == 45
    assert {
        (candidate["n_segments"], candidate["learning_rate"], candidate["d_model"])
        for candidate in candidates
    } == set(itertools.product(
        (16, 32, 64, 128, 256),
        (1e-4, 3e-4, 5e-4),
        (64, 128, 256),
    ))
    assert all(candidate["d_ff"] == 2 * candidate["d_model"] for candidate in candidates)


@pytest.mark.parametrize("horizon_mode", [HORIZON_FULL, HORIZON_WINDOW])
def test_fixed_horizon_cv_does_not_repeat_inert_n_segment_candidates(horizon_mode):
    config = TSConfig(horizon_mode=horizon_mode)
    candidates = cv._candidate_overrides(config)

    assert len(candidates) == 9
    assert all("n_segments" not in candidate for candidate in candidates)
    assert {
        (candidate["learning_rate"], candidate["d_model"])
        for candidate in candidates
    } == set(itertools.product(
        (1e-4, 3e-4, 5e-4),
        (64, 128, 256),
    ))


# ── Metrics ──────────────────────────────────────────────────────────────────


def test_spread_matches_the_gate_side_signed_spread():
    # metrics._spread is the VECTORISED twin of evaluation/stats.signed_spread (that one
    # is stdlib-only by design and would sort millions of boxed floats here). This seam
    # test is what makes "same statistic" a checked property instead of a mirror comment:
    # if either side changes its percentile method or keys, this fails.
    from evaluation.stats import signed_spread
    from metrics import _spread

    values = np.array([3.0, -1.5, 0.25, -7.0, 4.0, 2.5, -0.75])
    ours, theirs = _spread(values), signed_spread(values.tolist())
    assert set(ours) == set(theirs)
    for key in ours:
        assert ours[key] == pytest.approx(theirs[key])


def test_raw_kinematic_metrics_use_nonuniform_segment_durations():
    # Constant 1 m/s² eastward acceleration sampled at deliberately nonuniform node times.
    # Trapezoidal velocity integration is exact here, acceleration is constant, and jerk is
    # zero. A metric that silently assumes uniform N spacing fails this test.
    node_times = np.array([0.0, 1.0, 3.0, 6.0])
    nodes = np.zeros((1, len(node_times), len(ch.CHANNELS)), dtype=np.float64)
    nodes[0, :, ch.IDX["e"]] = 0.5 * node_times**2
    nodes[0, :, ch.IDX["edot"]] = node_times

    block = raw_kinematic_metrics(
        nodes[:, 0], nodes[:, 1:], np.diff(node_times)[None, :]
    )

    assert block["position_velocity_rmse_mps"] == pytest.approx(0.0, abs=1e-12)
    assert block["heading_consistency_p95_deg"] == pytest.approx(0.0, abs=1e-12)
    assert block["turn_rate_p95_deg_s"] == pytest.approx(0.0, abs=1e-12)
    assert block["acceleration_p95_mps2"] == pytest.approx(1.0)
    assert block["jerk_p95_mps3"] == pytest.approx(0.0, abs=1e-12)


def test_raw_kinematic_metrics_measure_geometric_turn_acceleration_and_jerk():
    # Three one-second position segments turn east -> north -> west. Node velocities are
    # chosen so their trapezoidal midpoint exactly matches each geometric segment velocity;
    # consistency and heading error must therefore be zero while the path still has a
    # 90 deg/s turn rate, sqrt(200) m/s² acceleration and 20 m/s³ jerk.
    nodes = np.zeros((1, 4, len(ch.CHANNELS)), dtype=np.float64)
    nodes[0][:, [ch.IDX["e"], ch.IDX["n"]]] = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],
    ])
    nodes[0][:, [ch.IDX["edot"], ch.IDX["ndot"]]] = np.array([
        [10.0, 0.0], [10.0, 0.0], [-10.0, 20.0], [-10.0, -20.0],
    ])

    block = raw_kinematic_metrics(
        nodes[:, 0], nodes[:, 1:], np.ones((1, 3), dtype=np.float64)
    )

    assert block["position_velocity_rmse_mps"] == pytest.approx(0.0, abs=1e-12)
    assert block["heading_consistency_p95_deg"] == pytest.approx(0.0, abs=1e-12)
    assert block["turn_rate_p95_deg_s"] == pytest.approx(90.0)
    assert block["acceleration_p95_mps2"] == pytest.approx(math.sqrt(200.0))
    assert block["jerk_p95_mps3"] == pytest.approx(20.0)


def test_raw_kinematic_metrics_detect_velocity_heading_disagreement():
    nodes = np.zeros((1, 4, len(ch.CHANNELS)), dtype=np.float64)
    nodes[0, :, ch.IDX["e"]] = np.arange(4) * 10.0
    nodes[0, :, ch.IDX["ndot"]] = 10.0

    block = raw_kinematic_metrics(
        nodes[:, 0], nodes[:, 1:], np.ones((1, 3), dtype=np.float64)
    )

    assert block["position_velocity_rmse_mps"] == pytest.approx(math.sqrt(200.0 / 3.0))
    assert block["heading_consistency_p95_deg"] == pytest.approx(90.0)


# ── Forecast ─────────────────────────────────────────────────────────────────

def test_batched_control_forecasts_match_independent_dense_rollouts(monkeypatch):
    series, config = _series(
        n_flights=4,
        prediction_output=PREDICTION_CONTROL,
        n_segments=2,
        seq_len=20,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)
    dense_batch_sizes: list[int] = []
    original_dense_rollout = control_rollout_module.rollout_control_dense

    def capture_dense_batch(controls, *args, **kwargs):
        dense_batch_sizes.append(len(controls))
        return original_dense_rollout(controls, *args, **kwargs)

    monkeypatch.setattr(
        control_rollout_module, "rollout_control_dense", capture_dense_batch
    )

    class HeterogeneousControlModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, history, dynamics):
            self.batch_sizes.append(len(history))
            total = 2.0 + torch.sigmoid(history[:, -1, ch.IDX["e"]])
            durations = total[:, None] * history.new_tensor([[0.375, 0.625]])
            controls = 0.5 * (
                dynamics["control_lower"] + dynamics["control_upper"]
            )
            return ControlPrediction(
                controls=controls[:, None, :].expand(-1, 2, -1).contiguous(),
                segment_durations=durations,
                final_time_s=total,
            )

    batched_model = HeterogeneousControlModel()
    batched = forecast_approaches(
        batched_model,
        series,
        config,
        normalizer,
        device=torch.device("cpu"),
    )
    assert batched_model.batch_sizes == [1] * len(series)
    assert dense_batch_sizes == [len(series)]

    independent_model = HeterogeneousControlModel()
    independent = [
        forecast_approach(
            independent_model,
            item,
            config,
            normalizer,
            device=torch.device("cpu"),
        )
        for item in series
    ]
    assert independent_model.batch_sizes == [1] * len(series)
    assert dense_batch_sizes == [len(series)] + [1] * len(series)

    for actual, expected in zip(batched, independent, strict=True):
        assert actual.anchor == expected.anchor
        assert actual.final_time_s == pytest.approx(expected.final_time_s, abs=1e-12)
        assert actual.predicted_final_time_s == pytest.approx(
            expected.predicted_final_time_s, abs=1e-12
        )
        np.testing.assert_allclose(actual.times, expected.times, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(
            actual.geodetic_values,
            expected.geodetic_values,
            rtol=1e-12,
            atol=1e-10,
        )
        np.testing.assert_allclose(actual.controls, expected.controls, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            actual.sample_durations_s,
            expected.sample_durations_s,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual.segment_durations_s,
            expected.segment_durations_s,
            rtol=0.0,
            atol=0.0,
        )


def test_forecast_uses_n_normalized_points_and_the_predicted_final_time():
    series, config = _series(n_flights=2, n_segments=10, seq_len=20)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    forecast = forecast_approach(model, series[0], config, normalizer,
                                 device=torch.device("cpu"))

    anchor_time = series[0].times[forecast.anchor]
    assert forecast.n_steps == config.n_segments
    assert forecast.normalized_progress == pytest.approx(np.arange(1, 11) / 10)
    assert forecast.times[-1] - anchor_time == pytest.approx(forecast.final_time_s)
    assert np.diff(np.concatenate([[anchor_time], forecast.times])) == pytest.approx(
        np.full(config.n_segments, forecast.final_time_s / config.n_segments)
    )


def test_full_forecast_uses_one_fixed_dt_pass_and_threshold_truncation():
    series, config = _series(
        n_flights=2,
        seq_len=20,
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=5,
    )
    normalizer = _identity_normalizer()

    class FixedPrediction(torch.nn.Module):
        def forward(self, history):
            batch = len(history)
            states = torch.zeros(
                batch, config.pred_len, len(config.channels), device=history.device
            )
            states[:, :, ch.IDX["e"]] = torch.tensor(
                [5.0, 2.0, 0.0, 3.0, 6.0], device=history.device
            )
            return StatePrediction(
                states=states,
                final_time_s=torch.full((batch,), 5.0, device=history.device),
            )

    forecast = forecast_approach(
        FixedPrediction(), series[0], config, normalizer, device=torch.device("cpu")
    )
    anchor_time = series[0].times[forecast.anchor]

    assert forecast.horizon_mode == HORIZON_FULL
    assert forecast.n_steps == 3
    assert forecast.times - anchor_time == pytest.approx([2.0, 4.0, 6.0])
    assert forecast.final_time_s == pytest.approx(6.0)
    assert forecast.passes == 1
    assert forecast.truncated_at_threshold
    assert not forecast.horizon_capped


def test_window_forecast_recurses_to_the_full_horizon():
    series, config = _series(
        n_flights=2,
        seq_len=20,
        horizon_mode=HORIZON_WINDOW,
        window_horizon_steps=2,
        full_horizon_steps=6,
    )
    normalizer = Normalizer.fit(series)

    class FixedPrediction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, history):
            self.calls += 1
            states = torch.zeros(
                len(history), config.pred_len, len(config.channels), device=history.device
            )
            states[:, :, ch.IDX["e"]] = 100.0 - self.calls
            return StatePrediction(
                states=states,
                final_time_s=torch.full((len(history),), 12.0, device=history.device),
            )

    model = FixedPrediction()
    forecast = forecast_approach(
        model, series[0], config, normalizer, device=torch.device("cpu"), truncate=False
    )

    assert forecast.horizon_mode == HORIZON_WINDOW
    assert forecast.n_steps == config.full_horizon_steps
    assert forecast.passes == 3
    assert model.calls == 3


def test_config_keeps_three_horizon_output_lengths_separate():
    normalized = TSConfig(model="itransformer")
    full = TSConfig(
        model="itransformer",
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=300,
    )
    window = TSConfig(
        model="itransformer",
        horizon_mode=HORIZON_WINDOW,
        window_horizon_steps=30,
    )

    assert normalized.horizon_mode == HORIZON_NORMALIZED
    assert normalized.pred_len == normalized.n_segments == 16
    assert full.pred_len == 300
    assert window.pred_len == 30


# ── Export seam ──────────────────────────────────────────────────────────────

def test_record_stem_disambiguates_flights_sharing_a_callsign():
    # The same callsign flies the same approach daily; id alone collides and one day's
    # result silently overwrites another's.
    monday = {"id": "AAL1", "runway": "05L", "icao24": "a1b2c3",
              "landing_time_utc": "2026-06-18T21:37:36Z"}
    tuesday = dict(monday, landing_time_utc="2026-06-19T21:31:02Z")
    assert record_stem(monday, 0) != record_stem(tuesday, 0)
    assert record_stem(monday, 0) == "AAL1_05L_a1b2c3_20260618T213736Z"


def test_exported_record_satisfies_the_evaluation_contract():
    # The real validator, not a copy of it: record_from_dict enforces the state keys, the
    # target, and final_time_s == states[-1]["t"] within 1e-6. Here final_time_s is learned,
    # so deriving it from N or dt would violate the record contract.
    series, config = _series(n_flights=3)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()

    forecast = forecast_approach(model, series[0], config, normalizer,
                                 device=torch.device("cpu"))
    record = build_prediction_record(series[0], forecast, index=0,
                                     model_name=config.model, horizon_mode=config.horizon_mode)

    parsed = record_from_dict(record.eval_record)
    assert parsed.solved
    assert parsed.source["subject"] == "predicted"
    assert parsed.controls == []            # a predictor emits no control schedule
    assert parsed.final_time_s == pytest.approx(parsed.states[-1]["t"], abs=1e-6)
    assert set(parsed.states[0]) == {"t", "lat", "lon", "alt", "V", "psi", "gamma", "m"}

    # t=0 is the anchor, and states[0] IS initial_state — same convention as an optimizer
    # record, so the two are readable side by side.
    assert parsed.states[0]["t"] == pytest.approx(0.0)
    for key, value in parsed.initial_state.items():
        assert parsed.states[0][key] == pytest.approx(value)


def test_normalized_state_export_preserves_a_tiny_relative_clock():
    """Relative offsets must not collapse when added to and subtracted from the anchor."""
    series, config = _series(n_flights=1)
    normalizer = Normalizer.fit(series)
    tiny_final_time_s = 3.7044681016305814e-13

    class TinyDurationStateModel(torch.nn.Module):
        def forward(self, history):
            batch = len(history)
            return StatePrediction(
                states=torch.zeros(
                    (batch, config.pred_len, config.enc_in),
                    dtype=history.dtype,
                    device=history.device,
                ),
                final_time_s=torch.full(
                    (batch,),
                    tiny_final_time_s,
                    dtype=history.dtype,
                    device=history.device,
                ),
            )

    forecast = forecast_approach(
        TinyDurationStateModel(),
        series[0],
        config,
        normalizer,
        device=torch.device("cpu"),
    )
    metrics = observed_series_metrics(series[0], forecast)
    record = build_prediction_record(
        series[0],
        forecast,
        index=0,
        model_name=config.model,
        horizon_mode=config.horizon_mode,
    )
    parsed = record_from_dict(record.eval_record)
    exported_times = np.array([state["t"] for state in parsed.states])

    assert metrics["raw_kinematics"]["predicted"]["segments"] == config.pred_len
    assert np.all(np.diff(exported_times) > 0.0)
    assert parsed.final_time_s == pytest.approx(tiny_final_time_s, rel=1e-6)


def test_post_training_accuracy_does_not_drop_the_tail_after_early_completion():
    series, config = _series(n_flights=1)
    item = series[0]
    anchor = config.seq_len - 1
    duration = float(item.times[anchor + 1] - item.times[anchor])
    forecast = Forecast(
        times=np.array([item.times[anchor + 1]]),
        values=item.values[anchor + 1 : anchor + 2].copy(),
        normalized_progress=np.array([1.0]),
        anchor=anchor,
        final_time_s=duration,
        predicted_final_time_s=duration,
        horizon_mode=HORIZON_NORMALIZED,
        passes=1,
        truncated_at_threshold=False,
        horizon_capped=False,
        sample_durations_s=np.array([duration]),
        segment_durations_s=np.array([duration]),
    )

    metrics = observed_series_metrics(item, forecast, points=8)

    assert metrics["n_steps"] == 8
    assert metrics["coverage_ratio"] < 0.1
    assert metrics["ade_m"] > 0.0
    assert metrics["fde_m"] > metrics["ade_m"]
    assert metrics["final_time_error_s"] < 0.0


def test_control_forecast_exports_optimizer_shaped_states_and_aligned_controls():
    series, config = _series(
        n_flights=1,
        prediction_output=PREDICTION_CONTROL,
        n_segments=2,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)

    class FixedControlModel(torch.nn.Module):
        def forward(self, history, dynamics):
            batch = len(history)
            controls = torch.tensor(
                [[[0.20, 0.04, 1.01], [0.16, -0.02, 0.99]]],
                dtype=history.dtype,
                device=history.device,
            ).expand(batch, -1, -1)
            durations = torch.tensor(
                [[0.75, 1.25]], dtype=history.dtype, device=history.device
            ).expand(batch, -1)
            return ControlPrediction(
                controls=controls,
                segment_durations=durations,
                final_time_s=durations.sum(dim=-1),
            )

    forecast = forecast_approach(
        FixedControlModel(),
        series[0],
        config,
        normalizer,
        device=torch.device("cpu"),
    )
    record = build_prediction_record(
        series[0], forecast, index=0, model_name=config.model,
        horizon_mode=config.horizon_mode,
    )
    parsed = record_from_dict(record.eval_record)

    assert parsed.solved
    assert len(parsed.states) == len(parsed.controls) == 6
    assert record.source["predictionOutput"] == "control"
    assert [state["t"] for state in parsed.states] == pytest.approx(
        [0.0, 0.5, 0.75, 1.0, 1.5, 2.0]
    )
    assert forecast.sample_durations_s.tolist() == pytest.approx(
        [0.5, 0.25, 0.25, 0.5, 0.5]
    )
    assert forecast.segment_durations_s.tolist() == pytest.approx([0.75, 1.25])
    # The model predicts thrust FRACTIONS; the exported record contract is newtons, so
    # each row must come back multiplied by this flight's installed thrust.
    max_thrust_n = series[0].scenario.aircraft.engine.max_thrust_total_n
    assert parsed.controls[0]["thrust"] == pytest.approx(0.20 * max_thrust_n)
    assert parsed.controls[2]["thrust"] == pytest.approx(0.20 * max_thrust_n)
    assert parsed.controls[3]["thrust"] == pytest.approx(0.16 * max_thrust_n)
    assert parsed.controls[-1]["thrust"] == pytest.approx(0.16 * max_thrust_n)
    assert [row["duration_s"] for row in record.states_payload["control_segments"]] \
        == pytest.approx([0.75, 1.25])
    assert parsed.final_time_s == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("duration_parameterization", "expected_contract"),
    [
        (
            CONTROL_DURATION_FACTORIZED,
            "bounded-control-nonuniform-duration-casadi-rollout-clock-aligned-v2"
            "+duration-uniform-floor=0.8-v1",
        ),
        (
            CONTROL_DURATION_UNIFORM,
            "bounded-control-uniform-duration-casadi-rollout-clock-aligned-v1",
        ),
    ],
)
def test_control_training_checkpoint_round_trip_keeps_output_identity(
    tmp_path, duration_parameterization, expected_contract
):
    series, config = _series(
        n_flights=12,
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=duration_parameterization,
        epochs=1,
        patience=1,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        n_segments=2,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
        device="cpu",
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "b" * 64,
            "source_records": [
                {"flight_key": item.flight_id, "source_sha256": f"{index + 100:064x}"}
                for index, item in enumerate(series)
            ],
        }],
    }
    train(series, config, output_dir=tmp_path, data_provenance=provenance, verbose=False)
    model, loaded_config, _normalizer, payload = load_checkpoint(tmp_path / "checkpoint.pt")
    metadata = json.loads((tmp_path / "checkpoint_metadata.json").read_text())

    assert loaded_config == config
    assert loaded_config.prediction_output == PREDICTION_CONTROL
    assert isinstance(model(torch.zeros(1, config.seq_len, config.enc_in), {
        "condition": torch.ones(1, len(dataset_module.DYNAMICS_CONDITION_NAMES)),
        "control_lower": torch.tensor([CONTROL_LOWER], dtype=torch.float32),
        "control_upper": torch.tensor([CONTROL_UPPER], dtype=torch.float32),
    }), ControlPrediction)
    assert payload["target_contract"] == expected_contract
    assert metadata["prediction_output"] == PREDICTION_CONTROL
    # control_recipe() IS the schema; restating it here would give a fixture that cannot
    # detect the field it forgot to add. Assert the identity plus the values this
    # particular run is meant to pin.
    assert metadata["control_recipe"] == control_recipe(loaded_config)
    assert metadata["control_recipe"]["duration_parameterization"] == (
        duration_parameterization
    )
    assert metadata["control_recipe"]["dynamics_backend"] == (
        CONTROL_DYNAMICS_REANCHORED_RK4
    )
    assert metadata["control_recipe"]["dynamics_model"] == CONTROL_DYNAMICS_POINT_MASS
    assert metadata["control_recipe"]["state_objective"] == (
        CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE
    )


def test_reference_comparison_detects_a_prediction_with_a_different_endpoint():
    # A shared anchor is necessary but not sufficient. An untrained prediction can finish
    # far from the observed endpoint; normalizing both full paths would compare different
    # physical locations and must therefore be skipped by batch evaluation.
    from evaluation.reference import reference_span

    series, config = _series(n_flights=3)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    forecast = forecast_approach(model, series[0], config, normalizer,
                                 device=torch.device("cpu"))
    record = build_prediction_record(series[0], forecast, index=0,
                                     model_name=config.model, horizon_mode=config.horizon_mode)

    predicted = record_from_dict(record.eval_record)
    reference = record_from_dict(record.reference_record)
    assert reference.states[0]["t"] == pytest.approx(0.0)
    # Same starting point, to the metre — they are the same observed anchor sample.
    assert reference.states[0]["lat"] == pytest.approx(predicted.states[0]["lat"], abs=1e-9)
    assert reference.states[0]["lon"] == pytest.approx(predicted.states[0]["lon"], abs=1e-9)

    span = reference_span(predicted, reference)
    assert span.start_gap_m == pytest.approx(0.0, abs=1e-6)
    assert span.comparable is False
    assert span.end_gap_m > 1.0


def test_batch_writes_a_manifest_that_evaluation_can_load_and_grade(tmp_path):
    series, config = _series(n_flights=4)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()

    records, overlap = [], []
    for index, s in enumerate(series):
        forecast = forecast_approach(model, s, config, normalizer, device=torch.device("cpu"))
        records.append(build_prediction_record(s, forecast, index=index,
                                               model_name=config.model,
                                               horizon_mode=config.horizon_mode))
        overlap.append(observed_series_metrics(s, forecast))
    write_batch(
        records, output_dir=tmp_path, config_dict=config.to_dict(), flight_metrics=overlap
    )

    # load_records is manifest-ONLY (no glob fallback), so this also proves summary.json
    # carries a results[] roster with resolvable eval_file entries.
    loaded = load_records(tmp_path)
    assert len(loaded) == len(series)
    report = evaluate_batch(loaded, contexts=_terminal_contexts())
    assert report["total"] == len(series) and report["solved"] == len(series)

    # Every record points at a reference that exists, so compare_to_reference works.
    for record in loaded:
        assert record.reference_file is not None
        assert (Path(tmp_path) / record.reference_file).is_file()


def test_manifest_carries_the_accuracy_the_run_printed(tmp_path):
    # The batch's error against the observed tracks is its headline result; it used to exist
    # only in terminal scrollback, which made any cross-batch comparison (the instance-norm
    # ablation) a stdout-scraping exercise.
    series, config = _series(n_flights=4)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()

    records, overlap = [], []
    for index, s in enumerate(series):
        forecast = forecast_approach(model, s, config, normalizer, device=torch.device("cpu"))
        records.append(build_prediction_record(s, forecast, index=index,
                                               model_name=config.model,
                                               horizon_mode=config.horizon_mode))
        overlap.append(observed_series_metrics(s, forecast))
    write_batch(
        records,
        output_dir=tmp_path,
        config_dict=config.to_dict(),
        flight_metrics=overlap,
        split="val",
    )

    summary = json.loads((Path(tmp_path) / "summary.json").read_text(encoding="utf-8"))
    assert summary["split"] == "val"
    assert all(row["split"] == "val" for row in summary["results"])
    states = json.loads(next(Path(tmp_path).glob("*_states.json")).read_text(encoding="utf-8"))
    assert states["source"]["predictionSplit"] == "val"
    assert summary["accuracy"] == accuracy_block(overlap)
    assert summary["accuracy"]["flights"] == len(series)
    assert summary["accuracy"]["ade_m"]["mean"] > 0.0
    assert summary["accuracy"]["final_time_s"]["mae"] >= 0.0
    assert set(summary["accuracy"]["raw_kinematics"]) == {
        "predicted", "observed_baseline", "delta"
    }
    # Per-flight too, so a batch can be re-aggregated (per runway, per capped/uncapped)
    # without re-running the forecast.
    for row, metrics in zip(summary["results"], overlap):
        assert row["ade_m"] == pytest.approx(metrics["ade_m"])
        assert row["metric_steps"] == metrics["n_steps"]
        assert row["arrival_endpoint_error_m"] == pytest.approx(
            metrics["arrival_endpoint_error_m"]
        )
        assert row["final_time_error_s"] == pytest.approx(metrics["final_time_error_s"])
        assert row["raw_kinematics"] == metrics["raw_kinematics"]


def test_accuracy_block_rejects_missing_or_nonfinite_flight_metrics():
    raw = {
        "predicted": {key: 2.0 for key in RAW_KINEMATIC_METRIC_KEYS},
        "observed_baseline": {key: 1.0 for key in RAW_KINEMATIC_METRIC_KEYS},
    }
    overlap = [{"ade_m": 100.0, "fde_m": 200.0,
                "arrival_endpoint_error_m": 175.0, "cross_track_p95_m": 50.0,
                "altitude_p95_m": 10.0, "n_steps": 30,
                "true_final_time_s": 300.0, "final_time_error_s": 20.0,
                "raw_kinematics": raw},
               {"ade_m": float("nan"), "fde_m": float("nan"),
                "arrival_endpoint_error_m": float("nan"),
                "cross_track_p95_m": float("nan"),
                "altitude_p95_m": float("nan"), "n_steps": 0,
                "true_final_time_s": 400.0, "final_time_error_s": -10.0,
                "raw_kinematics": raw}]
    with pytest.raises(ValueError, match="finite common-time metrics"):
        accuracy_block(overlap)
    with pytest.raises(ValueError, match="finite common-time metrics"):
        accuracy_block([overlap[1]])
    bad_time = dict(overlap[0], final_time_error_s=float("inf"))
    with pytest.raises(ValueError, match="finite final-time error"):
        accuracy_block([bad_time])


def test_accuracy_block_emits_empty_raw_metric_stats_when_all_values_are_nan():
    raw = {
        "predicted": {key: float("nan") for key in RAW_KINEMATIC_METRIC_KEYS},
        "observed_baseline": {key: float("nan") for key in RAW_KINEMATIC_METRIC_KEYS},
    }
    row = {
        "ade_m": 1.0,
        "fde_m": 2.0,
        "arrival_endpoint_error_m": 2.5,
        "cross_track_p95_m": 1.0,
        "altitude_p95_m": 1.0,
        "n_steps": 1,
        "true_final_time_s": 1.0,
        "final_time_error_s": 0.0,
        "raw_kinematics": raw,
    }

    block = accuracy_block([row])

    for role in ("predicted", "observed_baseline"):
        for key in RAW_KINEMATIC_METRIC_KEYS:
            stats = block["raw_kinematics"][role][key]
            assert stats["count"] == 0
            assert all(math.isnan(stats[name]) for name in ("median", "mean", "p95", "max"))
    assert all(
        math.isnan(value) for value in block["raw_kinematics"]["delta"].values()
    )


def test_write_batch_rejects_overlap_that_does_not_line_up_with_the_records(tmp_path):
    # Positional alignment is the whole contract — a short list would silently zip away the
    # tail of the batch, attributing metrics to the wrong flights.
    series, config = _series(n_flights=3)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    records = [
        build_prediction_record(
            s, forecast_approach(model, s, config, normalizer, device=torch.device("cpu")),
            index=index, model_name=config.model, horizon_mode=config.horizon_mode)
        for index, s in enumerate(series)
    ]
    with pytest.raises(ValueError, match="once per record"):
        write_batch(
            records, output_dir=tmp_path, config_dict=config.to_dict(), flight_metrics=[]
        )


def test_write_batch_rejects_non_finite_values_in_referenced_state_payload(tmp_path):
    series, config = _series(n_flights=3)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    forecast = forecast_approach(
        model, series[0], config, normalizer, device=torch.device("cpu")
    )
    record = build_prediction_record(
        series[0], forecast, index=0,
        model_name=config.model, horizon_mode=config.horizon_mode,
    )
    record.states_payload["predicted_states"][0]["alt"] = float("nan")
    overlap = [observed_series_metrics(series[0], forecast)]

    with pytest.raises(ValueError, match="Out of range float values"):
        write_batch(
            [record], output_dir=tmp_path,
            config_dict=config.to_dict(), flight_metrics=overlap,
        )


def test_write_batch_serializes_unavailable_raw_metrics_as_json_null(tmp_path):
    series, config = _series(n_flights=3)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    forecast = forecast_approach(
        model, series[0], config, normalizer, device=torch.device("cpu")
    )
    record = build_prediction_record(
        series[0], forecast, index=0,
        model_name=config.model, horizon_mode=config.horizon_mode,
    )
    overlap = [observed_series_metrics(series[0], forecast)]
    observed_raw = overlap[0]["raw_kinematics"]["observed_baseline"]
    observed_raw["heading_consistency_p95_deg"] = float("nan")
    observed_raw["turn_rate_p95_deg_s"] = float("nan")

    write_batch(
        [record], output_dir=tmp_path,
        config_dict=config.to_dict(), flight_metrics=overlap,
    )

    summary_text = (Path(tmp_path) / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in summary_text
    summary = json.loads(summary_text, parse_constant=lambda token: pytest.fail(token))
    row_raw = summary["results"][0]["raw_kinematics"]["observed_baseline"]
    assert row_raw["heading_consistency_p95_deg"] is None
    assert row_raw["turn_rate_p95_deg_s"] is None
    fleet_raw = summary["accuracy"]["raw_kinematics"]["observed_baseline"]
    assert fleet_raw["heading_consistency_p95_deg"] == {
        "count": 0, "median": None, "mean": None, "p95": None, "max": None,
    }
    assert summary["accuracy"]["raw_kinematics"]["delta"][
        "heading_consistency_p95_deg"
    ] is None


def test_stale_records_are_cleared_before_a_rerun(tmp_path):
    series, config = _series(n_flights=3)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()

    def batch(count):
        records, overlap = [], []
        for index, s in enumerate(series[:count]):
            forecast = forecast_approach(model, s, config, normalizer,
                                         device=torch.device("cpu"))
            records.append(build_prediction_record(s, forecast, index=index,
                                                   model_name=config.model,
                                                   horizon_mode=config.horizon_mode))
            overlap.append(observed_series_metrics(s, forecast))
        write_batch(
            records,
            output_dir=tmp_path,
            config_dict=config.to_dict(),
            flight_metrics=overlap,
        )

    batch(3)
    batch(1)   # a shrinking flight set must not leave orphans behind
    assert len(list(Path(tmp_path).glob("*_eval.json"))) == 1
    assert len(list((Path(tmp_path) / "references").glob("*_reference_eval.json"))) == 1


# ── End to end ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("horizon_mode", "contract", "expected_passes"),
    (
        (HORIZON_FULL, "full-horizon-physical-position-duration-v1", 1),
        (HORIZON_WINDOW, "recursive-window-physical-position-duration-v1", 3),
    ),
)
def test_fixed_time_modes_train_checkpoint_and_forecast(
    tmp_path, horizon_mode, contract, expected_passes
):
    series, config = _series(
        n_flights=12,
        model="itransformer",
        horizon_mode=horizon_mode,
        full_horizon_steps=12,
        window_horizon_steps=4,
        epochs=1,
        patience=1,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        device="cpu",
    )
    train(
        series,
        config,
        output_dir=tmp_path,
        data_provenance=_fake_data_provenance(),
        verbose=False,
    )

    model, loaded_config, normalizer, payload = load_checkpoint(tmp_path / "checkpoint.pt")
    forecast = forecast_approach(
        model,
        series[0],
        loaded_config,
        normalizer,
        device=torch.device("cpu"),
        truncate=False,
    )

    assert payload["target_contract"] == contract
    assert loaded_config.horizon_mode == horizon_mode
    assert forecast.passes == expected_passes
    assert forecast.n_steps == loaded_config.full_horizon_steps


def test_fit_evaluation_is_fixed_anchor_eval_mode_and_repeatable():
    series, config = _series(
        n_flights=12,
        random_train_anchor=True,
        dropout=0.5,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        n_segments=8,
        device="cpu",
    )
    train_series, val_series, _test_series = split_by_flight(series, config)
    normalizer = Normalizer.fit(train_series)
    model = build_model(config).train()

    first = evaluate_fit_splits(
        model, train_series, val_series, normalizer, config, torch.device("cpu")
    )
    second = evaluate_fit_splits(
        model, train_series, val_series, normalizer, config, torch.device("cpu")
    )

    assert not model.training
    assert first == second
    assert first["schema_version"] == FIT_EVALUATION_SCHEMA
    assert first["evaluation_contract"] == {
        "model_mode": "eval",
        "dropout": "disabled",
        "anchor": "fixed L-1",
        "batch_order": "sequential (shuffle disabled)",
        "splits": ["train", "val"],
        "metric_grid": (
            "Q=64 common true physical time; prediction endpoint held after early completion"
        ),
    }
    assert first["splits"]["train"]["flights"] == len(train_series)
    assert first["splits"]["val"]["flights"] == len(val_series)
    assert first["diagnostics"]["generalization"]["ade_m"]["ratio"] > 0.0


def test_evaluate_fit_cli_runs_train_and_validation_together(tmp_path, monkeypatch):
    series, config = _series(
        n_flights=12,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        n_segments=8,
        device="cpu",
    )
    train_series, val_series, test_series = split_by_flight(series, config)
    normalizer = Normalizer.fit(train_series)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint identity")
    provenance = _fake_data_provenance()
    payload = {
        "split": {
            "train": [item.dataset_id for item in train_series],
            "val": [item.dataset_id for item in val_series],
            "test": [item.dataset_id for item in test_series],
        },
        "data_provenance": provenance,
    }
    flights = [{"key": item.dataset_id} for item in (*train_series, *val_series)]
    report = SimpleNamespace(format=lambda: "built synthetic fit replay")

    monkeypatch.setattr(
        ts_cli, "load_checkpoint",
        lambda _path: (build_model(config), config, normalizer, payload),
    )
    monkeypatch.setattr(ts_cli, "arrival_data_provenance", lambda _data: provenance)
    monkeypatch.setattr(ts_cli, "require_matching_data_provenance", lambda *_args: None)
    loaded_keys = None

    def load_selected(_data, *, include_flight_keys=None):
        nonlocal loaded_keys
        loaded_keys = include_flight_keys
        return flights

    monkeypatch.setattr(ts_cli, "load_flight_dicts", load_selected)
    monkeypatch.setattr(ts_cli, "dataset_flight_key", lambda flight, _index: flight["key"])
    monkeypatch.setattr(ts_cli, "build_series", lambda *_args, **_kwargs: (series, report))
    monkeypatch.setattr(ts_cli, "resolve_device", lambda _device: torch.device("cpu"))

    assert ts_cli.main([
        "evaluate-fit",
        "--checkpoint", str(checkpoint),
        "--data", str(tmp_path / "manifest.json"),
        "--output-dir", str(tmp_path / "fit"),
        "--device", "cpu",
    ]) == 0
    replay = json.loads(
        (tmp_path / "fit" / FIT_EVALUATION_NAME).read_text(encoding="utf-8")
    )
    assert set(replay["splits"]) == {"train", "val"}
    assert loaded_keys == set(payload["split"]["train"] + payload["split"]["val"])


def test_train_refuses_to_replace_a_checkpoint_with_a_test_release(tmp_path):
    series, config = _series(
        n_flights=12,
        epochs=1,
        patience=1,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        device="cpu",
    )
    output = tmp_path / "released"
    output.mkdir()
    checkpoint = output / "checkpoint.pt"
    checkpoint.write_bytes(b"released checkpoint")
    (output / evaluation_protocol.TEST_RELEASE_NAME).write_text(
        json.dumps({"schema_version": evaluation_protocol.TEST_RELEASE_SCHEMA}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="test release"):
        train(
            series,
            config,
            output_dir=output,
            data_provenance=_fake_data_provenance(),
            verbose=False,
        )

    assert checkpoint.read_bytes() == b"released checkpoint"


@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_train_then_predict_produces_a_gradeable_batch(tmp_path, model_name):
    # Plumbing, not quality: two epochs on synthetic straight-ins. Proves a checkpoint
    # round-trips (config + normalizer + weights) into records `evaluation` can grade.
    series, config = _series(
        n_flights=12, model=model_name, epochs=2, patience=2,
        batch_size=32, d_model=32, n_heads=4, d_ff=64, e_layers=1, seq_len=20,
        n_segments=10,
        device="cpu",
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [
                {"flight_key": item.flight_id, "source_sha256": f"{index:064x}"}
                for index, item in enumerate(series)
            ],
        }],
    }
    summary = train(
        series,
        config,
        output_dir=tmp_path / "run",
        data_provenance=provenance,
        # Exercise the persisted metric block's human-readable publication seam too.
        verbose=True,
    )
    assert summary["epochs_run"] == 2
    assert set(summary["metrics"]) == {"train", "val"}
    assert summary["metrics"]["train"]["ade_m"] > 0.0
    assert summary["metrics"]["val"]["ade_m"] > 0.0
    assert "raw_kinematics" in summary["metrics"]["train"]
    assert "raw_kinematics" in summary["metrics"]["val"]
    fit_evaluation = json.loads(
        (tmp_path / "run" / FIT_EVALUATION_NAME).read_text(encoding="utf-8")
    )
    assert fit_evaluation["schema_version"] == FIT_EVALUATION_SCHEMA
    assert fit_evaluation["checkpoint"]["sha256"] == hashlib.sha256(
        (tmp_path / "run" / "checkpoint.pt").read_bytes()
    ).hexdigest()
    assert set(fit_evaluation["splits"]) == {"train", "val"}

    model, loaded_config, normalizer, payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded_config == config          # the config survives the round-trip verbatim
    assert payload["target_contract"] == (
        "normalized-output-true-time-physical-position-duration-v1"
    )
    assert payload[evaluation_protocol.TEST_RELEASE_PROTOCOL_FIELD] == (
        evaluation_protocol.TEST_RELEASE_SCHEMA
    )
    assert set(payload["split"]) == {"train", "val", "test"}
    assert payload["training_anchor_contract"] == summary["training_anchor_contract"]
    assert payload["training_cohort"] == summary["training_cohort"]
    assert payload["training_cohort"]["scope"] == "train only after by-flight split"
    assert payload["data_provenance"] == provenance
    checkpoint = tmp_path / "run" / "checkpoint.pt"
    metadata = json.loads(
        (tmp_path / "run" / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata == {
            "schema_version": CHECKPOINT_METADATA_SCHEMA,
        evaluation_protocol.TEST_RELEASE_PROTOCOL_FIELD:
            evaluation_protocol.TEST_RELEASE_SCHEMA,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "arrival_manifests": {AIRPORT: "a" * 64},
            "random_train_anchor": False,
            "training_anchor_contract": summary["training_anchor_contract"],
            "training_cohort_min_future_s": 0.0,
            "training_cohort_excluded_flights": 0,
            "random_train_anchor_min_future_s": 60.0,
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_COMMON_GRID_ADE,
            "validation_common_grid_points": 64,
        "horizon_mode": HORIZON_NORMALIZED,
        "prediction_output": "state",
        "aircraft_filter": "all",
        "pred_len": config.pred_len,
        "full_horizon_steps": config.full_horizon_steps,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": config.lr_plateau_factor,
            "patience": config.lr_plateau_patience,
        },
        "split_sha256": {
            split: hashlib.sha256("\n".join(sorted(payload["split"][split])).encode()).hexdigest()
            for split in ("train", "val", "test")
        },
    }
    assert set(summary["history"][0]["train_components"]) == {
        "state", "final_time", "kinematic", "terminal"
    }
    assert set(summary["history"][0]["val_components"]) == {
        "state", "final_time", "kinematic", "terminal"
    }
    assert summary["history"][0]["learning_rate"] == config.learning_rate
    assert summary["history"][0]["optimizer_updates"] > 0
    objective = fit_evaluation["diagnostics"]["training_objective"]
    assert objective["total_optimizer_updates"] == summary["history"][-1]["optimizer_updates"]
    assert objective["final_learning_rate"] == summary["history"][-1]["learning_rate"]

    records, overlap = [], []
    for index, s in enumerate(series[:4]):
        forecast = forecast_approach(model, s, loaded_config, normalizer,
                                     device=torch.device("cpu"))
        records.append(build_prediction_record(s, forecast, index=index,
                                               model_name=loaded_config.model,
                                               horizon_mode=loaded_config.horizon_mode))
        overlap.append(observed_series_metrics(s, forecast))
    out = tmp_path / "pred"
    write_batch(
        records,
        output_dir=out,
        config_dict=loaded_config.to_dict(),
        flight_metrics=overlap,
    )

    report = evaluate_batch(load_records(out), contexts=_terminal_contexts())
    assert report["total"] == 4
    # The gate outcome is not asserted — an undertrained model on synthetic data may or may
    # not land inside 106.75 m, and pinning that would make this a flaky quality test.
    assert "lateral_m" in report and "success_rate" in report

    history = json.loads((tmp_path / "run" / "history.json").read_text(encoding="utf-8"))
    assert history["config"]["model"] == model_name
    assert len(history["history"]) == 2
    assert history["data_provenance"]["source_record_count"] == len(series)
