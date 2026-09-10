"""The export seam: record stems, the evaluation record contract, the summary.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ts_transformer.outputs.control.conditioning import DYNAMICS_CONDITION_NAMES
from ts_transformer.config import (
    HORIZON_NORMALIZED,
    CONTROL_DYNAMICS_POINT_MASS,
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_DURATION_FACTORIZED,
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    PREDICTION_CONTROL,
    TSConfig,
    control_recipe,
)
from ts_transformer.outputs.control.envelope import CONTROL_LOWER, CONTROL_UPPER
from ts_transformer.data.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from ts_transformer.data.dataset import Normalizer, build_series
from evaluation.metrics import evaluate_batch
from evaluation.records import load_records, record_from_dict
from ts_transformer.inference.export import (
    accuracy_block, build_prediction_record, observed_series_metrics, record_stem, write_batch,
)
from ts_transformer.inference.forecast import Forecast, forecast_approach
from ts_transformer.geometry.metrics import RAW_KINEMATIC_METRIC_KEYS
from ts_transformer.backbone.adapters import build_model
from ts_transformer.outputs.control.heads import ControlPrediction
from ts_transformer.outputs.state.model import StatePrediction
from ts_transformer.data.synthetic import synthetic_arrivals
# Imported, never restated: a schema version pinned by hand in a fixture is a version
# the fixture cannot check, and this one gates every loader that reads the roster.
from ts_transformer.tests.support import terminal_contexts
from ts_transformer.training.train import load_checkpoint, train

AIRPORT, RUNWAY = "KRDU", "05L"


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


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
        "condition": torch.ones(1, len(DYNAMICS_CONDITION_NAMES)),
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
    report = evaluate_batch(loaded, contexts=terminal_contexts())
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
        "difficulty": {
            "anchor_range_m": 12000.0,
            "remaining_path_m": 12500.0,
            "route_tortuosity": 12500.0 / 12000.0,
            "anchor_cross_track_m": -40.0,
            "established_at_anchor": True,
        },
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
