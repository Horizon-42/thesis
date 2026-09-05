"""The closure prediction output: its config contract, the label ↔ decision round trip,
the closed-form reconstruction, and one whole chain — train, checkpoint, forecast,
export, evaluation — on synthetic arrivals labelled by ``fit_labels``."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (TS_DIR, REPO_ROOT, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import channels as ch  # noqa: E402
import closure_geometry as cg  # noqa: E402
import closure_output as co  # noqa: E402
import closure_profile as cp  # noqa: E402
from config import CHECKPOINT_SELECTION_OBJECTIVE, PREDICTION_CLOSURE, TSConfig  # noqa: E402
from dataset import ARRIVAL_DATA_PROVENANCE_SCHEMA, Normalizer, build_series  # noqa: E402
from evaluation.metrics import evaluate_batch  # noqa: E402
from evaluation.records import load_records  # noqa: E402
from evaluation.thresholds import AssessmentContext  # noqa: E402
from export import build_prediction_record, observed_series_metrics, write_batch  # noqa: E402
from forecast import forecast_approach, forecast_closure_from_labels  # noqa: E402
from models import build_model  # noqa: E402
from run_naming import run_display_name  # noqa: E402
from synthetic import synthetic_arrivals  # noqa: E402
from train import load_checkpoint, loss_component_names, prediction_loss_components, target_contract, train  # noqa: E402

AIRPORT, RUNWAY = "KRDU", "05L"
TINY = dict(seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1, final_time_scale_s=2.0,
            device="cpu", horizon_mode="normalized", checkpoint_selection_metric=CHECKPOINT_SELECTION_OBJECTIVE,
            epochs=1, patience=1, batch_size=8)


def _closure_config(labels_path: Path, **overrides) -> TSConfig:
    return TSConfig(prediction_output=PREDICTION_CLOSURE, closure_labels_path=str(labels_path), **{**TINY, **overrides})


def _closure_series(tmp_path: Path, n_flights: int = 4, **overrides):
    """Synthetic arrivals, their closure labels file, and a closure config pointing at it."""
    labels_path = tmp_path / "closure_labels.json"
    config = _closure_config(labels_path, **overrides)
    series, report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3), config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    anchor = config.seq_len - 1
    flights = {item.flight_id: co.fit_labels(item, anchor) for item in series}
    labels_path.write_text(json.dumps({"schema": co.LABEL_SCHEMA, "airport": AIRPORT, "flights": flights}))
    return series, config, co.ClosureLabels(AIRPORT, flights)


def _terminal_contexts():
    return {(AIRPORT, RUNWAY): AssessmentContext(
        benchmark="lpv", airport=AIRPORT, runway=RUNWAY, threshold_lat=35.8745003, threshold_lon=-78.802002,
        runway_course_deg=45.0, runway_width_m=45.72, runway_source="faa_nasr_apt_rwy",
        runway_source_cycle="2026-08-06", procedure_source="faa_cifp_path_point",
        procedure_source_cycle="2026-08-06", threshold_elevation_hae_m=141.86, threshold_elevation_msl_m=111.86,
        threshold_crossing_height_m=15.0, lpv_course_width_m=106.75)}


def _provenance():
    return {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
            "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64, "source_records": []}]}


def test_closure_config_contract(tmp_path):
    labels = tmp_path / "labels.json"
    config = _closure_config(labels)
    assert target_contract(config) == "closure-v1-slowness4-height4"
    assert loss_component_names(config) == ("state", "final_time", "kinematic", "terminal")
    assert co.decision_width(config) == 14 and len(co.decision_names(config)) == 14
    name = run_display_name(config.to_dict())
    assert name.startswith("closure ·") and "closed-form" in name and "closure-v1" in name
    assert f"labels={labels.parent.name}/labels.json" in name          # the file, with its directory
    assert "timing-scale" in run_display_name(_closure_config(labels, closure_timing_scale_s=30.0).to_dict()) or \
        "closure-timing-scale-s=30" in run_display_name(_closure_config(labels, closure_timing_scale_s=30.0).to_dict())
    assert "closure-v1(closure-slowness-knots=8" in run_display_name(_closure_config(labels, closure_slowness_knots=8).to_dict())
    with pytest.raises(ValueError, match="closure_labels_path"):
        TSConfig(prediction_output=PREDICTION_CLOSURE, **TINY)
    with pytest.raises(ValueError, match="knot widths"):
        _closure_config(labels, closure_slowness_knots=6)
    with pytest.raises(ValueError, match="ENU chart"):
        _closure_config(labels, coordinate_frame="airport-enu")
    with pytest.raises(ValueError, match="normalized horizon"):
        _closure_config(labels, horizon_mode="full")
    with pytest.raises(ValueError, match="checkpoint_selection_metric"):
        _closure_config(labels, checkpoint_selection_metric="fixed-anchor-common-grid-ade")
    with pytest.raises(ValueError, match="random_train_anchor"):
        _closure_config(labels, random_train_anchor=True)
    with pytest.raises(ValueError, match="belongs to the closure output"):
        TSConfig(closure_labels_path=str(labels), **TINY)
    # A serialised closure config must carry its own fields.
    data = config.to_dict()
    del data["closure_slowness_knots"]
    with pytest.raises(ValueError, match="closure_slowness_knots"):
        TSConfig.from_dict(data)


def test_labels_round_trip_through_the_decision_vector(tmp_path):
    series, config, labels = _closure_series(tmp_path, n_flights=2)
    label = labels.flights[series[0].flight_id]
    assert label["valid"] is True and label["geometry"]["params"]["canonical"] is True
    vector = co.decision_from_label(label, config)
    assert vector.shape == (14,)
    decision = co.split_decision(vector, config)
    params = label["geometry"]["params"]
    assert decision.d_join_m == params["d_join"] and decision.via_d_m == params["via_d"] and decision.via_xt_m == params["via_xt"]
    assert decision.via_heading_rel_rad == pytest.approx(params["via_heading_rel"])
    assert np.allclose(decision.slowness_knots, label["profile"]["4"]["slowness_knots"])
    assert np.allclose(decision.height_knots, label["profile"]["4"]["height_knots"]) and decision.height_knots[-1] == 0.0
    # The label carries both profile widths, each with its own residuals.
    assert set(label["profile"]) == {"4", "8"} and label["profile"]["8"]["time_error_s"] <= label["profile"]["4"]["time_error_s"] + 1e-9
    # The context rows: valid flights carry their vector, unknown flights a zero vector and weight 0.
    context = co.label_context(series[0], labels, config)
    assert tuple(context) == co.CLOSURE_CONTEXT_KEYS
    assert np.allclose(context[co.CONTEXT_DECISION], vector.astype(np.float32)) and context[co.CONTEXT_VALID] == 1.0
    assert context[co.CONTEXT_PATH_LENGTH] == pytest.approx(label["path_length_m"])
    assert context[co.CONTEXT_COURSE] == pytest.approx(float(series[0].scenario.target.psi))
    absent = co.label_context(series[1], co.ClosureLabels(AIRPORT, {}), config)
    assert absent[co.CONTEXT_VALID] == 0.0 and not absent[co.CONTEXT_DECISION].any()
    with pytest.raises(ValueError, match="fitted for 'KSJC'"):
        co.label_context(series[1], co.ClosureLabels("KSJC", labels.flights), config)
    # The airport is compared the way FlightSeries.airport normalises it.
    (tmp_path / "lower.json").write_text(json.dumps({"schema": co.LABEL_SCHEMA, "airport": " krdu ", "flights": labels.flights}))
    assert co.load_labels(tmp_path / "lower.json").airport == AIRPORT
    with pytest.raises(ValueError, match="schema"):
        (tmp_path / "bad.json").write_text(json.dumps({"schema": "other", "airport": AIRPORT, "flights": {}}))
        co.load_labels(tmp_path / "bad.json")


def test_reconstruction_is_a_valid_trajectory_that_ends_at_the_threshold(tmp_path):
    series, config, labels = _closure_series(tmp_path, n_flights=2)
    item = series[0]
    anchor = config.seq_len - 1
    vector = co.decision_from_label(labels.flights[item.flight_id], config)
    psi = float(item.scenario.target.psi)
    drawn = co.reconstruct(vector, item.values[anchor], psi, config)
    assert drawn.construction == cg.KIND_VIA_DUBINS
    assert np.all(np.diff(drawn.offsets_s) > 0.0) and drawn.offsets_s[0] > 0.0
    assert np.allclose(drawn.values[-1, :3], 0.0, atol=1e-6)                 # the threshold, height 0
    assert drawn.values.shape[1] == len(ch.CHANNELS)
    # Velocities are the chart derivatives of the positions (tangent × ground speed).
    speed = np.hypot(drawn.values[:, 3], drawn.values[:, 4])
    decision = co.split_decision(vector, config)
    assert cp.SPEED_MIN_MPS - 1e-6 <= speed.min() and speed.max() <= cp.SPEED_MAX_MPS + 1e-6
    assert drawn.final_time_s == pytest.approx(
        cp.times_from_slowness(drawn.path.arc / drawn.path.length, drawn.path.length, decision.slowness_knots)[-1], abs=1e-3)
    positions = np.concatenate([item.values[anchor][None, :2], drawn.values[:, :2]])
    times = np.concatenate([[0.0], drawn.offsets_s])
    derived = np.diff(positions, axis=0) / np.diff(times)[:, None]
    assert np.median(np.hypot(*(derived - drawn.values[:, 3:5]).T) / speed) < 0.15
    # The replay grid: the reconstruction on the target grid's fractions of its duration.
    sampled, durations = co.sample_at_progress(drawn, config.pred_len)
    assert sampled.shape == (config.pred_len, 6) and durations.sum() == pytest.approx(drawn.final_time_s)
    assert np.allclose(sampled[-1, :3], 0.0, atol=1e-6)
    # A decision no via-Dubins path can draw falls back to the plain CSC, then the straight line.
    at_anchor = vector.copy()
    a = cg.AnchorPose.from_state(item.values[anchor], psi)
    at_anchor[1], at_anchor[2] = a.d, a.xt
    at_anchor[3], at_anchor[4] = math.cos(cg.wrap_angle(a.heading - psi) + 0.3), math.sin(cg.wrap_angle(a.heading - psi) + 0.3)
    fallback = co.reconstruct(at_anchor, item.values[anchor], psi, config)
    assert fallback.construction == co.KIND_VIA_AT_ANCHOR and np.allclose(fallback.values[-1, :2], 0.0, atol=1e-6)
    # The head's outputs stay inside what the family draws, whatever the network emits.
    raw = torch.randn(64, co.decision_width(config)) * 50.0
    decoded = co.decode_raw(raw, config)
    assert decoded[:, 0].min() >= cg.D_JOIN_MIN_M and decoded[:, 0].max() <= cg.D_JOIN_MIN_M + co.D_JOIN_MAX_M
    assert decoded[:, 1:3].abs().max() <= co.VIA_MAX_M
    assert torch.allclose(decoded[:, 3:5].norm(dim=1), torch.ones(64), atol=1e-3)
    small = co.decode_raw(torch.zeros(1, co.decision_width(config)), config)   # the origin: softened, finite
    assert torch.isfinite(small).all() and small[0, 3:5].norm() < 1.0
    assert (1.0 / decoded[:, 5:5 + 5]).min() >= cp.SPEED_MIN_MPS - 1e-6 and (1.0 / decoded[:, 5:5 + 5]).max() <= cp.SPEED_MAX_MPS + 1e-6
    # The straight-line fallback: a decision whose via-Dubins and plain CSC both fail
    # cannot be produced by the bounded head, so it is exercised on the constructions.
    assert cg.straight_path(a).kind == cg.KIND_STRAIGHT
    # A via inside one turn radius of the anchor is not a decision: a straight-in
    # label puts it AT the anchor, and a predicted via 60 m off with 2° of heading error
    # would otherwise be a full circle. The decoder draws the plain CSC instead.
    near = vector.copy()
    off = 60.0 * cg._unit(a.heading + math.pi / 2)
    near[1], near[2] = cg.runway_axes_np(a.position[0] + off[0], a.position[1] + off[1], psi)
    rel = cg.wrap_angle(a.heading - psi) + math.radians(2.0)
    near[3], near[4] = math.cos(rel), math.sin(rel)
    dropped = co.reconstruct(near, item.values[anchor], psi, config)
    looped = cg.via_dubins(a, psi, near[0], *cg.chart_from_axes_np(near[1], near[2], psi), cg.wrap_angle(psi + rel))
    assert dropped.construction == co.KIND_VIA_AT_ANCHOR
    assert dropped.path.length < 0.8 * looped.length and dropped.path.length < 1.2 * drawn.path.length


def test_drawing_from_the_labels_reproduces_the_truth(tmp_path):
    """The oracle arm: every synthetic flight drawn from its own label sits within the
    label's residuals of its truth on the package's common-time grid."""
    series, config, labels = _closure_series(tmp_path, n_flights=3)
    forecasts = forecast_closure_from_labels(series, config, labels)
    for item, forecast in zip(series, forecasts):
        assert forecast.prediction_output == PREDICTION_CLOSURE and forecast.controls is None
        metrics = observed_series_metrics(item, forecast)
        assert metrics["ade_m"] < 250.0 and abs(metrics["final_time_error_s"]) < 5.0
    assert all(f.closure_from_labels and f.closure_construction in (cg.KIND_VIA_DUBINS, co.KIND_VIA_AT_ANCHOR) for f in forecasts)
    with pytest.raises(KeyError, match="no closure label"):
        forecast_closure_from_labels(series, config, co.ClosureLabels(AIRPORT, {}))
    with pytest.raises(ValueError, match="fitted for 'KSJC'"):
        forecast_closure_from_labels(series, config, co.ClosureLabels("KSJC", labels.flights))


def test_the_loss_regresses_the_labels_and_skips_invalid_flights(tmp_path):
    series, config, labels = _closure_series(tmp_path, n_flights=2)
    normalizer = Normalizer.fit(series)
    model = build_model(config, normalizer)
    context = {key: torch.stack([torch.as_tensor(co.label_context(item, labels, config)[key]) for item in series])
               for key in co.CLOSURE_CONTEXT_KEYS}
    # PatchTST feeds the same head (its pooled tokens are enc_in · d_model wide too).
    patch_config = _closure_config(tmp_path / "closure_labels.json", model="patchtst")
    assert build_model(patch_config)(history := torch.zeros(2, patch_config.seq_len, patch_config.enc_in)).decision.shape == (2, 14)
    history = torch.zeros(2, config.seq_len, config.enc_in)
    prediction = model(history, context)
    anchor = torch.zeros(2, len(ch.CHANNELS))
    y = torch.zeros(2, config.pred_len, len(ch.CHANNELS))
    weights = torch.ones_like(y)
    flight_weights = torch.ones(2)
    components = prediction_loss_components(prediction, anchor, y, weights, torch.ones(2), flight_weights, config, normalizer, context)
    assert set(components.tensors()) == {"state", "final_time", "kinematic", "terminal"}
    assert components.terminal.item() == 0.0 and components.total.item() > 0.0
    components.total.backward()
    assert all(p.grad is not None for p in model.head.parameters())
    assert components.diagnostics["closure_valid_share"].item() == 1.0
    # A flight without a valid label is in the batch but out of the loss.
    context[co.CONTEXT_VALID] = torch.tensor([1.0, 0.0])
    partial = prediction_loss_components(model(history, context), anchor, y, weights, torch.ones(2), flight_weights, config, normalizer, context)
    assert partial.diagnostics["closure_valid_share"].item() == 0.5
    with pytest.raises(ValueError, match="label context"):
        prediction_loss_components(prediction, anchor, y, weights, torch.ones(2), flight_weights, config, normalizer, None)


def test_train_checkpoint_predict_export_and_evaluate_one_closure_run(tmp_path):
    series, config, labels = _closure_series(tmp_path, n_flights=8)
    result = train(series, config, output_dir=tmp_path / "run", data_provenance=_provenance(), verbose=False)
    assert (tmp_path / "run" / "checkpoint.pt").is_file()
    model, loaded, normalizer, payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded.prediction_output == PREDICTION_CLOSURE and loaded.closure_labels_path == config.closure_labels_path
    assert payload["target_contract"] == "closure-v1-slowness4-height4"
    records, metrics = [], []
    for index, item in enumerate(series[:4]):
        forecast = forecast_approach(model, item, loaded, normalizer, device=torch.device("cpu"))
        assert forecast.prediction_output == PREDICTION_CLOSURE and forecast.controls is None
        assert forecast.values.shape[1] == len(ch.CHANNELS) and forecast.final_time_s == forecast.predicted_final_time_s
        records.append(build_prediction_record(item, forecast, index=index, model_name=loaded.model, horizon_mode=loaded.horizon_mode))
        metrics.append(observed_series_metrics(item, forecast))
    out = tmp_path / "pred"
    write_batch(records, output_dir=out, config_dict=loaded.to_dict(), flight_metrics=metrics)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["mode"].endswith(":closure:test") and all(row["ade_m"] is not None for row in summary["results"])
    states = json.loads((out / summary["results"][0]["states_file"]).read_text())
    assert states["control_segments"] == [] and states["source"]["predictionOutput"] == PREDICTION_CLOSURE
    assert np.allclose([states["predicted_states"][-1]["lat"], states["predicted_states"][-1]["lon"]],
                       [records[0].eval_record["target_state"]["lat"], records[0].eval_record["target_state"]["lon"]], atol=1e-5)
    assert states["source"]["closureFromLabels"] is False and states["source"]["closureConstruction"] in (cg.KIND_VIA_DUBINS, co.KIND_VIA_AT_ANCHOR)
    loaded_records = load_records(out)
    report = evaluate_batch(loaded_records, contexts=_terminal_contexts())
    assert report["total"] == 4 and report["solved"] == 4
    assert (tmp_path / "run" / "history.json").is_file() and isinstance(result, dict)
    # A labels file for none of the flights is refused before a single batch is built.
    (tmp_path / "other.json").write_text(json.dumps({"schema": co.LABEL_SCHEMA, "airport": AIRPORT, "flights": {}}))
    with pytest.raises(ValueError, match="carries none of these"):
        train(series, _closure_config(tmp_path / "other.json"), output_dir=tmp_path / "run2",
              data_provenance=_provenance(), verbose=False)
    invalid = {k: {**v, "valid": False} for k, v in labels.flights.items()}
    (tmp_path / "invalid.json").write_text(json.dumps({"schema": co.LABEL_SCHEMA, "airport": AIRPORT, "flights": invalid}))
    with pytest.raises(ValueError, match="nothing to regress"):
        train(series, _closure_config(tmp_path / "invalid.json"), output_dir=tmp_path / "run3",
              data_provenance=_provenance(), verbose=False)
