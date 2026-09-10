"""End to end: train on synthetic flights, checkpoint, predict, grade — a plumbing check, never a quality check.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_TS_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]

_CLI_SPEC = importlib.util.spec_from_file_location(
    "ts_transformer_cli_test", _TS_DIR / "__main__.py"
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
ts_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(ts_cli)

import ts_transformer.cli.common as cli_common  # noqa: E402
import ts_transformer.cli.evaluate_fit as cli_evaluate_fit  # noqa: E402
import ts_transformer.inference.evaluation_protocol as evaluation_protocol  # noqa: E402
from ts_transformer.outputs import ForecastOptions
from ts_transformer.config import (
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    PREDICTION_CONTROL,
    TSConfig,
)
from ts_transformer.data.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from ts_transformer.data.dataset import Normalizer, build_series
from ts_transformer.data.splits import split_by_flight
from evaluation.metrics import evaluate_batch  # noqa: E402
from evaluation.records import load_records
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.inference.forecast import forecast_approach
from ts_transformer.backbone.adapters import build_model
from ts_transformer.data.synthetic import synthetic_arrivals  # noqa: E402
# Imported, never restated: a schema version pinned by hand in a fixture is a version
# the fixture cannot check, and this one gates every loader that reads the roster.
from ts_transformer.outputs.state.loss import STATE_LOSS_COMPONENT_NAMES
from ts_transformer.tests.support import fake_data_provenance, terminal_contexts
from ts_transformer.training.train import (  # noqa: E402
    CHECKPOINT_METADATA_SCHEMA, FIT_EVALUATION_NAME, FIT_EVALUATION_SCHEMA,
    evaluate_fit_splits, load_checkpoint, train,
)

AIRPORT, RUNWAY = "KRDU", "05L"


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


L1_NATIVE32_CHECKPOINT = _REPO_ROOT / "4dTrajectory/outputs/KRDU/experiments/l1_lowdim_20260907/L1_native32/checkpoint.pt"


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
        data_provenance=fake_data_provenance(),
        verbose=False,
    )

    model, loaded_config, normalizer, payload = load_checkpoint(tmp_path / "checkpoint.pt")
    forecast = forecast_approach(
        model,
        series[0],
        loaded_config,
        normalizer,
        device=torch.device("cpu"),
        options=ForecastOptions(truncate=False),
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
    provenance = fake_data_provenance()
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
        cli_evaluate_fit, "load_checkpoint",
        lambda _path: (build_model(config), config, normalizer, payload),
    )
    monkeypatch.setattr(cli_common, "arrival_data_provenance", lambda _data: provenance)
    monkeypatch.setattr(cli_evaluate_fit, "require_matching_data_provenance", lambda *_args: None)
    loaded_keys = None

    def load_selected(_data, *, include_flight_keys=None):
        nonlocal loaded_keys
        loaded_keys = include_flight_keys
        return flights

    monkeypatch.setattr(cli_evaluate_fit, "load_flight_dicts", load_selected)
    monkeypatch.setattr(cli_evaluate_fit, "dataset_flight_key", lambda flight, _index: flight["key"])
    monkeypatch.setattr(cli_evaluate_fit, "build_series", lambda *_args, **_kwargs: (series, report))
    monkeypatch.setattr(cli_evaluate_fit, "resolve_device", lambda _device: torch.device("cpu"))

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
            data_provenance=fake_data_provenance(),
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
            # WHICH number it stepped on — the axis is read off the config here, so this
            # mirror cannot claim a metric the run did not use.
            "metric": config.lr_plateau_metric,
        },
        "split_sha256": {
            split: hashlib.sha256("\n".join(sorted(payload["split"][split])).encode()).hexdigest()
            for split in ("train", "val", "test")
        },
    }
    # Imported, never restated: the state objective's component list is one source.
    assert set(summary["history"][0]["train_components"]) == set(STATE_LOSS_COMPONENT_NAMES)
    assert set(summary["history"][0]["val_components"]) == set(STATE_LOSS_COMPONENT_NAMES)
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

    report = evaluate_batch(load_records(out), contexts=terminal_contexts())
    assert report["total"] == 4
    # The gate outcome is not asserted — an undertrained model on synthetic data may or may
    # not land inside 106.75 m, and pinning that would make this a flaky quality test.
    assert "lateral_m" in report and "success_rate" in report

    history = json.loads((tmp_path / "run" / "history.json").read_text(encoding="utf-8"))
    assert history["config"]["model"] == model_name
    assert len(history["history"]) == 2
    assert history["data_provenance"]["source_record_count"] == len(series)


def test_a_stored_config_carrying_a_retired_field_still_loads_and_an_unknown_one_does_not():
    """Checkpoints written before a field was retired keep loading; a genuinely unknown key
    is still refused, so the retired list stays honest."""
    from ts_transformer.config import RETIRED_CONSTANT_FIELDS, RETIRED_SERIALIZED_FIELDS
    from dataclasses import fields as dataclass_fields
    live = {field.name for field in dataclass_fields(TSConfig)}
    retired = set(RETIRED_SERIALIZED_FIELDS) | set(RETIRED_CONSTANT_FIELDS)
    assert not (retired & live), "a retired field is still declared"
    assert not (set(RETIRED_SERIALIZED_FIELDS) & set(RETIRED_CONSTANT_FIELDS)), (
        "a field cannot be both unread and a measured constant"
    )
    stored = TSConfig(prediction_output=PREDICTION_CONTROL).to_dict()
    for name in RETIRED_SERIALIZED_FIELDS:
        stored[name] = 0.25
    assert TSConfig.from_dict(stored).prediction_output == PREDICTION_CONTROL
    with pytest.raises(TypeError):
        TSConfig.from_dict({**stored, "never_a_field": 1})


def test_a_measured_constant_field_is_dropped_at_its_constant_and_refused_anywhere_else():
    """The other retirement kind: these three WERE read, so a different value is history.

    `procedure_loss_{lateral,vertical}_scale_m` and `closure_timing_scale_s` are loss
    denominators. They were retired because nothing on disk ever moved them — not because
    nothing read them — so dropping a non-default value by name would silently reinterpret
    an artifact produced under a scale this build no longer has.
    """
    from ts_transformer.config import RETIRED_CONSTANT_FIELDS

    base = TSConfig(prediction_output=PREDICTION_CONTROL).to_dict()
    at_constant = {**base, **RETIRED_CONSTANT_FIELDS}
    assert TSConfig.from_dict(at_constant).prediction_output == PREDICTION_CONTROL

    for name, constant in RETIRED_CONSTANT_FIELDS.items():
        with pytest.raises(ValueError, match=f"{name}=") as info:
            TSConfig.from_dict({**at_constant, name: constant * 2})
        assert repr(constant * 2) in str(info.value) and "retired" in str(info.value)


@pytest.mark.skipif(not L1_NATIVE32_CHECKPOINT.is_file(), reason="the L1 native32 checkpoint is not on this machine")
def test_the_l1_native32_checkpoint_written_with_the_retired_fields_still_loads():
    """The canary against the serialized contract: a REAL artifact from before a field was
    retired (the synthetic test above pins the rule, not an artifact) — its class, its
    strict state dict, and a config that round-trips without the retired keys."""
    from ts_transformer.config import RETIRED_SERIALIZED_FIELDS
    from ts_transformer.outputs.control.heads import ControlOutputModel
    model, config, _normalizer, payload = load_checkpoint(L1_NATIVE32_CHECKPOINT)
    assert isinstance(model, ControlOutputModel) and config.n_segments == 32
    assert set(RETIRED_SERIALIZED_FIELDS) <= set(payload["config"]), "the canary lost its point: pick an older artifact"
    assert not set(RETIRED_SERIALIZED_FIELDS) & set(config.to_dict())
    assert TSConfig.from_dict(config.to_dict()) == config
