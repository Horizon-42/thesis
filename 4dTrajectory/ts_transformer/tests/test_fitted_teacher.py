"""The imitation term's teacher axis (latent-intent design §六 L5.a).

`control_imitation_target="fitted"` swaps the schedule the imitation term imitates: instead
of inverting one out of the flown track at every sample, the dataset looks the flight up in
a table fitted THROUGH the rollout. L0 measured why: the inverted schedule flown open-loop
lands 2.5–7.8 km from the truth it was read off, the fitted one 88–433 m.

A table is specific to a width, an anchor and each flight's supervised duration, and a
flight it does not carry has no teacher at all — so everything here is about refusing the
wrong table loudly, at the dataset build, rather than training on a silently wrong target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_IMITATION_TARGET_FITTED,
    CONTROL_IMITATION_TARGET_INVERSE_DYNAMICS,
    CONTROL_RECIPE_NAMES,
    CONTROL_RECIPE_CUSTOM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
    control_recipe_overrides,
)
from control.basis_fit import FITTED_TEACHER_SCHEMA, DURATION_UNIFORM
from dataset import (
    ARRIVAL_DATA_PROVENANCE_SCHEMA,
    FixedAnchorTrajectoryWindows,
    Normalizer,
    build_series,
    truth_duration_s,
)
from forecast import forecast_approaches
from run_naming import run_display_name
from synthetic import synthetic_arrivals
from train import control_imitation_mse, load_checkpoint, train

AIRPORT, RUNWAY = "KRDU", "05L"
N_SEGMENTS, SEQ_LEN = 4, 8


def _config(**overrides) -> TSConfig:
    """simple-v3's supervision shape, shrunk to a CPU test: the imitation term is live."""
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_recipe_name=CONTROL_RECIPE_CUSTOM,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_imitation_loss_weight=64.0,
        seq_len=SEQ_LEN, n_segments=N_SEGMENTS, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=2, patience=2, batch_size=8, dropout=0.0,
        val_fraction=0.25, test_fraction=0.25,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def _provenance() -> dict:
    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }


def _series(config: TSConfig, n_flights: int = 12):
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3), config, airport=AIRPORT
    )
    return series


def _table(path: Path, series, *, n_segments: int = N_SEGMENTS, anchor_index: int = SEQ_LEN - 1,
           duration_delta_s: float = 0.0, drop: int = 0, **stamps) -> Path:
    """A teacher table in the fitter's own schema, one distinctive schedule per flight."""
    anchor = SEQ_LEN - 1
    flights = {}
    for index, item in enumerate(series[drop:]):
        # Every flight gets its own constant schedule inside the box, so a batch that
        # looked one flight up under another flight's key would be visible.
        value = 0.1 + 0.05 * index
        flights[item.flight_id] = {
            "controls": np.full((n_segments, 3), value).tolist(),
            "total_duration_s": truth_duration_s(item, anchor) + duration_delta_s,
            "fit_ade_m": 12.0, "seed_ade_m": 34.0, "best_step": 7, "split": "train",
        }
    payload = {
        "schema": FITTED_TEACHER_SCHEMA,
        "airport": AIRPORT,
        "n_segments": n_segments,
        "duration_mode": DURATION_UNIFORM,
        "anchor_index": anchor_index,
        "flights": flights,
        **stamps,
    }
    path.write_text(json.dumps(payload))
    return path


# ── the config axis ─────────────────────────────────────────────────────────

def test_the_default_and_every_named_recipe_imitate_the_inverse_dynamics():
    """The recipes are LITERALS: a published simple-v* comparison names one teacher."""
    assert TSConfig().control_imitation_target == CONTROL_IMITATION_TARGET_INVERSE_DYNAMICS
    assert TSConfig().control_fitted_teacher_path == ""
    for name in CONTROL_RECIPE_NAMES:
        overrides = control_recipe_overrides(name)
        if name == CONTROL_RECIPE_CUSTOM:
            assert overrides == {}
            continue
        assert overrides["control_imitation_target"] == (
            CONTROL_IMITATION_TARGET_INVERSE_DYNAMICS
        )


@pytest.mark.parametrize("overrides, message", [
    ({"control_imitation_target": CONTROL_IMITATION_TARGET_FITTED}, "set "),
    ({"control_fitted_teacher_path": "teacher.json"}, "belongs to"),
    ({"control_imitation_target": "fitted-and-inverted"}, "unknown control_imitation_target"),
    ({"control_imitation_target": CONTROL_IMITATION_TARGET_FITTED,
      "control_fitted_teacher_path": "teacher.json",
      "random_train_anchor": True}, "fitted AT the fixed anchor"),
])
def test_the_config_refuses_an_incoherent_teacher(overrides, message):
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_a_state_run_has_no_control_schedule_to_imitate():
    with pytest.raises(ValueError, match="emits none"):
        TSConfig(
            prediction_output=PREDICTION_STATE,
            control_imitation_target=CONTROL_IMITATION_TARGET_FITTED,
            control_fitted_teacher_path="teacher.json",
        )


def test_only_the_non_default_teacher_names_the_run():
    default = _config()
    assert "imit-target" not in run_display_name(default.to_dict())
    fitted = _config(control_imitation_target=CONTROL_IMITATION_TARGET_FITTED,
                     control_fitted_teacher_path="basis_fit.json")
    assert "imit-target=fitted" in run_display_name(fitted.to_dict())
    # A stored config from before the axis existed reads as the default, i.e. unchanged.
    stored = default.to_dict()
    del stored["control_imitation_target"], stored["control_fitted_teacher_path"]
    assert run_display_name(stored) == run_display_name(default.to_dict())


# ── the dataset swap ────────────────────────────────────────────────────────

def test_the_dataset_serves_the_table_with_unit_weights_and_the_loss_consumes_them(tmp_path):
    """Under ``fitted`` the batch's target IS the table's row — no inversion, no mask."""
    series = _series(_config())
    table = _table(tmp_path / "teacher.json", series)
    config = _config(control_imitation_target=CONTROL_IMITATION_TARGET_FITTED,
                     control_fitted_teacher_path=str(table))
    windows = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    indices = np.arange(len(series))
    _x, _y, _w, _final_time, _fw, dynamics = windows.batch(indices)

    stored = json.loads(table.read_text())["flights"]
    for row, item in enumerate(series):
        assert np.allclose(
            dynamics["reference_controls"][row].numpy(), stored[item.flight_id]["controls"]
        )
    assert torch.equal(
        dynamics["reference_control_weight"], torch.ones(len(series), N_SEGMENTS, dtype=torch.float64)
    )

    # The inverse-dynamics teacher is a different target in the same slot: the two are not
    # the same supervision, and the loss reads both without knowing which it got.
    inverted = FixedAnchorTrajectoryWindows(series, _config(), Normalizer.fit(series))
    _x, _y, _w, _t, _fw, inverse_dynamics = inverted.batch(indices)
    assert not torch.allclose(
        inverse_dynamics["reference_controls"], dynamics["reference_controls"]
    )

    from prediction_outputs import ControlPrediction
    prediction = ControlPrediction(
        controls=torch.zeros(len(series), N_SEGMENTS, 3),
        segment_durations=torch.ones(len(series), N_SEGMENTS),
        final_time_s=torch.full((len(series),), float(N_SEGMENTS)),
    )
    value = control_imitation_mse(prediction, config, dynamics)
    assert value.shape == (len(series),) and bool(torch.all(torch.isfinite(value)))


@pytest.mark.parametrize("kwargs, message", [
    ({"drop": 1}, "covers 11 of 12 flights"),
    ({"duration_delta_s": 1e-3}, "different horizon"),
    ({"n_segments": 8}, "fitted at N=8"),
    ({"anchor_index": 11}, "fitted at anchor 11"),
])
def test_the_dataset_build_refuses_a_table_that_is_not_this_cohorts(tmp_path, kwargs, message):
    series = _series(_config())
    table = _table(tmp_path / "teacher.json", series, **kwargs)
    config = _config(control_imitation_target=CONTROL_IMITATION_TARGET_FITTED,
                     control_fitted_teacher_path=str(table))
    with pytest.raises(ValueError, match=message):
        FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))


@pytest.mark.parametrize("field, value, message", [
    ("schema", "ts-basis-fit-v1", "expected fitted-teacher schema"),
    ("duration_mode", "free", "uniformly partitioned"),
])
def test_the_loader_refuses_another_files_schema(tmp_path, field, value, message):
    series = _series(_config())
    table = _table(tmp_path / "teacher.json", series)
    payload = json.loads(table.read_text())
    payload[field] = value
    table.write_text(json.dumps(payload))
    config = _config(control_imitation_target=CONTROL_IMITATION_TARGET_FITTED,
                     control_fitted_teacher_path=str(table))
    with pytest.raises(ValueError, match=message):
        FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))


# ── training end to end ─────────────────────────────────────────────────────

def test_training_from_the_table_stamps_it_and_predicts_without_it(tmp_path):
    """The checkpoint says which table taught it; prediction never needs the table back."""
    torch.manual_seed(0)
    series = _series(_config())
    table = _table(tmp_path / "teacher.json", series)
    config = _config(control_imitation_target=CONTROL_IMITATION_TARGET_FITTED,
                     control_fitted_teacher_path=str(table))
    out = tmp_path / "fitted_run"
    train(series, config, output_dir=out, data_provenance=_provenance(), verbose=False)

    digest = json.loads((out / "checkpoint_metadata.json").read_text())["fitted_teacher"]
    import hashlib
    assert digest == {
        "path": str(table),
        "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "n_segments": N_SEGMENTS,
        "anchor_index": SEQ_LEN - 1,
        "flights": len(series),
    }
    history = json.loads((out / "history.json").read_text())["history"]
    assert all(np.isfinite(row["train_components"]["imitation"]) for row in history)

    # The teacher is TRAINING-only: predict loads the checkpoint and forecasts with the
    # table deleted, because nothing downstream of the loss ever reads it.
    model, loaded, normalizer, payload = load_checkpoint(out / "checkpoint.pt")
    assert loaded.control_imitation_target == CONTROL_IMITATION_TARGET_FITTED
    assert payload["fitted_teacher"] == digest
    table.unlink()
    forecasts = forecast_approaches(model, series[:2], loaded, normalizer,
                                    device=torch.device("cpu"))
    assert len(forecasts) == 2 and all(f.n_steps > 1 for f in forecasts)


def test_the_two_teachers_train_to_different_imitation_numbers(tmp_path):
    """Same seed, same data, one axis: the component is finite and it is not the same."""
    series = _series(_config())
    table = _table(tmp_path / "teacher.json", series)
    components = {}
    for label, overrides in (
        ("inverse-dynamics", {}),
        ("fitted", {"control_imitation_target": CONTROL_IMITATION_TARGET_FITTED,
                    "control_fitted_teacher_path": str(table)}),
    ):
        torch.manual_seed(0)
        config = _config(**overrides)
        train(series, config, output_dir=tmp_path / label,
              data_provenance=_provenance(), verbose=False)
        history = json.loads((tmp_path / label / "history.json").read_text())["history"]
        components[label] = [row["train_components"]["imitation"] for row in history]
    assert all(np.isfinite(values).all() and (np.asarray(values) > 0.0).all()
               for values in components.values())
    assert components["inverse-dynamics"] != components["fitted"]
