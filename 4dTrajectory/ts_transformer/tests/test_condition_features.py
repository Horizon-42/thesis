"""How the airframe is written into the control head's condition vector (design §11, N4).

``control_condition_features`` is ``raw`` (every stored run) or ``ratios``: the same mass and
polar, with the installed thrust and the wing area replaced by the thrust-to-weight ratio and
the 1-g stall speed. What must hold for an N4 arm to measure PRESENTATION and nothing else:

* each set's channels divide by the unit their names state;
* the two sets carry the same information (one determines the other) and have one width, so
  an arm that moves only the set starts from its twin's weights;
* every surface that builds a condition row — the batch context, the forecast, the probe —
  writes the configured set, and none may omit it;
* the config refuses it where it means nothing, every named recipe pins the raw set, and the
  name and the recipe summary move only off the default.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aircraft.aero_params import GRAVITY_M_S2, RHO0_KG_M3
from ts_transformer.backbone.adapters import build_model
from ts_transformer.config import (
    CONTROL_CONDITION_FEATURES_RATIOS,
    CONTROL_CONDITION_FEATURES_RAW,
    CONTROL_DURATION_UNIFORM,
    CONTROL_RECIPE_CUSTOM,
    CONTROL_RECIPE_NAMES,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    PREDICTION_CONTROL,
    TSConfig,
    control_recipe,
    recipe_settings,
)
from ts_transformer.data.batch_contract import unpack_batch
from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows, Normalizer, build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.outputs.conditioning import (
    CONDITION_FEATURE_SETS,
    CONDITION_WIDTH,
    condition_names,
    condition_vector,
)
from ts_transformer.outputs.control.forecast import dynamics_batch
from ts_transformer.outputs.control.supervision import probe_dynamics
from ts_transformer.outputs.dynamics.context import dynamics_arrays
from ts_transformer.run_naming import run_display_name, run_parameter_rows, run_slug

AIRPORT, RUNWAY = "KRDU", "05L"
RATIOS = {"control_condition_features": CONTROL_CONDITION_FEATURES_RATIOS}
#: A light bizjet, the median narrowbody, a widebody: ``(mass, installed thrust, polar)``.
AIRFRAMES = (
    (6_800.0, 22_000.0, dict(S=30.0, Cl_max=2.2, Cd0=0.02, k=0.04, stall_threshold=0.9, k_stall=0.1)),
    (66_300.0, 233_000.0, dict(S=124.6, Cl_max=2.7, Cd0=0.02, k=0.04, stall_threshold=0.9, k_stall=0.1)),
    (251_000.0, 1_026_000.0, dict(S=427.8, Cl_max=2.4, Cd0=0.02, k=0.04, stall_threshold=0.9, k_stall=0.1)),
)


def _config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        n_segments=4, seq_len=8, d_model=16, d_ff=32, n_heads=4, e_layers=1,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def _series(config: TSConfig, n_flights: int = 2):
    series, report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3), config, airport=AIRPORT
    )
    assert report.built == n_flights, report.format()
    return series


def _named(features: str, mass_kg: float, thrust_n: float, aero) -> dict[str, float]:
    vector = condition_vector(mass_kg, thrust_n, aero, features=features)
    return dict(zip(condition_names(features), vector.tolist()))


# ── the two sets ─────────────────────────────────────────────────────────────


def test_each_ratio_channel_divides_by_the_unit_its_name_states():
    """An airframe whose installed thrust IS its weight and whose 1-g sea-level stall speed
    IS 100 m/s must read 1.0 on both new channels (the polar's divisors as in the raw set)."""
    polar = dict(Cl_max=3.0, Cd0=0.1, k=0.1, stall_threshold=0.8, k_stall=0.2)
    mass = 100_000.0
    weight = mass * GRAVITY_M_S2
    area = 2.0 * weight / (RHO0_KG_M3 * 100.0**2 * polar["Cl_max"])
    named = _named(CONTROL_CONDITION_FEATURES_RATIOS, mass, weight, SimpleNamespace(S=area, **polar))
    assert named["stall_threshold"] == pytest.approx(0.8)
    for name, value in named.items():
        if name != "stall_threshold":
            assert value == pytest.approx(1.0, rel=1e-6), name


def test_the_two_sets_are_the_same_information_and_one_width():
    """Given Cl_max, (m, T_max/W, V_s) determines (m, T_max, S): the ratios set loses
    nothing the raw set carried, so an N4 arm measures how the airframe is WRITTEN."""
    assert {len(channels) for channels in CONDITION_FEATURE_SETS.values()} == {CONDITION_WIDTH}
    for mass, thrust, polar in AIRFRAMES:
        aero = SimpleNamespace(**polar)
        raw = _named(CONTROL_CONDITION_FEATURES_RAW, mass, thrust, aero)
        ratios = _named(CONTROL_CONDITION_FEATURES_RATIOS, mass, thrust, aero)
        shared = set(raw) & set(ratios)
        assert shared == set(raw) - {"max_thrust_1MN", "wing_area_500m2"}
        assert all(raw[name] == ratios[name] for name in shared)
        m = ratios["mass_100t"] * 100_000.0
        stall_speed = ratios["stall_speed_100mps"] * 100.0
        cl_max = ratios["cl_max_3"] * 3.0
        recovered_thrust = ratios["thrust_to_weight"] * m * GRAVITY_M_S2
        recovered_area = 2.0 * m * GRAVITY_M_S2 / (RHO0_KG_M3 * stall_speed**2 * cl_max)
        # float32 channels: agreement to the vector's own precision.
        assert recovered_thrust / 1_000_000.0 == pytest.approx(raw["max_thrust_1MN"], rel=1e-6)
        assert recovered_area / 500.0 == pytest.approx(raw["wing_area_500m2"], rel=1e-6)


def test_a_ratios_head_starts_from_its_raw_twins_weights():
    """One width, one encoder shape: under one seed the two arms' initial parameters are
    identical, key for key and value for value."""
    torch.manual_seed(11)
    raw = build_model(_config()).state_dict()
    torch.manual_seed(11)
    ratios = build_model(_config(**RATIOS)).state_dict()
    assert raw.keys() == ratios.keys()
    assert all(torch.equal(raw[key], ratios[key]) for key in raw)


# ── every surface writes the configured set, and none may omit it ─────────────


def test_the_set_is_required_wherever_a_row_is_built():
    config = _config(**RATIOS)
    series = _series(config, n_flights=1)[0]
    mass, thrust, polar = AIRFRAMES[1]
    with pytest.raises(TypeError):
        condition_vector(mass, thrust, SimpleNamespace(**polar))
    with pytest.raises(TypeError):
        dynamics_arrays(series, config.seq_len - 1, parameterization=config.control_thrust_parameterization)


def test_the_batch_context_and_the_forecast_write_the_configured_set():
    config = _config(**RATIOS)
    series = _series(config)
    expected = np.stack([
        condition_vector(
            float(item.scenario.initial.m),
            float(item.scenario.aircraft.engine.max_thrust_total_n),
            item.scenario.aero,
            features=CONTROL_CONDITION_FEATURES_RATIOS,
        )
        for item in series
    ])
    windows = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    batch = unpack_batch(windows.batch(np.arange(len(series))))
    dynamics = batch[5]
    assert np.array_equal(dynamics["condition"].numpy(), expected)
    forecast = dynamics_batch(series, config.seq_len - 1, torch.device("cpu"), config)
    assert np.array_equal(forecast["condition"].numpy(), expected)
    raw = dynamics_batch(series, config.seq_len - 1, torch.device("cpu"), _config())
    assert not np.array_equal(raw["condition"].numpy(), expected)


def test_a_ratios_checkpoint_comes_back_reading_ratios(tmp_path):
    """Trained, saved and loaded: the set rides in the checkpoint's config and recipe, and
    the reloaded run's forecast rows are the ratios rows."""
    from ts_transformer.data.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
    from ts_transformer.training.train import load_checkpoint, train

    config = _config(**RATIOS, epochs=1, patience=1, batch_size=8, seq_len=20, n_segments=2,
                     final_time_scale_s=2.0, control_rollout_integrator_dt_s=0.5, device="cpu")
    series = _series(config, n_flights=12)
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
    model, loaded, _normalizer, _payload = load_checkpoint(tmp_path / "checkpoint.pt")
    metadata = json.loads((tmp_path / "checkpoint_metadata.json").read_text())
    assert loaded.control_condition_features == CONTROL_CONDITION_FEATURES_RATIOS
    assert metadata["control_recipe"]["condition_features"] == CONTROL_CONDITION_FEATURES_RATIOS
    rows = dynamics_batch(series, loaded.seq_len - 1, torch.device("cpu"), loaded)
    expected = dynamics_batch(series, config.seq_len - 1, torch.device("cpu"), config)
    assert torch.equal(rows["condition"], expected["condition"])
    assert model.condition_encoder[0].in_features == CONDITION_WIDTH


def test_the_probe_writes_its_condition_row_with_the_configured_set():
    """The raw probe row is the literal it was before the axis; the ratios row is the same
    airframe written the other way."""
    device = torch.device("cpu")
    raw = probe_dynamics(2, device, _config())["condition"][0]
    assert raw.tolist() == pytest.approx([0.66, 0.24, 0.2452, 0.9, 0.2, 0.4, 0.9, 0.5], rel=1e-6)
    ratios = dict(zip(
        condition_names(CONTROL_CONDITION_FEATURES_RATIOS),
        probe_dynamics(2, device, _config(**RATIOS))["condition"][0].tolist(),
    ))
    assert ratios["thrust_to_weight"] == pytest.approx(240_000.0 / (66_000.0 * GRAVITY_M_S2), rel=1e-6)
    stall = math.sqrt(2.0 * 66_000.0 * GRAVITY_M_S2 / (RHO0_KG_M3 * 122.6 * 2.7))
    assert ratios["stall_speed_100mps"] == pytest.approx(stall / 100.0, rel=1e-6)


# ── the config and the name ──────────────────────────────────────────────────


def test_the_axis_belongs_to_the_control_output_and_knows_its_values():
    assert TSConfig().control_condition_features == CONTROL_CONDITION_FEATURES_RAW
    with pytest.raises(ValueError, match="belongs to the control output"):
        TSConfig(**RATIOS)
    with pytest.raises(ValueError, match="control_condition_features"):
        _config(control_condition_features="dimensionless")


def test_every_named_recipe_pins_the_raw_set():
    for name in CONTROL_RECIPE_NAMES:
        if name == CONTROL_RECIPE_CUSTOM:
            continue
        settings = recipe_settings(name, keep_name=True)
        assert settings["control_condition_features"] == CONTROL_CONDITION_FEATURES_RAW, name
        with pytest.raises(ValueError, match="recipe fields are frozen"):
            TSConfig(**{**settings, **RATIOS})
    TSConfig(**{**recipe_settings("simple-v3", keep_name=False), **RATIOS})


def test_a_stored_config_without_the_field_reads_the_raw_set():
    stored = {key: value for key, value in _config().to_dict().items()
              if key != "control_condition_features"}
    assert TSConfig.from_dict(stored).control_condition_features == CONTROL_CONDITION_FEATURES_RAW


def test_the_control_recipe_names_the_set_only_off_its_default():
    """`experiments.pipeline` compares this dict for checkpoint reuse: writing the default
    would refuse every stored checkpoint."""
    assert "condition_features" not in control_recipe(_config())
    assert control_recipe(_config(**RATIOS))["condition_features"] == CONTROL_CONDITION_FEATURES_RATIOS


def test_the_name_moves_only_off_the_default():
    default = _config().to_dict()
    stored = {key: value for key, value in default.items() if key != "control_condition_features"}
    assert run_display_name(default) == run_display_name(stored)
    assert run_slug(default) == run_slug(stored)
    ratios = _config(**RATIOS).to_dict()
    assert "airframe=ratios" in run_display_name(ratios)
    assert run_slug(ratios) != run_slug(default)
    rows = [row for row in run_parameter_rows(ratios) if row["name"] == "control_condition_features"]
    assert [(row["section"], row["value"]) for row in rows] == [("Conditioning", "ratios")]


def test_the_train_cli_selects_the_set_and_a_recipe_refuses_it(capsys):
    import argparse

    import ts_transformer.cli.common as cli_common

    parser = argparse.ArgumentParser()
    cli_common.add_data_args(parser)
    cli_common.add_training_args(parser)
    base = ["--data", "unused.json", "--output-dir", "unused-output"]
    args = parser.parse_args([
        *base, "--prediction-output", "control", "--horizon-mode", "normalized",
        "--control-condition-features", "ratios",
    ])
    config, _batch_auto = cli_common.config_from_args(args, parser)
    assert config.control_condition_features == CONTROL_CONDITION_FEATURES_RATIOS
    pinned = parser.parse_args([
        *base, "--control-recipe-name", "simple-v3", "--control-condition-features", "ratios",
    ])
    with pytest.raises(SystemExit) as info:
        cli_common.config_from_args(pinned, parser)
    assert info.value.code == 2 and "control_condition_features" in capsys.readouterr().err
