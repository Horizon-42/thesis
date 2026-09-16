"""The specific-force control axis, ts side (docs/2026-09-14_specific_force_control_design.md §5).

``control_thrust_parameterization="specific-force"`` makes the head's first column the
specific force ``n_x = (T - D)/W`` instead of ``T/T_max``. What must hold for that to be the
same flight model in another coordinate, and for nothing stored to move:

* the teacher is the inverse of THIS forward model (a known n_x schedule comes back);
* the two coordinates describe the same motion (``n_x = (δ·T_max - D)/W`` on one track);
* the box, the neutral init and the initial actuator are the contract's, from ONE argument
  that no control call site may omit;
* the turn-rate supervision never reads the thrust column (so passing zero there is exact);
* the record stays in newtons and says which contract it came from — and a thrust-fraction
  record carries no new key, so every stored-config record reproduces to the bit;
* the config refuses the pairings not built, every named recipe pins the old law, and the
  name moves only off the default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (REPO_ROOT, TS_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2, heading_rate_rad_s  # noqa: E402
from ts_transformer.backbone.adapters import build_model  # noqa: E402
from ts_transformer.config import (  # noqa: E402
    CONTROL_DURATION_UNIFORM,
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_POINT_MASS,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    CONTROL_HOOKS_AVAILABLE,
    CONTROL_HOOK_MEMBERS,
    CONTROL_HOOK_SPEED_FLOOR,
    CONTROL_IMITATION_TARGET_FITTED,
    CONTROL_RECIPE_CUSTOM,
    CONTROL_RECIPE_NAMES,
    CONTROL_SPECIFIC_FORCE,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    CONTROL_CONDITION_FEATURES_RAW,
    CONTROL_THRUST_FRACTION,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
    control_recipe,
    recipe_settings,
)
from ts_transformer.data.dataset import Normalizer, build_series  # noqa: E402
from ts_transformer.data.synthetic import synthetic_arrivals  # noqa: E402
from ts_transformer.inference.export import build_prediction_record  # noqa: E402
from ts_transformer.outputs.conditioning import CONDITION_WIDTH  # noqa: E402
from ts_transformer.outputs.control.forecast import forecast_control_batch  # noqa: E402
from ts_transformer.outputs.control.supervision import probe_dynamics  # noqa: E402
from ts_transformer.outputs.dynamics.backends import RolloutInputs, control_dynamics_backend  # noqa: E402
from ts_transformer.outputs.dynamics.context import dynamics_arrays  # noqa: E402
from ts_transformer.outputs.dynamics.inverse import (  # noqa: E402
    _drag_force,
    actual_controls,
    reference_controls,
)
from ts_transformer.outputs.envelope import (  # noqa: E402
    MIN_THRUST_FRACTION,
    SPECIFIC_FORCE_CONTRACT,
    THRUST_FRACTION_CONTRACT,
    control_contract,
)
from ts_transformer.run_naming import run_display_name, run_slug  # noqa: E402

AIRPORT, RUNWAY = "KRDU", "05L"
LAG = {
    "control_dynamics_model": CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    "control_dynamics_backend": CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
}
SPECIFIC_FORCE = {**LAG, "control_thrust_parameterization": CONTROL_SPECIFIC_FORCE}
MAX_THRUST_N = 240_000.0
AERO = (122.6, 2.7, 0.02, 0.04, 0.9, 0.1)
FRAME = (37.36, -121.93, 18.0, 0.0)
INITIAL_STATE = (37.55, -121.70, 1800.0, 120.0, -2.4, -0.035, 62_000.0)
HORIZON_S = 240.0


def _config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        n_segments=8, seq_len=16, d_model=32, d_ff=64, n_heads=4, e_layers=1,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def _series(config: TSConfig, n_flights: int = 2):
    series, report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3), config, airport=AIRPORT
    )
    assert report.built == n_flights, report.format()
    return series


# ── the teacher is the inverse of this forward model ─────────────────────────


def test_a_specific_force_schedule_comes_back_through_its_own_inverse():
    """known n_x schedule -> lagged rollout -> dense reference -> inverse -> schedule back.

    The schedule is a near-coordinated turn on the fixture's own ~2° descent with a small
    speed-rate wobble, so the flight stays inside its thrust range (T/W 0.39, D/W ≈ 0.06) —
    where the clamp binds the command is not recoverable by construction — and the inverse
    is exact up to the 2 s grid.
    """
    config = _config(**SPECIFIC_FORCE, control_rollout_integrator_dt_s=0.1)
    n_segments = int(config.n_segments)
    progress = (np.arange(n_segments) + 0.5) / n_segments
    bank = np.deg2rad(10.0) * np.sin(np.pi * progress)
    controls = np.column_stack((
        -0.04 + 0.01 * np.sin(2.0 * np.pi * progress),
        bank,
        np.cos(INITIAL_STATE[5]) / np.cos(bank),
    ))
    inputs = RolloutInputs(
        initial_state=torch.tensor([INITIAL_STATE], dtype=torch.float64),
        initial_controls=torch.tensor(controls[:1], dtype=torch.float64),
        controls=torch.tensor(controls, dtype=torch.float64).unsqueeze(0),
        segment_durations_s=torch.full((1, n_segments), HORIZON_S / n_segments, dtype=torch.float64),
        aero_params=torch.tensor([AERO], dtype=torch.float64),
        frame_params=torch.tensor([FRAME], dtype=torch.float64),
        max_thrust_n=torch.tensor([MAX_THRUST_N], dtype=torch.float64),
    )
    offsets = np.arange(2.0, HORIZON_S + 1e-9, 2.0)
    rollout = control_dynamics_backend(config).dense_rollout(
        inputs, torch.tensor(offsets, dtype=torch.float64).unsqueeze(0),
        torch.ones((1, len(offsets)), dtype=torch.bool), config,
    )
    times = np.concatenate(([0.0], offsets))
    states = np.concatenate(
        (np.array([INITIAL_STATE]), rollout.query_geodetic_states[0].numpy()), axis=0
    )
    recovered = reference_controls(
        states, times, config=config, aero_params=np.array(AERO), max_thrust_n=MAX_THRUST_N
    )
    midpoints = (np.arange(n_segments) + 0.5) * (HORIZON_S / n_segments)
    sampled = np.column_stack([np.interp(midpoints, times, recovered[:, k]) for k in range(3)])
    error = np.abs(sampled - controls)
    assert error[:, 0].max() < 2e-3, f"specific force (g): {error[:, 0]}"
    assert error[:, 1].max() < np.deg2rad(0.5), f"bank: {np.rad2deg(error[:, 1])}"
    assert error[:, 2].max() < 5e-3, f"load factor: {error[:, 2]}"


def test_the_two_coordinates_describe_the_same_motion():
    """On one track: ``n_x = (δ·T_max - D)/W``, bank and load identical — the specific-force
    column is the thrust-fraction column read through the airframe, nothing else."""
    config = _config(**LAG)
    series = _series(config, n_flights=1)[0]
    anchor = config.seq_len - 1
    from ts_transformer.data.channels import states_from_channels
    samples = states_from_channels(
        series.times[anchor:] - series.times[anchor], series.values[anchor:], series.frame,
        mass_kg=float(series.scenario.initial.m),
    )
    times = np.array([t for t, _s in samples])
    states = np.array([[s.latitude, s.longitude, s.altitude, s.V, s.psi, s.gamma, s.m]
                       for _t, s in samples])
    aero = series.scenario.aero
    aero_row = np.array([aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall])
    thrust = float(series.scenario.aircraft.engine.max_thrust_total_n)
    kwargs = dict(aero_params=aero_row, max_thrust_n=thrust)
    fraction = actual_controls(states, times, parameterization=CONTROL_THRUST_FRACTION, **kwargs)
    specific = actual_controls(states, times, parameterization=CONTROL_SPECIFIC_FORCE, **kwargs)
    drag = _drag_force(states[:, 2], np.maximum(states[:, 3], 1e-3), states[:, 6], fraction[:, 2], aero_row)
    weight = states[:, 6] * GRAVITY_MPS2
    assert np.allclose(specific[:, 0], (fraction[:, 0] * thrust - drag) / weight, rtol=0.0, atol=1e-9)
    assert np.array_equal(specific[:, 1:], fraction[:, 1:])


def test_an_inversion_names_its_coordinate_or_refuses():
    states = np.tile(np.array(INITIAL_STATE), (4, 1))
    with pytest.raises(TypeError):
        actual_controls(states, np.arange(4.0), aero_params=np.array(AERO), max_thrust_n=MAX_THRUST_N)
    with pytest.raises(ValueError, match="no control contract for 'newtons'"):
        actual_controls(states, np.arange(4.0), aero_params=np.array(AERO),
                        max_thrust_n=MAX_THRUST_N, parameterization="newtons")


# ── the contract reaches the box, the actuator and the head from one argument ─


def test_dynamics_arrays_reads_the_contract_it_is_told_and_never_a_default():
    config = _config(**SPECIFIC_FORCE)
    series = _series(config, n_flights=1)[0]
    anchor = config.seq_len - 1
    with pytest.raises(TypeError):
        dynamics_arrays(series, anchor, condition_features=CONTROL_CONDITION_FEATURES_RAW)
    raw = {"condition_features": CONTROL_CONDITION_FEATURES_RAW}
    specific = dynamics_arrays(series, anchor, parameterization=CONTROL_SPECIFIC_FORCE, **raw)
    fraction = dynamics_arrays(series, anchor, parameterization=CONTROL_THRUST_FRACTION, **raw)
    assert np.allclose(specific["control_lower"], SPECIFIC_FORCE_CONTRACT.lower)
    assert np.allclose(specific["control_upper"], SPECIFIC_FORCE_CONTRACT.upper)
    assert np.allclose(fraction["control_lower"], THRUST_FRACTION_CONTRACT.lower)
    # The same actuator state at the anchor, in two coordinates (neither is clipped here).
    state = specific["initial_state"]
    aero_row = specific["aero_params"]
    thrust = float(specific["max_thrust_n"])
    delta, n_x = fraction["initial_controls"][0], specific["initial_controls"][0]
    assert THRUST_FRACTION_CONTRACT.lower[0] < delta < THRUST_FRACTION_CONTRACT.upper[0]
    drag = _drag_force(np.array([state[2]]), np.array([state[3]]), np.array([state[6]]),
                       np.array([fraction["initial_controls"][2]]), aero_row)[0]
    assert n_x == pytest.approx((delta * thrust - drag) / (state[6] * GRAVITY_MPS2), abs=1e-9)
    assert np.array_equal(specific["initial_controls"][1:], fraction["initial_controls"][1:])


def test_an_untrained_specific_force_head_starts_at_a_descent_speed_hold():
    """The zeroed head starts every flight at the contract's neutral: n_x = -0.05 (the
    speed hold on a ~2.9° descent, on every airframe), wings level, 1 g."""
    config = _config(**SPECIFIC_FORCE)
    model = build_model(config).eval()
    contract = control_contract(CONTROL_SPECIFIC_FORCE)
    lower = torch.tensor([contract.lower] * 2, dtype=torch.float32)
    upper = torch.tensor([contract.upper] * 2, dtype=torch.float32)
    dynamics = {
        "condition": torch.randn(2, CONDITION_WIDTH),
        "control_lower": lower,
        "control_upper": upper,
    }
    with torch.no_grad():
        prediction = model(torch.randn(2, config.seq_len, config.enc_in), dynamics)
    expected = torch.tensor(contract.neutral).view(1, 1, 3).expand_as(prediction.controls)
    assert torch.allclose(prediction.controls, expected, atol=1e-5)


def test_the_probe_batch_is_in_the_run_contract():
    probe = probe_dynamics(2, torch.device("cpu"), _config(**SPECIFIC_FORCE))
    assert torch.allclose(probe["control_lower"][0].double(), torch.tensor(SPECIFIC_FORCE_CONTRACT.lower, dtype=torch.float64))
    assert torch.allclose(probe["initial_controls"][0].double(), torch.tensor(SPECIFIC_FORCE_CONTRACT.neutral, dtype=torch.float64))


def test_the_turn_rate_row_never_reads_the_thrust_column():
    """What makes the heading-rate term's zero thrust column exact under both contracts."""
    generator = torch.Generator().manual_seed(4)
    states = torch.tensor([INITIAL_STATE] * 5, dtype=torch.float64)
    states[:, 3] += torch.rand(5, generator=generator, dtype=torch.float64) * 30.0
    controls = torch.stack((
        torch.rand(5, generator=generator, dtype=torch.float64) * 3e5,
        torch.rand(5, generator=generator, dtype=torch.float64) - 0.5,
        0.8 + torch.rand(5, generator=generator, dtype=torch.float64) * 0.6,
    ), dim=-1)
    aero = torch.tensor([AERO], dtype=torch.float64)
    zeroed = torch.cat((torch.zeros_like(controls[:, :1]), controls[:, 1:]), dim=-1)
    assert torch.equal(heading_rate_rad_s(states, controls, aero), heading_rate_rad_s(states, zeroed, aero))


# ── the record ───────────────────────────────────────────────────────────────


def _forecast_and_record(config: TSConfig):
    series = _series(config)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    anchor = config.seq_len - 1
    forecasts = forecast_control_batch(model, series, config, normalizer, anchor, torch.device("cpu"))
    record = build_prediction_record(series[0], forecasts[0], index=0, model_name=config.model,
                                     horizon_mode=config.horizon_mode)
    return series[0], forecasts[0], record


def test_a_specific_force_record_says_so_and_prices_its_thrust_where_each_segment_begins():
    config = _config(**SPECIFIC_FORCE)
    series, forecast, record = _forecast_and_record(config)
    assert forecast.control_parameterization == CONTROL_SPECIFIC_FORCE
    assert len(forecast.commands) == len(forecast.controls) == config.n_segments
    source = record.states_payload["source"]
    assert source["controlThrustParameterization"] == CONTROL_SPECIFIC_FORCE
    segments = record.states_payload["control_segments"]
    assert [s["specific_force"] for s in segments] == pytest.approx(list(forecast.commands[:, 0]))
    # Every segment's thrust, evaluated independently (numpy drag) at ITS start state: the
    # anchor for segment 0, the record's own row at the previous boundary after that (the
    # dense grid carries every boundary exactly).
    context = dynamics_arrays(
        series, config.seq_len - 1, parameterization=CONTROL_SPECIFIC_FORCE,
        condition_features=config.control_condition_features,
    )
    aero_row, thrust = context["aero_params"], float(context["max_thrust_n"])
    rows = {round(row["t"], 6): row for row in record.states_payload["predicted_states"]}
    for index, segment in enumerate(segments):
        if index == 0:
            start = context["initial_state"]
            alt, speed, mass = start[2], start[3], start[6]
            tolerance = 1e-9
        else:
            row = rows[round(segment["start_t"], 6)]
            alt, speed, mass = row["alt"], row["V"], row["m"]
            tolerance = 1e-6          # the record's serialized state precision
        drag = _drag_force(np.array([alt]), np.array([speed]), np.array([mass]),
                           np.array([segment["load_factor"]]), aero_row)[0]
        weight = mass * GRAVITY_MPS2
        expected = np.clip(weight * segment["specific_force"] + drag,
                           MIN_THRUST_FRACTION * thrust, thrust)
        # Absolute, in units of the weight: T is a small difference of W·n_x and D, so a
        # relative tolerance on T would amplify the state's rounding by W/T.
        assert segment["thrust"] == pytest.approx(expected, abs=tolerance * weight), index


def test_a_thrust_fraction_record_carries_no_new_key():
    config = _config(**LAG)
    _series_item, forecast, record = _forecast_and_record(config)
    assert forecast.control_parameterization == CONTROL_THRUST_FRACTION
    assert "controlThrustParameterization" not in record.states_payload["source"]
    assert all("specific_force" not in s for s in record.states_payload["control_segments"])


# ── one real training step ───────────────────────────────────────────────────


def test_a_specific_force_run_trains_through_the_whole_objective():
    """A real window batch, the simple-v3 terms (the trajectory, the velocity, the imitation
    teacher in n_x) plus the heading-rate term, through the lagged specific-force rollout:
    finite, and the gradient reaches the head's specific-force column."""
    from ts_transformer.data.batch_contract import anchor_state, unpack_batch
    from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows
    from ts_transformer.training.objective import prediction_loss

    config = _config(
        **SPECIFIC_FORCE, n_segments=4, seq_len=8, d_model=16, d_ff=32,
        control_velocity_loss_weight=1.0, control_imitation_loss_weight=64.0,
        control_heading_rate_loss_weight=1.0, control_rollout_integrator_dt_s=0.5,
    )
    series = _series(config)
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    x, target, weights, final_time, flight_weights, dynamics, dense = unpack_batch(
        dataset.batch(np.array([0, 1]))
    )
    assert torch.allclose(dynamics["control_upper"][0].double(),
                          torch.tensor(SPECIFIC_FORCE_CONTRACT.upper, dtype=torch.float64))
    model = build_model(config)
    loss = prediction_loss(
        model(x, dynamics), anchor_state(x, len(config.channels)), target, weights,
        final_time, flight_weights,
        config, normalizer, dynamics, dense,
    )
    loss.backward()
    assert torch.isfinite(loss)
    grad = model.control_head.control_projection.weight.grad.view(config.n_segments, 3, -1)
    assert torch.isfinite(grad).all() and grad[:, 0].abs().sum() > 0.0


# ── the config and the name ──────────────────────────────────────────────────


def test_specific_force_is_refused_where_it_is_not_built():
    with pytest.raises(ValueError, match="first-order-lag flight model only"):
        _config(control_dynamics_model=CONTROL_DYNAMICS_POINT_MASS,
                control_thrust_parameterization=CONTROL_SPECIFIC_FORCE)
    with pytest.raises(ValueError, match="fitted-teacher table"):
        _config(**SPECIFIC_FORCE, control_imitation_loss_weight=1.0,
                control_imitation_target=CONTROL_IMITATION_TARGET_FITTED,
                control_fitted_teacher_path="teacher.json")
    # Every hook that contains the speed floor is BUILT for the law since M2.
    floors = [hook for hook in CONTROL_HOOKS_AVAILABLE
              if CONTROL_HOOK_SPEED_FLOOR in CONTROL_HOOK_MEMBERS.get(hook, ())]
    assert floors, "no hook contains the speed floor; this check would pass vacuously"
    for hook in floors:
        _config(**SPECIFIC_FORCE, control_command_hook=hook)
    with pytest.raises(ValueError, match="belongs to the control output"):
        TSConfig(prediction_output=PREDICTION_STATE,
                 control_thrust_parameterization=CONTROL_SPECIFIC_FORCE)


def test_the_barrier_passes_the_specific_force_column_through():
    """The barrier (and the trombone) never write the first column, so they fly under the
    specific-force law unchanged: a hooked rollout's flown n_x IS the commanded n_x."""
    from ts_transformer.outputs.constraints import build_command_hook
    from ts_transformer.outputs.dynamics import rollout as control_rollout

    _config(**SPECIFIC_FORCE, control_command_hook="barrier+trombone")
    config = _config(**SPECIFIC_FORCE, control_command_hook="barrier")
    series = _series(config)
    rows = [dynamics_arrays(item, config.seq_len - 1, parameterization=CONTROL_SPECIFIC_FORCE,
                            condition_features=config.control_condition_features)
            for item in series]
    dynamics = {key: torch.from_numpy(np.stack([row[key] for row in rows])) for key in rows[0]}
    generator = torch.Generator().manual_seed(2)
    n = int(config.n_segments)
    controls = torch.stack((
        -0.08 + 0.06 * torch.rand(len(series), n, generator=generator, dtype=torch.float64),
        0.6 * (torch.rand(len(series), n, generator=generator, dtype=torch.float64) - 0.5),
        torch.ones(len(series), n, dtype=torch.float64),
    ), dim=-1)
    durations = torch.full((len(series), n), 20.0, dtype=torch.float64)
    hook = build_command_hook(config, dynamics)
    hooked = control_rollout.rollout_control_endpoints(
        controls, durations, dynamics, config, command_hook=hook
    )
    assert hook.diagnostics()["hook_steps"] == len(series) * n
    assert torch.equal(hooked.controls[..., 0], controls[..., 0])


def test_every_named_recipe_pins_the_thrust_fraction_law():
    for name in CONTROL_RECIPE_NAMES:
        if name == CONTROL_RECIPE_CUSTOM:
            continue
        settings = recipe_settings(name, keep_name=True)
        assert settings["control_thrust_parameterization"] == CONTROL_THRUST_FRACTION, name
        with pytest.raises(ValueError, match="recipe fields are frozen"):
            TSConfig(**{**settings, **SPECIFIC_FORCE})
    custom = recipe_settings("simple-v3", keep_name=False)
    TSConfig(**{**custom, **SPECIFIC_FORCE})


def test_the_train_cli_selects_the_law_and_a_recipe_refuses_it(capsys):
    import argparse
    import ts_transformer.cli.common as cli_common

    parser = argparse.ArgumentParser()
    cli_common.add_data_args(parser)
    cli_common.add_training_args(parser)
    base = ["--data", "unused.json", "--output-dir", "unused-output"]
    args = parser.parse_args([
        *base, "--prediction-output", "control", "--horizon-mode", "normalized",
        "--control-dynamics-model", "first-order-lag",
        "--control-dynamics-backend", "scaled-transport-chart-velocity",
        "--control-thrust-parameterization", "specific-force",
    ])
    config, _batch_auto = cli_common.config_from_args(args, parser)
    assert config.control_thrust_parameterization == CONTROL_SPECIFIC_FORCE
    pinned = parser.parse_args([
        *base, "--control-recipe-name", "simple-v3",
        "--control-thrust-parameterization", "specific-force",
    ])
    with pytest.raises(SystemExit) as info:
        cli_common.config_from_args(pinned, parser)
    assert info.value.code == 2 and "control_thrust_parameterization" in capsys.readouterr().err


def test_the_control_recipe_names_the_law_only_off_its_default():
    """`experiments.pipeline` compares this dict for checkpoint reuse: writing the default
    would refuse every stored checkpoint."""
    assert "thrust_parameterization" not in control_recipe(_config(**LAG))
    assert control_recipe(_config(**SPECIFIC_FORCE))["thrust_parameterization"] == CONTROL_SPECIFIC_FORCE


def test_the_name_moves_only_off_the_default():
    lag = _config(**LAG).to_dict()
    stored = {key: value for key, value in lag.items() if key != "control_thrust_parameterization"}
    assert run_display_name(lag) == run_display_name(stored)
    assert run_slug(lag) == run_slug(stored)
    specific = _config(**SPECIFIC_FORCE).to_dict()
    assert "first-order-lag+specific-force @scaled-transport-chart-velocity" in run_display_name(specific)
    assert "_lag-sf-stcv_" in run_slug(specific)
    assert run_slug(specific) != run_slug(lag)
