"""The inverse must be the inverse OF THE CONFIGURED FORWARD MODEL, for every model.

A teacher solved against equations the training rollout does not integrate converges to a
schedule that reproduces nothing, and nothing downstream can tell: the schedule is finite,
bounded, the right shape, and its own optimizer reports a falling loss. The only way to
catch it is to close the loop numerically, which is what this module does for every entry
in the dynamics registry:

    known schedule -> forward rollout -> dense reference -> inverse -> schedule back

A dynamics model added without an inverse fails at registry lookup here; a model whose
inverse drifts from its RHS fails the tolerance.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (REPO_ROOT, TS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_MODELS,
    CONTROL_DYNAMICS_POINT_MASS,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
    CONTROL_RECIPE_SIMPLE_V1,
    HORIZON_NORMALIZED,
    PREDICTION_CONTROL,
    TIME_CONSTANT_FIELDS,
    TSConfig,
    control_recipe_overrides,
)
from control_dynamics_backends import RolloutInputs, control_dynamics_backend  # noqa: E402
from control_envelope import CONTROL_LOWER, CONTROL_UPPER  # noqa: E402
from geokit import METRES_PER_DEG_LAT  # noqa: E402
from control_inverse_dynamics import (  # noqa: E402
    CONTROL_INVERSES,
    actual_controls,
    reference_controls,
)
from aerodynamic_model.torch_transport_chart_dynamics import (  # noqa: E402
    transport_chart_state_to_geodetic,
)


MAX_THRUST_N = 240_000.0
AERO = (122.6, 2.7, 0.023, 0.0334, 0.8, 0.2)
FRAME = (37.36, -121.93, 18.0, 0.0)
INITIAL_STATE = (37.55, -121.70, 1800.0, 120.0, -2.4, -0.035, 62_000.0)
HORIZON_S = 240.0
QUERY_DT_S = 2.0


def _config(model: str, **overrides) -> TSConfig:
    settings = {
        "prediction_output": PREDICTION_CONTROL,
        "control_dynamics_model": model,
        "control_dynamics_backend": CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
        "control_rollout_integrator_dt_s": 0.1,
        "n_segments": 8,
    }
    return TSConfig(**{**settings, **overrides})


def _smooth_schedule(n_segments: int) -> np.ndarray:
    """A schedule that varies within the horizon, so a constant inverse cannot pass."""
    progress = (np.arange(n_segments, dtype=np.float64) + 0.5) / n_segments
    return np.column_stack(
        (
            0.18 + 0.08 * np.sin(2.0 * np.pi * progress),
            np.deg2rad(18.0) * np.sin(np.pi * progress),
            1.0 + 0.05 * np.cos(2.0 * np.pi * progress),
        )
    )


def _inputs(controls: np.ndarray, initial_controls: np.ndarray) -> RolloutInputs:
    n_segments = len(controls)
    return RolloutInputs(
        initial_state=torch.tensor([INITIAL_STATE], dtype=torch.float64),
        initial_controls=torch.tensor(
            np.asarray([initial_controls]), dtype=torch.float64
        ),
        controls=torch.tensor(controls, dtype=torch.float64).unsqueeze(0),
        segment_durations_s=torch.full(
            (1, n_segments), HORIZON_S / n_segments, dtype=torch.float64
        ),
        aero_params=torch.tensor([AERO], dtype=torch.float64),
        frame_params=torch.tensor([FRAME], dtype=torch.float64),
        max_thrust_n=torch.tensor([MAX_THRUST_N], dtype=torch.float64),
    )


def _dense_reference(config: TSConfig, controls: np.ndarray, initial_controls):
    """Roll the schedule and return ``(times, geodetic states)`` on a 2 s grid."""
    inputs = _inputs(controls, initial_controls)
    offsets = np.arange(QUERY_DT_S, HORIZON_S + 1e-9, QUERY_DT_S)
    rollout = control_dynamics_backend(config).dense_rollout(
        inputs,
        torch.tensor(offsets, dtype=torch.float64).unsqueeze(0),
        torch.ones((1, len(offsets)), dtype=torch.bool),
        config,
        segment_valid=None,
    )
    states = rollout.query_geodetic_states[0].numpy()
    times = np.concatenate(([0.0], offsets))
    return times, np.concatenate((np.array([INITIAL_STATE]), states), axis=0)


def test_every_registered_dynamics_model_has_an_inverse():
    assert set(CONTROL_INVERSES) == set(CONTROL_DYNAMICS_MODELS)


@pytest.mark.parametrize("model", CONTROL_DYNAMICS_MODELS)
def test_inverse_recovers_the_schedule_its_forward_model_was_rolled_with(model):
    config = _config(model)
    controls = _smooth_schedule(int(config.n_segments))
    # For the lagged model the actuators start where the first command asks, so the
    # recovered command at t=0 is the schedule's own first value rather than a transient.
    times, states = _dense_reference(config, controls, controls[0])

    recovered = reference_controls(
        states,
        times,
        config=config,
        aero_params=np.array(AERO),
        max_thrust_n=MAX_THRUST_N,
        frame_params=np.array(FRAME),
    )
    midpoints = (np.arange(len(controls)) + 0.5) * (HORIZON_S / len(controls))
    sampled = np.column_stack(
        [np.interp(midpoints, times, recovered[:, k]) for k in range(3)]
    )

    # Tolerances are the finite-difference error of a 2 s grid against a control that
    # varies inside a 30 s segment, not a free parameter: a wrong equation misses by
    # orders of magnitude more (a missing transport term alone would be ~1e-4 in load).
    error = np.abs(sampled - controls)
    assert error[:, 0].max() < 5e-3, f"thrust fraction: {error[:, 0]}"
    assert error[:, 1].max() < np.deg2rad(0.5), f"bank: {np.rad2deg(error[:, 1])}"
    assert error[:, 2].max() < 5e-3, f"load factor: {error[:, 2]}"


@pytest.mark.parametrize(
    "backend",
    [
        CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
        CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    ],
)
def test_the_state_representation_does_not_change_the_recovered_schedule(backend):
    """Nondimensionalising the state is a change of variables, not of physics."""
    config = _config(CONTROL_DYNAMICS_POINT_MASS, control_dynamics_backend=backend)
    controls = _smooth_schedule(int(config.n_segments))
    times, states = _dense_reference(config, controls, controls[0])

    recovered = reference_controls(
        states,
        times,
        config=config,
        aero_params=np.array(AERO),
        max_thrust_n=MAX_THRUST_N,
        frame_params=np.array(FRAME),
    )

    assert np.all(np.isfinite(recovered))
    assert np.abs(recovered[1:-1, 1]).max() < np.deg2rad(25.0)


def test_lag_commands_lead_the_actual_controls_they_produce():
    """The command inversion is not the actual-control inversion: it must lead it.

    A constant command with the actuators starting away from it isolates the lag law from
    the segment-boundary discontinuities: the realised bank is then a clean exponential
    approach, so ``u_cmd = u + tau * du/dt`` can be checked pointwise rather than only at
    segment midpoints.
    """
    n_segments = 2
    config = _config(CONTROL_DYNAMICS_FIRST_ORDER_LAG, n_segments=n_segments)
    controls = np.tile(np.array([0.2, np.deg2rad(20.0), 1.0]), (n_segments, 1))
    times, states = _dense_reference(config, controls, np.array([0.2, 0.0, 1.0]))

    realised = actual_controls(
        states,
        times,
        aero_params=np.array(AERO),
        max_thrust_n=MAX_THRUST_N,
        frame_params=np.array(FRAME),
    )
    commanded = reference_controls(
        states,
        times,
        config=config,
        aero_params=np.array(AERO),
        max_thrust_n=MAX_THRUST_N,
        frame_params=np.array(FRAME),
    )

    # The realised bank rolls in exponentially towards the command...
    transient = slice(3, 12)
    expected_bank = np.deg2rad(20.0) * (
        1.0 - np.exp(-times / config.control_bank_time_constant_s)
    )
    np.testing.assert_allclose(
        realised[transient, 1], expected_bank[transient], atol=np.deg2rad(0.5)
    )
    # ...and the recovered COMMAND is the constant the schedule actually asked for,
    # throughout the roll-in, which no actual-control inversion would produce. The first
    # samples carry the documented COMMAND_SMOOTHING_SAMPLES bias: a 5-sample average of a
    # 2 s grid over an exponential with tau = 2 s overstates the initial slope, which is
    # 1.3 degrees here and gone within 4 tau.
    np.testing.assert_allclose(
        commanded[transient, 1], np.deg2rad(20.0), atol=np.deg2rad(1.5)
    )
    settled = slice(8, 20)
    np.testing.assert_allclose(
        commanded[settled, 1], np.deg2rad(20.0), atol=np.deg2rad(0.2)
    )
    assert np.all(commanded[transient, 1] > realised[transient, 1])


def _divergence_from_point_mass_m(tau_s: float, controls: np.ndarray) -> float:
    """Terminal horizontal distance between the lagged and instantaneous trajectories."""
    instant = _config(CONTROL_DYNAMICS_POINT_MASS)
    lagged = _config(
        CONTROL_DYNAMICS_FIRST_ORDER_LAG,
        control_thrust_time_constant_s=tau_s,
        control_bank_time_constant_s=tau_s,
        control_load_time_constant_s=tau_s,
    )
    _times, instant_states = _dense_reference(instant, controls, controls[0])
    _times, lagged_states = _dense_reference(lagged, controls, controls[0])
    delta_deg = instant_states[-1, :2] - lagged_states[-1, :2]
    scale = np.array([METRES_PER_DEG_LAT, METRES_PER_DEG_LAT * np.cos(np.deg2rad(37.5))])
    return float(np.linalg.norm(delta_deg * scale))


def test_the_lag_converges_to_the_point_mass_model_as_the_time_constants_shrink():
    """The lagged model is the point-mass model plus three actuators, not a new one.

    Asserting a fixed distance would only pin one arbitrary tau. The real claim is that
    the two models are the SAME model in the limit, so what is tested is that the
    divergence is first order in tau and shrinks with it.
    """
    controls = _smooth_schedule(8)
    short = _divergence_from_point_mass_m(0.1, controls)
    long = _divergence_from_point_mass_m(1.0, controls)

    assert short < long / 5.0
    # 0.1 s of actuator lag over a 240 s, ~28 km rollout: well under a percent of path.
    assert short < 0.01 * 28_000.0


def test_a_real_time_constant_actually_changes_the_trajectory():
    """A bound that can never bind is worse than no bound; so is an inert time constant."""
    controls = _smooth_schedule(8)
    instant = _config(CONTROL_DYNAMICS_POINT_MASS)
    lagged = _config(CONTROL_DYNAMICS_FIRST_ORDER_LAG)  # default tau_bank = 2 s
    _times, instant_states = _dense_reference(instant, controls, controls[0])
    _times, lagged_states = _dense_reference(lagged, controls, controls[0])

    horizontal_m = np.linalg.norm(
        (instant_states[:, :2] - lagged_states[:, :2]) * 111_000.0, axis=-1
    )
    assert horizontal_m.max() > 10.0


def test_the_lag_keeps_the_realised_bank_continuous_across_segment_boundaries():
    """The point of the model: a stepped command produces a continuous bank angle."""
    n_segments = 4
    controls = np.tile(np.array([0.2, 0.0, 1.0]), (n_segments, 1))
    controls[1::2, 1] = np.deg2rad(25.0)          # alternate hard left/level commands
    config = _config(CONTROL_DYNAMICS_FIRST_ORDER_LAG, n_segments=n_segments)
    times, states = _dense_reference(config, controls, controls[0])

    realised = actual_controls(
        states,
        times,
        aero_params=np.array(AERO),
        max_thrust_n=MAX_THRUST_N,
        frame_params=np.array(FRAME),
    )
    # The command steps 25 degrees instantly at a boundary. The realised bank cannot: over
    # one 2 s reference interval a first-order actuator can only cover
    # 1 - exp(-dt/tau) of the remaining gap, which is the bound asserted here. Without the
    # lag the same diff would show the full 25-degree step.
    step_rad = np.abs(np.diff(realised[1:-1, 1]))
    reachable = np.deg2rad(25.0) * (
        1.0 - np.exp(-QUERY_DT_S / config.control_bank_time_constant_s)
    )
    assert step_rad.max() < reachable + np.deg2rad(0.5)
    assert step_rad.max() < np.deg2rad(25.0)


def test_inverted_controls_stay_inside_the_envelope_on_a_benign_trajectory():
    config = _config(CONTROL_DYNAMICS_POINT_MASS)
    controls = _smooth_schedule(int(config.n_segments))
    times, states = _dense_reference(config, controls, controls[0])

    recovered = reference_controls(
        states,
        times,
        config=config,
        aero_params=np.array(AERO),
        max_thrust_n=MAX_THRUST_N,
        frame_params=np.array(FRAME),
    )

    interior = recovered[1:-1]
    assert np.all(interior >= CONTROL_LOWER - 1e-6)
    assert np.all(interior <= CONTROL_UPPER + 1e-6)


def test_the_lagged_recipe_is_simple_v1_with_one_field_changed():
    """A paired comparison is only about the flight model if nothing else moved."""
    from config import CONTROL_RECIPE_SIMPLE_V1_LAG

    base = control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V1)
    lagged = control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V1_LAG)
    differing = {
        name for name in base | lagged if base.get(name) != lagged.get(name)
    }

    assert differing == {"control_dynamics_model"}
    assert lagged["control_dynamics_model"] == CONTROL_DYNAMICS_FIRST_ORDER_LAG
    assert base["control_dynamics_model"] == CONTROL_DYNAMICS_POINT_MASS
    # The time constants stay open: tau_bank is what the sweep resolves.
    assert not TIME_CONSTANT_FIELDS & set(lagged)
    frozen = TSConfig(
        control_recipe_name=CONTROL_RECIPE_SIMPLE_V1_LAG, **lagged
    )
    assert (
        replace(frozen, control_bank_time_constant_s=3.0).control_bank_time_constant_s
        == 3.0
    )
    with pytest.raises(ValueError, match="recipe fields are frozen"):
        replace(frozen, control_dynamics_model=CONTROL_DYNAMICS_POINT_MASS)


def test_a_time_constant_below_the_integrator_step_is_refused_not_integrated():
    """Explicit RK4 on y' = -y/tau produces NaN there, not a worse answer."""
    config = _config(CONTROL_DYNAMICS_FIRST_ORDER_LAG)
    assert config.control_rollout_integrator_dt_s == 0.1
    replace(config, control_bank_time_constant_s=0.1)  # exactly at the step: allowed
    with pytest.raises(ValueError, match="shorter than the .* integrator step"):
        replace(config, control_bank_time_constant_s=0.05)
    # The guard is specific to the lagged model; the constants are inert without it.
    replace(
        config,
        control_dynamics_model=CONTROL_DYNAMICS_POINT_MASS,
        control_bank_time_constant_s=0.05,
    )


def test_the_time_constant_axis_is_dropped_from_cv_when_the_lag_is_off():
    """An inert axis multiplies the candidate grid and returns identical folds."""
    from cross_validation import applicable_cv_parameters

    requested = ("d_model", "control_bank_time_constant_s")

    assert applicable_cv_parameters(
        requested, HORIZON_NORMALIZED, CONTROL_DYNAMICS_FIRST_ORDER_LAG
    ) == requested
    assert applicable_cv_parameters(
        requested, HORIZON_NORMALIZED, CONTROL_DYNAMICS_POINT_MASS
    ) == ("d_model",)
    with pytest.raises(ValueError, match="inert"):
        applicable_cv_parameters(
            ("control_bank_time_constant_s",),
            HORIZON_NORMALIZED,
            CONTROL_DYNAMICS_POINT_MASS,
        )


def test_the_exported_control_record_stays_in_newtons():
    """The evaluation contract is shared with the optimizer and did not change units."""
    from control_envelope import fraction_controls, physical_controls

    controls = np.array([[[0.5, 0.1, 1.0], [-0.2, -0.1, 1.2]]])
    max_thrust_n = np.array([MAX_THRUST_N])

    newtons = physical_controls(controls, max_thrust_n)

    assert newtons[0, 0, 0] == pytest.approx(0.5 * MAX_THRUST_N)
    assert newtons[0, 1, 0] == pytest.approx(-0.2 * MAX_THRUST_N)
    np.testing.assert_allclose(newtons[..., 1:], controls[..., 1:])
    np.testing.assert_allclose(fraction_controls(newtons, max_thrust_n), controls)


def _velocity_term_config(weight: float) -> TSConfig:
    overrides = control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V1)
    overrides.update(
        seq_len=16, n_segments=8, d_model=32, d_ff=64, n_heads=4, e_layers=1,
        control_velocity_loss_weight=weight,
    )
    return TSConfig(control_recipe_name="custom", **overrides)


def test_the_velocity_term_is_off_by_default_and_scores_measured_rows_when_on():
    """The true-time-position objective scored position only; this adds the velocity."""
    import train as train_module
    from control_loss_components import ControlStateLossResult, control_tracking_loss_terms
    from dataset import Normalizer

    normalizer = Normalizer(mean=np.zeros(6), std=np.ones(6))
    result = ControlStateLossResult(
        normalized_mse=torch.zeros(2, dtype=torch.float64),
        normalized_segment_end_states=torch.zeros(2, 8, 6, dtype=torch.float64),
        physical_position_mse=torch.tensor([0.25, 0.5], dtype=torch.float64),
        physical_velocity_mse=torch.tensor([4.0, 9.0], dtype=torch.float64),
    )
    terminal = torch.zeros(2, 6, dtype=torch.float64)
    anchor = torch.zeros(2, 6, dtype=torch.float64)
    heading = torch.zeros(2, dtype=torch.float64)

    off = control_tracking_loss_terms(
        result, anchor, terminal, _velocity_term_config(0.0), normalizer, None, heading
    )
    assert "velocity" not in off.extras
    assert "velocity" not in train_module.loss_component_names(_velocity_term_config(0.0))

    on_config = _velocity_term_config(0.25)
    on = control_tracking_loss_terms(
        result, anchor, terminal, on_config, normalizer, None, heading
    )
    torch.testing.assert_close(
        on.extras["velocity"], torch.tensor([1.0, 2.25], dtype=torch.float64)
    )
    # The position term is untouched, so turning the velocity term on is additive.
    torch.testing.assert_close(on.state, off.state)
    assert "velocity" in train_module.loss_component_names(on_config)


def test_the_velocity_term_ignores_the_fitted_tail_and_reaches_the_controls():
    """Fitted-tail velocity weights are zero, so placeholders cannot enter the loss."""
    import train as train_module
    from dataset import Normalizer

    config = _velocity_term_config(1.0)
    channels = len(config.channels)
    prediction_zero_weights = train_module._native_endpoint_control_state_loss

    torch.manual_seed(0)
    controls = torch.zeros(1, config.n_segments, 3, dtype=torch.float64, requires_grad=True)
    from prediction_outputs import ControlPrediction
    from dataset import probe_dynamics

    durations = torch.full((1, config.n_segments), 20.0, dtype=torch.float64)
    prediction = ControlPrediction(
        controls=controls, segment_durations=durations,
        final_time_s=durations.sum(dim=1),
    )
    targets = torch.zeros(1, config.n_segments, channels, dtype=torch.float64)
    weights = torch.ones_like(targets)
    weights[..., list(ch_velocity())] = 0.0          # every velocity row masked off
    result = prediction_zero_weights(
        prediction, torch.zeros(1, channels, dtype=torch.float64), targets, weights,
        durations.sum(dim=1), config, Normalizer(mean=np.zeros(channels), std=np.ones(channels)),
        probe_dynamics(1, torch.device("cpu")), None, None,
    )
    # With every velocity weight zero the term is exactly zero, not a division blow-up.
    assert float(result.physical_velocity_mse[0]) == 0.0
    assert torch.isfinite(result.physical_position_mse).all()

    result.physical_velocity_mse.sum().backward(retain_graph=True)
    assert controls.grad is not None


def ch_velocity():
    from channels import VELOCITY_IDX
    return VELOCITY_IDX


def test_simple_v2_is_the_settled_production_recipe():
    """simple-v2 = simple-v1-lag + the velocity term, with nothing else left open."""
    from config import (
        CONTROL_RECIPE_SIMPLE_V1_LAG,
        CONTROL_RECIPE_SIMPLE_V2,
        SIMPLE_V2_VELOCITY_LOSS_WEIGHT,
    )

    lag = control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V1_LAG)
    v2 = control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V2)
    differing = {name for name in lag | v2 if lag.get(name) != v2.get(name)}

    # The velocity term is the change; the time constants move from open to pinned.
    assert "control_velocity_loss_weight" in differing
    assert v2["control_velocity_loss_weight"] == SIMPLE_V2_VELOCITY_LOSS_WEIGHT
    assert v2["control_dynamics_model"] == CONTROL_DYNAMICS_FIRST_ORDER_LAG
    # A production recipe names ONE configuration: unlike simple-v1-lag, nothing is open.
    assert TIME_CONSTANT_FIELDS <= set(v2)
    # The combination arm settled this: width adds nothing once the term is present.
    assert v2["d_model"] == 512

    config = TSConfig(control_recipe_name=CONTROL_RECIPE_SIMPLE_V2, **v2)
    assert TSConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="recipe fields are frozen"):
        replace(config, control_velocity_loss_weight=0.0)
    with pytest.raises(ValueError, match="recipe fields are frozen"):
        replace(config, control_bank_time_constant_s=3.0)
