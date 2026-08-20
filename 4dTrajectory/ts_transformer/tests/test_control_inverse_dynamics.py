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
from control_dynamics_backends import (  # noqa: E402
    _BACKENDS,
    RolloutInputs,
    control_dynamics_backend,
)
from control_envelope import CONTROL_LOWER, CONTROL_UPPER  # noqa: E402
from geokit import METRES_PER_DEG_LAT  # noqa: E402
import control_inverse_dynamics as inverse_module  # noqa: E402
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


@pytest.mark.parametrize(
    "model,backend",
    sorted(_BACKENDS),
    ids=lambda value: value,
)
def test_inverse_recovers_the_schedule_its_forward_model_was_rolled_with(model, backend):
    """Close the loop for EVERY registered (model, backend) pair, not just one backend.

    This used to be parametrized over models only, with the backend pinned to
    `transport-chart-velocity`, so the transport-FREE family (`reanchored-rk4`) was never
    numerically inverted — the claim that it carries no transport term rested on reading
    its forward code. Rolling each registered pair forward and inverting it is what turns
    `TRANSPORT_BACKENDS` from an assertion into a measurement.
    """
    config = _config(model, control_dynamics_backend=backend)
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
    )
    midpoints = (np.arange(len(controls)) + 0.5) * (HORIZON_S / len(controls))
    sampled = np.column_stack(
        [np.interp(midpoints, times, recovered[:, k]) for k in range(3)]
    )

    # Tolerances are the finite-difference error of a 2 s grid against a control that
    # varies inside a 30 s segment, not a free parameter: a wrong equation misses by
    # orders of magnitude more. They are NOT sensitive to the transport term, which is
    # ~1e-4 in load against a 5e-3 bound — verifying that needs the differential test
    # below, and a mutation that mislabels a backend passes right through this one.
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
    )
    commanded = reference_controls(
        states,
        times,
        config=config,
        aero_params=np.array(AERO),
        max_thrust_n=MAX_THRUST_N,
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


def _imitation_config(weight: float) -> TSConfig:
    overrides = control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V1)
    overrides.update(
        seq_len=16, n_segments=8, d_model=32, d_ff=64, n_heads=4, e_layers=1,
        control_imitation_loss_weight=weight,
    )
    return TSConfig(control_recipe_name="custom", **overrides)


def test_the_imitation_term_scores_the_schedule_and_masks_the_fitted_tail():
    """Bank is a second-order quantity; position and velocity supervision never see it.

    A schedule equal to the inverted one must cost nothing, a full-box error must cost the
    same in every channel, and segments past the last measured velocity must not enter --
    the fitted tail has no kinematics to invert.
    """
    from control_envelope import CONTROL_HALF_WIDTH
    from prediction_outputs import ControlPrediction
    from train import control_imitation_mse

    config = _imitation_config(0.05)
    target = torch.zeros(2, 8, 3, dtype=torch.float64)
    weight = torch.ones(2, 8, dtype=torch.float64)
    weight[1, 4:] = 0.0  # flight 1's fitted tail

    def predict(controls: torch.Tensor) -> ControlPrediction:
        return ControlPrediction(
            controls=controls,
            segment_durations=torch.full((2, 8), 10.0, dtype=torch.float64),
            final_time_s=torch.full((2,), 80.0, dtype=torch.float64),
        )

    dynamics = {"reference_controls": target, "reference_control_weight": weight}
    exact = control_imitation_mse(predict(target.clone()), config, dynamics)
    assert torch.allclose(exact, torch.zeros(2, dtype=torch.float64))

    # One half-box error in every channel costs 1.0 regardless of the channel's units.
    full = target + torch.as_tensor(CONTROL_HALF_WIDTH, dtype=torch.float64)
    assert torch.allclose(
        control_imitation_mse(predict(full), config, dynamics),
        torch.ones(2, dtype=torch.float64),
    )

    # An error confined to the masked tail is invisible on flight 1 and visible on flight 0.
    tail = target.clone()
    tail[:, 4:] = torch.as_tensor(CONTROL_HALF_WIDTH, dtype=torch.float64)
    scored = control_imitation_mse(predict(tail), config, dynamics)
    assert scored[0] == pytest.approx(0.5)
    assert scored[1] == pytest.approx(0.0)

    assert control_imitation_mse(predict(full), _imitation_config(0.0), dynamics) is None


def test_the_imitation_target_is_inverted_through_the_configured_flight_model():
    """The lagged model is supervised on COMMANDS, the point-mass model on actual controls.

    Same guarantee the teacher already has, now for the training loss: the target is built
    by the registry entry the forward rollout dispatches through, so the two can never be
    solutions of different equations.
    """
    import dataset as dataset_module

    seen: dict[str, str] = {}
    real = dataset_module.segment_controls

    def spy(*args, **kwargs):
        seen["model"] = kwargs["config"].control_dynamics_model
        return real(*args, **kwargs)

    for model in (CONTROL_DYNAMICS_POINT_MASS, CONTROL_DYNAMICS_FIRST_ORDER_LAG):
        overrides = control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V1)
        overrides.update(control_dynamics_model=model, control_imitation_loss_weight=0.05)
        config = TSConfig(control_recipe_name="custom", **overrides)
        assert config.control_dynamics_model in CONTROL_INVERSES
        seen.clear()
        dataset_module.segment_controls = spy
        try:
            from dataset import build_series
            from synthetic import synthetic_arrivals

            flights = synthetic_arrivals("KRDU", "05L", n_flights=1, seed=3)
            series, _report = build_series(flights, config, airport="KRDU")
            dataset_module.reference_control_supervision(
                series[0], config.seq_len - 1, config,
                total_duration_s=120.0, last_measured_time_s=120.0,
            )
        finally:
            dataset_module.segment_controls = real
        assert seen["model"] == model


def test_the_imitation_weight_is_not_a_required_serialized_field():
    """Its default reproduces every checkpoint trained before the term existed."""
    from config import REQUIRED_SERIALIZED_CONTROL_FIELDS

    assert "control_imitation_loss_weight" not in REQUIRED_SERIALIZED_CONTROL_FIELDS
    assert TSConfig(prediction_output=PREDICTION_CONTROL).control_imitation_loss_weight == 0.0


def test_an_enabled_loss_term_is_reported_as_its_own_component():
    """A term the trainer accumulates but never declared crashes the epoch loop.

    fit_model builds its accumulator from loss_component_names and then indexes it with
    whatever keys the objective actually returned, so an extra term with no name raises
    KeyError on the first batch -- after dataset build, which is the slow part.
    """
    from train import loss_component_names

    off = loss_component_names(_imitation_config(0.0))
    on = loss_component_names(_imitation_config(0.05))

    assert "imitation" not in off
    assert "imitation" in on
    assert set(off) < set(on)


def test_simple_v3_is_the_settled_production_recipe():
    """simple-v3 = simple-v2 + direct supervision of the control schedule.

    One field separates them, and it is the imitation weight: bank is an order-2
    quantity that simple-v2's position and velocity terms never named, and unsupervised
    it scored BELOW a random-other-flight baseline on KRDU (0.124 against a floor of
    0.170). The weight is 47x the position term, chosen off an eight-point ladder where
    the region below 11.8x is a noisy plateau and 188x overshoots the flown data.
    """
    from config import (
        CONTROL_RECIPE_SIMPLE_V2,
        CONTROL_RECIPE_SIMPLE_V3,
        SIMPLE_V3_IMITATION_LOSS_WEIGHT,
        SIMPLE_V2_VELOCITY_LOSS_WEIGHT,
    )

    v2 = control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V2)
    v3 = control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V3)

    assert {name for name in v2 | v3 if v2.get(name) != v3.get(name)} == {
        "control_imitation_loss_weight"
    }
    assert v3["control_imitation_loss_weight"] == SIMPLE_V3_IMITATION_LOSS_WEIGHT
    # Everything simple-v2 settled is inherited, not re-litigated.
    assert v3["control_velocity_loss_weight"] == SIMPLE_V2_VELOCITY_LOSS_WEIGHT
    assert v3["control_dynamics_model"] == CONTROL_DYNAMICS_FIRST_ORDER_LAG
    assert TIME_CONSTANT_FIELDS <= set(v3)
    assert v3["d_model"] == 512

    config = TSConfig(control_recipe_name=CONTROL_RECIPE_SIMPLE_V3, **v3)
    assert TSConfig.from_dict(config.to_dict()) == config
    # The term must actually be wired into training, not merely stored.
    from train import loss_component_names

    assert "imitation" in loss_component_names(config)
    with pytest.raises(ValueError, match="recipe fields are frozen"):
        replace(config, control_imitation_loss_weight=0.0)

@pytest.mark.parametrize("model,backend", sorted(_BACKENDS), ids=lambda value: value)
def test_the_transport_term_is_required_by_every_backend(model, backend, monkeypatch):
    """Measure that omega x v is needed, for every registered (model, backend) pair.

    Two things this exists to catch, both of which actually happened:

    1. The term used to be gated on ``frame_params is not None`` while ``_transport_rate``
       never read that array's values. Callers that omitted it silently inverted a
       different model -- the training target passed it and the scoring scripts did not.
    2. The obvious repair, keying it on the backend, is WRONG. ``reanchored-rk4`` writes no
       transport term in its RHS (it re-anchors into geodetic state each substep instead),
       so it looks transport-free in the source -- but its rolled trajectories still invert
       ~50x more accurately WITH the term, because the inverse works in a local ENU frame
       and a curved-earth trajectory carries omega x v there regardless.

    The round-trip test above cannot see any of this: the term moves recovered bank by
    ~0.007 deg against a 0.5 deg tolerance, and a mutation mislabelling a backend passes it
    unchanged (verified by trying exactly that). Zeroing the term and requiring the fit to
    get worse is sensitive by construction, since nothing else changes.
    """
    config = _config(model, control_dynamics_backend=backend)
    controls = _smooth_schedule(int(config.n_segments))
    times, states = _dense_reference(config, controls, controls[0])
    midpoints = (np.arange(len(controls)) + 0.5) * (HORIZON_S / len(controls))

    def residual() -> np.ndarray:
        raw = reference_controls(
            states, times, config=config,
            aero_params=np.array(AERO), max_thrust_n=MAX_THRUST_N,
        )
        sampled = np.column_stack(
            [np.interp(midpoints, times, raw[:, k]) for k in range(3)]
        )
        return np.abs(sampled - controls).mean(axis=0)

    with_transport = residual()
    monkeypatch.setattr(
        inverse_module, "_transport_rate",
        lambda states: np.zeros((len(states), 3), dtype=np.float64),
    )
    without = residual()

    # Load factor carries the term most directly and is least polluted by the
    # finite-difference error the coarse round-trip tolerances are sized for.
    assert with_transport[2] < without[2] / 10.0, (
        f"{model}/{backend}: load residual {with_transport[2]:.2e} with the transport "
        f"term vs {without[2]:.2e} without -- the term is not doing what it claims"
    )
    assert with_transport[1] < without[1], f"{model}/{backend}: bank"
