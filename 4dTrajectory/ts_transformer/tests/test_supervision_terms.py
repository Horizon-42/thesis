"""L1.b: the two supervision terms that replace the imitation teacher's job.

The teacher's one real job is to NAME THE BANK (position is derivative order 0, velocity
order 1, bank order 2), and its inverse-dynamics target is not consistent with the rollout.
These two terms price the same information through the rollout instead:

* ``heading_rate`` — the rollout's own turn rate at each segment endpoint against the flown
  track's, read out of the RHS the rollout integrates;
* ``bank_tv`` — the commanded bank's total variation, a structural penalty that names no
  value at all.

The heading-rate half is tested as TWO HALVES AGAINST EACH OTHER: a rollout at a constant
positive bank, the target rebuilt from the track that rollout flew, and the two required to
agree in sign and magnitude. Nothing here hard-codes which sign a positive bank turns —
that is the package's convention to state, not this file's to guess.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import objective
from aerodynamic_model.torch_dynamics import GRAVITY_MPS2, heading_rate_rad_s
from channels import CHANNELS, channels_from_states
from config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_RECIPE_SIMPLE_V3,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    PREDICTION_CONTROL,
    TSConfig,
    control_recipe_overrides,
    control_simple_v1_overrides,
    recipe_settings,
)
from control.dynamics import rollout as control_rollout
from control.envelope import BANK_INDEX, CONTROL_HALF_WIDTH, physical_controls
from control.loss.components import ControlStateLossResult, control_tracking_loss_terms
from coordinate_frames import ENUFrame
from dataset import (
    ARRIVAL_DATA_PROVENANCE_SCHEMA,
    HEADING_RATE_SMOOTHING_WINDOW_S,
    Normalizer,
    build_series,
    probe_dynamics,
    reference_heading_rate_supervision,
)
from forecast import forecast_approach
from run_naming import run_display_name
from synthetic import synthetic_arrivals
from train import load_checkpoint, train

from aerodynamic_model.common import GeodeticState


AIRPORT, RUNWAY = "KRDU", "05L"
MASS_KG = 66_000.0
FRAME = ENUFrame(lat0=35.9, lon0=-78.8, alt0=100.0)


# ── the observed-track target ────────────────────────────────────────────────

def _series_from_states(states: list[tuple[float, GeodeticState]]) -> SimpleNamespace:
    """The two attributes :func:`reference_heading_rate_supervision` reads off a series."""
    times, values = channels_from_states(states, FRAME)
    return SimpleNamespace(
        times=times,
        values=values,
        frame=FRAME,
        scenario=SimpleNamespace(initial=SimpleNamespace(m=MASS_KG)),
    )


def _constant_turn(rate_dps: float, *, psi0_rad: float, duration_s: float,
                   dt_s: float = 2.0, speed_mps: float = 90.0):
    """A level circular arc turning at exactly ``rate_dps``, as ``(t, GeodeticState)``."""
    omega = math.radians(rate_dps)
    times = np.arange(0.0, duration_s + dt_s, dt_s)
    radius = speed_mps / omega
    states = []
    for t in times:
        psi = psi0_rad + omega * t
        east = radius * (math.sin(psi) - math.sin(psi0_rad))
        north = radius * (math.cos(psi0_rad) - math.cos(psi))
        lat, lon = FRAME.latlon_from_horizontal(east, north)
        states.append((float(t), GeodeticState(
            latitude=lat, longitude=lon, altitude=FRAME.alt0 + 600.0,
            V=speed_mps, psi=psi, gamma=0.0, m=MASS_KG,
        )))
    return states


def _config(n_segments: int = 8, **overrides) -> TSConfig:
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        n_segments=n_segments,
        seq_len=16, d_model=32, d_ff=64, n_heads=4, e_layers=1,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def test_a_constant_rate_turn_reads_its_own_rate_at_every_endpoint():
    """A 3 deg/s turn reads 3 deg/s at all N endpoints, and a -3 deg/s turn reads -3.

    What this pins is the UNIT and the SIGN — rad/s off the velocity channels out as signed
    deg/s — and that the chain is rate-preserving end to end. It deliberately CANNOT catch a
    wrong smoothing width or a wrong endpoint index: on a constant-rate turn every window
    and every sample time gives the same answer. Those live in
    ``test_the_smoothing_window_is_a_stated_constant_and_survives_a_jittered_heading`` (a
    rate that is not constant under noise) and in
    ``test_endpoints_past_the_last_measured_velocity_carry_zero_weight`` /
    ``test_the_anchor_slice_is_the_only_past_the_target_reads``.
    """
    duration_s = 120.0
    config = _config(n_segments=8)

    for rate_dps in (3.0, -3.0):
        series = _series_from_states(
            _constant_turn(rate_dps, psi0_rad=0.4, duration_s=duration_s)
        )
        target = reference_heading_rate_supervision(
            series, 0, config,
            total_duration_s=duration_s, last_measured_time_s=duration_s,
        )
        np.testing.assert_allclose(
            target["reference_heading_rate_dps"], rate_dps, rtol=1e-6
        )
        assert target["reference_heading_rate_weight"].tolist() == [1.0] * 8


def test_the_heading_is_unwrapped_before_it_is_differentiated():
    """psi wraps at +/-pi; differencing the wrapped angle would put a 180 deg/s spike
    exactly where the turn crosses the branch cut, and nowhere else.

    Like the test above this is a constant-rate fixture, so it pins the branch cut ONLY —
    not the window width and not the endpoint placement."""
    duration_s = 120.0
    # Start 30 degrees short of +pi at 3 deg/s: the cut is crossed 10 s in.
    series = _series_from_states(
        _constant_turn(3.0, psi0_rad=math.pi - math.radians(30.0), duration_s=duration_s)
    )
    # The channels carry velocity, not psi; the branch cut appears when psi is recovered.
    from channels import states_from_channels

    recovered = np.asarray([
        state.psi for _t, state in states_from_channels(
            series.times, series.values, FRAME, mass_kg=MASS_KG
        )
    ])
    assert recovered.min() < -3.0 < 3.0 < recovered.max(), "the fixture must cross +/-pi"

    target = reference_heading_rate_supervision(
        series, 0, _config(n_segments=16),
        total_duration_s=duration_s, last_measured_time_s=duration_s,
    )

    np.testing.assert_allclose(target["reference_heading_rate_dps"], 3.0, rtol=1e-6)


def test_the_smoothing_window_is_a_stated_constant_and_survives_a_jittered_heading():
    """A single 2 s difference of a quantised heading is noise; the window is why the
    target is usable. The constant is public so a reader can price it."""
    assert HEADING_RATE_SMOOTHING_WINDOW_S == 10.0
    duration_s = 200.0
    clean = _constant_turn(1.0, psi0_rad=0.2, duration_s=duration_s)
    rng = np.random.default_rng(0)
    # ADS-B track angle is reported to ~0.35 deg; jitter of that size on a 1 deg/s turn is
    # 35 % of the per-step signal and 100 % of it after differencing two samples.
    jittered = [
        (t, replace(state, psi=state.psi + math.radians(rng.normal(0.0, 0.35))))
        for t, state in clean
    ]
    config = _config(n_segments=16)

    smoothed = reference_heading_rate_supervision(
        _series_from_states(jittered), 0, config,
        total_duration_s=duration_s, last_measured_time_s=duration_s,
    )["reference_heading_rate_dps"]

    # The one-step difference of the same jittered track, for comparison.
    psi = np.unwrap(np.asarray([s.psi for _t, s in jittered]))
    one_step = np.degrees(np.diff(psi) / 2.0)
    one_step_rms = float(np.sqrt(np.mean((one_step - 1.0) ** 2)))
    smoothed_rms = float(np.sqrt(np.mean((smoothed - 1.0) ** 2)))
    # Averaging the window's one-step slopes divides the quantisation noise by the number
    # of steps it spans; the single difference is not usable as a target at all.
    assert one_step_rms > 3.0 * smoothed_rms
    np.testing.assert_allclose(smoothed, 1.0, atol=0.35)


def test_a_sample_step_too_coarse_for_the_window_is_refused_with_its_numbers():
    """The window is a stated constant, so a dt_s that cannot realize even one sample of
    half-width fails loudly instead of being floored to one and silently widened."""
    duration_s = 600.0
    dt_s = 12.0
    series = _series_from_states(
        _constant_turn(1.0, psi0_rad=0.0, duration_s=duration_s, dt_s=dt_s)
    )

    with pytest.raises(ValueError, match="heading-rate smoothing window"):
        reference_heading_rate_supervision(
            series, 0, _config(n_segments=8, dt_s=dt_s),
            total_duration_s=duration_s, last_measured_time_s=duration_s,
        )


def test_endpoints_past_the_last_measured_velocity_carry_zero_weight():
    """The fitted tail has no measured velocity to differentiate, so it must not be
    scored — the same masking the velocity term relies on, read at the ENDPOINTS."""
    duration_s = 160.0
    series = _series_from_states(
        _constant_turn(2.0, psi0_rad=0.0, duration_s=duration_s)
    )

    target = reference_heading_rate_supervision(
        series, 0, _config(n_segments=8),
        total_duration_s=duration_s,
        # Endpoints sit at 20, 40, ... 160 s; the last measured velocity is at 100 s.
        last_measured_time_s=100.0,
    )

    assert target["reference_heading_rate_weight"].tolist() == [
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0
    ]


def test_the_anchor_slice_is_the_only_past_the_target_reads():
    """The target is built from the anchor onwards, on the anchor's own clock."""
    duration_s = 120.0
    series = _series_from_states(
        _constant_turn(3.0, psi0_rad=0.1, duration_s=duration_s)
    )
    anchor = 10
    remaining_s = float(series.times[-1] - series.times[anchor])

    target = reference_heading_rate_supervision(
        series, anchor, _config(n_segments=4),
        total_duration_s=remaining_s, last_measured_time_s=remaining_s,
    )

    assert len(target["reference_heading_rate_dps"]) == 4
    np.testing.assert_allclose(target["reference_heading_rate_dps"], 3.0, rtol=1e-6)


# ── the model side, through the rollout ──────────────────────────────────────

def _rollout_dynamics(bank_rad: float, *, config: TSConfig, thrust_fraction: float = 0.24,
                      segment_s: float = 10.0):
    """A one-flight batch flying a CONSTANT bank, and the rollout it produces."""
    dynamics = probe_dynamics(1, torch.device("cpu"))
    # A level, coordinated turn: the load factor that holds altitude at this bank.
    controls = torch.tensor(
        [[[thrust_fraction, bank_rad, 1.0 / math.cos(bank_rad)]]],
        dtype=torch.float64,
    ).expand(1, config.n_segments, 3).contiguous()
    durations = torch.full((1, config.n_segments), segment_s, dtype=torch.float64)
    return dynamics, controls, durations


def _model_side_dps(rollout, dynamics) -> torch.Tensor:
    return torch.rad2deg(
        heading_rate_rad_s(
            rollout.geodetic_states,
            physical_controls(
                rollout.actual_controls.to(torch.float64),
                dynamics["max_thrust_n"].to(torch.float64),
            ),
            dynamics["aero_params"].to(torch.float64).unsqueeze(-2),
        )
    )


def test_the_model_side_is_the_rhs_relation_on_the_rollout_endpoint_states():
    """dpsi/dt is read out of the integrated RHS, i.e. g*n*sin(bank)/(V*cos gamma) with the
    stall-limited load factor — NOT a restated g*tan(bank)/V. On a coordinated level turn
    (n = cos gamma / cos bank) the two coincide, and that is the only place they do."""
    config = _config(n_segments=6)
    bank = 0.35
    dynamics, controls, durations = _rollout_dynamics(bank, config=config)
    # Level flight, so the coordinated identity is exactly testable.
    initial = dynamics["initial_state"].to(torch.float64).clone()
    initial[:, 5] = 0.0
    dynamics = {**dynamics, "initial_state": initial}

    rollout = control_rollout.rollout_control_endpoints(
        controls, durations, dynamics, config
    )
    predicted_dps = _model_side_dps(rollout, dynamics)

    speed = rollout.geodetic_states[..., 3]
    gamma = rollout.geodetic_states[..., 5]
    load = rollout.actual_controls[..., 2]
    rhs_dps = torch.rad2deg(
        GRAVITY_MPS2 * load * math.sin(bank) / (speed * torch.cos(gamma))
    )
    torch.testing.assert_close(predicted_dps, rhs_dps)
    # And the textbook coordinated-turn form, which this rollout happens to satisfy.
    coordinated_dps = torch.rad2deg(GRAVITY_MPS2 * math.tan(bank) / speed)
    torch.testing.assert_close(predicted_dps, coordinated_dps, rtol=2e-3, atol=0.0)


def test_the_lagged_model_prices_the_bank_the_actuator_reached_not_the_command():
    """Under the first-order lag the commanded bank is not what the aircraft is doing;
    a term that read the command would price a turn the rollout never flew."""
    from config import CONTROL_DYNAMICS_FIRST_ORDER_LAG, CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY

    config = _config(
        n_segments=4,
        control_dynamics_model=CONTROL_DYNAMICS_FIRST_ORDER_LAG,
        control_dynamics_backend=CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
        control_bank_time_constant_s=2.0,
    )
    command = 0.4
    dynamics, controls, durations = _rollout_dynamics(
        command, config=config, segment_s=1.0
    )
    # The anchor is flying wings level; the actuator has 1 s to reach a 0.4 rad command.
    dynamics = {**dynamics, "initial_controls": torch.tensor(
        [[0.24, 0.0, 1.0]], dtype=torch.float64
    )}

    rollout = control_rollout.rollout_control_endpoints(
        controls, durations, dynamics, config
    )

    reached = rollout.actual_controls[0, 0, BANK_INDEX]
    assert 0.0 < float(reached) < command                # still rolling in
    assert float(rollout.controls[0, 0, BANK_INDEX]) == pytest.approx(command)
    # The priced turn rate is the one the reached bank produces, strictly smaller.
    predicted_dps = _model_side_dps(rollout, dynamics)
    speed = rollout.geodetic_states[0, 0, 3]
    gamma = rollout.geodetic_states[0, 0, 5]
    load = rollout.actual_controls[0, 0, 2]
    torch.testing.assert_close(
        predicted_dps[0, 0],
        torch.rad2deg(
            GRAVITY_MPS2 * load * torch.sin(reached) / (speed * torch.cos(gamma))
        ),
    )


def test_the_two_halves_agree_in_sign_and_size_on_the_track_the_rollout_flew():
    """The sign convention is pinned by construction, not by hand: fly a constant POSITIVE
    bank, rebuild the observed target from the resulting track, and require the target and
    the model side to be the same number. Either sign flipping would fail this."""
    config = _config(n_segments=6)
    bank = 0.3
    dynamics, controls, durations = _rollout_dynamics(bank, config=config)
    total_s = float(durations.sum())

    # The dense rollout refuses a t=0 query, so the anchor's own state supplies that row.
    query_offsets = torch.arange(2.0, total_s + 2.0, 2.0, dtype=torch.float64)[None, :]
    dense = control_rollout.rollout_control_dense(
        controls, durations, dynamics, query_offsets,
        torch.ones_like(query_offsets, dtype=torch.bool), config,
    )
    from aerodynamic_model.torch_dynamics import geodetic_states_to_channels

    flown_states = torch.cat(
        (dynamics["initial_state"].to(torch.float64).unsqueeze(1),
         dense.query_geodetic_states),
        dim=1,
    )
    flown = SimpleNamespace(
        times=np.arange(0.0, total_s + 2.0, 2.0),
        values=geodetic_states_to_channels(
            flown_states, dynamics["frame_params"].to(torch.float64),
            runway_aligned=False,
        )[0].numpy(),
        frame=ENUFrame(*[float(v) for v in dynamics["frame_params"][0, :3]]),
        scenario=SimpleNamespace(initial=SimpleNamespace(m=MASS_KG)),
    )

    target = reference_heading_rate_supervision(
        flown, 0, config, total_duration_s=total_s, last_measured_time_s=total_s,
    )["reference_heading_rate_dps"]
    endpoints = control_rollout.rollout_control_endpoints(
        controls, durations, dynamics, config
    )
    predicted = _model_side_dps(endpoints, dynamics)[0].numpy()

    assert np.all(np.sign(target) == np.sign(predicted))
    assert np.all(np.abs(predicted) > 1.0)              # a real turn, not numerical dust
    np.testing.assert_allclose(target, predicted, rtol=0.03)


def test_a_positive_bank_turns_the_way_the_channel_heading_says_it_does():
    """The convention itself, stated once: this package's psi is math-ENU (CCW positive),
    and the shared RHS turns CCW under a positive bank. Every consumer inherits it."""
    config = _config(n_segments=2)
    dynamics, controls, durations = _rollout_dynamics(0.3, config=config)
    rollout = control_rollout.rollout_control_endpoints(
        controls, durations, dynamics, config
    )
    assert float(_model_side_dps(rollout, dynamics)[0, 0]) > 0.0
    assert float(rollout.geodetic_states[0, 1, 4]) > float(
        dynamics["initial_state"][0, 4]
    )


# ── the two loss terms ───────────────────────────────────────────────────────

def _endpoint_result(config: TSConfig, bank_schedule: list[float], target_dps: list[float],
                     weight: list[float]):
    """Run the native-endpoint loss on a hand-built schedule and target."""
    dynamics = probe_dynamics(1, torch.device("cpu"))
    controls = torch.tensor(
        [[[0.24, bank, 1.0 / math.cos(bank)] for bank in bank_schedule]],
        dtype=torch.float64, requires_grad=True,
    )
    durations = torch.full((1, len(bank_schedule)), 10.0, dtype=torch.float64)
    from prediction_outputs import ControlPrediction

    prediction = ControlPrediction(
        controls=controls, segment_durations=durations,
        final_time_s=durations.sum(dim=1),
    )
    channels = len(config.channels)
    dynamics = {
        **dynamics,
        "reference_heading_rate_dps": torch.tensor([target_dps], dtype=torch.float64),
        "reference_heading_rate_weight": torch.tensor([weight], dtype=torch.float64),
    }
    targets = torch.zeros(1, len(bank_schedule), channels, dtype=torch.float64)
    weights = torch.full_like(targets, 1.0 / channels)
    result = objective._native_endpoint_control_state_loss(
        prediction, torch.zeros(1, channels, dtype=torch.float64), targets, weights,
        durations.sum(dim=1), config,
        Normalizer(mean=np.zeros(channels), std=np.ones(channels)),
        dynamics, None,
    )
    return result, controls


def test_the_heading_rate_term_is_zero_on_its_own_answer_and_reaches_the_bank():
    """Feed the term the rate the rollout itself produces: the residual is exactly zero,
    and a perturbed target puts a gradient on the bank column."""
    config = _config(n_segments=4, control_heading_rate_loss_weight=1.0)
    dynamics, controls, durations = _rollout_dynamics(0.3, config=config)
    truth = _model_side_dps(
        control_rollout.rollout_control_endpoints(controls, durations, dynamics, config),
        dynamics,
    )[0].tolist()

    result, schedule = _endpoint_result(config, [0.3] * 4, truth, [1.0] * 4)
    assert float(result.control_heading_rate_mse[0]) == pytest.approx(0.0, abs=1e-20)

    off_target, schedule = _endpoint_result(
        config, [0.3] * 4, [value + 1.0 for value in truth], [1.0] * 4
    )
    # One deg/s of error at the 1.5 deg/s scale.
    assert float(off_target.control_heading_rate_mse[0]) == pytest.approx(
        (1.0 / 1.5) ** 2, rel=1e-9
    )
    off_target.control_heading_rate_mse.sum().backward()
    assert schedule.grad is not None
    assert torch.all(schedule.grad[0, :, BANK_INDEX].abs() > 0.0)


def test_the_heading_rate_term_ignores_masked_endpoints():
    config = _config(n_segments=4, control_heading_rate_loss_weight=1.0)
    scored, _ = _endpoint_result(config, [0.3] * 4, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0, 0])
    all_zero, _ = _endpoint_result(config, [0.3] * 4, [0.0] * 4, [0.0] * 4)

    assert float(scored.control_heading_rate_mse[0]) > 0.0
    # Every endpoint masked: exactly zero, not a division blow-up.
    assert float(all_zero.control_heading_rate_mse[0]) == 0.0


def test_bank_total_variation_is_the_mean_absolute_step_in_half_box_units():
    config = _config(n_segments=4, control_bank_tv_loss_weight=1.0)
    schedule = [0.0, 0.2, 0.2, -0.1]
    result, controls = _endpoint_result(config, schedule, [0.0] * 4, [0.0] * 4)

    expected = (0.2 + 0.0 + 0.3) / 3.0 / float(CONTROL_HALF_WIDTH[BANK_INDEX])
    assert float(result.control_bank_tv[0]) == pytest.approx(expected)
    # A constant schedule has none of it, and the term reaches the commanded bank.
    flat, _ = _endpoint_result(config, [0.15] * 4, [0.0] * 4, [0.0] * 4)
    assert float(flat.control_bank_tv[0]) == 0.0
    result.control_bank_tv.sum().backward()
    assert torch.any(controls.grad[0, :, BANK_INDEX].abs() > 0.0)


def test_bank_total_variation_prices_reversals_and_is_stationary_when_flat():
    """What the term can and cannot do, as gradients — the reading rule for arm ③.

    ``|x|`` has subgradient ``sign(x)``, so on a monotone run the interior segments' two
    contributions cancel and only the ends are charged: a smooth roll-in costs exactly what
    a single step of the same total size costs. At EXACT flatness the value AND the gradient
    are both zero — a stationary point, and the one every run starts at, because
    ``control.heads._initialize_control_head`` zeroes the projection so all N commands are
    identical. Only the other terms leave it. "The TV term changed nothing" is this first,
    a too-small dose second.
    """
    config = _config(n_segments=4, control_bank_tv_loss_weight=1.0)

    def bank_grad(schedule):
        result, controls = _endpoint_result(config, schedule, [0.0] * 4, [0.0] * 4)
        result.control_bank_tv.sum().backward()
        return float(result.control_bank_tv[0]), controls.grad[0, :, BANK_INDEX]

    flat_value, flat_grad = bank_grad([0.15] * 4)
    assert flat_value == 0.0
    torch.testing.assert_close(flat_grad, torch.zeros_like(flat_grad))

    _monotone_value, monotone_grad = bank_grad([0.0, 0.1, 0.2, 0.3])
    assert torch.all(monotone_grad[1:3] == 0.0)          # the interior cancels
    assert float(monotone_grad[0]) < 0.0 < float(monotone_grad[3])

    _reversal_value, reversal_grad = bank_grad([0.0, 0.1, 0.0, 0.1])
    assert torch.all(reversal_grad.abs() > 0.0)          # every segment is charged


def test_both_terms_are_weighted_into_the_objective_and_are_additive():
    result = ControlStateLossResult(
        normalized_mse=torch.zeros(2, dtype=torch.float64),
        normalized_segment_end_states=torch.zeros(2, 4, 6, dtype=torch.float64),
        physical_position_mse=torch.tensor([0.25, 0.5], dtype=torch.float64),
        physical_velocity_mse=torch.tensor([4.0, 9.0], dtype=torch.float64),
        control_heading_rate_mse=torch.tensor([2.0, 3.0], dtype=torch.float64),
        control_bank_tv=torch.tensor([0.1, 0.2], dtype=torch.float64),
    )
    normalizer = Normalizer(mean=np.zeros(6), std=np.ones(6))
    terminal = torch.zeros(2, 6, dtype=torch.float64)
    off = control_tracking_loss_terms(result, terminal, terminal, _config(4), normalizer, None)
    on_config = _config(
        4, control_heading_rate_loss_weight=8.0, control_bank_tv_loss_weight=1.0
    )
    on = control_tracking_loss_terms(result, terminal, terminal, on_config, normalizer, None)

    assert "heading_rate" not in off.extras and "bank_tv" not in off.extras
    torch.testing.assert_close(
        on.extras["heading_rate"], torch.tensor([16.0, 24.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        on.extras["bank_tv"], torch.tensor([0.1, 0.2], dtype=torch.float64)
    )
    torch.testing.assert_close(on.state, off.state)


def test_the_components_are_registered_only_when_their_weight_is_non_zero():
    """A term missing from ``loss_component_names`` is a KeyError on the first batch,
    after the slow dataset build."""
    off = _config(4)
    assert "heading_rate" not in objective.loss_component_names(off)
    assert "bank_tv" not in objective.loss_component_names(off)

    on = _config(4, control_heading_rate_loss_weight=1.0, control_bank_tv_loss_weight=1.0)
    names = objective.loss_component_names(on)
    assert "heading_rate" in names and "bank_tv" in names
    # Beside, not instead of: the four base components are untouched.
    assert names[:4] == ("state", "final_time", "kinematic", "terminal")


def test_a_weight_that_no_objective_would_build_is_refused():
    """Both terms are built by the true-time-position objective only. Under any other the
    weight could not change an answer, so the config refuses it rather than ignoring it."""
    with pytest.raises(ValueError, match="control_heading_rate_loss_weight"):
        _config(
            4,
            control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
            control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
            control_heading_rate_loss_weight=1.0,
        )
    with pytest.raises(ValueError, match="control_bank_tv_loss_weight"):
        _config(
            4,
            control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
            control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
            control_bank_tv_loss_weight=1.0,
        )


def test_negative_weights_and_a_non_positive_scale_are_refused():
    with pytest.raises(ValueError, match="control_heading_rate_loss_weight"):
        _config(4, control_heading_rate_loss_weight=-1.0)
    with pytest.raises(ValueError, match="control_bank_tv_loss_weight"):
        _config(4, control_bank_tv_loss_weight=-0.5)
    for scale in (0.0, -1.5, math.inf, math.nan):
        with pytest.raises(ValueError, match="control_heading_rate_loss_scale_dps"):
            _config(4, control_heading_rate_loss_scale_dps=scale)


def test_the_scale_is_the_unit_the_residual_is_read_in():
    """1.5 deg/s is half a standard-rate turn: a full standard-rate error costs 4."""
    config = _config(4, control_heading_rate_loss_weight=1.0)
    assert config.control_heading_rate_loss_scale_dps == 1.5
    dynamics, controls, durations = _rollout_dynamics(0.3, config=config)
    truth = _model_side_dps(
        control_rollout.rollout_control_endpoints(controls, durations, dynamics, config),
        dynamics,
    )[0].tolist()
    off_by_a_standard_rate, _ = _endpoint_result(
        config, [0.3] * 4, [value + 3.0 for value in truth], [1.0] * 4
    )
    assert float(off_by_a_standard_rate.control_heading_rate_mse[0]) == pytest.approx(4.0)


# ── recipes, naming and the whole loop ───────────────────────────────────────

def test_the_recipe_definitions_pin_the_terms_off():
    """Every named recipe is simple-v3 or older, i.e. the TEACHER line. L1.b's terms are
    what the teacher is measured against, so a recipe must spell them off as literals."""
    literals = control_simple_v1_overrides()
    assert literals["control_heading_rate_loss_weight"] == 0.0
    assert literals["control_heading_rate_loss_scale_dps"] == 1.5
    assert literals["control_bank_tv_loss_weight"] == 0.0
    for recipe in ("simple-v1", "simple-v1-lag", "simple-v2", CONTROL_RECIPE_SIMPLE_V3):
        overrides = control_recipe_overrides(recipe)
        assert overrides["control_heading_rate_loss_weight"] == 0.0
        assert overrides["control_bank_tv_loss_weight"] == 0.0


def test_the_terms_name_the_run_only_when_they_leave_the_default():
    plain = TSConfig(**recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=True))
    assert run_display_name(plain.to_dict()).split(" · ")[3] == "simple-v3"

    settings = dict(recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False))
    settings.update(
        control_heading_rate_loss_weight=8.0, control_bank_tv_loss_weight=1.0
    )
    armed = TSConfig(**settings)
    loss = run_display_name(armed.to_dict()).split(" · ")[3]
    assert "hr=8" in loss and "bank-tv=1" in loss
    # The scale rides along only when it too is moved off the recipe's literal.
    assert "hr-scale" not in loss
    rescaled = replace(armed, control_heading_rate_loss_scale_dps=3.0)
    assert "hr-scale=3" in run_display_name(rescaled.to_dict())


def test_the_state_default_name_is_unchanged_by_the_new_fields():
    assert run_display_name(TSConfig().to_dict()) == (
        "state · iTransformer · kinematic · state-v1"
    )


def _screening_config(tmp_path: Path, **overrides) -> TSConfig:
    """The L1.b screening base, shrunk to a two-epoch synthetic run."""
    settings = dict(recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False))
    settings.update(
        n_segments=8,
        control_imitation_loss_weight=0.0,
        seq_len=8, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu",
        epochs=2, patience=2, batch_size=8, dropout=0.0,
        control_rollout_integrator_dt_s=0.5,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
    )
    settings.update(overrides)
    return TSConfig(**settings)


def _train_screening(tmp_path: Path, name: str, **overrides):
    config = _screening_config(tmp_path, **overrides)
    series, report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3), config, airport=AIRPORT
    )
    assert report.built == 8, report.format()
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64,
                       "source_records": []}],
    }
    torch.manual_seed(0)
    train(series, config, output_dir=tmp_path / name, data_provenance=provenance,
          verbose=False)
    return series, tmp_path / name


def test_a_screening_run_records_both_terms_and_they_change_the_schedule(tmp_path: Path):
    """The end-to-end contract: both components reach history.json with finite values, and
    training THROUGH them lands on a different schedule than the same seed without them."""
    _series, plain_dir = _train_screening(tmp_path, "plain")
    series, armed_dir = _train_screening(
        tmp_path, "armed",
        control_heading_rate_loss_weight=8.0,
        control_bank_tv_loss_weight=1.0,
    )

    armed = json.loads((armed_dir / "history.json").read_text())["history"]
    for epoch in armed:
        for name in ("heading_rate", "bank_tv"):
            assert name in epoch["train_components"], name
            assert math.isfinite(epoch["train_components"][name])
        assert epoch["train_components"]["heading_rate"] > 0.0
    # The control head starts with a zeroed projection, so every segment carries the same
    # neutral command and the FIRST epoch's total variation is exactly zero. It becomes
    # positive as soon as the schedule stops being flat — which is what the term prices.
    assert armed[0]["train_components"]["bank_tv"] == 0.0
    assert armed[-1]["train_components"]["bank_tv"] > 0.0
    plain = json.loads((plain_dir / "history.json").read_text())["history"]
    assert "heading_rate" not in plain[0]["train_components"]
    assert "bank_tv" not in plain[0]["train_components"]

    schedules = []
    for run in (plain_dir, armed_dir):
        model, config, normalizer, _payload = load_checkpoint(run / "checkpoint.pt")
        forecast = forecast_approach(
            model, series[0], config, normalizer, device=torch.device("cpu")
        )
        schedules.append(forecast.controls)
    assert not np.allclose(schedules[0], schedules[1])
