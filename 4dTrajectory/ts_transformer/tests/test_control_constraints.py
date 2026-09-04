"""The two command-hook constraint modules: barrier filter and nominal law + residual."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (TS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import final_approach_geometry as fag  # noqa: E402
from config import (  # noqa: E402
    CONTROL_DYNAMICS_FIRST_ORDER_LAG, CONTROL_HOOK_BARRIER, CONTROL_HOOK_NOMINAL_RESIDUAL,
    HOOK_SATURATION_HARD, PREDICTION_CONTROL, TSConfig, recipe_settings,
)
from control.constraints import BarrierFilter, NominalResidual, build_command_hook  # noqa: E402
import control.constraints.nominal_residual as nominal_residual_module  # noqa: E402
from control.constraints.gates import on_final_weight, runway_axes_view  # noqa: E402
from control.dynamics import rollout as control_rollout  # noqa: E402
from control.dynamics.hooks import RolloutStateView  # noqa: E402
from control.envelope import MAX_BANK_RAD  # noqa: E402
from control.guidance_laws import glidepath_load_factor, l1_bank, speed_hold_thrust  # noqa: E402
from coordinate_frames import ENUFrame  # noqa: E402
from dataset import Normalizer, build_series, dynamics_arrays  # noqa: E402
from evaluation.records import record_from_dict  # noqa: E402
from export import build_prediction_record  # noqa: E402
from flight_scenarios.runway_target import find_threshold  # noqa: E402
import forecast as forecast_module  # noqa: E402
from forecast import forecast_approach  # noqa: E402
from prediction_outputs import ControlPrediction  # noqa: E402
from run_naming import run_display_name  # noqa: E402
from synthetic import synthetic_arrivals  # noqa: E402
from train import fit_model  # noqa: E402

AIRPORT, RUNWAY = "KRDU", "05L"
TAN_GPA = math.tan(math.radians(3.0))


HOLD_S = 5.0   # about the deployed hold: 64 segments over a p50 328 s arrival


def _view(d_m, xt_m, *, heading_error_rad=0.0, height_above_gp_m=0.0, speed=70.0, psi_rwy=0.0,
          vertical_speed=None, hold_s=HOLD_S, bank_now_rad=0.0, reference_speed=None) -> RolloutStateView:
    """A chart state (runway course psi_rwy) at ``d`` back, ``xt`` right, on the final. The
    reference (the unhooked schedule's state) is the same state, at ``reference_speed`` if given."""
    d, xt = torch.as_tensor(d_m, dtype=torch.float64), torch.as_tensor(xt_m, dtype=torch.float64)
    ue, un = math.cos(psi_rwy), math.sin(psi_rwy)
    e, n = -d * ue + xt * un, -d * un - xt * ue
    u = d.clamp(min=0.0) * TAN_GPA + height_above_gp_m
    heading = psi_rwy + heading_error_rad
    vu = torch.full_like(d, -speed * TAN_GPA if vertical_speed is None else vertical_speed)
    chart = torch.stack([e, n, u, torch.full_like(d, speed * math.cos(heading)),
                         torch.full_like(d, speed * math.sin(heading)), vu, torch.full_like(d, 66000.0)], dim=-1)
    # Actuators being flown: trim thrust, the given bank, a level-flight load factor (the
    # barrier reads the lift factor n·cos μ from here — zeros would halve every bound).
    actuators = torch.tensor([[0.1, bank_now_rad, 1.0]], dtype=torch.float64).expand(len(d), -1).clone()
    reference_chart = chart.clone()
    if reference_speed is not None:
        reference_chart[:, 3:6] *= reference_speed / speed
    reference = RolloutStateView(chart=reference_chart, actuators=actuators.clone(), duration_s=torch.full_like(d, hold_s))
    return RolloutStateView(chart=chart, actuators=actuators, duration_s=torch.full_like(d, hold_s), reference=reference)


def _context(batch: int) -> dict[str, torch.Tensor]:
    return {"runway_heading_rad": torch.zeros(batch, dtype=torch.float64),
            "glidepath_tan": torch.full((batch,), TAN_GPA, dtype=torch.float64),
            "max_thrust_n": torch.full((batch,), 2.0e5, dtype=torch.float64)}


def _command(bank_rad, load=1.0, thrust=0.3) -> torch.Tensor:
    bank = torch.as_tensor(bank_rad, dtype=torch.float64)
    return torch.stack([torch.full_like(bank, thrust), bank, torch.full_like(bank, load)], dim=-1)


def _hook_config(**overrides) -> TSConfig:
    settings = {"prediction_output": PREDICTION_CONTROL, "control_dynamics_model": CONTROL_DYNAMICS_FIRST_ORDER_LAG,
                "control_dynamics_backend": "scaled-transport-chart-velocity", "seq_len": 8, "n_segments": 6}
    settings.update(overrides)
    return TSConfig(**settings)


# ── geometry adapter ────────────────────────────────────────────────────────

def test_runway_axes_view_reads_the_chart_state_correctly():
    view = runway_axes_view(_view([10_000.0, 5_000.0], [300.0, -200.0], heading_error_rad=0.1), torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(view.d, torch.tensor([10_000.0, 5_000.0], dtype=torch.float64))
    assert torch.allclose(view.xt, torch.tensor([300.0, -200.0], dtype=torch.float64))
    assert torch.allclose(view.heading_error, torch.full((2,), 0.1, dtype=torch.float64))
    assert torch.allclose(view.path_angle, torch.full((2,), -math.radians(3.0), dtype=torch.float64), atol=1e-6)
    # Gate: aligned rows inside the membership cone are on the final; a downwind row is not.
    assert on_final_weight(view, hard=True).tolist() == [1.0, 1.0]
    downwind = runway_axes_view(_view([5_000.0], [3_000.0], heading_error_rad=math.pi), torch.zeros(1, dtype=torch.float64))
    assert on_final_weight(downwind, hard=True).tolist() == [0.0]
    assert float(on_final_weight(downwind, hard=False)) < 1e-3


def test_corridor_halfwidth_slope_is_the_halfwidths_own_derivative():
    """The barrier's closing term must be d(hw)/dd of the SAME half-width the corridor uses —
    including the flat LTP width past the threshold, where the slope is zero."""
    d = torch.tensor([12_000.0, 3_000.0, 250.0, 0.0, -400.0], dtype=torch.float64)
    eps = 1e-3
    finite = (fag.corridor_halfwidth(d + eps) - fag.corridor_halfwidth(d - eps)) / (2 * eps)
    slope = fag.corridor_halfwidth_slope(d)
    assert torch.allclose(slope[:3], finite[:3], rtol=1e-6)
    assert slope[3] == 0.0 and slope[4] == 0.0 and finite[4] == 0.0


# ── barrier filter ───────────────────────────────────────────────────────────

def test_barrier_filter_leaves_a_centred_aligned_command_alone_and_bounds_a_diverging_one():
    config = _hook_config(control_command_hook=CONTROL_HOOK_BARRIER)
    context = _context(3)
    hook = BarrierFilter(config, context, hard=True)
    # On the centreline, aligned: any moderate bank passes (the interval is wide open).
    centred = hook(_view([8_000.0] * 3, [0.0] * 3), _command([0.0, 0.2, -0.2]), 0)
    assert torch.allclose(centred[:, 1], torch.tensor([0.0, 0.2, -0.2], dtype=torch.float64))
    # 20 m inside the right edge, heading 15° further right (psi_err < 0 moves right):
    # the filter demands a left turn, i.e. a bank at least some positive value.
    d = torch.tensor([8_000.0, 8_000.0, 8_000.0], dtype=torch.float64)
    edge = fag.K_MARGIN * fag.corridor_halfwidth(d) - 20.0
    diverging = hook(_view(d.tolist(), edge.tolist(), heading_error_rad=-math.radians(15.0)),
                     _command([-0.3, 0.0, 0.6]), 1)
    assert torch.all(diverging[:, 1] > 0.0)
    assert torch.all(diverging[:, 1] <= MAX_BANK_RAD)
    assert diverging[0, 1] == pytest.approx(diverging[1, 1])   # both lifted to the same demand
    assert diverging[2, 1] == pytest.approx(0.6)               # a command turning back hard enough passes
    # Outside the corridor (200 m past the right edge), aligned: still gated on (inside the
    # 500 m membership floor), and the barrier demands motion back toward the corridor.
    outside = hook(_view([8_000.0] * 3, [float(edge[0]) + 220.0] * 3), _command([0.0] * 3), 2)
    assert torch.all(outside[:, 1] > 0.0)
    # The same state held for a minute: the demanded rate is capped by the hold, so the
    # bound is milder — but still a left turn.
    long_hold = hook(_view([8_000.0] * 3, [float(edge[0]) + 220.0] * 3, hold_s=60.0), _command([0.0] * 3), 3)
    assert torch.all(long_hold[:, 1] > 0.0) and torch.all(long_hold[:, 1] < outside[:, 1])
    diagnostics = hook.diagnostics()
    # Clamped: the two diverging rows lifted to the demand and the three outside rows; the
    # minute-long hold's demand (~0.5°) sits at the "active" threshold and may not count.
    assert diagnostics["hook_steps"] == 12.0 and diagnostics["hook_clamped_steps"] >= 5.0


def test_barrier_filter_credits_the_bank_already_flown_and_keeps_the_vertical_lift():
    """Same state, same command; the aircraft already banked 25° left toward the interval
    needs less from the command than one flying wings level (the lag credit), and whatever
    bank the filter sets, the load factor keeps n·cos μ the network asked for."""
    config = _hook_config(control_command_hook=CONTROL_HOOK_BARRIER)
    hook = BarrierFilter(config, _context(1), hard=True)
    d = torch.tensor([17_000.0], dtype=torch.float64)
    edge = fag.K_MARGIN * fag.corridor_halfwidth(d) - 70.0
    command = _command([math.radians(7.0)], load=1.06)
    level = hook(_view(d.tolist(), edge.tolist(), heading_error_rad=-math.radians(30.0), speed=93.0, hold_s=7.0), command, 0)
    banked = hook(_view(d.tolist(), edge.tolist(), heading_error_rad=-math.radians(30.0), speed=93.0, hold_s=7.0,
                        bank_now_rad=math.radians(25.0)), command, 1)
    assert float(level[0, 1]) > math.radians(7.0)                  # the network's 7° is not enough
    assert float(banked[0, 1]) < float(level[0, 1])                 # the turn already under way counts
    for out in (level, banked):
        assert float(out[0, 2] * math.cos(float(out[0, 1]))) == pytest.approx(1.06 * math.cos(math.radians(7.0)), rel=1e-9)
    assert hook.diagnostics()["hook_load_change"] > 0.0


def test_barrier_filter_does_not_limit_cycle_through_the_lagged_rollout():
    """The first campaign's worst flight: joining the final 70 m inside the right edge at
    17 km, heading 25° right of the course, 7 s holds, τ_bank = 2 s, the network still
    banking +7° for three holds then wings level. The rate-only rule flipped +28° → −29° and
    steepened the path to 200 m/s; the lag-aware, load-coordinated rule must capture the
    heading in the first holds and then be quiet: no bank beyond 15° after the capture, the
    corridor kept, the glidepath kept, the speed kept."""
    config = _hook_config(control_command_hook=CONTROL_HOOK_BARRIER)
    d0, speed = 17_000.0, 93.0
    edge = fag.K_MARGIN * fag.corridor_halfwidth(torch.tensor([d0], dtype=torch.float64))[0].item() - 70.0
    dynamics, controls, durations = _final_batch(config, xt_m=edge, d_m=d0, segments=26, speed=speed,
                                                 heading_error_rad=-math.radians(25.0))
    network = controls.detach().clone()
    network[:, :3, 1] = math.radians(7.0)
    hook = build_command_hook(config.__class__(**{**config.to_dict(), "control_hook_saturation": HOOK_SATURATION_HARD}), dynamics)
    rollout = control_rollout.rollout_control_endpoints(network, durations, dynamics, config, command_hook=hook)
    psi = dynamics["runway_heading_rad"].to(rollout.channels.dtype)
    d_h, xt_h = fag.runway_axes(rollout.channels[..., 0], rollout.channels[..., 1], psi)
    last = int(_last_approach_index(d_h)[0])
    bank = rollout.controls[0, : last + 1, 1]
    assert float(bank[0]) > math.radians(20.0)                          # the capture: a real turn
    assert float(bank[2:].abs().max()) < math.radians(15.0), [round(math.degrees(b), 1) for b in bank.tolist()]
    bound = fag.K_MARGIN * fag.corridor_halfwidth(d_h)
    # Once captured the path may still bounce between the edges (a hold plus the lag makes
    # each correction about one hold late), but never by more than a hold's drift.
    assert torch.all((xt_h.abs() <= bound + 120.0)[0, 3 : last + 1]), (xt_h[0] - bound[0]).tolist()
    # Vertical: the fixture flies open loop (a trim load, no glidepath law), so the check is
    # that the coordinated load leaves the path angle and the speed alone — the rate-only
    # rule steepened this entry from −3° to −10° and doubled the speed.
    speeds = torch.hypot(rollout.channels[0, :, 3], rollout.channels[0, :, 4])
    path_angle = torch.atan2(rollout.channels[0, :, 5], speeds)
    assert torch.all(path_angle[: last + 1] > -math.radians(4.5)) and torch.all(path_angle[: last + 1] < 0.0)
    plain = control_rollout.rollout_control_endpoints(network, durations, dynamics, config)
    speeds_plain = torch.hypot(plain.channels[0, :, 3], plain.channels[0, :, 4])
    assert torch.all((speeds / speeds_plain)[: last + 1] < 1.05)         # no energy stolen from the vertical


def test_barrier_filter_soft_saturation_is_continuous_and_differentiable():
    config = _hook_config(control_command_hook=CONTROL_HOOK_BARRIER)
    hook = BarrierFilter(config, _context(2), hard=False)
    d = torch.tensor([8_000.0, 8_000.0], dtype=torch.float64)
    edge = fag.K_MARGIN * fag.corridor_halfwidth(d) - 20.0
    bank = torch.tensor([-0.3, 0.0], dtype=torch.float64, requires_grad=True)
    command = torch.stack([torch.full_like(bank, 0.3), bank, torch.ones_like(bank)], dim=-1)
    out = hook(_view(d.tolist(), edge.tolist(), heading_error_rad=-math.radians(15.0)), command, 0)
    out[:, 1].sum().backward()
    assert bank.grad is not None and torch.all(bank.grad > 0.0)     # soft: nonzero even when clamped
    hard = BarrierFilter(config, _context(2), hard=True)
    bank_h = bank.detach().clone().requires_grad_(True)
    command_h = torch.stack([torch.full_like(bank_h, 0.3), bank_h, torch.ones_like(bank_h)], dim=-1)
    hard(_view(d.tolist(), edge.tolist(), heading_error_rad=-math.radians(15.0)), command_h, 0)[:, 1].sum().backward()
    assert torch.all(bank_h.grad == 0.0)                             # hard: the dead zone


def test_barrier_filter_keeps_an_adversarial_rollout_inside_the_corridor():
    """Hard filter, gate on, a command that banks toward the right edge on every segment:
    the rollout may approach the edge but not cross it by more than one segment's drift."""
    config = _hook_config(control_command_hook=CONTROL_HOOK_BARRIER)
    dynamics, controls, durations = _final_batch(config, xt_m=100.0, d_m=9_000.0, segments=24)
    adversarial = controls.detach().clone()
    adversarial[:, :, 1] = -math.radians(20.0)                        # right turn every segment
    hook = build_command_hook(config.__class__(**{**config.to_dict(), "control_hook_saturation": HOOK_SATURATION_HARD}), dynamics)
    rollout = control_rollout.rollout_control_endpoints(adversarial, durations, dynamics, config, command_hook=hook)
    plain = control_rollout.rollout_control_endpoints(adversarial, durations, dynamics, config)
    psi = dynamics["runway_heading_rad"]
    def cross_track(channels):
        e, n = channels[..., 0], channels[..., 1]
        d, xt = fag.runway_axes(e, n, psi.to(e.dtype))
        return d, xt
    d_h, xt_h = cross_track(rollout.channels)
    d_p, xt_p = cross_track(plain.channels)
    bound = fag.K_MARGIN * fag.corridor_halfwidth(d_h)
    last = _last_approach_index(d_h)
    assert torch.all(_at(xt_p.abs(), last) > _at(fag.K_MARGIN * fag.corridor_halfwidth(d_p), last))   # unfiltered: leaves
    approach = d_h > 0.0
    assert torch.all((xt_h.abs() <= bound + 60.0)[approach])           # filtered: stays (segment-hold slack)
    assert torch.all(_at(xt_h.abs(), last) < _at(xt_p.abs(), last))


# ── nominal law + residual ───────────────────────────────────────────────────

def test_guidance_laws_point_back_to_the_centreline_and_glidepath():
    speed = torch.tensor([70.0], dtype=torch.float64)
    right = l1_bank(torch.tensor([300.0], dtype=torch.float64), torch.zeros(1, dtype=torch.float64), speed,
                    l1_distance_m=3000.0, bank_limit_rad=MAX_BANK_RAD)
    left = l1_bank(torch.tensor([-300.0], dtype=torch.float64), torch.zeros(1, dtype=torch.float64), speed,
                   l1_distance_m=3000.0, bank_limit_rad=MAX_BANK_RAD)
    assert float(right) > 0.0 and float(left) < 0.0 and float(right) == pytest.approx(-float(left))
    on_line = l1_bank(torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64), speed,
                      l1_distance_m=3000.0, bank_limit_rad=MAX_BANK_RAD)
    assert float(on_line) == 0.0
    gp = torch.tensor([TAN_GPA], dtype=torch.float64)
    gamma = torch.tensor([-math.radians(3.0)], dtype=torch.float64)
    high = glidepath_load_factor(torch.tensor([100.0], dtype=torch.float64), gamma, speed, torch.zeros(1, dtype=torch.float64),
                                 glidepath_tan=gp, lookahead_m=2000.0, gain_per_s=0.2, load_limits=(0.2, 2.0))
    low = glidepath_load_factor(torch.tensor([-100.0], dtype=torch.float64), gamma, speed, torch.zeros(1, dtype=torch.float64),
                                glidepath_tan=gp, lookahead_m=2000.0, gain_per_s=0.2, load_limits=(0.2, 2.0))
    on_path = glidepath_load_factor(torch.zeros(1, dtype=torch.float64), gamma, speed, torch.zeros(1, dtype=torch.float64),
                                    glidepath_tan=gp, lookahead_m=2000.0, gain_per_s=0.2, load_limits=(0.2, 2.0))
    assert float(high) < float(on_path) < float(low)                  # high → push over, low → pull up
    assert float(on_path) == pytest.approx(math.cos(float(gamma)))
    # Energy: slower than the unhooked schedule would be → more thrust, faster → less, the
    # same → the command's own; k·m·ΔV over the installed thrust, saturated at the box.
    mass, t_max = torch.tensor([66_000.0], dtype=torch.float64), torch.tensor([2.0e5], dtype=torch.float64)
    thrust = torch.tensor([0.1], dtype=torch.float64)
    def held(reference_speed):
        return speed_hold_thrust(thrust, speed, torch.tensor([reference_speed], dtype=torch.float64), gain_per_s=0.1,
                                 mass_kg=mass, max_thrust_n=t_max, thrust_limits=(-0.2, 1.0))
    assert float(held(70.0)) == pytest.approx(0.1)
    assert float(held(80.0)) == pytest.approx(0.1 + 0.1 * 66_000.0 * 10.0 / 2.0e5)
    assert float(held(60.0)) < 0.1
    assert float(held(200.0)) == 1.0 and float(held(0.0)) == -0.2


def test_nominal_residual_is_identity_off_the_final_and_a_bounded_band_on_it():
    config = _hook_config(control_command_hook=CONTROL_HOOK_NOMINAL_RESIDUAL)
    hook = NominalResidual(config, _context(2), hard=True)
    command = _command([0.4, -0.4], load=1.3)
    # Downwind (reversed heading): the gate is off, the command passes untouched.
    downwind = hook(_view([5_000.0, 5_000.0], [3_000.0, 3_000.0], heading_error_rad=math.pi), command, 0)
    assert torch.allclose(downwind, command)
    # On the final, 300 m right, aligned: the nominal bank is positive; the command is
    # held within ±5° of it, the load within ±0.1 of the glidepath law's.
    view = _view([8_000.0, 8_000.0], [300.0, 300.0])
    bank_nom, load_nom = hook.nominal(view)
    out = hook(view, command, 1)
    assert torch.all(bank_nom > 0.0)
    assert torch.allclose(out[:, 1], bank_nom + torch.tensor([1.0, -1.0], dtype=torch.float64) * config.control_nominal_residual_bank_max_rad)
    assert torch.allclose(out[:, 2], load_nom + config.control_nominal_residual_load_max)
    assert torch.allclose(out[:, 0], command[:, 0])                   # at the unhooked schedule's speed: thrust untouched
    # 10 m/s slower than the unhooked schedule would be: the thrust comes up by k·m·ΔV.
    slow = hook(_view([8_000.0, 8_000.0], [0.0, 0.0], reference_speed=80.0), command, 2)
    assert torch.allclose(slow[:, 0], command[:, 0] + config.control_nominal_speed_gain * 66_000.0 * 10.0 / 2.0e5, atol=2e-3)
    assert hook.diagnostics()["hook_thrust_change"] > 0.0
    diagnostics = hook.diagnostics()
    assert diagnostics["hook_gated_steps"] == 4.0 and diagnostics["hook_bank_residual_saturated_steps"] == 4.0


def test_nominal_residual_gradients_flow_through_the_residual_and_the_state():
    config = _hook_config(control_command_hook=CONTROL_HOOK_NOMINAL_RESIDUAL)
    hook = NominalResidual(config, _context(1), hard=False)
    bank = torch.tensor([0.02], dtype=torch.float64, requires_grad=True)
    command = torch.stack([torch.full_like(bank, 0.3), bank, torch.ones_like(bank)], dim=-1)
    view = _view([8_000.0], [300.0])
    chart = view.chart.clone().requires_grad_(True)
    out = hook(RolloutStateView(chart=chart, actuators=view.actuators, duration_s=view.duration_s, reference=view.reference), command, 0)
    out[:, 1].sum().backward()
    assert bank.grad is not None and float(bank.grad) > 0.0
    assert chart.grad is not None and torch.count_nonzero(chart.grad) > 0


@pytest.mark.parametrize("height_above_gp_m", [80.0, -120.0])
def test_nominal_law_converges_to_the_centreline_and_glidepath_through_the_rollout(monkeypatch, height_above_gp_m):
    """Residuals pinned near zero: from 300 m right and 80 m high (or 120 m low — the
    direction the first campaign's arm failed in), the tracked rollout ends much closer to
    both than the untracked one, at the untracked one's speed."""
    config = _hook_config(control_command_hook=CONTROL_HOOK_NOMINAL_RESIDUAL,
                          control_nominal_residual_bank_max_rad=1e-4, control_nominal_residual_load_max=1e-4)
    dynamics, controls, durations = _final_batch(config, xt_m=300.0, d_m=12_000.0, height_above_gp_m=height_above_gp_m, segments=32)
    hook = build_command_hook(config, dynamics)
    tracked = control_rollout.rollout_control_endpoints(controls, durations, dynamics, config, command_hook=hook)
    plain = control_rollout.rollout_control_endpoints(controls, durations, dynamics, config)
    psi = dynamics["runway_heading_rad"].to(tracked.channels.dtype)
    d_t, xt_t = fag.runway_axes(tracked.channels[..., 0], tracked.channels[..., 1], psi)
    d_p, xt_p = fag.runway_axes(plain.channels[..., 0], plain.channels[..., 1], psi)
    last = _last_approach_index(d_t)
    assert torch.all(_at(xt_t.abs(), last) < 60.0)
    assert torch.all(_at(xt_t.abs(), last) < _at(xt_p.abs(), last))
    height_error = tracked.channels[..., 2] - fag.glidepath_height(d_t, dynamics["glidepath_tan"].to(d_t.dtype))
    assert torch.all(_at(height_error.abs(), last) < 40.0)
    assert torch.all(_at(height_error.abs(), last) < height_error[:, 0].abs())
    # Energy: the plain rollout flies parallel to the glidepath at its offset; the law's
    # descent onto it (from above) releases height the schedule did not mean to release
    # and the climb onto it (from below) spends speed the schedule did not mean to spend —
    # with the thrust passed through the tracked rollout ends more than 10 m/s off the
    # plain one's speed, in opposite directions for the two starts (the first campaign's
    # arm, pulled up, arrived 30 m/s slow). The speed hold on the unhooked rollout keeps
    # it within 3 m/s.
    monkeypatch.setattr(nominal_residual_module, "speed_hold_thrust", lambda thrust, *args, **kwargs: thrust)
    passthrough = control_rollout.rollout_control_endpoints(
        controls, durations, dynamics, config, command_hook=build_command_hook(config, dynamics))
    speed_t = _at(torch.hypot(tracked.channels[..., 3], tracked.channels[..., 4]), last)
    speed_p = _at(torch.hypot(plain.channels[..., 3], plain.channels[..., 4]), last)
    speed_x = _at(torch.hypot(passthrough.channels[..., 3], passthrough.channels[..., 4]), last)
    assert torch.all((speed_x - speed_p).abs() > 10.0)
    assert torch.all((speed_t - speed_p).abs() < 3.0)


# ── batch fixture on the final ───────────────────────────────────────────────

def _last_approach_index(d: torch.Tensor) -> torch.Tensor:
    """Per row, the last segment endpoint still ≥ 300 m before the threshold (the fixture's
    holds are sized for a steady speed; the aircraft may reach the threshold a few segments
    early, and past it the corridor and the gate are undefined)."""
    on_approach = d > 300.0
    assert torch.all(on_approach[:, 0])
    return on_approach.sum(dim=1) - 1


def _at(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return values[torch.arange(values.shape[0]), index]


# Thrust fraction that roughly holds 72 m/s on the 3° path for this fleet's polar (measured on
# the fixture: idle decelerates into the stall handling, 0.3 accelerates to 190 m/s).
_TRIM_THRUST = 0.1


def _final_batch(config: TSConfig, *, xt_m: float, d_m: float, height_above_gp_m: float = 0.0, segments: int = 6,
                 speed: float = 72.0, heading_error_rad: float = 0.0):
    """A synthetic batch whose initial state sits on the final at (d, xt, +height)."""
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=2, seed=9)
    series, _ = build_series(flights, config, airport=AIRPORT)
    anchor = config.seq_len - 1
    rows = [dynamics_arrays(item, anchor) for item in series]
    dynamics = {key: torch.from_numpy(np.stack([row[key] for row in rows])) for key in rows[0]}
    threshold = find_threshold(AIRPORT, RUNWAY)
    frame = ENUFrame(lat0=float(threshold["lat"]), lon0=float(threshold["lon"]), alt0=float(threshold["elevation_m"]))
    psi = float(series[0].scenario.target.psi)
    ue, un = math.cos(psi), math.sin(psi)
    e, n = -d_m * ue + xt_m * un, -d_m * un - xt_m * ue
    u = d_m * TAN_GPA + series[0].scenario.target.altitude - float(threshold["elevation_m"]) + height_above_gp_m
    lat, lon = frame.latlon_from_horizontal(e, n)
    alt = frame.alt0 + u
    for row in range(len(series)):
        dynamics["initial_state"][row] = torch.tensor([lat, lon, alt, speed, psi + heading_error_rad, -math.radians(3.0), 66000.0], dtype=torch.float64)
        dynamics["initial_controls"][row] = torch.tensor([_TRIM_THRUST, 0.0, math.cos(math.radians(3.0))], dtype=torch.float64)
    controls = torch.zeros((len(series), segments, 3), dtype=torch.float32)
    controls[:, :, 0], controls[:, :, 2] = _TRIM_THRUST, math.cos(math.radians(3.0))
    controls.requires_grad_(True)
    durations = torch.full((len(series), segments), d_m / speed / segments)   # the hold that reaches the threshold
    return dynamics, controls, durations


# ── config, naming, training ─────────────────────────────────────────────────

def test_hook_config_is_guarded_and_named():
    assert build_command_hook(_hook_config(), _context(1)) is None
    with pytest.raises(ValueError, match="first-order-lag"):
        TSConfig(prediction_output=PREDICTION_CONTROL, control_command_hook=CONTROL_HOOK_BARRIER)
    with pytest.raises(ValueError, match="control output"):
        TSConfig(control_command_hook=CONTROL_HOOK_BARRIER)
    with pytest.raises(ValueError, match="on-final"):
        _hook_config(control_command_hook=CONTROL_HOOK_BARRIER, control_hook_gate="faf")
    with pytest.raises(ValueError, match="unknown control_hook_saturation"):
        _hook_config(control_command_hook=CONTROL_HOOK_BARRIER, control_hook_saturation="never")
    named = TSConfig(**recipe_settings("simple-v3", keep_name=True), control_command_hook=CONTROL_HOOK_NOMINAL_RESIDUAL)
    assert "hook=nominal-residual" in run_display_name(named.to_dict())
    assert "hook=" not in run_display_name(TSConfig(**recipe_settings("simple-v3", keep_name=True)).to_dict())


def test_training_refuses_hard_saturation_and_logs_the_hook(tmp_path):
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    hard = _hook_config(control_command_hook=CONTROL_HOOK_BARRIER, control_hook_saturation=HOOK_SATURATION_HARD,
                        epochs=1, patience=1, batch_size=32, d_model=16, n_heads=4, d_ff=32, e_layers=1, device="cpu")
    series, _ = build_series(flights, hard, airport=AIRPORT)
    with pytest.raises(ValueError, match="hard hook saturation"):
        fit_model(series[:8], series[8:], hard, verbose=False)
    soft = _hook_config(control_command_hook=CONTROL_HOOK_NOMINAL_RESIDUAL, n_segments=4,
                        epochs=1, patience=1, batch_size=32, d_model=16, n_heads=4, d_ff=32, e_layers=1, device="cpu")
    series, _ = build_series(flights, soft, airport=AIRPORT)
    fit = fit_model(series[:8], series[8:], soft, verbose=False)
    record = fit.history[0].command_hook
    assert record["steps"] > 0 and 0.0 <= record["gated_steps"] <= 1.0


# ── prediction-time export contract ──────────────────────────────────────────

def test_prediction_exports_the_effective_schedule_and_names_the_hook(monkeypatch):
    """What ``F_barrier_infer`` rests on: at predict time the hook rewrites the schedule, the
    record carries the FLOWN controls, and says which hook did it. The hook is a stand-in
    that rewrites every bank (the synthetic anchor is not on the final), so the assertion is
    about the export path, not the geometry."""
    config = _hook_config(control_command_hook=CONTROL_HOOK_BARRIER, control_hook_saturation=HOOK_SATURATION_HARD,
                          n_segments=2)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=1, seed=9)
    series, _ = build_series(flights, config, airport=AIRPORT)
    normalizer = Normalizer.fit(series)

    class RewritingHook:
        needs_reference = False

        def __call__(self, state, command, segment_index):
            return torch.stack((command[:, 0], torch.full_like(command[:, 1], 0.25), command[:, 2]), dim=-1)

        def diagnostics(self):
            return {}

    monkeypatch.setattr(forecast_module, "build_command_hook", lambda cfg, dynamics: RewritingHook())

    class FixedControlModel(torch.nn.Module):
        def forward(self, history, dynamics):
            controls = torch.tensor([[[0.20, 0.04, 1.01], [0.16, -0.02, 0.99]]], dtype=history.dtype).expand(len(history), -1, -1)
            durations = torch.tensor([[6.0, 6.0]], dtype=history.dtype).expand(len(history), -1)
            return ControlPrediction(controls=controls, segment_durations=durations, final_time_s=durations.sum(dim=-1))

    forecast = forecast_approach(FixedControlModel(), series[0], config, normalizer, device=torch.device("cpu"))
    assert forecast.command_hook == "barrier/hard"
    record = build_prediction_record(series[0], forecast, index=0, model_name=config.model, horizon_mode=config.horizon_mode)
    assert record.source["commandHook"] == "barrier/hard"
    parsed = record_from_dict(record.eval_record)
    assert [row["bank_rad"] for row in parsed.controls] == pytest.approx([0.25] * len(parsed.controls))
    assert [row["bank_rad"] for row in record.states_payload["control_segments"]] == pytest.approx([0.25, 0.25])
    max_thrust_n = series[0].scenario.aircraft.engine.max_thrust_total_n
    assert parsed.controls[0]["thrust"] == pytest.approx(0.20 * max_thrust_n)     # untouched channels pass
    assert parsed.controls[-1]["thrust"] == pytest.approx(0.16 * max_thrust_n)
