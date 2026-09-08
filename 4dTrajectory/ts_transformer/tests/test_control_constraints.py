"""The command-hook constraint module that stayed: the barrier filter and its gate.

The nominal tracking law that shared this file was never adopted and is archived
(`archive/nominal_law_hook_2026_09/`); what remains of it here is the contract that its
retired vocabulary value still NAMES a stored run and no longer BUILDS one.
"""

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
    CONTROL_DYNAMICS_FIRST_ORDER_LAG, CONTROL_HOOK_BARRIER,
    CONTROL_HOOK_BARRIER_SPEED_FLOOR, CONTROL_HOOK_MEMBERS, CONTROL_HOOK_NOMINAL_RESIDUAL,
    CONTROL_HOOK_SPEED_FLOOR, CONTROL_HOOKS, CONTROL_HOOKS_AVAILABLE,
    CONTROL_SPEED_FLOOR_MARGIN_DEFAULT, HOOK_SATURATION_HARD, PREDICTION_CONTROL, TSConfig,
    recipe_settings,
)
from control.constraints import (  # noqa: E402
    BarrierFilter, CompositeHook, SpeedFloor, build_command_hook,
)
from control.envelope import MAX_THRUST_FRACTION, MIN_THRUST_FRACTION  # noqa: E402
from flyability import flyability_summary, required_controls  # noqa: E402
from dataclasses import fields as dataclass_fields  # noqa: E402

from control.constraints.gates import on_final_weight, runway_axes_view  # noqa: E402
from control.dynamics import rollout as control_rollout  # noqa: E402
from control.dynamics.hooks import RolloutStateView  # noqa: E402
from control.envelope import MAX_BANK_RAD  # noqa: E402
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
    """The per-flight dynamics rows a hook may read. ``aero_params`` is `dataset`'s own
    probe row (S, Cl_max, ...) and the frame origin sits at sea level, so the speed floor's
    ISA density is read straight off the chart height ``_view`` builds."""
    return {"runway_heading_rad": torch.zeros(batch, dtype=torch.float64),
            "glidepath_tan": torch.full((batch,), TAN_GPA, dtype=torch.float64),
            "max_thrust_n": torch.full((batch,), 2.0e5, dtype=torch.float64),
            "aero_params": torch.tensor([[122.6, 2.7, 0.02, 0.04, 0.9, 0.1]], dtype=torch.float64).expand(batch, -1).clone(),
            "frame_params": torch.tensor([[35.9, -78.8, 0.0, 0.0]], dtype=torch.float64).expand(batch, -1).clone()}


def _floor_mps(height_m: float, *, mass_kg: float, area_m2: float, cl_max: float,
               load: float = 1.0, margin: float = CONTROL_SPEED_FLOOR_MARGIN_DEFAULT) -> float:
    """The hook's own floor, written out independently of it: the stall speed of the
    project's stall model at the ISA density of ``height_m``, times the margin. The
    airframe row is passed in — this fleet's Cl_max is per type, and reading a floor
    against the wrong one is exactly the mis-scoring `flyability_batch` exists to avoid."""
    density = 1.225 * ((288.15 - 0.0065 * height_m) / 288.15) ** 4.25588
    return margin * math.sqrt(2.0 * load * mass_kg * 9.81 / (density * area_m2 * cl_max))


def _stall_readout(rollout, dynamics, durations, aircraft):
    """``(flyability summary, the smallest V - V_floor over the samples)`` for row 0."""
    times = torch.cumsum(durations, dim=1)
    area, cl_max = (float(value) for value in dynamics["aero_params"][0, :2])
    states = [
        {"t": float(times[0, step]), "lat": float(row[0]), "lon": float(row[1]),
         "alt": float(row[2]), "V": float(row[3]), "psi": float(row[4]),
         "gamma": float(row[5]), "m": float(row[6])}
        for step, row in enumerate(rollout.geodetic_states[0])
    ]
    summary = flyability_summary(required_controls(states, aircraft), aircraft_code=aircraft.code)
    slack = min(
        state["V"] - _floor_mps(state["alt"], mass_kg=state["m"], area_m2=area, cl_max=cl_max)
        for state in states
    )
    return summary, slack


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


# ── speed floor ──────────────────────────────────────────────────────────────

def test_speed_floor_acts_on_the_thrust_alone_and_only_upward():
    """The floor owns ONE channel. Whatever it does to the thrust, the bank and the load
    factor it was handed come back bit-identical — that is what lets it compose with the
    barrier, which owns the other two — and a command already fast enough is not slowed."""
    config = _hook_config(control_command_hook=CONTROL_HOOK_SPEED_FLOOR)
    hook = SpeedFloor(config, _context(2), hard=True)
    command = _command([0.3, -0.2], load=1.15, thrust=0.05)
    # 70 m/s at 8 km back is above the floor there (about 63 m/s at n = 1.15); 45 m/s is not.
    fast = hook(_view([8_000.0] * 2, [0.0] * 2, speed=70.0), command, 0)
    slow = hook(_view([8_000.0] * 2, [0.0] * 2, speed=45.0), command, 1)
    for out in (fast, slow):
        assert torch.equal(out[:, 1], command[:, 1]) and torch.equal(out[:, 2], command[:, 2])
    assert torch.allclose(fast[:, 0], command[:, 0])          # nothing demanded, nothing changed
    assert torch.all(slow[:, 0] > command[:, 0])              # below the floor: thrust raised
    assert torch.all(slow[:, 0] <= MAX_THRUST_FRACTION)
    diagnostics = hook.diagnostics()
    assert diagnostics["hook_steps"] == 4.0 and diagnostics["hook_floor_bound_steps"] == 2.0
    assert diagnostics["hook_thrust_change"] > 0.0


def test_speed_floor_reads_the_commanded_load_factor_and_the_thrust_already_spooled():
    """Two properties the lesson of the barrier's v1/v2 pair demands of any hook here.

    (i) The floor is ``V_stall(n_commanded)``, so a command that pulls more g asks for more
    speed — this is what makes the composite well ordered (the barrier re-coordinates the
    load first, the floor then prices what will actually be flown). (ii) The command is
    HELD and the thrust actuator lags, so an engine already spooled up has committed part
    of the hold's acceleration and the command has to supply only the rest.
    """
    config = _hook_config(control_command_hook=CONTROL_HOOK_SPEED_FLOOR)
    hook = SpeedFloor(config, _context(1), hard=True)
    view = _view([8_000.0], [0.0], speed=58.0)
    level = hook(view, _command([0.0], load=1.0, thrust=0.0), 0)
    pulling = hook(view, _command([0.0], load=1.6, thrust=0.0), 1)
    assert float(pulling[0, 0]) > float(level[0, 0])
    # The idle engine has to be commanded harder than the one already at 60 % of installed
    # thrust, for the same state and the same demanded end-of-hold speed.
    idle_view = _view([8_000.0], [0.0], speed=58.0)
    idle_view.actuators[:, 0] = 0.0
    spooled_view = _view([8_000.0], [0.0], speed=58.0)
    spooled_view.actuators[:, 0] = 0.6
    idle = hook(idle_view, _command([0.0], thrust=0.0), 2)
    spooled = hook(spooled_view, _command([0.0], thrust=0.0), 3)
    assert float(idle[0, 0]) > float(spooled[0, 0])


def test_the_soft_speed_floor_is_inert_where_it_demands_nothing():
    """The regression the soft form was one clamp away from having.

    `soft_max` overshoots its bound by `softness x ln 2`, so parking a demand that binds
    NOTHING at `MIN_THRUST_FRACTION` would have added 0.0139 of installed thrust — 2.8 kN on
    this fixture, 0.042 m/s^2, ~12 m/s over a 300 s remainder — to every idle command, and
    `[-0.2, -0.1]` is the COMMON band on an approach, not a corner. `soft` is the ADOPTED
    delivery form, so the arm would have measured the bias rather than the floor. The demand
    is parked far enough below the box that the soft and the hard forms agree exactly."""
    config = _hook_config(control_command_hook=CONTROL_HOOK_SPEED_FLOOR)
    soft = SpeedFloor(config, _context(3), hard=False)
    hard = SpeedFloor(config, _context(3), hard=True)
    # 8 km back, comfortably fast: the floor is ~63 m/s and demands full negative thrust.
    view = _view([8_000.0] * 3, [0.0] * 3, speed=95.0)
    command = _command([0.0] * 3, thrust=MIN_THRUST_FRACTION)
    assert torch.equal(soft(view, command, 0)[:, 0], hard(view, command, 0)[:, 0])
    assert torch.equal(soft(view, command, 1)[:, 0], command[:, 0])
    # ...and neither form counts a step it did not act on.
    assert soft.diagnostics()["hook_floor_bound_steps"] == 0.0
    assert soft.diagnostics()["hook_thrust_change"] == 0.0
    # The bound case is unaffected: both forms still lift a command that would stall.
    slow = _view([8_000.0], [0.0], speed=45.0)
    lift = _command([0.0], thrust=MIN_THRUST_FRACTION)
    assert float(SpeedFloor(config, _context(1), hard=False)(slow, lift, 0)[0, 0]) == pytest.approx(MAX_THRUST_FRACTION)


def test_the_speed_floor_counts_a_demand_the_engine_cannot_meet():
    """`hook_floor_saturated_steps` is "full thrust is not enough", so it must fire on the
    DEMAND, not on the change: a network already commanding full thrust into a floor it
    cannot reach is the case the counter exists to surface, and gating it on the change
    would report zero exactly there."""
    config = _hook_config(control_command_hook=CONTROL_HOOK_SPEED_FLOOR)
    hook = SpeedFloor(config, _context(1), hard=True)
    view = _view([8_000.0], [0.0], speed=40.0)
    out = hook(view, _command([0.0], thrust=MAX_THRUST_FRACTION), 0)
    assert float(out[0, 0]) == pytest.approx(MAX_THRUST_FRACTION)   # nothing left to give
    diagnostics = hook.diagnostics()
    assert diagnostics["hook_floor_saturated_steps"] == 1.0
    assert diagnostics["hook_floor_bound_steps"] == 0.0             # it changed nothing
    assert diagnostics["hook_thrust_change"] == 0.0


def test_speed_floor_soft_saturation_is_continuous_and_differentiable():
    config = _hook_config(control_command_hook=CONTROL_HOOK_SPEED_FLOOR)
    view = _view([8_000.0, 8_000.0], [0.0, 0.0], speed=45.0)
    thrust = torch.tensor([-0.1, 0.0], dtype=torch.float64, requires_grad=True)
    command = torch.stack([thrust, torch.zeros_like(thrust), torch.ones_like(thrust)], dim=-1)
    SpeedFloor(config, _context(2), hard=False)(view, command, 0)[:, 0].sum().backward()
    assert thrust.grad is not None and torch.all(thrust.grad > 0.0)   # soft: alive when bound
    hard_thrust = thrust.detach().clone().requires_grad_(True)
    hard_command = torch.stack(
        [hard_thrust, torch.zeros_like(hard_thrust), torch.ones_like(hard_thrust)], dim=-1
    )
    SpeedFloor(config, _context(2), hard=True)(view, hard_command, 0)[:, 0].sum().backward()
    assert torch.all(hard_thrust.grad == 0.0)                        # hard: the dead zone


@pytest.mark.parametrize("saturation", ["soft", HOOK_SATURATION_HARD])
def test_speed_floor_keeps_a_decelerating_rollout_off_the_stall(saturation):
    """L3.d's whole point, end to end. The network commands idle-minus (the envelope's
    negative floor) for the length of a 14 km approach: the unhooked rollout decelerates
    through the stall speed and ``flyability`` reports stall samples on it. With the hook —
    which may only raise the thrust — the same schedule stays above the floor and the same
    check reports none. The tolerance is the mid-hold dip a per-segment command allows: the
    floor is imposed where the hold ENDS, and drag and density are frozen across it."""
    config = TSConfig(**{**_hook_config(control_command_hook=CONTROL_HOOK_SPEED_FLOOR).to_dict(),
                         "control_hook_saturation": saturation})
    dynamics, controls, durations = _final_batch(config, xt_m=0.0, d_m=14_000.0, segments=24, speed=95.0)
    decelerating = controls.detach().clone()
    decelerating[:, :, 0] = MIN_THRUST_FRACTION
    hook = build_command_hook(config, dynamics)
    hooked = control_rollout.rollout_control_endpoints(decelerating, durations, dynamics, config, command_hook=hook)
    plain = control_rollout.rollout_control_endpoints(decelerating, durations, dynamics, config)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=2, seed=9)
    series, _ = build_series(flights, config, airport=AIRPORT)
    aircraft = series[0].scenario.aircraft
    plain_summary, plain_slack = _stall_readout(plain, dynamics, durations, aircraft)
    hooked_summary, hooked_slack = _stall_readout(hooked, dynamics, durations, aircraft)
    assert plain_summary["violations"]["stall"] > 0 and plain_slack < -5.0
    assert hooked_summary["violations"].get("stall", 0) == 0 and hooked_summary["fully_flyable"]
    assert hooked_slack > -1.0, hooked_slack
    # Only the thrust moved: the bank and load schedules are the network's, unchanged.
    assert torch.equal(hooked.controls[:, :, 1:], decelerating.to(hooked.controls.dtype)[:, :, 1:])
    assert torch.all(hooked.controls[:, :, 0] >= decelerating.to(hooked.controls.dtype)[:, :, 0] - 1e-9)


# ── composition ──────────────────────────────────────────────────────────────

def test_the_combined_hook_is_the_barrier_then_the_floor_on_disjoint_channels():
    """One call, both modules: the bank and load are exactly what the barrier alone would
    have set (the floor never touches them) and the thrust is the floor's, above what the
    network asked for. The order is the vocabulary's, not a free choice."""
    config = _hook_config(control_command_hook=CONTROL_HOOK_BARRIER_SPEED_FLOOR,
                          control_hook_saturation=HOOK_SATURATION_HARD)
    context = _context(1)
    combined = build_command_hook(config, context)
    assert isinstance(combined, CompositeHook)
    barrier_only = BarrierFilter(config, context, hard=True)
    d = torch.tensor([8_000.0], dtype=torch.float64)
    edge = fag.K_MARGIN * fag.corridor_halfwidth(d) - 20.0
    view = _view(d.tolist(), edge.tolist(), heading_error_rad=-math.radians(15.0), speed=45.0)
    command = _command([0.0], thrust=-0.1)
    both = combined(view, command, 0)
    barrier = barrier_only(view, command, 0)
    assert torch.equal(both[:, 1], barrier[:, 1]) and torch.equal(both[:, 2], barrier[:, 2])
    assert float(both[0, 1]) > 0.0                      # the barrier did act
    assert float(both[0, 0]) > float(command[0, 0])     # and so did the floor
    merged = combined.diagnostics()
    assert merged["hook_steps"] == 1.0                  # one shared step count, not two
    assert merged["hook_clamped_steps"] == 1.0 and merged["hook_floor_bound_steps"] == 1.0
    per_flight = combined.per_flight_diagnostics()
    assert set(per_flight) == set(merged)
    assert all(value.shape == (1,) for value in per_flight.values())


def test_the_combined_hook_still_contains_the_corridor():
    """The barrier's own containment test, re-run under `barrier+speed-floor`: adding the
    speed floor must not cost the lateral guarantee the corridor arm was adopted for."""
    config = _hook_config(control_command_hook=CONTROL_HOOK_BARRIER_SPEED_FLOOR,
                          control_hook_saturation=HOOK_SATURATION_HARD)
    dynamics, controls, durations = _final_batch(config, xt_m=100.0, d_m=9_000.0, segments=24)
    adversarial = controls.detach().clone()
    adversarial[:, :, 1] = -math.radians(20.0)                        # right turn every segment
    adversarial[:, :, 0] = MIN_THRUST_FRACTION                        # ...and decelerating
    # ...with the load factor coordinated for that bank, so the fixture holds the 3° path
    # instead of diving (a dive would trade the deceleration back for speed and the floor
    # would have nothing to do).
    adversarial[:, :, 2] = math.cos(math.radians(3.0)) / math.cos(math.radians(20.0))
    hook = build_command_hook(config, dynamics)
    rollout = control_rollout.rollout_control_endpoints(adversarial, durations, dynamics, config, command_hook=hook)
    plain = control_rollout.rollout_control_endpoints(adversarial, durations, dynamics, config)
    psi = dynamics["runway_heading_rad"]
    d_h, xt_h = fag.runway_axes(rollout.channels[..., 0], rollout.channels[..., 1], psi.to(rollout.channels.dtype))
    d_p, xt_p = fag.runway_axes(plain.channels[..., 0], plain.channels[..., 1], psi.to(plain.channels.dtype))
    last = _last_approach_index(d_h)
    assert torch.all(_at(xt_p.abs(), _last_approach_index(d_p)) > _at(fag.K_MARGIN * fag.corridor_halfwidth(d_p), _last_approach_index(d_p)))
    approach = d_h > 0.0
    bound = fag.K_MARGIN * fag.corridor_halfwidth(d_h)
    assert torch.all((xt_h.abs() <= bound + 60.0)[approach])
    assert torch.all(_at(xt_h.abs(), last) < _at(xt_p.abs(), last))
    # ...and the floor did its own job on the same rollout: the unhooked one stalls, the
    # combined one holds its floor while the barrier is turning it.
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=2, seed=9)
    series, _ = build_series(flights, config, airport=AIRPORT)
    aircraft = series[0].scenario.aircraft
    plain_summary, plain_slack = _stall_readout(plain, dynamics, durations, aircraft)
    hooked_summary, hooked_slack = _stall_readout(rollout, dynamics, durations, aircraft)
    assert plain_summary["violations"]["stall"] > 0 and plain_slack < -5.0
    assert hooked_summary["violations"].get("stall", 0) == 0 and hooked_slack > -1.0
    diagnostics = hook.diagnostics()
    assert diagnostics["hook_floor_bound_steps"] > 0.0 and diagnostics["hook_clamped_steps"] > 0.0


def test_only_the_registered_hook_combination_can_be_spelled():
    """`+` in a hook value is a LOOKUP in `CONTROL_HOOK_MEMBERS`, never a split. The one
    registered combination is the one whose modules write disjoint channels in a defined
    order; every other spelling — the same two reversed, or a module that is archived — is
    simply not a member and is refused with the vocabulary."""
    for value in ("speed-floor+barrier", "barrier+nominal-residual", "barrier+barrier"):
        with pytest.raises(ValueError, match="unknown control_command_hook"):
            _hook_config(control_command_hook=value)
    assert CONTROL_HOOK_BARRIER_SPEED_FLOOR in CONTROL_HOOKS_AVAILABLE
    assert CONTROL_HOOK_NOMINAL_RESIDUAL not in CONTROL_HOOKS_AVAILABLE
    assert set(CONTROL_HOOKS_AVAILABLE) < set(CONTROL_HOOKS)
    with pytest.raises(ValueError, match="at least two"):
        CompositeHook((BarrierFilter(_hook_config(control_command_hook=CONTROL_HOOK_BARRIER), _context(1), hard=True),))


def test_a_hook_knob_is_refused_where_no_module_reads_it():
    """Until a second module existed, "a hook is on" and "the barrier is on" were the same
    condition, so the barrier's gains always bound. They no longer do, and a value that
    cannot change an answer is refused rather than serialized into a run name. The archived
    `nominal-residual` builds nothing and is deliberately outside the rule — its six stored
    2026-09-06 configs must keep loading exactly as they are."""
    with pytest.raises(ValueError, match="control_barrier_alpha=.* needs a command hook"):
        _hook_config(control_command_hook=CONTROL_HOOK_SPEED_FLOOR, control_barrier_alpha=0.5)
    with pytest.raises(ValueError, match="control_barrier_heading_gain=.* needs a command hook"):
        _hook_config(control_command_hook=CONTROL_HOOK_SPEED_FLOOR, control_barrier_heading_gain=0.5)
    # The barrier's own positivity check still applies wherever a barrier IS built.
    for hook in (CONTROL_HOOK_BARRIER, CONTROL_HOOK_BARRIER_SPEED_FLOOR):
        with pytest.raises(ValueError, match="control_barrier_alpha must be positive"):
            _hook_config(control_command_hook=hook, control_barrier_alpha=0.0)
    # The archived value keeps loading whatever it carries.
    assert _hook_config(control_command_hook=CONTROL_HOOK_NOMINAL_RESIDUAL,
                        control_barrier_alpha=0.5).control_barrier_alpha == 0.5


def test_the_hook_vocabulary_and_its_module_tables_agree():
    """Two import-time assertions, asserted here so the contract is readable: every hook a
    NEW run may select names its modules, and every module named has a class to build."""
    from control.constraints import _HOOKS

    assert set(CONTROL_HOOK_MEMBERS) == set(CONTROL_HOOKS_AVAILABLE) - {"off"}
    assert set().union(*CONTROL_HOOK_MEMBERS.values()) <= set(_HOOKS)
    assert CONTROL_HOOK_MEMBERS[CONTROL_HOOK_BARRIER_SPEED_FLOOR] == (
        CONTROL_HOOK_BARRIER, CONTROL_HOOK_SPEED_FLOOR
    )   # the order IS the value


def test_the_speed_floor_margin_is_guarded_and_names_its_run():
    config = _hook_config(control_command_hook=CONTROL_HOOK_SPEED_FLOOR)
    assert config.control_speed_floor_margin == CONTROL_SPEED_FLOOR_MARGIN_DEFAULT
    for bad in (0.9, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="not a floor"):
            _hook_config(control_command_hook=CONTROL_HOOK_SPEED_FLOOR,
                         control_speed_floor_margin=bad)
    # A margin without a hook that reads it cannot change an answer, so it is refused.
    with pytest.raises(ValueError, match="needs a command hook"):
        _hook_config(control_command_hook=CONTROL_HOOK_BARRIER, control_speed_floor_margin=1.2)
    with pytest.raises(ValueError, match="needs a command hook"):
        TSConfig(control_speed_floor_margin=1.2)
    named = TSConfig(**{
        **recipe_settings("simple-v3", keep_name=True),
        "control_dynamics_model": CONTROL_DYNAMICS_FIRST_ORDER_LAG,
        "control_dynamics_backend": "scaled-transport-chart-velocity",
        "control_command_hook": CONTROL_HOOK_BARRIER_SPEED_FLOOR,
        "control_speed_floor_margin": 1.25,
    })
    name = run_display_name(named.to_dict())
    assert "hook=barrier+speed-floor" in name and "floor-margin=1.25" in name
    # The default margin is silent: adding the field renames no stored run.
    default = TSConfig(**{**named.to_dict(), "control_speed_floor_margin": CONTROL_SPEED_FLOOR_MARGIN_DEFAULT})
    assert "floor-margin" not in run_display_name(default.to_dict())


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
    with pytest.raises(ValueError, match="unknown control_hook_saturation"):
        _hook_config(control_command_hook=CONTROL_HOOK_BARRIER, control_hook_saturation="never")
    # The archived nominal law's value still LOADS and still NAMES its six stored
    # 2026-09-06 runs (their checkpoints go through TSConfig.from_dict), and no longer
    # builds a hook — the whole of what T2 left of it.
    named = TSConfig(**recipe_settings("simple-v3", keep_name=True), control_command_hook=CONTROL_HOOK_NOMINAL_RESIDUAL)
    assert "hook=nominal-residual" in run_display_name(named.to_dict())
    assert "hook=" not in run_display_name(TSConfig(**recipe_settings("simple-v3", keep_name=True)).to_dict())
    with pytest.raises(ValueError, match="archived"):
        build_command_hook(_hook_config(control_command_hook=CONTROL_HOOK_NOMINAL_RESIDUAL), _context(1))


def test_the_predict_side_gains_are_a_guarded_table_and_need_the_hook():
    """The barrier's gains reach the ADOPTED delivery form, and only with the hook on.

    `predict --command-hook barrier --hook-saturation soft` is what `CLAUDE.md` names as
    the adopted use, so gains that existed only on `train` (where the hook cannot be
    enabled without also training through it) were unreachable where they matter. They are
    overrides of a checkpoint value rather than settings of a new run, so they live in
    `cli.predict.PREDICT_CONFIG_FLAGS`, asserted against `TSConfig` at import the way
    `cli.common.CLI_CONFIG_FIELDS` is.
    """
    import argparse

    import cli.predict as cli_predict

    live = {field.name for field in dataclass_fields(TSConfig)}
    assert set(cli_predict.PREDICT_CONFIG_FLAGS) <= live
    parser = argparse.ArgumentParser(allow_abbrev=False)
    cli_predict.add_cli_arguments(parser)
    flags = {option for action in parser._actions for option in action.option_strings}
    assert set(cli_predict.PREDICT_CONFIG_FLAGS.values()) <= flags

    # Two of the five keep a short name on purpose; the rest are named after the field.
    assert cli_predict.PREDICT_CONFIG_FLAGS["control_barrier_alpha"] == "--control-barrier-alpha"
    assert cli_predict.PREDICT_CONFIG_FLAGS["control_command_hook"] == "--command-hook"
    assert cli_predict.PREDICT_CONFIG_FLAGS["control_speed_floor_margin"] == "--control-speed-floor-margin"
    # The tuning fields are the overridable ones; the hook and its saturation ARE the choice.
    assert cli_predict.HOOK_TUNING_FIELDS == {
        "control_barrier_alpha", "control_barrier_heading_gain", "control_speed_floor_margin"
    }
    # The combined value is selectable at predict time — the delivery form L3.d measures.
    hook_choices = {
        option for action in parser._actions if "--command-hook" in action.option_strings
        for option in action.choices
    }
    assert hook_choices == {CONTROL_HOOK_BARRIER, CONTROL_HOOK_SPEED_FLOOR, CONTROL_HOOK_BARRIER_SPEED_FLOOR}

    # A gain without --command-hook changes nothing, so it is refused rather than ignored.
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--data", "x.json", "--output-dir", "out", "--checkpoint", "c.pt",
            "--control-barrier-alpha",
        ])


def test_training_refuses_hard_saturation_and_logs_the_hook(tmp_path):
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    hard = _hook_config(control_command_hook=CONTROL_HOOK_BARRIER, control_hook_saturation=HOOK_SATURATION_HARD,
                        epochs=1, patience=1, batch_size=32, d_model=16, n_heads=4, d_ff=32, e_layers=1, device="cpu")
    series, _ = build_series(flights, hard, airport=AIRPORT)
    with pytest.raises(ValueError, match="hard hook saturation"):
        fit_model(series[:8], series[8:], hard, verbose=False)
    soft = _hook_config(control_command_hook=CONTROL_HOOK_BARRIER, n_segments=4,
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
            return {"hook_steps": torch.zeros((), dtype=torch.float64)}

        def per_flight_diagnostics(self):
            return {"hook_steps": torch.ones(1, dtype=torch.float64)}

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


def test_a_prediction_record_carries_the_hooks_own_per_flight_counts():
    """The record surface of the hook: WHAT it did to THIS flight, not to the batch.

    `commandHook` already said which hook flew the schedule; on its own that cannot
    separate a flight the floor never touched from one it rewrote at every step. The
    counts are the flight's own — steps, then every other count as a share of it, the same
    normalisation an epoch record carries one level up.
    """
    config = _hook_config(control_command_hook=CONTROL_HOOK_SPEED_FLOOR, n_segments=2)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=1, seed=9)
    series, _ = build_series(flights, config, airport=AIRPORT)
    normalizer = Normalizer.fit(series)

    class IdleControlModel(torch.nn.Module):
        """Commands the envelope's negative thrust floor: the floor has to bind."""

        def forward(self, history, dynamics):
            controls = torch.tensor([[[MIN_THRUST_FRACTION, 0.0, 1.0]] * 2], dtype=history.dtype).expand(len(history), -1, -1)
            durations = torch.tensor([[30.0, 30.0]], dtype=history.dtype).expand(len(history), -1)
            return ControlPrediction(controls=controls, segment_durations=durations, final_time_s=durations.sum(dim=-1))

    forecast = forecast_approach(IdleControlModel(), series[0], config, normalizer, device=torch.device("cpu"))
    assert forecast.command_hook == "speed-floor/soft"
    record = build_prediction_record(series[0], forecast, index=0, model_name=config.model,
                                     horizon_mode=config.horizon_mode)
    counts = record.source["commandHookDiagnostics"]
    assert counts["steps"] == 2.0
    assert set(counts) == {"steps", "floorBoundSteps", "floorSaturatedSteps", "thrustChange"}
    assert 0.0 <= counts["floorBoundSteps"] <= 1.0 and counts["thrustChange"] > 0.0
    # A run without a hook makes no such claim: the key is absent, not zero.
    plain_config = _hook_config(n_segments=2)
    plain = forecast_approach(IdleControlModel(), series[0], plain_config, normalizer, device=torch.device("cpu"))
    plain_record = build_prediction_record(series[0], plain, index=0, model_name=plain_config.model,
                                           horizon_mode=plain_config.horizon_mode)
    assert plain_record.source["commandHook"] is None
    assert "commandHookDiagnostics" not in plain_record.source


def test_predicting_a_stored_nominal_law_checkpoint_refuses_instead_of_substituting():
    """The archived hook's checkpoints load; flying them does NOT quietly become a barrier.

    The value survives so `load_checkpoint` can rebuild the six 2026-09-06 configs, which
    means the whole predict path can reach a config asking for a hook that no longer exists.
    It must say so, not fall back to the hook that does.
    """
    config = _hook_config(control_command_hook=CONTROL_HOOK_NOMINAL_RESIDUAL, n_segments=2)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=1, seed=9)
    series, _ = build_series(flights, config, airport=AIRPORT)
    normalizer = Normalizer.fit(series)

    class AnyControlModel(torch.nn.Module):
        def forward(self, history, dynamics):
            controls = torch.tensor([[[0.20, 0.04, 1.01], [0.16, -0.02, 0.99]]], dtype=history.dtype).expand(len(history), -1, -1)
            durations = torch.tensor([[6.0, 6.0]], dtype=history.dtype).expand(len(history), -1)
            return ControlPrediction(controls=controls, segment_durations=durations, final_time_s=durations.sum(dim=-1))

    with pytest.raises(ValueError, match="archived"):
        forecast_approach(AnyControlModel(), series[0], config, normalizer, device=torch.device("cpu"))
