"""Wanted rates → the three commands, exactly, through the actuators' lag (executor design §4, §6.4, §7.5, §8.3).

The executor cannot read its actuators. It keeps an OBSERVER of them (§4.2): the lag law ``ȧ = (u − a) / τ`` solved in
closed form over each cycle on the commands it sent, ``â ← u + (â − u) e^(−Δt/τ)`` (the plant integrates the same law by
RK4 at 0.5 s: the two agree to about 1e-4 of the change, T1). With the observer an actuator is steered to a wanted value
``a*`` at the cycle's end by the dead-beat command (§4.3)

    u = (a* − â e^(−Δt/τ)) / (1 − e^(−Δt/τ))

inside the actuator's box (a command the box cuts is recorded as its limit binding: the actuator then falls short of
``a*`` at the cycle's end). The point-mass equations (`aerodynamic_model.torch_dynamics.enu_rhs`) give the wanted values:

    bank       the lateral law's turn rate r:      φ = atan(V r / g)   (γ̇* = 0; the dynamics' sign: positive turns left)
    load       the vertical law's γ̇*:              n* = (γ̇* V / g + cos γ) / cos φ̄   (φ̄ the bank's mean over the cycle)
    thrust     the speed law's V̇*:                 x* = (m (V̇* + g sin γ) + D(V, n*)) / T_max

The drag is the dynamics' own (`flight_aerodynamics` at n*, ISA density at the geometric height), so the inverse and the
integration share one polar and nothing here is tuned.

THE LIMITS, physical or an existing model value only (§0.3 L1), each on the quantity it bounds and recorded when it binds:

1. bank — the BANK the aircraft can turn at and still hold its path: ``φ_phys = min(arccos(cos γ / n_up), φ_env)`` with
   ``n_up = min(n_max, n_stall(V_end))`` (§6.4; ``φ_env`` the flyability envelope's 45°, the grader's; ``V_end`` the
   airspeed the cycle ends at, predicted from the observer, `load_ceiling`: the load factor is steered to its value at
   the cycle's END, so the stall it must stay under is the end's); the actual bank rolls
   at most ``p`` (§4.4: the command is ``φ̂ + τ_φ ρ``, ρ the wanted roll rate within ±p), and the bank is no more than
   the bank that can still be rolled out before the heading word's error is used up (`stop_bank_rad`, §6.3);
2. load factor — ``n* ∈ [n_min, n_max]`` (the grader's band) and not above ``n_stall(V_end)`` (§7.5);
3. thrust — the thrust-fraction contract's box (`outputs/envelope.py`); what a limited thrust costs is speed rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2, FlightCondition, flight_aerodynamics, isa_density
from ts_transformer.autopilot.frame import Kinematics
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.geometry.flyability import Envelope
from ts_transformer.outputs.envelope import MAX_THRUST_FRACTION, MIN_THRUST_FRACTION

_GRADER = {field.name: field.default for field in fields(Envelope)}
LOAD_FACTOR_MIN = float(_GRADER["min_load_factor"])
LOAD_FACTOR_MAX = float(_GRADER["max_load_factor"])
BANK_MAX_RAD = float(_GRADER["max_bank_rad"])
#: The actuators, in the order of a command row.
THRUST, BANK, LOAD = range(3)


def decays(params: ExecutorParams, like: torch.Tensor) -> torch.Tensor:
    """``[3]``: ``e^(−Δt/τ)`` of each actuator over one cycle."""
    return torch.exp(-params.cycle_s / like.new_tensor(params.time_constants_s))


def observe(observed: torch.Tensor, command: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """The observer over one cycle (§4.2): each actuator's value at the cycle's end under ``command``."""
    return command + (observed - command) * decay


def steer(wanted: torch.Tensor, observed: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """The dead-beat command that brings an actuator from ``observed`` to ``wanted`` at the cycle's end (§4.3)."""
    return (wanted - observed * decay) / (1.0 - decay)


def stall_load(state: Kinematics, speed_mps: torch.Tensor, aero_params: torch.Tensor) -> torch.Tensor:
    """``n_stall = ½ ρ V² S C_L,max / (m g)``: the largest load factor the wing gives at ``speed_mps`` (§7.5)."""
    return (0.5 * isa_density(state.height_m) * speed_mps.square() * aero_params[:, 0] * aero_params[:, 1]
            / (state.mass_kg * GRAVITY_MPS2))


def load_ceiling(state: Kinematics, observed: torch.Tensor, aero_params: torch.Tensor, max_thrust_n: torch.Tensor,
                 cycle_s: float) -> torch.Tensor:
    """``n_up = min(n_max, n_stall(V_end))``: ``V_end = V + V̇ Δt``, ``V̇ = (x̂ T_max − D(V, n_most)) / m − g sin γ`` —
    the observer's thrust against the drag of the largest load factor the cycle can ask for, ``n_most = min(n_max,
    n_stall(V))`` (the executor's own state and airframe, I2, I3). A PREDICTION, not a bound: the pull itself costs speed
    through its induced drag, which it holds; a thrust falling within the cycle (a deceleration) costs more, and there the
    speed law asks no speed below the stall at n* (§8.1). On a flight's first cycle, before any thrust is in effect (NaN),
    the speed is taken as held (V̇ = 0)."""
    condition = FlightCondition(speed_mps=state.speed_mps, sin_gamma=torch.sin(state.gamma_rad),
                                mass_kg=state.mass_kg, density=isa_density(state.height_m))
    most = torch.minimum(stall_load(state, state.speed_mps, aero_params), torch.full_like(state.speed_mps, LOAD_FACTOR_MAX))
    drag = flight_aerodynamics(condition, most, aero_params).drag_n
    rate = (observed[:, THRUST] * max_thrust_n - drag) / state.mass_kg - GRAVITY_MPS2 * condition.sin_gamma
    end = state.speed_mps + torch.where(observed[:, THRUST].isnan(), torch.zeros_like(rate), rate) * cycle_s
    return torch.minimum(stall_load(state, end, aero_params), torch.full_like(state.speed_mps, LOAD_FACTOR_MAX))


def bank_limit_rad(state: Kinematics, ceiling: torch.Tensor) -> torch.Tensor:
    """``φ_phys = min(arccos(cos γ / n_up), φ_env)`` (limit 1): 0 where the wing cannot even hold the path level."""
    return torch.acos((torch.cos(state.gamma_rad) / ceiling).clamp(max=1.0)).clamp(max=BANK_MAX_RAD)


def stop_bank_rad(error_deg: torch.Tensor, bank_rad: torch.Tensor, speed_mps: torch.Tensor,
                  params: ExecutorParams, decay_bank: torch.Tensor) -> torch.Tensor:
    """The largest bank from which the bank, rolled out at the mean roll rate ``p_mean`` the lag gives a cycle of
    peak-rate-``p`` commands, ``p τ_φ (1 − e^(−Δt/τ_φ)) / Δt``, turns the track no further than the heading error left at
    the cycle's end (§6.3): rolled out from φ the track turns ``−g ln(cos φ) / (V p_mean)`` (exact at constant V), so
    ``φ_stop = arccos(exp(−e' V p_mean / g))``. ``e'`` is the error at the cycle's end — less the turn of the cycle when
    the aircraft turns toward the word, more when it turns away; positive banks turn left, a positive error right."""
    cycle = params.cycle_s
    p_mean = math.radians(params.bank_rate_deg_s) * params.bank_time_constant_s * (1.0 - decay_bank) / cycle
    error = torch.deg2rad(error_deg.abs())
    turning = GRAVITY_MPS2 * torch.tan(bank_rad.abs()) / speed_mps * cycle
    toward = torch.sign(-error_deg) == torch.sign(bank_rad)
    left = torch.where(toward, (error - turning).clamp(min=0.0), error + turning)
    return torch.acos(torch.exp(-left * speed_mps * p_mean / GRAVITY_MPS2))


@dataclass(frozen=True)
class Bank:
    command: torch.Tensor           # [B] the bank sent (the dynamics' sign)
    mean: torch.Tensor              # [B] the observer's bank, the mean over the cycle (φ̄, what n* is solved at)
    binds: dict[str, torch.Tensor]  # "bank_cap", "bank_rate": [B] bool


def bank(state: Kinematics, track_rate_deg_s: torch.Tensor, error_deg: torch.Tensor, observed: torch.Tensor,
         limit_rad: torch.Tensor, params: ExecutorParams, decay_bank: torch.Tensor) -> Bank:
    """The bank command for the lateral law's track rate (§6.2 steps 2–5, §4.4): the bank of that turn rate within
    ``±φ_phys``, cut to the stopping bank, rolled toward from the observer's bank at no more than ``p``."""
    speed = state.speed_mps
    wanted = -torch.atan(speed * torch.deg2rad(track_rate_deg_s) / GRAVITY_MPS2)
    capped = torch.minimum(torch.maximum(wanted, -limit_rad), limit_rad)
    stop = stop_bank_rad(error_deg, observed, speed, params, decay_bank)
    target = torch.sign(capped) * torch.minimum(capped.abs(), stop)
    roll = math.radians(params.bank_rate_deg_s)
    rate = ((target - observed) / params.cycle_s).clamp(-roll, roll)
    rolled = observed + params.bank_time_constant_s * rate
    command = torch.minimum(torch.maximum(rolled, -limit_rad), limit_rad)
    end = observe(observed, command, decay_bank)
    return Bank(command=command, mean=0.5 * (observed + end),
                binds={"bank_cap": (capped != wanted) | (command != rolled),
                       "bank_rate": (target - observed).abs() > roll * params.cycle_s})


@dataclass(frozen=True)
class Load:
    wanted: torch.Tensor            # [B] n*, the load factor the cycle ends at (what drag and the speed law read)
    command: torch.Tensor           # [B]
    binds: dict[str, torch.Tensor]  # "load_factor": [B] bool


def load(state: Kinematics, gamma_rate_rad_s: torch.Tensor, bank_mean_rad: torch.Tensor, observed: torch.Tensor,
         ceiling: torch.Tensor, decay_load: torch.Tensor) -> Load:
    """The load factor for the vertical law's path-angle rate at the cycle's mean bank (§7.5 steps 3–5), in
    ``[n_min, n_up]``, steered to through its lag."""
    solved = (gamma_rate_rad_s * state.speed_mps / GRAVITY_MPS2 + torch.cos(state.gamma_rad)) / torch.cos(bank_mean_rad)
    wanted = torch.minimum(solved.clamp(min=LOAD_FACTOR_MIN), ceiling)           # the ceiling is at most n_max
    steered = steer(wanted, observed, decay_load)
    command = steered.clamp(LOAD_FACTOR_MIN, LOAD_FACTOR_MAX)
    return Load(wanted=wanted, command=command, binds={"load_factor": (wanted != solved) | (command != steered)})


@dataclass(frozen=True)
class Thrust:
    wanted: torch.Tensor            # [B] x*, the fraction of the installed thrust the cycle ends at
    command: torch.Tensor           # [B]
    binds: dict[str, torch.Tensor]  # "thrust_max", "thrust_min": [B] bool


def thrust_fraction(state: Kinematics, accel_mps2: torch.Tensor, load_factor: torch.Tensor, aero_params: torch.Tensor,
                    max_thrust_n: torch.Tensor) -> torch.Tensor:
    """``x* = (m (V̇* + g sin γ) + D(V, n)) / T_max`` (§8.3), unbounded."""
    condition = FlightCondition(speed_mps=state.speed_mps, sin_gamma=torch.sin(state.gamma_rad),
                                mass_kg=state.mass_kg, density=isa_density(state.height_m))
    drag = flight_aerodynamics(condition, load_factor, aero_params).drag_n
    return (state.mass_kg * (accel_mps2 + GRAVITY_MPS2 * condition.sin_gamma) + drag) / max_thrust_n


def thrust(solved: torch.Tensor, observed: torch.Tensor, decay_thrust: torch.Tensor) -> Thrust:
    """The thrust ``solved`` (`thrust_fraction`) within the contract's box (limit 3), steered to through its lag."""
    wanted = solved.clamp(MIN_THRUST_FRACTION, MAX_THRUST_FRACTION)
    steered = steer(wanted, observed, decay_thrust)
    command = steered.clamp(MIN_THRUST_FRACTION, MAX_THRUST_FRACTION)
    return Thrust(wanted=wanted, command=command,
                  binds={"thrust_max": (solved > MAX_THRUST_FRACTION) | (steered > MAX_THRUST_FRACTION),
                         "thrust_min": (solved < MIN_THRUST_FRACTION) | (steered < MIN_THRUST_FRACTION)})


def stalled(state: Kinematics, load_factor: torch.Tensor, aero_params: torch.Tensor) -> torch.Tensor:
    """``[B]`` bool: the wing cannot give ``load_factor`` at this state (the dynamics' own polar)."""
    condition = FlightCondition(speed_mps=state.speed_mps, sin_gamma=torch.sin(state.gamma_rad),
                                mass_kg=state.mass_kg, density=isa_density(state.height_m))
    return flight_aerodynamics(condition, load_factor, aero_params).stalled
