"""Transport-chart dynamics with first-order actuator lag on the three controls.

The point-mass model applies a piecewise-constant control instantly, so a learned schedule
produces a bank angle that steps at every segment boundary — a trajectory whose curvature
is discontinuous N times and whose roll rate is unbounded. Real aircraft roll into a turn,
spool an engine and load a wing over a finite time.

This model makes the three controls **states** driven towards the commanded value::

    d(a_x)/dt  = (a_x_cmd - a_x) /tau_thrust
    d(mu)/dt   = (mu_cmd - mu)   /tau_bank
    d(a_z)/dt  = (a_z_cmd - a_z) /tau_load

and flies the chart RHS (:func:`transport_chart_rate`) at the newton controls the ACTUAL values
resolve to. The force equations, the stall handling, the WGS84 transport term and the chart
projection are literally the same code as the instantaneous model — this module adds three
scalar ODEs and the control law, and reduces to the point-mass model as every tau goes to zero
under the thrust-fraction law. That equivalence is what makes the two models comparable rather
than two flight models with two sets of results.

State is the seven transport-chart values followed by the three actuator values::

    (e, n, u, ve, vn, vu, mass, a_x, bank_rad, a_z)

**The control law** (``control_law=`` on every rollout below) says what the first and third
actuators ARE, and it is the only thing that differs between contracts. A law resolves the
actuators to the newton thrust and the load factor the RHS flies, at every RK4 stage, through
two static methods — :meth:`resolve_load` then :meth:`resolve_thrust` (the latter sees the
drag at the resolved load, computed once per stage and subtracted again by the RHS):

* :class:`ThrustFractionLaw` — thrust as a fraction of the flight's installed thrust,
  ``T = a_x · T_max`` (``ts_transformer/outputs/envelope.py``); ``a_z`` is the load factor;
* :class:`SpecificForceLaw` — the specific force along the path, ``a_x = (T - D)/W``, flown by
  ``clamp(W·a_x + D, T_lo, T_max)``. Wherever the clamp does not bind ``V' = g·(a_x - sin γ)``
  and the airframe leaves the speed equation (``ts_transformer/docs/
  2026-09-14_specific_force_control_design.md``);
* :class:`SpeedCommandLaw` — a speed command relative to the anchor airspeed, ``a_Δ`` m/s,
  flown by a first-order speed loop through the specific-force thrust: wherever the clamp does
  not bind ``V' = (V₀ + a_Δ − V)/τ_V`` (the same design, §12);
* :class:`PathAngleLaw` — longitudinally the specific force; ``a_z`` is a path-angle target γ*
  and the load factor is re-solved as ``[cos γ + V(γ* − γ)/(g τ_γ)]/cos φ``, clipped to the load
  box, so wherever neither that box nor the stall clamp binds ``γ' = (γ* − γ)/τ_γ`` (§14).

A law's per-flight constants (the engine floor in newtons, the anchor airspeed, the loop's time
constant and box) travel in the rollout's ``step_context`` as :attr:`PARAMETERS` columns, so
ONE RHS, ONE RK4 step and ONE unpacking serve every law; the geodetic readers
(:meth:`geodetic_load`, :meth:`geodetic_controls`) call the same two static methods on a
geodetic state, which is how the exported record and the heading-rate loss resolve what the
rollout flew without restating a law.

The actuator states are order one under the first two laws, metres per second under the speed
command and radians under the path-angle law; a fixed-step RK4 is invariant to a linear
rescaling of a state, so none of them needs a scale of its own. ``state_scale``
nondimensionalises the seven point-mass coordinates and is all-ones for the physical variant.

Controls are the COMMANDS, in the same units as the actuator states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import ClassVar

import torch

from aerodynamic_model.torch_dense_rollout import (
    DenseControlRollout,
    rollout_piecewise_constant_at_times_with_step,
)
from aerodynamic_model.torch_dynamics import (
    CONTROL_NAMES,
    GRAVITY_MPS2,
    STATE_NAMES,
    FlightCondition,
    flight_aerodynamics,
    geodetic_flight_condition,
)
from aerodynamic_model.torch_piecewise_rollout import (
    RawCommandHook,
    rollout_piecewise_constant_hooked_with_step,
    rollout_piecewise_constant_with_step,
)
from aerodynamic_model.torch_transport_chart_dynamics import (
    TRANSPORT_CHART_STATE_NAMES,
    geodetic_to_transport_chart_state,
    transport_chart_kinematics,
    transport_chart_rate,
)


# The first actuator is the law's longitudinal command and the third its vertical one; the
# names are the default law's.
LAG_ACTUATOR_NAMES = ("thrust_fraction", "bank_rad", "load_factor")
LAG_STATE_NAMES = (*TRANSPORT_CHART_STATE_NAMES, *LAG_ACTUATOR_NAMES)
CHART_WIDTH = len(TRANSPORT_CHART_STATE_NAMES)
# The physical chart: every chart state divided by 1 (what a caller passes to fly unscaled).
UNSCALED_CHART = (1.0,) * CHART_WIDTH
# Actuator states are dimensionless, radians, (speed-command) m/s or (path-angle) radians;
# a fixed-step RK4 is invariant to a linear rescaling of a state, so only the chart half is
# rescaled.
UNIT_ACTUATOR_SCALE = (1.0, 1.0, 1.0)
THRUST_ACTUATOR, BANK_ACTUATOR, VERTICAL_ACTUATOR = range(len(LAG_ACTUATOR_NAMES))


# ── the laws' formulas ────────────────────────────────────────────────────────────────────


def specific_force_thrust_n(
    specific_force: torch.Tensor,
    drag_n: torch.Tensor,
    mass_kg: torch.Tensor,
    min_thrust_n: torch.Tensor,
    max_thrust_n: torch.Tensor,
) -> torch.Tensor:
    """The thrust that flies the specific force ``n_x = (T - D)/W``, clamped to the engine.

    ``T = m·g·n_x + D`` with ``D`` the drag the RHS subtracts at the same state and load
    (:func:`flight_aerodynamics`), so where the clamp does not bind ``V' = g·(n_x - sin γ)``:
    the mass, the installed thrust and the polar leave the speed equation. ``[min_thrust_n,
    max_thrust_n]`` is the engine's range; where it binds, the speed law is the thrust law's at
    that bound.
    """
    thrust = mass_kg * GRAVITY_MPS2 * specific_force + drag_n
    return torch.minimum(torch.maximum(thrust, min_thrust_n), max_thrust_n)


def speed_loop_specific_force(
    speed_command_mps: torch.Tensor,
    speed_mps: torch.Tensor,
    sin_gamma: torch.Tensor,
    speed_time_constant_s: torch.Tensor | float,
) -> torch.Tensor:
    """The specific force a first-order speed loop asks for, ``sin γ + (v_c − V)/(g·τ_V)``:
    flown through :func:`specific_force_thrust_n`, it makes ``V' = (v_c − V)/τ_V`` wherever
    the engine's clamp does not bind."""
    return sin_gamma + (speed_command_mps - speed_mps) / (GRAVITY_MPS2 * speed_time_constant_s)


def path_angle_load_factor(
    path_angle_command_rad: torch.Tensor,
    sin_gamma: torch.Tensor,
    speed_mps: torch.Tensor,
    bank_rad: torch.Tensor,
    path_angle_time_constant_s: torch.Tensor | float,
    min_load_factor: torch.Tensor | float,
    max_load_factor: torch.Tensor | float,
) -> torch.Tensor:
    """The load factor a first-order PATH loop asks for, the vertical analogue of
    :func:`speed_loop_specific_force`.

    ``n = [cos γ + V·(γ* − γ)/(g·τ_γ)] / cos φ``, clipped to the load box. Flown through the
    unchanged RHS it gives ``γ' = (γ* − γ)/τ_γ`` wherever neither the box nor the stall clamp
    binds, on every airframe. ``cos γ`` is ``√(1 − sin²γ)`` of the condition's own sine (the
    path angle is within ±90° by construction), so a chart and a geodetic caller resolve the
    same load from the same state."""
    cos_gamma = torch.sqrt(torch.clamp(1.0 - sin_gamma * sin_gamma, min=0.0))
    gamma = torch.atan2(sin_gamma, cos_gamma)
    vertical = cos_gamma + speed_mps * (path_angle_command_rad - gamma) / (
        GRAVITY_MPS2 * path_angle_time_constant_s
    )
    return torch.clamp(vertical / torch.cos(bank_rad), min_load_factor, max_load_factor)


# ── the laws ──────────────────────────────────────────────────────────────────────────────


class _ControlLaw:
    """What every law provides; the four below override the parts that differ.

    ``PARAMETERS`` names the per-flight columns :meth:`parameters` returns, in order; they
    ride in the step context after the installed thrust, and the static resolvers read them
    as ``parameters[..., i]``. The resolvers are static so the compiled step can call them on
    the CLASS (a law's float fields never become compile-time constants).
    """

    PARAMETERS: ClassVar[tuple[str, ...]] = ()
    #: The law resolves the load factor from the STATE and a vertical TARGET, dividing by the flown
    #: bank's cosine — so reading the load needs the state, its vertical lift ``n cos μ`` is the same
    #: at any bank, and a module that moves the bank must not re-coordinate the third column (it is
    #: not a load factor). False: the load factor IS the third actuator.
    RESOLVES_LOAD_AT_FLOWN_BANK: ClassVar[bool] = False

    def parameters(
        self, max_thrust_n: torch.Tensor, initial_geodetic_states: torch.Tensor
    ) -> tuple[torch.Tensor | float, ...]:
        """The :attr:`PARAMETERS` values, per flight (a scalar broadcasts)."""
        return ()

    @staticmethod
    def resolve_load(
        condition: FlightCondition, actual: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        """The load factor the RHS flies: the third actuator, unless the law resolves it."""
        return actual[..., VERTICAL_ACTUATOR]

    @staticmethod
    def held_load(
        condition: FlightCondition, command: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        """The load factor a command HELD over a segment is priced at by a stall floor, from the
        state where the segment starts: a load command is held as given."""
        return command[..., VERTICAL_ACTUATOR]

    @staticmethod
    def resolve_thrust(
        condition: FlightCondition,
        actual: torch.Tensor,
        drag_n: torch.Tensor,
        max_thrust_n: torch.Tensor,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def parameter_matrix(
        self,
        max_thrust_n: torch.Tensor,
        initial_geodetic_states: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """:meth:`parameters` as ``[B, len(PARAMETERS)]`` in ``dtype`` — the one builder of these
        columns, for the step context and the geodetic readers alike. ``dtype`` is the consumer's
        own (the frame's, the state's): cast FIRST, because ``as_tensor(0.2)`` is float32 and a
        later ``.to(float64)`` keeps its rounding, so a clip floor would differ from the law's
        own constant by 3e-9 (review §14.9, finding 6)."""
        values = self.parameters(max_thrust_n, initial_geodetic_states)
        if len(values) != len(self.PARAMETERS):
            raise ValueError(f"{type(self).__name__} returned {len(values)} of {self.PARAMETERS}")
        rows = max_thrust_n.shape[0]
        return torch.stack(
            [
                torch.as_tensor(value, dtype=dtype).to(device).reshape(-1).expand(rows)
                for value in values
            ],
            dim=-1,
        ) if values else torch.zeros((rows, 0), dtype=dtype, device=device)

    @staticmethod
    def _require_geodetic_batch(states_geo: torch.Tensor, actual: torch.Tensor) -> None:
        if states_geo.ndim != 3 or actual.shape != (*states_geo.shape[:2], len(LAG_ACTUATOR_NAMES)):
            raise ValueError(
                "the geodetic readers take [B, N, 7] states and [B, N, 3] actuators, got "
                f"{tuple(states_geo.shape)} and {tuple(actual.shape)}"
            )

    def geodetic_load(
        self,
        states_geo: torch.Tensor,
        actual: torch.Tensor,
        *,
        max_thrust_n: torch.Tensor,
        initial_geodetic_states: torch.Tensor,
    ) -> torch.Tensor:
        """The load factor the law resolves at geodetic ``[B, N, 7]`` states from ``[B, N, 3]``
        actuators — the heading-rate loss's reading of what the rollout flew."""
        self._require_geodetic_batch(states_geo, actual)
        parameters = self.parameter_matrix(
            max_thrust_n, initial_geodetic_states, dtype=states_geo.dtype, device=states_geo.device
        ).unsqueeze(-2)
        return self.resolve_load(geodetic_flight_condition(states_geo), actual, parameters)

    def geodetic_controls(
        self,
        states_geo: torch.Tensor,
        actual: torch.Tensor,
        *,
        max_thrust_n: torch.Tensor,
        initial_geodetic_states: torch.Tensor,
        aero_params: torch.Tensor,
    ) -> torch.Tensor:
        """Newton ``(thrust_N, bank, load)`` the law resolves at geodetic ``[B, N, 7]`` states
        from ``[B, N, 3]`` actuators (``max_thrust_n`` and ``initial_geodetic_states`` per flight,
        ``aero_params`` ``[B, 6]``)."""
        self._require_geodetic_batch(states_geo, actual)
        condition = geodetic_flight_condition(states_geo)
        parameters = self.parameter_matrix(
            max_thrust_n, initial_geodetic_states, dtype=states_geo.dtype, device=states_geo.device
        ).unsqueeze(-2)
        load = self.resolve_load(condition, actual, parameters)
        aerodynamics = flight_aerodynamics(condition, load, aero_params.unsqueeze(-2))
        thrust = self.resolve_thrust(
            condition, actual, aerodynamics.drag_n, max_thrust_n.unsqueeze(-1), parameters
        )
        return torch.stack((thrust, actual[..., BANK_ACTUATOR], load), dim=-1)


def _require_engine_floor(min_thrust_fraction: float) -> None:
    # 1.0 is the ceiling by definition: the fraction is of the INSTALLED thrust.
    if not math.isfinite(min_thrust_fraction) or min_thrust_fraction >= 1.0:
        raise ValueError(
            "min_thrust_fraction must be finite and below the installed thrust (1.0), "
            f"got {min_thrust_fraction!r}"
        )


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


# Each law carries its OWN two compile entry points (`step_inference`, `step_autograd`):
# torch.compile caches per code object, so a shared entry would put every law's graphs (and
# every static autograd shape) into one cache and exhaust Dynamo's recompile budget. They are
# two lines each and call the one generic `rk4_lag_step`.


@dataclass(frozen=True)
class ThrustFractionLaw(_ControlLaw):
    """The first actuator is thrust as a fraction of installed thrust, ``T = a_x · T_max``."""

    @staticmethod
    def resolve_thrust(condition, actual, drag_n, max_thrust_n, parameters):
        return actual[..., THRUST_ACTUATOR] * max_thrust_n

    @staticmethod
    def step_inference(state, commands, aero_params, dt_s, step_context):
        return rk4_lag_step(state, commands, aero_params, dt_s, step_context, ThrustFractionLaw)

    @staticmethod
    def step_autograd(state, commands, aero_params, dt_s, step_context):
        return rk4_lag_step(state, commands, aero_params, dt_s, step_context, ThrustFractionLaw)


@dataclass(frozen=True)
class SpecificForceLaw(_ControlLaw):
    """The first actuator is the specific force along the path, ``a_x = (T - D)/W``.

    The thrust that flies it is recomputed at every RHS evaluation — every RK4 stage — as
    ``clamp(W·a_x + D, min_thrust_fraction·T_max, T_max)``: a thrust held across a stage
    would not hold the specific force, because the drag moves with the state.
    ``min_thrust_fraction`` is the engine floor in the thrust-fraction law's own units, so a
    caller that passes that law's box floor makes the two laws admit the same thrusts.
    """

    min_thrust_fraction: float
    PARAMETERS: ClassVar[tuple[str, ...]] = ("min_thrust_n",)

    def __post_init__(self) -> None:
        _require_engine_floor(self.min_thrust_fraction)

    def parameters(self, max_thrust_n, initial_geodetic_states):
        return (self.min_thrust_fraction * max_thrust_n,)

    @staticmethod
    def resolve_thrust(condition, actual, drag_n, max_thrust_n, parameters):
        return specific_force_thrust_n(
            actual[..., THRUST_ACTUATOR], drag_n, condition.mass_kg, parameters[..., 0],
            max_thrust_n,
        )

    @staticmethod
    def step_inference(state, commands, aero_params, dt_s, step_context):
        return rk4_lag_step(state, commands, aero_params, dt_s, step_context, SpecificForceLaw)

    @staticmethod
    def step_autograd(state, commands, aero_params, dt_s, step_context):
        return rk4_lag_step(state, commands, aero_params, dt_s, step_context, SpecificForceLaw)


@dataclass(frozen=True)
class SpeedCommandLaw(_ControlLaw):
    """The first actuator is a speed command RELATIVE to the anchor's airspeed, ``a_Δ`` m/s,
    flown by a first-order speed loop (``ts_transformer/docs/
    2026-09-14_specific_force_control_design.md`` §12).

    At every RHS evaluation the loop asks for the specific force
    ``sin γ + (V₀ + a_Δ − V)/(g·τ_V)`` and the thrust that flies it is re-solved exactly as
    under :class:`SpecificForceLaw` — the same clamp, the same drag — so wherever the clamp
    does not bind ``V' = (V₀ + a_Δ − V)/τ_V`` on every airframe: the specific force's
    invariance, plus the restoring force ``∂V'/∂V = −1/τ_V`` that law lacks. ``V₀`` is each
    flight's anchor airspeed, read off the rollout's initial state.
    """

    min_thrust_fraction: float
    speed_time_constant_s: float
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "min_thrust_n", "reference_speed_mps", "speed_time_constant_s",
    )

    def __post_init__(self) -> None:
        _require_engine_floor(self.min_thrust_fraction)
        _require_positive("speed_time_constant_s", self.speed_time_constant_s)

    def parameters(self, max_thrust_n, initial_geodetic_states):
        return (
            self.min_thrust_fraction * max_thrust_n,
            initial_geodetic_states[..., STATE_NAMES.index("V")],
            self.speed_time_constant_s,
        )

    @staticmethod
    def resolve_thrust(condition, actual, drag_n, max_thrust_n, parameters):
        specific_force = speed_loop_specific_force(
            parameters[..., 1] + actual[..., THRUST_ACTUATOR], condition.speed_mps,
            condition.sin_gamma, parameters[..., 2],
        )
        return specific_force_thrust_n(
            specific_force, drag_n, condition.mass_kg, parameters[..., 0], max_thrust_n
        )

    @staticmethod
    def step_inference(state, commands, aero_params, dt_s, step_context):
        return rk4_lag_step(state, commands, aero_params, dt_s, step_context, SpeedCommandLaw)

    @staticmethod
    def step_autograd(state, commands, aero_params, dt_s, step_context):
        return rk4_lag_step(state, commands, aero_params, dt_s, step_context, SpeedCommandLaw)


@dataclass(frozen=True)
class PathAngleLaw(_ControlLaw):
    """The THIRD actuator is a path-angle target ``γ*`` in radians, flown by a first-order path
    loop through the load factor (``ts_transformer/docs/
    2026-09-14_specific_force_control_design.md`` §14); longitudinally it is
    :class:`SpecificForceLaw`.

    At every RHS evaluation the loop asks for :func:`path_angle_load_factor`, and the thrust
    is re-solved exactly as under :class:`SpecificForceLaw` at that load. So wherever neither
    the load box nor the stall clamp binds ``γ' = (γ* − γ)/τ_γ`` on every airframe: the
    vertical analogue of the speed command, and the one channel the specific force leaves as
    an open-loop double integrator (design §7.5.3).

    The actuator state is γ* itself, lagging toward the command with the LOAD actuator's own
    time constant, so the realised load never steps at a segment boundary. The stall clamp in
    :func:`aerodynamic_coefficients` is untouched: the loop may ask for lift the wing cannot
    make, and the RHS still refuses it.

    The loop is the flat-earth relation and the chart RHS adds the WGS84 transport term on top,
    which the loop does not subtract — so its fixed point sits about 0.002° off the target at
    85 m/s and 0.004° at 142 m/s (0.3 % of the rate), 30-50× below the contract's 0.11°
    break-even accuracy (design §13.4), and it cancels against the teacher, which carries the
    same term. Compensating it would put a chart term inside a law whose whole point is
    that it is the same on every airframe and every frame.
    """

    min_thrust_fraction: float
    path_angle_time_constant_s: float
    min_load_factor: float
    max_load_factor: float
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "min_thrust_n", "path_angle_time_constant_s", "min_load_factor", "max_load_factor",
    )
    RESOLVES_LOAD_AT_FLOWN_BANK: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _require_engine_floor(self.min_thrust_fraction)
        _require_positive("path_angle_time_constant_s", self.path_angle_time_constant_s)
        if not (math.isfinite(self.min_load_factor) and math.isfinite(self.max_load_factor)
                and self.min_load_factor < self.max_load_factor):
            raise ValueError(
                "the load box must be finite and ordered, got "
                f"({self.min_load_factor!r}, {self.max_load_factor!r})"
            )

    def parameters(self, max_thrust_n, initial_geodetic_states):
        return (
            self.min_thrust_fraction * max_thrust_n,
            self.path_angle_time_constant_s,
            self.min_load_factor,
            self.max_load_factor,
        )

    @staticmethod
    def resolve_load(condition, actual, parameters):
        return path_angle_load_factor(
            actual[..., VERTICAL_ACTUATOR], condition.sin_gamma, condition.speed_mps,
            actual[..., BANK_ACTUATOR], parameters[..., 1], parameters[..., 2],
            parameters[..., 3],
        )

    @staticmethod
    def held_load(condition, command, parameters):
        """Over a hold the loop flies loads between the one it resolves where the segment starts (the
        whole step ``γ* − γ`` in the pull term) and its fixed point once the path has reached the
        target, ``cos γ*/cos φ``; a stall floor prices the larger. The start value alone under-prices a
        DESCENDING target — the loop never flies that low a load within a hold, because the actuator
        and the loop lag (review 2026-09-16: a −6° step at 60 m/s left the speed 3.4 m/s under its
        floor at the hold's end, where the specific-force twin kept +3.1 m/s)."""
        steady = torch.clamp(
            torch.cos(command[..., VERTICAL_ACTUATOR]) / torch.cos(command[..., BANK_ACTUATOR]),
            parameters[..., 2], parameters[..., 3],
        )
        return torch.maximum(PathAngleLaw.resolve_load(condition, command, parameters), steady)

    @staticmethod
    def resolve_thrust(condition, actual, drag_n, max_thrust_n, parameters):
        return SpecificForceLaw.resolve_thrust(
            condition, actual, drag_n, max_thrust_n, parameters
        )

    @staticmethod
    def step_inference(state, commands, aero_params, dt_s, step_context):
        return rk4_lag_step(state, commands, aero_params, dt_s, step_context, PathAngleLaw)

    @staticmethod
    def step_autograd(state, commands, aero_params, dt_s, step_context):
        return rk4_lag_step(state, commands, aero_params, dt_s, step_context, PathAngleLaw)


THRUST_FRACTION_LAW = ThrustFractionLaw()
LagControlLaw = ThrustFractionLaw | SpecificForceLaw | SpeedCommandLaw | PathAngleLaw


# ── the augmented state ───────────────────────────────────────────────────────────────────


def _require_last_dim(tensor: torch.Tensor, expected: int, name: str) -> None:
    if tensor.shape[-1] != expected:
        raise ValueError(
            f"{name} must end in {expected} values, got shape {tuple(tensor.shape)}"
        )


def lag_state_scale(
    chart_scale: tuple[float, ...], reference: torch.Tensor
) -> torch.Tensor:
    """Return the ``[10]`` divisor for the augmented state (``UNSCALED_CHART``: the physical chart)."""
    chart = tuple(chart_scale)
    if len(chart) != CHART_WIDTH:
        raise ValueError(f"chart scale must contain {CHART_WIDTH} values")
    return reference.new_tensor((*chart, *UNIT_ACTUATOR_SCALE))


def lag_state_from_geodetic(
    states_geo: torch.Tensor,
    initial_controls: torch.Tensor,
    frame_params: torch.Tensor,
    state_scale: torch.Tensor,
) -> torch.Tensor:
    """Build the augmented state from a geodetic state and the controls in effect.

    ``initial_controls`` is what the aircraft is ALREADY doing at the anchor, not the
    first command: starting the actuators at the first command would place the aircraft in
    a bank it has not rolled into yet, and the lag would then be paid twice.
    """
    _require_last_dim(states_geo, len(STATE_NAMES), "states_geo")
    _require_last_dim(initial_controls, len(CONTROL_NAMES), "initial_controls")
    chart = geodetic_to_transport_chart_state(states_geo, frame_params)
    state = torch.cat((chart, initial_controls.to(chart.dtype)), dim=-1)
    return state / state_scale


def lag_state_to_transport_chart(
    state_lag: torch.Tensor, state_scale: torch.Tensor
) -> torch.Tensor:
    """Return the physical point-mass part every public conversion already understands."""
    _require_last_dim(state_lag, len(LAG_STATE_NAMES), "state_lag")
    return (state_lag * state_scale)[..., :CHART_WIDTH]


def lag_actuator_states(
    state_lag: torch.Tensor, state_scale: torch.Tensor
) -> torch.Tensor:
    """Return the realised actuators along the rollout, each in ITS OWN law's unit: ``a_x`` is
    the thrust fraction, the specific force or the speed command relative to the anchor's
    airspeed, and the third is the load factor under every law EXCEPT
    :class:`PathAngleLaw`, where it is the path-angle target in radians. A consumer that wants
    the load the rollout flew asks the law (:meth:`_ControlLaw.geodetic_load`), never reads
    column 2 blind."""
    _require_last_dim(state_lag, len(LAG_STATE_NAMES), "state_lag")
    return (state_lag * state_scale)[..., CHART_WIDTH:]


# ── the one RHS, RK4 step and step context ────────────────────────────────────────────────
#
# The rollout engine hands one ``step_context`` tensor to the step function, so every per-run
# constant travels in it, in ONE layout for every law:
#
#     [ frame (4) | time_constants_s (3) | max_thrust_n (1) | law PARAMETERS (k) | state_scale (10) ]
#
# state_scale rides along rather than being captured in a closure so the compiled step stays a
# module-level code object per law (a closure is a new code object per rollout, and
# torch.compile caches per code object).
_FRAME_WIDTH = 4
_TAU_WIDTH = len(CONTROL_NAMES)
_THRUST_AT = _FRAME_WIDTH + _TAU_WIDTH
_PARAMETERS_AT = _THRUST_AT + 1


def lag_step_context(
    law: LagControlLaw,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
    state_scale: torch.Tensor,
    initial_geodetic_states: torch.Tensor,
) -> torch.Tensor:
    """The layout above for ``law``, in the frame's dtype; every value is one column per flight
    (a scalar broadcasts)."""
    rows = len(frame_params)
    dtype, device = frame_params.dtype, frame_params.device
    return torch.cat(
        (
            frame_params,
            time_constants_s.reshape(1, -1).expand(rows, -1).to(dtype),
            torch.as_tensor(max_thrust_n, dtype=dtype).to(device).reshape(-1, 1).expand(rows, 1),
            law.parameter_matrix(max_thrust_n, initial_geodetic_states, dtype=dtype, device=device),
            state_scale.reshape(1, -1).expand(rows, -1).to(dtype),
        ),
        dim=-1,
    )


def lag_rhs(
    state_lag: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    step_context: torch.Tensor,
    law: LagControlLaw | type,
) -> torch.Tensor:
    """Continuous RHS of the chart state plus three first-order actuators under ``law``
    (an instance or its class), with the per-run constants in ``step_context``
    (:func:`lag_step_context`)."""
    _require_last_dim(state_lag, len(LAG_STATE_NAMES), "state_lag")
    _require_last_dim(commands, len(CONTROL_NAMES), "commands")
    scale_at = _PARAMETERS_AT + len(law.PARAMETERS)
    frame_params = step_context[..., :_FRAME_WIDTH]
    time_constants_s = step_context[..., _FRAME_WIDTH:_THRUST_AT]
    max_thrust_n = step_context[..., _THRUST_AT]
    parameters = step_context[..., _PARAMETERS_AT:scale_at]
    state_scale = step_context[0, scale_at:]
    physical = state_lag * state_scale
    chart, actual = physical[..., :CHART_WIDTH], physical[..., CHART_WIDTH:]
    kinematics = transport_chart_kinematics(chart, frame_params)
    load = law.resolve_load(kinematics.condition, actual, parameters)
    aerodynamics = flight_aerodynamics(kinematics.condition, load, aero_params)
    thrust = law.resolve_thrust(
        kinematics.condition, actual, aerodynamics.drag_n, max_thrust_n, parameters
    )
    rate = torch.cat(
        (
            transport_chart_rate(
                kinematics, thrust, actual[..., BANK_ACTUATOR], load, aerodynamics, aero_params
            ),
            (commands - actual) / time_constants_s,
        ),
        dim=-1,
    )
    return rate / state_scale


def _rk4(rhs, state: torch.Tensor, dt_s: torch.Tensor | float) -> torch.Tensor:
    """One explicit RK4 step of ``rhs`` from ``state``."""
    dt = torch.as_tensor(dt_s, dtype=state.dtype, device=state.device)
    while dt.ndim < state.ndim:
        dt = dt.unsqueeze(-1)
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_lag_step(
    state_lag: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor | float,
    step_context: torch.Tensor,
    law: LagControlLaw | type,
) -> torch.Tensor:
    """One explicit RK4 step over the coupled point-mass/actuator system; the law resolves its
    load and thrust at each of the four stages, which is what holds a specific force or a path
    angle across the step."""
    return _rk4(
        lambda state: lag_rhs(state, commands, aero_params, step_context, law),
        state_lag,
        dt_s,
    )


#: One compiled kernel per (law, grad mode) entry point, built on first use on CUDA and reused
#: for the life of the process.
_COMPILED_CUDA_STEPS: dict = {}


def _dispatch_step(
    law: type,
    state: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    """Eager on CPU, a fused Inductor graph on CUDA — as the point-mass backends do.

    Measured before this: a lagged epoch cost ~95 s against the point-mass backend's ~15 s
    on the same KSJC fold, entirely because this step ran eager while the others cached a
    compiled one.
    """
    if not state.is_cuda:
        return rk4_lag_step(state, commands, aero_params, dt_s, step_context, law)
    grad_enabled = torch.is_grad_enabled()
    source = law.step_autograd if grad_enabled else law.step_inference
    compiled = _COMPILED_CUDA_STEPS.get(source)
    if compiled is None:
        # ``dynamic=True`` in backward reproducibly segfaults on this Torch/CUDA stack
        # (see torch_dynamics), so the autograd kernel stays static.
        compiled = torch.compile(
            source,
            fullgraph=True,
            dynamic=not grad_enabled,
            mode="reduce-overhead",
        )
        _COMPILED_CUDA_STEPS[source] = compiled
    return compiled(state, commands, aero_params, dt_s, step_context)


def _rollout_step_and_context(
    law: LagControlLaw,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
    state_scale: torch.Tensor,
    initial_geodetic_states: torch.Tensor,
):
    """``(step function, step context)`` as the engines take them."""
    return partial(_dispatch_step, type(law)), lag_step_context(
        law, frame_params, time_constants_s, max_thrust_n, state_scale, initial_geodetic_states
    )


# ── the rollouts ──────────────────────────────────────────────────────────────────────────


def rollout_piecewise_constant(
    initial_geodetic_states: torch.Tensor,
    initial_controls: torch.Tensor,
    commands: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
    *,
    control_law: LagControlLaw = THRUST_FRACTION_LAW,
    chart_scale: tuple[float, ...],
    integrator_dt_s: float = 0.5,
    max_steps_per_segment: int = 4096,
) -> torch.Tensor:
    """Return augmented segment-end states ``[B,N,10]``."""
    state_scale = lag_state_scale(chart_scale, frame_params)
    step, context = _rollout_step_and_context(
        control_law, frame_params, time_constants_s, max_thrust_n, state_scale,
        initial_geodetic_states,
    )
    return rollout_piecewise_constant_with_step(
        lag_state_from_geodetic(
            initial_geodetic_states, initial_controls, frame_params, state_scale
        ),
        commands,
        segment_durations_s,
        aero_params,
        context,
        step,
        integrator_dt_s=integrator_dt_s,
        max_steps_per_segment=max_steps_per_segment,
    )


def rollout_piecewise_constant_hooked(
    initial_geodetic_states: torch.Tensor,
    initial_controls: torch.Tensor,
    commands: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
    command_hook: RawCommandHook,
    *,
    control_law: LagControlLaw = THRUST_FRACTION_LAW,
    track_reference: bool = False,
    chart_scale: tuple[float, ...],
    integrator_dt_s: float = 0.5,
    max_steps_per_segment: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Augmented segment-end states ``[B,N,10]`` and the effective commands ``[B,N,3]``
    when a hook rewrites each segment's command from the (scaled, augmented) state."""
    state_scale = lag_state_scale(chart_scale, frame_params)
    step, context = _rollout_step_and_context(
        control_law, frame_params, time_constants_s, max_thrust_n, state_scale,
        initial_geodetic_states,
    )
    return rollout_piecewise_constant_hooked_with_step(
        lag_state_from_geodetic(
            initial_geodetic_states, initial_controls, frame_params, state_scale
        ),
        commands,
        segment_durations_s,
        aero_params,
        context,
        step,
        command_hook,
        track_reference=track_reference,
        integrator_dt_s=integrator_dt_s,
        max_steps_per_segment=max_steps_per_segment,
    )


def rollout_piecewise_constant_at_times(
    initial_geodetic_states: torch.Tensor,
    initial_controls: torch.Tensor,
    commands: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
    query_offsets_s: torch.Tensor,
    query_valid: torch.Tensor,
    *,
    control_law: LagControlLaw = THRUST_FRACTION_LAW,
    chart_scale: tuple[float, ...],
    segment_valid: torch.Tensor | None = None,
    integrator_dt_s: float = 0.5,
    max_total_steps: int = 65536,
) -> DenseControlRollout:
    """Return event-aligned augmented states at queries and control boundaries."""
    state_scale = lag_state_scale(chart_scale, frame_params)
    step, context = _rollout_step_and_context(
        control_law, frame_params, time_constants_s, max_thrust_n, state_scale,
        initial_geodetic_states,
    )
    return rollout_piecewise_constant_at_times_with_step(
        lag_state_from_geodetic(
            initial_geodetic_states, initial_controls, frame_params, state_scale
        ),
        commands,
        segment_durations_s,
        aero_params,
        context,
        step,
        query_offsets_s,
        query_valid,
        segment_valid=segment_valid,
        integrator_dt_s=integrator_dt_s,
        max_total_steps=max_total_steps,
    )
