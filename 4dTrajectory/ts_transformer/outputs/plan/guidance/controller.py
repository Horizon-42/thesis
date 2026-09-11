"""`PlanGuidance`: the command hook that flies a plan on its route (design §4.3–4.6).

Called once per rollout segment with the aircraft's own state (`RolloutStateView`), it
returns the command flown for the hold: the BANK that tracks the route (a look-ahead point
on the polyline, turned onto through the lagging bank actuator exactly as the trombone
turns onto its dog-leg), the LOAD FACTOR that flies the height profile (the planned
descent to the capture height, then the glidepath from above or below), and the THRUST
that holds the speed schedule (`V_mid` to the deceleration point, then down to
`V_final`; never below the stall margin — the speed floor's own inversion of
``V̇ = (T − D)/m − g sin γ`` through the thrust actuator's lag).

Every command is inside `flyability`'s envelope (review C-12: the grader's floors, never
the learned head's box), so the record the rollout produces conforms by construction;
what the plan asked for and could not be flown — a bank the cap refused, thrust the
envelope refused, a load factor clamped — is counted per flight, never absorbed silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2, aerodynamic_coefficients
from ts_transformer.config import TSConfig
from ts_transformer.geometry.flyability import Envelope
from ts_transformer.outputs.constraints.barrier_filter import BarrierFilter
from ts_transformer.outputs.constraints.gates import runway_axes_view
from ts_transformer.outputs.constraints.speed_floor import floor_speed
from ts_transformer.outputs.dynamics.hooks import HOOK_STEPS_KEY, RolloutStateView
from ts_transformer.outputs.envelope import MAX_THRUST_FRACTION, MIN_THRUST_FRACTION
from ts_transformer.outputs.plan.guidance.route import Route, speed_schedule_mps

#: The guidance's own bounds, inside the grader's envelope (`flyability.Envelope`: bank
#: ±45°, load factor 0.5–2.0) with room for the lagging actuators' overshoot.
BANK_MAX_RAD = math.radians(25.0)
LOAD_FACTOR_MIN = 0.7
LOAD_FACTOR_MAX = 1.5
_ENVELOPE = Envelope.__dataclass_fields__
assert BANK_MAX_RAD < _ENVELOPE["max_bank_rad"].default
assert _ENVELOPE["min_load_factor"].default < LOAD_FACTOR_MIN < LOAD_FACTOR_MAX < _ENVELOPE["max_load_factor"].default
#: The stall margin the speed floor holds (the package's existing 1.10).
STALL_MARGIN = 1.10
#: The route point the bank aims at: this many seconds of flight ahead, at least this far —
#: the L1 point. The lateral acceleration that turns the velocity onto it,
#: ``a = 2 V² sin η / L1`` (η the angle from the velocity to the line of sight), is a
#: second-order tracker with damping 0.7 on a straight and a curvature feed-forward on an
#: arc; the coordinated bank that produces it is ``atan(a / g)``.
LOOKAHEAD_S = 8.0
LOOKAHEAD_MIN_M = 500.0
#: The speed schedule is approached at most this fast (an airliner's comfortable
#: deceleration on approach is ~0.5–1 m/s²).
ACCEL_MAX_MPS2 = 1.0
#: The height profile: a height error is closed over this many seconds, and the flight
#: path angle it asks for is capped (descent / climb).
HEIGHT_GAIN_S = 12.0
DESCENT_MAX_RAD = math.radians(6.0)
CLIMB_MAX_RAD = math.radians(2.0)
#: Capturing the glidepath from ABOVE: descend at this angle until on it.
CAPTURE_DESCENT_RAD = math.radians(4.5)

#: What the plan asked for and the envelope refused, counted per flight: the bank over the
#: cap, the thrust over the maximum (the schedule could not be held) and under idle (the
#: schedule needs more drag than the airframe has — a deceleration it cannot fly), the load
#: factor outside the guidance's own band; and the route offset summed over the steps.
_DIAGNOSTIC_KEYS = (
    HOOK_STEPS_KEY,
    "hook_plan_bank_capped_steps",
    "hook_plan_thrust_saturated_steps",
    "hook_plan_thrust_idle_steps",
    "hook_plan_load_clamped_steps",
    "hook_plan_route_cross_track_m",
)


@dataclass(frozen=True)
class PlanToFly:
    """The plan's operating parameters the controller reads, per flight."""

    v_mid_mps: float
    d_decel_m: float | None
    v_final_mps: float
    h_capture_m: float          # chart height above the aim point at the join
    #: `(remaining path, speed)` points the schedule runs through (the waypoints route:
    #: the anchor's and each fix's speed); empty for the plan's own law
    speed_points: tuple[tuple[float, float], ...] = ()


class PlanGuidance:
    needs_reference = False   # it flies its own route; the network's schedule is not consulted

    def __init__(
        self,
        config: TSConfig,
        dynamics: dict[str, torch.Tensor],
        routes: list[Route],
        plans: list[PlanToFly],
        anchor_heights_m: np.ndarray,
    ):
        if len(routes) != len(plans) or len(routes) != dynamics["runway_heading_rad"].shape[0]:
            raise ValueError("one route and one plan per flight in the batch")
        self.runway_heading = dynamics["runway_heading_rad"]
        self.aero_params = dynamics["aero_params"]
        self.origin_altitude_m = dynamics["frame_params"][:, 2]
        self.max_thrust_n = dynamics["max_thrust_n"]
        self.glidepath_tan = dynamics["glidepath_tan"]
        self.thrust_lag_s = config.control_thrust_time_constant_s
        self.routes = routes
        self.plans = plans
        self.anchor_heights_m = np.asarray(anchor_heights_m, dtype=np.float64)
        self._progress = np.zeros(len(routes), dtype=np.int64)
        self._counts: torch.Tensor | None = None
        # On the final the corridor is a hard bound, not a tracking target: the barrier's
        # bank interval (the same module the adopted predict-time hook runs, hard form)
        # filters the tracker's bank wherever the on-final gate is open, and re-coordinates
        # the load factor with it. Off the final it is silent. It runs BEFORE the thrust is
        # priced, so the floor and the drag read the load factor actually flown.
        self._barrier = BarrierFilter(config, dynamics, hard=True)

    # ── the route, per flight (NumPy: each flight has its own polyline) ─────────

    def _route_lookup(self, e: np.ndarray, n: np.ndarray, speed: np.ndarray):
        """For every flight: the bearing and distance to its L1 point on the route, the
        route's curvature at the nearest point (the feed-forward bank), the path still to
        fly to the threshold, and its offset from the route."""
        bearing = np.zeros(len(e))
        reach = np.zeros(len(e))
        curvature = np.zeros(len(e))
        remaining = np.zeros(len(e))
        offset = np.zeros(len(e))
        for i, route in enumerate(self.routes):
            start = int(self._progress[i])
            points = route.points[start:]
            distance = np.hypot(points[:, 0] - e[i], points[:, 1] - n[i])
            k = start + int(np.argmin(distance))
            self._progress[i] = k
            ahead = max(LOOKAHEAD_S * float(speed[i]), LOOKAHEAD_MIN_M)
            j = int(np.searchsorted(route.arc_m, route.arc_m[k] + ahead))
            j = min(j, len(route.points) - 1)
            de, dn = route.points[j, 0] - e[i], route.points[j, 1] - n[i]
            bearing[i] = math.atan2(dn, de)
            reach[i] = max(math.hypot(de, dn), LOOKAHEAD_MIN_M)
            curvature[i] = route.curvature_rad_per_m(k)
            remaining[i] = route.remaining_m(k)
            offset[i] = float(distance[k - start])
        return bearing, reach, curvature, remaining, offset

    def _reference_height(self, i: int, remaining: float) -> float:
        route, plan = self.routes[i], self.plans[i]
        s_join = route.remaining_m(route.join_index)
        s0 = route.remaining_m(0)
        glidepath = max(remaining, 0.0) * float(self.glidepath_tan[i])
        if remaining > s_join and s0 > s_join:
            # pre-final: a straight descent from the anchor height to the capture height
            fraction = (remaining - s_join) / (s0 - s_join)
            return plan.h_capture_m + (self.anchor_heights_m[i] - plan.h_capture_m) * fraction
        glidepath_at_join = s_join * float(self.glidepath_tan[i])
        if plan.h_capture_m > glidepath_at_join:
            # from above: descend at the capture angle until on the glidepath
            return max(glidepath, plan.h_capture_m - math.tan(CAPTURE_DESCENT_RAD) * (s_join - remaining))
        # from below: hold the capture height until the glidepath comes down to it
        return min(glidepath, plan.h_capture_m)

    # ── the hook ────────────────────────────────────────────────────────────────

    def __call__(
        self, state: RolloutStateView, command: torch.Tensor, segment_index: int
    ) -> torch.Tensor:
        if state.actuators is None:
            raise ValueError("the plan guidance needs the actuator states (a first-order-lag rollout)")
        view = runway_axes_view(state, self.runway_heading)
        real, dtype = view.d.dtype, command.dtype
        chart = state.chart.to(real)
        hold = view.hold_s
        e = chart[:, 0].detach().cpu().numpy()
        n = chart[:, 1].detach().cpu().numpy()
        speed_np = view.ground_speed.detach().cpu().numpy()
        target_heading, reach_np, curvature_np, remaining_np, offset_np = self._route_lookup(e, n, speed_np)
        hold_np = hold.detach().cpu().numpy()
        h_ref_now = np.array([self._reference_height(i, s) for i, s in enumerate(remaining_np)])
        h_ref_next = np.array([
            self._reference_height(i, s - v * h)
            for i, (s, v, h) in enumerate(zip(remaining_np, speed_np, hold_np, strict=True))
        ])
        device = chart.device
        target = torch.from_numpy(target_heading).to(device=device, dtype=real)
        reach = torch.from_numpy(reach_np).to(device=device, dtype=real)
        curvature = torch.from_numpy(curvature_np).to(device=device, dtype=real)
        h_now = torch.from_numpy(h_ref_now).to(device=device, dtype=real)
        h_next = torch.from_numpy(h_ref_next).to(device=device, dtype=real)

        # ── lateral: the L1 law — the bank whose lateral acceleration turns the velocity
        # onto the line of sight to the point ahead on the route ──────────────────────
        heading = torch.atan2(chart[:, 4], chart[:, 3])
        eta = target - heading
        eta = torch.atan2(torch.sin(eta), torch.cos(eta))
        actuators = state.actuators.to(real)
        # the arc's own turn as feed-forward (``ψ̇ = V κ``, bank ``atan(V² κ / g)``), the L1
        # term correcting the residual — a pure L1 tracker lags every arc by a steady error
        lateral_accel = 2.0 * view.ground_speed.square() * torch.sin(eta) / reach
        feed_forward = view.ground_speed.square() * curvature
        bank_demand = torch.atan((lateral_accel + feed_forward) / GRAVITY_MPS2)
        bank = bank_demand.clamp(-BANK_MAX_RAD, BANK_MAX_RAD)
        bank_capped = (bank_demand.abs() > BANK_MAX_RAD).detach()

        # ── vertical: the load factor that flies the height profile ─────────────────
        speed = view.speed
        vertical = (h_next - h_now) / hold + (h_now - view.height) / HEIGHT_GAIN_S
        vertical = torch.maximum(
            torch.minimum(vertical, speed * math.sin(CLIMB_MAX_RAD)),
            -speed * math.sin(DESCENT_MAX_RAD),
        )
        gamma_cmd = torch.asin((vertical / speed).clamp(-1.0, 1.0))
        gamma_dot = (gamma_cmd - view.path_angle) / hold
        n_vertical = torch.cos(view.path_angle) + speed * gamma_dot / GRAVITY_MPS2
        load_demand = n_vertical / torch.cos(bank)
        load = load_demand.clamp(LOAD_FACTOR_MIN, LOAD_FACTOR_MAX)
        load_clamped = ((load_demand < LOAD_FACTOR_MIN) | (load_demand > LOAD_FACTOR_MAX)).detach()

        # ── the corridor barrier on the final, before the thrust is priced: it may widen
        # the bank and re-coordinates the load with it, and the floor and the drag below
        # must read the manoeuvre flown, not the one it replaced ─────────────────────
        lateral = self._barrier(
            state, torch.stack((torch.zeros_like(bank), bank, load), dim=-1).to(dtype), segment_index,
        ).to(real)
        bank, load = lateral[:, 1], lateral[:, 2]

        # ── speed: the thrust that meets the schedule, never under the stall floor ──
        # The schedule is a GROUND-speed law (the extractors read it off the track); the
        # dynamics' V is the airspeed, which in this wind-free model is the ground speed
        # over cos γ — the target is converted, so the floor (an airspeed) and the thrust
        # law read one speed.
        v_ref = torch.stack([
            torch.as_tensor(float(speed_schedule_mps(
                s, v_mid=plan.v_mid_mps, d_decel_m=plan.d_decel_m, v_final=plan.v_final_mps,
                points=plan.speed_points,
            )), dtype=real, device=device)
            for s, plan in zip(remaining_np, self.plans, strict=True)
        ]) / torch.cos(view.path_angle)
        step = ACCEL_MAX_MPS2 * hold
        v_target = speed + (v_ref - speed).clamp(-step, step)
        aero = self.aero_params.to(real)
        floor, density = floor_speed(
            view, aero=aero, origin_altitude_m=self.origin_altitude_m.to(real),
            margin=STALL_MARGIN, commanded_load=load,
        )
        v_target = torch.maximum(v_target, floor)
        _cl, cd, _stalled = aerodynamic_coefficients(load, speed, view.mass, density, aero)
        drag = 0.5 * density * speed.square() * cd * aero[:, 0]
        mean_required = view.mass * (
            (v_target - speed) / hold + GRAVITY_MPS2 * torch.sin(view.path_angle)
        ) + drag
        max_thrust = self.max_thrust_n.to(real)
        tau_thrust = self.thrust_lag_s * (1.0 - torch.exp(-hold / self.thrust_lag_s))
        flying = actuators[:, 0]
        demand = (mean_required / max_thrust * hold - flying * tau_thrust) / (hold - tau_thrust)
        thrust = demand.clamp(MIN_THRUST_FRACTION, MAX_THRUST_FRACTION)
        thrust_saturated = (demand > MAX_THRUST_FRACTION).detach()
        thrust_idle = (demand < MIN_THRUST_FRACTION).detach()

        counts = torch.stack((
            torch.ones_like(bank, dtype=torch.float64),
            bank_capped.to(torch.float64),
            thrust_saturated.to(torch.float64),
            thrust_idle.to(torch.float64),
            load_clamped.to(torch.float64),
            torch.from_numpy(offset_np).to(device=device, dtype=torch.float64),
        ))
        self._counts = counts if self._counts is None else self._counts + counts
        assert self._counts.shape[1] == counts.shape[1], "a hook is built per batch"
        return torch.stack((thrust, bank, load), dim=-1).to(dtype)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        counts = (
            torch.zeros(len(_DIAGNOSTIC_KEYS), dtype=torch.float64)
            if self._counts is None else self._counts.sum(dim=1).cpu()
        )
        own = dict(zip(_DIAGNOSTIC_KEYS, counts.unbind()))
        barrier = {k: v for k, v in self._barrier.diagnostics().items() if k != HOOK_STEPS_KEY}
        return {**own, **barrier}

    def per_flight_diagnostics(self) -> dict[str, torch.Tensor]:
        if self._counts is None:
            return {name: torch.zeros(0, dtype=torch.float64) for name in _DIAGNOSTIC_KEYS}
        own = dict(zip(_DIAGNOSTIC_KEYS, self._counts.cpu().unbind()))
        barrier = {k: v for k, v in self._barrier.per_flight_diagnostics().items() if k != HOOK_STEPS_KEY}
        return {**own, **barrier}

    def diagnostic_labels(self) -> dict[str, str]:
        return {}
