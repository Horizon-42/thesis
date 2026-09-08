"""The per-step path stretch: the delay the remaining path cannot absorb, flown as a dog-leg.

The measured problem (L3.d, ``docs/2026-09-07_latent_intent_design.zh.md`` §六 L3.d). Under
a late CTA the duration head obeys the arrival time exactly, so the rollout's TOTAL TIME is
fixed; the network's own path is what it is; and a fixed path flown in a fixed time has a
fixed mean speed. The speed floor can refuse the infeasibly slow commands that produced the
stall samples, but it has nowhere to put the time it refuses to waste: the demand comes back
as thrust over max (0 samples before the floor, ~10 % of samples after), and the rollout
reaches the threshold EARLY and flies on: endpoint ``|xt|`` p95 43-63 km, pooled ADE
840 -> 3443 m at offset 0. **The missing degree of freedom is path length**, and the only
place a real arrival may buy it is BEFORE the final approach: a vectored dog-leg, a
trombone, an extended downwind. Inside the corridor there is nothing to buy — that is what
the corridor means. (The early arrival also has to stop being invisible in the readouts,
which is `forecast.cut_at_threshold_crossing` / `predict --truncate-at-threshold`, a
separate change: this hook is what gives the delay somewhere to go, that one is what makes
the reports read the approach rather than the flying after it.)

**The estimate.** At each segment start the hook reads the schedule's remaining time
``T_r`` (``RolloutStateView.remaining_s``; under a CTA-conditioned decoder ``T_cta − t``),
the beeline distance to the threshold ``D = hypot(d, xt)`` in runway axes, and the speed
floor for the COMMANDED load factor at this segment's own height
(``speed_floor.floor_speed`` — the same call the floor module makes on the same segment, so
the two cannot disagree). The delay the remaining path cannot absorb, and the extra path
length it is worth, are then one identity::

    V_e = max(V_floor, V_h)
    dt_unabsorbed = T_r − D / V_e              DL = V_e · dt_unabsorbed = V_e·T_r − D

``V_e·T_r`` is how far the aircraft covers in the time it has; ``D`` is how far it has to go;
the surplus has to be spent on path. **The speed is the one being flown, floored at the speed
floor** — and that ``max`` is not a detail, it is what makes the manoeuvre terminate. The
floor is a LOWER bound: the module only ever raises thrust, so ``V_h ≥ V_floor`` is the
invariant and ``V_h > V_floor`` is the ordinary case. Sizing against ``V_floor`` alone was the
first design and it fails exactly there: with the aircraft covering ground faster than the
estimate assumes, ``DL`` stops falling (measured on the fixture at commanded thrust 0.12, the
surplus rose from 2981 m and plateaued), the half-way switch never fires and the excursion
pins outbound until the threshold plane ends it, 2.9 km wide. With ``V_e``, ``dDL/dt =
−V_e + V_h·cos θ ≤ 0`` for any speed the aircraft flies, so the surplus is strictly spent and
the roll-out is guaranteed rather than hoped for. ``V_h`` is the HORIZONTAL speed, because
``D`` is a horizontal distance; the floor is an airspeed, and the two differ by ``cos γ`` —
0.999 on an approach.

**The manoeuvre.** Spend ``DL`` as a lateral excursion about the beeline. For a symmetric
dog-leg over a base ``D`` the extra length of a leg flown at ``theta`` off the base is
``D(sec theta − 1)``, so the offset that spends exactly ``DL`` is

    cos theta = D / (D + DL) = D / (V_floor · T_r)

— the heading at which flying the floor speed for the remaining time lands exactly on the
threshold. It is a pure function of the state: as the surplus is spent ``theta`` shrinks to
zero on its own, so the roll-out onto the beeline is guaranteed by the identity rather than
by a timer, and a flight with nothing to absorb never sees the hook at all (bit-identical
commands, in BOTH saturations). The offset is capped at :data:`OFFSET_MAX_RAD`.

It is a dog-leg, not a spiral: held on ONE side (``sigma``, latched at engagement as the
side the aircraft is already on, so the excursion stays WIDE of the extended centreline
rather than cutting across it) until half the stretch is flown, then mirrored. "Half is flown" needs no accumulator —
``DL`` itself burns down at the rate the extra length is bought, so ``DL <= DL*/2`` against
the value latched at engagement IS the half-way point. The mirrored leg converges: the
beeline is recomputed from the current position, so ``psi_bee − sigma·theta`` closes on it,
and ``theta -> 0`` ends the excursion pointing at the threshold.

**The hand-over rule (this is the one the spec left open).** The barrier owns the final; the
trombone owns nothing that is on it. An excursion may only be OPENED where the predicted
path is MORE than ``ALIGNMENT_MAX_DEG`` off the runway course. That is strictly inside "the
on-final gate is closed" (the gate REQUIRES alignment), so the hook can never open one
inside the gate; and it also declines the one state that is outside the gate yet already
lined up — wide of the membership cone but pointed down the final — which is exactly where a
dog-leg is a turn the barrier undoes as soon as the cone is entered.

Alignment gates the OPENING, not the continuation, and that asymmetry is not a loophole but
the same rule applied honestly: the dog-leg's own outbound leg can swing the aircraft
through the course (it is an offset from the beeline, and the beeline is not the course), and
a hook that fell silent there would leave the aircraft forty degrees off with no leg back —
measured on the fixture, exactly that: four engaged steps, then a parallel track past the
threshold 6.6 km wide. An open excursion therefore runs until its stretch is spent. Two
conditions bind either way: the hook never acts at or past the threshold (``d > 0``; past it
the runway frame's corridor is flat and the geometry means nothing), and once the on-final
gate has opened for a flight the hook is disabled for the rest of that rollout — a sticky
latch, so nothing can re-open a lateral excursion behind an aircraft already on final.

**The cost of the rule is real and is not hidden**: a flight already aligned with the course
at the anchor — a straight-in, which under the generous membership cone is inside the gate
from tens of kilometres out — cannot be stretched at all, and its unabsorbed delay comes back
in ``hook_trombone_delay_s`` next to an engaged share of zero. That is the readout to publish,
not a reason to let the hook into the corridor.

**Bank, lag and the load factor.** The heading change wanted this hold is inverted for the
bank exactly as the barrier does it — the command is HELD for ``dt``, the bank actuator is
first order, so the aircraft flies the bank it is in NOW for ``tau_eff = tau(1 − e^{−dt/tau})``
seconds' worth and the command for the rest, and only the remainder of the turn is asked of
the command::

    tan mu_c = [ (V_h / (g·L)) · dpsi − tan mu_0 · tau_eff ] / (dt − tau_eff)

with ``L = n cos mu`` the vertical lift factor being flown. The result is saturated into
``±`` :data:`TURN_BANK_MAX_RAD` (softly in the ``soft`` form, hard at inference), blended in
by the gate weight, clamped to the envelope, and the load factor is re-coordinated to keep
the vertical lift the network paired with it, ``n' = n cos mu / cos mu'``.

**Why the turn cap is 15° and not the envelope's 45°.** Under
``barrier+speed-floor+trombone`` the floor runs BEFORE this module — the value's ``+`` order
is its application order, which is the one thing the hook vocabulary exists to guarantee — so
the floor sets its thrust from the load factor the network commanded and does NOT price the
turn this module is about to add. The turn costs the margin twice: the coordinated load
factor raises the stall speed by ``sqrt(1/cos mu)``, and the induced drag it adds (``~tan^2
mu``) is drag the floor's thrust was not sized for, so the speed ends the hold below the
floor. Measured on the rollout fixture (48 segments, a base leg flown at the floor, ~6 s
holds, V_floor 64 m/s over a 58 m/s stall): the stall slack falls monotonically with the cap —
−0.6 m/s at 10°, −0.9 at 15°, −1.5 at 20°, −3.3 at 25° — and at 30° ``flyability`` reports its
first stall sample. The manoeuvre is indistinguishable all the way down to 10°, because the
OFFSET cap and not this one is what buys path; the bank only sets how many segments the
roll-in takes (34° of offset at 15° and 60 m/s is about four holds). 15° is a normal vectoring
bank with roughly twice the headroom the measurement needs, and
``hook_trombone_bank_capped_steps`` says how often it binds. Reordering the composite instead
would have let the floor price the turn exactly, and it is the better physics — but it would
have made a value's spelling disagree with what it does, which is worse.

**Under ``barrier+trombone`` the floor is an ASSUMPTION, not an enforced speed.** That stack
exists as the ablation that isolates the stretch from the floor, and there nothing holds the
speed up: ``V_floor`` is only the slowest speed the airframe MAY fly, so if the network
commands slower than that the ``max`` above still reports the floor and the detour is
over-sized for the ground the aircraft actually covers. Read that arm as "the stretch alone",
never as "the stretch under the floor it was sized against".

**What this hook does NOT do, and it matters for reading L3.e's gate (1).** It is GEOMETRY. The
rollout's speed is set by the thrust commands, the schedule is open loop, and the floor's demand
is a function of the speed and the height — so within one rollout the trombone barely touches
the thrust at all. Measured on the fixture: the floor's bound and saturated step counts are
IDENTICAL with and without it (47/48 and 0/48 either way). What the stretch changes is WHERE the
aircraft is when the schedule runs out — it lands at the threshold at ``T_cta`` instead of a
quarter of the schedule early and then flying on — and, through
``predict --truncate-at-threshold``, which part of the trajectory a report is scoring at all. If
``thrust_over_max`` falls in the arms, that is why; do not read it as the hook having relieved
the thrust demand.

**What ``hook_saturation`` does and does not soften.** It selects the saturation of the bank
into the turn cap and the ramp with which the stretch demand fades in (``soft`` is exactly
zero at zero surplus, so inertness survives), and the gate weight the command is blended by.
It does NOT soften the hand-over rule, the side, or the half-way switch: those are a MODE —
which module owns the command, and which way the dog-leg goes — and a mode that is half
taken is not a manoeuvre.
"""

from __future__ import annotations

import math

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from config import TSConfig
from control.constraints.gates import on_final_weight, runway_axes_view
from control.constraints.saturation import (
    ACTIVE_BANK_CHANGE_RAD,
    SATURATION_SOFTNESS_RAD,
    soft_max,
    soft_min,
)
from control.constraints.speed_floor import floor_speed
from control.dynamics.hooks import HOOK_STEPS_KEY, RolloutStateView
from control.envelope import MAX_BANK_RAD, MAX_LOAD_FACTOR, MIN_LOAD_FACTOR
from final_approach_geometry import hard_aligned

#: The largest heading offset from the beeline the hook will hold. ``sec 45° = 1.41``, so it
#: buys 41 % of extra path per metre of base — i.e. it BINDS once the delay to absorb exceeds
#: ~41 % of the time the remaining path needs at the floor speed. On the cohort this is
#: sized for that is far outside the arms (a 25 km run at ~70 m/s is 350 s, so the +90 s arm
#: asks for 26 % and lands at theta ~31°); past 45° the aircraft is flying more sideways than
#: forward, where ``sec theta`` stops being a useful parameterisation, and the honest answer
#: is that the delay is not absorbable before the final. ``hook_trombone_saturated_steps``
#: counts the steps where it bound, so "the stretch itself ran out of room" is readable — and
#: what a bound cap leaves is a lateral displacement at the end of the rollout (measured on
#: the fixture at 45 % of the remaining time: 3.3 km of cross-track, with the count pinned at
#: every engaged step), never a silent shortfall.
OFFSET_MAX_RAD = math.radians(45.0)
#: The largest bank the hook commands. Set by the speed floor, not by comfort — see the
#: module docstring's last paragraph but one. MEASURED on the rollout fixture (a 48-segment
#: base-leg arrival flown at the floor, `test_control_constraints`): the stall slack falls
#: monotonically with the cap (-0.6 m/s at 10 deg, -0.9 at 15, -1.5 at 20, -3.3 at 25) and at
#: 30 deg `flyability` reports its first stall sample; the arrival itself is unaffected all
#: the way down to 10 deg, because the OFFSET cap and not this one is what buys path. 15 deg
#: is the adopted value: a normal vectoring bank with about twice the headroom the
#: measurement needs.
TURN_BANK_MAX_RAD = math.radians(15.0)
#: The surplus over which the soft form's demand fades in, in metres of extra path. 200 m is
#: ~2 s at approach speed: below it there is no dog-leg worth flying, and the ramp is exactly
#: zero at zero surplus (and C¹ there), so a flight with nothing to absorb is untouched under
#: ``soft`` as well as ``hard``.
ENGAGE_SCALE_M = 200.0
#: A metre: the floor under the beeline distance and under the reach, so neither the root
#: nor the division has a NaN backward at exactly zero. Both are far inside the ``d > 0`` the
#: hook requires, so neither can change an answer.
_DISTANCE_FLOOR_M = 1.0
_DIAGNOSTIC_KEYS = (
    HOOK_STEPS_KEY, "hook_trombone_engaged_steps", "hook_trombone_bound_steps",
    "hook_trombone_saturated_steps", "hook_trombone_bank_capped_steps",
    "hook_trombone_offset_rad", "hook_trombone_stretch_m", "hook_trombone_delay_s",
)


class Trombone:
    needs_reference = False   # reads the hooked state and the schedule's own clock

    def __init__(self, config: TSConfig, dynamics: dict[str, torch.Tensor], *, hard: bool):
        self.runway_heading = dynamics["runway_heading_rad"]
        # (S, Cl_max, ...) and the frame origin's altitude: the floor is a density at a
        # geodetic height and the chart carries height above that origin.
        self.aero_params = dynamics["aero_params"]
        self.origin_altitude_m = dynamics["frame_params"][:, 2]
        self.margin = config.control_speed_floor_margin
        self.bank_lag_s = config.control_bank_time_constant_s
        self.hard = hard
        # ``[len(_DIAGNOSTIC_KEYS), B]`` — per row, so a prediction record carries its own.
        self._counts: torch.Tensor | None = None
        # Per-row MODE, latched across segments (never a gradient path): the side the
        # excursion is flown on, the stretch latched at engagement (0 = not engaged), the
        # anchor's estimate, and whether the on-final gate has ever opened for this flight.
        self._side: torch.Tensor | None = None
        self._target_m: torch.Tensor | None = None
        self._anchor_delay_s: torch.Tensor | None = None
        self._gate_seen: torch.Tensor | None = None

    def __call__(
        self, state: RolloutStateView, command: torch.Tensor, segment_index: int
    ) -> torch.Tensor:
        view = runway_axes_view(state, self.runway_heading)
        real, dtype = view.d.dtype, command.dtype
        hold = view.hold_s
        bank, load = command[:, 1], command[:, 2]
        # The floor the speed-floor module holds on this same segment, at the COMMANDED
        # load factor: one definition, so the detour is sized against the speed that will
        # actually be flown rather than against a second opinion of it.
        floor, _density = floor_speed(
            view,
            aero=self.aero_params.to(real),
            origin_altitude_m=self.origin_altitude_m.to(real),
            margin=self.margin,
            commanded_load=load.to(real),
        )
        # Beeline to the threshold. Clamped BEFORE the root, as `alignment_cosine` clamps its
        # own: hypot's backward is NaN at exactly (0, 0), and one NaN gradient poisons the
        # batch. A metre is far inside the `d > 0` the hook needs anyway.
        direct = torch.sqrt(
            (view.d.square() + view.xt.square()).clamp(min=_DISTANCE_FLOOR_M**2)
        )
        # The speed the surplus is measured at: the one being flown, floored at the speed
        # floor. See the module docstring — the max is what makes the excursion terminate.
        pace = torch.maximum(floor, view.ground_speed)
        # ...and floored again as a DENOMINATOR: `torch.where` evaluates both branches, so a
        # zero-length schedule would put a NaN gradient through the masked-out one.
        reach = (pace * state.remaining_s.to(real)).clamp(min=_DISTANCE_FLOOR_M)
        surplus = reach - direct                       # DL: the path length the delay needs
        delay_s = (surplus / pace).clamp(min=0.0)      # ...as the seconds it stands for
        if self._side is None or segment_index == 0:
            self._reset(view, delay_s)

        # ── who owns the command here ────────────────────────────────────────────────
        # A hand-over, so it is the HARD predicate under both saturations. "Not aligned
        # with the course" is strictly inside "the gate is closed" (the gate requires
        # alignment), which is why the hook can never act on the final; it also declines
        # the wide-but-lined-up state, where a dog-leg is a turn the barrier would undo.
        self._gate_seen |= on_final_weight(view, hard=True).detach() > 0.5
        available = ((view.d > 0.0) & ~self._gate_seen).detach()
        # Alignment gates the START of an excursion, not its continuation: the dog-leg's own
        # outbound leg can point the aircraft down the course for a while, and a hook that
        # stopped there would leave it 40° off its intended track with no leg back. Once
        # open, an excursion runs until its stretch is spent or the gate opens.
        opened = self._target_m > 0.0
        engaged = (
            available & (opened | ~hard_aligned(view.cos_align).detach())
            & (surplus.detach() > 0.0)
        )
        self._latch(engaged, opened, view, surplus)

        # ── the dog-leg ──────────────────────────────────────────────────────────────
        zero = torch.zeros_like(direct)
        # cos(theta) = D / (V_floor T_r): the offset that spends exactly the surplus. Where
        # nothing is engaged the ratio is parked at 0 rather than 1 — acos' is infinite at 1
        # and would put a NaN through a branch that is multiplied out anyway.
        ratio = torch.where(engaged, direct / reach, zero)
        demanded = torch.acos(ratio.clamp(0.0, 1.0))
        offset = torch.where(engaged, demanded.clamp(max=OFFSET_MAX_RAD), zero)
        # The heading error that points straight at the threshold (xt_dot = −V sin psi_err,
        # d_dot = −V cos psi_err, so the beeline is atan2(xt, d)), then the excursion: out on
        # the latched side while more than half the stretch is left, mirrored after.
        beeline = torch.atan2(view.xt, view.d)
        outbound = surplus.detach() > 0.5 * self._target_m
        target = beeline + torch.where(outbound, self._side, -self._side) * offset

        # ── the bank that turns onto it, over a hold, through a lagging actuator ──────
        tau_eff = self.bank_lag_s * (1.0 - torch.exp(-hold / self.bank_lag_s))
        actuators = state.actuators.to(real)
        committed = torch.tan(actuators[:, 1]) * tau_eff
        # The vertical lift factor being flown sets the turn rate a bank produces; the
        # coordination below keeps the commanded one, so the two agree once the actuators
        # settle. Floored well below any flown value, as the barrier floors it.
        lift = (actuators[:, 2] * torch.cos(actuators[:, 1])).clamp(min=0.5)
        scale = view.ground_speed / (GRAVITY_MPS2 * lift)
        change = target - view.heading_error
        change = torch.atan2(torch.sin(change), torch.cos(change))
        demand = torch.atan(
            (scale * change - committed) / (hold - tau_eff)
        ).to(dtype)
        cap = torch.full_like(demand, TURN_BANK_MAX_RAD)
        if self.hard:
            turn = torch.minimum(torch.maximum(demand, -cap), cap)
            weight = engaged.to(dtype)
        else:
            turn = soft_min(
                soft_max(demand, -cap, SATURATION_SOFTNESS_RAD), cap, SATURATION_SOFTNESS_RAD
            )
            # Continuous in the state: the demand fades in with the stretch it is for,
            # exactly zero and with zero slope at zero surplus. The ADMISSION stays the hard
            # predicate above — it is a hand-over, and blending it with the soft gate as well
            # would have let this module write a vanishing bank on steps the barrier owns.
            ramp = torch.tanh((surplus.clamp(min=0.0) / ENGAGE_SCALE_M) ** 2)
            weight = (engaged.to(real) * ramp).to(dtype)
        filtered = (bank + weight * (turn - bank)).clamp(min=-MAX_BANK_RAD, max=MAX_BANK_RAD)
        # Keep the vertical lift component the network paired with its load factor — but only
        # where the bank actually moved. `(load * cos b) / cos b` is NOT the identity in IEEE
        # (measured: it differs by an ULP for ~40 % of operand pairs), and a hook that is
        # supposed to be inert must not perturb the load of a row it never touched: the
        # arm-vs-arm "to the bit" comparison is exactly what that inertness is for.
        coordinated = torch.where(
            filtered == bank,
            load,
            (load * torch.cos(bank) / torch.cos(filtered)).clamp(
                min=MIN_LOAD_FACTOR, max=MAX_LOAD_FACTOR
            ),
        )

        moved = (filtered - bank).abs().detach()
        engaged_f = engaged.to(torch.float64)
        counts = torch.stack((
            torch.ones_like(engaged_f),
            engaged_f,
            (engaged & (moved > ACTIVE_BANK_CHANGE_RAD)).to(torch.float64),
            # The stretch ran out of room: the offset the surplus asks for is past the cap,
            # so this step cannot buy all of the delay it was asked to. The counterpart of
            # the floor's "full thrust is not enough" — without it, a run whose delay is
            # simply not absorbable before the final reads like one that absorbed it.
            (engaged & (demanded.detach() >= OFFSET_MAX_RAD)).to(torch.float64),
            # ...and the 15 deg TURN cap bound: the whole "why 15 deg" argument rests on this
            # cap, so how often it actually binds has to be readable off an arm.
            (engaged & (demand.detach().abs() >= TURN_BANK_MAX_RAD)).to(torch.float64),
            offset.detach().to(torch.float64),
            # The extra path this hold's offset commands: what the aircraft flies minus its
            # progress along the beeline. Summed and divided by `hook_steps` on the way out,
            # so the record's share x its step count is the total.
            (view.ground_speed * hold * (1.0 - torch.cos(offset))).detach().to(torch.float64),
            # Held at every step so the published share (value / steps) IS the anchor's
            # estimate — a flight the hand-over rule never admits still reports the delay it
            # was carrying, next to an engaged share of zero.
            self._anchor_delay_s.to(torch.float64),
        ))
        self._counts = counts if self._counts is None else self._counts + counts
        # One hook per batch: the rows ARE the flights, so a second batch through the same
        # hook would broadcast into the first one's rows instead of failing.
        assert self._counts.shape[1] == counts.shape[1], "a hook is built per batch"
        return torch.stack((command[:, 0], filtered, coordinated), dim=-1)

    def _reset(self, view, delay_s: torch.Tensor) -> None:
        """A fresh rollout: no excursion, no gate seen, and the anchor's own estimate."""
        self._side = torch.zeros_like(view.d)
        self._target_m = torch.zeros_like(view.d)
        self._gate_seen = torch.zeros_like(view.d, dtype=torch.bool)
        self._anchor_delay_s = delay_s.detach()

    def _latch(
        self, engaged: torch.Tensor, opened: torch.Tensor, view, surplus: torch.Tensor
    ) -> None:
        """Open an excursion where one starts, hold it, and forget it where it ends.

        The side is the one the aircraft is already on, so the dog-leg stays wide rather
        than cutting across the extended centreline; ``_target_m`` is the stretch the
        excursion was opened for and 0 means "no excursion", so a surplus that comes back
        later opens a fresh one instead of resuming a spent one.
        """
        opening = engaged & ~opened
        self._side = torch.where(
            opening, torch.where(view.xt >= 0.0, -1.0, 1.0).to(self._side.dtype), self._side
        )
        self._target_m = torch.where(
            engaged,
            torch.where(opening, surplus.detach(), self._target_m),
            torch.zeros_like(self._target_m),
        )

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Step counts (and the summed offset/stretch) over every call, summed over the batch."""
        counts = (
            torch.zeros(len(_DIAGNOSTIC_KEYS), dtype=torch.float64)
            if self._counts is None else self._counts.sum(dim=1).cpu()
        )
        return dict(zip(_DIAGNOSTIC_KEYS, counts.unbind()))

    def per_flight_diagnostics(self) -> dict[str, torch.Tensor]:
        """The same counts, one row per flight (``[B]`` each)."""
        if self._counts is None:
            return {name: torch.zeros(0, dtype=torch.float64) for name in _DIAGNOSTIC_KEYS}
        return dict(zip(_DIAGNOSTIC_KEYS, self._counts.cpu().unbind()))
