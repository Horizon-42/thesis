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

**What the surplus is measured against (``trombone_surplus_reference``, L3.f).** Everything
above sizes ``ΔL`` against the BEELINE ``D``, and L3.e measured what that costs: the beeline
is not the path a vectored flight intends to fly, so the whole of a downwind and a base reads
as time the aircraft does not need. At the TRUE CTA — where by construction there is nothing
to absorb — ``tromboneDelayS`` came out at p50 391 s, the endpoint ``|xt|`` p95 at 10 km, and
46 % of flights did not reach the threshold by ``T_cta``; the mechanism itself was clean
(delay absorbed exactly 30 s per 30 s of offset, thrust/stall/load violations −85/−84/−99 %).
The estimator, not the manoeuvre, was wrong.

Under ``reference-rollout`` the length the time is compared against is still the aircraft's
own beeline ``D``, plus **the detour the NETWORK ITSELF still intends to fly** on top of it::

    detour = L_ref − S_ref                     the reference's remaining path, less the
                                               straight line to the same endpoint
    Δt_unabsorbed = T_r − (D + detour) / V_e   ΔL = V_e·T_r − D − detour   (positive part)

— i.e. exactly the beeline surplus, less what the model was already going to spend on
vectoring. ``L_ref`` is the hook-free reference rollout's remaining horizontal path from this
segment's boundary, cut where it FIRST crosses the threshold ON THE FINAL (the shared
``final_approach_geometry.threshold_crossing_index`` — the plane ALONE is crossed abeam, on
a downwind, which is what this rule was until 2026-09-09 and why it measured ~12 km of
reference path on a 25 km approach; when the reference never crosses, the cut is the end of
its schedule and ``hook_trombone_ref_no_crossing`` says so), and
``S_ref`` is the straight line from that same boundary to that same cut. The dog-leg that
spends ``ΔL`` is unchanged, about the same base: ``cos θ = D / (D + ΔL)``. Under ``beeline``
that base IS ``V_e·T_r`` and the code keeps that spelling exactly, so the default reproduces
L3.e to the bit — records included, which is why the reference-only diagnostics are ABSENT
rather than zero under it.

**Why the DETOUR and not ``L_ref`` itself**, which is the more obvious reading of "size it
against the reference's remaining path". ``L_ref`` is indexed by the SCHEDULE: it knows the
step number and nothing about where the hooked aircraft has actually got to. The moment the
excursion opens, the aircraft leaves the reference's path — that is the point of the
manoeuvre — and the estimate stops responding to it, so the surplus never burns down; past
the reference's own threshold crossing ``L_ref`` is zero and ``ΔL`` degenerates to the whole
remaining reach, its largest possible value. Measured on the rollout fixture (48 segments,
60 s late, a flight whose own plan IS the beeline, where the axis should be a no-op): sizing
against ``L_ref`` pinned the offset at the 45° cap for 19 steps and ended 1.7 km wide of the
threshold, where ``beeline`` rolls out with zero saturated steps and ends 159 m out. The
detour form has neither problem, because ``D`` is the aircraft's own and ``detour`` is
non-increasing along the schedule by the triangle inequality (removing a leading chord takes
at least as much off the polyline as off the straight line) — so ``dΔL/dt ≤ 0`` survives as
the identity it is under ``beeline``, and the roll-out is guaranteed rather than hoped for.
It is also what makes the SHORTFALL case right: a reference that decelerates and stops short
of the threshold has a detour near zero, so the estimator falls back to the aircraft's own
beeline and the shortfall reads as the delay it is — sizing against ``L_ref`` would have read
those missing metres as surplus and stretched a flight that could not cover the path it
already had (measured on the on-time fixture: 9.4 s of invented delay, 15 of 48 steps
engaged, where ``beeline`` reports 0 and 0).

The reference comes from ``RolloutStateView.reference``, the hook-free schedule integrated
alongside (``needs_reference``; the composite carries it, and it costs one extra integration
of the schedule, once — ``control/constraints/composite.py``). One approximation, stated: the
path is summed as the CHORD between consecutive segment boundaries, not the arc. The command
is constant within a hold, so the segment is a circular arc and the chord is short by
``sinc(Δψ/2)`` — 0.15 % at a 15° bank over the deployed ~5 s hold, 2.8 % at 45°. Both terms
of ``detour`` are chords, so the two errors partly cancel; the alternative is a second dense
rollout for a sub-percent correction.
"""

from __future__ import annotations

import math

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from ts_transformer.config import TROMBONE_SURPLUS_REFERENCE_ROLLOUT, TSConfig
from ts_transformer.outputs.control.constraints.gates import on_final_weight, runway_axes_view
from ts_transformer.outputs.control.constraints.saturation import (
    ACTIVE_BANK_CHANGE_RAD,
    SATURATION_SOFTNESS_RAD,
    soft_max,
    soft_min,
)
from ts_transformer.outputs.control.constraints.speed_floor import floor_speed
from ts_transformer.outputs.control.dynamics.hooks import HOOK_STEPS_KEY, RolloutStateView
from ts_transformer.outputs.control.envelope import MAX_BANK_RAD, MAX_LOAD_FACTOR, MIN_LOAD_FACTOR
from ts_transformer.final_approach_geometry import (
    alignment_cosine,
    hard_aligned,
    runway_axes,
    threshold_crossing_index,
)

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
#: Reported ONLY under ``reference-rollout``, and that is deliberate: a key written on every
#: record distinguishes nothing, and adding one under the default would change the bytes of
#: every L3.e record on disk. Both are held at every step, so the published share (value /
#: ``hook_steps``) IS the anchor's value — the LENGTH the surplus was sized against there
#: (``D + detour``, which is the reference's own remaining path whenever it crosses at the
#: threshold point, so ``tromboneDelayS = T_r − tromboneRefPathM / V_e`` reconstructs), and
#: whether that reference ever reaches the threshold on the final at all.
_REFERENCE_DIAGNOSTIC_KEYS = (
    "hook_trombone_ref_path_m", "hook_trombone_ref_no_crossing",
)
#: The label that says which estimator produced the counts above. Same rule: present only
#: under ``reference-rollout``, so a record without it was sized against the beeline.
_SURPLUS_REFERENCE_LABEL_KEY = "hook_trombone_surplus_reference"


class Trombone:

    def __init__(self, config: TSConfig, dynamics: dict[str, torch.Tensor], *, hard: bool):
        self.runway_heading = dynamics["runway_heading_rad"]
        # (S, Cl_max, ...) and the frame origin's altitude: the floor is a density at a
        # geodetic height and the chart carries height above that origin.
        self.aero_params = dynamics["aero_params"]
        self.origin_altitude_m = dynamics["frame_params"][:, 2]
        self.margin = config.control_speed_floor_margin
        self.bank_lag_s = config.control_bank_time_constant_s
        self.hard = hard
        # WHAT the surplus is measured against, and the one thing that follows from it: the
        # `reference-rollout` estimator reads the hook-free schedule's own rollout, so this
        # instance asks the engine for it. `beeline` asks for nothing and is what every
        # stored artifact ran under.
        self.surplus_reference = config.trombone_surplus_reference
        self.needs_reference = (
            self.surplus_reference == TROMBONE_SURPLUS_REFERENCE_ROLLOUT
        )
        # The keys this instance reports; the reference pair only exists where a reference
        # was read, so a `beeline` record is byte-identical to L3.e's.
        self._keys = _DIAGNOSTIC_KEYS + (
            _REFERENCE_DIAGNOSTIC_KEYS if self.needs_reference else ()
        )
        # ``[len(self._keys), B]`` — per row, so a prediction record carries its own.
        self._counts: torch.Tensor | None = None
        # Per-row MODE, latched across segments (never a gradient path): the side the
        # excursion is flown on, the stretch latched at engagement (0 = not engaged), the
        # anchor's estimate, and whether the on-final gate has ever opened for this flight.
        self._side: torch.Tensor | None = None
        self._target_m: torch.Tensor | None = None
        self._anchor_delay_s: torch.Tensor | None = None
        self._gate_seen: torch.Tensor | None = None
        # The detour the reference still intends at every segment start ``[B,N]`` and
        # whether it ever crossed on the final ``[B]``, derived once per rollout from
        # the hook-free trajectory; None under `beeline`, where nothing reads them.
        self._ref_detour_m: torch.Tensor | None = None
        self._ref_no_crossing: torch.Tensor | None = None
        self._anchor_ref_path_m: torch.Tensor | None = None

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
        # DL: the path length the delay needs — the reach, less the length the aircraft
        # still has to cover. WHICH length is the axis: the beeline alone, or the beeline
        # plus the detour the hook-free rollout still intends to fly on top of it. Both
        # are anchored on the AIRCRAFT's own `direct`, which is what burns the surplus down.
        if self.needs_reference:
            if self._ref_detour_m is None or segment_index == 0:
                self._read_reference(state, view)
            path = direct + self._ref_detour_m[:, segment_index]
            surplus = reach - path
        else:
            path = None
            surplus = reach - direct
        delay_s = (surplus / pace).clamp(min=0.0)      # ...as the seconds it stands for
        if self._side is None or segment_index == 0:
            self._reset(view, delay_s, path)

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
        # cos(theta) = D / (D + DL): the offset that spends exactly the surplus over the
        # beeline base. Under `beeline` that denominator IS the reach, and the spelling stays
        # the one L3.e ran — `direct + (reach - direct)` is not `reach` in IEEE, and this
        # division is what the offset, the stretch and every downstream count come from.
        # Under `reference-rollout` the surplus was measured against the beeline PLUS a
        # detour, so the base has to be spelled out; `surplus` is clamped because
        # `torch.where` evaluates both branches and a spent surplus makes it negative, and
        # `direct` is already at least a metre, so nothing further is needed to keep the
        # division finite. Where nothing is engaged the ratio is parked at 0 rather than 1 —
        # acos' is infinite at 1 and would put a NaN through a branch multiplied out anyway.
        base = reach if not self.needs_reference else direct + surplus.clamp(min=0.0)
        ratio = torch.where(engaged, direct / base, zero)
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
            # ...and, where the surplus was sized against the reference rollout, the length
            # it was sized against and whether that reference ever reached the threshold
            # plane at all. Held at every step for the same reason as the line above.
            *((
                self._anchor_ref_path_m.to(torch.float64),
                self._ref_no_crossing.to(torch.float64),
            ) if self.needs_reference else ()),
        ))
        self._counts = counts if self._counts is None else self._counts + counts
        # One hook per batch: the rows ARE the flights, so a second batch through the same
        # hook would broadcast into the first one's rows instead of failing.
        assert self._counts.shape[1] == counts.shape[1], "a hook is built per batch"
        return torch.stack((command[:, 0], filtered, coordinated), dim=-1)

    def _reset(self, view, delay_s: torch.Tensor, path: torch.Tensor | None) -> None:
        """A fresh rollout: no excursion, no gate seen, and the anchor's own estimate.

        ``path`` is the length that estimate was sized against, or None under ``beeline``
        where there is no second length to publish.
        """
        self._side = torch.zeros_like(view.d)
        self._target_m = torch.zeros_like(view.d)
        self._gate_seen = torch.zeros_like(view.d, dtype=torch.bool)
        self._anchor_delay_s = delay_s.detach()
        self._anchor_ref_path_m = None if path is None else path.detach()

    def _read_reference(self, state: RolloutStateView, view) -> None:
        """Derive, once per rollout, the DETOUR the HOOK-FREE schedule still means to fly.

        ``state.reference`` is that schedule's chart at every segment BOUNDARY ``[B,N+1,7]``.
        Cut it where it FIRST crosses the threshold ON THE FINAL
        (``final_approach_geometry.threshold_crossing_index``, shared with
        ``forecast.cut_at_threshold_crossing``) — interpolated inside the segment that
        crosses, because a segment is tens of seconds and counting or dropping a whole one is
        a kilometre either way — or, where it never crosses, at its last boundary, and say
        which (``hook_trombone_ref_no_crossing``): the two are not the same claim.

        The remaining path from boundary ``i`` to that cut is ``L_ref(i)``, the sum of the
        chords from ``i`` on; the straight line from the same boundary to the same cut is
        ``S_ref(i)``; and what the estimator uses is the difference. See the module
        docstring for why it is the difference and not ``L_ref`` itself — in one line,
        ``L_ref`` is indexed by the schedule and cannot see the excursion the hook is
        flying, so the surplus never burns down.

        ``detour`` is non-negative by the triangle inequality (both terms end at the same
        point) and the ``clamp`` only absorbs the metre floors above; it is non-increasing
        along the schedule by the same inequality, which is what makes the roll-out an
        identity rather than a hope.
        """
        if state.reference is None:
            raise ValueError(
                "the trombone sizes its surplus against the hook-free reference rollout "
                "(trombone_surplus_reference='reference-rollout', needs_reference); the "
                "backend passed none"
            )
        reference = state.reference.to(view.d.dtype)
        position = reference[..., :2]                       # [B, N+1, 2] chart (e, n)
        psi = self.runway_heading.to(view.d.dtype)
        d_ref, xt_ref = runway_axes(position[..., 0], position[..., 1], psi)
        step = position[:, 1:] - position[:, :-1]
        # The chord flown in each segment. Clamped BEFORE the root for the same reason
        # `direct` is: sqrt' is infinite at zero and one NaN gradient poisons the batch. A
        # segment shorter than a metre contributes nothing either way — at the deployed
        # ~5 s hold a segment is ~350 m — so the floor cannot change an answer.
        chord = torch.sqrt(step.square().sum(-1).clamp(min=_DISTANCE_FLOOR_M**2))
        segments = chord.shape[1]
        # WHERE the reference lands, from the one definition of it. The threshold PLANE
        # alone is not a landing: a vectored downwind runs parallel to the course several
        # kilometres abeam and passes ``d = 0`` out there, and cutting the path at that
        # abeam point measured ``L_ref`` to the wrong end of it (2026-09-09: on L3.e's
        # arms, ``tromboneRefNoCrossing`` 31.6 % and ~12 km of reference path on a 25 km
        # approach). The direction the gate reads is the chord flown INTO each boundary —
        # the path the reference means to fly, which is what this estimate is about.
        first, _run_end, crossed = threshold_crossing_index(
            d_ref[:, 1:], xt_ref[:, 1:],
            alignment_cosine(step[..., 0], step[..., 1], psi),
        )
        no_crossing = ~crossed
        # The segment the crossing happens in, or `segments` (past the last) for a reference
        # that never gets there — which makes the whole schedule count, below.
        first = torch.where(no_crossing, torch.full_like(first, segments), first)
        index = first.clamp(max=segments - 1).unsqueeze(1)
        before = d_ref[:, :-1].gather(1, index)
        after = d_ref[:, 1:].gather(1, index)
        # Where in that segment ``d`` reaches zero. The denominator is floored for a segment
        # that barely moves; the ordinary crossing has ``before > 0 >= after``, so it never
        # binds there. A reference that crossed the PLANE earlier, off the final, arrives at
        # its real crossing segment with ``before <= 0``: the ratio goes negative and clamps
        # to 0, i.e. the cut is that segment's start, which is the earliest the on-final rule
        # allows. A reference that never crosses is FORCED to 1 — its cut is the last
        # boundary, which is where ``weight`` below already counts the schedule to, and the
        # two must end at the same point or ``remaining − straight`` is a detour that is not
        # there. (Under the plane-only rule this line was a saturation: no crossing meant no
        # boundary past the plane, so ``before > 0`` and the ratio exceeded 1 on its own. The
        # on-final rule broke that — an overshoot IS past the plane and still never on the
        # final — and left a straight reference reporting ~one chord of invented detour at
        # every step, which never burns down. Measured on the 48-segment fixture before the
        # fix: 340 m of detour on a perfectly straight path.)
        fraction = torch.where(
            no_crossing.unsqueeze(1),
            torch.ones_like(before),
            (before / (before - after).clamp(min=_DISTANCE_FLOOR_M)).clamp(0.0, 1.0),
        )
        pair = index.unsqueeze(-1).expand(-1, -1, 2)
        start = position[:, :-1].gather(1, pair)
        end = position[:, 1:].gather(1, pair)
        cut = start + fraction.unsqueeze(-1) * (end - start)          # [B, 1, 2]
        column = torch.arange(segments, device=chord.device).unsqueeze(0)
        weight = (column < first.unsqueeze(1)).to(chord.dtype) + (
            (column == first.unsqueeze(1)).to(chord.dtype) * fraction
        )
        remaining = torch.flip(
            torch.cumsum(torch.flip(chord * weight, dims=(1,)), dim=1), dims=(1,)
        )
        straight = torch.sqrt(
            (position[:, :-1] - cut).square().sum(-1).clamp(min=_DISTANCE_FLOOR_M**2)
        )
        self._ref_detour_m = (remaining - straight).clamp(min=0.0)
        self._ref_no_crossing = no_crossing.to(chord.dtype).detach()

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
            torch.zeros(len(self._keys), dtype=torch.float64)
            if self._counts is None else self._counts.sum(dim=1).cpu()
        )
        return dict(zip(self._keys, counts.unbind()))

    def per_flight_diagnostics(self) -> dict[str, torch.Tensor]:
        """The same counts, one row per flight (``[B]`` each)."""
        if self._counts is None:
            return {name: torch.zeros(0, dtype=torch.float64) for name in self._keys}
        return dict(zip(self._keys, self._counts.cpu().unbind()))

    def diagnostic_labels(self) -> dict[str, str]:
        """Which estimator sized the surplus — reported only where it is not the default.

        A record carrying no such key was sized against the beeline, which is what the whole
        L3.e campaign ran under and what the default still is.
        """
        if not self.needs_reference:
            return {}
        return {_SURPLUS_REFERENCE_LABEL_KEY: self.surplus_reference}
