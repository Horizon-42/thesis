"""The bounded control box the learned head predicts in, and its physical conversion.

Three numbers with three different natural magnitudes — thrust in hundreds of kilonewtons,
bank in radians, load factor around one — cannot share a sigmoid, a regularizer or a
teacher-imitation MSE without one of them dominating. So the CONTRACT this package
predicts in is one box per ``control_thrust_parameterization`` (:func:`control_contract`),
the same on every airframe — four since 2026-09-16:

    thrust-fraction   delta_T = T / T_max     normalised by the ACTUATOR (T_max installed)
    specific-force    n_x = (T - D) / W       normalised by the EFFECT (speed-rate in g)
    speed-command     Δv = v_c − V₀ (m/s)     a target speed relative to the anchor's, flown by
                                              a speed loop (design §12); the one column in
                                              physical units, since a speed IS airframe-free
    specific-force+path-angle                 the same first column, and the THIRD becomes the
                      γ* (rad)                target path angle, flown by a path loop through
                                              the load the RHS re-solves (design §14)
    bank_rad          unchanged; a quarter turn is already order one
    load_factor       unchanged; it is a ratio by definition — except under the path-angle
                      contract, where that column is γ* and the load is resolved, not named

``thrust-fraction`` made the box mean the same ACTUATOR setting on every airframe (before
2026-08-18 one sigmoid output meant 100 kN on a small jet and 400 kN on a heavy).
``specific-force`` makes it mean the same MOTION: the lag RHS re-solves the thrust at every
stage and the drag cancels, so the same n_x moves a C550 and a B77W identically
(``docs/2026-09-14_specific_force_control_design.md``). ``speed-command`` keeps that
invariance and adds the restoring force n_x lacks; ``specific-force+path-angle`` leaves the
speed to n_x and closes the VERTICAL channel instead, which is the one the open-loop load
factor turns into a double integrator (design §7.5.3). Under thrust-fraction the newton
conversion is :func:`physical_controls`, once, on the way into the dynamics and out to the
record; under the other three it needs the state's drag and lives in the lag RHS
(``aerodynamic_model.torch_lag_dynamics``) and in the record export. The evaluation record
contract is newtons either way, shared with the CasADi optimizer.

**Negative thrust is deliberate.** ``MIN_THRUST_FRACTION`` is below zero because a real
approach needs net-negative propulsive force: idle thrust plus the drag of speedbrake,
flaps and gear, none of which this clean-configuration polar models. Measured on the KSJC
outer-train cohort with a non-negative floor, **40 % of inverted teacher segments pinned at
0 N**, i.e. the teacher could not reproduce the deceleration the aircraft actually flew.
``flyability.py`` already treats negative required thrust as a SOFT violation for exactly
this reason (median required thrust on a real arrival is 0.43 kN — idle).

**This is not the optimizer's envelope.** ``optimization/casadi_optimizer.make_control_bounds``
keeps a non-negative thrust floor and a 0.5 load floor. That is deliberate: the optimizer
publishes a trajectory it claims is flyable, so its box is a certification statement, while
this box is the search space of a learned head that is judged afterwards. Changing one does
not imply changing the other, and the optimizer's published artifacts are not restaled by
anything here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ts_transformer.config import (
    CONTROL_SPECIFIC_FORCE,
    CONTROL_SPECIFIC_FORCE_PATH_ANGLE,
    CONTROL_SPEED_COMMAND,
    CONTROL_THRUST_FRACTION,
)


CONTROL_NAMES = ("thrust_fraction", "bank_rad", "load_factor")
# The bank column, so a term acting on bank alone (the total-variation penalty) never
# hard-codes a 1. The same column under both contracts.
BANK_INDEX = CONTROL_NAMES.index("bank_rad")

MIN_THRUST_FRACTION = -0.2
MAX_THRUST_FRACTION = 1.0
# The optimizer's general bound, and the value flyability judges against.
MAX_BANK_RAD = math.pi / 4.0
# Meeting 2026-08-17: 0.2-2 g is the reasonable band for transport arrivals; negative and
# above-2 g are both very rare. Measured on real inverted KSJC tracks the required load
# factor spans only [0.92, 1.18], so this box never binds on observed data — it binds on
# what the LEARNED head may emit, which is the point.
#
# This floor is NOT the grader's: `flyability.Envelope.min_load_factor` is 0.5, so a segment the
# head emits at n in [0.2, 0.5) is unflyable by construction on the published metric. Decided
# 2026-09-09 (package review C-12) that neither number moves — the grader's floor is the
# flyability claim every published number was read against, this box is the old path's search
# space. A layer that COMMANDS the load factor (the plan-and-guidance design's guidance layer)
# reads the grader's envelope, never this one.
MIN_LOAD_FACTOR = 0.2
MAX_LOAD_FACTOR = 2.0

# The specific-force box (design §2.3). Its half width is the thrust-fraction box's (0.6)
# times the fleet-median installed thrust-to-weight at landing mass (0.358, 28 OpenAP types,
# `docs/literature/control_normalization/measurements/fleet_spread.py`), so a physical
# speed-rate error costs the imitation MSE the same under both contracts for the median
# airframe and one imitation weight reads the same in both arms. The floor sits below the
# truth teacher's p0.1 (-0.179 g on KRDU val) and below the feasible floor of 99.7 % of its
# states (`docs/specific_force_teacher_distribution.py`); the box cuts 0.10 % of that
# teacher. Like the thrust-fraction box it is the head's SEARCH SPACE: feasibility is the
# thrust clamp inside the lag RHS, which admits the thrust-fraction box's thrust RANGE at
# every state — the same range, not the same trajectories (see NEUTRAL_SPECIFIC_FORCE).
MIN_SPECIFIC_FORCE = -0.20
MAX_SPECIFIC_FORCE = 0.23
# The specific-force neutral: n_x = sin(gamma) holds speed, and -0.05 is the hold on a
# ~2.9 deg descent — an approach's dominant condition, beside the truth teacher's median
# (-0.059 g, KRDU val, `docs/specific_force_teacher_distribution.py`). NOT level trim, which
# is what the thrust-fraction neutral (0.2) means for the median airframe: under that law
# the drag feedback bounds the overspeed a level-trim command builds on a descent (~210 m/s
# on four synthetic 2.5-3 deg KRDU descents over 416 s), while under specific-force the speed
# has no drag feedback at all (dV'/dV = 0 where the thrust clamp does not bind), so level
# trim there flew the same untrained heads to 311-328 m/s, stopped only by the T_max clamp
# (M1 review, 2026-09-14). At -0.05 they stay within 103-144 m/s from 118-136 m/s.
NEUTRAL_SPECIFIC_FORCE = -0.05
# The speed-command contract (design §12): the head's first column is the target airspeed
# RELATIVE to the anchor's, Δv in m/s, flown by a first-order speed loop of time constant
# SPEED_LOOP_TIME_CONSTANT_S. The box holds the truth's own commands (KRDU val, 1404 x 32:
# p0.1 -85 m/s, p99.9 +15 m/s, at every loop constant from 5 to 12 s — §12.3) and the neutral
# 0 is "hold the speed you have", so an untrained head flies no transient on any airframe.
# The loop constant is part of the CONTRACT, like the specific-force box, not a config field:
# the inverse reads it at every call site and the teacher barely moves with it, so a run
# that wants another value is another contract.
MIN_SPEED_COMMAND_DELTA_MPS = -90.0
MAX_SPEED_COMMAND_DELTA_MPS = 20.0
SPEED_LOOP_TIME_CONSTANT_S = 8.0
# The path-angle contract (design §14): the head's THIRD column is the target path angle γ*
# in radians, flown by a first-order path loop of time constant PATH_ANGLE_TIME_CONSTANT_S
# through the load factor the lag RHS re-solves at every stage. Unlike the speed command it
# is ABSOLUTE, not anchor-relative: the anchor's own γ comes from an ADS-B vertical rate and
# its noise is the size of the whole error budget (§13.4: the break-even per-flight bias is
# 0.11°), so an anchor-relative target would carry that noise into exactly the quantity
# §7.5.3 shows integrating. The box is a flight-envelope bound, not a data bound: −15° is
# ~3,600 fpm at 70 m/s and +10° a go-around climb, against a teacher whose p1/p99 are
# −4.89/+0.25° (§13.2).
MIN_PATH_ANGLE_COMMAND_RAD = math.radians(-15.0)
MAX_PATH_ANGLE_COMMAND_RAD = math.radians(10.0)
# The neutral is the teacher's OWN median on KRDU val (−2.94°, §13.2), so a zeroed head flies a
# steady ~3° descent on every airframe: no zoom climb and no stall, and nothing airframe-
# specific. It is data-derived, not read from any published procedure — the head learns the
# profile; this is only where an untrained one starts.
NEUTRAL_PATH_ANGLE_RAD = math.radians(-2.9)
# Measured insensitive between 2 and 5 s (§13.2); 3 s sits above the 0.8 s load actuator and
# below the ~5.3 s segment hold, and keeps the load transient of a 1° step near 0.04 g. Like
# the speed loop's constant it is part of the CONTRACT, not a config field.
PATH_ANGLE_TIME_CONSTANT_S = 3.0


@dataclass(frozen=True)
class ControlContract:
    """One longitudinal parameterisation's box: what each column means, where the head's
    sigmoid maps it, and the "doing nothing" command a zeroed head starts every flight at."""

    parameterization: str
    names: tuple[str, str, str]
    units: tuple[str, str, str]
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    #: The neutral command, wings level at 1 g: under thrust-fraction 20 % of installed
    #: thrust — the level-flight trim of the median airframe (0.2 x 0.358 ≈ its D/W 0.076);
    #: under specific-force :data:`NEUTRAL_SPECIFIC_FORCE`, a descent's speed hold on EVERY
    #: airframe (why not level trim: that constant's comment); under speed-command Δv = 0,
    #: the anchor's own speed held; under the path-angle contract that same speed hold and
    #: :data:`NEUTRAL_PATH_ANGLE_RAD`, the teacher's own median descent.
    neutral: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not all(lo < mid < hi for lo, mid, hi in zip(self.lower, self.neutral, self.upper)):
            raise ValueError(f"{self.parameterization}: neutral must lie inside the box")

    @property
    def lower_array(self) -> np.ndarray:
        return np.asarray(self.lower, dtype=np.float64)

    @property
    def upper_array(self) -> np.ndarray:
        return np.asarray(self.upper, dtype=np.float64)

    @property
    def half_width(self) -> np.ndarray:
        """Half the box width per channel. An imitation MSE divides by these, so a
        full-scale error costs the same in each channel instead of the widest one
        drowning the others."""
        return (self.upper_array - self.lower_array) / 2.0


THRUST_FRACTION_CONTRACT = ControlContract(
    parameterization=CONTROL_THRUST_FRACTION,
    names=CONTROL_NAMES,
    units=("1", "rad", "1"),
    lower=(MIN_THRUST_FRACTION, -MAX_BANK_RAD, MIN_LOAD_FACTOR),
    upper=(MAX_THRUST_FRACTION, MAX_BANK_RAD, MAX_LOAD_FACTOR),
    neutral=(0.2, 0.0, 1.0),
)
SPECIFIC_FORCE_CONTRACT = ControlContract(
    parameterization=CONTROL_SPECIFIC_FORCE,
    names=("specific_force", "bank_rad", "load_factor"),
    units=("g", "rad", "1"),
    lower=(MIN_SPECIFIC_FORCE, -MAX_BANK_RAD, MIN_LOAD_FACTOR),
    upper=(MAX_SPECIFIC_FORCE, MAX_BANK_RAD, MAX_LOAD_FACTOR),
    neutral=(NEUTRAL_SPECIFIC_FORCE, 0.0, 1.0),
)
SPEED_COMMAND_CONTRACT = ControlContract(
    parameterization=CONTROL_SPEED_COMMAND,
    names=("speed_command_delta", "bank_rad", "load_factor"),
    units=("m/s", "rad", "1"),
    lower=(MIN_SPEED_COMMAND_DELTA_MPS, -MAX_BANK_RAD, MIN_LOAD_FACTOR),
    upper=(MAX_SPEED_COMMAND_DELTA_MPS, MAX_BANK_RAD, MAX_LOAD_FACTOR),
    neutral=(0.0, 0.0, 1.0),
)
PATH_ANGLE_CONTRACT = ControlContract(
    parameterization=CONTROL_SPECIFIC_FORCE_PATH_ANGLE,
    names=("specific_force", "bank_rad", "path_angle_command"),
    units=("g", "rad", "rad"),
    lower=(MIN_SPECIFIC_FORCE, -MAX_BANK_RAD, MIN_PATH_ANGLE_COMMAND_RAD),
    upper=(MAX_SPECIFIC_FORCE, MAX_BANK_RAD, MAX_PATH_ANGLE_COMMAND_RAD),
    neutral=(NEUTRAL_SPECIFIC_FORCE, 0.0, NEUTRAL_PATH_ANGLE_RAD),
)
_CONTRACTS = {
    contract.parameterization: contract
    for contract in (
        THRUST_FRACTION_CONTRACT,
        SPECIFIC_FORCE_CONTRACT,
        SPEED_COMMAND_CONTRACT,
        PATH_ANGLE_CONTRACT,
    )
}


def speed_command_identity() -> str:
    """The speed-command contract's constants, spelled for a checkpoint's target contract.

    The loop constant and the box are constants of this module, not config fields, so
    nothing else a checkpoint stores would change if one moved: a speed-command checkpoint
    trained under other constants would load and fly under these. Spelled into the target
    contract (``outputs.control.strategy``), a moved constant is refused at load instead."""
    contract = SPEED_COMMAND_CONTRACT
    return (
        f"speed-command(tau-v={SPEED_LOOP_TIME_CONSTANT_S:g}s,"
        f"box={contract.lower[0]:g}..{contract.upper[0]:g}m/s,"
        f"neutral={contract.neutral[0]:g})-v1"
    )


def path_angle_identity() -> str:
    """The path-angle contract's constants, spelled for a checkpoint's target contract.

    Same reason as :func:`speed_command_identity`: the loop constant, the box and the neutral
    are constants of this module rather than config fields, so a checkpoint trained under other
    values would otherwise load and fly under these without a word."""
    contract = PATH_ANGLE_CONTRACT
    return (
        f"specific-force+path-angle(tau-gamma={PATH_ANGLE_TIME_CONSTANT_S:g}s,"
        f"box={math.degrees(contract.lower[2]):g}..{math.degrees(contract.upper[2]):g}deg,"
        f"neutral={math.degrees(contract.neutral[2]):g}deg)-v1"
    )


def control_contract(parameterization: str) -> ControlContract:
    """The contract for a ``control_thrust_parameterization`` value."""
    try:
        return _CONTRACTS[parameterization]
    except KeyError:
        raise ValueError(
            f"no control contract for {parameterization!r}; known: {sorted(_CONTRACTS)}"
        ) from None


# The thrust-fraction contract's arrays under their original names: what the plan path's
# guidance, the speed floor and every stored run read.
CONTROL_LOWER = THRUST_FRACTION_CONTRACT.lower_array
CONTROL_UPPER = THRUST_FRACTION_CONTRACT.upper_array
CONTROL_HALF_WIDTH = THRUST_FRACTION_CONTRACT.half_width


def _rescaled_thrust(controls, max_thrust_n, *, to_newtons: bool):
    """Replace the thrust column, leaving bank and load factor untouched."""
    if controls.shape[-1] != len(CONTROL_NAMES):
        raise ValueError(
            f"controls must end in {len(CONTROL_NAMES)} values, got {controls.shape}"
        )
    scale = max_thrust_n.reshape(
        max_thrust_n.shape + (1,) * (controls.ndim - max_thrust_n.ndim)
    )
    thrust = controls[..., :1] * scale if to_newtons else controls[..., :1] / scale
    rest = controls[..., 1:]
    if isinstance(controls, np.ndarray):
        return np.concatenate((thrust, rest), axis=-1)
    import torch

    return torch.cat((thrust, rest), dim=-1)


def physical_controls(controls, max_thrust_n):
    """Convert ``(thrust_fraction, bank, load)`` to the newton contract the RHS takes.

    Works for torch tensors and numpy arrays alike. ``controls`` is ``[...,3]`` and
    ``max_thrust_n`` is ``[B]`` (or a scalar), broadcast over the leading axes.
    """
    return _rescaled_thrust(controls, max_thrust_n, to_newtons=True)


def fraction_controls(controls_n, max_thrust_n):
    """Inverse of :func:`physical_controls`; the newton contract back to the unit box."""
    return _rescaled_thrust(controls_n, max_thrust_n, to_newtons=False)
