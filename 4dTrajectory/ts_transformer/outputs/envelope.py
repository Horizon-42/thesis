"""The bounded control box the learned head predicts in, and its physical conversion.

Three numbers with three different natural magnitudes — thrust in hundreds of kilonewtons,
bank in radians, load factor around one — cannot share a sigmoid, a regularizer or a
teacher-imitation MSE without one of them dominating. So the CONTRACT this package
predicts in is dimensionless, and since 2026-09-14 there are two of them — one per
``control_thrust_parameterization`` (:func:`control_contract`):

    thrust-fraction   delta_T = T / T_max     normalised by the ACTUATOR (T_max installed)
    specific-force    n_x = (T - D) / W       normalised by the EFFECT (speed-rate in g)
    bank_rad          unchanged; a quarter turn is already order one
    load_factor       unchanged; it is a ratio by definition

``thrust-fraction`` made the box mean the same ACTUATOR setting on every airframe (before
2026-08-18 one sigmoid output meant 100 kN on a small jet and 400 kN on a heavy).
``specific-force`` makes it mean the same MOTION: the lag RHS re-solves the thrust at every
stage and the drag cancels, so the same n_x moves a C550 and a B77W identically
(``docs/2026-09-14_specific_force_control_design.md``). Under thrust-fraction the newton
conversion is :func:`physical_controls`, once, on the way into the dynamics and out to the
record; under specific-force it needs the state's drag and lives in the lag RHS
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

from ts_transformer.config import CONTROL_SPECIFIC_FORCE, CONTROL_THRUST_FRACTION


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
    #: airframe (why not level trim: that constant's comment).
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
_CONTRACTS = {
    contract.parameterization: contract
    for contract in (THRUST_FRACTION_CONTRACT, SPECIFIC_FORCE_CONTRACT)
}


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
