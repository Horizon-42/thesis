"""The bounded control box the learned head predicts in, and its physical conversion.

Three numbers with three different natural magnitudes — thrust in hundreds of kilonewtons,
bank in radians, load factor around one — cannot share a sigmoid, a regularizer or a
teacher-imitation MSE without one of them dominating. So the CONTRACT this package
predicts in is dimensionless:

    thrust_fraction   delta_T = T / T_max,  where T_max is the flight's installed thrust
    bank_rad          unchanged; a quarter turn is already order one
    load_factor       unchanged; it is a ratio by definition

The same box then means the same thing on every airframe: before this, one sigmoid output
meant 100 kN on a small jet and 400 kN on a heavy, so a teacher schedule or a learned bias
was not transferable across the fleet. Conversion to newtons happens in exactly one place,
:func:`physical_controls`, on the way into the dynamics; records exported for evaluation go
back through it, because the evaluation record contract is newtons and is shared with the
CasADi optimizer.

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

import numpy as np


CONTROL_NAMES = ("thrust_fraction", "bank_rad", "load_factor")

MIN_THRUST_FRACTION = -0.2
MAX_THRUST_FRACTION = 1.0
# The optimizer's general bound, and the value flyability judges against.
MAX_BANK_RAD = math.pi / 4.0
# Meeting 2026-08-17: 0.2-2 g is the reasonable band for transport arrivals; negative and
# above-2 g are both very rare. Measured on real inverted KSJC tracks the required load
# factor spans only [0.92, 1.18], so this box never binds on observed data — it binds on
# what the LEARNED head may emit, which is the point.
MIN_LOAD_FACTOR = 0.2
MAX_LOAD_FACTOR = 2.0

CONTROL_LOWER = np.array(
    [MIN_THRUST_FRACTION, -MAX_BANK_RAD, MIN_LOAD_FACTOR], dtype=np.float64
)
CONTROL_UPPER = np.array(
    [MAX_THRUST_FRACTION, MAX_BANK_RAD, MAX_LOAD_FACTOR], dtype=np.float64
)

# Half the box width per channel. An imitation MSE that compares a predicted schedule with
# an inverted one divides by these, so a full-scale error costs the same in each channel
# instead of thrust (span 1.2) drowning bank (span 1.57) and load factor (span 1.8).
CONTROL_HALF_WIDTH = (CONTROL_UPPER - CONTROL_LOWER) / 2.0


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
