"""The per-flight conditioning vector: what the control head is told beyond the track.

Control-output conditioning is dimensionless and deliberately small. The model still sees
the full observed state history; these values supply only what ADS-B cannot identify —
aircraft mass, installed thrust, and the aerodynamic model the rollout will use.

The names and the scalings live together because they ARE one contract: ``mass_100t`` means
"divided by 100 t" and nothing enforced that when the tuple sat in ``dataset`` and the
divisors sat in ``dynamics_arrays`` fifty lines apart. Renaming a channel without changing
its divisor, or vice versa, was a silent mislabel; now both come from one feature set.

**Two feature sets, one per value of ``control_condition_features``** (design N4,
``docs/2026-09-14_specific_force_control_design.md`` §11):

- ``raw`` scales each quantity on its own. It is what every run before 2026-09-15 read.
- ``ratios`` replaces the installed thrust and the wing area by the groups the rollout
  actually depends on: the thrust-to-weight ratio ``T_max/(m g)`` and the 1-g sea-level
  stall speed ``√(2 m g/(ρ₀ S Cl_max))``, which carries the wing loading. The point-mass
  equations read the airframe only through ``T_max/W``, ``W/S`` and the polar, so under
  ``raw`` the head has to learn the ratio of two of its inputs to know how hard a thrust
  fraction pushes this aircraft.

The mass stays in both. It carries what the rollout cannot see (wake category, the speeds
controllers give heavies), and with it the two sets are the same information: given
``Cl_max``, ``(m, T_max/W, V_s)`` and ``(m, T_max, S)`` determine each other. They also have
the same width (asserted below). An arm that moves only this field therefore starts from the
same weights as its twin and differs only in how the airframe is written.

The head declares what it needs and the data plane fills it: this module is shared under
``outputs/`` (the control strategy and the plan guidance both build the row), and imports
nothing from a path.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from aircraft.aero_params import GRAVITY_M_S2, stall_speed_ms
from ts_transformer.config import (
    CONTROL_CONDITION_FEATURES,
    CONTROL_CONDITION_FEATURES_RATIOS,
    CONTROL_CONDITION_FEATURES_RAW,
)

Channel = tuple[str, Callable[[float, float, object], float]]

# (name, value) pairs; the name states the unit the divisor produces.
_MASS: Channel = ("mass_100t", lambda mass_kg, _thrust, _aero: mass_kg / 100_000.0)
_POLAR: tuple[Channel, ...] = (
    ("cl_max_3", lambda _mass, _thrust, aero: aero.Cl_max / 3.0),
    ("cd0_0p1", lambda _mass, _thrust, aero: aero.Cd0 / 0.1),
    ("induced_k_0p1", lambda _mass, _thrust, aero: aero.k / 0.1),
    ("stall_threshold", lambda _mass, _thrust, aero: aero.stall_threshold),
    ("stall_k_0p2", lambda _mass, _thrust, aero: aero.k_stall / 0.2),
)

CONDITION_FEATURE_SETS: dict[str, tuple[Channel, ...]] = {
    CONTROL_CONDITION_FEATURES_RAW: (
        _MASS,
        ("max_thrust_1MN", lambda _mass, thrust_n, _aero: thrust_n / 1_000_000.0),
        ("wing_area_500m2", lambda _mass, _thrust, aero: aero.S / 500.0),
        *_POLAR,
    ),
    CONTROL_CONDITION_FEATURES_RATIOS: (
        _MASS,
        # Already O(1): 0.31–0.42 over the cohort's types at landing mass.
        ("thrust_to_weight", lambda mass_kg, thrust_n, _aero: thrust_n / (mass_kg * GRAVITY_M_S2)),
        # The repository's one stall-speed definition (the optimizer's floor and evaluation's
        # gate call it too; `outputs.control.strategy.airborne_control_candidates` still
        # restates it — docs/code-health-followups.md §35).
        ("stall_speed_100mps", lambda mass_kg, _thrust, aero: stall_speed_ms(
            mass_kg, wing_area_m2=aero.S, cl_max=aero.Cl_max) / 100.0),
        *_POLAR,
    ),
}
if set(CONDITION_FEATURE_SETS) != set(CONTROL_CONDITION_FEATURES):
    raise RuntimeError("every control_condition_features value needs a feature set")

#: The width of every feature set: the head's condition encoder is one shape under all of
#: them, which is what lets an arm that moves only the set start from its twin's weights.
CONDITION_WIDTH = len(CONDITION_FEATURE_SETS[CONTROL_CONDITION_FEATURES_RAW])
if {len(channels) for channels in CONDITION_FEATURE_SETS.values()} != {CONDITION_WIDTH}:
    raise RuntimeError("every conditioning feature set must have the same width")


def condition_names(features: str) -> tuple[str, ...]:
    """The channel names of feature set ``features``, in vector order."""
    return tuple(name for name, _ in CONDITION_FEATURE_SETS[features])


def condition_vector(mass_kg: float, max_thrust_n: float, aero, *, features: str) -> np.ndarray:
    """Return the ``[CONDITION_WIDTH]`` conditioning vector of feature set ``features``.

    ``features`` is REQUIRED: the sets have one width, so a call site that forgot it would
    hand a head trained on one set the other's numbers without any shape error.
    """
    return np.array(
        [scale(mass_kg, max_thrust_n, aero) for _name, scale in CONDITION_FEATURE_SETS[features]],
        dtype=np.float32,
    )
