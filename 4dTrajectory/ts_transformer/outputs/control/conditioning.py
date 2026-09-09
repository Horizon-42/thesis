"""The per-flight conditioning vector: what the control head is told beyond the track.

Control-output conditioning is dimensionless and deliberately small. The model still sees
the full observed state history; these values supply only what ADS-B cannot identify —
aircraft mass, installed thrust, and the aerodynamic model the rollout will use.

The names and the scalings live together because they ARE one contract: ``mass_100t`` means
"divided by 100 t" and nothing enforced that when the tuple sat in ``dataset`` and the
divisors sat in ``dynamics_arrays`` fifty lines apart. Renaming a channel without changing
its divisor, or vice versa, was a silent mislabel; now both come from ``CONDITION_CHANNELS``.

It lives under ``control`` and is imported DOWNWARD by ``dataset`` because the head declares
what it needs and the data plane fills it — the reverse (``control`` importing ``dataset``)
made the package reach up into the data plane for its own input width.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# (name, value) pairs; the name states the unit the divisor produces.
CONDITION_CHANNELS: tuple[tuple[str, Callable[[float, float, object], float]], ...] = (
    ("mass_100t", lambda mass_kg, _thrust, _aero: mass_kg / 100_000.0),
    ("max_thrust_1MN", lambda _mass, thrust_n, _aero: thrust_n / 1_000_000.0),
    ("wing_area_500m2", lambda _mass, _thrust, aero: aero.S / 500.0),
    ("cl_max_3", lambda _mass, _thrust, aero: aero.Cl_max / 3.0),
    ("cd0_0p1", lambda _mass, _thrust, aero: aero.Cd0 / 0.1),
    ("induced_k_0p1", lambda _mass, _thrust, aero: aero.k / 0.1),
    ("stall_threshold", lambda _mass, _thrust, aero: aero.stall_threshold),
    ("stall_k_0p2", lambda _mass, _thrust, aero: aero.k_stall / 0.2),
)

DYNAMICS_CONDITION_NAMES = tuple(name for name, _ in CONDITION_CHANNELS)


def condition_vector(mass_kg: float, max_thrust_n: float, aero) -> np.ndarray:
    """Return the ``[len(DYNAMICS_CONDITION_NAMES)]`` dimensionless conditioning vector."""
    return np.array(
        [scale(mass_kg, max_thrust_n, aero) for _name, scale in CONDITION_CHANNELS],
        dtype=np.float32,
    )
