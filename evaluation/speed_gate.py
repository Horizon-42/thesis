"""Stall-anchored threshold-crossing speed gate (policy + per-record bounds).

The lateral and vertical gates ask WHERE the crossing was; this gate asks how much
ENERGY the aircraft carried across the threshold. The window is anchored on the
project's own 1-g stall model at the record's crossing mass:

    V_s     = sqrt(2 m g / (rho0 S Cl_max_landing))     # aircraft.aero_params, ONE source
    V_ref   = 1.23 x V_s                                # 14 CFR 25.125(b)(2)(i)
    window  = [V_ref, V_ref + 20 kt]  (inclusive)       # FSF ALAR Briefing Note 7.1

Design, sources, worked numbers and the observed-subject exclusion are documented in
``docs/THRESHOLD_SPEED_GATE.md`` (this package's docs directory).

Policy lives HERE (the multiplier and the additive); the aircraft FACTS (wing area,
landing Cl_max) come from the record's producer-written ``source.landing_aero`` block —
the same supplied-then-checked pattern as ``hae_minus_msl_m``. A computed record
without the block is gradable on geometry but not on speed, and says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from aircraft.aero_params import stall_speed_ms
from geokit import kt_to_ms

# The criterion id serialized next to every speed bound, mirroring how the lateral
# criterion is named. "vs1g" says which stall speed anchors it (the model's 1-g level
# stall, the analogue of the Part 25 reference stall speed V_SR).
SPEED_CRITERION_ID = "vref_1p23_vs1g_to_vref_plus_20kt"

# V_REF may not be less than 1.23 V_SR0 (14 CFR 25.125(b)(2)(i); EASA CS-25.125 is
# identical). The model's V_s is a 1-g stall speed, which is what V_SR references
# (14 CFR 25.103), so the multiplier applies to it directly.
VREF_STALL_MULTIPLIER = 1.23

# Stabilized-approach speed element: "not more than V_REF + 20 knots indicated
# airspeed and not less than V_REF" (FSF ALAR Briefing Note 7.1, Table 1, element 3).
SPEED_GATE_UPPER_ADDITIVE_MS = kt_to_ms(20.0)

# Producer-written aircraft facts on the record (flight_scenarios.build_scenario).
LANDING_AERO_KEY = "landing_aero"

# Why observed subjects are never speed-graded (serialized into the methodology and
# used as the per-row result reason): the record's V is ground-referenced (wind is not
# modelled), and ADS-B coverage ends a median ~325 m before the threshold, so no
# observed crossing speed was ever measured.
OBSERVED_SPEED_POLICY = (
    "observed records are not speed-graded: track V is ground speed (wind unmodelled) "
    "and coverage ends before the threshold, so no crossing airspeed was measured"
)
MISSING_LANDING_AERO_REASON = (
    "record carries no source.landing_aero block; crossing speed cannot be judged "
    "against a stall-anchored window"
)


@dataclass(frozen=True)
class SpeedGateBounds:
    """The per-record window, plus the stall speed that anchored it."""

    stall_speed_ms: float
    lower_ms: float
    upper_ms: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "speed_criterion": SPEED_CRITERION_ID,
            "stall_speed_ms": self.stall_speed_ms,
            "speed_lower_ms": self.lower_ms,
            "speed_upper_ms": self.upper_ms,
        }


def speed_gate_bounds(
    crossing_mass_kg: float, landing_aero: Mapping[str, Any]
) -> SpeedGateBounds:
    """Resolve one record's speed window from its own crossing mass + aircraft facts.

    A PRESENT but malformed block raises: unlike an absent block (a record predating
    the contract, honestly indeterminate), a broken one means the producer wrote
    something and it cannot be trusted — same absent-vs-invalid split as the observed
    threshold event.
    """
    if not isinstance(landing_aero, Mapping):
        raise ValueError(
            f"source.landing_aero must be an object, got {landing_aero!r}"
        )
    values: dict[str, float] = {}
    for key in ("wing_area_m2", "cl_max_landing"):
        value = landing_aero.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(
                f"source.landing_aero.{key} must be a positive finite number, "
                f"got {value!r}"
            )
        values[key] = float(value)
    stall_ms = stall_speed_ms(
        crossing_mass_kg,
        wing_area_m2=values["wing_area_m2"],
        cl_max=values["cl_max_landing"],
    )
    lower = VREF_STALL_MULTIPLIER * stall_ms
    return SpeedGateBounds(
        stall_speed_ms=stall_ms,
        lower_ms=lower,
        upper_ms=lower + SPEED_GATE_UPPER_ADDITIVE_MS,
    )
