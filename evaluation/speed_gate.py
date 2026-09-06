"""Stall-anchored threshold-crossing speed gate (policy + per-record bounds).

The lateral and vertical gates ask WHERE the crossing was; this gate asks how much
ENERGY the aircraft carried across the threshold, and whether the wing could carry the
lift the crossing manoeuvre demanded. The window is anchored on the project's own
stall model at the record's crossing mass AND crossing load factor:

    V_s1g   = sqrt(2 m g / (rho0 S Cl_max_landing))    # aircraft.aero_params, ONE source
    V_s(n)  = V_s1g * sqrt(max(n, 1))                   # the same model under n·m·g of lift
    lower   = 1.23 x V_s(n)                             # 14 CFR 25.125(b)(2)(i) at the crossing n
    upper   = 1.23 x V_s1g + 20 kt                      # FSF ALAR Briefing Note 7.1, at 1 g
    window  = [lower, upper]  (inclusive)

Design, sources, the measured load-factor distribution and the observed-subject proxy
are documented in ``docs/THRESHOLD_SPEED_GATE.md`` (this package's docs directory).

Policy lives HERE (the multiplier, the additive, the ``n >= 1`` clamp); the aircraft
FACTS (wing area, landing Cl_max) come from the record's producer-written
``source.landing_aero`` block -- the same supplied-then-checked pattern as
``hae_minus_msl_m`` -- and the load factor from the record's own controls
(``evaluation.arrival``). A computed record without the block is gradable on geometry
but not on speed, and says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from aircraft.aero_params import stall_speed_ms
from geokit import kt_to_ms

# The criterion id serialized next to every speed bound, mirroring how the lateral
# criterion is named: the lower edge is V_ref at the crossing load factor ("vs_at_n"),
# the upper edge V_ref at 1 g plus the ALAR additive.
SPEED_CRITERION_ID = "vref_1p23_vs_at_n_to_vref_1g_plus_20kt"

# V_REF may not be less than 1.23 V_SR0 (14 CFR 25.125(b)(2)(i); EASA CS-25.125 is
# identical). The model's V_s is a 1-g stall speed, which is what V_SR references
# (14 CFR 25.103), so the multiplier applies to it directly -- and to the accelerated
# stall speed V_s1g * sqrt(n), which is the same wing at the lift the manoeuvre needs.
VREF_STALL_MULTIPLIER = 1.23

# Stabilized-approach speed element: "not more than V_REF + 20 knots indicated
# airspeed and not less than V_REF" (FSF ALAR Briefing Note 7.1, Table 1, element 3).
# An energy / overrun criterion, defined at 1 g: it does not move with the load factor.
SPEED_GATE_UPPER_ADDITIVE_MS = kt_to_ms(20.0)

# The load factor the LOWER bound is evaluated at never drops below 1: a push-over at
# the threshold precedes a flare that needs n >= 1, so the 1-g floor stays binding
# (docs/THRESHOLD_SPEED_GATE.md section 3.5). The measured n is still reported.
MIN_BOUND_LOAD_FACTOR = 1.0

# Producer-written aircraft facts on the record (flight_scenarios.build_scenario).
LANDING_AERO_KEY = "landing_aero"

# ``ArrivalDeviation.crossing_load_factor_source`` values: the control active over the
# final rollout step (records that carry controls); the inversion of the flight's own
# ADS-B kinematics over the final window before the crossing (observed baselines,
# ``evaluation.arrival._observed_load_factor``); or the declared 1-g assumption
# (state-output predictions, and observed tracks whose window is too short).
LOAD_FACTOR_FROM_CONTROLS = "controls_last_step"
LOAD_FACTOR_FROM_ADSB = "adsb_kinematics"
LOAD_FACTOR_ASSUMED_1G = "assumed_1g"

# Observed subjects are judged on their estimated crossing GROUND speed as a STATED
# PROXY for airspeed (owner decision 2026-08-24, superseding the original exclusion):
# wind is unmodelled, so an ordinary 10 kt headwind is half the 20 kt window and a
# baseline speed fail can be the day's wind rather than the flight. The proxy is
# declared everywhere it appears -- its own criterion id, the methodology block, and
# this string -- never silently equated with the airspeed the computed subjects are
# judged on. The mass anchoring the window is the flight's own resolved airframe's
# landing mass (the same identity->OpenAP chain the scenarios use), so baseline and
# modeled twins share one set of stall assumptions.
OBSERVED_SPEED_POLICY = (
    "observed records are speed-graded on the fitted crossing ground speed CORRECTED "
    "by the field's METAR headwind into an airspeed estimate (criterion "
    "..._metar_airspeed_estimate, +/-5 kt declared) when a report within 30 min with "
    "a non-variable direction exists, and otherwise on the raw ground speed as a "
    "stated proxy (..._ground_speed_proxy, wind unmodelled) -- each row's "
    "bounds.speed_criterion and wind block say which; the window is anchored on the "
    "resolved airframe's landing mass -- the same stall assumptions the flight's "
    "modeled twins fly with -- at the load factor inverted from the flight's own "
    "ADS-B kinematics over the final 20 s before the crossing (a declared 1 g when "
    "that window holds too few samples)"
)
# Distinct criterion id for the proxy, mirroring how the lateral criterion is named:
# a reader of one row can tell WHAT was judged without consulting the subject. The
# window is the same load-factor-anchored one the computed subjects get -- the
# observed n is measured, not assumed -- so the id shares its stem.
OBSERVED_SPEED_CRITERION_ID = SPEED_CRITERION_ID + "_ground_speed_proxy"
# The proxy corrected by the field's METAR headwind component (``evaluation.wind``):
# an airspeed ESTIMATE with a declared uncertainty, judged in the same window.
OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID = SPEED_CRITERION_ID + "_metar_airspeed_estimate"
MISSING_LANDING_AERO_REASON = (
    "record carries no source.landing_aero block; crossing speed cannot be judged "
    "against a stall-anchored window"
)
OBSERVED_UNRESOLVED_AIRFRAME_REASON = (
    "airframe could not be resolved from icao24, so no stall-anchored window exists; "
    "crossing ground speed is not judged"
)
OBSERVED_NO_CROSSING_SPEED_REASON = (
    "the threshold event fitted no crossing ground speed; nothing to judge"
)


@dataclass(frozen=True)
class SpeedGateBounds:
    """The per-record window, plus the stall speeds that anchored it."""

    stall_speed_ms: float          # 1 g -- what the upper edge and the optimizer floor use
    stall_speed_at_n_ms: float     # at max(n, 1) -- what the lower edge uses
    lower_ms: float
    upper_ms: float

    @property
    def empty(self) -> bool:
        """No speed satisfies the window: the crossing load factor lifted V_ref past the
        1-g energy limit (n above ~1.25-1.40 on this fleet). A verdict, not an error --
        the manoeuvre itself has no stabilized-approach speed -- reported by both
        bounds on the row and failed by ``metrics._component``."""
        return self.lower_ms > self.upper_ms

    def to_dict(self, criterion: str) -> dict[str, float | str]:
        return {
            "speed_criterion": criterion,
            "stall_speed_ms": self.stall_speed_ms,
            "stall_speed_at_n_ms": self.stall_speed_at_n_ms,
            "speed_lower_ms": self.lower_ms,
            "speed_upper_ms": self.upper_ms,
        }


def validate_landing_aero(landing_aero: Any) -> dict[str, float]:
    """The stall facts as floats, or a ValueError naming the broken field.

    Called ONCE, at the record boundary (``records.record_from_dict``). A PRESENT but
    malformed block raises: unlike an absent block (a record predating the contract,
    honestly indeterminate), a broken one means the producer wrote something and it
    cannot be trusted -- same absent-vs-invalid split as the observed threshold event.
    """
    if not isinstance(landing_aero, Mapping):
        raise ValueError(
            f"source.{LANDING_AERO_KEY} must be an object, got {landing_aero!r}"
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
                f"source.{LANDING_AERO_KEY}.{key} must be a positive finite number, "
                f"got {value!r}"
            )
        values[key] = float(value)
    return values


def speed_gate_bounds(
    crossing_mass_kg: float,
    landing_aero: Mapping[str, Any],
    *,
    load_factor: float,
) -> SpeedGateBounds:
    """Resolve one record's speed window from its crossing mass, load factor and
    aircraft facts (a block :func:`validate_landing_aero` accepted).

    A non-finite or non-positive load factor raises: the record's controls said
    something impossible, which is a malformed record, not an open question.
    """
    values = {
        key: float(landing_aero[key]) for key in ("wing_area_m2", "cl_max_landing")
    }
    if not math.isfinite(load_factor) or load_factor <= 0.0:
        raise ValueError(
            f"crossing load factor must be a positive finite number, got {load_factor!r}"
        )
    bound_load_factor = max(load_factor, MIN_BOUND_LOAD_FACTOR)
    stall_1g_ms = stall_speed_ms(
        crossing_mass_kg,
        wing_area_m2=values["wing_area_m2"],
        cl_max=values["cl_max_landing"],
    )
    stall_at_n_ms = stall_speed_ms(
        crossing_mass_kg,
        wing_area_m2=values["wing_area_m2"],
        cl_max=values["cl_max_landing"],
        load_factor=bound_load_factor,
    )
    return SpeedGateBounds(
        stall_speed_ms=stall_1g_ms,
        stall_speed_at_n_ms=stall_at_n_ms,
        lower_ms=VREF_STALL_MULTIPLIER * stall_at_n_ms,
        upper_ms=VREF_STALL_MULTIPLIER * stall_1g_ms + SPEED_GATE_UPPER_ADDITIVE_MS,
    )
