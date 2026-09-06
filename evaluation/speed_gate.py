"""Published-V_ref threshold-crossing speed gate (policy + per-record bounds).

The lateral and vertical gates ask WHERE the crossing was; this gate asks whether the
aircraft crossed the threshold at a speed its TYPE lands at. The window is anchored on
the approach speed the type publishes (``aircraft/reference_speeds.json``: the FAA
Aircraft Characteristics Database's FSB approach speed at the Maximum Allowable
Landing Weight, with the dual flap-configuration values where the FAA gives them;
provenance in ``docs/reference_speeds/README.md``), scaled to the mass in question by
the square-root law and lifted on the lower edge by the crossing load factor:

    V_ref,lo(m) = V_min · sqrt(m / MALW)      # lowest published flap-configuration value
    V_ref,hi(m) = V_max · sqrt(m / MALW)      # highest
    computed record (crossing mass m known):
        [V_ref,lo(m)     · sqrt(max(n, 1)),   V_ref,hi(m)    + 20 kt]
    observed record (mass unknown -- the type's published mass range):
        [V_ref,lo(m_min) · sqrt(max(n, 1)),   V_ref,hi(MALW) + 20 kt]

Every bound is a determinate number from a cited document plus one physical law; the
verdict is pass/fail against it, and indeterminate only when there is nothing to judge
with (no type on the record, no published entry for the type, no minimum mass for an
observed row, no crossing speed). Design, sources and the measured results are in
``docs/THRESHOLD_SPEED_GATE.md``.

Policy lives HERE (the additive, the ``n >= 1`` clamp, which record key names the
type); the aircraft FACTS come from the published table, the mass and load factor from
the record (``evaluation.arrival``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from aircraft.reference_speeds import ReferenceSpeed
from geokit import kt_to_ms

# The criterion id serialized next to every speed bound, mirroring how the lateral
# criterion is named: what anchors the lower edge, what anchors the upper.
SPEED_CRITERION_ID = "published_vref_at_crossing_mass_and_n_to_vref_plus_20kt"
# Observed rows are judged over the type's published mass range (their mass is not
# measured); the suffix says which measured quantity was judged.
OBSERVED_SPEED_CRITERION_STEM = "published_vref_over_type_mass_range_and_n_to_vref_plus_20kt"
OBSERVED_SPEED_CRITERION_ID = OBSERVED_SPEED_CRITERION_STEM + "_ground_speed_proxy"
OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID = OBSERVED_SPEED_CRITERION_STEM + "_metar_airspeed_estimate"

# Stabilized-approach speed element: "not more than V_REF + 20 knots indicated
# airspeed and not less than V_REF" (FSF ALAR Briefing Note 7.1, Table 1, element 3).
# An energy / overrun criterion, defined at 1 g: it does not move with the load factor.
SPEED_GATE_UPPER_ADDITIVE_MS = kt_to_ms(20.0)

# The load factor the LOWER bound is evaluated at never drops below 1: a push-over at
# the threshold precedes a flare that needs n >= 1, so the 1-g floor stays binding
# (docs/THRESHOLD_SPEED_GATE.md section 3.5). The measured n is still reported.
MIN_BOUND_LOAD_FACTOR = 1.0

# Which mass framed the window (``SpeedGateBounds.mass_basis``).
MASS_BASIS_CROSSING = "crossing_mass"
MASS_BASIS_TYPE_RANGE = "type_mass_range"

# The record's ICAO type designator, by producer: ``flight_scenarios.build`` writes the
# type the model FLEW as ``dynamics_typecode`` (optimizer solves, ts predictions --
# the aircraft whose mass the states carry); the harvest's observed writer
# (``trajectory_data_process/harvest/observed.py``) writes the resolved airframe's
# identity typecode as ``aircraft_type``. A record carries one or the other.
TYPECODE_KEYS = ("dynamics_typecode", "aircraft_type")

# ``ArrivalDeviation.crossing_load_factor_source`` values: the control active over the
# final rollout step (records that carry controls); the inversion of the flight's own
# ADS-B kinematics over the final window before the crossing (observed baselines,
# ``evaluation.arrival._observed_load_factor``); or the declared 1-g assumption
# (state-output predictions, and observed tracks whose window is too short).
LOAD_FACTOR_FROM_CONTROLS = "controls_last_step"
LOAD_FACTOR_FROM_ADSB = "adsb_kinematics"
LOAD_FACTOR_ASSUMED_1G = "assumed_1g"

# Observed subjects: the fitted crossing GROUND speed corrected by the field's METAR
# headwind into an airspeed estimate when a usable report exists, else the raw ground
# speed as a STATED proxy. The window spans the type's published mass range because
# an ADS-B track carries no mass -- the honest window for a flight of unknown weight,
# and one every correctly flown landing of the type sits inside.
OBSERVED_SPEED_POLICY = (
    "observed records are speed-graded on the fitted crossing ground speed CORRECTED "
    "by the field's METAR headwind into an airspeed estimate (criterion "
    "..._metar_airspeed_estimate) when a report within 30 min with a non-variable "
    "direction exists, and otherwise on the raw ground speed as a stated proxy "
    "(..._ground_speed_proxy, wind unmodelled) -- each row's bounds.speed_criterion "
    "and wind block say which; the window is the type's PUBLISHED approach-speed "
    "window over its published mass range [minimum operating mass, MALW] (the "
    "flight's own mass is not measured), at the load factor inverted from the "
    "flight's own ADS-B kinematics over the final 20 s before the crossing (a "
    "declared 1 g when that window holds too few samples)"
)
NO_TYPECODE_REASON = (
    "record names no aircraft type (source.dynamics_typecode / source.aircraft_type); "
    "no published speed window exists"
)
OBSERVED_UNRESOLVED_AIRFRAME_REASON = (
    "airframe could not be resolved from icao24, so no published speed window exists; "
    "crossing ground speed is not judged"
)
NO_REFERENCE_SPEED_REASON = (
    "no published approach speed for type {typecode} in aircraft/reference_speeds.json; "
    "crossing speed is not judged"
)
NO_MIN_MASS_REASON = (
    "type {typecode} publishes no minimum operating mass, so the observed window's "
    "lower edge is undefined; crossing speed is not judged"
)
OBSERVED_NO_CROSSING_SPEED_REASON = (
    "the threshold event fitted no crossing ground speed; nothing to judge"
)
# Rows with no measured crossing at all (unsolved, ended short of the plane, event
# unavailable): speed is indeterminate for the same reason lateral and vertical are.
NO_CROSSING_REASON = (
    "no threshold crossing was measured (unsolved, not reached, or event "
    "unavailable); nothing to judge"
)


@dataclass(frozen=True)
class SpeedGateBounds:
    """The per-record window, plus the published speeds that anchored it."""

    reference_typecode: str
    # Source ids (keys of the table's ``sources`` block) of the three published facts
    # the window came from -- each row traces to its documents without the table.
    reference_sources: dict[str, str | None]
    mass_basis: str            # MASS_BASIS_CROSSING or MASS_BASIS_TYPE_RANGE
    vref_low_ms: float         # V_ref,lo at the basis's lower mass, 1 g
    vref_high_ms: float        # V_ref,hi at the basis's upper mass
    lower_ms: float            # vref_low at max(n, 1)
    upper_ms: float            # vref_high + the additive

    @property
    def empty(self) -> bool:
        """No speed satisfies the window: the crossing load factor lifted V_ref past the
        1-g energy limit. A verdict, not an error -- the manoeuvre itself has no
        stabilized-approach speed -- reported by both bounds on the row and failed by
        ``metrics._component``."""
        return self.lower_ms > self.upper_ms

    def to_dict(self, criterion: str) -> dict[str, Any]:
        return {
            "speed_criterion": criterion,
            "reference_typecode": self.reference_typecode,
            "reference_sources": dict(self.reference_sources),
            "mass_basis": self.mass_basis,
            "vref_low_ms": self.vref_low_ms,
            "vref_high_ms": self.vref_high_ms,
            "speed_lower_ms": self.lower_ms,
            "speed_upper_ms": self.upper_ms,
        }


def record_typecode(source: Mapping[str, Any]) -> str | None:
    """The ICAO type designator the record names (``TYPECODE_KEYS``), or None."""
    for key in TYPECODE_KEYS:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return None


def speed_gate_bounds(
    reference: ReferenceSpeed,
    *,
    load_factor: float,
    crossing_mass_kg: float | None,
) -> SpeedGateBounds:
    """Resolve one record's speed window from the type's published speeds.

    ``crossing_mass_kg`` frames a computed record's window at its own crossing mass;
    None frames an observed record's window over the type's published mass range,
    which needs the type's minimum mass (a table without one is the caller's
    indeterminate case, not this function's). A non-finite or non-positive load
    factor raises: the record said something impossible.
    """
    if not math.isfinite(load_factor) or load_factor <= 0.0:
        raise ValueError(
            f"crossing load factor must be a positive finite number, got {load_factor!r}"
        )
    if crossing_mass_kg is None:
        if reference.min_mass_kg is None:
            raise ValueError(f"{reference.typecode}: no minimum mass for a type-range window")
        basis = MASS_BASIS_TYPE_RANGE
        low_mass, high_mass = reference.min_mass_kg, reference.malw_kg
    else:
        basis = MASS_BASIS_CROSSING
        low_mass = high_mass = crossing_mass_kg
    vref_low = kt_to_ms(reference.vref_kt(low_mass, edge="low"))
    vref_high = kt_to_ms(reference.vref_kt(high_mass, edge="high"))
    return SpeedGateBounds(
        reference_typecode=reference.typecode,
        reference_sources={
            "approach_speed": reference.approach_speed_source,
            "malw": reference.malw_source,
            "min_mass": reference.min_mass_source,
        },
        mass_basis=basis,
        vref_low_ms=vref_low,
        vref_high_ms=vref_high,
        lower_ms=vref_low * math.sqrt(max(load_factor, MIN_BOUND_LOAD_FACTOR)),
        upper_ms=vref_high + SPEED_GATE_UPPER_ADDITIVE_MS,
    )
