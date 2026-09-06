import math
from dataclasses import dataclass

from .aircraft_sets import Aircraft

# The one gravity/density pair the stall model is defined with. Sea-level ISA density is
# deliberate even though thresholds sit up to ~190 m: V here is TAS and the operational
# reference speeds it is compared against are CAS, and at this fleet's threshold
# elevations TAS/CAS differ by <1% — the consumers that care state that approximation.
GRAVITY_M_S2 = 9.81
RHO0_KG_M3 = 1.225


@dataclass
class AeroParams:
    S: float
    Cl_max: float = 1.5
    Cd0: float = 0.02
    k: float = 0.04
    stall_threshold: float = 0.9
    k_stall: float = 0.1


# Landing-configuration Cl_max for the A320 family, replacing the generic
# narrow-body bucket below. The calibration source is Airbus's own speed logic:
# on final in CONF FULL the FBW lowest selectable speed IS VLS = 1.23·Vs1g, and
# published VLS figures (≈128 kt CAS at 64 t for the A320) invert through this
# module's stall formula to Cl_max ≈ 2.9–3.0 — the slats + fowler-flap high-lift
# system genuinely outperforms the 737's. Measured consequence of the old shared
# 2.7 (evaluation/docs/BASELINE_SPEED_GATE_RESULTS.md §5): the family's
# crossing-speed windows sat ~7 kt high and 45–92 % of its real crossings graded
# "too slow", while the 737 family measured healthy at 2.7 — weather cannot tell
# Airbus from Boeing, so the anchor was the error. 3.0 centres the measured fleet.
_A320_FAMILY = frozenset({"A318", "A319", "A320", "A321", "A19N", "A20N", "A21N"})
_A320_FAMILY_LANDING_CL_MAX = 3.0


def aero_params_for_aircraft(aircraft: Aircraft) -> AeroParams:
    """AeroParams for an aircraft with a landing-configuration maximum lift coefficient.

    The single source of truth for the stall model: the optimiser AND the
    playback (``CasadiSimulator``) must use the SAME ``Cl_max`` or an optimized
    trajectory will not replay consistently. The A320 family carries its own
    calibrated value (above); other types keep the mass-class buckets — heavier
    types reach a lower terminal Cl_max; 737-class flies ~2.7 with landing flaps.
    """
    if aircraft.code in _A320_FAMILY:
        cl_max = _A320_FAMILY_LANDING_CL_MAX
    elif aircraft.mass.max_takeoff_kg > 100_000.0:
        cl_max = 2.4
    elif aircraft.mass.max_takeoff_kg >= 30_000.0:   # e.g. 737, E-jets, CRJs
        cl_max = 2.7
    else:
        cl_max = 2.2
    return AeroParams(S=aircraft.geometry.wing_area_m2, Cl_max=cl_max)


def stall_speed_ms(
    mass_kg: float, *, wing_area_m2: float, cl_max: float, load_factor: float = 1.0
) -> float:
    """Stall speed of the project's stall model at load factor ``n`` (TAS, m/s).

    ``V_s(n) = sqrt(2 n m g / (rho0 S Cl_max))`` — the SINGLE definition: the speed at
    which the lift the manoeuvre demands, ``n m g``, needs the whole ``Cl_max``. The
    default ``n = 1`` is the 1-g level-flight stall speed the optimizer's velocity floor
    anchors on (and the analogue of the regulatory reference stall speed V_SR);
    evaluation's threshold speed gate passes the crossing load factor
    (``evaluation/docs/THRESHOLD_SPEED_GATE.md`` §3.5). Both call this one function, so
    a solve admitted by the floor and the gate that judges it share one stall model by
    construction. ``cl_max`` is the LANDING-configuration value
    (:func:`aero_params_for_aircraft`).
    """
    return math.sqrt(
        2.0 * load_factor * mass_kg * GRAVITY_M_S2
        / (RHO0_KG_M3 * wing_area_m2 * cl_max)
    )
