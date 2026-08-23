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


def aero_params_for_aircraft(aircraft: Aircraft) -> AeroParams:
    """AeroParams for an aircraft with a mass-based maximum lift coefficient.

    The single source of truth for the stall model: the optimiser AND the
    playback (``CasadiSimulator``) must use the SAME ``Cl_max`` or an optimized
    trajectory will not replay consistently.  Heavier types reach a lower
    terminal Cl_max; A320/737-class fly ~2.7 with landing flaps deployed.
    """
    if aircraft.mass.max_takeoff_kg > 100_000.0:
        cl_max = 2.4
    elif aircraft.mass.max_takeoff_kg >= 30_000.0:   # e.g. A320, 737
        cl_max = 2.7
    else:
        cl_max = 2.2
    return AeroParams(S=aircraft.geometry.wing_area_m2, Cl_max=cl_max)


def stall_speed_ms(mass_kg: float, *, wing_area_m2: float, cl_max: float) -> float:
    """1-g level-flight stall speed of the project's stall model (TAS, m/s).

    ``V_s = sqrt(2 m g / (rho0 S Cl_max))`` — the SINGLE definition. The optimizer's
    velocity floor and evaluation's threshold speed gate both anchor on it, so a solve
    admitted by the floor and the gate that judges it share one stall model by
    construction. ``cl_max`` is the LANDING-configuration value
    (:func:`aero_params_for_aircraft`).
    """
    return math.sqrt(
        2.0 * mass_kg * GRAVITY_M_S2 / (RHO0_KG_M3 * wing_area_m2 * cl_max)
    )
