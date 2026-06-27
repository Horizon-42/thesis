from dataclasses import dataclass

from .aircraft_sets import AircraftSpec


@dataclass
class AeroParams:
    S: float
    Cl_max: float = 1.5
    Cd0: float = 0.02
    k: float = 0.04
    stall_threshold: float = 0.9
    k_stall: float = 0.1


def aero_params_for_aircraft(aircraft: AircraftSpec) -> AeroParams:
    """AeroParams for an aircraft with a mass-based maximum lift coefficient.

    The single source of truth for the stall model: the optimiser AND the
    playback (``CasadiSimulator``) must use the SAME ``Cl_max`` or an optimized
    trajectory will not replay consistently.  Heavier types reach a lower
    terminal Cl_max; A320/737-class fly ~2.7 with landing flaps deployed.
    """
    if aircraft.mass_kg > 100_000.0:
        cl_max = 2.4
    elif aircraft.mass_kg >= 30_000.0:   # e.g. A320, 737
        cl_max = 2.7
    else:
        cl_max = 2.2
    return AeroParams(S=aircraft.wing_area_m2, Cl_max=cl_max)
