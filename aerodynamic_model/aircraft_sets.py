from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AircraftSpec:
    code: str
    name: str
    category: str
    wing_area_m2: float
    mass_kg: float

A320 = AircraftSpec(
    code="A320",
    name="Airbus A320-200",
    category="narrow_body",
    wing_area_m2=122.6,
    mass_kg=78000.0,
)

B77W = AircraftSpec(
    code="B77W",
    name="Boeing 777-300ER",
    category="wide_body",
    wing_area_m2=436.8,
    mass_kg=351530.0,
)

C172 = AircraftSpec(
    code="C172",
    name="Cessna 172",
    category="general_aviation",
    wing_area_m2=16.2,
    mass_kg=1157.0,
)

AIRCRAFT_PRESETS = {
    aircraft.code: aircraft
    for aircraft in [A320, B77W, C172]
}