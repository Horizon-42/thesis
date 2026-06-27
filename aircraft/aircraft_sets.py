from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AircraftSpec:
    code: str
    name: str
    category: str
    wing_area_m2: float
    mass_kg: float
    max_thrust_n: float
    approach_thrust_guess_n: float
    terminal_speed_kt: float
    terminal_speed_min_kt: float
    terminal_speed_max_kt: float
    final_approach_min_nm: float
    final_approach_max_nm: float
    final_approach_lateral_half_width_nm: float
    final_approach_glide_angle_deg: float
    threshold_crossing_height_m: float

A320 = AircraftSpec(
    code="A320",
    name="Airbus A320-200",
    category="narrow_body",
    wing_area_m2=122.6,
    mass_kg=78000.0,
    max_thrust_n=240000.0,
    approach_thrust_guess_n=40000.0,
    terminal_speed_kt=145.0,
    terminal_speed_min_kt=135.0,
    terminal_speed_max_kt=155.0,
    final_approach_min_nm=5.0,
    final_approach_max_nm=10.0,
    final_approach_lateral_half_width_nm=0.8,
    final_approach_glide_angle_deg=3.0,
    threshold_crossing_height_m=15.0,
)

B77W = AircraftSpec(
    code="B77W",
    name="Boeing 777-300ER",
    category="wide_body",
    wing_area_m2=436.8,
    mass_kg=351530.0,
    max_thrust_n=1026000.0,
    approach_thrust_guess_n=140000.0,
    terminal_speed_kt=155.0,
    terminal_speed_min_kt=145.0,
    terminal_speed_max_kt=165.0,
    final_approach_min_nm=6.0,
    final_approach_max_nm=12.0,
    final_approach_lateral_half_width_nm=1.0,
    final_approach_glide_angle_deg=3.0,
    threshold_crossing_height_m=15.0,
)

C172 = AircraftSpec(
    code="C172",
    name="Cessna 172",
    category="general_aviation",
    wing_area_m2=16.2,
    mass_kg=1157.0,
    max_thrust_n=3200.0,
    approach_thrust_guess_n=800.0,
    terminal_speed_kt=65.0,
    terminal_speed_min_kt=60.0,
    terminal_speed_max_kt=75.0,
    final_approach_min_nm=2.0,
    final_approach_max_nm=5.0,
    final_approach_lateral_half_width_nm=0.5,
    final_approach_glide_angle_deg=3.0,
    threshold_crossing_height_m=15.0,
)

AIRCRAFT_PRESETS = {
    aircraft.code: aircraft
    for aircraft in [A320, B77W, C172]
}
