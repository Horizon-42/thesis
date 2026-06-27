from dataclasses import dataclass, field

from geokit import kt_to_ms, nm_to_m


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

    # SI mirrors of the aviation-unit fields above, derived in __post_init__ so
    # callers never re-derive kt->m/s or nm->m inline. Not constructor args:
    # they are computed from the kt/nm fields, the existing fields are unchanged.
    terminal_speed_ms: float = field(init=False)
    terminal_speed_min_ms: float = field(init=False)
    terminal_speed_max_ms: float = field(init=False)
    final_approach_min_m: float = field(init=False)
    final_approach_max_m: float = field(init=False)
    final_approach_lateral_half_width_m: float = field(init=False)

    def __post_init__(self) -> None:
        # frozen dataclass: bypass the blocked __setattr__ to set derived fields.
        # Conversions come from geokit (single source) — see the SI-mirror comment above.
        object.__setattr__(self, "terminal_speed_ms", kt_to_ms(self.terminal_speed_kt))
        object.__setattr__(self, "terminal_speed_min_ms", kt_to_ms(self.terminal_speed_min_kt))
        object.__setattr__(self, "terminal_speed_max_ms", kt_to_ms(self.terminal_speed_max_kt))
        object.__setattr__(self, "final_approach_min_m", nm_to_m(self.final_approach_min_nm))
        object.__setattr__(self, "final_approach_max_m", nm_to_m(self.final_approach_max_nm))
        object.__setattr__(
            self, "final_approach_lateral_half_width_m", nm_to_m(self.final_approach_lateral_half_width_nm)
        )

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
