"""The unified aircraft model.

``Aircraft`` is one structured, readable model used everywhere — geometry / mass /
engine / approach / drag, mirroring the OpenAP parameter structure
(``query_aircraft_parameters``) and adding the hand-tuned **approach** group OpenAP lacks.
It replaces the old flat ``AircraftSpec``.

Access is nested and explicit, e.g. ``aircraft.geometry.wing_area_m2``,
``aircraft.mass.max_takeoff_kg``, ``aircraft.engine.max_thrust_total_n``,
``aircraft.approach.reference_speed_ms``.
"""

from __future__ import annotations

from dataclasses import dataclass

from geokit import kt_to_ms, nm_to_m


# ── Nested groups ─────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Geometry:
    wing_area_m2: float
    wing_span_m: float | None = None
    wing_mean_chord_m: float | None = None      # OpenAP wing_mac_m
    wing_sweep_deg: float | None = None
    fuselage_length_m: float | None = None
    fuselage_width_m: float | None = None
    fuselage_height_m: float | None = None


@dataclass(frozen=True, slots=True)
class Mass:
    max_takeoff_kg: float                        # OpenAP mtow_kg
    max_landing_kg: float | None = None          # OpenAP mlw_kg
    operating_empty_kg: float | None = None      # OpenAP oew_kg
    max_fuel_kg: float | None = None             # OpenAP maximum_fuel_capacity_kg


@dataclass(frozen=True, slots=True)
class Engine:
    count: int                                   # OpenAP number
    max_thrust_n_each: float
    model: str | None = None                     # OpenAP default/type
    cruise_thrust_n_each: float | None = None
    cruise_sfc: float | None = None

    @property
    def max_thrust_total_n(self) -> float:
        """Total installed max thrust across all engines."""
        return self.max_thrust_n_each * self.count


@dataclass(frozen=True, slots=True)
class Drag:
    zero_lift_cd0: float | None = None           # OpenAP cd0
    induced_drag_factor: float | None = None     # OpenAP k
    oswald_efficiency: float | None = None       # OpenAP e
    landing_gear_drag_increment: float | None = None


@dataclass(frozen=True, slots=True)
class Approach:
    """The hand-tuned approach/landing envelope OpenAP does not provide."""

    reference_speed_kt: float                    # Vref (old terminal_speed_kt)
    min_speed_kt: float                          # old terminal_speed_min_kt
    max_speed_kt: float                          # old terminal_speed_max_kt
    final_segment_min_nm: float                  # old final_approach_min_nm
    final_segment_max_nm: float                  # old final_approach_max_nm
    protection_half_width_nm: float              # old final_approach_lateral_half_width_nm
    glide_angle_deg: float                       # old final_approach_glide_angle_deg
    threshold_crossing_height_m: float
    thrust_guess_n: float                        # old approach_thrust_guess_n

    # SI mirrors (derived; geokit is the single conversion source).
    @property
    def reference_speed_ms(self) -> float:
        return kt_to_ms(self.reference_speed_kt)

    @property
    def min_speed_ms(self) -> float:
        return kt_to_ms(self.min_speed_kt)

    @property
    def max_speed_ms(self) -> float:
        return kt_to_ms(self.max_speed_kt)

    @property
    def final_segment_min_m(self) -> float:
        return nm_to_m(self.final_segment_min_nm)

    @property
    def final_segment_max_m(self) -> float:
        return nm_to_m(self.final_segment_max_nm)

    @property
    def protection_half_width_m(self) -> float:
        return nm_to_m(self.protection_half_width_nm)


@dataclass(frozen=True, slots=True)
class Aircraft:
    code: str
    name: str
    category: str
    geometry: Geometry
    mass: Mass
    engine: Engine
    approach: Approach
    drag: Drag | None = None


# ── Presets (hand-tuned; values match the former AircraftSpec) ──

A320 = Aircraft(
    code="A320",
    name="Airbus A320-200",
    category="narrow_body",
    geometry=Geometry(wing_area_m2=122.6),
    mass=Mass(max_takeoff_kg=78000.0),
    engine=Engine(count=2, max_thrust_n_each=120000.0),   # 240000 N total
    approach=Approach(
        reference_speed_kt=145.0,
        min_speed_kt=135.0,
        max_speed_kt=155.0,
        final_segment_min_nm=5.0,
        final_segment_max_nm=10.0,
        protection_half_width_nm=0.8,
        glide_angle_deg=3.0,
        threshold_crossing_height_m=15.0,
        thrust_guess_n=40000.0,
    ),
)

B77W = Aircraft(
    code="B77W",
    name="Boeing 777-300ER",
    category="wide_body",
    geometry=Geometry(wing_area_m2=436.8),
    mass=Mass(max_takeoff_kg=351530.0),
    engine=Engine(count=2, max_thrust_n_each=513000.0),   # 1026000 N total
    approach=Approach(
        reference_speed_kt=155.0,
        min_speed_kt=145.0,
        max_speed_kt=165.0,
        final_segment_min_nm=6.0,
        final_segment_max_nm=12.0,
        protection_half_width_nm=1.0,
        glide_angle_deg=3.0,
        threshold_crossing_height_m=15.0,
        thrust_guess_n=140000.0,
    ),
)

C172 = Aircraft(
    code="C172",
    name="Cessna 172",
    category="general_aviation",
    geometry=Geometry(wing_area_m2=16.2),
    mass=Mass(max_takeoff_kg=1157.0),
    engine=Engine(count=1, max_thrust_n_each=3200.0),
    approach=Approach(
        reference_speed_kt=65.0,
        min_speed_kt=60.0,
        max_speed_kt=75.0,
        final_segment_min_nm=2.0,
        final_segment_max_nm=5.0,
        protection_half_width_nm=0.5,
        glide_angle_deg=3.0,
        threshold_crossing_height_m=15.0,
        thrust_guess_n=800.0,
    ),
)

AIRCRAFT_PRESETS = {
    aircraft.code: aircraft
    for aircraft in [A320, B77W, C172]
}
