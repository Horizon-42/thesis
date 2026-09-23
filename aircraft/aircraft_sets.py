"""The unified aircraft model.

``Aircraft`` is one structured, readable model used everywhere — geometry / mass /
engine / approach / drag, mirroring the OpenAP parameter structure
(``query_aircraft_parameters``) and adding the **approach** group OpenAP lacks. Its speeds
are the airframe's PUBLISHED approach speed (``aircraft/reference_speeds.json``, the FAA Aircraft
Characteristics Database), used at a mass (``approach.reference_speed_ms(m)``); its procedure
geometry and thrust guess are hand-tuned for the presets below and MTOW-class defaults for
OpenAP types. It replaces the old flat ``AircraftSpec``.

Access is nested and explicit, e.g. ``aircraft.geometry.wing_area_m2``,
``aircraft.mass.max_takeoff_kg``, ``aircraft.engine.max_thrust_total_n``,
``aircraft.approach.reference_speed_ms(mass_kg)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from geokit import kt_to_ms, nm_to_m

from aircraft.reference_speeds import ReferenceSpeed, reference_speed

# Used by ``Aircraft.landing_mass`` when no max-landing weight is known: a typical
# landing/take-off weight ratio (e.g. A320 66 t / 78 t ≈ 0.85).
_LANDING_MASS_FRACTION_OF_MTOW = 0.85


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
    """The approach/landing envelope OpenAP does not provide.

    ``speeds`` is the PUBLISHED approach-speed row of the airframe this envelope belongs to
    (``aircraft/reference_speeds.json``: the FAA Aircraft Characteristics Database
    ``Approach_Speed_knot`` at the Maximum Allowable Landing Weight, its dual
    flap-configuration values, and that MALW). The speed is never a class default, and it is
    only meaningful at a mass: :meth:`reference_speed_ms` rescales it by sqrt(m / MALW), the
    same law the threshold speed gate uses (``aircraft.reference_speeds.ReferenceSpeed.vref_kt``).
    The target speed is the upper edge, the speed floor the lower one, so for a record the gate
    judges against THIS row (a preset, a direct OpenAP type, C56X) a target flown at
    ``reference_speed_ms(m)`` is ``evaluation.speed_gate.speed_gate_bounds(...).vref_high_ms`` at
    that crossing mass: inside the gate's window, on its lower edge when the type publishes one
    value (FAA AC 91-79B §5.2.2: V_ref plus wind and gust additives until 50 ft over the
    threshold; the model flies no wind, so no additive). An OpenAP synonym flies its
    surrogate's row while the gate keys on the synonym's own code; the performance index
    (``aircraft/performance_index.json``) replaces synonyms with explicit substitutes.

    One stated approximation: the published speed is an indicated airspeed and the model
    flies it as a true airspeed; at this fleet's threshold elevations they differ by < 1 %.
    The procedure fields below are hand-tuned (presets) or MTOW-class defaults (OpenAP types).
    """

    speeds: ReferenceSpeed                       # the airframe's published approach speeds
    final_segment_min_nm: float                  # old final_approach_min_nm
    final_segment_max_nm: float                  # old final_approach_max_nm
    protection_half_width_nm: float              # old final_approach_lateral_half_width_nm
    glide_angle_deg: float                       # old final_approach_glide_angle_deg
    threshold_crossing_height_m: float
    thrust_guess_n: float                        # old approach_thrust_guess_n

    def reference_speed_ms(self, mass_kg: float) -> float:
        """V_ref at ``mass_kg`` (m/s): the published approach speed scaled by sqrt(m / MALW)."""
        return kt_to_ms(self.speeds.vref_kt(mass_kg, edge="high"))

    def minimum_speed_ms(self, mass_kg: float) -> float:
        """The lowest published approach speed at ``mass_kg`` (m/s): the speed gate's 1-g lower
        edge (the lower flap-configuration value, equal to :meth:`reference_speed_ms` when the
        type publishes one)."""
        return kt_to_ms(self.speeds.vref_kt(mass_kg, edge="low"))

    # SI mirrors (derived; geokit is the single conversion source).
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

    @property
    def landing_mass(self) -> float:
        """Representative mass on approach/landing (kg) — the initial mass for approach
        scenarios fed to the simulator and optimizer.

        Max-landing weight when OpenAP provides it, else a typical fraction of MTOW. NOT
        MTOW: a landing aircraft is much lighter, and using MTOW inflates the stall speed so
        realistic approach speeds become infeasible. Computed (``@property``), so the rule
        lives in one place and is easy to change.
        """
        if self.mass.max_landing_kg is not None:
            return float(self.mass.max_landing_kg)
        return _LANDING_MASS_FRACTION_OF_MTOW * self.mass.max_takeoff_kg


def published_speeds(typecode: str) -> ReferenceSpeed:
    """The type's published approach-speed row; ``KeyError`` when the table has none."""
    speeds = reference_speed(typecode)
    if speeds is None:
        raise KeyError(f"{typecode}: no published approach speed in aircraft/reference_speeds.json")
    return speeds


# ── Presets (hand-tuned airframe and procedure values; speeds are the published ones) ──

A320 = Aircraft(
    code="A320",
    name="Airbus A320-200",
    category="narrow_body",
    geometry=Geometry(wing_area_m2=122.6),
    mass=Mass(max_takeoff_kg=78000.0),
    engine=Engine(count=2, max_thrust_n_each=120000.0),   # 240000 N total
    approach=Approach(
        speeds=published_speeds("A320"),
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
        speeds=published_speeds("B77W"),
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
        speeds=published_speeds("C172"),
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
