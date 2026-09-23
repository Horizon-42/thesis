import unittest

from geokit import kt_to_ms, nm_to_m

from aircraft.query_aircraft_parameters import (
    AircraftLookupError,
    get_aircraft_parameters,
    load_json,
    openap_direct_typecodes,
    openap_performance_metadata,
    openap_support_kind,
    PARAMETERS_PATH,
)
from aircraft.reference_speeds import reference_speed

from aircraft.aircraft_sets import (
    A320,
    AIRCRAFT_PRESETS,
    Aircraft,
    Approach,
    Engine,
    Geometry,
    Mass,
    published_speeds,
)


class TestApproachSiMirrors(unittest.TestCase):
    def test_target_takes_the_upper_and_floor_the_lower_flap_value(self):
        # B738 publishes 140 kt (full flap) and 144 kt (reduced flap) at its MALW.
        approach = get_aircraft_parameters("B738").approach
        malw = approach.speeds.malw_kg
        self.assertAlmostEqual(approach.reference_speed_ms(malw), kt_to_ms(144.0))
        self.assertAlmostEqual(approach.minimum_speed_ms(malw), kt_to_ms(140.0))

    def test_reference_speed_is_the_published_speed_scaled_to_mass(self):
        speeds = A320.approach.speeds
        self.assertAlmostEqual(A320.approach.reference_speed_ms(speeds.malw_kg),
                               kt_to_ms(speeds.approach_speed_kt))
        mass = 0.81 * speeds.malw_kg
        self.assertAlmostEqual(A320.approach.reference_speed_ms(mass),
                               kt_to_ms(speeds.approach_speed_kt) * 0.9)

    def test_distance_mirrors_are_nm_converted(self):
        self.assertAlmostEqual(A320.approach.final_segment_min_m, nm_to_m(A320.approach.final_segment_min_nm))
        self.assertAlmostEqual(A320.approach.final_segment_max_m, nm_to_m(A320.approach.final_segment_max_nm))
        self.assertAlmostEqual(A320.approach.protection_half_width_m, nm_to_m(A320.approach.protection_half_width_nm))


class TestEngine(unittest.TestCase):
    def test_total_thrust_is_per_engine_times_count(self):
        self.assertEqual(A320.engine.max_thrust_total_n, A320.engine.max_thrust_n_each * A320.engine.count)
        self.assertEqual(A320.engine.max_thrust_total_n, 240000.0)


class TestPresets(unittest.TestCase):
    def test_preset_values_unchanged(self):
        self.assertEqual(A320.geometry.wing_area_m2, 122.6)
        self.assertEqual(A320.mass.max_takeoff_kg, 78000.0)
        self.assertEqual(A320.approach.final_segment_min_nm, 5.0)

    def test_preset_speeds_are_the_published_ones(self):
        for code, aircraft in AIRCRAFT_PRESETS.items():
            self.assertEqual(aircraft.approach.speeds, reference_speed(code))

    def test_all_presets_are_aircraft_with_consistent_si_mirrors(self):
        for aircraft in AIRCRAFT_PRESETS.values():
            self.assertIsInstance(aircraft, Aircraft)
            self.assertAlmostEqual(
                aircraft.approach.final_segment_max_m, nm_to_m(aircraft.approach.final_segment_max_nm)
            )


class TestPublishedApproachSpeed(unittest.TestCase):
    """Every approach speed the model flies is a published one, never a default, and it
    belongs to the airframe whose mass the model carries (it is rescaled by that mass)."""

    def test_every_openap_type_flies_the_published_speed_of_its_mass_airframe(self):
        records = load_json(PARAMETERS_PATH)["typecodes"]
        supported = sorted(code for code, record in records.items() if record.get("openap_supported"))
        refused = []
        for code in supported:
            try:
                aircraft = get_aircraft_parameters(code)
            except AircraftLookupError:
                refused.append(code)
                continue
            if openap_support_kind(code) == "direct":
                self.assertEqual(aircraft.approach.speeds, reference_speed(code), code)
            self.assertAlmostEqual(aircraft.approach.speeds.malw_kg, aircraft.landing_mass,
                                   delta=0.25 * aircraft.landing_mass, msg=code)
        # The one OpenAP type the FAA Aircraft Characteristics Database does not list.
        self.assertEqual(refused, ["B3XM"])

    def test_a_synonym_type_flies_the_speed_of_the_airframe_whose_mass_it_carries(self):
        # A synonym is cached with its surrogate's performance AND mass (LJ45 with GLF6's), so
        # it lands at its surrogate's speed; C56X's own mass was restored, so it keeps its own.
        records = load_json(PARAMETERS_PATH)["typecodes"]
        synonyms = sorted(code for code in records if openap_support_kind(code) == "synonym")
        self.assertEqual(len(synonyms), 21)
        for code in synonyms:
            surrogate = openap_performance_metadata(code)["performance_typecode"]
            expected = code if code == "C56X" else surrogate
            self.assertEqual(get_aircraft_parameters(code).approach.speeds,
                             reference_speed(expected), code)

    def test_a_type_without_a_published_speed_is_refused(self):
        with self.assertRaisesRegex(AircraftLookupError, "no published"):
            get_aircraft_parameters("B3XM")
        self.assertIsNone(openap_support_kind("B3XM"))
        self.assertNotIn("B3XM", openap_direct_typecodes())
        with self.assertRaisesRegex(KeyError, "no published approach speed"):
            published_speeds("ZZZZ")


class TestLandingMass(unittest.TestCase):
    def test_uses_max_landing_kg_when_set(self):
        aircraft = Aircraft(
            code="X", name="X", category="x",
            geometry=Geometry(wing_area_m2=122.6),
            mass=Mass(max_takeoff_kg=78000.0, max_landing_kg=66000.0),
            engine=Engine(count=2, max_thrust_n_each=120000.0),
            approach=A320.approach,
        )
        self.assertEqual(aircraft.landing_mass, 66000.0)

    def test_falls_back_to_fraction_of_mtow(self):
        # A320 preset carries no max_landing_kg -> 0.85 * MTOW.
        self.assertAlmostEqual(A320.landing_mass, 0.85 * 78000.0)
        self.assertLess(A320.landing_mass, A320.mass.max_takeoff_kg)


class TestFrozen(unittest.TestCase):
    def test_aircraft_is_frozen(self):
        with self.assertRaises(Exception):
            A320.code = "X"  # type: ignore[misc]


class TestConstruct(unittest.TestCase):
    def test_build_custom_aircraft_from_nested_groups(self):
        aircraft = Aircraft(
            code="TEST",
            name="Test",
            category="test",
            geometry=Geometry(wing_area_m2=100.0),
            mass=Mass(max_takeoff_kg=50000.0),
            engine=Engine(count=2, max_thrust_n_each=100000.0),
            approach=Approach(
                speeds=reference_speed("E190"),
                final_segment_min_nm=4.0,
                final_segment_max_nm=9.0,
                protection_half_width_nm=0.7,
                glide_angle_deg=3.0,
                threshold_crossing_height_m=15.0,
                thrust_guess_n=30000.0,
            ),
        )
        self.assertEqual(aircraft.engine.max_thrust_total_n, 200000.0)
        self.assertAlmostEqual(aircraft.approach.reference_speed_ms(reference_speed("E190").malw_kg),
                               kt_to_ms(reference_speed("E190").approach_speed_kt))
        self.assertAlmostEqual(aircraft.approach.final_segment_min_m, nm_to_m(4.0))


if __name__ == "__main__":
    unittest.main()
