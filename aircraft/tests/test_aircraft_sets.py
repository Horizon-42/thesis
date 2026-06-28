import unittest

from geokit import kt_to_ms, nm_to_m

from aircraft.aircraft_sets import (
    A320,
    AIRCRAFT_PRESETS,
    Aircraft,
    Approach,
    Engine,
    Geometry,
    Mass,
)


class TestApproachSiMirrors(unittest.TestCase):
    def test_speed_mirrors_are_kt_converted(self):
        self.assertAlmostEqual(A320.approach.reference_speed_ms, kt_to_ms(A320.approach.reference_speed_kt))
        self.assertAlmostEqual(A320.approach.min_speed_ms, kt_to_ms(A320.approach.min_speed_kt))
        self.assertAlmostEqual(A320.approach.max_speed_ms, kt_to_ms(A320.approach.max_speed_kt))

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
        self.assertEqual(A320.approach.reference_speed_kt, 145.0)
        self.assertEqual(A320.approach.final_segment_min_nm, 5.0)

    def test_all_presets_are_aircraft_with_consistent_si_mirrors(self):
        for aircraft in AIRCRAFT_PRESETS.values():
            self.assertIsInstance(aircraft, Aircraft)
            self.assertAlmostEqual(
                aircraft.approach.reference_speed_ms, kt_to_ms(aircraft.approach.reference_speed_kt)
            )
            self.assertAlmostEqual(
                aircraft.approach.final_segment_max_m, nm_to_m(aircraft.approach.final_segment_max_nm)
            )


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
                reference_speed_kt=140.0,
                min_speed_kt=130.0,
                max_speed_kt=150.0,
                final_segment_min_nm=4.0,
                final_segment_max_nm=9.0,
                protection_half_width_nm=0.7,
                glide_angle_deg=3.0,
                threshold_crossing_height_m=15.0,
                thrust_guess_n=30000.0,
            ),
        )
        self.assertEqual(aircraft.engine.max_thrust_total_n, 200000.0)
        self.assertAlmostEqual(aircraft.approach.reference_speed_ms, kt_to_ms(140.0))
        self.assertAlmostEqual(aircraft.approach.final_segment_min_m, nm_to_m(4.0))


if __name__ == "__main__":
    unittest.main()
