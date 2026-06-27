import unittest

from geokit import KT_MS, NM_M

from aircraft.aircraft_sets import (
    A320,
    AIRCRAFT_PRESETS,
    AircraftSpec,
    knots_to_ms,
    nm_to_m,
)


class TestUnitConversions(unittest.TestCase):
    def test_knots_to_ms_uses_geokit_factor(self):
        self.assertAlmostEqual(knots_to_ms(100.0), 100.0 * KT_MS)
        self.assertEqual(knots_to_ms(0.0), 0.0)

    def test_nm_to_m_uses_geokit_factor(self):
        self.assertAlmostEqual(nm_to_m(1.0), NM_M)
        self.assertAlmostEqual(nm_to_m(2.5), 2.5 * NM_M)


class TestAircraftSpecDerivedSiFields(unittest.TestCase):
    def test_speed_fields_are_kt_converted_to_ms(self):
        self.assertAlmostEqual(A320.terminal_speed_ms, knots_to_ms(A320.terminal_speed_kt))
        self.assertAlmostEqual(A320.terminal_speed_min_ms, knots_to_ms(A320.terminal_speed_min_kt))
        self.assertAlmostEqual(A320.terminal_speed_max_ms, knots_to_ms(A320.terminal_speed_max_kt))

    def test_approach_fields_are_nm_converted_to_m(self):
        self.assertAlmostEqual(A320.final_approach_min_m, nm_to_m(A320.final_approach_min_nm))
        self.assertAlmostEqual(A320.final_approach_max_m, nm_to_m(A320.final_approach_max_nm))
        self.assertAlmostEqual(
            A320.final_approach_lateral_half_width_m,
            nm_to_m(A320.final_approach_lateral_half_width_nm),
        )

    def test_all_presets_have_consistent_si_mirrors(self):
        for spec in AIRCRAFT_PRESETS.values():
            self.assertAlmostEqual(spec.terminal_speed_ms, spec.terminal_speed_kt * KT_MS)
            self.assertAlmostEqual(spec.final_approach_max_m, spec.final_approach_max_nm * NM_M)

    def test_original_aviation_unit_fields_are_unchanged(self):
        self.assertEqual(A320.terminal_speed_kt, 145.0)
        self.assertEqual(A320.final_approach_min_nm, 5.0)

    def test_spec_remains_frozen(self):
        with self.assertRaises(Exception):
            A320.terminal_speed_ms = 0.0  # type: ignore[misc]

    def test_derived_fields_are_not_constructor_arguments(self):
        # The SI mirrors are computed, never passed in: building a spec with only
        # the aviation-unit fields must succeed and populate the SI fields.
        spec = AircraftSpec(
            code="TEST",
            name="Test",
            category="test",
            wing_area_m2=100.0,
            mass_kg=50000.0,
            max_thrust_n=200000.0,
            approach_thrust_guess_n=30000.0,
            terminal_speed_kt=140.0,
            terminal_speed_min_kt=130.0,
            terminal_speed_max_kt=150.0,
            final_approach_min_nm=4.0,
            final_approach_max_nm=9.0,
            final_approach_lateral_half_width_nm=0.7,
            final_approach_glide_angle_deg=3.0,
            threshold_crossing_height_m=15.0,
        )
        self.assertAlmostEqual(spec.terminal_speed_ms, knots_to_ms(140.0))
        self.assertAlmostEqual(spec.final_approach_min_m, nm_to_m(4.0))


if __name__ == "__main__":
    unittest.main()
