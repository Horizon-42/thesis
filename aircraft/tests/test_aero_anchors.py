"""The stall-model anchors: per-family landing Cl_max and airframe-identity facts.

These pins exist because the anchors were measured wrong once
(``evaluation/docs/BASELINE_SPEED_GATE_RESULTS.md`` §5): a shared narrow-body
Cl_max graded 45–92 % of real A320-family crossings "too slow", and OpenAP's
C550 surrogate understated the C56X airframe by 25 % of its mass.
"""

from __future__ import annotations

import unittest

from aircraft.aero_params import aero_params_for_aircraft, stall_speed_ms
from aircraft.query_aircraft_parameters import get_aircraft_parameters
from geokit import ms_to_kt


class A320FamilyLandingClMax(unittest.TestCase):
    def test_family_carries_the_calibrated_landing_cl_max(self):
        for code in ("A319", "A320", "A321", "A20N", "A21N"):
            aircraft = get_aircraft_parameters(code)
            self.assertEqual(
                aero_params_for_aircraft(aircraft).Cl_max, 3.0, code
            )

    def test_boeing_narrow_bodies_keep_the_bucket(self):
        for code in ("B737", "B738", "B38M", "B739"):
            aircraft = get_aircraft_parameters(code)
            self.assertEqual(
                aero_params_for_aircraft(aircraft).Cl_max, 2.7, code
            )

    def test_a320_floor_sits_at_or_just_below_published_vls(self):
        """The calibration claim itself: 1.23·Vs1g(64 t) must land at the published
        ≈128 kt CONF FULL VLS, or up to ~3 kt BELOW it — slightly conservative is
        the right direction for a gate that judges wind-uncorrected ground speed.
        If this drifts, the Cl_max above no longer means what its comment says."""
        aircraft = get_aircraft_parameters("A320")
        aero = aero_params_for_aircraft(aircraft)
        floor_kt = ms_to_kt(
            1.23 * stall_speed_ms(64_000.0, wing_area_m2=aero.S, cl_max=aero.Cl_max)
        )
        self.assertGreaterEqual(floor_kt, 124.0)
        self.assertLessEqual(floor_kt, 129.0)


class AirframeIdentityCorrections(unittest.TestCase):
    def test_c56x_carries_certificated_masses_not_the_c550_surrogate(self):
        aircraft = get_aircraft_parameters("C56X")
        self.assertEqual(aircraft.mass.max_takeoff_kg, 9072.0)
        self.assertEqual(aircraft.mass.max_landing_kg, 8482.0)
        self.assertEqual(aircraft.geometry.wing_area_m2, 34.35)
        # landing_mass follows the corrected MLW.
        self.assertEqual(aircraft.landing_mass, 8482.0)


if __name__ == "__main__":
    unittest.main()
