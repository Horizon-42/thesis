import math
import unittest

from aerodynamic_model.coordinates_convertor import (
    CoordinateConverter,
    ECEFCoordinate,
    ENUCoordinate,
    GeodeticCoordinate,
)


def assert_close(
    testcase: unittest.TestCase,
    actual: float,
    expected: float,
    abs_tol: float = 1e-9,
) -> None:
    testcase.assertTrue(
        math.isclose(actual, expected, abs_tol=abs_tol),
        f"{actual!r} != {expected!r} within absolute tolerance {abs_tol}",
    )


def assert_geodetic_close(
    testcase: unittest.TestCase,
    actual: GeodeticCoordinate,
    expected: GeodeticCoordinate,
    angle_tol: float = 1e-9,
    altitude_tol: float = 1e-4,
) -> None:
    assert_close(testcase, actual.latitude, expected.latitude, angle_tol)
    assert_close(testcase, actual.longitude, expected.longitude, angle_tol)
    assert_close(testcase, actual.altitude, expected.altitude, altitude_tol)


class TestCoordinateConverter(unittest.TestCase):
    def test_geodetic_to_ecef_equator_prime_meridian_uses_semimajor_axis(self):
        # This anchor case guards the WGS84 x-axis convention at latitude 0 and longitude 0.
        ecef = CoordinateConverter.geodetic_to_ecef(GeodeticCoordinate(0.0, 0.0, 0.0))

        assert_close(self, ecef.x, CoordinateConverter.a)
        assert_close(self, ecef.y, 0.0)
        assert_close(self, ecef.z, 0.0)

    def test_geodetic_to_ecef_north_pole_uses_semiminor_axis(self):
        # This anchor case guards polar conversion, where the ellipsoid semi-minor axis defines z.
        ecef = CoordinateConverter.geodetic_to_ecef(GeodeticCoordinate(90.0, 0.0, 0.0))

        assert_close(self, ecef.x, 0.0, abs_tol=1e-8)
        assert_close(self, ecef.y, 0.0, abs_tol=1e-8)
        assert_close(self, ecef.z, CoordinateConverter.b, abs_tol=1e-6)

    def test_ecef_to_geodetic_equator_prime_meridian_recovers_zero_altitude(self):
        # This reverse anchor case proves the simple surface point returns to geodetic origin.
        geo = CoordinateConverter.ecef_to_geodetic(
            ECEFCoordinate(CoordinateConverter.a, 0.0, 0.0)
        )

        assert_geodetic_close(self, geo, GeodeticCoordinate(0.0, 0.0, 0.0))

    def test_ecef_to_geodetic_exact_pole_keeps_altitude_finite(self):
        # This edge case catches divide-by-cos(latitude) failures at exact polar ECEF points.
        geo = CoordinateConverter.ecef_to_geodetic(
            ECEFCoordinate(0.0, 0.0, CoordinateConverter.b + 500.0)
        )

        assert_geodetic_close(self, geo, GeodeticCoordinate(90.0, 0.0, 500.0))

    def test_geodetic_ecef_round_trip_preserves_airport_area_coordinate(self):
        # This realistic CYYC-area point checks negative longitude and nonzero altitude together.
        original = GeodeticCoordinate(51.1139, -114.0203, 1084.0)

        restored = CoordinateConverter.ecef_to_geodetic(
            CoordinateConverter.geodetic_to_ecef(original)
        )

        assert_geodetic_close(self, restored, original)

    def test_geodetic_to_enu_same_point_is_local_origin(self):
        # This verifies a reference point converts to the ENU origin relative to itself.
        ref = GeodeticCoordinate(51.1139, -114.0203, 1084.0)

        enu = CoordinateConverter.geodetic_to_enu(ref, ref)

        assert_close(self, enu.east, 0.0)
        assert_close(self, enu.north, 0.0)
        assert_close(self, enu.up, 0.0)

    def test_geodetic_to_enu_altitude_delta_projects_to_up_axis(self):
        # This isolates altitude so east and north stay zero while up equals the height change.
        ref = GeodeticCoordinate(51.1139, -114.0203, 1084.0)
        above_ref = GeodeticCoordinate(ref.latitude, ref.longitude, ref.altitude + 250.0)

        enu = CoordinateConverter.geodetic_to_enu(above_ref, ref)

        assert_close(self, enu.east, 0.0, abs_tol=1e-8)
        assert_close(self, enu.north, 0.0, abs_tol=1e-8)
        assert_close(self, enu.up, 250.0, abs_tol=1e-8)

    def test_enu_geodetic_round_trip_preserves_local_displacement(self):
        # This round trip protects the ENU basis-vector signs used for local trajectory offsets.
        ref = GeodeticCoordinate(51.1139, -114.0203, 1084.0)
        original = ENUCoordinate(east=1200.0, north=-800.0, up=150.0)

        restored = CoordinateConverter.geodetic_to_enu(
            CoordinateConverter.enu_to_geodetic(original, ref),
            ref,
        )

        assert_close(self, restored.east, original.east, abs_tol=1e-6)
        assert_close(self, restored.north, original.north, abs_tol=1e-6)
        assert_close(self, restored.up, original.up, abs_tol=1e-6)

    def test_enu_east_axis_increases_longitude_near_equator(self):
        # This directional case confirms positive local east maps to positive longitude.
        ref = GeodeticCoordinate(0.0, 0.0, 0.0)

        geo = CoordinateConverter.enu_to_geodetic(ENUCoordinate(1000.0, 0.0, 0.0), ref)

        self.assertGreater(geo.longitude, 0.0)
        assert_close(self, geo.latitude, 0.0)
        assert_close(
            self,
            geo.altitude,
            math.hypot(CoordinateConverter.a, 1000.0) - CoordinateConverter.a,
            abs_tol=1e-6,
        )
