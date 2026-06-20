import math
import unittest

import casadi as ca
import numpy as np

from aerodynamic_model.casadi_coordinates_converter import (
    WGS84_A,
    WGS84_B,
    ecef_to_geodetic_expr,
    ecef_to_enu_rotation_matrix_expr,
    enu_to_geodetic_expr,
    geodetic_deg_to_ecef_expr,
    geodetic_to_enu_expr,
)
from aerodynamic_model.coordinates_convertor import (
    CoordinateConverter,
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


class TestCasadiCoordinatesConverter(unittest.TestCase):
    def setUp(self):
        lat = ca.SX.sym("lat")
        lon = ca.SX.sym("lon")
        alt = ca.SX.sym("alt")
        ref_lat = ca.SX.sym("ref_lat")
        ref_lon = ca.SX.sym("ref_lon")
        ref_alt = ca.SX.sym("ref_alt")
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        z = ca.SX.sym("z")
        east = ca.SX.sym("east")
        north = ca.SX.sym("north")
        up = ca.SX.sym("up")

        self.geodetic_to_ecef = ca.Function(
            "geodetic_to_ecef",
            [lat, lon, alt],
            list(geodetic_deg_to_ecef_expr(lat, lon, alt)),
            ["lat", "lon", "alt"],
            ["x", "y", "z"],
        )
        self.ecef_to_geodetic = ca.Function(
            "ecef_to_geodetic",
            [x, y, z],
            list(ecef_to_geodetic_expr(x, y, z)),
            ["x", "y", "z"],
            ["lat", "lon", "alt"],
        )
        self.ecef_to_enu_rotation_matrix = ca.Function(
            "ecef_to_enu_rotation_matrix",
            [ref_lat, ref_lon],
            [ecef_to_enu_rotation_matrix_expr(ref_lat, ref_lon)],
        )
        self.geodetic_to_enu = ca.Function(
            "geodetic_to_enu",
            [lat, lon, alt, ref_lat, ref_lon, ref_alt],
            list(geodetic_to_enu_expr(lat, lon, alt, ref_lat, ref_lon, ref_alt)),
            ["lat", "lon", "alt", "ref_lat", "ref_lon", "ref_alt"],
            ["east", "north", "up"],
        )
        self.enu_to_geodetic = ca.Function(
            "enu_to_geodetic",
            [east, north, up, ref_lat, ref_lon, ref_alt],
            list(enu_to_geodetic_expr(east, north, up, ref_lat, ref_lon, ref_alt)),
            ["east", "north", "up", "ref_lat", "ref_lon", "ref_alt"],
            ["lat", "lon", "alt"],
        )

        ecef = ca.vertcat(*geodetic_deg_to_ecef_expr(lat, lon, alt))
        self.geodetic_to_ecef_jacobian = ca.Function(
            "geodetic_to_ecef_jacobian",
            [lat, lon, alt],
            [ca.jacobian(ecef, ca.vertcat(lat, lon, alt))],
        )

    def _geodetic_to_ecef_values(
        self,
        geo: GeodeticCoordinate,
    ) -> tuple[float, float, float]:
        result = self.geodetic_to_ecef(
            lat=geo.latitude,
            lon=geo.longitude,
            alt=geo.altitude,
        )
        return tuple(float(result[name]) for name in ("x", "y", "z"))

    def _ecef_to_geodetic_value(
        self,
        x: float,
        y: float,
        z: float,
    ) -> GeodeticCoordinate:
        result = self.ecef_to_geodetic(x=x, y=y, z=z)
        return GeodeticCoordinate(
            latitude=float(result["lat"]),
            longitude=float(result["lon"]),
            altitude=float(result["alt"]),
        )

    def _geodetic_to_enu_value(
        self,
        geo: GeodeticCoordinate,
        ref: GeodeticCoordinate,
    ) -> ENUCoordinate:
        result = self.geodetic_to_enu(
            lat=geo.latitude,
            lon=geo.longitude,
            alt=geo.altitude,
            ref_lat=ref.latitude,
            ref_lon=ref.longitude,
            ref_alt=ref.altitude,
        )
        return ENUCoordinate(
            east=float(result["east"]),
            north=float(result["north"]),
            up=float(result["up"]),
        )

    def _enu_to_geodetic_value(
        self,
        enu: ENUCoordinate,
        ref: GeodeticCoordinate,
    ) -> GeodeticCoordinate:
        result = self.enu_to_geodetic(
            east=enu.east,
            north=enu.north,
            up=enu.up,
            ref_lat=ref.latitude,
            ref_lon=ref.longitude,
            ref_alt=ref.altitude,
        )
        return GeodeticCoordinate(
            latitude=float(result["lat"]),
            longitude=float(result["lon"]),
            altitude=float(result["alt"]),
        )

    def test_geodetic_to_ecef_equator_prime_meridian_uses_semimajor_axis(self):
        # This anchor proves latitude 0 and longitude 0 lie on the ECEF x-axis.
        x, y, z = self._geodetic_to_ecef_values(GeodeticCoordinate(0.0, 0.0, 0.0))

        assert_close(self, x, WGS84_A)
        assert_close(self, y, 0.0)
        assert_close(self, z, 0.0)

    def test_geodetic_to_ecef_north_pole_uses_semiminor_axis(self):
        # This anchor proves the polar surface point uses the ellipsoid semi-minor axis.
        x, y, z = self._geodetic_to_ecef_values(GeodeticCoordinate(90.0, 0.0, 0.0))

        assert_close(self, x, 0.0, abs_tol=1e-8)
        assert_close(self, y, 0.0, abs_tol=1e-8)
        assert_close(self, z, WGS84_B, abs_tol=1e-6)

    def test_geodetic_to_ecef_matches_numeric_converter_for_airport_point(self):
        # This realistic airport-area point checks nonzero altitude and negative longitude.
        geo = GeodeticCoordinate(51.1139, -114.0203, 1084.0)

        actual = self._geodetic_to_ecef_values(geo)
        expected = CoordinateConverter.geodetic_to_ecef(geo)

        assert_close(self, actual[0], expected.x, abs_tol=1e-6)
        assert_close(self, actual[1], expected.y, abs_tol=1e-6)
        assert_close(self, actual[2], expected.z, abs_tol=1e-6)

    def test_ecef_to_geodetic_matches_numeric_converter_for_airport_point(self):
        # Avoiding the exact pole keeps this symbolic expression in its smooth operating range.
        geo = GeodeticCoordinate(51.1139, -114.0203, 1084.0)
        ecef = CoordinateConverter.geodetic_to_ecef(geo)

        actual = self._ecef_to_geodetic_value(ecef.x, ecef.y, ecef.z)

        assert_geodetic_close(self, actual, geo)

    def test_ecef_to_enu_rotation_matrix_uses_projection_rows(self):
        # At the equator/prime meridian, ECEF y is east, z is north, and x is up.
        matrix = np.array(self.ecef_to_enu_rotation_matrix(0.0, 0.0), dtype=float)

        expected = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(matrix, expected, atol=1e-12)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)

    def test_geodetic_to_enu_same_point_is_local_origin(self):
        # The reference geodetic point must convert to zero displacement in its own ENU frame.
        ref = GeodeticCoordinate(51.1139, -114.0203, 1084.0)

        actual = self._geodetic_to_enu_value(ref, ref)

        assert_close(self, actual.east, 0.0)
        assert_close(self, actual.north, 0.0)
        assert_close(self, actual.up, 0.0)

    def test_geodetic_to_enu_matches_numeric_converter_for_local_offset(self):
        # This catches sign/order mistakes in the ECEF-to-ENU basis projection.
        ref = GeodeticCoordinate(51.1139, -114.0203, 1084.0)
        expected = ENUCoordinate(east=1200.0, north=-800.0, up=150.0)
        geo = CoordinateConverter.enu_to_geodetic(expected, ref)

        actual = self._geodetic_to_enu_value(geo, ref)

        assert_close(self, actual.east, expected.east, abs_tol=1e-5)
        assert_close(self, actual.north, expected.north, abs_tol=1e-5)
        assert_close(self, actual.up, expected.up, abs_tol=1e-5)

    def test_enu_to_geodetic_matches_numeric_converter_for_local_offset(self):
        # The inverse path should match the existing numeric converter for the same ENU offset.
        ref = GeodeticCoordinate(51.1139, -114.0203, 1084.0)
        enu = ENUCoordinate(east=1200.0, north=-800.0, up=150.0)

        actual = self._enu_to_geodetic_value(enu, ref)
        expected = CoordinateConverter.enu_to_geodetic(enu, ref)

        assert_geodetic_close(self, actual, expected)

    def test_geodetic_to_ecef_jacobian_is_symbolically_buildable(self):
        # This protects the OCP use case: downstream code can differentiate the expression.
        jacobian = np.array(
            self.geodetic_to_ecef_jacobian(51.1139, -114.0203, 1084.0),
            dtype=float,
        )

        self.assertEqual(jacobian.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(jacobian)))


if __name__ == "__main__":
    unittest.main()
