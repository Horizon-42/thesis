"""Geometry primitives ①–④: along/cross-track, bearing, intercept (scalar-component API)."""

import numpy as np

from approach_constraints import geometry as geo

A = np.array([0.0, 0.0])
B = np.array([10.0, 0.0])      # leg points due-north (+n) by 10 m


def test_along_track_scalar_and_array():
    assert np.isclose(geo.along_track(3.0, 7.0, A, B), 3.0)
    n = np.array([3.0, 10.0, -1.0])
    e = np.array([7.0, -2.0, 0.0])
    assert np.allclose(geo.along_track(n, e, A, B), [3.0, 10.0, -1.0])


def test_cross_track_is_signed():
    assert np.isclose(geo.cross_track(3.0, 7.0, A, B), -7.0)     # +e side -> negative
    assert np.isclose(geo.cross_track(3.0, -7.0, A, B), 7.0)
    n = np.array([3.0, 5.0])
    e = np.array([7.0, 0.0])
    assert np.allclose(np.abs(geo.cross_track(n, e, A, B)), [7.0, 0.0])


def test_course_bearing_uses_the_model_heading_convention():
    # Model convention (aerodynamic_model.casadi_simulator): 0 = East (+e), CCW toward North (+n).
    assert np.isclose(geo.course_bearing([0.0, 0.0], [0.0, 5.0]), 0.0)             # due east
    assert np.isclose(geo.course_bearing([0.0, 0.0], [5.0, 0.0]), np.pi / 2)       # due north
    assert np.isclose(geo.course_bearing([0.0, 0.0], [-5.0, 0.0]), -np.pi / 2)     # due south
    assert np.isclose(abs(geo.course_bearing([0.0, 0.0], [0.0, -5.0])), np.pi)     # due west


def test_intercept_angle_wraps():
    assert np.isclose(geo.intercept_angle_deg(np.deg2rad(10.0), np.deg2rad(350.0)), 20.0)
    assert np.isclose(geo.intercept_angle_deg(np.deg2rad(30.0), 0.0), 30.0)


def test_state_psi_compares_directly_against_course_bearing():
    # REGRESSION (convention schism): an aircraft flying due north (model psi = +pi/2) along a
    # due-north leg must read as a ZERO intercept. With the old compass-convention course
    # (atan2(de, dn)) this read as 90 deg — exactly the KRDU RW32 bug class.
    psi_due_north = np.pi / 2
    course = geo.course_bearing([0.0, 0.0], [5.0, 0.0])   # due-north leg
    assert np.isclose(geo.intercept_angle_deg(psi_due_north, course), 0.0)
