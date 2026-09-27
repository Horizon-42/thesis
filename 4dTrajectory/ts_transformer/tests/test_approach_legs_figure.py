"""The prior design's §1.1 figure (`experiments/approach_legs_figure`): the file in the docs is what the runner draws;
the vectored arrival it draws is legs at the stated headings joined by left turns tangent to them, intercepting the
course where a 30° intercept is allowed; the straight-in reaches the course along it."""

import math

import pytest
from geokit import NM_M

from ts_transformer.experiments import approach_legs_figure as figure
from ts_transformer.experiments.approach_legs_figure import Point
from ts_transformer.instructions.spec import ATC_MAX_INTERCEPT_DEG


def test_the_figure_in_the_docs_is_what_the_runner_draws():
    """Change the example or the drawing, then run `python run_ts.py approach_legs_figure` and commit the SVG with it."""
    assert figure.FIGURE.read_text(encoding="utf-8") == figure.render()


def heading_deg(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x)) % 360.0


def assert_left_turn_tangent(centre: Point, joint: Point, heading: float) -> None:
    """The turn's radius at the joint is `TURN_RADIUS_M` long and lies 90° to the left of the leg flown there."""
    dx, dy = math.cos(math.radians(heading)), math.sin(math.radians(heading))
    vx, vy = centre.x - joint.x, centre.y - joint.y
    assert math.hypot(vx, vy) == pytest.approx(figure.TURN_RADIUS_M)
    assert dx * vy - dy * vx == pytest.approx(figure.TURN_RADIUS_M)


def test_the_legs_fly_their_headings_and_every_turn_is_tangent_to_both_legs_it_joins():
    path = figure.vectored_path()
    intercept = 360.0 - ATC_MAX_INTERCEPT_DEG
    assert heading_deg(path.downwind_start, path.downwind_end) == pytest.approx(180.0)
    assert path.downwind_start.y == path.downwind_end.y == figure.DOWNWIND_Y_M
    assert heading_deg(path.base_start, path.base_end) == pytest.approx(270.0)
    assert path.base_end.y == pytest.approx(figure.BASE_END_Y_M)
    assert heading_deg(path.intercept_start, path.clearance) == pytest.approx(intercept)
    assert (path.capture.x, path.capture.y) == (figure.CAPTURE_X_M, 0.0)
    for centre, joint, heading in ((path.base_turn_centre, path.downwind_end, 180.0),
                                   (path.base_turn_centre, path.base_start, 270.0),
                                   (path.intercept_turn_centre, path.base_end, 270.0),
                                   (path.intercept_turn_centre, path.intercept_start, intercept),
                                   (path.capture_turn_centre, path.clearance, intercept),
                                   (path.capture_turn_centre, path.capture, 0.0)):
        assert_left_turn_tangent(centre, joint, heading)


def test_a_drawn_turn_is_points_of_its_computed_circle_from_one_leg_to_the_next():
    path = figure.vectored_path()
    points = figure.left_turn(path.base_turn_centre, 180.0, 270.0)
    assert (points[0], points[-1]) == (path.downwind_end, path.base_start)
    for point in points:
        assert math.dist((point.x, point.y), (path.base_turn_centre.x, path.base_turn_centre.y)) == pytest.approx(
            figure.TURN_RADIUS_M)


def test_the_30_degree_intercept_meets_the_course_at_least_two_miles_outside_the_approach_gate():
    """7110.65BB 5-9-2 TBL 5-9-1: closer than 2 miles to the gate the limit is 20°."""
    gate = figure.approach_gate_x_m()
    assert gate <= -5.0 * NM_M
    assert figure.interception_point(figure.vectored_path()).x <= gate - 2.0 * NM_M


def test_the_straight_in_reaches_the_course_along_it_at_less_than_the_intercept_angle():
    start, control, join = figure.straight_in()
    assert control.y == join.y == 0.0 and control.x < join.x
    assert heading_deg(start, control) < ATC_MAX_INTERCEPT_DEG
