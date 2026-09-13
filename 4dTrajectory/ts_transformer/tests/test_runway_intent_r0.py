"""Runway-intent R0's own readout logic (`experiments.runway_intent_r0`): anchors, flips."""

from datetime import datetime, timedelta, timezone

from ts_transformer.data.runway_context import ContextLanding
from ts_transformer.experiments.runway_intent_r0 import (
    anchors,
    direction_flips,
    remaining_path_m,
    track_course_at,
)

T0 = datetime(2026, 6, 1, 6, 0, tzinfo=timezone.utc)
GROUPS = {"05L": 0, "05R": 0, "23L": 1, "23R": 1}


def _bins(*directions: str, start: datetime = T0) -> list[ContextLanding]:
    """Two landings per 15-min bin, one bin per entry, on the given runways."""
    landings = []
    for i, runway in enumerate(directions):
        at = start + timedelta(minutes=15 * i + 1)
        landings += [ContextLanding(at, runway), ContextLanding(at + timedelta(minutes=5), runway)]
    return landings


def test_a_flip_needs_the_new_direction_held_two_bins():
    flips, across = direction_flips(_bins("23R", "23R", "05L", "23R", "05L", "05L"), GROUPS)
    assert flips == {"2026-06-01": 1} and across == {}


def test_a_change_across_a_long_gap_is_not_a_flip():
    evening = _bins("23R", "23R")
    morning = _bins("05L", "05L", start=T0 + timedelta(hours=10))
    flips, across = direction_flips(evening + morning, GROUPS)
    assert flips == {} and across == {"2026-06-01": 1}


def test_the_course_at_an_anchor_looks_back_not_ahead():
    # north, north, then a turn east AFTER the anchor at index 1
    waypoints = [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.01, 0.0], [2.0, 0.01, 0.01, 0.0]]
    assert round(track_course_at(waypoints, 1)) == 0      # still north: the turn is later
    assert round(track_course_at(waypoints, 2)) == 90
    assert round(track_course_at(waypoints, 0)) == 0      # entry: the first segment


def test_anchors_skip_distances_the_slice_never_had():
    # a straight 12 km track: no 15 km anchor; the 10 km anchor is the first sample within 10 km
    waypoints = [[float(i), 0.0, i * 0.001, 0.0] for i in range(109)]   # ~0.111 km per step
    remaining = remaining_path_m(waypoints)
    got = anchors(waypoints, [15.0, 10.0, 3.0])
    assert "15km" not in got and got["entry"] == 0
    assert remaining[got["10km"]] <= 10_000.0 < remaining[got["10km"] - 1]
    assert remaining[got["3km"]] <= 3_000.0 < remaining[got["3km"] - 1]
