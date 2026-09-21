"""The box vocabulary's contract: the boxes tile, the outside is refused, and the track is INSIDE.

The regression these tests exist for: the first reading smoothed the heading on the WRAPPED
course, so a moving average across ±180° averaged +179° and −179° to 0° — the final approach
course itself. A third of the fleet got heading words pointing the opposite way somewhere on the
downwind, and every artefact read that way had to be thrown out.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ts_transformer.manoeuvre.box_vocabulary import (
    BoxVocabulary, TERMINAL_CONTINUE, TERMINAL_LANDED, altitude_words, read_boxes,
)


def test_every_kind_tiles_with_no_gap_and_no_overlap():
    v = BoxVocabulary()
    for name, edges in (("heading", v.heading_edges), ("speed", v.speed_edges)):
        assert (np.diff(edges) > 0).all(), f"{name} edges are strictly increasing"
    # the heading's boxes cover the whole circle, the course box is the floor's width
    assert v.heading_edges[0] == pytest.approx(-v.heading_range_deg)
    assert v.heading_edges[-1] == pytest.approx(v.heading_range_deg)
    course_box = np.searchsorted(v.heading_edges, 0.0, side="right") - 1
    width = v.heading_edges[course_box + 1] - v.heading_edges[course_box]
    assert width == pytest.approx(2 * v.heading_floor_deg)


def test_the_altitude_ladder_spans_the_threshold_both_ways():
    """A track passing under the threshold elevation is an ordinary arrival, not an error."""
    v = BoxVocabulary()
    targets = v.altitude_targets
    assert targets[0] < 0.0 < targets[-1]
    assert (np.diff(targets) > 0).all()


def test_the_ladder_tiles_its_own_boxes_so_a_one_row_segment_always_has_a_target():
    """The reading cannot fail: this is what makes an end floor of ±5 % the only legal choice."""
    v = BoxVocabulary()
    targets = v.altitude_targets
    for height in np.linspace(targets[0] + 1.0, targets[-1] - 1.0, 400):
        half = v.altitude_half_width(targets)
        assert (np.abs(targets - height) <= half).any(), f"no target contains {height:.1f} m"


def test_a_narrower_end_floor_would_break_that():
    """Why the floor is not a free parameter: a flat 15 m end box is narrower than the ladder's
    own spacing above ~300 m, and the reading would refuse ordinary flights."""
    v = BoxVocabulary()
    targets = v.altitude_targets
    high = targets[targets > 1000.0][0]
    assert v.altitude_half_width(high) > 15.0


def test_out_of_range_is_refused_not_clamped():
    v = BoxVocabulary()
    height = np.array([100.0, 200.0])
    remaining = np.array([1000.0, 0.0])
    with pytest.raises(ValueError, match="leaves the vocabulary's range"):
        altitude_words(np.array([v.altitude_top_m * 2, 0.0]), remaining, v)


def test_the_wedge_opens_backwards_and_closes_on_the_target():
    """One straight 2° descent is one word, and the envelope shrinks to the ±5 % box at the end."""
    v = BoxVocabulary(altitude_down_deg=2.5)
    remaining = np.linspace(20_000.0, 0.0, 200)
    height = remaining * math.tan(math.radians(2.0)) + 30.0
    words = altitude_words(height, remaining, v)
    assert len(set(words.tolist())) == 1
    target = v.altitude_targets[words[0]]
    assert abs(target - height[-1]) <= v.altitude_half_width(target) + 1e-9


def test_the_wedge_must_use_remaining_path_not_the_along_course_projection():
    """A downwind leg grows the along-course distance; using it cuts the flight to shreds."""
    v = BoxVocabulary()
    seconds = np.arange(0.0, 600.0, 2.0)
    height = 1500.0 - seconds * 2.0
    path = seconds * 80.0
    by_path = altitude_words(height, path[-1] - path, v)
    downwind = np.where(seconds < 200.0, -path, path - 2 * 200.0 * 80.0)  # to-go grows, then falls
    by_projection = altitude_words(height, np.abs(downwind[-1] - downwind), v)
    assert len(set(by_path.tolist())) < len(set(by_projection.tolist()))


class _Series:
    """The smallest thing `course_frame` is asked for, built straight in the course frame."""

    def __init__(self, relative_course_deg, height_m, speed_mps, seconds=2.0):
        self.n = len(relative_course_deg)
        self.times = np.arange(self.n) * seconds
        self.course = np.asarray(relative_course_deg, dtype=float)
        self.height = np.asarray(height_m, dtype=float)
        self.speed = np.asarray(speed_mps, dtype=float)
        self.dataset_id = self.flight_id = "TEST"


@pytest.fixture()
def course_frame_of(monkeypatch):
    def install(series: _Series):
        unwrapped = np.degrees(np.unwrap(np.radians(series.course)))
        frame = {
            "t": series.times, "to_go_m": np.zeros(series.n), "cross_m": np.zeros(series.n),
            "relative_course_deg": series.course, "course_unwrapped_deg": unwrapped,
            "height_m": series.height, "ground_speed_mps": series.speed,
            "established": np.zeros(series.n, dtype=bool),
        }
        monkeypatch.setattr("ts_transformer.manoeuvre.box_vocabulary.course_frame", lambda _: frame)
    return install


def test_the_course_is_smoothed_unwrapped(course_frame_of):
    """THE regression. A track sitting on the reciprocal must not read as the COURSE.

    Smoothing the wrapped signal averages +179° and −179° to 0°, which is the final approach
    course — the exact opposite of where the aircraft is pointing."""
    course = np.array([179.0, -179.0] * 60)
    series = _Series(course, np.full(120, 900.0), np.full(120, 80.0))
    course_frame_of(series)
    v = BoxVocabulary()
    _, words, _ = read_boxes(series, v, runway_word=0)
    edges = v.heading_edges
    course_word = np.searchsorted(edges, 0.0, side="right") - 1
    assert (words[:, 0] != course_word).all(), "the reciprocal read as the course: wrapped smoothing"
    lower, upper = edges[words[0, 0]], edges[words[0, 0] + 1]
    assert abs(lower) > 90.0 and abs(upper) > 90.0


def test_the_track_is_inside_the_words_in_force(course_frame_of):
    from ts_transformer.manoeuvre.box_vocabulary import contains
    rng = np.random.default_rng(1337)
    n = 300
    course = np.concatenate([np.full(120, 150.0), np.linspace(150.0, 0.0, 60), np.zeros(120)])
    course = course + rng.normal(0.0, 0.3, n)
    height = np.linspace(1800.0, 25.0, n)
    speed = np.linspace(120.0, 70.0, n)
    series = _Series(course, height, speed)
    course_frame_of(series)
    v = BoxVocabulary()
    moments, words, holds = read_boxes(series, v, runway_word=0)
    assert contains(series, v, moments, words)
    assert holds.sum() == pytest.approx(series.times[-1] - series.times[0])
    assert words[-1, KINDS_TERMINAL] == TERMINAL_LANDED
    assert (words[:-1, KINDS_TERMINAL] == TERMINAL_CONTINUE).all()


KINDS_TERMINAL = 5


def test_the_duration_sits_on_the_row_it_describes(course_frame_of):
    """The holds sum to the flight, so the sentence can say when it lands without a landing row."""
    series = _Series(np.zeros(200), np.linspace(900.0, 20.0, 200), np.full(200, 75.0))
    course_frame_of(series)
    v = BoxVocabulary()
    moments, words, holds = read_boxes(series, v, runway_word=0)
    assert holds.sum() == pytest.approx(series.times[-1])
    rebuilt = np.cumsum(np.concatenate([[0.0], words[:-1, 4] * v.duration_bin_s]))
    assert np.allclose(rebuilt, moments - moments[0], atol=v.duration_bin_s)


def test_the_sha_moves_with_the_spec():
    assert BoxVocabulary().sha256 != BoxVocabulary(altitude_down_deg=2.0).sha256


def test_a_descent_angle_at_or_past_the_fleet_median_is_refused():
    with pytest.raises(ValueError, match="fleet"):
        BoxVocabulary(altitude_down_deg=3.0)
