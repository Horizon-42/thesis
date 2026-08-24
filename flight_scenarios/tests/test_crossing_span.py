"""The crossing-span builder: event → marker (+ the one inferred crossing row)."""

from __future__ import annotations

import pytest

from flight_scenarios.crossing_span import (
    FITTED_TAIL_KIND,
    MEASURED_BRACKET_KIND,
    V_SOURCE_EVENT_GROUND_SPEED,
    V_SOURCE_LAST_MEASURED,
    crossing_span_from_event,
)


def _states(n: int = 3) -> list[dict[str, float]]:
    return [
        {
            "t": float(10 * index),
            "lat": 35.0 + 0.001 * index,
            "lon": -78.0,
            "alt": 500.0 - 100.0 * index,
            "V": 72.0,
            "psi": 1.5,
            "gamma": -0.05,
            "m": 60_000.0,
        }
        for index in range(n)
    ]


def _censored_event(**overrides) -> dict:
    event = {
        "status": "estimated",
        "method": "censored_robust_line",
        "threshold_crossing_lat": 35.01,
        "threshold_crossing_lon": -78.001,
        "threshold_crossing_altitude_m": 160.0,
        "extrapolation_distance_m": 350.0,
        "crossing_ground_speed_m_s": 68.0,
        "source_sample_range": [0, 2],
        "interpolation_fraction": None,
    }
    event.update(overrides)
    return event


def test_direct_event_becomes_a_measured_bracket_marker_with_no_appended_rows():
    marker, rows = crossing_span_from_event(
        _censored_event(
            method="direct_linear_bracket",
            extrapolation_distance_m=0.0,
            interpolation_fraction=0.25,
            source_sample_range=[1, 2],
        ),
        _states(),
        hae_minus_msl_m=30.0,
    )
    assert rows == []
    assert marker == {
        "kind": MEASURED_BRACKET_KIND, "left_index": 1, "fraction": 0.25,
    }


def test_censored_event_appends_the_inferred_crossing_row_in_msl():
    states = _states()
    marker, rows = crossing_span_from_event(
        _censored_event(), states, hae_minus_msl_m=30.0
    )
    assert marker["kind"] == FITTED_TAIL_KIND
    assert marker["start_index"] == len(states)
    assert marker["v_source"] == V_SOURCE_EVENT_GROUND_SPEED
    assert marker["extrapolation_m"] == pytest.approx(350.0)
    [row] = rows
    # HAE event altitude → record MSL datum, converted exactly once, here.
    assert row["alt"] == pytest.approx(160.0 - 30.0)
    assert row["lat"] == pytest.approx(35.01)
    assert row["V"] == pytest.approx(68.0)
    # Trapezoidal timing over the deceleration: 350 m / mean(72, 68) = 5.0 s.
    assert row["t"] == pytest.approx(states[-1]["t"] + 5.0)
    # Heading/path angle carry from the last established sample.
    assert row["psi"] == pytest.approx(1.5)
    assert row["m"] == pytest.approx(60_000.0)


def test_missing_ground_speed_carries_the_last_measured_v_and_says_so():
    marker, [row] = crossing_span_from_event(
        _censored_event(crossing_ground_speed_m_s=None),
        _states(),
        hae_minus_msl_m=30.0,
    )
    assert marker["v_source"] == V_SOURCE_LAST_MEASURED
    assert row["V"] == pytest.approx(72.0)


def test_only_estimated_events_and_fitting_sample_ranges_are_accepted():
    with pytest.raises(ValueError, match="estimated"):
        crossing_span_from_event(
            _censored_event(status="unavailable"), _states(), hae_minus_msl_m=30.0
        )
    with pytest.raises(ValueError, match="sample range"):
        crossing_span_from_event(
            _censored_event(
                method="direct_linear_bracket",
                extrapolation_distance_m=0.0,
                interpolation_fraction=0.5,
                source_sample_range=[2, 3],
            ),
            _states(3),
            hae_minus_msl_m=30.0,
        )
