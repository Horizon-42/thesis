"""The shared crossing interpolation and the crossing-span marker contract."""

from __future__ import annotations

import math

import pytest

from final_approach.crossing import (
    FITTED_TAIL_KIND,
    MEASURED_BRACKET_KIND,
    bracket_fraction,
    interpolate_channels,
    validate_crossing_span,
)


def test_bracket_fraction_is_the_zero_crossing_of_the_along_coordinate():
    assert bracket_fraction(-100.0, 100.0) == pytest.approx(0.5)
    assert bracket_fraction(-30.0, 90.0) == pytest.approx(0.25)
    assert bracket_fraction(0.0, 50.0) == 0.0


@pytest.mark.parametrize("before, after", [(10.0, 100.0), (-100.0, -1.0), (5.0, 5.0)])
def test_bracket_fraction_refuses_a_non_bracketing_segment(before, after):
    with pytest.raises(ValueError, match="bracket"):
        bracket_fraction(before, after)


def test_interpolate_channels_blends_linearly_and_wraps_angles_on_the_short_arc():
    before = {"alt": 100.0, "psi": math.pi - 0.1, "V": 70.0}
    after = {"alt": 90.0, "psi": -math.pi + 0.1, "V": 68.0}
    blended = interpolate_channels(
        before, after, 0.5, keys=("alt", "psi", "V"), angular_keys=("psi",)
    )
    assert blended["alt"] == pytest.approx(95.0)
    assert blended["V"] == pytest.approx(69.0)
    # Short arc across the ±π seam — a plain blend would return 0 (the long way).
    assert blended["psi"] == pytest.approx(math.pi, abs=1e-12)


def test_interpolate_channels_rejects_bad_fraction_and_unknown_angular_keys():
    state = {"psi": 0.0}
    with pytest.raises(ValueError, match="fraction"):
        interpolate_channels(state, state, 1.5, keys=("psi",))
    with pytest.raises(ValueError, match="angular"):
        interpolate_channels(state, state, 0.5, keys=("psi",), angular_keys=("gamma",))


def test_validate_crossing_span_accepts_both_kinds_and_names_the_offence():
    assert validate_crossing_span(
        {"kind": MEASURED_BRACKET_KIND, "left_index": 3, "fraction": 0.4}, 10
    ) == MEASURED_BRACKET_KIND
    assert validate_crossing_span(
        {"kind": FITTED_TAIL_KIND, "start_index": 9, "v_source": "x"}, 10
    ) == FITTED_TAIL_KIND

    with pytest.raises(ValueError, match="left_index"):
        validate_crossing_span(
            {"kind": MEASURED_BRACKET_KIND, "left_index": 9, "fraction": 0.4}, 10
        )
    with pytest.raises(ValueError, match="fraction"):
        validate_crossing_span(
            {"kind": MEASURED_BRACKET_KIND, "left_index": 3, "fraction": 0.0}, 10
        )
    # Exactly one appended row is the current contract.
    with pytest.raises(ValueError, match="exactly one"):
        validate_crossing_span({"kind": FITTED_TAIL_KIND, "start_index": 8}, 10)
    with pytest.raises(ValueError, match="start_index"):
        validate_crossing_span({"kind": FITTED_TAIL_KIND, "start_index": 0}, 1)
    with pytest.raises(ValueError, match="kind"):
        validate_crossing_span({"kind": "surprise"}, 10)
