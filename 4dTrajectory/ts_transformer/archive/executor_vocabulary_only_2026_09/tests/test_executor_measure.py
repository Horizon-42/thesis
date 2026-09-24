"""Cut verbatim from `tests/test_autopilot.py` (2026-09-24, stage 2): the tests of the executor's data measurement
(`autopilot/measure.py`, archived beside it). Off the import path; nothing here runs."""


def test_the_data_parameters_are_read_from_the_labellers_reading_of_each_flight():
    from ts_transformer.autopilot import measure
    from ts_transformer.instructions.labeller.read import admit, read_flight

    one, words, geometry = spec(), Words(spec()), instruction_airport()
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    reading = read_flight(signals, geometry, one, words)
    out = measure.flight_measurements(admit(signals, geometry, one), reading, geometry, one, words)
    # the legs change speed in a step: no transition lasts the MIN_SPAN_S a rate is read from
    assert out["transition_accel_mps2"] == []
    # the final is a straight 3° line, so the extrapolated crossing is the line's height at the threshold
    height = signals.altitude_m[-1] - geometry.candidates[0].elevation_m
    assert out["crossing_height_m"] == [pytest.approx(height - 400.0 * math.tan(math.radians(3.0)), abs=0.5)]
    # the measured sentence must be the stored one
    stored = [(reading.words, reading.runway_index)]
    measure.measure_flights([signals], stored, one, words, {"KXXX": geometry})
    changed = reading.words.copy()
    changed[1, SPEED] = words.speed_unspecified
    with pytest.raises(ValueError, match="differs from the stored one"):
        measure.measure_flights([signals], [(changed, 0)], one, words, {"KXXX": geometry})


def test_the_data_parameters_are_the_pooled_medians_rounded_as_the_spec_says():
    from ts_transformer.autopilot import measure

    one = spec()
    pooled = {"transition_accel_mps2": [-0.3, -0.24, -0.2, 0.1, 0.16, 0.5], "unspecified_slope_mps2": [-0.26, -0.3, -0.2],
              "crossing_height_m": [16.0, 20.84, 26.0]}
    measured = measure.measured_values(pooled, one)
    assert measured["values"] == {"decel_mps2": 0.24, "accel_mps2": 0.16, "unspecified_decel_mps2": 0.26,
                                  "land_aim_height_m": 20.8, "land_window_low_m": 16.5, "land_window_high_m": 25.5}
    assert measured["counts"]["decelerations"] == measured["counts"]["accelerations"] == 3
