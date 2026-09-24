"""Cut verbatim from `tests/test_autopilot.py` (2026-09-24): the test of method A's flown checks (`autopilot/derive.py`,
archived beside it). Off the import path; nothing here runs."""


def test_method_a_takes_the_lead_for_the_executors_own_turns_and_the_gentlest_roll_rate_the_checks_allow():
    from dataclasses import replace

    from ts_transformer.autopilot import derive

    one = spec()
    # the executor's own turns settle on the lead a word gives
    assert derive.heading_time_constant_s(one, 1.0) == one.heading_lead_s == 4.0
    with pytest.raises(ValueError, match="under 2 Δt"):
        derive.heading_time_constant_s(spec(heading_lead_s=0.0), 1.0)
    params = _params(heading_time_constant_s=4.0)
    largest = derive.largest_own_turn_deg(one)
    assert largest == 90.0 + one.heading_tolerance_deg + one.intercept_angle_deg
    slow, fast = (derive.turn_overshoot_deg(replace(params, bank_rate_deg_s=p), 140.0, largest) for p in (1.0, 5.0))
    assert slow > one.heading_tolerance_deg > fast >= 0.0                        # a slower roll passes further
    # a slow roll-in lags the first words of a worded turn past their envelope; a brisk one follows them
    steady = derive.follow_rates_deg_s(params, one, 140.0)["steady"]
    assert derive.follow_excess_deg(replace(params, bank_rate_deg_s=1.0), one, 140.0, steady, rolled=False) > 0.0
    assert derive.follow_excess_deg(replace(params, bank_rate_deg_s=5.0), one, 140.0, steady, rolled=False) <= 0.0
    rate, measures = derive.roll_rate_deg_s(params, one)
    assert set(measures) == {"overshoot_deg", "steady_excess_deg", "fastest_excess_deg"}
    assert all(len(values) == len(derive.ROLL_CHECK_SPEEDS_MPS) for values in measures.values())
    assert max(measures["overshoot_deg"].values()) <= one.heading_tolerance_deg
    assert max(measures["steady_excess_deg"].values()) <= 0.0 and max(measures["fastest_excess_deg"].values()) <= 0.0
    less = replace(params, bank_rate_deg_s=rate - derive.ROLL_RATE_STEP_DEG_S)
    assert rate == derive.ROLL_RATE_STEP_DEG_S or not derive.roll_checks(less, one, stop_at_first_failure=True)[0]
