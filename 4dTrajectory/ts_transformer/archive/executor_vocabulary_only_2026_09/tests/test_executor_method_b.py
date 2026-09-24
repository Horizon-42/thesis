"""Cut verbatim from `tests/test_autopilot.py` (2026-09-24): the tests of method B (`autopilot/observe.py`) and the
helper they used (`_observed_series`, which stays live there too). Off the import path; nothing here runs."""


def test_a_received_word_is_matched_to_the_nearest_reread_word_of_its_column_and_value():
    from ts_transformer.autopilot.observe import word_leads

    received = [(HEADING, 7, 100.0), (ALTITUDE, 3, 200.0), (SPEED, 5, 300.0)]
    reread = [(HEADING, 7, 96.0), (HEADING, 7, 150.0), (HEADING, 8, 100.0), (ALTITUDE, 3, 206.0), (SPEED, 5, 400.0)]
    assert word_leads(received, reread, 30.0) == [(HEADING, 4.0), (ALTITUDE, -6.0)]


def _observed_series(geometry):
    from types import SimpleNamespace

    from ts_transformer.data.coordinate_frames import ENUFrame
    from ts_transformer.data.dataset import FlightSeries

    candidate = geometry.candidates[0]
    lat, lon = geometry.frame.latlon_from_horizontal(candidate.threshold_e_m, candidate.threshold_n_m)
    scenario = SimpleNamespace(source={"arr_airport": "KXXX", "runway": "09", "resolved_typecode": "A320"},
                               initial=SimpleNamespace(m=62000.0),
                               target=SimpleNamespace(latitude=lat, longitude=lon, psi=0.0), aircraft=SimpleNamespace(code="A320"))
    return FlightSeries(flight_id="TEST1", scenario=scenario, frame=ENUFrame(lat0=lat, lon0=lon, alt0=candidate.elevation_m),
                        times=np.zeros(1), values=np.zeros((1, 6)))


def test_the_observation_operator_reads_a_flown_track_as_the_data_plane_reads_a_radar_track():
    """Method B's O: the flown track, sampled once a cycle, through the data plane's own fit and grid, reads
    back as the track that was flown, and the labeller re-reads its words from it: the descent LATER than the
    executor began it (the reading needs height lost before it sees a descent — method B's finding, flown with no
    delay). Heading words are not measured: each carries its own lead (§10.1)."""
    from ts_transformer.autopilot.observe import flight_leads, observe
    from ts_transformer.autopilot.judge import flown_track

    one, words, geometry = spec(), Words(spec()), instruction_airport()
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    flown, verdict, reading = _fly_sentence(signals)
    states = flown.states[0, : verdict.end_row + 1].numpy()
    seen = observe(states, 1.0, _observed_series(geometry), geometry, one.step_s)
    truth = flown_track(states, geometry)
    rows = (seen.time_s / 1.0).astype(int)
    assert len(seen.time_s) == len(states[::2]) and seen.time_s[1] - seen.time_s[0] == one.step_s
    assert np.abs(seen.e_m - truth["e"][rows]).max() < 5.0 and np.abs(seen.altitude_m - truth["height"][rows]).max() < 2.0
    leads, received = flight_leads(states, flown.sentence_s[0].numpy(), 1.0, _observed_series(geometry), geometry,
                                   reading, one, words, 30.0)
    said = [i for i in reading.instructions if i.row > 0 and i.column in (ALTITUDE, ANGLE, SPEED)]
    assert sum(received.values()) == len(said) == 5 and HEADING not in received
    by_column = dict(leads)
    assert by_column[ALTITUDE] < 0.0 and by_column[ANGLE] < 0.0


def test_method_bs_delays_are_the_groups_median_leads_floored_at_zero():
    from ts_transformer.autopilot.observe import delays_from_leads

    leads = {ALTITUDE: [-6.0, -4.0], ANGLE: [-6.0], SPEED: [-9.0, 3.0, -8.0]}
    assert delays_from_leads(leads) == Delays(vertical_s=0.0, speed_s=0.0)
    assert delays_from_leads({**leads, ALTITUDE: [5.0, 7.0], ANGLE: [6.0]}).vertical_s == 6.0
    with pytest.raises(ValueError, match="matched no word of speed_s"):
        delays_from_leads({**leads, SPEED: []})
