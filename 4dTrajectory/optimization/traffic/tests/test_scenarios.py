"""The scenario census (design §10.6): a record judged as the flown track, and the catalog's lists."""

import numpy as np
import pytest

from traffic.check import FlownTrack
from traffic.frame import TargetFrame
from traffic.scene import RecordedFlight

import traffic_scenarios as ts
from aerodynamic_model.common import GeodeticState


def _window(own, recorded=()):
    from traffic.check import Window
    frame = TargetFrame(GeodeticState(35.0, -78.0, 100.0, 70.0, 0.0, 0.0, 60000.0))
    return Window(flight_key=own.flight_key, t0_utc_s=own.start_utc_s, frame=frame, runways={}, rules=None,
                  runway=own.runway, category=None, recorded=tuple(recorded), categories=(None,) * len(recorded),
                  uncategorised_types=())


def test_a_record_judged_as_flown_has_the_states_it_has_as_a_neighbour():
    """At every check instant the flown track's state equals the same record's state as a recorded aircraft
    (a turning record sampled at fractional UTC seconds: a straight line would agree under any differencing)."""
    from traffic.check import _record_state, check_times
    t = 1_800_000_000.437 + np.arange(0.0, 120.0, 0.97)
    turn = np.radians(np.linspace(0.0, 120.0, len(t)))                     # a 120° turn
    record = RecordedFlight("A", "09", "A320", t, 35.0 + 0.002 * np.cumsum(np.cos(turn)) / 10,
                            -78.2 + 0.002 * np.cumsum(np.sin(turn)) / 10, np.linspace(1500.0, 900.0, len(t)))
    window = _window(record)
    flown = FlownTrack.from_record(window, record, 1.0)
    times = check_times(window, record.end_utc_s - window.t0_utc_s, 1.0)
    assert np.array_equal(flown.t_s, times)
    for t_s in times:
        lat, lon, alt, track = flown.at(t_s)
        n, e = window.frame.to_ne(lat, lon)
        assert (float(n), float(e), alt, track) == pytest.approx(_record_state(window.frame, record, window.t0_utc_s + t_s),
                                                                 abs=1e-9)


def test_judge_record_counts_only_the_instants_it_answers_for(monkeypatch):
    from types import SimpleNamespace
    from traffic import rules
    from traffic.check import Check, Conflict
    own = RecordedFlight("A", "09", "A320", np.array([0.0, 10.0]), np.zeros(2), np.zeros(2), np.zeros(2))
    window = _window(own, recorded=(own,))
    loss = dict(kind=rules.RADAR_OR_VERTICAL, required_m=5556.0, vertical_m=0.0, above=True)
    conflicts = (Conflict(1.0, 0, distance_m=3000.0, responsible=True, **loss),
                 Conflict(1.0, 0, distance_m=3000.0, responsible=True, **loss),     # the same instant twice
                 Conflict(2.0, 0, distance_m=4000.0, responsible=True, **loss),
                 Conflict(3.0, 0, distance_m=1000.0, responsible=False, **loss))    # not its to answer for
    monkeypatch.setattr(ts, "make_window", lambda *a, **k: window)
    monkeypatch.setattr(ts, "check", lambda w, flown, *, reading, step_s: Check(np.zeros(1), conflicts, 0))
    scenario = SimpleNamespace(traffic=SimpleNamespace(flight=lambda key: own))
    row = ts.judge_record((scenario, "root", 2000.0, 1.0))
    assert row == {"flightKey": "A", "lossInstants": 2, "kinds": [rules.RADAR_OR_VERTICAL],
                   "tightest": pytest.approx(3000.0 / 5556.0), "recordedAircraft": 1}


def _arrival(key, runway, landing):
    return {"flightKey": key, "callsign": key, "runway": runway, "type": "A320", "entryUtc": landing,
            "landingUtc": landing}


def test_the_catalog_lists_every_arrival_with_a_loss_and_every_block_by_its_losses():
    arrivals = [_arrival("A", "05L", "2026-05-01T10:05:00Z"), _arrival("B", "05R", "2026-05-01T10:20:00Z"),
                _arrival("C", "05L", "2026-05-01T11:10:00Z"), _arrival("NODYN", "32", "2026-05-01T11:50:00Z")]
    judged = {"A": {"flightKey": "A", "lossInstants": 3, "kinds": ["in_trail"], "tightest": 0.8, "recordedAircraft": 4},
              "B": {"flightKey": "B", "lossInstants": 7, "kinds": ["radar_or_vertical"], "tightest": 0.6,
                    "recordedAircraft": 5},
              "C": {"flightKey": "C", "lossInstants": 0, "kinds": [], "tightest": None, "recordedAircraft": 2}}
    lists = ts.build_catalog(arrivals, judged)
    assert [r["flightKey"] for r in lists["m1"]] == ["B", "A"]                    # by loss instants; C has none
    assert lists["m1"][0]["runway"] == "05R" and lists["m1"][0]["lossInstants"] == 7
    hours = lists["m2"]["3600"]
    assert [(b["startUtc"], b["arrivals"], b["commandable"], b["lossInstants"]) for b in hours] == [
        ("2026-05-01T10:00:00Z", 2, 2, 10), ("2026-05-01T11:00:00Z", 2, 1, 0)]
    assert hours[1]["runways"] == ["05L", "32"]
    quarters = lists["m2"]["900"]
    assert [b["startUtc"] for b in quarters][:2] == ["2026-05-01T10:15:00Z", "2026-05-01T10:00:00Z"]
    assert set(lists["m2"]) == {"900", "1800", "3600"}
