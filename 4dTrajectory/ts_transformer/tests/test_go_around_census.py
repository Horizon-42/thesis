"""The go-around census (`experiments/go_around_census`, multi-aircraft design §6.6 step 8 item 7): what it finds on
hand-built tracks in a runway frame, and its airport pass on a tmp harvest root (never a live one)."""

import json
import math

import numpy as np
import pytest

from final_approach import TrackPoint
from final_approach.frame import Projected, RunwayFrame

from ts_transformer.data.day_split import DaySplit

SPEED = 70.0
#: a 3° glidepath crossing the threshold 15 m up
GLIDE = math.tan(math.radians(3.0))
FRAME = RunwayFrame(ident="09", lat=35.0, lon=-78.0, elevation_m=100.0, course_deg=90.0)


def _leg(start, end, speed=SPEED):
    """(along, cross, height) points from ``start`` to ``end`` one second apart at ``speed``, ``start`` included."""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    n = max(1, int(round(float(np.hypot(*(b - a)[:2])) / speed)))
    return [tuple(a + (b - a) * k / n) for k in range(n)]


def _approach(from_along, to_along=0.0):
    return _leg((from_along, 0.0, 15.0 - GLIDE * from_along), (to_along, 0.0, 15.0 - GLIDE * to_along))


def _track(legs, frame=FRAME):
    """The legs joined end to end, each point unprojected; the last point is the landing sample."""
    rows = [p for leg in legs for p in leg]
    points = [frame.unproject(Projected(*p)) for p in rows]
    return points, [float(t) for t in range(len(points))]


def _go_around_then_land(*, low_at=-1_000.0, cross=0.0):
    """Down the 09 glidepath to ``low_at``, climb straight ahead to 700 m by 8 km past the threshold, a downwind 4 km
    north back to 12 km before it, a base onto the centreline, and the landing."""
    low_h = 15.0 - GLIDE * low_at
    return [
        _leg((-15_000.0, cross, 15.0 + GLIDE * 15_000.0), (low_at, cross, low_h)),
        _leg((low_at, cross, low_h), (8_000.0, cross, 700.0)),
        _leg((8_000.0, cross, 700.0), (8_000.0, 4_000.0, 700.0)),
        _leg((8_000.0, 4_000.0, 700.0), (-12_000.0, 4_000.0, 700.0)),
        _leg((-12_000.0, 4_000.0, 700.0), (-12_000.0, 0.0, 15.0 + GLIDE * 12_000.0)),
        _approach(-12_000.0) + [(0.0, 0.0, 15.0)],
    ]


def test_a_straight_in_landing_has_no_go_around():
    from ts_transformer.experiments.go_around_census import go_arounds

    points, times = _track([_approach(-15_000.0) + [(0.0, 0.0, 15.0)]])
    assert go_arounds(points, times, [FRAME], len(points) - 1, 600.0) == []


def test_a_go_around_is_found_at_its_lowest_sample_and_timed_to_the_landing():
    from ts_transformer.experiments.go_around_census import go_arounds

    legs = _go_around_then_land()
    points, times = _track(legs)
    landing = len(points) - 1
    found = go_arounds(points, times, [FRAME], landing, 600.0)
    assert len(found) == 1
    event = found[0]
    assert event.runway == "09"
    assert event.index == len(legs[0])                       # the first sample of the climb leg: the lowest
    assert event.along_m == pytest.approx(-1_000.0, abs=1.0)
    assert event.height_m == pytest.approx(15.0 + GLIDE * 1_000.0, abs=0.5)
    assert event.climb_m == pytest.approx(700.0 - event.height_m, abs=1.0)
    assert times[landing] - times[event.index] > 0


def test_a_takeoff_that_comes_back_to_land_is_not_a_go_around():
    from ts_transformer.experiments.go_around_census import go_arounds

    legs = [_leg((500.0, 0.0, 0.0), (8_000.0, 0.0, 700.0)), *_go_around_then_land()[2:]]
    points, times = _track(legs)
    assert go_arounds(points, times, [FRAME], len(points) - 1, 600.0) == []


def test_crossing_the_centreline_low_is_not_a_pass():
    """Down to 300 m, straight across the final 6 km out, up to 900 m: with no along-track progress rule it would be a
    go-around."""
    from ts_transformer.experiments.go_around_census import go_arounds, low_passes

    legs = [_leg((-6_000.0, -20_000.0, 900.0), (-6_000.0, -8_000.0, 300.0)),
            _leg((-6_000.0, -8_000.0, 300.0), (-6_000.0, 8_000.0, 300.0)),     # across the final at 300 m
            _leg((-6_000.0, 8_000.0, 300.0), (-12_000.0, 8_000.0, 900.0)),
            _leg((-12_000.0, 8_000.0, 900.0), (-12_000.0, 0.0, 15.0 + GLIDE * 12_000.0)),
            _approach(-12_000.0) + [(0.0, 0.0, 15.0)]]
    points, times = _track(legs)
    landing = len(points) - 1
    assert len(low_passes(points[:landing], [FRAME], 600.0)) == 1         # only the landing's own final
    assert go_arounds(points, times, [FRAME], landing, 600.0) == []


def test_a_few_samples_of_another_aircraft_are_not_a_held_climb():
    """KRDU RPA5593: samples 15 km off and 400 m up between samples on the final split its pass in two, and the stray
    samples read as a climb after the first part. A level held 20 s is a climb; five seconds of strays are not."""
    from ts_transformer.experiments.go_around_census import go_arounds, low_passes

    points, times = _track([_approach(-15_000.0) + [(0.0, 0.0, 15.0)]])
    landing = len(points) - 1
    first = int(round(9_400.0 / SPEED))                                       # 5.6 km before the threshold
    for i in range(first, first + 5):
        stray = FRAME.unproject(Projected(-10_000.0, -12_000.0, points[i].alt_m - FRAME.elevation_m + 400.0))
        points[i] = TrackPoint(stray.lat, stray.lon, stray.alt_m)
    assert len(low_passes(points[:landing], [FRAME], 600.0)) == 2            # the strays split the final
    assert go_arounds(points, times, [FRAME], landing, 600.0) == []


def test_a_dip_and_a_shallow_level_off_are_not_a_go_around():
    """Down to 400 m on the final, out of the corridor while levelling 80 m higher, back in and down to land: two passes,
    and the first one's climb is short of `MIN_CLIMB_M`."""
    from ts_transformer.experiments.go_around_census import go_arounds, low_passes

    legs = [_leg((-15_000.0, 0.0, 800.0), (-8_000.0, 0.0, 400.0)),
            _leg((-8_000.0, 0.0, 400.0), (-7_000.0, 900.0, 480.0)),         # 80 m back up, off the centreline
            _leg((-7_000.0, 900.0, 480.0), (-5_000.0, 900.0, 480.0)),
            _leg((-5_000.0, 900.0, 480.0), (-4_000.0, 0.0, 15.0 + GLIDE * 4_000.0)),
            _approach(-4_000.0) + [(0.0, 0.0, 15.0)]]
    points, times = _track(legs)
    landing = len(points) - 1
    assert len(low_passes(points[:landing], [FRAME], 600.0)) == 2
    assert go_arounds(points, times, [FRAME], landing, 600.0) == []


def test_close_parallels_count_a_go_around_once_on_the_nearer_runway():
    from ts_transformer.experiments.go_around_census import go_arounds, low_passes

    north = FRAME.unproject(Projected(0.0, -200.0, 0.0))                     # 200 m left of 09's centreline
    parallel = RunwayFrame(ident="09L", lat=north.lat, lon=north.lon, elevation_m=100.0, course_deg=90.0)
    points, times = _track(_go_around_then_land(cross=-150.0))              # flown 50 m right of 09L, landed on 09
    landing = len(points) - 1
    for frames in ([FRAME, parallel], [parallel, FRAME]):
        assert [p.runway for p in low_passes(points[:landing], frames, 600.0)] == ["09L", "09"]
        assert [event.runway for event in go_arounds(points, times, frames, landing, 600.0)] == ["09L"]


def test_the_ceiling_decides_which_passes_are_low():
    from ts_transformer.experiments.go_around_census import go_arounds

    high = _go_around_then_land(low_at=-9_500.0)                             # bottoms out ~513 m up 9.5 km out
    points, times = _track(high)
    assert len(go_arounds(points, times, [FRAME], len(points) - 1, 600.0)) == 1
    assert go_arounds(points, times, [FRAME], len(points) - 1, 500.0) == []


def _with_strays(points, first, count, along, cross, height):
    """``count`` samples from ``first`` replaced by another aircraft's, at (along, cross, height) in FRAME."""
    stray = FRAME.unproject(Projected(along, cross, height))
    return [TrackPoint(stray.lat, stray.lon, stray.alt_m) if first <= i < first + count else p
            for i, p in enumerate(points)]


def test_strays_on_the_runway_ahead_are_set_aside_before_they_become_a_go_around_point():
    """Review finding 2: three samples of the aircraft ahead rolling on the runway (600 m past the threshold, 2 m up),
    6.5 km out on a straight-in. The altitude repair cannot see a run of three; read without the strays there is none."""
    from ts_transformer.experiments.go_around_census import go_arounds, read_flight

    points, times = _track([_approach(-15_000.0) + [(0.0, 0.0, 15.0)]])
    landing = len(points) - 1
    points = _with_strays(points, int(round(8_500.0 / SPEED)), 3, 600.0, 0.0, 2.0)
    assert len(go_arounds(points, times, [FRAME], landing, 600.0)) == 1     # the strays alone make one
    reading = read_flight(points, times, [FRAME], landing, 600.0)
    assert reading.go_arounds == [] and reading.strays == 3 and not reading.landing_not_last


def test_strays_on_the_climb_out_do_not_move_the_go_around_point():
    """Review finding 3: five strays 1.5 km after the true point split the pass; read without them the point is the true
    lowest sample, 1 km before the threshold."""
    from ts_transformer.experiments.go_around_census import read_flight

    legs = _go_around_then_land()
    points, times = _track(legs)
    point = len(legs[0])
    points = _with_strays(points, point + int(round(1_500.0 / SPEED)), 5, -10_000.0, -12_000.0, 900.0)
    reading = read_flight(points, times, [FRAME], len(points) - 1, 600.0)
    assert reading.strays == 5
    assert [(event.index, round(event.along_m)) for event in reading.go_arounds] == [(point, -1_000)]


def test_a_landing_followed_by_a_held_climb_is_not_the_last():
    """A touch-and-go and away is LEFT; a low go-around the harvest took for the landing, flown round to land, is
    LANDED LATER (review finding 6: KSMF SWA1521)."""
    from ts_transformer.experiments.go_around_census import LANDED_LATER, LEFT, THE_LAST, read_flight

    pattern = _go_around_then_land()
    landing = sum(len(leg) for leg in pattern) - 1                            # the pattern's own last point
    away = [*pattern, _leg((0.0, 0.0, 15.0), (3_000.0, 0.0, 15.0)), _leg((3_000.0, 0.0, 15.0), (12_000.0, 0.0, 600.0))]
    reading = read_flight(*_track(away), [FRAME], landing, 600.0)
    assert reading.after_landing == LEFT and len(reading.go_arounds) == 1
    again = [*pattern, _leg((0.0, 0.0, 15.0), (8_000.0, 0.0, 700.0)), *_go_around_then_land()[2:],
             _leg((0.0, 0.0, 2.0), (1_500.0, 0.0, 2.0))]                     # round again, and the rollout
    points, times = _track(again)
    reading = read_flight(points, times, [FRAME], landing, 600.0)
    assert reading.after_landing == LANDED_LATER and reading.kept[-1] == len(points) - 1
    # reception ending on short final, 500 m before the threshold and ~41 m up (most KRDU tracks end so)
    short = [*pattern, _leg((0.0, 0.0, 15.0), (8_000.0, 0.0, 700.0)), *_go_around_then_land()[2:5],
             _leg((-12_000.0, 0.0, 15.0 + GLIDE * 12_000.0), (-500.0, 0.0, 15.0 + GLIDE * 500.0))]
    assert read_flight(*_track(short), [FRAME], landing, 600.0).after_landing == LANDED_LATER
    straight = _track([_approach(-15_000.0) + [(0.0, 0.0, 15.0)], _leg((0.0, 0.0, 0.0), (2_000.0, 0.0, 0.0))])
    assert read_flight(*straight, [FRAME], len(_approach(-15_000.0)), 600.0).after_landing == THE_LAST


def test_held_level_needs_the_whole_hold():
    from ts_transformer.experiments.go_around_census import HOLD_S, held_level

    times = [float(t) for t in range(60)]
    altitudes = [100.0] * 60
    altitudes[30] = 900.0                                                    # one sample: not held
    assert held_level(times, altitudes, 0, 60) == 100.0
    altitudes[20:20 + int(HOLD_S) + 1] = [500.0] * (int(HOLD_S) + 1)       # HOLD_S seconds at 500 m
    assert held_level(times, altitudes, 0, 60) == 500.0
    assert held_level(times, altitudes, 0, int(HOLD_S)) == float("-inf")    # spans less than HOLD_S
    # review finding 4: two strays either side of a reception gap span HOLD_S on two samples
    assert held_level([0.0, 1.0, 50.0, 51.0], [100.0, 900.0, 900.0, 100.0], 0, 4) == float("-inf")


def _days():
    return DaySplit(seed=1, days={"test": ("2026-05-01",), "val": ("2026-05-02",), "select": ("2026-05-03",),
                                  "train": ("2026-05-04", "2026-05-05")})


def test_only_training_day_landings_are_taken_and_an_unlisted_day_is_refused():
    from ts_transformer.experiments.go_around_census import training_landings

    roster = {"records": [
        {"flight_key": "A", "outcome": "assigned", "landing_time_utc": "2026-05-04T12:00:00Z"},
        {"flight_key": "B", "outcome": "assigned", "landing_time_utc": "2026-05-01T12:00:00Z"},   # test day
        {"flight_key": "C", "outcome": "not_landing", "landing_time_utc": None},
        {"flight_key": "D", "outcome": "assigned", "landing_time_utc": "2026-05-06T05:00:00Z"},   # 05-05 operating day
        {"flight_key": "E", "outcome": "assigned", "landing_time_utc": "2026-05-03T12:00:00Z"},   # select day
    ]}
    assert [r["flight_key"] for r in training_landings(roster, _days())] == ["A", "D"]
    roster["records"].append({"flight_key": "F", "outcome": "assigned", "landing_time_utc": "2026-06-01T12:00:00Z"})
    with pytest.raises(KeyError, match="2026-06-01"):
        training_landings(roster, _days())


def _write_track(tracks, key, legs, start="2026-05-04T11:00:00Z", strays=None):
    points, times = _track(legs)
    if strays is not None:
        points = _with_strays(points, *strays)
    samples = [[t, p.lon, p.lat, p.alt_m] for t, p in zip(times, points)]
    record = {"flight_key": key, "start_time_utc": start, "landing_sample_index": len(samples) - 1, "samples": samples,
              "reported_ground_speeds_m_s": [SPEED] * len(samples)}
    path = tracks / "assigned" / "09" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return {"flight_key": key, "file": f"assigned/09/{key}.json", "outcome": "assigned", "runway": "09",
            "landing_time_utc": "2026-05-04T12:00:00Z"}, len(_go_around_then_land()[0])


def test_the_airport_pass_places_each_go_around_against_the_slice_and_the_sentence(tmp_path):
    from ts_transformer.experiments.go_around_census import AirportInputs, census_airport, summarise

    tracks = tmp_path / "KXXX" / "tracks"
    point = len(_go_around_then_land()[0])
    # five strays 15 km out on GA1's climb-out: read without them, distances included
    around, _ = _write_track(tracks, "GA1", _go_around_then_land(),
                             strays=(point + int(round(1_500.0 / SPEED)), 5, -10_000.0, -12_000.0, 900.0))
    later, _ = _write_track(tracks, "GA2", _go_around_then_land())
    straight, _ = _write_track(tracks, "SI1", [_approach(-15_000.0) + [(0.0, 0.0, 15.0)]])
    circuit = _go_around_then_land(low_at=300.0)
    circuit[0] = _leg((-15_000.0, 0.0, 15.0 + GLIDE * 15_000.0), (300.0, 0.0, 2.0))   # down to the runway
    touch, _ = _write_track(tracks, "TG1", circuit)
    inputs = AirportInputs(
        "KXXX", tmp_path, (FRAME,), (FRAME.lat, FRAME.lon), (around, later, straight, touch),
        # GA1's slice starts before its go-around, GA2's after it
        {"GA1": 0, "GA2": point + 10, "SI1": 0},
        # GA1 labelled from the track's start for long enough; GA2 labelled from after the go-around; SI1 refused
        {"GA1": ("labelled", "2026-05-04T11:00:00Z", 2_000), "GA2": ("labelled", "2026-05-04T11:10:00Z", 50),
         "SI1": ("refused", "2026-05-04T11:00:00Z", None)},
        2.0, 600.0)
    result = census_airport(inputs)
    assert result["landings"] == 4 and result["stray_samples"] == 5
    assert result["landings_not_the_last"] == {"landed later": [], "left": []}
    rows = {row["flight_key"]: row for row in result["go_arounds"]}
    assert set(rows) == {"GA1", "GA2", "TG1"}
    assert rows["TG1"]["kind"] == "on the runway" and rows["TG1"]["artefact"] == "absent"
    first = rows["GA1"]
    assert first["kind"] == "go-around"
    assert first["index"] == point and first["in_arrival_slice"] and first["in_sentence"]
    assert first["artefact"] == "labelled" and first["same_runway"]
    assert first["time_utc"] == f"2026-05-04T11:{point // 60:02d}:{point % 60:02d}Z"
    assert first["still_needed_s"] == pytest.approx(1_000.0 / SPEED, abs=0.1)
    assert first["cost_s"] == pytest.approx(first["to_landing_s"] - first["still_needed_s"])
    assert first["farthest_km"] == pytest.approx(math.hypot(12.0, 4.0), abs=0.02)   # the downwind's far corner
    assert not rows["GA2"]["in_arrival_slice"] and not rows["GA2"]["in_sentence"]

    summary = summarise(result["go_arounds"], result["landings"], 600.0)
    assert summary["go_arounds"] == 2 and summary["per_1000_landings"] == pytest.approx(500.0)
    assert summary["flights_with_a_go_around"] == 2 and summary["flights_with_several"] == 0
    assert summary["standing"] == {
        "labelled, inside the sentence, inside the arrival slice": 1,
        "labelled, outside the sentence, outside the arrival slice": 1}
    assert summary["set_aside"] == {"on the runway: absent": 1}
    assert set(summary["by"]["height"]) == {"<=300 m"}


def test_height_bands_stop_at_the_runs_ceiling():
    from ts_transformer.experiments.go_around_census import height_band

    assert height_band(250.0, 600.0) == "<=300 m"
    assert height_band(450.0, 600.0) == "300-600 m"
    assert height_band(750.0, 900.0) == "600-900 m"
    assert height_band(450.0, 500.0) == "300-500 m"
    with pytest.raises(ValueError):
        height_band(650.0, 600.0)


def test_the_artefact_index_reads_labelled_and_refused_flights_per_airport(monkeypatch):
    from ts_transformer.experiments import go_around_census as census

    meta = [{"dataset_id": "KXXX:A", "airport": "KXXX", "entry_time_utc": "t0"},
            {"dataset_id": "KYYY:B", "airport": "KYYY", "entry_time_utc": "t1"},
            {"dataset_id": "KXXX:C", "airport": "KXXX", "entry_time_utc": "t2"}]
    monkeypatch.setattr(census, "load_spec", lambda directory: object())
    monkeypatch.setattr(census, "signals_flights", lambda directory, split: meta)
    monkeypatch.setattr(census, "load_sentences", lambda directory, split, spec: {
        "offsets": np.array([0, 7, 12]), "signal_index": np.array([1, 2])})
    index = census.artefact_index(None, ["KXXX"])
    assert index == {"KXXX": {"A": ("refused", "t0", None), "C": ("labelled", "t2", 5)}}


def test_the_slack_mirrors_the_free_generation_time_limit():
    from types import SimpleNamespace

    from ts_transformer.experiments import prior_free_generation as free
    from ts_transformer.experiments.go_around_census import limit_s, observed_remaining_s, slack

    for words in (9, 40, 251):
        reading = SimpleNamespace(words=[None] * words)
        assert observed_remaining_s(words, 2.0) == free.observed_remaining_s(reading, 2.0)
        assert limit_s(words, 2.0, 1.5) == free.limits_s(SimpleNamespace(readings=[reading]),
                                                         SimpleNamespace(timeout_factor=1.5), 2.0, augmented=False)[0]
    # 108 rows: 198 s observed from the first predicted step, a 300 s limit, 102 s of slack; 408 rows: 798, 1200, 402
    result = slack([108, 408], 2.0, 1.5, {"pooled": [50.0, 150.0, 250.0]})
    assert result["slack_s"]["min"] == pytest.approx(102.0) and result["slack_s"]["max"] == pytest.approx(402.0)
    assert result["covers"]["pooled"]["covers_p50"] == 0.5 and result["covers"]["pooled"]["covers_p95"] == 0.5


def test_main_writes_the_census_and_refuses_a_foreign_manifest_and_an_existing_directory(tmp_path, monkeypatch):
    """The runner end to end on a tmp harvest of KRDU (the airport's real runway ends, synthetic tracks on 05L's final);
    the artefact and the executor spec are stood in for."""
    from types import SimpleNamespace

    from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
    from trajectory_data_process.harvest.airports import load_airport
    from ts_transformer.experiments import go_around_census as census
    from ts_transformer.io_utils import file_sha256

    frame = load_airport("KRDU", config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP).runway("05L").frame("hae")
    tracks = tmp_path / "KRDU" / "tracks"
    records = []
    for key, legs in (("GA1_05L", _go_around_then_land()), ("SI1_05L", [_approach(-15_000.0) + [(0.0, 0.0, 15.0)]])):
        points, times = _track(legs, frame)
        path = tracks / "assigned" / "05L" / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"start_time_utc": "2026-05-04T11:00:00Z", "landing_sample_index": len(points) - 1,
                                    "samples": [[t, p.lon, p.lat, p.alt_m] for t, p in zip(times, points)],
                                    "reported_ground_speeds_m_s": [SPEED] * len(points)}), encoding="utf-8")
        records.append({"flight_key": key, "file": f"assigned/05L/{key}.json", "outcome": "assigned", "runway": "05L",
                        "landing_time_utc": "2026-05-04T12:00:00Z"})
    records.append({"flight_key": "SEALED", "file": "assigned/05L/absent.json", "outcome": "assigned", "runway": "05L",
                    "landing_time_utc": "2026-05-01T12:00:00Z"})                 # a test day: never opened
    (tracks / "manifest.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    arrivals = tmp_path / "KRDU" / "arrivals" / "manifest.json"
    arrivals.parent.mkdir(parents=True)
    arrivals.write_text(json.dumps({"records": [{"flight_key": "GA1_05L", "first_sample_index": 0}]}), encoding="utf-8")

    recorded = {"KRDU": file_sha256(arrivals)}
    monkeypatch.setattr(census, "arrival_manifest_sha256s", lambda directory: recorded)
    monkeypatch.setattr(census, "load_day_split", lambda directory: _days())
    monkeypatch.setattr(census, "load_spec", lambda directory: SimpleNamespace(step_s=2.0))
    monkeypatch.setattr(census, "signals_flights", lambda directory, split: [
        {"dataset_id": "KRDU:GA1_05L", "airport": "KRDU", "entry_time_utc": "2026-05-04T11:00:00Z"}])
    monkeypatch.setattr(census, "load_sentences", lambda directory, split, spec: {
        "offsets": np.array([0, 1_500]), "signal_index": np.array([0])})
    monkeypatch.setattr(census, "load_executor_spec", lambda directory: (SimpleNamespace(timeout_factor=1.5),
                                                                         {"sha256": "e" * 64}))
    argv = ["--instructions", str(tmp_path / "artefact"), "--executor", str(tmp_path / "executor"),
            "--harvest-root", str(tmp_path), "--workers", "1"]
    assert census.main([*argv, "--out", str(tmp_path / "out")]) == 0
    payload = json.loads((tmp_path / "out" / "go_arounds.json").read_text(encoding="utf-8"))
    assert payload["schema"] == census.SCHEMA and payload["read"]["KRDU"]["train_days_assigned"] == 2
    assert [(r["flight_key"], r["kind"], r["in_sentence"]) for r in payload["go_arounds"]] == [
        ("GA1_05L", "go-around", True)]
    assert payload["pooled"]["go_arounds"] == 1 and payload["pooled"]["time_limit"]["sentences"] == 1
    assert payload["criteria"]["max_height_m"] == census.MAX_HEIGHT_M

    with pytest.raises(SystemExit):
        census.main([*argv, "--out", str(tmp_path / "out")])                    # never overwritten
    recorded["KRDU"] = "0" * 64
    with pytest.raises(ValueError, match="not the one the artefact"):
        census.main([*argv, "--out", str(tmp_path / "out2")])
