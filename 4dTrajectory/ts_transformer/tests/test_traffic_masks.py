"""The closed loop's two separation masks (`inference/separation_masks`, multi-aircraft design §3.4 layer 2) on hand-built
aircraft, and the step-6 runner that measures them on labelled words (`experiments/traffic_masks`) on a tmp artefact."""

import json

import numpy as np
import pytest
from geokit import NM_M

from ts_transformer.inference.runway_schedule import CWT_ON_APPROACH_NM, DEPENDENT, SINGLE, Separation
from ts_transformer.inference.separation import Traffic
from ts_transformer.inference.separation_masks import (
    ahead, clearance_check, ramp_distance_m, ramp_time_s, speed_check,
)

#: Runway "R", a pair "S1"/"S2" separated as one, a dependent pair "L1"/"L2", all thresholds at the clock's origin.
SEPARATION = Separation(same_nm=3.0, speed_mps=70.0,
                        relations={frozenset(("S1", "S2")): SINGLE, frozenset(("L1", "L2")): DEPENDENT},
                        spacing_nm={frozenset(("S1", "S2")): 0.1, frozenset(("L1", "L2")): 0.6},
                        right_nm={("S1", "S2"): -0.1, ("S2", "S1"): 0.1, ("L1", "L2"): -0.6, ("L2", "L1"): 0.6},
                        diagonal_nm={frozenset(("L1", "L2")): 1.0}, wake_nm=CWT_ON_APPROACH_NM,
                        along_nm={"R": 0.0, "S1": 0.0, "S2": 0.0, "L1": 0.0, "L2": 0.0})
PACE = 0.25
FIVE_NM = 5.0 * NM_M


def traffic(*aircraft) -> Traffic:
    """Each aircraft: (approach clock m — its east, on its centreline, flying its course —, runway, established,
    category)."""
    along, runway, established, category = zip(*aircraft)
    along = np.array(along, dtype=float)
    return Traffic(along, np.zeros(len(along)), np.full(len(along), 300.0), tuple(runway), along, np.zeros(len(along)),
                   np.zeros(len(along)), np.array(established, dtype=bool), tuple(category))


@pytest.mark.parametrize("speed, target", [(80.0, 70.0), (60.0, 75.0), (70.0, 70.0)])
def test_the_ramp_is_a_constant_pace_to_the_target_and_its_time_inverts_its_distance(speed, target):
    ramp_s = abs(target - speed) / PACE
    ramp_m = 0.5 * (speed + target) * ramp_s
    assert float(ramp_distance_m(ramp_s, speed, np.array([target]), PACE)[0]) == pytest.approx(ramp_m)
    beyond = float(ramp_distance_m(ramp_s + 10.0, speed, np.array([target]), PACE)[0])
    assert beyond == pytest.approx(ramp_m + 10 * target)
    for distance in (0.3 * ramp_m + 100.0, ramp_m + 700.0):
        t = ramp_time_s(distance, speed, target, PACE)
        assert float(ramp_distance_m(t, speed, np.array([target]), PACE)[0]) == pytest.approx(distance)
    assert ramp_time_s(3_700.0, 80.0, 70.0, PACE) == pytest.approx(40.0 + 700.0 / 70.0)


def test_the_one_ahead_is_the_nearest_eligible_on_one_runway():
    scene = traffic((-10_000.0, "S1", True, "F"), (-6_000.0, "S1", True, "F"), (-3_000.0, "S1", False, "F"),
                    (-8_000.0, "S2", True, "F"), (-9_000.0, "L1", True, "F"), (-12_000.0, "S1", True, "F"))
    established = np.asarray(scene.established, dtype=bool)
    # S2 is one runway with S1; L1 is not; 2 is not eligible
    assert ahead(scene, 0, SEPARATION, established) == 3
    assert ahead(scene, 0, SEPARATION, np.ones(6, dtype=bool)) == 3
    assert ahead(scene, 1, SEPARATION, established) is None
    assert ahead(scene, 1, SEPARATION, np.ones(6, dtype=bool)) == 2
    assert ahead(scene, 4, SEPARATION, established) is None           # nothing on L1 or L2 ahead of it
    assert ahead(traffic((-10_000.0, "R", True, "F"), (-8_000.0, "S2", True, "F")), 0, SEPARATION,
                 np.ones(2, dtype=bool)) is None                      # R and S2 are not one runway


def test_a_speed_word_is_masked_when_it_closes_the_gap_under_the_minimum_at_the_crossing():
    """The one ahead is 8 km out at 70 m/s holding 70: it crosses in 114.3 s. From 14 km at 80 m/s: holding 80 closes to
    4.86 km (under 3 NM), 70 leaves 5.80 km, 60 leaves 6.34 km, 90 less than 80."""
    scene = traffic((-8_000.0, "R", True, "F"), (-14_000.0, "R", True, "F"))
    speeds, targets = np.array([70.0, 80.0]), np.array([70.0, 80.0])
    check = speed_check(scene, 1, SEPARATION, speeds, targets, np.array([60.0, 70.0, 80.0, 90.0]), 2, PACE, FIVE_NM)
    assert check.leader == 0 and check.required_m == pytest.approx(3.0 * NM_M)
    assert check.gaps_m[:3] == pytest.approx([14_000.0 - 5_600.0 - 34.2857 * 60.0, 14_000.0 - 3_000.0 - 74.2857 * 70.0,
                                              14_000.0 - 80.0 * 8_000.0 / 70.0], abs=1.0)
    assert check.allowed.tolist() == [True, True, False, False] and not check.unchanged_allowed
    # with 70 in force it may stay
    assert speed_check(scene, 1, SEPARATION, speeds, targets, np.array([60.0, 70.0, 80.0, 90.0]), 1, PACE,
                       FIVE_NM).unchanged_allowed


def test_the_speed_mask_waits_for_both_to_be_established_and_stops_at_five_miles():
    words = np.array([60.0, 70.0, 80.0])

    def check(*aircraft):
        return speed_check(traffic(*aircraft), 1, SEPARATION, np.array([70.0, 80.0]), np.array([70.0, 80.0]), words, 2,
                           PACE, FIVE_NM)

    assert check((-8_000.0, "R", True, "F"), (-14_000.0, "R", False, "F")) is None          # it is not established
    assert check((-8_000.0, "R", False, "F"), (-14_000.0, "R", True, "F")) is None          # the one ahead is not
    assert check((-3_000.0, "R", True, "F"), (-9_000.0, "R", True, "F")) is None            # inside 5 NM
    assert check((-3_000.0, "R", True, "F"), (-9_300.0, "R", True, "F")) is not None        # outside it
    # behind a heavy (B → F: TBL 5-5-2 5 NM) the required distance is the wake one
    heavy = check((-8_000.0, "R", True, "B"), (-14_000.0, "R", True, "F"))
    assert heavy.required_m == pytest.approx(5.0 * NM_M) and heavy.fallback and heavy.allowed.all()


def test_a_clearance_is_masked_while_the_nearest_cleared_one_ahead_is_under_the_in_trail_minimum():
    scene = traffic((-8_000.0, "R", True, "F"), (-13_000.0, "R", False, "F"), (-9_000.0, "R", False, "F"))
    cleared = np.array([True, False, False])
    gate = clearance_check(scene, 1, SEPARATION, cleared)
    assert (gate.leader, gate.gap_m, gate.allowed) == (0, 5_000.0, False)
    assert gate.required_m == pytest.approx(3.0 * NM_M)
    # the one between them is not cleared: it does not count; 6 km behind the cleared one is enough
    assert clearance_check(traffic((-8_000.0, "R", True, "F"), (-14_000.0, "R", False, "F")), 1, SEPARATION,
                           np.array([True, False])).allowed
    # directly behind a heavy the minimum is TBL 5-5-1's 5 NM
    assert not clearance_check(traffic((-8_000.0, "R", True, "B"), (-16_000.0, "R", False, "F")), 1, SEPARATION,
                               np.array([True, False])).allowed
    assert clearance_check(scene, 0, SEPARATION, cleared) is None                          # nothing cleared ahead of it


def test_in_a_fallback_nothing_is_masked():
    """A B 2 km out holding 70 crosses in 28.6 s; an F 11 km out at 85 m/s needs 5 NM behind it and no word gives it (the
    best would be the slowest): every word stays, "unchanged" too (design §9 item 20)."""
    scene = traffic((-2_000.0, "R", True, "B"), (-11_000.0, "R", True, "F"))
    words = np.arange(20.0, 255.0, 5.0)
    check = speed_check(scene, 1, SEPARATION, np.array([70.0, 85.0]), np.array([70.0, 70.0]), words,
                        int(np.flatnonzero(words == 70.0)[0]), PACE, FIVE_NM)
    assert check.fallback and check.unchanged_allowed and check.allowed.all()
    assert check.gaps_m.max() < check.required_m


def test_the_leader_slowing_to_its_own_target_closes_the_gap():
    """From 14 km at 70 m/s holding 70 behind one 8 km out at 70: 6 km at its crossing if it holds 70 (allowed), 4.9 km
    if its word in force is 60 (ramp 40 s over 2.6 km, then 60 m/s: crossing at 130 s)."""
    scene = traffic((-8_000.0, "R", True, "F"), (-14_000.0, "R", True, "F"))
    holding = speed_check(scene, 1, SEPARATION, np.array([70.0, 70.0]), np.array([70.0, 70.0]), np.array([70.0]), 0,
                          PACE, FIVE_NM)
    slowing = speed_check(scene, 1, SEPARATION, np.array([70.0, 70.0]), np.array([60.0, 70.0]), np.array([70.0]), 0,
                          PACE, FIVE_NM)
    assert holding.gaps_m[0] == pytest.approx(6_000.0) and not holding.fallback
    assert slowing.gaps_m[0] == pytest.approx(14_000.0 - 70.0 * 130.0) and slowing.fallback


def test_the_distance_required_at_the_crossing_is_tbl_5_5_2s():
    """An I 6.5 km behind an F at the F's crossing: TBL 5-5-2 asks 4 NM (7.4 km) where TBL 5-5-1, in trail, asks none
    beyond 3 NM; an F behind the F is clear."""
    def check(category):
        return speed_check(traffic((-8_000.0, "R", True, "F"), (-14_500.0, "R", True, category)), 1, SEPARATION,
                           np.array([70.0, 70.0]), np.array([70.0, 70.0]), np.array([70.0]), 0, PACE, FIVE_NM)

    assert check("I").required_m == pytest.approx(4.0 * NM_M) and check("I").fallback and check("I").allowed.all()
    assert check("F").required_m == pytest.approx(3.0 * NM_M) and check("F").allowed[0]


STEP_S = 2.0


def _spoken(key: str, along, speed, said: dict[int, dict[int, int]], *, typecode: str | None = "A320",
            landing_s: float, roster_landing_s: float | None = None):
    """A flight with a sentence on R, on the loop's steps from 0 s, established throughout: its approach clock position
    and speed along its course per row; its sentence says every column at row 0 (not cleared, the speed "unspecified")
    and ``said`` — row → {column: word} — after that."""
    from ts_transformer.experiments.traffic_loop import Controlled
    from ts_transformer.experiments.traffic_masks import Spoken
    from ts_transformer.instructions.words import APPROACH_NOT_CLEARED, UNCHANGED, Words
    from ts_transformer.prior.scene import Presence
    from ts_transformer.tests.support import instruction_spec

    words = Words(instruction_spec())
    along = np.asarray(along, dtype=float)
    speed = np.broadcast_to(np.asarray(speed, dtype=float), along.shape).copy()
    count = len(along)
    times = STEP_S * np.arange(count)
    grid = np.full((count, 6), UNCHANGED, dtype=np.int64)
    grid[0] = [0, APPROACH_NOT_CLEARED, 0, 0, 0, words.speed_unspecified]
    for row, columns in said.items():
        for column, word in columns.items():
            grid[row, column] = word
    seen = Presence(key, "KXXX", "R", landing_s if roster_landing_s is None else roster_landing_s, True, times,
                    np.abs(along))
    aircraft = Controlled(seen, times, along.copy(), np.zeros(count), np.full(count, 300.0), ("R",) * count, along,
                          np.zeros(count), np.zeros(count), speed, np.ones(count, dtype=bool), "F", "landed", landing_s)
    return Spoken(aircraft, grid, words, typecode), words


def _measure(spoken, replayed=()):
    from ts_transformer.experiments.traffic_masks import measure_airport

    return measure_airport(list(spoken), list(replayed), SEPARATION, STEP_S, PACE)


def test_the_runner_reads_each_flights_words_and_speed_at_its_row_and_the_record_at_the_crossing():
    """L — no published speed, "unspecified" in force, so it holds its 70 m/s — crosses at 85.7 s (its roster time 5 s
    earlier); F follows 11 km out at row 8 at 80 m/s (90 before it), its 80 in force there, and says 90 at row 11: both
    words close to under 3 NM (5,423 m, 4,986 m); every silent step to 5 NM keeps a masked word in force (forced). Held,
    80 is exactly what F flew: the prediction for the word in force errs by nothing until row 11's 90."""
    from ts_transformer.instructions.words import APPROACH, APPROACH_CLEARED, SPEED

    t = STEP_S * np.arange(43)
    words = _spoken("L", -6_000.0 + 70.0 * t, 70.0, {}, typecode=None, landing_s=6_000.0 / 70.0)[1]
    leader, _ = _spoken("L", -6_000.0 + 70.0 * t, 70.0, {0: {APPROACH: APPROACH_CLEARED}}, typecode=None,
                        landing_s=6_000.0 / 70.0, roster_landing_s=6_000.0 / 70.0 - 5.0)
    t_f = STEP_S * np.arange(46)
    follower, _ = _spoken("F", -12_280.0 + 80.0 * t_f, np.where(np.arange(46) < 8, 90.0, 80.0),
                          {0: {SPEED: words.speed_index(80.0)}, 11: {SPEED: words.speed_index(90.0)},
                           20: {APPROACH: APPROACH_CLEARED}}, landing_s=500.0)
    read = _measure([leader, follower])
    assert (read["speed_words"], read["speed_words_checked"], read["speed_words_masked"]) == (3, 2, 2)
    assert [(m["row"], round(m["gap_m"], 1)) for m in read["masked"]] == [(8, 5422.9), (11, 4985.7)]
    assert (read["silent_steps_checked"], read["forced_steps"], read["fallbacks"]) == (9, 9, 0)
    assert read["masked_speed_words_the_record_then_broke"] == 2                 # F really was 5,423 m behind
    assert read["speed_prediction_minus_recorded_m"]["max"] == pytest.approx(0.0, abs=1e-6)
    assert (read["clearances"], read["clearances_behind_a_cleared_aircraft"], read["clearances_masked"]) == (2, 1, 0)


def test_the_runner_takes_the_leaders_speed_word_and_tells_a_masked_word_the_record_did_not_break():
    """L's word in force is 60: it slows from 70 and crosses at 94 s; F, holding 80, is predicted 4,860 m behind then (the
    slowest words keep 3 NM: no fallback) — but in the record it slowed to 60 m/s at 24 s and was 6.8 km behind."""
    from ts_transformer.instructions.words import SPEED

    t = STEP_S * np.arange(43)
    words = _spoken("L", -6_000.0 + 70.0 * t, 70.0, {}, landing_s=6_000.0 / 70.0)[1]
    leader, _ = _spoken("L", -6_000.0 + 70.0 * t, 70.0, {0: {SPEED: words.speed_index(60.0)}}, landing_s=6_000.0 / 70.0)
    t_f = STEP_S * np.arange(46)
    along = -12_380.0 + np.where(t_f <= 24.0, 80.0 * t_f, 80.0 * 24.0 + 60.0 * (t_f - 24.0))
    follower, _ = _spoken("F", along, np.where(t_f < 24.0, 80.0, 60.0), {0: {SPEED: words.speed_index(80.0)}},
                          landing_s=500.0)
    read = _measure([leader, follower])
    assert [(m["row"], round(m["gap_m"], 1)) for m in read["masked"]] == [(8, 4860.0)]
    assert read["masked"][0]["actual_gap_m"] > 3.0 * NM_M and read["masked_speed_words_the_record_then_broke"] == 0


def test_the_runner_takes_a_background_aircraft_at_its_speed_and_as_cleared_once_established():
    """B, background, established, 6 km ahead of F at 70 m/s: F's leader. Held at 70 it crosses in 71.4 s from row 8, so
    F's 80 in force leaves 5,286 m (masked) where slower words keep 3 NM; F's clearance at row 32 (B 5,520 m ahead) is
    masked, B counting as cleared."""
    from ts_transformer.experiments.traffic_census import Track
    from ts_transformer.instructions.words import APPROACH, APPROACH_CLEARED, SPEED
    from ts_transformer.prior.scene import Presence, hung_span

    t = STEP_S * np.arange(46)
    words = _spoken("F", -12_280.0 + 80.0 * t, 80.0, {}, landing_s=500.0)[1]
    follower, _ = _spoken("F", -12_280.0 + 80.0 * t, 80.0,
                          {0: {SPEED: words.speed_index(80.0)}, 32: {APPROACH: APPROACH_CLEARED}}, landing_s=500.0)
    along = -6_120.0 + 70.0 * t
    seen = Presence("B", "KXXX", "R", 6_120.0 / 70.0, False, t, np.abs(along))
    first, last = hung_span(seen, STEP_S)
    background = Track(seen, first, last, along.copy(), np.zeros(len(t)), np.full(len(t), 300.0), along,
                       np.zeros(len(t)), np.zeros(len(t)), np.full(len(t), 70.0), 0.0, "raw", "F")
    read = _measure([follower], [background])
    speed = [m for m in read["masked"] if m["column"] == "speed"]
    assert [(m["row"], m["leader"], round(m["gap_m"], 1)) for m in speed] == [(8, "B", 5285.7)]
    assert read["fallbacks"] == 0 and read["speed_checks"] == 11                     # rows 8–18, to 5 NM
    clear = [m for m in read["masked"] if m["column"] == "approach"]
    assert [(m["row"], m["leader"], round(m["gap_m"], 1)) for m in clear] == [(32, "B", 5520.0)]


def test_the_runner_masks_the_clearance_said_30_s_behind_a_cleared_aircraft(tmp_path, monkeypatch):
    """The step-5 fixture's four labelled flights: "b" flies "a"'s approach 30 s behind it, so when b is cleared at the
    start of its turn onto the final, a — cleared there 30 s earlier, its quarter turn flown since — is 1.4 km ahead on the
    approach clock; "c" and "d" are alone. Behind a, b says no speed word while it is checked: at first, far enough out,
    slowing hard could still open 3 NM — its word in force cannot, so a new one is forced — then nothing can (a fallback,
    which forces nothing)."""
    from ts_transformer.experiments import traffic_masks
    from ts_transformer.tests.test_traffic_loop import _labelled_artefact

    directory, manifest, flights = _labelled_artefact(tmp_path)
    monkeypatch.setattr(traffic_masks, "arrival_manifest_path", lambda code: manifest)
    out = tmp_path / "masks"
    args = ["--instructions", str(directory), "--out", str(out)]
    assert traffic_masks.main(args) == 0
    payload = json.loads((out / "masks.json").read_text(encoding="utf-8"))
    assert payload["schema"] == traffic_masks.SCHEMA and payload["accel_mps2"] == pytest.approx(PACE)
    airport = payload["airports"]["KXXX"]
    assert (airport["flights_with_a_sentence"], airport["background"]) == (4, 0)
    assert (airport["clearances"], airport["clearances_behind_a_cleared_aircraft"], airport["clearances_masked"]) == (
        4, 1, 1)
    event = [m for m in airport["masked"] if m["column"] == "approach"]
    assert [(m["key"], m["leader"]) for m in event] == [("KXXX:b", "KXXX:a")]
    assert 1_000.0 < event[0]["gap_m"] < 2_000.0
    assert airport["speed_words_checked"] == airport["speed_words_masked"] == 0       # b says no speed word then
    assert airport["speed_checks"] == airport["silent_steps_checked"] == airport["forced_steps"] + airport["fallbacks"]
    assert airport["forced_steps"] > 0 and airport["fallbacks"] > 0
    assert airport["speed_words"] >= 4                                                   # each flight's word in force
    assert payload["totals"]["clearances_masked"] == 1
    assert payload["pass_line"]["masked_share"] == pytest.approx(
        (airport["speed_words_masked"] + 1) / (airport["speed_words"] + 4))
    with pytest.raises(SystemExit):                                                   # never overwritten
        traffic_masks.main(args)
