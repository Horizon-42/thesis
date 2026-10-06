"""Stage C, C3: "established", the separation judge on v4 and the speed-word mask (post-training §3, §8 C3; D92)."""

from __future__ import annotations


import numpy as np
import pytest
from geokit import NM_M

from dataclasses import replace

from ts_transformer.inference.separation import RADAR_OR_VERTICAL, IN_TRAIL
from ts_transformer.instructions.grammar import column_words
from ts_transformer.instructions.spec import ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M
from ts_transformer.instructions.words import SPEED, UNCHANGED, Words
from ts_transformer.post.established import established
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import AircraftAt
from ts_transformer.post.speed_mask import along_course_speeds, ramp_distance_m, ramp_time_s, speed_check
from ts_transformer.post.traffic import commanded_loss, joined, traffic
from ts_transformer.tests.post_support import FAF_M, airport, finals
from ts_transformer.tests.support import instruction_spec

SPEC = instruction_spec()
WORDS = Words(SPEC)


def _one(key, e, n=0.0, h=600.0, *, track_deg=90.0, speed=75.0, runway=0, known=True, category="F", last=False,
         go_around=False):
    """One aircraft at ``(e, n, h)`` that moved ``speed`` m/s on ``track_deg`` in the 2 s before."""
    angle = np.radians(track_deg)
    before = (e - 2.0 * speed * np.sin(angle), n - 2.0 * speed * np.cos(angle), h + 3.0) if known else (0.0, 0.0, 0.0)
    return (key, (e, n, h), before, known, runway, category, last, go_around)


def _set(*items):
    return AircraftAt.of(list(items))


def test_established_is_g_false_inside_the_region_and_lined_up():
    geometry = airport()
    fin = finals(geometry)
    aircraft = _set(_one("in", -5_000.0), _one("beyond_faf", -(FAF_M + 500.0)), _one("off", -5_000.0, n=2_000.0),
                    _one("turning", -5_000.0, track_deg=90.0 + 21.0), _one("lined", -5_000.0, track_deg=90.0 - 19.0),
                    _one("no_runway", -5_000.0, runway=-1), _one("row0", -5_000.0, known=False))
    got = established(aircraft, geometry, fin, 2.0)
    assert got.tolist() == [True, False, False, False, True, False, False]           # within 20° (TBL 5-9-1)
    in_go_around = replace(aircraft, go_around=np.ones(7, dtype=bool))
    assert not established(in_go_around, geometry, fin, 2.0).any()                    # G true


def test_established_reads_one_row_and_is_the_same_for_every_aircraft():
    geometry = airport()
    fin = finals(geometry)
    rng = np.random.default_rng(0)
    items = [_one(f"a{k}", float(rng.uniform(-9_500, 0)), float(rng.uniform(-400, 400)),
                  track_deg=float(rng.uniform(40, 140)), runway=int(rng.integers(0, 2))) for k in range(40)]
    together = established(_set(*items), geometry, fin, 2.0)
    alone = [bool(established(_set(item), geometry, fin, 2.0)[0]) for item in items]
    assert together.tolist() == alone and together.any() and not together.all()
    # the commanded aircraft (first of a joined set) reads the same as a recorded one
    own = _set(items[3])
    both = joined(own, _set(*items[3:5]))
    assert established(both, geometry, fin, 2.0)[0] == together[3]


def test_the_judge_on_v4_makes_the_joining_commanded_aircraft_answer():
    geometry = airport()
    separation, fin = airport_separation(geometry), finals(geometry)
    leader = _one("leader", -6_000.0, h=500.0)                                  # established on 09
    joiner = _one("me", -7_000.0, n=-3_000.0, h=600.0, track_deg=45.0)           # turning in, not established
    scene = traffic(joined(_set(joiner), _set(leader)), geometry, separation, fin, 2.0)
    assert scene.established.tolist() == [False, True]
    assert scene.runway == ("09", "09")
    assert scene.along_m[1] - scene.along_m[0] == pytest.approx(1_000.0)
    loss = commanded_loss(scene, np.zeros(2, dtype=bool), separation)
    assert loss.kind == RADAR_OR_VERTICAL and loss.responsible == (0,)
    # the same two the other way round: the recorded aircraft joins behind an established commanded one — not ours
    me = _one("me", -6_000.0, h=500.0)
    other = _one("other", -7_000.0, n=-3_000.0, h=600.0, track_deg=45.0)
    scene = traffic(joined(_set(me), _set(other)), geometry, separation, fin, 2.0)
    assert commanded_loss(scene, np.zeros(2, dtype=bool), separation) is None
    # in trail on one final, the aircraft behind answers
    scene = traffic(joined(_set(_one("me", -7_000.0)), _set(_one("lead", -3_000.0, h=300.0))),
                    geometry, separation, fin, 2.0)
    loss = commanded_loss(scene, np.zeros(2, dtype=bool), separation)
    assert loss.kind == IN_TRAIL and loss.responsible == (0,)


def test_the_wake_at_the_threshold_counts_only_when_the_commanded_aircraft_follows():
    geometry = airport()
    separation, fin = airport_separation(geometry), finals(geometry)
    heavy = _one("heavy", -50.0, h=115.0, category="B", last=True)              # over its threshold now
    me = _one("me", -50.0 - 6.0 * NM_M, h=600.0, category="F")                   # 6 NM behind: B → F needs 5 NM
    scene = traffic(joined(_set(me), _set(heavy)), geometry, separation, fin, 2.0)
    assert scene.established.tolist() == [False, True]                           # 6 NM out: beyond the FAF
    assert commanded_loss(scene, np.array([False, True]), separation) is None
    close = _one("me", -50.0 - 4.0 * NM_M, h=500.0, category="F")                # 4 NM behind, inside the FAF
    scene = traffic(joined(_set(close), _set(heavy)), geometry, separation, fin, 2.0)
    assert scene.established.all()
    loss = commanded_loss(scene, np.array([False, True]), separation)
    assert loss is not None and loss.responsible == (0,)


#: A FAF far enough out that an established aircraft can be more than 5 NM from its threshold (the mask applies only
#: there: with the FAF of `post_support.FAF_M`, 9,000 m, it never does).
FAR_FAF_M = 14_000.0


def _speed_scene(me_e, leader_e, *, me_speed=75.0, leader_speed=65.0, me_track=90.0, leader_category="F"):
    geometry = airport()
    separation, fin = airport_separation(geometry), finals(geometry, FAR_FAF_M)
    aircraft = joined(_set(_one("me", me_e, h=700.0, speed=me_speed, track_deg=me_track)),
                      _set(_one("leader", leader_e, h=400.0, speed=leader_speed, category=leader_category)))
    scene = traffic(aircraft, geometry, separation, fin, 2.0)
    return scene, along_course_speeds(aircraft, scene, 2.0), separation


def test_the_speed_mask_masks_the_words_that_close_the_gap():
    scene, speeds, separation = _speed_scene(-12_000.0, -5_000.0)
    assert scene.established.all()
    check = speed_check(scene, speeds, separation, WORDS, len(airport().candidates))
    words_of = column_words(SPEED, WORDS, 2)
    assert check.applies and check.leader == 1 and not check.fallback
    assert check.required_m == pytest.approx(3.0 * NM_M)
    assert check.permitted[list(words_of).index(UNCHANGED)] and check.permitted[-1]          # unchanged, unspecified
    level = [k for k, w in enumerate(words_of) if w not in (UNCHANGED, WORDS.speed_unspecified)]
    fast = max(level, key=lambda k: WORDS.speed_mps(int(words_of[k])))
    slow = min(level, key=lambda k: WORDS.speed_mps(int(words_of[k])))
    assert not check.permitted[fast] and check.permitted[slow]
    assert all(check.permitted[k] == (check.gaps_m[k] >= check.required_m) for k in level)


def test_the_speed_mask_masks_nothing_outside_its_conditions_or_when_every_word_falls_short():
    n = len(airport().candidates)
    every = np.ones(len(column_words(SPEED, WORDS, n)), dtype=bool)
    for scene, speeds, separation in (_speed_scene(-12_000.0, -5_000.0, me_track=150.0),     # not lined up
                                      _speed_scene(-(ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M - 200.0), -2_800.0),
                                      _speed_scene(-12_000.0, -13_000.0)):                  # nobody ahead
        check = speed_check(scene, speeds, separation, WORDS, n)
        assert not check.applies and np.array_equal(check.permitted, every)
    # the aircraft next ahead not established (turning in, 2 km ahead): nothing masked, though an established one is
    # 6 km ahead (the mask reads the aircraft next ahead, §3)
    geometry = airport()
    separation, fin = airport_separation(geometry), finals(geometry, FAR_FAF_M)
    aircraft = joined(_set(_one("me", -12_000.0, h=700.0)),
                      _set(_one("joiner", -10_000.0, n=-1_500.0, h=650.0, track_deg=50.0), _one("lead", -6_000.0, h=400.0)))
    scene = traffic(aircraft, geometry, separation, fin, 2.0)
    assert scene.established.tolist() == [True, False, True]
    check = speed_check(scene, along_course_speeds(aircraft, scene, 2.0), separation, WORDS, n)
    assert not check.applies and np.array_equal(check.permitted, every)
    # a heavy leader 1 km from its threshold, 8.5 km ahead: B → F needs 5 NM (9,260 m) when it crosses (TBL 5-5-2)
    scene, speeds, separation = _speed_scene(-9_500.0, -1_000.0, leader_category="B")
    check = speed_check(scene, speeds, separation, WORDS, n)
    assert check.applies and check.fallback and np.array_equal(check.permitted, every)


def test_the_ramp_reaches_its_target_and_holds_it():
    assert ramp_distance_m(10.0, 70.0, np.array([70.0]), 1.0)[0] == pytest.approx(700.0)
    assert ramp_distance_m(30.0, 70.0, np.array([60.0]), 1.0)[0] == pytest.approx(70 * 10 - 50 + 60 * 20)
    for target in (55.0, 70.0, 85.0):
        t = ramp_time_s(5_000.0, 70.0, target, 0.8)
        assert ramp_distance_m(t, 70.0, np.array([target]), 0.8)[0] == pytest.approx(5_000.0)
    with pytest.raises(ValueError):
        ramp_time_s(1_000.0, -5.0, 60.0, 1.0)
