"""The multi-runway arrival scheduler and its separation rules (`inference.runway_schedule`)."""

import math
import random

import pytest
from geokit import FT_M, METRES_PER_DEG_LAT, NM_M, metres_per_deg_lon

from ts_transformer.inference.runway_schedule import (
    DEPENDENT,
    INDEPENDENT,
    SINGLE,
    UNRELATED,
    Arrival,
    ParallelRegime,
    Separation,
    Slot,
    earliest_time,
    eligible_runways,
    faa_separation,
    fcfs_by_eta,
    parallel_relations,
    schedule,
    violations,
    wake_category,
)

SPEED = 70.0
ONE_RUNWAY = Separation(same_nm=3.0, speed_mps=SPEED)
GAP = 3.0 * NM_M / SPEED      # ~79.4 s
EPS = 0.01


def _at(slots: list[Slot], separation: Separation) -> list[Slot]:
    """``placed`` as `earliest_time` takes it: sorted on the approach clock."""
    return sorted(slots, key=lambda s: (separation.approach_s(s), s.key))


def test_two_arrivals_at_one_eta_on_one_runway_land_one_minimum_apart():
    slots = schedule(fcfs_by_eta([Arrival("b", {"05L": 100.0}, {"05L": 0.0}, "F"), Arrival("a", {"05L": 100.0}, {"05L": 0.0}, "F")], EPS),
                     ONE_RUNWAY, delay_weight_per_s=1 / 60, min_probability=EPS)
    assert [s.key for s in slots] == ["a", "b"]                            # the key breaks the ETA tie
    assert slots[1].time_s == pytest.approx(100.0 + GAP)
    assert violations(slots, ONE_RUNWAY) == []


def test_a_late_comer_fits_a_gap_wide_enough_and_waits_otherwise():
    placed = [Slot("a", "05L", 0.0, 0.0, "F"), Slot("b", "05L", 300.0, 300.0, "F")]
    assert earliest_time(placed, "05L", "F", 50.0, ONE_RUNWAY) == pytest.approx(GAP)          # after a, before b
    narrow = [Slot("a", "05L", 0.0, 0.0, "F"), Slot("b", "05L", 120.0, 120.0, "F")]
    assert earliest_time(narrow, "05L", "F", 50.0, ONE_RUNWAY) == pytest.approx(120.0 + GAP)  # no room between


def test_the_wake_minimum_depends_on_who_leads():
    wake = Separation(same_nm=3.0, speed_mps=SPEED, wake_nm={("B", "F"): 5.0})   # heavy leader, large follower
    assert earliest_time([Slot("h", "05L", 0.0, 0.0, "B")], "05L", "F", 0.0, wake) == pytest.approx(5.0 * NM_M / SPEED)
    assert earliest_time([Slot("l", "05L", 0.0, 0.0, "F")], "05L", "B", 0.0, wake) == pytest.approx(GAP)
    # a heavy may not slip in just ahead of a large that is already placed
    assert earliest_time([Slot("l", "05L", 400.0, 400.0, "F")], "05L", "B", 350.0, wake) == pytest.approx(400.0 + GAP)


def test_the_runway_choice_trades_probability_against_delay():
    independent = Separation(same_nm=3.0, speed_mps=SPEED, relations={frozenset(("17L", "17R")): INDEPENDENT})
    first = Arrival("a", {"17L": 0.0, "17R": 0.0}, {"17L": math.log(0.9), "17R": math.log(0.1)}, "F")
    unsure = Arrival("b", {"17L": 1.0, "17R": 1.0}, {"17L": math.log(0.6), "17R": math.log(0.4)}, "F")
    sure = Arrival("c", {"17L": 2.0, "17R": 2.0}, {"17L": math.log(0.999), "17R": math.log(0.001)}, "F")
    slots = {s.key: s for s in schedule(fcfs_by_eta([sure, unsure, first], EPS), independent,
                                        delay_weight_per_s=1 / 60, min_probability=EPS)}
    assert slots["a"].runway == "17L"
    assert slots["b"].runway == "17R" and slots["b"].delay_s == 0.0      # 0.4 beats waiting ~79 s behind a
    assert slots["c"].runway == "17L"                                     # 0.001 is under the floor: it waits


def test_the_fcfs_order_reads_only_the_runways_the_scheduler_tries():
    # b's earliest ETA is on a runway under the floor; on the runways it can get, a comes first
    a = Arrival("a", {"17L": 100.0}, {"17L": 0.0}, "F")
    b = Arrival("b", {"17L": 110.0, "35R": 50.0}, {"17L": math.log(0.999), "35R": math.log(0.001)}, "F")
    assert eligible_runways(b, EPS) == ["17L"]
    assert [x.key for x in fcfs_by_eta([b, a], EPS)] == ["a", "b"]
    with pytest.raises(ValueError):
        eligible_runways(a, 0.0)
    with pytest.raises(ValueError):
        Arrival("c", {"17L": 1.0}, {"17L": 0.0, "17R": -1.0}, "F")        # the two maps must name the same runways


def test_dependent_parallels_keep_the_diagonal_through_an_along_track_stagger():
    pair = frozenset(("05L", "05R"))
    s = 3498 * FT_M / NM_M
    dependent = Separation(same_nm=3.0, speed_mps=SPEED, relations={pair: DEPENDENT}, spacing_nm={pair: s}, diagonal_nm={pair: 1.5})
    assert dependent.distance_nm("05L", "F", "05R", "F") == pytest.approx(math.sqrt(1.5 ** 2 - s ** 2))
    assert dependent.distance_nm("05L", "F", "05L", "F") == 3.0
    assert Separation(same_nm=3.0, speed_mps=SPEED).relation("05L", "23R") == UNRELATED
    # the largest minimum bounds a dependent stagger too (a 4-NM diagonal here, over the 3-NM radar)
    wide = Separation(same_nm=3.0, speed_mps=SPEED, relations={pair: DEPENDENT}, spacing_nm={pair: 0.1}, diagonal_nm={pair: 4.0})
    assert wide.max_gap_s == pytest.approx(math.sqrt(16.0 - 0.01) * NM_M / SPEED)


def test_staggered_thresholds_are_separated_on_the_approach_clock():
    # KRDU-like: 23L's threshold 0.67 NM further along the course than 23R's, a 1.0 NM diagonal
    pair = frozenset(("23L", "23R"))
    s = 3500 * FT_M / NM_M
    stagger = math.sqrt(1.0 - s * s) * NM_M / SPEED
    offset = 0.67 * NM_M / SPEED
    rules = Separation(same_nm=3.0, speed_mps=SPEED, relations={pair: DEPENDENT}, spacing_nm={pair: s},
                       diagonal_nm={pair: 1.0}, along_nm={"23R": 0.0, "23L": 0.67})
    # behind a 23R landing at 0, a 23L landing needs the stagger PLUS the time to fly the offset
    assert earliest_time([Slot("r", "23R", 0.0, 0.0, "F")], "23L", "F", 0.0, rules) == pytest.approx(offset + stagger)
    # ahead of it on the approach clock by more than the stagger: a 23L landing ~9 s BEFORE the 23R one
    assert earliest_time([Slot("r", "23R", 0.0, 0.0, "F")], "23L", "F", offset - stagger - 5.0, rules) == pytest.approx(offset - stagger - 5.0)
    short = [Slot("r", "23R", 0.0, 0.0, "F"), Slot("l", "23L", offset + stagger - 2.0, 0.0, "F")]
    assert [(a.key, b.key) for a, b, _ in violations(short, rules)] == [("r", "l")]
    # the stagger is read off the thresholds: along the course from the first runway's threshold
    targets = {"23L": {"lat": 35.0, "lon": -80.0, "course_deg": 225.0},
               "23R": {"lat": 35.0 + 0.67 * NM_M * math.cos(math.radians(225.0)) / -METRES_PER_DEG_LAT,
                       "lon": -80.0 + 0.67 * NM_M * math.sin(math.radians(225.0)) / -metres_per_deg_lon(35.0), "course_deg": 225.0}}
    along = parallel_relations(targets, [ParallelRegime(2500.0, SINGLE)])[3]
    assert along["23L"] - along["23R"] == pytest.approx(0.67, abs=1e-3)


def test_earliest_time_is_the_earliest_feasible_time_on_random_streams():
    pair_d, pair_s = frozenset(("A", "B")), frozenset(("A", "C"))
    rules = Separation(same_nm=3.0, speed_mps=SPEED, relations={pair_d: DEPENDENT, pair_s: SINGLE},
                       spacing_nm={pair_d: 0.55, pair_s: 0.2}, diagonal_nm={pair_d: 1.0},
                       wake_nm={("B", "F"): 5.0, ("C", "F"): 3.5, ("F", "I"): 4.0}, along_nm={"A": 0.0, "B": 0.6, "C": -0.4})
    rng = random.Random(7)
    for _ in range(150):
        placed = _at([Slot(f"s{i}", rng.choice("ABC"), rng.uniform(0.0, 900.0), 0.0, rng.choice("BCFFI")) for i in range(12)], rules)
        runway, category, eta = rng.choice("ABC"), rng.choice("BCFI"), rng.uniform(0.0, 900.0)
        t = earliest_time(placed, runway, category, eta, rules)
        new = Slot("new", runway, t, eta, category)
        assert t >= eta and not any("new" in (a.key, b.key) for a, b, _ in violations([*placed, new], rules))
        # nothing earlier on a 0.05 s grid is feasible
        for k in range(int((t - eta) / 0.05)):
            probe = Slot("new", runway, eta + 0.05 * k, eta, category)
            assert any("new" in (a.key, b.key) for a, b, _ in violations([*placed, probe], rules))


def test_the_parallel_relation_follows_the_centerline_spacing_bands():
    regimes = [ParallelRegime(2500.0, SINGLE), ParallelRegime(3600.0, DEPENDENT, 1.0), ParallelRegime(4300.0, DEPENDENT, 1.5)]

    def targets(offset_ft: float) -> dict:
        # two north-bound runways, the second offset east by ``offset_ft``
        east_deg = offset_ft * FT_M / (metres_per_deg_lon(35.0))
        return {"36L": {"lat": 35.0, "lon": -80.0, "course_deg": 0.0},
                "36R": {"lat": 35.0, "lon": -80.0 + east_deg, "course_deg": 0.0},
                "09": {"lat": 35.0, "lon": -80.0, "course_deg": 90.0}}

    for offset, relation, diag in ((700.0, SINGLE, None), (3000.0, DEPENDENT, 1.0), (4000.0, DEPENDENT, 1.5), (6000.0, INDEPENDENT, None)):
        relations, spacing, diagonal, along = parallel_relations(targets(offset), regimes)
        pair = frozenset(("36L", "36R"))
        assert relations[pair] == relation
        assert spacing[pair] == pytest.approx(offset * FT_M / NM_M, rel=1e-6)
        assert diagonal.get(pair) == diag
        assert along["36L"] == pytest.approx(along["36R"], abs=1e-9)          # level thresholds: no stagger
        assert frozenset(("36L", "09")) not in relations                    # another direction: no parallel relation


def test_violations_list_the_pairs_closer_than_their_minimum():
    slots = [Slot("a", "05L", 0.0, 0.0, "F"), Slot("b", "05L", 60.0, 60.0, "F"), Slot("c", "05L", 200.0, 200.0, "F")]
    found = violations(slots, ONE_RUNWAY)
    assert [(l.key, f.key) for l, f, _ in found] == [("a", "b")]
    assert found[0][2] == pytest.approx(GAP - 60.0)
    assert violations(slots, ONE_RUNWAY, tolerance_s=30.0) == []


def test_the_order_given_is_the_order_placed_and_a_placed_slot_is_frozen():
    early = Arrival("early", {"05L": 100.0}, {"05L": 0.0}, "F")
    late = Arrival("late", {"05L": 130.0}, {"05L": 0.0}, "F")
    by_eta = {s.key: s.time_s for s in schedule(fcfs_by_eta([late, early], EPS), ONE_RUNWAY, delay_weight_per_s=1 / 60, min_probability=EPS)}
    assert by_eta == pytest.approx({"early": 100.0, "late": 100.0 + GAP})
    # known first, the later ETA keeps its own time and the earlier one cannot land before it
    by_arrival = {s.key: s.time_s for s in schedule([late, early], ONE_RUNWAY, delay_weight_per_s=1 / 60, min_probability=EPS)}
    assert by_arrival == pytest.approx({"late": 130.0, "early": 130.0 + GAP})


def test_only_landings_within_the_largest_minimum_constrain_a_time():
    wake = Separation(same_nm=3.0, speed_mps=SPEED, wake_nm={("B", "F"): 5.0})
    assert wake.max_gap_s == pytest.approx(5.0 * NM_M / SPEED)
    stream = [Slot(f"s{i}", "05L", 1000.0 * i, 1000.0 * i, "F") for i in range(50)]
    assert earliest_time(stream, "05L", "F", 20_500.0, wake) == 20_500.0                  # between two, far from both
    assert earliest_time(stream, "05L", "F", 20_010.0, wake) == pytest.approx(20_000.0 + GAP)
    assert earliest_time(stream, "05L", "F", 19_950.0, wake) == pytest.approx(20_000.0 + GAP)  # too close ahead of s20


def test_a_stream_on_the_wall_clock_keeps_its_minima_to_float_resolution():
    # at ~1.8e9 s a sum and a difference are off by ~2e-7 s: a slot placed AT its minimum must read as kept
    epoch = 1_784_000_000.0
    arrivals = [Arrival(f"a{i:02d}", {"05L": epoch + 10.0 * i}, {"05L": 0.0}, "F" if i % 3 else "B") for i in range(30)]
    wake = Separation(same_nm=3.0, speed_mps=SPEED, wake_nm={("B", "F"): 5.0})
    slots = schedule(fcfs_by_eta(arrivals, EPS), wake, delay_weight_per_s=1 / 60, min_probability=EPS)
    assert violations(slots, wake) == []
    assert all(b.time_s > a.time_s for a, b in zip(slots, slots[1:]))


def _parallel_pair(offset_ft: float) -> dict:
    # two runways on a 050 compass course, the second offset to the right of the course by ``offset_ft``
    d, course = offset_ft * FT_M, math.radians(50.0)
    return {"05L": {"lat": 35.0, "lon": -80.0, "course_deg": 50.0},
            "05R": {"lat": 35.0 - d * math.sin(course) / METRES_PER_DEG_LAT,
                    "lon": -80.0 + d * math.cos(course) / metres_per_deg_lon(35.0), "course_deg": 50.0}}


def test_the_faa_rules_by_spacing_and_category():
    # one runway: 3 NM, or TBL 5-5-2 at the threshold — heavy B before large F is 5, C before small I 6
    one = faa_separation({"17": {"lat": 38.0, "lon": -121.0, "course_deg": 170.0}}, speed_mps=SPEED)
    assert one.distance_nm("17", "F", "17", "G") == 3.0
    assert one.distance_nm("17", "B", "17", "F") == 5.0
    assert one.distance_nm("17", "C", "17", "I") == 6.0
    assert one.distance_nm("17", "F", "17", "I") == 4.0
    assert one.distance_nm("17", "G", "17", "I") == 3.0        # a blank cell: the radar minimum only
    assert one.distance_nm("17", "E", "17", "F") == 3.0
    assert wake_category("B38M") == "F" and wake_category("E75L") == "G" and wake_category("B763") == "C"
    with pytest.raises(KeyError):
        wake_category("ZZZZ")
    # a KSJC-like pair is one runway, wake included; KRDU-like dependent at 1.0 NM; KSMF-like independent
    close = faa_separation(_parallel_pair(700.0), speed_mps=SPEED)
    assert close.one_runway("05L", "05R") and close.distance_nm("05L", "B", "05R", "F") == 5.0
    krdu = faa_separation(_parallel_pair(3500.0), speed_mps=SPEED)
    s = krdu.spacing_nm[frozenset(("05L", "05R"))]
    assert krdu.relation("05L", "05R") == DEPENDENT
    assert krdu.distance_nm("05L", "B", "05R", "F") == pytest.approx(math.sqrt(1.0 - s * s))   # no wake across
    assert faa_separation(_parallel_pair(4000.0), speed_mps=SPEED).diagonal_nm[frozenset(("05L", "05R"))] == 1.5
    assert faa_separation(_parallel_pair(6000.0), speed_mps=SPEED).relation("05L", "05R") == INDEPENDENT
    # the options: the reduced 2.5 NM, and the visual reading with nothing across a parallel pair
    assert faa_separation(_parallel_pair(700.0), speed_mps=SPEED, radar_nm=2.5).distance_nm("05L", "F", "05L", "F") == 2.5
    visual = faa_separation(_parallel_pair(700.0), speed_mps=SPEED, visual_parallels=True)
    assert visual.gap_s("05L", "B", "05R", "F") == 0.0 and visual.distance_nm("05L", "B", "05L", "F") == 5.0
