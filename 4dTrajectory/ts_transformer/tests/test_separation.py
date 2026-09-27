"""Loss of separation at one instant (`inference.separation`, multi-aircraft design §3.2)."""

import itertools
import math

import numpy as np
import pytest
from geokit import FT_M, NM_M

from ts_transformer.inference.runway_schedule import (
    CWT_DIRECTLY_BEHIND_NM,
    CWT_ON_APPROACH_NM,
    DEPENDENT,
    FAA_VISUAL_INTERCEPT_MAX_DEG,
    INDEPENDENT,
    SAME,
    SINGLE,
    UNRELATED,
    Separation,
)
from ts_transformer.inference.separation import (
    AT_THRESHOLD,
    DIAGONAL,
    IFR,
    IN_TRAIL,
    NO_RUNWAY,
    RADAR_OR_VERTICAL,
    VISUAL,
    Traffic,
    losses,
    next_behind,
    wake_at_threshold,
)

RADAR_M = 3.0 * NM_M
VERTICAL_M = 1_000.0 * FT_M
#: One runway "R", a pair separated as one "S1"/"S2", a dependent parallel pair "L1"/"L2" (1.0 NM diagonal), an
#: independent pair "I1"/"I2", all landing east, the second of each pair north of (left of) the first; "X" is a runway
#: of another direction.
SEPARATION = Separation(same_nm=3.0, speed_mps=70.0,
                        relations={frozenset(("S1", "S2")): SINGLE, frozenset(("L1", "L2")): DEPENDENT,
                                   frozenset(("I1", "I2")): INDEPENDENT},
                        spacing_nm={frozenset(("S1", "S2")): 0.1, frozenset(("L1", "L2")): 0.6,
                                    frozenset(("I1", "I2")): 1.5},
                        right_nm={("S1", "S2"): -0.1, ("S2", "S1"): 0.1, ("L1", "L2"): -0.6, ("L2", "L1"): 0.6,
                                  ("I1", "I2"): -1.5, ("I2", "I1"): 1.5},
                        diagonal_nm={frozenset(("L1", "L2")): 1.0}, wake_nm=CWT_ON_APPROACH_NM)


def traffic(*aircraft) -> Traffic:
    """Each aircraft: (east m, north m, height m, runway, established, category[, track less course °[, metres right of
    the centreline]]); on the approach clock at its east; on its centreline, flying its course, unless said otherwise."""
    e, n, h, runway, established, category, angle, right = zip(*((*a, 0.0, 0.0)[:8] for a in aircraft))

    def said(values) -> np.ndarray:
        return np.array([x if r is not None else np.nan for x, r in zip(values, runway)], dtype=float)

    return Traffic(np.array(e, float), np.array(n, float), np.array(h, float), tuple(runway), said(e), said(angle),
                   said(right), np.array(established, bool), tuple(category))


def test_the_wake_tables_are_transcribed_as_the_order_prints_them():
    """TBL 5-5-2 is TBL 5-5-1 except the cells printed bold in the literature (larger at the threshold)."""
    larger = {pair for pair in CWT_ON_APPROACH_NM if CWT_ON_APPROACH_NM[pair] != CWT_DIRECTLY_BEHIND_NM.get(pair)}
    assert larger == {("B", "I"), ("C", "I"), ("D", "H"), ("D", "I"), ("F", "I")}
    assert all(CWT_ON_APPROACH_NM[pair] >= value for pair, value in CWT_DIRECTLY_BEHIND_NM.items())


def test_in_trail_on_one_final_is_horizontal_only_and_the_one_behind_answers():
    # F behind F 5 km on the same final: under 3 NM although 400 m apart vertically
    found = losses(traffic((-5_000, 0, 700, "R", True, "F"), (-10_000, 0, 300, "R", True, "F")), SEPARATION, IFR)
    assert [(loss.kind, loss.responsible) for loss in found] == [(IN_TRAIL, (1,))]      # the one 10 km out
    assert found[0].required_m == pytest.approx(RADAR_M)
    # 6 km: clear of 3 NM; behind a B the directly-behind wake minimum (5 NM) binds
    assert losses(traffic((-4_000, 0, 700, "R", True, "F"), (-10_000, 0, 700, "R", True, "F")), SEPARATION, IFR) == []
    wake = losses(traffic((-2_000, 0, 700, "R", True, "B"), (-10_000, 0, 700, "R", True, "F")), SEPARATION, IFR)
    assert wake[0].required_m == pytest.approx(5.0 * NM_M) and wake[0].responsible == (1,)


def test_a_pair_not_both_established_keeps_radar_or_vertical_and_the_joining_one_answers():
    joining = losses(traffic((-5_000, 0, 700, "R", True, "F"), (-9_000, 1_000, 600, "R", False, "F")), SEPARATION, IFR)
    assert [(loss.kind, loss.responsible) for loss in joining] == [(RADAR_OR_VERTICAL, (1,))]
    stacked = traffic((-5_000, 0, 700, "R", False, "F"), (-9_000, 1_000, 700 + VERTICAL_M + 1, "R", False, "F"))
    assert losses(stacked, SEPARATION, IFR) == []                                   # vertically separated
    converging = losses(traffic((0, 4_000, 900, None, False, "F"), (0, 0, 800, "R", False, "F")), SEPARATION, IFR)
    assert [(loss.relation, loss.responsible) for loss in converging] == [(NO_RUNWAY, (0, 1))]


def test_parallel_finals_dependent_by_the_diagonal_independent_not_at_all():
    diagonal = losses(traffic((-5_000, 0, 700, "L1", True, "F"), (-6_000, 1_100, 700, "L2", True, "F")), SEPARATION, IFR)
    assert [(loss.kind, loss.responsible) for loss in diagonal] == [(DIAGONAL, (1,))]
    assert diagonal[0].distance_m == pytest.approx(math.hypot(1_000, 1_100))
    assert losses(traffic((-5_000, 0, 700, "I1", True, "F"), (-5_100, 800, 700, "I2", True, "F")), SEPARATION, IFR) == []


def test_an_untyped_aircraft_in_trail_is_judged_on_the_radar_minimum_and_says_so():
    found = losses(traffic((-2_000, 0, 700, "R", True, None), (-7_000, 0, 700, "R", True, "F")), SEPARATION, IFR)
    assert found[0].required_m == pytest.approx(RADAR_M) and not found[0].wake_known


def test_at_the_threshold_the_established_one_next_behind_must_be_the_on_approach_wake_minimum():
    # a D over the threshold, an H 9 km behind: TBL 5-5-2 wants 6 NM (TBL 5-5-1 only 5)
    scene = traffic((0, 0, 50, "R", True, "D"), (-9_000, 0, 550, "R", True, "H"), (-12_000, 0, 700, "R", True, "H"))
    loss = wake_at_threshold(scene, 0, SEPARATION)
    assert (loss.kind, loss.j, loss.responsible, loss.relation) == (AT_THRESHOLD, 1, (1,), SAME)
    assert loss.required_m == pytest.approx(6.0 * NM_M) and loss.distance_m == pytest.approx(9_000)
    # not yet established, another runway, an untyped follower: not judged
    assert wake_at_threshold(traffic((0, 0, 50, "R", True, "D"), (-9_000, 0, 550, "R", False, "H")), 0, SEPARATION) is None
    assert wake_at_threshold(traffic((0, 0, 50, "R", True, "D"), (-9_000, 0, 550, "I1", True, "H")), 0, SEPARATION) is None
    assert wake_at_threshold(traffic((0, 0, 50, "R", True, "D"), (-9_000, 0, 550, "R", True, None)), 0, SEPARATION) is None


def test_in_trail_uses_the_directly_behind_table_and_the_one_runway_radar_minimum():
    # D before H: 5 NM directly behind (TBL 5-5-1), 6 NM only at the threshold (TBL 5-5-2)
    assert losses(traffic((0, 0, 700, "R", True, "D"), (-9_500, 0, 700, "R", True, "H")), SEPARATION, IFR) == []
    close = losses(traffic((0, 0, 700, "R", True, "D"), (-9_000, 0, 700, "R", True, "H")), SEPARATION, IFR)
    assert close[0].required_m == pytest.approx(5.0 * NM_M)
    # a pair separated as one is in trail too
    single = losses(traffic((-5_000, 0, 700, "S1", True, "F"), (-9_000, 150, 700, "S2", True, "F")), SEPARATION, IFR)
    assert [(loss.kind, loss.relation, loss.responsible) for loss in single] == [(IN_TRAIL, SINGLE, (1,))]
    # the one-runway radar minimum is the Separation's (2.5 NM where authorized, 5-5-4 j)
    reduced = Separation(same_nm=2.5, speed_mps=70.0, wake_nm=CWT_ON_APPROACH_NM)
    pair = traffic((0, 0, 700, "R", True, "F"), (-2.6 * NM_M, 0, 700, "R", True, "F"))
    assert losses(pair, reduced, IFR) == [] and losses(pair, SEPARATION, IFR)[0].required_m == pytest.approx(RADAR_M)


def test_the_diagonal_is_the_dependent_minimum_and_vertical_separation_still_counts():
    diagonal = losses(traffic((-5_000, 0, 700, "L1", True, "F"), (-6_000, 1_100, 700, "L2", True, "F")), SEPARATION, IFR)
    assert diagonal[0].required_m == pytest.approx(1.0 * NM_M)
    # 2.5 km apart: inside 3 NM, outside the 1.0 NM diagonal
    assert losses(traffic((-5_000, 0, 700, "L1", True, "F"), (-7_300, 1_000, 700, "L2", True, "F")), SEPARATION, IFR) == []
    stacked = traffic((-5_000, 0, 700, "L1", True, "F"), (-6_000, 1_100, 700 + VERTICAL_M + 1, "L2", True, "F"))
    assert losses(stacked, SEPARATION, IFR) == []


def test_established_on_runways_of_other_directions_is_radar_or_vertical_both_answering():
    found = losses(traffic((-3_000, 0, 700, "R", True, "F"), (-3_000, 3_000, 700, "X", True, "F")), SEPARATION, IFR)
    assert [(loss.kind, loss.relation, loss.responsible) for loss in found] == [(RADAR_OR_VERTICAL, UNRELATED, (0, 1))]


def test_the_vertical_minimum_is_1000_ft():
    below = losses(traffic((0, 4_000, 900, None, False, "F"), (0, 0, 900 + 200, "R", False, "F")), SEPARATION, IFR)
    assert below[0].vertical_m == pytest.approx(200) and below[0].relation == NO_RUNWAY
    assert losses(traffic((0, 4_000, 900, None, False, "F"), (0, 0, 900 + VERTICAL_M + 0.01, "R", False, "F")),
                  SEPARATION, IFR) == []


def test_at_the_threshold_only_the_on_approach_table_binds_and_exactly_the_minimum_is_enough():
    # F behind F: the table is blank at the threshold, so 3 km is not judged there (in trail judges it)
    assert wake_at_threshold(traffic((0, 0, 50, "R", True, "F"), (-3_000, 0, 250, "R", True, "F")), 0, SEPARATION) is None
    exactly = traffic((0, 0, 50, "R", True, "D"), (-6.0 * NM_M, 0, 550, "R", True, "H"))
    assert wake_at_threshold(exactly, 0, SEPARATION) is None
    across = wake_at_threshold(traffic((0, 0, 50, "S1", True, "D"), (-9_000, 150, 550, "S2", True, "H")), 0, SEPARATION)
    assert across.relation == SINGLE


NAN = float("nan")


@pytest.mark.parametrize("runway, along, angle, right, established", [
    (None, 0.0, NAN, NAN, False),        # on the approach clock with no runway in force
    ("R", NAN, 0.0, 0.0, False),         # a runway in force, not on the approach clock
    ("R", 0.0, NAN, 0.0, False),         # a runway in force, no track less its course
    ("R", 0.0, 0.0, NAN, False),         # a runway in force, no distance off its centreline
    (None, NAN, 0.0, NAN, False),        # a track less a course with no runway in force
    (None, NAN, NAN, 0.0, False),        # a distance off a centreline with no runway in force
    ("R", 0.0, 190.0, 0.0, False),       # a track less its course outside [-180, 180]
    ("R", 0.0, -340.0, 0.0, False),
    (None, NAN, NAN, NAN, True),         # established with no runway
])
def test_traffic_refuses_a_broken_contract(runway, along, angle, right, established):
    with pytest.raises(ValueError):
        Traffic(np.zeros(1), np.zeros(1), np.zeros(1), (runway,), np.array([along]), np.array([angle]),
                np.array([right]), np.array([established]), ("F",))


def test_the_visual_reading_frees_parallels_once_both_intercept_at_30_deg_or_less():
    """7-4-4 c2 a / c3 a: approved separation until each is on a heading intercepting its centreline at 30° or less."""
    diagonal = traffic((-5_000, 0, 700, "L1", True, "F"), (-6_000, 1_100, 700, "L2", True, "F"))
    # the second aircraft 500 m outside its centreline (left, away from the other final), or 1,500 m inside it — past
    # the midline between the two finals (0.3 NM for L, 0.75 NM for I)
    for angle, right, lost in ((20.0, -500.0, False), (FAA_VISUAL_INTERCEPT_MAX_DEG, -500.0, False),
                               (31.0, -500.0, True), (75.0, -500.0, True), (-20.0, -500.0, False),
                               (0.0, 1_500.0, True), (-20.0, 1_500.0, True)):
        for runways in (("I1", "I2"), ("L1", "L2")):
            turning_in = traffic((-8_000, 0, 700, runways[0], True, "F"),
                                 (-9_000, 1_800, 800, runways[1], False, "F", angle, right))
            assert [(loss.kind, loss.responsible) for loss in losses(turning_in, SEPARATION, IFR)] == [
                (RADAR_OR_VERTICAL, (1,))]
            assert losses(turning_in, SEPARATION, VISUAL) == (losses(turning_in, SEPARATION, IFR) if lost else [])
    assert losses(diagonal, SEPARATION, IFR) and losses(diagonal, SEPARATION, VISUAL) == []
    # neither established, the one 35° off: the IFR rule
    both_turning = traffic((-8_000, 0, 700, "I1", False, "F", 10.0), (-9_000, 1_800, 800, "I2", False, "F", 35.0))
    assert losses(both_turning, SEPARATION, VISUAL) == losses(both_turning, SEPARATION, IFR) != []


#: One pair of runways per relation, and a pair with one runway not said yet.
PAIRS = {SAME: ("R", "R"), SINGLE: ("S1", "S2"), DEPENDENT: ("L1", "L2"), INDEPENDENT: ("I1", "I2"),
         UNRELATED: ("R", "X"), NO_RUNWAY: ("R", None)}


def test_the_visual_reading_is_the_ifr_reading_less_exactly_what_7_4_4_c_frees():
    """Every relation × who is established × track less course × distance off the centreline × order: VISUAL is IFR,
    except none between parallels 2,500 ft or more apart that are both turned in (within 30° of the course, on their own
    side of the midline), and none between established finals of other directions."""
    limit = FAA_VISUAL_INTERCEPT_MAX_DEG

    def turned_in(runway: str, other: str, angle: float, right: float) -> bool:
        other_right_m = SEPARATION.right_nm[(runway, other)] * NM_M
        return abs(angle) <= limit and right * math.copysign(1.0, other_right_m) < abs(other_right_m) / 2

    kinematics = list(itertools.product((0.0, limit, limit + 1.0, -limit, -limit - 1.0, 90.0),
                                        (-400.0, 0.0, 400.0, 1_500.0, -1_500.0)))
    for relation, (ra, rb) in PAIRS.items():
        for est_a, est_b in itertools.product((False, True), repeat=2):
            if rb is None and est_b:
                continue
            for (angle_a, right_a), (angle_b, right_b) in itertools.product(kinematics, repeat=2):
                a = (-3_000.0, 0.0, 600.0, ra, est_a, "F", angle_a, right_a)
                b = (-4_000.0, 1_100.0, 600.0, rb, est_b, "F", angle_b, right_b)
                freed = ((relation in (DEPENDENT, INDEPENDENT) and turned_in(ra, rb, angle_a, right_a)
                          and turned_in(rb, ra, angle_b, right_b))
                         or (relation == UNRELATED and est_a and est_b))
                for scene in (traffic(a, b), traffic(b, a)):
                    ifr = losses(scene, SEPARATION, IFR)
                    assert ifr or (relation == INDEPENDENT and est_a and est_b)      # the scene is not vacuous
                    assert losses(scene, SEPARATION, VISUAL) == ([] if freed else ifr), (relation, a, b)


def test_the_visual_reading_keeps_a_close_pair_as_one_runway():
    """7-4-4 c1 (N JO 7110.805): a visual approach beside a pair under 2,500 ft needs visual separation (c1 b)."""
    side_by_side = traffic((-5_000, 0, 700, "S1", True, "F"), (-8_000, 150, 700, "S2", True, "F"))
    assert [loss.kind for loss in losses(side_by_side, SEPARATION, VISUAL)] == [IN_TRAIL]
    assert losses(side_by_side, SEPARATION, VISUAL) == losses(side_by_side, SEPARATION, IFR)
    across = traffic((0, 0, 50, "S1", True, "D"), (-9_000, 150, 550, "S2", True, "H"))
    assert wake_at_threshold(across, 0, SEPARATION) is not None


def test_the_visual_reading_keeps_one_runway_and_vectored_traffic_and_drops_established_converging_finals():
    in_trail = traffic((-5_000, 0, 700, "R", True, "F"), (-10_000, 0, 300, "R", True, "F"))
    assert [loss.kind for loss in losses(in_trail, SEPARATION, VISUAL)] == [IN_TRAIL]
    converging_established = traffic((-3_000, 0, 700, "R", True, "F"), (-3_000, 3_000, 700, "X", True, "F"))
    converging_vectored = traffic((-3_000, 0, 700, "R", True, "F"), (-3_000, 3_000, 700, "X", False, "F"))
    assert losses(converging_established, SEPARATION, VISUAL) == []
    assert [loss.kind for loss in losses(converging_vectored, SEPARATION, VISUAL)] == [RADAR_OR_VERTICAL]
    # one aircraft joining the other's runway, and one with no runway yet: radar or vertical, as under IFR
    joining = traffic((-5_000, 0, 700, "R", True, "F"), (-7_000, 2_000, 800, "R", False, "F"))
    no_runway = traffic((-5_000, 0, 700, "R", True, "F"), (-7_000, 2_000, 800, None, False, "F"))
    for scene in (joining, no_runway):
        assert [loss.kind for loss in losses(scene, SEPARATION, VISUAL)] == [RADAR_OR_VERTICAL]
        assert losses(scene, SEPARATION, VISUAL) == losses(scene, SEPARATION, IFR)


@pytest.mark.parametrize("scene", [
    Traffic(*[np.zeros(0)] * 3, (), *[np.zeros(0)] * 3, np.zeros(0, bool), ()),              # nobody to judge
    traffic((-5_000, 0, 700, "R", True, "F"), (-7_000, 2_000, 800, "R", False, "F")),   # not both established
])
def test_a_reading_not_listed_is_refused_whatever_the_scene(scene):
    with pytest.raises(ValueError, match="reading"):
        losses(scene, SEPARATION, "Visual")


def test_the_visual_reading_never_judges_a_pair_the_ifr_reading_would_not():
    random = np.random.default_rng(7)
    runways = [None, "R", "S1", "S2", "L1", "L2", "I1", "I2", "X"]
    for _ in range(2_000):
        aircraft = []
        for _ in range(3):
            runway = runways[random.integers(len(runways))]
            aircraft.append((random.uniform(-12_000, 0), random.uniform(-3_000, 3_000), random.uniform(0, 900), runway,
                             runway is not None and bool(random.integers(2)), "FDH"[random.integers(3)],
                             random.uniform(-180, 180), random.uniform(-2_000, 2_000)))
        scene = traffic(*aircraft)
        assert set(losses(scene, SEPARATION, VISUAL)) <= set(losses(scene, SEPARATION, IFR))


def test_next_behind_finds_the_established_one_on_the_runway_or_its_close_pair():
    scene = traffic((0, 0, 50, "S1", True, "F"), (-4_000, 150, 250, "S2", True, "F"), (-3_000, 0, 200, "S1", False, "F"),
                    (-2_000, 1_800, 150, "I1", True, "F"))
    assert next_behind(scene, 0, SEPARATION) == 1
