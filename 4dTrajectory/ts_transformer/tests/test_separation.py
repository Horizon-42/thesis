"""Loss of separation at one instant (`inference.separation`, multi-aircraft design §3.2)."""

import math

import numpy as np
import pytest
from geokit import FT_M, NM_M

from ts_transformer.inference.runway_schedule import (
    CWT_DIRECTLY_BEHIND_NM,
    CWT_ON_APPROACH_NM,
    DEPENDENT,
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
    wake_at_threshold,
)

RADAR_M = 3.0 * NM_M
VERTICAL_M = 1_000.0 * FT_M
#: One runway "R", a pair separated as one "S1"/"S2", a dependent parallel pair "L1"/"L2" (1.0 NM diagonal), an
#: independent pair "I1"/"I2", all landing east; "X" is a runway of another direction.
SEPARATION = Separation(same_nm=3.0, speed_mps=70.0,
                        relations={frozenset(("S1", "S2")): SINGLE, frozenset(("L1", "L2")): DEPENDENT,
                                   frozenset(("I1", "I2")): INDEPENDENT},
                        spacing_nm={frozenset(("S1", "S2")): 0.1, frozenset(("L1", "L2")): 0.6,
                                    frozenset(("I1", "I2")): 1.5},
                        diagonal_nm={frozenset(("L1", "L2")): 1.0}, wake_nm=CWT_ON_APPROACH_NM)


def traffic(*aircraft) -> Traffic:
    """Each aircraft: (east m, north m, height m, runway, established, category); on the approach clock at its east."""
    e, n, h, runway, established, category = zip(*aircraft)
    along = np.array([x if r is not None else np.nan for x, r in zip(e, runway)], dtype=float)
    return Traffic(np.array(e, float), np.array(n, float), np.array(h, float), tuple(runway), along,
                   np.array(established, bool), tuple(category))


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
    loss = wake_at_threshold(scene, 0, SEPARATION, IFR)
    assert (loss.kind, loss.j, loss.responsible, loss.relation) == (AT_THRESHOLD, 1, (1,), SAME)
    assert loss.required_m == pytest.approx(6.0 * NM_M) and loss.distance_m == pytest.approx(9_000)
    # not yet established, another runway, an untyped follower: not judged
    assert wake_at_threshold(traffic((0, 0, 50, "R", True, "D"), (-9_000, 0, 550, "R", False, "H")), 0, SEPARATION, IFR) is None
    assert wake_at_threshold(traffic((0, 0, 50, "R", True, "D"), (-9_000, 0, 550, "I1", True, "H")), 0, SEPARATION, IFR) is None
    assert wake_at_threshold(traffic((0, 0, 50, "R", True, "D"), (-9_000, 0, 550, "R", True, None)), 0, SEPARATION, IFR) is None


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
    assert wake_at_threshold(traffic((0, 0, 50, "R", True, "F"), (-3_000, 0, 250, "R", True, "F")), 0, SEPARATION, IFR) is None
    exactly = traffic((0, 0, 50, "R", True, "D"), (-6.0 * NM_M, 0, 550, "R", True, "H"))
    assert wake_at_threshold(exactly, 0, SEPARATION, IFR) is None
    across = wake_at_threshold(traffic((0, 0, 50, "S1", True, "D"), (-9_000, 150, 550, "S2", True, "H")), 0, SEPARATION, IFR)
    assert across.relation == SINGLE


@pytest.mark.parametrize("runway, along, established", [
    (None, 0.0, False),              # on the approach clock with no runway in force
    ("R", float("nan"), False),      # a runway in force, not on the approach clock
    (None, float("nan"), True),      # established with no runway
])
def test_traffic_refuses_a_broken_contract(runway, along, established):
    with pytest.raises(ValueError):
        Traffic(np.zeros(1), np.zeros(1), np.zeros(1), (runway,), np.array([along]), np.array([established]), ("F",))


def test_the_visual_reading_sets_no_minimum_between_runways_of_one_direction():
    # side by side on a close pair, turning in beside a parallel final, both established on dependent parallels
    side_by_side = traffic((-5_000, 0, 700, "S1", True, "F"), (-5_000, 150, 700, "S2", True, "F"))
    turning_in = traffic((-8_000, 0, 700, "I1", True, "F"), (-9_000, 1_800, 800, "I2", False, "F"))
    diagonal = traffic((-5_000, 0, 700, "L1", True, "F"), (-6_000, 1_100, 700, "L2", True, "F"))
    for scene in (side_by_side, turning_in, diagonal):
        assert losses(scene, SEPARATION, IFR) and losses(scene, SEPARATION, VISUAL) == []
    # no wake across a close pair at the threshold either
    across = traffic((0, 0, 50, "S1", True, "D"), (-9_000, 150, 550, "S2", True, "H"))
    assert wake_at_threshold(across, 0, SEPARATION, IFR) is not None
    assert wake_at_threshold(across, 0, SEPARATION, VISUAL) is None


def test_the_visual_reading_keeps_one_runway_and_vectored_traffic_and_drops_established_converging_finals():
    in_trail = traffic((-5_000, 0, 700, "R", True, "F"), (-10_000, 0, 300, "R", True, "F"))
    assert [loss.kind for loss in losses(in_trail, SEPARATION, VISUAL)] == [IN_TRAIL]
    converging_established = traffic((-3_000, 0, 700, "R", True, "F"), (-3_000, 3_000, 700, "X", True, "F"))
    converging_vectored = traffic((-3_000, 0, 700, "R", True, "F"), (-3_000, 3_000, 700, "X", False, "F"))
    assert losses(converging_established, SEPARATION, VISUAL) == []
    assert [loss.kind for loss in losses(converging_vectored, SEPARATION, VISUAL)] == [RADAR_OR_VERTICAL]
    with pytest.raises(ValueError, match="reading"):
        losses(in_trail, SEPARATION, "vfr")
