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
    Separation,
)
from ts_transformer.inference.separation import (
    AT_THRESHOLD,
    DIAGONAL,
    IN_TRAIL,
    NO_RUNWAY,
    RADAR_OR_VERTICAL,
    Traffic,
    losses,
    wake_at_threshold,
)

RADAR_M = 3.0 * NM_M
VERTICAL_M = 1_000.0 * FT_M
#: One runway "R", a dependent parallel pair "L1"/"L2" (1.0 NM diagonal), an independent pair "I1"/"I2", all landing east.
SEPARATION = Separation(same_nm=3.0, speed_mps=70.0,
                        relations={frozenset(("L1", "L2")): DEPENDENT, frozenset(("I1", "I2")): INDEPENDENT},
                        spacing_nm={frozenset(("L1", "L2")): 0.6, frozenset(("I1", "I2")): 1.5},
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
    found = losses(traffic((-5_000, 0, 700, "R", True, "F"), (-10_000, 0, 300, "R", True, "F")), SEPARATION)
    assert [(loss.kind, loss.responsible) for loss in found] == [(IN_TRAIL, (1,))]      # the one 10 km out
    assert found[0].required_m == pytest.approx(RADAR_M)
    # 6 km: clear of 3 NM; behind a B the directly-behind wake minimum (5 NM) binds
    assert losses(traffic((-4_000, 0, 700, "R", True, "F"), (-10_000, 0, 700, "R", True, "F")), SEPARATION) == []
    wake = losses(traffic((-2_000, 0, 700, "R", True, "B"), (-10_000, 0, 700, "R", True, "F")), SEPARATION)
    assert wake[0].required_m == pytest.approx(5.0 * NM_M) and wake[0].responsible == (1,)


def test_a_pair_not_both_established_keeps_radar_or_vertical_and_the_joining_one_answers():
    joining = losses(traffic((-5_000, 0, 700, "R", True, "F"), (-9_000, 1_000, 600, "R", False, "F")), SEPARATION)
    assert [(loss.kind, loss.responsible) for loss in joining] == [(RADAR_OR_VERTICAL, (1,))]
    stacked = traffic((-5_000, 0, 700, "R", False, "F"), (-9_000, 1_000, 700 + VERTICAL_M + 1, "R", False, "F"))
    assert losses(stacked, SEPARATION) == []                                   # vertically separated
    converging = losses(traffic((0, 4_000, 900, None, False, "F"), (0, 0, 800, "R", False, "F")), SEPARATION)
    assert [(loss.relation, loss.responsible) for loss in converging] == [(NO_RUNWAY, (0, 1))]


def test_parallel_finals_dependent_by_the_diagonal_independent_not_at_all():
    diagonal = losses(traffic((-5_000, 0, 700, "L1", True, "F"), (-6_000, 1_100, 700, "L2", True, "F")), SEPARATION)
    assert [(loss.kind, loss.responsible) for loss in diagonal] == [(DIAGONAL, (1,))]
    assert diagonal[0].distance_m == pytest.approx(math.hypot(1_000, 1_100))
    assert losses(traffic((-5_000, 0, 700, "I1", True, "F"), (-5_100, 800, 700, "I2", True, "F")), SEPARATION) == []


def test_an_untyped_aircraft_in_trail_is_judged_on_the_radar_minimum_and_says_so():
    found = losses(traffic((-2_000, 0, 700, "R", True, None), (-7_000, 0, 700, "R", True, "F")), SEPARATION)
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
