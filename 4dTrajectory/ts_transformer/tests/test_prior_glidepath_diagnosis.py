"""The glidepath diagnosis (`experiments.prior_glidepath_diagnosis`, prior readouts §12): the what-if's one-line change of
the vertical law, the executor and the observed aircraft paired at the state the word clock matched, the words a cycle
flew, the reading of a stop, and the measures over a replay's cycles."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot import vertical
from ts_transformer.autopilot.frame import Kinematics
from ts_transformer.experiments import prior_glidepath_diagnosis as diagnosis
from ts_transformer.experiments.prior_glidepath_diagnosis import (
    SHALLOWEST, WHAT_IFS, at_faf, flight_series, glidepath_m, height_lost, longest_run, observed_class,
    read_stop, run_angles,
)
from ts_transformer.experiments.prior_free_generation import in_force
from ts_transformer.instructions.words import ALTITUDE, ANGLE, RUNWAY, UNCHANGED, Words
from ts_transformer.prior.procedure import GLIDEPATH_BELOW_M, track_tolerance_m
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.support import instruction_airport, instruction_spec
from ts_transformer.tests.test_autopilot import TEST_TCH_M, _params
from ts_transformer.tests.test_prior_procedure import RUNWAY_09, TAN, _final, _flown

F64 = torch.float64
WORDS = Words(instruction_spec())


def _aim(captured: bool, above_threshold_m: float = 300.0, to_go_m: float = 8_000.0,
         glidepath_tan: float = 0.0) -> tuple[float, bool]:
    """The path angle the vertical law asks for (descending positive), deg, on its first cycle under "descend to land"
    with the shallowest class, level at 75 m/s ``above_threshold_m`` over a threshold ``to_go_m`` ahead on the course,
    and whether it left the tube. From 300 m at 8 km the line to the crossing point (about 2°) is steeper than the class
    and the tube can still reach the landing; from 600 m at 3 km it cannot. The glidepath is given at 0°, which puts
    its lower edge (`vertical`) flat at the TCH less 60 m, out of the way: the what-if is about the tube."""
    params = _params()
    law = vertical.Vertical(1, params, WORDS, torch.device("cpu"))
    state = Kinematics(*(torch.tensor([v], dtype=F64)
                         for v in (0.0, 0.0, 100.0 + above_threshold_m, 75.0, 90.0, 0.0, 75.0, 60_000.0)))
    _, wanted, modes = law.rate(state, torch.tensor([math.nan], dtype=F64), torch.tensor([True]),
                                torch.tensor([SHALLOWEST]), torch.tensor([WORDS.angle_deg(SHALLOWEST)], dtype=F64),
                                torch.tensor([[0, 0]]), torch.tensor([to_go_m], dtype=F64),
                                torch.tensor([100.0], dtype=F64), torch.tensor([TEST_TCH_M], dtype=F64),
                                torch.tensor([glidepath_tan], dtype=F64), torch.zeros(1, dtype=F64),
                                torch.tensor([to_go_m], dtype=F64),
                                torch.tensor([captured]), torch.tensor([False]))
    # level now: the reference is −aim
    return -math.degrees(float(wanted[0]) * params.path_time_constant_s), bool(modes["aim_left_tube"][0])


def test_the_what_if_flies_the_class_centre_inside_the_tube_after_the_capture_only_and_restores_the_law():
    law = vertical.Vertical.rate
    current = {case: _aim(*case) for case in ((True,), (False,), (True, 600.0, 3_000.0))}
    with diagnosis.law_changed("class_centre_in_tube"):
        assert vertical.Vertical.rate is not law
        centre = {case: _aim(*case) for case in current}
    assert vertical.Vertical.rate is law
    # captured, in reach: the law aims at the crossing point, past the class's steep edge within the tube's tolerance;
    # the what-if flies the class's nominal angle
    assert not current[(True,)][1] and current[(True,)][0] > WORDS.angle_bounds(SHALLOWEST)[1]
    assert centre[(True,)][0] == pytest.approx(WORDS.angle_deg(SHALLOWEST))
    # before the capture both fly the nominal angle (never steeper than the straight line); out of the tube's reach both
    # leave it for the crossing point
    assert current[(False,)][0] == pytest.approx(WORDS.angle_deg(SHALLOWEST))
    assert current[(True, 600.0, 3_000.0)][1]
    for case in ((False,), (True, 600.0, 3_000.0)):
        assert centre[case] == pytest.approx(current[case])


def test_the_law_joins_the_glidepath_from_below_after_the_capture_and_aims_at_the_crossing_point_on_or_above_it():
    """Executor v11 (the shallow classes' readout, 2026-09-27): captured, below the published glidepath (3°: 434 m up at
    8 km) but above its lower edge (374 m, where the floor already levels it), the shallowest class flies level — the
    approach joined from below, as the observed aircraft fly it — where the law up to v10 (`toward_below_glidepath`)
    descended past the class's steep edge toward the crossing point; on or above the glidepath, and before the capture,
    the two are one. `join_from_below_centre` flies the class's nominal angle above it."""
    tan3 = math.tan(math.radians(3.0))
    below, above = (True, 400.0, 8_000.0, tan3), (True, 450.0, 8_000.0, tan3)   # above: in the tube's reach
    uncaptured = (False, 400.0, 8_000.0, tan3)
    law = {case: _aim(*case) for case in (below, above, uncaptured)}
    with diagnosis.law_changed("toward_below_glidepath"):
        before = {case: _aim(*case) for case in law}
    with diagnosis.law_changed("join_from_below_centre"):
        centred = _aim(*above)
    assert law[below][0] == pytest.approx(0.0) and before[below][0] > 1.0
    assert law[above] == pytest.approx(before[above]) and law[uncaptured] == pytest.approx(before[uncaptured])
    assert centred[0] == pytest.approx(min(WORDS.angle_deg(SHALLOWEST), law[above][0]))
    assert set(WHAT_IFS) == {"toward_below_glidepath", "class_centre_in_tube", "join_from_below_centre",
                             "class_centre_own_reach", "join_from_below_centre_own_reach"}
    # every what-if's lines are the law's, each once
    for name in WHAT_IFS:
        with diagnosis.law_changed(name):
            pass


def test_the_what_if_refuses_a_law_that_does_not_hold_its_line_once_and_restores_on_an_error(monkeypatch):
    law = vertical.Vertical.rate
    monkeypatch.setitem(diagnosis.WHAT_IFS, "class_centre_in_tube", (("no such line", "x"),))
    with pytest.raises(RuntimeError, match="0 copies"):
        with diagnosis.law_changed("class_centre_in_tube"):
            pass
    monkeypatch.undo()
    with pytest.raises(KeyError):
        with diagnosis.law_changed("class_centre_in_tube"):
            raise KeyError("inside")
    assert vertical.Vertical.rate is law


def _replay(clock: np.ndarray | None):
    """20 cycles of 1 s down runway 09's centreline, 100 m a cycle on a 3° path 80 m under the glidepath; observed rows
    every 200 m on the same path, so state ``c`` sits on observed row ``c / 2`` at every even ``c``. The sentence: runway
    1 from row 3, the shallowest class and "descend to land" from row 5; the lateral capture from cycle 10."""
    final, cycles = _final(), 20
    d = np.linspace(9_000.0, 7_000.0, cycles + 1)
    flown = _flown(d, final.crossing_m + d * TAN - 80.0, cycles - 1, clock)
    flown.modes = {"captured": torch.arange(cycles)[None] >= 10}
    grid = np.full((cycles // 2 + 1, 6), UNCHANGED, dtype=np.int64)       # the last cycle may hear row 10
    grid[0] = 0                                                           # the first row says every column
    grid[3, RUNWAY], grid[5, ALTITUDE], grid[5, ANGLE] = 1, WORDS.altitude_land, SHALLOWEST
    d_o = 9_000.0 - 200.0 * np.arange(len(grid))
    # the rows before the first predicted step are not the sentence's: a series reading them reads 99,999 m
    lead = np.full(N_LOOK, 99_999.0)
    signal = SimpleNamespace(e_m=np.concatenate([lead, -d_o]), n_m=np.concatenate([lead, 0.0 * d_o]),
                             altitude_m=np.concatenate([lead, final.crossing_m + d_o * TAN - 80.0]))
    away = replace(final, candidate=replace(RUNWAY_09, threshold_e_m=50_000.0))
    return flight_series(flown, 0, grid, signal, instruction_airport(), (final, away), WORDS), d, final, away


def test_a_replay_s_series_pairs_each_state_with_the_row_the_clock_matched_to_it():
    """State ``c`` (the start of cycle ``c``) against the row heard in cycle ``c``: on the time clock, row ``c // 2`` —
    the same point at every even ``c``, the observed aircraft 100 m farther out (higher) at every odd one."""
    one, d, final, away = _replay(None)
    even = np.arange(20) % 2 == 0
    gap = one["h_x"] - one["h_o"]
    assert np.allclose(gap[even], 0.0) and np.allclose(gap[~even], -100.0 * TAN)
    assert np.allclose(one["h_x"], final.crossing_m + d[:20] * TAN - 80.0)
    # the runway that brought the executor to each state: row 3 is flown from cycle 6, so states from 7 are runway 1's
    ours = np.arange(20) < 7
    assert np.array_equal(one["into"], (~ours).astype(int))
    assert np.allclose((one["h_x"] - one["gp_x"])[ours], -80.0)
    assert np.allclose(one["d_x"][~ours], away.axes(-d[:20][~ours], 0.0 * d[:20][~ours])[0])
    assert np.array_equal(one["land"], np.arange(20) >= 10)
    assert np.array_equal(one["captured"], np.arange(20) >= 10)
    assert np.allclose(glidepath_m(final, [0.0, 1_000.0]), [final.crossing_m, final.crossing_m + 1_000.0 * TAN])


def test_a_cycle_flies_the_words_heard_at_its_step_s_start_and_reads_the_row_heard_in_it():
    """A clock one row ahead in the second cycle of each step (as the track clock can be): the observed aircraft is the
    row heard in that cycle, but the words are still the step's first cycle's."""
    one, _d, _final_, _away = _replay(2.0 * ((np.arange(20) + 1) // 2))
    even = np.arange(20) % 2 == 0
    gap = one["h_x"] - one["h_o"]
    assert np.allclose(gap[even], 0.0) and np.allclose(gap[~even], 100.0 * TAN)
    assert np.array_equal(one["land"], np.arange(20) >= 10)               # not from cycle 9, which heard row 5
    assert np.array_equal(one["angle"] == SHALLOWEST, np.arange(20) >= 10)
    assert np.array_equal(one["runway"], (np.arange(20) >= 6).astype(int))


def _stop_series(executor_below_m: float, observed_below_m: float, d_o: float = 8_000.0) -> dict[str, np.ndarray]:
    """Six states on runway 09's final; at state 4 the executor 8 km out and ``executor_below_m`` under the glidepath,
    the observed aircraft ``d_o`` out and ``observed_below_m`` under it."""
    final = _final()
    d_x = np.array([9_600.0, 9_200.0, 8_800.0, 8_400.0, 8_000.0, 7_600.0])
    d_obs = d_x.copy()
    d_obs[4] = d_o
    gp_x, gp_o = glidepath_m(final, d_x), glidepath_m(final, d_obs)
    h_x, h_o = gp_x.copy(), gp_o.copy()
    h_x[4] -= executor_below_m
    h_o[4] -= observed_below_m
    return {"h_x": h_x, "h_o": h_o, "e_o": -d_obs, "n_o": 0.0 * d_obs, "d_x": d_x, "d_o": d_obs, "gp_x": gp_x,
            "gp_o": gp_o, "faf_d": np.full(6, final.faf_d_m), "into": np.zeros(6, dtype=int),
            "captured": np.array([0, 0, 1, 1, 1, 1], dtype=bool)}


def test_a_stop_is_read_at_its_state_under_the_step_s_words_and_refused_where_it_is_not_the_check_s():
    spec, final = WORDS.spec, _final()
    grid = np.zeros((4, 6), dtype=np.int64)
    grid[2, ALTITUDE], grid[2, ANGLE] = WORDS.altitude_land, SHALLOWEST
    force = in_force(grid)
    line = GLIDEPATH_BELOW_M + track_tolerance_m(spec)
    read = read_stop(_stop_series(line + 20.0, -3.0), 4, 2, force, (final,), spec, WORDS)
    assert read["executor_vs_glidepath_m"] == pytest.approx(-(line + 20.0))
    assert read["observed_vs_glidepath_m"] == pytest.approx(3.0)
    assert read["executor_minus_observed_m"] == pytest.approx(-(line + 23.0))
    assert read["start_executor_minus_observed_m"] == pytest.approx(0.0)
    assert (read["observed"], read["altitude"], read["angle_class"], read["captured"]) == (
        "on the glidepath", "descend to land", SHALLOWEST, True)
    assert read["runway"] == final.ident and read["d_m"] == pytest.approx(8_000.0)
    with pytest.raises(ValueError, match="not beyond the stop line"):
        read_stop(_stop_series(line - 1.0, 0.0), 4, 2, force, (final,), spec, WORDS)
    pointed = force.copy()
    pointed[2, RUNWAY] = 1
    with pytest.raises(ValueError, match="the check 1"):
        read_stop(_stop_series(line + 20.0, 0.0), 4, 2, pointed, (final, final), spec, WORDS)
    with pytest.raises(ValueError, match="past the last state"):
        read_stop(_stop_series(line + 20.0, 0.0), 6, 2, force, (final,), spec, WORDS)


def test_the_observed_aircraft_is_stopped_itself_only_where_the_check_would_stop_it():
    spec, final = WORDS.spec, _final()
    line = GLIDEPATH_BELOW_M + track_tolerance_m(spec)

    def seen(observed_below_m, d_o=8_000.0):
        return observed_class(_stop_series(line + 20.0, observed_below_m, d_o), 4, final, spec)

    assert seen(-50.0) == seen(30.0) == "on the glidepath"
    assert seen(30.5) == seen(GLIDEPATH_BELOW_M) == "above the lower edge"
    assert seen(GLIDEPATH_BELOW_M + 1.0) == seen(line) == "below the lower edge"
    assert seen(line + 1.0) == "stopped itself"
    assert seen(line + 1.0, d_o=final.faf_d_m + 500.0) == "below the lower edge"    # outside the FAF: no edge there


def _series(h_x, h_o, land, angle, captured, d_x=None, d_o=None, faf_d=10_000.0):
    n = len(h_x)
    d_x = np.linspace(12_000.0, 12_000.0 - 150.0 * (n - 1), n) if d_x is None else np.asarray(d_x, dtype=float)
    d_o = d_x.copy() if d_o is None else np.asarray(d_o, dtype=float)
    return {"h_x": np.asarray(h_x, dtype=float), "h_o": np.asarray(h_o, dtype=float), "d_x": d_x, "d_o": d_o,
            "gp_x": 15.0 + d_x * TAN, "gp_o": 15.0 + d_o * TAN, "faf_d": np.full(n, faf_d),
            "land": np.asarray(land, dtype=bool), "angle": np.asarray(angle), "captured": np.asarray(captured, dtype=bool),
            "runway": np.zeros(n, dtype=int), "into": np.zeros(n, dtype=int)}


def test_the_height_lost_is_each_cycle_s_fall_below_the_observed_aircraft_summed_by_what_it_flew():
    one = _series([500, 490, 470, 460], [500, 500, 500, 490], [0, 1, 1, 1], [3, 1, 1, 3], [0, 0, 1, 1])
    two = _series([300, 290], [300, 300], [1, 1], [1, 1], [0, 0])
    rows = height_lost([one, two], cycle_s=1.0)
    by = {(r["altitude"], r["angle_class"], r["captured"]): (r["executor_lost_m"], r["seconds"]) for r in rows}
    # cycle c's words take the change from state c to state c + 1; the last state's words take none
    assert by == {("level", 3, False): (10.0, 1.0), ("descend to land", 1, False): (30.0, 2.0),
                  ("descend to land", 1, True): (0.0, 1.0)}
    assert [r["executor_lost_m"] for r in rows] == sorted((r["executor_lost_m"] for r in rows), reverse=True)


def test_the_longest_run_is_the_earliest_of_equals():
    assert longest_run(np.array([0, 1, 1, 0, 1, 1], dtype=bool)) == (1, 2)
    assert longest_run(np.array([1, 0, 1, 1, 1], dtype=bool)) == (2, 4)
    assert longest_run(np.array([1], dtype=bool)) == (0, 0)
    assert longest_run(np.zeros(3, dtype=bool)) is None


def test_a_class_s_run_after_the_capture_reads_both_angles_over_the_distance_covered():
    n, step = 41, 150.0                                                   # 40 s, 6 km
    d = 12_000.0 - step * np.arange(n)
    h_x = 800.0 - np.tan(np.radians(2.0)) * step * np.arange(n)
    h_o = 850.0 - np.tan(np.radians(1.0)) * step * np.arange(n)
    one = _series(h_x, h_o, np.ones(n), np.full(n, SHALLOWEST), np.ones(n), d_x=d)
    read = run_angles(one, SHALLOWEST, cycle_s=1.0)
    assert read["executor_deg"] == pytest.approx(2.0) and read["observed_deg"] == pytest.approx(1.0)
    assert read["observed_vs_glidepath_m"] == pytest.approx(850.0 - 15.0 - 12_000.0 * TAN)
    lost = (np.tan(np.radians(2.0)) - np.tan(np.radians(1.0))) * step * 40                 # m over 40 s
    assert read["executor_lost_m_per_min"] == pytest.approx(lost / 40.0 * 60.0) and lost > 0.0
    assert run_angles(one, SHALLOWEST + 1, cycle_s=1.0) is None                # no run of that class
    short = _series(h_x[:15], h_o[:15], np.ones(15), np.full(15, SHALLOWEST), np.ones(15), d_x=d[:15])
    assert run_angles(short, SHALLOWEST, cycle_s=1.0) is None                  # 14 s < 20 s
    slow = _series(h_x, h_o, np.ones(n), np.full(n, SHALLOWEST), np.ones(n), d_x=12_000.0 - 10.0 * np.arange(n))
    assert run_angles(slow, SHALLOWEST, cycle_s=1.0) is None                   # 400 m < 500 m along the course
    moved = _series(h_x, h_o, np.ones(n), np.full(n, SHALLOWEST), np.ones(n), d_x=d)
    moved["into"][:5] = 1                                  # the first states were reached under another runway's words
    assert run_angles(moved, SHALLOWEST, cycle_s=1.0)["executor_deg"] == pytest.approx(2.0)
    moved["into"][15] = 1                                  # a state mid-run under another runway splits the run: 16–40 left
    assert run_angles(moved, SHALLOWEST, cycle_s=1.0)["executor_deg"] == pytest.approx(2.0)
    moved["into"][28] = 1                                  # now every piece is under 20 s
    assert run_angles(moved, SHALLOWEST, cycle_s=1.0) is None


def test_the_faf_reading_is_the_first_state_inside_it_reached_on_a_captured_cycle():
    d = np.array([12_000.0, 9_950.0, 9_900.0, 9_000.0])
    one = _series([900, 700, 600, 500], [900, 750, 640, 520], [1, 1, 1, 1], [SHALLOWEST, 3, 3, 3], [0, 1, 0, 1], d_x=d)
    read = at_faf(one)                     # state 1 is inside but cycle 0 was not captured; cycle 1 brought state 2
    assert read["executor_minus_observed_m"] == pytest.approx(-40.0)
    assert read["executor_vs_glidepath_m"] == pytest.approx(600.0 - 15.0 - 9_900.0 * TAN)
    assert read["shallowest_before"]
    one["angle"][0] = 3
    assert not at_faf(one)["shallowest_before"]
    one["captured"][:] = False
    assert at_faf(one) is None
