"""The faults of an observed track (vocabulary D111, §12.1 A40): `instructions/faults.py`."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ts_transformer.instructions.faults import (
    AROUND_STEPS, HELD, JUMP, JUMP_RATIO, REVERSAL, REVERSAL_DEG, Fault, faulty_flights, track_faults,
)

from ts_transformer.tests.support import (
    fixture_days, fly_legs, instruction_airport, instruction_flight, instruction_spec,
)


def _east(e_m) -> object:
    """A level track east along n = 0 through the positions ``e_m`` (2 s rows)."""
    e = np.asarray(e_m, dtype=np.float64)
    n = np.zeros_like(e)
    return instruction_flight(e, n, np.full_like(e, 1000.0), np.full_like(e, 90.0), np.full_like(e, 50.0))


def test_a_clean_track_has_no_fault():
    """No fault in a flight's own manoeuvres: a holding turn at the standard rate (3°/s), a speed change at the
    vocabulary's a_max, a level-off, a descent and a final — the labeller's synthetic approach included."""
    a_max = instruction_spec().speed_accel_max_mps2
    speeds = [100.0 - a_max * 2.0 * k for k in range(15)]                       # slowed at a_max, row by row
    legs = [(30, 6.0, 100.0, 0.0), (20, 0.0, 100.0, -5.0), (10, 0.0, 100.0, 0.0),
            *[(1, 0.0, v, 0.0) for v in speeds], (40, -6.0, speeds[-1], -3.0), (60, 0.0, speeds[-1], -3.9)]
    for legs_flown in (legs, [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0)]):
        assert track_faults(instruction_flight(*fly_legs(legs_flown, 270.0, 1500.0, -400.0, 0.0))) == ()


def test_a_jump_a_held_position_and_a_reversal_are_marked_with_their_row():
    straight = 100.0 * np.arange(40)
    jumped = straight.copy()
    jumped[20] += 1000.0                                         # one position 1 km ahead, then back on the track
    assert track_faults(_east(jumped)) == (Fault(JUMP, 19), Fault(JUMP, 20), Fault(REVERSAL, 20), Fault(REVERSAL, 21))
    held = straight.copy()
    held[11:14] = held[10]                                       # held for three rows, then the catch-up of 4 steps
    assert track_faults(_east(held)) == (Fault(HELD, 10), Fault(HELD, 11), Fault(HELD, 12), Fault(JUMP, 13))
    back = straight.copy()
    back[21:] -= 200.0                                           # a step back, then on again
    assert track_faults(_east(back)) == (Fault(REVERSAL, 20), Fault(REVERSAL, 21))


def test_the_rules_edges_decide_as_written():
    """A step of exactly `JUMP_RATIO` times the median around it is no jump, nor one of a third a held position; a turn
    a degree under `REVERSAL_DEG` is no reversal, a degree over one is."""
    assert (JUMP_RATIO, AROUND_STEPS, REVERSAL_DEG) == (3.0, 5, 120.0)
    straight = 100.0 * np.arange(40)
    held = straight.copy()
    held[11:13] = held[10]                                       # held two rows: the catch-up is exactly 3 steps
    assert track_faults(_east(held)) == (Fault(HELD, 10), Fault(HELD, 11))
    steps = np.full(39, 96.0)
    steps[20] = 32.0                                             # exactly a third (both exact in binary)
    assert track_faults(_east(np.concatenate([[0.0], np.cumsum(steps)]))) == ()      # not held
    steps[20] = 31.0
    assert track_faults(_east(np.concatenate([[0.0], np.cumsum(steps)]))) == (Fault(HELD, 20),)
    for turn_deg, expected in ((REVERSAL_DEG - 1.0, ()), (REVERSAL_DEG + 1.0, (Fault(REVERSAL, 20),))):
        bearing = np.radians(90.0 + turn_deg)                    # a turn right from east, at row 20
        e = np.concatenate([straight[:21], straight[20] + 100.0 * np.sin(bearing) * np.arange(1, 20)])
        n = np.concatenate([np.zeros(21), 100.0 * np.cos(bearing) * np.arange(1, 20)])
        assert track_faults(replace(_east(np.zeros(40)), e_m=e, n_m=n)) == expected, turn_deg


def _steps(steps) -> object:
    """A level track east whose 2 s steps are ``steps``."""
    return _east(np.concatenate([[0.0], np.cumsum(np.asarray(steps, dtype=np.float64))]))


def test_the_median_reads_five_steps_each_side_never_the_step_itself_and_fewer_at_the_ends():
    # never itself: 90 m ×5 and 110 m ×5 around a 301 m step have the median 100 (with itself: 110, no jump)
    assert Fault(JUMP, 20) in track_faults(_steps([100.0] * 15 + [90.0] * 5 + [301.0] + [110.0] * 5 + [100.0] * 15))
    # the reach: a 400 m step with 10 m ×4 / 200 m ×4 within 4 a side, 200 m at distance 5 and 10 m at distance 6 on
    # each side — 5 a side reads 10 ×4 and 200 ×6 (median 200: no jump); 4 or 6 a side would read a median of 105 (a jump)
    left, right = [10.0, 200.0, 10.0, 10.0, 10.0, 10.0], [200.0, 200.0, 200.0, 200.0, 200.0, 10.0]
    steps = [100.0] * 14 + left + [400.0] + right + [100.0] * 14
    assert Fault(JUMP, 20) not in track_faults(_steps(steps))
    # fewer at the end: the last step against the five before it (the end step's own copies would hide it)
    assert track_faults(_steps([100.0] * 30 + [30.0])) == (Fault(HELD, 30),)


def test_a_reversal_is_read_across_a_held_stretch_and_a_held_step_is_no_move():
    east = 100.0 * np.arange(21)
    back = np.concatenate([east, [2000.0, 2000.0], 2000.0 - 100.0 * np.arange(1, 20)])
    assert track_faults(_east(back)) == (Fault(HELD, 20), Fault(HELD, 21), Fault(REVERSAL, 22))
    e = np.concatenate([east, [2000.0], np.full(19, 2000.0)])                  # held a row, then south
    n = np.concatenate([np.zeros(22), -100.0 * np.arange(1, 20)])
    assert track_faults(replace(_east(np.zeros(41)), e_m=e, n_m=n)) == (Fault(HELD, 20),)


def test_a_step_among_held_positions_is_a_jump_and_a_turn_just_over_120_degrees_a_reversal():
    assert track_faults(_east([0.0] * 12 + [100.0] * 12)) == (Fault(JUMP, 11),)
    straight = 100.0 * np.arange(21)
    for turn_deg, expected in ((REVERSAL_DEG - 1e-6, ()), (REVERSAL_DEG + 1e-6, (Fault(REVERSAL, 20),))):
        bearing = np.radians(90.0 + turn_deg)
        e = np.concatenate([straight, straight[-1] + 100.0 * np.sin(bearing) * np.arange(1, 20)])
        n = np.concatenate([np.zeros(21), 100.0 * np.cos(bearing) * np.arange(1, 20)])
        assert track_faults(replace(_east(np.zeros(40)), e_m=e, n_m=n)) == expected, turn_deg


def test_a_track_too_short_or_not_finite_is_refused():
    for e in ([0.0, 100.0], [0.0] * 20 + [np.nan] + [0.0] * 20):
        with pytest.raises(ValueError, match="all positions finite, is needed"):
            track_faults(_east(e))


def test_the_flights_of_an_artefact_split_by_their_place_in_its_signals(tmp_path):
    """`faulty_flights` reads a split's stored signals and keys each marked flight by its place in them (the key of a
    closed-loop sentence)."""
    from ts_transformer.instructions.artefact import write_signals

    clean = instruction_flight(*fly_legs([(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0)], 270.0, 1110.0, -400.0, 0.0),
                               dataset_id="KXXX:clean")
    jumped = replace(clean, dataset_id="KXXX:jumped", e_m=clean.e_m + np.where(np.arange(len(clean.e_m)) == 30, 2000.0, 0.0))
    (tmp_path / "artefact").mkdir()
    write_signals(tmp_path / "artefact", {"train": [clean, jumped]},
                  {"counts": {"train": {"built_usable": 2}}, "test_days": {"flights_not_opened": 0},
                   "sources": [{"airport": "KXXX", "arrival_manifest_sha256": "0" * 64}]}, fixture_days())
    marked = faulty_flights(tmp_path / "artefact", "train")
    assert list(marked) == [1] and Fault(JUMP, 29) in marked[1] and Fault(JUMP, 30) in marked[1]
