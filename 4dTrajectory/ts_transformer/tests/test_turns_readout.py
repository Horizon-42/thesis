"""Post-training D214: how often a commanded aircraft's heading words and flown bank turn back (`post_window_loop.Turns`,
`heading_reversals`, `bank_reversals`, `turns_readout`), on built words and states and on the synthetic window of three
commanded aircraft of stage D's loop tests."""

from __future__ import annotations

import numpy as np
import pytest

from ts_transformer.experiments.post_window_loop import (
    BANK_FLOOR_DEG, Turns, bank_reversals, heading_reversals, turns_readout,
)
from ts_transformer.instructions.artefact import STATE_COLUMNS
from ts_transformer.instructions.words import COLUMNS, HEADING, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED
from ts_transformer.tests.test_post_window_loop import _with_module, setup  # noqa: F401
from ts_transformer.tests.test_post_window_multi import JOINS, _multi, _numbers

N_HEADING = 72


def _words(*rows):
    """Said rows: each ``(runway, heading)``, the other columns unchanged."""
    out = np.full((len(rows), len(COLUMNS)), UNCHANGED, dtype=np.int64)
    for k, (runway, heading) in enumerate(rows):
        out[k, RUNWAY], out[k, HEADING] = runway, heading
    return out


@pytest.mark.parametrize("rows, expected", [
    (((0, 0), (UNCHANGED, 1), (UNCHANGED, 0), (UNCHANGED, 1)), 2),                    # left, right, left: two back
    (((0, 0), (UNCHANGED, 1), (UNCHANGED, 2), (UNCHANGED, 4)), 0),                    # one turn in steps
    (((0, 71), (UNCHANGED, 0), (UNCHANGED, 71)), 1),                                  # across the course: short way
    (((0, 0), (UNCHANGED, UNCHANGED), (UNCHANGED, 2), (UNCHANGED, UNCHANGED), (UNCHANGED, 1)), 1),
    (((0, 0), (UNCHANGED, 1), (1, 0), (UNCHANGED, 1)), 0),                             # a runway word starts again
    (((0, 0), (UNCHANGED, 1), (RUNWAY_GO_AROUND, UNCHANGED), (UNCHANGED, 0)), 0),     # so does a go-around
])
def test_a_heading_word_turning_back_is_counted_under_its_runway_word(rows, expected):
    assert heading_reversals(_words(*rows), N_HEADING) == expected


def _states(track_deg, speed_mps=70.0):
    states = np.zeros((len(track_deg), len(STATE_COLUMNS)))
    states[:, STATE_COLUMNS.index("track_deg")] = track_deg
    states[:, STATE_COLUMNS.index("ground_speed_mps")] = speed_mps
    return states


def test_the_bank_changes_side_only_above_its_floor():
    """A turn one way and then the other is one reversal (a 3 °/s turn at 70 m/s banks 20°); a straight track and a
    wiggle under `BANK_FLOOR_DEG` none; a turn across north is one turn."""
    s_turn = np.concatenate((np.arange(0, 30, 6.0), 30 - np.arange(6, 30, 6.0)))
    assert bank_reversals(_states(s_turn), 2.0) == 1
    assert bank_reversals(_states(np.full(10, 90.0)), 2.0) == 0
    wiggle = 90.0 + 0.4 * (np.arange(12) % 2)                          # 0.2 °/s: under 1.5° of bank
    assert bank_reversals(_states(wiggle), 2.0) == 0 and BANK_FLOOR_DEG > 1.5
    assert bank_reversals(_states(np.array([350.0, 356.0, 2.0, 8.0, 14.0])), 2.0) == 0
    assert bank_reversals(_states(np.array([350.0, 356.0, 2.0, 356.0, 350.0])), 2.0) == 1


def test_the_readout_sums_by_airport_and_over_all_and_reads_per_minute():
    one, two = Turns(3, 120.0, 2, 60.0), Turns(1, 60.0, 0, 0.0)
    out = turns_readout([("KAAA", _End(one)), ("KBBB", _End(two)), ("KAAA", _End(one))])
    assert out["KAAA"] == {"aircraft": 2, "heading_reversals": 6, "spoken_s": 240.0, "bank_reversals": 4,
                           "flown_s": 120.0, "heading_reversals_per_min": 1.5, "bank_reversals_per_min": 2.0}
    assert out["KBBB"]["bank_reversals_per_min"] is None and out["KBBB"]["heading_reversals_per_min"] == 1.0
    assert out["all"]["aircraft"] == 3 and out["all"]["heading_reversals_per_min"] == pytest.approx(60 * 7 / 300)


class _End:
    def __init__(self, turns):
        self.turns = turns


def test_each_commanded_aircraft_of_a_window_carries_the_turns_of_its_words_and_flown_track(setup):
    """The loop's end of each aircraft: the heading reversals of its words said, over the rows said times Δ; the bank
    reversals of its states from its first predicted step on, over the time flown."""
    s = setup
    _, loop_of = _multi(s)
    loop = loop_of(_with_module(s["base"]))
    ends = loop.run(_numbers(len(JOINS)))
    speaking = loop.speaking
    first, step_s = speaking.start * speaking.every, speaking.words.spec.step_s
    for end in ends:
        flown = end.states[first:]
        assert end.turns == Turns(heading_reversals(end.words, speaking.words.n_heading),
                                  len(end.words) * speaking.every * step_s, bank_reversals(flown, step_s),
                                  (len(flown) - 1) * step_s)
        assert end.turns.spoken_s > 0 and end.turns.flown_s > 0
