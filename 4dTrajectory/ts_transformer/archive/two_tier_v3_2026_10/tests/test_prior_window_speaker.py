"""The prior speaking to every commanded aircraft of a window (`prior/window_speaker`, multi-aircraft design §6.6 step
7.2): each commanded aircraft's words stay its own, word for word, when another of the batch stops; a later aircraft is
placed and speaks from its own rows; a later round's masks read the words an earlier round just said. (With one commanded
aircraft and no other it says what single-aircraft free generation says: the window loop's tests,
`test_traffic_window.py`.)"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.instructions.words import SPEED, Words
from ts_transformer.prior.data import VARIANTS
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.model import with_traffic
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.prior.scene_data import Node
from ts_transformer.prior.window_speaker import WindowSpeaker
from ts_transformer.tests.test_prior_speaker import _flight, _model

STEPS = 6


def _other(first_step, rows=40, seed=3, width=None, slots=1):
    """A replayed aircraft with random inputs, its first row at batch step ``first_step``."""
    generator = np.random.default_rng(seed)
    relative = len(VARIANTS["no-context"].relative_features)
    return Node(f"other{seed}", False, first_step, generator.normal(size=(rows, width)).astype(np.float32),
                generator.normal(size=(rows, slots, relative)).astype(np.float32), np.zeros(0, dtype=np.float32),
                np.zeros((rows, 6), dtype=np.int64), np.zeros((rows, 6), dtype=np.float32),
                np.zeros((rows, 6), dtype=np.int64), np.zeros((rows, 6), dtype=bool))


def _self_edges(speaker_ref, calls=None):
    def edges(first, last):
        if calls is not None:
            calls.append((first, last))
        speaker = speaker_ref[0]
        out = np.zeros((len(speaker.others), last - first, speaker.aircraft, speaker.aircraft, len(EDGE_FEATURES)),
                       dtype=np.float32)
        out[..., 0] = np.eye(speaker.aircraft)
        return out
    return edges


def _window_speaker(model, one, geometry, flights, scenes, starts, others, *, masks=None, mask_columns=(),
                    steps=STEPS, seed=4, calls=None):
    reference = [None]
    speaker = WindowSpeaker(model, flights, [geometry] * len(flights), None, Words(one), scenes=scenes,
                            starts=starts, others=others, edges=_self_edges(reference, calls),
                            masks=masks or (lambda column, chosen, now: np.ones((len(chosen), 1), dtype=bool)),
                            mask_columns=mask_columns, max_rows=N_LOOK + max(starts) - min(starts) + steps + 1,
                            generator=torch.Generator().manual_seed(seed), procedure_masks=ProcedureMasks.none())
    reference[0] = speaker
    return speaker


def _traffic(one):
    torch.manual_seed(1)
    return with_traffic(_model(Words(one)), EDGE_FEATURES).eval()


def test_each_commanded_aircraft_s_words_stay_its_own_when_another_of_the_batch_leaves():
    """Two scenes of one commanded aircraft each (the replayed ones in the air before the first and entering later
    beside the second): the second leaves after its third step in one run and flies on in the other; the first says the
    same words in both, the second the same up to its leaving and nothing after, and the masks of the first remove the
    same."""
    one, geometry, signals, _ = _flight()
    traffic = _traffic(one)
    width = traffic.state.in_features - 6
    pre, leaves = 5, 3
    others = [[_other(pre - 5, width=width, seed=3)], [_other(pre + 2, rows=6, width=width, seed=5)]]
    runs = []
    for leaving in (False, True):
        window = _window_speaker(traffic, one, geometry, [signals, signals], [0, 1], [pre, pre], others)
        assert window.aircraft == 2
        said = []
        for step in range(STEPS):
            active = np.array([True, not leaving or step < leaves])
            if step:
                row = N_LOOK + step
                flying = np.flatnonzero(active)
                window.advance(flying, np.full(len(flying), signals.e_m[row]), np.full(len(flying), signals.n_m[row]),
                               np.full(len(flying), signals.altitude_m[row]))
            said.append(window.speak(np.where(active, 0, -1), np.zeros(2, dtype=bool)))
        runs.append((np.stack(said), window))
    (stays, staying), (left, leaving) = runs
    assert np.array_equal(left[:, 0], stays[:, 0])                  # the one that flies on: every word
    assert np.array_equal(left[:leaves, 1], stays[:leaves, 1])      # the one that left: to its end
    assert (left[leaves:, 1] == 0).all() and (stays[leaves:, 1] != 0).any()
    for column, masses in staying.forbidden.items():                 # what the masks removed from the first
        assert np.array_equal(leaving.forbidden[column][0, :STEPS], masses[0, :STEPS])


def test_a_later_aircraft_enters_at_its_offset_and_speaks_from_its_own_first_predicted_step():
    one, geometry, signals, _ = _flight()
    traffic = _traffic(one)
    width = traffic.state.in_features - 6
    seen = []
    extend = traffic.extend
    traffic.extend = lambda *args: (seen.append({"present": args[6].clone(), "rows": args[7].clone()}),
                                    extend(*args))[1]
    later = 3
    calls = []
    window = _window_speaker(traffic, one, geometry, [signals, signals], [0, 0], [2, 2 + later],
                             [[_other(0, rows=30, width=width)]], steps=STEPS, calls=calls)
    # the two commanded aircraft, then the replayed one
    assert (window.aircraft, window.slot.tolist(), window.step) == (3, [0, 1], 2 + N_LOOK)
    said = []
    for step in range(STEPS + later):
        if step:
            row = window.row + 1                      # each one's row at the next step
            flying = np.flatnonzero(row > N_LOOK)
            window.advance(flying, signals.e_m[row[flying]], signals.n_m[row[flying]],
                           signals.altitude_m[row[flying]])
        row = window.row
        speaking = (row >= N_LOOK) & window.there
        # front first: the later one behind the reference
        said.append(window.speak(np.where(speaking, np.cumsum(speaking) - 1, -1), np.zeros(2, dtype=bool)))
    said = np.stack(said)
    assert (said[:later, 1] == 0).all() and (said[later, 1] > 0).all()   # its first predicted step says every column
    assert (said[0, 0] > 0).all()
    first = seen[0]
    present, rows = first["present"][0], first["rows"][0]
    # the first block: the replayed one from step 0, the first commanded one from step 2, the later one from step 5
    assert present[2, :].all() and not present[0, :2].any() and present[0, 2:].all()
    assert not present[1, :2 + later].any() and rows[1, 2 + later:].tolist() == list(range(N_LOOK + 1 - later))
    assert calls[0] == (0, 2 + N_LOOK + 1)
    # its words are recorded at its own steps
    assert window.allowed[next(iter(window.allowed))][1, :STEPS].any()


def test_a_forced_column_is_said_in_place_of_its_draw_where_allowed_and_moves_nobody_elses():
    """Multi-aircraft design §6.6 step 8 item 10: a probe's word is said in place of the aircraft's draw, the draw still
    made — the others' words do not move; a word the masks forbid there is not forced."""
    from ts_transformer.instructions.words import APPROACH, APPROACH_CLEARED, APPROACH_GO_AROUND

    one, geometry, signals, _ = _flight()
    traffic = _traffic(one)

    def first_step(forced):
        window = _window_speaker(traffic, one, geometry, [signals, signals], [0, 1], [2, 2], [[], []])
        return window.speak(np.array([0, 0]), np.zeros(2, dtype=bool), None, forced)

    plain = first_step(None)
    cleared = np.full((2, 6), -1)
    cleared[0, APPROACH] = APPROACH_CLEARED + 1
    got = first_step(cleared)
    assert got[0, APPROACH] == APPROACH_CLEARED + 1 and (got[1] == plain[1]).all()
    # a go-around at the first step: forbidden there (nothing in force), so the draw stands
    around = np.full((2, 6), -1)
    around[0, APPROACH] = APPROACH_GO_AROUND + 1
    assert (first_step(around) == plain).all()


def test_a_later_round_s_masks_read_the_words_an_earlier_round_just_said():
    one, geometry, signals, _ = _flight()
    traffic = _traffic(one)
    classes = traffic.config.classes[SPEED]
    asked, leader = [], []
    window_ref = [None]

    def masks(column, chosen, now):
        assert column == SPEED
        out = np.ones((len(chosen), classes), dtype=bool)
        if now[1]:
            # the second aircraft may only say the class the first has in force — its word of this step included
            leader.append(int(window_ref[0].value[0, SPEED]))
            out[1] = False
            out[1, leader[-1]] = True
        asked.append(now.copy())
        return out

    window = _window_speaker(traffic, one, geometry, [signals, signals], [0, 0], [0, 0], [[]], masks=masks,
                             mask_columns=(SPEED,))
    window_ref[0] = window
    said = []
    for step in range(STEPS):
        if step:
            row = N_LOOK + step
            window.advance(np.arange(2), signals.e_m[[row, row]], signals.n_m[[row, row]],
                           signals.altitude_m[[row, row]])
        said.append(window.speak(np.array([0, 1]), np.zeros(2, dtype=bool)))
    said = np.stack(said)
    first = said[:, 0, SPEED]
    in_force = [int(first[: k + 1][first[: k + 1] > 0][-1]) for k in range(STEPS)]
    assert leader == in_force and said[:, 1, SPEED].tolist() == in_force
    assert [a.tolist() for a in asked] == [[True, False], [False, True]] * STEPS


def test_the_rounds_are_one_aircraft_a_scene_and_only_there_from_the_first_predicted_step():
    one, geometry, signals, _ = _flight()
    traffic = _traffic(one)
    together = _window_speaker(traffic, one, geometry, [signals, signals], [0, 0], [0, 0], [[]])
    with pytest.raises(ValueError, match="two aircraft of one scene"):
        together.speak(np.array([0, 0]), np.zeros(2, dtype=bool))
    later = _window_speaker(traffic, one, geometry, [signals, signals], [0, 0], [0, 2], [[]])
    with pytest.raises(ValueError, match="from its first predicted step"):
        later.speak(np.array([0, 1]), np.zeros(2, dtype=bool))
    with pytest.raises(ValueError, match="every scene commands an aircraft"):
        _window_speaker(traffic, one, geometry, [signals, signals], [0, 2], [0, 0], [[], []])
    with pytest.raises(ValueError, match="traffic attention"):
        _window_speaker(_model(Words(one)), one, geometry, [signals], [0], [0], [[]])
