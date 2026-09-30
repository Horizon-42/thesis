"""The prior speaking to every commanded aircraft of a window (`prior/window_speaker`, multi-aircraft design §6.6 step
7.2): one commanded aircraft a scene says what `SceneSpeaker` says, word for word, and each one's words stay its own
when another of the batch stops; a later aircraft is placed and speaks from its own rows; a later round's masks read
the words an earlier round just said."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.instructions.words import SPEED, Words
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.model import with_traffic
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.prior.window_speaker import WindowSpeaker
from ts_transformer.prior.scene_speaker import SceneSpeaker
from ts_transformer.tests.test_prior_scene_speaker import _other
from ts_transformer.tests.test_prior_speaker import _flight, _model

STEPS = 6


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


def test_one_commanded_aircraft_a_scene_says_what_the_scene_speaker_says_and_its_words_outlive_another_leaving():
    one, geometry, signals, _ = _flight()
    traffic = _traffic(one)
    width = traffic.state.in_features - 6
    others = [[_other(-5, width=width, seed=3)], [_other(2, rows=6, width=width, seed=5)]]
    leaves = 3                                            # the second scene's aircraft stops after its third step
    # the scene speaker: both flights to the end, the second frozen and silent from its step `leaves`
    reference = [None]
    scene = SceneSpeaker(traffic, [signals, signals], [geometry] * 2, None, Words(one), others=others,
                         edges=_self_edges(reference), masks=lambda column, chosen: None, mask_columns=(),
                         history=100, max_rows=N_LOOK + STEPS + 1, generator=torch.Generator().manual_seed(4),
                         procedure_masks=ProcedureMasks.none())
    reference[0] = scene
    # the same places on the batch's steps: the speaking aircraft's row 0 at the pre-roll's end, the others from there
    pre = scene.pre
    window = _window_speaker(traffic, one, geometry, [signals, signals], [0, 1], [pre, pre],
                             [[dataclasses.replace(node, first_step=pre + node.first_step) for node in nodes]
                              for nodes in others])
    assert (pre, window.aircraft) == (5, scene.aircraft) == (5, 2)
    scene_said, window_said = [], []
    for step in range(STEPS):
        active = np.array([True, step < leaves])
        if step:
            row = N_LOOK + step
            scene.append(signals.e_m[[row, row]], signals.n_m[[row, row]], signals.altitude_m[[row, row]],
                         frozen=~active)
            flying = np.flatnonzero(active)
            window.advance(flying, np.full(len(flying), signals.e_m[row]), np.full(len(flying), signals.n_m[row]),
                           np.full(len(flying), signals.altitude_m[row]))
        scene_said.append(scene.speak(active=active, runway_locked=np.zeros(2, dtype=bool)))
        window_said.append(window.speak(np.where(active, 0, -1), np.zeros(2, dtype=bool)))
    scene_said, window_said = np.stack(scene_said), np.stack(window_said)
    assert np.array_equal(window_said[:, 0], scene_said[:, 0])            # the one that flies on: every word
    assert np.array_equal(window_said[:leaves, 1], scene_said[:leaves, 1])  # the one that left: to its end
    assert (window_said[leaves:, 1] == 0).all()
    # what the masks removed, by each aircraft's own step
    for column, masses in scene.forbidden.items():
        assert np.array_equal(window.forbidden[column][0, :STEPS], np.stack(masses)[:, 0])


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
