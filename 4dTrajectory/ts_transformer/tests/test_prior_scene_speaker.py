"""The prior speaking in a scene (`prior/scene_speaker`, multi-aircraft design §6.6 step 2): one aircraft per scene speaks,
the others are replayed; alone it says what the `Speaker` says, it asks the caller's edge features only of the steps it
encodes, and the caller's masks take words away as the others do."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.instructions.words import SPEED, Words
from ts_transformer.prior.data import VARIANTS
from ts_transformer.prior.generate import Speaker, allowed_classes
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.model import with_traffic
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.prior.scene_data import Node
from ts_transformer.prior.scene_speaker import SceneSpeaker
from ts_transformer.tests.test_prior_speaker import _flight, _model

STEPS = 6


def _other(first_step, rows=40, seed=3, width=None, slots=1):
    """A replayed aircraft with random inputs, its first row at ``first_step`` from the speaking aircraft's row 0."""
    generator = np.random.default_rng(seed)
    relative = len(VARIANTS["no-context"].relative_features)
    return Node(f"other{seed}", False, first_step, generator.normal(size=(rows, width)).astype(np.float32),
                generator.normal(size=(rows, slots, relative)).astype(np.float32), np.zeros(0, dtype=np.float32),
                np.zeros((rows, 6), dtype=np.int64), np.zeros((rows, 6), dtype=np.float32),
                np.zeros((rows, 6), dtype=np.int64), np.zeros((rows, 6), dtype=bool))


def _edges(speaker_ref, calls):
    """Edge features with only "self" set, the calls recorded."""
    def edges(first, last):
        calls.append((first, last))
        speaker = speaker_ref[0]
        out = np.zeros((1, last - first, speaker.aircraft, speaker.aircraft, len(EDGE_FEATURES)), dtype=np.float32)
        out[..., 0] = np.eye(speaker.aircraft)
        return out
    return edges


def _drive(speaker, signals, steps=STEPS):
    """Speak ``steps`` steps, appending the flight's own later rows as where it is."""
    said = []
    for step in range(steps):
        if step:
            row = N_LOOK + step
            speaker.append(signals.e_m[row: row + 1], signals.n_m[row: row + 1], signals.altitude_m[row: row + 1],
                           frozen=np.zeros(1, dtype=bool))
        said.append(speaker.speak(active=np.ones(1, dtype=bool), runway_locked=np.zeros(1, dtype=bool)))
    return np.concatenate(said)


def _scene_speaker(model, one, geometry, signals, others, calls, masks=None, mask_columns=()):
    reference = [None]
    speaker = SceneSpeaker(model, [signals], [geometry], None, Words(one), others=[others],
                           edges=_edges(reference, calls), masks=masks or (lambda column, chosen: None),
                           mask_columns=mask_columns, max_rows=N_LOOK + STEPS + 1,
                           generator=torch.Generator().manual_seed(4), procedure_masks=ProcedureMasks.none())
    reference[0] = speaker
    return speaker


def test_alone_a_scene_speaker_with_a_zero_traffic_attention_says_what_the_speaker_says():
    one, geometry, signals, _ = _flight()
    single = _model(Words(one))
    single_said = _drive(Speaker(single, [signals], [geometry], None, Words(one), max_rows=N_LOOK + STEPS + 1,
                                 generator=torch.Generator().manual_seed(4), procedure_masks=ProcedureMasks.none()),
                         signals)
    torch.manual_seed(1)
    calls = []
    scene_said = _drive(_scene_speaker(with_traffic(single, EDGE_FEATURES).eval(), one, geometry, signals, [], calls),
                        signals)
    assert np.array_equal(single_said, scene_said)
    # the edge features are asked of the steps as they are encoded, in order, never of a later one
    assert calls == [(0, N_LOOK + 1)] + [(N_LOOK + k, N_LOOK + k + 1) for k in range(1, STEPS)]


def test_the_others_are_placed_at_their_own_rows_from_a_pre_roll_and_read_through_the_traffic_attention():
    one, geometry, signals, _ = _flight()
    single = _model(Words(one))
    torch.manual_seed(1)
    traffic = with_traffic(single, EDGE_FEATURES).eval()
    width = single.state.in_features - 6
    early, late = _other(-5, width=width, seed=3), _other(3, rows=4, width=width, seed=5)
    seen = []
    extend = traffic.extend

    def spy(*args):
        seen.append({"present": args[6].clone(), "rows": args[7].clone()})
        return extend(*args)

    traffic.extend = spy
    calls = []
    speaker = _scene_speaker(traffic, one, geometry, signals, [early, late], calls)
    assert speaker.pre == 5 and speaker.aircraft == 3
    _drive(speaker, signals)
    first = seen[0]
    # the pre-roll: 5 steps of the early one alone, then the speaking aircraft from its row 0
    assert first["present"][0, 1, :].all() and first["rows"][0, 1].tolist() == list(range(5 + N_LOOK + 1))
    assert not first["present"][0, 0, :5].any() and first["rows"][0, 0, 5:].tolist() == list(range(N_LOOK + 1))
    # the late one enters at the speaking aircraft's row 3 and leaves after 4 rows
    assert first["present"][0, 2].tolist() == [False] * 8 + [True] * 4 + [False] * (5 + N_LOOK + 1 - 12)
    assert calls[0] == (0, 5 + N_LOOK + 1)
    # at zero the others change the speaking aircraft's encoding only to rounding; with weights they move it
    traffic.extend = extend
    alone = _newest_h(_scene_speaker(traffic, one, geometry, signals, [], []), signals)
    among = _newest_h(_scene_speaker(traffic, one, geometry, signals, [early, late], []), signals)
    for a, b in zip(alone, among):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-5)
    for layer in traffic.layers:
        layer.traffic.out.weight.data.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(9))
    moved = _newest_h(_scene_speaker(traffic, one, geometry, signals, [early, late], []), signals)
    assert max(float((a - b).abs().max()) for a, b in zip(alone, moved)) > 1e-2


def _newest_h(speaker, signals):
    """The speaking aircraft's newest encoding at every step of `_drive`."""
    out, original = [], speaker._newest

    def spy():
        h, tokens, valid = original()
        out.append(h.clone())
        return h, tokens, valid

    speaker._newest = spy
    _drive(speaker, signals)
    return out


def test_the_caller_s_masks_take_words_away_and_are_recorded():
    one, geometry, signals, _ = _flight()
    torch.manual_seed(1)
    traffic = with_traffic(_model(Words(one)), EDGE_FEATURES).eval()
    keep = 2

    def masks(column, chosen):
        assert column == SPEED
        out = np.zeros((len(chosen), traffic.config.classes[SPEED]), dtype=bool)
        out[:, keep] = True
        return out

    speaker = _scene_speaker(traffic, one, geometry, signals, [], [], masks=masks, mask_columns=(SPEED,))
    said = _drive(speaker, signals)
    assert (said[:, SPEED] == keep).all()
    assert len(speaker.forbidden[SPEED]) == STEPS and all(0.0 <= float(m[0]) <= 1.0 for m in speaker.forbidden[SPEED])
    assert all(allowed_classes(packed, traffic.config.classes[SPEED])[0].tolist()
               == [k == keep for k in range(traffic.config.classes[SPEED])] for packed in speaker.allowed[SPEED])


def test_a_scene_is_spoken_to_through_a_traffic_prior_only():
    one, geometry, signals, _ = _flight()
    with pytest.raises(ValueError, match="traffic attention"):
        _scene_speaker(_model(Words(one)), one, geometry, signals, [], [])
    with pytest.raises(ValueError, match="scene prior"):
        Speaker(with_traffic(_model(Words(one)), EDGE_FEATURES), [signals], [geometry], None, Words(one), max_rows=20,
                generator=torch.Generator(), procedure_masks=ProcedureMasks.none())
