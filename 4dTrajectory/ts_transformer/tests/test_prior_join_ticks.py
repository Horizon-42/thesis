"""Aircraft that join a loop at their own ticks (multi-aircraft control D150, §6.2 items 2–4, §6.3 items 2–5; prior §7
items 2, 3, 7): `prior.loop.LoopRows`, `prior.speaker.Speaker.say`, `experiments.prior_speaking_loop.SpeakingLoop` — on
A26's synthetic flight (`test_start._artefact`), several copies of it in one loop, each under its own key."""

from __future__ import annotations

import hashlib
from dataclasses import fields
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from flight_scenarios.fas_geometry import fas_course_geometry
from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.start import NO_MOVE, Loop, start_moved
from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop
from ts_transformer.instructions.artefact import load_day_split, signals_flights
from ts_transformer.instructions.grammar import column_words
from ts_transformer.instructions.words import COLUMNS, RUNWAY, UNCHANGED
from ts_transformer.prior.batch import collate
from ts_transformer.prior.landings import Landing, LandingIndex, utc_s
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.procedure import Final
from ts_transformer.prior.speaker import ABSENT, MOST_GO_AROUNDS, OBSERVED, SAID
from ts_transformer.prior.train import masked_log_probability
from ts_transformer.tests import test_start

CPU = torch.device("cpu")
INTERVAL_S = 4.0
#: The keys of the copies of the flight; every loop's landings hold all of them.
KEYS = tuple(f"F{i}" for i in range(5))
#: The join ticks of the loops below: one joins at the first tick, one inside the first's observed rows, one after the
#: first's first predicted step, one later still.
JOIN_TICKS = np.array([0, 2, 5, 9])


def _setup(tmp_path, monkeypatch):
    """``speaking(join_ticks, keys)``: a `SpeakingLoop` of copies of A26's flight, copy b under the key ``keys[b]`` and
    joining at ``join_ticks[b]`` (None: a loop without join ticks); the model."""
    directory, words, batch, stored, _ = test_start._artefact(tmp_path, monkeypatch, INTERVAL_S)
    one, _, observed = start_moved(directory, "train", INTERVAL_S, {0: stored}, tmp_path / "executor", {0: NO_MOVE},
                                   most_go_arounds=MOST_GO_AROUNDS, device=CPU)
    geometry = batch.geometries[0]
    record = signals_flights(directory, "train")[0]
    idents = tuple(c.ident for c in geometry.candidates)
    landing = utc_s(record["landing_time_utc"])
    landings = LandingIndex(idents, tuple(Landing(landing + k, idents[0], key) for k, key in enumerate(KEYS)), 0,
                            load_day_split(directory))
    finals = {geometry.code: tuple(Final(geometry, k, 9_000.0, fas_course_geometry(c.length_m))
                                   for k, c in enumerate(geometry.candidates))}
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64)).eval()
    inputs, limit = one.executor.inputs, float(one.executor.time_limit_s[0])

    def speaking(join_ticks, keys=KEYS[:4]):
        count = len(keys)
        many = FlightInputs(**{f.name: getattr(inputs, f.name).expand(count, *getattr(inputs, f.name).shape[1:]).clone()
                               for f in fields(FlightInputs)})
        joined = {} if join_ticks is None else {"join_ticks": join_ticks}
        loop = Loop(many, [geometry] * count, [test_start.A320_SPEEDS] * count, ["A320"] * count, [limit] * count,
                    test_start._params(), words, interval_s=INTERVAL_S, most_go_arounds=MOST_GO_AROUNDS, device=CPU,
                    **joined)
        flights = {b: {**record, "dataset_id": f"{geometry.code}:{key}"} for b, key in enumerate(keys)}
        return SpeakingLoop(model, loop, list(range(count)), {b: stored for b in range(count)},
                            {b: observed[0] for b in range(count)}, flights, {geometry.code: geometry},
                            [landings] * count, finals, words, interval_s=INTERVAL_S, variant="full", device=CPU)

    return speaking, model, words


def _sources(keys):
    """Each copy's own random numbers, by its key (the same whatever loop it is in)."""
    return [np.random.default_rng([7, KEYS.index(key)]) for key in keys]


def _fly(loop: SpeakingLoop, sources, *, steps: int | None = None, caller=None) -> SpeakingLoop:
    """``loop`` observed (a loop without join ticks) and stepped until every flight has ended (or ``steps`` ticks), each
    flight drawing its numbers only at the ticks it is said (its own rows: the same draws as alone)."""
    while loop.observing:
        loop.observe()
    k = 0
    while loop.alive.any() and (steps is None or k < steps):
        roles = loop.roles()
        numbers = np.stack([s.random(len(COLUMNS)) if r == SAID else np.zeros(len(COLUMNS))
                            for s, r in zip(sources, roles)])
        loop.step(numbers, None if caller is None else caller(loop))
        k += 1
    return loop


def _fly_plain(loop: SpeakingLoop, sources, *, steps: int | None = None) -> SpeakingLoop:
    """`_fly` as a loop without join ticks is flown (every flight draws its numbers at every row): only the names the
    loop had before join ticks, so that the code before them runs it too (`BEFORE_JOIN_TICKS`)."""
    while loop.observing:
        loop.observe()
    k = 0
    while loop.alive.any() and (steps is None or k < steps):
        loop.step(np.stack([s.random(len(COLUMNS)) for s in sources]))
        k += 1
    return loop


def _digest(loop: SpeakingLoop) -> str:
    """The sha256 of what a loop said and flew (a digest of output data, outline D139 item 5): each flight's words,
    states and sentence rows, the speaker's drawn probabilities and records."""
    out = hashlib.sha256()
    for b in range(len(loop.order)):
        out.update(loop.said(b).tobytes())
        out.update(loop.states(b).tobytes())
    for rows in loop.sentences("train"):
        for name in ("time_s", "own", "candidates", "runway_in_force", "go_around", "heading_in_force",
                     "words_in_force", "since", "targets"):
            out.update(np.ascontiguousarray(getattr(rows, name)).tobytes())
    out.update(np.stack(loop.speaker.drawn_probability).tobytes())
    permitted = loop.speaker.permitted()             # the speaker's whole record, as it was pinned
    for mask in permitted.masks:
        out.update(mask.tobytes())
    out.update(permitted.time_s.tobytes())
    out.update(permitted.own.tobytes())
    return out.hexdigest()


def _zero_join_scenario(speaking, join_ticks) -> str:
    """Three copies flown without join ticks (``join_ticks`` None) or with every one 0: copy 1 ended by the caller after
    three rows, a copy of copies 2 and 0 taken after five rows and flown on with the same numbers; the digest of both."""
    keys = KEYS[:3]
    loop = speaking(join_ticks, keys)
    sources = _sources(keys)
    _fly_plain(loop, sources, steps=3)
    loop.end(np.array([False, True, False]))
    _fly_plain(loop, sources, steps=2)
    state = [s.bit_generator.state for s in sources]
    copy = loop.copy([2, 0])
    _fly_plain(loop, sources)
    twin = [np.random.default_rng() for _ in (2, 0)]
    for generator, k in zip(twin, (2, 0)):
        generator.bit_generator.state = state[k]
    _fly_plain(copy, twin)
    return _digest(loop) + _digest(copy)


#: `_zero_join_scenario`'s digests, written by the code before join ticks (dev-two-tier-v4 c7a0b6b0, the CPU, one
#: thread, the loop without join ticks): the reference of D150's "every join tick 0 is today's loop, bit for bit".
#: Written again 2026-10-13: the first-principles executor (223f1e30, ts-executor-spec-v11) flies the words otherwise.
#: The loop with no join ticks and the loop with every join tick 0 give this same digest from 223f1e30 to 288b779b.
#: Written again on 2026-10-14 (two-tier2-glidepath, prior D204): the offset right of the centreline goes into the
#: prior on the glidepath's 100 m scale, so the synthetic prior says other words; every part it says moved.
#: Written again on 2026-10-14 (prior D206): the prior reads no landings, so its candidate input is one feature
#: narrower and the synthetic prior says other words.
BEFORE_JOIN_TICKS = ("a4eec3e7f766d11df3c3702808b6687ff4ac59b29f35ba07e782d78e1d545eb0"
                     "30e387d8583e5e49a831ff7b8d916d52a904aaa6e8676e64a9ba738abba6b47d")


def test_every_join_tick_zero_says_and_flies_what_the_loop_said_before_join_ticks(tmp_path, monkeypatch):
    """D150: a loop of three flights (one ended by the caller, a copy taken and flown on) says and flies, with every join
    tick 0 and with none given, what the code before join ticks said and flew — words, states, sentence rows, drawn
    probabilities and records, bit for bit (their digest)."""
    speaking, _, _ = _setup(tmp_path, monkeypatch)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        plain, zeros = _zero_join_scenario(speaking, None), _zero_join_scenario(speaking, [0, 0, 0])
    finally:
        torch.set_num_threads(threads)
    assert plain == zeros == BEFORE_JOIN_TICKS


def test_a_flight_that_joins_at_its_tick_says_and_flies_what_it_says_alone(tmp_path, monkeypatch):
    """D150: flights that join one loop at their own ticks — before, inside and after another's observed rows — each say,
    with their own numbers, the words they say alone, with the same drawn probabilities and records (to the float
    tolerance of post-training §6.4), and fly the states they fly alone within STATE_BOUND_M, to the same outcome."""
    speaking, _, _ = _setup(tmp_path, monkeypatch)
    together = _fly(speaking(JOIN_TICKS), _sources(KEYS[:4]))
    drawn = np.stack(together.speaker.drawn_probability, axis=1)       # [B, rows of the batch]
    said_rows = np.stack(together.speaker.said_rows, axis=1)
    for b, key in enumerate(KEYS[:4]):
        alone = _fly(speaking(None, (key,)), _sources((key,)))
        assert np.array_equal(together.said(b), alone.said(0))
        assert together.states(b).shape == alone.states(0).shape
        assert np.allclose(together.states(b), alone.states(0), rtol=0.0, atol=STATE_BOUND_M)
        assert together.generated([b])[0].outcome == alone.generated()[0].outcome
        # its rows said to its end (the speaker says a done flight's rows on while the batch is said: not its sentence's)
        its_drawn = np.stack(alone.speaker.drawn_probability, axis=1)[0]
        rows = len(its_drawn)
        np.testing.assert_allclose(drawn[b, said_rows[b]][:rows], its_drawn, rtol=1e-5, atol=1e-7)
        mine, its = together.permitted([b])[0], alone.permitted([0])[0]       # each the flight's own record (S6)
        assert np.array_equal(mine.time_s, its.time_s)
        for a, c in zip(mine.masks, its.masks):
            assert np.array_equal(a, c)
        (sentence,), (single,) = together.sentences("train")[b: b + 1], alone.sentences("train")
        assert sentence.rows == single.rows and np.array_equal(sentence.time_s, single.time_s)
        np.testing.assert_allclose(sentence.own, single.own, rtol=0.0, atol=1e-6)


def test_an_absent_flight_is_never_encoded_heard_or_recorded(tmp_path, monkeypatch):
    """D150: before its join tick a flight's rows are written into the speaker's cache as not present (no later row reads
    them), it is not started in the executor, and no record or sentence row of it is kept; its roles go absent,
    observed, said."""
    speaking, _, _ = _setup(tmp_path, monkeypatch)
    loop = speaking(JOIN_TICKS)
    sources = _sources(KEYS[:4])
    seen = []
    for _ in range(12):
        seen.append(loop.roles())
        _fly(loop, sources, steps=1)
    roles = np.stack(seen, axis=1)                                      # [B, ticks]
    for b, tick in enumerate(JOIN_TICKS):
        start = loop.start
        assert (roles[b, :tick] == ABSENT).all() and (roles[b, tick: tick + start] == OBSERVED).all()
        assert (roles[b, tick + start:] == SAID).all()
        present = loop.speaker.past[0].present[b, : loop.t].numpy()
        assert not present[:tick].any() and present[tick:].all()
        assert len(loop.said(b)) == max(0, 12 - tick - start)
    assert loop.loop.steps == 12 - loop.start and np.array_equal(loop.loop.started(), JOIN_TICKS <= loop.loop.steps)
    said_rows = np.stack(loop.speaker.said_rows, axis=1)
    assert np.array_equal(said_rows.sum(axis=1), [len(loop.said(b)) for b in range(4)])


def test_a_copy_of_a_loop_with_join_ticks_says_what_its_original_says(tmp_path, monkeypatch):
    """D150: a copy taken while a flight is still absent and another observed keeps the join ticks and the landings; flown
    on with the same numbers, it says and flies what its original says and flies."""
    speaking, _, _ = _setup(tmp_path, monkeypatch)
    loop = speaking(JOIN_TICKS)
    sources = _sources(KEYS[:4])
    _fly(loop, sources, steps=6)
    assert loop.roles().tolist() == [SAID, SAID, OBSERVED, ABSENT]
    taken = [3, 2, 0, 0]
    state = [s.bit_generator.state for s in sources]
    copy = loop.copy(taken)
    assert np.array_equal(copy.join_ticks, JOIN_TICKS[taken]) and np.array_equal(copy.loop.join_ticks, JOIN_TICKS[taken])
    _fly(loop, sources)
    twin = [np.random.default_rng() for _ in taken]
    for generator, k in zip(twin, taken):
        generator.bit_generator.state = state[k]
    _fly(copy, twin)
    for j, b in enumerate(taken):
        assert np.array_equal(copy.said(j), loop.said(b))
        assert np.allclose(copy.states(j), loop.states(b), rtol=0.0, atol=STATE_BOUND_M)


def test_a_landing_added_goes_to_the_chosen_flights_landings_and_changes_no_input(tmp_path, monkeypatch):
    """D150, §6.3 item 5 (multi-aircraft control D147), D206: a landing added to chosen flights' landings while the loop
    runs goes to their indexes only, and no row of any flight changes (the prior reads no landings); a copy keeps it;
    the index keeps its checks — a second landing of a flight and a landing on a day outside the split are refused; one
    on a sealed test day is left out and counted with the sealed landings (requests item 6, the user, 2026-10-07)."""
    speaking, _, _ = _setup(tmp_path, monkeypatch)
    loop = speaking(JOIN_TICKS)
    rows_of = loop.rows_of
    at = loop.observed[:, 4]
    before = loop.observed[:, 3]
    tick = 20
    first = rows_of(tick, at, before, True, loop.speaker.heard)[0]
    kept = [loop.landings(b) for b in range(4)]
    idents = kept[0].runways
    when = max(landing.time_s for landing in kept[0].landings) + 1.0
    loop.add_landing([1, 3], Landing(when, idents[0], "NEW"))
    assert [loop.landings(b).digest() == kept[b].digest() for b in range(4)] == [True, False, True, False]
    assert all("NEW" in {x.flight_key for x in loop.landings(b).landings} for b in (1, 3))
    second = rows_of(tick, at, before, True, loop.speaker.heard)[0]
    for name, x, y in zip(first._fields, first, second):
        assert torch.equal(x, y), name
    copy = loop.copy([1, 0])
    assert [copy.landings(j).digest() for j in range(2)] == [loop.landings(1).digest(), loop.landings(0).digest()]
    with pytest.raises(ValueError, match="a flight lands twice"):
        loop.landings(1).with_landing(Landing(when, idents[0], "NEW"))
    days = kept[0].days
    test_day = datetime.strptime(sorted(days.days["test"])[0], "%Y-%m-%d").replace(hour=21, tzinfo=timezone.utc)
    sealed = kept[0].with_landing(Landing(test_day.timestamp(), idents[0], "SEALED"))
    assert sealed.landings == kept[0].landings and sealed.sealed == kept[0].sealed + 1
    early = test_day.replace(hour=5) + timedelta(days=1)                   # UTC − 9 h: still the test day's
    assert kept[0].with_landing(Landing(early.timestamp(), idents[0], "EARLY")).sealed == sealed.sealed
    with pytest.raises(ValueError, match="lands twice"):                    # the other checks first
        loop.landings(1).with_landing(Landing(test_day.timestamp(), idents[0], "NEW"))
    outside = datetime(1990, 1, 1, 21, tzinfo=timezone.utc)                # a day outside the split: refused
    with pytest.raises(KeyError, match="not in this day split"):
        kept[0].with_landing(Landing(outside.timestamp(), idents[0], "OUTSIDE"))


def test_the_loop_keeps_each_flights_landings_for_its_caller(tmp_path, monkeypatch):
    """D105, D147, D206: the loop keeps each flight's landings for its caller (the post-training reward reads them; the
    prior reads none). A landing added goes to the chosen flights still held — an ended flight is held until the loop
    is joined again — and never to a flight that left; `together` and `copy` (repeats permitted) give each flight its
    own landings."""
    from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop as Loops

    speaking, _, _ = _setup(tmp_path, monkeypatch)
    keys = KEYS[:3]
    sources = _sources(keys)
    loop = speaking(None, keys)
    runway = loop.landings(0).runways[0]
    late = max(x.time_s for x in loop.landings(0).landings)

    def added(of, flight):
        return {"A", "B", "C", "D"} & {x.flight_key for x in of.landings(flight).landings}

    loop.add_landing([0], Landing(late + 10.0, runway, "A"))
    loop.add_landing([1, 2], Landing(late + 20.0, runway, "B"))
    _fly(loop, sources, steps=2)
    loop.end(np.array([False, True, False]))
    loop.add_landing([1, 2], Landing(late + 30.0, runway, "C"))     # flight 1 ended, still held: it gets it
    assert [added(loop, b) for b in range(3)] == [{"A"}, {"B", "C"}, {"B", "C"}]
    joint = Loops.together([(loop, [0, 1, 1, 2])], CPU, None)
    assert joint.held.tolist() == [True, False, False, True]           # flight 1 left the speaker and the executor
    assert [joint.landings(k).digest() for k in range(4)] == [loop.landings(b).digest() for b in (0, 1, 1, 2)]
    joint.add_landing([0, 1, 2, 3], Landing(late + 40.0, runway, "D"))
    assert ["D" in added(joint, k) for k in range(4)] == [True, False, False, True]
    copy = joint.copy([3, 3, 1])
    assert [copy.landings(j).digest() for j in range(3)] == [joint.landings(k).digest() for k in (3, 3, 1)]


def test_a_row_of_several_roles_is_refused_out_of_order_or_unlike_its_rows(tmp_path, monkeypatch):
    """D150: `Speaker.say` refuses, before any change, roles of the wrong shape or going back (said, then observed) and
    a row whose presence is not its aircraft's role."""
    speaking, _, _ = _setup(tmp_path, monkeypatch)
    loop = speaking(JOIN_TICKS)
    _fly(loop, _sources(KEYS[:4]), steps=6)
    speaker = loop.speaker
    at, before, known = loop._at(loop.current, loop.before)
    tensors, position = loop.rows_of(loop.t, at, before, known, speaker.heard)
    numbers = np.zeros((4, len(COLUMNS)))
    count = len(speaker.permitted_rows)
    for roles, match in (([SAID, SAID, OBSERVED], "a role"), ([OBSERVED, SAID, OBSERVED, ABSENT], "absent, then observed"),
                         ([SAID, SAID, OBSERVED, OBSERVED], "present for each aircraft not absent")):
        with pytest.raises(ValueError, match=match):
            speaker.say(tensors, position, np.array(roles), numbers)
    assert len(speaker.permitted_rows) == count and loop.roles().tolist() == [SAID, SAID, OBSERVED, ABSENT]


def test_a_silent_flight_says_unchanged_with_probability_one_and_flies_its_words_in_force(tmp_path, monkeypatch):
    """Multi-aircraft control §6.2 item 4 (D144): a caller's mask that permits only "unchanged" in every column, from a
    row after a flight's first predicted step, is taken in every column (the grammar permits "unchanged" in every
    column there); the flight says "unchanged" with probability 1 — its rows' log-probability under the speaker's
    records 0 — and flies its words in force; the other flights are said as before."""
    speaking, model, words = _setup(tmp_path, monkeypatch)
    silent_from = 3

    def silence(loop):
        masks = {}
        for c in range(len(COLUMNS)):
            values = column_words(c, words, int(loop.speaker.n_candidates.max()))
            mask = np.ones((len(loop.order), len(values)), dtype=bool)
            if len(loop.said(1)) >= silent_from:
                mask[1] = values == UNCHANGED
            masks[c] = mask
        return masks

    loop = _fly(speaking(JOIN_TICKS), _sources(KEYS[:4]), caller=silence)
    said = loop.said(1)
    assert (said[silent_from:] == UNCHANGED).all() and len(said) > silent_from + 3
    drawn = np.stack(loop.speaker.drawn_probability, axis=1)[1][np.stack(loop.speaker.said_rows, axis=1)[1]]
    assert (drawn[silent_from:] == 1.0).all()
    sentence = loop.sentences("train")[1]
    with torch.no_grad():
        log_p = masked_log_probability(model, collate([sentence], CPU), loop.permitted([1])[0])
    start = loop.start
    assert float(log_p[0, start + silent_from:].abs().max()) == 0.0
    alone = _fly(speaking(None, (KEYS[0],)), _sources((KEYS[0],)))
    assert np.array_equal(loop.said(0), alone.said(0))              # the others are said as before


def test_a_row_refused_at_a_flights_first_predicted_step_leaves_the_loop_as_it_was(tmp_path, monkeypatch):
    """B12 (vocabulary D80) with join ticks: a row the executor refuses at the tick where a flight that joined later
    reaches its first predicted step is refused whole — the speaker's records, the loop's states and records, the
    flights' states kept so far — and the loop said on with the same numbers says and flies what a loop never refused
    does."""
    from ts_transformer.autopilot.start import RowRefused

    speaking, _, _ = _setup(tmp_path, monkeypatch)
    refuse_tick = int(JOIN_TICKS[1]) + 4                     # flight 1's first predicted step (s = 4 at Δ = 4 s)

    def state(loop):
        return (len(loop.speaker.permitted_rows), loop.t, len(loop._inputs), [len(x) for x in loop._said],
                [len(x) for x in loop._flown], loop.loop.steps, None if loop.current is None else loop.current.copy())

    def flown(refuse: bool):
        loop = speaking(JOIN_TICKS)
        sources = _sources(KEYS[:4])
        _fly(loop, sources, steps=refuse_tick)
        assert loop.own_row()[1] == loop.start
        if refuse:
            step = loop.loop.step

            def refusing(words_row):
                loop.loop.step = step
                raise RowRefused("flight 1: a stand-in's refusal", "stand-in")

            loop.loop.step = refusing
            before = state(loop)
            with pytest.raises(RowRefused, match="stand-in"):
                loop.step(np.full((4, len(COLUMNS)), 0.5))
            after = state(loop)
            assert after[:6] == before[:6] and np.array_equal(after[6], before[6])
        return _fly(loop, sources)

    never, refused = flown(False), flown(True)
    for b in range(4):
        assert np.array_equal(never.said(b), refused.said(b)) and np.array_equal(never.states(b), refused.states(b))


def test_loops_flown_together_move_each_parts_rows_and_pad_a_copys_narrower_rows():
    """D188: the records of the rows said (`Speaker.together`) and the inputs of every tick (`SpeakingLoop.together`)
    of loops flown together: each part's rows moved by its shift (a row before its first: the fill, no row present), its
    first rows left out below 0, and a row narrower than the widest (a copy's batch has fewer candidates than the batch
    it was copied from) padded to it."""
    from types import SimpleNamespace

    from ts_transformer.experiments.prior_speaking_loop import _inputs_together
    from ts_transformer.prior.batch import RowTensors
    from ts_transformer.prior.speaker import _rows_together

    wide = [np.full((2, 4), r + 1.0) for r in range(2)] + [np.full((2, 3), 3.0)]       # copied before its third row
    other = [np.full((1, 4), 10.0 + r) for r in range(5)]
    out = _rows_together([wide, other], [np.array([1]), np.array([0, 0])], [1, -1], -1.0)
    assert len(out) == 4 and all(row.shape == (3, 4) for row in out)
    assert out[0][0].tolist() == [-1.0] * 4 and out[1][0].tolist() == [1.0] * 4
    assert out[3][0].tolist() == [3.0, 3.0, 3.0, -1.0]                                   # the narrower row padded
    assert [row[1:, 0].tolist() for row in out] == [[11.0, 11.0], [12.0, 12.0], [13.0, 13.0], [14.0, 14.0]]

    def row(b, k, value):                                     # one tick's inputs of b aircraft and k candidates
        return RowTensors(np.full((b, 1), value), np.zeros((b, 1, 4)), np.full((b, 1, k, 7), value),
                          np.ones((b, k), bool), np.zeros((b, 1), np.int64), np.zeros((b, 1), bool),
                          np.zeros((b, 1, 2)), np.zeros((b, 1, 3), np.int64), np.zeros((b, 1, 5)),
                          np.ones((b, 1), bool), np.zeros((b, 1), bool), np.ones((b, 1), bool),
                          np.zeros((b, 1, 5), np.int64))

    copy = SimpleNamespace(_inputs=[row(1, 3, 1.0), row(1, 2, 2.0)])                  # narrower after it was copied
    later = SimpleNamespace(_inputs=[row(1, 3, 5.0 + t) for t in range(4)])
    ticks = _inputs_together([(copy, np.array([0]), 1), (later, np.array([0]), -1)], 3)
    assert len(ticks) == 3 and all(t.candidates.shape == (2, 1, 3, 7) and t.valid.shape == (2, 3) for t in ticks)
    assert not ticks[0].present[0, 0] and ticks[0].present[1, 0] and ticks[0].time_s[1, 0] == 6.0
    assert ticks[2].time_s[0, 0] == 2.0 and ticks[2].valid[0].tolist() == [True, True, False]
    assert ticks[2].candidates[0, 0, 2].tolist() == [0.0] * 7 and ticks[2].time_s[1, 0] == 8.0


def test_a_loop_whose_said_flights_left_keeps_a_row_of_records_each_tick_and_goes_with_others(tmp_path, monkeypatch):
    """S6 with join ticks: flight 1 joins inside flight 0's observed rows; flight 0 ends and leaves at a rebuild
    (`SpeakingLoop.together`); at the next tick no flight is said. The speaker keeps a row of records for that tick (said
    by none), so the loop goes with others again (MC A's batch of 2026-10-11 stopped here), and flight 1 says what it
    says alone."""
    from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop as Loops

    speaking, _, _ = _setup(tmp_path, monkeypatch)
    keys = KEYS[:2]
    sources = _sources(keys)
    loop = speaking(np.array([0, 2]), keys)
    while loop.roles()[0] != SAID:
        _fly(loop, sources, steps=1)
    assert loop.roles().tolist() == [SAID, OBSERVED]
    _fly(loop, sources, steps=1)                                    # flight 0 said once
    assert loop.roles().tolist() == [SAID, OBSERVED]
    loop.end(np.array([True, False]))
    joint = Loops.together([(loop, [0, 1])], CPU, None)
    assert 0 in joint.left and joint.inside.tolist() == [1]
    _fly(joint, sources, steps=1)
    assert not joint.speaker.said_rows[-1].any()
    assert len(joint.speaker.said_rows) == joint.t - joint.start
    again = _fly(Loops.together([(joint, [0, 1])], CPU, None), sources)       # a window goes whole, its left flight too
    alone = _fly(speaking(None, keys[1:]), _sources(keys[1:]))
    assert np.array_equal(again.said(1), alone.said(0))
    assert np.allclose(again.states(1), alone.states(0), rtol=0.0, atol=STATE_BOUND_M)
