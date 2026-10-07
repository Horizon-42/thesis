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
#: The keys of the copies of the flight; every loop's landings hold all of them, so a copy's inputs do not depend on
#: the other copies of its loop.
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
        loop = Loop(many, [geometry] * count, [test_start.A320_IAS] * count, [limit] * count, test_start._params(),
                    words, interval_s=INTERVAL_S, most_go_arounds=MOST_GO_AROUNDS, device=CPU, **joined)
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
    permitted = loop.permitted()
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
BEFORE_JOIN_TICKS = ("03dbe9d489c6ec3bcba1a2589d02bc1f469d3c1d9257918e291b6ab94f4879c7"
                     "6bc66c9fac4c54e902d2650198d46034470e13e117e3b6e3a4552ed5e972accc")


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
    records = together.permitted()
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
        mine, its = records.select([b]), alone.permitted()
        assert np.array_equal(mine.time_s[0, :rows], its.time_s[0])
        for a, c in zip(mine.masks, its.masks):
            assert np.array_equal(a[0, :rows, : c.shape[2]], c[0])
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


def test_a_landing_added_changes_only_the_counts_of_the_chosen_flights_rows_after_it(tmp_path, monkeypatch):
    """D150, §6.3 item 5 (multi-aircraft control D147): a landing added to chosen flights' landings while the loop runs
    changes only their rows whose time is after it (within the 30 min of the input), and nothing else of any row; a
    copy keeps it; the index keeps its checks — a second landing of a flight and a landing on a day outside the split
    are refused; one on a sealed test day is left out and counted with the sealed landings (requests item 6, the user,
    2026-10-07)."""
    speaking, _, _ = _setup(tmp_path, monkeypatch)
    loop = speaking(JOIN_TICKS)
    rows_of = loop.rows_of
    at = loop.observed[:, 4]
    before = loop.observed[:, 3]
    tick = 20
    first = rows_of(tick, at, before, True, loop.speaker.heard)[0]
    utc = rows_of.entry_utc_s + (rows_of.first_rows + (tick - rows_of.join_ticks) * rows_of.every) * rows_of.step_s
    idents = rows_of.landings[0].runways
    rows_of.add_landing([1, 3], Landing(float(utc[1]) - 1.0, idents[0], "NEW"))      # before flight 1's row, after 3's
    second = rows_of(tick, at, before, True, loop.speaker.heard)[0]
    for name, a, b in zip(first._fields, first, second):
        if name != "candidates":
            assert torch.equal(a, b), name
    changed = (first.candidates != second.candidates).any(dim=-1).any(dim=-1)[:, 0]
    assert utc[3] < utc[1] - 1.0 and changed.tolist() == [False, True, False, False]   # 3's row is earlier
    copy = loop.copy([1, 0])
    assert [index.digest() for index in copy.rows_of.landings] == [rows_of.landings[1].digest(),
                                                                   rows_of.landings[0].digest()]
    with pytest.raises(ValueError, match="a flight lands twice"):
        rows_of.landings[1].with_landing(Landing(float(utc[1]), idents[0], "NEW"))
    days = rows_of.landings[0].days
    test_day = datetime.strptime(sorted(days.days["test"])[0], "%Y-%m-%d").replace(hour=21, tzinfo=timezone.utc)
    sealed = rows_of.landings[0].with_landing(Landing(test_day.timestamp(), idents[0], "SEALED"))
    assert sealed.landings == rows_of.landings[0].landings and sealed.sealed == rows_of.landings[0].sealed + 1
    early = test_day.replace(hour=5) + timedelta(days=1)                   # UTC − 9 h: still the test day's
    assert rows_of.landings[0].with_landing(Landing(early.timestamp(), idents[0], "EARLY")).sealed == sealed.sealed
    with pytest.raises(ValueError, match="lands twice"):                    # the other checks first
        rows_of.landings[1].with_landing(Landing(test_day.timestamp(), idents[0], "NEW"))
    outside = datetime(1990, 1, 1, 21, tzinfo=timezone.utc)                # a day outside the split: refused
    with pytest.raises(KeyError, match="not in this day split"):
        rows_of.landings[0].with_landing(Landing(outside.timestamp(), idents[0], "OUTSIDE"))


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
        log_p = masked_log_probability(model, collate([sentence], CPU), loop.permitted().select([1]))
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
