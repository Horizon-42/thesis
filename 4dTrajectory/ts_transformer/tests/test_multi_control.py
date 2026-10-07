"""Stage D, MC2 and MC3: stage D's rules of the window loop and of branch training (multi-aircraft control §2, §3; D141–
D145, D148, D152; `multi/separation.py`, `multi/tokens.py`, `multi/credit.py`) — on the synthetic flights of stage C's
scene tests and the synthetic windows of stage C's and stage D's loop tests."""

from __future__ import annotations

import copy as copying

import numpy as np
import pytest

from ts_transformer.experiments.post_branches import Rules, branch_round, stage_c_rules
from ts_transformer.experiments.post_window_loop import start_move_of
from ts_transformer.instructions.artefact import load_candidates
from ts_transformer.instructions.grammar import column_words
from ts_transformer.instructions.words import ALTITUDE, ANGLE, COLUMNS, SPEED, UNCHANGED
from ts_transformer.multi import credit
from ts_transformer.multi.credit import Aircraft, event, points, spoken_again, varied, window_reward
from ts_transformer.multi.separation import Answering, losses_on_records
from ts_transformer.multi.tokens import PART_FEATURES, TOKENS_SCHEMA, TokenPart, commanded_part
from ts_transformer.post import edges
from ts_transformer.post.branches import branch_points, continuation_numbers, first_numbers
from ts_transformer.post.edges import TOKEN_FEATURES
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import AircraftAt, airport_scenes, real_windows
from ts_transformer.post.traffic import scene_aircraft, step_losses, traffic
from ts_transformer.post.traffic_attention import add_token_part
from ts_transformer.prior.speaker import SAID
from ts_transformer.tests.post_support import categories, finals, scene_artefact, straight_in, train_noon
from ts_transformer.tests.support import INSTRUCTION_STEP_S
from ts_transformer.tests.test_post_branches import _ahead
from ts_transformer.tests.test_post_generalised import _Digest
from ts_transformer.tests.test_post_window_loop import CPU, DELTA, _with_module, setup  # noqa: F401
from ts_transformer.tests.test_post_window_multi import JOINS, _multi, _numbers

#: The observed rows before a first predicted step at Δ = 4 s.
START = 4


# ---- D145: who answers for a loss with a recorded aircraft
@pytest.fixture
def stream(tmp_path):
    """Four train flights onto 09: b 40 s behind a (the records lose separation), d 120 s behind c (they keep it)."""
    flights = {"train": [straight_in("KXXX:a", train_noon(0.0)), straight_in("KXXX:b", train_noon(40.0)),
                         straight_in("KXXX:c", train_noon(1000.0)), straight_in("KXXX:d", train_noon(1120.0))]}
    spec = scene_artefact(tmp_path / "art", flights, interval_s=DELTA)
    geometries = load_candidates(tmp_path / "art")
    scenes, signals = airport_scenes(tmp_path / "art", "train", spec, DELTA, geometries, categories)
    windows = real_windows(tmp_path / "art", "train", spec, DELTA, scenes, signals)
    geometry = geometries["KXXX"]
    return windows, geometry, airport_separation(geometry), finals(geometry)


def _judged(window, step, own, geometry, separation, fin):
    """The loop's judged set of a window of one commanded aircraft at ``step``, the commanded one at ``own`` (its
    `AircraftAt` fields), and its losses."""
    aircraft = scene_aircraft(AircraftAt.of([own]), window.others_at(step))
    return aircraft, step_losses(traffic(aircraft, geometry, separation, fin, INSTRUCTION_STEP_S), aircraft.last_step,
                                 separation)


def test_a_loss_only_the_recorded_aircraft_answers_for_is_the_commanded_ones_where_the_records_kept_their_separation(
        stream):
    """D145, §2.4: of a loss between a commanded and a recorded aircraft that only the recorded follower is responsible
    for, the commanded aircraft answers when the two records kept their separation at that step (c flown 20 s ahead of
    its follower d, whose records are 120 s apart), and nobody answers when the records also lose it (a on its record,
    b 40 s behind); a loss that the commanded aircraft is responsible for is its own, as in stage C."""
    windows, geometry, separation, fin = stream
    a, _, c, _ = windows
    answering = Answering({geometry.code: separation}, {geometry.code: fin}, INSTRUCTION_STEP_S)
    own = list(a.commanded.at_step(a.step_s(60), DELTA))
    own[6] = False
    aircraft, (loss,) = _judged(a, 60, tuple(own), geometry, separation, fin)
    assert loss.responsible == (1,)                                     # the recorded follower alone
    assert frozenset(aircraft.keys) in losses_on_records(a, 60, separation, fin, INSTRUCTION_STEP_S)
    assert answering(a, 60, loss, aircraft, 1) == ()                    # the records lose it too: nothing for W
    d = next(r for r in c.scene.flights if r.key == "KXXX:d")
    step = 76
    assert c.commanded.start_s <= c.step_s(step) <= c.commanded.end_s     # both records in the air at the step
    ahead = list(d.at_step(c.step_s(step) + 20.0, DELTA))
    ahead[0], ahead[6] = c.commanded.key, False
    aircraft, (loss,) = _judged(c, step, tuple(ahead), geometry, separation, fin)
    assert loss.responsible == (1,) and losses_on_records(c, step, separation, fin, INSTRUCTION_STEP_S) == frozenset()
    assert answering(c, step, loss, aircraft, 1) == (0,)                # the model made it: the commanded one answers
    both = type(loss)(*[(0, 1) if f == "responsible" else getattr(loss, f) for f in loss.__dataclass_fields__])
    assert answering(c, step, both, aircraft, 1) == (0,)
    assert answering(a, 60, both, aircraft, 1) == (0,)                  # its own, whatever the records did


def test_two_commanded_aircraft_answer_by_the_rules_alone(stream):
    """D145, §2.4: a loss of two commanded aircraft is answered by each responsible one; the records are not read."""
    windows, geometry, separation, fin = stream
    answering = Answering({geometry.code: separation}, {geometry.code: fin}, INSTRUCTION_STEP_S)
    a = windows[0]
    own = list(a.commanded.at_step(a.step_s(60), DELTA))
    own[6] = False
    aircraft, (loss,) = _judged(a, 60, tuple(own), geometry, separation, fin)
    assert answering(a, 60, loss, aircraft, 2) == (1,)                  # b counted as commanded: it is responsible
    assert answering.records(a, 60) is answering.records(a, 60)          # judged once a step


# ---- D152: stage D's token part
def test_the_token_parts_scales_mirror_the_edge_features():
    from ts_transformer.multi import tokens

    assert tokens.HEIGHT_SCALE_M == edges.HEIGHT_SCALE_M and tokens.SPEED_SCALE_MPS == edges.SPEED_SCALE_MPS


def test_the_part_of_another_commanded_aircraft_is_its_flags_and_its_words_in_force(setup):
    """D152: the flags commanded and silent; no word before anything is said; then the altitude level over 1 km, the
    angle over 3°, the speed over 100 m/s, the heading as the prior reads it, and the flags of "no level-off" and of an
    unspecified speed."""
    words = setup["words"]
    index = {name: k for k, name in enumerate(PART_FEATURES)}
    nothing = commanded_part(True, np.zeros(2), np.full(3, -1), words)
    assert nothing.tolist() == [1.0, 1.0] + [0.0] * (len(PART_FEATURES) - 2)
    part = commanded_part(False, np.array([0.6, 0.8]), np.array([3, 1, 2]), words)
    assert part[index["commanded"]] == 1.0 and part[index["silent"]] == 0.0 and part[index["in_force"]] == 1.0
    assert part[index["heading_sin"]] == np.float32(0.6) and part[index["heading_cos"]] == np.float32(0.8)
    assert part[index["altitude"]] == np.float32(words.altitude_level_m(3) / 1000.0)
    assert part[index["angle"]] == np.float32(words.angle_deg(1) / 3.0)
    assert part[index["speed"]] == np.float32(words.speed_mps(2) / 100.0)
    assert part[index["no_level_off"]] == part[index["speed_unspecified"]] == 0.0
    free = commanded_part(False, np.zeros(2), np.array([words.altitude_no_level_off, 0, words.speed_unspecified]),
                          words)
    assert free[index["no_level_off"]] == free[index["speed_unspecified"]] == 1.0
    assert free[index["altitude"]] == free[index["speed"]] == free[index["angle"]] == 0.0


def _stepped(loop, numbers, until):
    """Step ``loop`` to tick ``until`` (or its end), each row said with its own numbers; before each step, the silent
    rows, each row's last altitude, angle and speed words said (None before any), and the rows joined and flown."""
    seen = {}
    while loop.speaking.alive.any() and loop.speaking.t < until:
        t = loop.speaking.t
        last = {}
        for b in range(len(loop.order)):
            said = loop.speaking.said(b)
            last[b] = None if not len(said) else [
                next((int(w) for w in said[::-1, c] if w != UNCHANGED), None) for c in (ALTITUDE, ANGLE, SPEED)]
        flown = [b for b in range(len(loop.order)) if loop.speaking.joined()[b] and loop.speaking.alive[b]]
        seen[t] = (loop.silent.copy(), last, flown)
        roles = loop.speaking.roles()
        loop.step(np.stack([n.random(len(COLUMNS)) if r == SAID else np.zeros(len(COLUMNS))
                            for n, r in zip(numbers, roles)]))
    return seen


def test_another_commanded_aircrafts_part_carries_its_words_from_the_row_after_they_are_said(setup):
    """D152, D148: in the loop, a commanded aircraft's token of another commanded aircraft carries the flag commanded,
    the flag silent of the start of the row, and that aircraft's words in force from the row after its first row said
    (the words it was told up to the row before) — none before."""
    s = setup
    _, loop_of = _multi(s)
    model = _with_module(s["base"])
    add_token_part(model, TokenPart.width, TOKENS_SCHEMA)
    loop = loop_of(model.eval(), token_part=TokenPart(s["words"]), part_width=TokenPart.width)
    seen = _stepped(loop, _numbers(len(JOINS)), JOINS[2] + START + 6)
    index = {name: len(TOKEN_FEATURES) + k for k, name in enumerate(PART_FEATURES)}
    words, checked, informed = s["words"], 0, 0
    for b, join in enumerate(JOINS):
        for own, row in enumerate(loop._tokens[b]):
            t = join + own
            silent, last, flown = seen[t]
            others = [c for c in flown if c != b] if b in flown else []    # the synthetic window has no recorded one
            assert len(row) == len(others), (t, b)
            for c, token in zip(others, row):
                assert token[index["commanded"]] == 1.0
                said_before = t > JOINS[c] + START                   # its first row said, at its own row START
                assert token[index["in_force"]] == float(said_before), (t, b, c)
                assert token[index["silent"]] == float(silent[c])
                if said_before:
                    altitude, angle, speed = last[c]
                    level = words.altitude_level_m(altitude)
                    assert token[index["altitude"]] == (np.float32(level / 1000.0) if level is not None else 0.0)
                    assert token[index["angle"]] == np.float32(words.angle_deg(angle) / 3.0)
                    target = words.speed_mps(speed)
                    assert token[index["speed"]] == (np.float32(target / 100.0) if target is not None else 0.0)
                    informed += 1
                checked += 1
    assert checked >= 10 and informed >= 3


def test_with_the_part_at_zero_the_loop_says_what_it_says_without_it(setup):
    """D152: stage D's token part starts at zero, so the module with it says and flies, aircraft by aircraft, what the
    module without it says and flies (the start is stage C's chosen round's words, bit for bit)."""
    s = setup
    _, loop_of = _multi(s)
    plain = _with_module(s["base"])
    with_part = copying.deepcopy(plain)
    add_token_part(with_part, TokenPart.width, TOKENS_SCHEMA)
    a = loop_of(plain.eval()).run(_numbers(len(JOINS)))
    b = loop_of(with_part.eval(), token_part=TokenPart(s["words"]), part_width=TokenPart.width).run(
        _numbers(len(JOINS)))
    for x, y in zip(a, b):
        assert np.array_equal(x.words, y.words) and np.array_equal(x.states, y.states)
        assert (x.outcome, x.reward, x.loss_step, x.other) == (y.outcome, y.reward, y.loss_step, y.other)


def test_a_change_of_one_aircrafts_words_at_its_first_predicted_step_leaves_the_scene_up_to_that_step(setup):
    """D23, D31, D148 with several commanded aircraft (§3.2): the words that copy 1 says at its first predicted step
    (its speed word forced to two permitted values; the synthetic airport has one runway, so its R is the same in both)
    leave every aircraft's inputs and tokens at the rows up to that step the same, bit for bit; from the next row the
    others' token of it carries the word. Its R is in force in the others' tokens of it only from the row after that
    step: no relation of the runways before it, one after."""
    s = setup
    c, tick = 1, JOINS[1] + START                                      # copy 1's first predicted step, a loop tick
    relation = [TOKEN_FEATURES.index(name) for name in edges.RELATIONS]
    speed = len(TOKEN_FEATURES) + PART_FEATURES.index("speed")
    runs = []
    for pick in (0, -1):
        _, loop_of = _multi(s)
        model = _with_module(s["base"])
        add_token_part(model, TokenPart.width, TOKENS_SCHEMA)
        loop = loop_of(model.eval(), token_part=TokenPart(s["words"]), part_width=TokenPart.width,
                       answering=lambda *args: ())                       # nobody silent: the scene stays whole
        real = loop._masks

        def forced(permitted, loop=loop, real=real, pick=pick):
            masks = dict(real(permitted))
            if loop.speaking.t == tick:
                mask = masks[SPEED].copy()
                allowed = np.flatnonzero(mask[c] & (column_words(SPEED, loop.words, 1) != UNCHANGED))
                assert len(allowed) >= 2
                mask[c] = False
                mask[c, allowed[pick]] = True
                masks[SPEED] = mask
            return masks

        loop._masks = forced
        _stepped(loop, _numbers(len(JOINS)), JOINS[2] + START + 1)          # every aircraft said: its sentence
        runs.append(loop)
    one, other = runs
    assert one.speaking.said(c)[0, SPEED] != other.speaking.said(c)[0, SPEED]
    for b, join in enumerate(JOINS):
        upto = tick - join + 1                                           # its rows up to the tick, inclusive
        if upto <= 0:
            continue
        x, y = one._tokens[b][:upto], other._tokens[b][:upto]
        assert len(x) == len(y) == upto and all(np.array_equal(p, q) for p, q in zip(x, y)), b
        rows_x, rows_y = one.speaking.sentences("train")[b], other.speaking.sentences("train")[b]
        for name in ("time_s", "own", "candidates", "runway_in_force", "go_around", "heading_in_force",
                     "words_in_force", "since"):
            assert np.array_equal(getattr(rows_x, name)[:upto], getattr(rows_y, name)[:upto]), (b, name)
    anchor = one._tokens[0]                                             # the anchor's token of copy 1 (its only other)
    assert all(len(anchor[t]) == 1 and not anchor[t][0, relation].any() for t in range(JOINS[1], tick + 1))
    assert anchor[tick + 1][0, relation].sum() == 1.0                    # R in force from the row after
    assert one._tokens[0][tick + 1][0, speed] != other._tokens[0][tick + 1][0, speed]


# ---- D141–D143: the window's reward and its credit
def _a(key, reward, first, end, loss=None, other=None):
    return Aircraft(key, reward, first, end, loss, other)


def test_the_windows_reward_is_the_sum_and_a_window_below_its_count_is_spoken_again():
    assert window_reward([1.0, 0.9, 0.0]) == 1.9 and window_reward([0.81]) == 0.81
    assert spoken_again([1.0, 0.9]) and not spoken_again([1.0, 1.0]) and spoken_again([0.0])


def test_the_varied_aircraft_are_those_below_one_and_the_other_commanded_aircraft_of_their_losses():
    """D143: each aircraft with r < 1, and the commanded aircraft of a loss that one of them answers for (here the
    anchor, r = 1, that the third aircraft lost separation against); a recorded aircraft of a loss is not one."""
    aircraft = [_a("x", 1.0, 4, 90), _a("y", 0.9, 12, 80), _a("z", 0.0, 20, 40, loss=40, other="x"),
                _a("w", 0.0, 24, 30, loss=30, other="recorded")]
    assert varied(aircraft) == [0, 1, 2, 3]
    assert varied([_a("x", 1.0, 4, 90), _a("y", 1.0, 12, 80)]) == []
    assert varied([_a("x", 1.0, 4, 90), _a("w", 0.0, 24, 30, loss=30, other="recorded")]) == [1]


def test_the_branch_points_of_a_varied_aircraft():
    """D143: its own first predicted step, then the window's grid (the anchor's first predicted step + every 120 s,
    30 ticks at Δ = 4 s) after it, before its event — its end, or the earliest loss it is in, answering or not."""
    anchor = _a("x", 1.0, 4, 200)
    late = _a("y", 0.0, 50, 150, loss=150, other="x")
    early = _a("z", 0.0, 10, 20, loss=20, other="recorded")
    aircraft = [anchor, late, early]
    assert points(aircraft, 1, DELTA) == [50, 64, 94, 124]
    assert event(aircraft, 0) == 150 and points(aircraft, 0, DELTA) == [4, 34, 64, 94, 124]   # in late's loss
    assert points(aircraft, 2, DELTA) == [10]
    assert points([_a("x", 0.0, 4, 100)], 0, DELTA) == branch_points(4, 100, DELTA)   # one aircraft: stage C's
    assert points([anchor, _a("y", 0.0, 50, 50)], 1, DELTA) == []        # ended at its first step: none


def test_stage_ds_numbers_are_each_aircrafts_own():
    """§3.3: each aircraft's own stream (the seed, the round, the window's place, its place in the window), and a
    continuation's from these, the branch point and k."""
    def draws(rng):
        return rng.random(5)

    assert np.array_equal(draws(credit.first_numbers(1, 2, 3, 0)), draws(credit.first_numbers(1, 2, 3, 0)))
    assert not np.array_equal(draws(credit.first_numbers(1, 2, 3, 0)), draws(credit.first_numbers(1, 2, 3, 1)))
    base = draws(credit.continuation_numbers(1, 2, 3, 1, 40, 0))
    for other in ((1, 2, 3, 0, 40, 0), (1, 2, 3, 1, 41, 0), (1, 2, 3, 1, 40, 1)):
        assert not np.array_equal(base, draws(credit.continuation_numbers(*other)))


def _round_c(s, model, windows, rules=None, loop_options=None):
    """`branch_round` of stage C's windows of one commanded aircraft (`test_post_branches._round`), under ``rules``."""
    return branch_round(model, lambda flights: s["moved_loop"](start_move_of(windows[0])), windows, [0],
                        {0: s["stored"]}, s["flights"], s["geometries"], {s["geometry"].code: s["roster"]}, s["finals"],
                        s["words"], interval_s=DELTA, variant="full", edges_reference=s["reference"],
                        faults={s["geometry"].code: {}}, device=CPU, seed=1337, round_=0, split="train",
                        continuations=2, rules=rules, loop_options=loop_options)


def _round_digest(round_):
    digest = _Digest()
    for result in round_.first:
        digest.result(result)
    digest.add(round_.spoken_again, round_.differed, len(round_.groups))
    for group in round_.groups:
        digest.add(group.window, group.branch, group.varied, group.informative)
        for sentence in group.sentences:
            digest.sentence(sentence.rows, sentence.permitted, sentence.tokens)
            digest.add(sentence.reward, sentence.until)
    return digest.hash.hexdigest()


def test_with_one_commanded_aircraft_stage_ds_rules_give_stage_cs_groups(setup):
    """D149, MC3: with one commanded aircraft in a window, stage D's reward, its rule of a window spoken again, its
    varied aircraft and their branch points, and its rule of who answers, give stage C's round bit for bit (stage D's
    numbers replaced by stage C's: the only part that differs by design, §3.3) — on window A (lost at its first row)
    and the real window (its time limit), each with its groups."""
    s = setup
    model = _with_module(s["base"])
    (window,) = s["windows"]
    stage_d = {**credit.rules(1337, 0, DELTA), "first": lambda place, member: first_numbers(1337, 0, place),
               "continuation": lambda place, member, v, tick, k: continuation_numbers(1337, 0, place, tick, k)}
    answering = Answering({s["geometry"].code: airport_separation(s["geometry"])},
                          {s["geometry"].code: s["finals"][s["geometry"].code]}, s["words"].spec.step_s)
    for chosen in (_ahead(window), window):
        c_round = _round_c(s, model, [chosen], stage_c_rules(1337, 0, DELTA))
        d_round = _round_c(s, model, [chosen], Rules(**stage_d), {"answering": answering})
        assert _round_digest(c_round) == _round_digest(d_round) and d_round.groups    # not vacuous: groups


def test_stage_ds_rules_vary_the_aircraft_of_d143_at_their_points(setup):
    """D142, D143 on the window of three commanded aircraft (each lost: W = 0): a group for each varied aircraft at each
    of its branch points (in its own rows), its sentences' rewards the windows' sums, its rows counted up to its
    event; only the varied aircraft is given new numbers."""
    s = setup
    multi, loop_of = _multi(s)
    parts = loop_of.parts
    asked = []
    fields = credit.rules(1337, 0, DELTA)
    real_continuation = fields["continuation"]

    def continuation(place, member, v, tick, k):
        asked.append((member, v))
        return real_continuation(place, member, v, tick, k)

    model = _with_module(s["base"])
    round_ = branch_round(model, parts["start"], [multi], [5], parts["sentences"], parts["flights"], s["geometries"],
                          parts["rosters"], s["finals"], s["words"], interval_s=DELTA, variant="full",
                          edges_reference=s["reference"], faults={s["geometry"].code: {}}, device=CPU, seed=1337,
                          round_=0, split="train", continuations=2, rules=Rules(**{**fields,
                                                                                   "continuation": continuation}))
    first = round_.first
    assert round_.spoken_again == [5] and round_.differed == []
    keys = [r.key for r in multi.commanded_all]
    aircraft = [Aircraft(keys[b], r.reward, JOINS[b] + START,
                         r.loss_step if r.loss_step is not None else JOINS[b] + START + len(r.words), r.loss_step,
                         r.other) for b, r in enumerate(first)]
    expected = sorted((v, p - JOINS[v]) for v in varied(aircraft) for p in points(aircraft, v, DELTA))
    assert expected and sorted((g.varied, g.branch) for g in round_.groups) == expected
    assert {m for m, _ in asked} == {v for v, _ in asked}                # only the varied aircraft is asked
    for group in round_.groups:
        assert group.window == 5 and group.first.reward == sum(r.reward for r in first)
        assert group.first.until == (aircraft[group.varied].end - JOINS[group.varied])
