"""The speaker and its masks (prior design §4, §7 item 3; milestone B4): the procedure masks rule by rule, G, the masks
of a caller, the grammar's column mask, and the same seed giving the same sentence."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from flight_scenarios.fas_geometry import fas_course_geometry
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.grammar import InForce, Ungrammatical, apply, column_words
from ts_transformer.instructions.words import ALTITUDE, ANGLE, COLUMNS, HEADING, RUNWAY, SPEED, UNCHANGED, Words
from ts_transformer.prior.batch import collate
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.procedure import GLIDEPATH_BELOW_M, THRESHOLD_TOLERANCE_M, Final, ProcedureMasks
from ts_transformer.prior.speaker import Position, Speaker
from ts_transformer.repo_layout import REPO_ROOT
from ts_transformer.tests.support import instruction_spec, parallel_airport, prior_sentence

CPU = torch.device("cpu")
FAF_M = 9_000.0


@pytest.fixture(scope="module")
def words():
    return Words(instruction_spec())


def finals():
    geometry = parallel_airport()
    return tuple(Final(geometry, k, FAF_M, fas_course_geometry(3000.0)) for k in range(2))


def masks(words, count=1):
    return ProcedureMasks([finals()] * count, words)


def level_words(words, heights):
    return [words.altitude_index(h) for h in heights]


def permitted(m, words, column, *, e, height, runway=0, go_around=False, n=0.0):
    return m.permitted(column, np.array([runway]), np.array([go_around]), np.array([e]), np.array([n]),
                       np.array([height]))[0]


def test_the_glidepath_lower_edge_binds_inside_the_faf_and_the_cone(words):
    final = finals()[0]
    m = masks(words)
    e = -5_000.0                                               # 5 km before "09"'s threshold, on its centreline
    edge = float(final.glidepath_m(np.array(5_000.0))) - GLIDEPATH_BELOW_M
    m.track(np.array([e]), np.array([0.0]), np.array([edge + 200.0]), np.array([False]))
    ok = permitted(m, words, ALTITUDE, e=e, height=edge + 200.0)
    for v in range(words.n_altitude_levels):
        level, band = words.altitude_level_m(v), words.altitude_tolerance_m(v)
        assert ok[1 + v] == (level >= edge - band), (v, level, edge)
    assert ok[1 + words.altitude_no_level_off]
    low = permitted(m, words, ALTITUDE, e=e, height=edge - words.altitude_tolerance_m(words.altitude_no_level_off) - 1)
    assert not low[1 + words.altitude_no_level_off]
    # inside the region only the edge binds, not the DA: 500 m before the threshold the edge is below the DA, and level
    # 0 is permitted there
    near = float(final.glidepath_m(np.array(500.0))) - GLIDEPATH_BELOW_M
    assert near < final.decision_m - words.altitude_tolerance_m(0)
    m.track(np.array([-500.0]), np.array([0.0]), np.array([near + 30.0]), np.array([False]))
    assert permitted(m, words, ALTITUDE, e=-500.0, height=near + 30.0)[1 + level_words(words, [0.0])[0]]

def test_a_level_below_the_da_is_blocked_after_the_join_when_the_aircraft_has_left_the_cone(words):
    """D64: the DA binds wherever the aircraft is not inside the region — also after the join, out of the LPV cone."""
    final = finals()[0]
    m = masks(words)
    e = -5_000.0
    edge = float(final.glidepath_m(np.array(5_000.0))) - GLIDEPATH_BELOW_M
    m.track(np.array([e]), np.array([0.0]), np.array([edge + 100.0]), np.array([False]))
    assert m.joined[0, 0]
    low = level_words(words, [final.decision_m - 100.0, final.decision_m + 60.0])
    out = permitted(m, words, ALTITUDE, e=e, n=-2_000.0, height=edge + 100.0)       # 2 km right of the final
    assert not out[1 + low[0]] and out[1 + low[1]]
    for v in range(words.n_altitude_levels):
        assert out[1 + v] == (words.altitude_level_m(v) >= final.decision_m - words.altitude_tolerance_m(v))


def test_before_the_join_the_decision_altitude_binds(words):
    final = finals()[0]
    m = masks(words)
    e = -20_000.0                                              # beyond the FAF: never joined
    m.track(np.array([e]), np.array([0.0]), np.array([1_200.0]), np.array([False]))
    ok = permitted(m, words, ALTITUDE, e=e, height=1_200.0)
    for v in range(words.n_altitude_levels):
        assert ok[1 + v] == (words.altitude_level_m(v) >= final.decision_m - words.altitude_tolerance_m(v))
    assert not ok[1 + level_words(words, [0.0])[0]] and ok[1 + level_words(words, [600.0])[0]]


def test_no_climb_below_the_entry_height_before_the_join_and_none_of_it_while_g(words):
    final = finals()[0]
    m = masks(words)
    e, height = -20_000.0, final.entry_m - 100.0
    m.track(np.array([e]), np.array([0.0]), np.array([height]), np.array([False]))
    up = level_words(words, [height + 300.0])[0]
    assert not permitted(m, words, ALTITUDE, e=e, height=height)[1 + up]
    assert not permitted(m, words, ANGLE, e=e, height=height)[1 + words.angle_climb]
    # D14: while G is true the rule does not apply …
    assert permitted(m, words, ALTITUDE, e=e, height=height, go_around=True)[1 + up]
    assert permitted(m, words, ANGLE, e=e, height=height, go_around=True)[1 + words.angle_climb]
    # … and its stretch starts again after the go-around
    m.track(np.array([e]), np.array([0.0]), np.array([final.entry_m + 200.0]), np.array([True]))
    m.track(np.array([e]), np.array([0.0]), np.array([final.entry_m + 200.0]), np.array([False]))
    assert permitted(m, words, ANGLE, e=e, height=final.entry_m + 200.0)[1 + words.angle_climb]


def test_the_row_whose_runway_word_ends_g_starts_the_stretch_again(words):
    """D64: at the row whose runway word ends G (G false after it), the masks read the row's own state — below the
    entry height before the join, "no climb" binds at that row — and keep it after the row, not one row later; a row
    whose word keeps G keeps nothing."""
    final = finals()[0]
    e, low, high = -20_000.0, final.entry_m - 100.0, final.entry_m + 200.0
    climb = 1 + words.angle_climb
    up = 1 + level_words(words, [low + 300.0])[0]
    m = masks(words)
    m.track(np.array([e]), np.array([0.0]), np.array([low]), np.array([True]))         # a row under G
    assert permitted(m, words, ANGLE, e=e, height=low, go_around=True)[climb]         # its word keeps G
    assert not permitted(m, words, ANGLE, e=e, height=low)[climb]                     # its word ends G: this row
    assert not permitted(m, words, ALTITUDE, e=e, height=low)[up]
    m.after_row(np.array([e]), np.array([0.0]), np.array([low]), np.array([False]))
    m.track(np.array([e]), np.array([0.0]), np.array([high]), np.array([False]))
    assert not permitted(m, words, ANGLE, e=e, height=high)[climb]                    # kept: it passed below
    kept_g = masks(words)
    kept_g.track(np.array([e]), np.array([0.0]), np.array([low]), np.array([True]))
    kept_g.after_row(np.array([e]), np.array([0.0]), np.array([low]), np.array([True]))
    kept_g.track(np.array([e]), np.array([0.0]), np.array([high]), np.array([False]))
    assert permitted(kept_g, words, ANGLE, e=e, height=high)[climb]                   # nothing kept under G


def test_a_procedure_mask_never_blocks_unchanged(words):
    """D64: a mask blocks a word when it is said, never "unchanged" — also where the word in force breaks a limit (a
    level under the edge, the aircraft sunk under it with "no level-off" in force, under the DA, a climb barred)."""
    final = finals()[0]
    m = masks(words)
    e = -5_000.0
    edge = float(final.glidepath_m(np.array(5_000.0))) - GLIDEPATH_BELOW_M
    for height in (edge - 200.0, edge + 200.0):
        m.track(np.array([e]), np.array([0.0]), np.array([height]), np.array([False]))
        assert permitted(m, words, ALTITUDE, e=e, height=height)[0] and permitted(m, words, ANGLE, e=e, height=height)[0]
    sunk = permitted(m, words, ALTITUDE, e=e, height=edge - 200.0)
    assert not sunk[1 + words.altitude_no_level_off] and sunk[0]
    far = ProcedureMasks([finals()], words)
    far.track(np.array([-20_000.0]), np.array([0.0]), np.array([final.entry_m - 300.0]), np.array([False]))
    barred = permitted(far, words, ANGLE, e=-20_000.0, height=final.entry_m - 300.0)
    assert not barred[1 + words.angle_climb] and barred[0]


@pytest.mark.parametrize("faf_m, segment_band_m", [(FAF_M, 40.0), (26_000.0, 70.0)])
def test_at_the_level_nearest_the_entry_height_the_aircraft_may_still_climb(words, faf_m, segment_band_m):
    """D64: "no climb" starts once the aircraft has been below the entry height by more than the band ε of the level
    nearest it — an aircraft held at that level, up to half a step below the entry height, has not passed under it. A
    FAF 26 km out puts the entry height in the 120 m segment (ε 70 m), not the 60 m one (ε 40 m)."""
    geometry = parallel_airport()
    final = Final(geometry, 0, faf_m, fas_course_geometry(3000.0))
    nearest = words.altitude_index(final.entry_m)
    band = words.altitude_tolerance_m(nearest)
    assert band == segment_band_m
    e = -faf_m - 10_000.0
    for below, barred in ((band - 1.0, False), (band + 1.0, True)):
        m = ProcedureMasks([(final, Final(geometry, 1, faf_m, fas_course_geometry(3000.0)))], words)
        height = final.entry_m - below
        m.track(np.array([e]), np.array([0.0]), np.array([height]), np.array([False]))
        assert permitted(m, words, ANGLE, e=e, height=height)[1 + words.angle_climb] == (not barred), below


def test_the_threshold_tolerance_mirrors_the_optimizers():
    """`procedure.THRESHOLD_TOLERANCE_M` is a mirror (the optimizer is not on the package's import path), pinned the way
    `test_guidance_skeleton_mirrors.py` pins the guidance's."""
    optimization = REPO_ROOT / "4dTrajectory" / "optimization"
    if str(optimization) not in sys.path:
        sys.path.insert(0, str(optimization))
    import scenario_optimization

    assert THRESHOLD_TOLERANCE_M == scenario_optimization._FRAME_ANCHOR_TOLERANCE_M


# ---- the speaker

SMALL = {"d_model": 32, "layers": 2, "heads": 4, "feedforward": 64}


def setup(words, seed=0, count=4, first_step=8, rows=40):
    torch.manual_seed(seed)
    model = Prior(PriorConfig.from_words(words, "full", **SMALL)).eval()
    rng = np.random.default_rng(seed)
    sentences = [prior_sentence(rng, words, candidates=2, rows=rows, first_step=first_step) for _ in range(count)]
    return model, collate(sentences, CPU)


def position(count, row):
    return Position(np.full(count, -15_000.0 + 140.0 * row), np.zeros(count), np.full(count, 700.0 - 4.0 * row))


def spread(count, row):
    """Each aircraft its own place: aircraft b 4 km nearer the threshold and 40 m lower than b − 1 (their masks
    differ)."""
    b = np.arange(count)
    return Position(-15_000.0 + 140.0 * row + 4_000.0 * b, np.zeros(count), 700.0 - 4.0 * row - 40.0 * b)


def numbers(seed, count, rows):
    """Each aircraft's own uniform numbers, ``[rows, count, 5]``: aircraft b's from (seed, b) alone (D96)."""
    return np.stack([np.random.default_rng([seed, b]).random((rows, len(COLUMNS))) for b in range(count)], axis=1)


def speak(model, rows, words, *, seed, caller=None, count_rows=20, extra=None):
    count = rows.present.shape[0]
    speaker = Speaker(model, words, [finals()] * count, capacity=rows.present.shape[1])
    first = int(rows.first[0].nonzero()[0, 0])
    speaker.observe(rows.between(0, first), [position(count, r) for r in range(first)], extra)
    drawn = numbers(seed, count, count_rows)
    said = [speaker.speak(rows.between(r, r + 1), position(count, r), drawn[r - first], caller, extra)
            for r in range(first, first + count_rows)]
    return speaker, np.stack(said, axis=1)


def test_the_same_seed_says_the_same_sentence(words):
    model, rows = setup(words)
    _, a = speak(model, rows, words, seed=7)
    _, b = speak(model, rows, words, seed=7)
    _, c = speak(model, rows, words, seed=8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_every_row_said_passes_the_grammar_and_the_first_says_every_column(words):
    model, rows = setup(words, count=6)
    speaker, said = speak(model, rows, words, seed=3, count_rows=30)
    assert (said[:, 0] != UNCHANGED).all() and (said[:, 0, RUNWAY] >= 0).all()
    for b in range(said.shape[0]):
        state = None
        for r in range(said.shape[1]):
            state = apply(state, said[b, r], float(position(1, 8 + r).height_m[0]), words, 2)
        assert state == speaker.in_force[b]
        # the speaker's words in force are the inputs' own walk: the next row's inputs come from it
        runway, go_around, _, levels, _ = speaker.heard[b].inputs(float(rows.time_s[b, 8 + said.shape[1]]))
        assert (runway, go_around, tuple(levels)) == (state.runway, state.go_around,
                                                      (state.altitude, state.angle, state.speed))
    assert all(len(speaker.forbidden[c]) == 30 for c in range(len(COLUMNS)))
    # at the first step the model blocks "unchanged" and "go-around" itself: the masks remove nothing more there
    assert not speaker.forbidden[RUNWAY][0].any()


def test_the_masks_of_a_caller_are_applied(words):
    model, rows = setup(words)
    count = rows.present.shape[0]
    heading = np.zeros((count, 1 + words.n_heading), dtype=bool)
    heading[:, [0, 1 + 3]] = True                              # "unchanged" or heading class 3 only
    speed = np.ones((count, 1 + words.n_speed_levels + 1), dtype=bool)
    speed[:, 1:6] = False
    speaker, said = speak(model, rows, words, seed=1, caller={HEADING: heading, SPEED: speed})
    assert all((value > 0.5).all() for value in speaker.forbidden[HEADING][:1])  # most of the heading words removed
    assert set(np.unique(said[:, :, HEADING])) <= {UNCHANGED, 3} and (said[:, 0, HEADING] == 3).all()
    assert not np.isin(said[:, :, SPEED], range(5)).any()


def test_no_row_reaches_a_column_with_no_permitted_word(words):
    """D62: the grammar is asked with the later columns' other masks, so random caller masks (each column keeping some
    words) never leave a later column empty, with the procedure masks on (the DA before the join; "no climb" from the
    rows below the entry height; the glidepath edge once inside the FAF)."""
    model, rows = setup(words, count=8, rows=60)
    rng = np.random.default_rng(5)
    count = rows.present.shape[0]
    widths = {HEADING: 1 + words.n_heading, ALTITUDE: 2 + words.n_altitude_levels, ANGLE: 3 + words.n_descent,
              SPEED: 2 + words.n_speed_levels}
    for trial in range(3):
        caller = {c: rng.random((count, w)) < 0.6 for c, w in widths.items()}
        for c in caller:
            caller[c][:, 0] = True
        speak(model, rows, words, seed=trial, caller=caller, count_rows=40)


def test_a_dead_end_raises(words):
    """A caller that permits no word of a column at the first predicted step: the speaker refuses, never says a word
    no mask permits."""
    model, rows = setup(words)
    count = rows.present.shape[0]
    nothing = np.zeros((count, 1 + words.n_heading), dtype=bool)
    nothing[:, 0] = True                                       # only "unchanged", which the first step cannot say
    with pytest.raises(ValueError, match="no permitted word"):
        speak(model, rows, words, seed=0, caller={HEADING: nothing}, count_rows=1)


def test_the_finals_read_from_the_cifp_for_an_airport_of_the_artefact():
    """`airport_finals` on KRDU's live data (read only): one final for each candidate, its FAF 5–20 km out, the entry
    height above the DA; a candidate whose CIFP threshold is more than the tolerance away is refused; one procedure
    document digest for each candidate."""
    from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
    from trajectory_data_process.harvest.airports import load_airport
    from ts_transformer.instructions.airport import airport_geometry
    from ts_transformer.prior.procedure import airport_finals, procedure_digests

    geometry = airport_geometry("KRDU", load_airport("KRDU", config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP).runways)
    finals_ = airport_finals(geometry)
    assert [f.index for f in finals_] == list(range(len(geometry.candidates)))
    for final in finals_:
        assert 5_000.0 < final.faf_m < 20_000.0 and final.entry_m > final.decision_m > 0.0
    digests = procedure_digests({"KRDU": geometry})["KRDU"]
    assert set(digests) == {c.ident for c in geometry.candidates}
    assert all(len(d) == 64 for d in digests.values())
    data = geometry.to_dict()
    data["candidates"][0]["threshold_n_m"] += THRESHOLD_TOLERANCE_M + 50.0
    with pytest.raises(ValueError, match="from the candidate's"):
        airport_finals(AirportGeometry.from_dict(data))


def test_the_runway_column_asks_the_masks_under_each_runway_word(words):
    """D62 with the procedure masks: the aircraft is on "09"'s final 5 km out (joined: the glidepath edge binds), not on
    "09L"'s (the DA binds); a caller permits only the levels between the DA and the edge. Under "09L" they are
    permitted, under "09" (in force) not — so the runway word "09L" is permitted, "unchanged" not. The speaker's mask
    equals the brute force: every completion of the row through `apply`, each word among its masks under the runway
    word asked."""
    from itertools import product

    model, rows = setup(words, count=1)
    speaker = Speaker(model, words, [finals()], capacity=rows.present.shape[1])
    final = finals()[0]
    edge = float(final.glidepath_m(np.array(5_000.0))) - GLIDEPATH_BELOW_M
    at = Position(np.array([-5_000.0]), np.array([0.0]), np.array([edge + 60.0]))
    speaker.procedure.track(at.e_m, at.n_m, at.height_m, np.array([False]))
    speaker._heard[0].state = InForce(runway=0, go_around=False, heading=0, altitude=words.altitude_index(900.0),
                                     angle=words.angle_index(3.0), speed=words.speed_index(75.0))
    between = [v for v in range(words.n_altitude_levels)
               if final.decision_m < words.altitude_level_m(v) < edge - words.altitude_tolerance_m(v)]
    assert between
    altitude = np.zeros((1, 2 + words.n_altitude_levels), dtype=bool)
    altitude[0, [1 + v for v in between]] = True
    caller = {ALTITUDE: altitude}
    mask = speaker._allowed(RUNWAY, np.zeros((1, 0), dtype=np.int64), at, caller, speaker.procedure)[0]
    runway_words = column_words(RUNWAY, words, 2)
    assert mask[list(runway_words).index(1)] and not mask[list(runway_words).index(UNCHANGED)]
    for w, word in enumerate(runway_words):
        runway, go_around = speaker._runway_after(np.array([word]))
        others = {c: speaker._others(c, runway, go_around, at, caller, speaker.procedure)[0] for c in range(1, len(COLUMNS))}
        options = [column_words(c, words, 2)[others[c]] for c in range(1, len(COLUMNS))]
        found = False
        for rest in product(*options):
            try:
                apply(speaker.in_force[0], [word, *rest], float(at.height_m[0]), words, 2)
            except Ungrammatical:
                continue
            found = True
            break
        assert mask[w] == found, (word, mask[w], found)


def test_after_its_second_go_around_a_flight_may_not_say_another(words):
    """D68: a caller that permits in the runway column only "go-around" and the candidates (never "unchanged") makes
    each flight alternate go-around and a runway word; with the bound of D68 as a caller mask too, no flight says more
    than 2 go-arounds, and every flight says exactly 2."""
    from ts_transformer.prior.speaker import MOST_GO_AROUNDS, go_around_bound

    model, rows = setup(words, count=3, rows=40)
    count = rows.present.shape[0]
    speaker = Speaker(model, words, [finals()] * count, capacity=rows.present.shape[1])
    first = int(rows.first[0].nonzero()[0, 0])
    speaker.observe(rows.between(0, first), [position(count, r) for r in range(first)])
    no_unchanged = np.ones((count, 4), dtype=bool)
    no_unchanged[:, 0] = False
    drawn = numbers(4, count, 12)
    for r in range(first, first + 12):
        caller = {RUNWAY: no_unchanged & go_around_bound(speaker.go_arounds, words, 2)}
        said = speaker.speak(rows.between(r, r + 1), position(count, r), drawn[r - first], caller)
        assert (said[:, RUNWAY] != UNCHANGED).all() and (speaker.go_arounds <= MOST_GO_AROUNDS).all()
    assert (speaker.go_arounds == MOST_GO_AROUNDS).all()
    assert go_around_bound(np.array([0, 1, 2]), words, 2)[:, 1].tolist() == [True, True, False]


def test_the_speaker_records_the_words_the_procedure_masks_blocked_and_says_none_of_them(words):
    """B6: at each row said, for each column the procedure masks rule, the words they blocked (the Training view shows
    them): never "unchanged", never a word said; before the join the DA blocks the levels below it."""
    model, rows = setup(words, count=4)
    speaker, said = speak(model, rows, words, seed=5, count_rows=20)
    assert len(speaker.procedure_blocked) == 20
    for r, blocked in enumerate(speaker.procedure_blocked):
        assert set(blocked) == set(ProcedureMasks.columns)
        for column, mask in blocked.items():
            classes = list(column_words(column, words, 2))
            assert mask.shape == (4, len(classes)) and not mask[:, classes.index(UNCHANGED)].any()
            for b in range(4):
                assert not mask[b, classes.index(int(said[b, r, column]))]
    da = finals()[0].decision_m
    below = [v for v in range(words.n_altitude_levels) if words.altitude_level_m(v) < da - words.altitude_tolerance_m(v)]
    assert below
    classes = list(column_words(ALTITUDE, words, 2))
    assert all(speaker.procedure_blocked[0][ALTITUDE][:, classes.index(v)].all() for v in below)


# ---- B9: the interface for the post-training (D96)

def subset(rows, indices):
    """The rows of the aircraft ``indices`` of a batch, in that order."""
    return type(rows)(*(value[list(indices)] for value in rows))


def speak_with(model, rows, words, ids, count_rows, *, seed=11, extra=None):
    """The aircraft ``ids`` of ``rows`` spoken together, each with its own numbers (`numbers`: aircraft b's from (seed,
    b)); the speaker and ``[B, rows, 5]`` words."""
    speaker = Speaker(model, words, [finals()] * len(ids), capacity=8)
    part = subset(rows, ids)
    first = int(part.first[0].nonzero()[0, 0])
    speaker.observe(part.between(0, first), [position(len(ids), r) for r in range(first)], extra)
    drawn = numbers(seed, max(ids) + 1, count_rows)[:, list(ids)]
    said = [speaker.speak(part.between(r, r + 1), position(len(ids), r), drawn[r - first], None, extra)
            for r in range(first, first + count_rows)]
    return speaker, np.stack(said, axis=1)


def test_an_aircraft_says_the_same_words_alone_and_in_a_batch(words):
    """D96 item 2: an aircraft's words depend on its own numbers and inputs only. Each of 6 aircraft spoken alone and
    in the batch of all 6 says the same words at every row (a difference could come only from a number within the float
    tolerance of a boundary, post-training §6.4: counted, none on this batch)."""
    model, rows = setup(words, seed=2, count=6)
    _, together = speak_with(model, rows, words, list(range(6)), 25)
    differ = sum(int((speak_with(model, rows, words, [b], 25)[1][0] != together[b]).any(axis=-1).sum()) for b in range(6))
    assert differ == 0
    _, reversed_ = speak_with(model, rows, words, list(range(5, -1, -1)), 25)
    assert np.array_equal(reversed_[::-1], together)


def test_the_log_probability_under_the_records_is_the_probability_the_speaker_drew_from(words):
    """D96 item 3: teacher-forced on the words the speaker said, under its records of the permitted words, the
    probability of each word is the one it was drawn from (the encodings agree within the float tolerance); a word a
    record blocks has probability 0; gradients flow."""
    from ts_transformer.prior.batch import target_classes
    from ts_transformer.prior.train import masked_log_probability

    model, rows = setup(words, seed=3, count=4)
    speaker = Speaker(model, words, [finals()] * 4, capacity=8)
    first = int(rows.first[0].nonzero()[0, 0])
    speaker.observe(rows.between(0, first), [spread(4, r) for r in range(first)])
    drawn_numbers = numbers(11, 4, 20)
    said = np.stack([speaker.speak(rows.between(r, r + 1), spread(4, r), drawn_numbers[r - first])
                     for r in range(first, first + 20)], axis=1)
    first = int(rows.first[0].nonzero()[0, 0])
    part = rows.between(0, first + 20)
    targets = part.targets.clone()
    targets[:, first:] = torch.as_tensor(target_classes(said))
    told = part._replace(targets=targets)
    log_p = masked_log_probability(model, told, speaker.permitted())
    drawn = torch.as_tensor(np.stack(speaker.drawn_probability, axis=1), dtype=torch.float32)
    torch.testing.assert_close(log_p[:, first:].exp(), drawn, rtol=1e-4, atol=1e-6)
    assert (log_p[:, :first] == 0).all()
    log_p[:, first:].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
    # a word the records block (not "unchanged", which the model blocks itself at the first step): probability 0 under
    # the records, positive under records that permit every word
    from ts_transformer.prior.speaker import Permitted

    record = speaker.permitted()
    blocked = [k for k in np.flatnonzero(~record.masks[ALTITUDE][0, 0]) if k != 0]
    assert blocked
    targets[0, first, ALTITUDE] = int(blocked[0])
    every = Permitted(tuple(np.ones_like(m) for m in record.masks), record.time_s, record.own, record.temperature)
    with torch.no_grad():
        changed = part._replace(targets=targets)
        assert masked_log_probability(model, changed, record)[0, first, ALTITUDE] == float("-inf")
        assert torch.isfinite(masked_log_probability(model, changed, every)[0, first, ALTITUDE])
        with pytest.raises(ValueError, match="records of 3 aircraft"):
            masked_log_probability(model, changed, record.select([0, 1, 2]))
        later = part._replace(time_s=part.time_s + 1000.0)
        with pytest.raises(ValueError, match="no record of the rows"):
            masked_log_probability(model, later, record)
        with pytest.raises(ValueError, match="another aircraft's records"):          # the records of other aircraft
            masked_log_probability(model, changed, record.select([3, 2, 1, 0]))
        # rows asked from 5 rows after the first said: each reads its own row's record
        asked = told.asked.clone()
        asked[:, first: first + 5] = False
        torch.testing.assert_close(masked_log_probability(model, told._replace(asked=asked), record)[:, first + 5:].exp(),
                                   drawn[:, 5:], rtol=1e-4, atol=1e-6)


def test_a_copy_continued_with_the_same_inputs_and_numbers_says_what_the_original_says(words):
    """D96 item 5: a copy of every aircraft (the same layout) continued with the same inputs and numbers says the same
    words with the same probabilities, bit for bit; a copy of chosen aircraft, repeated, carries their state: records,
    words in force, go-arounds, procedure masks."""
    model, rows = setup(words, seed=4, count=4, rows=50)
    speaker = Speaker(model, words, [finals()] * 4, capacity=8)
    first = int(rows.first[0].nonzero()[0, 0])
    speaker.observe(rows.between(0, first), [spread(4, r) for r in range(first)])
    drawn = numbers(5, 4, 30)
    for r in range(first, first + 10):
        speaker.speak(rows.between(r, r + 1), spread(4, r), drawn[r - first])
    assert not (speaker.procedure.joined == speaker.procedure.joined[0]).all()      # the aircraft's masks differ
    twin = speaker.copy(range(4))
    picked = speaker.copy([2, 2, 0])
    assert np.array_equal(picked.permitted().masks[ALTITUDE], speaker.permitted().masks[ALTITUDE][[2, 2, 0]])
    assert picked.in_force == [speaker.in_force[i] for i in (2, 2, 0)]
    assert np.array_equal(picked.go_arounds, speaker.go_arounds[[2, 2, 0]])
    assert np.array_equal(picked.procedure.joined, speaker.procedure.joined[[2, 2, 0]])
    chosen = subset(rows, [2, 2, 0])
    for r in range(first + 10, first + 30):
        a = speaker.speak(rows.between(r, r + 1), spread(4, r), drawn[r - first])
        b = twin.speak(rows.between(r, r + 1), spread(4, r), drawn[r - first])
        assert np.array_equal(a, b) and np.array_equal(speaker.drawn_probability[-1], twin.drawn_probability[-1])
        at = spread(4, r)
        c = picked.speak(chosen.between(r, r + 1), Position(*(v[[2, 2, 0]] for v in at)), drawn[r - first][[2, 2, 0]])
        assert np.array_equal(c, a[[2, 2, 0]])        # another layout: the same words (none near a boundary here)
    assert np.array_equal(speaker.permitted().masks[RUNWAY], twin.permitted().masks[RUNWAY])


class Recorder(torch.nn.Module):
    """An added module (§7 item 5) whose output is zero; it keeps every input of the caller it is given."""

    def __init__(self, seen):
        super().__init__()
        self.seen = seen

    def forward(self, x, extra):
        self.seen.append(extra)
        return torch.zeros_like(x)


def test_the_speaker_gives_the_added_modules_the_caller_s_input_and_a_zero_module_changes_no_word(words):
    """D96 item 1: every layer's added module gets the caller's input at every row the speaker encodes or says; one
    whose output is zero changes no word."""
    model, rows = setup(words, seed=6, count=3)
    _, plain = speak_with(model, rows, words, [0, 1, 2], 15)
    seen = []
    model.add_at_each_layer(lambda i: Recorder(seen))
    model.eval()                                            # D107: the added modules too
    token = object()
    _, added = speak_with(model, rows, words, [0, 1, 2], 15, extra=token)
    assert np.array_equal(added, plain)
    assert len(seen) == len(model.layers) * (1 + 15) and all(item is token for item in seen)   # observe once, 15 rows


def test_draw_takes_the_first_class_whose_cumulative_probability_passes_the_number():
    from ts_transformer.prior.speaker import draw

    probabilities = torch.tensor([[0.25, 0.0, 0.5, 0.25]] * 6, dtype=torch.float32)     # exact in binary
    numbers_ = np.array([0.0, 0.2499, 0.25, 0.7499, 0.75, 0.999999999])
    assert draw(probabilities, numbers_)[:, 0].tolist() == [0, 0, 2, 2, 3, 3]     # never class 1 (probability 0)
    one = torch.tensor([[0.0, 1.0, 0.0]])
    assert draw(one, np.array([0.999999999]))[:, 0].tolist() == [1]
    # a float sum short of 1: a number past it says the last class of positive probability, never a zero one
    short = torch.tensor([[0.3, 0.6999999, 0.0]], dtype=torch.float32)
    assert draw(short, np.array([0.99999999]))[:, 0].tolist() == [1]


def test_records_of_airports_with_other_numbers_of_candidates_go_with_their_aircraft(words):
    """The runway column's width is a batch's (2 + its most candidates): a speaker of a 2-candidate and a 1-candidate
    aircraft, a copy of the 1-candidate one twice spoken on in a batch of its own width, gives its records; under them
    the log-probability of its words, scored in a batch beside a 2-candidate aircraft, is the probability it drew."""
    import dataclasses

    from ts_transformer.prior.batch import target_classes
    from ts_transformer.prior.speaker import Permitted
    from ts_transformer.prior.train import masked_log_probability

    torch.manual_seed(7)
    model = Prior(PriorConfig.from_words(words, "full", **SMALL)).eval()
    rng = np.random.default_rng(7)
    sentences = [prior_sentence(rng, words, candidates=k, rows=40, first_step=8) for k in (2, 1)]
    rows = collate(sentences, CPU)
    one = dataclasses.replace(parallel_airport(), candidates=parallel_airport().candidates[:1])
    finals_one = (Final(one, 0, FAF_M, fas_course_geometry(3000.0)),)
    speaker = Speaker(model, words, [finals(), finals_one], capacity=8)
    speaker.observe(rows.between(0, 8), [spread(2, r) for r in range(8)])
    drawn = numbers(9, 2, 20)
    said = [speaker.speak(rows.between(r, r + 1), spread(2, r), drawn[r - 8]) for r in range(8, 13)]
    copy = speaker.copy([1, 1])
    alone = collate([sentences[1]] * 2, CPU)                                   # the copy's own width: 1 candidate
    for r in range(13, 20):
        at = spread(2, r)
        said.append(copy.speak(alone.between(r, r + 1), Position(*(v[[1, 1]] for v in at)), drawn[r - 8][[1, 1]]))
    copied = copy.permitted()
    assert copied.masks[RUNWAY].shape[-1] == 4 and not copied.masks[RUNWAY][:, 5:, 3].any()   # no candidate 2 after
    for r in range(13, 20):                                     # the original speaks on: the 2-candidate aircraft's rows
        speaker.speak(rows.between(r, r + 1), spread(2, r), drawn[r - 8])
    original = speaker.permitted().select([0])
    first_ = copied.select([0])
    record = Permitted(tuple(np.concatenate(pair) for pair in zip(first_.masks, original.masks)),
                       np.concatenate([first_.time_s, original.time_s]), np.concatenate([first_.own, original.own]),
                       first_.temperature)
    # the copy's first aircraft scored first in a batch with the 2-candidate aircraft (4 runway classes); its words: the
    # original's aircraft 1 to row 12, the copy's from row 13
    told = [said[k][1] for k in range(5)] + [said[k][0] for k in range(5, 12)]
    batch = collate([sentences[1], sentences[0]], CPU).between(0, 20)
    targets = batch.targets.clone()
    targets[0, 8:20] = torch.as_tensor(target_classes(np.stack(told)))
    batch = batch._replace(targets=targets)
    with torch.no_grad():
        log_p = masked_log_probability(model, batch, record)
        torch.testing.assert_close(log_p[0, 8:20].exp(), torch.as_tensor(np.stack(copy.drawn_probability)[:, 0],
                                                                         dtype=torch.float32), rtol=1e-4, atol=1e-6)
        # a record narrower than the batch's classes permits none of the extra ones (here: 3 of 4 runway classes)
        narrow = Permitted(tuple(m[..., :3] if c == RUNWAY else m for c, m in enumerate(record.masks)), record.time_s,
                           record.own, record.temperature)
        torch.testing.assert_close(masked_log_probability(model, batch, narrow)[0], log_p[0])
        # one wider than the batch is refused only when it permits a class past it
        alone_batch = collate([sentences[1]] * 2, CPU).between(0, 20)
        masked_log_probability(model, alone_batch, copied)                    # 4 wide, 3 classes: allowed
        wide = Permitted(tuple(m.copy() for m in copied.masks), copied.time_s, copied.own, copied.temperature)
        wide.masks[RUNWAY][:, 0, 3] = True
        with pytest.raises(ValueError, match="past the batch"):
            masked_log_probability(model, alone_batch, wide)


def test_the_speaker_refuses_a_training_model_and_a_row_not_after_the_last(words):
    model, rows = setup(words, seed=8, count=2)
    with pytest.raises(ValueError, match="training"):
        Speaker(model.train(), words, [finals()] * 2, capacity=8)
    speaker = Speaker(model.eval(), words, [finals()] * 2, capacity=8)
    first = int(rows.first[0].nonzero()[0, 0])
    speaker.observe(rows.between(0, first), [position(2, r) for r in range(first)])
    speaker.speak(rows.between(first, first + 1), position(2, first), numbers(1, 2, 1)[0])
    speaker.speak(rows.between(first + 1, first + 2), position(2, first + 1), numbers(1, 2, 1)[0])
    with pytest.raises(ValueError, match="time order"):
        speaker.speak(rows.between(first + 1, first + 2), position(2, first + 1), numbers(1, 2, 1)[0])
    model.train()
    with pytest.raises(ValueError, match="training"):
        speaker.speak(rows.between(first + 2, first + 3), position(2, first + 2), numbers(1, 2, 1)[0])
    model.eval()


def test_what_a_loop_reads_of_the_speaker_does_not_change_it(words):
    """D106 item 6: the words in force, the go-arounds said and the number of candidates a loop reads are copies."""
    model, rows = setup(words, seed=9, count=2)
    speaker = Speaker(model, words, [finals()] * 2, capacity=8)
    first = int(rows.first[0].nonzero()[0, 0])
    speaker.observe(rows.between(0, first), [position(2, r) for r in range(first)])
    said = speaker.speak(rows.between(first, first + 1), position(2, first), numbers(1, 2, 1)[0])
    in_force = speaker.in_force
    heard, go_arounds, candidates = speaker.heard, speaker.go_arounds, speaker.n_candidates
    heard[0].hear(said[1], 500.0, 99.0)                      # another row heard by the copy
    go_arounds[:] = 7
    candidates[:] = 0
    assert speaker.in_force == in_force and (speaker.go_arounds == 0).all() and (speaker.n_candidates == 2).all()
    assert heard[0].state != in_force[0] or heard[0].said_s.max() == 99.0
    assert speaker.heard[0].said_s.max() < 99.0


def test_a_refused_row_leaves_the_speaker_as_it_was(words):
    """A row refused after the speaker began it (here: no word of a column permitted, D62) changes nothing — its cache,
    procedure masks, words heard, go-arounds and records — so the next row is said as by a speaker that never saw it
    (a copy taken before), bit for bit."""
    model, rows = setup(words, seed=10, count=2)
    speaker = Speaker(model, words, [finals()] * 2, capacity=8)
    first = int(rows.first[0].nonzero()[0, 0])
    speaker.observe(rows.between(0, first), [spread(2, r) for r in range(first)])
    drawn = numbers(3, 2, 6)
    for r in range(first, first + 3):
        speaker.speak(rows.between(r, r + 1), spread(2, r), drawn[r - first])
    twin = speaker.copy(range(2))
    r = first + 3
    nothing = np.zeros((2, 1 + words.n_heading), dtype=bool)
    with pytest.raises(ValueError, match="no permitted word"):
        speaker.speak(rows.between(r, r + 1), spread(2, r), drawn[3], {HEADING: nothing})
    assert [p.rows for p in speaker.past] == [p.rows for p in twin.past]
    assert speaker.in_force == twin.in_force and np.array_equal(speaker.go_arounds, twin.go_arounds)
    for name in ("joined", "dipped", "cleared"):
        assert np.array_equal(getattr(speaker.procedure, name), getattr(twin.procedure, name)), name
    assert len(speaker.drawn_probability) == len(twin.drawn_probability) == 3
    assert all(len(rows_) == 3 for rows_ in speaker.forbidden.values()) and len(speaker.procedure_blocked) == 3
    a = speaker.speak(rows.between(r, r + 1), spread(2, r), drawn[3])
    b = twin.speak(rows.between(r, r + 1), spread(2, r), drawn[3])
    assert np.array_equal(a, b) and np.array_equal(speaker.drawn_probability[-1], twin.drawn_probability[-1])


def test_a_module_in_training_mode_inside_an_eval_model_is_refused(words):
    """D107: the speaker and the log-probability under its records refuse a model of which any module is in training
    mode — an added module too; the teacher-forced loss (`batch_nll`) does not."""
    from ts_transformer.prior.train import batch_nll, masked_log_probability

    model, rows = setup(words, seed=11, count=2)
    model.layers[1].train()
    with pytest.raises(ValueError, match="D107"):
        Speaker(model, words, [finals()] * 2, capacity=8)
    model.eval()
    speaker = Speaker(model, words, [finals()] * 2, capacity=8)
    first = int(rows.first[0].nonzero()[0, 0])
    speaker.observe(rows.between(0, first), [spread(2, r) for r in range(first)])
    speaker.speak(rows.between(first, first + 1), spread(2, first), numbers(1, 2, 1)[0])
    model.add_at_each_layer(lambda i: Recorder([]))                       # added modules, in training mode
    with pytest.raises(ValueError, match="D107"):
        speaker.speak(rows.between(first + 1, first + 2), spread(2, first + 1), numbers(1, 2, 1)[0])
    told = rows.between(0, first + 1)
    with pytest.raises(ValueError, match="D107"):
        masked_log_probability(model, told, speaker.permitted())
    batch_nll(model, told)                                                # the data term reads either mode
    model.eval()
    masked_log_probability(model, told._replace(targets=told.targets), speaker.permitted())


def test_rows_come_in_time_order_and_none_is_observed_after_one_is_said(words):
    model, rows = setup(words, seed=12, count=2)
    speaker = Speaker(model, words, [finals()] * 2, capacity=8)
    first = int(rows.first[0].nonzero()[0, 0])
    speaker.observe(rows.between(0, first - 1), [spread(2, r) for r in range(first - 1)])
    with pytest.raises(ValueError, match="time order"):                   # not after the last observed row
        speaker.observe(rows.between(first - 2, first - 1), [spread(2, first - 2)])
    speaker.observe(rows.between(first - 1, first), [spread(2, first - 1)])
    with pytest.raises(ValueError, match="time order"):                   # the first row said: after it too
        speaker.speak(rows.between(first - 1, first)._replace(first=rows.between(first, first + 1).first),
                      spread(2, first - 1), numbers(1, 2, 1)[0])
    speaker.speak(rows.between(first, first + 1), spread(2, first), numbers(1, 2, 1)[0])
    with pytest.raises(ValueError, match="observed before it"):
        speaker.observe(rows.between(first + 1, first + 2)._replace(first=torch.zeros_like(rows.first[:, :1])),
                        [spread(2, first + 1)])


def test_the_speaker_keeps_the_row_whose_runway_word_ends_g(words):
    """D64 through the speaker: a row said under G whose runway word (a candidate) ends it, below the entry height before
    the join, is kept as the masks' state — the passage below the entry height starts at that row."""
    model, rows = setup(words, seed=13, count=1)
    speaker = Speaker(model, words, [finals()], capacity=8)
    first = int(rows.first[0].nonzero()[0, 0])
    speaker.observe(rows.between(0, first), [position(1, r) for r in range(first)])
    speaker._heard[0].state = InForce(runway=0, go_around=True, heading=0, altitude=words.altitude_index(900.0),
                                     angle=words.angle_index(3.0), speed=words.speed_index(75.0))
    low = finals()[0].entry_m - 100.0
    at = Position(np.array([-20_000.0]), np.array([0.0]), np.array([low]))
    runway_words = column_words(RUNWAY, words, 2)
    said = speaker.speak(rows.between(first + 1, first + 2), at, numbers(1, 1, 1)[0], {RUNWAY: (runway_words == 0)[None]})
    assert said[0, RUNWAY] == 0 and not speaker.in_force[0].go_around
    assert speaker.procedure.dipped[0, 0] and not speaker.procedure.cleared[0]


def test_the_speaker_refuses_a_row_marked_first_otherwise_and_positions_not_one_for_each_row(words):
    """B12 (§7 item 3): `observe` refuses positions that are not one for each row and each aircraft; `speak` refuses a
    row whose mark of the first predicted step is not, for an aircraft, whether nothing is in force yet (a later row
    marked first would draw without "unchanged" while its record permits it). Both before any change."""
    import torch

    model, rows = setup(words, seed=14, count=2)
    speaker = Speaker(model, words, [finals()] * 2, capacity=8)
    first = int(rows.first[0].nonzero()[0, 0])
    with pytest.raises(ValueError, match="one for each of the"):
        speaker.observe(rows.between(0, first), [position(2, r) for r in range(first - 1)])
    with pytest.raises(ValueError, match="one for each of the"):
        speaker.observe(rows.between(0, first), [position(1, r) for r in range(first)])
    speaker.observe(rows.between(0, first), [position(2, r) for r in range(first)])      # the refusals changed nothing
    unmarked = rows.between(first, first + 1)._replace(first=torch.zeros_like(rows.first[:, :1]))
    with pytest.raises(ValueError, match="mark of the first predicted step"):
        speaker.speak(unmarked, position(2, first), numbers(1, 2, 1)[0])
    assert not speaker.permitted_rows and all(state is None for state in speaker.in_force)
    speaker.speak(rows.between(first, first + 1), position(2, first), numbers(1, 2, 1)[0])
    marked = rows.between(first + 1, first + 2)._replace(first=rows.between(first, first + 1).first)
    with pytest.raises(ValueError, match=r"aircraft \[0, 1\]: the row's mark of the first predicted step"):
        speaker.speak(marked, position(2, first + 1), numbers(1, 2, 1)[0])
    assert len(speaker.permitted_rows) == 1
    speaker.speak(rows.between(first + 1, first + 2), position(2, first + 1), numbers(1, 2, 1)[0])


def test_a_row_the_caller_refuses_leaves_the_speaker_as_it_was(words):
    """B12 (§7 item 7, vocabulary D80): when the caller's last step (`accept`: a closed loop's executor) refuses the
    row's words, the speaker keeps nothing of it; the row said again then says what a speaker never refused says."""
    model, rows = setup(words, seed=15, count=2)
    first = int(rows.first[0].nonzero()[0, 0])

    def said(refuse):
        speaker = Speaker(model, words, [finals()] * 2, capacity=8)
        speaker.observe(rows.between(0, first), [position(2, r) for r in range(first)])
        out = []
        for r in range(first, first + 4):
            if refuse and r == first + 1:
                def no(words_row):
                    raise ValueError("the executor refuses the row")
                with pytest.raises(ValueError, match="executor refuses"):
                    speaker.speak(rows.between(r, r + 1), position(2, r), numbers(3, 2, 4)[r - first], accept=no)
                assert len(speaker.permitted_rows) == 1 and len(speaker.drawn_probability) == 1
            out.append(speaker.speak(rows.between(r, r + 1), position(2, r), numbers(3, 2, 4)[r - first]))
        return np.stack(out), np.stack(speaker.drawn_probability), speaker.go_arounds

    a, b = said(False), said(True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def test_a_change_of_runway_outside_a_go_around_keeps_each_candidate_s_join_and_passage(words):
    """D122: the masks keep, for each candidate, its join and its passage below its entry height before the join from
    row 0 or the last go-around, whatever runway was in force at those rows — a change of runway outside a go-around
    starts neither again (the new runway has the rows flown under the one before); a go-around does."""
    finals_b = finals()
    masks = ProcedureMasks([finals_b], words)
    far = np.array([-40_000.0]), np.array([0.0])
    low = np.array([masks.entry_low[0, 1] - 50.0])                   # under candidate 1's entry height, not joined
    masks.track(*far, low, np.array([False]))                       # a row flown under runway 0 in force
    masks.after_row(*far, low, np.array([False]))
    assert masks.dipped[0, 1] and not masks.joined[0, 1]
    climb = 1 + words.angle_climb
    # the runway changes to candidate 1 (no go-around): its passage, made under runway 0, bars the climb
    assert not masks.permitted(ANGLE, np.array([1]), np.array([False]), *far, low)[0, climb]
    assert ProcedureMasks([finals_b], words).permitted(ANGLE, np.array([1]), np.array([False]), *far, low)[0, climb]
    grid = [(e, n) for e in np.arange(-20_000.0, 0.0, 250.0) for n in np.arange(-3_000.0, 3_000.0, 100.0)
            if finals_b[1].inside(np.array(e), np.array(n)) and not finals_b[0].inside(np.array(e), np.array(n))]
    e, n = (np.array([value]) for value in grid[0])                  # inside candidate 1's region alone
    masks.track(e, n, low, np.array([False]))
    masks.after_row(e, n, low, np.array([False]))
    assert masks.joined[0, 1] and not masks.joined[0, 0]             # joined under runway 0 in force: kept for 1
    masks.track(*far, low, np.array([True]))                        # a go-around starts both again
    assert not masks.joined.any() and not masks.dipped.any()
