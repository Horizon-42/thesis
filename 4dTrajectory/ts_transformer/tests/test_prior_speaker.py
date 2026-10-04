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
    model = Prior(PriorConfig.from_words(words, "full", **SMALL))
    rng = np.random.default_rng(seed)
    sentences = [prior_sentence(rng, words, candidates=2, rows=rows, first_step=first_step) for _ in range(count)]
    return model, collate(sentences, CPU)


def position(count, row):
    return Position(np.full(count, -15_000.0 + 140.0 * row), np.zeros(count), np.full(count, 700.0 - 4.0 * row))


def speak(model, rows, words, *, seed, caller=None, count_rows=20):
    count = rows.present.shape[0]
    speaker = Speaker(model, words, [finals()] * count, capacity=rows.present.shape[1],
                      generator=torch.Generator().manual_seed(seed))
    first = int(rows.first[0].nonzero()[0, 0])
    speaker.observe(rows.between(0, first), [position(count, r) for r in range(first)])
    said = [speaker.speak(rows.between(r, r + 1), position(count, r), caller) for r in range(first, first + count_rows)]
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
    import json

    from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
    from trajectory_data_process.harvest.airports import load_airport
    from ts_transformer.instructions.airport import airport_geometry
    from ts_transformer.prior.procedure import airport_finals, procedure_digests
    from ts_transformer.repo_layout import HARVEST_ROOT

    manifest = HARVEST_ROOT / "KRDU" / "arrivals" / "manifest.json"
    geometry = airport_geometry("KRDU", json.loads(manifest.read_text(encoding="utf-8"))["runway_targets"],
                                load_airport("KRDU", config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP).runways)
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
    speaker = Speaker(model, words, [finals()], capacity=rows.present.shape[1],
                      generator=torch.Generator().manual_seed(0))
    final = finals()[0]
    edge = float(final.glidepath_m(np.array(5_000.0))) - GLIDEPATH_BELOW_M
    at = Position(np.array([-5_000.0]), np.array([0.0]), np.array([edge + 60.0]))
    speaker.procedure.track(at.e_m, at.n_m, at.height_m, np.array([False]))
    speaker.heard[0].state = InForce(runway=0, go_around=False, heading=0, altitude=words.altitude_index(900.0),
                                     angle=words.angle_index(3.0), speed=words.speed_index(75.0))
    between = [v for v in range(words.n_altitude_levels)
               if final.decision_m < words.altitude_level_m(v) < edge - words.altitude_tolerance_m(v)]
    assert between
    altitude = np.zeros((1, 2 + words.n_altitude_levels), dtype=bool)
    altitude[0, [1 + v for v in between]] = True
    caller = {ALTITUDE: altitude}
    mask = speaker._allowed(RUNWAY, np.zeros((1, 0), dtype=np.int64), at, caller)[0]
    runway_words = column_words(RUNWAY, words, 2)
    assert mask[list(runway_words).index(1)] and not mask[list(runway_words).index(UNCHANGED)]
    for w, word in enumerate(runway_words):
        runway, go_around = speaker._runway_after(np.array([word]))
        others = {c: speaker._others(c, runway, go_around, at, caller)[0] for c in range(1, len(COLUMNS))}
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
    speaker = Speaker(model, words, [finals()] * count, capacity=rows.present.shape[1],
                      generator=torch.Generator().manual_seed(4))
    first = int(rows.first[0].nonzero()[0, 0])
    speaker.observe(rows.between(0, first), [position(count, r) for r in range(first)])
    no_unchanged = np.ones((count, 4), dtype=bool)
    no_unchanged[:, 0] = False
    for r in range(first, first + 12):
        caller = {RUNWAY: no_unchanged & go_around_bound(speaker.go_arounds, words, 2)}
        said = speaker.speak(rows.between(r, r + 1), position(count, r), caller)
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
