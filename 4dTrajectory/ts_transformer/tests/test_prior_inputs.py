"""The prior's inputs of a row (prior design §2, §7 item 2; milestone B1): the landings before a step."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from ts_transformer.instructions.airport import AirportGeometry, curvature_radius_m
from ts_transformer.instructions.artefact import STATE_COLUMNS, signals_flights
from ts_transformer.instructions.words import HEADING, RUNWAY, Words
from ts_transformer.prior.batch import CANDIDATE_FEATURES, OWN_FEATURES
from ts_transformer.prior.inputs import (
    GLIDEPATH_HEIGHT_SCALE_M, SPEED_SCALE_MPS, VERTICAL_RATE_SCALE_MPS, sentence_rows, since_input,
)
from ts_transformer.prior.landings import (
    LANDINGS_WINDOW_S, Landing, LandingIndex, read_roster_landings, roster_landings, utc_s,
)
from ts_transformer.prior.source import ArtefactSource, artefact_identity
from ts_transformer.tests.support import (
    INSTRUCTION_AIRPORT_ELEVATION_M, TEST_GLIDEPATH_DEG, TEST_TCH_M, fixture_days, fly_legs, instruction_flight,
    instruction_spec, parallel_airport, prior_artefact, prior_closed_loop_sentence,
)


def record(key, runway, time_utc, outcome="assigned"):
    return {"flight_key": key, "outcome": outcome, "runway": runway, "landing_time_utc": time_utc}


def test_the_landings_index_counts_the_30_min_before_a_step_without_the_test_days(tmp_path):
    days = fixture_days()
    train_day, test_day = days.days["train"][0], days.days["test"][0]
    records = [
        record("A", "05L", f"{train_day}T12:00:00Z"),
        record("B", "05L", f"{train_day}T12:20:00Z"),
        record("C", "23R", f"{train_day}T12:25:00Z"),
        record("D", "05L", f"{test_day}T12:00:00Z"),            # a sealed test day: never counted
        record("E", None, f"{train_day}T12:10:00Z", outcome="not_landing"),
        record("F", "32", f"{train_day}T12:10:00Z"),             # not a candidate
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"records": records}))
    index = read_roster_landings(path, ("05L", "23R"), days)
    assert [landing.flight_key for landing in index.landings] == ["A", "B", "C"] and index.sealed == 1
    noon = utc_s(f"{train_day}T12:00:00Z")
    times = np.array([noon, noon + 1.0, noon + 25 * 60.0, noon + LANDINGS_WINDOW_S, noon + LANDINGS_WINDOW_S + 1.0])
    # [t − 30 min, t): a landing at t is not before it; one exactly 30 min before is
    assert index.counts_before(times, without="C").tolist() == [[0, 0], [1, 0], [2, 0], [2, 0], [1, 0]]
    assert index.counts_before(times, without="B").tolist() == [[0, 0], [1, 0], [1, 0], [1, 1], [0, 1]]
    with pytest.raises(ValueError, match="does not land in the index"):
        index.counts_before(times, without="D")
    assert index.digest() != roster_landings(records[:2], ("05L", "23R"), days).digest()


def test_the_landings_index_refuses_a_day_outside_the_split():
    with pytest.raises(KeyError, match="not in this day split"):
        roster_landings([record("A", "05L", "2031-01-01T12:00:00Z")], ("05L",), fixture_days())


# ---- the inputs of a sentence's rows (B1)

INTERVALS = (2.0, 4.0, 8.0)


def own(rows, name):
    return rows.own[:, OWN_FEATURES.index(name)]


def candidate(rows, k, name):
    return rows.candidates[:, k, CANDIDATE_FEATURES.index(name)]


def flight_and_index(split="train", i=1):
    words = Words(instruction_spec())
    legs = [(160, 0.0, 75.0 - 2.0 * i, -1.5)]
    signals = instruction_flight(*fly_legs(legs, 90.0, 750.0 + 30.0 * i, -400.0, 0.0), dataset_id=f"KXXX:F{split}{i}",
                                 split=split)
    record = {"dataset_id": signals.dataset_id, "airport": "KXXX", "runway": "09",
              "entry_time_utc": signals.entry_time_utc, "landing_time_utc": signals.landing_time_utc}
    index = roster_landings([{"flight_key": f"F{split}{i}", "outcome": "assigned", "runway": "09",
                              "landing_time_utc": signals.landing_time_utc}], ("09", "09L"), fixture_days())
    return words, signals, record, index


def rows_of(sentence, record, index, words, interval_s, geometry=None, variant="full"):
    return sentence_rows(sentence, record, geometry or parallel_airport(), index, words, interval_s=interval_s,
                         split="train", variant=variant)


@pytest.mark.parametrize("interval_s", INTERVALS)
def test_the_motion_comes_from_the_2_s_displacement_and_row_0_has_none(interval_s):
    words, signals, record, index = flight_and_index()
    every = int(interval_s / 2)
    for first_row in (0, 1):            # D60: also when the data have a 2 s row before row 0
        sentence = prior_closed_loop_sentence(signals, words, interval_s=interval_s, first_row=first_row)
        rows = rows_of(sentence, record, index, words, interval_s)
        assert own(rows, "no_motion")[0] == 1.0 and not own(rows, "no_motion")[1:].any()
        assert own(rows, "ground_speed")[0] == 0.0 and own(rows, "vertical_rate")[0] == 0.0
        assert not rows.candidates[0, :, [CANDIDATE_FEATURES.index("motion_minus_course_sin"),
                                          CANDIDATE_FEATURES.index("motion_minus_course_cos")]].any()
        e, n, h = (sentence.states[:, STATE_COLUMNS.index(name)] for name in ("e_m", "n_m", "height_m"))
        for j in (1, 5, rows.rows - 1):
            r = j * every
            speed = np.hypot(e[r] - e[r - 1], n[r] - n[r - 1]) / 2.0
            assert own(rows, "ground_speed")[j] == pytest.approx(speed / SPEED_SCALE_MPS, rel=1e-6)
            assert own(rows, "vertical_rate")[j] == pytest.approx((h[r] - h[r - 1]) / 2.0 / VERTICAL_RATE_SCALE_MPS,
                                                                  rel=1e-5)
            # course 090°, the flight flies 090°: motion minus course is 0
            assert candidate(rows, 0, "motion_minus_course_cos")[j] == pytest.approx(1.0)


def test_only_positions_and_heights_give_the_motion():
    """The stored track, ground speed and vertical rate (on observed rows, a fit with 7.5 s of the future) change no
    input."""
    words, signals, record, index = flight_and_index()
    sentence = prior_closed_loop_sentence(signals, words, interval_s=2.0)
    states = sentence.states.copy()
    for name in ("track_deg", "ground_speed_mps", "vertical_rate_mps"):
        states[:, STATE_COLUMNS.index(name)] += 17.0
    a, b = (rows_of(s, record, index, words, 2.0) for s in (sentence, replace(sentence, states=states)))
    for name in ("own", "candidates", "heading_in_force", "since"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


@pytest.mark.parametrize("interval_s", INTERVALS)
def test_a_change_of_the_runway_word_leaves_the_rows_up_to_the_first_predicted_step_bit_for_bit(interval_s):
    """D23: the artefact writes the landed runway at the first predicted step; no input up to that step reads it."""
    words, signals, record, index = flight_and_index()
    sentence = prior_closed_loop_sentence(signals, words, interval_s=interval_s, go_around=True)
    grid = sentence.grid.copy()
    grid[0, RUNWAY], grid[20, RUNWAY] = 1, 1
    a, b = (rows_of(s, record, index, words, interval_s) for s in (sentence, replace(sentence, grid=grid)))
    upto = slice(0, a.first_step + 1)
    for name in ("time_s", "own", "candidates", "runway_in_force", "go_around", "heading_in_force", "words_in_force",
                 "since"):
        assert np.array_equal(getattr(a, name)[upto], getattr(b, name)[upto]), name
    assert a.runway_in_force[a.first_step + 1] == 0 and b.runway_in_force[b.first_step + 1] == 1


def test_the_height_above_the_glidepath_against_a_hand_computation():
    words, signals, record, index = flight_and_index()
    sentence = prior_closed_loop_sentence(signals, words, interval_s=2.0)
    rows = rows_of(sentence, record, index, words, 2.0)
    j = 20
    e, h = sentence.states[j, STATE_COLUMNS.index("e_m")], sentence.states[j, STATE_COLUMNS.index("height_m")]
    d = -e                                                     # "09": threshold at the origin, course 090°
    radius = curvature_radius_m(35.0, 90.0)
    glidepath = TEST_TCH_M + d * np.tan(np.radians(TEST_GLIDEPATH_DEG)) + d * d / (2 * radius)
    expected = ((h - 100.0) - glidepath) / GLIDEPATH_HEIGHT_SCALE_M
    assert candidate(rows, 0, "height_above_glidepath")[j] == pytest.approx(expected, rel=1e-5)
    assert own(rows, "height_above_elevation")[j] == pytest.approx((h - INSTRUCTION_AIRPORT_ELEVATION_M) / 1000.0)


def test_the_rows_from_the_first_predicted_step_are_the_flown_states():
    words, signals, record, index = flight_and_index()
    sentence = prior_closed_loop_sentence(signals, words, interval_s=4.0)
    states = sentence.states.copy()
    flown = int(np.flatnonzero(sentence.on_interval)[sentence.start])
    states[flown:, STATE_COLUMNS.index("height_m")] += 50.0
    a, b = (rows_of(s, record, index, words, 4.0) for s in (sentence, replace(sentence, states=states)))
    first = a.first_step
    assert np.array_equal(a.own[:first], b.own[:first])
    heights = own(b, "height_above_elevation")[first:] - own(a, "height_above_elevation")[first:]
    assert heights == pytest.approx(np.full(len(heights), 0.05))


def test_the_words_in_force_follow_the_sentence_through_a_go_around():
    words, signals, record, index = flight_and_index()
    sentence = prior_closed_loop_sentence(signals, words, interval_s=2.0, go_around=True)
    rows = rows_of(sentence, record, index, words, 2.0)
    s = rows.first_step
    assert (rows.runway_in_force[: s + 1] == -1).all() and (rows.runway_in_force[s + 1:] == 0).all()
    assert not rows.go_around[: s + 13].any() and rows.go_around[s + 13: s + 21].all() and not rows.go_around[s + 21:].any()
    # the heading word of row s + 5 (5° right of the course) is in force from the row after it
    assert rows.heading_in_force[s + 5] == pytest.approx([0.0, 1.0])
    assert rows.heading_in_force[s + 6] == pytest.approx([np.sin(np.radians(5.0)), np.cos(np.radians(5.0))], abs=1e-6)
    assert rows.words_in_force[s + 13, 1] == words.angle_climb
    # the time since each word, in seconds from the row it was said at; the runway's since the candidate, not the
    # go-around
    assert rows.since[s + 7, HEADING] == pytest.approx(since_input(np.array([4.0]))[0])
    assert rows.since[s + 15, RUNWAY] == pytest.approx(since_input(np.array([30.0]))[0])
    assert rows.since[s + 22, RUNWAY] == pytest.approx(since_input(np.array([4.0]))[0])
    assert not rows.since[: s + 1].any()


def permuted_airport(geometry, order):
    data = geometry.to_dict()
    data["candidates"] = [data["candidates"][k] for k in order]
    return AirportGeometry.from_dict(data)


def test_a_permutation_of_the_candidates_permutes_their_vectors_and_nothing_else():
    words, signals, record, index = flight_and_index()
    sentence = prior_closed_loop_sentence(signals, words, interval_s=2.0)
    swapped = sentence.grid.copy()
    swapped[0, RUNWAY] = 1                                     # "09" is candidate 1 of the permuted airport
    a = rows_of(sentence, record, index, words, 2.0)
    b = rows_of(replace(sentence, grid=swapped), record, index, words, 2.0,
                geometry=permuted_airport(parallel_airport(), [1, 0]))
    assert np.array_equal(b.candidates, a.candidates[:, [1, 0]])
    for name in ("own", "heading_in_force", "words_in_force", "since", "go_around", "time_s"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert (b.runway_in_force[a.first_step + 1:] == 1).all()


def test_an_airport_with_more_candidates_than_any_training_airport_is_read():
    words, signals, record, _ = flight_and_index()
    data = parallel_airport().to_dict()
    extra = [{**data["candidates"][0], "ident": f"X{k}", "threshold_n_m": 2000.0 + 900.0 * k} for k in range(7)]
    data["candidates"] += extra
    data["runway_ends"] += [{key: c[key] for key in ("ident", "threshold_e_m", "threshold_n_m", "course_deg")}
                            for c in extra]
    geometry = AirportGeometry.from_dict(data)
    index = roster_landings([{"flight_key": "Ftrain1", "outcome": "assigned", "runway": "09",
                              "landing_time_utc": record["landing_time_utc"]}],
                            [c.ident for c in geometry.candidates], fixture_days())
    rows = rows_of(prior_closed_loop_sentence(signals, words, interval_s=2.0), record, index, words, 2.0, geometry)
    assert rows.candidates.shape[1] == 9


def test_the_constants_variant_adds_the_length_and_the_threshold_elevation():
    words, signals, record, index = flight_and_index()
    sentence = prior_closed_loop_sentence(signals, words, interval_s=2.0)
    full, constants = (rows_of(sentence, record, index, words, 2.0, variant=v) for v in ("full", "constants"))
    assert np.array_equal(constants.candidates[..., : full.candidates.shape[2]], full.candidates)
    assert constants.candidates[3, 0, -2:] == pytest.approx([3.0, 0.1])


def test_a_flight_never_counts_its_own_landing():
    """Its own landing is the answer (a closed-loop sentence can fly past the observed landing time): here it is put 20 s
    after the entry, inside the rows, and still never counted; another flight's landing on "09L" is."""
    words, signals, record, _ = flight_and_index()
    sentence = prior_closed_loop_sentence(signals, words, interval_s=2.0)
    entry = utc_s(record["entry_time_utc"])
    index = LandingIndex(("09", "09L"), (Landing(entry + 10.0, "09L", "OTHER"), Landing(entry + 20.0, "09", "Ftrain1")),
                         0)
    rows = rows_of(sentence, record, index, words, 2.0)
    assert not candidate(rows, 0, "landings_30min").any()
    counted = candidate(rows, 1, "landings_30min")
    assert not counted[:6].any() and (counted[6:] == 0.1).all()      # row 5 is at 10 s: [t − 30 min, t) excludes it
    with pytest.raises(ValueError, match="does not land in the index"):
        rows_of(sentence, record, LandingIndex(("09", "09L"), (), 0), words, 2.0)


def test_the_artefact_source_reads_every_sentence_and_its_identity(tmp_path):
    directory = tmp_path / "artefact"
    words, records = prior_artefact(directory, interval_s=4.0)
    days = fixture_days()
    source = ArtefactSource(directory, 4.0, "full", {"KXXX": roster_landings(records, ("09", "09L"), days)})
    for split in ("train", "select", "val"):
        sentences = source.sentences(split, "KXXX")
        assert [s.flight_key for s in sentences] == [f["dataset_id"] for f in signals_flights(directory, split)]
        assert {s.split for s in sentences} == {split} and sentences[1].go_around.any()
    with pytest.raises(ValueError, match="development splits"):
        source.sentences("test", "KXXX")
    identity = artefact_identity(directory, 4.0, source.landings)
    assert identity["spec_sha256"] == words.spec.sha256 and set(identity["sentence_files"]) == {"train", "select", "val"}
    assert identity["candidates"]["airports"]["KXXX"]["candidates"][0]["vertical_path"]["crossing_height_m"] == TEST_TCH_M
    with pytest.raises(ValueError, match="landings of"):
        ArtefactSource(directory, 4.0, "full", {})


def test_in_a_turn_the_motion_is_the_2_s_displacement_not_the_rows():
    """At Δ = 8 s a row's motion is still the displacement in the 2 s before it (D25): in a 3°/2 s turn its direction is
    the track of now, not the chord since the Δ row before."""
    words = Words(instruction_spec())
    signals = instruction_flight(*fly_legs([(160, 3.0, 75.0, -1.5)], 90.0, 750.0, -400.0, 0.0),
                                 dataset_id="KXXX:Ftrain1", split="train")
    _, _, record, index = flight_and_index()
    sentence = prior_closed_loop_sentence(signals, words, interval_s=8.0)
    rows = rows_of(sentence, record, index, words, 8.0)
    e, n = (sentence.states[:, STATE_COLUMNS.index(name)] for name in ("e_m", "n_m"))
    for j in (2, 7):
        r = 4 * j
        now = np.degrees(np.arctan2(e[r] - e[r - 1], n[r] - n[r - 1]))
        chord = np.degrees(np.arctan2(e[r] - e[r - 4], n[r] - n[r - 4]))
        assert abs(now - chord) > 3.0
        assert candidate(rows, 0, "motion_minus_course_sin")[j] == pytest.approx(np.sin(np.radians(now - 90.0)),
                                                                                abs=1e-6)


def crossing_airport():
    """`parallel_airport` with its second candidate "18" on another course (180°), 3 km north."""
    data = parallel_airport().to_dict()
    second = {**data["candidates"][1], "ident": "18", "threshold_e_m": 0.0, "threshold_n_m": 3_000.0,
              "course_deg": 180.0}
    data["candidates"][1] = second
    data["runway_ends"][1] = {key: second[key] for key in ("ident", "threshold_e_m", "threshold_n_m", "course_deg")}
    return AirportGeometry.from_dict(data)


def test_a_heading_word_keeps_the_track_it_was_heard_with_and_reads_against_the_new_runway():
    """D46: the heading word said under "09" (course 090°, +5°: track 095°) keeps its track when the runway changes to
    "18" (course 180°): the input becomes 095° − 180° = −85°; a heading word heard under "18" says a track from 180°."""
    words, signals, record, _ = flight_and_index()
    geometry = crossing_airport()
    index = roster_landings([{"flight_key": "Ftrain1", "outcome": "assigned", "runway": "09",
                              "landing_time_utc": record["landing_time_utc"]}], ("09", "18"), fixture_days())
    sentence = prior_closed_loop_sentence(signals, words, interval_s=2.0)
    grid = sentence.grid.copy()
    grid[10, RUNWAY] = 1
    grid[15, HEADING] = words.heading_index(-10.0)
    rows = rows_of(replace(sentence, grid=grid), record, index, words, 2.0, geometry)
    s = rows.first_step

    def angle(deg):
        return [np.sin(np.radians(deg)), np.cos(np.radians(deg))]

    assert rows.heading_in_force[s + 10] == pytest.approx(angle(5.0), abs=1e-6)
    assert rows.heading_in_force[s + 11] == pytest.approx(angle(-85.0), abs=1e-6)
    assert rows.heading_in_force[s + 16] == pytest.approx(angle(-10.0), abs=1e-6)


def test_the_rows_utc_time_is_the_flights_entry_and_its_2_s_rows():
    """The landings window is read at each Δ row's UTC time: the entry plus 2 s for each signals row, the sentence's
    first row included — at Δ = 4 s from signals row 1, Δ row j is signals row 1 + 2j."""
    words, signals, record, _ = flight_and_index()
    sentence = prior_closed_loop_sentence(signals, words, interval_s=4.0, first_row=1)
    entry = utc_s(record["entry_time_utc"])
    index = LandingIndex(("09", "09L"), (Landing(entry + 13.0, "09L", "OTHER"),
                                         Landing(entry + 3_600.0, "09", "Ftrain1")), 0)
    counted = candidate(rows_of(sentence, record, index, words, 4.0), 1, "landings_30min")
    # Δ row 3 is signals row 7, at entry + 14 s: the first after the landing at 13 s
    assert not counted[:3].any() and (counted[3:] == 0.1).all()


def test_the_flight_key_mirror_reads_the_dataset_id_as_the_data_plane_writes_it():
    from flight_scenarios.identity import flight_key
    from ts_transformer.data.dataset import dataset_flight_key
    from ts_transformer.prior.inputs import own_flight_key

    source = {"arr_airport": "KRDU", "id": "AAL1007", "runway": "05L", "icao24": "abe2a3",
              "landing_time_utc": "2026-05-18T04:17:18Z"}
    assert own_flight_key({"dataset_id": dataset_flight_key(source, 0), "airport": "KRDU"}) == flight_key(source, 0)
    with pytest.raises(ValueError, match="flight key"):
        own_flight_key({"dataset_id": dataset_flight_key(source, 0), "airport": "KSJC"})


def test_the_landings_digest_holds_each_landings_flight_runway_and_time_only():
    """D63: a change of a landing's time or runway in the roster changes the digest; a change of another field of the
    roster does not; the landings left out on the sealed test days count."""
    days = fixture_days()
    train_day, test_day = days.days["train"][0], days.days["test"][0]
    records = [{**record("A", "05L", f"{train_day}T12:00:00Z"), "icao24": "abc123", "file": "a.json"},
               {**record("B", "23R", f"{train_day}T12:20:00Z"), "icao24": "def456", "file": "b.json"},
               record("C", "05L", f"{test_day}T12:00:00Z")]

    def digest(rows):
        return roster_landings(rows, ("05L", "23R"), days).digest()

    base = digest(records)
    assert digest([{**records[0], "icao24": "zzz999", "file": "elsewhere.json"}, *records[1:]]) == base
    assert digest([{**records[0], "landing_time_utc": f"{train_day}T12:00:02Z"}, *records[1:]]) != base
    assert digest([{**records[0], "runway": "23R"}, *records[1:]]) != base
    assert digest(records[:2]) != base                                    # one sealed landing fewer


def test_a_run_refuses_landings_whose_digest_differs_from_the_identity(tmp_path):
    """D63: the identity is computed again from the landings read again; a checkpoint of other landings is refused."""
    from ts_transformer.prior.checkpoint import load_checkpoint, save_checkpoint
    from ts_transformer.prior.model import Prior, PriorConfig

    directory = tmp_path / "artefact"
    words, records = prior_artefact(directory, interval_s=2.0)
    days = fixture_days()
    landings = {"KXXX": roster_landings(records, ("09", "09L"), days)}
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64))
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, model.state_dict(), identity=artefact_identity(directory, 2.0, landings),
                    run={"airports": ["KXXX"], "held_out": None}, train_config={})
    load_checkpoint(path, artefact_identity(directory, 2.0, landings))
    moved = [{**records[0], "landing_time_utc": records[0]["landing_time_utc"].replace(":00Z", ":04Z")}, *records[1:]]
    with pytest.raises(ValueError, match=r"\['landings'\] differ"):
        load_checkpoint(path, artefact_identity(directory, 2.0, {"KXXX": roster_landings(moved, ("09", "09L"), days)}))
