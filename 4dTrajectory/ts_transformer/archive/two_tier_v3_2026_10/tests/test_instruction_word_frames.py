"""The word-frame read (`experiments/instruction_word_frames`, R48): the divergence, the runway frame's re-indexing, the
intercept heading, the grid offsets, and the runner on a tmp artefact (never a live one)."""

import json
import math
from collections import Counter

import numpy as np
import pytest

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.words import ALTITUDE, HEADING, RUNWAY, UNCHANGED, Words


def _airport(code, course_deg, elevation_m):
    return AirportGeometry.from_dict({
        "code": code, "reference": {"lat": 35.0, "lon": -78.0, "elevation_m": elevation_m},
        "candidates": [{"ident": "R1", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": course_deg,
                        "elevation_m": elevation_m, "length_m": 3000.0}],
        "runway_ends": [{"ident": "R1", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": course_deg}],
    })


def _sentence(heading, levels, join_row):
    """Rows of a sentence: the runway at row 0, ``heading`` words and ``levels`` (MSL classes) from row 0 one a row."""
    rows = np.full((max(len(heading), len(levels)) + 1, 6), UNCHANGED, dtype=np.int16)
    rows[0, RUNWAY] = 0
    rows[: len(heading), HEADING] = heading
    rows[: len(levels), ALTITUDE] = levels
    return rows, join_row


def _sentences(items):
    """The artefact's sentence arrays for ``items`` = [(rows, join_row)], one flight each in order."""
    offsets = np.cumsum([0] + [len(rows) for rows, _ in items])
    return {"words": np.concatenate([rows for rows, _ in items]), "offsets": offsets,
            "signal_index": np.arange(len(items)), "runway_index": np.zeros(len(items), dtype=int),
            "join_row": np.array([join for _, join in items])}


def test_the_divergence_is_zero_for_one_distribution_and_one_bit_for_disjoint_ones():
    from ts_transformer.experiments.instruction_word_frames import js_divergence_bits

    assert js_divergence_bits({0: 2, 1: 6}, {0: 1, 1: 3}) == pytest.approx(0.0)
    assert js_divergence_bits({0: 5}, {1: 5}) == pytest.approx(1.0)
    expected = 0.5 * (0.5 * math.log2(0.5 / 0.75) + 0.5 * math.log2(0.5 / 0.25)) + 0.5 * math.log2(1 / 0.75)
    assert js_divergence_bits({0: 1, 1: 1}, {0: 1}) == pytest.approx(expected)
    assert js_divergence_bits({0: 1, 1: 1}, {0: 1}) == pytest.approx(js_divergence_bits({0: 1}, {0: 1, 1: 1}))


def test_the_runway_frame_shifts_by_the_course_rounded_to_the_grid():
    from ts_transformer.experiments.instruction_word_frames import course_shift

    assert course_shift(225.0, 5.0, 72) == 45
    assert course_shift(122.3, 5.0, 72) == 24          # 120°
    assert course_shift(0.75, 5.0, 72) == 0
    assert course_shift(358.0, 5.0, 72) == 0           # 360° wraps


def test_the_intercept_heading_is_the_last_word_said_at_or_before_the_clearance():
    from ts_transformer.experiments.instruction_word_frames import intercept_heading

    column = np.array([10, UNCHANGED, UNCHANGED, 12, UNCHANGED, 14, UNCHANGED])
    assert intercept_heading(column, 4) == 12
    assert intercept_heading(column, 5) == 14           # said at the clearance row itself
    assert intercept_heading(column, 0) is None         # cleared at row 0


def test_one_manoeuvre_at_two_airports_differs_only_in_the_absolute_frame():
    from ts_transformer.experiments.instruction_word_frames import compare, count_words
    from ts_transformer.tests.support import instruction_spec

    words = Words(instruction_spec())
    geometries = {"KAAA": _airport("KAAA", 90.0, 0.0), "KBBB": _airport("KBBB", 180.0, 150.0)}
    flights = [{"dataset_id": "KAAA:a", "airport": "KAAA", "runway": "R1"},
               {"dataset_id": "KBBB:b", "airport": "KBBB", "runway": "R1"}]
    # downwind (+180), base (+90), final (0) relative to each course; the same MSL levels at both
    relative = np.array([36, 18, 0])
    items = [_sentence((relative + 18) % 72, [40, 30], 1), _sentence((relative + 36) % 72, [40, 30], 1)]
    counts = count_words(_sentences(items), flights, geometries, words)
    heading = {frame: compare(counts["heading"][frame]) for frame in ("absolute", "runway")}
    assert heading["runway"]["mean_pairwise_js_bits"] == pytest.approx(0.0)
    assert heading["absolute"]["mean_pairwise_js_bits"] > 0.0
    intercept = {frame: compare(counts["intercept_heading"][frame]) for frame in ("absolute", "runway")}
    assert counts["intercept_heading"]["runway"]["KAAA"] == Counter({18: 1})    # the base word, said at row 1
    assert intercept["runway"]["mean_pairwise_js_bits"] == pytest.approx(0.0)
    assert intercept["absolute"]["mean_pairwise_js_bits"] == pytest.approx(1.0)
    level = {frame: compare(counts["level"][frame]) for frame in ("absolute", "runway")}
    assert level["absolute"]["mean_pairwise_js_bits"] == pytest.approx(0.0)
    assert counts["level"]["runway"]["KBBB"] == Counter({35: 1, 25: 1})        # 150 m = 5 levels
    assert level["runway"]["mean_pairwise_js_bits"] > 0.0
    assert level["runway"]["rare_share"] == {"KAAA": 1.0, "KBBB": 1.0}


def test_descend_to_land_is_not_a_level_and_a_sentence_off_its_runway_is_refused():
    from ts_transformer.experiments.instruction_word_frames import count_words
    from ts_transformer.tests.support import instruction_spec

    words = Words(instruction_spec())
    geometries = {"KAAA": _airport("KAAA", 90.0, 0.0)}
    items = [_sentence([18], [40, words.altitude_land], 0)]
    flight = {"dataset_id": "KAAA:a", "airport": "KAAA", "runway": "R1"}
    counts = count_words(_sentences(items), [flight], geometries, words)
    assert counts["level"]["absolute"]["KAAA"] == Counter({40: 1})
    assert not counts["intercept_heading"]["absolute"].get("KAAA")                 # cleared at row 0
    with pytest.raises(ValueError, match="not the landed"):                       # the flight landed elsewhere
        count_words(_sentences(items), [{**flight, "runway": "R2"}], geometries, words)
    rows, join = items[0]
    rows = rows.copy()
    rows[0, RUNWAY] = 1
    with pytest.raises(ValueError, match="row 0 says runway slot 1"):              # row 0 is not the recorded slot
        count_words(_sentences([(rows, join)]), [flight], geometries, words)


def test_a_course_off_the_grid_by_more_than_the_corridor_tolerance_is_flagged():
    from ts_transformer.experiments.instruction_word_frames import grid_offsets

    offsets = grid_offsets({"KAAA": _airport("KAAA", 122.3, 0.0), "KBBB": _airport("KBBB", 45.01, 0.0)}, 5.0, 2.0)
    (a,), (b,) = offsets["KAAA"], offsets["KBBB"]
    assert (a["nearest_grid_deg"], a["within_course_tolerance"]) == (120.0, False)
    assert a["offset_deg"] == pytest.approx(2.3)
    assert (b["nearest_grid_deg"], b["within_course_tolerance"]) == (45.0, True)


def test_the_runner_reads_a_tmp_artefact_and_never_overwrites(tmp_path):
    from ts_transformer.experiments import instruction_word_frames
    from ts_transformer.tests.support import labelled_instruction_artefact

    directory = tmp_path / "artefact"
    spec = labelled_instruction_artefact(directory)
    out = tmp_path / "frames"
    assert instruction_word_frames.main(["--instructions", str(directory), "--out", str(out)]) == 0
    read = json.loads((out / "word_frames.json").read_text(encoding="utf-8"))
    assert (read["schema"], read["split"], read["spec_sha256"], read["sentences"]) == (
        instruction_word_frames.SCHEMA, "train", spec.sha256, 1)
    assert read["measures"]["heading"]["runway"]["words"]["KXXX"] > 0
    assert read["measures"]["heading"]["runway"]["mean_pairwise_js_bits"] is None       # one airport, no pair
    assert read["grid_offsets"]["KXXX"][0]["within_course_tolerance"] is True
    with pytest.raises(SystemExit):
        instruction_word_frames.main(["--instructions", str(directory), "--out", str(out)])
