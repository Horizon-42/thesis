"""The Training view's sample exporter, under the BOX vocabulary (`box-v3`).

The track rebuild needs the arrival store, so it is exercised by the runner itself; what is
pinned here is everything that decides WHICH flights, WHAT A BOX IS, and WHAT SHAPE the file has
— the parts that can silently disagree with the artefact or with the frontend reader.

THE ARTEFACT'S LABELLER IS NOT IN THIS REPOSITORY, so the boxes the exporter draws are rebuilt
from the artefact's own spec. There is no original to compare them against; what stands in for one
is the containment verdict, measured on the real artefact (39 670 of 39 670 rows at five
airports). These tests pin the SHAPE of that rebuild, so a change to it is a change somebody made
on purpose.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ts_transformer.experiments.instruction_sample_export import (
    BOX_SCHEMA, INDEX_SCHEMA, INSIDE_EPSILON, KINDS, POOL_FACTOR, READING_RULE, SAMPLE_SCHEMA,
    SECTOR_ARC_MAX, SECTOR_ARC_MIN, altitude_envelope, check_columns, containment,
    cumulative_path_m, drawn_flights, hae_offset_m, load_box_vocabulary, lonlat_from_frame,
    read_signals, reachable_sector, reading_block, runway_sha256, sentences_by_flight,
    update_index, word_runs,
)

# The frontend reader's own constants (`src/data/trainingSample.ts`). Declared MIRRORS: the two
# sides are one contract, and a fixture that restated them could not catch it moving.
FRONTEND_INDEX_SCHEMA = "aeroviz-training-index-v1"
FRONTEND_SAMPLE_SCHEMA = "aeroviz-training-sample-v2"
FRONTEND_READING_RULE = "box-v3"
FRONTEND_KINDS = ("heading", "altitude", "speed", "runway", "duration", "terminal")
FRONTEND_INSIDE_EPSILON = 1e-3


def test_the_contract_is_the_frontend_reader_s():
    """Six positional columns, two schema names, one reading rule and one epsilon. If any of
    these fails, one side moved without the other — and every one of them is a silent failure:
    the file still parses, it just means something else."""
    assert KINDS == FRONTEND_KINDS
    assert INDEX_SCHEMA == FRONTEND_INDEX_SCHEMA
    assert SAMPLE_SCHEMA == FRONTEND_SAMPLE_SCHEMA
    assert READING_RULE == FRONTEND_READING_RULE
    # the verdict is computed on BOTH sides and the two are compared, so the tolerance that
    # decides a row sitting on a box's edge has to be the same number
    assert INSIDE_EPSILON == FRONTEND_INSIDE_EPSILON
    block = reading_block()
    assert block["insideEpsilon"] == INSIDE_EPSILON
    # and the block says, in the file, that the artefact's producer is not in this tree
    assert "NOT in this repository" in block["producedBy"]
    # `box-v3` stopped carrying the wedge's prose, so it moved here — to the block that says who
    # rebuilt the boxes, which is what it was describing all along. The reader requires it HERE.
    for field in ("altitudeForm", "altitudeReading"):
        assert field in block, field


# ── the artefact ─────────────────────────────────────────────────────────────

def _spec() -> dict:
    return {
        "schema": BOX_SCHEMA,
        "reading_rule": READING_RULE,
        "redundancy_fraction": 0.05,
        "heading_floor_deg": 1.0,
        "speed_low_mps": 30.0, "speed_high_mps": 250.0,
        "altitude_h0_m": 50.0, "altitude_top_m": 6000.0, "altitude_bottom_m": -150.0,
        "altitude_down_deg": 1.5, "altitude_up_deg": 1.0,
        "duration_bin_s": 2.0, "duration_max_s": 600.0,
        "course_smoothing_s": 6.0, "smoothing_s": 10.0,
    }


def _artefact(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema": BOX_SCHEMA, "spec": _spec(), "sha256": "a" * 64,
        "words": {"heading": 3, "altitude": 4, "speed": 2, "runway": 1, "duration": 301, "terminal": 3},
        "runway_idents": ["KRDU:05L"],
        "boxes": {"heading_edges_deg": [-180.0, -10.0, 10.0, 180.0],
                  "speed_edges_mps": [60.0, 90.0, 125.0],
                  "altitude_targets_m": [-20.0, 0.0, 100.0, 400.0]},
    }
    payload.update(overrides)
    path = tmp_path / "instruction_vocabulary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_an_artefact_of_another_schema_is_refused_by_name(tmp_path: Path):
    with pytest.raises(SystemExit, match="has schema"):
        load_box_vocabulary(_artefact(tmp_path, schema="something-else"))


def test_a_reading_rule_this_exporter_is_not_written_for_is_refused(tmp_path: Path):
    spec = _spec()
    spec["reading_rule"] = "segment-v14"
    with pytest.raises(SystemExit, match="written for 'box-v3'"):
        load_box_vocabulary(_artefact(tmp_path, spec=spec))


def test_an_edge_table_that_does_not_tile_its_words_is_refused(tmp_path: Path):
    """The heading and speed tables TILE their words, so they hold one more value than there are
    words; the altitude table is a LADDER OF TARGETS, one per word. Reading one as the other
    shifts every word by half a box and still draws."""
    with pytest.raises(SystemExit, match="boxes.heading_edges_deg holds"):
        load_box_vocabulary(_artefact(
            tmp_path, boxes={"heading_edges_deg": [-180.0, 0.0, 180.0],   # 3 edges, 3 words claimed
                             "speed_edges_mps": [60.0, 90.0, 125.0],
                             "altitude_targets_m": [-20.0, 0.0, 100.0, 400.0]}))


def test_the_altitude_table_is_one_target_per_word(tmp_path: Path):
    with pytest.raises(SystemExit, match="boxes.altitude_targets_m holds"):
        load_box_vocabulary(_artefact(
            tmp_path, boxes={"heading_edges_deg": [-180.0, -10.0, 10.0, 180.0],
                             "speed_edges_mps": [60.0, 90.0, 125.0],
                             "altitude_targets_m": [-20.0, 0.0, 100.0, 400.0, 900.0]}))


def test_a_good_artefact_loads_whole(tmp_path: Path):
    payload = load_box_vocabulary(_artefact(tmp_path))
    assert payload["spec"]["reading_rule"] == READING_RULE
    assert len(payload["boxes"]["heading_edges_deg"]) == payload["words"]["heading"] + 1
    assert len(payload["boxes"]["altitude_targets_m"]) == payload["words"]["altitude"]


def _sentence(words: list[list[int]], holds: list[float], runway: str = "KRDU:05L") -> dict:
    return {"F": {"dataset_id": "KRDU:F", "flight_id": "F", "runway": runway,
                  "event_times_s": [0.0, holds[0]], "hold_s": holds, "words": words}}


def test_the_columns_are_pinned_against_the_files_own_data():
    """`box-v3` stopped stating which column is which (`spec.kinds` went with the rename), so the
    order is checked against the DATA — which is the better check anyway.

    Three of the six are pinned here; heading, altitude and speed are pinned by the containment
    verdict, because a swapped pair would put nearly every row outside its box.
    """
    good = [[1, 2, 1, 0, 5, 0], [1, 2, 1, 0, 3, 1]]
    check_columns(_sentence(good, [10.0, 6.0]), ["KRDU:05L"], _spec())          # no raise

    # the runway column must hold THIS flight's runway, as an index into the class list
    with pytest.raises(SystemExit, match="the runway column does not hold"):
        check_columns(_sentence(good, [10.0, 6.0]), ["KRDU:23R", "KRDU:05L"], _spec())

    # the duration column times the bin must be the hold beside it
    with pytest.raises(SystemExit, match="is not the hold"):
        check_columns(_sentence(good, [12.0, 6.0]), ["KRDU:05L"], _spec())

    # the terminal column is `continue` up to a single `landed` at the last event
    with pytest.raises(SystemExit, match="not 'continue' up to a single 'landed'"):
        check_columns(_sentence([[1, 2, 1, 0, 5, 1], [1, 2, 1, 0, 3, 1]], [10.0, 6.0]),
                      ["KRDU:05L"], _spec())
    with pytest.raises(SystemExit, match="not 'continue' up to a single 'landed'"):
        check_columns(_sentence([[1, 2, 1, 0, 5, 0], [1, 2, 1, 0, 3, 0]], [10.0, 6.0]),
                      ["KRDU:05L"], _spec())


def test_swapping_two_columns_is_caught_rather_than_drawn():
    """The failure this exists for: the runway and terminal columns both hold small integers, so
    a swap is in range everywhere and would draw a whole flight against the wrong runway."""
    swapped = [[1, 2, 1, 0, 5, 0], [1, 2, 1, 1, 3, 0]]     # runway changes, terminal never lands
    with pytest.raises(SystemExit, match="check their ORDER first"):
        check_columns(_sentence(swapped, [10.0, 6.0]), ["KRDU:05L", "KRDU:23R"], _spec())


def test_the_runway_classes_have_a_sha_of_their_own():
    """The classes sit OUTSIDE the spec so one vocabulary serves five airports — which also means
    two artefacts with the same spec sha can disagree about what word 1 means."""
    assert runway_sha256(["KRDU:05L", "KRDU:23R"]) != runway_sha256(["KRDU:05L", "KSTL:23R"])
    assert runway_sha256(["b", "a"]) == runway_sha256(["b", "a"])


# ── the draw ─────────────────────────────────────────────────────────────────

def test_the_draw_is_reproducible_from_the_artefact_alone():
    sentences = {f"F{i:03d}": {} for i in range(200)}
    first = drawn_flights(sentences, 20, seed=1337)
    assert first == drawn_flights(sentences, 20, seed=1337)
    assert first != drawn_flights(sentences, 20, seed=2024)
    # it is a POOL to stratify from, not the final draw: the stratum is a property of the
    # rebuilt track, so the draw has to rebuild before it can pick
    assert len(first) == min(len(sentences), 20 * 2 * POOL_FACTOR)
    assert len(set(first)) == len(first)
    assert set(first) <= set(sentences)


def test_a_split_smaller_than_the_draw_is_refused():
    with pytest.raises(SystemExit, match="fewer than the 4 asked for"):
        drawn_flights({f"F{i}": {} for i in range(3)}, 2, seed=1337)


class _Difficulty:
    def __init__(self, tortuosity: float, established: bool):
        self.route_tortuosity = tortuosity
        self.established_at_anchor = established


def test_stratify_takes_the_first_of_each_stratum_and_drops_neither(monkeypatch):
    import ts_transformer.experiments.instruction_sample_export as module

    table = {
        "a": _Difficulty(1.01, False), "b": _Difficulty(1.50, False), "c": _Difficulty(1.60, True),
        "d": _Difficulty(1.02, False), "e": _Difficulty(1.70, False), "f": _Difficulty(1.03, False),
    }
    monkeypatch.setattr(module, "approach_difficulty", lambda item, anchor: table[item])
    monkeypatch.setattr(module, "default_anchor", lambda config: 0)
    drawn = module.stratify(list(table), list(table), None, per_stratum=2)

    assert [flight for flight, _, _ in drawn] == ["a", "d", "b", "e"]
    assert [stratum for _, stratum, _ in drawn] == ["straight-in", "straight-in", "vectored", "vectored"]
    assert "c" not in [flight for flight, _, _ in drawn]       # tortuous but established


def test_a_stratum_the_pool_cannot_fill_is_refused(monkeypatch):
    import ts_transformer.experiments.instruction_sample_export as module

    table = {"a": _Difficulty(1.01, False), "b": _Difficulty(1.02, False), "c": _Difficulty(1.03, False)}
    monkeypatch.setattr(module, "approach_difficulty", lambda item, anchor: table[item])
    monkeypatch.setattr(module, "default_anchor", lambda config: 0)
    with pytest.raises(SystemExit, match="filled"):
        module.stratify(list(table), list(table), None, per_stratum=2)


def test_sentences_read_under_another_vocabulary_are_refused(tmp_path: Path):
    (tmp_path / "sentences_train.json").write_text(json.dumps({
        "vocabulary_sha256": "b" * 64, "flights": [],
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="not one vocabulary"):
        sentences_by_flight(tmp_path, "train", "a" * 64)


def test_sentences_are_keyed_by_flight_id(tmp_path: Path):
    (tmp_path / "sentences_train.json").write_text(json.dumps({
        "vocabulary_sha256": "a" * 64,
        "flights": [{"flight_id": "AAL1_05L_x_t", "dataset_id": "KRDU:AAL1_05L_x_t"}],
    }), encoding="utf-8")
    sentences = sentences_by_flight(tmp_path, "train", "a" * 64)
    # the sentences name flights by flight_id, the rebuild wants dataset_id: the join is here
    assert sentences["AAL1_05L_x_t"]["dataset_id"] == "KRDU:AAL1_05L_x_t"


def test_a_missing_split_is_refused_by_path(tmp_path: Path):
    with pytest.raises(SystemExit, match="sentences_val.json"):
        sentences_by_flight(tmp_path, "val", "a" * 64)


# ── the manifest ─────────────────────────────────────────────────────────────

def _entry(set_id: str) -> dict:
    return {"id": set_id, "kind": "vocabulary-readback", "title": set_id, "file": f"{set_id}/sample.json",
            "vocabularySha256": "a" * 64, "runwaySha256": "b" * 64, "readingRule": READING_RULE,
            "flights": 40}


def test_the_manifest_keeps_the_other_sets(tmp_path: Path):
    """A second export must not erase the first — the frontend lists them side by side, and a
    superseded set stays listed (marked) rather than disappearing."""
    update_index(tmp_path, "KRDU", _entry("box_v3"))
    update_index(tmp_path, "KRDU", _entry("box_v4"))
    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert payload["schema"] == INDEX_SCHEMA
    assert [item["id"] for item in payload["sets"]] == ["box_v3", "box_v4"]


def test_re_exporting_a_set_replaces_its_entry_once(tmp_path: Path):
    update_index(tmp_path, "KRDU", _entry("box_v3"))
    update_index(tmp_path, "KRDU", {**_entry("box_v3"), "flights": 100})
    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert len(payload["sets"]) == 1
    assert payload["sets"][0]["flights"] == 100


def test_another_airports_manifest_is_refused(tmp_path: Path):
    update_index(tmp_path, "KRDU", _entry("box_v3"))
    with pytest.raises(SystemExit, match="not KSJC's"):
        update_index(tmp_path, "KSJC", _entry("box_v3"))


def test_a_foreign_schema_is_refused_rather_than_rewritten(tmp_path: Path):
    (tmp_path / "index.json").write_text(json.dumps({"schema": "something-else", "sets": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="something-else"):
        update_index(tmp_path, "KRDU", _entry("box_v3"))


# ── what a box is ────────────────────────────────────────────────────────────

def test_word_runs_recovers_the_segments_by_run_length():
    """The altitude segment the wedge is anchored to is recovered by run-length, not by
    re-reading the profile: every event inside one segment repeats its word."""
    assert word_runs([5, 5, 4, 4, 4, 1]) == [(0, 1), (2, 4), (5, 5)]
    assert word_runs([3]) == [(0, 0)]
    assert word_runs([]) == []


def _flat_frame(rows: int = 61, speed: float = 80.0) -> dict:
    times = np.arange(rows, dtype=np.float64) * 2.0
    return {
        "t": times,
        "to_go_m": 20000.0 - speed * times,
        "cross_m": np.zeros(rows),
        "height_m": np.full(rows, 600.0),
        "ground_speed_mps": np.full(rows, speed),
        "relative_course_deg": np.zeros(rows),
        "course_unwrapped_deg": np.zeros(rows),
        "established": np.ones(rows, dtype=bool),
    }


def test_the_read_signals_are_smoothed_with_the_artefacts_own_windows():
    """WHICH signal is the whole game: the same track against the same boxes is 100 % inside on
    these and 93 % on the raw rows. The windows come from the artefact, never from here."""
    signals = read_signals(_flat_frame(), _spec())
    assert signals["dt_s"] == 2.0
    assert signals["course_window_rows"] == 4          # round(6 / 2) + 1
    assert signals["signal_window_rows"] == 6          # round(10 / 2) + 1
    # a constant track smooths to itself, so the shape is checked rather than the values
    assert len(signals["height_m"]) == len(_flat_frame()["t"])
    # the path axis is the SMOOTHED speed integrated: 80 m/s over 120 s
    assert signals["path_m"][-1] == pytest.approx(80.0 * 120.0, rel=1e-6)
    assert (np.diff(signals["path_m"]) > 0).all()


def test_the_path_axis_is_floored_so_a_stopped_row_cannot_pull_it_back():
    from ts_transformer.manoeuvre.segments import MINIMUM_GROUND_SPEED_MPS
    times = np.array([0.0, 2.0, 4.0])
    path = cumulative_path_m(times, np.array([80.0, 0.0, 80.0]))
    assert path[1] - path[0] == pytest.approx(2.0 * MINIMUM_GROUND_SPEED_MPS)
    assert (np.diff(path) > 0).all()


def test_the_wedge_closes_onto_its_target_and_opens_asymmetrically_going_back():
    """`T - r·tan(up) - f ≤ h ≤ T + r·tan(down) + f`, with f = redundancy · (T + h0).

    Down is the WIDER side and it opens ABOVE the target: the wedge is the set the target is
    backward-reachable from, and losing height is the manoeuvre with the most room. Swapping the
    two draws a corridor of exactly the same width with the slack on the wrong side."""
    spec = _spec()
    targets = np.array([-20.0, 0.0, 100.0, 400.0])
    times = np.array([0.0, 2.0, 4.0, 6.0])
    path = np.array([0.0, 100.0, 200.0, 300.0])
    words = np.zeros((1, len(KINDS)), dtype=np.int64)
    words[0, KINDS.index("altitude")] = 3                      # target 400 m
    low, high = altitude_envelope(np.array([0.0]), words, times, path, spec, targets)

    floor = 0.05 * (400.0 + 50.0)
    # the LAST row is the segment's end: r = 0, so the box is the target's own tolerance
    assert high[-1] == pytest.approx(400.0 + floor)
    assert low[-1] == pytest.approx(400.0 - floor)
    # and going back it opens, more above than below
    remaining = path[-1] - path[0]
    assert high[0] == pytest.approx(400.0 + remaining * math.tan(math.radians(1.5)) + floor)
    assert low[0] == pytest.approx(400.0 - remaining * math.tan(math.radians(1.0)) - floor)
    assert high[0] - 400.0 > 400.0 - low[0]


def test_a_segment_ends_at_the_next_altitude_events_instant():
    """Not at the last row before it. Taking the last row instead leaves rows outside a box the
    artefact accepted (measured: 3 in 2 670)."""
    spec = _spec()
    targets = np.array([-20.0, 0.0, 100.0, 400.0])
    times = np.arange(5, dtype=np.float64) * 2.0               # 0 2 4 6 8
    path = times * 50.0
    words = np.zeros((2, len(KINDS)), dtype=np.int64)
    words[0, KINDS.index("altitude")] = 3
    words[1, KINDS.index("altitude")] = 2
    low, high = altitude_envelope(np.array([0.0, 4.0]), words, times, path, spec, targets)

    # rows 0 and 1 belong to the first segment, and its `r` runs to t = 4 s (the instant the word
    # changes), not to t = 2 s (its own last row)
    floor = 0.05 * (400.0 + 50.0)
    assert high[1] == pytest.approx(400.0 + (path[2] - path[1]) * math.tan(math.radians(1.5)) + floor)
    # the last segment runs to the track's last row
    assert high[-1] == pytest.approx(100.0 + 0.05 * 150.0)


def test_a_row_in_no_segment_is_refused():
    """The events tile the track, so a row with no box in force means the sentence and the track
    are not the same flight — which would otherwise draw as a gap nobody notices."""
    spec = _spec()
    targets = np.array([-20.0, 0.0, 100.0, 400.0])
    times = np.array([0.0, 2.0, 4.0])
    words = np.zeros((1, len(KINDS)), dtype=np.int64)
    with pytest.raises(SystemExit, match="fall in no altitude segment"):
        altitude_envelope(np.array([10.0]), words, times, times * 50.0, spec, targets)


def _reach(to_go: list[float], cross: list[float]) -> list[float]:
    """How far each outline point sits from the apex."""
    return [math.hypot(to_go[i] - to_go[0], cross[i] - cross[0]) for i in range(1, len(to_go))]


def test_the_footprint_is_a_SECTOR_from_the_aircraft_not_a_box_around_it():
    """The words bound the STATE, and the positions that follow over a hold are a pie slice with
    its apex at the aircraft — every point of it at a bearing inside the heading box.

    Drawing the bounding box of this instead (which the first version did) puts flyable-looking
    ground beside the apex that no heading in the box can reach.
    """
    to_go, cross = reachable_sector(-1.0, 1.0, 100.0, 10.0)
    assert (to_go[0], cross[0]) == (0.0, 0.0)          # the apex IS the aircraft
    # every other point sits on the arc, at the hold times the speed box's upper edge
    assert _reach(to_go, cross) == pytest.approx([1000.0] * (len(to_go) - 1))
    # and at a bearing inside the heading box — `to_go` counts DOWN toward the threshold, so a
    # displacement along the course is negative there
    for index in range(1, len(to_go)):
        bearing = math.degrees(math.atan2(-cross[index], -to_go[index]))
        assert -1.0 - 1e-9 <= bearing <= 1.0 + 1e-9

    # the corner of the bounding box that the rectangle would have added is NOT in the sector:
    # it sits beside the apex, at 90° to the course, which no heading in a 2° box can produce
    assert max(abs(value) for value in cross) < 20.0
    assert min(to_go) == pytest.approx(-1000.0, rel=1e-3)


def test_the_speed_words_LOWER_edge_does_not_bound_the_footprint():
    """It says where the aircraft is at the END of the hold; at every earlier instant it is
    nearer, so the swept region runs all the way back to the apex. Only `v_hi` sets the radius —
    which is why `reachable_sector` does not take `v_lo` at all."""
    slow = reachable_sector(-1.0, 1.0, 100.0, 10.0)
    assert max(_reach(*slow)) == pytest.approx(1000.0)
    faster = reachable_sector(-1.0, 1.0, 120.0, 10.0)
    assert max(_reach(*faster)) == pytest.approx(1200.0)


def test_a_wide_box_fans_and_a_narrow_one_is_a_sliver():
    """The arc is drawn at one point per degree of its own opening, between a floor and a cap: a
    constant would either facet the wide boxes or spend sixteen points on a sliver."""
    narrow_to_go, narrow_cross = reachable_sector(-1.0, 1.0, 122.0, 4.0)
    assert len(narrow_to_go) == 1 + SECTOR_ARC_MIN
    width = math.hypot(narrow_to_go[-1] - narrow_to_go[1], narrow_cross[-1] - narrow_cross[1])
    # 488 m deep and 17 m across at its far edge: horizontally one word says almost nothing,
    # and that is the reading, not a drawing fault
    assert max(_reach(narrow_to_go, narrow_cross)) == pytest.approx(488.0)
    assert width < 20.0

    wide_to_go, _ = reachable_sector(-90.0, 90.0, 100.0, 20.0)
    assert len(wide_to_go) == 1 + SECTOR_ARC_MAX


def test_a_zero_hold_sector_collapses_onto_the_aircraft():
    to_go, cross = reachable_sector(-10.0, 10.0, 100.0, 0.0)
    assert _reach(to_go, cross) == pytest.approx([0.0] * (len(to_go) - 1))


def test_containment_counts_rows_and_forgives_only_the_edge():
    read = np.array([10.0, 20.0, 30.0])
    low = np.array([10.0, 10.0, 10.0])
    high = np.array([25.0, 25.0, 25.0])
    # a row exactly ON an edge is inside — the labeller's greedy reader leaves rows there by
    # construction, and without the epsilon they read as violations of a millionth of a metre
    assert containment(read, low, high) == {"rows": 3, "outside": 1}
    assert containment(np.array([25.0 + INSIDE_EPSILON / 2]), np.array([10.0]), np.array([25.0]))["outside"] == 0
    assert containment(np.array([25.0 + INSIDE_EPSILON * 10]), np.array([10.0]), np.array([25.0]))["outside"] == 1


# ── the geodetic columns the 3D layer draws from ────────────────────────────

class _StubSeries:
    """A series carrying only what the geodesy reads: the runway's inbound course, the chart
    frame, and the target's place in it."""

    def __init__(self, lat: float, lon: float, alt_m: float, psi_rad: float,
                 threshold_above_anchor_m: float = 0.0, rotated: bool = False):
        from ts_transformer.data.coordinate_frames import AirportENUFrame, RunwayAlignedFrame

        self.frame = (
            RunwayAlignedFrame(lat0=lat, lon0=lon, alt0=alt_m, heading_rad=psi_rad)
            if rotated
            else AirportENUFrame(lat0=lat, lon0=lon, alt0=alt_m, code="TEST")
        )
        self.target_chart = (0.0, 0.0, threshold_above_anchor_m)
        self.scenario = type("S", (), {"target": type("T", (), {"psi": psi_rad})()})()


@pytest.mark.parametrize("psi_deg", [52.0, -131.0, 17.0])
@pytest.mark.parametrize("rotated", [False, True])
def test_the_geodetic_inverse_is_course_frame_rows_run_backwards(psi_deg: float, rotated: bool):
    """A ROUND TRIP, which is the only shape that pins all four terms of the inverse.

    The angles have both sin and cos non-zero, because at ψ = 0 two of the four terms vanish and
    a sign error in either survives; and it runs under the rotated frame as well, because on an
    unrotated one `from_world_horizontal` and its opposite agree and the wrong one would pass.
    """
    from ts_transformer.data.approach_difficulty import course_frame_rows

    psi = math.radians(psi_deg)
    series = _StubSeries(35.8776, -78.7875, 132.0, psi, rotated=rotated)
    offsets = [(4000.0, -2500.0), (-1200.0, 800.0), (25000.0, 6400.0)]

    east = np.array([e for e, _ in offsets])
    north = np.array([n for _, n in offsets])
    frame = course_frame_rows(east, north, np.ones(len(offsets)), np.zeros(len(offsets)), psi)

    lons, lats = lonlat_from_frame(series, frame["to_go_m"], frame["cross_m"])
    for index, (east_m, north_m) in enumerate(offsets):
        first, second = series.frame.from_world_horizontal(east_m, north_m)
        lat, lon = series.frame.latlon_from_horizontal(
            series.target_chart[0] + first, series.target_chart[1] + second
        )
        assert lats[index] == pytest.approx(lat, abs=1e-6)
        assert lons[index] == pytest.approx(lon, abs=1e-6)


def test_the_geodetic_inverse_puts_the_threshold_back_where_it_is():
    for psi in (0.0, math.radians(52.0), math.radians(-131.0), math.pi):
        series = _StubSeries(35.8776, -78.7875, 132.0, psi)
        lons, lats = lonlat_from_frame(series, [0.0], [0.0])
        assert lats[0] == pytest.approx(35.8776, abs=1e-6)
        assert lons[0] == pytest.approx(-78.7875, abs=1e-6)


def test_a_point_down_the_course_lands_up_the_inbound_bearing():
    """10 km still to fly on a due-EAST inbound course (math-ENU 0) is 10 km WEST of the
    threshold: `to_go` counts what is left to fly, so the aircraft is behind it."""
    series = _StubSeries(35.8776, -78.7875, 132.0, 0.0)
    lons, lats = lonlat_from_frame(series, [10000.0], [0.0])
    assert lats[0] == pytest.approx(35.8776, abs=1e-5)
    assert lons[0] < -78.7875                                   # west
    _, right_lats = lonlat_from_frame(series, [0.0], [1000.0])  # 1 km right of the course
    assert right_lats[0] < 35.8776                              # is 1 km SOUTH of it


def test_the_altitude_leaves_as_hae_not_msl():
    """A record is MSL and Cesium reads the ellipsoid. At KRDU N = -33.5 m and h = H + N, so a
    line handed MSL where HAE was wanted renders 33.5 m too HIGH — above its own terrain.

    It is an OFFSET rather than a converted column because the envelope has four more height
    columns over the same ground track; recomputing the geodesy per column would be the same
    inverse five times, and the five could drift."""
    from flight_scenarios.datum import geoid_undulation_m

    series = _StubSeries(35.8776, -78.7875, 132.0, math.radians(52.0), threshold_above_anchor_m=3.0)
    lons, lats = lonlat_from_frame(series, [0.0], [0.0])
    offset = hae_offset_m(series, lats, lons)
    undulation = geoid_undulation_m([35.8776], [-78.7875])[0]
    assert undulation == pytest.approx(-33.5, abs=0.5)
    # the anchor's elevation + the threshold above it + the geoid — an altitude that forgot the
    # anchor put touchdown below the ellipsoid
    assert offset[0] == pytest.approx(132.0 + 3.0 + undulation, abs=0.1)
    assert offset[0] + 500.0 == pytest.approx(132.0 + 3.0 + 500.0 + undulation, abs=0.1)
