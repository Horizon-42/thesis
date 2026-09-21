"""The Training view's sample exporter: the draw, the artefact join and the manifest.

The track rebuild needs a checkpoint and the arrival store, so it is exercised by the runner
itself; what is pinned here is everything that decides WHICH flights and WHAT SHAPE — the parts
that can silently disagree with the artefact or with the frontend reader.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from ts_transformer.experiments.instruction_sample_export import (
    INDEX_SCHEMA, POOL_FACTOR, cumulative_path_m, drawn_flights, readings_by_flight, update_index,
)
import numpy as np

from ts_transformer.experiments.instruction_sample_export import (
    flown_sentence, geodetic_columns,
)
from ts_transformer.manoeuvre import instruction_kinematics
from ts_transformer.manoeuvre.instructions import INSTRUCTION_KINDS, Vocabulary

# The frontend reader's own constants (`src/data/trainingSample.ts`). Declared MIRRORS: the two
# sides are one contract, and a fixture that restated them could not catch it moving.
FRONTEND_INDEX_SCHEMA = "aeroviz-training-index-v1"
FRONTEND_KINDS = ("heading", "vertical", "speed", "runway", "duration", "terminal")


def test_the_column_order_is_the_frontend_reader_s():
    """The six columns are positional on both sides; if this fails, one of them moved."""
    assert INSTRUCTION_KINDS == FRONTEND_KINDS
    assert INDEX_SCHEMA == FRONTEND_INDEX_SCHEMA
    assert "intercept" not in INSTRUCTION_KINDS       # deleted 2026-09-20 (D73)


def test_the_draw_is_reproducible_from_the_artefact_alone():
    """The draw moved off the hand check (2026-09-21: the `segment-v12` artefacts were written
    with `--hand-check 0`, so there are no pages to be a prefix of). What replaced it has to be
    reproducible from the artefact and the seed the manifest records — otherwise nobody can say
    which flights a published set holds, or get them back."""
    readings = {f"F{i:03d}": {} for i in range(200)}
    first = drawn_flights(readings, 20, seed=1337)
    assert first == drawn_flights(readings, 20, seed=1337)
    assert first != drawn_flights(readings, 20, seed=2024)
    # it is a POOL to stratify from, not the final draw: the stratum is a property of the
    # rebuilt track, so the draw has to rebuild before it can pick
    assert len(first) == min(len(readings), 20 * 2 * POOL_FACTOR)
    assert len(set(first)) == len(first)
    assert set(first) <= set(readings)


def test_a_split_smaller_than_the_draw_is_refused():
    with pytest.raises(SystemExit, match="fewer than the 4 asked for"):
        drawn_flights({f"F{i}": {} for i in range(3)}, 2, seed=1337)


class _Difficulty:
    def __init__(self, tortuosity: float, established: bool):
        self.route_tortuosity = tortuosity
        self.established_at_anchor = established


def test_stratify_takes_the_first_of_each_stratum_and_drops_neither(monkeypatch):
    """Half straight-in, half vectored, in the draw's own order — and a flight in NEITHER
    stratum (tortuous but already established at the anchor) is not drawn, which is the same
    rule the vocabulary runner's hand-check draw applies."""
    import ts_transformer.experiments.instruction_sample_export as module

    table = {
        "a": _Difficulty(1.01, False), "b": _Difficulty(1.50, False), "c": _Difficulty(1.60, True),
        "d": _Difficulty(1.02, False), "e": _Difficulty(1.70, False), "f": _Difficulty(1.03, False),
    }
    monkeypatch.setattr(module, "approach_difficulty", lambda item, anchor: table[item])
    monkeypatch.setattr(module, "default_anchor", lambda config: 0)
    drawn = module.stratify(list(table), list(table), Vocabulary(), per_stratum=2)

    assert [flight for flight, _, _ in drawn] == ["a", "d", "b", "e"]
    assert [stratum for _, stratum, _ in drawn] == ["straight-in", "straight-in", "vectored", "vectored"]
    assert "c" not in [flight for flight, _, _ in drawn]       # tortuous but established


def test_a_stratum_the_pool_cannot_fill_is_refused(monkeypatch):
    """Publishing 3 straight-in and 1 vectored under a `perStratum: 2` manifest would make the
    stratum column say something the draw did not do."""
    import ts_transformer.experiments.instruction_sample_export as module

    table = {"a": _Difficulty(1.01, False), "b": _Difficulty(1.02, False), "c": _Difficulty(1.03, False)}
    monkeypatch.setattr(module, "approach_difficulty", lambda item, anchor: table[item])
    monkeypatch.setattr(module, "default_anchor", lambda config: 0)
    with pytest.raises(SystemExit, match="filled"):
        module.stratify(list(table), list(table), Vocabulary(), per_stratum=2)


def test_sentences_read_under_another_vocabulary_are_refused(tmp_path: Path):
    """The view claims to show ONE artefact; sentences from another sha are not it."""
    (tmp_path / "sentences_train.json").write_text(json.dumps({
        "vocabulary_sha256": "b" * 64, "flights": [],
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="not one vocabulary"):
        readings_by_flight(tmp_path, "train", "a" * 64)


def test_sentences_are_keyed_by_flight_id(tmp_path: Path):
    (tmp_path / "sentences_train.json").write_text(json.dumps({
        "vocabulary_sha256": "a" * 64,
        "flights": [{"flight_id": "AAL1_05L_x_t", "dataset_id": "KRDU:AAL1_05L_x_t"}],
    }), encoding="utf-8")
    readings = readings_by_flight(tmp_path, "train", "a" * 64)
    # the hand check names flights by flight_id, the rebuild wants dataset_id: the join is here
    assert readings["AAL1_05L_x_t"]["dataset_id"] == "KRDU:AAL1_05L_x_t"


def test_a_missing_split_is_refused_by_path(tmp_path: Path):
    with pytest.raises(SystemExit, match="sentences_val.json"):
        readings_by_flight(tmp_path, "val", "a" * 64)


def _entry(set_id: str) -> dict:
    return {"id": set_id, "kind": "vocabulary-readback", "title": set_id, "file": f"{set_id}/sample.json",
            "vocabularySha256": "a" * 64, "runwaySha256": "b" * 64, "readingRule": "plateau-v11", "flights": 40}


def test_the_manifest_keeps_the_other_sets(tmp_path: Path):
    """A second export must not erase the first — the frontend lists them side by side."""
    update_index(tmp_path, "KRDU", _entry("vocabulary_tau10"))
    update_index(tmp_path, "KRDU", _entry("vocabulary_tau5"))
    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert payload["schema"] == INDEX_SCHEMA
    assert [item["id"] for item in payload["sets"]] == ["vocabulary_tau10", "vocabulary_tau5"]


def test_re_exporting_a_set_replaces_its_entry_once(tmp_path: Path):
    update_index(tmp_path, "KRDU", _entry("vocabulary_tau10"))
    replacement = {**_entry("vocabulary_tau10"), "flights": 100}
    update_index(tmp_path, "KRDU", replacement)
    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert len(payload["sets"]) == 1
    assert payload["sets"][0]["flights"] == 100


def test_another_airports_manifest_is_refused(tmp_path: Path):
    update_index(tmp_path, "KRDU", _entry("vocabulary_tau10"))
    with pytest.raises(SystemExit, match="not KSJC's"):
        update_index(tmp_path, "KSJC", _entry("vocabulary_tau10"))


def test_a_foreign_schema_is_refused_rather_than_rewritten(tmp_path: Path):
    (tmp_path / "index.json").write_text(json.dumps({"schema": "something-else", "sets": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="something-else"):
        update_index(tmp_path, "KRDU", _entry("vocabulary_tau10"))


# ── the flown sentence in the payload (T5) ───────────────────────────────────

def _reading(events: list[float], words: list[list[int]]) -> dict:
    return {
        "dataset_id": "KRDU:TEST", "flight_id": "TEST", "runway": "05L",
        "event_times_s": events, "words": words,
        "established_from_start": True, "duration_s": 200.0,
    }


def _frame(rows: int = 101) -> dict:
    times = np.arange(rows, dtype=np.float64) * 2.0
    return {
        "t": times,
        "to_go_m": 20000.0 - 70.0 * times,
        "cross_m": np.zeros(rows),
        "height_m": np.full(rows, 600.0),
        "ground_speed_mps": np.full(rows, 70.0),
        "relative_course_deg": np.zeros(rows),
        "course_unwrapped_deg": np.zeros(rows),
        "established": np.ones(rows, dtype=bool),
    }


def test_the_flown_sentence_is_the_artefacts_own_sentence_not_a_re_reading():
    """V19 for the geometric track too: the reading handed to the kinematics is rebuilt from the
    artefact's event times and words, so the line drawn is the sentence the view shows."""
    columns = {kind: index for index, kind in enumerate(INSTRUCTION_KINDS)}
    level = Vocabulary().vertical_modes_deg.index(0.0)   # word 0 is the CLIMB mode, not level
    words = [[0] * len(INSTRUCTION_KINDS), [0] * len(INSTRUCTION_KINDS)]
    words[1][columns["heading"]] = 18                    # +90° from 100 s (5° a bin)
    words[0][columns["vertical"]] = words[1][columns["vertical"]] = level
    words[0][columns["speed"]] = words[1][columns["speed"]] = 4
    words[1][columns["terminal"]] = 1
    _, flown = flown_sentence(_reading([0.0, 100.0], words), Vocabulary(), _frame())

    assert flown["tS"][0] == 0.0
    # it flies the course until the second event, then turns
    assert abs(flown["relCourseDeg"][50]) < 1e-9
    assert flown["relCourseDeg"][-1] > 45.0
    # having turned 90° off the course it never reaches the runway, and says so
    assert flown["endReason"] == "time-cap"
    assert flown["finalGapM"] > 1000.0


def test_the_geometry_block_states_every_assumption_the_line_was_drawn_under():
    """An approximation nobody can see stated is worse than none (design §5.4), so the block
    travels with the export and names the files its constants came from."""
    from ts_transformer.outputs.guidance.controller import CLIMB_MAX_RAD, DESCENT_MAX_RAD
    from ts_transformer.geometry.flyability import G

    block = instruction_kinematics.assumptions()
    for field in ("method", "dtS", "bankDeg", "gravityMps2", "descentMaxDeg",
                  "climbMaxDeg", "accelMaxMps2", "startsAt", "stopRule", "windModelled",
                  "aircraftTypeModelled", "constantsFrom",
                  # word 0 is not flown as a direction (2026-09-21): a reader told only the bank
                  # and the step would still not know why the track curves back to the centreline
                  "establishedWordTracksTheCentreline", "interceptMaxDeg", "bankAndAccelFrom"):
        assert field in block, field
    # the stated numbers ARE the imported ones — a block that drifted from the code it
    # describes is worse than no block
    assert block["gravityMps2"] == G
    # the bank and the acceleration are the kinematics module's OWN measured values, not the
    # guidance controller's: the block must state what was flown, and they are different numbers
    assert block["accelMaxMps2"] == instruction_kinematics.ACCEL_MAX_MPS2
    assert block["bankDeg"] == pytest.approx(math.degrees(instruction_kinematics.TURN_BANK_RAD))
    assert block["interceptMaxDeg"] == instruction_kinematics.INTERCEPT_MAX_DEG
    assert block["descentMaxDeg"] == pytest.approx(math.degrees(DESCENT_MAX_RAD))
    assert block["climbMaxDeg"] == pytest.approx(math.degrees(CLIMB_MAX_RAD))
    # The vertical word IS the commanded angle now, so the block says so and no longer carries a
    # height gain — there is no height error for a time constant to close.
    assert block["verticalIsCommandedAngle"] is True and "heightGainS" not in block
    assert block["dtS"] == instruction_kinematics.STEP_S
    assert str(int(instruction_kinematics.OVERRUN_S)) in block["stopRule"]
    assert block["windModelled"] is False
    assert block["aircraftTypeModelled"] is False


def test_the_gap_is_reported_with_the_fraction_of_the_approach_it_covers():
    """A mean gap over a flown track that stopped early would otherwise read as if it spanned
    the whole approach."""
    columns = {kind: index for index, kind in enumerate(INSTRUCTION_KINDS)}
    words = [[0] * len(INSTRUCTION_KINDS)]
    words[0][columns["speed"]] = 4
    words[0][columns["terminal"]] = 1
    # A frame that outlives the flown sentence: it starts 3 km out, so the words cross the
    # threshold long before the observation ends and the comparison covers only part of it.
    frame = _frame()
    frame["to_go_m"] = 3000.0 - 70.0 * frame["t"]
    _, flown = flown_sentence(_reading([0.0], words), Vocabulary(), frame)

    assert flown["endReason"] == "crossed-threshold"
    assert flown["comparedS"] == pytest.approx(flown["tS"][-1], abs=2.0)
    assert flown["comparedFraction"] < 0.5          # a PARTIAL cover, which is the point
    assert flown["comparedFraction"] == pytest.approx(flown["comparedS"] / frame["t"][-1], abs=0.01)


# ── the geodetic columns the 3D layer draws from (T6) ───────────────────────

class _StubSeries:
    """A series carrying only what `geodetic_columns` reads: the runway's inbound course, the
    chart frame, and the target's place in it."""

    def __init__(self, lat: float, lon: float, alt_m: float, psi_rad: float,
                 threshold_above_anchor_m: float = 0.0, rotated: bool = False):
        from ts_transformer.data.coordinate_frames import AirportENUFrame, RunwayAlignedFrame

        # `alt_m` is the AIRPORT reference elevation (the chart's vertical anchor); the
        # threshold sits a little above or below it, and the chart's z counts from the anchor.
        # A ROTATED frame is the one that tells `from_world_horizontal` apart from its
        # opposite: on the unrotated default the rotation is the identity and both agree.
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

    Push an arbitrary offset from the threshold forward through `course_frame_rows`, feed the
    `to_go / cross` it produces back through `geodetic_columns`, and require the latitude and
    longitude to come back. The angles have both sin and cos non-zero, because at ψ = 0 two of
    the four terms vanish and a sign error in either survives; and it runs under the rotated
    frame as well, because on an unrotated one `from_world_horizontal` and its opposite agree
    and the wrong one would pass.
    """
    from ts_transformer.data.approach_difficulty import course_frame_rows

    psi = math.radians(psi_deg)
    series = _StubSeries(35.8776, -78.7875, 132.0, psi, rotated=rotated)
    offsets = [(4000.0, -2500.0), (-1200.0, 800.0), (25000.0, 6400.0)]

    east = np.array([e for e, _ in offsets])
    north = np.array([n for _, n in offsets])
    frame = course_frame_rows(east, north, np.ones(len(offsets)), np.zeros(len(offsets)), psi)

    columns = geodetic_columns(series, frame["to_go_m"], frame["cross_m"], np.zeros(len(offsets)))
    for index, (east_m, north_m) in enumerate(offsets):
        first, second = series.frame.from_world_horizontal(east_m, north_m)
        lat, lon = series.frame.latlon_from_horizontal(
            series.target_chart[0] + first, series.target_chart[1] + second
        )
        assert columns["lat"][index] == pytest.approx(lat, abs=1e-6)
        assert columns["lon"][index] == pytest.approx(lon, abs=1e-6)


def test_the_geodetic_inverse_puts_the_threshold_back_where_it_is():
    """`to_go = cross = 0` is the threshold, so it must invert to the frame's own anchor —
    whatever the inbound course is."""
    for psi in (0.0, math.radians(52.0), math.radians(-131.0), math.pi):
        series = _StubSeries(35.8776, -78.7875, 132.0, psi)
        columns = geodetic_columns(series, [0.0], [0.0], [0.0])
        assert columns["lat"][0] == pytest.approx(35.8776, abs=1e-6)
        assert columns["lon"][0] == pytest.approx(-78.7875, abs=1e-6)


def test_a_point_down_the_course_lands_up_the_inbound_bearing():
    """10 km still to fly on a due-EAST inbound course (math-ENU 0) is 10 km WEST of the
    threshold: `to_go` counts what is left to fly, so the aircraft is behind it."""
    series = _StubSeries(35.8776, -78.7875, 132.0, 0.0)
    columns = geodetic_columns(series, [10000.0], [0.0], [0.0])
    assert columns["lat"][0] == pytest.approx(35.8776, abs=1e-5)
    assert columns["lon"][0] < -78.7875                       # west
    # and 1 km right of the course is 1 km SOUTH of it
    right = geodetic_columns(series, [0.0], [1000.0], [0.0])
    assert right["lat"][0] < 35.8776


def test_the_altitude_leaves_as_hae_not_msl():
    """A record is MSL and Cesium reads the ellipsoid. At KRDU N = -33.5 m and h = H + N, so
    a line handed MSL where HAE was wanted renders 33.5 m too HIGH — above its own terrain and
    above the observed CZML it is read against."""
    from flight_scenarios.datum import geoid_undulation_m

    series = _StubSeries(35.8776, -78.7875, 132.0, math.radians(52.0), threshold_above_anchor_m=3.0)
    columns = geodetic_columns(series, [0.0], [0.0], [500.0])
    undulation = geoid_undulation_m([35.8776], [-78.7875])[0]
    assert undulation == pytest.approx(-33.5, abs=0.5)
    # the anchor's elevation + the threshold above it + the height above the threshold, then
    # the geoid — an altitude that forgot the anchor put touchdown below the ellipsoid
    assert columns["altHaeM"][0] == pytest.approx(132.0 + 3.0 + 500.0 + undulation, abs=0.1)
