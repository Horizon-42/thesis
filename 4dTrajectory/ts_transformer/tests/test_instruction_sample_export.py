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
    INDEX_SCHEMA, drawn_flights, readings_by_flight, update_index,
)
import numpy as np

from ts_transformer.experiments.instruction_sample_export import geometric_track
from ts_transformer.manoeuvre import instruction_kinematics
from ts_transformer.manoeuvre.instructions import INSTRUCTION_KINDS, Vocabulary

# The frontend reader's own constants (`src/data/trainingSample.ts`). Declared MIRRORS: the two
# sides are one contract, and a fixture that restated them could not catch it moving.
FRONTEND_INDEX_SCHEMA = "aeroviz-training-index-v1"
FRONTEND_KINDS = ("heading", "altitude", "speed", "runway", "duration", "terminal")


def _hand_check(directory: Path, rows: list[tuple[str, str]]) -> None:
    folder = directory / "hand_check"
    folder.mkdir(parents=True)
    with (folder / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "flight_id", "stratum"])
        for flight_id, stratum in rows:
            writer.writerow([f"{flight_id}.png", flight_id, stratum])


def test_the_column_order_is_the_frontend_reader_s():
    """The six columns are positional on both sides; if this fails, one of them moved."""
    assert INSTRUCTION_KINDS == FRONTEND_KINDS
    assert INDEX_SCHEMA == FRONTEND_INDEX_SCHEMA
    assert "intercept" not in INSTRUCTION_KINDS       # deleted 2026-09-20 (D73)


def test_the_draw_is_a_prefix_of_the_hand_check(tmp_path: Path):
    """Taking the first N per stratum must give the pages a human already checked — that is the
    whole reason the two views match, so it is pinned rather than reasoned about."""
    rows = [(f"S{i}", "straight-in") for i in range(10)] + [(f"V{i}", "vectored") for i in range(10)]
    _hand_check(tmp_path, rows)

    four = drawn_flights(tmp_path, 2)
    assert four == [("S0", "straight-in"), ("S1", "straight-in"), ("V0", "vectored"), ("V1", "vectored")]

    # and a bigger draw EXTENDS it rather than reshuffling: the small draw stays a prefix
    eight = drawn_flights(tmp_path, 4)
    by_stratum = {name: [f for f, s in eight if s == name] for name in ("straight-in", "vectored")}
    assert by_stratum["straight-in"][:2] == ["S0", "S1"]
    assert by_stratum["vectored"][:2] == ["V0", "V1"]


def test_a_short_stratum_is_refused_not_topped_up(tmp_path: Path):
    _hand_check(tmp_path, [("S0", "straight-in"), ("V0", "vectored"), ("V1", "vectored")])
    with pytest.raises(SystemExit, match="fewer than the 2 per stratum"):
        drawn_flights(tmp_path, 2)


def test_a_missing_hand_check_is_refused_by_path(tmp_path: Path):
    with pytest.raises(SystemExit, match="hand_check"):
        drawn_flights(tmp_path, 1)


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
    words = [[0] * len(INSTRUCTION_KINDS), [0] * len(INSTRUCTION_KINDS)]
    words[1][columns["heading"]] = 9                     # +90° from 100 s
    words[0][columns["speed"]] = words[1][columns["speed"]] = 4
    words[1][columns["terminal"]] = 1
    flown = geometric_track(_reading([0.0, 100.0], words), Vocabulary(), _frame())

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
    from ts_transformer.outputs.guidance.controller import (
        ACCEL_MAX_MPS2, CLIMB_MAX_RAD, DESCENT_MAX_RAD, HEIGHT_GAIN_S,
    )
    from ts_transformer.geometry.flyability import G

    block = instruction_kinematics.assumptions()
    for field in ("method", "dtS", "bankDeg", "gravityMps2", "heightGainS", "descentMaxDeg",
                  "climbMaxDeg", "accelMaxMps2", "startsAt", "stopRule", "windModelled",
                  "aircraftTypeModelled", "constantsFrom"):
        assert field in block, field
    # the stated numbers ARE the imported ones — a block that drifted from the code it
    # describes is worse than no block
    assert block["gravityMps2"] == G
    assert block["heightGainS"] == HEIGHT_GAIN_S
    assert block["accelMaxMps2"] == ACCEL_MAX_MPS2
    assert block["descentMaxDeg"] == pytest.approx(math.degrees(DESCENT_MAX_RAD))
    assert block["climbMaxDeg"] == pytest.approx(math.degrees(CLIMB_MAX_RAD))
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
    flown = geometric_track(_reading([0.0], words), Vocabulary(), frame)

    assert flown["endReason"] == "crossed-threshold"
    assert flown["comparedS"] == pytest.approx(flown["tS"][-1], abs=2.0)
    assert flown["comparedFraction"] < 0.5          # a PARTIAL cover, which is the point
    assert flown["comparedFraction"] == pytest.approx(flown["comparedS"] / frame["t"][-1], abs=0.01)
