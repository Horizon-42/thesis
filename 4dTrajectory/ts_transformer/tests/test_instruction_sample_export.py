"""The Training view's sample exporter: the draw, the artefact join and the manifest.

The track rebuild needs a checkpoint and the arrival store, so it is exercised by the runner
itself; what is pinned here is everything that decides WHICH flights and WHAT SHAPE — the parts
that can silently disagree with the artefact or with the frontend reader.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ts_transformer.experiments.instruction_sample_export import (
    INDEX_SCHEMA, drawn_flights, readings_by_flight, update_index,
)
from ts_transformer.manoeuvre.instructions import INSTRUCTION_KINDS

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
