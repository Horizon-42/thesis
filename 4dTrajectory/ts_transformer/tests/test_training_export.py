"""The Training export of stage A (`experiments/training_export.py`, `instructions/training_files.py`; vocabulary §12.1
A23, outline §6): who is drawn, the words as events, the envelopes the views draw, and the set and its index beside the
old ones — never over an existing set, never into the instruction-v3 view's index."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ts_transformer.experiments import training_export as export
from ts_transformer.instructions import training_files as files
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import UNCHANGED


def _rows(airport, strata):
    """Formal replay rows: ``strata`` a list of (stratum, count) for ``airport``, ids F0, F1, …"""
    out, k = {}, 0
    for stratum, count in strata:
        for _ in range(count):
            out[f"F{k}"] = {"airport": airport, "stratum": stratum, "kind": "without go-around"}
            k += 1
    return out


def test_each_split_draws_per_stratum_flights_flown_at_every_row_interval_seeded():
    two = _rows("KAAA", [("straight-in", 6), ("vectored", 5)])
    rows = {2.0: two, 4.0: dict(two), 8.0: {k: v for k, v in two.items() if k != "F0"}}   # F0 not flown at 8 s
    chosen = export.choose(rows, "KAAA", 3, 1337)
    assert len(chosen) == 6 and "F0" not in chosen
    assert [rows[2.0][d]["stratum"] for d in chosen] == ["straight-in"] * 3 + ["vectored"] * 3
    assert export.choose(rows, "KAAA", 3, 1337) == chosen != export.choose(rows, "KAAA", 3, 7)
    with pytest.raises(ValueError, match="5 straight-in flights flown at every row interval, 6 asked"):
        export.choose(rows, "KAAA", 6, 1337)


def test_the_words_of_a_sentence_are_its_events_decoded_with_the_corrections_marked():
    """Each word as the views show it, decoded by `Words` (the views decode nothing): the runway's candidate or
    go-around, a heading relative to the course of the runway in force and its track, a level above E and in MSL or "no
    level-off", an angle's nominal, a speed or "unspecified"."""
    from ts_transformer.instructions.words import Words
    from ts_transformer.tests.support import instruction_airport, instruction_spec

    words, geometry = Words(instruction_spec()), instruction_airport()
    grid = np.full((3, 5), UNCHANGED)
    grid[0] = [0, 2, 10, 0, words.speed_unspecified]
    grid[2, 1], grid[2, 2], grid[2, 0] = 3, words.altitude_no_level_off, -2
    correction = np.zeros((3, 5), bool)
    correction[2, 1] = True
    said = export.events(grid, None, geometry, words)
    assert [(e["row"], e["column"], e["value"], e["correction"]) for e in said] == [
        (0, 0, 0, False), (0, 1, 2, False), (0, 2, 10, False), (0, 3, 0, False), (0, 4, words.speed_unspecified, False),
        (2, 0, -2, False), (2, 1, 3, False), (2, 2, words.altitude_no_level_off, False)]
    course = geometry.candidates[0].course_deg
    step = words.spec.heading_step_deg
    assert said[0]["says"] == {"runway": geometry.candidates[0].ident, "runwayIndex": 0}
    assert said[1]["says"] == {"relativeDeg": 2 * step, "trackDeg": round((course + 2 * step) % 360.0, 3)}
    level = words.altitude_level_m(10)
    assert said[2]["says"] == {"noLevelOff": False, "levelM": level, "mslM": round(level + geometry.elevation_m, 1)}
    assert said[3]["says"] == {"angleDeg": 0.0, "climb": False, "level": True}
    assert said[4]["says"] == {"speedMps": None}
    # go-around changes no runway: the heading word after it is still relative to candidate 0's course
    assert said[5]["says"] == {"goAround": True}
    assert said[6]["says"]["trackDeg"] == round((course + 3 * step) % 360.0, 3)
    assert said[7]["says"] == {"noLevelOff": True, "levelM": None, "mslM": None}
    assert export.events(grid, correction, geometry, words)[6]["correction"] is True


def test_a_heading_band_is_the_judges_rows_and_verdicts():
    """The rows and the verdicts are `envelope.heading_words_inside`'s (a word judged from a lead after it to a lead
    after the next one, never past the end), drawn row by row."""
    from ts_transformer.instructions import envelope

    track = np.array([90.0] * 6 + [100.0, 120.0, 130.0, 130.0])
    words = [(0, 90.0), (5, 130.0)]
    bands = files.heading_bands(track, words, 2, len(track), 4.5)
    judged = envelope.heading_words_inside(track, words, 2, len(track), 4.5)
    assert [(b["row"], len(b["inside"]), sum(b["inside"])) for b in bands] == [
        (j["row"], j["rows"], j["inside"]) for j in judged]
    assert (bands[0]["firstRow"], bands[0]["stopRow"], bands[1]["firstRow"], bands[1]["stopRow"]) == (2, 7, 7, 10)
    assert bands[1]["inside"] == [0, 1, 1]


def test_a_tube_is_written_in_msl_beside_its_level_above_the_airport():
    tube = files.tube_payload(4, 9, 600.0, np.array([550.0, 560.0]), np.array([640.0, 650.0]), 60.0,
                              {"rows": 5, "inside": 4, "contained": False})
    assert tube == {"row": 4, "endRow": 9, "levelM": 600.0, "lowMslM": [610.0, 620.0], "highMslM": [700.0, 710.0],
                    "rows": 5, "inside": 4, "contained": False}
    assert files.nullable([1.04, float("nan")], 1) == [1.0, None]


def _entry(set_id):
    return {"id": set_id, "kind": files.SET_KIND, "readingRule": READING_RULE, "file": f"{set_id}/{files.SAMPLE_FILE}"}


def _sample(set_id, airport):
    return {"schema": files.SAMPLE_SCHEMA, "setId": set_id, "airport": airport, "vocabulary": {"readingRule": READING_RULE}}


def test_a_set_goes_beside_the_old_index_and_is_never_overwritten(tmp_path):
    """Outline §6 item 3: the set and its entry go into `index_v4.json`; the instruction-v3 view's `index.json` is never
    read or written; a set already listed, or an index another run changed, is refused."""
    training = tmp_path / "KAAA" / "training"
    training.mkdir(parents=True)
    old = training / "index.json"
    old.write_text('{"schema": "aeroviz-training-index-v1", "sets": []}', encoding="utf-8")
    before = old.read_bytes()
    existing = files.read_index(training, "KAAA", "set_a")
    assert existing == []
    files.write_set(training, "KAAA", _entry("set_a"), files.serialise(_sample("set_a", "KAAA")), existing)
    assert old.read_bytes() == before
    index = json.loads((training / files.INDEX_FILE).read_text(encoding="utf-8"))
    assert index["schema"] == files.INDEX_SCHEMA and [s["id"] for s in index["sets"]] == ["set_a"]
    entry, sample = files.listed_set(training, "KAAA", "set_a")
    assert entry["id"] == sample["setId"] == "set_a"
    with pytest.raises(ValueError, match="already lists set set_a"):
        files.read_index(training, "KAAA", "set_a")
    stale = files.read_index(training, "KAAA", "set_b")
    files.write_set(training, "KAAA", _entry("set_c"), files.serialise(_sample("set_c", "KAAA")), stale)
    with pytest.raises(ValueError, match="changed since this run read it"):
        files.write_set(training, "KAAA", _entry("set_b"), files.serialise(_sample("set_b", "KAAA")), stale)
    with pytest.raises(files.NotListed):
        files.listed_set(training, "KAAA", "set_b")
    with pytest.raises(ValueError, match="not KBBB's"):
        files.read_index(training, "KBBB", "set_d")
    with pytest.raises(ValueError, match="refusing NaN|Out of range float"):
        files.serialise({"x": float("nan")})


def test_a_listed_set_of_another_format_is_refused_by_name(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    files.write_set(training, "KAAA", _entry("set_a"), files.serialise({**_sample("set_a", "KAAA"),
                                                                        "schema": "aeroviz-training-sample-v8"}), [])
    with pytest.raises(ValueError, match="aeroviz-training-sample-v8 file, not a closed-loop-readback set"):
        files.listed_set(training, "KAAA", "set_a")


def test_the_datum_is_the_artefacts_own_manifests(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runway_targets": {"09": {"hae_minus_msl_m": -33.0}}}), encoding="utf-8")
    recorded = hashlib.sha256(manifest.read_bytes()).hexdigest()
    sources = [{"airport": "KAAA", "arrival_manifest_sha256": recorded}]
    assert files.runway_hae_minus_msl_m(sources, "KAAA", manifest) == {"09": -33.0}
    manifest.write_text(json.dumps({"runway_targets": {"09": {"hae_minus_msl_m": -32.0}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not the arrival manifest"):
        files.runway_hae_minus_msl_m(sources, "KAAA", manifest)


# ---- the frontend's fixtures, written by the export code (outline §6 item 2)
#: Where the frontend's Vitest readers find the stage-A fixtures; `AEROVIZ_WRITE_FIXTURES=1` writes them again.
FIXTURES = Path(__file__).resolve().parents[3] / "aeroviz-4d" / "src" / "data" / "__tests__" / "fixtures" / "stage_a"
FIXTURE_SET = "fixture_set"


def stage_a_fixture() -> tuple[dict, dict]:
    """A set of one synthetic flight (`support.closed_loop_flight`: a downwind, base and final that ends unstable at the
    DA) and its index, written by the export's own functions: the observed track, the open-loop sentence and the
    closed-loop sentence at Δ = 2, 4, 8 s, each flown again and judged. The observed attitude is the no-airframe one
    (heading and path angle; a synthetic flight has no data-plane series)."""
    import torch

    from ts_transformer.autopilot import replay
    from ts_transformer.tests.support import closed_loop_flight

    flights = {interval: closed_loop_flight(interval) for interval in export.ROW_INTERVALS_S}
    one = flights[2.0]
    assert one.sentence.first_row == 0   # the observed rows from the flight's first: the open-loop sentence's own
    signals, geometry, words = one.signals, one.geometry, one.words
    attitude = {"headingDeg": signals.track_deg,
                "pathAngleDeg": np.degrees(np.arctan2(signals.vertical_rate_mps, signals.ground_speed_mps)),
                "bankRightDeg": None, "attackDeg": None}
    flight = export.flight_head(signals, f"{signals.dataset_id}_fixture", "train", "vectored", "without go-around",
                                "own", one.reading, geometry, attitude, words)
    for interval, item in flights.items():
        flown, (verdict,) = replay.fly_batch(item.batch, item.params, words, device=torch.device("cpu"))
        replayed = export.replay_payload(flown, 0, verdict, item.batch, item.sentence, {"outcome": verdict.outcome},
                                         item.inputs.aero_params[0].numpy(), words.spec, words)
        flight["closedLoop"][f"{interval:g}"] = export.closed_loop_payload(item.sentence, replayed, interval, geometry,
                                                                              words)
    source = {"instructions": "fixture/instruction_language", "executor": "fixture/executor", "specSha256": "fixture",
              "executorSpecSha256": "fixture", "git": {"head": "fixture", "dirty": False}}
    cohort = {"splits": {"train": 1, "select": 0}, "perStratum": 1, "strata": list(export.STRATA), "seed": 1337,
              "drawnFrom": "a synthetic flight"}
    sample = export.sample_of(FIXTURE_SET, geometry, {c.ident: -33.0 for c in geometry.candidates}, source, cohort,
                              words, [flight])
    index = {"schema": files.INDEX_SCHEMA, "writtenUtc": "fixture", "airport": geometry.code,
             "sets": [export.index_entry(FIXTURE_SET, sample)]}
    return index, sample


def test_the_frontend_fixtures_are_what_the_export_writes():
    """The fixture the frontend's readers are tested on is the export's own output today; a change of the export moves
    it, and this test says so until it is written again (``AEROVIZ_WRITE_FIXTURES=1``)."""
    index, sample = stage_a_fixture()
    texts = {files.INDEX_FILE: json.dumps(index, indent=1) + "\n",
             f"{FIXTURE_SET}/{files.SAMPLE_FILE}": json.dumps(sample, separators=(",", ":"), allow_nan=False) + "\n"}
    if os.environ.get("AEROVIZ_WRITE_FIXTURES") == "1":
        for name, text in texts.items():
            (FIXTURES / name).parent.mkdir(parents=True, exist_ok=True)
            (FIXTURES / name).write_text(text, encoding="utf-8")
    for name, text in texts.items():
        assert (FIXTURES / name).read_text(encoding="utf-8") == text, (
            f"{FIXTURES / name} is not what the export writes now: AEROVIZ_WRITE_FIXTURES=1 writes it again")
    flight = sample["flights"][0]
    assert sorted(flight["closedLoop"]) == ["2", "4", "8"]
    replayed = flight["closedLoop"]["2"]["replay"]
    assert replayed["outcome"] == "unstable_at_minimums" and replayed["crossing"]["decision"]["passed"] is False
    assert any(event["correction"] for event in flight["closedLoop"]["2"]["events"])
