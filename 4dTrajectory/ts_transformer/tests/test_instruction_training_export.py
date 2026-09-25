"""The Training module's export of instruction sentences, and the display geometry it draws from
(`instructions/display.py`, `experiments/instruction_training_export.py`).

Every write goes into ``tmp_path``: the artefact is built there from synthetic flights, and the
frontend's airports directory is a ``tmp_path`` directory too.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np
import pytest

from ts_transformer.experiments import instruction_training_export as export
from ts_transformer.instructions import display, envelope
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import (
    labeller_source_sha256, write_candidates, write_sentences, write_signals, write_spec,
)
from ts_transformer.instructions.labeller.read import admit, read_flight
from ts_transformer.instructions.labeller.vertical import tube_bounds
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import COLUMNS, HEADING, UNCHANGED, Words, wrap180
from ts_transformer.repo_layout import REPO_ROOT
from ts_transformer.tests.support import (
    fixture_days, fly_legs, instruction_airport, instruction_flight, instruction_spec as spec,
)

#: The set the exporter names after the reading rule by default.
SET_ID = READING_RULE.replace("-", "_")

TRAINING_SAMPLE_TS = REPO_ROOT / "aeroviz-4d" / "src" / "data" / "trainingSample.ts"

#: downwind west, left onto a base south, left onto the final east (course 090): two 90° turns
VECTORED = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
#: on the final from the first row, decelerating, then descending
STRAIGHT = [(40, 0.0, 90.0, 0.0), (30, 0.0, 80.0, 0.0), (110, 0.0, 72.0, -72.0 * np.tan(np.radians(3.0)))]


def _key(callsign: str) -> str:
    """A dataset id in the harvest's shape: airport, then ``id_runway_icao24_landingTime``."""
    return f"KXXX:{callsign}_09_abc123_20260101T000000Z"


def _vectored(identifier: str):
    return instruction_flight(*fly_legs(VECTORED, 270.0, 1110.0, -400.0, 0.0), dataset_id=identifier, split="val")


def _straight(identifier: str):
    return instruction_flight(*fly_legs(STRAIGHT, 90.0, 1200.0, -300.0, 0.0), dataset_id=identifier, split="val")


def _artefact(directory, flights, readings=None, also=()):
    """A frozen-artefact directory: val signals, the candidates (KXXX's, and the same geometry
    under each code of ``also``), a spec measured by this labeller, and the val sentences (the
    flights' own readings unless ``readings`` replaces them)."""
    one = spec()
    directory.mkdir(parents=True)
    write_signals(directory, {"val": flights}, {"note": "test"}, fixture_days())
    geometries = {"KXXX": instruction_airport()}
    for code in also:
        geometries[code] = AirportGeometry.from_dict({**instruction_airport().to_dict(), "code": code})
    write_candidates(directory, geometries)
    write_spec(directory, one, {"n": 1}, {"labeller_source_sha256": labeller_source_sha256(),
                                          "git": {"head": "test", "dirty": False}})
    readings = readings or [read_flight(flight, instruction_airport(), one) for flight in flights]
    write_sentences(directory, "val", one, readings, list(range(len(flights))))
    return one


def _run(tmp_path, *extra: str) -> int:
    return export.main(["--dir", str(tmp_path / "artefact"), "--airports-root", str(tmp_path / "airports"),
                        "--airport", "KXXX", "--per-stratum", "1", *extra])


# ---- the contract with the frontend reader
def _ts_constant(name: str) -> str:
    source = TRAINING_SAMPLE_TS.read_text(encoding="utf-8")
    match = re.search(rf"export const {name}\b[^=]*=\s*(?P<value>[^;]+);", source)
    assert match is not None, f"{name} not found in {TRAINING_SAMPLE_TS}"
    return match.group("value").strip()


def test_the_contract_is_the_frontend_reader_s():
    """The schema names, the reading rule, the six columns IN ORDER and "unchanged" are one
    contract across the two languages; each side refuses the other's file by name if they part."""
    assert json.loads(_ts_constant("TRAINING_INDEX_SCHEMA")) == export.INDEX_SCHEMA
    assert json.loads(_ts_constant("TRAINING_SAMPLE_SCHEMA")) == export.SAMPLE_SCHEMA
    assert json.loads(_ts_constant("TRAINING_READING_RULE")) == READING_RULE
    assert json.loads(_ts_constant("TRAINING_READABLE_SET_KIND")) == export.KIND_READBACK
    assert tuple(re.findall(r'"([^"]+)"', _ts_constant("TRAINING_COLUMNS"))) == COLUMNS
    assert int(_ts_constant("TRAINING_UNCHANGED")) == UNCHANGED
    assert tuple(re.findall(r'"([^"]+)"', _ts_constant("TRAINING_WORD_KINDS"))) == export.WORD_KINDS


# ---- display geometry
def test_a_heading_word_s_rows_run_from_a_lead_after_it_to_the_next_word_s_and_stop_at_the_clearance():
    one, words = spec(), Words(spec())
    flight, geometry = _vectored(_key("V")), instruction_airport()
    reading = read_flight(flight, geometry, one, words)
    admitted = admit(flight, geometry, one)
    envelopes = display.flight_envelopes(admitted, reading, one, words)
    said = sorted((i for i in reading.instructions if i.column == HEADING), key=lambda item: item.row)
    assert [h.word for h in envelopes.heading] == said and len(said) > 2          # a turn said step by step
    lead = one.rows_exact(one.heading_lead_s)
    track, tolerance = admitted.smoothed.track_deg, one.heading_tolerance_deg
    for number, (item, check) in enumerate(zip(envelopes.heading, reading.checks["heading"])):
        band = item.band
        following = said[number + 1].row + lead if number + 1 < len(said) else reading.join_row
        assert band.first_row == item.word.row + lead
        assert band.stop_row == max(band.first_row, min(following, reading.join_row))
        assert item.check is check and (band.stop_row - band.first_row, int(band.inside.sum())) == (check["rows"], check["inside"])
        # each row's verdict is the track against θ ± the tolerance, circularly
        rows = slice(band.first_row, band.stop_row)
        assert list(band.inside) == list(np.abs(wrap180(track[rows] - item.target_deg)) <= tolerance)
        assert band.band_deg == pytest.approx((band.target_on_track_deg - tolerance, band.target_on_track_deg + tolerance))
        assert float(wrap180(band.target_on_track_deg - item.target_deg)) == pytest.approx(0.0, abs=1e-9)
        if band.stop_row > band.first_row:
            assert abs(band.target_on_track_deg - track[band.first_row]) <= 180.0
    # the observed track is inside its words by construction, and the rows tile the approach to the clearance
    assert all(h.band.inside.all() for h in envelopes.heading)
    judged = [h.band for h in envelopes.heading if h.band.stop_row > h.band.first_row]
    assert all(a.stop_row == b.first_row for a, b in zip(judged, judged[1:])) and judged[-1].stop_row == reading.join_row


def test_the_row_verdicts_are_the_envelope_s_own_check_asked_row_by_row():
    # rows 1–6: 4.5° off counts inside, a whole turn off is the same heading (circular)
    track = np.array([88.0, 91.0, 94.5, 99.4, 86.0, 360.0 + 91.0, 90.0])
    flags = display.rows_inside(track, 90.0, 1, 7, 4.5)
    assert [bool(flag) for flag in flags] == [True, True, False, True, True, True]
    assert int(flags.sum()) == envelope.heading_words_inside(track, [(0, 90.0)], 1, 7, 4.5)[0]["inside"]
    assert len(display.rows_inside(track, 90.0, 3, 3, 4.5)) == 0
    band = display.heading_band(track + 360.0, 0, 90.0, 1, 7, 4.5, {"rows": 6, "inside": 5})
    assert band.target_on_track_deg == pytest.approx(450.0) and band.band_deg == pytest.approx((445.5, 454.5))
    with pytest.raises(ValueError, match="5 of 6 rows inside its band, its check counts 4 of 6"):
        display.heading_band(track, 0, 90.0, 1, 7, 4.5, {"rows": 6, "inside": 4})


def test_a_reading_whose_heading_checks_are_not_its_words_is_refused():
    one, words = spec(), Words(spec())
    flight, geometry = _vectored(_key("V")), instruction_airport()
    reading = read_flight(flight, geometry, one, words)
    admitted = admit(flight, geometry, one)
    checks = dict(reading.checks, heading=reading.checks["heading"][:-1])
    with pytest.raises(ValueError, match="heading checks are not the heading words said"):
        display.flight_envelopes(admitted, replace(reading, checks=checks), one, words)


def test_the_capture_turn_runs_from_the_clearance_to_the_capture_and_a_straight_in_has_none():
    one, words = spec(), Words(spec())
    geometry = instruction_airport()
    vectored, straight = _vectored(_key("V")), _straight(_key("S"))
    reading = read_flight(vectored, geometry, one, words)
    admitted = admit(vectored, geometry, one)
    capture = display.flight_envelopes(admitted, reading, one, words).capture_turn
    assert (capture.start_row, capture.end_row) == (reading.join_row, reading.capture_row)
    assert capture.check is reading.checks["capture_turn"]
    assert float(wrap180(capture.course_on_track_deg - geometry.candidates[0].course_deg)) == pytest.approx(0.0, abs=1e-9)
    assert abs(capture.course_on_track_deg - admitted.smoothed.track_deg[capture.start_row]) <= 180.0
    reading = read_flight(straight, geometry, one, words)
    envelopes = display.flight_envelopes(admit(straight, geometry, one), reading, one, words)
    assert reading.capture_row == 0 and envelopes.capture_turn is None
    # on the final from row 0: the one heading word is cleared at once, its rows the lead carries past the clearance
    (heading,) = envelopes.heading
    assert heading.band.stop_row == heading.band.first_row and heading.check["rows"] == 0


def test_the_flight_envelopes_follow_the_labeller():
    one, words = spec(), Words(spec())
    flight, geometry = _vectored(_key("V")), instruction_airport()
    reading = read_flight(flight, geometry, one, words)
    admitted = admit(flight, geometry, one)
    envelopes = display.flight_envelopes(admitted, reading, one, words)
    # the tubes are `tube_bounds` itself, with the labeller's own verdict
    for tube, (word, end, low, high) in zip(envelopes.altitude, tube_bounds(reading.instructions, admitted.smoothed.distance_m,
                                                                            admitted.smoothed.altitude_m, one, words)):
        assert (tube.word, tube.end_row) == (word, end) and np.array_equal(tube.lower_m, low) and np.array_equal(tube.upper_m, high)
        assert tube.check["contained"] == bool(tube.inside.all())
    assert envelopes.corridor.rows == len(reading.words) - reading.capture_row
    assert envelopes.corridor.half_width_at_threshold_m == one.corridor_half_width_m
    assert envelopes.speed[-1].target_mps is None and envelopes.speed[-1].range_mps is not None


# ---- the runner
def test_the_export_draws_both_strata_and_adds_its_set_to_the_index(tmp_path):
    flights = [_straight(_key("S1")), _vectored(_key("V1")), _straight(_key("S2")), _vectored(_key("V2"))]
    one = _artefact(tmp_path / "artefact", flights)
    training = tmp_path / "airports" / "KXXX" / "training"
    training.mkdir(parents=True)
    old = {"id": "box_v3", "kind": "vocabulary-readback", "title": "old", "file": "box_v3/sample.json",
           "vocabularySha256": "a" * 64, "runwaySha256": "b" * 64, "readingRule": "box-v3", "flights": 40,
           "cohort": {"split": "val", "perStratum": 20, "seed": 1337, "drawnFrom": "old"}, "source": {"any": 1}}
    (training / "index.json").write_text(json.dumps({"schema": export.INDEX_SCHEMA, "writtenUtc": "x",
                                                      "airport": "KXXX", "sets": [old]}), encoding="utf-8")
    assert _run(tmp_path) == 0

    index = json.loads((training / "index.json").read_text(encoding="utf-8"))
    assert index["schema"] == export.INDEX_SCHEMA and index["sets"][0] == old          # the old set, untouched
    entry = index["sets"][1]
    assert entry["id"] == SET_ID and entry["readingRule"] == READING_RULE
    assert entry["vocabularySha256"] == one.sha256
    assert entry["runwaySha256"] == export.candidates_sha256(instruction_airport())
    assert entry["flights"] == 2 and entry["cohort"]["perStratum"] == 1 and "4 labelled val flights" in entry["cohort"]["drawnFrom"]

    sample = json.loads((training / SET_ID / "sample.json").read_text(encoding="utf-8"))
    assert sample["schema"] == export.SAMPLE_SCHEMA and sample["vocabulary"]["specSha256"] == one.sha256
    assert sample["vocabulary"]["columns"] == list(COLUMNS)
    assert sample["vocabulary"]["headingLeadS"] == one.heading_lead_s
    assert not {"headingMaxTurnDeg", "turnStartDelayMaxS"} & set(sample["vocabulary"])
    assert sorted(f["stratum"] for f in sample["flights"]) == ["straight-in", "vectored"]
    assert sample["cohort"]["pool"] == 4 and sample["candidatesSha256"] == entry["runwaySha256"]
    counts = sample["vocabulary"]["classCounts"]
    assert len(sample["vocabulary"]["headingTargetsDeg"]) == counts["heading"]
    assert len(sample["vocabulary"]["altitudeTargetsM"]) + 1 == counts["altitude"]
    assert len(sample["vocabulary"]["speedTargetsMps"]) + 1 == counts["speed"]
    assert len(sample["vocabulary"]["angleClasses"]) == counts["angle"]

    # a second run refuses: the set's directory exists
    with pytest.raises(SystemExit):
        _run(tmp_path)


def test_one_flight_s_file_holds_its_words_its_envelopes_and_the_geodesy(tmp_path):
    _artefact(tmp_path / "artefact", [_straight(_key("S1")), _vectored(_key("V1"))])
    assert _run(tmp_path) == 0
    sample = json.loads((tmp_path / "airports" / "KXXX" / "training" / SET_ID / "sample.json").read_text())
    flight = next(f for f in sample["flights"] if f["stratum"] == "vectored")
    rows = flight["rows"]
    assert flight["datasetId"] == _key("V1") and flight["flightKey"] == "V1_09_abc123_20260101T000000Z"
    assert flight["callsign"] == "V1" and flight["runway"] == "09" and flight["runwayIndex"] == 0
    signals = flight["signals"]
    assert all(len(signals[k]) == rows for k in ("tS", "eM", "nM", "lon", "lat", "altitudeHaeM"))
    # the lon/lat are the airport frame's own projection of the metres
    frame = instruction_airport().frame
    e, n = frame.horizontal_from_latlon(np.array(signals["lat"]), np.array(signals["lon"]))
    assert np.allclose(e, signals["eM"], atol=0.05) and np.allclose(n, signals["nM"], atol=0.05)
    # one geoid undulation per row, the same for the track and for the tube over that row
    offset = np.array(signals["altitudeHaeM"]) - np.array(signals["raw"]["altitudeM"])
    assert -40.0 < offset.mean() < -25.0                                        # EGM96 near 35° N, 78° W
    for tube in flight["envelopes"]["altitude"]:
        span = slice(tube["row"], tube["endRow"])
        assert np.allclose(np.array(tube["lowerHaeM"]) - np.array(tube["lowerM"]), offset[span], atol=0.02)
        assert sum(tube["inside"]) == tube["check"]["inside"]
    # the words: step 0 complete, events sparse, and the in-force table their forward fill
    events = flight["words"]["events"]
    assert sorted(e["column"] for e in events if e["row"] == 0) == list(range(len(COLUMNS)))
    for column in range(len(COLUMNS)):
        issued = {e["row"]: e["value"] for e in events if e["column"] == column}
        value = None
        for row in range(rows):
            value = issued.get(row, value)
            assert flight["words"]["inForce"][column][row] == value
    # every heading word: its judged rows a lead after it, its band θ ± the tolerance, a verdict per row that its
    # check counts; the words said step by step through the turns
    heading = flight["envelopes"]["heading"]
    lead = sample["vocabulary"]["headingLeadRows"]
    assert lead * sample["vocabulary"]["stepS"] == sample["vocabulary"]["headingLeadS"]
    tolerance = sample["vocabulary"]["headingToleranceDeg"]
    assert [h["kind"] for h in heading] == ["initial"] + ["per-step"] * (len(heading) - 1) and len(heading) > 2
    for item in heading:
        assert item["firstRow"] == item["row"] + lead and item["stopRow"] >= item["firstRow"]
        assert len(item["inside"]) == item["stopRow"] - item["firstRow"] == item["check"]["rows"]
        assert sum(item["inside"]) == item["check"]["inside"]
        assert item["bandDeg"] == pytest.approx([item["targetOnTrackDeg"] - tolerance, item["targetOnTrackDeg"] + tolerance])
        assert not {"turn", "funnel", "holdCheck", "split"} & set(item)
    approach = flight["envelopes"]["approach"]
    assert "interceptInserted" not in approach
    capture = approach["captureTurn"]
    assert (capture["startRow"], capture["endRow"]) == (flight["joinRow"], flight["captureRow"]) and "turn" not in capture
    assert capture["check"]["progressOk"] is True and heading[-1]["stopRow"] <= flight["joinRow"]
    assert len(approach["corridor"]["outline"]["eM"]) == 4
    assert approach["landing"]["cutAtCrossing"] is False and approach["landing"]["crossing"] is None
    speeds = flight["envelopes"]["speed"]
    assert speeds[-1]["targetMps"] is None and speeds[-1]["rangeMps"] is not None
    for span in speeds[:-1]:
        assert sum(span["bandInside"]) == span["check"]["bandInside"]
    assert {c["ident"] for c in sample["candidates"]} == {"09"}
    assert sample["centrelineLengthM"] % 1000 == 0


def test_a_flight_whose_stored_sentence_differs_from_its_reading_stops_the_export(tmp_path):
    flights = [_straight(_key("S1")), _vectored(_key("V1"))]
    one = spec()
    readings = [read_flight(flight, instruction_airport(), one) for flight in flights]
    readings[1].words = readings[1].words.copy()
    readings[1].words[3, HEADING] = 0                                  # a word the flight never said
    _artefact(tmp_path / "artefact", flights, readings)
    with pytest.raises(SystemExit, match="V1_09_abc123_20260101T000000Z: re-read heading word"):
        _run(tmp_path)
    assert not (tmp_path / "airports" / "KXXX" / "training" / SET_ID).exists()


def test_a_refusal_at_a_later_airport_writes_nothing_at_an_earlier_one(tmp_path, monkeypatch):
    """Every airport is built before any is written: a flight the export stops on leaves no set anywhere."""
    _artefact(tmp_path / "artefact", [_straight(_key("S1")), _vectored(_key("V1"))], also=("KYYY",))
    calls = []

    def refuse_on_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise SystemExit("stopped at the second airport")
        return real(*args, **kwargs)

    real = export.draw
    monkeypatch.setattr(export, "draw", refuse_on_second)
    with pytest.raises(SystemExit, match="stopped at the second airport"):
        export.main(["--dir", str(tmp_path / "artefact"), "--airports-root", str(tmp_path / "airports"),
                     "--airport", "KXXX", "--airport", "KYYY", "--per-stratum", "1"])
    assert calls == [1, 1] and not (tmp_path / "airports").exists()


def test_an_index_already_listing_the_set_is_refused_before_anything_is_written(tmp_path):
    _artefact(tmp_path / "artefact", [_straight(_key("S1")), _vectored(_key("V1"))])
    training = tmp_path / "airports" / "KXXX" / "training"
    training.mkdir(parents=True)
    listed = {"schema": export.INDEX_SCHEMA, "writtenUtc": "x", "airport": "KXXX",
              "sets": [{"id": SET_ID}]}
    (training / "index.json").write_text(json.dumps(listed), encoding="utf-8")
    with pytest.raises(SystemExit, match="already lists set " + SET_ID):
        _run(tmp_path)
    assert json.loads((training / "index.json").read_text(encoding="utf-8")) == listed
    assert not (training / SET_ID).exists()
