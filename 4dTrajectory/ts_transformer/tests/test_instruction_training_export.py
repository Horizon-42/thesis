"""The Training module's export of instruction sentences, and the display geometry it draws from
(`instructions/display.py`, `experiments/instruction_training_export.py`).

Every write goes into ``tmp_path``: the artefact is built there from synthetic flights, and the
frontend's airports directory is a ``tmp_path`` directory too.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

from ts_transformer.experiments import instruction_training_export as export
from ts_transformer.instructions import display, envelope
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import (
    labeller_source_sha256, write_candidates, write_sentences, write_signals, write_spec,
)
from ts_transformer.instructions.labeller.read import admit, read_flight, turn_ends_at
from ts_transformer.instructions.labeller.vertical import tube_bounds
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import COLUMNS, HEADING, UNCHANGED, Words, wrap180
from ts_transformer.repo_layout import REPO_ROOT
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec as spec

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
    return instruction_flight(*fly_legs(VECTORED, 270.0, 1110.0, -400.0, 0.0), dataset_id=identifier)


def _straight(identifier: str):
    return instruction_flight(*fly_legs(STRAIGHT, 90.0, 1200.0, -300.0, 0.0), dataset_id=identifier)


def _artefact(directory, flights, readings=None, also=()):
    """A frozen-artefact directory: val signals, the candidates (KXXX's, and the same geometry
    under each code of ``also``), a spec measured by this labeller, and the val sentences (the
    flights' own readings unless ``readings`` replaces them)."""
    one = spec()
    directory.mkdir(parents=True)
    write_signals(directory, {"val": flights}, {"note": "test"})
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
def test_the_turn_region_is_the_labeller_s_turn_ends_from_the_issue_point():
    one = spec()
    flight = admit(_vectored(_key("V")), instruction_airport(), one)
    row, target = 70, 180.0                                   # on the downwind, told to turn left onto the base
    region = display.turn_region(flight, row, target, one)
    ends = turn_ends_at(flight, row, target, one)
    start = np.array([flight.signals.e_m[row], flight.signals.n_m[row]])
    assert region.turn_deg == pytest.approx(float(wrap180(target - flight.smoothed.track_deg[row])))
    assert region.slow_finished == ends.finished
    corners = np.column_stack((region.end.e_m, region.end.n_m))
    assert corners == pytest.approx(start + ends.corners)
    # the region runs from the issue point round the fastest turn, and back along the slowest turn
    # begun as late as allowed to the late start: the issue speed × the latest delay along the track
    outline = np.column_stack((region.outline.e_m, region.outline.n_m))
    track = math.radians(float(flight.smoothed.track_deg[row]))
    late = float(flight.smoothed.ground_speed_mps[row]) * one.turn_start_delay_max_s * np.array([math.sin(track),
                                                                                                 math.cos(track)])
    assert outline[0] == pytest.approx(start) and outline[-1] == pytest.approx(start + late)
    assert (region.fast.e_m[-1], region.fast.n_m[-1]) == pytest.approx(tuple(corners[0]))


def test_the_funnel_starts_where_the_turn_may_end_and_widens_by_the_tolerance():
    one = spec()
    tolerance = one.heading_tolerance_deg
    funnel = display.funnel(envelope.hold_funnel(np.array([[0.0, 0.0], [0.0, 400.0]]), 90.0, tolerance, 5000.0),
                            90.0)                                                    # 400 m across a hold heading east
    assert funnel.start_half_width_m == pytest.approx(200.0)
    assert funnel.end_half_width_m == pytest.approx(200.0 + 5000.0 * math.tan(math.radians(one.heading_tolerance_deg)))
    assert (funnel.axis.e_m[0], funnel.axis.n_m[0]) == pytest.approx((0.0, 200.0))
    assert (funnel.axis.e_m[1], funnel.axis.n_m[1]) == pytest.approx((5000.0, 200.0))
    # the start SWEPT along θ: the funnel starts where the turn may end, not ahead of it
    east = np.array(funnel.outline.e_m)
    assert east.min() == pytest.approx(0.0) and east.max() == pytest.approx(5000.0)
    assert {(0.0, 0.0), (0.0, 400.0)} <= {(round(e, 6), round(n, 6)) for e, n in zip(funnel.outline.e_m, funnel.outline.n_m)}
    # no turn flown: the funnel is one cone from the point itself
    cone = display.funnel(envelope.hold_funnel(np.array([[3.0, 4.0]]), 0.0, tolerance, 1000.0), 0.0)
    assert cone.start_half_width_m == 0.0 and len(cone.outline.e_m) == 3


def test_the_flight_envelopes_follow_the_labeller():
    one, words = spec(), Words(spec())
    flight, geometry = _vectored(_key("V")), instruction_airport()
    reading = read_flight(flight, geometry, one, words)
    admitted = admit(flight, geometry, one)
    envelopes = display.flight_envelopes(admitted, reading, one, words)
    heading = [i for i in reading.instructions if i.column == HEADING]
    assert [h.word for h in envelopes.heading] == sorted(heading, key=lambda item: item.row)
    first, turned = envelopes.heading[0], envelopes.heading[1]
    assert first.turn is None and first.turn_check is None and first.hold_start_row == 0
    assert turned.turn is not None and turned.turn_check["departure_row"] == turned.word.row
    assert turned.hold_start_row == turned.turn_check["arrival_row"] == turned.turn_end_row
    # the last hold ends where the labeller's capture turn begins, and the turn region ends on θ
    assert turned.hold_end_row == reading.checks["capture_turn"]["start_row"] == envelopes.capture_turn.start_row
    assert turned.turn.turn_deg == pytest.approx(float(wrap180(180.0 - admitted.smoothed.track_deg[turned.word.row])))
    # a judged hold is drawn over the rows the labeller judged, with the labeller's own funnel
    for item in envelopes.heading:
        if item.hold_check is not None:
            assert (item.hold_start_row, item.hold_end_row) == (item.hold_check["hold_start"], item.hold_check["hold_end"])
            assert item.funnel.end_half_width_m == pytest.approx(item.hold_check["half_width_end_m"])
    assert sum(1 for item in envelopes.heading if item.funnel is not None) == reading.checks["holds"]
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
    heading = flight["envelopes"]["heading"]
    assert heading[0]["turn"] is None and heading[0]["check"] is None
    turned = heading[1]
    assert turned["turn"]["rateMinDegS"] < turned["turn"]["rateMaxDegS"] and turned["turn"]["slowFinished"]
    assert len(turned["turn"]["slowPath"]["eM"]) > len(turned["turn"]["fastPath"]["eM"])
    assert len(turned["turn"]["region"]["lon"]) == len(turned["turn"]["region"]["eM"]) >= 3
    assert turned["check"]["progressOk"] is True
    approach = flight["envelopes"]["approach"]
    assert approach["captureTurn"]["startRow"] == turned["holdEndRow"]
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
