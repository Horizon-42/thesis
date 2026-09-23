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
from ts_transformer.instructions.artefact import (
    labeller_source_sha256, write_candidates, write_sentences, write_signals, write_spec,
)
from ts_transformer.instructions.labeller.read import admit, read_flight
from ts_transformer.instructions.labeller.vertical import tube_bounds
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import COLUMNS, HEADING, UNCHANGED, Words, wrap180
from ts_transformer.repo_layout import REPO_ROOT
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec as spec

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


def _artefact(directory, flights, readings=None):
    """A frozen-artefact directory: val signals, the candidates, a spec measured by this labeller,
    and the val sentences (the flights' own readings unless ``readings`` replaces them)."""
    one = spec()
    directory.mkdir(parents=True)
    write_signals(directory, {"val": flights}, {"note": "test"})
    write_candidates(directory, {"KXXX": instruction_airport()})
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
def test_the_turn_radius_is_the_envelope_s_bank_read_backwards():
    radius = display.turn_radius_m(105.0, 25.0)
    assert radius == pytest.approx(105.0 ** 2 / (9.81 * math.tan(math.radians(25.0))))
    assert float(envelope.bank_deg_from_turn_rate(math.degrees(105.0 / radius), 105.0)) == pytest.approx(25.0)


@pytest.mark.parametrize("turn", [90.0, -60.0, 7.0])
def test_an_arc_leaves_on_the_track_and_ends_on_the_target(turn):
    start, radius = np.array([1000.0, -500.0]), 2000.0
    path = display.arc(start, 45.0, turn, radius)
    assert np.allclose(path[0], start)
    bearing = lambda v: math.degrees(math.atan2(v[0], v[1])) % 360.0  # noqa: E731
    half = math.copysign(abs(turn) / (len(path) - 1) / 2, turn)       # a chord points half a step into the turn
    assert float(wrap180(bearing(path[1] - path[0]) - (45.0 + half))) == pytest.approx(0.0, abs=1e-6)
    assert float(wrap180(bearing(path[-1] - path[-2]) - (45.0 + turn - half))) == pytest.approx(0.0, abs=1e-6)
    side = 45.0 + math.copysign(90.0, turn)                           # the centre lies on the turn's side
    centre = start + radius * np.array([math.sin(math.radians(side)), math.cos(math.radians(side))])
    assert np.allclose(np.hypot(*(path - centre).T), radius)


def test_the_turn_region_lies_between_the_tightest_and_the_widest_bank():
    one = spec()
    region = display.turn_region(np.zeros(2), 0.0, 270.0, 100.0, one)     # a left turn of 90°
    assert region.turn_deg == pytest.approx(-90.0)
    assert region.radius_min_m == pytest.approx(display.turn_radius_m(100.0, one.turn_bank_max_deg))
    assert region.radius_max_m == pytest.approx(display.turn_radius_m(100.0, one.turn_bank_min_deg))
    # a left turn from north ends west of the start, the widest arc further out than the tightest
    (e0, e1), (n0, n1) = region.end.e_m, region.end.n_m
    assert (e0, n0) == pytest.approx((-region.radius_min_m, region.radius_min_m))
    assert (e1, n1) == pytest.approx((-region.radius_max_m, region.radius_max_m))
    assert len(region.outline.e_m) == len(region.inner.e_m) + len(region.outer.e_m) - 1


def test_the_funnel_starts_as_the_turn_end_and_widens_by_the_tolerance():
    one = spec()
    end = display.Line.of([[0.0, 0.0], [0.0, 400.0]])                    # 400 m across a hold heading east
    funnel = display.funnel(end, 90.0, 5000.0, one)
    assert funnel.start_half_width_m == pytest.approx(200.0)
    assert funnel.end_half_width_m == pytest.approx(200.0 + 5000.0 * math.tan(math.radians(one.heading_tolerance_deg)))
    assert (funnel.axis.e_m[0], funnel.axis.n_m[0]) == pytest.approx((0.0, 200.0))
    assert (funnel.axis.e_m[1], funnel.axis.n_m[1]) == pytest.approx((5000.0, 200.0))
    # no turn flown: the funnel opens from the point itself
    assert display.funnel(display.Line.of([[3.0, 4.0]]), 0.0, 1000.0, one).start_half_width_m == 0.0


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
    assert entry["id"] == "instruction_v1" and entry["readingRule"] == READING_RULE
    assert entry["vocabularySha256"] == one.sha256
    assert entry["runwaySha256"] == export.candidates_sha256(instruction_airport())
    assert entry["flights"] == 2 and entry["cohort"]["perStratum"] == 1 and "4 labelled val flights" in entry["cohort"]["drawnFrom"]

    sample = json.loads((training / "instruction_v1" / "sample.json").read_text(encoding="utf-8"))
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
    sample = json.loads((tmp_path / "airports" / "KXXX" / "training" / "instruction_v1" / "sample.json").read_text())
    flight = next(f for f in sample["flights"] if f["stratum"] == "vectored")
    rows = flight["rows"]
    assert flight["flightKey"] == "v1" and flight["callsign"] == "v1" or flight["datasetId"] == _key("V1")
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
    assert turned["turn"]["radiusMinM"] < turned["turn"]["radiusMaxM"]
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
    assert not (tmp_path / "airports" / "KXXX" / "training" / "instruction_v1").exists()


def test_an_index_already_listing_the_set_is_refused_before_anything_is_written(tmp_path):
    _artefact(tmp_path / "artefact", [_straight(_key("S1")), _vectored(_key("V1"))])
    training = tmp_path / "airports" / "KXXX" / "training"
    training.mkdir(parents=True)
    listed = {"schema": export.INDEX_SCHEMA, "writtenUtc": "x", "airport": "KXXX",
              "sets": [{"id": "instruction_v1"}]}
    (training / "index.json").write_text(json.dumps(listed), encoding="utf-8")
    with pytest.raises(SystemExit, match="already lists set instruction_v1"):
        _run(tmp_path)
    assert json.loads((training / "index.json").read_text(encoding="utf-8")) == listed
    assert not (training / "instruction_v1").exists()
