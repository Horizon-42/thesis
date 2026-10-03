"""The final-approach read (`experiments/instruction_final_approach`, R49): the glidepath deviation on hand-built
flights, the clearance and "descend to land" shares, and the runner on a tmp artefact (never a live one)."""

import json
import math

import numpy as np
import pytest

from ts_transformer.autopilot.runway_data import VerticalPath
from ts_transformer.instructions.words import ALTITUDE, RUNWAY, UNCHANGED, Words

PATH = VerticalPath(crossing_height_m=15.0, glidepath_deg=3.0)
GLIDE = math.tan(math.radians(3.0))
LAND = 181          # "descend to land" at the fixture spec's 0–5,400 m, 30 m levels


def _straight_in(rows, above_m, start_before_m=20_000.0, speed=75.0):
    """`FlightSignals` onto `instruction_airport`'s runway 09 (threshold at the origin, course 090°, 100 m), flying the
    centreline from ``start_before_m`` at ``above_m`` over the published glidepath, 2 s a row."""
    from ts_transformer.tests.support import instruction_flight

    before = start_before_m - speed * 2.0 * np.arange(rows)
    altitude = 100.0 + PATH.crossing_height_m + before * GLIDE + above_m
    return instruction_flight(-before, np.zeros(rows), altitude, np.full(rows, 90.0), np.full(rows, speed))


def _words(rows, land_row=None):
    grid = np.full((rows, 6), UNCHANGED, dtype=np.int16)
    grid[0, RUNWAY] = 0
    if land_row is not None:
        grid[land_row, ALTITUDE] = LAND
    return grid


RADIUS = 6_380_000.0


def test_the_flat_glidepath_is_the_tch_plus_the_distance_times_its_slope_and_the_straight_line_adds_the_curvature():
    from ts_transformer.experiments.instruction_final_approach import glidepath_heights_m

    heights = glidepath_heights_m(np.array([0.0, 1_000.0, 20_000.0]), PATH, RADIUS)
    flat = [15.0, 15.0 + 1_000.0 * GLIDE, 15.0 + 20_000.0 * GLIDE]
    assert heights["flat"] == pytest.approx(flat)
    assert heights["straight_line"] - heights["flat"] == pytest.approx([0.0, 1e6 / (2 * RADIUS), 4e8 / (2 * RADIUS)])
    assert 4e8 / (2 * RADIUS) == pytest.approx(31.3, abs=0.1)                       # the size at 20 km


def test_the_radius_of_curvature_is_the_meridional_one_along_a_meridian_and_the_prime_vertical_one_across():
    from geokit import wgs84_curvature_radii

    from ts_transformer.experiments.instruction_final_approach import curvature_radius_m

    meridional, prime_vertical = wgs84_curvature_radii(35.0)
    assert curvature_radius_m(35.0, 0.0) == pytest.approx(meridional)
    assert curvature_radius_m(35.0, 270.0) == pytest.approx(prime_vertical)
    assert meridional < curvature_radius_m(35.0, 45.0) < prime_vertical


def test_a_flight_20_m_over_the_glidepath_reads_20_m_after_its_capture_row_outside_the_flare():
    from ts_transformer.experiments.instruction_final_approach import FLARE_EXCLUDED_M, read_sentence
    from ts_transformer.tests.support import instruction_airport

    rows = 134                                         # the last row 50 m before the threshold
    flight = _straight_in(rows, 20.0)
    reading = read_sentence(flight, _words(rows, land_row=30), instruction_airport().candidates[0], PATH, RADIUS,
                            join_row=10, capture_row=40, land_class=LAND)
    before = 20_000.0 - 150.0 * np.arange(rows)
    kept = (np.arange(rows) >= 40) & (before > FLARE_EXCLUDED_M)
    assert len(reading.deviation_m["flat"]) == kept.sum() and reading.deviation_m["flat"] == pytest.approx(20.0)
    assert reading.deviation_m["straight_line"] == pytest.approx(20.0 - before[kept] ** 2 / (2 * RADIUS))
    assert reading.capture_deviation_m["flat"] == pytest.approx(20.0)
    assert reading.capture_before_threshold_m == pytest.approx(20_000.0 - 150.0 * 40)
    assert (reading.land_row, reading.join_row) == (30, 10)
    assert reading.land_before_threshold_m == pytest.approx(20_000.0 - 150.0 * 30)


def test_a_sentence_with_too_few_rows_after_its_capture_is_not_read_on_the_glidepath():
    from ts_transformer.experiments.instruction_final_approach import read_sentence
    from ts_transformer.tests.support import instruction_airport

    rows = 134
    reading = read_sentence(_straight_in(rows, 0.0), _words(rows), instruction_airport().candidates[0], PATH, RADIUS,
                            join_row=0, capture_row=128, land_class=LAND)
    assert len(reading.deviation_m["flat"]) == len(reading.deviation_m["straight_line"]) == 0
    assert (reading.capture_deviation_m, reading.land_row, reading.land_before_threshold_m) == (None, None, None)


def test_the_shares_and_the_lead_of_descend_to_land():
    from ts_transformer.experiments.instruction_final_approach import SentenceReading, summarise

    def reading(rows, join, land, capture, deviation):
        both = {"flat": np.asarray(deviation, dtype=float), "straight_line": np.asarray(deviation, dtype=float) - 25.0}
        return SentenceReading("KXXX", rows, join, land, None if land is None else 9_000.0, capture, both,
                               {k: v[0] for k, v in both.items()} if deviation else None,
                               5_000.0 if deviation else None)

    readings = [reading(100, 40, 30, 50, [10.0, 50.0, 70.0, 0.0, 0.0]),     # said 10 rows before the clearance
                reading(100, 0, 60, 0, [-80.0, 0.0, 0.0, 0.0, 0.0]),        # cleared at row 0
                reading(50, 20, None, 30, [])]                              # never says it, not read
    summary = summarise(readings, 2.0)
    assert summary["clearance"] == {"rows_before_share": pytest.approx(60 / 250), "cleared_at_row_0_share":
                                    pytest.approx(1 / 3)}
    land = summary["descend_to_land"]
    assert (land["sentences"], land["before_clearance_share"], land["before_capture_share"]) == (2, 0.5, 0.5)
    assert land["lead_on_clearance_s"] == {"p10": 20.0, "p50": 20.0, "p90": 20.0}
    assert (summary["glidepath"]["flights"], summary["glidepath"]["rows"]) == (2, 10)
    assert summary["glidepath"]["capture_before_threshold_m"]["p50"] == pytest.approx(5_000.0)
    flat, straight = summary["glidepath"]["flat"], summary["glidepath"]["straight_line"]
    assert flat["rows_within_share"] == {"30": pytest.approx(0.7), "60": pytest.approx(0.8)}
    assert flat["flights_on_glidepath_share"] == pytest.approx(0.0)         # 4/5 within 60 m each, under 0.9
    assert flat["at_capture"]["above_widest_band_share"] == pytest.approx(0.0)
    assert flat["at_capture"]["below_widest_band_share"] == pytest.approx(0.5)
    assert straight["rows_within_share"] == {"30": pytest.approx(0.8), "60": pytest.approx(0.9)}
    assert straight["deviation_m"]["p50"] == pytest.approx(flat["deviation_m"]["p50"] - 25.0)


def test_a_sentence_off_its_runway_is_refused():
    import dataclasses

    from ts_transformer.experiments.instruction_final_approach import read_split
    from ts_transformer.tests.support import instruction_airport, instruction_spec

    flight = dataclasses.replace(_straight_in(20, 0.0), runway="27")
    sentences = {"words": _words(20), "offsets": np.array([0, 20]), "signal_index": np.array([0]),
                 "runway_index": np.array([0]), "join_row": np.array([0]), "capture_row": np.array([0])}
    geometries, paths, words = {"KXXX": instruction_airport()}, {"KXXX": [PATH]}, Words(instruction_spec())
    with pytest.raises(ValueError, match="not the landed"):                       # the flight landed elsewhere
        read_split(sentences, [flight], geometries, paths, words)
    sentences["words"] = sentences["words"].copy()
    sentences["words"][0, RUNWAY] = 1
    with pytest.raises(ValueError, match="row 0 says runway slot 1"):              # row 0 is not the recorded slot
        read_split(sentences, [dataclasses.replace(flight, runway="09")], geometries, paths, words)


def test_the_runner_reads_a_tmp_artefact_and_never_overwrites(tmp_path, monkeypatch):
    from ts_transformer.experiments import instruction_final_approach
    from ts_transformer.tests.support import labelled_instruction_artefact

    directory = tmp_path / "artefact"
    spec = labelled_instruction_artefact(directory)
    monkeypatch.setattr(instruction_final_approach, "published_vertical_paths",
                        lambda geometry: tuple(PATH for _ in geometry.candidates))
    out = tmp_path / "final"
    assert instruction_final_approach.main(["--instructions", str(directory), "--out", str(out)]) == 0
    read = json.loads((out / "final_approach.json").read_text(encoding="utf-8"))
    assert (read["schema"], read["split"], read["spec_sha256"]) == (instruction_final_approach.SCHEMA, "train",
                                                                    spec.sha256)
    assert read["pooled"]["sentences"] == 1 and set(read["airports"]) == {"KXXX"}
    path = read["published_vertical_paths"]["KXXX"]["09"]
    assert (path["crossing_height_m"], path["glidepath_deg"]) == (15.0, 3.0) and path["curvature_radius_m"] > 6.3e6
    # the fixture flies its final on the flat 3° line 30.7 m above the glidepath: main → read_split → summarise
    glide = read["pooled"]["glidepath"]
    assert (glide["flights"], glide["flat"]["deviation_m"]["p10"], glide["flat"]["deviation_m"]["p90"]) == (
        1, pytest.approx(30.7, abs=0.05), pytest.approx(30.7, abs=0.05))
    assert glide["flat"]["rows_within_share"] == {"30": 0.0, "60": 1.0}
    assert glide["straight_line"]["deviation_m"]["p90"] < 30.7
    assert read["pooled"]["descend_to_land"]["before_threshold_m"]["p50"] == pytest.approx(18_100.0, abs=100.0)
    with pytest.raises(SystemExit):
        instruction_final_approach.main(["--instructions", str(directory), "--out", str(out)])
