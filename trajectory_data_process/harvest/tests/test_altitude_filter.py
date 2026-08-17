"""Altitude outliers are repaired in the VIEW; the stored track never changes."""

from __future__ import annotations

import json

import pytest

from trajectory_data_process.harvest.altitude_filter import (
    HELD,
    INTERPOLATED,
    detect_altitude_outliers,
    filter_altitude_outliers,
    filtered_track,
)
from trajectory_data_process.harvest.arrivals import load_arrival_flights, write_arrival_records
from trajectory_data_process.harvest.czml import observed_czml_flights
from trajectory_data_process.harvest.store import HarvestPaths, iter_records, read_track_view
from trajectory_data_process.harvest.tests.test_arrivals import (
    _airport,
    _source_track,
    _write_source_manifest,
)

# A 1 Hz descent at a realistic 8 m/s, which is what the policy was measured against.
def _descent(count: int = 8, *, start_alt: float = 1000.0) -> list[list[float]]:
    return [
        [float(i), 0.30 - i * 0.045, 0.0, start_alt - 8.0 * i]
        for i in range(count)
    ]


def test_an_unreachable_single_sample_is_replaced_by_time_interpolation():
    samples = _descent()
    samples[4][3] = 20147.0

    result = filter_altitude_outliers(samples)

    assert [outlier.index for outlier in result.outliers] == [4]
    assert result.outliers[0].method == INTERPOLATED
    assert result.outliers[0].observed_alt_m == 20147.0
    # The neighbours bracket it in time, so the replacement lands back on the descent.
    assert result.samples[4][3] == pytest.approx(968.0)


def test_repair_keeps_the_sample_count_times_and_horizontal_positions():
    samples = _descent()
    samples[4][3] = 20147.0

    result = filter_altitude_outliers(samples)

    assert len(result.samples) == len(samples)
    assert [row[0] for row in result.samples] == [row[0] for row in samples]
    assert [row[1:3] for row in result.samples] == [row[1:3] for row in samples]


def test_the_caller_s_samples_are_never_mutated():
    samples = _descent()
    samples[4][3] = 20147.0

    filter_altitude_outliers(samples)

    assert samples[4][3] == 20147.0


def test_adjacent_outliers_interpolate_across_the_whole_run():
    samples = _descent()
    samples[3][3] = 5000.0
    samples[4][3] = -4000.0

    result = filter_altitude_outliers(samples)

    assert [outlier.index for outlier in result.outliers] == [3, 4]
    # Both are repaired from samples 2 and 5, never from each other.
    assert result.samples[3][3] == pytest.approx(976.0)
    assert result.samples[4][3] == pytest.approx(968.0)


def test_an_outlier_at_the_track_edge_holds_rather_than_extrapolating():
    samples = _descent()
    samples[0][3] = 9000.0

    result = filter_altitude_outliers(samples)

    assert [outlier.index for outlier in result.outliers] == [0]
    assert result.outliers[0].method == HELD
    assert result.samples[0][3] == pytest.approx(992.0)


def test_a_real_descent_across_a_coverage_gap_is_not_an_outlier():
    # Measured shape: reception resumes 10 s later, 107 m lower. That is 10.7 m/s and the
    # aircraft never came back up, so it is flight, not a needle.
    samples = [
        [0.0, 0.30, 0.0, 1303.0],
        [10.2, 0.20, 0.0, 1196.3],
        [11.2, 0.19, 0.0, 1188.7],
        [12.2, 0.18, 0.0, 1181.1],
    ]

    assert detect_altitude_outliers(samples) == ()


def test_the_reporting_lattices_are_not_outliers():
    # 25 ft (7.62 m) and 100 ft (30.5 m) steps, which every stored altitude sits on.
    samples = [[float(i), 0.3 - i * 0.01, 0.0, alt] for i, alt in enumerate(
        [868.7, 838.2, 868.7, 838.2, 807.7, 838.2, 807.7, 800.1, 792.5]
    )]

    assert detect_altitude_outliers(samples) == ()


def test_a_track_with_no_retained_sample_refuses_to_invent_one():
    samples = [[0.0, 0.3, 0.0, 0.0], [0.1, 0.29, 0.0, 1000.0]]

    with pytest.raises(ValueError, match="no retained neighbour"):
        filter_altitude_outliers(samples)


def test_filtered_track_reports_the_repair_and_leaves_the_record_alone():
    samples = _descent()
    samples[4][3] = 20147.0
    track = {"flight_key": "T", "samples": samples}

    view = filtered_track(track)

    assert view["altitude_filter"]["outlier_count"] == 1
    assert view["altitude_filter"]["outliers"][0]["index"] == 4
    assert track["samples"][4][3] == 20147.0
    assert "altitude_filter" not in track


# ── the seams: every derived view is filtered, the stored bytes are not ───────────

def _harvest_with_a_spike(tmp_path) -> tuple[HarvestPaths, dict]:
    paths = HarvestPaths(tmp_path, "KAAA")
    samples = _descent()
    samples[4][3] = 20147.0
    row = _source_track(
        paths,
        callsign="ARR1",
        icao24="aaa001",
        runway="18",
        samples=samples,
        landing_sample_index=6,
    )
    _write_source_manifest(paths, [row])
    return paths, row


def test_the_observed_czml_layer_renders_repaired_altitudes(tmp_path):
    paths, _ = _harvest_with_a_spike(tmp_path)

    flight = next(iter(observed_czml_flights(paths)))

    assert flight["waypoints"][4][3] == pytest.approx(968.0)
    assert flight["altitude_outliers"] == 1


def test_the_arrival_roster_hashes_the_source_and_serves_the_filtered_slice(tmp_path):
    paths, row = _harvest_with_a_spike(tmp_path)

    manifest = write_arrival_records(_airport(), paths)

    assert manifest["altitude_filter"]["repaired_samples"] == 1
    assert manifest["altitude_filter"]["repaired_records"] == 1
    assert manifest["records"][0]["altitude_outliers"] == 1

    # The hash was taken over the source bytes, so the loader's integrity check passes and
    # the training waypoints still come back repaired.
    flights = load_arrival_flights(paths.airport)
    first = manifest["records"][0]["first_sample_index"]
    assert flights[0]["waypoints"][4 - first][3] == pytest.approx(968.0)

    stored = json.loads((paths.tracks / manifest["records"][0]["source_file"]).read_text())
    assert stored["samples"][4][3] == 20147.0
    assert stored["flight_key"] == row["flight_key"]


def test_the_store_separates_the_raw_record_from_the_derived_view(tmp_path):
    paths, row = _harvest_with_a_spike(tmp_path)

    raw = next(iter_records(paths))
    view = read_track_view(paths, row["file"])

    assert raw["samples"][4][3] == 20147.0
    assert view["samples"][4][3] == pytest.approx(968.0)
