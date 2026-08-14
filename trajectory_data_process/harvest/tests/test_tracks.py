"""Track reconstruction — one pinned test per defect the shipped artifacts showed."""

from __future__ import annotations

import pytest

from trajectory_data_process.harvest.tracks import reconstruct_tracks

KRDU_LAT, KRDU_LON = 35.8776, -78.7875
FT_M = 0.3048


def row(
    t,
    *,
    lat=KRDU_LAT,
    lon=KRDU_LON,
    alt_ft=3000.0,
    ground=False,
    dep="KATL",
    arr="KRDU",
    velocity=70.0,
    lastposupdate=None,
    lastcontact=None,
):
    return {
        "icao24": "abc123",
        "callsign": "TEST123",
        "timestamp": float(t),
        "latitude": lat,
        "longitude": lon,
        "geoaltitude": alt_ft,
        "onground": ground,
        "velocity": velocity,
        "lastposupdate": float(t) - 0.25 if lastposupdate is None else lastposupdate,
        "lastcontact": float(t) - 0.1 if lastcontact is None else lastcontact,
        "estdepartureairport": dep,
        "estarrivalairport": arr,
    }


def approach_rows(start_t=0.0, n=60, *, dep="KATL", arr="KRDU", ground=False):
    """A descending track walking in toward the field."""
    return [
        row(
            start_t + i,
            lat=KRDU_LAT - (n - i) * 0.002,
            alt_ft=3000.0 - i * 20.0,
            ground=ground,
            dep=dep,
            arr=arr,
        )
        for i in range(n)
    ]


def build(rows, **kwargs):
    kwargs.setdefault("altitude_units", "ft")
    return reconstruct_tracks(rows, airport_lat=KRDU_LAT, airport_lon=KRDU_LON, **kwargs)


# --- units ------------------------------------------------------------------------

def test_altitude_units_are_required_and_validated():
    """No default and no sniffing: guessing wrong scales every altitude by 3.28 silently."""
    with pytest.raises(TypeError):
        reconstruct_tracks(approach_rows(), airport_lat=KRDU_LAT, airport_lon=KRDU_LON)
    with pytest.raises(ValueError, match="altitude_units"):
        build(approach_rows(), altitude_units="metres")


def test_feet_are_converted_and_metres_are_not():
    feet = build(approach_rows(), altitude_units="ft")[0]
    metres = build(approach_rows(), altitude_units="m")[0]
    assert feet.samples[0].alt_hae_m == pytest.approx(3000.0 * FT_M)
    assert metres.samples[0].alt_hae_m == pytest.approx(3000.0)


def test_adsb_velocity_and_position_freshness_stay_attached_to_the_sample():
    track = build(approach_rows())[0]
    sample = track.samples[0]
    assert sample.reported_ground_speed_m_s == 70.0
    assert sample.time_s == pytest.approx(sample.last_position_update_s)
    assert sample.last_contact_s == pytest.approx(sample.time_s + 0.15)


def test_held_state_rows_are_collapsed_onto_last_position_update_time():
    rows = approach_rows(n=20)
    held = dict(rows[-1])
    held["timestamp"] = 20.0
    held["lastcontact"] = 19.9
    rows.append(held)

    track = build(rows, min_samples=2)[0]

    assert len(track.samples) == 20
    assert [sample.time_s for sample in track.samples] == pytest.approx(
        [row["lastposupdate"] for row in rows[:-1]]
    )
    assert track.source_integrity is not None
    assert track.source_integrity.held_rows_removed == 1


def test_geoaltitude_change_without_position_update_is_audited_not_refit():
    rows = approach_rows(n=20)
    asynchronous = dict(rows[5])
    asynchronous["timestamp"] = rows[5]["timestamp"] + 0.5
    asynchronous["lastcontact"] = asynchronous["timestamp"] - 0.05
    asynchronous["geoaltitude"] = rows[5]["geoaltitude"] + 25.0
    rows.insert(6, asynchronous)

    track = build(rows, min_samples=2)[0]

    assert len(track.samples) == 20
    assert track.samples[5].alt_hae_m == pytest.approx(rows[5]["geoaltitude"] * FT_M)
    assert track.source_integrity is not None
    assert track.source_integrity.geoaltitude_async_groups == 1
    assert track.source_integrity.held_rows_removed == 1


def test_freshness_filter_uses_both_lastcontact_and_lastposupdate():
    rows = approach_rows(n=20)
    rows[3]["lastcontact"] = rows[3]["timestamp"] - 16.0
    rows[4]["lastposupdate"] = rows[4]["timestamp"] - 16.0

    track = build(rows, min_samples=2)[0]

    assert len(track.samples) == 18
    assert track.source_integrity is not None
    assert track.source_integrity.stale_last_contact_rows == 1
    assert track.source_integrity.stale_position_rows == 1
    assert track.source_integrity.coverage_gap_count == 0


def test_only_final_contiguous_source_position_block_survives():
    first = approach_rows(0.0, 12)
    second = approach_rows(40.0, 12)

    track = build([*first, *second], min_samples=2, max_gap_s=900.0)[0]

    assert track.start_s == pytest.approx(second[0]["lastposupdate"])
    assert len(track.samples) == len(second)
    assert track.source_integrity is not None
    assert track.source_integrity.coverage_gap_count == 1


def test_position_update_time_rewind_starts_a_new_block_without_reordering():
    rows = approach_rows(0.0, 12)
    later = approach_rows(20.0, 12)
    for index, item in enumerate(later):
        item["lastposupdate"] = 5.0 + index

    track = build([*rows, *later], min_samples=2, max_gap_s=900.0)[0]

    assert track.start_s == pytest.approx(5.0)
    assert len(track.samples) == len(later)
    assert track.source_integrity is not None
    assert track.source_integrity.coverage_gap_count == 1


# --- defect 1: a track glued to a later pass --------------------------------------

def test_a_sustained_ground_run_ends_the_flight():
    """The root cause of the 6598 s hole: an aircraft at the gate keeps transmitting, so
    the time-gap rule never fires and the next departure gets glued on."""
    landing = approach_rows(0.0, 40)
    parked = [row(40.0 + i, ground=True) for i in range(120)]
    departure = approach_rows(1000.0, 40)
    tracks = build([*landing, *parked, *departure])
    assert len(tracks) >= 2
    assert all(t.max_gap_s < 100.0 for t in tracks)


def test_a_spatial_crop_can_never_invent_a_discontinuity():
    """Samples outside the radius are removed; only the FINAL contiguous run survives, so
    two passes separated by an out-of-range excursion cannot be spliced together."""
    first_pass = approach_rows(0.0, 30)
    away = [row(100.0 + i, lat=KRDU_LAT + 5.0, alt_ft=30000.0) for i in range(30)]
    second_pass = approach_rows(5000.0, 30)
    tracks = build([*first_pass, *away, *second_pass], max_gap_s=1e9)
    assert tracks, "the final pass must survive"
    for track in tracks:
        assert track.max_gap_s < 100.0


def test_max_gap_s_reports_a_clean_track_as_clean():
    track = build(approach_rows())[0]
    assert track.max_gap_s == pytest.approx(1.0)


def test_a_real_time_gap_still_splits():
    tracks = build([*approach_rows(0.0, 30), *approach_rows(10_000.0, 30)])
    assert len(tracks) == 2


# --- defect 2: one approach split in two ------------------------------------------

def test_filling_in_the_origin_estimate_does_not_split_an_approach():
    """(None,'KRDU') -> ('KATL','KRDU') is the SAME flight with the origin merely
    resolved. The predecessor read it as a different flight and cut the approach."""
    unknown = approach_rows(0.0, 30, dep=None)
    known = approach_rows(30.0, 30, dep="KATL")
    assert len(build([*unknown, *known])) == 1


def test_a_genuine_metadata_change_still_splits():
    first = approach_rows(0.0, 30, dep="KATL", arr="KRDU")
    second = approach_rows(30.0, 30, dep="KRDU", arr="KJFK")
    assert len(build([*first, *second])) == 2


# --- datum ------------------------------------------------------------------------

def test_altitudes_are_kept_as_broadcast_not_converted_to_msl():
    """The harvest is the faithful record; the datum choice belongs to the consumer."""
    track = build(approach_rows())[0]
    assert track.samples[0].alt_hae_m == pytest.approx(3000.0 * FT_M)


# --- general ----------------------------------------------------------------------

def test_samples_without_a_geometric_altitude_are_dropped():
    rows = approach_rows()
    rows[5]["geoaltitude"] = None
    track = build(rows)[0]
    assert len(track.samples) == len(rows) - 1


def test_non_finite_position_or_altitude_rows_are_dropped():
    rows = approach_rows()
    rows[3]["geoaltitude"] = float("nan")
    rows[4]["longitude"] = float("inf")

    track = build(rows)[0]

    assert len(track.samples) == len(rows) - 2


def test_tracks_are_time_ordered_even_from_shuffled_rows():
    rows = approach_rows()
    track = build(list(reversed(rows)))[0]
    times = [s.time_s for s in track.samples]
    assert times == sorted(times)


def test_a_track_shorter_than_min_samples_is_dropped():
    assert build(approach_rows(0.0, 5), min_samples=10) == []


def test_callsign_survives_blank_leading_rows():
    rows = approach_rows()
    rows[0]["callsign"] = "   "
    assert build(rows)[0].callsign == "TEST123"
