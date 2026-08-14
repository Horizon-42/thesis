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
    assert sample.last_position_update_s == pytest.approx(sample.time_s - 0.25)
    assert sample.last_contact_s == pytest.approx(sample.time_s - 0.1)


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
