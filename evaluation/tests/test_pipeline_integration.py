"""One real-airport observed path through harvest, evaluation, and CZML."""

from __future__ import annotations

import math
from pathlib import Path

from final_approach import Projected

from evaluation import contexts_for_airport, evaluate_batch
from trajectory_data_process.harvest.airports import load_airport
from trajectory_data_process.harvest.arrivals import write_arrival_records
from trajectory_data_process.harvest.classify import classify_track
from trajectory_data_process.harvest.czml import render_observed_czml
from trajectory_data_process.harvest.observed import iter_observed_records, write_observed_records
from trajectory_data_process.harvest.store import HarvestPaths, write_tracks
from trajectory_data_process.harvest.tracks import (
    Sample,
    Track,
    source_timed_final_block,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_current_faa_krdu_observed_pipeline_reuses_one_threshold_event(
    tmp_path, monkeypatch
):
    airport = load_airport(
        "KRDU",
        config_file=REPO_ROOT / "trajectory_data_process/config/runway_thresholds.json",
        cifp_file=REPO_ROOT / "data/CIFP/CIFP_260806/FAACIFP18",
    )
    runway = airport.runway("05L")
    frame = runway.frame("hae")
    slope = math.tan(math.radians(runway.published_glidepath_deg))
    samples = []
    for index, along_m in enumerate(range(-30_000, 0, 250)):
        point = frame.unproject(Projected(
            float(along_m),
            0.0,
            runway.threshold_crossing_height_m - slope * along_m,
        ))
        samples.append(Sample(
            time_s=1_786_492_800.0 + index * 2.5,
            lat=point.lat,
            lon=point.lon,
            alt_hae_m=point.alt_m,
            on_ground=False,
            reported_ground_speed_m_s=200.0,
            last_position_update_s=1_786_492_800.0 + index * 2.5,
            last_contact_s=1_786_492_800.0 + index * 2.5,
        ))
    crossing = frame.unproject(Projected(
        250.0,
        0.0,
        runway.threshold_crossing_height_m - slope * 250.0,
    ))
    samples.append(Sample(
        time_s=1_786_492_800.0 + len(samples) * 2.5,
        lat=crossing.lat,
        lon=crossing.lon,
        alt_hae_m=crossing.alt_m,
        on_ground=False,
        reported_ground_speed_m_s=200.0,
        last_position_update_s=1_786_492_800.0 + len(samples) * 2.5,
        last_contact_s=1_786_492_800.0 + len(samples) * 2.5,
    ))
    source_timed, integrity = source_timed_final_block(samples)
    classified = classify_track(
        Track("abc123", "PIPE1", tuple(source_timed), integrity), airport
    )
    assert classified.runway == "05L"
    assert classified.observed_threshold_event["status"] == "estimated"
    assert classified.observed_threshold_event["method"] == \
        "direct_linear_bracket"
    assert classified.observed_threshold_event["observability"] == \
        "within_observed_support"

    paths = HarvestPaths(tmp_path / "harvest", "KRDU")
    write_tracks([classified], paths, provenance={"test": True})
    arrivals = write_arrival_records(airport, paths)
    assert arrivals["counts"]["included"] == 1
    observed = write_observed_records(airport, paths)
    assert observed["total"] == 1

    def fail_if_evaluation_refits(*_args, **_kwargs):
        raise AssertionError("evaluation must consume the serialized event without refitting")

    monkeypatch.setattr("final_approach.fit_final_segment", fail_if_evaluation_refits)
    monkeypatch.setattr(
        "final_approach.fit.fit_final_segment", fail_if_evaluation_refits
    )
    monkeypatch.setattr(
        "final_approach.assign.fit_final_segment", fail_if_evaluation_refits
    )

    report = evaluate_batch(
        iter_observed_records(paths),
        contexts=contexts_for_airport(airport),
        observed_availability=observed["event_availability"],
    )
    row = report["trajectories"][0]
    assert row["lateral_result"] == "pass"
    assert row["vertical_result"] == "pass"
    # The synthetic icao24 resolves to no airframe, so the baseline speed gate has
    # no stall window: the geometry passes, speed grades indeterminate with the
    # reason named, and the three-gate composite is honestly indeterminate — the
    # real-fleet outcome for an unregistered airframe.
    assert row["speed_result"] == "indeterminate"
    assert row["verdict"] == "indeterminate"
    assert "airframe" in (row.get("reason") or "")
    assert row["bounds"]["vertical_lower_m"] == -22.0
    assert row["bounds"]["vertical_upper_m"] == 22.0
    assert row["observed_threshold_event"] == classified.observed_threshold_event
    assert report["observed"]["event_denominator"] == 1

    rendered = render_observed_czml(
        paths, frontend_data_root=tmp_path / "frontend"
    )
    assert rendered.flights == 1
    assert rendered.combined_czml.is_file()


def test_constrained_optimizer_target_is_graded_against_the_same_threshold(tmp_path):
    """The seam that broke: a CONSTRAINED-IAF record must survive the 1 cm target check.

    ``evaluation.arrival._require_target_agrees_with_runway_data`` gained its POSITION half
    on 2026-08-17 and was validated against observed and prediction records only — there was
    no optimizer comparison tree on disk at the time. The constrained optimizer was the one
    producer that moved its target (onto the procedure document's rendering of the same CIFP
    threshold, 0.05-0.22 m away), so every ``runway_cons`` record was rejected on the first
    row and took the whole sweep down with it. This pins the two ends together: the state
    ``scenario_optimization`` hands the solver IS the state ``evaluation`` measures against.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "4dTrajectory" / "optimization"))
    import scenario_optimization as so
    from aerodynamic_model.common import GeodeticState
    from flight_scenarios.runway_target import threshold_target_state
    from flight_scenarios.scenario import aircraft_for_code

    airport = load_airport(
        "KRDU",
        config_file=REPO_ROOT / "trajectory_data_process/config/runway_thresholds.json",
        cifp_file=REPO_ROOT / "data/CIFP/CIFP_260806/FAACIFP18",
    )
    contexts = contexts_for_airport(airport)
    checked = 0

    for runway in airport.runways:
        try:
            path = so._resolve_procedure_path(
                so.DEFAULT_PROCEDURE_ROOT, "KRDU", runway.ident
            )
        except (ValueError, OSError):
            continue                      # no published RNAV(GPS) procedure for this runway
        import json as _json

        paths = so._iaf_full_paths(_json.loads(path.read_text(encoding="utf-8")))
        if not paths:
            continue
        context = contexts[("KRDU", runway.ident)]
        # The scenario target, built exactly as the arrival manifest builds it.
        target = threshold_target_state(
            "KRDU", runway.ident, aircraft_for_code("A320"), mass_kg=60_000.0,
            published_target={
                "lat": runway.lat,
                "lon": runway.lon,
                "elevation_msl_m": runway.elevation_msl_m,
                "course_deg": runway.course_deg,
                "threshold_crossing_height_m": runway.threshold_crossing_height_m,
                "published_glidepath_deg": runway.published_glidepath_deg,
            },
        )
        if target is None:
            continue    # no published TCH/glidepath — the arrival manifest excludes it too
        assert isinstance(target, GeodeticState)
        checked += 1
        # (a) the constrained path leaves it alone, and only warns about the procedure gap
        gap_m = so._require_procedure_threshold_agrees(target, paths)
        assert gap_m <= so._FRAME_ANCHOR_TOLERANCE_M
        # (b) that same state passes evaluation's authoritative-threshold check
        from geokit import haversine_m

        assert haversine_m(
            target.latitude, target.longitude,
            context.threshold_lat, context.threshold_lon,
        ) <= 0.01

    assert checked >= 4, f"expected KRDU's four in-service runways, checked {checked}"
