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
from trajectory_data_process.harvest.tracks import Sample, Track


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_current_faa_krdu_observed_pipeline_reuses_one_threshold_event(tmp_path):
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
        ))
    classified = classify_track(Track("abc123", "PIPE1", tuple(samples)), airport)
    assert classified.runway == "05L"
    assert classified.observed_threshold_event["status"] == "estimated"

    paths = HarvestPaths(tmp_path / "harvest", "KRDU")
    write_tracks([classified], paths, provenance={"test": True})
    arrivals = write_arrival_records(airport, paths)
    assert arrivals["counts"]["included"] == 1
    observed = write_observed_records(airport, paths)
    assert observed["total"] == 1

    report = evaluate_batch(
        iter_observed_records(paths),
        contexts=contexts_for_airport(airport),
        observed_availability=observed["event_availability"],
    )
    row = report["trajectories"][0]
    assert row["lateral_result"] == "pass"
    assert row["vertical_result"] == "indeterminate"
    assert row["verdict"] == "indeterminate"
    assert row["observed_threshold_event"] == classified.observed_threshold_event
    assert report["observed"]["event_denominator"] == 1

    rendered = render_observed_czml(
        paths, frontend_data_root=tmp_path / "frontend"
    )
    assert rendered.flights == 1
    assert rendered.combined_czml.is_file()
