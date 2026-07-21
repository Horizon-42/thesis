"""The arrival event: subject dispatch, the observed fit, and not-established."""

from __future__ import annotations

import math

import pytest

from evaluation.arrival import EstablishedCriteria, arrival_deviation, subject_of
from evaluation.metrics import evaluate_batch, evaluate_record
from evaluation.records import TrajectoryRecord
from final_approach.tests.test_fit import FRAME, synthetic_approach

TARGET_TCH_M = 17.5


def _state(lat, lon, alt, t=0.0):
    return {"t": t, "lat": lat, "lon": lon, "alt": alt, "V": 70.0, "psi": 0.8, "gamma": -0.05, "m": 60000.0}


def observed_record(**kwargs) -> TrajectoryRecord:
    """An observed track in the record contract, target = threshold + published TCH.

    ``tch_m`` overrides the height the aircraft ACTUALLY crossed at; the target stays at
    the published TCH, so passing it produces a known vertical deviation.
    """
    kwargs.setdefault("tch_m", TARGET_TCH_M)
    points = synthetic_approach(**kwargs)
    states = [_state(p.lat, p.lon, p.alt_m, t=float(i)) for i, p in enumerate(points)]
    return TrajectoryRecord(
        source={
            "id": "TEST123",
            "subject": "observed",
            "runway": FRAME.ident,
            "runway_course_deg": FRAME.course_deg,
        },
        initial_state={k: v for k, v in states[0].items() if k != "t"},
        target_state=_state(FRAME.lat, FRAME.lon, FRAME.elevation_m + TARGET_TCH_M),
        final_time_s=states[-1]["t"],
        states=states,
        controls=[],
    )


def solved_record() -> TrajectoryRecord:
    """A solve: it terminates exactly at its target."""
    target = _state(FRAME.lat, FRAME.lon, FRAME.elevation_m + TARGET_TCH_M)
    states = [_state(FRAME.lat - 0.05, FRAME.lon - 0.05, 500.0, t=0.0), {**target, "t": 300.0}]
    return TrajectoryRecord(
        source={"id": "SOLVE1"},
        initial_state={k: v for k, v in states[0].items() if k != "t"},
        target_state=target,
        final_time_s=300.0,
        states=states,
        controls=[{"thrust": 1.0}, {"thrust": 1.0}],
    )


# --- subject dispatch --------------------------------------------------------------

def test_subject_defaults_to_optimized():
    assert subject_of(solved_record()) == "optimized"


def test_an_unknown_subject_raises_rather_than_being_guessed():
    record = solved_record()
    record.source["subject"] = "measured"
    with pytest.raises(ValueError, match="unknown subject"):
        subject_of(record)


def test_a_solve_is_still_measured_at_its_final_state():
    """The regression that matters most: existing numbers must not move."""
    outcome = arrival_deviation(solved_record())
    assert outcome.deviation is not None
    assert not outcome.deviation.extrapolated
    assert outcome.established is None
    assert outcome.deviation.lateral_m == pytest.approx(0.0, abs=1e-6)
    assert outcome.deviation.vertical_m == pytest.approx(0.0, abs=1e-6)
    assert outcome.deviation.lateral_sigma_m is None


def test_a_prediction_takes_the_solve_path_too():
    record = solved_record()
    record.source["subject"] = "predicted"
    assert not arrival_deviation(record).deviation.extrapolated


# --- the observed arrival ----------------------------------------------------------

def test_an_observed_track_is_measured_at_its_extrapolated_crossing():
    """The track stops 325 m short; the crossing is recovered, not read off the end."""
    outcome = arrival_deviation(observed_record())
    assert outcome.established is True
    deviation = outcome.deviation
    assert deviation.extrapolated
    assert deviation.vertical_m == pytest.approx(0.0, abs=0.3)  # it crossed AT the TCH
    assert deviation.lateral_m == pytest.approx(0.0, abs=1.0)
    assert deviation.glidepath_deg == pytest.approx(3.0, abs=0.05)
    assert deviation.extrapolation_m > 200.0


def test_the_final_sample_would_have_been_badly_wrong():
    """Quantifies why the dispatch exists, on the same track."""
    record = observed_record()
    fitted = arrival_deviation(record).deviation.vertical_m
    final = record.states[-1]["alt"] - record.target_state["alt"]
    assert abs(fitted) < 1.0
    assert final > 15.0  # the last sample is still high, by the truncation distance


def test_an_observed_arrival_carries_its_uncertainty():
    deviation = arrival_deviation(observed_record(quantise=True)).deviation
    assert deviation.vertical_sigma_m is not None and deviation.vertical_sigma_m > 0.0
    assert deviation.lateral_sigma_m is not None


def test_a_missing_runway_course_raises_rather_than_guessing():
    record = observed_record()
    del record.source["runway_course_deg"]
    with pytest.raises(ValueError, match="runway_course_deg"):
        arrival_deviation(record)


# --- not established ---------------------------------------------------------------

def test_an_unstabilised_approach_is_not_established_and_says_why():
    outcome = arrival_deviation(observed_record(glidepath_deg=6.0))
    assert outcome.established is False
    assert outcome.deviation is None
    assert "glidepath" in outcome.reason


def test_a_track_too_short_to_fit_is_not_established():
    outcome = arrival_deviation(observed_record(start_along_m=-600.0, end_along_m=-320.0))
    assert outcome.established is False
    assert outcome.deviation is None


def test_established_criteria_are_caller_supplied():
    strict = EstablishedCriteria(glidepath_range_deg=(2.95, 3.05))
    assert arrival_deviation(observed_record(glidepath_deg=3.5), criteria=strict).established is False


def test_the_rms_criterion_tolerates_a_single_blip():
    """A max-residual test threw away 12% of otherwise clean KRDU approaches."""
    record = observed_record()
    record.states[3]["alt"] += 40.0  # one bad sample in a long clean descent
    assert arrival_deviation(record).established is True


# --- batch reporting ---------------------------------------------------------------

def test_an_observed_batch_reports_an_established_rate_not_a_solve_rate():
    records = [observed_record(), observed_record(glidepath_deg=6.0)]
    report = evaluate_batch(records)
    assert report["subject"] == "observed"
    assert report["observed"]["established"] == 1
    assert report["observed"]["not_established"] == 1
    assert report["observed"]["established_rate"] == pytest.approx(0.5)
    # Every observed track "has states", so this would be a meaningless 1.0.
    assert report["solve_rate"] == 1.0
    assert report["measured"] == 1


def test_not_established_is_counted_never_dropped():
    report = evaluate_batch([observed_record(glidepath_deg=6.0)])
    assert report["total"] == 1
    assert report["observed"]["not_established"] == 1
    assert report["trajectories"][0]["violations"] == ["not_established"]
    assert report["trajectories"][0]["established"] is False


def test_a_solved_batch_report_is_unchanged_in_shape():
    report = evaluate_batch([solved_record()])
    assert report["subject"] == "optimized"
    assert "observed" not in report
    assert report["solved"] == 1 and report["successful"] == 1
    assert "marginal" not in report["trajectories"][0]


def test_marginal_is_never_set_for_a_computed_trajectory():
    assert evaluate_record(solved_record()).marginal is False


def test_a_crossing_near_the_gate_edge_is_marked_marginal():
    """On real data this is the majority case, and hiding it overstates the pass rate."""
    # Put the crossing just inside the +6.10 m gate, with quantisation noise for sigma.
    record = observed_record(tch_m=TARGET_TCH_M + 6.0, quantise=True)
    evaluation = evaluate_record(record)
    assert evaluation.deviation.vertical_m == pytest.approx(6.0, abs=1.5)
    assert evaluation.marginal is True
