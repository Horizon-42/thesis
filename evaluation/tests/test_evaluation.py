"""Evaluation package: record contract, deviation math, gates, batch metrics, CLI."""

import json
import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation import (  # noqa: E402
    DeviationThresholds,
    compare_to_reference,
    evaluate_batch,
    evaluate_record,
    final_state_deviation,
    load_record,
    load_records,
    percentile,
    record_from_dict,
    resample_by_arc_length,
)
from evaluation.__main__ import main as evaluation_main  # noqa: E402

# 0.001° of latitude on the geokit sphere (WGS84 a) ≈ 111.32 m.
_LAT_STEP_DEG = 0.001
_LAT_STEP_M = math.radians(_LAT_STEP_DEG) * 6378137.0

_TARGET = {"lat": 35.9, "lon": -78.8, "alt": 130.0, "V": 70.0, "psi": 0.8, "gamma": -0.05, "m": 60000.0}


def _payload(*, final_lat=35.9, final_lon=-78.8, final_alt=130.0, final_t=380.0,
             target=_TARGET, source=None):
    """A minimal valid solved record: two states + two aligned controls."""
    first = {"t": 0.0, "lat": 35.6, "lon": -78.5, "alt": 2000.0, "V": 130.0,
             "psi": 1.5, "gamma": -0.05, "m": 60000.0}
    last = {"t": final_t, "lat": final_lat, "lon": final_lon, "alt": final_alt,
            "V": 71.0, "psi": 0.85, "gamma": -0.05, "m": 60000.0}
    control = {"thrust": 1e5, "bank_rad": 0.0, "load_factor": 1.0}
    return {
        "source": source or {"id": "AFR074"},
        "initial_state": {k: v for k, v in first.items() if k != "t"},
        "target_state": target,
        "final_time_s": final_t,
        "states": [first, last],
        "controls": [control, control],
    }


def _failed_payload():
    return {
        "source": {"id": "BAD001"},
        "initial_state": {k: v for k, v in _TARGET.items()},
        "target_state": _TARGET,
        "final_time_s": None,
        "states": [],
        "controls": [],
        "reason": "ValueError: infeasible",
    }


def _manifest(dirpath, *eval_files):
    """The batch roster load_records requires — results[].eval_file names the run's records."""
    (dirpath / "summary.json").write_text(json.dumps({
        "results": [{"eval_file": name} for name in eval_files],
    }), encoding="utf-8")


# ── record contract ───────────────────────────────────────────────────────────

def test_record_round_trip_and_solved_flag():
    record = record_from_dict(_payload())
    assert record.solved
    assert len(record.states) == len(record.controls) == 2
    unsolved = record_from_dict(_failed_payload())
    assert not unsolved.solved
    assert unsolved.reason.startswith("ValueError")


def test_evaluate_batch_accepts_a_one_pass_record_iterator():
    records = (record_from_dict(payload) for payload in [_payload(), _failed_payload()])

    report = evaluate_batch(records)

    assert report["total"] == 2
    assert report["solved"] == 1
    assert [row["id"] for row in report["trajectories"]] == ["AFR074", "BAD001"]


def test_misaligned_controls_raise():
    payload = _payload()
    payload["controls"] = payload["controls"][:1]
    with pytest.raises(ValueError, match="align 1:1"):
        record_from_dict(payload)


def test_solved_record_requires_target():
    payload = _payload()
    payload["target_state"] = None
    with pytest.raises(ValueError, match="target_state"):
        record_from_dict(payload)


def test_state_samples_missing_key_raise():
    payload = _payload()
    del payload["states"][-1]["psi"]
    with pytest.raises(ValueError, match="psi"):
        record_from_dict(payload)


# ── deviation math ────────────────────────────────────────────────────────────

def test_final_state_deviation_lateral_and_vertical():
    record = record_from_dict(
        _payload(final_lat=_TARGET["lat"] + _LAT_STEP_DEG, final_alt=_TARGET["alt"] + 10.0)
    )
    deviation = final_state_deviation(record)
    assert deviation.lateral_m == pytest.approx(_LAT_STEP_M, rel=1e-3)
    assert deviation.vertical_m == pytest.approx(10.0)
    assert deviation.speed_ms == pytest.approx(1.0)
    assert deviation.flight_time_s == pytest.approx(380.0)


def test_heading_deviation_wraps_2pi():
    payload = _payload(target={**_TARGET, "psi": math.pi - 0.05})
    payload["states"][-1]["psi"] = -math.pi + 0.05
    deviation = final_state_deviation(record_from_dict(payload))
    assert deviation.heading_rad == pytest.approx(0.1, abs=1e-9)


# ── gates ─────────────────────────────────────────────────────────────────────

def test_gates_pass_and_fail_each_side():
    thresholds = DeviationThresholds()
    on_target = evaluate_record(record_from_dict(_payload()), thresholds)
    assert on_target.solved and on_target.success and on_target.violations == ()

    wide = evaluate_record(
        record_from_dict(_payload(final_lat=_TARGET["lat"] + 0.002)), thresholds
    )  # ~222 m lateral
    assert not wide.success and "lateral" in wide.violations[0]

    low = evaluate_record(
        record_from_dict(_payload(final_alt=_TARGET["alt"] - 5.0)), thresholds
    )
    assert not low.success and "below" in low.violations[0]

    slightly_high = evaluate_record(
        record_from_dict(_payload(final_alt=_TARGET["alt"] + 5.0)), thresholds
    )  # +5 m is inside the +6.10 m window
    assert slightly_high.success

    too_high = evaluate_record(
        record_from_dict(_payload(final_alt=_TARGET["alt"] + 7.0)), thresholds
    )
    assert not too_high.success and "above" in too_high.violations[0]


def test_unsolved_record_fails_with_reason():
    verdict = evaluate_record(record_from_dict(_failed_payload()))
    assert not verdict.solved and not verdict.success
    assert verdict.violations == ("unsolved",)
    assert verdict.deviation is None
    assert verdict.reason.startswith("ValueError")


# ── batch metrics ─────────────────────────────────────────────────────────────

def test_percentile_linear_interpolation():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert percentile([5.0], 0.95) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        percentile([], 0.95)


def test_batch_rates_and_spreads():
    records = [
        record_from_dict(_payload(source={"id": "OK1"})),
        record_from_dict(_payload(final_t=400.0, source={"id": "OK2"})),
        record_from_dict(_payload(final_lat=_TARGET["lat"] + 0.002, source={"id": "WIDE"})),
        record_from_dict(_failed_payload()),
    ]
    report = evaluate_batch(records)
    assert report["total"] == 4
    assert report["solved"] == 3
    assert report["solve_rate"] == pytest.approx(0.75)
    assert report["successful"] == 2
    assert report["success_rate"] == pytest.approx(0.5)
    assert report["success_rate_among_solved"] == pytest.approx(2.0 / 3.0)
    assert report["final_time_s"]["mean"] == pytest.approx((380.0 + 400.0 + 380.0) / 3.0)
    assert report["lateral_m"]["max"] == pytest.approx(2 * _LAT_STEP_M, rel=1e-3)
    rows = {row["id"]: row for row in report["trajectories"]}
    assert rows["WIDE"]["solved"] and not rows["WIDE"]["success"]
    assert not rows["BAD001"]["solved"] and rows["BAD001"]["reason"].startswith("ValueError")


def test_all_unsolved_batch_has_null_spreads():
    report = evaluate_batch([record_from_dict(_failed_payload())])
    assert report["solve_rate"] == 0.0
    assert report["success_rate_among_solved"] is None
    assert report["lateral_m"] is None and report["final_time_s"] is None


# ── reference comparison ──────────────────────────────────────────────────────

def _path_states(*, lon_offset_deg=0.0, alt_top=1000.0, t_end=100.0, n=5):
    """A straight due-north descending path: lat 35.0→35.1 at fixed lon, alt→130."""
    states = []
    for k in range(n):
        f = k / (n - 1)
        states.append({
            "t": t_end * f, "lat": 35.0 + 0.1 * f, "lon": -78.5 + lon_offset_deg,
            "alt": alt_top + (130.0 - alt_top) * f,
            "V": 100.0, "psi": math.pi / 2, "gamma": -0.05, "m": 60000.0,
        })
    return states


def _reference_payload(**kwargs):
    """An observed-track record: same contract, EMPTY controls."""
    states = _path_states(**kwargs)
    return {
        "source": {"id": "AFR074"},
        "initial_state": {k: v for k, v in states[0].items() if k != "t"},
        "target_state": {k: v for k, v in states[-1].items() if k != "t"},
        "final_time_s": states[-1]["t"],
        "states": states,
        "controls": [],
    }


def test_reference_record_with_empty_controls_validates():
    record = record_from_dict(_reference_payload())
    assert record.solved and record.controls == []


def test_record_can_anchor_states_in_a_sibling_states_file(tmp_path):
    payload = _payload()
    states = payload["states"]
    (tmp_path / "flight_states.json").write_text(
        json.dumps({"simulator_states": states}), encoding="utf-8"
    )
    payload["states"] = []
    payload["states_ref"] = {
        "file": "flight_states.json",
        "key": "simulator_states",
        "start_index": 1,
    }
    payload["initial_state"] = {k: v for k, v in states[1].items() if k != "t"}
    payload["final_time_s"] = states[1]["t"]
    payload["controls"] = [payload["controls"][-1]]
    path = tmp_path / "flight_eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    record = load_record(path)

    assert record.states == states[1:]
    assert record.path == path


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (2, None),
        (1, 3),
        (1, 1),
    ],
)
def test_states_ref_rejects_empty_or_out_of_bounds_slices(tmp_path, start, stop):
    payload = _payload()
    states = payload["states"]
    (tmp_path / "flight_states.json").write_text(
        json.dumps({"simulator_states": states}), encoding="utf-8"
    )
    payload["states"] = []
    payload["controls"] = []
    payload["states_ref"] = {
        "file": "flight_states.json",
        "key": "simulator_states",
        "start_index": start,
    }
    if stop is not None:
        payload["states_ref"]["stop_index"] = stop
    path = tmp_path / "flight_eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="states_ref slice"):
        load_record(path)


def test_unsolved_record_cannot_keep_a_final_time():
    payload = _payload()
    payload["states"] = []
    payload["controls"] = []
    with pytest.raises(ValueError, match="unsolved record.*final_time_s"):
        record_from_dict(payload)


def test_solved_record_requires_final_time():
    payload = _payload()
    payload["final_time_s"] = None
    with pytest.raises(ValueError, match="final_time_s"):
        record_from_dict(payload)


def test_resample_by_arc_length_is_even_and_endpoint_exact():
    states = _path_states(n=4)
    points = resample_by_arc_length(states, n=5)
    assert len(points) == 5
    assert points[0][:2] == (35.0, -78.5) and points[-1][0] == pytest.approx(35.1)
    # even arc spacing on a straight path = even latitude spacing
    lats = [p[0] for p in points]
    assert lats == pytest.approx([35.0, 35.025, 35.05, 35.075, 35.1])
    with pytest.raises(ValueError, match="zero horizontal length"):
        resample_by_arc_length([_path_states()[0], _path_states()[0]])
    with pytest.raises(ValueError, match="n must be"):
        resample_by_arc_length(states, n=1)


def test_horizontal_arc_length():
    from evaluation import horizontal_arc_length_m
    states = _path_states(n=4)  # 0.1° of latitude ≈ 100 × _LAT_STEP_M
    assert horizontal_arc_length_m(states) == pytest.approx(100 * _LAT_STEP_M, rel=1e-3)
    assert horizontal_arc_length_m(states[:1]) == 0.0
    assert horizontal_arc_length_m([]) == 0.0


def test_solved_record_requires_final_time_matching_last_sample():
    payload = _payload()
    payload["final_time_s"] = payload["states"][-1]["t"] + 5.0
    with pytest.raises(ValueError, match="must equal the last state"):
        record_from_dict(payload)


def test_compare_to_reference_measures_offset_and_time_delta(tmp_path):
    # Optimized path parallel to the reference, offset 0.001° of longitude
    # (≈ 111.32·cos(35°) m ≈ 91.2 m), 20 s faster, flown 50 m lower at the top.
    record_payload = _payload()
    record_payload["states"] = _path_states(lon_offset_deg=0.001, alt_top=950.0, t_end=80.0)
    record_payload["controls"] = record_payload["controls"][:1] * len(record_payload["states"])
    record_payload["final_time_s"] = 80.0
    record_payload["reference_file"] = "refs/AFR074_reference_eval.json"
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "AFR074_reference_eval.json").write_text(
        json.dumps(_reference_payload()), encoding="utf-8"
    )
    record_file = tmp_path / "AFR074_eval.json"
    record_file.write_text(json.dumps(record_payload), encoding="utf-8")
    _manifest(tmp_path, "AFR074_eval.json")

    records = load_records(tmp_path)
    assert len(records) == 1 and records[0].reference_file == "refs/AFR074_reference_eval.json"
    from evaluation import load_reference
    reference = load_reference(records[0])
    comparison = compare_to_reference(records[0], reference)

    expected_lateral = math.radians(0.001) * 6378137.0 * math.cos(math.radians(35.05))
    assert comparison.flight_time_delta_s == pytest.approx(-20.0)
    assert comparison.reference_flight_time_s == pytest.approx(100.0)
    assert comparison.path_lateral_m["mean"] == pytest.approx(expected_lateral, rel=1e-2)
    # 50 m low at the start, converging to 0 at the end -> mean_signed ≈ -25 m
    assert comparison.path_vertical_m["mean_signed"] == pytest.approx(-25.0, abs=1.0)
    assert comparison.path_vertical_m["max_abs"] == pytest.approx(50.0, abs=1e-6)


def test_load_reference_rejects_a_different_flight_identity(tmp_path):
    record_payload = _payload()
    record_payload["source"].update(
        icao24="aaa001",
        runway="05L",
        landing_time_utc="2026-01-01T00:00:00Z",
    )
    record_payload["reference_file"] = "reference.json"
    reference_payload = _reference_payload()
    reference_payload["source"].update(
        icao24="bbb002",
        runway="05L",
        landing_time_utc="2026-01-01T00:00:00Z",
    )
    (tmp_path / "record.json").write_text(json.dumps(record_payload), encoding="utf-8")
    (tmp_path / "reference.json").write_text(json.dumps(reference_payload), encoding="utf-8")

    from evaluation import load_reference

    with pytest.raises(ValueError, match="identity"):
        load_reference(load_record(tmp_path / "record.json"))


def test_batch_integrates_reference_rows_and_aggregate(tmp_path):
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "AFR074_reference_eval.json").write_text(
        json.dumps(_reference_payload()), encoding="utf-8"
    )
    failed_reference = _reference_payload()
    failed_reference["source"]["id"] = "BAD001"
    (tmp_path / "references" / "BAD001_reference_eval.json").write_text(
        json.dumps(failed_reference), encoding="utf-8"
    )
    solved = _payload()
    solved["states"] = _path_states(lon_offset_deg=0.0002, t_end=90.0)
    solved["controls"] = solved["controls"][:1] * len(solved["states"])
    solved["final_time_s"] = 90.0
    solved["target_state"] = {k: v for k, v in solved["states"][-1].items() if k != "t"}
    solved["reference_file"] = "references/AFR074_reference_eval.json"
    unsolved = _failed_payload()
    unsolved["reference_file"] = "references/BAD001_reference_eval.json"
    (tmp_path / "a_eval.json").write_text(json.dumps(solved), encoding="utf-8")
    (tmp_path / "b_eval.json").write_text(json.dumps(unsolved), encoding="utf-8")
    _manifest(tmp_path, "a_eval.json", "b_eval.json")

    report = evaluate_batch(load_records(tmp_path))
    rows = {row["id"]: row for row in report["trajectories"]}
    assert rows["AFR074"]["reference"]["flight_time_delta_s"] == pytest.approx(-10.0)
    # The unsolved record still reports the observed baseline duration.
    assert rows["BAD001"]["reference"] == {
        "file": "references/BAD001_reference_eval.json", "flight_time_s": 100.0,
    }
    aggregate = report["reference"]
    assert aggregate["compared"] == 1
    assert aggregate["flight_time_delta_s"]["mean"] == pytest.approx(-10.0)
    assert aggregate["path_lateral_m"]["mean"] > 0.0


def test_batch_without_references_has_null_reference_block():
    report = evaluate_batch([record_from_dict(_payload())])
    assert report["reference"] is None
    assert "reference" not in report["trajectories"][0]


def test_degenerate_solved_record_does_not_abort_the_batch(tmp_path):
    # A rollout truncated at its first step yields a single-sample "solved" record
    # with zero horizontal extent — the batch must SKIP its comparison with a note,
    # not raise out of evaluate_batch and lose the whole report.
    (tmp_path / "references").mkdir()
    for flight_id in ("DGN1", "OK1"):
        reference = _reference_payload()
        reference["source"]["id"] = flight_id
        (tmp_path / "references" / f"{flight_id}_reference_eval.json").write_text(
            json.dumps(reference), encoding="utf-8"
        )
    single = {"t": 0.0, "lat": 35.6, "lon": -78.5, "alt": 2000.0, "V": 130.0,
              "psi": 1.5, "gamma": -0.05, "m": 60000.0}
    degenerate = {
        "source": {"id": "DGN1"},
        "initial_state": {k: v for k, v in single.items() if k != "t"},
        "target_state": _TARGET,
        "final_time_s": 0.0,
        "states": [single],
        "controls": [{"thrust": 1e5, "bank_rad": 0.0, "load_factor": 1.0}],
        "reference_file": "references/DGN1_reference_eval.json",
    }
    healthy = _payload(source={"id": "OK1"})
    healthy["states"] = _path_states(lon_offset_deg=0.0002, t_end=90.0)
    healthy["controls"] = healthy["controls"][:1] * len(healthy["states"])
    healthy["final_time_s"] = 90.0
    healthy["target_state"] = {k: v for k, v in healthy["states"][-1].items() if k != "t"}
    healthy["reference_file"] = "references/OK1_reference_eval.json"
    (tmp_path / "a_eval.json").write_text(json.dumps(degenerate), encoding="utf-8")
    (tmp_path / "b_eval.json").write_text(json.dumps(healthy), encoding="utf-8")
    _manifest(tmp_path, "a_eval.json", "b_eval.json")

    report = evaluate_batch(load_records(tmp_path))
    rows = {row["id"]: row for row in report["trajectories"]}
    assert rows["DGN1"]["solved"] and not rows["DGN1"]["success"]
    assert rows["DGN1"]["reference"]["flight_time_s"] == 100.0
    assert "skipped" in rows["DGN1"]["reference"]["note"]
    assert "flight_time_delta_s" not in rows["DGN1"]["reference"]
    # The healthy record still gets its full comparison + drives the aggregate.
    assert rows["OK1"]["reference"]["flight_time_delta_s"] == pytest.approx(-10.0)
    assert report["reference"]["compared"] == 1


# ── file IO + CLI ─────────────────────────────────────────────────────────────

def test_load_records_directory_and_cli(tmp_path, capsys):
    (tmp_path / "a_eval.json").write_text(json.dumps(_payload()), encoding="utf-8")
    (tmp_path / "b_eval.json").write_text(json.dumps(_failed_payload()), encoding="utf-8")
    (tmp_path / "a_states.json").write_text("{}", encoding="utf-8")  # must be ignored
    _manifest(tmp_path, "a_eval.json", "b_eval.json")

    records = load_records(tmp_path)
    assert len(records) == 2
    assert records[0].path.name == "a_eval.json"

    report_path = tmp_path / "report.json"
    evaluation_main(["--input", str(tmp_path), "--output", str(report_path)])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["total"] == 2 and report["solved"] == 1 and report["successful"] == 1
    out = capsys.readouterr().out
    assert "solve rate" in out and "1/2" in out


def test_load_records_directory_without_a_manifest_raises(tmp_path):
    # No glob mode: guessing a directory's batch membership by filename is exactly
    # how orphan records get counted — a manifest-less directory is an error.
    (tmp_path / "a_eval.json").write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(ValueError, match="no summary.json manifest"):
        load_records(tmp_path)


def test_load_records_follows_the_batch_manifest_and_ignores_orphans(tmp_path):
    # A batch directory is read via summary.json's results[].eval_file roster — a
    # record left behind by an EARLIER batch (different filename) must not be
    # counted into the report (a KRDU report once counted 27 such orphans).
    (tmp_path / "a_eval.json").write_text(json.dumps(_payload()), encoding="utf-8")
    (tmp_path / "b_eval.json").write_text(json.dumps(_failed_payload()), encoding="utf-8")
    (tmp_path / "ORPHAN_eval.json").write_text(json.dumps(_payload()), encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps({
        "total": 2, "solved": 1, "failed": 1,
        "results": [
            {"id": "AFR074", "status": "solved", "eval_file": "a_eval.json"},
            {"id": "BAD001", "status": "failed", "eval_file": "b_eval.json"},
        ],
    }), encoding="utf-8")

    records = load_records(tmp_path)
    assert [r.path.name for r in records] == ["a_eval.json", "b_eval.json"]


def test_load_records_manifest_pointing_at_a_missing_file_fails_loudly(tmp_path):
    # The manifest and the records must come from the same run — a listed file
    # missing on disk is corruption, not something to skip over.
    (tmp_path / "summary.json").write_text(json.dumps({
        "results": [{"id": "AFR074", "status": "solved", "eval_file": "gone_eval.json"}],
    }), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_records(tmp_path)


def test_load_records_manifest_with_no_results_raises(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="lists no records"):
        load_records(tmp_path)


def test_cli_flags_reach_gates_and_established_criteria():
    # Shared by BOTH entry points (cli.py) so the JSON and HTML reports cannot
    # be produced with silently different knobs.
    import argparse

    from evaluation.cli import add_judgement_args, criteria_from_args, thresholds_from_args

    parser = argparse.ArgumentParser()
    add_judgement_args(parser)
    args = parser.parse_args([
        "--fit-window-m", "-8000", "-300",
        "--glidepath-range-deg", "2.5", "3.5",
        "--lateral-max-m", "556.0",
    ])
    criteria = criteria_from_args(args)
    assert criteria.window_m == (-8000.0, -300.0)
    assert criteria.glidepath_range_deg == (2.5, 3.5)
    assert criteria.max_cross_track_m == 400.0          # untouched default
    assert thresholds_from_args(args).lateral_max_m == 556.0
    assert thresholds_from_args(args).vertical_above_max_m == 6.10
