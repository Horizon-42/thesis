"""Two-tier v3's relative gate (§3.3 rows A3 and B; D45): `gates.gate_relative` judges a
candidate reading against its baseline on the same flights per seed, and the runner
`executor_relative_gate` intersects the two readings' flights, recomputes both readings there
and names the seed line's source."""

from __future__ import annotations

import json

import pytest

from ts_transformer.experiments import executor_relative_gate as runner
from ts_transformer.experiments.manoeuvre_lockstep import LOCKSTEP_SCHEMA
from ts_transformer.manoeuvre import gates

LINE = {"established_all": 0.05, "established_vectored": 0.10, "vectored_ade_mean_m": 100.0}


def _reading(est_all: float, est_vectored: float, ade: float, flyable: float = 0.99) -> dict[str, float]:
    return {"n": 10, "established_all": est_all, "established_vectored": est_vectored, "established_straight": 0.9,
            "fully_flyable": flyable, "vectored_ade_mean_m": ade, "fde_p50_m": 300.0, "segment_s": 20.0, "executed_s": 20.0}


def test_the_relative_gate_reads_improvements_against_the_seed_line_per_seed():
    baseline = {1337: _reading(0.70, 0.40, 2400.0), 2024: _reading(0.72, 0.45, 2300.0)}
    # both seeds better on all flights, one beyond the line; vectored not worse; ADE within the line
    candidate = {1337: _reading(0.78, 0.42, 2350.0), 2024: _reading(0.75, 0.44, 2320.0)}
    result = gates.gate_relative(baseline, candidate, seed_line=LINE)
    assert result["improvement"]["established_all"] == {1337: pytest.approx(0.08), 2024: pytest.approx(0.03)}
    assert result["improvement"]["vectored_ade_mean_m"] == {1337: pytest.approx(50.0), 2024: pytest.approx(-20.0)}
    assert result["verdicts"]["a3"]["pass"] is True and result["verdicts"]["a3"]["on"] == ["established_all"]
    assert result["verdicts"]["b"]["pass"] is False and result["verdicts"]["b"]["beyond_on"] == []      # beyond on ONE seed only
    # beyond the line on both seeds on one metric, nothing worse: row B passes too
    candidate = {1337: _reading(0.78, 0.42, 2350.0), 2024: _reading(0.80, 0.44, 2320.0)}
    result = gates.gate_relative(baseline, candidate, seed_line=LINE)
    assert result["verdicts"]["b"]["pass"] is True and result["verdicts"]["b"]["beyond_on"] == ["established_all"]
    # the vectored ADE worse beyond its line: row B fails (not worse on every metric), row A3 does not read the ADE,
    # and gate B1 (the truth-fed upper bound, plan v3 §10 item 5: both established not worse, beyond on one metric on
    # both seeds — the ADE may be worse) still passes
    candidate = {1337: _reading(0.78, 0.42, 2550.0), 2024: _reading(0.80, 0.44, 2320.0)}
    result = gates.gate_relative(baseline, candidate, seed_line=LINE)
    assert result["verdicts"]["b"]["pass"] is False and result["verdicts"]["b"]["not_worse"] is False
    assert result["verdicts"]["a3"]["pass"] is True
    assert result["verdicts"]["b1"]["pass"] is True and result["verdicts"]["b1"]["beyond_on"] == ["established_all"]
    assert result["verdicts"]["b1"]["primaries_not_worse"] is True
    # beyond on ONE seed only: gate B1 fails like row B
    candidate = {1337: _reading(0.78, 0.42, 2350.0), 2024: _reading(0.75, 0.44, 2320.0)}
    assert gates.gate_relative(baseline, candidate, seed_line=LINE)["verdicts"]["b1"]["pass"] is False
    # the vectored established share worse beyond its line while all-flights established is far beyond: gate B1 fails
    # (plan v3 §10 item 5: a truth feed that loses established on either group is not an upper bound of anything)
    candidate = {1337: _reading(0.85, 0.25, 2350.0), 2024: _reading(0.85, 0.30, 2320.0)}
    result = gates.gate_relative(baseline, candidate, seed_line=LINE)
    assert result["verdicts"]["b1"]["pass"] is False and result["verdicts"]["b1"]["primaries_not_worse"] is False
    assert result["verdicts"]["b1"]["beyond_on"] == ["established_all"]
    # the other primary worse beyond its line: row A3 fails on that primary
    candidate = {1337: _reading(0.78, 0.25, 2350.0), 2024: _reading(0.80, 0.44, 2320.0)}
    assert gates.gate_relative(baseline, candidate, seed_line=LINE)["verdicts"]["a3"]["pass"] is False
    # not fully flyable on one seed: nothing passes
    candidate = {1337: _reading(0.78, 0.42, 2350.0, flyable=0.9), 2024: _reading(0.80, 0.44, 2320.0)}
    verdicts = gates.gate_relative(baseline, candidate, seed_line=LINE)["verdicts"]
    assert not verdicts["a3"]["pass"] and not verdicts["b"]["pass"] and not verdicts["b1"]["pass"]
    with pytest.raises(ValueError, match="two seeds"):
        gates.gate_relative({1337: baseline[1337]}, candidate, seed_line=LINE)
    with pytest.raises(ValueError, match="missing"):
        gates.gate_relative(baseline, candidate, seed_line={"established_all": 0.05})
    with pytest.raises(ValueError, match="non-negative"):
        gates.gate_relative(baseline, candidate, seed_line={**LINE, "established_all": -0.05})


def _row(established: bool, vectored: bool, ade: float, flyable: bool = True) -> dict:
    return {"ade_m": ade, "fde_m": ade / 2, "chamfer_m": ade / 3, "final_time_error_s": 1.0, "ended": "crossed" if established else "horizon",
            "predictions": 5, "token_refreshes": 5, "prior_landed_at_s": None, "prior_landed_error_s": None,
            "rounds": [{"round": 0, "token_index": 0, "phase": 0, "e_track_m": 50.0}], "at": {"60": 100.0},
            "reference": {"fully_flyable": flyable, "established": established},
            "route_tortuosity": 1.5 if vectored else 1.0, "established_at_anchor": not vectored, "remaining_path_m": 20_000.0}


def _payload(rows: dict, protocol: str = "none", segment_s: float = 20.0, executed_s: float = 20.0) -> dict:
    return {"schema": LOCKSTEP_SCHEMA, "protocol": protocol, "segment_s": segment_s, "executed_s": executed_s,
            "first_prediction": {"rule": "fixed L-1 (row 29)"},
            "executor": f"{protocol}-executor", "executor_sha256": "0" * 64, "executor_name": "twin", "flights": len(rows), "rows": rows}


def test_the_runner_intersects_the_flights_recomputes_both_readings_and_names_the_seed_line(tmp_path, capsys):
    flights = {f"KRDU:F{i}": (i % 2 == 0) for i in range(8)}                       # even = vectored
    baseline_rows = {key: _row(established=(i % 4 == 1), vectored=vectored, ade=2000.0) for i, (key, vectored) in enumerate(flights.items())}
    candidate_rows = {key: _row(established=(i % 2 == 1) or i == 0, vectored=vectored, ade=1800.0) for i, (key, vectored) in enumerate(flights.items())}
    baseline_rows["KRDU:EXTRA"] = _row(established=True, vectored=True, ade=1000.0)  # only the baseline flew it: dropped by the intersection
    dirs = {}
    for side, rows in (("baseline", baseline_rows), ("candidate", candidate_rows)):
        for seed in (1337, 2024):
            d = tmp_path / side / str(seed)
            d.mkdir(parents=True)
            # the candidate is a receding reading (A3-a): a 60 s forecast of which 20 s are flown
            payload = _payload(rows) if side == "baseline" else _payload(rows, segment_s=60.0, executed_s=20.0)
            (d / "manoeuvre_lockstep.json").write_text(json.dumps(payload), encoding="utf-8")
            dirs[(side, seed)] = d
    out = tmp_path / "gate"
    argv = ["--baseline", f"1337={dirs[('baseline', 1337)]}", f"2024={dirs[('baseline', 2024)]}",
            "--candidate", f"1337={dirs[('candidate', 1337)]}", f"2024={dirs[('candidate', 2024)]}",
            "--seed-line", "0.05", "0.10", "100", "--out", str(out)]
    assert runner.main(argv) == 0
    result = json.loads((out / "relative_gate.json").read_text(encoding="utf-8"))
    assert result["flights"]["1337"] == {"baseline": 9, "candidate": 8, "common": 8} and result["seed_line_source"] == "given on the command line"
    assert result["baseline"]["1337"]["n"] == 8 and result["baseline"]["1337"]["established_all"] == pytest.approx(2 / 8)
    assert result["candidate"]["1337"]["established_all"] == pytest.approx(5 / 8) and result["candidate"]["1337"]["vectored_ade_mean_m"] == pytest.approx(1800.0)
    assert result["candidate"]["1337"]["executed_s"] == 20.0 and result["candidate"]["1337"]["segment_s"] == 60.0
    assert result["improvement"]["established_all"]["1337"] == pytest.approx(3 / 8) and result["verdicts"]["a3"]["pass"] is True
    assert result["candidate_sources"]["1337"] == {"dir": str(dirs[("candidate", 1337)]), "executor": "none-executor", "executor_sha256": "0" * 64}
    printed = capsys.readouterr().out
    assert "common 8" in printed and "executed s 20 / 20 vs 20 / 60" in printed and "row A3: PASS" in printed and "row B:  PASS" in printed
    assert "gate B1: PASS" in printed and result["verdicts"]["b1"]["pass"] is True
    # the verdict is never overwritten
    with pytest.raises(SystemExit):
        runner.main(argv)
    # a candidate that starts its closed loop by another rule is refused — for that reason
    other = tmp_path / "candidate_row59"
    other.mkdir()
    payload = _payload(candidate_rows)
    payload["first_prediction"]["rule"] = "fixed row 59 (common start)"
    (other / "manoeuvre_lockstep.json").write_text(json.dumps(payload), encoding="utf-8")
    mismatched = argv[:4] + [f"1337={other}", f"2024={dirs[('candidate', 2024)]}"] + argv[6:-1] + [str(tmp_path / "gate2")]
    with pytest.raises(SystemExit):
        runner.main(mismatched)
    assert "different rules" in capsys.readouterr().err and not (tmp_path / "gate2").exists()
    # a payload of another schema (a reading flown by older code) is refused by name, never read around
    older = tmp_path / "older"
    older.mkdir()
    (older / "manoeuvre_lockstep.json").write_text(json.dumps({**_payload(candidate_rows), "schema": "ts-manoeuvre-lockstep-v2"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        runner.main(argv[:4] + [f"1337={older}", f"2024={dirs[('candidate', 2024)]}"] + argv[6:-1] + [str(tmp_path / "gate3")])
    assert f"is not {LOCKSTEP_SCHEMA!r}" in capsys.readouterr().err
