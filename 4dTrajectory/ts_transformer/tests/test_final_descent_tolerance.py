"""Vocabulary A24 (D66): the readout of the final descent's vertical tolerance (`experiments/final_descent_tolerance.py`)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.speed import approach_speed_ias_mps
from ts_transformer.experiments import final_descent_tolerance as readout
from ts_transformer.instructions.airport import published_glidepath_height_m
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words
from ts_transformer.tests.support import (
    TEST_DA_M, executor_inputs, fly_legs, instruction_airport, instruction_flight, instruction_spec as spec,
)

CPU = torch.device("cpu")
#: A downwind, a base and a final on a 3° descent, ending 400 m before runway 09's threshold 36 m above it: below the
#: DA (`TEST_DA_M`, 60 m), on the published glidepath.
LEGS = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
        (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]


def _final(geometry, offsets_m):
    """Straight-in passes on runway 09's final, 8 km to 50 m before the threshold, each at the published glidepath +
    its offset; the rows of all passes in order and the row each pass starts at."""
    before = np.linspace(8000.0, 50.0, 160)
    glidepath = published_glidepath_height_m(geometry, 0, before) + geometry.candidates[0].elevation_m
    e = np.concatenate([-before] * len(offsets_m))
    altitude = np.concatenate([glidepath + offset for offset in offsets_m])
    n = len(e)
    flight = instruction_flight(e, np.zeros(n), altitude, np.full(n, 90.0), np.full(n, 75.0))
    return flight, [k * len(before) for k in range(len(offsets_m))]


def test_the_observed_decision_reads_the_landing_approach_after_the_last_runway_word():
    """The judge's DA check on the observed track (raw heights): the approach after the last runway word — a pass 30 m
    high, then a go-around and a pass on the glidepath, reads the second; read from the first runway word alone, the
    first pass fails high."""
    geometry, one = instruction_airport(), spec()
    flight, starts = _final(geometry, (30.0, 0.0))
    words = np.full((len(flight.e_m), 5), UNCHANGED, dtype=np.int16)
    words[0, RUNWAY] = 0
    words[starts[1] - 1, RUNWAY] = RUNWAY_GO_AROUND
    words[starts[1], RUNWAY] = 0
    landing = readout.observed_decision(flight, SimpleNamespace(words=words, runway_index=0), geometry, one)
    assert landing["passed"] and landing["row"] > starts[1] and abs(landing["above_glidepath_m"]) < 1.0
    words[starts[1] - 1:, RUNWAY] = UNCHANGED
    first = readout.observed_decision(flight, SimpleNamespace(words=words, runway_index=0), geometry, one)
    assert not first["vertical_ok"] and first["row"] < starts[1] and first["above_glidepath_m"] == pytest.approx(30.0, abs=1.0)
    assert readout.unstable_kind(first) == "high"


def test_the_kinds_of_unstable_and_the_table_by_airport():
    decision = {"vertical_ok": True, "lateral_ok": False, "above_glidepath_m": 3.0, "passed": False}
    assert [readout.unstable_kind(d) for d in (None, {**decision, "vertical_ok": False, "above_glidepath_m": 30.0},
                                               {**decision, "vertical_ok": False, "above_glidepath_m": -30.0},
                                               decision)] == list(readout.UNSTABLE_KINDS)
    passed = {"vertical_ok": True, "lateral_ok": True, "above_glidepath_m": 5.0, "passed": True}
    high = {"vertical_ok": False, "lateral_ok": True, "above_glidepath_m": 25.0, "passed": False}
    base = {"final_descents": 1, "final_descent_rows": 30, "angle_corrections": 3}
    rows = [{**base, "airport": "KAAA", "outcome": "landed", "decision": passed, "observed_decision": passed,
             "final_descent_angle_corrections": 2},
            {**base, "airport": "KAAA", "outcome": "unstable_at_minimums", "decision": high, "observed_decision": passed,
             "final_descent_angle_corrections": 4},
            {**base, "airport": "KBBB", "outcome": "ground_contact", "decision": None, "observed_decision": None,
             "final_descents": 2, "final_descent_angle_corrections": 0}]
    cells = readout.table(rows)
    assert list(cells) == ["KAAA", "KBBB", "all"]
    pooled = cells["all"]
    assert pooled["flights"] == 3 and pooled["landed"] == pytest.approx(1 / 3)
    assert pooled["unstable_at_minimums"] == {"no DA point": 0, "high": 1, "low": 0, "lateral only": 0}
    assert pooled["above_glidepath_at_da_m"]["n"] == 2 and pooled["above_glidepath_at_da_m"]["p50"] == pytest.approx(15.0)
    assert pooled["da_check_observed_vs_replay"] == {"both pass": 1, "observed passes, replay does not": 1,
                                                      "replay passes, observed does not": 0, "neither": 1}
    assert pooled["final_descents"] == 4 and pooled["angle_corrections_per_final_descent"] == pytest.approx(6 / 4)
    assert cells["KBBB"]["angle_corrections_per_final_descent"] == 0.0 and cells["KBBB"]["landed"] == 0.0


def _flown(monkeypatch, final_vertical_m, interval_s):
    """The readout of one row interval on a synthetic flight (no harvest to rebuild it from: its executor inputs at the
    first predicted step, the A320's approach speed)."""
    one, geometry = spec(closed_loop_final_vertical_m=final_vertical_m), instruction_airport()
    words = Words(one)
    flight = instruction_flight(*fly_legs(LEGS, 270.0, 1079.0, -400.0, 0.0))
    reading = read_flight(flight, geometry, one, words)
    drawn = replay.Drawn(indices=[5], signals=[flight], series=[None], groups=[replay.OWN],
                         geometries={flight.airport: geometry}, description={"excluded": {}})
    monkeypatch.setattr(replay, "flight_approach_ias_mps", lambda series, group: approach_speed_ias_mps("A320", 62000.0))
    sentence = replay.sentence_on_interval(reading, flight, interval_s, geometry, words)
    start = sentence.first_row + closed_loop.start_row(interval_s) * int(round(interval_s / one.step_s))
    inputs = executor_inputs(flight, geometry, start)
    monkeypatch.setattr(closed_loop, "start_inputs", lambda part, step_s, device: inputs)
    monkeypatch.setattr(replay.Batch, "inputs", lambda self, device: inputs)
    observed = {5: readout.observed_decision(flight, reading, geometry, one)}
    from ts_transformer.experiments.executor_spec import ROLL_RATE_DEG_S
    from ts_transformer.autopilot.params import ExecutorParams

    params = ExecutorParams(cycle_s=1.0, bank_rate_deg_s=ROLL_RATE_DEG_S, path_time_constant_s=2.0,
                            path_rate_factor=2.0, timeout_factor=1.5)
    return readout.read_interval(drawn, [reading], observed, interval_s, params, words, chunk=8, device=CPU)


@pytest.mark.parametrize("interval_s", [2.0, 4.0, 8.0])
def test_a_flight_read_in_closed_loop_flown_again_and_judged_at_each_row_interval(monkeypatch, interval_s):
    """Each row interval: the closed-loop numbers of the split, and the flight flown again on its stored sentence (it must
    fly its stored states again) and judged; at H_final = 5 m its final descent says angle corrections that H leaves out."""
    rows = {}
    for final_vertical_m in (15.0, 5.0):
        numbers, (row,) = _flown(monkeypatch, final_vertical_m, interval_s)
        assert numbers["sentences"] == 1
        rows[final_vertical_m] = row
    for row in rows.values():
        assert row["dataset_id"] == "KXXX:test" and row["stratum"] == "vectored" and row["go_arounds"] == 0
        assert row["final_descents"] == 1 and row["final_descent_rows"] > 0
        assert row["observed_decision"] is not None
        assert (row["decision"] is not None) == (row["outcome"] in readout.DECIDED)
    assert rows[5.0]["final_descent_angle_corrections"] > rows[15.0]["final_descent_angle_corrections"]


def _readout(h_final, split="select", **changes):
    one = spec(closed_loop_final_vertical_m=h_final).to_dict()
    one.update(changes)
    cells = {"closed_loop": {"outside_the_tolerance": {"vertical": {"share": 0.1}}},
             "table": readout.table([{"airport": "KAAA", "outcome": "landed", "decision": None, "observed_decision": None,
                                      "final_descents": 1, "final_descent_rows": 3, "final_descent_angle_corrections": 1,
                                      "angle_corrections": 2}])}
    return {"schema": readout.READOUT_SCHEMA, "spec": one, "executor_params_sha256": "a", "closed_loop_final_vertical_m": h_final, "split": split,
            "drawn": {"flights": 1}, "row_intervals_s": [2.0], "intervals": {"2": cells}}


def test_readouts_side_by_side_differ_only_in_h_final(tmp_path):
    tables = readout.side_by_side([_readout(5.0), _readout(15.0), _readout(10.0), _readout(15.0, "train")])
    assert list(tables) == ["select", "train"]
    assert list(tables["select"]["intervals"]["2"]) == ["15", "10", "5"]
    text = readout.markdown(tables)
    assert text.index("| 15 | 1 |") < text.index("| 10 | 1 | 100.0 |") < text.index("| 5 | 1 |")
    with pytest.raises(ValueError, match="differ in more than H_final"):
        readout.side_by_side([_readout(5.0), _readout(15.0, closed_loop_lateral_m=20.0)])
    with pytest.raises(ValueError, match="other executor parameters"):
        readout.side_by_side([_readout(5.0), {**_readout(15.0), "executor_params_sha256": "b"}])
    with pytest.raises(ValueError, match="drew other flights"):
        readout.side_by_side([_readout(5.0), {**_readout(15.0), "drawn": {"flights": 2}}])
    with pytest.raises(ValueError, match="two select readouts at one H_final"):
        readout.side_by_side([_readout(5.0), _readout(5.0)])
    for value in (5.0, 15.0):
        (tmp_path / f"h{value:g}").mkdir()
        (tmp_path / f"h{value:g}" / "readout.json").write_text(json.dumps(_readout(value)), encoding="utf-8")
    assert readout.main(["--table", str(tmp_path / "h5"), str(tmp_path / "h15"), "--out", str(tmp_path / "t")]) == 0
    assert (tmp_path / "t" / "table.md").read_text(encoding="utf-8").startswith("## select, Δ = 2 s")
    for extra in (["--split", "train"], ["--per-airport", "400"]):
        with pytest.raises(SystemExit):
            readout.main(["--table", str(tmp_path / "h5"), *extra, "--out", str(tmp_path / "u")])
    assert readout.table([]) == {}


def test_the_da_of_the_test_runway_lies_on_the_test_final():
    """The fixture's DA (above the threshold) is crossed by `_final`'s passes before their last row."""
    geometry = instruction_airport()
    assert published_glidepath_height_m(geometry, 0, 50.0) + 30.0 < TEST_DA_M < published_glidepath_height_m(geometry, 0, 8000.0)
