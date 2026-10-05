"""Vocabulary A33 (D77): the start rules measured (`experiments/start_rules.py`, R55)."""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.params import START_RULES, ExecutorParams
from ts_transformer.autopilot.speed import approach_speed_ias_mps
from ts_transformer.experiments import start_rules
from ts_transformer.experiments.executor_spec import ROLL_RATE_DEG_S
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import Words
from ts_transformer.tests.support import (
    executor_inputs, fly_legs, instruction_airport, instruction_flight, instruction_spec as spec,
)

#: Eastbound at 70 m/s for 60 s, then a right turn at 3°/s (6° a 2 s row).
TURNING = [(30, 0.0, 70.0, 0.0), (20, 6.0, 70.0, 0.0)]
#: A downwind, a base and a final on a 3° descent, ending 400 m before runway 09's threshold.
LEGS = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
        (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]


def _batch(flight, rows):
    """A batch at Δ = 4 s whose flights' first predicted steps are ``rows``: each sentence's first row 16 s before."""
    geometry = instruction_airport()
    before = closed_loop.start_row(4.0) * 2
    return SimpleNamespace(observed=[flight] * len(rows), row_interval_s=4.0,
                           readings=[SimpleNamespace(words=np.zeros((flight.n_rows - 3, 5)))] * len(rows),
                           sentences=[SimpleNamespace(first_row=r - before) for r in rows], geometries=[geometry] * len(rows))


def test_the_start_track_against_the_observed_direction_after_a_turning_first_predicted_step():
    """8 s into a turn of 3°/s the observed directions of the 8 s before and after the row differ by 24° (a turning
    flight); the start track by the displacement of the last 2 s is 3° behind the track there, 15° from the direction
    of the 8 s after, and a trailing fit lags more the longer its window. Straight ahead nothing turns; a flight whose
    labelled rows (they end before the landing) end within 8 s of the row is counted apart."""
    flight = instruction_flight(*fly_legs(TURNING, 90.0, 900.0, 0.0, 0.0))
    turn_row = 30 + 4
    errors = {}
    for rule in START_RULES:
        # the labelled rows end 3 rows before the observed ones: 8 s after the last row is past them
        rows = start_rules.turns(_batch(flight, [turn_row, 10, flight.n_rows - 3 - 4]), rule, 2.0)
        assert rows["turn_deg"][0] == pytest.approx(24.0, abs=0.5)
        assert rows["turn_deg"][1] == pytest.approx(0.0, abs=1e-6)
        assert np.isnan(rows["turn_deg"][2]) and np.isnan(rows["start_from_after_deg"][2])
        errors[rule] = rows["start_from_after_deg"][0]
        summary = start_rules.turn_summary(rows)
        assert (summary["flights"], summary["turning"], summary["labelled_rows_end_within_look"]) == (3, 1, 1)
        assert summary["turning_share"] == 0.5 and summary["start_from_after_deg"]["n"] == 1
    assert errors["displacement-2s"] == pytest.approx(15.0, abs=0.5)
    assert errors["displacement-2s"] < errors["trailing-fit-8s"] < errors["trailing-fit-15s"]


def test_a_direction_is_measured_in_metres_on_the_ground():
    """A displacement due north-east in the airport frame's metres points east of 45° on the ground: a metre of the
    frame's east is more ground than a metre of its north away from the frame's latitude (`flights.ground_scale`)."""
    geometry = instruction_airport()
    flight = instruction_flight(np.array([0.0, 1000.0]), np.array([0.0, 1000.0]), np.full(2, 900.0), np.full(2, 45.0),
                                np.full(2, 70.0))
    direction = start_rules.directions_deg(flight, np.array([0]), np.array([1]), geometry)[0]
    from ts_transformer.autopilot.flights import ground_scale

    to_east, to_north = ground_scale(flight, [0], geometry)
    assert direction == pytest.approx(np.degrees(np.arctan2(to_east[0], to_north[0])))
    assert start_rules.wrapped_deg(np.array([350.0 - 10.0, -190.0])).tolist() == [20.0, 170.0]


#: As `LEGS`, but turning onto the base 10 s after the first row: the first predicted step (16 s on) is 6 s into the turn.
TURNING_LEGS = [(5, 0.0, 100.0, 0.0), *LEGS[1:]]


def _read(monkeypatch, tmp_path, parts, name):
    """The runner on two synthetic train flights (no harvest to rebuild them from: their executor inputs at the first
    predicted step by each rule, the A320's approach speed), in ``parts`` parts; its readout and what it was called
    with."""
    one, geometry = spec(), instruction_airport()
    words = Words(one)
    flights = [instruction_flight(*fly_legs(legs, 270.0, 1079.0, -400.0, 0.0), dataset_id=f"KXXX:{k}")
               for k, legs in enumerate((LEGS, TURNING_LEGS))]
    readings = [read_flight(flight, geometry, one, words) for flight in flights]
    params = ExecutorParams(cycle_s=1.0, bank_rate_deg_s=ROLL_RATE_DEG_S, path_time_constant_s=2.0, path_rate_factor=2.0,
                            timeout_factor=1.5, start_rule="trailing-fit-8s")

    def draw(directory, split, spec_, words_, **kw):
        k, n = kw["part"]
        taken = list(range(len(flights)))[k::n] if n == 1 else [k]
        description = {"split": "train", "seed": start_rules.SEED, "per_airport": 0, "groups": [replay.OWN],
                       "threshold_crossing_heights_m": None, "pool": len(taken), "read": len(taken),
                       "flights": len(taken), "excluded": {}, "by_group": {replay.OWN: len(taken)}}
        calls.append(("draw", split, kw["part"]))
        return (replay.Drawn(indices=taken, signals=[flights[i] for i in taken], series=[None] * len(taken),
                             groups=[replay.OWN] * len(taken), geometries={"KXXX": geometry}, description=description,
                             excluded_seen=Counter()), [readings[i] for i in taken])

    def inputs(batch, params_, step_s, device):
        calls.append(("start", params_.start_rule))
        each = [executor_inputs(flight, geometry, row, rule=params_.start_rule)
                for flight, row in zip(batch.observed, closed_loop.first_predicted_rows(batch, step_s))]
        return FlightInputs(*(torch.cat([getattr(i, field) for i in each]) for field in
                              ("initial_state", "aero_params", "frame_params", "max_thrust_n")))

    calls = []
    monkeypatch.setattr(replay, "open_executor", lambda executor, instructions: (
        params, {"sha256": "s", "checks": {"labeller": "stand-in"}}, words))
    monkeypatch.setattr(replay, "draw_readings", draw)
    monkeypatch.setattr(replay, "flight_approach_ias_mps", lambda series, group: approach_speed_ias_mps("A320", 62000.0))
    monkeypatch.setattr(closed_loop, "start_inputs", inputs)
    out = tmp_path / name
    assert start_rules.main(["--instructions", str(tmp_path / "artefact"), "--executor", str(tmp_path / "spec"),
                             "--row-interval-s", "4", "--chunk", "8", "--parts", str(parts), "--out", str(out)]) == 0
    return json.loads((out / "readout.json").read_text()), calls


def test_the_runner_reads_every_rule_on_one_draw_the_same_in_parts(monkeypatch, tmp_path):
    """Each rule of `START_RULES` reads the drawn train flights in closed loop with the spec's parameters but for the
    start rule (the executor's start state taken by that rule); the readout holds every rule's numbers, its parameters'
    sha and the spec's own rule; the draw is the split's, train only; read in two parts it is the same. The flight that
    turns at its first predicted step gives each rule's own start track."""
    got, calls = _read(monkeypatch, tmp_path, 1, "whole")
    assert calls == [("draw", "train", (0, 1)), *(("start", rule) for rule in START_RULES)]
    assert got["schema"] == start_rules.READOUT_SCHEMA and got["split"] == "train" and got["spec_start_rule"] == \
        "trailing-fit-8s"
    assert list(got["rules"]) == list(START_RULES)
    assert len({rule["executor_params_sha256"] for rule in got["rules"].values()}) == len(START_RULES)
    for rule in got["rules"].values():
        assert rule["closed_loop"]["sentences"] == 2 and sum(rule["closed_loop"]["outcomes"].values()) == 2
        assert sum(rule["by_airport"]["KXXX"]["outcomes"].values()) == 2
        assert (rule["start_track"]["flights"], rule["start_track"]["turning"]) == (2, 1)
    lag = {name: rule["start_track"]["start_from_after_deg"]["p50"] for name, rule in got["rules"].items()}
    assert lag["displacement-2s"] < lag["trailing-fit-8s"] < lag["trailing-fit-15s"]
    parted, _ = _read(monkeypatch, tmp_path, 2, "parts")
    assert parted["rules"] == got["rules"] and parted["drawn"] == got["drawn"]
    with pytest.raises(SystemExit):
        start_rules.main(["--instructions", "a", "--executor", "b", "--row-interval-s", "4", "--out",
                          str(tmp_path / "whole")])
