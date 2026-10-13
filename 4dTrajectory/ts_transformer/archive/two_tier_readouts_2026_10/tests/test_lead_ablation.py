"""Executor design V12: the readout of the words' leads, one column at a time (`experiments/lead_ablation.py`)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.speed import approach_speeds
from ts_transformer.experiments import lead_ablation as readout
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import ANGLE, Words
from ts_transformer.tests.support import (
    executor_inputs, fly_legs, instruction_airport, instruction_flight, instruction_spec as spec,
)

CPU = torch.device("cpu")
#: A downwind, a base and a final on a 3° descent, ending 400 m before runway 09's threshold (`test_final_descent_tolerance`).
LEGS = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
        (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]


def test_a_lead_changes_its_own_field_alone_and_the_spec_refuses_a_lead_off_the_steps():
    base = spec()
    for column, (field, _) in readout.COLUMNS.items():
        led = readout.spec_with_lead(base, column, 6.0)
        assert getattr(led, field) == 6.0
        assert {k: v for k, v in led.to_dict().items() if k != field} == \
            {k: v for k, v in base.to_dict().items() if k != field}
        with pytest.raises(ValueError, match="not a whole number of steps"):
            readout.spec_with_lead(base, column, 3.0)


def _params():
    from ts_transformer.autopilot.params import ExecutorParams
    from ts_transformer.experiments.executor_spec import ROLL_RATE_DEG_S

    return ExecutorParams(cycle_s=1.0, bank_rate_deg_s=ROLL_RATE_DEG_S, thrust_time_constant_s=1.5,
                          bank_time_constant_s=2.0, load_time_constant_s=0.8, timeout_factor=1.5,
                          start_rule="trailing-fit-8s")


def _value(monkeypatch, column, seconds, interval_s=2.0):
    """One value of a lead on a synthetic flight (no harvest to rebuild it from: its executor inputs at the first
    predicted step under that lead's reading, the A320's approach speed)."""
    geometry = instruction_airport()
    flight = instruction_flight(*fly_legs(LEGS, 270.0, 1079.0, -400.0, 0.0))
    drawn = replay.Drawn(indices=[5], signals=[flight], series=[None], groups=[replay.OWN],
                         geometries={flight.airport: geometry}, description={"excluded": {}}, excluded_seen=Counter())
    monkeypatch.setattr(replay, "flight_approach_speeds", lambda series: approach_speeds("A320"))
    monkeypatch.setattr(replay, "flown_typecode", lambda series: "A320")
    led = readout.spec_with_lead(spec(), column, seconds)
    sentence = replay.sentence_on_interval(read_flight(flight, geometry, led, Words(led)), flight, interval_s, geometry,
                                           Words(led))
    start = sentence.first_row + closed_loop.start_row(interval_s) * int(round(interval_s / led.step_s))
    inputs = executor_inputs(flight, geometry, start)
    monkeypatch.setattr(closed_loop, "start_inputs", lambda part, params, step_s, device: inputs)
    return readout.read_value(drawn, spec(), column, seconds, [interval_s], _params(), chunk=8, device=CPU)


@pytest.mark.parametrize("column", list(readout.COLUMNS))
def test_each_value_is_read_again_flown_in_closed_loop_and_its_words_errors_read(monkeypatch, column):
    """Each value: the labeller's reading under that lead (the flight read, the column's envelopes on the observed
    track), the closed loop at the row interval (one sentence, the corrections per sentence, the outcome) and the errors
    around the column's observed words at every offset."""
    cell = _value(monkeypatch, column, 4.0)
    assert cell["lead_s"] == 4.0 and cell["labeller"]["read"] == 1 and not cell["labeller"]["refused"]
    assert cell["labeller"]["envelopes"]["words"] > 0
    part = cell["intervals"]["2"]
    assert part["closed_loop"]["sentences"] == 1 and part["corrections_per_sentence"] is not None
    assert part["landed"] in (0.0, 1.0)
    assert set(part["after_words"]) == {str(o) for o in readout.OFFSETS}
    assert part["after_words"]["0"] is not None and part["after_words"]["0"]["n"] > 0


def test_an_angle_lead_says_the_descent_earlier_in_the_closed_loop_too(monkeypatch):
    """The value read is the value flown: under L_γ = 8 s the flight's observed angle words come 4 rows earlier than at
    L_γ = 0 (the labeller's shift), and the closed loop says them about there (at the row of the matched point where
    the observed aircraft heard each, D42, D45: within a row)."""
    said = {}
    for seconds in (0.0, 8.0):
        captured = {}
        original = closed_loop.read_chunked

        def keep(batch, params, words, *, chunk, device):
            results = original(batch, params, words, chunk=chunk, device=device)
            captured["rows"] = [int(m) for m in np.flatnonzero((results[0].rows.grid[:, ANGLE] >= 0)
                                                                & ~results[0].rows.correction[:, ANGLE])]
            return results

        monkeypatch.setattr(closed_loop, "read_chunked", keep)
        _value(monkeypatch, "angle", seconds)
        monkeypatch.setattr(closed_loop, "read_chunked", original)
        said[seconds] = captured["rows"]
    assert said[0.0][0] == said[8.0][0] == 0                      # the first step says every column
    later = [r for r in said[0.0] if r > 4]
    earlier = [r for r in said[8.0] if r > 0][-len(later):]
    assert later and len(earlier) == len(later) and all(3 <= a - b <= 5 for a, b in zip(later, earlier))


def test_the_runner_refuses_a_value_twice_and_an_unsampled_train_split(capsys):
    for argv in (["--leads-s", "0", "0", "--per-airport", "10"], ["--leads-s", "0", "2", "--per-airport", "0"]):
        with pytest.raises(SystemExit) as exited:
            readout.main(["--instructions", "i", "--executor", "e", "--column", "angle", *argv,
                          "--row-interval-s", "2", "--out", "o"])
        assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "--leads-s holds a value twice" in err and "--per-airport must be positive" in err
