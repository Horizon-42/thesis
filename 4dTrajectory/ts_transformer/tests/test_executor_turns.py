"""The turn of the executor measured (design §14.2 A13, `experiments/executor_turns.py`): three ways to fly the open-loop
words and the offset each leaves across a turn."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.judge import flown_track
from ts_transformer.experiments import executor_turns
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.words import HEADING, UNCHANGED
from ts_transformer.tests.support import executor_inputs, fly_legs, instruction_flight
from ts_transformer.tests.test_closed_loop import _batch, _params

CPU = torch.device("cpu")


def test_the_exact_model_reaches_each_word_s_track_the_lead_after_it_at_the_observed_speed():
    """Way (c): the track ramps linearly from where it is to the word's track over the lead (4 s), at the observed ground
    speed, from the observed position at row 0; a later word is measured from the word before it."""
    signals = instruction_flight(*fly_legs([(12, 0.0, 100.0, 0.0)], 90.0, 900.0, -4000.0, 0.0))
    grid = np.full((12, 5), UNCHANGED, dtype=np.int16)
    said = [Instruction(HEADING, 0, 0, "said", {"target_deg": 90.0}),
            Instruction(HEADING, 2, 2, "said", {"target_deg": 100.0})]
    sentence = replay.Sentence(grid=grid, instructions=said, first_row=0)
    e, n = executor_turns.exact_positions(sentence, signals, 4.0, 2.0, 1.0)
    t = np.linspace(0.0, 22.0, 22001)
    track = np.radians(np.interp(t, [0.0, 4.0, 8.0, 22.0], [90.0, 90.0, 100.0, 100.0]))
    expected_n = np.concatenate(([0.0], np.cumsum(100.0 * np.cos(track[1:]) * np.diff(t))))
    assert n[-1] - signals.n_m[0] == pytest.approx(expected_n[-1], abs=0.5)
    assert e[0] == signals.e_m[0] and len(e) == 12
    assert e[2] - e[0] == pytest.approx(400.0, abs=1e-6)                    # 4 s due east at 100 m/s


def test_the_turns_of_the_observed_flight_are_its_runs_faster_than_the_onset_rate():
    batch, _, words = _batch()
    signals = batch.signals[0]
    found = executor_turns.turns(signals, len(signals.e_m), words)
    assert len(found) == 2
    (s1, e1, turned1, speed1), (s2, e2, turned2, speed2) = found
    assert turned1 == pytest.approx(-90.0, abs=3.0) and turned2 == pytest.approx(-90.0, abs=3.0)
    assert speed1 == pytest.approx(100.0, abs=1.0) and speed2 == pytest.approx(85.0, abs=1.0) and e1 < s2


def test_three_ways_fly_the_words_at_the_observed_speed_and_read_the_offset_across_each_turn(monkeypatch):
    """Ways (a) and (b) fly at the observed ground speed whatever the speed words say (told 70 m/s on a 100 m/s
    downwind); (b) turns without the stopping-rate limit; the record of each turn reads every way."""
    params = _params()

    def flown_ways(speeds):
        batch, _, words = _batch(speeds=speeds)
        inputs = executor_inputs(batch.signals[0], batch.geometries[0])         # the observed state at row 0
        monkeypatch.setattr(replay.Batch, "inputs", lambda self, device: inputs)
        return batch, words, {
            "executor": executor_turns.fly_way(batch, params, words, stopping=True, device=CPU),
            "executor_no_stopping": executor_turns.fly_way(batch, params, words, stopping=False, device=CPU),
            "exact_words": [executor_turns.exact_positions(batch.sentences[0], batch.signals[0],
                                                           words.spec.heading_lead_s, words.spec.step_s, params.cycle_s)]}

    batch, words, flown = flown_ways({0: 70.0})
    _, _, observed_words = flown_ways(None)
    signals = batch.signals[0]
    states = flown["executor"].states[0, :400]
    assert torch.equal(states, observed_words["executor"].states[0, :400])      # the speed words move nothing
    track = flown_track(states.numpy(), batch.geometries[0])
    observed = np.interp(np.arange(400), signals.time_s, signals.ground_speed_mps)
    assert np.median(np.abs(track["ground_speed"] - observed)) < 0.01           # the observed ground speed
    wanted = {way: flown[way].wanted[0, :400, 0].abs() for way in ("executor", "executor_no_stopping")}
    assert (wanted["executor_no_stopping"] - wanted["executor"]).max() > 0.1     # the stopping rate lifted
    records = executor_turns.turn_rows(batch, flown, words)
    assert len(records) == 2 and all(r["stratum"] == "vectored" for r in records)
    for record in records:
        for way in executor_turns.WAYS:
            assert record[way]["outward_m"] == pytest.approx(-math.copysign(1.0, record["turn_deg"])
                                                             * record[way]["change_m"])
    # information, pinned: the words alone leave 119 m and 141 m across these turns, the executor 108 m and 157 m
    change = [{way: r[way]["change_m"] for way in executor_turns.WAYS} for r in records]
    assert change[0]["exact_words"] == pytest.approx(-118.9, abs=1.0)
    assert change[1]["exact_words"] == pytest.approx(140.9, abs=1.0)
    assert change[0]["executor"] == pytest.approx(-107.5, abs=5.0)
    assert change[1]["executor"] == pytest.approx(157.0, abs=5.0)


def test_the_readout_reads_each_way_by_stratum_and_speed_band():
    def record(part, speed, change, turned=-90.0):
        way = {"change_m": change, "outward_m": -math.copysign(1.0, turned) * change}
        return {"stratum": part, "ground_speed_mps": speed, "turn_deg": turned,
                "executor": way, "executor_no_stopping": way, "exact_words": None}

    records = [record("vectored", 95.0, 100.0), record("vectored", 95.0, -20.0), record("straight-in", 60.0, 10.0)]
    table = executor_turns.readout_table(records)
    cell = table["vectored"]["85-100 m/s"]
    assert cell["turns"] == 2 and cell["executor"]["measured"] == 2 and cell["exact_words"]["not_measured"] == 2
    assert cell["executor"]["abs_change_m"]["p50"] == pytest.approx(60.0)
    assert cell["executor"]["outward_m"]["mean"] == pytest.approx(40.0)     # a left turn: right is outward
    assert table["all"]["all"]["turns"] == 3 and table["straight-in"]["0-70 m/s"]["turns"] == 1
    assert table["vectored"]["100-inf m/s"]["executor"]["abs_change_m"] is None
