"""The turn of the executor measured (vocabulary §12.1 A13, `experiments/executor_turns.py`): three ways to fly the open-loop
words and a control, and the offset each leaves across a turn."""

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


def _sentence(rows, said):
    grid = np.full((rows, 5), UNCHANGED, dtype=np.int16)
    return replay.Sentence(grid=grid, instructions=[Instruction(HEADING, 0, row, "said", {"target_deg": target})
                                                    for row, target in said], first_row=0)


def _integrated_n(times, tracks, until_s, speed=100.0):
    t = np.linspace(0.0, until_s, 100001)
    track = np.radians(np.interp(t, times, tracks))
    return float(np.sum(speed * np.cos(0.5 * (track[1:] + track[:-1])) * np.diff(t))), \
        float(np.sum(speed * np.sin(0.5 * (track[1:] + track[:-1])) * np.diff(t)))


def test_the_exact_model_reaches_each_word_s_track_the_lead_after_it_at_the_observed_speed():
    """Way (c): the track ramps linearly from where it is to the word's track over the lead (4 s), at the observed ground
    speed, from the observed position at row 0."""
    signals = instruction_flight(*fly_legs([(12, 0.0, 100.0, 0.0)], 90.0, 900.0, -4000.0, 0.0))
    e, n = executor_turns.exact_positions(_sentence(12, [(0, 90.0), (2, 100.0)]), signals, 4.0, 2.0, 1.0)
    north, _ = _integrated_n([0.0, 4.0, 8.0, 22.0], [90.0, 90.0, 100.0, 100.0], 22.0)
    assert n[-1] - signals.n_m[0] == pytest.approx(north, abs=0.5)
    assert e[0] == signals.e_m[0] and len(e) == 12
    assert e[2] - e[0] == pytest.approx(400.0, abs=1e-6)                    # 4 s due east at 100 m/s


def test_a_word_heard_mid_turn_is_measured_from_the_word_before_it_and_starts_from_the_track_there():
    """§5.4: the first word turns the shorter way from the track (0° → 170°: right); a word heard while the aircraft is
    still turning (at 2 s, the track 85°) starts its ramp from the track there and is measured from the word before it:
    300° is 130° further right of 170°, not 145° left of 85°."""
    signals = instruction_flight(*fly_legs([(12, 0.0, 100.0, 0.0)], 0.0, 900.0, -4000.0, 0.0))
    e, n = executor_turns.exact_positions(_sentence(12, [(0, 170.0), (1, 300.0)]), signals, 4.0, 2.0, 0.1)  # 42°/s
    north, east = _integrated_n([0.0, 2.0, 6.0, 22.0], [0.0, 85.0, 300.0, 300.0], 22.0)
    assert n[-1] - signals.n_m[0] == pytest.approx(north, abs=0.5)
    assert e[-1] - signals.e_m[0] == pytest.approx(east, abs=0.5)


def test_the_turns_of_the_observed_flight_are_its_runs_faster_than_the_onset_rate():
    batch, _, words = _batch()
    signals = batch.signals[0]
    found = executor_turns.turns(signals, len(signals.e_m), words)
    assert len(found) == 2
    (s1, e1, turned1, speed1), (s2, e2, turned2, speed2) = found
    assert turned1 == pytest.approx(-90.0, abs=3.0) and turned2 == pytest.approx(-90.0, abs=3.0)
    assert speed1 == pytest.approx(100.0, abs=1.0) and speed2 == pytest.approx(85.0, abs=1.0) and e1 < s2


def test_three_ways_and_the_control_fly_at_the_observed_speed_and_read_the_offset_across_each_turn(monkeypatch):
    """Ways (a) and (b) fly at the observed ground speed whatever the speed words say (told 70 m/s on a 100 m/s
    downwind); (b) turns without the stopping-rate limit. The second turn's change of e_y holds what the first left
    (its sign even), its own part does not."""
    params = _params()

    def flown_ways(speeds):
        batch, _, words = _batch(speeds=speeds)
        inputs = executor_inputs(batch.signals[0], batch.geometries[0])         # the observed state at row 0
        monkeypatch.setattr(replay.Batch, "inputs", lambda self, device: inputs)
        sentence, signals = batch.sentences[0], batch.signals[0]
        return batch, words, {
            "executor": executor_turns.fly_way(batch, params, words, stopping=True, device=CPU),
            "executor_no_stopping": executor_turns.fly_way(batch, params, words, stopping=False, device=CPU),
            "exact_words": [executor_turns.exact_positions(sentence, signals, words.spec.heading_lead_s, 2.0,
                                                           params.cycle_s)],
            "observed_track": [executor_turns.observed_positions(sentence, signals, 2.0, params.cycle_s)]}

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
    first, second = executor_turns.turn_rows(batch, flown, words)
    assert first["stratum"] == second["stratum"] == "vectored" and first["group"] == replay.OWN
    for way in executor_turns.WAYS:
        assert first[way]["start_lateral_m"] == pytest.approx(0.0, abs=0.5)
        assert second[way]["change_m"] > 0.0 > second[way]["own_m"]             # carried from the first turn
        assert second[way]["start_lateral_m"] == pytest.approx(first[way]["change_m"], abs=1.0)
        # a left turn: right is outward — the change ends outside, the turn's own part inside
        assert second[way]["outward_m"] == pytest.approx(second[way]["change_m"])
        assert second[way]["own_outward_m"] == pytest.approx(second[way]["own_m"])
    assert not first["overlaps_next"] and first["next_start_row"] == second["start_row"]
    assert second["next_start_row"] is None
    # information, pinned: the fixture's track leads its positions by 1 s (each 2 s segment flown at its first row's
    # track), so even the observed track integrated (the control) ends 99 m inside each turn; the words add 20 m and
    # 17 m; the executor takes back 14 m and 20 m
    own = [{way: r[way]["own_m"] for way in executor_turns.WAYS} for r in (first, second)]
    assert [round(o["observed_track"]) for o in own] == [-99, -89]
    assert [round(o["exact_words"]) for o in own] == [-119, -106]
    assert own[0]["executor"] == pytest.approx(-104.5, abs=3.0) and own[1]["executor"] == pytest.approx(-85.8, abs=3.0)


def test_the_outside_of_a_turn_and_a_flight_that_ends_before_it_is_read():
    """A path that ends 50 m outside the observed (left) turn reads +50 m outward, its change and its own part alike; a
    way whose flight ends before the second turn's end + 30 s does not measure it, and the paired block leaves it out."""
    batch, _, words = _batch()
    signals = batch.signals[0]
    (start, stop, turned, _), (later_start, *_) = executor_turns.turns(signals, len(signals.e_m), words)
    right = np.radians(signals.track_deg)
    shift = np.where(np.arange(len(signals.e_m)) > start, 50.0, 0.0)       # 50 m right of the track after the start
    outside = (signals.e_m + shift * np.cos(right), signals.n_m - shift * np.sin(right))
    short = (signals.e_m[:later_start + 5], signals.n_m[:later_start + 5])
    # 30 m right from the read row itself (the end + 30 s): read there, not a row early
    late = np.where(np.arange(len(signals.e_m)) >= stop + 15, 30.0, 0.0)
    at_read_row = (signals.e_m + late * np.cos(right), signals.n_m - late * np.sin(right))
    flown = {"executor": [outside], "executor_no_stopping": [at_read_row], "exact_words": [outside],
             "observed_track": [short]}
    first, second = executor_turns.turn_rows(batch, flown, words)
    assert turned < 0.0                                                       # a left turn: its outside is right
    for name in ("change_m", "own_m"):
        assert first["executor"][name] == pytest.approx(50.0, abs=1.0)
    assert first["executor"]["outward_m"] == pytest.approx(50.0, abs=1.0) == first["executor"]["own_outward_m"]
    assert first["executor_no_stopping"]["change_m"] == pytest.approx(30.0, abs=1.0)
    assert second["observed_track"] is None and second["executor"] is not None
    table = executor_turns.readout_table([first, second])
    assert table["all"]["all"]["turns"] == 2 and table["all"]["all"]["paired"]["turns"] == 1
    assert table["all"]["all"]["observed_track"]["not_measured"] == 1


def test_the_readout_reads_each_way_by_stratum_and_speed_band():
    def record(part, speed, change, own, turned=-90.0):
        way = {"start_lateral_m": 0.0, "change_m": change, "outward_m": -math.copysign(1.0, turned) * change,
               "own_m": own, "own_outward_m": -math.copysign(1.0, turned) * own}
        return {"stratum": part, "ground_speed_mps": speed, "turn_deg": turned, "overlaps_next": speed > 90.0,
                "executor": way, "executor_no_stopping": way, "exact_words": {**way, "own_outward_m": 5.0},
                "observed_track": None}

    records = [record("vectored", 95.0, 100.0, 30.0), record("vectored", 95.0, -20.0, 10.0),
               record("straight-in", 60.0, 10.0, 1.0)]
    table = executor_turns.readout_table(records)
    cell = table["vectored"]["85-100 m/s"]
    assert cell["turns"] == 2 and cell["executor"]["measured"] == 2 and cell["observed_track"]["not_measured"] == 2
    assert cell["overlapping_the_next"] == 2 and table["straight-in"]["all"]["overlapping_the_next"] == 0
    assert cell["executor"]["abs_change_m"]["p50"] == pytest.approx(60.0)
    assert cell["executor"]["outward_m"]["mean"] == pytest.approx(40.0)     # a left turn: right is outward
    assert cell["executor"]["own_outward_m"]["mean"] == pytest.approx(20.0)
    assert cell["paired"]["turns"] == 0                                      # the control measured none
    assert table["all"]["all"]["turns"] == 3 and table["straight-in"]["0-70 m/s"]["turns"] == 1
    assert table["vectored"]["100-inf m/s"]["executor"]["abs_change_m"] is None
    for r in records:
        r["observed_track"] = r["executor"]
    paired = executor_turns.readout_table(records)["vectored"]["85-100 m/s"]["paired"]
    assert paired["turns"] == 2
    assert paired["executor_minus_exact_words_own_outward_m"]["mean"] == pytest.approx(20.0 - 5.0)
    assert paired["exact_words_minus_observed_track_own_outward_m"]["mean"] == pytest.approx(5.0 - 20.0)
