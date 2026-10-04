"""Free generation (prior design §12 B4; vocabulary §6 items 5, 6; D67, D68): the prior speaks, the executor flies
through the start of a closed loop, the judge decides — on a synthetic artefact of one flight (A26's test artefact)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flight_scenarios.fas_geometry import fas_course_geometry
from ts_transformer.autopilot.start import start
from ts_transformer.experiments.prior_free_generation import speak_and_fly
from ts_transformer.instructions.artefact import load_candidates, signals_flights
from ts_transformer.instructions.grammar import apply
from ts_transformer.instructions.words import COLUMNS, UNCHANGED
from ts_transformer.prior.landings import Landing, LandingIndex, utc_s
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.procedure import Final
from ts_transformer.prior.speaker import MOST_GO_AROUNDS
from ts_transformer.tests import test_start

CPU = torch.device("cpu")


def generate(tmp_path, monkeypatch, *, interval_s=4.0, seed=0, most=MOST_GO_AROUNDS, spy=None):
    directory, words, _, stored, _ = test_start._artefact(tmp_path, monkeypatch, interval_s)
    loop, order = start(directory, "train", interval_s, {0: stored}, test_start._params(), words,
                        most_go_arounds=most, device=CPU)
    flights = {0: signals_flights(directory, "train")[0]}
    geometries = load_candidates(directory)
    geometry = geometries[flights[0]["airport"]]
    landing = utc_s(flights[0]["landing_time_utc"])
    key = flights[0]["dataset_id"].split(":", 1)[1]
    entry = utc_s(flights[0]["entry_time_utc"])
    landings = {geometry.code: LandingIndex(tuple(c.ident for c in geometry.candidates),
                                            (Landing(entry + 30.0, geometry.candidates[0].ident, "OTHER"),
                                             Landing(landing, geometry.candidates[0].ident, key)), 0)}
    if spy is not None:
        spy.update(flights=flights, geometry=geometry, landings=landings, stored=stored, words=words)
    finals = {geometry.code: tuple(Final(geometry, k, 9_000.0, fas_course_geometry(c.length_m))
                                   for k, c in enumerate(geometry.candidates))}
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64))
    (generated,) = speak_and_fly(model, loop, order, {0: stored}, flights, geometries, landings, finals, words,
                                 interval_s=interval_s, variant="full", generator=torch.Generator().manual_seed(seed),
                                 device=CPU)
    return generated, stored, words, geometry


@pytest.mark.parametrize("interval_s", [2.0, 4.0])
def test_one_flight_spoken_and_flown_to_its_outcome(tmp_path, monkeypatch, interval_s):
    generated, stored, words, geometry = generate(tmp_path, monkeypatch, interval_s=interval_s)
    every = int(round(interval_s / words.spec.step_s))
    assert generated.outcome and len(generated.words) >= 1
    assert (generated.words[0] != UNCHANGED).all()                       # the first predicted step says every column
    state = None
    for row, words_row in enumerate(generated.words):                      # every row passes the grammar
        height = generated.states[(stored.start + row) * every, 2] - geometry.elevation_m
        state = apply(state, words_row, float(height), words, len(geometry.candidates))
    # the observed rows, then the flown ones from the first predicted step; one row of states for each word row
    assert np.array_equal(generated.states[: stored.start * every], stored.states[: stored.start * every])
    said_rows = (stored.start + len(generated.words) - 1) * every + 1          # the states to the last row's start
    assert said_rows < len(generated.states) <= said_rows + every
    assert np.isfinite(generated.states[:said_rows]).all()
    assert generated.go_arounds <= MOST_GO_AROUNDS
    assert (len(generated.go_around_probability) == len(generated.words) == len(generated.on_final)
            == len(generated.go_around_permitted))
    assert generated.words.shape[1] == len(COLUMNS)


def test_the_same_seed_flies_the_same_sentence(tmp_path, monkeypatch):
    a, _, _, _ = generate(tmp_path / "a", monkeypatch, seed=3)
    b, _, _, _ = generate(tmp_path / "b", monkeypatch, seed=3)
    assert np.array_equal(a.words, b.words) and np.array_equal(a.states, b.states) and a.outcome == b.outcome


def test_a_loop_started_with_another_bound_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="D68"):
        generate(tmp_path, monkeypatch, most=1)


def test_the_readout_by_airport_and_stratum():
    from ts_transformer.experiments.prior_free_generation import Generated, readout

    def sentence(index, outcome, words, go_arounds, probability, on_final):
        return Generated(index=index, words=np.array(words), states=np.zeros((1, 6)), outcome=outcome, crossing=None,
                         timed_out=outcome == "timeout", go_arounds=go_arounds,
                         go_around_probability=np.array(probability), go_around_permitted=np.ones(len(words), bool),
                         on_final=np.array(on_final))

    u = UNCHANGED
    generated = [sentence(0, "landed", [[0, 1, 2, 1, 3], [u, 2, u, u, u]], 0, [0.0, 0.1], [False, True]),
                 sentence(0, "timeout", [[0, 1, 2, 1, 3], [-2, u, 4, 5, u]], 2, [0.0, 0.5], [False, False]),
                 sentence(1, "landed", [[1, 0, 2, 1, 3]], 0, [0.0], [True])]
    out = readout(generated, {0: "KXXX", 1: "KXXX"}, {0: "vectored", 1: "straight-in"},
                  {0: np.array([[0, 1, 2, 1, 3], [u, 3, u, u, u], [u, 4, u, u, u]]), 1: np.array([[1, 0, 2, 1, 3]])})
    vectored = out["KXXX"]["vectored"]
    assert vectored["sentences"] == 2 and vectored["outcomes"] == {"landed": 1, "timeout": 1}
    assert vectored["timed_out"] == 1 and vectored["go_arounds"] == 2 and vectored["at_the_bound"] == 1
    assert vectored["words_per_sentence"]["heading"] == 1.5 and vectored["labelled_words_per_sentence"]["heading"] == 3
    assert vectored["go_around_probability_on_final"] == {"rows": 1, "mean": pytest.approx(0.1)}
    assert out["KXXX"]["straight-in"]["go_around_probability_on_final"] == {"rows": 1, "mean": 0.0}


@pytest.mark.parametrize("interval_s", [2.0, 4.0])
def test_the_loop_gives_the_speaker_the_rows_the_sentence_gives(tmp_path, monkeypatch, interval_s):
    """§7 item 2: every row the loop gives the speaker — observed and flown — is, bit for bit, the row `sentence_rows`
    gives the sentence it said, read back from the flight's words and states: the time, the motion from the 2 s row
    before, the landings at the row's UTC, the words in force, the first predicted step."""
    from dataclasses import replace

    from ts_transformer.instructions.labeller.interval import on_interval_rows
    from ts_transformer.prior.batch import collate
    from ts_transformer.prior.inputs import sentence_rows
    from ts_transformer.prior.speaker import Speaker

    rows = []
    observe, speak = Speaker.observe, Speaker.speak

    def observed(self, tensors, positions):
        rows.append(tensors)
        return observe(self, tensors, positions)

    def spoken(self, tensors, at, caller=None):
        rows.append(tensors)
        return speak(self, tensors, at, caller)

    monkeypatch.setattr(Speaker, "observe", observed)
    monkeypatch.setattr(Speaker, "speak", spoken)
    spy = {}
    generated, stored, words, geometry = generate(tmp_path, monkeypatch, interval_s=interval_s, spy=spy)
    every = int(round(interval_s / words.spec.step_s))
    count = (stored.start + len(generated.words) - 1) * every + 1
    sentence = replace(stored, grid=generated.words.astype(np.int16), states=generated.states[:count],
                       on_interval=on_interval_rows(count, every))
    expected = collate([sentence_rows(sentence, spy["flights"][0], geometry, spy["landings"][geometry.code], words,
                                      interval_s=interval_s, split="train", variant="full")], CPU)
    assert len(rows) == stored.start + len(generated.words)
    assert expected.candidates[..., 6].abs().sum() > 0                  # a landing counted: the UTC times are read
    for r, row in enumerate(rows):
        for name in row._fields:
            if name != "targets":
                assert torch.equal(getattr(row, name), getattr(expected.between(r, r + 1), name)), (r, name)


class FakeLoop:
    """A loop of flights that each fly on in a straight line and are done after ``ends[b]`` rows; once done, a flight's
    states are not a number (as an executor's can be, which flies a done flight on)."""

    def __init__(self, starts, ends, every, most):
        self.state = np.array(starts, dtype=np.float64)
        self.ends, self.every, self.most_go_arounds, self.row_cycles = np.array(ends), every, most, 2
        self.steps, self.halted = 0, np.zeros(len(starts), dtype=bool)
        cycles = 2 * every * self.ends - 1
        self.executor = type("E", (), {"time_limit_s": torch.tensor([600.0] * len(starts)),
                                       "done_cycle": torch.as_tensor(cycles)})()

    def rows(self):
        return self.state.copy()

    def step(self, words_row):
        flown = []
        for _ in range(self.every):
            self.state[:, 0] += 100.0
            self.state[self.steps >= self.ends] = np.nan
            flown.append(self.state.copy())
        self.steps += 1
        return np.stack(flown, axis=1), self.steps >= self.ends

    def halt(self, flights):
        self.halted |= flights

    def timed_out(self):
        return np.zeros(len(self.ends), dtype=bool)

    def outcome(self, b):
        return type("O", (), {"outcome": "crossed", "crossing": None})()


def test_flights_done_at_different_rows_end_apart_and_never_feed_a_state_that_is_not_a_number(tmp_path, monkeypatch):
    from ts_transformer.experiments.prior_free_generation import speak_and_fly

    directory, words, _, stored, _ = test_start._artefact(tmp_path, monkeypatch, 4.0)
    record = signals_flights(directory, "train")[0]
    flights = {0: record, 1: {**record, "dataset_id": record["dataset_id"] + "B"}}
    geometries = load_candidates(directory)
    geometry = geometries[record["airport"]]
    keys = [flights[i]["dataset_id"].split(":", 1)[1] for i in (0, 1)]
    landings = {geometry.code: LandingIndex(tuple(c.ident for c in geometry.candidates),
                                            tuple(Landing(utc_s(record["landing_time_utc"]) + k, "09", key)
                                                  for k, key in enumerate(keys)), 0)}
    finals = {geometry.code: tuple(Final(geometry, k, 9_000.0, fas_course_geometry(c.length_m))
                                   for k, c in enumerate(geometry.candidates))}
    first = stored.states[stored.start * 2]
    loop = FakeLoop([first, first], [3, 6], 2, MOST_GO_AROUNDS)
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64))
    a, b = speak_and_fly(model, loop, [0, 1], {0: stored, 1: stored}, flights, geometries, landings, finals, words,
                         interval_s=4.0, variant="full", generator=torch.Generator().manual_seed(1), device=CPU)
    assert (len(a.words), len(b.words)) == (3, 6)
    assert loop.halted.all()
    assert np.isfinite(a.states).all() and np.isfinite(b.states).all()   # trimmed to the row each was done in
    assert len(a.states) == (stored.start + 3) * 2 + 1 and len(b.states) == (stored.start + 6) * 2 + 1
