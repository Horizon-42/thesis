"""Free generation (prior design §12 B4; vocabulary §6 items 5, 6; D67, D68): the prior speaks, the executor flies
through the start of a closed loop, the judge decides — on a synthetic artefact of one flight (A26's test artefact)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flight_scenarios.fas_geometry import fas_course_geometry
from ts_transformer.autopilot.start import start
from ts_transformer.experiments.prior_free_generation import flight_numbers, speak_and_fly
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
    directory, words, batch, stored, _ = test_start._artefact(tmp_path, monkeypatch, interval_s)
    loop, order = start(directory, "train", interval_s, {0: stored}, tmp_path / "executor", most_go_arounds=most,
                        device=CPU)
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
        spy.update(flights=flights, geometry=geometry, landings=landings, stored=stored, words=words, batch=batch,
                   directory=directory)
    finals = {geometry.code: tuple(Final(geometry, k, 9_000.0, fas_course_geometry(c.length_m))
                                   for k, c in enumerate(geometry.candidates))}
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64))
    (generated,) = speak_and_fly(model, loop, order, {0: stored}, flights, geometries, landings, finals, words,
                                 interval_s=interval_s, variant="full", numbers=[flight_numbers(seed, 0, 0)],
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
        height = generated.states[(stored.rows.start + row) * every, 2] - geometry.elevation_m
        state = apply(state, words_row, float(height), words, len(geometry.candidates))
    # the observed rows, then the flown ones from the first predicted step; one row of states for each word row
    assert np.array_equal(generated.states[: stored.rows.start * every], stored.rows.states[: stored.rows.start * every])
    said_rows = (stored.rows.start + len(generated.words) - 1) * every + 1          # the states to the last row's start
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
                         on_final=np.array(on_final), blocked={})

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

    def observed(self, tensors, positions, extra=None):
        rows.append(tensors)
        return observe(self, tensors, positions, extra)

    def spoken(self, tensors, at, numbers, caller=None, extra=None):
        rows.append(tensors)
        return speak(self, tensors, at, numbers, caller, extra)

    monkeypatch.setattr(Speaker, "observe", observed)
    monkeypatch.setattr(Speaker, "speak", spoken)
    spy = {}
    generated, stored, words, geometry = generate(tmp_path, monkeypatch, interval_s=interval_s, spy=spy)
    every = int(round(interval_s / words.spec.step_s))
    count = (stored.rows.start + len(generated.words) - 1) * every + 1
    sentence = replace(stored.rows, grid=generated.words.astype(np.int16), states=generated.states[:count],
                       on_interval=on_interval_rows(count, every))
    expected = collate([sentence_rows(sentence, spy["flights"][0], geometry, spy["landings"][geometry.code], words,
                                      interval_s=interval_s, split="train", variant="full")], CPU)
    assert len(rows) == stored.rows.start + len(generated.words)
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
    from ts_transformer.experiments.prior_free_generation import flight_numbers, speak_and_fly

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
    first = stored.rows.states[stored.rows.start * 2]
    loop = FakeLoop([first, first], [3, 6], 2, MOST_GO_AROUNDS)
    from ts_transformer.experiments import prior_free_generation as runner
    from ts_transformer.instructions.words import RUNWAY, RUNWAY_GO_AROUND

    class LateGoAround(runner.Speaker):
        """Says "go-around" for flight 0 once it is done (row 3 on): words the executor never hears."""

        def speak(self, row, at, numbers, caller=None, extra=None):
            made.append(self)
            said = super().speak(row, at, numbers, caller, extra)
            if len(self.go_around_probability) > 3 and self.go_arounds[0] < MOST_GO_AROUNDS:
                said[0, RUNWAY] = RUNWAY_GO_AROUND
                self.go_arounds[0] += 1
            return said

    made = []
    monkeypatch.setattr(runner, "Speaker", LateGoAround)
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64))
    a, b = speak_and_fly(model, loop, [0, 1], {0: stored, 1: stored}, flights, geometries, landings, finals, words,
                         interval_s=4.0, variant="full",
                         numbers=[flight_numbers(1, 0, i) for i in (0, 1)], device=CPU)
    assert (len(a.words), len(b.words)) == (3, 6)
    # its own words' go-arounds, none of those said after it was done
    assert a.go_arounds == int((a.words[:, RUNWAY] == RUNWAY_GO_AROUND).sum()) < made[0].go_arounds[0]
    assert loop.halted.all()
    assert np.isfinite(a.states).all() and np.isfinite(b.states).all()   # trimmed to the row each was done in
    assert len(a.states) == (stored.rows.start + 3) * 2 + 1 and len(b.states) == (stored.rows.start + 6) * 2 + 1


def test_the_runner_writes_its_sentences_and_readout(tmp_path, monkeypatch):
    """`main` on A26's synthetic artefact: every live root replaced (the landings, the CIFP finals and digests, the
    closed-loop check, the identity of a one-split artefact), a prior written as `prior_train` writes one; two samples of
    the one flight, the files read back. The flight's stored outcome is set to a landing and its generated ones are
    not: the readout puts it inside the prior's selection by the stored outcome (D75)."""
    import dataclasses
    import json

    from ts_transformer.experiments import prior_free_generation as runner
    from ts_transformer.io_utils import file_sha256
    from ts_transformer.prior.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
    from ts_transformer.prior.procedure import PROCEDURE_MASKS

    directory, words, _, stored, _ = test_start._artefact(tmp_path, monkeypatch, 4.0)
    flight = signals_flights(directory, "train")[0]
    geometry = load_candidates(directory)[flight["airport"]]
    landings = {geometry.code: LandingIndex(tuple(c.ident for c in geometry.candidates),
                                            (Landing(utc_s(flight["landing_time_utc"]), geometry.candidates[0].ident,
                                                     flight["dataset_id"].split(":", 1)[1]),), 0)}
    identity = {"row_interval_s": 4.0, "artefact": "synthetic", "selection": {"rule": "landed"}}
    digests = {geometry.code: {c.ident: "0" * 64 for c in geometry.candidates}}
    checked = []
    monkeypatch.setattr(runner, "airport_landings", lambda geometries, days: landings)
    monkeypatch.setattr(runner, "artefact_identity", lambda d, interval, given, rule: identity)
    monkeypatch.setattr(runner, "require_conforming_closed_loop", lambda *given: checked.append(given) or (None, {"checks": {"stub": True}}, None))
    monkeypatch.setattr(runner, "procedure_digests", lambda geometries: digests)
    read = runner.closed_loop_sentences
    monkeypatch.setattr(runner, "closed_loop_sentences", lambda *given: {
        index: dataclasses.replace(sentence, withheld=dataclasses.replace(sentence.withheld, outcome="landed"))
        for index, sentence in read(*given).items()})
    monkeypatch.setattr(runner, "airport_finals", lambda g: tuple(Final(g, k, 9_000.0, fas_course_geometry(c.length_m))
                                                                  for k, c in enumerate(g.candidates)))
    prior = tmp_path / "prior"
    prior.mkdir()
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64))
    save_checkpoint(prior / "checkpoint.pt", model, model.state_dict(), identity=identity,
                    run={"airports": [geometry.code], "held_out": None, "sample": None}, train_config={})
    (prior / "config.json").write_text(json.dumps({"schema": CHECKPOINT_SCHEMA, "identity": identity}))
    (prior / "procedure_masks.json").write_text(json.dumps({
        "set": PROCEDURE_MASKS, "checkpoint_sha256": file_sha256(prior / "checkpoint.pt"), "procedure_data": digests}))
    out = tmp_path / "readout"
    argv = ["--prior", str(prior), "--instructions", str(directory), "--executor", str(tmp_path / "executor"),
            "--split", "train", "--per-airport", "1", "--samples", "2", "--device", "cpu", "--out", str(out), "--smoke"]
    assert runner.main(argv) == 0
    assert checked == [(directory, tmp_path / "executor")]
    rows = [json.loads(line) for line in (out / "sentences.jsonl").read_text().splitlines()]
    assert [r["sample"] for r in rows] == [0, 1] and {r["index"] for r in rows} == {0}
    with np.load(out / "sentences.npz") as arrays:
        assert arrays["offsets"][-1] == len(arrays["words"]) == sum(r["rows"] for r in rows)
    # B6: read back by the format's reader, each sentence with the words the procedure masks blocked at each of its rows
    from ts_transformer.instructions.grammar import column_words
    from ts_transformer.prior.procedure import ProcedureMasks

    _, read_back = runner.read_sentences(out)
    assert [(s.index, s.sample) for s in read_back] == [(0, 0), (0, 1)] and [s.row for s in read_back] == rows
    for s in read_back:
        assert len(s.words) == s.row["rows"] and set(s.blocked) == set(ProcedureMasks.columns)
        for column, mask in s.blocked.items():
            classes = list(column_words(column, words, len(geometry.candidates)))
            assert mask.shape == (len(s.words), len(classes))
            assert not any(mask[r, classes.index(int(s.words[r, column]))] for r in range(len(s.words)))
    config = json.loads((out / "config.json").read_text())
    assert config["checks"] == {"stub": True} and config["chunk"] == 400 and config["identity"] == identity
    assert config["selection"] == "landed"                             # the prior's own rule (D75)
    assert config["checkpoint_sha256"] == file_sha256(prior / "checkpoint.pt")
    readout = json.loads((out / "readout.json").read_text())
    # free generation starts from every flight; the flights outside the prior's selection are given apart (D75), by
    # their stored outcome, not the generated one
    assert all(r["outcome"] != "landed" for r in rows)
    assert readout["selection"] == "landed" and readout["outside"] == {}
    (stratum,) = readout["inside"][geometry.code]
    assert readout["inside"][geometry.code][stratum]["sentences"] == 2
    with pytest.raises(SystemExit):
        runner.main(argv)                                              # never over an existing readout
    smoke_on_val = [*argv[:-1]]
    smoke_on_val[smoke_on_val.index("train")] = "val"
    smoke_on_val[smoke_on_val.index(str(out))] = str(tmp_path / "val")
    with pytest.raises(SystemExit):                                    # a smoke never reads the val days (D85)
        runner.main([*smoke_on_val, "--smoke"])
    assert not list(prior.glob("val_read_*"))
    for airports in (["KZZZ"], [geometry.code, "KZZZ"]):
        with pytest.raises(SystemExit):                                # an airport not the artefact's
            runner.main([*argv[:-3], "--airports", *airports, "--out", str(tmp_path / "unknown"), "--smoke"])
    config_text = (out / "config.json").read_text()
    (out / "config.json").write_text(config_text.replace(runner.FREE_GENERATION_SCHEMA, "ts-prior-free-generation-v0"))
    with pytest.raises(ValueError, match=runner.FREE_GENERATION_SCHEMA):                 # another format, by name
        runner.read_sentences(out)
    (out / "config.json").write_text(config_text)
    lines = (out / "sentences.jsonl").read_text()
    (out / "sentences.jsonl").write_text(lines.splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="holds 1 sentences"):                          # a sentence missing
        runner.read_sentences(out)
    (out / "sentences.jsonl").write_text(lines)
    (prior / "config.json").write_text(json.dumps({"schema": "ts-prior-checkpoint-v6", "identity": identity}))
    with pytest.raises(SystemExit, match="ts-prior-checkpoint-v6"):                # a prior of another format, by name
        runner.main([*argv[:-3], "--out", str(tmp_path / "older"), "--smoke"])
    (prior / "config.json").write_text(json.dumps({"schema": CHECKPOINT_SCHEMA, "identity": identity}))
    (prior / "procedure_masks.json").write_text(json.dumps({"set": "another", "checkpoint_sha256": "", "procedure_data": {}}))
    with pytest.raises(SystemExit, match="procedure masks"):
        runner.main([*argv[:-3], "--out", str(tmp_path / "other"), "--smoke"])
