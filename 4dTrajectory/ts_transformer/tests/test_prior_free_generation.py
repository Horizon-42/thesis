"""Free generation (prior design §12 B4; vocabulary §6 items 5, 6; D67, D68): the prior speaks, the executor flies
through the start of a closed loop, the judge decides — on a synthetic artefact of one flight (A26's test artefact)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flight_scenarios.fas_geometry import fas_course_geometry
from ts_transformer.autopilot.start import start
from ts_transformer.experiments.prior_free_generation import speak_and_fly
from ts_transformer.experiments.prior_speaking_loop import flight_numbers
from ts_transformer.instructions.artefact import load_candidates, load_day_split, signals_flights
from ts_transformer.instructions.grammar import apply
from ts_transformer.instructions.words import COLUMNS, UNCHANGED
from ts_transformer.prior.landings import Landing, LandingIndex, utc_s
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.procedure import Final
from ts_transformer.prior.speaker import MOST_GO_AROUNDS
from ts_transformer.tests import test_start

CPU = torch.device("cpu")


def generate(tmp_path, monkeypatch, *, interval_s=4.0, seed=0, most=MOST_GO_AROUNDS, spy=None, speaking=False):
    """One flight of A26's synthetic artefact spoken and flown (`speak_and_fly`): ``(generated, stored, words,
    geometry)``; with ``speaking``, the `SpeakingLoop` flown to its end instead of the sentence ("start": at its first
    predicted step)."""
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
                                             Landing(landing, geometry.candidates[0].ident, key)), 0,
                                            load_day_split(directory))}
    if spy is not None:
        spy.update(flights=flights, geometry=geometry, landings=landings, stored=stored, words=words, batch=batch,
                   directory=directory)
    finals = {geometry.code: tuple(Final(geometry, k, 9_000.0, fas_course_geometry(c.length_m))
                                   for k, c in enumerate(geometry.candidates))}
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64)).eval()
    if speaking:
        from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop

        loop_ = SpeakingLoop(model, loop, order, {0: stored}, flights, geometries, [landings[geometry.code]], finals,
                             words, interval_s=interval_s, variant="full", device=CPU)
        numbers = flight_numbers(seed, 0, 0)
        while loop_.observing:
            loop_.observe()
        if speaking == "start":                                   # at its first predicted step, for the caller
            return loop_, model
        while loop_.alive.any():
            loop_.step(numbers.random((1, len(COLUMNS))))
        return loop_, model
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
    from ts_transformer.experiments.prior_free_generation import readout
    from ts_transformer.experiments.prior_speaking_loop import Generated

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


@pytest.mark.parametrize("interval_s", [2.0, 4.0, 8.0])
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
    directory, words, _, stored, _ = test_start._artefact(tmp_path, monkeypatch, 4.0)
    record = signals_flights(directory, "train")[0]
    flights = {0: record, 1: {**record, "dataset_id": record["dataset_id"] + "B"}}
    geometries = load_candidates(directory)
    geometry = geometries[record["airport"]]
    keys = [flights[i]["dataset_id"].split(":", 1)[1] for i in (0, 1)]
    landings = {geometry.code: LandingIndex(tuple(c.ident for c in geometry.candidates),
                                            tuple(Landing(utc_s(record["landing_time_utc"]) + k, "09", key)
                                                  for k, key in enumerate(keys)), 0, load_day_split(directory))}
    finals = {geometry.code: tuple(Final(geometry, k, 9_000.0, fas_course_geometry(c.length_m))
                                   for k, c in enumerate(geometry.candidates))}
    first = stored.rows.states[stored.rows.start * 2]
    loop = FakeLoop([first, first], [3, 6], 2, MOST_GO_AROUNDS)
    from ts_transformer.experiments import prior_speaking_loop as speaking
    from ts_transformer.instructions.words import RUNWAY, RUNWAY_GO_AROUND

    class LateGoAround(speaking.Speaker):
        """Says "go-around" for flight 0 once it is done (row 3 on): words the executor never hears."""

        def speak(self, row, at, numbers, caller=None, extra=None):
            made.append(self)
            said = super().speak(row, at, numbers, caller, extra)
            if len(self.go_around_probability) > 3 and self.go_arounds[0] < MOST_GO_AROUNDS:
                said[0, RUNWAY] = RUNWAY_GO_AROUND
                self._go_arounds[0] += 1
            return said

    made = []
    monkeypatch.setattr(speaking, "Speaker", LateGoAround)
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64)).eval()
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
    from ts_transformer.prior import checkpoint as opening
    from ts_transformer.prior.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
    from ts_transformer.prior.procedure import PROCEDURE_MASKS

    directory, words, _, stored, _ = test_start._artefact(tmp_path, monkeypatch, 4.0)
    flight = signals_flights(directory, "train")[0]
    geometry = load_candidates(directory)[flight["airport"]]
    landings = {geometry.code: LandingIndex(tuple(c.ident for c in geometry.candidates),
                                            (Landing(utc_s(flight["landing_time_utc"]), geometry.candidates[0].ident,
                                                     flight["dataset_id"].split(":", 1)[1]),), 0,
                                            load_day_split(directory))}
    identity = {"row_interval_s": 4.0, "artefact": "synthetic", "selection": {"rule": "landed", "counts": {}}}
    digests = {geometry.code: {c.ident: "0" * 64 for c in geometry.candidates}}
    checked = []
    # the prior opens through `checkpoint.open_prior`: its live roots replaced there
    monkeypatch.setattr(opening, "airport_landings", lambda geometries, days: landings)
    monkeypatch.setattr(opening, "artefact_identity", lambda d, interval, given, rule, counted: identity)
    monkeypatch.setattr(runner, "require_conforming_closed_loop", lambda *given: checked.append(given) or (None, {"checks": {"stub": True}}, None))
    monkeypatch.setattr(opening, "procedure_digests", lambda geometries, root: digests)
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
    assert readout["selection"] == "landed" and readout["outside_outcome"] == readout["outside_fault"] == {}
    (stratum,) = readout["inside"][geometry.code]
    assert readout["inside"][geometry.code][stratum]["sentences"] == 2
    # D111: the flight marked as having a faulty track — landed, its stored outcome — is given apart for its fault
    monkeypatch.setattr(runner, "faulty_flights", lambda d, split: {0: ("a fault",)})
    assert runner.main([*argv[:-2], str(tmp_path / "marked"), "--smoke"]) == 0
    marked = json.loads((tmp_path / "marked" / "readout.json").read_text())
    assert marked["inside"] == marked["outside_outcome"] == {}
    assert marked["outside_fault"][geometry.code][stratum]["sentences"] == 2
    # the mark keeps or leaves a sentence in the selection, never an input: the marked flight says and flies the same
    with np.load(out / "sentences.npz") as a, np.load(tmp_path / "marked" / "sentences.npz") as b:
        assert a.files == b.files and all(np.array_equal(a[name], b[name]) for name in a.files)
    monkeypatch.setattr(runner, "faulty_flights", lambda d, split: {})
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
    with pytest.raises(ValueError, match="ts-prior-checkpoint-v6"):                # a prior of another format, by name
        runner.main([*argv[:-3], "--out", str(tmp_path / "older"), "--smoke"])
    (prior / "config.json").write_text(json.dumps({"schema": CHECKPOINT_SCHEMA, "identity": identity}))
    (prior / "procedure_masks.json").write_text(json.dumps({"set": "another", "checkpoint_sha256": "", "procedure_data": {}}))
    with pytest.raises(ValueError, match="procedure masks"):
        runner.main([*argv[:-3], "--out", str(tmp_path / "other"), "--smoke"])


def test_the_rows_a_loop_said_and_its_records_give_the_probabilities_the_speaker_drew(tmp_path, monkeypatch):
    """D106 items 3, 4: a loop's sentences (`SpeakingLoop.sentences`: its rows, its words as targets) under the
    speaker's records give, teacher-forced (`train.masked_log_probability`), the probability each word was drawn with;
    two loops' sentences collated in one batch and their records joined (`Permitted.join`, padded to the longer) give
    each its own."""
    import dataclasses

    from ts_transformer.prior.batch import collate
    from ts_transformer.prior.speaker import Permitted
    from ts_transformer.prior.train import masked_log_probability

    loops = [generate(tmp_path / str(seed), monkeypatch, seed=seed, speaking=True) for seed in (0, 5)]
    model = loops[0][1]
    lengths = [len(loop.speaker.drawn_probability) for loop, _ in loops]
    assert lengths[0] != lengths[1]                       # the join pads a shorter record
    start = loops[0][0].start
    with torch.no_grad():
        for loop, _ in loops:
            (sentence,) = loop.sentences("train")
            assert sentence.rows == start + len(loop.speaker.drawn_probability)
            log_p = masked_log_probability(model, collate([sentence], CPU), loop.permitted())
            drawn = torch.as_tensor(np.stack(loop.speaker.drawn_probability, axis=1), dtype=torch.float32)
            torch.testing.assert_close(log_p[:, start:].exp(), drawn, rtol=1e-4, atol=1e-6)
        batch = collate([loop.sentences("train")[0] for loop, _ in loops], CPU)
        joined = Permitted.join([loop.permitted() for loop, _ in loops])
        assert joined.time_s.shape == (2, max(lengths)) and np.isnan(joined.time_s[np.argmin(lengths), -1])
        log_p = masked_log_probability(model, batch, joined)
        for b, (loop, _) in enumerate(loops):
            drawn = torch.as_tensor(np.stack(loop.speaker.drawn_probability, axis=1)[0], dtype=torch.float32)
            torch.testing.assert_close(log_p[b, start: start + lengths[b]].exp(), drawn, rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError, match="temperatures"):
        Permitted.join([loops[0][0].permitted(), dataclasses.replace(loops[1][0].permitted(), temperature=0.5)])


# ---- what free generation must not read (vocabulary D90, D82; prior D23, D63)

def spoken_rows(monkeypatch):
    """Every row the speaker is given (`Speaker.observe`, `Speaker.speak`), in order, into the list returned."""
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
    return rows


def fly_changed(tmp_path, monkeypatch, interval_s, *, signals_after=False, limit_factor=1.0, seed=0):
    """`generate`'s flight, its stored signals after the first predicted step changed by up to 300 m (every field the
    start reads) or its time limit scaled: the rows the speaker is given and the sentence (the start reads the changed
    signals, as `test_start._artefact` would not)."""
    from dataclasses import replace

    from ts_transformer.autopilot import start as start_module
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.tests.support import executor_inputs

    real, loop_class = load_signals, start_module.Loop
    anchor = {}

    def signals(directory, split):
        out = real(directory, split)
        if not signals_after:
            return out
        flight = out[0]
        rng = np.random.default_rng(5)
        cut = anchor["row"] + 1
        changed = {}
        for name in ("e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps", "vertical_rate_mps"):
            values = getattr(flight, name).copy()
            values[cut:] += rng.normal(size=len(values) - cut) * 300.0
            changed[name] = values
        return [replace(flight, **changed)]

    rows = spoken_rows(monkeypatch)
    spy = {}
    original = test_start._artefact

    def artefact(*given):
        out = original(*given)
        anchor["row"] = out[4]
        directory = out[0]
        geometry = load_candidates(directory)[signals_flights(directory, "train")[0]["airport"]]
        monkeypatch.setattr(start_module, "load_signals", signals)
        monkeypatch.setattr(start_module, "flight_inputs", lambda series, flights, anchors, airports, rule, device:
                            executor_inputs(flights[0], geometry, anchors[0], rule=rule))
        monkeypatch.setattr(start_module, "Loop", lambda inputs, geometries, ias, limits, *a, **k:
                            loop_class(inputs, geometries, ias, [x * limit_factor for x in limits], *a, **k))
        return out

    monkeypatch.setattr(test_start, "_artefact", artefact)
    generated, *_ = generate(tmp_path, monkeypatch, interval_s=interval_s, seed=seed, spy=spy)
    monkeypatch.undo()
    return rows, generated


@pytest.mark.parametrize("interval_s", [2.0, 4.0, 8.0])
def test_the_observed_samples_after_the_first_predicted_step_and_the_time_limit_change_no_input(
        tmp_path, monkeypatch, interval_s):
    """Vocabulary D90: the stored signals after the first predicted step (every field the start reads, moved by up to
    300 m) change no row the speaker is given and no word; the time limit (here 0.35 of its own) only ends the flight
    sooner: every row both runs give is the same, bit for bit."""
    a, ga = fly_changed(tmp_path / "a", monkeypatch, interval_s)
    b, gb = fly_changed(tmp_path / "b", monkeypatch, interval_s, signals_after=True)
    assert len(a) == len(b) > 6 and np.array_equal(ga.words, gb.words)
    for r, (x, y) in enumerate(zip(a, b)):
        for name in x._fields:
            assert torch.equal(getattr(x, name), getattr(y, name)), (r, name)
    c, gc = fly_changed(tmp_path / "c", monkeypatch, interval_s, limit_factor=0.35)
    assert 0 < len(c) <= len(a)
    for r, (x, y) in enumerate(zip(a, c)):
        for name in x._fields:
            assert torch.equal(getattr(x, name), getattr(y, name)), (r, name)
    assert np.array_equal(gc.words, ga.words[: len(gc.words)])


@pytest.mark.parametrize("interval_s", [2.0, 4.0])
@pytest.mark.parametrize("seed", [0, 3])
def test_free_generation_reads_none_of_the_fields_it_must_not(tmp_path, monkeypatch, interval_s, seed):
    """D82, D23, D63: the stored flown states after the first predicted step (moved by 5 km), the words and the
    correction marks, every withheld field (vocabulary D82), the flight record's landing time and runway, and its own
    landing in the index (600 s earlier) change no word, state, outcome or record of free generation."""
    import dataclasses

    spy = {}
    a, stored, words, geometry = generate(tmp_path / "a", monkeypatch, interval_s=interval_s, seed=seed, spy=spy)
    every = int(round(interval_s / words.spec.step_s))
    cut = stored.rows.start * every
    rng = np.random.default_rng(99)
    states = stored.rows.states.copy()
    states[cut:] += rng.normal(0.0, 5_000.0, states[cut:].shape)
    rows = dataclasses.replace(stored.rows, states=states, grid=rng.integers(-1, 3, stored.rows.grid.shape).astype(
        stored.rows.grid.dtype), correction=~stored.rows.correction)
    w = stored.withheld
    withheld = dataclasses.replace(
        w, runway="XX", runway_index=w.runway_index + 7, landing_time_utc="2001-01-01T00:00:00Z", capture_row=0,
        go_around_rows=np.array([1, 2, 3]), stratum="other", outcome="crashed", timed_out=not w.timed_out,
        lateral_m=w.lateral_m + 1e3, vertical_m=w.vertical_m - 1e3, uncorrectable=~w.uncorrectable,
        observed_row=w.observed_row * 0, matched_row=w.matched_row * 0)
    perturbed = dataclasses.replace(stored, rows=rows, withheld=withheld)
    record = {**spy["flights"][0], "landing_time_utc": "2026-06-01T00:00:00Z", "runway": "XX"}
    index = spy["landings"][geometry.code]
    own = record["dataset_id"].split(":", 1)[1]
    moved = LandingIndex(index.runways, tuple(sorted(
        (dataclasses.replace(x, time_s=x.time_s - 600.0) if x.flight_key == own else x for x in index.landings),
        key=lambda x: x.time_s)), index.sealed, index.days)
    loop, order = start(spy["directory"], "train", interval_s, {0: stored}, tmp_path / "a" / "executor",
                        most_go_arounds=MOST_GO_AROUNDS, device=CPU)
    finals = {geometry.code: tuple(Final(geometry, k, 9_000.0, fas_course_geometry(c.length_m))
                                   for k, c in enumerate(geometry.candidates))}
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64)).eval()
    (b,) = speak_and_fly(model, loop, order, {0: perturbed}, {0: record}, load_candidates(spy["directory"]),
                         {geometry.code: moved}, finals, words, interval_s=interval_s, variant="full",
                         numbers=[flight_numbers(seed, 0, 0)], device=CPU)
    for name in ("words", "states", "go_around_probability", "go_around_permitted", "on_final"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert (a.outcome, a.timed_out, a.go_arounds) == (b.outcome, b.timed_out, b.go_arounds)


def two_candidate_loop(tmp_path, monkeypatch, starts, ends):
    """A `SpeakingLoop` of ``len(starts)`` flights of a two-candidate airport (`support.parallel_airport`) on a
    `FakeLoop` (each flight from its start state, done after ``ends[b]`` rows), the observed rows A26's synthetic
    flight's; its words, airport and finals."""
    from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop
    from ts_transformer.tests.support import parallel_airport

    directory, words, _, stored, _ = test_start._artefact(tmp_path, monkeypatch, 4.0)
    record = signals_flights(directory, "train")[0]
    geometry = parallel_airport()
    flights = {i: {**record, "airport": geometry.code, "dataset_id": f"{geometry.code}:F{i}"} for i in range(len(starts))}
    landings = LandingIndex(tuple(c.ident for c in geometry.candidates),
                            tuple(Landing(utc_s(record["landing_time_utc"]) + i, geometry.candidates[0].ident, f"F{i}")
                                  for i in range(len(starts))), 0, load_day_split(directory))
    finals = {geometry.code: tuple(Final(geometry, k, 9_000.0, fas_course_geometry(c.length_m))
                                   for k, c in enumerate(geometry.candidates))}
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64)).eval()
    loop = SpeakingLoop(model, FakeLoop(starts, ends, 2, MOST_GO_AROUNDS), list(flights), {i: stored for i in flights},
                        flights, {geometry.code: geometry}, [landings] * len(starts), finals, words, interval_s=4.0,
                        variant="full", device=CPU)
    while loop.observing:
        loop.observe()
    return loop, words, geometry, finals[geometry.code], stored


def test_on_the_final_is_read_under_the_runway_in_force_before_the_row(tmp_path, monkeypatch):
    """D72: a row is "on the final" inside the region of the runway in force before it — the runway under which the
    speaker drew its runway word — not of the runway the row says: at the row that changes the runway from a candidate
    whose region holds the aircraft to one whose does not, the row is on the final, the next one not; the first
    predicted step has no runway before it."""
    from ts_transformer.experiments.prior_training_export import runway_point
    from ts_transformer.instructions.grammar import column_words
    from ts_transformer.instructions.words import RUNWAY
    from ts_transformer.tests.support import parallel_airport

    geometry = parallel_airport()
    e, n = runway_point(geometry.candidates[0], np.array([6_000.0]), np.array([0.0]))
    start_state = np.array([e[0], n[0], geometry.elevation_m + 350.0, 90.0, 70.0, -3.0])
    loop, words, geometry, finals, _ = two_candidate_loop(tmp_path, monkeypatch, [start_state], [4])
    for at_e in e[0] + 100.0 * np.arange(9):                # the rows flown: inside candidate 0's region only
        assert finals[0].inside(np.array([at_e]), n)[0] and not finals[1].inside(np.array([at_e]), n)[0]
    runway_words = column_words(RUNWAY, words, len(geometry.candidates))
    numbers = flight_numbers(0, 0, 0)
    for said in (0, UNCHANGED, 1, UNCHANGED):                     # candidate 0, then a change to candidate 1
        loop.step(numbers.random((1, len(COLUMNS))), {RUNWAY: (runway_words == said)[None]})
    (generated,) = loop.generated()
    assert generated.words[:, RUNWAY].tolist() == [0, UNCHANGED, 1, UNCHANGED]
    assert generated.on_final.tolist() == [False, True, True, False]


def test_a_flight_the_caller_ends_is_halted_and_keeps_its_sentence_to_the_row_it_ended_in(tmp_path, monkeypatch):
    """D106 item 1 (post-training D93): the caller ends a flight after a row; it is halted, said no more of its own
    rows, its words and states run to the row it ended in, the others fly on; `generated` refuses it by name (its outcome
    is the caller's)."""
    start = np.array([0.0, 0.0, 600.0, 90.0, 70.0, -3.0])
    loop, words, *_ = two_candidate_loop(tmp_path, monkeypatch, [start, start], [6, 6])
    numbers = [flight_numbers(0, 0, i) for i in (0, 1)]
    for _ in range(2):
        loop.step(np.stack([n.random(len(COLUMNS)) for n in numbers]))
    loop.end(np.array([False, True]))
    assert loop.alive.tolist() == [True, False] and loop.ended.tolist() == [False, True]
    assert loop.loop.halted.tolist() == [False, True]
    while loop.alive.any():
        loop.step(np.stack([n.random(len(COLUMNS)) for n in numbers]))
    assert len(loop.said(0)) == 6 and len(loop.said(1)) == 2
    assert len(loop.states(1)) == len(loop.observed[1]) + 1 + 2 * 2
    assert [s.rows for s in loop.sentences("train")] == [loop.start + 6, loop.start + 2]
    with pytest.raises(ValueError, match=r"flights \[1\] are still flown or were ended by the caller"):
        loop.generated()
    (done,) = loop.generated([0])                                       # the others' records, the judge's outcome
    assert np.array_equal(done.words, loop.said(0)) and np.array_equal(done.states, loop.states(0))


def test_a_copy_of_the_loop_says_and_flies_what_its_original_does(tmp_path, monkeypatch):
    """D106 item 1 (post-training D94, on vocabulary D97's `Loop.copy`): copies of a flight taken after three rows — one
    flown first with other numbers (it shares nothing with the original: the original then flies as a loop never copied
    does), one flown after the original with its numbers (it says the original's words and flies its states within the
    executor's bound, vocabulary D97 (3), to the same outcome) — and a copy taken before the first predicted step."""
    from ts_transformer.autopilot.conformance import STATE_BOUND_M

    numbers = np.random.default_rng(7).random((2000, len(COLUMNS)))
    other = np.random.default_rng(8).random((2000, len(COLUMNS)))

    def fly(loop, draws, start=0):
        k = start
        while loop.alive.any():
            loop.step(np.repeat(draws[k][None], len(loop.order), axis=0))
            k += 1
        return loop

    def same(a, b):
        assert np.array_equal(a.said(0), b.said(0)) and a.states(0).shape == b.states(0).shape
        assert np.allclose(a.states(0), b.states(0), rtol=0.0, atol=STATE_BOUND_M, equal_nan=True)
        assert a.generated()[0].outcome == b.generated()[0].outcome

    plain = fly(generate(tmp_path / "plain", monkeypatch, interval_s=4.0, speaking="start")[0], numbers)
    original = generate(tmp_path / "copied", monkeypatch, interval_s=4.0, speaking="start")[0]
    for k in range(3):
        original.step(numbers[k][None])
    diverged, twin = original.copy([0, 0]), original.copy([0])
    fly(diverged, other, 3)                                    # another sentence, flown before the original goes on
    assert not np.array_equal(diverged.said(0), plain.said(0))
    fly(original, numbers, 3)
    assert np.array_equal(original.said(0), plain.said(0)) and np.array_equal(original.states(0), plain.states(0))
    same(fly(twin, numbers, 3), original)
    observing = generate(tmp_path / "observing", monkeypatch, interval_s=4.0, speaking="start")[0]
    early = observing.copy([0])                                # (here at the first predicted step: `observe` is done)
    same(fly(early, numbers), plain)
