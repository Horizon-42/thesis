"""Training the prior (prior design §5; milestone B3's loop, on synthetic sentences): the loss, the batches, the stop on
the select days, and a fold that never reads its held-out airport in training."""

from __future__ import annotations

import json
import zlib

import numpy as np
import pytest
import torch
from torch import nn

from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.prior import train as training
from ts_transformer.prior.batch import collate
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.runs import Run, RunData, held_out_sentences, run_data
from ts_transformer.prior.train import TrainConfig, column_nll, evaluate, length_groups, train
from ts_transformer.tests.support import instruction_spec, prior_sentence

CPU = torch.device("cpu")
SMALL = {"d_model": 32, "layers": 1, "heads": 4, "feedforward": 64, "dropout": 0.0}
AIRPORTS = ("KAAA", "KBBB", "KCCC")


@pytest.fixture(scope="module")
def words():
    return Words(instruction_spec())


class RecordingSource:
    """A `runs.SentenceSource` of synthetic sentences that records every split and airport it is asked for."""

    def __init__(self, words, per_airport=4):
        self.words, self.per_airport, self.asked = words, per_airport, []

    def sentences(self, split, airport):
        self.asked.append((split, airport))
        rng = np.random.default_rng(zlib.crc32(f"{split}:{airport}".encode()))
        return [prior_sentence(rng, self.words, airport=airport, split=split, rows=int(rng.integers(15, 40)))
                for _ in range(self.per_airport)]


def test_a_run_names_its_training_airports():
    assert Run(AIRPORTS).training_airports == AIRPORTS
    assert Run(AIRPORTS, "KBBB").training_airports == ("KAAA", "KCCC")
    with pytest.raises(ValueError, match="not one of"):
        Run(AIRPORTS, "KZZZ")
    with pytest.raises(ValueError, match="training airport"):
        Run(("KAAA",), "KAAA")


def test_a_fold_reads_only_the_train_and_select_days_of_its_training_airports(words):
    source = RecordingSource(words)
    run = Run(AIRPORTS, held_out="KBBB")
    data = run_data(source, run)
    assert sorted(source.asked) == [("select", "KAAA"), ("select", "KCCC"), ("train", "KAAA"), ("train", "KCCC")]
    assert {s.airport for s in data.train + data.select} == {"KAAA", "KCCC"}
    source.asked.clear()
    held_out = held_out_sentences(source, run)
    assert source.asked == [("select", "KBBB")] and {s.airport for s in held_out} == {"KBBB"}
    with pytest.raises(ValueError, match="no held-out airport"):
        held_out_sentences(source, Run(AIRPORTS))


def test_the_run_data_refuses_a_sentence_of_another_airport_or_split(words):
    source = RecordingSource(words)
    run = Run(AIRPORTS, held_out="KBBB")
    train_days, select_days = source.sentences("train", "KAAA"), source.sentences("select", "KAAA")
    with pytest.raises(ValueError, match="train sentences are not"):
        RunData(run, train_days + source.sentences("train", "KBBB"), select_days)
    with pytest.raises(ValueError, match="select sentences are not"):
        RunData(run, train_days, select_days + source.sentences("val", "KAAA"))
    with pytest.raises(ValueError, match="no select sentence"):
        RunData(run, train_days, [])


def test_the_batches_hold_every_sentence_once_within_the_padded_rows():
    lengths = [5, 30, 12, 12, 7, 100, 3]
    groups = length_groups(lengths, 40, np.random.default_rng(0))
    assert sorted(i for group in groups for i in group) == list(range(len(lengths)))
    for group in groups:
        assert len(group) == 1 or max(lengths[i] for i in group) * len(group) <= 40
    assert length_groups(lengths, 40, None) == length_groups(lengths, 40, None)
    # each epoch draws new groups: sentences of one length are not always grouped with the same others
    same = [6] * 12
    rng = np.random.default_rng(1)
    draws = {tuple(sorted(tuple(sorted(g)) for g in length_groups(same, 18, rng))) for _ in range(5)}
    assert len(draws) > 1


def test_the_loss_counts_only_the_asked_rows(words):
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", **SMALL)).eval()
    rng = np.random.default_rng(3)
    group = [prior_sentence(rng, words, rows=n) for n in (12, 30)]
    rows = collate(group, CPU)
    with torch.no_grad():
        logits = model(rows)
        nll = column_nll(logits, rows)
        expected = torch.zeros(len(COLUMNS))
        for b, sentence in enumerate(group):
            for r in range(sentence.first_step, sentence.rows):
                for c, logit in enumerate(logits):
                    expected[c] -= torch.log_softmax(logit[b, r], -1)[rows.targets[b, r, c]]
    torch.testing.assert_close(nll, expected, rtol=1e-5, atol=1e-4)
    # the per-step loss does not depend on the batches it is summed over
    small, large = evaluate(model, group, 30, CPU), evaluate(model, group, 10_000, CPU)
    assert small["steps"] == large["steps"] == sum(s.rows - s.first_step for s in group)
    assert small["loss_per_step"] == pytest.approx(large["loss_per_step"], rel=1e-6)


def test_the_loop_learns_and_keeps_the_best_select_epoch(words):
    data = run_data(RecordingSource(words, per_airport=6), Run(AIRPORTS))
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", **SMALL))
    config = TrainConfig(tokens_per_batch=200, learning_rate=3e-3, warmup_steps=5, max_epochs=6, patience=6)
    lines = []
    result = train(model, data, config, CPU, lines.append)
    history = result.history
    assert len(history) == 6 and len(lines) == 6
    assert history[-1]["train_loss_per_step"] < 0.8 * history[0]["train_loss_per_step"]
    losses = [row["select_loss_per_step"] for row in history]
    assert result.best_epoch == 1 + int(np.argmin(losses))
    model.load_state_dict(result.state)
    assert evaluate(model, data.select, 10_000, CPU)["loss_per_step"] == pytest.approx(min(losses), rel=1e-6)


def test_the_stop_reads_only_the_select_days_and_waits_patience_epochs(words, monkeypatch):
    data = run_data(RecordingSource(words), Run(AIRPORTS, held_out="KCCC"))
    read = []
    real = training.evaluate

    def recording(model, sentences, tokens, device):
        read.append([s.flight_key for s in sentences])
        return real(model, sentences, tokens, device)

    monkeypatch.setattr(training, "evaluate", recording)
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", **SMALL))
    # no learning: the select loss never improves after the first epoch
    config = TrainConfig(tokens_per_batch=400, learning_rate=0.0, warmup_steps=1, max_epochs=30, patience=3)
    result = train(model, data, config, CPU, lambda line: None)
    assert len(result.history) == 1 + 3 and result.best_epoch == 1
    assert read == [[s.flight_key for s in data.select]] * 4


def test_the_state_kept_is_the_best_select_epochs_not_the_last(words, monkeypatch):
    """The select losses scripted: epoch 2 is the best, epochs 3–5 worse; the state returned is the weights after epoch
    2, not those the model ends with."""
    data = run_data(RecordingSource(words), Run(AIRPORTS))
    scripted = iter([5.0, 3.0, 4.0, 4.5, 6.0])
    weights = []

    def scripted_evaluate(model, sentences, tokens, device):
        weights.append({name: value.clone() for name, value in model.state_dict().items()})
        return {"loss_per_step": next(scripted), "per_column": {}, "steps": 1}

    monkeypatch.setattr(training, "evaluate", scripted_evaluate)
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", **SMALL))
    config = TrainConfig(tokens_per_batch=400, learning_rate=1e-2, warmup_steps=1, max_epochs=30, patience=3)
    result = train(model, data, config, CPU, lambda line: None)
    assert result.best_epoch == 2 and len(result.history) == 5
    final = model.state_dict()
    assert any(not torch.equal(result.state[name], final[name]) for name in final)
    assert all(torch.equal(result.state[name], weights[1][name]) for name in final)


def test_a_non_finite_loss_stops_the_training(words):
    data = run_data(RecordingSource(words), Run(AIRPORTS))
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", **SMALL))
    with torch.no_grad():
        nn.init.constant_(model.heads["speed"].weight, float("nan"))
    with pytest.raises(FloatingPointError, match="epoch 1"):
        train(model, data, TrainConfig(tokens_per_batch=400, max_epochs=2), CPU, lambda line: None)



def procedure_root(tmp_path, airport, runways):
    """A tmp CIFP procedure tree with an RNAV(GPS) document for each of ``runways`` at ``airport`` (only its bytes are
    read here)."""
    root = tmp_path / "procedures"
    details = root / airport / "procedure-details"
    details.mkdir(parents=True)
    index = {"runways": [{"runwayIdent": f"RW{r}", "procedures": [{"procedureFamily": "RNAV_GPS",
                                                                   "procedureUid": f"R{r}"}]} for r in runways]}
    (details / "index.json").write_text(json.dumps(index))
    for r in runways:
        (details / f"R{r}.json").write_text(json.dumps({"runway": r}))
    return root


def run_runner(tmp_path, monkeypatch, *more, airports=("KXXX",), selection="landed", outcomes=("landed", "landed")):
    """`prior_train.main` on a synthetic artefact of ``airports``: every write root under tmp, the landings from the
    artefact's roster records (never a live roster), the closed-loop check of the runner stubbed (the synthetic artefact
    has no reference), under the rule ``selection`` (D75), the artefact's first and second flight of each split and
    airport of stored outcomes ``outcomes``. Returns ``(artefact, landings, out, argv, checked)``."""
    from ts_transformer.experiments import prior_train
    from ts_transformer.io_utils import file_sha256
    from ts_transformer.prior.landings import roster_landings
    from ts_transformer.tests.support import fixture_days, prior_artefact

    artefact = tmp_path / "artefact"
    _, records = prior_artefact(artefact, interval_s=4.0, airports=airports, outcomes=outcomes)
    landings = {code: roster_landings(records[code], ("09", "09L"), fixture_days()) for code in airports}
    checked = []
    monkeypatch.setattr(prior_train, "require_conforming_closed_loop", lambda *given: checked.append(given) or (None, {"checks": {"stub": True}}, None))
    monkeypatch.setattr(prior_train, "airport_landings", lambda geometries, days: landings)
    procedures = tmp_path / "procedures"
    for code in airports:
        procedure_root(procedures, code, ("09", "09L"))
    files = {path: file_sha256(path) for path in sorted(artefact.rglob("*")) if path.is_file()}
    out = tmp_path / "run"
    argv = ["--instructions", str(artefact), "--row-interval-s", "4", "--executor", str(tmp_path / "executor"),
            "--variant", "full", "--selection", selection, "--out", str(out),
            "--sample", "1", "--device", "cpu", "--procedure-root", str(procedures / "procedures"),
            "--d-model", "32", "--layers", "1", "--heads", "4", "--feedforward", "64", "--max-epochs", "2",
            "--warmup-steps", "1", *more]
    assert prior_train.main(argv) == 0
    # B8, D75: the selection is applied as the sentences are read; the artefact is never changed
    assert {path: file_sha256(path) for path in sorted(artefact.rglob("*")) if path.is_file()} == files
    return artefact, landings, out, argv, checked


def test_the_runner_trains_a_smoke_run_into_a_new_directory(tmp_path, monkeypatch):
    from ts_transformer.experiments import prior_train
    from ts_transformer.prior.checkpoint import load_checkpoint
    from ts_transformer.prior.source import artefact_identity

    artefact, landings, out, argv, checked = run_runner(tmp_path, monkeypatch)
    assert checked == [(artefact, tmp_path / "executor")]
    assert {p.name for p in out.iterdir()} == {"checkpoint.pt", "config.json", "memory.json", "history.json",
                                              "procedure_masks.json"}
    config = json.loads((out / "config.json").read_text())
    assert config["sample"] == {"per_airport_and_split": 1, "seed": 1337} and config["checks"] == {"stub": True}
    assert config["sentences"] == {"train": 1, "select": 1}
    memory = json.loads((out / "memory.json").read_text())
    assert memory["batches"][0]["sentences"] == 1 and memory["gpu_peak_reserved_bytes"] is None
    masks = json.loads((out / "procedure_masks.json").read_text())
    assert set(masks["procedure_data"]["KXXX"]) == {"09", "09L"}
    loaded = load_checkpoint(out / "checkpoint.pt", artefact_identity(artefact, 4.0, landings, "landed"))
    assert loaded.model.config.d_model == 32
    assert loaded.run["sample"] == config["sample"]                  # a smoke checkpoint says so itself
    with pytest.raises(SystemExit):
        prior_train.main(argv)                                       # never over an existing run


def test_a_landed_run_trains_on_the_landed_sentences_and_a_run_under_another_rule_is_refused(tmp_path, monkeypatch):
    """B8, D75: of two flights a split, the second's stored outcome not a landing, `landed` trains and stops on the
    first alone and records the counts; the checkpoint is refused against the identity of `all`, by name."""
    from ts_transformer.instructions.artefact import SPLITS
    from ts_transformer.prior.checkpoint import load_checkpoint
    from ts_transformer.prior.selection import selection_totals
    from ts_transformer.prior.source import artefact_identity

    artefact, landings, out, _, _ = run_runner(tmp_path, monkeypatch, "--sample", "2",
                                               outcomes=("landed", "crossed_too_high"))
    config = json.loads((out / "config.json").read_text())
    assert config["sentences"] == {"train": 1, "select": 1}
    assert config["identity"]["selection"]["rule"] == "landed"
    assert selection_totals(config["identity"]["selection"]) == {
        split: {"kept": 1, "left_out_fault": 0, "left_out_outcome": 1} for split in SPLITS}
    load_checkpoint(out / "checkpoint.pt", artefact_identity(artefact, 4.0, landings, "landed"))
    with pytest.raises(ValueError, match=r"\['selection'\] differ"):
        load_checkpoint(out / "checkpoint.pt", artefact_identity(artefact, 4.0, landings, "all"))


def test_the_memory_check_only_writes_the_check_and_trains_nothing(tmp_path, monkeypatch):
    _, _, out, _, _ = run_runner(tmp_path, monkeypatch, "--memory-check-only")
    assert {p.name for p in out.iterdir()} == {"config.json", "memory.json"}
    assert json.loads((out / "memory.json").read_text())["batches"]


def test_a_fold_trains_without_its_held_out_airport_and_scores_it_on_its_select_days(tmp_path, monkeypatch):
    _, _, out, _, _ = run_runner(tmp_path, monkeypatch, "--held-out", "KYYY", "--sample", "2",
                                 airports=("KXXX", "KYYY"))
    config = json.loads((out / "config.json").read_text())
    assert config["run"] == {"airports": ["KXXX", "KYYY"], "held_out": "KYYY"}
    assert config["sentences"] == {"train": 2, "select": 2}         # KXXX's alone
    held_out = json.loads((out / "held_out.json").read_text())
    assert held_out["airport"] == "KYYY" and held_out["steps"] > 0 and np.isfinite(held_out["loss_per_step"])
    assert held_out["first_step_runway"]["sentences"] == 2                   # §5: a readout of a fold


def test_the_first_step_runway_counts_the_sentences_whose_top_class_is_the_labelled_one(words):
    """§5's first-step runway: a model whose runway head puts its largest logit on the labelled class at the first
    predicted step scores every sentence, one that puts it on another class none; only the first step is read."""
    from ts_transformer.instructions.words import RUNWAY
    from ts_transformer.prior.train import first_step_runway

    rng = np.random.default_rng(0)
    sentences = [prior_sentence(rng, words, candidates=3, rows=20, first_step=6) for _ in range(5)]

    class Oracle(nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.shift = shift

        def forward(self, rows):
            classes = 2 + 3                                                  # unchanged, go-around, 3 candidates
            target = (rows.targets[..., RUNWAY] + torch.where(rows.first, self.shift, 0)) % classes
            runway = nn.functional.one_hot(target, classes).float()
            runway[~rows.first] = nn.functional.one_hot((target[~rows.first] + 1) % classes, classes).float()
            return [runway]

    assert first_step_runway(Oracle(0), sentences, 4096, CPU) == {"sentences": 5, "top1": 5, "share": 1.0}
    assert first_step_runway(Oracle(1), sentences, 4096, CPU)["top1"] == 0


def test_the_memory_check_takes_the_batch_of_the_most_row_candidates(words):
    """§12 B3: the batches checked are the one with the most padded rows × sentences × candidates and the one of the
    longest sentences."""
    from ts_transformer.prior.train import largest_batches

    rng = np.random.default_rng(0)
    sentences = [prior_sentence(rng, words, rows=n, candidates=k) for n, k in ((30, 2), (30, 2), (60, 2), (31, 8))]
    most, longest = largest_batches(sentences, 64)
    assert [sentences[i].rows for i in most] == [31] and sentences[most[0]].candidates.shape[1] == 8
    assert [sentences[i].rows for i in longest] == [60]


def test_the_runner_prints_the_selection_of_train_and_select_only(tmp_path, monkeypatch, capsys):
    """D85, D120: the run counts the selection of every split for its identity, and prints train's and select's alone —
    nothing of val is shown before the base's one validation readout."""
    run_runner(tmp_path, monkeypatch, outcomes=("landed", "crossed_too_high"))
    printed = capsys.readouterr().out
    (line,) = [json.loads(text) for text in printed.splitlines() if text.startswith('{"artefact_selection"')]
    assert set(line["artefact_selection"]) == {"train", "select"}
    assert '"val"' not in printed


def test_a_fold_through_the_runner_reads_nothing_of_its_held_out_airport_to_train(tmp_path, monkeypatch):
    """§5 (D39): a fold's training reads nothing of its held-out airport — its sentences changed (every height 500 m up),
    the checkpoint's weights and the history are the same, bit for bit; only the held-out score moves."""
    import dataclasses

    import torch

    from ts_transformer.prior import source as source_module
    from ts_transformer.prior.batch import OWN_FEATURES

    a = run_runner(tmp_path / "a", monkeypatch, "--held-out", "KYYY", "--sample", "2", airports=("KXXX", "KYYY"))[2]
    real = source_module.ArtefactSource.sentences

    def moved(self, split, airport, *given, **named):
        out = real(self, split, airport, *given, **named)
        if airport != "KYYY":
            return out
        height = OWN_FEATURES.index("height_above_elevation")
        return [dataclasses.replace(s, own=s.own + 0.5 * (np.arange(s.own.shape[1]) == height)) for s in out]

    monkeypatch.setattr(source_module.ArtefactSource, "sentences", moved)
    b = run_runner(tmp_path / "b", monkeypatch, "--held-out", "KYYY", "--sample", "2", airports=("KXXX", "KYYY"))[2]
    weights = [torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)["state"] for run in (a, b)]
    assert weights[0].keys() == weights[1].keys()
    assert all(torch.equal(weights[0][name], weights[1][name]) for name in weights[0])
    histories = [json.loads((run / "history.json").read_text()) for run in (a, b)]
    assert [[{k: v for k, v in row.items() if k != "seconds"} for row in h["epochs"]] for h in histories] == [
        [{k: v for k, v in row.items() if k != "seconds"} for row in histories[0]["epochs"]]] * 2
    held = [json.loads((run / "held_out.json").read_text())["loss_per_step"] for run in (a, b)]
    assert held[0] != held[1]
