"""The base's validation readout (prior §12 B5): the procedure masks on the labelled words, and the runner — the
teacher-forced loss of the val sentences its selection keeps, read once, of the base alone."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from flight_scenarios.fas_geometry import fas_course_geometry
from ts_transformer.experiments import prior_validation as runner
from ts_transformer.instructions.artefact import load_signals
from ts_transformer.instructions.words import ALTITUDE, ANGLE, Words
from ts_transformer.prior.procedure import Final
from ts_transformer.tests.support import (
    fixture_days, instruction_spec, parallel_airport, prior_artefact, prior_closed_loop_sentence,
)

CPU = torch.device("cpu")


def finals_of(geometry):
    return tuple(Final(geometry, k, 9_000.0, fas_course_geometry(c.length_m)) for k, c in enumerate(geometry.candidates))


@pytest.fixture(scope="module")
def words():
    return Words(instruction_spec())


def test_the_masks_count_the_labelled_words_they_block_and_never_unchanged(tmp_path, words):
    """A sentence whose first predicted step says "no level-off" and 3°: one altitude and one angle word said, none
    blocked; the same row saying the lowest level (below the DA and the glidepath edge wherever the aircraft is): that
    word blocked."""
    directory = tmp_path / "artefact"
    prior_artefact(directory, interval_s=2.0)
    (signals, *_) = load_signals(directory, "train")
    rows = prior_closed_loop_sentence(signals, words, interval_s=2.0).rows
    finals = finals_of(parallel_airport())
    assert runner.blocked_by_masks(rows, finals, words, 2.0) == {ALTITUDE: (1, 0), ANGLE: (1, 0)}
    grid = rows.grid.copy()
    grid[0, ALTITUDE] = 0
    assert runner.blocked_by_masks(dataclasses.replace(rows, grid=grid), finals, words, 2.0)[ALTITUDE] == (1, 1)
    with pytest.raises(ValueError, match="Δ rows"):
        runner.blocked_by_masks(rows, finals, words, 4.0)


def test_the_masks_on_a_go_around_read_g_and_the_runway_after_the_word(tmp_path, words):
    """A sentence with a go-around (row 12: a level 300 m above and the climb, then "09" again at row 20): while G is
    in force the masks' rule of no climb does not apply, and the level above is permitted; the words of the go-around
    are counted."""
    directory = tmp_path / "artefact"
    prior_artefact(directory, interval_s=2.0)
    (signals, *_) = load_signals(directory, "train")
    rows = prior_closed_loop_sentence(signals, words, interval_s=2.0, go_around=True).rows
    assert runner.blocked_by_masks(rows, finals_of(parallel_airport()), words, 2.0) == {ALTITUDE: (2, 0),
                                                                                         ANGLE: (2, 0)}


def base_prior(tmp_path, monkeypatch, *, held_out=None):
    """A synthetic artefact (two val flights, the second's stored outcome not a landing), a base prior written as
    `prior_train` writes one, every live root of the runner replaced. Returns ``(artefact, prior directory)``."""
    from ts_transformer.prior import checkpoint as opening
    from ts_transformer.prior.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
    from ts_transformer.prior.landings import roster_landings
    from ts_transformer.prior.model import Prior, PriorConfig
    from ts_transformer.prior.procedure import PROCEDURE_MASKS
    from ts_transformer.prior.source import artefact_identity
    from ts_transformer.io_utils import file_sha256

    artefact = tmp_path / "artefact"
    words, records = prior_artefact(artefact, interval_s=2.0, outcomes=("landed", "crossed_too_high"))
    landings = {"KXXX": roster_landings(records["KXXX"], ("09", "09L"), fixture_days())}
    digests = {"KXXX": {"09": "0" * 64, "09L": "0" * 64}}
    monkeypatch.setattr(runner, "require_conforming_closed_loop", lambda *given: (None, {"checks": {"stub": True}}, None))
    # the prior opens through `checkpoint.open_prior`: its live roots replaced there
    monkeypatch.setattr(opening, "airport_landings", lambda geometries, days: landings)
    monkeypatch.setattr(opening, "procedure_digests", lambda geometries, root: digests)
    monkeypatch.setattr(runner, "airport_finals", lambda geometry, root: finals_of(geometry))
    monkeypatch.setattr(runner, "git_state", lambda: {"head": "fixture", "dirty": False})
    identity = artefact_identity(artefact, 2.0, landings, "landed")
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64))
    prior = tmp_path / "prior"
    prior.mkdir()
    run = {"airports": ["KXXX"], "held_out": held_out}
    save_checkpoint(prior / "checkpoint.pt", model, model.state_dict(), identity=identity, run={**run, "sample": None},
                    train_config={"tokens_per_batch": 4096})
    (prior / "config.json").write_text(json.dumps({"schema": CHECKPOINT_SCHEMA, "identity": identity, "run": run}))
    (prior / "procedure_masks.json").write_text(json.dumps({
        "set": PROCEDURE_MASKS, "checkpoint_sha256": file_sha256(prior / "checkpoint.pt"), "procedure_data": digests}))
    return artefact, prior


def test_the_readout_reads_the_val_days_once_and_gives_the_flights_outside_the_selection_apart(tmp_path, monkeypatch):
    from ts_transformer.tests.support import PRIOR_ARTEFACT_ROWS

    artefact, prior = base_prior(tmp_path, monkeypatch)
    argv = ["--prior", str(prior), "--instructions", str(artefact), "--executor", str(tmp_path / "executor"),
            "--device", "cpu"]
    assert runner.main([*argv, "--out", str(tmp_path / "readout")]) == 0
    readout = json.loads((tmp_path / "readout" / "readout.json").read_text())
    assert readout["schema"] == runner.VALIDATION_SCHEMA and readout["split"] == "val"
    teacher = readout["teacher_forced"]
    # the landed val flight alone (its word rows from the first predicted step): the selection of D75
    assert teacher["pooled"]["steps"] == teacher["airports"]["KXXX"]["steps"] == PRIOR_ARTEFACT_ROWS
    masks = readout["masks_on_labelled_words"]
    assert set(masks["inside"]["KXXX"]) == set(masks["outside_outcome"]["KXXX"]) == {"altitude", "angle"}
    assert masks["outside_fault"] == {}
    config = json.loads((tmp_path / "readout" / "config.json").read_text())
    assert config["checks"] == {"stub": True} and config["selection"] == "landed"
    assert set(config["identity"]["selection"]["counts"]) == {"train", "select"}       # no val count shown (D85)
    with pytest.raises(ValueError, match="run again only to its own output"):     # read once (D85, D119)
        runner.main([*argv, "--out", str(tmp_path / "again")])
    assert not (tmp_path / "again").exists()


VAL_READERS = ("closed_loop_sentences", "faulty_flights", "signals_flights", "load_signals", "load_sentences")


def val_reads_after_the_claim(monkeypatch, reader_module):
    """Each read of the val days' sentences, stored outcomes, faulty-track marks or signals (`VAL_READERS`, wherever
    they are imported) recorded with whether ``reader_module``'s claim of the val read came before it."""
    from ts_transformer.instructions import artefact as artefact_module
    from ts_transformer.instructions import faults as faults_module
    from ts_transformer.prior import checkpoint as checkpoint_module
    from ts_transformer.prior import source as source_module

    claimed, reads = [], []
    for module in (artefact_module, faults_module, source_module, checkpoint_module, reader_module):
        for name in VAL_READERS:
            if hasattr(module, name):
                def read(directory, split, *given, _real=getattr(module, name), _name=name, **named):
                    if split == "val":
                        reads.append((_name, bool(claimed)))
                    return _real(directory, split, *given, **named)

                monkeypatch.setattr(module, name, read)
    claim = reader_module.claim_validation_read
    monkeypatch.setattr(reader_module, "claim_validation_read", lambda *given: (claimed.append(True), claim(*given)))
    return reads


def test_a_claim_is_written_and_spent_only_with_its_readout(tmp_path):
    """D119: `written_claim` names a claimed output once its readout.json is there (None before: no claim, or a read
    that stopped); `spend_validation_claim` refuses an output the claim does not name, or one without its readout."""
    from ts_transformer.prior.checkpoint import (CLAIM_SPENT_BY, claim_validation_read, spend_validation_claim,
                                                 validation_claim, written_claim)

    prior, out = tmp_path / "prior", tmp_path / "readout"
    prior.mkdir()
    assert written_claim(prior, "reader") is None
    claim_validation_read(prior, "reader", out, {})
    assert validation_claim(prior, "reader") is not None and written_claim(prior, "reader") is None
    out.mkdir()
    with pytest.raises(ValueError, match=f"holds no {CLAIM_SPENT_BY}"):
        spend_validation_claim(prior, "reader", out)
    (out / CLAIM_SPENT_BY).write_text("{}")
    other = tmp_path / "other"
    other.mkdir()
    (other / CLAIM_SPENT_BY).write_text("{}")
    with pytest.raises(ValueError, match="names"):
        spend_validation_claim(prior, "reader", other)
    assert written_claim(prior, "reader") == validation_claim(prior, "reader")
    spend_validation_claim(prior, "reader", out)
    assert "spent_utc" in json.loads((prior / "val_read_reader.json").read_text())


def test_the_val_readout_reads_nothing_of_val_before_its_claim(tmp_path, monkeypatch):
    """D85, D119: the sentences, the stored outcomes and the faulty-track marks of the val days are read only after the
    claim."""
    artefact, prior = base_prior(tmp_path, monkeypatch)
    reads = val_reads_after_the_claim(monkeypatch, runner)
    assert runner.main(["--prior", str(prior), "--instructions", str(artefact), "--executor",
                        str(tmp_path / "executor"), "--device", "cpu", "--out", str(tmp_path / "readout")]) == 0
    names = {name for name, _ in reads}
    assert {"closed_loop_sentences", "faulty_flights"} <= names and all(after for _, after in reads), reads


def test_a_val_read_that_stopped_before_its_readout_runs_again_to_its_own_output_only(tmp_path, monkeypatch, capsys):
    """D119: a claim is spent when its readout is written. A read that stops after the claim (a crash, a kill) may run
    again to the output it claimed, never to another; once written, the claim refuses every read."""
    from ts_transformer.prior.checkpoint import claim_validation_read, validation_claim

    artefact, prior = base_prior(tmp_path, monkeypatch)
    argv = ["--prior", str(prior), "--instructions", str(artefact), "--executor", str(tmp_path / "executor"),
            "--device", "cpu"]
    evaluate = runner.evaluate

    def crash(*given, **named):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(runner, "evaluate", crash)
    with pytest.raises(RuntimeError, match="out of memory"):
        runner.main([*argv, "--out", str(tmp_path / "readout")])
    assert validation_claim(prior, "prior_validation") is not None and not (tmp_path / "readout").exists()
    monkeypatch.setattr(runner, "evaluate", evaluate)
    with pytest.raises(ValueError, match="run again only to its own output"):
        runner.main([*argv, "--out", str(tmp_path / "other")])
    (tmp_path / "readout").mkdir()                                  # what a run stopped while writing leaves
    with pytest.raises(SystemExit):
        runner.main([*argv, "--out", str(tmp_path / "readout")])
    assert "a run that stopped" in capsys.readouterr().err
    (tmp_path / "readout").rmdir()                                  # moved aside: run again to its own output
    assert runner.main([*argv, "--out", str(tmp_path / "readout")]) == 0
    record = json.loads((prior / "val_read_prior_validation.json").read_text())
    assert record["out"] == validation_claim(prior, "prior_validation") and "spent_utc" in record
    with pytest.raises(ValueError, match="read by prior_validation already"):
        claim_validation_read(prior, "prior_validation", tmp_path / "readout", {"split": "val"})
    (tmp_path / "readout").rename(tmp_path / "archived")           # the readout moved away later: still spent
    with pytest.raises(ValueError, match="read by prior_validation already"):
        claim_validation_read(prior, "prior_validation", tmp_path / "readout", {"split": "val"})


def test_a_smoke_reads_the_select_days_never_the_val_days(tmp_path, monkeypatch):
    artefact, prior = base_prior(tmp_path, monkeypatch)
    argv = ["--prior", str(prior), "--instructions", str(artefact), "--executor", str(tmp_path / "executor"),
            "--device", "cpu"]
    with pytest.raises(SystemExit):
        runner.main([*argv, "--smoke", "--out", str(tmp_path / "a")])               # a smoke on val
    with pytest.raises(SystemExit):
        runner.main([*argv, "--split", "select", "--out", str(tmp_path / "b")])     # the formal one on select
    assert runner.main([*argv, "--split", "select", "--smoke", "--out", str(tmp_path / "c")]) == 0
    assert json.loads((tmp_path / "c" / "readout.json").read_text())["split"] == "select"
    assert not list(prior.glob("val_read_*"))                                         # val not read


def test_a_smoke_reads_no_outcome_of_a_val_sentence(tmp_path, monkeypatch):
    """D85: before the base's one validation readout, nothing reads the val days' sentences — not their stored outcomes
    for the identity either (`checkpoint.open_prior` compares the val counts by the val file's sha256): every read of
    val's closed-loop sentences fails here, and the smoke runs."""
    from ts_transformer.instructions import artefact as artefact_module
    from ts_transformer.prior import source as source_module

    artefact, prior = base_prior(tmp_path, monkeypatch)
    read = artefact_module.closed_loop_sentences
    opened = []

    def guarded(directory, split, *args, **kwargs):
        opened.append(split)
        if split == "val":
            raise AssertionError("a val closed-loop sentence read")
        return read(directory, split, *args, **kwargs)

    monkeypatch.setattr(source_module, "closed_loop_sentences", guarded)
    monkeypatch.setattr(runner, "closed_loop_sentences", guarded)
    argv = ["--prior", str(prior), "--instructions", str(artefact), "--executor", str(tmp_path / "executor"),
            "--device", "cpu", "--split", "select", "--smoke", "--out", str(tmp_path / "c")]
    assert runner.main(argv) == 0
    assert set(opened) == {"train", "select"}


def test_the_readout_is_the_base_s_alone(tmp_path, monkeypatch):
    artefact, prior = base_prior(tmp_path, monkeypatch, held_out="KXXX")
    with pytest.raises(SystemExit, match="is a fold"):
        runner.main(["--prior", str(prior), "--instructions", str(artefact), "--executor", str(tmp_path / "executor"),
                     "--out", str(tmp_path / "readout"), "--device", "cpu"])
    assert not (tmp_path / "readout").exists()


def test_the_masks_readout_gives_a_flight_with_a_faulty_track_apart(tmp_path):
    """D111: a val flight stage A marks as faulty is counted under `outside_fault`, whatever its outcome; the flight
    left out by its outcome stays under `outside_outcome`."""
    from ts_transformer.instructions.artefact import closed_loop_sentences, load_spec, signals_flights

    artefact = tmp_path / "artefact"
    words, _ = prior_artefact(artefact, interval_s=2.0, outcomes=("landed", "crossed_too_high"))
    sentences = closed_loop_sentences(artefact, "val", 2.0, load_spec(artefact))
    finals = {"KXXX": finals_of(parallel_airport())}
    args = (sentences, signals_flights(artefact, "val"), finals, words, 2.0, "landed")
    plain, marked = runner.masks_readout(*args, {}), runner.masks_readout(*args, {0: ("a fault",)})
    assert plain["outside_fault"] == {} and plain["inside"] and plain["outside_outcome"]
    assert marked["outside_fault"] == plain["inside"] and marked["inside"] == {}
    assert marked["outside_outcome"] == plain["outside_outcome"]


def test_after_its_claim_the_val_readout_refuses_val_marks_that_are_not_the_prior_s(tmp_path, monkeypatch):
    """D85, D111: the identity's val counts are compared by the val readers after the claim of the val read, from the
    artefact's outcomes and marks (a mark comes from the val signals, which no sha256 of the identity holds)."""
    from ts_transformer.prior import source as source_module

    artefact, prior = base_prior(tmp_path, monkeypatch)
    monkeypatch.setattr(source_module, "faulty_flights", lambda d, split: {0: ("a fault",)} if split == "val" else {})
    argv = ["--prior", str(prior), "--instructions", str(artefact), "--executor", str(tmp_path / "executor"),
            "--device", "cpu", "--out", str(tmp_path / "readout")]
    with pytest.raises(ValueError, match="faulty-track marks changed"):
        runner.main(argv)
    assert (prior / "val_read_prior_validation.json").exists() and not (tmp_path / "readout").exists()


def test_a_claim_names_its_readout_relative_to_the_repository(tmp_path):
    """D85: the claim of the val read names the readout as `repo_layout.repo_relative` does — relative inside the
    repository, so a worktree removed after its campaign leaves the name valid; as given outside it."""
    from ts_transformer.prior.checkpoint import claim_validation_read, validation_claim
    from ts_transformer.repo_layout import REPO_ROOT

    inside, outside = tmp_path / "inside", tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    claim_validation_read(inside, "reader", REPO_ROOT / "4dTrajectory/outputs/POOLED/prior/x/base/free_generation", {})
    assert validation_claim(inside, "reader") == "4dTrajectory/outputs/POOLED/prior/x/base/free_generation"
    claim_validation_read(outside, "reader", tmp_path / "readout", {})
    assert validation_claim(outside, "reader") == str(tmp_path / "readout")
    assert validation_claim(outside, "another") is None


def test_a_kill_between_the_readout_and_the_spent_mark_is_marked_spent_by_the_next_run(tmp_path, monkeypatch):
    """D128: killed after its readout.json and before the claim is marked spent, the read is not left open — the next
    run finds the readout written, marks the claim spent, then refuses; the output moved away later opens nothing."""
    from ts_transformer.prior.checkpoint import claim_validation_read

    artefact, prior = base_prior(tmp_path, monkeypatch)
    argv = ["--prior", str(prior), "--instructions", str(artefact), "--executor", str(tmp_path / "executor"),
            "--device", "cpu", "--out", str(tmp_path / "readout")]
    spend = runner.spend_validation_claim

    def killed(*given):
        raise KeyboardInterrupt("killed between the readout and the spent mark")

    monkeypatch.setattr(runner, "spend_validation_claim", killed)
    with pytest.raises(KeyboardInterrupt):
        runner.main(argv)
    claim = prior / "val_read_prior_validation.json"
    record = json.loads(claim.read_text())
    assert (tmp_path / "readout" / "readout.json").exists() and "spent_utc" not in record
    assert record["options"] == {"split": "val", "device": "cpu"}
    monkeypatch.setattr(runner, "spend_validation_claim", spend)
    with pytest.raises(SystemExit):
        runner.main(argv)
    assert "spent_utc" in json.loads(claim.read_text())
    (tmp_path / "readout").rename(tmp_path / "archived")
    with pytest.raises(ValueError, match="read by prior_validation already"):
        claim_validation_read(prior, "prior_validation", tmp_path / "readout", {"split": "val"})


def test_a_second_run_to_one_output_is_refused_while_the_first_holds_the_claim(tmp_path):
    """D128: the val read is held under an exclusive lock while its reader runs: a second run is refused by name, also
    from another thread; once the first ends, a rerun with the same options may run, one with others may not; a claim
    written before D128 (no options) is not run again."""
    import threading

    from ts_transformer.prior.checkpoint import claim_validation_read, lock_val_read

    prior, out = tmp_path / "prior", tmp_path / "readout"
    prior.mkdir()
    first = lock_val_read(prior, "reader")
    claim_validation_read(prior, "reader", out, {"seed": 1})
    refused = []

    def second():
        try:
            lock_val_read(prior, "reader")
        except ValueError as error:
            refused.append(str(error))

    thread = threading.Thread(target=second)
    thread.start()
    thread.join()
    assert refused and "another run holds the claim" in refused[0]
    first.close()                                                         # the first run ended (stopped)
    with lock_val_read(prior, "reader"):
        with pytest.raises(ValueError, match=r"other \['seed'\]"):
            claim_validation_read(prior, "reader", out, {"seed": 2})
        claim_validation_read(prior, "reader", out, {"seed": 1})          # the same read again
    assert not list(prior.glob("*.tmp"))                                  # written whole, nothing left beside it
    old = tmp_path / "old"
    old.mkdir()
    (old / "val_read_reader.json").write_text(json.dumps({"reader": "reader", "out": str(out), "utc": "x"}))
    with pytest.raises(ValueError, match="records no options"):
        claim_validation_read(old, "reader", out, {"seed": 1})
