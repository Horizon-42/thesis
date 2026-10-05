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


def base_prior(tmp_path, monkeypatch, *, held_out=None):
    """A synthetic artefact (two val flights, the second's stored outcome not a landing), a base prior written as
    `prior_train` writes one, every live root of the runner replaced. Returns ``(artefact, prior directory)``."""
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
    monkeypatch.setattr(runner, "airport_landings", lambda geometries, days: landings)
    monkeypatch.setattr(runner, "procedure_digests", lambda geometries, root: digests)
    monkeypatch.setattr(runner, "airport_finals", lambda geometry, root: finals_of(geometry))
    identity = artefact_identity(artefact, 2.0, landings, "landed")
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64))
    prior = tmp_path / "prior"
    prior.mkdir()
    run = {"airports": ["KXXX"], "held_out": held_out}
    save_checkpoint(prior / "checkpoint.pt", model, model.state_dict(), identity=identity, run={**run, "sample": None},
                    train_config={})
    (prior / "config.json").write_text(json.dumps({"schema": CHECKPOINT_SCHEMA, "identity": identity, "run": run}))
    (prior / "procedure_masks.json").write_text(json.dumps({
        "set": PROCEDURE_MASKS, "checkpoint_sha256": file_sha256(prior / "checkpoint.pt"), "procedure_data": digests}))
    return artefact, prior


def test_the_readout_reads_the_val_days_once_and_gives_the_flights_outside_the_selection_apart(tmp_path, monkeypatch):
    artefact, prior = base_prior(tmp_path, monkeypatch)
    out = tmp_path / "readout"
    argv = ["--prior", str(prior), "--instructions", str(artefact), "--executor", str(tmp_path / "executor"),
            "--out", str(out), "--device", "cpu", "--smoke"]
    assert runner.main(argv) == 0
    readout = json.loads((out / "readout.json").read_text())
    assert readout["schema"] == runner.VALIDATION_SCHEMA and readout["split"] == "val"
    teacher = readout["teacher_forced"]
    assert teacher["pooled"]["steps"] == teacher["airports"]["KXXX"]["steps"] > 0     # the landed val flight alone
    masks = readout["masks_on_labelled_words"]
    assert set(masks["inside"]["KXXX"]) == set(masks["outside"]["KXXX"]) == {"altitude", "angle"}
    config = json.loads((out / "config.json").read_text())
    assert config["checks"] == {"stub": True} and config["selection"] == "landed"
    with pytest.raises(SystemExit):
        runner.main(argv)                                              # the validation days are read once (D85)


def test_the_readout_is_the_base_s_alone(tmp_path, monkeypatch):
    artefact, prior = base_prior(tmp_path, monkeypatch, held_out="KXXX")
    with pytest.raises(SystemExit, match="is a fold"):
        runner.main(["--prior", str(prior), "--instructions", str(artefact), "--executor", str(tmp_path / "executor"),
                     "--out", str(tmp_path / "readout"), "--device", "cpu", "--smoke"])
    assert not (tmp_path / "readout").exists()
