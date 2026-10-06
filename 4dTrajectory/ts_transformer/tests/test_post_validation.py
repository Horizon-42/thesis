"""Stage C, C10: the validation readout of the chosen round (post-training §8 C10; prior D119, D128) — a campaign of one
round on A26's one-flight synthetic artefact (the fixture of `test_post_window_loop`), whose train split stands in for
the val days; the val days are never read. Every write root under tmp."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest
import torch

from ts_transformer.experiments import post_train
from ts_transformer.experiments import post_validation as validation
from ts_transformer.experiments.post_train import KINDS, done_rounds, open_campaign, run_campaign
from ts_transformer.post.scene import REAL
from ts_transformer.tests.test_post_branches import _ahead, _round
from ts_transformer.tests.test_post_train import _context, _settings  # noqa: F401
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401


@pytest.fixture
def campaign(setup, tmp_path, monkeypatch):  # noqa: F811
    """A campaign of one round (its speaking replaced, as the export's test; its selection readout run, on the train
    split standing in for the select days) and the runner's world: the checks stubbed, the context opened without
    splits, and each split asked for read as the fixture's train split, with every split asked for and the claim file at
    that moment recorded (``asked``)."""
    s = setup
    # the base's identity with its selection's counts, val's among them (held, never shown: prior D85, D120)
    context = replace(_context(s), base_identity={"selection": {"rule": "all", "counts": {"train": 1, "val": 2}}})
    settings = _settings(rounds=1, per_kind={**dict.fromkeys(KINDS, 0), REAL: 1})
    model, _ = post_train.start_model(context, settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))
    with monkeypatch.context() as patch:
        patch.setattr(post_train, "speak_round", lambda model, context, windows, settings, round_, directory, speakers: (
            torch.save([rewarded], directory / "groups_0.pt"), {"windows": 1})[1])
        out = tmp_path / "campaign"
        inputs = {"prior": str(tmp_path / "prior"), "instructions": str(s["directory"]),
                  "executor": str(tmp_path / "executor"), "windows": str(tmp_path / "census"),
                  "procedure_root": str(tmp_path / "cifp"), "settings": asdict(settings), "smoke": False}
        open_campaign(out, inputs, {"head": "x", "dirty": False}, {})
        run_campaign(out, settings, context)
    assert done_rounds(out) == 1
    asked = []
    train = context.splits["train"]
    claim = out / f"val_read_{validation.CLAIM_READER}.json"
    monkeypatch.setattr(validation, "git_state", lambda: {"head": "x", "dirty": False})
    monkeypatch.setattr(validation, "require_conforming_closed_loop", lambda *a: (None, {"checks": {"stub": True}}, None))
    monkeypatch.setattr(validation, "checked_edges", lambda path: None)
    monkeypatch.setattr(validation, "open_context", lambda *a, **k: (
        asked.append(("context", k["splits"], k["data"], k["formal"], claim.exists())),
        replace(context, splits={}))[1])
    monkeypatch.setattr(validation, "require_selection_of", lambda *a: asked.append(("recount", a[-1], claim.exists())))
    def split_data(instructions, split, words, interval_s, geometries, executor):
        assert executor == tmp_path / "executor"           # the campaign's executor opens the val days' start
        asked.append(("split", split, claim.exists()))
        return train

    monkeypatch.setattr(validation, "split_data", split_data)
    return out, asked, claim


def _argv(campaign, out, *more):
    return ["--campaign", str(campaign), "--round", "0", "--out", str(out), "--device", "cpu", *more]


def test_the_chosen_round_reads_the_val_days_once(campaign, tmp_path):
    out, asked, claim = campaign
    readout_dir = tmp_path / "validation"
    assert validation.main(_argv(out, readout_dir)) == 0
    # only the val days opened, after the claim; neither the train nor the select days
    # the formal read: the base checked as stage B's formal base (D132), before the claim
    assert asked == [("context", (), False, True, False), ("recount", "val", True), ("split", "val", True)]
    readout = json.loads((readout_dir / "readout.json").read_text())
    assert readout["split"] == "val" and readout["round"] == 0 and len(readout["windows"]) == 1
    (coverage,) = readout["coverage"].values()
    assert coverage == {"real": 1, "left_out_inside_loss": 0, "read": 1}
    (airport,) = readout["readout"].values()
    assert airport["windows"] == 1 and sum(airport["outcomes"].values()) == 1
    held = json.loads(claim.read_text())
    assert "spent_utc" in held and held["options"] == {"round": 0, "device": "cpu"}
    config = json.loads((readout_dir / "config.json").read_text())
    assert config["schema"] == validation.VALIDATION_SCHEMA and config["checks"] == {"stub": True}
    assert config["base"]["selection"]["counts"] == {"train": 1}                  # val's counts not shown
    # a second read is refused: to the same output, to another
    with pytest.raises(SystemExit):
        validation.main(_argv(out, readout_dir))
    with pytest.raises(ValueError, match="claimed by"):
        validation.main(_argv(out, tmp_path / "again"))
    assert not (tmp_path / "again").exists() and [a for a in asked if a[0] == "split"] == [("split", "val", True)]


def test_a_read_that_stopped_runs_again_only_as_the_same_read(campaign, tmp_path, monkeypatch):
    """A run that stopped before its readout (the claim made, no output): run again to the same output with the same
    options, and only so (D119, D128); a round the campaign does not hold is refused before the claim."""
    out, asked, claim = campaign
    readout_dir = tmp_path / "validation"
    real = validation.selection_readout
    monkeypatch.setattr(validation, "selection_readout", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("killed")))
    with pytest.raises(RuntimeError, match="killed"):
        validation.main(_argv(out, readout_dir))
    assert claim.exists() and not readout_dir.exists() and "spent_utc" not in json.loads(claim.read_text())
    monkeypatch.setattr(validation, "selection_readout", real)
    with pytest.raises(ValueError, match="other"):
        validation.main(_argv(out, readout_dir)[:-2] + ["--device", "meta"])
    with pytest.raises(SystemExit):
        validation.main(["--campaign", str(out), "--round", "1", "--out", str(readout_dir), "--device", "cpu"])
    assert validation.main(_argv(out, readout_dir)) == 0
    assert "spent_utc" in json.loads(claim.read_text())


def test_a_smoke_reads_the_select_days_and_claims_nothing(campaign, tmp_path, monkeypatch):
    out, asked, claim = campaign
    monkeypatch.setattr(validation, "git_state", lambda: {"head": "x", "dirty": True})
    with pytest.raises(SystemExit):
        validation.main(_argv(out, tmp_path / "formal"))                        # a formal read needs a clean tree
    assert not claim.exists()
    assert validation.main(_argv(out, tmp_path / "smoke", "--smoke")) == 0
    assert [a for a in asked if a[0] in ("split", "recount")] == [("split", "select", False)] and not claim.exists()
    assert asked[-2] == ("context", (), False, False, False)                     # a smoke: the base is not checked
    assert json.loads((tmp_path / "smoke" / "readout.json").read_text())["split"] == "select"


def test_the_lock_and_a_kill_after_the_readout(campaign, tmp_path, monkeypatch):
    """D128: a second run while one holds the read's lock is refused; a run killed between its readout and the spent mark
    leaves the claim open, and the next run marks it spent, then refuses."""
    out, asked, claim = campaign
    readout_dir = tmp_path / "validation"
    held = validation.lock_val_read(out, validation.CLAIM_READER)
    with pytest.raises(ValueError, match="another run holds"):
        validation.main(_argv(out, readout_dir))
    held.close()
    real = validation.spend_validation_claim
    monkeypatch.setattr(validation, "spend_validation_claim", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        validation.main(_argv(out, readout_dir))
    assert (readout_dir / "readout.json").exists() and "spent_utc" not in json.loads(claim.read_text())
    monkeypatch.setattr(validation, "spend_validation_claim", real)
    with pytest.raises(SystemExit):
        validation.main(_argv(out, readout_dir))
    assert "spent_utc" in json.loads(claim.read_text())


def test_the_read_is_the_selection_readout_and_waits_for_every_round(campaign, tmp_path):
    """A smoke of round 0 (the select days) gives back the readout the campaign's round 0 wrote: the same windows and
    random numbers (the round's weights: `round_model`'s identity check, its own test). A campaign with rounds still to
    do has no formal read; a smoke reads it."""
    out, asked, claim = campaign
    assert validation.main(_argv(out, tmp_path / "smoke", "--smoke")) == 0
    smoke = json.loads((tmp_path / "smoke" / "readout.json").read_text())
    assert smoke["readout"] == json.loads((out / "round_0" / "round.json").read_text())["selection_readout"]
    record = json.loads((out / "campaign.json").read_text())
    record["inputs"]["settings"]["rounds"] = 2
    (out / "campaign.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit):
        validation.main(_argv(out, tmp_path / "early"))
    assert not claim.exists() and not (tmp_path / "early").exists()
    assert validation.main(_argv(out, tmp_path / "smoke_early", "--smoke")) == 0
