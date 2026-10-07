"""Stage D, MC4: the validation readout of stage D's chosen round (multi-aircraft control §5 item 3; outline D85;
post-training D132; prior D119, D128) — a campaign of one round of stage D from round 0 of a stage C campaign, on A26's
one-flight synthetic artefact (the fixture of `test_post_window_loop`), whose train split stands in for the val days;
the val days are never read. Every write root under tmp."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from ts_transformer.experiments import multi_train, post_train
from ts_transformer.experiments import multi_validation as validation
from ts_transformer.experiments.multi_train import MULTI_CAMPAIGN_SCHEMA, MULTI_CLAIM_READER, MultiSettings, stage_d
from ts_transformer.multi.windows import REAL_KIND
from ts_transformer.tests.test_multi_train import _multi_settings
from ts_transformer.tests.test_post_branches import _ahead
from ts_transformer.tests.test_post_train import _context, _settings, short_round
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401


@pytest.fixture
def campaign(setup, tmp_path, monkeypatch):  # noqa: F811
    """A stage D campaign of one round (its draw replaced by window A) from round 0 of a stage C campaign, and the
    runner's world: the checks stubbed, the context opened without splits, each split asked for read as the fixture's
    train split, with every split asked for and the claim file at that moment recorded (``asked``)."""
    s = setup
    short_round(monkeypatch, s)
    context = replace(_context(s), base_identity={"selection": {"rule": "all", "counts": {"train": 1, "val": 2}}})
    source = tmp_path / "stage_c"
    post_train.open_campaign(source, {"settings": asdict(_settings(rounds=1, continuations=2)), "smoke": False},
                             {"head": "x", "dirty": False}, {})
    post_train.run_campaign(source, _settings(rounds=1, continuations=2), context)
    start = post_train.start_of(source, 0, formal=False)
    window = replace(_ahead(s["windows"][0]), span_s=1.0)                  # of the campaign's span
    with monkeypatch.context() as patch:
        patch.setattr(multi_train, "draw_windows", lambda context, settings, rng: ([window], {"drawn": {REAL_KIND: 1}}))
        settings = _multi_settings(start=start, spans_s=[1.0, 2.0])            # two spans: the coverage by span
        out = tmp_path / "multi"
        inputs = {"prior": str(tmp_path / "prior"), "instructions": str(s["directory"]),
                  "executor": str(tmp_path / "executor"), "windows": str(tmp_path / "census"),
                  "procedure_root": str(tmp_path / "cifp"), "settings": asdict(settings), "smoke": False}
        post_train.open_campaign(out, inputs, {"head": "x", "dirty": False}, {}, schema=MULTI_CAMPAIGN_SCHEMA,
                                 reader=MULTI_CLAIM_READER, settings_type=MultiSettings)
        post_train.run_campaign(out, settings, context, stage=stage_d())
    assert post_train.done_rounds(out) == 1
    asked = []
    train = context.splits["train"]
    claim = out / f"val_read_{MULTI_CLAIM_READER}.json"
    monkeypatch.setattr(validation, "git_state", lambda: {"head": "x", "dirty": False})
    monkeypatch.setattr(validation, "require_conforming_closed_loop", lambda *a: (None, {"checks": {"stub": True}}, None))
    monkeypatch.setattr(validation, "checked_edges", lambda path: None)
    monkeypatch.setattr(validation, "open_context", lambda *a, **k: (
        asked.append(("context", k["splits"], k["data"], k["formal"], claim.exists())),
        replace(context, splits={}))[1])
    monkeypatch.setattr(validation, "require_selection_of", lambda *a: asked.append(("recount", a[-1], claim.exists())))

    def split_data(instructions, split, words, interval_s, geometries, executor):
        assert executor == tmp_path / "executor"
        asked.append(("split", split, claim.exists()))
        return train

    monkeypatch.setattr(validation, "split_data", split_data)
    return out, asked, claim


def _argv(campaign, out, *more):
    return ["--campaign", str(campaign), "--round", "0", "--out", str(out), "--device", "cpu", *more]


def test_stage_ds_chosen_round_reads_the_val_days_once(campaign, tmp_path):
    """§5 item 3: the round's model checked, the context opened formal without splits, the claim made, the val days'
    selection recounted and only then opened; the readout by airport and kind, the coverage stated; the claim spent; a
    second read refused, to the same output and to another."""
    out, asked, claim = campaign
    readout_dir = tmp_path / "validation"
    assert validation.main(_argv(out, readout_dir)) == 0
    assert asked == [("context", (), False, True, False), ("recount", "val", True), ("split", "val", True)]
    readout = json.loads((readout_dir / "readout.json").read_text())
    assert readout["schema"] == validation.MULTI_VALIDATION_SCHEMA and readout["split"] == "val"
    assert readout["round"] == 0 and [w["span_s"] for w in readout["windows"]] == [1.0, 2.0]
    (coverage,) = readout["coverage"].values()
    assert coverage == {"anchors": 1, "read": {"1": 1, "2": 1}}           # by span
    (airport,) = readout["readout"].values()
    assert airport["all"]["all"]["windows"] == 2 and airport["1"]["all"]["aircraft"] == 1 and "reward_mean" in airport
    assert airport["2"]["all"]["windows"] == 1
    held = json.loads(claim.read_text())
    assert "spent_utc" in held and held["options"] == {"round": 0, "device": "cpu"}
    config = json.loads((readout_dir / "config.json").read_text())
    assert config["schema"] == validation.MULTI_VALIDATION_SCHEMA and config["checks"] == {"stub": True}
    assert config["base"]["selection"]["counts"] == {"train": 1}                  # val's counts not shown
    with pytest.raises(SystemExit):
        validation.main(_argv(out, readout_dir))
    with pytest.raises(ValueError, match="claimed by"):
        validation.main(_argv(out, tmp_path / "again"))
    assert not (tmp_path / "again").exists()


def test_a_read_is_refused_by_name_before_anything_opens(campaign, tmp_path, capsys):
    """A round not done, a campaign of stage C, and a campaign not done with all its rounds are refused before the
    context is opened."""
    out, asked, _ = campaign
    with pytest.raises(SystemExit):
        validation.main(["--campaign", str(out), "--round", "1", "--out", str(tmp_path / "v"), "--device", "cpu"])
    assert "holds the checkpoints of rounds 0–0, not round 1" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        validation.main(_argv(tmp_path / "stage_c", tmp_path / "v"))
    assert f"not {MULTI_CAMPAIGN_SCHEMA}" in capsys.readouterr().err
    record = json.loads((out / "campaign.json").read_text())
    record["inputs"]["settings"]["rounds"] = 2
    (out / "campaign.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit):
        validation.main(_argv(out, tmp_path / "v"))
    assert "has done 1 of its 2 rounds" in capsys.readouterr().err
    assert asked == []


def test_a_smoke_read_reads_the_select_days_claims_nothing_and_is_the_rounds_readout(campaign, tmp_path):
    """``--smoke`` (the review of MC4's timing, S2-3): the select days read, no claim made, the context opened as a smoke;
    its readout is the round's own selection readout (the same windows and numbers: the validation read is the
    selection readout on other days)."""
    out, asked, claim = campaign
    readout_dir = tmp_path / "smoke"
    assert validation.main(_argv(out, readout_dir, "--smoke")) == 0
    assert asked == [("context", (), False, False, False), ("split", "select", False)] and not claim.exists()
    readout = json.loads((readout_dir / "readout.json").read_text())
    assert readout["split"] == "select"
    round_ = json.loads((out / "round_0" / "round.json").read_text())
    assert readout["readout"] == round_["selection_readout"]
