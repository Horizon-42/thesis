"""Stage C, P48: the ceiling readout — a campaign of one round on A26's one-flight synthetic artefact (the fixture of
`test_post_window_loop`), whose train split stands in for the select days. Every write root under tmp."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.experiments import post_ceiling as ceiling
from ts_transformer.experiments import post_train
from ts_transformer.experiments.post_train import KINDS, done_rounds, open_campaign, readout_numbers, run_campaign
from ts_transformer.experiments.post_window_loop import LOST_SEPARATION
from ts_transformer.post.reward import LANDED
from ts_transformer.post.scene import REAL
from ts_transformer.tests.test_post_branches import _ahead, _round
from ts_transformer.tests.test_post_train import _context, _settings
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401


@pytest.fixture
def campaign(setup, tmp_path, monkeypatch):  # noqa: F811
    """A campaign of one round (its speaking replaced, as the validation readout's test; its selection readout run) and
    the runner's world: the checks stubbed and the context the fixture's."""
    s = setup
    context = _context(s)
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
    opened = []
    monkeypatch.setattr(ceiling, "git_state", lambda: {"head": "x", "dirty": False})
    monkeypatch.setattr(ceiling, "require_conforming_closed_loop", lambda *a: (None, {"checks": {"stub": True}}, None))
    monkeypatch.setattr(ceiling, "checked_edges", lambda path: None)
    monkeypatch.setattr(ceiling, "open_context", lambda *a, **k: (opened.append((k["splits"], k["formal"])), context)[1])
    return out, opened


def _argv(campaign, out, *more):
    return ["--campaign", str(campaign), "--out", str(out), "--device", "cpu", *more]


def test_each_model_reads_every_window_n_times_and_draw_0_is_the_round_s_readout(campaign, tmp_path):
    """The start and round 0, two draws each: draw 0 of round 0 is the round's selection readout (checked by the runner
    before its other draws); each window's outcome and reward of every draw kept; the curve and the summary written.
    Only the select days are opened."""
    out, opened = campaign
    assert ceiling.main(_argv(out, tmp_path / "ceiling", "--models", "start", "0", "--draws", "2")) == 0
    assert opened == [(("select",), True)]
    config = json.loads((tmp_path / "ceiling" / "config.json").read_text())
    assert config["schema"] == ceiling.CEILING_SCHEMA and config["models"] == ["start", "0"] and len(config["windows"]) == 1
    for name, checked in (("start", False), ("0", True)):
        model = json.loads((tmp_path / "ceiling" / f"model_{name}.json").read_text())
        assert model["draw0_checked"] is checked and len(model["outcomes"]) == 1 and len(model["outcomes"][0]) == 2
        assert set(model["all"]["landed_in_first"]) == {"1", "2"} and model["all"]["windows"] == 1
        assert model["all"]["landed_in_first"]["1"] <= model["all"]["landed_in_first"]["2"]
    summary = json.loads((tmp_path / "ceiling" / "summary.json").read_text())
    assert set(summary["curves"]) == {"start", "0"} and set(summary["never_landed_pairs"]) == {"start&0"}


def test_a_draw_0_that_is_not_the_round_s_readout_stops_the_read(campaign, tmp_path):
    out, _ = campaign
    held = json.loads((out / "round_0" / "round.json").read_text())
    code = next(iter(held["selection_readout"]))
    held["selection_readout"][code]["reward_sum"] += 1.0
    (out / "round_0" / "round.json").write_text(json.dumps(held))
    with pytest.raises(SystemExit, match="draw 0 is not the round's selection readout"):
        ceiling.main(_argv(out, tmp_path / "ceiling", "--models", "0", "--draws", "3"))
    assert not (tmp_path / "ceiling" / "model_0.json").exists()


def test_the_options_are_refused_by_name_before_anything_is_read(campaign, tmp_path):
    out, opened = campaign
    for more in (("--models", "1", "--draws", "2"), ("--models", "0", "0", "--draws", "2"),
                 ("--models", "0", "--draws", "0")):
        with pytest.raises(SystemExit):
            ceiling.main(_argv(out, tmp_path / "ceiling", *more))
    (tmp_path / "taken").mkdir()
    with pytest.raises(SystemExit):
        ceiling.main(_argv(out, tmp_path / "taken", "--models", "0", "--draws", "2"))
    assert opened == [] and not (tmp_path / "ceiling").exists()


def test_draw_0_keeps_the_readout_s_numbers_and_each_draw_has_its_own():
    first = [readout_numbers(1337, 5, d).random() for d in range(3)]
    assert first[0] == np.random.default_rng([1337, 1 << 30, 5]).random()          # the readout's, as before P48
    assert len(set(first)) == 3


def test_the_curve_counts_a_window_landed_once_in_its_first_n_draws():
    windows = [SimpleNamespace(scene=SimpleNamespace(geometry=SimpleNamespace(code=c))) for c in ("KA", "KA", "KB")]
    outcomes = [[LOST_SEPARATION, LANDED, LOST_SEPARATION, LOST_SEPARATION],
                [LOST_SEPARATION] * 4,
                [LANDED, "timeout", LANDED, LANDED]]
    rewards = [[0.0, 1.0, 0.0, 0.0], [0.0] * 4, [1.0, 0.0, 1.0, 0.9]]
    reading = ceiling.ceiling_of(windows, outcomes, rewards)
    assert reading["all"]["landed_in_first"] == {"1": 1 / 3, "2": 2 / 3, "4": 2 / 3}
    assert reading["all"]["best_reward_of_first"]["1"] == pytest.approx(1 / 3)
    assert (reading["all"]["never_landed"], reading["all"]["lost_every_draw"]) == (1, 1)
    assert reading["airports"]["KA"]["landed_in_first"]["2"] == 0.5 and reading["airports"]["KB"]["never_landed"] == 0
    assert ceiling.firsts(32) == [1, 2, 4, 8, 16, 32] and ceiling.firsts(5) == [1, 2, 4, 5] and ceiling.firsts(1) == [1]
