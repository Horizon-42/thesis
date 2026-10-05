"""Stage C, C10: the post-training campaign (post-training §2, §8 C10) — on A26's one-flight synthetic artefact (the
fixture of `test_post_window_loop`); every write root under tmp."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.experiments import post_train
from ts_transformer.experiments.post_train import (
    KINDS, Context, Settings, batches, done_rounds, draw_round, open_campaign, run_campaign, start_model, train_pass,
    update_pairs,
)
from ts_transformer.instructions.artefact import closed_loop_sentences
from ts_transformer.post.scene import INSERTED, LEADER_MOVED, MOVED_START, NO_START_MOVE, REAL
from ts_transformer.prior.source import ArtefactSource
from ts_transformer.tests.test_post_branches import _ahead, _round
from ts_transformer.tests.test_post_window_loop import CPU, DELTA, setup  # noqa: F401

PER_KIND = {REAL: 1, INSERTED: 1, LEADER_MOVED: 1, MOVED_START: 1}


def _settings(**changed):
    values = dict(rounds=2, per_kind=PER_KIND, batch_windows=2, continuations=2, seed=1337, prior_lr=1e-4,
                  traffic_lr=1e-3, weight_decay=0.0, update_groups=1, data_sentences=1, select_per_airport=1,
                  traffic_hidden=16, traffic_heads=4)
    return Settings(**{**values, **changed})


def _context(s):
    code = s["geometry"].code
    train = {"windows": s["windows"], "sentences": closed_loop_sentences(s["directory"], "train", DELTA,
                                                                         s["words"].spec),
             "flights": s["flights"], "signals": {}, "faults": {code: {}}}
    from ts_transformer.instructions.artefact import load_signals

    train["signals"] = {x.dataset_id: x for x in load_signals(s["directory"], "train")}
    data = ArtefactSource(s["directory"], DELTA, "full", {code: s["roster"]}, "all").sentences("train", code)
    return Context(s["directory"], s["directory"].parent / "executor", s["reference"], CPU, s["words"], DELTA, s["base"],
                   {"base": "the fixture's"}, s["geometries"], {code: s["roster"]}, s["finals"],
                   {"train": train, "select": train}, data)


@pytest.fixture
def select_is_train(monkeypatch):
    """The fixture's artefact has a train split only: the selection readout's start reads it as the select days."""
    real = post_train.start_moved
    monkeypatch.setattr(post_train, "start_moved", lambda instructions, split, *a, **k: real(instructions, "train",
                                                                                             *a, **k))


def test_the_settings_take_a_count_of_every_kind_and_positive_sizes():
    assert set(_settings().per_kind) == set(KINDS)
    with pytest.raises(ValueError, match="each kind"):
        _settings(per_kind={REAL: 1})
    with pytest.raises(ValueError, match="not all zero"):
        _settings(per_kind=dict.fromkeys(KINDS, 0))
    with pytest.raises(ValueError, match="positive"):
        _settings(update_groups=0)


def test_a_batch_commands_each_flight_once_in_signal_order():
    windows = [SimpleNamespace(signal_index=i) for i in (3, 3, 1, 2, 3, 1)]
    out = batches(windows, 2)
    assert sorted(p for b in out for p in b) == list(range(6))
    for b in out:
        flights = [windows[p].signal_index for p in b]
        assert len(b) <= 2 and flights == sorted(set(flights))
    assert out == [[2, 0], [3, 1], [5, 4]]


def test_the_draw_counts_each_kind_and_its_shortfall(setup):
    """One flight: a real window and its window B; no flight for A to insert, no aircraft ahead for D — a shortfall,
    recorded. The same seed draws the same windows."""
    context = _context(setup)
    windows, counted = draw_round(context, PER_KIND, np.random.default_rng([1, 0]))
    assert [w.kind for w in windows] == [REAL, MOVED_START]
    assert counted["drawn"] == {REAL: 1, INSERTED: 0, LEADER_MOVED: 0, MOVED_START: 1}
    assert counted["shortfall"] == {INSERTED: 1, LEADER_MOVED: 1} and counted["left_out_inside_loss"] == {}
    assert windows[1].start_move != NO_START_MOVE and windows[0].start_move == NO_START_MOVE
    again, _ = draw_round(context, PER_KIND, np.random.default_rng([1, 0]))
    assert [asdict(w.start_move) for w in again] == [asdict(w.start_move) for w in windows]
    assert batches(windows, 2) == [[0], [1]]                     # the real window and its B command the same flight


def test_the_pass_reads_the_groups_file_by_file_a_few_at_a_time(setup, tmp_path):
    s = setup
    context = _context(s)
    settings = _settings(update_groups=2)
    model, optimizer = start_model(context, settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))
    directory = tmp_path / "round"
    directory.mkdir()
    torch.save([rewarded] * 3, directory / "groups_0.pt")          # 3 groups: an update of 2, then of 1
    torch.save([], directory / "groups_1.pt")                      # a batch without an informative group
    torch.save([rewarded], directory / "groups_10.pt")             # read after groups_1 (by number, not by name)
    sizes = [p.rows.asked.shape[0] for p, _ in update_pairs(directory, context.data, settings,
                                                            np.random.default_rng(0), CPU)]
    assert sizes == [6, 3, 3]                                      # 3 sentences a group
    before = [p.detach().clone() for p in model.parameters()]
    passed = train_pass(model, context, optimizer, directory, settings, np.random.default_rng(0))
    assert passed["updates"] == 3 and np.isfinite(passed["loss"]) and not model.training
    assert any(not torch.equal(a, p) for a, p in zip(before, model.parameters()))
    empty = tmp_path / "empty"
    empty.mkdir()
    assert train_pass(model, context, optimizer, empty, settings, np.random.default_rng(0)) == {"updates": 0}


def short_round(monkeypatch, s):
    """The round's windows replaced by one short window (its own flight inserted 8 s ahead: lost at its first row flown,
    one branch point), so a round flies only a few rows; the draw has its own test, the real window's whole flight
    `test_post_branches`."""
    window = _ahead(s["windows"][0])
    monkeypatch.setattr(post_train, "draw_round", lambda context, per_kind, rng: ([window], {"drawn": {INSERTED: 1}}))
    return window


def test_a_campaign_round_end_to_end(setup, tmp_path, monkeypatch, select_is_train):
    """One round through every step but the draw (`short_round`): the two passes, the pass, the selection readout (the
    real window of the select days), the record and the checkpoint with its identity."""
    s = setup
    short_round(monkeypatch, s)
    settings = _settings(rounds=1, continuations=2)
    out = tmp_path / "campaign"
    open_campaign(out, {"settings": asdict(settings)}, {"head": "x", "dirty": False}, {})
    run_campaign(out, settings, _context(s))
    assert done_rounds(out) == 1
    record = json.loads((out / "round_0" / "round.json").read_text())
    assert [w["kind"] for w in record["windows"]] == [INSERTED] and record["speaking"]["spoken_again"] == 1
    assert record["speaking"]["groups"] == 1 and record["speaking"]["outcomes"] == {"lost_separation": 1}
    assert record["groups_bytes"] > 0 and not list((out / "round_0").glob("groups_*.pt"))     # deleted after the round
    assert record["speaking"]["windows"] == 1 and sum(record["speaking"]["outcomes"].values()) == 1
    readout = record["selection_readout"][s["geometry"].code]
    assert readout["windows"] == 1 and sum(readout["outcomes"].values()) == 1
    identity = torch.load(out / "round_0" / "checkpoint.pt", weights_only=False)["identity"]
    assert identity["schema"] == post_train.POST_CHECKPOINT_SCHEMA and identity["base"] == {"base": "the fixture's"}
    assert identity["rounds"] == [record["windows"]]


def test_a_resumed_campaign_is_the_campaign_run_through(setup, tmp_path, monkeypatch):
    """The resume's mechanics, with the speaking and the readout replaced (each round writes the same informative
    groups, so every round updates the model, its data term with dropout): a round broken off is moved aside and run
    again from the last checkpoint, and the weights after the last round are those of the campaign run through, bit
    for bit; other settings are refused, and a directory that holds no campaign."""
    s = setup
    context = _context(s)
    settings = _settings(rounds=2, per_kind={**dict.fromkeys(KINDS, 0), REAL: 1}, update_groups=1)
    model, _ = start_model(context, settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))

    def speak(model, context, windows, settings, round_, directory):
        torch.save([rewarded, rewarded], directory / "groups_0.pt")
        return {"windows": len(windows)}

    monkeypatch.setattr(post_train, "speak_round", speak)
    monkeypatch.setattr(post_train, "selection_readout", lambda *a, **k: {"KXXX": {"reward_mean": 0.0}})
    inputs = {"settings": asdict(settings)}
    through = tmp_path / "through"
    open_campaign(through, inputs, {"head": "x", "dirty": False}, {})
    run_campaign(through, settings, context)
    record = json.loads((through / "round_1" / "round.json").read_text())
    assert record["pass"]["updates"] == 2 and done_rounds(through) == 2
    state = torch.load(through / "round_1" / "checkpoint.pt", weights_only=False)
    broken = tmp_path / "broken"
    open_campaign(broken, inputs, {"head": "x", "dirty": False}, {})
    passes = []
    real = post_train.train_pass

    def killed_in_round_1(*a, **k):
        passes.append(1)
        if len(passes) == 2:
            raise RuntimeError("killed")
        return real(*a, **k)

    monkeypatch.setattr(post_train, "train_pass", killed_in_round_1)
    with pytest.raises(RuntimeError, match="killed"):
        run_campaign(broken, settings, context)
    assert done_rounds(broken) == 1 and (broken / "round_1").exists()
    monkeypatch.setattr(post_train, "train_pass", real)
    reopened = open_campaign(broken, inputs, {"head": "y", "dirty": False}, {})
    assert len(reopened["aborted"]) == 1 and (broken / reopened["aborted"][0]).is_dir()
    assert reopened["resumed"][0]["git"] == {"head": "y", "dirty": False}
    run_campaign(broken, settings, context)
    resumed = torch.load(broken / "round_1" / "checkpoint.pt", weights_only=False)
    assert all(torch.equal(resumed["model"][k], state["model"][k]) for k in state["model"])
    assert any(not torch.equal(state["model"][k], v) for k, v in model.state_dict().items())   # the rounds moved it
    with pytest.raises(SystemExit, match="other inputs"):
        open_campaign(broken, {"settings": asdict(_settings(rounds=3))}, {"head": "y", "dirty": False}, {})
    (tmp_path / "stray").mkdir()
    with pytest.raises(SystemExit, match="holds no campaign"):
        open_campaign(tmp_path / "stray", inputs, {"head": "x", "dirty": False}, {})


def test_a_formal_campaign_needs_a_clean_tree_and_its_intent(tmp_path, monkeypatch):
    """Refused before anything is opened: a tree with changes (unless a smoke), a campaign without an intent."""
    argv = ["--prior", str(tmp_path / "p"), "--instructions", str(tmp_path / "i"), "--executor", str(tmp_path / "e"),
            "--windows", str(tmp_path / "w"), "--procedure-root", str(tmp_path / "cifp"),
            "--out", str(tmp_path / "no_such_campaign_20991231"), "--rounds", "1", "--batch-windows", "1",
            "--seed", "1", "--prior-lr", "1e-4", "--traffic-lr", "1e-3", "--weight-decay", "0", "--update-groups", "1",
            "--data-sentences", "1", "--select-per-airport", "1", "--traffic-hidden", "16", "--traffic-heads", "4"]
    argv += [x for kind in KINDS for x in (f"--windows-{kind.lower()}", "1")]
    monkeypatch.setattr(post_train, "git_state", lambda: {"head": "x", "dirty": True})
    with pytest.raises(SystemExit):
        post_train.main(argv)
    monkeypatch.setattr(post_train, "git_state", lambda: {"head": "x", "dirty": False})
    with pytest.raises(SystemExit):
        post_train.main(argv)
    assert not (tmp_path / "no_such_campaign_20991231").exists()
