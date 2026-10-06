"""Stage C, C8: the profile of a round (post-training §8 C8) — on A26's one-flight synthetic artefact."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from ts_transformer.experiments import post_profile
from ts_transformer.experiments.post_profile import APPROXIMATION, PARTS, memory, part_times, pass_memory, profile
from ts_transformer.tests.test_post_train import _context, _settings, select_is_train, short_round  # noqa: F401
from ts_transformer.tests.test_post_window_loop import CPU, setup  # noqa: F401


def test_a_profile_times_each_part_of_a_batch_and_the_round(setup, tmp_path, monkeypatch, select_is_train):  # noqa: F811
    window = short_round(monkeypatch, setup)
    monkeypatch.setattr(post_profile, "draw_round", lambda context, per_kind, rng: ([window], {}))
    saved = []
    record = profile(_context(setup), _settings(rounds=1, continuations=2), tmp_path / "profile", [1, 2],
                     lambda parts: saved.append(set(parts)))
    batch = record["one_batch"]
    assert set(batch["parts_s"]) == set(PARTS) and batch["approximation"] == APPROXIMATION
    for part in ("prior_step", "executor_step", "speaker_masks", "edge_features", "separation_scene"):
        assert 0.0 < batch["parts_s"][part] <= batch["wall_s"], part
    assert record["batches"] == [1] and batch["windows"] == 1 and batch["groups"] == 1
    rounds = record["round"]
    assert rounds["speaking"]["windows"] == 1 and rounds["selection_windows"] == 1
    assert min(rounds["speaking_s"], rounds["pass_s"], rounds["selection_readout_s"]) >= 0.0
    assert (tmp_path / "profile" / "round_0").is_dir() and rounds["groups_bytes"] > 0
    assert {"host_peak_rss_gib", "host_available_gib"} <= set(record["memory_before"])
    assert "gpu_peak_allocated_gib" not in memory(CPU)
    # one update's memory, before the pass: the round wrote no informative group (its rewards alike), so no k is held
    # (an update measured: the next test)
    assert rounds["speaking"]["informative_groups"] == record["pass_memory"]["groups_written"] == 0
    assert record["pass_memory"]["updates"] == {"1": "fewer groups held", "2": "fewer groups held"}
    # the record handed on after each part: a failing part leaves the parts before it
    assert "one_batch" in saved[0] and "round" not in saved[0] and "pass_memory" in saved[2]
    assert len(saved) == 4 and saved[-1] == set(record)


def _group(rows, width):
    """A stand-in group of one sentence: ``rows`` rows of ``width`` traffic tokens."""
    return SimpleNamespace(sentences=[SimpleNamespace(tokens=[np.zeros((width, 4), dtype=np.float32)] * rows)])


def test_an_update_out_of_memory_is_recorded_and_stops_the_larger_ones(tmp_path, monkeypatch):
    """The k largest groups (longest sentence, then widest traffic) are measured, read file by file; an out-of-memory
    is recorded and stops the larger k; the record is handed on after each k."""
    directory = tmp_path / "round_0"
    directory.mkdir()
    torch.save([_group(3, 1), _group(9, 1), _group(5, 2)], directory / "groups_0.pt")
    torch.save([_group(9, 4), _group(1, 1), _group(2, 1)], directory / "groups_1.pt")
    tried = []

    def update(model, start, base, batch, data):
        tried.append(batch.chosen)
        if len(batch.chosen) >= 4:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return SimpleNamespace(loss=(model.weight * 0).sum())

    def batch_of(groups, device):
        rows = max(len(g.sentences[0].tokens) for g in groups)
        return SimpleNamespace(chosen=[post_profile.group_size(g) for g in groups],
                               rows=SimpleNamespace(asked=torch.zeros(len(groups), rows, dtype=torch.bool)),
                               traffic=SimpleNamespace(tokens=torch.zeros(len(groups), rows, 2, 4)))

    monkeypatch.setattr(post_profile, "update_loss", update)
    monkeypatch.setattr(post_profile, "samples", batch_of)
    monkeypatch.setattr(post_profile, "collate", lambda data, device: SimpleNamespace(asked=torch.zeros(3, 7)))
    monkeypatch.setattr(post_profile, "PassStart", lambda model: None)
    model = torch.nn.Linear(2, 2)
    context = SimpleNamespace(device=CPU, data=list(range(5)), base=None)
    saved = []
    out = pass_memory(model, context, directory, SimpleNamespace(seed=1, data_sentences=3), [8, 1, 2, 4],
                      lambda part: saved.append(dict(part["updates"])))
    assert tried == [[(9, 4)], [(9, 4), (9, 1)], [(9, 4), (9, 1), (5, 2), (3, 1)]]
    assert out["updates"]["4"] == "out of memory" and "8" not in out["updates"] and model.weight.grad is None
    two = out["updates"]["2"]
    assert (two["rows"], two["sentences"], two["data_rows"], two["traffic"]) == (9, 2, 7, 2)
    assert (out["groups_written"], out["largest_rows"], out["largest_traffic"], out["data_sentences"]) == (6, 9, 4, 3)
    assert [len(s) for s in saved] == [1, 2, 3]
    # the widest group is not the longest: it is measured alone for k = 1 and joins the longest for k = 2
    other = tmp_path / "round_1"
    other.mkdir()
    torch.save([_group(10, 1), _group(8, 1), _group(2, 6), _group(7, 1)], other / "groups_0.pt")
    tried.clear()
    out = pass_memory(model, context, other, SimpleNamespace(seed=1, data_sentences=3), [1, 2])
    assert tried == [[(10, 1)], [(2, 6)], [(10, 1), (2, 6)]] and list(out["updates"]) == ["1", "1_widest", "2"]
    assert out["updates"]["2"]["rows"] == out["largest_rows"] == 10 and out["largest_traffic"] == 6


def test_the_update_measured_leaves_the_model_as_it_was(setup, tmp_path):  # noqa: F811
    """A real update (a rewarded group of the synthetic window): measured, and the model's weights, gradients and mode
    afterwards those before it."""
    from dataclasses import replace

    from ts_transformer.experiments.post_train import start_model
    from ts_transformer.tests.test_post_branches import _ahead, _round

    s = setup
    context = _context(s)
    settings = _settings(rounds=1, continuations=2, data_sentences=1)
    model, _ = start_model(context, settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))
    directory = tmp_path / "round_0"
    directory.mkdir()
    torch.save([rewarded], directory / "groups_0.pt")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = pass_memory(model, context, directory, settings, [1, 2])
    assert out["updates"]["1"]["s"] >= 0.0 and out["updates"]["1"]["sentences"] == 3
    assert out["updates"]["1"]["rows"] == out["largest_rows"] and out["updates"]["2"] == "fewer groups held"
    assert out["updates"]["1"]["traffic"] == out["largest_traffic"] and "1_widest" not in out["updates"]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters()) and not model.training


def test_a_part_is_the_largest_of_its_functions_never_their_sum():
    class Stats:
        stats = {("/x/ts_transformer/prior/model.py", 1, "extend"): (1, 1, 0.1, 2.0, {}),
                 ("/x/ts_transformer/prior/model.py", 9, "extend"): (1, 1, 0.1, 3.0, {}),
                 ("/x/other/prior/model.py", 1, "extend"): (1, 1, 0.1, 9.0, {})}
    times = part_times(Stats())
    assert times["prior_step"] == 3.0 and times["executor_step"] == 0.0
