"""Which part of the loss sets a round's pass (`pass_gradients`, the user 2026-10-10): each part's gradient at the start
of a pass, on the loss tests' small prior and synthetic spoken samples. Every write root under tmp."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.experiments import pass_gradients as gradients
from ts_transformer.experiments import post_train
from ts_transformer.post.loss import PassStart, update_gradients
from ts_transformer.post.traffic_attention import parameter_groups
from ts_transformer.prior.batch import collate
from ts_transformer.tests.support import prior_sentence
from ts_transformer.tests.test_post_loss import CPU, FIRST, LENGTH, WORDS, _base, _samples, _spoken, _trained


def test_each_part_s_share_of_the_direction_adds_up_to_one_by_group():
    """By group and over all: each part's norm, its share ⟨g_part, g⟩ / |g|² (the shares add up to 1) and the cosine of
    each pair; a part that reaches none of a group's parameters has norm 0 and its cosines None."""
    parameters = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))]
    parts = {"surrogate": [torch.tensor([3.0, 0.0]), None], "pull": [torch.tensor([0.0, 4.0]), None],
             "data": [torch.tensor([3.0, 0.0]), torch.tensor([2.0])]}
    got = gradients.measured(parts, {"prior": range(0, 1), "traffic": range(1, 2)}, parameters)
    prior = got["prior"]
    assert prior["norm"] == {"surrogate": 3.0, "pull": 4.0, "data": 3.0}
    assert prior["whole_norm"] == pytest.approx(52 ** 0.5)
    assert prior["share"] == pytest.approx({"surrogate": 18 / 52, "pull": 16 / 52, "data": 18 / 52})
    assert prior["cosine"] == pytest.approx({"surrogate|pull": 0.0, "surrogate|data": 1.0, "pull|data": 0.0})
    traffic = got["traffic"]
    assert traffic["share"] == {"surrogate": 0.0, "pull": 0.0, "data": 1.0}
    assert traffic["cosine"] == {"surrogate|pull": None, "surrogate|data": None, "pull|data": None}
    assert sum(got["all"]["share"].values()) == pytest.approx(1.0)


def _close(a, b):
    """Nested dicts of numbers equal to float rounding, a None only where the other is None."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    return (a is None) == (b is None) and (a is None or a == pytest.approx(b, rel=1e-6, abs=1e-9))


def test_the_updates_consistency_reads_one_way_as_one_and_independent_ways_as_zero():
    """`Consistency`: four updates pointing one way read pairwise and halves' cosines 1 and coherence 1; four orthogonal
    ones read 0, 0 and 1/√4; by span (a group's slice of the flat gradient) apart from the whole; a part that reaches no
    parameter of a span reads None there."""
    spans = {"prior": slice(0, 4), "traffic": slice(4, 6), "all": slice(0, 6)}
    eye = torch.eye(6, dtype=torch.float64)
    one, apart = gradients.Consistency(spans), gradients.Consistency(spans)
    for i in range(4):
        same = torch.tensor([1.0, 2.0, 0.0, 0.0, 3.0, 0.0], dtype=torch.float64) * (i + 1)
        one.add({"surrogate": same, "data": same, "whole": same})
        apart.add({"surrogate": eye[i], "data": eye[i], "whole": eye[i]})
    aligned, orthogonal = one.result(), apart.result()
    for name in ("prior", "traffic", "all"):
        assert aligned[name]["surrogate"] == {"updates": 4, "pairwise_cosine": pytest.approx(1.0),
                                              "halves_cosine": pytest.approx(1.0), "coherence": pytest.approx(1.0)}
    assert orthogonal["prior"]["data"] == {"updates": 4, "pairwise_cosine": 0.0, "halves_cosine": 0.0,
                                           "coherence": pytest.approx(0.5)}
    assert orthogonal["traffic"]["whole"] == {"updates": 0, "pairwise_cosine": None, "halves_cosine": None,
                                              "coherence": None}            # e0..e3 reach no traffic parameter
    parameters = [torch.nn.Parameter(torch.zeros(2, 2)), torch.nn.Parameter(torch.zeros(2))]
    assert gradients.spans_of({"prior": range(0, 1), "traffic": range(1, 2)}, parameters) == {
        "prior": slice(0, 4), "traffic": slice(4, 6), "all": slice(0, 6)}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A campaign of one round done and the next left half done (its groups on disk), the runner's world: the stage's
    settings, its model and context, its pass's updates (three, synthetic spoken samples) and the checks stubbed."""
    base = _base()
    model = _trained(base)
    rows, permitted, _, traffic = _spoken(model)
    rng = np.random.default_rng(9)
    data = collate([prior_sentence(rng, WORDS, candidates=2, rows=LENGTH, first_step=FIRST) for _ in range(4)], CPU)
    advantages = [torch.as_tensor(np.random.default_rng(k).normal(size=rows.asked.shape), dtype=torch.float32)
                  for k in range(3)]
    spoken = [_samples(rows, permitted, traffic, a) for a in advantages]
    updates = [([one.select([0]), one.select([1, 2])], data) for one in spoken]
    campaign = tmp_path / "campaign"
    (campaign / "round_0").mkdir(parents=True)
    (campaign / "round_0" / "checkpoint.pt").write_bytes(b"")
    (campaign / "round_1").mkdir()
    (campaign / "round_1" / "groups_0.pt").write_bytes(b"")
    (campaign / "campaign.json").write_text(json.dumps({"schema": "stub", "inputs": {
        "smoke": False, "prior": "p", "instructions": "i", "executor": "e", "windows": "w", "procedure_root": "c"}}))
    settings = SimpleNamespace(seed=2035, prior_lr=1e-5, traffic_lr=1e-4, data_weight=0.5)
    asked, built = [], []
    stage = SimpleNamespace(part_width=5)                                  # stage D's: a token part
    monkeypatch.setattr(gradients.window_behaviour, "opened_campaign", lambda record: (settings, stage))
    monkeypatch.setattr(gradients.window_behaviour, "model_of", lambda context, s, stage, c, r: (asked.append(r),
                                                                                              model)[1])
    monkeypatch.setattr(post_train, "inputs_here", lambda inputs: inputs)
    monkeypatch.setattr(post_train, "open_context", lambda *a, **k: SimpleNamespace(base=base, data=[], interval_s=4.0))
    monkeypatch.setattr(post_train, "segment_rows_of", lambda s, interval_s: 7)

    def update_pairs(directory, data_, s, rng_, device, **k):
        built.append({"directory": directory.name, "words": list(rng_.bit_generator.seed_seq.entropy), **k})
        return iter(updates)

    monkeypatch.setattr(post_train, "update_pairs", update_pairs)
    monkeypatch.setattr(gradients, "git_state", lambda: {"head": "x", "dirty": False})
    monkeypatch.setattr(gradients, "require_conforming_closed_loop",
                        lambda *a: (None, {"checks": {"stub": True}}, None))
    monkeypatch.setattr(gradients, "checked_edges", lambda path: None)
    return campaign, model, base, updates, asked, built


def test_the_read_takes_every_kth_update_at_the_model_that_spoke_the_round(world, tmp_path):
    """Round 1 read at round 0's model, every 2nd update (0 and 2 of 3), each update's parts the ones
    `update_gradients` gives at that model with the round's dropout and the campaign's data weight, the means and the
    learning rates written."""
    campaign, model, base, updates, asked, built = world
    out = tmp_path / "g.json"
    assert gradients.main(["--campaign", str(campaign), "--round", "1", "--out", str(out), "--every", "2"]) == 0
    held = json.loads(out.read_text())
    assert asked == [0] and held["model_round"] == 0 and held["schema"] == gradients.PASS_GRADIENTS_SCHEMA
    assert built == [{"directory": "round_1", "words": [2035, 1, 1], "part_width": 5, "segment_rows": None}]
    assert [u["update"] for u in held["updates"]] == [0, 2] and held["updates_read"] == 2 and held["updates_seen"] == 3
    assert held["learning_rates"] == {"prior": 1e-5, "traffic": 1e-4} and held["checks"] == {"stub": True}
    named = parameter_groups(model, 1e-5, 1e-4)
    parameters = [p for group in named for p in group["params"]]
    groups = {"prior": range(0, len(named[0]["params"])),
              "traffic": range(len(named[0]["params"]), len(parameters))}
    torch.manual_seed(post_train.pass_seed(2035, 1))
    pieces, data = updates[0]
    expected = gradients.measured(update_gradients(model, PassStart(model), base, pieces, data, parameters,
                                                   data_weight=0.5), groups, parameters)
    assert _close(held["updates"][0]["parts"], json.loads(json.dumps(expected)))
    for update in held["updates"]:
        for name in ("prior", "traffic", "all"):
            assert sum(update["parts"][name]["share"].values()) == pytest.approx(1.0)
    assert set(held["means"]) == {"prior", "traffic", "all"}
    for name in ("prior", "traffic", "all"):                # two updates read: the halves' cosine is their cosine
        surrogate = held["consistency"][name]["surrogate"]
        assert surrogate["updates"] == 2 and surrogate["halves_cosine"] == pytest.approx(surrogate["pairwise_cosine"])
    assert held["consistency"]["traffic"]["data"]["pairwise_cosine"] is None       # the data term has no traffic
    for name in ("prior", "traffic", "all"):        # each part's own gradients: two updates' coherence from their norms
        for part in ("surrogate", "data", "whole"):
            c = held["consistency"][name][part]
            if c["updates"] < 2:
                continue
            n0, n1 = (u["parts"][name]["whole_norm"] if part == "whole" else u["parts"][name]["norm"][part]
                      for u in held["updates"])
            assert c["coherence"] == pytest.approx((n0 ** 2 + n1 ** 2 + 2 * n0 * n1 * c["halves_cosine"]) ** 0.5
                                                   / (n0 + n1))
    assert held["means"]["all"]["share"]["data"] == pytest.approx(
        np.mean([u["parts"]["all"]["share"]["data"] for u in held["updates"]]))


def test_a_read_is_refused_by_name_before_anything_opens(world, tmp_path, capsys, monkeypatch):
    """A round with no groups on disk (a done round's are deleted after its pass), a round past the done ones, an
    output that exists, and a tree with changes are refused before the model is read."""
    campaign, _, _, _, asked, _ = world
    (campaign / "round_3").mkdir()
    (campaign / "round_3" / "groups_0.pt").write_bytes(b"")
    taken = tmp_path / "taken.json"
    taken.write_text("{}")
    for argv, says in ((["--round", "0", "--out", str(tmp_path / "a.json")], "holds no groups"),
                       (["--round", "3", "--out", str(tmp_path / "a.json")], "has done 1 rounds"),
                       (["--round", "1", "--out", str(taken)], "exists")):
        with pytest.raises(SystemExit):
            gradients.main(["--campaign", str(campaign), *argv])
        assert says in capsys.readouterr().err
    monkeypatch.setattr(gradients, "git_state", lambda: {"head": "x", "dirty": True})
    with pytest.raises(SystemExit):
        gradients.main(["--campaign", str(campaign), "--round", "1", "--out", str(tmp_path / "a.json")])
    assert "needs a clean tree" in capsys.readouterr().err and asked == []


def test_a_stage_c_campaign_s_updates_count_its_rows_as_its_pass_does(world, tmp_path, monkeypatch):
    """Stage C's pass counts a group's rows up to the next branch point where its settings say so (`segment_rows_of`,
    as `STAGE_C.train`), and its stage has no token part."""
    campaign, _, _, _, _, built = world
    settings = SimpleNamespace(seed=11, prior_lr=1e-5, traffic_lr=1e-4)
    monkeypatch.setattr(gradients.window_behaviour, "opened_campaign", lambda record: (settings, post_train.STAGE_C))
    assert gradients.main(["--campaign", str(campaign), "--round", "1", "--out", str(tmp_path / "c.json")]) == 0
    assert built == [{"directory": "round_1", "words": [11, 1, 1], "part_width": 0, "segment_rows": 7}]
