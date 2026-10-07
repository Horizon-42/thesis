"""Stage C, C17 (P49): a campaign that trains on the landed sentences — on A26's one-flight synthetic artefact (the
fixture of `test_post_window_loop`); every write root under tmp."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.experiments import post_train
from ts_transformer.experiments.post_train import (
    KINDS, LANDED_SENTENCES, STAGE_C_LANDED, Speakers, best_landed, done_rounds, landed_pairs, open_campaign,
    run_campaign, speak_round, start_model,
)
from ts_transformer.post.branches import Landed, first_numbers, landed_numbers, landed_samples
from ts_transformer.post.loss import KL_WEIGHT, DATA_WEIGHT, PassStart, data_term, landed_step, _pull_words
from ts_transformer.post.reward import LANDED
from ts_transformer.post.scene import REAL
from ts_transformer.prior.batch import collate
from ts_transformer.prior.train import masked_log_probability
from ts_transformer.tests.test_post_branches import _ahead, _round
from ts_transformer.tests.test_post_train import _context, _digest, _inputs, _same, _settings
from ts_transformer.tests.test_post_window_loop import CPU, setup  # noqa: F401


def _landed_settings(**changed):
    return _settings(**{"method": LANDED_SENTENCES, "per_kind": {**dict.fromkeys(KINDS, 0), REAL: 1}, "continuations": 2,
                        "update_groups": 1, **changed})


def test_the_kept_sentence_is_the_best_landed_one_a_tie_the_lowest_draw():
    """P49: of a window's draws its landed sentence of the highest reward (a tie: the lowest draw); a window with no
    landing of a reward above 0 keeps none (a landing the reward scores 0, D105, is not learned)."""
    def ends(*pairs):
        return [SimpleNamespace(outcome=o, reward=r) for o, r in pairs]

    lost = ("lost_separation", 0.0)
    drawn = [(ends(lost, lost, (LANDED, 1.0)), [("d0w0",), ("d0w1",), ("d0w2",)]),
             (ends((LANDED, 0.9), lost, ("timeout", 0.0)), [("d1w0",), ("d1w1",), ("d1w2",)]),
             (ends((LANDED, 0.9), lost, (LANDED, 1.0)), [("d2w0",), ("d2w1",), ("d2w2",)]),
             (ends((LANDED, 1.0), (LANDED, 0.0), (LANDED, 0.9)), [("d3w0",), ("d3w1",), ("d3w2",)])]
    best = {b: (s.rows, s.reward) for b, s in
            best_landed([(e, [(m[0], None, []) for m in made]) for e, made in drawn]).items()}
    assert best == {0: ("d3w0", 1.0), 2: ("d0w2", 1.0)}      # window 1: only a landing the reward scores 0, nothing


def test_draw_0_is_the_window_s_first_sentence_and_each_draw_has_its_own_numbers():
    assert landed_numbers(1337, 3, 5, 0).random() == first_numbers(1337, 3, 5).random()
    assert len({landed_numbers(1337, 3, 5, d).random() for d in range(4)}) == 4


def test_the_loss_of_a_kept_sentence_is_its_words_log_likelihood(setup):  # noqa: F811
    """`landed_step`: the negative log-likelihood of the kept words a counted row, the pull to the base and the data
    term — its value and its gradient those of the whole expression computed at once."""
    s = setup
    context = _context(s)
    settings = _landed_settings()
    model, _ = start_model(context, settings)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)                                    # not the base: the pull has a gradient
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    kept = [Landed(0, group.first), Landed(1, group.continuations[0])]
    data = collate(context.data[:1], CPU)
    torch.manual_seed(1)
    model.zero_grad(set_to_none=True)
    parts = landed_step(model, PassStart(model), context.base, [landed_samples([k], CPU) for k in kept], data)
    stepped = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    whole = landed_samples(kept, CPU)
    torch.manual_seed(1)
    model.zero_grad(set_to_none=True)
    model.eval()
    log_p = masked_log_probability(model, whole.rows, whole.permitted, whole.traffic)
    with torch.no_grad():
        base_log_p = masked_log_probability(context.base, whole.rows, whole.permitted)
    counted = whole.counted[..., None].expand_as(log_p)
    rows = whole.counted.sum()
    nll = -log_p[counted].sum() / rows
    kl = _pull_words(log_p, base_log_p)[counted].sum() / rows
    teacher = data_term(model, data)
    (nll + KL_WEIGHT * kl + DATA_WEIGHT * teacher).backward()
    assert float(parts.surrogate) == pytest.approx(float(nll), rel=1e-5) and parts.clipped == 0
    assert float(parts.kl) == pytest.approx(float(kl), rel=1e-4, abs=1e-8)
    assert parts.words == int(counted.sum())
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.allclose(stepped[n], p.grad, rtol=1e-4, atol=1e-7), n


def test_a_round_speaks_each_window_n_times_and_writes_the_kept_sentences(setup, tmp_path, monkeypatch):  # noqa: F811
    """`speak_landed_batch` on the short window (lost at its first row: nothing kept) through the round's speaking, here
    and through two workers: the same kept files, byte for byte, and the same record; draw 0's ends are the record's."""
    s = setup
    context = _context(s)
    settings = _landed_settings(per_kind=dict.fromkeys(KINDS, 1), continuations=3)
    model, _ = start_model(context, settings)
    windows = [_ahead(s["windows"][0])] * 2
    monkeypatch.setattr(post_train, "draw_round", lambda context, per_kind, rng: (windows, {}))
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    record = speak_round(model, context, windows, settings, 0, here, stage=STAGE_C_LANDED)
    assert record["batches"] == 2 and record["windows"] == 2 and record["kept"] == 0 and record["landings"] == 0
    assert record["outcomes"] == {"lost_separation": 2}
    speakers = Speakers(context, settings, 2, CPU, stage=STAGE_C_LANDED)
    try:
        assert speak_round(model, context, windows, settings, 0, there, speakers, stage=STAGE_C_LANDED) == record
    finally:
        speakers.close()
    assert sorted(p.name for p in there.iterdir()) == ["kept_0.pt", "kept_1.pt"]
    assert all((here / n).read_bytes() == (there / n).read_bytes() for n in ("kept_0.pt", "kept_1.pt"))


def test_a_landed_campaign_resumed_is_the_campaign_run_through(setup, tmp_path, monkeypatch):  # noqa: F811
    """The speaking replaced (each round keeps the same two sentences, so every round updates the model), a landed
    campaign's round broken off is moved aside and run again from the last checkpoint: the weights after the last round
    are those of the campaign run through, bit for bit; its pass learns the kept sentences (``nll``, no clipped share);
    a resume with the other method is refused; it starts from a round of another campaign (D162) as stage C's does."""
    from ts_transformer.io_utils import file_sha256

    s = setup
    context = _context(s)
    source_settings = _settings(rounds=1, per_kind={**dict.fromkeys(KINDS, 0), REAL: 1}, update_groups=1)
    model, _ = start_model(context, source_settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    kept = [Landed(0, replace(group.first, reward=1.0)), Landed(0, replace(group.continuations[0], reward=1.0))]

    def speak(model, context, windows, places, settings, round_, directory, k):
        torch.save(kept, directory / f"kept_{k}.pt")
        return {"kept": len(kept), "landings": len(kept), "reward_sum": 1.0, "faulty_steps": 0,
                "losses_reading_fault": 0, "outcomes": {LANDED: 1}, "weights": _digest(model)}

    monkeypatch.setattr(post_train, "speak_landed_batch", speak)
    monkeypatch.setattr(post_train, "selection_readout", lambda model, *a, **k: {"KXXX": {"reward_mean": _digest(model)}})
    source = tmp_path / "source"
    open_campaign(source, _inputs(source_settings, tmp_path), {"head": "x", "dirty": False}, {})
    run_campaign(source, source_settings, context)
    start = {"campaign": str(source), "round": 0, "checkpoint_sha256": file_sha256(source / "round_0" / "checkpoint.pt")}
    settings = _landed_settings(rounds=2, start=start, seed=2024)
    through, broken = tmp_path / "through", tmp_path / "broken"
    open_campaign(through, _inputs(settings, tmp_path), {"head": "x", "dirty": False}, {})
    run_campaign(through, settings, context, stage=STAGE_C_LANDED)
    record = json.loads((through / "round_1" / "round.json").read_text())
    assert record["pass"]["updates"] == 2 and "nll" in record["pass"] and "clipped_share" not in record["pass"]
    assert record["speaking"]["kept"] == 2 and not list((through / "round_1").glob("kept_*.pt"))
    held = torch.load(source / "round_0" / "checkpoint.pt", weights_only=False)
    first = json.loads((through / "round_0" / "round.json").read_text())["speaking"]["weights"]
    probe = start_model(context, source_settings)[0]
    probe.load_state_dict(held["model"])
    assert first == _digest(probe)                                     # round 0 spoke with the start's weights
    state = torch.load(through / "round_1" / "checkpoint.pt", weights_only=False)
    open_campaign(broken, _inputs(settings, tmp_path), {"head": "x", "dirty": False}, {})
    passes = []
    real = post_train.landed_train_pass

    def killed_in_round_1(*a, **k):
        passes.append(1)
        if len(passes) == 2:
            raise RuntimeError("killed")
        return real(*a, **k)

    monkeypatch.setattr(post_train, "landed_train_pass", killed_in_round_1)
    with pytest.raises(RuntimeError, match="killed"):
        run_campaign(broken, settings, context, stage=STAGE_C_LANDED)
    monkeypatch.setattr(post_train, "landed_train_pass", real)
    assert done_rounds(broken) == 1
    reopened = open_campaign(broken, _inputs(settings, tmp_path), {"head": "y", "dirty": False}, {})
    assert len(reopened["aborted"]) == 1
    run_campaign(broken, settings, context, stage=STAGE_C_LANDED)
    assert _same(torch.load(broken / "round_1" / "checkpoint.pt", weights_only=False)["model"], state["model"])
    with pytest.raises(SystemExit, match="other inputs"):
        open_campaign(broken, _inputs(replace(settings, method=post_train.BRANCH), tmp_path),
                      {"head": "y", "dirty": False}, {})


def test_the_landed_pass_reads_every_kept_sentence_a_piece_each(setup, tmp_path):  # noqa: F811
    s = setup
    context = _context(s)
    settings = _landed_settings(update_groups=2, data_sentences=1)
    model, _ = start_model(context, settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    torch.save([Landed(p, group.first) for p in range(3)], tmp_path / "kept_0.pt")
    torch.save([Landed(9, group.first)], tmp_path / "kept_1.pt")
    updates = list(landed_pairs(tmp_path, context.data, settings, np.random.default_rng(0), CPU))
    assert [len(pieces) for pieces, _ in updates] == [2, 1, 1]           # a file's last update may hold fewer


def test_a_method_is_named_and_one_of_the_two():
    with pytest.raises(ValueError, match="none of"):
        _settings(method="other")


def test_the_readout_reads_with_its_own_seed_so_a_campaign_of_another_seed_reads_the_same_windows(setup, monkeypatch):  # noqa: F811
    """The user, 2026-10-07: the selection readout's windows and numbers come from `Settings.select_seed`, so a campaign
    of seed 2024 (D162) reads C10's select windows with C10's numbers, and another readout seed reads other windows (on
    a pool of 500 stand-in windows: `selection_windows` only picks among them)."""
    s = setup
    context = _context(s)
    c10 = _settings(seed=1337, select_seed=1337, select_per_airport=200)
    landed = _landed_settings(seed=2024, select_seed=1337, select_per_airport=200)
    with monkeypatch.context() as patch:
        patch.setattr(post_train, "readout_pool", lambda context, split: {"KXXX": list(range(500))})
        picked = [post_train.selection_windows(context, c) for c in (c10, landed, replace(landed, select_seed=99))]
    assert picked[0] == picked[1] and picked[0] != picked[2] and len(picked[0]) == 200
    asked = []
    numbers = post_train.readout_numbers
    monkeypatch.setattr(post_train, "readout_numbers", lambda seed, place, draw=0: (asked.append(seed),
                                                                                     numbers(seed, place, draw))[1])
    model, _ = start_model(context, landed)
    windows = post_train.selection_windows(context, landed)
    post_train.read_batch(model, context, windows, [0], landed, "select")
    assert asked == [1337]


def test_a_kept_sentence_goes_from_the_speaking_through_the_workers_to_the_pass(setup, tmp_path, monkeypatch):  # noqa: F811
    """A real sentence kept: each draw's ends read as landed with reward 1 (`best_landed` wrapped, before the workers
    fork, so they keep the same), the round's speaking here and through two workers writes the same kept files, byte
    for byte, and the same record; the landed pass then makes its updates from them."""
    s = setup
    context = _context(s)
    settings = _landed_settings(per_kind=dict.fromkeys(KINDS, 1), continuations=2, update_groups=1, data_sentences=1)
    model, optimizer = start_model(context, settings)
    windows = [_ahead(s["windows"][0])] * 2
    monkeypatch.setattr(post_train, "draw_round", lambda context, per_kind, rng: (windows, {}))
    real = post_train.best_landed
    monkeypatch.setattr(post_train, "best_landed", lambda drawn: real(
        [([SimpleNamespace(outcome=LANDED, reward=1.0) for _ in ends], made) for ends, made in drawn]))
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    record = speak_round(model, context, windows, settings, 0, here, stage=STAGE_C_LANDED)
    assert record["kept"] == 2
    speakers = Speakers(context, settings, 2, CPU, stage=STAGE_C_LANDED)
    try:
        assert speak_round(model, context, windows, settings, 0, there, speakers, stage=STAGE_C_LANDED) == record
    finally:
        speakers.close()
    assert all((here / n).read_bytes() == (there / n).read_bytes() for n in ("kept_0.pt", "kept_1.pt"))
    kept = [k for n in ("kept_0.pt", "kept_1.pt") for k in torch.load(here / n, weights_only=False)]
    assert [k.window for k in kept] == [0, 1] and all(k.sentence.reward == 1.0 for k in kept)
    passed = post_train.landed_train_pass(model, context, optimizer, here, settings, np.random.default_rng(0))
    assert passed["updates"] == 2 and passed["words"] > 0 and passed["nll"] > 0.0


def test_the_runner_hands_the_method_s_stage_to_the_workers_and_the_campaign(tmp_path, monkeypatch):
    """`post_train --method landed` runs `STAGE_C_LANDED`: its speaking workers and its rounds (a run recording
    "landed" never trains with branch groups)."""
    handed = {}

    class Workers:
        workers = 2

        def __init__(self, context, settings, workers, device, *, stage):
            handed["speakers"] = stage
            handed["speak_device"] = device

        def close(self):
            pass

    monkeypatch.setattr(post_train, "git_state", lambda: {"head": "x", "dirty": False})
    monkeypatch.setattr(post_train, "require_conforming_closed_loop", lambda *a: (None, {"checks": {}}, None))
    monkeypatch.setattr(post_train, "checked_edges", lambda path: None)
    context = SimpleNamespace(base=torch.nn.Linear(1, 1), device=CPU)
    monkeypatch.setattr(post_train, "open_context", lambda *a, **k: context)
    monkeypatch.setattr(post_train, "replace", lambda c, **k: c)
    monkeypatch.setattr(post_train, "Speakers", Workers)
    monkeypatch.setattr(post_train, "require_workers_fit", lambda *a: None)
    monkeypatch.setattr(post_train, "run_campaign", lambda out, settings, context, speakers, *, stage:
                        handed.update(campaign=stage, method=settings.method))
    argv = ["--prior", str(tmp_path / "p"), "--instructions", str(tmp_path / "i"), "--executor", str(tmp_path / "e"),
            "--windows", str(tmp_path / "w"), "--out", str(tmp_path / "campaign"), "--rounds", "1",
            "--batch-windows", "1", "--seed", "2024", "--prior-lr", "1e-4", "--traffic-lr", "1e-3",
            "--weight-decay", "0", "--update-groups", "1", "--data-sentences", "1", "--select-per-airport", "1",
            "--traffic-hidden", "16", "--traffic-heads", "4", "--method", "landed", "--select-seed", "1337",
            "--speak-workers", "2", "--device", "cpu"]
    argv += [x for kind in KINDS for x in (f"--windows-{kind.lower()}", "1")]
    assert post_train.main(argv) == 0
    assert handed == {"speakers": STAGE_C_LANDED, "campaign": STAGE_C_LANDED, "method": LANDED_SENTENCES,
                      "speak_device": CPU}
    # the workers' device apart from the campaign's: `--speak-device meta` reaches `Speakers`, the campaign stays on the
    # CPU (a device that is neither, so that the test tells the two apart)
    handed.clear()
    (tmp_path / "campaign").rename(tmp_path / "first")
    assert post_train.main(argv + ["--speak-device", "meta"]) == 0 and handed["speak_device"] == torch.device("meta")
    with pytest.raises(SystemExit):                                     # workers' device without workers
        post_train.main([a for a in argv if a not in ("--speak-workers", "2")] + ["--speak-device", "cpu"])
    with pytest.raises(SystemExit):                                     # workers on the GPU, the campaign on the CPU
        post_train.main(argv + ["--speak-device", "cuda"])


def test_workers_on_the_cpu_beside_a_pass_on_the_gpu_hold_nothing_of_the_gpu():
    """O15 with ``--speak-device cpu`` and a pass on the GPU: the workers' measure has no GPU part, and the pass needs
    only its own growth (each worker holds 0 of the GPU)."""
    measured = {"host": {"peak": 1 << 30, "now": 1 << 29}, "gpu": None,
                "held": {"reader_model": 1 << 20, "series": 1 << 20}}
    passed = {"peak": 3 << 30, "now": 1 << 30}
    assert post_train.workers_fit(16, measured, passed, {"host": 64 << 30, "gpu": 2 << 30}) == []    # exactly the pass's
    short = post_train.workers_fit(16, measured, passed, {"host": 64 << 30, "gpu": (2 << 30) - 1})
    assert len(short) == 1 and short[0].startswith("gpu: the pass beside 16 workers")
