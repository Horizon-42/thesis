"""Training with a value function (post-training §2 item 10, §8 C22; D171): V, what it reads beyond the model, the
advantages."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.post.branches import Sentence
from ts_transformer.post.edges import EDGE_FEATURES, TOKEN_FEATURES
from ts_transformer.post.value import (
    GAE_LAMBDA, LANDING_SCALE_S, VALUE_FEATURES, Value, ValueBatch, ValueSample, advantages, value_batch, value_loss,
    value_part,
)
from ts_transformer.prior.train import masked_log_probability
from ts_transformer.tests.test_post_branches import _ahead
from ts_transformer.tests.test_post_window_loop import CPU, _window_loop, _with_module, setup  # noqa: F401


def test_the_advantages_on_hand_computed_values():
    """λ = 1: the reward less v_t; λ = 0: δ_t (v_{t+1} − v_t, the reward at the last counted row); 0 at rows not
    counted; the target is A_t + v_t."""
    values = torch.tensor([[9.0, 0.2, 0.5, 0.4, 9.0]])
    counted = torch.tensor([[False, True, True, True, False]])
    reward = torch.tensor([1.0])
    a1, r1 = advantages(values, counted, reward, lam=1.0)
    assert torch.allclose(a1, torch.tensor([[0.0, 0.8, 0.5, 0.6, 0.0]]))
    assert torch.allclose(r1, torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0]]))
    a0, _ = advantages(values, counted, reward, lam=0.0)
    assert torch.allclose(a0, torch.tensor([[0.0, 0.3, -0.1, 0.6, 0.0]]))
    a, _ = advantages(values, counted, reward)                                      # λ = 0.95
    assert float(a[0, 1]) == pytest.approx(0.3 + GAE_LAMBDA * -0.1 + GAE_LAMBDA ** 2 * 0.6)
    with pytest.raises(ValueError, match="one run"):
        advantages(values, torch.tensor([[True, False, True, False, False]]), reward)


def _value_samples(s, model, seed=1):
    """The fixture's window with an aircraft inserted 8 s ahead (`test_post_branches._ahead`) flown by ``model`` three
    times (its own numbers each), each sentence kept as a `ValueSample`, V's part random (so that it is read)."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(3):
        flown = _window_loop(s, model, [_ahead(s["windows"][0])])
        (end,) = flown.run([np.random.default_rng([seed, k])])
        ((rows, permitted, tokens),) = flown.samples("train")
        part = [rng.normal(size=(len(t), len(VALUE_FEATURES))).astype(np.float32) for t in tokens]
        sentence = Sentence(rows, permitted, tokens, end.reward)
        out.append(ValueSample(k, sentence, part, rng.uniform(0.2, 1.0, size=len(tokens)).astype(np.float32)))
    return out


def test_v_reads_its_part_and_the_time_left_and_the_model_never_does(setup):
    """D90's exception (D171): the model's samples hold the tokens only, so a change of V's features changes V's values
    and leaves the model's log-probabilities the same; V shares no weight with the model; V refuses training mode."""
    model = _with_module(setup["base"])
    kept = _value_samples(setup, model)
    value = Value.of(model)
    assert not {p.data_ptr() for p in value.parameters()} & {p.data_ptr() for p in model.parameters()}
    torch.manual_seed(0)
    with torch.no_grad():
        for p in value.prior.parameters():                  # V's part read: its projection away from zero
            p.add_(torch.randn_like(p) * 0.05)
    samples, traffic, left, _ = value_batch(kept, CPU)
    assert samples.traffic.tokens.shape[-1] == len(TOKEN_FEATURES) and samples.traffic.part is None
    changed = [ValueSample(k.window, k.sentence, [p + 1.0 for p in k.part], k.time_left + 0.5) for k in kept]
    samples2, traffic2, left2, _ = value_batch(changed, CPU)
    with torch.no_grad():
        assert torch.equal(masked_log_probability(model, samples.rows, samples.permitted, samples.traffic),
                           masked_log_probability(model, samples2.rows, samples2.permitted, samples2.traffic))
        v1 = value(samples.rows, traffic, left)
        v2 = value(samples2.rows, traffic2, left2)
    assert not torch.allclose(v1, v2)
    with pytest.raises(ValueError, match="dropout off"):
        value.train()(samples.rows, traffic, left)


def test_v_s_loss_is_the_mean_of_its_squared_error_over_the_counted_rows(setup):
    model = _with_module(setup["base"])
    kept = _value_samples(setup, model)
    value = Value.of(model)
    samples, traffic, left, reward = value_batch(kept, CPU)
    with torch.no_grad():
        v = value(samples.rows, traffic, left)
    advantage, target = advantages(v, samples.counted, reward)
    batch = ValueBatch(samples, traffic, left, target)
    rows = samples.counted.sum().to(torch.float32)
    with torch.no_grad():
        loss = value_loss(value, batch, rows)
    assert float(loss) == pytest.approx(float((advantage[samples.counted] ** 2).mean()), rel=1e-5)    # (v − R)² = A²


def test_v_s_part_reads_each_recorded_aircraft_s_future_against_the_present_state(setup):
    """V's token part of a window's recorded aircraft: at +30, 60 and 120 s the edge features of where each will be,
    against the commanded aircraft's state now, with the flag that it is in the air then; the time to its landing."""
    from ts_transformer.post.edges import tokens
    from ts_transformer.post.runways import airport_separation

    s = setup
    window = _ahead(s["windows"][0])
    step = next(t for t in range(200) if len(window.others_at(t)))
    others = window.others_at(step)
    own = others.__class__.of([(window.commanded.key, *window.commanded.at_step(window.step_s(step),
                                                                            window.scene.interval_s)[1:])])
    separation = airport_separation(window.scene.geometry)
    part = value_part(window, step, own, others, separation, 2.0)
    assert part.shape == (len(others), len(VALUE_FEATURES))
    width = len(EDGE_FEATURES) + 1
    time_s = window.step_s(step)
    for i, key in enumerate(others.keys):
        record = window.scene.flight(key)
        assert part[i, -1] == pytest.approx(max(record.landing_s - time_s, 0.0) / LANDING_SCALE_S)
        for k, horizon in enumerate((30.0, 60.0, 120.0)):
            later = time_s + horizon
            if record.start_s <= later <= record.end_s:
                then = others.__class__.of([record.at_step(later, window.scene.interval_s)])
                expected = tokens(own, then, window.scene.geometry, separation, 2.0)[0, :len(EDGE_FEATURES)]
                assert np.array_equal(part[i, k * width:k * width + len(EDGE_FEATURES)], expected)
                assert part[i, k * width + len(EDGE_FEATURES)] == 1.0
            else:
                assert not part[i, k * width:(k + 1) * width].any()


def _value_settings(**changed):
    from ts_transformer.experiments.post_train import KINDS, VALUE
    from ts_transformer.post.scene import REAL
    from ts_transformer.tests.test_post_train import _settings

    return _settings(**{"method": VALUE, "per_kind": {**dict.fromkeys(KINDS, 0), REAL: 1}, "continuations": 1,
                        "update_groups": 1, "value_lr": 1e-3, "value_warmup": 1, **changed})


def test_value_lr_and_value_warmup_go_with_the_value_method_only():
    from ts_transformer.experiments import post_train
    from ts_transformer.tests.test_post_train import _settings

    _value_settings()
    for wrong in (dict(value_lr=None), dict(value_warmup=None), dict(continuations=2), dict(value_lr=0.0),
                  dict(segment_only=True), dict(branch_every_s=60.0)):
        with pytest.raises(ValueError):
            _value_settings(**wrong)
    for method in (post_train.BRANCH, post_train.LANDED_SENTENCES):
        with pytest.raises(ValueError, match="value_lr and value_warmup are required with method 'value'"):
            _settings(method=method, value_lr=1e-3, value_warmup=1)


def test_a_value_campaign_warms_up_v_alone_keeps_v_beside_the_checkpoint_and_resumes_as_one_run(setup, tmp_path,
                                                                                                  monkeypatch):
    """D171 on the fixture's short round (one window, lost at its first row): round 0 warms V up — the model and its
    optimizer do not move — and round 1 trains both; ``value.pt`` is written before each checkpoint; a run killed in
    round 1's pass and resumed gives round 1 as the run without a stop (the model and V); no readout opens
    ``value.pt`` (round 1's selection readout again, the file removed); a start from a value round reads no V."""
    from ts_transformer.experiments import post_train
    from ts_transformer.experiments.post_train import (
        STAGE_C_VALUE, campaign_start, done_rounds, open_campaign, round_model, run_campaign, selection_readout,
        selection_windows, start_model,
    )
    from ts_transformer.io_utils import file_sha256
    from ts_transformer.tests.test_post_train import _context, _inputs, _same, short_round

    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    settings = _value_settings(rounds=2, epochs=2, clip_norm=1.0)            # V's clip and two passes too
    saved = []
    real_save = torch.save
    monkeypatch.setattr(post_train.torch, "save", lambda obj, path, *a, **k: (saved.append(Path(path).name),
                                                                              real_save(obj, path, *a, **k))[1])
    through = tmp_path / "through"
    open_campaign(through, _inputs(settings, tmp_path), {"head": "x", "dirty": False}, {})
    run_campaign(through, settings, context, stage=STAGE_C_VALUE)
    assert done_rounds(through) == 2
    for r in range(2):                                                     # V's file, then the checkpoint
        names = [n for n in saved if n in ("value.pt", "checkpoint.pt")]
        assert names[2 * r:2 * r + 2] == ["value.pt", "checkpoint.pt"]
    start, _ = start_model(context, settings)
    held = torch.load(through / "round_0" / "checkpoint.pt", weights_only=False)
    assert _same(held["model"], start.state_dict()) and not held["optimizer"]["state"]       # warm-up: not moved
    record = [json.loads((through / f"round_{r}" / "round.json").read_text()) for r in range(2)]
    assert record[0]["pass"]["warmup"] is True and "loss" not in record[0]["pass"]
    assert record[0]["warmup_readout"] == {"against": None, "same": None}                 # the base: nothing before
    assert record[1]["pass"]["warmup"] is False and record[1]["pass"]["updates"] == 2           # two passes
    assert record[1]["pass"]["value"]["updates"] == 2 and "grad_clipped_share" in record[1]["pass"]
    assert {"loss_before", "loss_after", "explained_after", "advantage_mean", "advantage_std"} <= set(
        record[1]["pass"]["value"])
    assert not list((through / "round_1").glob("sentences_*.pt")) and record[1]["speaking"]["windows"] == 1
    value = [torch.load(through / f"round_{r}" / "value.pt", weights_only=False) for r in range(2)]
    assert value[0]["schema"] == "ts-post-value-v1" and not _same(value[0]["value"], value[1]["value"])
    assert value[1]["identity"] == torch.load(through / "round_1" / "checkpoint.pt", weights_only=False)["identity"]

    broken = tmp_path / "broken"
    open_campaign(broken, _inputs(settings, tmp_path), {"head": "x", "dirty": False}, {})
    real = post_train.value_train_pass
    calls = []

    def killed_in_round_1(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("killed")
        return real(*a, **k)

    monkeypatch.setattr(post_train, "value_train_pass", killed_in_round_1)
    with pytest.raises(RuntimeError, match="killed"):
        run_campaign(broken, settings, context, stage=STAGE_C_VALUE)
    monkeypatch.setattr(post_train, "value_train_pass", real)
    open_campaign(broken, _inputs(settings, tmp_path), {"head": "y", "dirty": False}, {})
    run_campaign(broken, settings, context, stage=STAGE_C_VALUE)
    again = torch.load(broken / "round_1" / "checkpoint.pt", weights_only=False)
    assert _same(again["model"], torch.load(through / "round_1" / "checkpoint.pt", weights_only=False)["model"])
    assert _same(torch.load(broken / "round_1" / "value.pt", weights_only=False)["value"], value[1]["value"])

    (through / "round_1" / "value.pt").unlink()                             # no readout opens it
    model = round_model(context, settings, through, 1)
    readout = selection_readout(model, context, selection_windows(context, settings, "select"), settings)
    assert json.loads(json.dumps(readout)) == record[1]["selection_readout"]
    start_from = {"campaign": str(through), "round": 1,
                  "checkpoint_sha256": file_sha256(through / "round_1" / "checkpoint.pt")}
    started, _ = campaign_start(context, _value_settings(start=start_from, seed=2024))      # reads no V
    assert _same(started.state_dict(), again["model"])


def test_the_speaking_workers_give_each_sample_v_s_reading_as_one_process(setup, tmp_path, monkeypatch):
    """D171: a value round's batches spoken by the workers (`Speakers`) write the same samples — the sentence, V's part
    and the time left at each row — as one process; every row of a sentence has its reading."""
    from ts_transformer.experiments import post_train
    from ts_transformer.experiments.post_train import STAGE_C_VALUE, Speakers, speak_round, start_model
    from ts_transformer.tests.test_post_train import _context

    s = setup
    context = _context(s)
    settings = _value_settings(batch_windows=1)
    model, _ = start_model(context, settings)
    windows = [_ahead(s["windows"][0])] * 2
    monkeypatch.setattr(post_train, "draw_round", lambda context, per_kind, rng: (windows, {}))
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    record = speak_round(model, context, windows, settings, 0, here, stage=STAGE_C_VALUE)
    assert record["windows"] == 2 and record["outcomes"] == {"lost_separation": 2}
    speakers = Speakers(context, settings, 2, CPU, stage=STAGE_C_VALUE)
    try:
        assert speak_round(model, context, windows, settings, 0, there, speakers, stage=STAGE_C_VALUE) == record
    finally:
        speakers.close()
    names = sorted(p.name for p in there.iterdir())
    assert names == ["sentences_0.pt", "sentences_1.pt"]
    assert all((here / n).read_bytes() == (there / n).read_bytes() for n in names)
    (one,) = torch.load(here / "sentences_0.pt", weights_only=False)
    assert len(one.part) == len(one.sentence.tokens) == len(one.time_left) and (one.time_left > 0).all()
    assert any(len(p) for p in one.part)                                     # the inserted aircraft read


def test_a_warm_up_round_s_readout_is_checked_against_the_start_s(tmp_path):
    """§2 item 10 point 6: in a warm-up round the selection readout is the start's — a start from a round: that round's
    readout, read with the same select windows (otherwise not compared); the base: round 0's, from round 1 on;
    `close_value_round` says whether it is the same, and writes V's file with the round's identity."""
    from ts_transformer.experiments import post_train
    from ts_transformer.tests.test_post_train import _inputs

    def campaign(path, settings, readouts):
        path.mkdir(parents=True)
        (path / "campaign.json").write_text(json.dumps({"inputs": _inputs(settings, tmp_path)}))
        for r, readout in enumerate(readouts):
            (path / f"round_{r}").mkdir()
            (path / f"round_{r}" / "round.json").write_text(json.dumps({"windows": [], "selection_readout": readout}))

    context = SimpleNamespace(base_identity={"base": 1})
    companion = SimpleNamespace(value=SimpleNamespace(state_dict=lambda: {}, shape=lambda: {}),
                                optimizer=SimpleNamespace(state_dict=lambda: {}))
    a, b = {"KXXX": {"reward_mean": 0.5}}, {"KXXX": {"reward_mean": 0.4}}
    source = tmp_path / "source"
    campaign(source, _value_settings(), [a])
    start = {"campaign": str(source), "round": 0, "checkpoint_sha256": "0" * 64}
    for select, readout, expected in ((1, a, True), (1, b, False), (2, a, None)):
        out = tmp_path / f"campaign_{select}_{expected}"
        settings = _value_settings(start=start, seed=2024, select_per_airport=select)
        campaign(out, settings, [readout])
        closed = post_train.close_value_round(context, settings, out, 0, companion, {"selection_readout": readout})
        assert closed["warmup_readout"]["against"] == f"{source}/round_0"
        assert closed["warmup_readout"]["same"] is expected
        held = torch.load(out / "round_0" / "value.pt", weights_only=False)
        assert held["identity"] == post_train.identity(context, settings, out, 1)
    base = tmp_path / "base"
    settings = _value_settings(value_warmup=2)
    campaign(base, settings, [a, a])
    assert post_train.close_value_round(context, settings, base, 0, companion, {"selection_readout": a})[
        "warmup_readout"] == {"against": None, "same": None}                       # round 0: nothing before it
    assert post_train.close_value_round(context, settings, base, 1, companion, {"selection_readout": a})[
        "warmup_readout"] == {"against": "round_0", "same": True}
    assert post_train.close_value_round(context, _value_settings(value_warmup=1), base, 1, companion,
                                        {"selection_readout": b}) == {"value_file": "value.pt"}   # after warm-up


def test_v_s_reading_changes_nothing_the_loop_says_or_flies(setup):
    """D90 at speaking time: a loop with V's reader says and flies the same words and states as one without; the time
    left falls with the rows flown."""
    from ts_transformer.experiments.post_train import read_value

    s = setup
    model = _with_module(s["base"])
    ends = []
    for reader in (None, read_value):
        flown = _window_loop(s, model, [s["windows"][0]])
        if reader is not None:
            flown.value_reader, flown._values, flown._pending = reader, [[]], [None]
        (end,) = flown.run([np.random.default_rng(7)])
        ends.append((end, flown))
    (a, _), (b, flown) = ends
    assert np.array_equal(a.words, b.words) and np.array_equal(a.states, b.states) and a.outcome == b.outcome
    ((part, left),) = flown.values("train")
    start = flown.speaking.start
    said = np.diff(left[start:start + 20])
    assert np.allclose(said[said < 0.5], -4.0 / 900.0, atol=1e-5)      # 4 s a row
    assert np.allclose(said[said >= 0.5], 896.0 / 900.0, atol=1e-5)     # a go-around's 900 s, less the row


def _weights(model) -> str:
    import hashlib

    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


#: A value campaign of 2 rounds (warm-up then training, two passes, a clip) on the fixture's short round, as the code
#: before D173 made it (2026-10-08, dev-two-tier-v4-post c288ad77): its models' and V's weights and the old keys of
#: V's record. D173's defaults must give it bit for bit.
VALUE_BEFORE_D173 = "e38ec0f923c83eed37d97172616d3c09676fe8d4f946d73d1842f26503cd2efb"


def test_with_d173_s_defaults_a_value_campaign_is_the_code_s_before_bit_for_bit(setup, tmp_path, monkeypatch):
    import hashlib

    from ts_transformer.experiments.post_train import STAGE_C_VALUE, open_campaign, run_campaign
    from ts_transformer.tests.test_post_train import _context, _inputs, short_round

    s = setup
    short_round(monkeypatch, s)
    settings = _value_settings(rounds=2, epochs=2, clip_norm=1.0)
    out = tmp_path / "campaign"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)                          # the digest was taken on one thread (as test_post_generalised's)
    try:
        open_campaign(out, _inputs(settings, tmp_path), {"head": "x", "dirty": False}, {})
        run_campaign(out, settings, _context(s), stage=STAGE_C_VALUE)
    finally:
        torch.set_num_threads(threads)
    h = hashlib.sha256()
    for r in range(2):
        for name in ("checkpoint.pt", "value.pt"):
            state = torch.load(out / f"round_{r}" / name, weights_only=False)
            for key in ("model", "value"):
                if key in state:
                    for k, v in state[key].items():
                        h.update(k.encode())
                        h.update(v.detach().cpu().contiguous().numpy().tobytes())
        value = json.loads((out / f"round_{r}" / "round.json").read_text())["pass"]["value"]
        old = {k: value[k] for k in ("loss_before", "advantage_mean", "advantage_std", "rows", "loss_after",
                                     "explained_after", "updates")}
        h.update(json.dumps(old, sort_keys=True).encode())
        assert value["centred_by"] is None and value["passes"] == 2
    assert h.hexdigest() == VALUE_BEFORE_D173


def _spoken_round(s, tmp_path, monkeypatch, **changed):
    """The fixture's value round spoken into a directory, a batch (a file) each: the real window (its whole flight, many
    counted rows) and the window with an aircraft inserted ahead (one counted row); the context, the settings, the
    start model and its optimizer."""
    from ts_transformer.experiments.post_train import STAGE_C_VALUE, speak_round, start_model
    from ts_transformer.tests.test_post_train import _context

    context = _context(s)
    settings = _value_settings(batch_windows=1, **changed)
    model, optimizer = start_model(context, settings)
    windows = [s["windows"][0], _ahead(s["windows"][0])]
    directory = tmp_path / "round"
    directory.mkdir()
    speak_round(model, context, windows, settings, 0, directory, stage=STAGE_C_VALUE)
    return context, settings, model, optimizer, directory


def test_the_centring_takes_the_round_s_mean_off_the_advantages_and_keeps_the_targets(setup, tmp_path, monkeypatch):
    """D173 (a): after V at the round's start has read the samples, each counted row's advantage less the round's mean
    of them — their mean 0, the rows not counted 0 — and the targets those before the centring."""
    from ts_transformer.experiments.post_train import round_advantages

    context, settings, model, _, directory = _spoken_round(setup, tmp_path, monkeypatch)
    torch.manual_seed(0)
    value = Value.of(model)
    plain, before = round_advantages(directory, CPU, value)
    centred, after = round_advantages(directory, CPU, value, centering=True)
    assert before["centred_by"] is None and after["centred_by"] == pytest.approx(before["advantage_mean"])
    assert before["advantage_mean"] != 0.0 and before["advantage_std"] > 0.0 and len(plain) == 2   # two files
    assert any(bool((c[m] != 0).any()) for c, _, m in
               [(centred[k][0], plain[k][0], plain[k][0] != 0) for k in plain])
    rows = [(centred[k][0], plain[k][0], plain[k][0] != 0) for k in plain]
    total = sum(float(c[m].sum()) for c, _, m in rows)
    assert abs(total) < 1e-4                                                       # their mean 0
    assert all(torch.allclose(c[m], p[m] - before["advantage_mean"]) and not c[~m].any() for c, p, m in rows)
    assert all(torch.equal(centred[k][1], plain[k][1]) for k in plain)              # the targets kept


def test_v_steps_in_its_first_passes_only_and_the_model_in_every_pass(setup, tmp_path, monkeypatch):
    """D173 (b): with `value_epochs` 1 and `epochs` 4, V's weights change in the first pass only and the model's in all
    four; a warm-up round makes V's pass only; `value_epochs` above `epochs` is refused."""
    from ts_transformer.experiments import post_train
    from ts_transformer.experiments.post_train import ValueRun, value_train_pass

    context, settings, model, optimizer, directory = _spoken_round(setup, tmp_path, monkeypatch, epochs=4,
                                                                    value_epochs=1, value_warmup=0)
    companion = ValueRun.of(context, settings, tmp_path, 0, model)
    seen = []
    real = post_train.value_pairs

    def watched(*a, **k):
        seen.append((_weights(model), _weights(companion.value)))
        yield from real(*a, **k)

    monkeypatch.setattr(post_train, "value_pairs", watched)
    record = value_train_pass(model, context, optimizer, directory, settings, np.random.default_rng(0), part_width=0,
                              round_=0, companion=companion)
    seen.append((_weights(model), _weights(companion.value)))
    models, values = [m for m, _ in seen], [v for _, v in seen]
    assert len(seen) == 5 and len(set(models)) == 5                               # the model moves in every pass
    assert values[0] != values[1] and len(set(values[1:])) == 1                   # V in the first pass only
    assert record["value"]["passes"] == 1 and len(record["passes"]) == 4
    assert record["value"]["updates"] == record["updates"] // 4                   # V's updates: the first pass's
    seen.clear()
    warm = replace(settings, value_warmup=1)
    value_train_pass(model, context, optimizer, directory, warm, np.random.default_rng(0), part_width=0, round_=0,
                     companion=companion)
    assert len(seen) == 1                                                         # warm-up: V's one pass only
    with pytest.raises(ValueError, match="value_epochs 5: from 1 to epochs"):
        _value_settings(epochs=4, value_epochs=5)


def test_d173_s_settings_go_with_the_value_method_and_a_record_without_them_reads_the_defaults(tmp_path):
    from ts_transformer.experiments import post_train
    from ts_transformer.experiments.post_train import open_campaign
    from ts_transformer.tests.test_post_train import _inputs, _settings

    for wrong in (dict(advantage_centering=True), dict(value_epochs=1)):
        with pytest.raises(ValueError, match="advantage_centering and value_epochs are the value method's"):
            _settings(**wrong)
    settings = _value_settings()
    out = tmp_path / "campaign"
    open_campaign(out, _inputs(settings, tmp_path), {"head": "x", "dirty": False}, {})
    record = json.loads((out / "campaign.json").read_text())
    for added in ("advantage_centering", "value_epochs"):
        del record["inputs"]["settings"][added]                                   # a record from before D173
    (out / "campaign.json").write_text(json.dumps(record))
    assert post_train.settings_of(record) == settings
    assert len(open_campaign(out, _inputs(settings, tmp_path), {"head": "x", "dirty": False}, {})["resumed"]) == 1
    for changed in (dict(advantage_centering=True), dict(value_epochs=1)):
        with pytest.raises(SystemExit, match="other inputs or settings"):
            open_campaign(out, _inputs(_value_settings(**changed), tmp_path), {"head": "x", "dirty": False}, {})

