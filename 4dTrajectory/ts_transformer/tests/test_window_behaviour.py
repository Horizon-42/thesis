"""The behaviour check of the window loop (`experiments/window_behaviour.py`; multi-aircraft control D181, D183, D186;
post-training D185) on the synthetic artefact of stage C's campaign tests, on the CPU: a stage's fixed batches and
readout are digested the same twice, other random numbers' are named where they differ, the capture of the branch rounds
comes off, and records of other inputs are refused."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from ts_transformer.experiments import multi_train, post_train, window_behaviour
from ts_transformer.experiments.window_behaviour import behaviour, chosen_batches, differences, readout_places
from ts_transformer.multi.windows import REAL_KIND
from ts_transformer.tests.test_multi_train import _multi_settings, _stage_c
from ts_transformer.tests.test_post_branches import _ahead
from ts_transformer.tests.test_post_train import _context, _settings, short_round
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401


def _record(digests, **inputs):
    return {"inputs": {"campaign": "c", **inputs}, "digests": digests}


def _checked(context, settings, stage):
    """The digests of a stage's start twice (the same), and with other random numbers (named)."""
    real = post_train.branch_round
    model = stage.start(context, settings)[0].eval()
    once = behaviour(context, settings, stage, model, 0, None, 1)
    assert post_train.branch_round is real and multi_train.branch_round is real          # the capture came off
    assert once == behaviour(context, settings, stage, model, 0, None, 1)
    assert list(once["batches"]) == ["0"] and once["batches"]["0"]["groups"] and list(once["readout"]) == ["0"]
    seeds = {"seed": settings.seed + 1, **({"select_seed": settings.select_seed + 1}
                                          if isinstance(settings, post_train.Settings) else {})}
    moved = behaviour(context, replace(settings, **seeds), stage, model, 0, None, 1)       # other numbers
    assert differences(_record(once), _record(once)) == []
    named = differences(_record(once), _record(moved))
    assert "readout window 0" in named and any(n.startswith("batch 0: ") for n in named)
    if isinstance(settings, multi_train.MultiSettings):    # D219 (3): the readout's seed held, the training's moved
        held = behaviour(context, replace(settings, seed=settings.seed + 1, select_seed=settings.seed), stage, model, 0,
                         None, 1)
        named = differences(_record(once), _record(held))
        assert not any(n.startswith("readout") for n in named) and any(n.startswith("batch 0: ") for n in named)
    with pytest.raises(ValueError, match="other inputs"):
        differences(_record(once), _record(once, model_round=1))
    return once


def test_stage_c_s_fixed_batch_and_readout(setup, monkeypatch):
    s = setup
    short_round(monkeypatch, s)
    _checked(_context(s), _settings(rounds=1, continuations=2), post_train.STAGE_C)


def test_stage_d_s_fixed_batch_and_readout(setup, tmp_path, monkeypatch):
    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    settings = _multi_settings(start=_stage_c(s, tmp_path, context), continuations=2)
    window = replace(_ahead(s["windows"][0]), span_s=1.0)
    monkeypatch.setattr(multi_train, "draw_windows", lambda context, settings, rng: ([window], {"drawn": {REAL_KIND: 1}}))
    found = _checked(context, settings, multi_train.stage_d())
    assert found["batches"]["0"]["span_s"] == 1.0


def test_the_batches_and_readout_windows_chosen():
    windows = [SimpleNamespace(span_s=s) for s in (300.0, 300.0, 600.0, 300.0, 600.0, 1200.0)]
    found = [[0, 1], [2, 4], [3], [5]]
    assert chosen_batches(windows, found, None) == [(0, [0, 1]), (1, [2, 4]), (3, [5])]
    assert chosen_batches(windows, found, 1) == [(0, [0]), (1, [2]), (3, [5])]
    assert readout_places(windows, 1) == [0, 2, 5] and readout_places(windows, 2) == [0, 1, 2, 4, 5]
    assert readout_places(windows, 0) == []


def test_a_stage_c_campaign_of_another_method_than_branch_is_refused_by_name():
    record = {"schema": post_train.CAMPAIGN_SCHEMA, "inputs": {"settings": {}}}
    value = {"value_lr": 1e-4, "value_warmup": 0, "continuations": 1}
    for method in (post_train.LANDED_SENTENCES, post_train.VALUE):
        settings = _settings(method=method, **(value if method == post_train.VALUE else {}))
        with pytest.raises(ValueError, match=f"method {method!r} speaks none"):
            window_behaviour.opened_campaign({**record, "inputs": {"settings": asdict(settings)}})
    assert window_behaviour.opened_campaign({**record, "inputs": {"settings": asdict(_settings())}})[1] is \
        post_train.STAGE_C


def test_the_runner_refuses_an_output_that_exists_and_no_windows(tmp_path, capsys):
    (tmp_path / "there").mkdir()
    for argv, message in ((["--out", str(tmp_path / "there")], "exists"),
                          (["--out", str(tmp_path / "new"), "--windows", "0"], "--windows is at least 1")):
        with pytest.raises(SystemExit):
            window_behaviour.main(["--campaign", str(tmp_path)] + argv)
        assert message in capsys.readouterr().err
