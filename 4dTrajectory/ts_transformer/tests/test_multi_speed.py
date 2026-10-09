"""Stage D, MC11: the speed runner (multi-aircraft control D181 (56); `experiments/multi_speed.py`) on a smoke batch of
the synthetic artefact of stage C's campaign tests, on the CPU: the parts add up to the batch, the steps inside them,
the record is the batch speaker's, the timers come off; the profile only inside its part and only on the CPU."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ts_transformer.autopilot.start import Loop
from ts_transformer.experiments import multi_speed, multi_train
from ts_transformer.experiments.post_window_loop import WindowLoop
from ts_transformer.multi.windows import REAL_KIND
from ts_transformer.prior.speaker import Speaker
from ts_transformer.tests.test_multi_train import _multi_settings, _stage_c
from ts_transformer.tests.test_post_branches import _ahead
from ts_transformer.tests.test_post_train import _context, short_round
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401


def test_a_batch_is_timed_by_its_parts_and_a_profile_runs_inside_one(setup, tmp_path, monkeypatch):  # noqa: F811
    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    settings = _multi_settings(start=_stage_c(s, tmp_path, context), continuations=2)
    window = replace(_ahead(s["windows"][0]), span_s=1.0)
    monkeypatch.setattr(multi_train, "draw_windows", lambda context, settings, rng: ([window], {"drawn": {REAL_KIND: 1}}))
    model, _ = multi_train.stage_d().start(context, settings)
    methods = {(owner, name): getattr(owner, name) for owner, name in
               [(WindowLoop, name) for name in multi_speed.ENTRIES] + [m for ms in multi_speed.STEPS.values() for m in ms]}
    plain = multi_train.stage_d().speak_batch(model, context, *multi_train.stage_d().draw(context, settings, 0)[:1],
                                              [0], settings, 0, tmp_path, 0)
    speed = multi_speed.timed_batch(model.eval(), context, settings, 0, 0)
    assert {getattr(owner, name) for owner, name in methods} == set(methods.values())       # the timers came off
    assert speed["record"] == plain and (speed["windows"], speed["rows"], speed["span"]) == (1, 1, "1")
    parts = speed["parts"]
    assert parts["first_pass"]["entries"] == 1 and parts["first_pass"]["executor_s"] > 0
    assert parts["first_pass"]["speaker_s"] > 0 and parts["first_pass"]["gpu_reserved_peak"] is None
    if plain["spoken_again"]:
        assert parts["second_pass"]["entries"] > 0 and parts["continuations"]["entries"] > 0
    for held in parts.values():
        assert held["window_loop_s"] >= -1e-9
    assert abs(sum(held["s"] for held in parts.values()) + speed["other_s"] - speed["s"]) < 1e-9
    profiled = multi_speed.timed_batch(model, context, settings, 0, 0, profiled="first_pass")
    assert profiled["profile"]["part"] == "first_pass" and "say" in profiled["profile"]["text"]
    with pytest.raises(ValueError, match="the CPU only"):
        multi_speed.timed_batch(model, replace(context, device=multi_speed.torch.device("cuda")), settings, 0, 0,
                                profiled="first_pass")
    assert {Speaker, Loop} <= {owner for owner, _ in methods}


def test_the_runner_refuses_a_profile_on_the_gpu_and_an_output_that_exists(tmp_path, capsys):
    (tmp_path / "there").mkdir()
    for argv, message in ((["--device", "cuda", "--cprofile", "first_pass", "--out", str(tmp_path / "new")],
                           "--cprofile runs on the CPU only"),
                          (["--device", "cpu", "--out", str(tmp_path / "there")], "exists")):
        with pytest.raises(SystemExit):
            multi_speed.main(["--campaign", str(tmp_path), "--batch", "0"] + argv)
        assert message in capsys.readouterr().err
