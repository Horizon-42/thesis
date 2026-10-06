"""D136 (outline §6.2 item 10): the model's speed — the prior's step of a row and the executor's steps of it timed apart,
on A26's synthetic artefact of one flight; the timing changes nothing that is said or flown."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.autopilot.start import NO_MOVE, start_moved
from ts_transformer.experiments import model_speed
from ts_transformer.experiments.model_speed import TimedLoop, Warmed, prior_groups, summary
from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop, flight_numbers
from ts_transformer.instructions.words import COLUMNS
from ts_transformer.tests import test_prior_free_generation as free
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401

CPU = torch.device("cpu")


def spoken(tmp_path, monkeypatch, *, timed: bool, limit=None):
    """The synthetic flight spoken to its end (`test_prior_free_generation.generate`'s setup), its loop ``timed`` or not:
    the speaking loop and the timed loop (None when not timed)."""
    spy: dict = {}
    _, model = free.generate(tmp_path, monkeypatch, spy=spy, speaking="start")
    loop, order, observed = start_moved(spy["directory"], "train", 4.0, {0: spy["stored"]}, tmp_path / "executor",
                                        {0: NO_MOVE}, most_go_arounds=free.MOST_GO_AROUNDS, device=CPU)
    wrapped = TimedLoop(loop, lambda: None, limit) if timed else loop
    finals = {spy["geometry"].code: tuple(free.Final(spy["geometry"], k, 9_000.0, free.fas_course_geometry(c.length_m))
                                          for k, c in enumerate(spy["geometry"].candidates))}
    speaking = SpeakingLoop(model, wrapped, order, {0: spy["stored"]}, observed, spy["flights"],
                            {spy["geometry"].code: spy["geometry"]}, [spy["landings"][spy["geometry"].code]], finals,
                            spy["words"], interval_s=4.0, variant="full", device=CPU)
    numbers = flight_numbers(5, 0, 0)
    try:
        while speaking.observing:
            speaking.observe()
        while speaking.alive.any():
            speaking.step(numbers.random((1, len(COLUMNS))))
    except Warmed:
        pass
    return speaking, (wrapped if timed else None)


def test_the_timed_loop_says_and_flies_what_the_loop_does_and_times_each_row_apart(tmp_path, monkeypatch):
    plain, _ = spoken(tmp_path / "plain", monkeypatch, timed=False)
    speaking, timed = spoken(tmp_path / "timed", monkeypatch, timed=True)
    said, flown = plain.generated()[0], speaking.generated()[0]
    assert np.array_equal(said.words, flown.words) and np.array_equal(said.states, flown.states)
    assert said.outcome == flown.outcome
    rows = len(flown.words)
    assert len(timed.prior_s) == len(timed.executor_s) == len(timed.flying) == rows
    assert all(t > 0 for t in timed.prior_s) and all(t > 0 for t in timed.executor_s)
    assert [bool(f[0]) for f in timed.flying] == [True] * rows           # one flight, flying in every row it said
    # every moment from the end of the observed rows to the last halt is in one timing or the other
    assert sum(timed.prior_s) + sum(timed.executor_s) == pytest.approx(timed._since - timed.started, abs=1e-9)


def test_the_warm_up_stops_after_its_rows(tmp_path, monkeypatch):
    _, timed = spoken(tmp_path, monkeypatch, timed=True, limit=2)
    assert len(timed.prior_s) == 2


def test_a_step_before_the_observed_rows_end_is_refused():
    class Executor:
        done = halted = torch.zeros(1, dtype=torch.bool)

    class Loop:
        executor = Executor()

        def step(self, row):
            return row, np.zeros(1, bool)

    with pytest.raises(ValueError, match="no start"):
        TimedLoop(Loop(), lambda: None).step(np.zeros((1, 5)))


def test_every_drawn_flight_is_spoken_in_loops_of_the_batch_or_repeated_to_it():
    assert prior_groups([7, 9, 11], 7) == [[(7, 0), (9, 0), (11, 0), (7, 1), (9, 1), (11, 1), (7, 2)]]
    assert prior_groups([7, 9, 11], 2) == [[(7, 0), (9, 0)], [(11, 0)]]
    assert prior_groups([7, 9, 11], 1) == [[(7, 0)], [(9, 0)], [(11, 0)]]


def test_the_runner_s_loop_says_what_free_generation_says_copy_k_its_sample_k(tmp_path, monkeypatch):
    """`speak_prior` and `prior_setting` on the synthetic flight (a train flight: the split given as train here): a loop
    of three copies of it, copy k saying what a plain loop says with sample k's numbers."""
    from types import SimpleNamespace

    from ts_transformer.experiments import prior_speaking_loop

    spy: dict = {}
    _, model = free.generate(tmp_path, monkeypatch, spy=spy, speaking="start")
    geometry = spy["geometry"]
    finals = {geometry.code: tuple(free.Final(geometry, k, 9_000.0, free.fas_course_geometry(c.length_m))
                                   for k, c in enumerate(geometry.candidates))}
    prior = SimpleNamespace(interval_s=4.0, geometries={geometry.code: geometry}, landings=spy["landings"],
                            checkpoint=SimpleNamespace(model=model))
    run = SimpleNamespace(prior=prior, instructions=spy["directory"], executor=tmp_path / "executor",
                          sentences={0: spy["stored"]}, flights=spy["flights"], finals=finals, drawn=[0])
    monkeypatch.setattr(model_speed, "SPLIT", "train")
    seen = []

    class Seen(SpeakingLoop):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen.append(self)

    monkeypatch.setattr(prior_speaking_loop, "SpeakingLoop", Seen)
    timed = model_speed.speak_prior(run, model, [(0, 0), (0, 1), (0, 2)], 1337, CPU)
    monkeypatch.setattr(prior_speaking_loop, "SpeakingLoop", SpeakingLoop)
    said = seen[0].generated()
    for k in range(3):
        reference = _plain(spy, tmp_path, model, finals, flight_numbers(1337, k, 0))
        assert np.array_equal(said[k].words, reference.words) and np.array_equal(said[k].states, reference.states)
        assert sum(bool(f[k]) for f in timed.flying) == len(reference.words)
    setting = model_speed.prior_setting(run, CPU, 3, 1337)
    assert setting["loopSizes"] == [3] and setting["sentences"] == 3 and setting["rowsTimed"] >= 1


def _plain(spy, tmp_path, model, finals, numbers):
    """The synthetic flight spoken in a plain loop with ``numbers``: what free generation says."""
    loop, order, observed = start_moved(spy["directory"], "train", 4.0, {0: spy["stored"]}, tmp_path / "executor",
                                        {0: NO_MOVE}, most_go_arounds=free.MOST_GO_AROUNDS, device=CPU)
    geometry = spy["geometry"]
    speaking = SpeakingLoop(model, loop, order, {0: spy["stored"]}, observed, spy["flights"],
                            {geometry.code: geometry}, [spy["landings"][geometry.code]], finals, spy["words"],
                            interval_s=4.0, variant="full", device=CPU)
    while speaking.observing:
        speaking.observe()
    while speaking.alive.any():
        speaking.step(numbers.random((1, len(COLUMNS))))
    return speaking.generated()[0]


def test_a_round_s_model_is_timed_in_the_window_loop(setup, monkeypatch):
    """`window_setting` on the synthetic windows (their split given as train here): the window loop under the timed
    loop, one window a loop."""
    from ts_transformer.experiments import post_train
    from ts_transformer.tests.test_post_branches import _ahead
    from ts_transformer.tests.test_post_train import _context, _settings

    s = setup
    context = _context(s)
    model, _ = post_train.start_model(context, _settings(rounds=1))
    monkeypatch.setattr(model_speed, "SPLIT", "train")
    monkeypatch.setattr(model_speed, "WARMUP_ROWS", 1)
    setting = model_speed.window_setting(context, model, [_ahead(s["windows"][0])], 1337, CPU, 1)
    assert setting["loopSizes"] == [1] and setting["rowsTimed"] >= 1 and setting["priorStepMs"]["p50"] > 0


def test_the_summary_of_fixed_times():
    one = TimedLoop(None, lambda: None, prior_s=[0.010, 0.030], executor_s=[0.002, 0.004],
                    flying=[np.array([True, True]), np.array([True, False])])
    out = summary([one], 4.0)
    assert out["rowsTimed"] == 2 and out["sentences"] == 2
    assert out["priorStepMs"]["max"] == pytest.approx(30.0) and out["executorStepsMs"]["p50"] == pytest.approx(3.0)
    assert out["rowMs"]["max"] == pytest.approx(34.0)
    # the first flight flew both rows (0.046 s), the second only the first (0.012 s)
    assert out["sentenceS"]["p50"] == pytest.approx((0.012 + 0.046) / 2)
    assert out["sentenceS"]["max"] == pytest.approx(0.046)
    assert out["flightRowsPerS"] == pytest.approx(3 / 0.046)
    assert out["shareOfInterval"]["row"]["p50"] == pytest.approx(0.023 / 4.0)


def test_the_runner_reads_the_select_days_only_and_refuses_a_bad_call(tmp_path, capsys):
    assert model_speed.SPLIT == "select"
    for argv, said in ((["--campaign", str(tmp_path)], "--round goes with --campaign"),
                       (["--prior", str(tmp_path)], "--prior needs --instructions and --executor"),
                       (["--prior", str(tmp_path), "--instructions", str(tmp_path), "--executor", str(tmp_path),
                         "--batches", "0"], "at least 1")):
        with pytest.raises(SystemExit):
            model_speed.main([*argv, "--out", str(tmp_path / "o"), "--smoke"])
        assert said in capsys.readouterr().err
    (tmp_path / "o").mkdir()
    with pytest.raises(SystemExit):
        model_speed.main(["--prior", str(tmp_path), "--instructions", str(tmp_path), "--executor", str(tmp_path),
                          "--out", str(tmp_path / "o"), "--smoke"])
    assert "never overwritten" in capsys.readouterr().err
    (tmp_path / "campaign.json").write_text('{"schema": "ts-post-train-v0", "inputs": {}}')
    with pytest.raises(SystemExit):
        model_speed.main(["--campaign", str(tmp_path), "--round", "start", "--out", str(tmp_path / "c"), "--smoke"])
    assert "campaign, not" in capsys.readouterr().err
