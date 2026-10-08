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
    setting = model_speed.window_setting(context, model, [_ahead(s["windows"][0])], [0], 1337, CPU, 1)
    assert setting["loopSizes"] == [1] and setting["rowsTimed"] >= 1 and setting["priorStepMs"]["p50"] > 0


def test_stage_c_flies_the_drawn_windows_at_their_places_with_their_readout_numbers(monkeypatch):
    """`window_setting`: the windows flown are those at the drawn places among the selection windows, at batch 1 and in
    a batch (the batch's indices mapped back to the places), each with the selection readout's numbers at its place."""
    from types import SimpleNamespace

    from ts_transformer.experiments import post_train, post_window_loop

    windows = [SimpleNamespace(name=f"w{k}", signal_index=k) for k in range(8)]
    flown, numbered = [], []

    class Loop:
        def __init__(self, model, timed, order, chosen, *a, **k):
            flown.append([w.name for w in chosen])

        def run(self, numbers):
            pass

    context = SimpleNamespace(start_loop=lambda split, chosen: lambda indices: (None, indices, {}),
                              splits={model_speed.SPLIT: {"sentences": None, "flights": None, "faults": None}},
                              geometries=None, rosters=None, finals=None, words=None, interval_s=4.0, variant="full",
                              edges_reference=None)
    monkeypatch.setattr(post_window_loop, "WindowLoop", Loop)
    monkeypatch.setattr(post_train, "readout_numbers", lambda seed, place: numbered.append((seed, place)))
    batched = []
    monkeypatch.setattr(post_train, "batches", lambda chosen, size: (batched.append([w.name for w in chosen]),
                                                                     [[1, 0]])[1])
    monkeypatch.setattr(model_speed, "TimedLoop", lambda loop, sync, limit: SimpleNamespace())
    monkeypatch.setattr(model_speed, "summary", lambda timed, interval_s: {})
    monkeypatch.setattr(model_speed, "host_info", lambda device: {})
    for batch, expected in ((1, [["w3"], ["w3"], ["w7"]]), (400, [["w7", "w3"], ["w7", "w3"]])):
        flown.clear(), numbered.clear()
        model_speed.window_setting(context, None, windows, [3, 7], 1337, CPU, batch)
        assert flown == expected                                  # the warm-up's group first, then every group
        assert [place for _, place in numbered] == [p for group in expected for p in (int(w[1]) for w in group)]
        assert {seed for seed, _ in numbered} == {1337}
    assert batched == [["w3", "w7"]]                              # the batch is made of the drawn windows only


def test_each_setting_draws_its_own_windows_a_batch_at_least_its_size():
    """`setting_draws`: batch 1 takes ``--per-airport`` of each airport; a batch at least the batch, an equal share an
    airport (400 over 5 airports: 80 each), fewer where an airport holds fewer."""
    from types import SimpleNamespace

    def windows(sizes):
        return [SimpleNamespace(scene=SimpleNamespace(geometry=SimpleNamespace(code=f"K{k}")))
                for k, size in enumerate(sizes) for _ in range(size)]

    draws = model_speed.setting_draws(windows([200] * 5), 20, [1, 400], 1337)
    assert {batch: len(places) for batch, places in draws.items()} == {1: 100, 400: 400}
    short = model_speed.setting_draws(windows([200, 200, 200, 200, 50]), 20, [1, 400], 1337)
    assert {batch: len(places) for batch, places in short.items()} == {1: 100, 400: 370}


def test_stage_c_times_a_seeded_draw_of_each_airport_s_selection_windows():
    """Stage C times ``--per-airport`` of each airport's selection windows (the user, 2026-10-08: a sample, never all),
    drawn with the seed, by their places among the selection windows (each keeps its readout's numbers)."""
    from types import SimpleNamespace

    windows = [SimpleNamespace(scene=SimpleNamespace(geometry=SimpleNamespace(code=code)))
               for code in ["KAAA"] * 5 + ["KBBB"] * 2 + ["KAAA"] * 3]
    places = model_speed.drawn_places(windows, 3, 1337)
    assert len(places) == 3 + 2 and places == sorted(places[:3]) + sorted(places[3:])
    assert all(windows[p].scene.geometry.code == "KAAA" for p in places[:3]) and places[3:] == [5, 6]
    assert places == model_speed.drawn_places(windows, 3, 1337) != model_speed.drawn_places(windows, 3, 7)


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
