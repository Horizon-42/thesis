"""Stage D, MC5: the memory measure of stage D at the formal size (multi-aircraft control §11 MC5, D179, D180;
`experiments/multi_profile.py`) — from a round of stage C's campaign on the synthetic artefact of stage C's campaign
tests: one worker's measure and the pass, the margin, no round spoken; the runner's refusals. Every write root under
tmp."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ts_transformer.experiments import multi_profile, multi_train, post_train
from ts_transformer.multi.windows import REAL_KIND
from ts_transformer.tests.test_multi_train import _multi_settings, _stage_c
from ts_transformer.tests.test_post_branches import _ahead
from ts_transformer.tests.test_post_train import _context, short_round
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401


def test_a_profile_with_workers_measures_them_and_stops_where_they_do_not_fit(setup, tmp_path, monkeypatch):  # noqa: F811
    """Two workers on the CPU: one worker's measure spoken with round 0's model, the pass's (none on the CPU), its
    device and margin recorded, what does not fit for each N with each peak times the margin (D179); no round spoken,
    no readout read; where ``--speak-workers`` do not fit, the measure stops by name, its record written."""
    import torch

    from ts_transformer.experiments.post_train import Speakers

    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    start = _stage_c(s, tmp_path, context)
    window = replace(_ahead(s["windows"][0]), span_s=1.0)                  # of the campaign's span
    monkeypatch.setattr(multi_train, "draw_windows", lambda context, settings, rng: ([window], {"drawn": {REAL_KIND: 1}}))
    settings = _multi_settings(start=start)
    stage = multi_train.stage_d()
    speakers = Speakers(context, settings, 2, torch.device("cpu"), stage=stage)
    try:
        out = tmp_path / "fits"
        out.mkdir()
        record = multi_profile.profile(context, settings, out, speakers, stage)
        workers = record["workers"]
        assert workers["measured"]["windows"] == 1 and workers["pass"] is None and workers["measured_batches"] == [0]
        assert "one_batch" not in record                                # each span's batch spoken once, by the worker
        (batch,) = workers["measured"]["batches"]
        assert batch["span_s"] == 1.0 and batch["rows"] == 1 and batch["s"] > 0 and batch["gpu_reserved_peak"] is None
        assert set(workers["short"]) == {"1", "2"} and not any(workers["short"].values())
        assert workers["speak_device"] == "cpu" and workers["margin"] == multi_train.MEASURE_MARGIN == 1.3
        assert {"round", "spread"}.isdisjoint(record) and not (out / "round_0").exists()      # D179: no round
        # the margin counts: a host that holds two workers at their measured peaks, not at 1.3 times them
        measured = workers["measured"]
        held = measured["held"]["series"] + measured["held"]["reader_model"]
        tight = 2 * (measured["host"]["peak"] + held) - measured["host"]["now"] + measured["host"]["peak"] // 10
        assert not post_train.workers_fit(2, measured, None, {"host": tight, "gpu": None}, held_now=True)
        monkeypatch.setattr(multi_profile, "available_memory", lambda device: {"host": tight, "gpu": None})
        saved = []
        stopped = tmp_path / "short"
        stopped.mkdir()
        with pytest.raises(SystemExit, match=r"2 speaking workers do not fit \(O15, measured peaks × 1.3\): host"):
            multi_profile.profile(context, settings, stopped, speakers, stage, saved.append)
        assert saved[-1]["workers"]["short"]["2"] and not saved[-1]["workers"]["short"]["1"]
    finally:
        speakers.close()


def test_the_runner_refuses_one_worker_and_cuda_workers_beside_a_campaign_on_the_cpu(tmp_path, capsys):
    """The measure is a speaking worker's (two or more); ``--speak-device`` refused as stage C's (D180)."""
    argv = ["--prior", "p", "--instructions", "i", "--executor", "e", "--windows", "w", "--out", str(tmp_path / "m"),
            "--rounds", "1", "--spans-s", "300", "--c-min", "0.6", "--batch-rows", "1", "--seed", "2024", "--prior-lr",
            "1e-4", "--traffic-lr", "1e-3", "--weight-decay", "0", "--update-groups", "1", "--data-sentences", "1",
            "--select-per-airport", "1", "--windows-real", "1", "--windows-compressed", "0", "--start-campaign", "s",
            "--start-round", "0"]
    for extra, message in ((["--speak-workers", "1"], "--speak-workers is 2 or more"),
                           (["--speak-workers", "2", "--device", "cpu", "--speak-device", "cuda"],
                            "--speak-device cuda needs --device cuda")):
        with pytest.raises(SystemExit):
            multi_profile.main(argv + extra)
        assert message in capsys.readouterr().err
    assert not (tmp_path / "m").exists()


def test_the_runner_measures_its_worker_on_the_speaking_device(tmp_path, monkeypatch):
    """D180: the profile's worker is built on ``--speak-device``, by default on ``--device``; the context opened on
    the CPU before the workers are forked."""
    from dataclasses import dataclass
    from types import SimpleNamespace

    import torch

    @dataclass
    class FakeContext:
        device: torch.device
        base: SimpleNamespace

    base = SimpleNamespace(to=lambda device: base, eval=lambda: base)    # no CUDA touched
    calls = []
    monkeypatch.setattr(multi_profile, "start_of", lambda source, round_, formal: {"campaign": "c", "round": 0,
                                                                                  "checkpoint_sha256": "0" * 64})
    monkeypatch.setattr(multi_profile, "git_state", lambda: {"head": "x", "dirty": False})
    monkeypatch.setattr(multi_profile, "require_conforming_closed_loop", lambda i, e: (None, {"checks": {}}, None))
    monkeypatch.setattr(multi_profile, "checked_edges", lambda reference: None)
    monkeypatch.setattr(multi_profile, "open_context", lambda *a, **k: (calls.append(("context", a[4].type)),
                                                                        FakeContext(a[4], base))[1])
    monkeypatch.setattr(multi_profile, "stage_d", lambda: SimpleNamespace(start=lambda c, s: None))

    class FakeSpeakers:
        def __init__(self, context, settings, workers, device, *, stage):
            calls.append(("speakers", context.device.type, workers, device.type))

        def close(self):
            pass

    monkeypatch.setattr(multi_profile, "Speakers", FakeSpeakers)
    monkeypatch.setattr(multi_profile, "profile", lambda context, settings, out, speakers, stage, save: {
        "workers": {"speak_device": "x", "margin": 1.3, "measured": {"batches": []}, "pass": None, "short": {}}})
    argv = ["--prior", "p", "--instructions", "i", "--executor", "e", "--windows", "w", "--rounds", "1",
            "--spans-s", "300", "--c-min", "0.6", "--batch-rows", "1", "--seed", "2024", "--prior-lr", "1e-4",
            "--traffic-lr", "1e-3", "--weight-decay", "0", "--update-groups", "1", "--data-sentences", "1",
            "--select-per-airport", "1", "--windows-real", "1", "--windows-compressed", "0", "--start-campaign", "s",
            "--start-round", "0", "--speak-workers", "2", "--device", "cuda"]
    assert multi_profile.main(argv + ["--out", str(tmp_path / "a")]) == 0
    assert multi_profile.main(argv + ["--out", str(tmp_path / "b"), "--speak-device", "cpu"]) == 0
    assert [c for c in calls if c[0] == "speakers"] == [("speakers", "cpu", 2, "cuda"), ("speakers", "cpu", 2, "cpu")]
    assert [c for c in calls if c[0] == "context"] == [("context", "cpu")] * 2
