"""Stage D, MC5: the profile of stage D at the formal size (multi-aircraft control §11 MC5;
`experiments/multi_profile.py`) — the spread of W per aircraft on hand-worked numbers, and one profile end to end from
a round of stage C's campaign on the synthetic artefact of stage C's campaign tests (one process). Every write root under
tmp."""

from __future__ import annotations

import json
import math

import pytest

from ts_transformer.experiments import multi_profile, multi_train
from ts_transformer.multi.windows import REAL_KIND
from ts_transformer.tests.test_multi_train import _multi_settings, _stage_c
from ts_transformer.tests.test_post_branches import _ahead
from ts_transformer.tests.test_post_train import _context, short_round
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401


def test_the_spread_of_w_per_aircraft_is_by_unit_and_paired_between_the_draws():
    """Part 4: W per aircraft is Σ W over Σ aircraft; its standard error the ratio estimator's by units (a window of a
    span; over all an anchor, its nested windows summed); the paired one of the draws' difference; the units for a
    target scale as the square of the standard errors' ratio, and an airport's are their share."""
    first = [("KXXX", "300", "a", 2.0, 2), ("KXXX", "300", "b", 1.0, 2), ("KXXX", "600", "a", 3.0, 3),
             ("KYYY", "300", "c", 1.0, 1)]
    second = [("KXXX", "300", "a", 1.0, 2), ("KXXX", "300", "b", 1.0, 2), ("KXXX", "600", "a", 3.0, 3),
              ("KYYY", "300", "c", 1.0, 1)]
    out = multi_profile.spread(first[:3], second[:3])
    assert set(out) == {"300", "600", "all"}
    five = out["300"]
    assert (five["units"], five["aircraft"], five["mean_0"], five["mean_1"]) == (2, 4, 0.75, 0.5)
    assert five["se_0"] == pytest.approx(0.25) and five["se_1"] == 0.0     # 2 − 0.75·2 = 0.5, 1 − 1.5 = −0.5
    assert five["difference"] == 0.25 and five["paired_se"] == pytest.approx(0.25)
    assert five["units_for"]["0.01"]["se_0"] == {"all": 1250, "an_airport": 1250}      # 2 × (0.25 / 0.01)², 1 airport
    assert five["units_for"]["0.02"]["paired_se"]["all"] == math.ceil(2 * (0.25 / 0.02) ** 2)
    assert out["600"]["units"] == 1 and math.isnan(out["600"]["se_0"]) and out["600"]["units_for"] is None
    every = out["all"]                                     # anchor a's two windows one unit: (5 W, 5 aircraft), b (1, 2)
    assert (every["units"], every["windows"], every["aircraft"]) == (2, 3, 7)
    assert every["mean_0"] == pytest.approx(6.0 / 7.0)
    m = 6.0 / 7.0
    assert every["se_0"] == pytest.approx(math.sqrt(2 * ((5 - 5 * m) ** 2 + (1 - 2 * m) ** 2)) / 7)
    two = multi_profile.spread(first, second)["300"]
    assert two["airports"] == 2 and two["units_for"]["0.01"]["se_0"]["an_airport"] == math.ceil(
        two["units_for"]["0.01"]["se_0"]["all"] / 2)
    with pytest.raises(ValueError, match="other windows"):
        multi_profile.spread(first, second[:3] + [("KYYY", "300", "c", 1.0, 2)])


def test_a_profile_measures_each_spans_batch_the_round_and_the_spread(setup, tmp_path, monkeypatch):  # noqa: F811
    """MC5 in one process: from round 0 of a stage C campaign, one batch of each span spoken and timed with its rows and
    groups' bytes, the round's speaking, the select windows read with draws 0 and 1 and their spread by span, the pass;
    the record written after each part; no workers' measure without workers."""
    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    start = _stage_c(s, tmp_path, context)
    window = _ahead(s["windows"][0])
    monkeypatch.setattr(multi_train, "draw_windows", lambda context, settings, rng: ([window], {"drawn": {REAL_KIND: 1}}))
    out = tmp_path / "profile"
    out.mkdir()
    saved = []
    record = multi_profile.profile(context, _multi_settings(start=start), out, None, multi_train.stage_d(),
                                   lambda part: saved.append(json.loads(json.dumps(part, default=str))))
    assert record["batches"] == {"0": {"batches": 1, "windows": 1, "rows": [1]}}
    batch = record["one_batch"]["0"]
    assert batch["windows"] == 1 and batch["rows"] == 1 and batch["wall_s"] > 0 and batch["groups_bytes"] > 0
    assert batch["speaking"]["outcomes"] == {"lost_separation": 1}
    assert "workers" not in record and record["round"]["speak_workers"] == 1
    assert record["round"]["speaking"]["windows"] == 1 and record["round"]["groups_bytes"] > 0
    assert record["round"]["selection_windows"] == 1 and record["round"]["pass_s"] > 0
    assert {"selection_readout_0_s", "selection_readout_1_s", "speaking_s"} <= set(record["round"])
    assert set(record["spread"]) == {"1", "all"} and record["spread"]["1"]["units"] == 1
    assert len(saved) >= 4 and "spread" in saved[-1]
    assert not list(out.glob("multi_profile_*"))                           # the batches' scratch removed


def test_a_profile_with_workers_measures_them_and_stops_where_they_do_not_fit(setup, tmp_path, monkeypatch):  # noqa: F811
    """Part 2 with two workers on the CPU: one worker's measure spoken with round 0's model, the pass's (none on the
    CPU), what does not fit for each N; where ``--speak-workers`` do not fit, the profile stops by name after part 2."""
    import torch

    from ts_transformer.experiments.post_train import Speakers

    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    start = _stage_c(s, tmp_path, context)
    window = _ahead(s["windows"][0])
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
        assert set(workers["short"]) == {"1", "2"} and record["round"]["speak_workers"] == 2
        assert set(record["spread"]) == {"1", "all"}
        monkeypatch.setattr(multi_profile, "available_memory", lambda device: {"host": 0, "gpu": None})
        saved = []
        stopped = tmp_path / "short"
        stopped.mkdir()
        with pytest.raises(SystemExit, match=r"2 speaking workers do not fit \(O15\): host"):
            multi_profile.profile(context, settings, stopped, speakers, stage, saved.append)
        assert saved[-1]["workers"]["short"]["2"] and "round" not in saved[-1]
        assert not (stopped / "round_0").exists()
    finally:
        speakers.close()
