"""Lockstep (`manoeuvre/lockstep.py`): under every protocol each flight is re-asked once per
segment on its own flown rows and ends by crossing, by the budget or by the prior's landing;
the flown legs are contiguous; protocol A tokenises every whole leg back and reads e_plan;
the row carries the reference verdicts and the leads. Untrained models: the mechanics are what
is pinned, never a number."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.backbone.adapters import build_model
from ts_transformer.config import CONTROL_RECIPE_SIMPLE_V3, PREDICTION_CONTROL, TSConfig, recipe_settings
from ts_transformer.data.dataset import Normalizer, build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.manoeuvre import lockstep as ls
from ts_transformer.manoeuvre import tokenizer as tok
from ts_transformer.manoeuvre.context import TypeVocabulary
from ts_transformer.manoeuvre.prior import ManoeuvrePrior, PriorConfig
from ts_transformer.tests.support import AIRPORT, RUNWAY

SEGMENT_S, DT_S = 20.0, 2.0
IDENTITY = {"eligible_set_sha256": {"KRDU": "b" * 64}}


def _config(codebook_dir: str) -> TSConfig:
    settings = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False)
    settings.update(dict(
        prediction_output=PREDICTION_CONTROL, plan_conditioning="manoeuvre-code", manoeuvre_fsq_levels=(4, 4),
        manoeuvre_codebook=codebook_dir, control_horizon_s=SEGMENT_S, n_segments=2, control_imitation_loss_weight=0.0,
        final_time_loss_weight=0.0, state_endpoint_loss_weight=0.0, seq_len=8, d_model=16, n_heads=4, d_ff=32,
        e_layers=1, dropout=0.0, device="cpu", epochs=1, patience=1,
    ))
    return TSConfig(**settings)


def _no_token_config() -> TSConfig:
    settings = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False)
    settings.update(dict(
        prediction_output=PREDICTION_CONTROL, control_horizon_s=SEGMENT_S, n_segments=2, control_imitation_loss_weight=0.0,
        final_time_loss_weight=0.0, state_endpoint_loss_weight=0.0, seq_len=8, d_model=16, n_heads=4, d_ff=32,
        e_layers=1, dropout=0.0, device="cpu", epochs=1, patience=1,
    ))
    return TSConfig(**settings)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    torch.manual_seed(0)
    codebook = tok.write_codebook(tmp_path_factory.mktemp("cb") / "cb", tok.tokenizer_for("learned", levels=(4, 4), segment_s=SEGMENT_S, dt_s=DT_S),
                                  segment_s=SEGMENT_S, dt_s=DT_S, data_identity=IDENTITY, source={})
    config = _config(str(codebook.path))
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=3, seed=3), config, airport=AIRPORT)
    model = build_model(config).eval()
    with torch.no_grad():
        model.control_head.control_projection.weight.normal_(std=0.05)
    executor = ls.Executor(model=model, config=config, normalizer=Normalizer.fit(series))
    vocabulary = TypeVocabulary.from_typecodes(str(item.scenario.aircraft.code) for item in series)
    prior = ls.Prior(model=ManoeuvrePrior(PriorConfig(code_count=16, z_dim=2, type_count=vocabulary.size, d_model=32, n_heads=4,
                                                       n_layers=1, d_ff=64, dropout=0.0, max_positions=64)).eval(), vocabulary=vocabulary)
    return codebook, executor, series, prior


def test_the_closing_budget_is_the_archived_rule_and_the_prior_is_sized_for_it():
    assert ls.closing_horizon_s(100.0) == 130.0 and ls.closing_horizon_s(400.0) == 440.0
    # a 700 s truth at Δ = 20 s: the budget is 770 s = 39 rounds → 41 positions, more than the
    # truth's 35 segments + 2 (the overflow a truth-sized prior hit under protocol A)
    assert ls.required_positions(700.0, 20.0, 35) == 41
    assert ls.required_positions(100.0, 60.0, 1) == 5           # 130 s / 60 → 3 rounds + 2
    assert ls.required_positions(100.0, 60.0, 9) == 11          # never below the longest truth + 2


def _check_runs(runs, protocol):
    for run in runs:
        assert run.ended in (ls.ENDED_CROSSED, ls.ENDED_HORIZON, ls.ENDED_LANDED, ls.ENDED_TRUTH_EXHAUSTED)
        assert run.asks == len(run.legs) == len(run.codes) == len(run.asks_e)
        if run.legs:
            # contiguous legs, each one segment long except the last
            times = np.concatenate([leg.times for leg in run.legs])
            assert bool((np.diff(times) > 0).all())
            for leg in run.legs[:-1]:
                assert float(np.sum(leg.sample_durations_s)) == pytest.approx(SEGMENT_S)
            assert run.flown_s == pytest.approx(float(np.sum(np.concatenate([leg.sample_durations_s for leg in run.legs]))))
            assert run.flown_s <= run.horizon_s + 1e-6
            row, metrics = ls.flight_row(run, points=16)
            whole = ls.whole_forecast(run)
            assert whole.predicted_final_time_s == pytest.approx(run.flown_s)      # the span IS the arrival
            assert whole.final_time_s == pytest.approx(run.flown_s)
            # the leads hold the forecast's last row past its end: 60 s after the anchor is
            # readable for every flight, however early it ended (the truth reaches it)
            assert row["at"]["60"] is not None
            assert set(row["reference"]) == {"fully_flyable", "violations", "established"}
            assert row["ended"] == run.ended and len(row["codes"]) == run.asks and set(row["at"]) == {"60", "120", "180", "300"}
            assert row["established_at_anchor"] in (True, False) and metrics["ade_m"] >= 0.0
            assert (row["reference"]["established"]) == (run.ended == ls.ENDED_CROSSED)
        # every whole leg is tokenised back under every protocol (the closed-loop input)
        whole = [leg for leg in run.legs if float(np.sum(leg.sample_durations_s)) == pytest.approx(SEGMENT_S)]
        assert len(run.flown_codes) == len(whole) and len(run.flown_states) == len(whole) + 1
        if run.legs:
            assert len(row["flown_states"]) == len(run.flown_states) and row["truth_length"] == run.truth.length
        if protocol == ls.PROTOCOL_C:
            assert all(code == run.truth.codes[min(i, run.truth.length - 1)] for i, code in enumerate(run.codes))
            assert run.held_asks == max(run.asks - run.truth.length, 0)
        elif protocol == ls.PROTOCOL_NONE:
            assert all(code == ls.CODE_NONE for code in run.codes) and run.held_asks == 0
            assert all("e_plan_m" not in ask for ask in run.asks_e)
            assert all(leg.manoeuvre_code is None for leg in run.legs)      # nothing was handed to the executor
        else:
            assert len(run.landed_probabilities) >= 1
            assert all("e_plan_m" in ask and "truth_code" in ask for ask in run.asks_e)
            if run.ended == ls.ENDED_LANDED:
                assert run.landed_fraction is not None and float(np.sum(run.legs[-1].sample_durations_s)) <= SEGMENT_S + 1e-6


def test_protocol_c_flies_the_truth_codes_and_ends_by_crossing_or_budget(world):
    codebook, executor, series, _prior = world
    runs = ls.fly(executor, codebook, series, ls.PROTOCOL_C, prior=None, device=torch.device("cpu"), batch_size=2)
    assert len(runs) == 3
    _check_runs(runs, ls.PROTOCOL_C)
    with pytest.raises(ValueError, match="prior"):
        ls.fly(executor, codebook, series, ls.PROTOCOL_C, prior=world[3], device=torch.device("cpu"), batch_size=2)


def test_protocol_a_tokenises_every_whole_leg_back_and_reads_e_plan(world):
    codebook, executor, series, prior = world
    lines = []
    runs = ls.fly(executor, codebook, series, ls.PROTOCOL_A, prior=prior, device=torch.device("cpu"), batch_size=2, log=lines.append)
    _check_runs(runs, ls.PROTOCOL_A)
    assert lines and lines[0].startswith("    A round 0:")
    for run in runs:
        whole = [leg for leg in run.legs if float(np.sum(leg.sample_durations_s)) == pytest.approx(SEGMENT_S)]
        assert len(run.flown_codes) == len(whole) and len(run.flown_states) == len(whole) + 1
        assert all(0 <= code < 16 for code in run.flown_codes)
    with pytest.raises(ValueError, match="prior"):
        ls.fly(executor, codebook, series, ls.PROTOCOL_A, prior=None, device=torch.device("cpu"), batch_size=2)


def test_protocol_a_truth_reads_the_truth_prefix_and_ends_when_it_runs_out(world):
    codebook, executor, series, prior = world
    runs = ls.fly(executor, codebook, series, ls.PROTOCOL_A_TRUTH, prior=prior, device=torch.device("cpu"), batch_size=3)
    _check_runs(runs, ls.PROTOCOL_A_TRUTH)
    for run in runs:
        assert run.asks <= run.truth.length + 1 or run.ended != ls.ENDED_TRUTH_EXHAUSTED
    with pytest.raises(ValueError, match="protocol"):
        ls.fly(executor, codebook, series, "B", prior=prior, device=torch.device("cpu"), batch_size=3)


def test_protocol_none_flies_the_no_token_twin_without_a_code_and_labels_its_legs(world):
    """The control (2026-09-18): a no-token executor under the same rounds and budget; the
    codebook only labels the truth's and the flown legs' codes. A coded executor is refused
    under `none`, the no-token one under every coded protocol, and a prior under `none`."""
    codebook, _executor, series, prior = world
    config = _no_token_config()
    torch.manual_seed(1)
    model = build_model(config).eval()
    with torch.no_grad():
        model.control_head.control_projection.weight.normal_(std=0.05)
    twin = ls.Executor(model=model, config=config, normalizer=Normalizer.fit(series))
    runs = ls.fly(twin, codebook, series, ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=2)
    assert len(runs) == 3
    _check_runs(runs, ls.PROTOCOL_NONE)
    for run in runs:
        assert run.truth.length >= 1 and all(0 <= code < 16 for code in run.flown_codes)
        row, _metrics = ls.flight_row(run, points=16)
        assert row["codes"] == [ls.CODE_NONE] * run.asks and row["truth_codes"] == run.truth.codes.tolist()
    with pytest.raises(ValueError, match="plan_conditioning"):
        ls.fly(twin, codebook, series, ls.PROTOCOL_C, prior=None, device=torch.device("cpu"), batch_size=2)
    with pytest.raises(ValueError, match="plan_conditioning"):
        ls.fly(_executor, codebook, series, ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=2)
    with pytest.raises(ValueError, match="prior"):
        ls.fly(twin, codebook, series, ls.PROTOCOL_NONE, prior=prior, device=torch.device("cpu"), batch_size=2)


def test_a_round_the_budget_leaves_no_row_for_flies_nothing_and_records_nothing(world):
    """`_fly_leg` returns None when the remaining budget is below the first query step: no leg,
    no code, no ask row — the bookkeeping stays aligned (review 2026-09-18 M4)."""
    codebook, executor, series, _prior = world
    runs = ls.fly(executor, codebook, series[:1], ls.PROTOCOL_C, prior=None, device=torch.device("cpu"), batch_size=1)
    run = runs[0]
    forecast = run.legs[0]
    run.ended, run.flown_s = None, run.horizon_s - 0.1          # 0.1 s of budget left: under one 0.5 s step
    before = (len(run.legs), len(run.codes), len(run.asks_e), run.asks)
    leg = ls._fly_leg(run, run.series, forecast, segment_s=SEGMENT_S, round_index=99, code=7)
    assert leg is None and run.ended == ls.ENDED_HORIZON
    assert (len(run.legs), len(run.codes), len(run.asks_e), run.asks) == before
