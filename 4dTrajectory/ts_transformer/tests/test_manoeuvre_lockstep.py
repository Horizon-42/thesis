"""Lockstep (`manoeuvre/lockstep.py`): under every protocol each flight is predicted once per
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
        assert run.predictions == len(run.legs) == len(run.codes) == len(run.rounds)
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
            assert row["ended"] == run.ended and len(row["codes"]) == run.predictions and set(row["at"]) == {"60", "120", "180", "300"}
            assert row["established_at_anchor"] in (True, False) and metrics["ade_m"] >= 0.0
            assert (row["reference"]["established"]) == (run.ended == ls.ENDED_CROSSED)
        # every whole leg is tokenised back under every coded protocol (the closed-loop input);
        # under none there is no codebook to tokenise with
        whole = [leg for leg in run.legs if float(np.sum(leg.sample_durations_s)) == pytest.approx(SEGMENT_S)]
        if protocol == ls.PROTOCOL_NONE:
            assert run.flown_codes == [] and len(run.flown_states) == 1
        else:
            assert len(run.flown_codes) == len(whole) and len(run.flown_states) == len(whole) + 1
        if run.legs:
            assert len(row["flown_states"]) == len(run.flown_states)
            assert ("truth_length" in row) == (run.truth is not None)
            if run.truth is not None:
                assert row["truth_length"] == run.truth.length
        if protocol == ls.PROTOCOL_C:
            assert all(code == run.truth.codes[min(i, run.truth.length - 1)] for i, code in enumerate(run.codes))
            assert run.held_predictions == max(run.predictions - run.truth.length, 0)
        elif protocol == ls.PROTOCOL_NONE:
            assert all(code == ls.CODE_NONE for code in run.codes) and run.held_predictions == 0
            assert all("e_plan_m" not in record for record in run.rounds)
            assert all(leg.manoeuvre_code is None for leg in run.legs)      # nothing was handed to the executor
        else:
            assert len(run.landed_probabilities) >= 1
            assert all("e_plan_m" in record and "truth_code" in record for record in run.rounds)
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
        assert run.predictions <= run.truth.length + 1 or run.ended != ls.ENDED_TRUTH_EXHAUSTED
    with pytest.raises(ValueError, match="protocol"):
        ls.fly(executor, codebook, series, "B", prior=prior, device=torch.device("cpu"), batch_size=3)


def _no_token_executor(series) -> ls.Executor:
    config = _no_token_config()
    torch.manual_seed(1)
    model = build_model(config).eval()
    with torch.no_grad():
        model.control_head.control_projection.weight.normal_(std=0.05)
    return ls.Executor(model=model, config=config, normalizer=Normalizer.fit(series))


def test_protocol_none_flies_the_no_token_twin_without_a_code_or_a_codebook(world):
    """The baseline (2026-09-18; two-tier v3 stage A): a no-token executor under the same rounds
    and budget, no code handed over and no codebook at all — the row has no code columns. The
    first prediction is at L−1 (v3 D2: no floor). A coded executor is refused under `none`, the no-token
    one under every coded protocol, a codebook under `none`, and a prior under `none`."""
    codebook, _executor, series, prior = world
    twin = _no_token_executor(series)
    runs = ls.fly(twin, None, series, ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=2)
    assert len(runs) == 3
    _check_runs(runs, ls.PROTOCOL_NONE)
    for run in runs:
        assert run.truth is None and run.anchor == twin.config.seq_len - 1 == 7 and run.flown_codes == []
        row, _metrics = ls.flight_row(run, points=16)
        assert row["codes"] == [ls.CODE_NONE] * run.predictions
        assert not {"truth_codes", "flown_codes", "truth_length", "landed_fraction_truth"} & set(row)
    with pytest.raises(ValueError, match="codebook"):
        ls.fly(twin, codebook, series, ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=2)
    with pytest.raises(ValueError, match="codebook"):
        ls.fly(_executor, None, series, ls.PROTOCOL_C, prior=None, device=torch.device("cpu"), batch_size=2)
    with pytest.raises(ValueError, match="plan_conditioning"):
        ls.fly(twin, codebook, series, ls.PROTOCOL_C, prior=None, device=torch.device("cpu"), batch_size=2)
    with pytest.raises(ValueError, match="plan_conditioning"):
        ls.fly(_executor, None, series, ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=2)
    with pytest.raises(ValueError, match="prior"):
        ls.fly(twin, None, series, ls.PROTOCOL_NONE, prior=prior, device=torch.device("cpu"), batch_size=2)


def test_the_remaining_path_reading_cuts_each_flight_at_its_bin_row(world):
    """v3 §3.1's second reading (D3): the cohort first seen where each flight has X km of path
    left — the bin's row becomes the cut flight's L−1, the truth-side readings move with it, and
    a flight with no admissible row at the bin is absent, not flown from elsewhere."""
    from ts_transformer.data.approach_difficulty import remaining_path_profile_m

    _codebook, _executor, series, _prior = world
    twin = _no_token_executor(series)
    config = twin.config
    cut, first_rows = ls.from_remaining_path(series, config, 12_000.0)
    assert len(cut) == len(first_rows) == 3
    for item, whole in zip(cut, series, strict=True):
        row = first_rows[whole.dataset_id]
        assert row >= config.seq_len - 1 and item.n_samples == whole.n_samples - (row - config.seq_len + 1)
        # the cut flight's L−1 IS the whole flight's bin row: same time, same remaining path
        assert item.times[config.seq_len - 1] == whole.times[row]     # no floor: the fixed anchor is L−1
        assert remaining_path_profile_m(item)[config.seq_len - 1] == pytest.approx(remaining_path_profile_m(whole)[row])
        assert abs(remaining_path_profile_m(whole)[row] - 12_000.0) == pytest.approx(np.abs(remaining_path_profile_m(whole) - 12_000.0).min())
    runs = ls.fly(twin, None, cut, ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=3)
    _check_runs(runs, ls.PROTOCOL_NONE)
    for run in runs:
        row, _metrics = ls.flight_row(run, points=16)
        assert row["remaining_path_m"] == pytest.approx(remaining_path_profile_m(run.series)[run.anchor])
    # a bin no flight has 20 s of truth after (the synthetic tracks end at the threshold): absent, not flown
    empty, rows = ls.from_remaining_path(series, config, 2.0)
    assert empty == [] and rows == {}


def test_the_common_row_reading_starts_every_flight_at_the_same_row(world):
    """v3's reading (c): every flight cut so that ONE row of the whole flight becomes the
    executor's L−1 — cells of different lookback then fly the same segment; a flight without the
    executor's horizon of truth after that row is absent, not flown from elsewhere."""
    from ts_transformer.data.dataset import effective_min_future_s

    _codebook, _executor, series, _prior = world
    twin = _no_token_executor(series)
    config = twin.config
    a0, horizon_s = ls.default_anchor(config), effective_min_future_s(config)

    def truth_after(whole, row):                                # seconds of truth after a row of the whole flight
        return float(whole.supervision_times[-1] - whole.times[row])

    row = a0 + 2                                                # two rows after the executor's own first row
    assert all(truth_after(whole, row) >= horizon_s for whole in series)
    cut, first_rows = ls.from_row(series, config, row)
    assert [item.dataset_id for item in cut] == [whole.dataset_id for whole in series] and set(first_rows.values()) == {row}
    for item, whole in zip(cut, series, strict=True):
        assert item.n_samples == whole.n_samples - (row - a0)
        assert item.times[a0] == whole.times[row]               # the cut flight's fixed anchor IS the common row
        assert truth_after(item, a0) == pytest.approx(truth_after(whole, row))
    runs = ls.fly(twin, None, cut, ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=3)
    _check_runs(runs, ls.PROTOCOL_NONE)
    # a row the shortest flight has less than the horizon of truth after, the two others more: kept and dropped in one call
    by_length = sorted(series, key=lambda whole: whole.n_samples)
    row = by_length[0].n_samples - 1 - int(horizon_s / config.dt_s / 2)
    assert truth_after(by_length[0], row) < horizon_s <= min(truth_after(whole, row) for whole in by_length[1:])
    cut, first_rows = ls.from_row(series, config, row)
    assert {item.dataset_id for item in cut} == set(first_rows) == {whole.dataset_id for whole in by_length[1:]}
    # the longest flight's last row: no truth after it, and no other flight has that row — nobody is flown
    empty, rows = ls.from_row(series, config, by_length[-1].n_samples - 1)
    assert empty == [] and rows == {}


def test_a_round_the_budget_leaves_no_row_for_flies_nothing_and_records_nothing(world):
    """`_fly_leg` returns None when the remaining budget is below the first query step: no leg,
    no code, no round record — the bookkeeping stays aligned (review 2026-09-18 M4)."""
    codebook, executor, series, _prior = world
    runs = ls.fly(executor, codebook, series[:1], ls.PROTOCOL_C, prior=None, device=torch.device("cpu"), batch_size=1)
    run = runs[0]
    forecast = run.legs[0]
    run.ended, run.flown_s = None, run.horizon_s - 0.1          # 0.1 s of budget left: under one 0.5 s step
    before = (len(run.legs), len(run.codes), len(run.rounds), run.predictions)
    leg = ls._fly_leg(run, run.series, forecast, segment_s=SEGMENT_S, round_index=99, code=7)
    assert leg is None and run.ended == ls.ENDED_HORIZON
    assert (len(run.legs), len(run.codes), len(run.rounds), run.predictions) == before
