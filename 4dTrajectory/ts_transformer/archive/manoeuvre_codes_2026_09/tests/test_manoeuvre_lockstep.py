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


def test_executing_a_prefix_of_each_forecast_is_a_shorter_round_of_the_no_token_reading(world, monkeypatch):
    """A3-a (v3 §5.3, D43): the executor forecasts its whole horizon every round but only the
    first ``execute_s`` is flown and the next prediction is made there — shorter rounds, the same
    budget, more predictions. Refused under a coded protocol (its codes are per whole segment)
    and for a step that is not a whole number of rows inside the horizon."""
    from dataclasses import replace
    from types import SimpleNamespace

    codebook, executor, series, _prior = world
    twin = _no_token_executor(series)
    half = SEGMENT_S / 2
    whole_runs = ls.fly(twin, None, series, ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=3)
    same_runs = ls.fly(twin, None, series, ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=3, execute_s=SEGMENT_S)
    runs = ls.fly(twin, None, series, ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=3, execute_s=half)
    for run, whole, same in zip(runs, whole_runs, same_runs, strict=True):
        # execute_s = the horizon IS the whole-segment reading (the grid's readings rest on this)
        assert same.predictions == whole.predictions and same.ended == whole.ended
        assert np.allclose(ls.whole_forecast(same).values, ls.whole_forecast(whole).values)
        assert run.legs and run.predictions == len(run.legs) == len(run.rounds)
        for leg in run.legs[:-1]:                                    # every leg but the last is exactly the executed prefix
            assert float(np.sum(leg.sample_durations_s)) == pytest.approx(half)
        assert float(np.sum(run.legs[-1].sample_durations_s)) <= half + 1e-6
        assert [r["lead_s"] for r in run.rounds][:2] == pytest.approx([half, 2 * half])
        assert run.horizon_s == whole.horizon_s and run.predictions >= 2 * whole.predictions - 1
    for bad in (SEGMENT_S + DT_S, DT_S * 1.5, 0.0):
        with pytest.raises(ValueError, match="whole number of rows"):
            ls.fly(twin, None, series, ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=3, execute_s=bad)
    # a step that is a whole number of rows but not of integrator steps: the leg could not be cut on the rollout grid
    with pytest.raises(ValueError, match="integrator steps"):
        ls.executed_step_s(SimpleNamespace(control_horizon_s=20.0, dt_s=2.0, control_rollout_integrator_dt_s=3.0), 4.0)
    with pytest.raises(ValueError, match="no-token reading"):
        ls.fly(executor, codebook, series, ls.PROTOCOL_C, prior=None, device=torch.device("cpu"), batch_size=3, execute_s=half)
    # a crossing the forecast predicts AFTER the executed prefix is not flown: the flight goes on and predicts again
    def crosses_at_15_s(forecast, history):
        rows = int(np.searchsorted(np.cumsum(forecast.sample_durations_s), 15.0 + 1e-6, side="right"))
        return replace(ls.cut_rows(forecast, rows), truncated_at_threshold=True, final_time_s=15.0)

    monkeypatch.setattr(ls, "cut_at_threshold_crossing", crosses_at_15_s)
    (whole_run,) = ls.fly(twin, None, series[:1], ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=1)
    (prefix_run,) = ls.fly(twin, None, series[:1], ls.PROTOCOL_NONE, prior=None, device=torch.device("cpu"), batch_size=1, execute_s=half)
    assert whole_run.ended == ls.ENDED_CROSSED and len(whole_run.legs) == 1
    assert float(np.sum(whole_run.legs[0].sample_durations_s)) == pytest.approx(15.0)
    assert prefix_run.ended == ls.ENDED_HORIZON and len(prefix_run.legs) > 1
    assert all(float(np.sum(leg.sample_durations_s)) == pytest.approx(half) for leg in prefix_run.legs[:-1])


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
    leg = ls._fly_leg(run, run.series, forecast, step_s=SEGMENT_S, round_index=99, code=7, token_index=99, phase=0)
    assert leg is None and run.ended == ls.ENDED_HORIZON
    assert (len(run.legs), len(run.codes), len(run.rounds), run.predictions) == before


def test_the_cohort_option_restricts_the_split_to_a_subset_in_the_checkpoints_order():
    """B0 (v3 D33): `--cohort` flies the development cohort's roster of the split, in the
    checkpoint's order; a cohort flight the checkpoint does not hold refuses."""
    from ts_transformer.data.development_cohorts import DevelopmentCohort
    from ts_transformer.experiments.manoeuvre_lockstep import cohort_keys

    payload = {"split": {"train": ["t1", "t2", "t3"], "val": ["v3", "v1", "v2"]}}
    cohort = DevelopmentCohort(name="b", train_flight_ids=("t3", "t1"), val_flight_ids=("v1", "v3"), selection={})
    assert cohort_keys(payload, "val", cohort) == ["v3", "v1"]
    assert cohort_keys(payload, "train", cohort) == ["t1", "t3"]
    with pytest.raises(ValueError, match="not in the executor's val split"):
        cohort_keys(payload, "val", DevelopmentCohort(name="b", train_flight_ids=("t1",), val_flight_ids=("v9",), selection={}))


# ── a held token (two-tier v3 stage B, D38) ──────────────────────────────────

SPAN_S = 60.0


def _coded_world(tmp_path_factory, *, name: str, horizon_s: float, n_segments: int, token_s: float, token_step_s: float):
    """A coded executor whose token span is 60 s: S60-held (horizon 20, S 60) or S60-h60 (horizon 60, step 20)."""
    torch.manual_seed(0)
    codebook = tok.write_codebook(tmp_path_factory.mktemp(name) / "cb", tok.tokenizer_for("learned", levels=(4, 4), segment_s=SPAN_S, dt_s=DT_S),
                                  segment_s=SPAN_S, dt_s=DT_S, data_identity=IDENTITY, source={})
    settings = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False)
    settings.update(dict(
        prediction_output=PREDICTION_CONTROL, plan_conditioning="manoeuvre-code", manoeuvre_fsq_levels=(4, 4),
        manoeuvre_codebook=str(codebook.path), control_horizon_s=horizon_s, n_segments=n_segments,
        manoeuvre_token_s=token_s, manoeuvre_token_step_s=token_step_s, random_train_anchor=True, random_train_anchor_min_future_s=horizon_s,
        control_imitation_loss_weight=0.0, final_time_loss_weight=0.0, state_endpoint_loss_weight=0.0, seq_len=8,
        d_model=16, n_heads=4, d_ff=32, e_layers=1, dropout=0.0, device="cpu", epochs=1, patience=1,
    ))
    config = TSConfig(**settings)
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=3, seed=3), config, airport=AIRPORT)
    model = build_model(config).eval()
    with torch.no_grad():
        model.control_head.control_projection.weight.normal_(std=0.05)
    executor = ls.Executor(model=model, config=config, normalizer=Normalizer.fit(series))
    vocabulary = TypeVocabulary.from_typecodes(str(item.scenario.aircraft.code) for item in series)
    prior = ls.Prior(model=ManoeuvrePrior(PriorConfig(code_count=16, z_dim=2, type_count=vocabulary.size, d_model=32, n_heads=4,
                                                       n_layers=1, d_ff=64, dropout=0.0, max_positions=64)).eval(), vocabulary=vocabulary)
    return codebook, executor, series, prior


@pytest.fixture(scope="module")
def held_world(tmp_path_factory):
    return _coded_world(tmp_path_factory, name="held", horizon_s=20.0, n_segments=2, token_s=SPAN_S, token_step_s=0.0)


@pytest.fixture(scope="module")
def h60_world(tmp_path_factory):
    return _coded_world(tmp_path_factory, name="h60", horizon_s=60.0, n_segments=6, token_s=0.0, token_step_s=20.0)


def _check_held(runs, hold: int = 3):
    for run in runs:
        assert run.truth.segment_s == SPAN_S and run.predictions == len(run.legs) == len(run.codes) == len(run.rounds)
        for k, (code, record) in enumerate(zip(run.codes, run.rounds, strict=True)):
            assert (record["token_index"], record["phase"]) == divmod(k, hold)
        assert run.token_refreshes == (run.predictions + hold - 1) // hold
        for leg in run.legs[:-1]:
            assert float(np.sum(leg.sample_durations_s)) == pytest.approx(20.0)      # the step, never the span
        # spans tokenised back: one per completed 60 s of the flown polyline, the boundary states beside
        t0 = float(run.series.times[run.anchor])
        times, _values = ls._polyline(run)
        complete = int(np.floor((float(times[-1]) - t0 + 1e-6) / SPAN_S))
        assert len(run.flown_codes) == complete and len(run.flown_states) == complete + 1
        row, _metrics = ls.flight_row(run, points=16)
        assert row["token_refreshes"] == run.token_refreshes and row["rounds"][0]["phase"] == 0


def test_a_held_token_is_flown_for_three_rounds_and_the_truth_codes_are_read_per_span(held_world):
    """S60-held (D38): the truth's codes are spaced one span apart, round k flies code k // 3 at
    phase k % 3, past the truth's last full span the last code is held, and the flown history is
    tokenised once per completed span — the executor's rounds stay 20 s."""
    codebook, executor, series, _prior = held_world
    from ts_transformer.config import token_hold
    assert token_hold(executor.config) == 3 and ls.round_step_s(executor.config, ls.PROTOCOL_C, None) == 20.0
    runs = ls.fly(executor, codebook, series, ls.PROTOCOL_C, prior=None, device=torch.device("cpu"), batch_size=2)
    _check_held(runs)
    for run in runs:
        assert run.predictions >= 4                                                   # more than one span
        assert all(code == run.truth.codes[min(k // 3, run.truth.length - 1)] for k, code in enumerate(run.codes))
        assert run.held_predictions == sum(1 for k in range(run.predictions) if k // 3 >= run.truth.length)
        assert all("e_plan_m" not in record for record in run.rounds)
    with pytest.raises(ValueError, match="no-token reading"):
        ls.fly(executor, codebook, series, ls.PROTOCOL_C, prior=None, device=torch.device("cpu"), batch_size=2, execute_s=20.0)
    with pytest.raises(ValueError, match="only the prior"):
        ls.fly(executor, codebook, series, ls.PROTOCOL_C, prior=None, device=torch.device("cpu"), batch_size=2, landed_ends_flight=True)


def test_the_prior_is_asked_at_each_refresh_and_its_landing_only_records_a_time_unless_asked(held_world, monkeypatch):
    """Protocol A under a held token: the prior answers once per span (a refresh), its code is
    flown for the span's three rounds, e_plan is read every round; its landing (D37) is recorded
    on the first refresh that says so and never ends the flight — unless ``landed_ends_flight``,
    the 09-18 rule, under which the flight ends at that fraction of the span."""
    from ts_transformer.data.dataset import truth_duration_s
    codebook, executor, series, prior = held_world
    runs = ls.fly(executor, codebook, series, ls.PROTOCOL_A, prior=prior, device=torch.device("cpu"), batch_size=2)
    _check_held(runs)
    for run in runs:
        assert len(run.landed_probabilities) == run.token_refreshes and run.ended in (ls.ENDED_CROSSED, ls.ENDED_HORIZON)
        assert all("e_plan_m" in record and "truth_code" in record for record in run.rounds)
        assert all(run.rounds[k]["code"] == run.rounds[k - 1]["code"] for k in range(1, run.predictions) if k % 3)   # held inside a span
    monkeypatch.setattr(ls, "LANDED_THRESHOLD", -1.0)                                   # every refresh says "lands within this span"
    runs = ls.fly(executor, codebook, series, ls.PROTOCOL_A, prior=prior, device=torch.device("cpu"), batch_size=2)
    for run in runs:
        assert run.prior_landed_at_s is not None and 0.0 <= run.prior_landed_at_s <= SPAN_S and run.ended != ls.ENDED_LANDED
        row, _metrics = ls.flight_row(run, points=16)
        assert row["prior_landed_error_s"] == pytest.approx(run.prior_landed_at_s - truth_duration_s(run.series, run.anchor))
    runs = ls.fly(executor, codebook, series, ls.PROTOCOL_A, prior=prior, device=torch.device("cpu"), batch_size=2, landed_ends_flight=True)
    for run in runs:
        assert run.ended in (ls.ENDED_LANDED, ls.ENDED_CROSSED)
        if run.ended == ls.ENDED_LANDED:
            assert run.landed_fraction == run.landing_fraction and run.flown_s <= run.landing_fraction * SPAN_S + 1e-6
            assert run.predictions <= 3


def test_a_truth_prefix_runs_out_only_at_a_refresh(held_world):
    codebook, executor, series, prior = held_world
    runs = ls.fly(executor, codebook, series, ls.PROTOCOL_A_TRUTH, prior=prior, device=torch.device("cpu"), batch_size=2)
    _check_held(runs)
    for run in runs:
        if run.ended == ls.ENDED_TRUTH_EXHAUSTED:
            assert run.predictions == 3 * (run.truth.length + 1)


def test_the_h60_configuration_flies_twenty_seconds_of_each_sixty_second_forecast(h60_world):
    """S60-h60: the token span IS the horizon (60 s, spelled 0), the step 20 s — every round flies
    the first 20 s of a 60 s forecast under the span's token, three rounds per token."""
    codebook, executor, series, prior = h60_world
    from ts_transformer.config import token_hold, token_span_s
    assert token_span_s(executor.config) == SPAN_S and token_hold(executor.config) == 3
    assert ls.round_step_s(executor.config, ls.PROTOCOL_A, None) == 20.0 and ls.round_step_s(executor.config, ls.PROTOCOL_NONE, 20.0) == 20.0
    runs = ls.fly(executor, codebook, series, ls.PROTOCOL_C, prior=None, device=torch.device("cpu"), batch_size=2)
    _check_held(runs)
    for run in runs:
        assert all(code == run.truth.codes[min(k // 3, run.truth.length - 1)] for k, code in enumerate(run.codes))
    runs = ls.fly(executor, codebook, series, ls.PROTOCOL_A, prior=prior, device=torch.device("cpu"), batch_size=2)
    _check_held(runs)
