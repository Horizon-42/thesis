"""Lockstep (`manoeuvre/lockstep.py`): the no-token closed loop predicts each flight once per
round on its own flown rows and ends by crossing or by the budget; the flown legs are
contiguous; the row carries the reference verdicts and the leads. Untrained models: the
mechanics are what is pinned, never a number.

The intent-code protocols C / A / A-truth and their tests were ARCHIVED 2026-09-20
(`archive/manoeuvre_codes_2026_09/tests/test_manoeuvre_lockstep.py` is this file's original,
unmodified).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.backbone.adapters import build_model
from ts_transformer.config import CONTROL_RECIPE_SIMPLE_V3, PREDICTION_CONTROL, TSConfig, recipe_settings
from ts_transformer.data.dataset import Normalizer, build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.manoeuvre import lockstep as ls
from ts_transformer.tests.support import AIRPORT, RUNWAY

SEGMENT_S, DT_S = 20.0, 2.0


def _config() -> TSConfig:
    settings = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False)
    settings.update(dict(
        prediction_output=PREDICTION_CONTROL, control_horizon_s=SEGMENT_S, n_segments=2, control_imitation_loss_weight=0.0,
        final_time_loss_weight=0.0, state_endpoint_loss_weight=0.0, seq_len=8, d_model=16, n_heads=4, d_ff=32,
        e_layers=1, dropout=0.0, device="cpu", epochs=1, patience=1,
    ))
    return TSConfig(**settings)


@pytest.fixture(scope="module")
def world():
    config = _config()
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=3, seed=3), config, airport=AIRPORT)
    torch.manual_seed(1)
    model = build_model(config).eval()
    with torch.no_grad():
        model.control_head.control_projection.weight.normal_(std=0.05)
    return ls.Executor(model=model, config=config, normalizer=Normalizer.fit(series)), series


def test_the_closing_budget_is_the_archived_rule():
    assert ls.closing_horizon_s(100.0) == 130.0 and ls.closing_horizon_s(400.0) == 440.0


def _check_runs(runs):
    for run in runs:
        assert run.ended in (ls.ENDED_CROSSED, ls.ENDED_HORIZON)
        assert run.predictions == len(run.legs) == len(run.rounds)
        if not run.legs:
            continue
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
        assert row["ended"] == run.ended and set(row["at"]) == {"60", "120", "180", "300"}
        assert row["established_at_anchor"] in (True, False) and metrics["ade_m"] >= 0.0
        assert (row["reference"]["established"]) == (run.ended == ls.ENDED_CROSSED)
        # a no-token row carries no code columns at all: nothing labelled anything
        assert not {"codes", "truth_codes", "flown_codes", "truth_length", "flown_states",
                    "token_refreshes", "held_predictions", "prior_landed_at_s"} & set(row)
        assert all(set(record) == {"round", "lead_s", "e_track_m"} for record in run.rounds)


def test_the_closed_loop_flies_the_executor_on_its_own_rows_without_a_code(world):
    """Two-tier v3 stage A: the executor predicts again every round on the rows it just flew,
    under one budget, with no second layer — the first prediction at L−1 (v3 D2: no floor)."""
    executor, series = world
    lines = []
    runs = ls.fly(executor, series, device=torch.device("cpu"), batch_size=2, log=lines.append)
    assert len(runs) == 3
    _check_runs(runs)
    assert lines and lines[0].startswith("    none round 0:")
    for run in runs:
        assert run.anchor == executor.config.seq_len - 1 == 7


def test_executing_a_prefix_of_each_forecast_is_a_shorter_round(world, monkeypatch):
    """A3-a (v3 §5.3, D43): the executor forecasts its whole horizon every round but only the
    first ``execute_s`` is flown and the next prediction is made there — shorter rounds, the same
    budget, more predictions. Refused for a step that is not a whole number of rows inside the
    horizon, or not a whole number of integrator steps."""
    from dataclasses import replace
    from types import SimpleNamespace

    executor, series = world
    half = SEGMENT_S / 2
    whole_runs = ls.fly(executor, series, device=torch.device("cpu"), batch_size=3)
    same_runs = ls.fly(executor, series, device=torch.device("cpu"), batch_size=3, execute_s=SEGMENT_S)
    runs = ls.fly(executor, series, device=torch.device("cpu"), batch_size=3, execute_s=half)
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
            ls.fly(executor, series, device=torch.device("cpu"), batch_size=3, execute_s=bad)
    # a step that is a whole number of rows but not of integrator steps: the leg could not be cut on the rollout grid
    with pytest.raises(ValueError, match="integrator steps"):
        ls.executed_step_s(SimpleNamespace(control_horizon_s=20.0, dt_s=2.0, control_rollout_integrator_dt_s=3.0), 4.0)
    # a crossing the forecast predicts AFTER the executed prefix is not flown: the flight goes on and predicts again
    def crosses_at_15_s(forecast, history):
        rows = int(np.searchsorted(np.cumsum(forecast.sample_durations_s), 15.0 + 1e-6, side="right"))
        return replace(ls.cut_rows(forecast, rows), truncated_at_threshold=True, final_time_s=15.0)

    monkeypatch.setattr(ls, "cut_at_threshold_crossing", crosses_at_15_s)
    (whole_run,) = ls.fly(executor, series[:1], device=torch.device("cpu"), batch_size=1)
    (prefix_run,) = ls.fly(executor, series[:1], device=torch.device("cpu"), batch_size=1, execute_s=half)
    assert whole_run.ended == ls.ENDED_CROSSED and len(whole_run.legs) == 1
    assert float(np.sum(whole_run.legs[0].sample_durations_s)) == pytest.approx(15.0)
    assert prefix_run.ended == ls.ENDED_HORIZON and len(prefix_run.legs) > 1
    assert all(float(np.sum(leg.sample_durations_s)) == pytest.approx(half) for leg in prefix_run.legs[:-1])


def test_the_remaining_path_reading_cuts_each_flight_at_its_bin_row(world):
    """v3 §3.1's second reading (D3): the cohort first seen where each flight has X km of path
    left — the bin's row becomes the cut flight's L−1, the truth-side readings move with it, and
    a flight with no admissible row at the bin is absent, not flown from elsewhere."""
    from ts_transformer.data.approach_difficulty import remaining_path_profile_m

    executor, series = world
    config = executor.config
    cut, first_rows = ls.from_remaining_path(series, config, 12_000.0)
    assert len(cut) == len(first_rows) == 3
    for item, whole in zip(cut, series, strict=True):
        row = first_rows[whole.dataset_id]
        assert row >= config.seq_len - 1 and item.n_samples == whole.n_samples - (row - config.seq_len + 1)
        # the cut flight's L−1 IS the whole flight's bin row: same time, same remaining path
        assert item.times[config.seq_len - 1] == whole.times[row]     # no floor: the fixed anchor is L−1
        assert remaining_path_profile_m(item)[config.seq_len - 1] == pytest.approx(remaining_path_profile_m(whole)[row])
        assert abs(remaining_path_profile_m(whole)[row] - 12_000.0) == pytest.approx(np.abs(remaining_path_profile_m(whole) - 12_000.0).min())
    runs = ls.fly(executor, cut, device=torch.device("cpu"), batch_size=3)
    _check_runs(runs)
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

    executor, series = world
    config = executor.config
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
    runs = ls.fly(executor, cut, device=torch.device("cpu"), batch_size=3)
    _check_runs(runs)
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
    no round record — the bookkeeping stays aligned (review 2026-09-18 M4)."""
    executor, series = world
    runs = ls.fly(executor, series[:1], device=torch.device("cpu"), batch_size=1)
    run = runs[0]
    forecast = run.legs[0]
    run.ended, run.flown_s = None, run.horizon_s - 0.1          # 0.1 s of budget left: under one 0.5 s step
    before = (len(run.legs), len(run.rounds), run.predictions)
    leg = ls._fly_leg(run, run.series, forecast, step_s=SEGMENT_S, round_index=99)
    assert leg is None and run.ended == ls.ENDED_HORIZON
    assert (len(run.legs), len(run.rounds), run.predictions) == before


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
