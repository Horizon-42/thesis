"""`run_ts.py chain_sensitivity` (two-tier feasibility T0(b), design §10.1).

The chain is only a measurement of the model if a re-ask on the ROLLED series is the model's
ordinary predict path on that history — so the load-bearing test here re-asks on a rolled
series built from the observed track itself and requires the re-anchored forecast back.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ts_transformer.experiments.anytime_curve as anytime
import ts_transformer.experiments.chain_sensitivity as runner
from ts_transformer.config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    PREDICTION_CONTROL,
    TSConfig,
    default_anchor,
)
from ts_transformer.data.dataset import build_series, dataset_flight_key, window_anchors
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.forecast import Forecast, concatenate, cut_rows, forecast_approaches
from ts_transformer.training.train import load_checkpoint, train
from ts_transformer.tests.support import fake_data_provenance

AIRPORT, RUNWAY = "KRDU", "05L"
STEP_S = 4.0


def _config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_rollout_integrator_dt_s=0.5,
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=40.0, device="cpu", horizon_mode="normalized",
        epochs=1, patience=1, batch_size=8, dropout=0.0,
        val_fraction=0.25, test_fraction=0.25,
    )
    settings.update(overrides)
    return TSConfig(**settings)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    torch.manual_seed(0)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    config = _config()
    series, _report = build_series(flights, config, airport=AIRPORT)
    out = tmp_path_factory.mktemp("chain_run")
    train(series, config, output_dir=out, data_provenance=fake_data_provenance(), verbose=False)
    model, loaded_config, normalizer, payload = load_checkpoint(out / "checkpoint.pt")
    arm = anytime.Arm(label="tiny", path=out / "checkpoint.pt", model=model, config=loaded_config,
                      normalizer=normalizer, payload=payload, airports=(AIRPORT,),
                      manifests=[out / "manifest.json"])
    return flights, series, arm


def _truth_forecast(series, anchor: int, config: TSConfig) -> Forecast:
    """The observed track after the anchor dressed as a forecast (one segment per sample)."""
    origin = float(series.times[anchor])
    offsets = np.asarray(series.times[anchor + 1 :], dtype=np.float64) - origin
    durations = np.diff(np.concatenate(([0.0], offsets)))
    return Forecast(
        times=origin + offsets, values=np.asarray(series.values[anchor + 1 :], dtype=np.float64),
        normalized_progress=offsets / offsets[-1], anchor=anchor, final_time_s=float(offsets[-1]),
        predicted_final_time_s=float(offsets[-1]), horizon_mode=config.horizon_mode, passes=1,
        truncated_at_threshold=False, horizon_capped=False, sample_durations_s=durations,
        segment_durations_s=durations, controls=np.zeros((len(offsets), 3)),
        geodetic_values=np.zeros((len(offsets), 7)), prediction_output=PREDICTION_CONTROL,
    )


# ── the rolled series ─────────────────────────────────────────────────────────

def test_a_rolled_series_built_from_the_truth_is_the_observed_series(trained) -> None:
    _flights, series, arm = trained
    item = series[0]
    a0 = default_anchor(arm.config)
    until = a0 + 5
    rolled = runner.rolled_series(item, a0, _truth_forecast(item, a0, arm.config), until, arm.config.dt_s)
    assert len(rolled.times) == until + 1
    assert rolled.times == pytest.approx(item.times[: until + 1])
    assert rolled.values == pytest.approx(item.values[: until + 1])
    # a rolled series has no truth of its own
    assert rolled.supervision_times is rolled.times


def test_a_reask_on_that_rolled_series_is_the_reanchored_forecast(trained) -> None:
    """The instrument's whole claim: the chain's re-ask IS the model's predict path on the
    history it is given — window, anchor state and the inverted actuator state alike."""
    _flights, series, arm = trained
    a0 = default_anchor(arm.config)
    until = a0 + 5
    items = [item for item in series if until in window_anchors(item, arm.config)][:3]
    assert items
    rolled = [runner.rolled_series(item, a0, _truth_forecast(item, a0, arm.config), until, arm.config.dt_s)
              for item in items]
    device = torch.device("cpu")
    on_rolled = forecast_approaches(arm.model, rolled, arm.config, arm.normalizer, anchor=until, device=device)
    on_truth = forecast_approaches(arm.model, items, arm.config, arm.normalizer, anchor=until, device=device)
    for a, b in zip(on_rolled, on_truth, strict=True):
        assert a.times == pytest.approx(b.times)
        assert a.values == pytest.approx(b.values, abs=1e-6)
        assert a.controls == pytest.approx(b.controls, abs=1e-3)


def test_a_rolled_series_carries_the_flown_rows_not_the_truth(trained) -> None:
    """The review's mutation (2026-09-16): a `rolled_series` that ignored ``flown`` and returned
    the observed rows passed every truth-built test, and the chain would silently have been
    protocol B. Flown rows displaced from the truth must reach the history AND the re-ask."""
    _flights, series, arm = trained
    a0 = default_anchor(arm.config)
    until = a0 + 5
    items = [item for item in series if until in window_anchors(item, arm.config)][:2]
    assert items
    shift = np.zeros(6)
    shift[0], shift[1] = 300.0, -200.0
    device = torch.device("cpu")
    for item in items:
        truth = _truth_forecast(item, a0, arm.config)
        flown = replace(truth, values=truth.values + shift)
        rolled = runner.rolled_series(item, a0, flown, until, arm.config.dt_s)
        assert rolled.values[: a0 + 1] == pytest.approx(item.values[: a0 + 1])
        assert rolled.values[a0 + 1 : until + 1] == pytest.approx(item.values[a0 + 1 : until + 1] + shift)
        by_hand = replace(item, times=item.times[: until + 1].copy(),
                          values=np.concatenate((item.values[: a0 + 1], item.values[a0 + 1 : until + 1] + shift)),
                          supervision_times=None, supervision_values=None, supervision_weights=None)
        ours = forecast_approaches(arm.model, [rolled], arm.config, arm.normalizer, anchor=until, device=device)[0]
        theirs = forecast_approaches(arm.model, [by_hand], arm.config, arm.normalizer, anchor=until, device=device)[0]
        truth_reask = forecast_approaches(arm.model, [item], arm.config, arm.normalizer, anchor=until, device=device)[0]
        assert ours.values == pytest.approx(theirs.values, abs=1e-6)
        # and the displaced history really moved the forecast off the truth-history one
        assert not np.allclose(ours.values[:, :2], truth_reask.values[:, :2], atol=1.0)


def test_a_rolled_series_refuses_flown_rows_that_end_short(trained) -> None:
    _flights, series, arm = trained
    item = series[0]
    a0 = default_anchor(arm.config)
    short = cut_rows(_truth_forecast(item, a0, arm.config), 2)
    with pytest.raises(ValueError, match="flown rows end"):
        runner.rolled_series(item, a0, short, a0 + 5, arm.config.dt_s)


# ── cutting and joining hook-free forecasts ───────────────────────────────────

def test_cut_at_lead_lands_on_the_row_and_keeps_the_clocks_aligned(trained) -> None:
    _flights, series, arm = trained
    a0 = default_anchor(arm.config)
    forecast = forecast_approaches(arm.model, series[:1], arm.config, arm.normalizer, anchor=a0,
                                   device=torch.device("cpu"))[0]
    assert forecast.command_hook_diagnostics is None
    cut = runner.cut_at_lead(forecast, STEP_S)
    assert cut.final_time_s == pytest.approx(STEP_S)
    assert float(np.sum(cut.segment_durations_s)) == pytest.approx(STEP_S)
    assert len(cut.controls) == len(cut.segment_durations_s)
    assert cut.command_hook_diagnostics is None
    with pytest.raises(ValueError, match="no rollout row"):
        runner.cut_at_lead(forecast, STEP_S + 0.25)


def test_hook_free_legs_concatenate_into_one_record_clock(trained) -> None:
    _flights, series, arm = trained
    a0 = default_anchor(arm.config)
    forecast = forecast_approaches(arm.model, series[:1], arm.config, arm.normalizer, anchor=a0,
                                   device=torch.device("cpu"))[0]
    first = runner.cut_at_lead(forecast, STEP_S)
    second = replace(forecast, times=forecast.times + STEP_S)       # the next link, asked at the cut
    whole = concatenate([first, second], a0, 99.0)
    assert np.all(np.diff(whole.times) > 0.0)
    assert len(whole.times) == len(first.times) + len(second.times)
    assert whole.final_time_s == pytest.approx(STEP_S + forecast.final_time_s)
    assert whole.predicted_final_time_s == 99.0
    assert whole.command_hook_diagnostics is None
    assert len(whole.controls) == len(whole.segment_durations_s)


def test_the_chain_links_are_independent_reasks_on_the_flown_rows(trained) -> None:
    """Link 1 and link 2 of `chain_forecasts`, rebuilt by hand from the pieces."""
    _flights, series, arm = trained
    a0 = default_anchor(arm.config)
    device = torch.device("cpu")
    plan = runner.ChainPlan(split="train", step_s=STEP_S, links=2, limit=0, batch_size=4, write_records=False)
    step_rows = int(round(STEP_S / arm.config.dt_s))
    one = forecast_approaches(arm.model, series, arm.config, arm.normalizer, anchor=a0, device=device)
    chains, held = runner.chain_forecasts(arm, series, one, plan, device, 4)
    checked = 0
    for item, first, chain, links in zip(series, one, chains, held, strict=True):
        if links < 3:
            continue
        leg0 = runner.cut_at_lead(first, STEP_S)
        leg1_full = forecast_approaches(
            arm.model, [runner.rolled_series(item, a0, leg0, a0 + step_rows, arm.config.dt_s)],
            arm.config, arm.normalizer, anchor=a0 + step_rows, device=device)[0]
        leg1 = runner.cut_at_lead(leg1_full, STEP_S)
        leg2 = forecast_approaches(
            arm.model, [runner.rolled_series(item, a0, concatenate([leg0, leg1], a0, 0.0), a0 + 2 * step_rows, arm.config.dt_s)],
            arm.config, arm.normalizer, anchor=a0 + 2 * step_rows, device=device)[0]
        expected = concatenate([leg0, leg1, leg2], a0, 2 * STEP_S + leg2.predicted_final_time_s)
        assert chain.times == pytest.approx(expected.times)
        assert chain.values == pytest.approx(expected.values, abs=1e-6)
        assert chain.passes == 3
        assert chain.predicted_final_time_s == pytest.approx(expected.predicted_final_time_s)
        checked += 1
    assert checked


def test_epsilon_at_a_later_link_is_the_observed_reask_at_its_anchor(trained) -> None:
    _flights, series, arm = trained
    a0 = default_anchor(arm.config)
    device = torch.device("cpu")
    plan = runner.ChainPlan(split="train", step_s=STEP_S, links=2, limit=0, batch_size=4, write_records=False)
    step_rows = int(round(STEP_S / arm.config.dt_s))
    one = forecast_approaches(arm.model, series, arm.config, arm.normalizer, anchor=a0, device=device)
    eps = runner.reanchored_errors(arm, series, one, plan, device, 4)
    checked = 0
    for item, first, row in zip(series, one, eps, strict=True):
        assert row.get(1) == runner.displacement_at(item, first, a0, float(item.times[a0]) + STEP_S)
        anchor = a0 + 2 * step_rows
        if 3 not in row:
            continue
        forecast = forecast_approaches(arm.model, [item], arm.config, arm.normalizer, anchor=anchor, device=device)[0]
        assert row[3] == pytest.approx(runner.displacement_at(item, forecast, anchor, float(item.times[anchor]) + STEP_S))
        checked += 1
    assert checked


# ── the estimator ─────────────────────────────────────────────────────────────

def test_the_l_estimates_recover_a_known_contraction() -> None:
    plan = runner.ChainPlan(split="val", step_s=60.0, links=2, limit=0, batch_size=None, write_records=False)
    rows = {}
    rng = np.random.default_rng(0)
    for index in range(20):
        eps = float(rng.uniform(50, 150))
        e1 = float(rng.uniform(100, 400))
        e2 = eps + 0.8 * e1
        e3 = eps + 0.8 * e2
        rows[f"f{index}"] = {"links": 3, "at": {
            "1": {"e_one_m": e1, "e_chain_m": e1, "epsilon_m": e1},
            "2": {"e_one_m": 2 * e1, "e_chain_m": e2, "epsilon_m": eps},
            "3": {"e_one_m": None, "e_chain_m": e3, "epsilon_m": eps},
        }}
    summary = runner.stratum_summary(rows, sorted(rows), plan)
    assert "L" not in summary["1"]
    for k in ("2", "3"):
        assert summary[k]["L"]["n"] == 20
        assert summary[k]["L"]["paired_median"] == pytest.approx(0.8)
        assert summary[k]["L"]["least_squares_through_origin"] == pytest.approx(0.8)
    # a lead absent from the one-shot is absent from the ratio, never scored
    assert summary["3"]["e_one"]["n"] == 0
    assert summary["3"]["chain_over_one_n"] == 0


def test_displacement_is_absent_past_either_end(trained) -> None:
    _flights, series, arm = trained
    item = series[0]
    a0 = default_anchor(arm.config)
    truth = _truth_forecast(item, a0, arm.config)
    assert runner.displacement_at(item, truth, a0, float(item.times[a0]) + 4.0) == pytest.approx(0.0)
    assert runner.displacement_at(item, truth, a0, float(item.times[-1]) + 1.0) is None
    short = cut_rows(truth, 2)
    assert runner.displacement_at(item, short, a0, float(short.times[-1]) + 1.0) is None


# ── the command line ──────────────────────────────────────────────────────────

def _patch_data_plane(monkeypatch, flights, tmp_path):
    manifest = tmp_path / "manifest.json"
    indexed = {dataset_flight_key(flight, index): flight for index, flight in enumerate(flights)}
    monkeypatch.setattr(anytime.pipeline, "arrival_manifest_path", lambda _airport: manifest)
    monkeypatch.setattr(anytime, "checkpoint_data_provenance", lambda _payload, _manifests: fake_data_provenance())
    monkeypatch.setattr(
        anytime, "load_flight_dicts",
        lambda _paths, include_flight_keys, verbose=True: [
            flight for key, flight in indexed.items() if key in include_flight_keys
        ],
    )


def test_the_readout_runs_end_to_end_and_writes_both_record_sets(monkeypatch, tmp_path, trained) -> None:
    flights, _series, arm = trained
    _patch_data_plane(monkeypatch, flights, tmp_path)
    out = tmp_path / "chain"
    assert runner.main([
        "--checkpoint", f"tiny={arm.path}", "--out", str(out), "--device", "cpu", "--split", "train",
        "--step-s", str(STEP_S), "--links", "3", "--write-records", "--batch-size", "4",
    ]) == 0
    payload = json.loads((out / "chain_sensitivity.json").read_text())
    assert payload["schema"] == runner.RESULT_SCHEMA
    block = payload["checkpoints"]["tiny"]
    assert block["anchor_policy"] == anytime.ARM_FIXED
    rows = block["flights_rows"]
    assert rows
    # the chain really re-asked: some flight holds more than its first link
    assert any(int(n) > 1 for n in block["links_held"])
    for row in rows.values():
        first = row["at"]["1"]
        # link 0 IS the one-shot cut at the step, and eps_1 is the one-shot at the step
        if first["e_one_m"] is not None:
            assert first["e_chain_m"] == pytest.approx(first["e_one_m"])
            assert first["epsilon_m"] == pytest.approx(first["e_one_m"])
    for variant in ("oneshot", f"chain_{STEP_S:g}s"):
        summary = json.loads((out / "records" / "tiny" / variant / "summary.json").read_text())
        assert summary["chain"]["variant"] == variant
        assert summary["chain"]["records"] == len(rows)
    assert (out / "chain_sensitivity.txt").read_text().startswith("Chain sensitivity")


@pytest.mark.parametrize("argv, message", [
    (["--split", "test"], "sealed"),
    (["--links", "0"], "at least 1"),
    (["--step-s", "-5"], "positive"),
])
def test_the_command_line_is_refused_before_anything_is_created(tmp_path, capsys, trained, argv, message) -> None:
    _flights, _series, arm = trained
    with pytest.raises(SystemExit):
        runner.main(["--checkpoint", f"tiny={arm.path}", "--out", str(tmp_path / "never"), "--device", "cpu", *argv])
    assert message in capsys.readouterr().err
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("overrides, message", [
    ({"prediction_output": "state"}, "CONTROL"),
    ({"control_command_hook": "barrier"}, "hooked"),
    ({"latent_dim": 8}, "latent"),
    ({"duration_head": "quantile"}, "point head"),
])
def test_the_chain_is_refused_off_the_deterministic_hook_free_control_path(overrides, message) -> None:
    config = SimpleNamespace(**{
        "prediction_output": PREDICTION_CONTROL, "control_command_hook": "off", "latent_dim": 0,
        "duration_head": "point", **overrides,
    })
    with pytest.raises(SystemExit, match=message):
        runner.check_arm(SimpleNamespace(label="x", path="x", config=config))


def test_a_step_off_the_integrator_step_is_refused_even_on_the_series_grid(capsys) -> None:
    arm = SimpleNamespace(label="x", config=SimpleNamespace(dt_s=2.0, control_rollout_integrator_dt_s=1.5))
    plan = runner.ChainPlan(split="val", step_s=4.0, links=1, limit=0, batch_size=None, write_records=False)
    with pytest.raises(SystemExit):
        runner.check_step(runner.build_parser(), arm, plan)
    assert "control_rollout_integrator_dt_s" in capsys.readouterr().err


def test_a_step_off_the_integrator_grid_is_refused(monkeypatch, tmp_path, capsys, trained) -> None:
    flights, _series, arm = trained
    _patch_data_plane(monkeypatch, flights, tmp_path)
    with pytest.raises(SystemExit):
        runner.main(["--checkpoint", f"tiny={arm.path}", "--out", str(tmp_path / "never"), "--device", "cpu",
                     "--split", "train", "--step-s", "3"])
    assert "whole multiple" in capsys.readouterr().err
    assert not (tmp_path / "never").exists()
