"""`run_ts.py tracker_lockstep` (two-tier T1, protocol C, design §10.3).

The runner's first ask must be the checkpoint's own predict path — the same truth plan token and
arrival time the training rows and `predict` hand it — and every later ask a re-ask on the flown
rows with the truth expert read at the flown pose.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ts_transformer.experiments.anytime_curve as anytime
import ts_transformer.experiments.tracker_lockstep as runner
import ts_transformer.experiments.two_tier_gates as gates
from ts_transformer.experiments.plan_oracle import closing_horizon_s
from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    CTA_CONDITIONING_GIVEN,
    PLAN_CONDITIONING_TRUTH_NEXT,
    PREDICTION_CONTROL,
    TSConfig,
    default_anchor,
)
from ts_transformer.data.approach_difficulty import STRATUM_ALL
from ts_transformer.inference.forecast import Forecast
from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows, build_series, dataset_flight_key, truth_duration_s
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.forecast import forecast_approaches
from ts_transformer.outputs.control.plan_token import PLAN_TOKEN_KEY
from ts_transformer.training.train import load_checkpoint, train
from ts_transformer.tests.support import AIRPORT, RUNWAY, fake_data_provenance

pytestmark = pytest.mark.skipif(
    not (DEFAULT_PROCEDURE_ROOT / AIRPORT / "procedure-details").is_dir(),
    reason="the KRDU procedure documents are not on this machine",
)

STEP_S = 20.0


def _config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        cta_conditioning=CTA_CONDITIONING_GIVEN,
        plan_conditioning=PLAN_CONDITIONING_TRUTH_NEXT,
        plan_conditioning_dropout=0.5,
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1, final_time_scale_s=100.0,
        device="cpu", horizon_mode="normalized", epochs=1, patience=1, batch_size=8, dropout=0.0,
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
    out = tmp_path_factory.mktemp("tracker_run")
    train(series, config, output_dir=out, data_provenance=fake_data_provenance(), verbose=False)
    model, loaded, normalizer, payload = load_checkpoint(out / "checkpoint.pt")
    arm = anytime.Arm(label="t1", path=out / "checkpoint.pt", model=model, config=loaded,
                      normalizer=normalizer, payload=payload, airports=(AIRPORT,), manifests=[out / "manifest.json"])
    return flights, series, arm


def _plan(**overrides) -> runner.LockstepPlan:
    settings = dict(split="train", step_s=STEP_S, variants=runner.VARIANTS, limit=0,
                    batch_size=4, write_records=False)
    settings.update(overrides)
    return runner.LockstepPlan(**settings)


def test_the_first_ask_is_the_checkpoints_own_predict_path(trained) -> None:
    """At a0 the runner's row (truth expert) must equal what training and `predict` hand the
    checkpoint: the plan head's label as the token, the truth duration as the CTA."""
    _flights, series, arm = trained
    a0 = default_anchor(arm.config)
    windows = FixedAnchorTrajectoryWindows(series, arm.config, arm.normalizer)
    runs = runner.fly_variant(arm, series, runner.VARIANT_ONE_SHOT, _plan(), torch.device("cpu"), 4)
    one_shot = forecast_approaches(arm.model, series, arm.config, arm.normalizer, device=torch.device("cpu"))
    for i, (run, expected) in enumerate(zip(runs, one_shot, strict=True)):
        fresh = runner.FlightRun(series=run.series, skeleton=run.skeleton)
        plan_at = runner.TruthPlans([fresh], a0).at_ask([fresh], [run.series], a0, first=True)[0]
        row = runner.ask_row(fresh, run.series, a0, arm.config, plan_at, with_plan=True,
                              floor_s=runner.ask_floor_s(arm.config, STEP_S))
        training = windows.context.row(i)
        assert row[PLAN_TOKEN_KEY] == pytest.approx(training[PLAN_TOKEN_KEY])
        assert float(row["cta_s"]) == pytest.approx(float(training["cta_s"]))
        # the one ask is flown to its last whole step (or its crossing), then closed by re-asks
        first = run.legs[0]
        assert first.values == pytest.approx(expected.values[: len(first.values)], abs=1e-6)
        assert first.truncated_at_threshold or first.final_time_s == pytest.approx(
            STEP_S * int((expected.final_time_s + 1e-6) // STEP_S))


def test_receding_flights_fly_one_step_per_ask_and_end_by_a_stated_rule(trained) -> None:
    _flights, series, arm = trained
    runs = runner.fly_variant(arm, series, runner.VARIANT_RECEDING, _plan(), torch.device("cpu"), 4)
    assert any(run.asks > 1 for run in runs)
    for run in runs:
        # a given CTA is never below one step, so no flight ends as a short forecast
        assert run.ended in (runner.ENDED_CROSSED, runner.ENDED_HORIZON)
        for leg in run.legs[:-1]:
            assert leg.final_time_s == pytest.approx(STEP_S)
        whole = runner.whole_forecast(run, default_anchor(arm.config))
        assert np.all(np.diff(whole.times) > 0.0)
        assert whole.passes == run.asks
        # the horizon is the guidance's: the first ask's arrival time (the truth duration) plus its slack
        assert run.horizon_s == pytest.approx(closing_horizon_s(truth_duration_s(run.series, default_anchor(arm.config))))
        if run.ended == runner.ENDED_HORIZON:
            assert run.horizon_s - 1.0 <= run.flown_s <= run.horizon_s + 1e-6
        # the record's own crossing is the one the flight ended on: re-cutting the whole record agrees
        recut = runner.cut_at_threshold_crossing(whole, run.series)
        assert recut.truncated_at_threshold == run.truncated
        assert recut.n_steps == whole.n_steps


def test_the_no_plan_variant_hands_the_absent_token(trained) -> None:
    _flights, series, arm = trained
    run = runner.fly_variant(arm, series[:1], runner.VARIANT_ONE_SHOT, _plan(), torch.device("cpu"), 4)[0]
    a0 = default_anchor(arm.config)
    fresh = runner.FlightRun(series=run.series, skeleton=run.skeleton)
    plan_at = runner.TruthPlans([fresh], a0).at_ask([fresh], [run.series], a0, first=True)[0]
    row = runner.ask_row(fresh, run.series, a0, arm.config, plan_at, with_plan=False,
                         floor_s=runner.ask_floor_s(arm.config, STEP_S))
    assert np.all(row[PLAN_TOKEN_KEY] == 0.0)
    assert "cta_s" in row                              # the arrival time is still handed over


def _capture_asks(monkeypatch):
    """Every ask's dynamics rows, keyed by (anchor, flight)."""
    seen: dict[tuple[int, str], dict] = {}
    original = runner.forecast_control_batch

    def spy(model, histories, config, normalizer, anchor, device, *, dynamics):
        for i, item in enumerate(histories):
            seen[(int(anchor), item.dataset_id)] = {name: value[i].cpu().numpy() for name, value in dynamics.items()}
        return original(model, histories, config, normalizer, anchor, device, dynamics=dynamics)

    monkeypatch.setattr(runner, "forecast_control_batch", spy)
    return seen


def test_the_variants_wire_the_token_present_and_absent(monkeypatch, trained) -> None:
    _flights, series, arm = trained
    a0 = default_anchor(arm.config)
    seen = _capture_asks(monkeypatch)
    by_variant = {}
    for variant in (runner.VARIANT_RECEDING, runner.VARIANT_NO_PLAN):
        seen.clear()
        runner.fly_variant(arm, series[:3], variant, _plan(), torch.device("cpu"), 4)
        by_variant[variant] = dict(seen)
    for item in series[:3]:
        assert by_variant[runner.VARIANT_RECEDING][(a0, item.dataset_id)][PLAN_TOKEN_KEY][-1] == 1.0
        assert np.all(by_variant[runner.VARIANT_NO_PLAN][(a0, item.dataset_id)][PLAN_TOKEN_KEY] == 0.0)


def test_a_later_ask_reads_the_truth_expert_at_the_flown_pose(monkeypatch, trained) -> None:
    """Ask 1: the arrival time is the expert's time-to-go and the token `targets_at` at the pose
    the first leg ended in — read by an expert built at a0 and asked in order, independently."""
    import math
    from dataclasses import replace

    from ts_transformer.data.channels import IDX
    from ts_transformer.inference.receding import rolled_series
    from ts_transformer.outputs.control.plan_token import plan_token
    from ts_transformer.outputs.plan.extractors import extract_plan, ground_speeds
    from ts_transformer.outputs.plan.labels import TruthExpert, on_final_pose, targets_at

    _flights, series, arm = trained
    a0 = default_anchor(arm.config)
    step_rows = int(round(STEP_S / arm.config.dt_s))
    seen = _capture_asks(monkeypatch)
    runs = runner.fly_variant(arm, series, runner.VARIANT_RECEDING, _plan(), torch.device("cpu"), 4)
    checked = 0
    for run in runs:
        if run.asks < 2:
            continue
        expert = TruthExpert(extract_plan(run.series, a0, run.skeleton), run.series, a0, run.skeleton)
        for history, anchor in ((run.series, a0), (rolled_series(run.series, a0, run.legs[0], a0 + step_rows, arm.config.dt_s), a0 + step_rows)):
            row = np.asarray(history.values[anchor], dtype=np.float64)
            e, n = float(row[IDX["e"]]), float(row[IDX["n"]])
            heading = math.atan2(row[IDX["ndot"]], row[IDX["edot"]])
            instruction, operating = expert.order_at(
                e, n, heading, float(ground_speeds(history, slice(anchor, anchor + 1))[0]),
            )
        if on_final_pose(run.skeleton, e, n, heading):          # the shared rule after the first ask
            operating = replace(operating, h_capture_m=None)
        asked = seen[(a0 + step_rows, run.series.dataset_id)]
        assert float(asked["cta_s"]) == pytest.approx(float(operating.T_s))
        assert asked[PLAN_TOKEN_KEY] == pytest.approx(plan_token(targets_at(operating, instruction, e, n, run.skeleton)))
        checked += 1
    assert checked


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


def test_the_readout_runs_end_to_end_with_records(monkeypatch, tmp_path, trained) -> None:
    flights, _series, arm = trained
    _patch_data_plane(monkeypatch, flights, tmp_path)
    out = tmp_path / "lockstep"
    assert runner.main([
        "--checkpoint", f"t1={arm.path}", "--out", str(out), "--device", "cpu", "--split", "train",
        "--step-s", str(STEP_S), "--write-records", "--batch-size", "4",
    ]) == 0
    payload = json.loads((out / "tracker_lockstep.json").read_text())
    assert payload["schema"] == runner.RESULT_SCHEMA
    block = payload["checkpoints"]["t1"]
    assert set(block["variants"]) == set(runner.VARIANTS)
    for variant in runner.VARIANTS:
        summary = json.loads((out / "records" / "t1" / variant / "summary.json").read_text())
        assert summary[runner.RECORDS_BLOCK]["variant"] == variant
        assert summary[runner.RECORDS_BLOCK]["records"] == block["flights"]
        # every row carries the plan oracle's reference reading and its strata covariates, and the
        # blocks their shares — established IS the crossing on the final, as the plan oracle defines it
        rows = block["variants"][variant]["flights"]
        assert all(set(row["reference"]) >= set(runner.REFERENCE_SHARES) and "difficulty" in row for row in rows.values())
        everyone = block["variants"][variant]["strata"][STRATUM_ALL]
        assert everyone["fully_flyable_share"] == pytest.approx(
            np.mean([row["reference"]["fully_flyable"] for row in rows.values()]))
    # …and the gate readout reads the artifact (a guidance baseline made of the tracker's own rows)
    # a gate refuses a train-split artifact (the readout on this fixture is a train-split smoke)
    with pytest.raises(SystemExit, match="never a smoke run"):
        gates.load_lockstep(out)


# ── the leg rule ────────────────────────────────────────────────────────────────

def _truth_leg(series, anchor: int, config) -> Forecast:
    """The observed rows after ``anchor`` dressed as a forecast (one segment per row)."""
    origin = float(series.times[anchor])
    offsets = np.asarray(series.times[anchor + 1 :], dtype=np.float64) - origin
    durations = np.diff(np.concatenate(([0.0], offsets)))
    return Forecast(
        times=origin + offsets, values=np.asarray(series.values[anchor + 1 :], dtype=np.float64),
        normalized_progress=offsets / offsets[-1], anchor=anchor, final_time_s=float(offsets[-1]),
        predicted_final_time_s=float(offsets[-1]), horizon_mode=config.horizon_mode, passes=1,
        truncated_at_threshold=False, horizon_capped=False, sample_durations_s=durations,
        segment_durations_s=durations, controls=np.zeros((len(offsets), 3)), commands=np.zeros((len(offsets), 3)),
        control_parameterization=config.control_thrust_parameterization,
        geodetic_values=np.zeros((len(offsets), 7)), prediction_output=PREDICTION_CONTROL,
    )


def test_a_leg_is_whole_steps_and_a_flight_ends_only_by_crossing_or_at_its_horizon(trained) -> None:
    _flights, series, arm = trained
    item = next(s for s in series if float(s.times[-1] - s.times[default_anchor(arm.config)]) > 3 * STEP_S)
    a0 = default_anchor(arm.config)
    leg = _truth_leg(item, a0, arm.config)
    skeleton = runner.SkeletonCache().for_series(item)
    # far from its horizon: one step flown, asked again next
    run = runner.FlightRun(series=item, skeleton=skeleton, horizon_s=1e6)
    runner.fly_leg(run, item, leg, steps=1, step_s=STEP_S, ask=3)
    assert run.ended is None and run.next_ask == 4 and run.legs[-1].final_time_s == pytest.approx(STEP_S)
    # a one-shot leg of two whole steps: the next ask waits two steps
    run = runner.FlightRun(series=item, skeleton=skeleton, horizon_s=1e6)
    runner.fly_leg(run, item, leg, steps=2, step_s=STEP_S, ask=0)
    assert run.ended is None and run.next_ask == 2 and run.legs[-1].final_time_s == pytest.approx(2 * STEP_S)
    # its horizon inside the step: cut there, ended at the horizon, not established
    run = runner.FlightRun(series=item, skeleton=skeleton, horizon_s=0.5 * STEP_S)
    runner.fly_leg(run, item, leg, steps=1, step_s=STEP_S, ask=0)
    assert run.ended == runner.ENDED_HORIZON and not run.truncated
    assert run.flown_s <= 0.5 * STEP_S + 1e-6 and run.flown_s > 0.5 * STEP_S - float(np.max(leg.sample_durations_s)) - 1e-6
    # a horizon before the forecast's first row: the flight ends there and flies (and counts) nothing
    first_row_s = float(leg.sample_durations_s[0])
    run = runner.FlightRun(series=item, skeleton=skeleton, horizon_s=0.5 * first_row_s)
    runner.fly_leg(run, item, leg, steps=1, step_s=STEP_S, ask=0)
    assert run.ended == runner.ENDED_HORIZON and run.legs == [] and run.asks == 0
    # a forecast shorter than one step (only without a given CTA) is flown whole and ends the flight…
    short = runner.cut_at_lead(leg, float(np.cumsum(leg.sample_durations_s)[2]))
    run = runner.FlightRun(series=item, skeleton=skeleton, horizon_s=1e6)
    runner.fly_leg(run, item, short, steps=1, step_s=STEP_S, ask=0)
    assert run.ended in (runner.ENDED_FORECAST, runner.ENDED_CROSSED) and run.asks == 1
    # …unless its horizon comes first
    run = runner.FlightRun(series=item, skeleton=skeleton, horizon_s=float(np.cumsum(leg.sample_durations_s)[1]))
    runner.fly_leg(run, item, short, steps=1, step_s=STEP_S, ask=0)
    assert run.ended == runner.ENDED_HORIZON and run.flown_s <= run.horizon_s + 1e-6


def test_an_ask_hands_at_least_one_step_and_counts_a_raised_arrival(trained) -> None:
    _flights, series, arm = trained
    a0 = default_anchor(arm.config)
    floor = runner.ask_floor_s(arm.config, STEP_S)
    assert floor == max(arm.config.random_train_anchor_min_future_s, STEP_S)
    run = runner.FlightRun(series=series[0], skeleton=runner.SkeletonCache().for_series(series[0]))
    plan_at = runner.TruthPlans([run], a0).at_ask([run], [series[0]], a0, first=True)[0]
    low = runner.AskPlan(arrival_s=0.25 * floor, operating=plan_at.operating, instruction=plan_at.instruction)
    row = runner.ask_row(run, series[0], a0, arm.config, low, with_plan=True, floor_s=floor)
    assert float(row["cta_s"]) == floor and run.asks_below_floor == 1
    assert run.cta_raised_max_s == pytest.approx(0.75 * floor)
    row = runner.ask_row(run, series[0], a0, arm.config, plan_at, with_plan=True, floor_s=floor)
    assert float(row["cta_s"]) == pytest.approx(max(plan_at.arrival_s, floor)) and run.asks_below_floor == 1


# ── T2: the plan head's own plan ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def head(trained, tmp_path_factory):
    """A tiny plan head on the same synthetic flights and split rule as the tracker, with a
    SHORTER lookback than the tracker's (the real pair is 30 against 60)."""
    from ts_transformer.config import CHECKPOINT_SELECTION_OBJECTIVE, PREDICTION_PLAN

    flights, _series, arm = trained
    config = TSConfig(prediction_output=PREDICTION_PLAN, seq_len=6, n_segments=4, d_model=16, n_heads=4, d_ff=32,
                      e_layers=1, final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
                      checkpoint_selection_metric=CHECKPOINT_SELECTION_OBJECTIVE, epochs=1, patience=1, batch_size=8,
                      val_fraction=0.25, test_fraction=0.25)
    series, _report = build_series(flights, config, airport=AIRPORT)
    out = tmp_path_factory.mktemp("plan_head")
    torch.manual_seed(0)
    train(series, config, output_dir=out, data_provenance=fake_data_provenance(), verbose=False)
    model, loaded, normalizer, payload = load_checkpoint(out / "checkpoint.pt")
    return anytime.Arm(label="plan-head", path=out / "checkpoint.pt", model=model, config=loaded,
                       normalizer=normalizer, payload=payload, airports=(AIRPORT,), manifests=[out / "manifest.json"])


def test_a_head_ask_hands_over_the_heads_own_order_as_token_and_arrival(trained, head) -> None:
    """Independently of `HeadPlans`: the head's prediction on the ask's history, through
    `order_from_prediction`, the on-final capture rule and `targets_at` — at a0 (off the final)
    and late in each flight (on it), so both branches of the rule are exercised."""
    from ts_transformer.inference.forecast import history_batch
    from ts_transformer.outputs.control.plan_token import plan_token
    from ts_transformer.outputs.plan.labels import T_MIN_S, Operating, on_final_pose, order_from_prediction, targets_at
    from ts_transformer.outputs.plan.model import prediction_rows

    _flights, series, arm = trained
    branches = set()
    for anchor_of in (lambda item: default_anchor(arm.config), lambda item: item.n_samples - 6):
        items = [item for item in series if anchor_of(item) == anchor_of(series[0])][:4]
        anchor = anchor_of(items[0])
        runs = [runner.FlightRun(series=item, skeleton=runner.SkeletonCache().for_series(item)) for item in items]
        asked = runner.HeadPlans(head, 2, torch.device("cpu")).at_ask(runs, items, anchor, first=True)
        with torch.no_grad():
            values, probability = prediction_rows(head.model(torch.from_numpy(
                history_batch(items, head.config, head.normalizer, anchor))))
        for i, (run, plan_at) in enumerate(zip(runs, asked, strict=True)):
            e, n, heading, _v = runner.pose(run.series, anchor)
            order = order_from_prediction(values[i], float(probability[i]), e, n, run.skeleton)
            assert plan_at.arrival_s == pytest.approx(order.T_s) and plan_at.arrival_s >= T_MIN_S
            row = runner.ask_row(run, run.series, anchor, arm.config, plan_at, with_plan=True,
                                 floor_s=runner.ask_floor_s(arm.config, STEP_S))
            on_final = on_final_pose(run.skeleton, e, n, heading)
            branches.add(on_final)
            operating = Operating(T_s=order.T_s, V_mid_mps=order.V_mid_mps, d_decel_m=order.d_decel_m,
                                  V_final_mps=order.V_final_mps, h_capture_m=None if on_final else order.h_capture_m,
                                  d_join_m=order.d_join_m, remaining_m=order.remaining_m)
            assert row[PLAN_TOKEN_KEY] == pytest.approx(plan_token(targets_at(operating, order.instruction, e, n, run.skeleton)))
            assert float(row["cta_s"]) == pytest.approx(max(order.T_s, runner.ask_floor_s(arm.config, STEP_S)))
    assert branches == {True, False}


def test_a_head_run_reads_no_truth_and_ends_at_its_own_horizon(monkeypatch, trained, head) -> None:
    """On series with the truth stripped (no supervision rows) and no `TruthExpert` to build, a head
    run flies, and each flight's horizon is the head's own first arrival time plus the slack."""
    from dataclasses import replace

    import ts_transformer.outputs.plan.labels as labels_module

    _flights, series, arm = trained
    blind = [replace(item, supervision_times=None, supervision_values=None, supervision_weights=None) for item in series[:3]]
    a0 = default_anchor(arm.config)
    probe = [runner.FlightRun(series=item, skeleton=runner.SkeletonCache().for_series(item)) for item in blind]
    first = runner.HeadPlans(head, 4, torch.device("cpu")).at_ask(probe, blind, a0, first=True)
    monkeypatch.setattr(runner, "TruthExpert", None)
    monkeypatch.setattr(labels_module, "TruthExpert", None)
    monkeypatch.setattr(runner, "truth_duration_s", None)
    runs = runner.fly_variant(arm, blind, runner.VARIANT_RECEDING, _plan(), torch.device("cpu"), 4, head)
    for run, plan_at in zip(runs, first, strict=True):
        assert run.ended is not None
        assert run.horizon_s == pytest.approx(closing_horizon_s(plan_at.arrival_s))


def test_a_rolled_head_window_is_the_plan_paths_own(trained, head) -> None:
    """At a later ask the head reads `history_batch` on the rolled series — the window the plan
    path's own lockstep builds with `rolled_history` from the same flown leg."""
    from ts_transformer.data.target_conditioning import conditioned_history
    from ts_transformer.inference.forecast import history_batch
    from ts_transformer.inference.receding import cut_at_lead, rolled_series
    from ts_transformer.outputs.plan.forecast import rolled_history

    _flights, series, arm = trained
    a0 = default_anchor(arm.config)
    step_rows = int(round(STEP_S / arm.config.dt_s))
    item = series[0]
    leg = cut_at_lead(forecast_approaches(arm.model, [item], arm.config, arm.normalizer, device=torch.device("cpu"))[0], STEP_S)
    rolled = rolled_series(item, a0, leg, a0 + step_rows, arm.config.dt_s)
    ours = history_batch([rolled], head.config, head.normalizer, a0 + step_rows)[0]
    theirs = conditioned_history(head.normalizer.encode(rolled_history(item, a0, [leg], head.config)), None)
    assert ours == pytest.approx(theirs, abs=1e-5)


def test_a_head_is_refused_on_a_flight_it_trained_on_or_another_series_contract(trained, head) -> None:
    from dataclasses import replace

    _flights, _series, arm = trained
    grid = anytime.Grid(split="train", bins_m=(), min_future_s=0.0, batch_size=None, limit=0)
    assert set(arm.payload["split"]["train"]) & set(head.payload["split"]["train"])
    with pytest.raises(SystemExit, match="TRAIN split"):
        runner.check_head(head, [arm], grid)
    held_out = replace(grid, split="val")
    assert not set(arm.payload["split"]["val"]) & set(head.payload["split"]["train"])
    runner.check_head(head, [arm], held_out)
    for overrides, message in (({"dt_s": 1.0}, "dt_s"), ({"seq_len": 9}, "lookback")):
        with pytest.raises(SystemExit, match=message):
            runner.check_head(replace(head, config=replace(head.config, **overrides)), [arm], held_out)
    with pytest.raises(SystemExit, match="asked about"):
        runner.check_head(replace(head, airports=("KSJC",)), [arm], held_out)


@pytest.mark.parametrize("config_overrides, variants, message", [
    ({"plan_conditioning": "off", "plan_conditioning_dropout": 0.0}, runner.VARIANTS, "does not read"),
    ({"prediction_output": "state"}, (runner.VARIANT_RECEDING,), "CONTROL"),
    ({"cta_conditioning": "self-q"}, (runner.VARIANT_RECEDING,), "arrival time"),
])
def test_the_lockstep_refuses_what_it_cannot_fly(config_overrides, variants, message) -> None:
    config = SimpleNamespace(**{
        "prediction_output": PREDICTION_CONTROL, "control_command_hook": "off", "latent_dim": 0,
        "duration_head": "point", "cta_conditioning": CTA_CONDITIONING_GIVEN,
        "plan_conditioning": PLAN_CONDITIONING_TRUTH_NEXT, "dt_s": 2.0, "control_rollout_integrator_dt_s": 0.5,
        **config_overrides,
    })
    with pytest.raises(SystemExit, match=message):
        runner.check_arm(SimpleNamespace(label="x", path="x", config=config), _plan(variants=variants))
