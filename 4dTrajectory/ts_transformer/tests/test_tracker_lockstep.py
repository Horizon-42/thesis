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
from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows, build_series, dataset_flight_key
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
    settings = dict(split="train", step_s=STEP_S, cap_factor=1.5, variants=runner.VARIANTS, limit=0,
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
        fresh = runner.FlightRun(series=run.series, labels=run.labels, skeleton=run.skeleton,
                                 expert=type(run.expert)(run.labels, run.series, a0, run.skeleton), cap_s=run.cap_s)
        row = runner.ask_row(fresh, run.series, a0, arm.config, first=True, with_plan=True)
        training = windows.context.row(i)
        assert row[PLAN_TOKEN_KEY] == pytest.approx(training[PLAN_TOKEN_KEY])
        assert float(row["cta_s"]) == pytest.approx(float(training["cta_s"]))
        assert run.ended == runner.ENDED_ONE_SHOT and run.asks == 1
        assert run.legs[0].values == pytest.approx(expected.values, abs=1e-6)


def test_receding_flights_fly_one_step_per_ask_and_end_by_a_stated_rule(trained) -> None:
    _flights, series, arm = trained
    runs = runner.fly_variant(arm, series, runner.VARIANT_RECEDING, _plan(), torch.device("cpu"), 4)
    assert any(run.asks > 1 for run in runs)
    for run in runs:
        assert run.ended in (runner.ENDED_CROSSED, runner.ENDED_FORECAST, runner.ENDED_CAPPED)
        for leg in run.legs[:-1]:
            assert leg.final_time_s == pytest.approx(STEP_S)
        whole = runner.whole_forecast(run, default_anchor(arm.config))
        assert np.all(np.diff(whole.times) > 0.0)
        assert whole.passes == run.asks
        if run.ended == runner.ENDED_CAPPED:
            assert run.flown_s >= run.cap_s - 1e-6


def test_the_no_plan_variant_hands_the_absent_token(trained) -> None:
    _flights, series, arm = trained
    run = runner.fly_variant(arm, series[:1], runner.VARIANT_ONE_SHOT, _plan(), torch.device("cpu"), 4)[0]
    a0 = default_anchor(arm.config)
    fresh = runner.FlightRun(series=run.series, labels=run.labels, skeleton=run.skeleton,
                             expert=type(run.expert)(run.labels, run.series, a0, run.skeleton), cap_s=run.cap_s)
    row = runner.ask_row(fresh, run.series, a0, arm.config, first=True, with_plan=False)
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

    from ts_transformer.data.channels import IDX
    from ts_transformer.inference.receding import rolled_series
    from ts_transformer.outputs.control.plan_token import plan_token
    from ts_transformer.outputs.plan.extractors import ground_speeds
    from ts_transformer.outputs.plan.labels import TruthExpert, targets_at

    _flights, series, arm = trained
    a0 = default_anchor(arm.config)
    step_rows = int(round(STEP_S / arm.config.dt_s))
    seen = _capture_asks(monkeypatch)
    runs = runner.fly_variant(arm, series, runner.VARIANT_RECEDING, _plan(), torch.device("cpu"), 4)
    checked = 0
    for run in runs:
        if run.asks < 2:
            continue
        expert = TruthExpert(run.labels, run.series, a0, run.skeleton)
        for history, anchor in ((run.series, a0), (rolled_series(run.series, a0, run.legs[0], a0 + step_rows, arm.config.dt_s), a0 + step_rows)):
            row = np.asarray(history.values[anchor], dtype=np.float64)
            e, n = float(row[IDX["e"]]), float(row[IDX["n"]])
            instruction, operating = expert.order_at(
                e, n, math.atan2(row[IDX["ndot"]], row[IDX["edot"]]), float(ground_speeds(history, slice(anchor, anchor + 1))[0]),
            )
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
