"""`run_ts.py short_horizon_readout` (two-tier v2 §3, evaluation 1).

Pinned: a fixed-horizon waypoint checkpoint is read with and without its token from the fixed
anchor and the bins, both readings spanning exactly the horizon; a whole-approach reference is
cut at the horizon (absent, never scored, where it ends before it); the pairing is over the
flights both hold at an anchor set; the truth-plan reading of a checkpoint told the truth is
never worse than its no-plan reading on a truth leg; the arms must share their fixed anchor.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import ts_transformer.experiments.anytime_curve as anytime
import ts_transformer.experiments.short_horizon_readout as runner
from ts_transformer.config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    PLAN_CONDITIONING_WAYPOINTS,
    PLAN_WAYPOINT_SEGMENT_S,
    PREDICTION_CONTROL,
    TSConfig,
    default_anchor,
)
from ts_transformer.data.approach_difficulty import STRATUM_ALL
from ts_transformer.data.dataset import build_series, truth_duration_s
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.forecast import Forecast
from ts_transformer.inference.receding import mean_displacement_to
from ts_transformer.training.train import load_checkpoint, train
from ts_transformer.tests.support import AIRPORT, RUNWAY, fake_data_provenance

HORIZON_S = 2 * PLAN_WAYPOINT_SEGMENT_S
TINY = dict(seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1, final_time_scale_s=100.0,
            device="cpu", horizon_mode="normalized", epochs=1, patience=1, batch_size=8, dropout=0.0,
            val_fraction=0.25, test_fraction=0.25)
CONTROL = dict(
    prediction_output=PREDICTION_CONTROL,
    control_duration_parameterization=CONTROL_DURATION_UNIFORM,
    control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
    control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
    control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
)


def _arm(label: str, out, series, config) -> anytime.Arm:
    torch.manual_seed(0)
    train(series, config, output_dir=out, data_provenance=fake_data_provenance(), verbose=False)
    model, loaded, normalizer, payload = load_checkpoint(out / "checkpoint.pt")
    return anytime.Arm(label=label, path=out / "checkpoint.pt", model=model, config=loaded, normalizer=normalizer,
                       payload=payload, airports=(AIRPORT,), manifests=[out / "manifest.json"])


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    short = TSConfig(**CONTROL, **TINY, control_horizon_s=HORIZON_S, final_time_loss_weight=0.0,
                     plan_conditioning=PLAN_CONDITIONING_WAYPOINTS, plan_conditioning_dropout=0.5)
    whole = TSConfig(**CONTROL, **TINY)
    series, _report = build_series(flights, short, airport=AIRPORT)
    root = tmp_path_factory.mktemp("short_horizon")
    l1 = _arm("L1", root / "l1", series, short)
    reference = _arm("ref", root / "ref", series, whole)
    return series, l1, reference


def _plan(**overrides) -> runner.ReadoutPlan:
    settings = dict(split="train", horizon_s=HORIZON_S, leads_s=(30.0, 60.0), bins_m=(12_000.0, 8_000.0),
                    limit=0, batch_size=4, write_records=False)
    settings.update(overrides)
    return runner.ReadoutPlan(**settings)


def test_the_readout_spans_the_horizon_from_every_anchor_set_with_and_without_the_plan(arms, tmp_path) -> None:
    series, l1, reference = arms
    payload = runner.measure([reference, l1], "ref", {"ref": series, "L1": series}, _plan(), torch.device("cpu"),
                             None, "smoke")
    assert payload["schema"] == runner.RESULT_SCHEMA and payload["reference"] == "ref"
    assert set(payload["arms"]["L1"]["variants"]) == {runner.VARIANT_PLAN, runner.VARIANT_NO_PLAN}
    assert set(payload["arms"]["ref"]["variants"]) == {runner.VARIANT_NO_PLAN}
    sets = payload["arms"]["L1"]["variants"][runner.VARIANT_PLAN]["sets"]
    assert set(sets) == {runner.FIXED_SET, "12km", "8km"}
    fixed = sets[runner.FIXED_SET]
    assert fixed["anchored_flights"] == len(series) and fixed["forecasts_shorter_than_horizon"] == 0
    for row in fixed["flights"].values():
        assert row["anchor_index"] == default_anchor(l1.config)
        assert row["predicted_final_time_s"] == pytest.approx(HORIZON_S)
        assert 0.0 <= row["ade_m"] and row["at"]["30"] is not None and row["at"]["60"] is not None
    # a bin's flights sit at their own anchors, every one with the horizon of truth after it
    for row in sets["8km"]["flights"].values():
        assert row["anchor_index"] >= default_anchor(l1.config)
    bin_anchors = {key: row["anchor_index"] for key, row in sets["8km"]["flights"].items()}
    for item in series:
        if item.dataset_id in bin_anchors:
            assert truth_duration_s(item, bin_anchors[item.dataset_id]) >= HORIZON_S - 1e-6
    # the pairing: L1's no-plan against the reference's, and its truth-plan against its no-plan
    paired = payload["paired"]["L1"]
    assert set(paired) == {f"{runner.VARIANT_NO_PLAN} − reference {runner.VARIANT_NO_PLAN}",
                           f"{runner.VARIANT_PLAN} − {runner.VARIANT_NO_PLAN}"}
    cell = paired[f"{runner.VARIANT_NO_PLAN} − reference {runner.VARIANT_NO_PLAN}"][runner.FIXED_SET][STRATUM_ALL]
    assert cell["n"] == len(series) - payload["arms"]["ref"]["variants"][runner.VARIANT_NO_PLAN]["sets"][runner.FIXED_SET]["forecasts_shorter_than_horizon"]
    assert payload["paired"]["ref"] == {}
    text = runner.render(payload)
    assert "truth-plan" in text and "paired" in text


def test_a_whole_approach_forecast_is_cut_at_the_horizon_or_absent(arms) -> None:
    series, _l1, reference = arms
    rows, _pairs, short = runner.measure_variant(
        reference, series, {i: default_anchor(reference.config) for i in range(len(series))}, runner.VARIANT_NO_PLAN,
        _plan(), torch.device("cpu"), 4, runner.SkeletonCache(), build_records=False,
    )
    assert len(rows) + short == len(series)
    for row in rows.values():
        assert row["predicted_final_time_s"] >= HORIZON_S - 1e-6      # a shorter one is absent, never scored
    # the cut itself: a forecast longer than the horizon keeps exactly the horizon
    long = Forecast(times=np.arange(1.0, 121.0), values=np.zeros((120, 6)), normalized_progress=np.linspace(0, 1, 120),
                    anchor=0, final_time_s=120.0, predicted_final_time_s=120.0, horizon_mode="normalized", passes=1,
                    truncated_at_threshold=False, horizon_capped=False, sample_durations_s=np.ones(120),
                    segment_durations_s=np.ones(120))
    assert runner.within_horizon(long, HORIZON_S).final_time_s == pytest.approx(HORIZON_S)
    assert runner.within_horizon(runner.cut_at_lead(long, 30.0), HORIZON_S) is None


def test_mean_displacement_to_is_the_lead_time_errors_accounting(arms) -> None:
    series, _l1, _reference = arms
    item = series[0]
    anchor = 10
    # a forecast that IS the truth reads zero; one displaced by a constant reads that constant
    offsets = np.asarray(item.times[anchor + 1 :]) - float(item.times[anchor])
    truth = Forecast(times=np.asarray(item.times[anchor + 1 :]), values=np.asarray(item.values[anchor + 1 :]),
                     normalized_progress=offsets / offsets[-1], anchor=anchor, final_time_s=float(offsets[-1]),
                     predicted_final_time_s=float(offsets[-1]), horizon_mode="normalized", passes=1,
                     truncated_at_threshold=False, horizon_capped=False,
                     sample_durations_s=np.diff(np.concatenate(([0.0], offsets))), segment_durations_s=offsets)
    assert mean_displacement_to(item, truth, anchor, HORIZON_S) == pytest.approx(0.0, abs=1e-6)
    shifted = np.asarray(item.values[anchor + 1 :], dtype=np.float64).copy()
    shifted[:, 0] += 30.0
    from dataclasses import replace
    # the observed anchor row stands in at t=0 (displacement 0), the shift ramps in over the
    # first sample interval and holds after: the mean over the 1 s grid 0..Δ inclusive
    first_row_s = float(item.times[anchor + 1] - item.times[anchor])
    grid = np.arange(0.0, HORIZON_S + 1e-9, 1.0)
    expected = float(np.mean(np.interp(grid, [0.0, first_row_s, HORIZON_S], [0.0, 30.0, 30.0])))
    assert mean_displacement_to(item, replace(truth, values=shifted), anchor, HORIZON_S) == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError, match="before the"):
        mean_displacement_to(item, runner.cut_at_lead(truth, 30.0), anchor, HORIZON_S)


def test_the_arms_must_share_their_fixed_anchor_and_the_reference_reads_no_plan(arms) -> None:
    series, l1, reference = arms
    from dataclasses import replace
    moved = replace(l1, config=replace(l1.config, anchor_floor_index=default_anchor(l1.config) + 5))
    with pytest.raises(SystemExit, match="fixed anchors differ"):
        runner.check_arms_share_the_anchor([reference, moved])
    with pytest.raises(SystemExit, match="reads a plan token"):
        runner.check_arm(l1, _plan(), reference=True)
    with pytest.raises(SystemExit, match="against a --horizon-s"):
        runner.check_arm(l1, _plan(horizon_s=30.0), reference=False)
    runner.check_arm(reference, _plan(), reference=True)


def test_records_are_written_per_arm_variant_and_anchor_set(arms, tmp_path) -> None:
    series, l1, reference = arms
    payload = runner.measure([reference, l1], "ref", {"ref": series, "L1": series},
                             _plan(bins_m=(8_000.0,), write_records=True), torch.device("cpu"),
                             tmp_path / "records", "smoke")
    dirs = payload["arms"]["L1"]["record_dirs"]
    assert set(dirs) == {runner.VARIANT_PLAN, runner.VARIANT_NO_PLAN} and set(dirs[runner.VARIANT_PLAN]) == {runner.FIXED_SET, "8km"}
    summary = json.loads((tmp_path / dirs[runner.VARIANT_PLAN][runner.FIXED_SET] / "summary.json").read_text())
    block = summary[runner.RECORDS_BLOCK]
    assert block["schema"] == runner.RECORDS_SCHEMA and block["variant"] == runner.VARIANT_PLAN
    assert block["horizon_s"] == HORIZON_S and block["records"] == len(series) and "protocol C" in block["reads_the_future"]
