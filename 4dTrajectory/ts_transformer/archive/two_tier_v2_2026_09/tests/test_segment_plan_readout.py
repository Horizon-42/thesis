"""`run_ts.py segment_plan_readout` (two-tier v2 §4, the L2 evaluation).

Pinned: every checkpoint is read at ONE common fixed anchor (the latest of their own) and the
bins; a forecast's last row is held past its end and only the truth's end makes a lead absent;
the constant-velocity reference is exactly the anchor state carried on; a segment-plan head's
plan block rides beside its lead cells; the pairing is over the flights both hold, per lead,
with each side's p50 over those flights; the references must be whole-approach heads.
"""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

import ts_transformer.experiments.anytime_curve as anytime
import ts_transformer.experiments.segment_plan_readout as runner
from ts_transformer.config import (
    CHECKPOINT_SELECTION_OBJECTIVE,
    PREDICTION_SEGMENT_PLAN,
    PREDICTION_STATE,
    TSConfig,
    default_anchor,
)
from ts_transformer.data.approach_difficulty import STRATUM_ALL
from ts_transformer.data.channels import POSITION_IDX, VELOCITY_IDX
from ts_transformer.data.dataset import build_series, truth_duration_s
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.forecast import Forecast
from ts_transformer.training.train import load_checkpoint, train
from ts_transformer.tests.support import AIRPORT, RUNWAY, fake_data_provenance

TINY = dict(n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1, final_time_scale_s=100.0, device="cpu",
            horizon_mode="normalized", epochs=1, patience=1, batch_size=8, dropout=0.0, val_fraction=0.25,
            test_fraction=0.25)
LEADS = (30.0, 60.0, 120.0)


def _arm(label: str, out, series, config) -> anytime.Arm:
    torch.manual_seed(0)
    train(series, config, output_dir=out, data_provenance=fake_data_provenance(), verbose=False)
    model, loaded, normalizer, payload = load_checkpoint(out / "checkpoint.pt")
    return anytime.Arm(label=label, path=out / "checkpoint.pt", model=model, config=loaded, normalizer=normalizer,
                       payload=payload, airports=(AIRPORT,), manifests=[out / "manifest.json"])


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    # the plan head: two coarse segments of history (seq_len 31), six segments ahead
    plan = TSConfig(prediction_output=PREDICTION_SEGMENT_PLAN, seq_len=31, segment_plan_segments=6,
                    checkpoint_selection_metric=CHECKPOINT_SELECTION_OBJECTIVE, **TINY)
    # the reference: a state head with a SHORTER lookback (its own fixed anchor is earlier)
    state = TSConfig(prediction_output=PREDICTION_STATE, seq_len=8, **TINY)
    series, _report = build_series(flights, plan, airport=AIRPORT)
    root = tmp_path_factory.mktemp("segment_plan_readout")
    # the reference's cohort is a SUBSET of the plan head's (native32's openap-direct split against an
    # all-aircraft one): the pairing is over the flights both hold
    return series, _arm("A", root / "a", series, plan), _arm("state", root / "state", series[:9], state)


def _plan(**overrides) -> runner.ReadoutPlan:
    settings = dict(split="train", leads_s=LEADS, bins_m=(12_000.0, 8_000.0), limit=0, batch_size=4, write_records=False)
    settings.update(overrides)
    return runner.ReadoutPlan(**settings)


def test_the_readout_reads_every_checkpoint_at_one_anchor_and_pairs_them(arms) -> None:
    series, plan_arm, state_arm = arms
    payload = runner.measure([plan_arm], [state_arm], {"A": series, "state": series[:9]}, _plan(), torch.device("cpu"),
                             None, "smoke")
    assert payload["schema"] == runner.RESULT_SCHEMA
    assert payload["cohorts"]["A"]["flights"] == len(series) and payload["cohorts"]["state"]["flights"] == 9
    assert payload["cohorts"][runner.CONSTANT_VELOCITY]["flights"] == len(series)
    # the common anchor is the LATEST of the checkpoints' own; the state head's own is earlier
    assert payload["anchor"] == default_anchor(plan_arm.config) > default_anchor(state_arm.config)
    assert payload["checkpoints"]["state"]["own_fixed_anchor"] == default_anchor(state_arm.config)
    assert payload["references"] == ["state", runner.CONSTANT_VELOCITY] and payload["arms"] == ["A"]
    assert set(payload["checkpoints"]) == {"A", "state", runner.CONSTANT_VELOCITY}
    sets = payload["checkpoints"]["A"]["sets"]
    assert set(sets) == {runner.FIXED_SET, "12km", "8km"}
    fixed = sets[runner.FIXED_SET]
    assert fixed["anchored_flights"] == len(series)
    for row in fixed["flights"].values():
        assert row["anchor_index"] == payload["anchor"]
        assert set(row["at"]) == {"30", "60", "120"}
    # the plan block rides beside the plan head's cells, and only there
    assert fixed["plan"][STRATUM_ALL]["flights"] == len(series)
    assert "plan" not in payload["checkpoints"]["state"]["sets"][runner.FIXED_SET]
    assert "plan" not in payload["checkpoints"][runner.CONSTANT_VELOCITY]["sets"][runner.FIXED_SET]
    # every bin anchor is at or after the common anchor with the first lead of truth after it
    for row in sets["8km"]["flights"].values():
        assert row["anchor_index"] >= payload["anchor"]
    by_id = {item.dataset_id: item for item in series}
    for key, row in sets["8km"]["flights"].items():
        assert truth_duration_s(by_id[key], row["anchor_index"]) >= LEADS[0] - 1e-6
    # the pairing: A against the state head and against the constant-velocity reference, per lead
    paired = payload["paired"]["A"]
    assert set(paired) == {"state", runner.CONSTANT_VELOCITY}
    cell = paired["state"][runner.FIXED_SET][STRATUM_ALL]["30"]
    assert cell["n"] == 9 and cell["arm_p50_m"] is not None and cell["reference_p50_m"] is not None
    assert paired[runner.CONSTANT_VELOCITY][runner.FIXED_SET][STRATUM_ALL]["30"]["n"] == len(series)
    assert cell["delta_of_p50_m"] == pytest.approx(cell["arm_p50_m"] - cell["reference_p50_m"])
    assert 0.0 <= cell["arm_better_share"] <= 1.0
    text = runner.render(payload)
    assert "paired" in text and runner.CONSTANT_VELOCITY in text and "plan:" in text


def test_a_lead_is_held_past_the_forecasts_end_and_absent_only_where_the_truth_landed(arms) -> None:
    series, plan_arm, _state_arm = arms
    item = series[0]
    # an anchor 75 s before the OBSERVED track's end: 30 and 60 s are inside the truth, 120 s is past it
    anchor = int(np.searchsorted(item.times, item.times[-1] - 75.0))
    rows, _pairs, readings = runner.measure_set(plan_arm, [item], {0: anchor}, _plan(), torch.device("cpu"), 4, 6,
                                                build_records=False)
    row = rows[item.dataset_id]
    assert row["at"]["120"] is None                       # the truth has landed: absent
    assert row["at"]["30"] is not None and row["at"]["60"] is not None
    # a lead past the forecast's own end is READ (its last row held) and counted as held
    assert row["held"]["60"] == (row["forecast_end_s"] < 60.0 - 1e-6)
    assert row["held"]["30"] == (row["forecast_end_s"] < 30.0 - 1e-6)
    assert dict(readings)[item.dataset_id]["truth_arrives"]


def test_the_constant_velocity_reference_is_the_anchor_state_carried_on(arms) -> None:
    series, _plan_arm, _state_arm = arms
    item = series[0]
    anchor = 40
    forecast = runner.constant_velocity_forecast(item, anchor, LEADS, 6)
    state = item.values[anchor]
    for k, lead in enumerate(LEADS):
        np.testing.assert_allclose(forecast.values[k, list(POSITION_IDX)],
                                   state[list(POSITION_IDX)] + lead * state[list(VELOCITY_IDX)])
    assert forecast.horizon_capped and not forecast.truncated_at_threshold
    assert forecast.final_time_s == pytest.approx(LEADS[-1])


def _short_copy(item, samples: int):
    """The flight cut to ``samples`` observed rows (its supervision cut with it)."""
    return replace(item, times=item.times[:samples], values=item.values[:samples],
                   supervision_times=item.supervision_times[:samples], supervision_values=item.supervision_values[:samples],
                   supervision_weights=item.supervision_weights[:samples])


def test_the_fixed_set_admits_only_flights_that_hold_the_anchor_with_the_first_lead_after_it(arms) -> None:
    series, plan_arm, state_arm = arms
    fixed = runner.common_anchor([plan_arm, state_arm])
    short = _short_copy(series[0], fixed)                                   # one sample short of the anchor
    thin = _short_copy(series[1], fixed + int(LEADS[0] / 2.0 / 2.0))       # holds it, under 30 s of track after
    cohort = runner.cohort_at([short, thin, *series[2:]], _plan(), anchor=fixed, seq_len=31)
    assert short.dataset_id not in cohort.keys                              # not even stratified there
    assert thin.dataset_id in cohort.keys
    admitted = {cohort.keys[i] for i in cohort.sets[runner.FIXED_SET]}
    assert thin.dataset_id not in admitted and len(admitted) == len(series) - 2


def test_a_forecast_that_ends_before_a_lead_is_read_at_its_held_last_row_and_counted(arms) -> None:
    series, _plan_arm, _state_arm = arms
    item = series[0]
    anchor = 40
    origin = float(item.times[anchor])
    # a two-row forecast ending 45 s after the anchor, its last row at the threshold
    rows = np.zeros((2, 6))
    rows[0, list(POSITION_IDX)] = item.values[anchor, list(POSITION_IDX)]
    rows[1, list(POSITION_IDX)] = item.target_chart
    forecast = Forecast(times=origin + np.array([30.0, 45.0]), values=rows, normalized_progress=np.array([2 / 3, 1.0]),
                        anchor=anchor, final_time_s=45.0, predicted_final_time_s=45.0, horizon_mode="normalized", passes=1,
                        truncated_at_threshold=True, horizon_capped=False, sample_durations_s=np.array([30.0, 15.0]),
                        segment_durations_s=np.array([30.0, 15.0]))
    row = runner.lead_row(item, forecast, anchor, LEADS)
    assert row["held"] == {"30": False, "60": True, "120": True}
    truth_at_60 = np.array([np.interp(origin + 60.0, item.times, item.values[:, c]) for c in POSITION_IDX])
    assert row["at"]["60"] == pytest.approx(float(np.linalg.norm(item.target_chart - truth_at_60)))
    # ...and the held counts reach the paired cell
    keys = [item.dataset_id]
    masks = {s: np.array([True]) for s in runner.STRATA}
    cell = runner.paired({keys[0]: row}, {keys[0]: runner.lead_row(item, forecast, anchor, LEADS)}, masks, keys, LEADS)
    assert cell[STRATUM_ALL]["60"] == {"n": 1, "arm_p50_m": row["at"]["60"], "reference_p50_m": row["at"]["60"],
                                        "delta_of_p50_m": 0.0, "delta_p50_m": 0.0, "arm_better_share": 0.0,
                                        "arm_held": 1, "reference_held": 1}


def test_records_are_written_for_the_arms_and_the_references_with_the_block(arms, tmp_path) -> None:
    series, plan_arm, state_arm = arms
    payload = runner.measure([plan_arm], [state_arm], {"A": series, "state": series[:9]}, _plan(bins_m=(), write_records=True),
                             torch.device("cpu"), tmp_path / "records", "smoke")
    assert set(payload["checkpoints"]["A"]["record_dirs"]) == {runner.FIXED_SET}
    assert set(payload["checkpoints"]["state"]["record_dirs"]) == {runner.FIXED_SET}
    assert payload["checkpoints"][runner.CONSTANT_VELOCITY]["record_dirs"] == {}
    summary = json.loads((tmp_path / "records" / "A" / runner.FIXED_SET / "summary.json").read_text())
    block = summary[runner.RECORDS_BLOCK]
    assert block["schema"] == runner.RECORDS_SCHEMA and block["anchor_set"] == runner.FIXED_SET
    assert block["fixed_anchor"] == payload["anchor"] and block["records"] == len(series)
    assert block["prediction_output"] == PREDICTION_SEGMENT_PLAN and block["campaign"] == "smoke"


def test_refusals(arms) -> None:
    _series, plan_arm, state_arm = arms
    with pytest.raises(SystemExit, match="segment-plan head"):
        runner.check_checkpoint(plan_arm, reference=True, leads_s=LEADS)
    runner.check_checkpoint(state_arm, reference=True, leads_s=LEADS)
    runner.check_checkpoint(plan_arm, reference=False, leads_s=LEADS)
    window = replace(state_arm, config=TSConfig(prediction_output=PREDICTION_STATE, seq_len=8, **{**TINY, "horizon_mode": "window",
                                                                                                  "window_horizon_steps": 5}))
    with pytest.raises(SystemExit, match="window forecast ends before"):
        runner.check_checkpoint(window, reference=True, leads_s=LEADS)
    parser = runner.build_parser()
    with pytest.raises(SystemExit):
        runner.parse_plan(parser, parser.parse_args(["--out", "x", "--leads-s", "60,30"]))
    with pytest.raises(SystemExit):
        runner.parse_plan(parser, parser.parse_args(["--out", "x", "--bins-km", "12,12"]))
