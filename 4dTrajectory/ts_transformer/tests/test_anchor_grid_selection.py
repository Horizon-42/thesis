"""`anchor-grid-common-grid-ade`: what the number IS, and what it is not.

The metric averages the SAME common-grid ADE over five anchor sets — L−1 and
`anchor_grid`'s four remaining-path bins — instead of L−1 alone (A1 of
`docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`). Four things decide
whether that number means what it says, and each is a test here:

* every anchor set contributes its value AND its coverage, and the published mean is the
  mean of exactly those five values (equal weight per SET, never per flight-anchor pair);
* the L−1 set is bit-for-bit the value `fixed-anchor-common-grid-ade` selects on, so the
  two metrics stay comparable and the L−1 veto stays readable;
* a flight that has no admissible anchor at a bin is ABSENT from that bin's mean — never
  scored zero, never carried in from another bin — and a bin no flight reaches is refused
  rather than quietly dropped from the average;
* the training loop keeps the epoch with the lowest GRID mean, which is the whole point:
  the fixed metric froze the A0-random arm at epoch 10 on the one anchor it was least
  specialised for.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

import run_naming as rn
import validation as val
from anchor_grid import (
    DEFAULT_GRID_MIN_FUTURE_S,
    VALIDATION_ANCHOR_GRID_KM,
    VALIDATION_ANCHOR_GRID_M,
    anchors_for_bin,
    remaining_path_profiles,
)
from batch_contract import anchor_state, unpack_batch
from config import (
    CHECKPOINT_SELECTION_ANCHOR_GRID_ADE,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CHECKPOINT_SELECTION_METRICS,
    TSConfig,
)
from data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from dataset import Normalizer, build_series
from fixed_anchor_validation import fixed_anchor_common_grid_ade_metrics
from models import build_model
from prediction_outputs import StatePrediction
from splits import split_by_flight
from synthetic import synthetic_arrivals
from train import fit_model, train
from validation import (
    ANCHOR_GRID_L1_KEY,
    ValidationSelection,
    build_anchor_grid_validation_plans,
    replay_validation_plan,
    validation_datasets,
)

AIRPORT, RUNWAY = "KRDU", "05L"
#: The common grid and the model's own node grid deliberately coincide, so the
#: truth-copying model below is exact rather than exact-up-to-resampling.
POINTS = 8


def _config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output="state",
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ANCHOR_GRID_ADE,
        validation_common_grid_points=POINTS,
        seq_len=8, n_segments=POINTS, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=1, patience=1, batch_size=8, dropout=0.0,
        val_fraction=0.25, test_fraction=0.25,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def _provenance() -> dict:
    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }


@pytest.fixture(scope="module")
def cohort():
    """Twelve synthetic arrivals and one untrained model — enough for every anchor set."""
    config = _config()
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3), config, airport=AIRPORT
    )
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    torch.manual_seed(0)
    model = build_model(config, normalizer).eval()
    val_sets = validation_datasets(series, config, normalizer)
    return config, series, normalizer, model, val_sets


def _select(cohort, plans, model=None) -> ValidationSelection:
    config, _series, normalizer, built, val_sets = cohort
    return val.VALIDATION_SELECTIONS[CHECKPOINT_SELECTION_ANCHOR_GRID_ADE](
        model=model or built,
        val_sets=val_sets,
        normalizer=normalizer,
        config=config,
        device=torch.device("cpu"),
        val_by_airport={AIRPORT: 0.0},
        anchor_grid_plans=plans,
    )


@pytest.fixture(scope="module")
def plans(cohort):
    _config_, _series, _normalizer, _model, val_sets = cohort
    return build_anchor_grid_validation_plans(val_sets, 8)


# ── what the published number is made of ────────────────────────────────────

def test_every_anchor_set_publishes_its_value_and_its_coverage(cohort, plans) -> None:
    selection = _select(cohort, plans)
    block = selection.anchor_grid
    assert block["selection_metric"] == CHECKPOINT_SELECTION_ANCHOR_GRID_ADE
    assert block["grid_km"] == list(VALIDATION_ANCHOR_GRID_KM)
    assert block["min_future_s"] == DEFAULT_GRID_MIN_FUTURE_S
    assert list(block["anchor_sets"]) == [ANCHOR_GRID_L1_KEY, "16000", "12000", "8000", "6000"]
    for name, cell in block["anchor_sets"].items():
        assert cell["flights"] == 12, name          # this cohort reaches every bin
        assert cell["ade_m"] > 0.0, name
        assert cell["by_airport"][AIRPORT]["flights"] == 12
    # Five different numbers: the sets are scored independently, not copied from L−1.
    assert len({cell["ade_m"] for cell in block["anchor_sets"].values()}) == 5


def test_the_value_is_the_mean_of_the_recorded_anchor_sets(cohort, plans) -> None:
    """Equal weight per SET. Pooling (flight, anchor) pairs would weight bins by coverage."""
    selection = _select(cohort, plans)
    recorded = [cell["ade_m"] for cell in selection.anchor_grid["anchor_sets"].values()]
    assert len(recorded) == 5
    assert selection.value == pytest.approx(float(np.mean(recorded)))
    assert selection.anchor_grid["mean_ade_m"] == pytest.approx(selection.value)
    assert selection.by_airport[AIRPORT] == pytest.approx(selection.value)


def test_the_l_minus_one_set_is_the_fixed_metrics_own_value(cohort, plans) -> None:
    """The L−1 veto stays readable: the two metrics quote the same number for it."""
    config, _series, normalizer, model, val_sets = cohort
    fixed = val.VALIDATION_SELECTIONS[CHECKPOINT_SELECTION_COMMON_GRID_ADE](
        model=model, val_sets=val_sets, normalizer=normalizer,
        config=config, device=torch.device("cpu"), val_by_airport={AIRPORT: 0.0},
    )
    block = _select(cohort, plans).anchor_grid
    assert block["anchor_sets"][ANCHOR_GRID_L1_KEY]["ade_m"] == fixed.value
    assert block["fixed_anchor_common_grid_ade_m"] == fixed.value


class _TruthCopier(nn.Module):
    """The model that cannot be wrong: it answers each row with that row's own future.

    Looked up by the anchor state it was shown, which is unique per (flight, anchor) — so
    it is a genuine forward pass, re-anchored bin by bin like any other.
    """

    def __init__(self, table: dict[tuple[float, ...], tuple[torch.Tensor, torch.Tensor]],
                 channels: int):
        super().__init__()
        self.table = table
        self.channels = channels

    def forward(self, history: torch.Tensor) -> StatePrediction:
        rows = [self.table[tuple(row.tolist())]
                for row in anchor_state(history, self.channels)]
        return StatePrediction(
            states=torch.stack([item[0] for item in rows]),
            final_time_s=torch.stack([item[1] for item in rows]),
        )


def _truth_table(plans, config) -> dict:
    table = {}
    for by_airport in plans.values():
        for plan in by_airport.values():
            for batch in plan.batches:
                x, y, _mask, final_time_s, *_rest = unpack_batch(batch.raw_batch)
                anchors = anchor_state(x, len(config.channels))
                for row in range(len(x)):
                    table[tuple(anchors[row].tolist())] = (y[row], final_time_s[row])
    return table


def test_a_model_that_copies_the_truth_scores_zero_at_every_bin(cohort) -> None:
    """The per-bin truth cache really is the truth AFTER that bin's anchor.

    A cache still built at L−1 would make this model look kilometres wrong at 6 km. The
    residual is float32 on a ~20 km chart, not a modelling error.
    """
    config, _series, _normalizer, _model, val_sets = cohort
    plans = build_anchor_grid_validation_plans(val_sets, 8)
    l1_plans = {0.0: {AIRPORT: val.build_validation_batch_plan(val_sets[AIRPORT], 8)}}
    copier = _TruthCopier(
        {**_truth_table(plans, config), **_truth_table(l1_plans, config)},
        len(config.channels),
    )
    for target_m, by_airport in {**l1_plans, **plans}.items():
        plan = by_airport[AIRPORT]
        replay = replay_validation_plan(copier, plan, torch.device("cpu"))
        block = fixed_anchor_common_grid_ade_metrics(
            plan.dataset.series, config, replay.anchors, replay.predicted,
            replay.predicted_time_s, replay.segment_durations_s,
            points=config.validation_common_grid_points, common_truth=plan.common_truth,
        )
        assert block["ade_m"] == pytest.approx(0.0, abs=1.0), target_m


# ── coverage: absent is absent, and empty is refused ────────────────────────

def test_a_flight_without_an_anchor_is_absent_from_that_bins_mean(cohort, monkeypatch,
                                                                  plans) -> None:
    """Absent, not zero, and not diluted.

    The future floor is a module constant (one grid, two consumers), so the cohort is held
    fixed and the FLOOR is moved: at 68.5 s only some of these flights still have 60+ s of
    truth left at 6 km. Their per-flight ADEs do not change — the anchor is chosen on
    remaining path and only then tested — so the bin's value must be exactly their mean.
    """
    config, _series, _normalizer, model, val_sets = cohort
    full = plans[6_000.0][AIRPORT]
    replay = replay_validation_plan(model, full, torch.device("cpu"))
    per_flight = fixed_anchor_common_grid_ade_metrics(
        full.dataset.series, config, replay.anchors, replay.predicted,
        replay.predicted_time_s, replay.segment_durations_s,
        points=config.validation_common_grid_points, common_truth=full.common_truth,
    )["ade_per_flight_m"]

    monkeypatch.setattr(val, "DEFAULT_GRID_MIN_FUTURE_S", 68.5)
    narrow = build_anchor_grid_validation_plans(val_sets, 8)
    kept = {item.dataset_id for item in narrow[6_000.0][AIRPORT].dataset.series}
    assert 0 < len(kept) < len(full.dataset.series)          # the construction bites
    expected = float(np.mean([
        value for item, value in zip(full.dataset.series, per_flight)
        if item.dataset_id in kept
    ]))

    cell = _select(cohort, narrow).anchor_grid["anchor_sets"]["6000"]
    assert cell["flights"] == len(kept)
    assert cell["ade_m"] == pytest.approx(expected, rel=1e-9)
    # ...and neither of the two ways of getting it wrong:
    assert cell["ade_m"] != pytest.approx(float(np.mean(per_flight)))
    zero_filled = float(np.sum([
        value for item, value in zip(full.dataset.series, per_flight)
        if item.dataset_id in kept
    ]) / len(full.dataset.series))
    assert cell["ade_m"] != pytest.approx(zero_filled)


def test_a_bin_no_flight_reaches_is_refused_not_dropped(cohort, monkeypatch) -> None:
    """A five-set metric that quietly became a four-set one is not comparable across arms."""
    _config_, _series, _normalizer, _model, val_sets = cohort
    monkeypatch.setattr(val, "DEFAULT_GRID_MIN_FUTURE_S", 1e6)
    with pytest.raises(ValueError, match="cannot support the anchor grid"):
        build_anchor_grid_validation_plans(val_sets, 8)


def test_the_bins_are_the_shared_grids_and_the_anchors_are_its_rule(cohort, plans) -> None:
    """The metric anchors where `anchor_grid` says, flight by flight — not at a common index."""
    config, _series, _normalizer, _model, val_sets = cohort
    series = val_sets[AIRPORT].series
    profiles = remaining_path_profiles(series)
    assert tuple(plans) == VALIDATION_ANCHOR_GRID_M
    for target_m, by_airport in plans.items():
        expected = anchors_for_bin(
            series, profiles, target_m,
            seq_len=config.seq_len, min_future_s=DEFAULT_GRID_MIN_FUTURE_S,
        )
        dataset = by_airport[AIRPORT].dataset
        assert [item.dataset_id for item in dataset.series] == [
            series[index].dataset_id for index in expected
        ]
        assert [anchor for _row, anchor in dataset.index] == list(expected.values())
    # Per flight, not per bin: the anchors move with the geometry, so they differ.
    assert len({tuple(a for _r, a in by_airport[AIRPORT].dataset.index)
                for by_airport in plans.values()}) == len(plans)


# ── the loop keeps the epoch the GRID prefers ───────────────────────────────

def test_the_loop_keeps_the_epoch_with_the_lowest_grid_mean(cohort, monkeypatch) -> None:
    """Constructed so the two metrics disagree: the grid's best epoch is not L−1's.

    This is the failure the metric exists for. `A0_random_hr8_tv1` early-stopped with its
    best at epoch 10 because the L−1 score stalled while the arm kept improving at every
    other anchor; the loop must follow the grid mean, not the L−1 number beside it.
    """
    config = _config(prediction_output="state", epochs=3, patience=3)
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3), config, airport=AIRPORT
    )
    train_series, val_series, _test = split_by_flight(series, config)
    grid_values = [40.0, 10.0, 25.0]                 # best at epoch 2
    l1_values = [100.0, 900.0, 500.0]                # best at epoch 1
    calls = {"n": 0}

    def scripted(**kwargs) -> ValidationSelection:
        index = calls["n"]
        calls["n"] += 1
        return ValidationSelection(
            metric=CHECKPOINT_SELECTION_ANCHOR_GRID_ADE,
            value=grid_values[index],
            by_airport={AIRPORT: grid_values[index]},
            anchor_grid={
                "anchor_sets": {ANCHOR_GRID_L1_KEY: {
                    "flights": len(val_series), "ade_m": l1_values[index], "by_airport": {},
                }},
                "mean_ade_m": grid_values[index],
                "fixed_anchor_common_grid_ade_m": l1_values[index],
            },
        )

    monkeypatch.setitem(
        val.VALIDATION_SELECTIONS, CHECKPOINT_SELECTION_ANCHOR_GRID_ADE, scripted
    )
    fit = fit_model(train_series, val_series, config, verbose=False)
    assert calls["n"] == 3
    assert fit.best_validation_selection == min(grid_values)
    recorded = [row.validation_selection_value for row in fit.history]
    assert recorded == grid_values
    assert int(np.argmin(recorded)) == 1
    assert int(np.argmin([
        row.validation_anchor_grid["fixed_anchor_common_grid_ade_m"]
        for row in fit.history
    ])) == 0                                        # L−1 would have kept a different epoch


def test_a_two_epoch_train_writes_the_block(tmp_path) -> None:
    config = _config(epochs=2, patience=2)
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3), config, airport=AIRPORT
    )
    torch.manual_seed(0)
    train(series, config, output_dir=tmp_path, data_provenance=_provenance(), verbose=False)
    history = json.loads((tmp_path / "history.json").read_text())["history"]
    assert len(history) == 2
    for row in history:
        block = row["validation_anchor_grid"]
        assert row["validation_selection_metric"] == CHECKPOINT_SELECTION_ANCHOR_GRID_ADE
        assert list(block["anchor_sets"]) == [
            ANCHOR_GRID_L1_KEY, "16000", "12000", "8000", "6000"
        ]
        assert all(cell["flights"] > 0 for cell in block["anchor_sets"].values())
        assert row["validation_selection_value"] == pytest.approx(block["mean_ade_m"])


def test_the_fixed_anchor_metric_records_no_grid_block(tmp_path) -> None:
    """Every other metric leaves the block empty — like `latent` and `procedure` before it."""
    config = _config(
        epochs=1, patience=1,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    )
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3), config, airport=AIRPORT
    )
    torch.manual_seed(0)
    train(series, config, output_dir=tmp_path, data_provenance=_provenance(), verbose=False)
    history = json.loads((tmp_path / "history.json").read_text())["history"]
    assert all(row["validation_anchor_grid"] == {} for row in history)


# ── the config axis and the run name ────────────────────────────────────────

def test_the_metric_is_a_selectable_value_and_an_unknown_one_is_refused() -> None:
    assert CHECKPOINT_SELECTION_ANCHOR_GRID_ADE in CHECKPOINT_SELECTION_METRICS
    assert TSConfig().checkpoint_selection_metric == CHECKPOINT_SELECTION_COMMON_GRID_ADE
    with pytest.raises(ValueError, match="unknown checkpoint_selection_metric"):
        TSConfig(checkpoint_selection_metric="anchor-grid")


def test_the_metric_names_the_run() -> None:
    """A run selected on the grid is a DIFFERENT run, and the name grammar says so.

    `checkpoint_selection_metric` is already a `META_FIELDS` entry, so a non-default value
    is spelled `select=…` in the display name — and on a run with more than
    ``_MAX_LISTED_META`` other deviations it folds into ``+N more``, where `run_slug`
    hashes it so two such runs cannot be handed one directory.
    """
    assert "checkpoint_selection_metric" in rn.META_FIELDS
    spelled = TSConfig(
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ANCHOR_GRID_ADE
    ).to_dict()
    assert f"select={CHECKPOINT_SELECTION_ANCHOR_GRID_ADE}" in rn.run_display_name(spelled)

    grid = _config().to_dict()
    fixed = grid | {"checkpoint_selection_metric": CHECKPOINT_SELECTION_COMMON_GRID_ADE}
    assert rn.run_slug(fixed) != rn.run_slug(grid)
    assert rn.run_display_name(fixed) != rn.run_display_name(grid)
    assert (
        f"select={CHECKPOINT_SELECTION_ANCHOR_GRID_ADE}" in rn.meta_items(grid)
        or dict(rn.dropped_meta_diffs(grid)).get("checkpoint_selection_metric")
        == CHECKPOINT_SELECTION_ANCHOR_GRID_ADE
    )
