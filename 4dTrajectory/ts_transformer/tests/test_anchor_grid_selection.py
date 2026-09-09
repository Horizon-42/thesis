"""`anchor-grid-common-grid-ade`: what the number IS, and what it is not.

The metric averages the SAME common-grid ADE over the anchor sets a cohort can support —
L−1 plus whichever of `anchor_grid`'s candidate remaining-path bins clear the coverage gate
(A1 of `docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`). Five things
decide whether that number means what it says, and each is a test here:

* every anchor set contributes its value AND its coverage, and the published mean is the
  mean of exactly the surviving sets (equal weight per SET, never per flight-anchor pair);
* a bin the cohort cannot cover is DROPPED and recorded, not averaged in at 37 % of the
  split — and the whole metric is refused when too few bins survive to be a curve;
* the L−1 set is bit-for-bit the value `fixed-anchor-common-grid-ade` selects on, so the
  two metrics stay comparable and the L−1 veto stays readable;
* a flight that has no admissible anchor at a bin is ABSENT from that bin's mean — never
  scored zero, never carried in from another bin;
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

import ts_transformer.run_naming as rn
import ts_transformer.validation as val
from ts_transformer.anchor_grid import (
    DEFAULT_GRID_MIN_FUTURE_S,
    PARTIAL_COVERAGE,
    VALIDATION_ANCHOR_GRID_KM,
    VALIDATION_ANCHOR_GRID_M,
    anchors_for_bin,
    bin_label,
    remaining_path_profiles,
)
from ts_transformer.batch_contract import anchor_state, unpack_batch
from ts_transformer.config import (
    CHECKPOINT_SELECTION_ANCHOR_GRID_ADE,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CHECKPOINT_SELECTION_METRICS,
    TSConfig,
)
from ts_transformer.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from ts_transformer.dataset import ExplicitAnchorTrajectoryWindows, Normalizer, build_series
from ts_transformer.fixed_anchor_validation import FIXED_ANCHOR_LABEL, fixed_anchor_common_grid_ade_metrics
from ts_transformer.models import build_model
from ts_transformer.outputs.state.model import StatePrediction
from ts_transformer.splits import split_by_flight
from ts_transformer.synthetic import synthetic_arrivals
from ts_transformer.train import fit_model, train
from ts_transformer.validation import (
    ANCHOR_GRID_L1_KEY,
    MINIMUM_ANCHOR_GRID_BINS,
    ValidationSelection,
    build_anchor_grid_validation_plans,
    replay_validation_plan,
    validation_datasets,
)

AIRPORT, RUNWAY = "KRDU", "05L"
SECOND_AIRPORT, SECOND_RUNWAY = "KSJC", "30L"
#: The common grid and the model's own node grid deliberately coincide, so the
#: truth-copying model below is exact rather than exact-up-to-resampling.
POINTS = 8
#: This synthetic cohort covers every candidate bin, so nothing is gated away by default;
#: the tests that DO want a drop move the future FLOOR rather than the threshold.
ALL_BINS = [ANCHOR_GRID_L1_KEY] + [bin_label(value) for value in VALIDATION_ANCHOR_GRID_M]


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


def _provenance(*airports: str) -> dict:
    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [
            {"airport": airport, "arrival_manifest_sha256": "a" * 64, "source_records": []}
            for airport in (airports or (AIRPORT,))
        ],
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
        val_by_airport={airport: 0.0 for airport in val_sets},
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
    assert block["minimum_coverage"] == PARTIAL_COVERAGE
    assert block["dropped_bins"] == []
    assert list(block["anchor_sets"]) == ALL_BINS
    for name, cell in block["anchor_sets"].items():
        assert cell["flights"] == 12, name          # this cohort reaches every bin
        assert cell["ade_m"] > 0.0, name
        assert cell["by_airport"][AIRPORT]["flights"] == 12
    # Five different numbers: the sets are scored independently, not copied from L−1.
    assert len({cell["ade_m"] for cell in block["anchor_sets"].values()}) == len(ALL_BINS)


def test_the_value_is_the_mean_of_the_recorded_anchor_sets(cohort, plans) -> None:
    """Equal weight per SET. Pooling (flight, anchor) pairs would weight bins by coverage."""
    selection = _select(cohort, plans)
    recorded = [cell["ade_m"] for cell in selection.anchor_grid["anchor_sets"].values()]
    assert len(recorded) == len(ALL_BINS)
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


def _truth_table(bins: dict, config) -> dict:
    table = {}
    for by_airport in bins.values():
        for plan in by_airport.values():
            for batch in plan.batches:
                x, y, _mask, final_time_s, *_rest = unpack_batch(batch.raw_batch)
                anchors = anchor_state(x, len(config.channels))
                for row in range(len(x)):
                    table[tuple(anchors[row].tolist())] = (y[row], final_time_s[row])
    return table


def test_a_model_that_copies_the_truth_scores_zero_at_every_bin(cohort, plans) -> None:
    """The per-bin truth cache really is the truth AFTER that bin's anchor.

    A cache still built at L−1 would make this model look kilometres wrong at 6 km. The
    residual is float32 on a ~20 km chart, not a modelling error.
    """
    config, _series, _normalizer, _model, val_sets = cohort
    l1_bins = {0.0: {AIRPORT: val.build_validation_batch_plan(val_sets[AIRPORT], 8)}}
    copier = _TruthCopier(
        {**_truth_table(plans.bins, config), **_truth_table(l1_bins, config)},
        len(config.channels),
    )
    for target_m, by_airport in {**l1_bins, **plans.bins}.items():
        plan = by_airport[AIRPORT]
        replay = replay_validation_plan(copier, plan, torch.device("cpu"))
        block = fixed_anchor_common_grid_ade_metrics(
            plan.dataset.series, config, replay.anchors, replay.predicted,
            replay.predicted_time_s, replay.segment_durations_s,
            points=config.validation_common_grid_points, common_truth=plan.common_truth,
        )
        assert block["ade_m"] == pytest.approx(0.0, abs=1.0), target_m


def test_a_block_names_the_anchor_it_was_scored_at(cohort, plans) -> None:
    """An "anchor" field that always says `L-1` is read as a claim, so it must be true."""
    config, _series, _normalizer, model, val_sets = cohort
    l1 = val.build_validation_batch_plan(val_sets[AIRPORT], 8)
    replay = replay_validation_plan(model, l1, torch.device("cpu"))
    default = fixed_anchor_common_grid_ade_metrics(
        l1.dataset.series, config, replay.anchors, replay.predicted,
        replay.predicted_time_s, replay.segment_durations_s,
        points=config.validation_common_grid_points, common_truth=l1.common_truth,
    )
    assert default["anchor"] == FIXED_ANCHOR_LABEL

    plan = plans.bins[8_000.0][AIRPORT]
    replay = replay_validation_plan(model, plan, torch.device("cpu"))
    named = fixed_anchor_common_grid_ade_metrics(
        plan.dataset.series, config, replay.anchors, replay.predicted,
        replay.predicted_time_s, replay.segment_durations_s,
        points=config.validation_common_grid_points, common_truth=plan.common_truth,
        anchor_label="remaining path 8km",
    )
    assert named["anchor"] == "remaining path 8km"


# ── coverage: absent is absent, thin is dropped, empty is refused ───────────

def test_a_flight_without_an_anchor_is_absent_from_that_bins_mean(cohort, monkeypatch,
                                                                  plans) -> None:
    """Absent, not zero, and not diluted.

    The future floor is a module constant (one grid, two consumers), so the cohort is held
    fixed and the FLOOR is moved: at 66 s only some of these flights still have enough
    truth left at 6 km. Their per-flight ADEs do not change — the anchor is chosen on
    remaining path and only then tested — so the bin's value must be exactly their mean.
    """
    config, _series, _normalizer, model, val_sets = cohort
    full = plans.bins[6_000.0][AIRPORT]
    replay = replay_validation_plan(model, full, torch.device("cpu"))
    per_flight = fixed_anchor_common_grid_ade_metrics(
        full.dataset.series, config, replay.anchors, replay.predicted,
        replay.predicted_time_s, replay.segment_durations_s,
        points=config.validation_common_grid_points, common_truth=full.common_truth,
    )["ade_per_flight_m"]

    # 66 s still leaves 9 of these 12 flights a 6 km anchor — 75 %, above the
    # coverage gate — so the bin survives and its MEAN is what is under test.
    monkeypatch.setattr(val, "DEFAULT_GRID_MIN_FUTURE_S", 66.0)
    narrow = build_anchor_grid_validation_plans(val_sets, 8)
    kept = {item.dataset_id for item in narrow.bins[6_000.0][AIRPORT].dataset.series}
    assert 0 < len(kept) < len(full.dataset.series)          # the construction bites
    expected = float(np.mean([
        value for item, value in zip(full.dataset.series, per_flight)
        if item.dataset_id in kept
    ]))

    cell = _select(cohort, narrow).anchor_grid["anchor_sets"]["6km"]
    assert cell["flights"] == len(kept)
    assert cell["ade_m"] == pytest.approx(expected, rel=1e-6)
    # ...and neither of the two ways of getting it wrong:
    assert cell["ade_m"] != pytest.approx(float(np.mean(per_flight)))
    zero_filled = float(np.sum([
        value for item, value in zip(full.dataset.series, per_flight)
        if item.dataset_id in kept
    ]) / len(full.dataset.series))
    assert cell["ade_m"] != pytest.approx(zero_filled)


def test_a_bin_the_cohort_cannot_cover_is_dropped_and_recorded(cohort, monkeypatch,
                                                               capsys) -> None:
    """The 16 km bin covers 37 % of the real KRDU val cohort — a long-haul SUBCOHORT.

    An equal-weighted fifth of the selection value must not come from it, so a bin under
    `PARTIAL_COVERAGE` is dropped, printed and recorded; the mean is over what survives.
    Here the future floor stands in for the geometry that thins the real 16 km bin.
    """
    _config_, _series, _normalizer, _model, val_sets = cohort
    # 69 s leaves only 5 of these 12 flights a 6 km anchor: 41.7 %, under the gate.
    monkeypatch.setattr(val, "DEFAULT_GRID_MIN_FUTURE_S", 69.0)
    gated = build_anchor_grid_validation_plans(val_sets, 8)
    assert 6_000.0 not in gated.bins
    assert [item["bin"] for item in gated.dropped] == ["6km"]
    dropped = gated.dropped[0]
    assert dropped["coverage"][AIRPORT] < PARTIAL_COVERAGE
    assert dropped["flights"][AIRPORT] == len(anchors_for_bin(
        val_sets[AIRPORT].series,
        remaining_path_profiles(val_sets[AIRPORT].series),
        6_000.0, seq_len=8, min_future_s=69.0,
    ))
    assert "50%" in dropped["reason"] and "partial" in dropped["reason"]
    assert "grid drop" in capsys.readouterr().out

    block = _select(cohort, gated).anchor_grid
    assert list(block["anchor_sets"]) == [ANCHOR_GRID_L1_KEY, "16km", "12km", "8km"]
    assert block["grid_km"] == [16.0, 12.0, 8.0]
    assert [item["bin"] for item in block["dropped_bins"]] == ["6km"]
    assert block["mean_ade_m"] == pytest.approx(float(np.mean(
        [cell["ade_m"] for cell in block["anchor_sets"].values()]
    )))


def test_a_grid_too_thin_to_be_a_curve_is_refused(cohort, monkeypatch) -> None:
    """Fewer than two surviving bins is the L−1 metric with a companion, not an average."""
    _config_, _series, _normalizer, _model, val_sets = cohort
    # 130 s leaves only the 16 km bin: one surviving set is not a curve.
    monkeypatch.setattr(val, "DEFAULT_GRID_MIN_FUTURE_S", 130.0)
    with pytest.raises(ValueError, match="candidate bins clear"):
        build_anchor_grid_validation_plans(val_sets, 8)
    assert MINIMUM_ANCHOR_GRID_BINS == 2


def test_the_bins_are_the_shared_grids_and_the_anchors_are_its_rule(cohort, plans) -> None:
    """The metric anchors where `anchor_grid` says, flight by flight — not at a common index."""
    config, _series, _normalizer, _model, val_sets = cohort
    series = val_sets[AIRPORT].series
    profiles = remaining_path_profiles(series)
    assert tuple(plans.bins) == VALIDATION_ANCHOR_GRID_M
    for target_m, by_airport in plans.bins.items():
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
                for by_airport in plans.bins.values()}) == len(plans.bins)


def test_the_grid_windows_carry_no_control_supervision() -> None:
    """A replay set builds no imitation or heading-rate target.

    Not an optimisation: the imitation target is ANCHOR-BOUND, so a `fitted` run whose grid
    rows built one would silently carry the inverse-dynamics teacher at these anchors
    instead of its own table. Nothing in the replay reads either key.
    """
    config = _config(
        prediction_output="control",
        control_duration_parameterization="uniform",
        control_state_supervision_clock="observed",
        control_state_loss_grid="native-segment-endpoints",
        control_state_objective="true-time-position",
        control_rollout_integrator_dt_s=0.5,
        control_imitation_loss_weight=1.0,
        control_heading_rate_loss_weight=1.0,
    )
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=4, seed=3), config, airport=AIRPORT
    )
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    anchors = {item.dataset_id: 9 for item in series}
    supervised, replay_only = (
        ExplicitAnchorTrajectoryWindows(
            series, config, normalizer, anchors=anchors, supervision=flag
        )
        for flag in (True, False)
    )
    keys = {
        "reference_controls", "reference_heading_rate_dps", "reference_heading_rate_weight",
    }
    rows = np.array([0, 1])
    assert keys <= set(unpack_batch(supervised.batch(rows))[5])
    assert not (keys & set(unpack_batch(replay_only.batch(rows))[5]))
    # ...and the flight state the replay DOES read is still there.
    assert "max_thrust_n" in unpack_batch(replay_only.batch(rows))[5]


# ── more than one airport: the macro is a macro ─────────────────────────────

def test_the_set_value_is_the_airport_macro_of_flight_means() -> None:
    """Two airports, deliberately different sizes: mean over AIRPORTS, not over flights.

    A flight mean over the pool would let the bigger airport carry the metric — the same
    rule `fixed-anchor-common-grid-ade` already follows, which is why the L−1 term of this
    block is bit-for-bit that metric's value.
    """
    config = _config()
    series = []
    for airport, runway, count in (
        (AIRPORT, RUNWAY, 10), (SECOND_AIRPORT, SECOND_RUNWAY, 4)
    ):
        built, _report = build_series(
            synthetic_arrivals(airport, runway, n_flights=count, seed=3),
            config, airport=airport,
        )
        series += built
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    torch.manual_seed(0)
    model = build_model(config, normalizer).eval()
    val_sets = validation_datasets(series, config, normalizer)
    assert {airport: len(dataset) for airport, dataset in val_sets.items()} == {
        AIRPORT: 10, SECOND_AIRPORT: 4
    }
    plans = build_anchor_grid_validation_plans(val_sets, 8)
    selection = val.VALIDATION_SELECTIONS[CHECKPOINT_SELECTION_ANCHOR_GRID_ADE](
        model=model, val_sets=val_sets, normalizer=normalizer, config=config,
        device=torch.device("cpu"),
        val_by_airport={AIRPORT: 0.0, SECOND_AIRPORT: 0.0},
        anchor_grid_plans=plans,
    )
    sets = selection.anchor_grid["anchor_sets"]
    for name, cell in sets.items():
        per_airport = [item["ade_m"] for item in cell["by_airport"].values()]
        assert len(per_airport) == 2, name
        assert cell["ade_m"] == pytest.approx(float(np.mean(per_airport))), name
        assert cell["flights"] == sum(
            item["flights"] for item in cell["by_airport"].values()
        )
    # The two airports really do score differently, so a flight-pooled mean would differ.
    l1 = sets[ANCHOR_GRID_L1_KEY]["by_airport"]
    assert l1[AIRPORT]["ade_m"] != pytest.approx(l1[SECOND_AIRPORT]["ade_m"])
    assert selection.value == pytest.approx(
        float(np.mean([cell["ade_m"] for cell in sets.values()]))
    )
    for airport in (AIRPORT, SECOND_AIRPORT):
        assert selection.by_airport[airport] == pytest.approx(float(np.mean(
            [cell["by_airport"][airport]["ade_m"] for cell in sets.values()]
        )))


# ── the loop keeps the epoch the GRID prefers ───────────────────────────────

def test_the_loop_keeps_the_epoch_with_the_lowest_grid_mean() -> None:
    """A real training run where the two metrics disagree about which epoch to keep.

    Not a scripted stub: 24 synthetic arrivals, a state model at lr 3e-3 for 10 epochs.
    The L−1 ADE and the grid mean bottom out at DIFFERENT epochs, and the loop follows the
    grid. This is the failure the metric exists for — `A0_random_hr8_tv1` early-stopped
    with its best at epoch 10 because the L−1 score stalled while the arm kept improving at
    every other anchor.
    """
    config = _config(epochs=10, patience=10, learning_rate=1e-2, batch_size=16, seed=1)
    # Data seed 4: under the one-contract terminal supervision (2026-09-09, review A-3) seed
    # 3's run improves on both metrics through epoch 10, so it can no longer show the
    # disagreement; seed 4 separates them for every torch seed 0-5 (grid 7, L-1 9).
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=24, seed=4), config, airport=AIRPORT
    )
    train_series, val_series, _test = split_by_flight(series, config)
    torch.manual_seed(0)
    fit = fit_model(train_series, val_series, config, verbose=False)

    grid = [row.validation_selection_value for row in fit.history]
    l1 = [row.validation_anchor_grid["fixed_anchor_common_grid_ade_m"]
          for row in fit.history]
    assert len(grid) == 10
    # Deterministic under this seed: the grid bottoms out at epoch 7, L-1 at epoch 9.
    assert (int(np.argmin(grid)) + 1, int(np.argmin(l1)) + 1) == (7, 9)
    assert fit.best_validation_selection == pytest.approx(min(grid))
    assert fit.best_validation_selection != pytest.approx(grid[int(np.argmin(l1))])
    # Each epoch's value really is the mean of its own recorded sets.
    for row in fit.history:
        assert row.validation_selection_value == pytest.approx(float(np.mean([
            cell["ade_m"] for cell in row.validation_anchor_grid["anchor_sets"].values()
        ])))


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
        assert list(block["anchor_sets"]) == ALL_BINS
        assert all(cell["flights"] > 0 for cell in block["anchor_sets"].values())
        assert block["dropped_bins"] == []
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


def _oracle_control(**overrides) -> dict:
    return dict(
        prediction_output="control",
        control_duration_parameterization="uniform",
        control_state_supervision_clock="observed",
        control_state_loss_grid="fixed-dt",
        control_state_objective="normalized-mse",
        control_rollout_integrator_dt_s=0.5,
        **overrides,
    )


def test_an_oracle_input_is_refused_because_the_grid_re_reads_it_at_every_anchor() -> None:
    """`cta=given` and `intent=truth-…` read the FUTURE, and the grid reads it again per bin.

    The A0 runner refuses exactly these two checkpoints for the same reason; selecting an
    epoch on them would be selecting on how fast the oracle converges.
    """
    with pytest.raises(ValueError, match="cta_conditioning=given reads the future"):
        _config(**_oracle_control(cta_conditioning="given"))
    with pytest.raises(ValueError, match="reads the FUTURE"):
        _config(**_oracle_control(intent_conditioning="truth-join-duration"))
    # ...and both stay allowed under the L−1 metric, which reads the oracle once.
    for overrides in (
        {"cta_conditioning": "given"}, {"intent_conditioning": "truth-join-duration"},
    ):
        _config(
            checkpoint_selection_metric=CHECKPOINT_SELECTION_COMMON_GRID_ADE,
            **_oracle_control(**overrides),
        )


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


# ── review 2026-09-09 A-2: the truth is taken where the windows were anchored ──

def test_the_l1_plan_truth_follows_a_common_anchor_floor(cohort) -> None:
    """`run_ts_history_ablation.py` trains every candidate seq_len at `max(L) - 1`; the
    L-1 plan's truth used to be built at `seq_len - 1` regardless, i.e. `(max L - L) * dt`
    EARLIER than the anchor the prediction was made from, which corrupted the kept epoch
    and every metric that runner published."""
    config, series, normalizer, _model, _val_sets = cohort
    floor = config.seq_len - 1 + 3
    floored = validation_datasets(series, config, normalizer, minimum_anchor_index=floor)[AIRPORT]
    assert floored.anchor == floor
    assert floored.anchor_indices == [floor] * len(floored.series)
    assert all(anchor == floor for _s, anchor in floored.index)
    plan = val.build_validation_batch_plan(floored, 8)
    _truth, durations, _progress = plan.common_truth
    expected = [item.supervision_times[-1] - item.times[floor] for item in floored.series]
    assert np.allclose(durations, expected)
    # ...and the unfloored set anchors at L-1, where it always did.
    plain = validation_datasets(series, config, normalizer)[AIRPORT]
    assert plain.anchor == config.seq_len - 1
