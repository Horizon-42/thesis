"""`random_train_anchor_sampling`: WHERE along the approach a random train anchor is drawn.

A0.b of `docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md` §2.4c. The
uniform policy draws over a flight's admissible SAMPLES, which is uniform over time and so
biased toward the runway; the KRDU A0-random arm's 6 km validation set improved for 180
epochs while L−1 and 12 km degraded after epoch 10, which is what that skew looks like.

What the tests hold:

* the strata ARE `anchor_grid`'s bins read as edges — nothing restates 2/4/6/8/12/16/20 km,
  and a sample's stratum is the same remaining path the bins are chosen from;
* the stratified draw is uniform over the strata a flight HAS anchors in, the uniform draw
  is proportional to how many samples sit in each — the two laws, measured;
* both draws are the flight's own per-epoch hash: same seed and epoch, same anchor;
* admissibility is untouched, so the two policies store the same anchors for the same
  cohort (the 20 s membership guard behaves identically);
* every epoch records the realised distribution, under BOTH policies, and the counts sum to
  the flights — one draw per flight is the invariant that makes the numbers readable.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from anchor_grid import (
    DEFAULT_ANCHOR_GRID_KM,
    REMAINING_PATH_STRATA_EDGES_M,
    REMAINING_PATH_STRATA_LABELS,
    remaining_path_strata,
)
from approach_difficulty import remaining_path_profile_m
from config import (
    CONTROL_RECIPE_SIMPLE_V3,
    RANDOM_TRAIN_ANCHOR_SAMPLING_STRATA,
    RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM,
    RANDOM_TRAIN_ANCHOR_SAMPLINGS,
    TSConfig,
    control_recipe_overrides,
    control_simple_v1_overrides,
    recipe_settings,
)
from data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from dataset import (
    FixedAnchorTrajectoryWindows,
    Normalizer,
    RandomAnchorTrajectoryWindows,
    StratifiedRandomAnchorTrajectoryWindows,
    build_series,
    training_window_class,
)
from run_naming import run_display_name
from synthetic import synthetic_arrivals
from train import train

AIRPORT, RUNWAY = "KRDU", "05L"


# ── the strata are the grid ──────────────────────────────────────────────────

def test_the_strata_edges_are_the_measurement_grid_read_as_edges():
    assert REMAINING_PATH_STRATA_EDGES_M == tuple(
        sorted(float(km) * 1000.0 for km in DEFAULT_ANCHOR_GRID_KM)
    )
    # Seven edges, eight strata: the two open ends are strata of their own, so every
    # admissible anchor lands in exactly one.
    assert len(REMAINING_PATH_STRATA_LABELS) == len(REMAINING_PATH_STRATA_EDGES_M) + 1
    assert REMAINING_PATH_STRATA_LABELS[0] == "<2km"
    assert REMAINING_PATH_STRATA_LABELS[-1] == ">=20km"


def test_a_samples_stratum_is_the_remaining_path_the_bins_are_chosen_from():
    config = _config()
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=2, seed=3), config, airport=AIRPORT
    )
    for item in series:
        strata = remaining_path_strata(item)
        profile = remaining_path_profile_m(item)
        assert strata.shape == profile.shape
        assert np.array_equal(
            strata, np.digitize(profile, REMAINING_PATH_STRATA_EDGES_M)
        )


# ── the config axis ──────────────────────────────────────────────────────────

def test_the_axis_defaults_to_todays_policy():
    assert RANDOM_TRAIN_ANCHOR_SAMPLINGS == (
        RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM, RANDOM_TRAIN_ANCHOR_SAMPLING_STRATA
    )
    assert TSConfig().random_train_anchor_sampling == RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM


def test_an_unknown_sampling_policy_is_refused():
    with pytest.raises(ValueError, match="unknown random_train_anchor_sampling"):
        TSConfig(random_train_anchor=True, random_train_anchor_sampling="stratified")


def test_strata_sampling_is_refused_without_random_anchors():
    with pytest.raises(ValueError, match="random_train_anchor=False draws"):
        TSConfig(random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_STRATA)


def test_every_named_recipe_pins_the_default_as_a_literal():
    assert control_simple_v1_overrides()["random_train_anchor_sampling"] == (
        RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM
    )
    for recipe in ("simple-v1", "simple-v1-lag", "simple-v2", CONTROL_RECIPE_SIMPLE_V3):
        assert control_recipe_overrides(recipe)["random_train_anchor_sampling"] == (
            RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM
        )


def test_the_policy_names_the_run_and_selects_the_window_class():
    plain = TSConfig(random_train_anchor=True)
    assert "anchors=" not in run_display_name(plain.to_dict())
    assert training_window_class(plain) is RandomAnchorTrajectoryWindows
    assert training_window_class(TSConfig()) is FixedAnchorTrajectoryWindows
    stratified = replace(
        plain, random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_STRATA
    )
    assert "anchors=remaining-path-strata" in run_display_name(stratified.to_dict())
    assert training_window_class(stratified) is StratifiedRandomAnchorTrajectoryWindows


# ── the two draw laws ────────────────────────────────────────────────────────

def _config(**overrides) -> TSConfig:
    settings = dict(
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=2, patience=2, batch_size=8, dropout=0.0,
        val_fraction=0.25, test_fraction=0.25,
        random_train_anchor=True, random_train_anchor_min_future_s=4.0,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def _windows(sampling: str, *, n_flights: int = 1):
    config = _config(random_train_anchor_sampling=sampling)
    series, report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3),
        config, airport=AIRPORT,
    )
    assert report.built == n_flights, report.format()
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    return training_window_class(config)(series, config, normalizer)


#: One flight's anchors, hand-assigned to three strata in wildly unequal numbers — a
#: near-runway-heavy track, which is exactly the shape a real short arrival has.
_HAND_STRATA = (0, 4, 7)
_HAND_TAIL_SIZES = (10, 3)


def _hand_built(sampling: str):
    windows = _windows(sampling)
    stored = len(windows.index)
    sizes = (stored - sum(_HAND_TAIL_SIZES), *_HAND_TAIL_SIZES)
    windows.anchor_strata = np.repeat(_HAND_STRATA, sizes)
    return windows, dict(zip(_HAND_STRATA, sizes))


def _drawn_strata(windows, epochs: int = 3000) -> Counter:
    counts: Counter = Counter()
    for epoch in range(1, epochs + 1):
        counts.update(windows.anchor_strata[windows.epoch_indices(epoch)].tolist())
    return counts


def test_the_stratified_draw_is_uniform_over_the_strata_a_flight_has():
    windows, sizes = _hand_built(RANDOM_TRAIN_ANCHOR_SAMPLING_STRATA)
    draws = _drawn_strata(windows)
    assert set(draws) == set(_HAND_STRATA), "a stratum the flight has no anchors in was drawn"
    total = sum(draws.values())
    for stratum in _HAND_STRATA:
        assert draws[stratum] / total == pytest.approx(1 / len(_HAND_STRATA), abs=0.03)
    # ...and it is emphatically NOT the sample-count law the uniform policy follows.
    assert draws[_HAND_STRATA[0]] / total < 0.5 < sizes[_HAND_STRATA[0]] / len(windows.index)


def test_the_uniform_draw_is_proportional_to_the_samples_in_each_stratum():
    windows, sizes = _hand_built(RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM)
    draws = _drawn_strata(windows)
    total = sum(draws.values())
    stored = len(windows.index)
    for stratum in _HAND_STRATA:
        assert draws[stratum] / total == pytest.approx(sizes[stratum] / stored, abs=0.03)


@pytest.mark.parametrize("sampling", RANDOM_TRAIN_ANCHOR_SAMPLINGS)
def test_the_draw_is_the_flights_own_per_epoch_hash(sampling: str):
    """Same seed and epoch, same anchors — from a window set built independently."""
    first, second = _windows(sampling, n_flights=4), _windows(sampling, n_flights=4)
    assert np.array_equal(first.epoch_indices(11), second.epoch_indices(11))
    assert not np.array_equal(first.epoch_indices(11), first.epoch_indices(12))


def test_the_two_policies_store_the_same_anchors_for_the_same_cohort():
    """Admissibility is not part of this axis: the 20 s guard admits the same population."""
    uniform = _windows(RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM, n_flights=4)
    strata = _windows(RANDOM_TRAIN_ANCHOR_SAMPLING_STRATA, n_flights=4)
    assert uniform.index == strata.index
    assert np.array_equal(uniform.anchor_strata, strata.anchor_strata)
    assert uniform.eligible_candidate_anchors == strata.eligible_candidate_anchors
    assert uniform.sampling_version != strata.sampling_version


def test_the_stored_strata_are_the_strata_of_the_stored_anchors():
    windows = _windows(RANDOM_TRAIN_ANCHOR_SAMPLING_STRATA, n_flights=4)
    expected = [
        int(remaining_path_strata(windows.series[s_idx])[anchor])
        for s_idx, anchor in windows.index
    ]
    assert windows.anchor_strata.tolist() == expected


# ── the per-epoch record ─────────────────────────────────────────────────────

@pytest.mark.parametrize("sampling", RANDOM_TRAIN_ANCHOR_SAMPLINGS)
def test_the_epoch_record_counts_the_realised_anchors_per_stratum(sampling: str):
    windows = _windows(sampling, n_flights=4)
    statistics = windows.anchor_statistics(5)
    block = statistics["remaining_path_strata"]
    assert list(block) == list(REMAINING_PATH_STRATA_LABELS)
    # One draw per flight: the counts are a distribution over the epoch's flights, so they
    # sum to it. A count that summed to the WINDOWS would be reporting the population.
    assert sum(block.values()) == len(windows.eligible_series) == statistics["samples"]


def test_a_fixed_anchor_window_set_records_no_strata_block():
    """One anchor per flight by definition: there is no distribution to report."""
    config = TSConfig(seq_len=8, device="cpu", horizon_mode="normalized")
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=2, seed=3), config, airport=AIRPORT
    )
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    assert "remaining_path_strata" not in windows.anchor_statistics(1)


@pytest.mark.parametrize("sampling", RANDOM_TRAIN_ANCHOR_SAMPLINGS)
def test_a_two_epoch_run_trains_and_records_the_distribution(
    tmp_path: Path, sampling: str
):
    settings = dict(recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False))
    settings.update(
        n_segments=8, control_imitation_loss_weight=0.0,
        seq_len=8, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu",
        epochs=2, patience=2, batch_size=8, dropout=0.0,
        control_rollout_integrator_dt_s=0.5,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        random_train_anchor=True, random_train_anchor_min_future_s=4.0,
        random_train_anchor_sampling=sampling,
    )
    config = TSConfig(**settings)
    series, report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3), config, airport=AIRPORT
    )
    assert report.built == 8, report.format()
    torch.manual_seed(0)
    train(
        series, config, output_dir=tmp_path / sampling,
        data_provenance={
            "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
            "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64,
                           "source_records": []}],
        },
        verbose=False,
    )
    summary = json.loads((tmp_path / sampling / "history.json").read_text())
    flights = summary["flights"]["train"]
    for epoch in summary["history"]:
        block = epoch["train_anchor_sampling"]["remaining_path_strata"]
        assert sum(block.values()) == flights
    contract = summary["training_anchor_contract"]
    assert contract["sampling_version"] == (
        "per-flight-hash-v3-remaining-path-strata"
        if sampling == RANDOM_TRAIN_ANCHOR_SAMPLING_STRATA
        else "per-flight-hash-v2-output-eligibility"
    )
