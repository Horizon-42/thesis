"""`random_train_anchor_sampling`: WHERE along the approach a random train anchor is drawn.

A0.b of `docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md` §2.4c. The
default `uniform` policy draws over a flight's admissible SAMPLES, i.e. uniformly in TIME;
pooled over flights that over-weights the near end relative to the stored anchor population,
because every flight gets one draw whatever its length and a kilometre near the runway holds
more samples than a kilometre at 25 km. `remaining-path-uniform` places the draw uniformly
across the flight's own admissible remaining-path span instead.

What the tests hold:

* the strata are `anchor_grid`'s bins read as edges — nothing restates 2/4/6/8/12/16/20 km —
  and each label owns a HAND-PINNED interval, not a restatement of `np.digitize`;
* the two draw LAWS, measured on a flight whose anchors are dense near the runway and sparse
  far out: `uniform` follows the sample density, `remaining-path-uniform` follows the span;
* both draws are the flight's own per-epoch hash: same seed and epoch, same anchor;
* admissibility is untouched, so the two policies store the same anchors for the same cohort
  (the 20 s membership guard behaves identically);
* every epoch records the realised distribution AND the population it was drawn from, under
  both policies — the drawn counts sum to the flights, the population counts to the stored
  anchors, and neither means anything without the other.
"""

from __future__ import annotations

import json
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
from anchor_strata import remaining_path_uniform_offset
from approach_difficulty import remaining_path_profile_m
from config import (
    CONTROL_RECIPE_SIMPLE_V3,
    RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM,
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
    RemainingPathUniformAnchorTrajectoryWindows,
    build_series,
    training_window_class,
)
from run_naming import run_display_name
from synthetic import synthetic_arrivals
from train import train

AIRPORT, RUNWAY = "KRDU", "05L"


# ── the strata are the grid, and each label owns an interval ─────────────────

def test_the_strata_edges_are_the_measurement_grid_read_as_edges():
    assert REMAINING_PATH_STRATA_EDGES_M == tuple(
        sorted(float(km) * 1000.0 for km in DEFAULT_ANCHOR_GRID_KM)
    )
    # Seven edges, eight strata: the two open ends are strata of their own, so every
    # admissible anchor is counted in exactly one column.
    assert len(REMAINING_PATH_STRATA_LABELS) == len(REMAINING_PATH_STRATA_EDGES_M) + 1


@pytest.mark.parametrize(
    "remaining_path_m, label",
    [
        (0.0, "<2km"),
        (1999.0, "<2km"),
        (2000.0, "2km-4km"),        # an edge belongs to the stratum ABOVE it
        (7999.9, "6km-8km"),
        (8000.0, "8km-12km"),
        (19999.0, "16km-20km"),
        (20000.0, ">=20km"),
        (123_000.0, ">=20km"),      # the far end is open, and really is that wide
    ],
)
def test_each_label_owns_its_interval(remaining_path_m: float, label: str):
    """Pinned by hand, not by restating `np.digitize`: a silently shifted edge would move
    every published histogram column by one stratum."""
    index = int(np.digitize([remaining_path_m], REMAINING_PATH_STRATA_EDGES_M)[0])
    assert REMAINING_PATH_STRATA_LABELS[index] == label


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
        RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM, RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM
    )
    assert TSConfig().random_train_anchor_sampling == RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM


def test_an_unknown_sampling_policy_is_refused():
    with pytest.raises(ValueError, match="unknown random_train_anchor_sampling"):
        TSConfig(random_train_anchor=True, random_train_anchor_sampling="stratified")


def test_path_uniform_sampling_is_refused_without_random_anchors():
    with pytest.raises(ValueError, match="random_train_anchor=False draws"):
        TSConfig(random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM)


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
    path_uniform = replace(
        plain, random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM
    )
    assert "anchors=remaining-path-uniform" in run_display_name(path_uniform.to_dict())
    assert training_window_class(path_uniform) is (
        RemainingPathUniformAnchorTrajectoryWindows
    )


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


#: A flight whose admissible anchors are DENSE near the runway and SPARSE far out — the real
#: shape, exaggerated so the two laws cannot agree: most samples sit in the first eighth of
#: the span. Uniform-in-sample must follow the counts, uniform-in-path the span.
_NEAR_SAMPLES = 90
_NEAR_SPAN_M, _FAR_SPAN_M = 5_000.0, 40_000.0


def _hand_built(sampling: str):
    """One flight, its stored anchors relabelled with a hand-made remaining-path profile."""
    windows = _windows(sampling)
    stored = len(windows.index)
    assert stored >= _NEAR_SAMPLES + 10
    near = np.linspace(0.0, _NEAR_SPAN_M, _NEAR_SAMPLES, endpoint=False)
    far = np.linspace(_NEAR_SPAN_M, _FAR_SPAN_M, stored - _NEAR_SAMPLES)
    # Descending with the index, as a real profile is: the last anchor is the nearest.
    windows.anchor_remaining_path_m = np.concatenate((far[::-1], near[::-1]))
    return windows


def _near_share(windows, epochs: int = 2000) -> float:
    """The share of drawn anchors inside the near block, over many epochs."""
    drawn = np.concatenate([
        windows.anchor_remaining_path_m[windows.epoch_indices(epoch)]
        for epoch in range(1, epochs + 1)
    ])
    return float(np.mean(drawn < _NEAR_SPAN_M))


def test_the_uniform_draw_follows_the_sample_density():
    windows = _hand_built(RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM)
    expected = _NEAR_SAMPLES / len(windows.index)
    assert expected > 0.6, "the fixture must make the two laws disagree"
    assert _near_share(windows) == pytest.approx(expected, abs=0.03)


def test_the_path_uniform_draw_follows_the_span():
    """Equal weight per kilometre: the near block is an eighth of the span, so it takes
    about an eighth of the draws — not the share its sample count would take."""
    windows = _hand_built(RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM)
    assert _near_share(windows) == pytest.approx(_NEAR_SPAN_M / _FAR_SPAN_M, abs=0.03)


def test_the_offset_law_is_nearest_within_the_flights_own_span():
    """The draw is placed across [min, max] and snapped to the nearest admissible anchor —
    so both ends are reachable and a single-anchor flight still returns it."""
    remaining = np.array([20_000.0, 12_000.0, 4_000.0])
    assert remaining_path_uniform_offset(remaining, 0.0) == 2      # the minimum
    assert remaining_path_uniform_offset(remaining, 0.999) == 0    # the maximum
    assert remaining_path_uniform_offset(remaining, 0.5) == 1      # 12 km is nearest
    assert remaining_path_uniform_offset(np.array([7_000.0]), 0.4) == 0


@pytest.mark.parametrize("sampling", RANDOM_TRAIN_ANCHOR_SAMPLINGS)
def test_the_draw_is_the_flights_own_per_epoch_hash(sampling: str):
    """Same seed and epoch, same anchors — from a window set built independently."""
    first, second = _windows(sampling, n_flights=4), _windows(sampling, n_flights=4)
    assert np.array_equal(first.epoch_indices(11), second.epoch_indices(11))
    assert not np.array_equal(first.epoch_indices(11), first.epoch_indices(12))


def test_the_two_policies_store_the_same_anchors_for_the_same_cohort():
    """Admissibility is not part of this axis: the 20 s guard admits the same population."""
    uniform = _windows(RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM, n_flights=4)
    path_uniform = _windows(RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, n_flights=4)
    assert uniform.index == path_uniform.index
    assert np.array_equal(uniform.anchor_strata, path_uniform.anchor_strata)
    assert np.array_equal(
        uniform.anchor_remaining_path_m, path_uniform.anchor_remaining_path_m
    )
    assert uniform.eligible_candidate_anchors == path_uniform.eligible_candidate_anchors
    assert uniform.sampling_version != path_uniform.sampling_version
    # ...and they must not draw the same anchors, or the axis changes nothing.
    assert not np.array_equal(uniform.epoch_indices(7), path_uniform.epoch_indices(7))


def test_the_stored_path_and_strata_are_those_of_the_stored_anchors():
    windows = _windows(RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, n_flights=4)
    expected_strata = [
        int(remaining_path_strata(windows.series[s_idx])[anchor])
        for s_idx, anchor in windows.index
    ]
    expected_path = [
        float(remaining_path_profile_m(windows.series[s_idx])[anchor])
        for s_idx, anchor in windows.index
    ]
    assert windows.anchor_strata.tolist() == expected_strata
    assert windows.anchor_remaining_path_m.tolist() == pytest.approx(expected_path)


# ── the per-epoch record ─────────────────────────────────────────────────────

@pytest.mark.parametrize("sampling", RANDOM_TRAIN_ANCHOR_SAMPLINGS)
def test_the_epoch_record_counts_the_draws_beside_the_population(sampling: str):
    windows = _windows(sampling, n_flights=4)
    statistics = windows.anchor_statistics(5)
    drawn = statistics["remaining_path_strata"]
    population = statistics["remaining_path_strata_population"]
    assert list(drawn) == list(REMAINING_PATH_STRATA_LABELS)
    assert list(population) == list(REMAINING_PATH_STRATA_LABELS)
    # One draw per flight; the population is every stored admissible anchor. A drawn count
    # that summed to the windows would be reporting the population, and vice versa.
    assert sum(drawn.values()) == len(windows.eligible_series) == statistics["samples"]
    assert sum(population.values()) == len(windows.index)


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
        block = epoch["train_anchor_sampling"]
        assert sum(block["remaining_path_strata"].values()) == flights
        assert sum(block["remaining_path_strata_population"].values()) > flights
    contract = summary["training_anchor_contract"]
    assert contract["sampling_version"] == (
        "per-flight-hash-v3-remaining-path-uniform"
        if sampling == RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM
        else "per-flight-hash-v2-output-eligibility"
    )
