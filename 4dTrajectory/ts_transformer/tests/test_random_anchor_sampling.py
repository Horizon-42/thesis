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
import math
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
    default_anchor,
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


def _windows(sampling: str, *, n_flights: int = 1, minimum_anchor_index=None, **overrides):
    config = _config(random_train_anchor_sampling=sampling, **overrides)
    series, report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3),
        config, airport=AIRPORT,
    )
    assert report.built == n_flights, report.format()
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    return training_window_class(config)(
        series, config, normalizer, minimum_anchor_index=minimum_anchor_index
    )


#: A flight whose admissible anchors are DENSE near the runway and SPARSE far out — the real
#: shape, exaggerated so the two laws cannot agree: most samples sit in the first eighth of
#: the span. Uniform-in-sample must follow the counts, uniform-in-path the span.
_NEAR_SAMPLES = 90
_NEAR_SPAN_M, _FAR_SPAN_M = 5_000.0, 40_000.0


def _hand_built(sampling: str, **overrides):
    """One flight, its stored anchors relabelled with a hand-made remaining-path profile."""
    windows = _windows(sampling, **overrides)
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


# ── A2b: the share of the draws RESERVED for L-1 ─────────────────────────────
#
# A0.b's random-anchor arms win at every re-anchored bin and still lose at L-1
# (`A0b_lr_objective_path_uniform` 1782 m pooled ADE against the fixed-anchor native32's
# 1322 m), because under a law spread over the whole approach L-1 is one point among many
# draws. `random_train_anchor_l1_share` mixes a reserved share of L-1 draws into
# `remaining-path-uniform` and asks whether that recovers L-1 without flattening the curve.
#
# What the tests below hold:
#
# * the default 0 draws EXACTLY what the pure law drew before the axis existed — the epoch
#   digests are pinned by hand, measured at the commit before it;
# * a share of 1 draws nothing but L-1, and a share of 0.3 realises 0.3 within binomial
#   noise over 200 flights x 100 epochs;
# * the draws the coin passes over are the SAME anchors the share-0 arm drew (a mixture,
#   not a reweighting) and still follow the span law;
# * the two refusals, both of which would otherwise be silent no-ops;
# * a flight that stores no L-1 at all is reserved at its earliest admissible anchor and
#   COUNTED into the epoch record (10 of 9,720 on the real KRDU roster), never silently.

#: The epoch-5 sample digest of the four-flight fixture under each policy at the DEFAULT
#: share, measured at 9b1f130 (the commit before this axis) and unchanged after it. The
#: mixture reads a SECOND 8 bytes of the same per-flight digest for its coin and salts the
#: first with the class's own version rather than the reported one, precisely so these do
#: not move: an arm trained under the pure law must stay comparable with A0.b's.
_DEFAULT_SHARE_SAMPLE_SHA256 = {
    RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM:
        "19504377e3fecd34c1f9d8f3e95afd13c637d7180ef13826d978808d91cfc00d",
    RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM:
        "94bdfef457e95bbe223ac0781e2797f3e39d543c7f8587a20cabd0dcf307eb51",
}

_L1_SHARE_SAMPLING_VERSION = "per-flight-hash-v4-remaining-path-uniform-l1-share"


def _path_uniform(**overrides) -> TSConfig:
    return _config(
        random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, **overrides
    )


def _drawn_anchor_by_flight(windows, epoch: int) -> dict[int, int]:
    """Each flight's drawn anchor for one epoch, keyed by flight — never by position.

    `epoch_indices` shuffles, so two window sets' arrays line up only by accident.
    """
    return {
        windows.index[int(index)][0]: windows.index[int(index)][1]
        for index in windows.epoch_indices(epoch)
    }


def test_the_reserved_share_defaults_to_none_of_the_draws():
    assert TSConfig().random_train_anchor_l1_share == 0.0


@pytest.mark.parametrize("sampling", RANDOM_TRAIN_ANCHOR_SAMPLINGS)
def test_the_default_share_draws_what_the_pure_law_drew_before_the_axis(sampling: str):
    """Byte-identical draws, pinned by digest — and no bookkeeping column that never moves."""
    windows = _windows(sampling, n_flights=4)
    statistics = windows.anchor_statistics(5)
    assert statistics["sample_sha256"] == _DEFAULT_SHARE_SAMPLE_SHA256[sampling]
    assert statistics["sampling_version"] == type(windows).sampling_version
    assert windows.reported_sampling_version == type(windows).sampling_version
    assert "l1_share_drawn" not in statistics
    assert "l1_share" not in statistics


def test_a_full_share_draws_nothing_but_the_fixed_anchor():
    windows = _windows(
        RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, n_flights=4,
        random_train_anchor_l1_share=1.0,
    )
    anchor = default_anchor(windows.config)
    for epoch in (1, 7, 40):
        drawn = _drawn_anchor_by_flight(windows, epoch)
        assert len(drawn) == len(windows.eligible_series)
        assert set(drawn.values()) == {anchor}
        statistics = windows.anchor_statistics(epoch)
        assert statistics["l1_share_drawn"] == 1.0
        assert statistics["fixed_anchor_fraction"] == 1.0


def test_a_partial_share_is_realised_within_binomial_noise():
    """200 flights x 100 epochs of the recorded share against the 0.3 it was asked for."""
    share, epochs = 0.3, 100
    windows = _windows(
        RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, n_flights=200,
        random_train_anchor_l1_share=share,
    )
    flights = len(windows.eligible_series)
    assert flights == 200
    realised = [
        windows.anchor_statistics(epoch)["l1_share_drawn"]
        for epoch in range(1, epochs + 1)
    ]
    pooled = float(np.mean(realised))
    sigma = math.sqrt(share * (1.0 - share) / (flights * epochs))
    assert abs(pooled - share) < 4.0 * sigma
    # ...and every epoch's record says which share it was asked for, beside what it got.
    assert {windows.anchor_statistics(epoch)["l1_share"] for epoch in (1, 50)} == {share}
    # `fixed_anchor_fraction` counts every drawn anchor that IS L-1 — the coin's, plus the
    # span draws that landed at the far end anyway — so on a cohort where every flight
    # stores L-1 it can only be the larger number.
    assert windows.flights_without_default_anchor == 0
    for epoch in (1, 50, 100):
        statistics = windows.anchor_statistics(epoch)
        assert statistics["fixed_anchor_fraction"] >= statistics["l1_share_drawn"]


def test_the_draws_the_coin_passes_over_are_the_share_zero_draws():
    """A MIXTURE, not a reweighting: the arm differs from its base arm only in the draws the
    coin replaced, which is what makes A2b a one-field contrast against A0.b."""
    base = _windows(RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, n_flights=8)
    mixed = _windows(
        RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, n_flights=8,
        random_train_anchor_l1_share=0.5,
    )
    anchor = default_anchor(mixed.config)
    replaced = 0
    for epoch in range(1, 40):
        before, after = _drawn_anchor_by_flight(base, epoch), _drawn_anchor_by_flight(mixed, epoch)
        assert before.keys() == after.keys()
        _unit_draws, reserved = mixed._epoch_draws(epoch)
        # WHICH flights the coin fired on, not merely that some anchors moved: a mixture
        # that reserved the wrong flights would satisfy "L-1 or the base draw" everywhere.
        for series_index, take_l1 in zip(mixed.eligible_series, reserved, strict=True):
            flight = int(series_index)
            assert after[flight] == (anchor if take_l1 else before[flight])
            replaced += after[flight] != before[flight]
    assert replaced, "a 0.5 share that never replaced a draw is not a mixture"


def test_the_unreserved_draws_still_follow_the_span_law():
    """The pure law's own population test, re-read on the mixture.

    L-1 is the FAR end of the hand-made profile (the first stored anchor), so no reserved
    draw can land in the near block: read against all the draws the near share would simply
    be diluted by the share, which says nothing about the law the rest obey.
    """
    windows = _hand_built(
        RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, random_train_anchor_l1_share=0.3
    )
    drawn = []
    for epoch in range(1, 4001):
        path_by_flight = {
            windows.index[int(index)][0]: windows.anchor_remaining_path_m[int(index)]
            for index in windows.epoch_indices(epoch)
        }
        _unit_draws, reserved = windows._epoch_draws(epoch)
        drawn += [
            path_by_flight[int(series_index)]
            for series_index, take_l1 in zip(windows.eligible_series, reserved)
            if not take_l1
        ]
    near_share = float(np.mean(np.array(drawn) < _NEAR_SPAN_M))
    assert near_share == pytest.approx(_NEAR_SPAN_M / _FAR_SPAN_M, abs=0.03)


def test_a_reserved_share_reports_its_own_law_and_keeps_the_pure_laws_salt():
    windows = _windows(
        RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, n_flights=4,
        random_train_anchor_l1_share=0.3,
    )
    assert windows.reported_sampling_version == _L1_SHARE_SAMPLING_VERSION
    assert windows.anchor_statistics(5)["sampling_version"] == _L1_SHARE_SAMPLING_VERSION
    # ...while the SALT stays the pure law's, which is what keeps the unreserved draws
    # equal to the share-0 arm's.
    assert windows.sampling_version == (
        RemainingPathUniformAnchorTrajectoryWindows.sampling_version
    )


def test_a_share_is_refused_without_random_anchors():
    with pytest.raises(ValueError, match="reserves a share of the RANDOM anchor draws"):
        TSConfig(random_train_anchor_l1_share=0.3)


def test_a_share_is_refused_under_the_uniform_draw():
    """Never ignored: `uniform` draws over the samples, where L-1 is already one of them."""
    with pytest.raises(ValueError, match="mixing a reserved share into it"):
        TSConfig(random_train_anchor=True, random_train_anchor_l1_share=0.3)


@pytest.mark.parametrize("share", (-0.1, 1.5))
def test_a_share_outside_the_unit_interval_is_refused(share: float):
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        _path_uniform(random_train_anchor_l1_share=share)
    # ...and it is the message a user sees even when the cross-field refusals also apply:
    # "not a share at all" is the more actionable complaint about `1.5`.
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        TSConfig(random_train_anchor_l1_share=share)


def test_a_flight_that_stores_no_l1_is_reserved_at_its_earliest_anchor_and_COUNTED():
    """Bounded coverage is stated, not silent — and not refused either.

    On the real KRDU roster 10 of 9,720 eligible flights store no L-1 at all: output
    eligibility (`airborne-1.10-stall-margin-v1`) removed it because the observed state
    there is outside the airborne model domain, so there is no valid L-1 window to train.
    Their reserved draw is the earliest anchor they DO have, and the count reaches every
    epoch's record. A refusal here would have aborted both A2b arms; changing the roster to
    avoid it would have destroyed the pairing against the arm they are one field from.

    The `minimum_anchor_index` floor reproduces the shape cheaply — and there it is not
    even a deviation: `FixedAnchorTrajectoryWindows` anchors at the same `anchors[0]`.
    """
    floor = _path_uniform().seq_len + 2
    windows = _windows(
        RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, n_flights=4,
        minimum_anchor_index=floor, random_train_anchor_l1_share=1.0,
    )
    flights = len(windows.eligible_series)
    assert flights == 4
    assert windows.flights_without_default_anchor == flights
    assert windows.anchor_statistics(3)["l1_share_flights_without_l1"] == flights
    # Every reserved draw is the flight's FIRST admissible anchor, which the floor moved.
    assert set(_drawn_anchor_by_flight(windows, 3).values()) == {floor}
    assert windows.anchor_statistics(3)["fixed_anchor_fraction"] == 0.0


def test_an_unfloored_cohort_reserves_l1_and_counts_no_exceptions():
    windows = _windows(
        RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, n_flights=4,
        random_train_anchor_l1_share=0.3,
    )
    assert windows.flights_without_default_anchor == 0
    assert windows.anchor_statistics(3)["l1_share_flights_without_l1"] == 0


def test_the_share_names_the_run_only_when_it_is_set():
    plain = TSConfig(
        random_train_anchor=True,
        random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM,
    )
    assert "l1-share=" not in run_display_name(plain.to_dict())
    mixed = replace(plain, random_train_anchor_l1_share=0.3)
    assert "l1-share=0.3" in run_display_name(mixed.to_dict())


def test_a_two_epoch_run_records_the_realised_share(tmp_path: Path):
    """The seed the epoch's draws came from reaches the record through `train()` too."""
    settings = dict(recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False))
    settings.update(
        n_segments=8, control_imitation_loss_weight=0.0,
        seq_len=8, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu",
        epochs=2, patience=2, batch_size=8, dropout=0.0,
        control_rollout_integrator_dt_s=0.5,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        random_train_anchor=True, random_train_anchor_min_future_s=4.0,
        random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM,
        random_train_anchor_l1_share=0.5,
    )
    config = TSConfig(**settings)
    series, report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3), config, airport=AIRPORT
    )
    assert report.built == 8, report.format()
    torch.manual_seed(0)
    train(
        series, config, output_dir=tmp_path / "l1_share",
        data_provenance={
            "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
            "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64,
                           "source_records": []}],
        },
        verbose=False,
    )
    summary = json.loads((tmp_path / "l1_share" / "history.json").read_text())
    for epoch in summary["history"]:
        block = epoch["train_anchor_sampling"]
        assert block["l1_share"] == 0.5
        assert 0.0 <= block["l1_share_drawn"] <= 1.0
        assert block["fixed_anchor_fraction"] >= block["l1_share_drawn"]
    assert summary["training_anchor_contract"]["sampling_version"] == (
        _L1_SHARE_SAMPLING_VERSION
    )
