from types import SimpleNamespace

import pytest

import run_ts_kinematic_ablation as ablation


def _candidate(weight, ade, smoothness):
    return {
        "kinematic_weight": weight,
        "validation_accuracy": {"ade_m": {"mean": ade}},
        "smoothness_score": smoothness,
    }


def test_selection_applies_accuracy_band_before_smoothness():
    candidates = [
        _candidate(0.0, 100.0, 4.0),
        _candidate(0.3, 108.0, 1.5),
        _candidate(3.0, 120.0, 0.2),
    ]

    selected = ablation.select_candidate(candidates, accuracy_tolerance=0.10)

    assert selected["kinematic_weight"] == 0.3


def test_selection_prefers_accuracy_and_smaller_model_inside_physics_equivalence_band():
    candidates = [
        {**_candidate(3.0, 100.0, 1.01), "d_model": 256, "n_segments": 64},
        {**_candidate(3.0, 101.0, 1.00), "d_model": 512, "n_segments": 64},
    ]

    selected = ablation.select_candidate(
        candidates, accuracy_tolerance=0.10, smoothness_tolerance=0.02
    )

    assert selected["d_model"] == 256


def test_balanced_subsets_are_split_label_specific_and_deterministic():
    series = [
        SimpleNamespace(airport=airport, dataset_id=f"{airport}:{index}")
        for airport in ("KAAA", "KBBB")
        for index in range(10)
    ]
    first = ablation.select_balanced_subset(
        series, ("KAAA", "KBBB"), samples_per_airport=3, seed=7, label="outer-train"
    )
    second = ablation.select_balanced_subset(
        list(reversed(series)), ("KAAA", "KBBB"), samples_per_airport=3,
        seed=7, label="outer-train",
    )
    validation = ablation.select_balanced_subset(
        series, ("KAAA", "KBBB"), samples_per_airport=3, seed=7,
        label="outer-validation",
    )

    assert [row.dataset_id for row in first] == [row.dataset_id for row in second]
    assert [row.dataset_id for row in first] != [row.dataset_id for row in validation]
    assert sum(row.airport == "KAAA" for row in first) == 3


def test_smoothness_score_is_unitless_geometric_mean():
    ratios = {key: value for key, value in zip(
        ablation.RAW_KINEMATIC_METRIC_KEYS, (1.0, 1.0, 1.0, 4.0, 4.0)
    )}
    assert ablation.smoothness_score(ratios) == pytest.approx(16.0 ** (1.0 / 5.0))


def test_raw_metric_ratios_use_fleet_p95_not_outlier_sensitive_mean():
    accuracy = {"raw_kinematics": {"predicted": {}, "observed_baseline": {}}}
    for key in ablation.RAW_KINEMATIC_METRIC_KEYS:
        accuracy["raw_kinematics"]["predicted"][key] = {
            "mean": 1_000_000.0, "p95": 6.0,
        }
        accuracy["raw_kinematics"]["observed_baseline"][key] = {
            "mean": 2.0, "p95": 3.0,
        }

    ratios = ablation.raw_metric_ratios(accuracy)

    assert set(ratios.values()) == {2.0}
