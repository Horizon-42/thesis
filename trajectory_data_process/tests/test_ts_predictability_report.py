from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
for path in (REPO_ROOT, TS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_ts_predictability_report as report  # noqa: E402
from config import HORIZON_FULL, HORIZON_WINDOW, TSConfig  # noqa: E402
from time_grids import output_time_grid  # noqa: E402


def test_full_grid_shortens_the_last_segment_and_masks_padding():
    config = TSConfig(
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=4,
        dt_s=2.0,
    )
    grid = output_time_grid(5.0, config)

    assert grid.offsets_s == pytest.approx([2.0, 4.0, 5.0, 5.0])
    assert grid.segment_durations_s == pytest.approx([2.0, 2.0, 1.0, 0.0])
    assert grid.active.tolist() == [True, True, True, False]
    assert not grid.horizon_capped


def test_window_grid_reports_a_capped_physical_horizon():
    config = TSConfig(
        horizon_mode=HORIZON_WINDOW,
        window_horizon_steps=3,
        dt_s=2.0,
    )
    grid = output_time_grid(8.0, config)

    assert grid.offsets_s == pytest.approx([2.0, 4.0, 6.0])
    assert grid.active.all()
    assert grid.horizon_capped


def test_recursive_window_exceed_rate_uses_the_total_full_horizon():
    config = TSConfig(
        horizon_mode=HORIZON_WINDOW,
        window_horizon_steps=3,
        full_horizon_steps=10,
        dt_s=2.0,
    )

    assert report.true_horizon_exceed_rate(np.array([6.0, 20.0, 21.0]), config) == pytest.approx(1 / 3)


def test_route_complexity_labels_are_geometric_proxies():
    straight = np.column_stack((np.linspace(1000, 0, 20), np.zeros(20), np.zeros(20)))
    theta = np.linspace(0, np.pi / 2, 30)
    single_turn = np.column_stack((1000 * np.cos(theta), 1000 * np.sin(theta), np.zeros(30)))
    circle = np.column_stack((1000 * np.cos(np.linspace(0, 2 * np.pi, 60)),
                              1000 * np.sin(np.linspace(0, 2 * np.pi, 60)),
                              np.zeros(60)))

    assert report.classify_trajectory(straight) == "straight-in"
    assert report.classify_trajectory(single_turn) == "single-turn"
    assert report.classify_trajectory(circle) == "holding-like"


def test_remaining_time_report_uses_true_time_not_predicted_time():
    model_results = {"model": {"error_grid_m": np.array([[10.0, 20.0]])}}
    rows = report.remaining_time_rows(
        model_results,
        durations=np.array([100.0]),
        progress=np.array([0.5, 1.0]),
        edges=(0.0, 30.0, 60.0),
    )
    values = {row["remaining_time_bin_s"]: row["mean_error_m"] for row in rows}

    assert values["0–30"] == pytest.approx(20.0)
    assert values["30–60"] == pytest.approx(10.0)


def test_validation_comparison_allows_non_evaluated_split_differences():
    config = TSConfig()
    reference = SimpleNamespace(
        label="normalized",
        config=config,
        payload={"split": {"train": ["a", "b"], "val": ["v1"], "test": ["t1"]}},
    )
    window = SimpleNamespace(
        label="window",
        config=config,
        payload={"split": {"train": ["a"], "val": ["v1"], "test": []}},
    )

    assert report.comparison_identity_error(reference, window) is None


def test_validation_comparison_rejects_a_different_validation_cohort():
    config = TSConfig()
    reference = SimpleNamespace(
        label="normalized",
        config=config,
        payload={"split": {"train": ["a"], "val": ["v1"], "test": ["t1"]}},
    )
    candidate = SimpleNamespace(
        label="window",
        config=config,
        payload={"split": {"train": ["a"], "val": ["v2"], "test": ["t1"]}},
    )

    assert report.comparison_identity_error(reference, candidate) == (
        "checkpoint window has a different validation split"
    )


def test_fit_evaluation_is_bound_to_the_exact_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"best checkpoint")
    evaluation = {
        "schema_version": report.FIT_EVALUATION_SCHEMA,
        "checkpoint": {
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    }
    (tmp_path / report.FIT_EVALUATION_NAME).write_text(
        json.dumps(evaluation), encoding="utf-8"
    )

    assert report.fit_evaluation_for_checkpoint(checkpoint) == evaluation

    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="different checkpoint"):
        report.fit_evaluation_for_checkpoint(checkpoint)
