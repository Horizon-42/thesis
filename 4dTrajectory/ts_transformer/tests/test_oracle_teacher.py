from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


TS_DIR = Path(__file__).resolve().parents[1]
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

from config import (  # noqa: E402
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    PREDICTION_CONTROL,
    TSConfig,
)
from control.oracle.imitation import control_imitation_loss  # noqa: E402
from control.oracle.evaluation import observed_clock_prediction  # noqa: E402
from control.oracle.pretraining import validate_teacher_durations  # noqa: E402
from prediction_outputs import ControlOutputHead, ControlPrediction  # noqa: E402

REPO_ROOT = TS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_ts_simple_teacher_paired_cv import (  # noqa: E402
    RNGNeutralPretrainer,
    select_fold_teacher_series,
    summarize_pairs,
    validate_fold_isolation,
)
from run_ts_oracle_teacher_optimize import _balanced_cohort_sizes  # noqa: E402


def test_control_imitation_loss_balances_physical_control_ranges():
    lower = torch.tensor([[0.0, -1.0, 0.5]])
    upper = torch.tensor([[200_000.0, 1.0, 2.0]])
    target_controls = torch.tensor([[[100_000.0, 0.0, 1.0]]])
    predicted_controls = target_controls + torch.tensor([[[20_000.0, 0.2, 0.15]]])
    prediction = ControlPrediction(
        controls=predicted_controls,
        segment_durations=torch.tensor([[5.0]]),
        final_time_s=torch.tensor([5.0]),
    )

    loss = control_imitation_loss(
        prediction,
        target_controls,
        torch.tensor([[5.0]]),
        torch.tensor([5.0]),
        lower,
        upper,
        final_time_scale_s=600.0,
    )

    # Every channel differs by exactly 10% of its own aircraft-specific range.
    torch.testing.assert_close(loss.control, torch.tensor(0.01))
    torch.testing.assert_close(loss.duration_fraction, torch.tensor(0.0))
    torch.testing.assert_close(loss.final_time, torch.tensor(0.0))


def test_pooled_teacher_cohort_is_balanced_and_keeps_requested_total():
    allocation = _balanced_cohort_sizes(
        ["KSTL", "KSJC", "KRDU", "KSMF", "KMSY"], 32
    )

    assert allocation == {
        "KMSY": 7,
        "KRDU": 7,
        "KSJC": 6,
        "KSMF": 6,
        "KSTL": 6,
    }
    assert sum(allocation.values()) == 32


def test_observed_clock_teacher_evaluation_preserves_partition_and_replaces_total():
    prediction = ControlPrediction(
        controls=torch.zeros(1, 2, 3),
        segment_durations=torch.tensor([[2.0, 6.0]]),
        final_time_s=torch.tensor([8.0]),
    )

    state_clock = observed_clock_prediction(prediction, torch.tensor([20.0]))

    torch.testing.assert_close(
        state_clock.segment_durations, torch.tensor([[5.0, 15.0]])
    )
    torch.testing.assert_close(state_clock.final_time_s, torch.tensor([20.0]))


def test_uniform_recipe_rejects_nonuniform_teacher_durations():
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_duration_gradient=False,
        n_segments=2,
    )

    validate_teacher_durations(
        torch.tensor([[5.0, 5.0]]), torch.tensor([10.0]), config
    )
    with pytest.raises(ValueError, match="requires uniform teacher durations"):
        validate_teacher_durations(
            torch.tensor([[2.0, 8.0]]), torch.tensor([10.0]), config
        )




def test_fold_teacher_selection_is_deterministic_and_train_local():
    fold_train = [SimpleNamespace(dataset_id=f"KSJC:train-{index}") for index in range(8)]

    first = select_fold_teacher_series(
        fold_train, fold_index=1, cohort_size=3, split_seed=1337
    )
    second = select_fold_teacher_series(
        list(reversed(fold_train)), fold_index=1, cohort_size=3, split_seed=1337
    )

    assert [item.dataset_id for item in first] == [item.dataset_id for item in second]
    assert {item.dataset_id for item in first} <= {
        item.dataset_id for item in fold_train
    }


def test_fold_isolation_rejects_teacher_from_fold_validation():
    folds = [
        [SimpleNamespace(dataset_id="KSJC:a"), SimpleNamespace(dataset_id="KSJC:b")],
        [SimpleNamespace(dataset_id="KSJC:c"), SimpleNamespace(dataset_id="KSJC:d")],
    ]
    validate_fold_isolation(
        usable_outer_train_ids=["KSJC:a", "KSJC:b", "KSJC:c", "KSJC:d"],
        outer_validation_ids=["KSJC:outer-val"],
        outer_test_ids=["KSJC:outer-test"],
        folds=folds,
        teacher_ids_by_fold=[["KSJC:c"], ["KSJC:a"]],
    )
    with pytest.raises(ValueError, match="not confined to fold training"):
        validate_fold_isolation(
            usable_outer_train_ids=["KSJC:a", "KSJC:b", "KSJC:c", "KSJC:d"],
            outer_validation_ids=["KSJC:outer-val"],
            outer_test_ids=["KSJC:outer-test"],
            folds=folds,
            teacher_ids_by_fold=[["KSJC:a"], ["KSJC:a"]],
        )


def test_rng_neutral_pretrainer_changes_model_but_restores_random_streams():
    torch.manual_seed(17)
    np.random.seed(17)
    model = nn.Linear(1, 1, bias=False)

    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    expected_torch = torch.rand(3)
    expected_numpy = np.random.random(3)
    torch.random.set_rng_state(torch_state)
    np.random.set_state(numpy_state)

    def delegate(target, *_args, **_kwargs):
        with torch.no_grad():
            target.weight.add_(1.0)
        torch.rand(20)
        np.random.random(20)
        return {"delegate": True}

    original = model.weight.detach().clone()
    audit = RNGNeutralPretrainer(delegate)(model)

    assert not torch.equal(model.weight, original)
    torch.testing.assert_close(torch.rand(3), expected_torch)
    np.testing.assert_allclose(np.random.random(3), expected_numpy)
    assert "restored" in audit["downstream_rng_policy"]


def test_paired_summary_uses_within_fold_teacher_improvements():
    folds = []
    for teacher_ade, baseline_ade in ((80.0, 100.0), (120.0, 100.0), (60.0, 100.0)):
        arms = {}
        for arm, ade in (("teacher", teacher_ade), ("no_teacher", baseline_ade)):
            arms[arm] = {
                "validation_metrics": {
                    "ade_m": ade,
                    "fde_m": ade + 10.0,
                    "invalid_flights": 0,
                }
            }
        folds.append({"arms": arms})

    summary = summarize_pairs(folds)

    assert summary["ade_m"]["paired_improvement_by_fold"] == [20.0, -20.0, 40.0]
    assert summary["ade_m"]["paired_improvement_mean"] == pytest.approx(40.0 / 3.0)
    assert summary["ade_m"]["teacher_wins"] == 2
