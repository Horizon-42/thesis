"""The state objective and the common-true-time metric it is selected on.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""


import numpy as np
import pytest
import torch

import ts_transformer.data.channels as ch
from ts_transformer.config import HORIZON_FULL, CHECKPOINT_SELECTION_COMMON_GRID_ADE, TSConfig
from ts_transformer.data.dataset import Normalizer
from ts_transformer.geometry.metrics import (
    common_physical_time_flight_metrics,
    states_with_derived_velocity,
)
from ts_transformer.outputs.state.model import StatePrediction
from ts_transformer.outputs.state.loss import state_prediction_loss_components
from ts_transformer.training.objective import prediction_loss


def _identity_normalizer() -> Normalizer:
    return Normalizer(
        mean=np.zeros(len(ch.CHANNELS), dtype=np.float64),
        std=np.ones(len(ch.CHANNELS), dtype=np.float64),
    )


def test_prediction_loss_adds_scaled_final_time_error():
    config = TSConfig(
        final_time_scale_s=600.0,
        final_time_loss_weight=2.0,
        kinematic_consistency_loss_weight=0.0,
        terminal_loss_weight=0.0,
    )
    states = torch.zeros((1, 2, len(ch.CHANNELS)))
    prediction = StatePrediction(states=states, final_time_s=torch.tensor([900.0]))
    loss = prediction_loss(
        prediction,
        torch.zeros((1, len(ch.CHANNELS))),
        states,
        torch.ones_like(states),
        torch.tensor([600.0]),
        torch.ones(1),
        config,
        _identity_normalizer(),
    )
    assert float(loss) == pytest.approx(0.5)


def test_prediction_loss_applies_per_flight_airport_weights():
    config = TSConfig(
        n_segments=2,
        validation_common_grid_points=2,
        final_time_loss_weight=0.0,
        state_endpoint_loss_weight=0.0,
        kinematic_consistency_loss_weight=0.0,
        terminal_loss_weight=0.0,
    )
    target = torch.zeros((2, 2, len(ch.CHANNELS)))
    prediction = StatePrediction(
        states=torch.stack((torch.ones_like(target[0]), 3.0 * torch.ones_like(target[0]))),
        final_time_s=torch.full((2,), 10.0),
    )
    loss = prediction_loss(
        prediction,
        torch.zeros((2, len(ch.CHANNELS))),
        target,
        torch.ones_like(target),
        torch.full((2,), 10.0),
        torch.tensor([1.5, 0.5]),
        config,
        _identity_normalizer(),
    )

    expected_physical_position_mse = (3.0 * 1.5 + 27.0 * 0.5) / 2.0
    assert float(loss) == pytest.approx(expected_physical_position_mse / 10_000.0**2)


def test_default_checkpoint_selection_is_common_true_time_ade():
    assert TSConfig().checkpoint_selection_metric == CHECKPOINT_SELECTION_COMMON_GRID_ADE


def test_state_loss_is_isotropic_physical_position_plus_time_only():
    normalizer = Normalizer(
        mean=np.zeros(len(ch.CHANNELS)),
        std=np.array([100.0, 200.0, 10.0, 2.0, 3.0, 4.0]),
    )
    config = TSConfig(
        n_segments=2,
        validation_common_grid_points=2,
        final_time_loss_weight=1.0,
        state_endpoint_loss_weight=0.0,
        kinematic_consistency_loss_weight=99.0,
        terminal_loss_weight=99.0,
    )
    target = torch.zeros((1, 2, len(ch.CHANNELS)))
    predicted = target.clone()
    predicted[..., ch.IDX["e"]] = 1.0       # 100 m
    predicted[..., ch.IDX["u"]] = 10.0      # 100 m despite a different std
    predicted[..., list(ch.VELOCITY_IDX)] = 999.0
    components = state_prediction_loss_components(
        StatePrediction(states=predicted, final_time_s=torch.tensor([600.0])),
        torch.zeros((1, len(ch.CHANNELS))),
        target,
        torch.ones_like(target),
        torch.tensor([600.0]),
        torch.ones(1),
        config,
        normalizer,
    )

    assert float(components.state) == pytest.approx(
        (100.0**2 + 100.0**2) / 10_000.0**2
    )
    assert float(components.final_time) == pytest.approx(0.0)
    assert float(components.kinematic) == pytest.approx(0.0)
    assert float(components.terminal) == pytest.approx(0.0)


def test_state_position_loss_uses_true_time_not_equal_progress():
    config = TSConfig(
        n_segments=2,
        validation_common_grid_points=2,
        final_time_loss_weight=0.0,
    )
    target = torch.zeros((1, 2, len(ch.CHANNELS)))
    target[0, :, ch.IDX["e"]] = torch.tensor([50.0, 100.0])
    prediction = StatePrediction(
        states=target.clone(),
        final_time_s=torch.tensor([5.0]),
    )
    components = state_prediction_loss_components(
        prediction,
        torch.zeros((1, len(ch.CHANNELS))),
        target,
        torch.ones_like(target),
        torch.tensor([10.0]),
        torch.ones(1),
        config,
        _identity_normalizer(),
    )

    # At true t=5 s, the short prediction has already reached its 100 m endpoint while
    # truth is at 50 m.  Equal-progress loss would incorrectly be zero.
    assert float(components.state) == pytest.approx(
        (50.0**2 / 2.0) / 10_000.0**2
    )


def test_common_physical_time_metric_penalizes_an_early_ending_prediction():
    anchor = np.zeros(len(ch.CHANNELS))
    truth = np.zeros((2, len(ch.CHANNELS)))
    truth[:, ch.IDX["e"]] = [50.0, 100.0]
    predicted = np.zeros((1, len(ch.CHANNELS)))
    predicted[0, ch.IDX["e"]] = 50.0

    block = common_physical_time_flight_metrics(
        anchor_values=anchor,
        predicted_values=predicted,
        predicted_offsets_s=np.array([5.0]),
        predicted_final_time_s=5.0,
        truth_values=truth,
        truth_offsets_s=np.array([5.0, 10.0]),
        true_final_time_s=10.0,
        points=2,
    )

    # The old overlap-only metric saw only the exact 5 s point and returned zero.  The
    # common grid holds the predicted endpoint through the true 10 s horizon.
    assert block["ade_m"] == pytest.approx(25.0)
    assert block["fde_m"] == pytest.approx(50.0)
    assert block["final_time_error_s"] == pytest.approx(-5.0)
    assert block["coverage_ratio"] == pytest.approx(0.5)


def test_common_physical_time_metric_uses_the_truth_path_frame():
    anchor = np.zeros(len(ch.CHANNELS))
    truth = np.zeros((2, len(ch.CHANNELS)))
    truth[:, ch.IDX["e"]] = [100.0, 200.0]
    predicted = truth.copy()
    predicted[:, ch.IDX["n"]] = 50.0

    block = common_physical_time_flight_metrics(
        anchor_values=anchor,
        predicted_values=predicted,
        predicted_offsets_s=np.array([1.0, 2.0]),
        predicted_final_time_s=2.0,
        truth_values=truth,
        truth_offsets_s=np.array([1.0, 2.0]),
        true_final_time_s=2.0,
        points=2,
    )

    assert block["along_track_m"]["mean_abs"] == pytest.approx(0.0, abs=1e-9)
    assert block["cross_track_m"]["mean_signed"] == pytest.approx(50.0)
    assert block["vertical_m"]["mean_abs"] == pytest.approx(0.0, abs=1e-9)


def test_common_physical_time_metric_rejects_nonfinite_or_nonmonotonic_paths():
    anchor = np.zeros(len(ch.CHANNELS))
    path = np.zeros((2, len(ch.CHANNELS)))
    arguments = dict(
        anchor_values=anchor,
        predicted_values=path,
        predicted_final_time_s=2.0,
        truth_values=path,
        truth_offsets_s=np.array([1.0, 2.0]),
        true_final_time_s=2.0,
        points=2,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        common_physical_time_flight_metrics(
            **arguments, predicted_offsets_s=np.array([1.0, 1.0])
        )
    bad = path.copy()
    bad[0, ch.IDX["u"]] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        common_physical_time_flight_metrics(
            **{**arguments, "predicted_values": bad},
            predicted_offsets_s=np.array([1.0, 2.0]),
        )


def test_state_velocity_is_derived_from_the_same_piecewise_linear_position_curve():
    anchor = np.zeros(len(ch.CHANNELS))
    predicted = np.zeros((2, len(ch.CHANNELS)))
    predicted[:, ch.IDX["e"]] = [10.0, 40.0]
    predicted[:, ch.IDX["n"]] = [0.0, 8.0]
    predicted[:, list(ch.VELOCITY_IDX)] = 999.0

    derived = states_with_derived_velocity(
        anchor, predicted, np.array([2.0, 4.0])
    )

    assert derived[:, ch.IDX["edot"]].tolist() == pytest.approx([5.0, 7.5])
    assert derived[:, ch.IDX["ndot"]].tolist() == pytest.approx([0.0, 2.0])
    assert derived[:, ch.IDX["udot"]].tolist() == pytest.approx([0.0, 0.0])
    assert derived[:, list(ch.POSITION_IDX)] == pytest.approx(
        predicted[:, list(ch.POSITION_IDX)]
    )


def test_state_loss_has_one_explicit_physical_output_endpoint_task():
    config = TSConfig(
        final_time_loss_weight=0.0,
        state_endpoint_loss_weight=1.0,
        kinematic_consistency_loss_weight=0.0,
        terminal_loss_weight=99.0,
        validation_common_grid_points=2,
    )
    target = torch.zeros((1, 2, len(ch.CHANNELS)))
    predicted = target.clone()
    predicted[0, -1, ch.IDX["e"]] = 1.0

    loss = prediction_loss(
        StatePrediction(states=predicted, final_time_s=torch.tensor([2.0])),
        torch.zeros((1, len(ch.CHANNELS))),
        target,
        torch.ones_like(target),
        torch.tensor([2.0]),
        torch.ones(1),
        config,
        _identity_normalizer(),
    )

    assert float(loss) == pytest.approx((1.0 / 2.0 + 1.0) / 10_000.0**2)


def test_full_position_and_endpoint_loss_ignore_the_padded_suffix():
    config = TSConfig(
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=3,
        final_time_loss_weight=0.0,
        state_endpoint_loss_weight=1.0,
        kinematic_consistency_loss_weight=0.0,
        terminal_loss_weight=99.0,
    )
    target = torch.zeros((1, 3, len(ch.CHANNELS)))
    predicted = target.clone()
    predicted[0, 1, ch.IDX["e"]] = 1.0
    predicted[0, 2, ch.IDX["e"]] = 1000.0
    weights = torch.zeros_like(target)
    # The first two rows are valid. The 1000 m padded suffix must not enter the loss.
    weights[0, :2, ch.IDX["n"]] = 1.0

    loss = prediction_loss(
        StatePrediction(states=predicted, final_time_s=torch.tensor([3.0])),
        torch.zeros((1, len(ch.CHANNELS))),
        target,
        weights,
        torch.tensor([3.0]),
        torch.ones(1),
        config,
        _identity_normalizer(),
    )

    # Row 1 is the last supervised output endpoint, so it contributes once to the path
    # average and once to the explicit endpoint task. Row 2 is padding and contributes 0.
    assert float(loss) == pytest.approx((1.0 / 2.0 + 1.0) / 10_000.0**2)


def test_state_endpoint_rejects_noncontiguous_position_supervision():
    config = TSConfig(
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=3,
        final_time_loss_weight=0.0,
    )
    target = torch.zeros((1, 3, len(ch.CHANNELS)))
    weights = torch.zeros_like(target)
    weights[0, (0, 2), ch.IDX["e"]] = 1.0

    with pytest.raises(ValueError, match="contiguous prefix"):
        prediction_loss(
            StatePrediction(states=target.clone(), final_time_s=torch.tensor([3.0])),
            torch.zeros((1, len(ch.CHANNELS))),
            target,
            weights,
            torch.tensor([3.0]),
            torch.ones(1),
            config,
            _identity_normalizer(),
        )
