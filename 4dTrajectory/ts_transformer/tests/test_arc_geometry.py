"""The arc-length geometry metrics and the terminal-state emphasis.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ts_transformer.data.channels as ch
from ts_transformer.geometry.arc_length_geometry import (
    arc_length_geometry_metrics,
    arc_length_velocity_metrics,
    resample_horizontal_arc_length_numpy,
)
from ts_transformer.config import TSConfig
from ts_transformer.geometry.terminal_state_loss import (
    last_reliable_terminal_velocity_target,
    terminal_state_metrics_numpy,
)
from ts_transformer.data.dataset import Normalizer
from ts_transformer.data.fixed_dt_supervision import (
    FixedDTControlSupervision,
    build_fixed_dt_supervision,
)
from ts_transformer.training.fixed_anchor_validation import (
    ARC_LENGTH_POSITION_END_WEIGHT,
    TERMINAL_CROSS_TRACK_EMPHASIS,
    TERMINAL_VERTICAL_EMPHASIS,
    fixed_anchor_arc_length_geometry_metrics,
    fixed_anchor_common_weights_and_terminal_velocity,
)


def _identity_normalizer() -> Normalizer:
    return Normalizer(
        mean=np.zeros(len(ch.CHANNELS), dtype=np.float64),
        std=np.ones(len(ch.CHANNELS), dtype=np.float64),
    )


def test_horizontal_arc_resampling_is_independent_of_node_spacing():
    sparse = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 3.0]], dtype=np.float64
    )
    uneven = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [3.0, 0.0, 3.0]],
        dtype=np.float64,
    )
    expected = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        resample_horizontal_arc_length_numpy(sparse, points=4), expected
    )
    np.testing.assert_allclose(
        resample_horizontal_arc_length_numpy(uneven, points=4), expected
    )
    metrics = arc_length_geometry_metrics(
        uneven, sparse, _identity_normalizer(), points=4
    )
    assert metrics["loss"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["distance_mean_m"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["path_length_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["path_length_log_error"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["horizontal_mean_m"] == pytest.approx(0.0, abs=1e-12)


def test_arc_length_geometry_detects_shape_error_with_matching_endpoints():
    straight = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64
    )
    dogleg = np.array(
        [[0.0, 0.0, 0.0], [1.5, 1.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    metrics = arc_length_geometry_metrics(
        straight, dogleg, _identity_normalizer(), points=9
    )

    assert metrics["horizontal_mean_m"] > 0.25
    assert metrics["terminal_position_m"] == pytest.approx(0.0)
    assert metrics["path_length_log_error"] > 0.0


def test_arc_position_progress_weight_emphasizes_late_geometry_error():
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    early_error = reference.copy()
    early_error[0, 2] = 2.0
    late_error = reference.copy()
    late_error[-1, 2] = 2.0

    early = arc_length_geometry_metrics(
        early_error,
        reference,
        _identity_normalizer(),
        points=3,
        position_end_weight=ARC_LENGTH_POSITION_END_WEIGHT,
    )
    late = arc_length_geometry_metrics(
        late_error,
        reference,
        _identity_normalizer(),
        points=3,
        position_end_weight=ARC_LENGTH_POSITION_END_WEIGHT,
    )

    assert early["unweighted_loss"] == pytest.approx(late["unweighted_loss"])
    assert late["loss"] > early["loss"]


def test_terminal_state_metrics_numpy_applies_the_frozen_emphasis():
    """The shape constants used to be config fields (retired with the arc-length objective,
    T1-11); they are frozen so the ``arc_length_*`` diagnostic keys stay comparable across
    the artifact history. A non-zero error pins the arithmetic — and the numbers."""
    from ts_transformer.config import COORDINATE_FRAME_RUNWAY_ALIGNED
    predicted = np.zeros(len(ch.CHANNELS))
    predicted[list(ch.POSITION_IDX)] = 1.0          # 1 m along, 1 m cross, 1 m vertical
    reference = np.zeros(len(ch.CHANNELS))
    metrics = terminal_state_metrics_numpy(
        predicted, reference, np.zeros(len(ch.VELOCITY_IDX)), 0.0,
        coordinate_frame=COORDINATE_FRAME_RUNWAY_ALIGNED,
        cross_track_emphasis=TERMINAL_CROSS_TRACK_EMPHASIS,
        vertical_emphasis=TERMINAL_VERTICAL_EMPHASIS,
    )
    assert (ARC_LENGTH_POSITION_END_WEIGHT, TERMINAL_CROSS_TRACK_EMPHASIS, TERMINAL_VERTICAL_EMPHASIS) == (4.0, 3.0, 5.0)
    assert metrics["position_runway_components_m"] == pytest.approx(1.0 + 3.0 * 1.0 + 5.0 * 1.0)
    assert metrics["position_vector_m"] == pytest.approx(np.sqrt(3.0))


def test_fixed_anchor_arc_geometry_filters_the_same_sparse_reference_rows():
    channels = len(ch.CHANNELS)
    config = TSConfig(seq_len=1, n_segments=2)
    anchor = np.zeros((1, channels), dtype=np.float32)
    predicted = np.zeros((1, 2, channels), dtype=np.float32)
    predicted[0, :, ch.POSITION_IDX[0]] = [1.0, 4.0]
    reference = np.zeros((5, channels), dtype=np.float32)
    reference[:, ch.POSITION_IDX[0]] = [0.0, 1.0, 999.0, 999.0, 4.0]
    weights = np.zeros_like(reference)
    weights[:2] = 1.0 / channels
    weights[-1, list(ch.POSITION_IDX)] = 1.0 / channels
    item = SimpleNamespace(
        dataset_id="KAAA:SPARSE",
        scenario=SimpleNamespace(target=SimpleNamespace(psi=0.0)),
        times=np.array([0.0]),
        values=reference[:1],
        supervision_times=np.arange(5, dtype=np.float64),
        supervision_values=reference,
        supervision_weights=weights,
    )

    metrics = fixed_anchor_arc_length_geometry_metrics(
        [item],
        config,
        anchor,
        predicted,
        np.zeros((1, len(ch.VELOCITY_IDX)), dtype=np.float32),
        _identity_normalizer(),
        points=3,
        anchor=0,
    )

    assert metrics["arc_length_geometry_loss"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["arc_length_distance_mean_m"] == pytest.approx(0.0, abs=1e-12)


def test_arc_length_velocity_metrics_follow_position_alignment_and_mask_tail():
    predicted_positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    reference_positions = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    predicted_velocity = np.array(
        [[10.0, 2.0, -1.0], [10.0, 2.0, -1.0], [99.0, 99.0, 99.0]]
    )
    reference_velocity = np.array(
        [[10.0, 0.0, -2.0], [999.0, 999.0, 999.0]]
    )

    metrics = arc_length_velocity_metrics(
        predicted_positions,
        predicted_velocity,
        reference_positions,
        reference_velocity,
        np.array([True, False]),
        points=4,
    )

    assert metrics["velocity_valid_points"] == 1
    assert metrics["horizontal_velocity_mae_mps"] == pytest.approx(2.0)
    assert metrics["horizontal_tangent_mean"] == pytest.approx(
        1.0 - 10.0 / math.sqrt(104.0)
    )
    assert metrics["horizontal_speed_mae_mps"] == pytest.approx(
        math.sqrt(104.0) - 10.0
    )
    assert metrics["vertical_velocity_mae_mps"] == pytest.approx(1.0)


def test_terminal_velocity_target_falls_back_to_observed_anchor():
    states = torch.full((1, 2, len(ch.CHANNELS)), 999.0)
    weights = torch.zeros_like(states)
    supervision = FixedDTControlSupervision(
        query_offsets_s=torch.tensor([[2.0, 4.0]], dtype=torch.float64),
        states=states,
        weights=weights,
        valid=torch.ones(1, 2, dtype=torch.bool),
    )
    anchor = torch.zeros(1, len(ch.CHANNELS))
    anchor[0, list(ch.VELOCITY_IDX)] = torch.tensor([80.0, 2.0, -3.0])

    target = last_reliable_terminal_velocity_target(anchor, supervision)

    torch.testing.assert_close(target, torch.tensor([[80.0, 2.0, -3.0]]))


def test_validation_terminal_velocity_matches_fixed_dt_training_before_off_grid_crossing():
    config = TSConfig(seq_len=2, dt_s=2.0)
    supervision_values = np.zeros((4, len(ch.CHANNELS)), dtype=np.float32)
    supervision_values[2, list(ch.VELOCITY_IDX)] = [10.0, 20.0, 30.0]
    supervision_values[3, list(ch.VELOCITY_IDX)] = [100.0, 200.0, 300.0]
    item = SimpleNamespace(
        dataset_id="KAAA:OFFGRID",
        times=np.array([0.0, 2.0, 4.0]),
        values=supervision_values[:3],
        supervision_times=np.array([0.0, 2.0, 4.0, 5.0]),
        supervision_values=supervision_values,
        supervision_weights=np.full_like(
            supervision_values, 1.0 / len(ch.CHANNELS)
        ),
    )
    dense = build_fixed_dt_supervision(
        [item], [supervision_values], [(0, config.seq_len - 1)], dt_s=config.dt_s
    )
    training_target = last_reliable_terminal_velocity_target(
        torch.from_numpy(item.values[-1:]), dense
    )
    _weights, validation_target = fixed_anchor_common_weights_and_terminal_velocity(
        [item], config, np.array([1.0]), np.array([3.0]), anchor=config.seq_len - 1
    )

    np.testing.assert_allclose(
        validation_target, training_target.numpy(), rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(validation_target[0], [10.0, 20.0, 30.0])
