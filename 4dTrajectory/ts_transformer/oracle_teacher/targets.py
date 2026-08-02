"""Future-aware inverse-dynamics schedules used only as train-only teacher targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from channels import POSITION_IDX, states_from_channels
from control_oracle_initialization import inverse_dynamics_controls
from dataset import FixedAnchorTrajectoryWindows
from prediction_outputs import ControlPrediction
from reference_velocity import (
    REFERENCE_VELOCITY_SMOOTHED_POSITION_DIFFERENCE,
    rebuild_reference_velocities,
)


@dataclass(frozen=True)
class OracleTeacherTarget:
    dataset_id: str
    controls: np.ndarray
    segment_durations_s: np.ndarray
    final_time_s: float
    clipped_fraction: np.ndarray
    reference_points: int

    def prediction(self, device: torch.device) -> ControlPrediction:
        return ControlPrediction(
            controls=torch.as_tensor(
                self.controls, dtype=torch.float32, device=device
            ).unsqueeze(0),
            segment_durations=torch.as_tensor(
                self.segment_durations_s, dtype=torch.float32, device=device
            ).unsqueeze(0),
            final_time_s=torch.tensor(
                [self.final_time_s], dtype=torch.float32, device=device
            ),
        )


def build_inverse_dynamics_target(
    dataset: FixedAnchorTrajectoryWindows,
    index: int,
) -> OracleTeacherTarget:
    """Build one uniform-clock N-segment teacher from a known outer-train future.

    Fitted-tail velocity placeholders are never treated as observations.  All future
    velocity rows are rebuilt from the complete position reference before applying the
    algebraic inverse dynamics, which is an explicit teacher-only privilege.
    """
    x, _target, _weights, final_time, _flight_weights, dynamics, supervision = (
        dataset.batch(np.array([index]))
    )
    valid = supervision.valid[0].numpy()
    anchor = dataset.normalizer.decode(x[0, -1].numpy().astype(np.float64))
    future = dataset.normalizer.decode(
        supervision.states[0, valid].numpy().astype(np.float64)
    )
    reference_channels = np.concatenate((anchor[None, :], future), axis=0)
    reference_times = np.concatenate(
        (
            np.array([0.0], dtype=np.float64),
            supervision.query_offsets_s[0, valid].numpy(),
        )
    )
    reference_channels = rebuild_reference_velocities(
        reference_times,
        reference_channels,
        source=REFERENCE_VELOCITY_SMOOTHED_POSITION_DIFFERENCE,
        valid_rows=np.ones(len(reference_times), dtype=bool),
    )
    mass_kg = float(dynamics["initial_state"][0, -1])
    reference_states = states_from_channels(
        reference_times,
        reference_channels,
        dataset.series[dataset.index[index][0]].frame,
        mass_kg=mass_kg,
    )
    physical_states = np.asarray(
        [
            [
                state.latitude,
                state.longitude,
                state.altitude,
                state.V,
                state.psi,
                state.gamma,
                state.m,
            ]
            for _time, state in reference_states
        ],
        dtype=np.float64,
    )
    total_duration_s = float(final_time[0])
    inverse = inverse_dynamics_controls(
        physical_states,
        reference_times,
        aero_params=dynamics["aero_params"][0].numpy(),
        control_lower=dynamics["control_lower"][0].numpy(),
        control_upper=dynamics["control_upper"][0].numpy(),
        n_segments=int(dataset.config.n_segments),
        total_duration_s=total_duration_s,
    )
    durations = np.full(
        int(dataset.config.n_segments),
        total_duration_s / int(dataset.config.n_segments),
        dtype=np.float64,
    )
    series_index, _anchor_index = dataset.index[index]
    return OracleTeacherTarget(
        dataset_id=dataset.series[series_index].dataset_id,
        controls=inverse.controls,
        segment_durations_s=durations,
        final_time_s=total_duration_s,
        clipped_fraction=inverse.clipped_fraction,
        reference_points=int(valid.sum()),
    )


def neutral_prediction(
    dynamics: dict[str, torch.Tensor],
    *,
    n_segments: int,
    final_time_s: float,
    device: torch.device,
) -> ControlPrediction:
    """Return the deterministic control head's pre-training neutral schedule."""
    lower = dynamics["control_lower"][0].to(device)
    upper = dynamics["control_upper"][0].to(device)
    unit = torch.tensor((0.2, 0.5, 1.0 / 3.0), device=device)
    control = lower + unit * (upper - lower)
    durations = torch.full(
        (1, n_segments), final_time_s / n_segments, device=device
    )
    return ControlPrediction(
        controls=control.view(1, 1, 3).expand(1, n_segments, 3),
        segment_durations=durations,
        final_time_s=torch.tensor([final_time_s], device=device),
    )
