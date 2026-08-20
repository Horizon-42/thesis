"""Future-aware inverse-dynamics schedules used only as train-only teacher targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from channels import states_from_channels
from control.dynamics.inverse import segment_controls
from control.heads import NEUTRAL_CONTROLS
from dataset import FixedAnchorTrajectoryWindows
from prediction_outputs import ControlPrediction
from reference_velocity import rebuild_reference_velocities
from batch_contract import unpack_batch


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
    # unpack_batch, not a fixed-width tuple: dense supervision is present only under the
    # fixed-dt loss grid, and hard-unpacking seven fields made this builder unusable under
    # every native-grid recipe (simple-v1 among them) with a bare ValueError.
    x, _target, _weights, final_time, _flight_weights, dynamics, supervision = (
        unpack_batch(dataset.batch(np.array([index])))
    )
    if supervision is None:
        raise ValueError(
            "inverse-dynamics teacher targets need the dense reference future; build the "
            "cohort with control_state_loss_grid='fixed-dt'"
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
        source=dataset.config.reference_velocity_source,
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
    # The inverse is selected by the SAME config field as the forward rollout, so a teacher
    # can never be solved against equations the training rollout does not integrate.
    inverse = segment_controls(
        physical_states,
        reference_times,
        config=dataset.config,
        aero_params=dynamics["aero_params"][0].numpy(),
        max_thrust_n=float(dynamics["max_thrust_n"][0]),
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
    del dynamics
    control = torch.tensor(NEUTRAL_CONTROLS, dtype=torch.float32, device=device)
    durations = torch.full(
        (1, n_segments), final_time_s / n_segments, device=device
    )
    return ControlPrediction(
        controls=control.view(1, 1, 3).expand(1, n_segments, 3),
        segment_durations=durations,
        final_time_s=torch.tensor([final_time_s], device=device),
    )
