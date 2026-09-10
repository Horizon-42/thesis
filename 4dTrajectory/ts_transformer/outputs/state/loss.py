"""The state objective: the physical-position / duration airport-macro loss."""

from __future__ import annotations

import torch

from ts_transformer.data.batch_contract import LossComponents
from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.config import HORIZON_FULL, HORIZON_NORMALIZED, HORIZON_WINDOW, TSConfig
from ts_transformer.data.dataset import Normalizer
from ts_transformer.data.fixed_dt_supervision import FixedDTControlSupervision
from ts_transformer.training.objective import ProcedureMultipliers, procedure_loss
from ts_transformer.outputs.state.model import StatePrediction


STATE_TARGET_CONTRACTS = {
    HORIZON_NORMALIZED: "normalized-output-true-time-physical-position-duration-v1",
    HORIZON_FULL: "full-horizon-physical-position-duration-v1",
    HORIZON_WINDOW: "recursive-window-physical-position-duration-v1",
}

STATE_LOSS_COMPONENT_NAMES = ("state", "final_time", "kinematic", "terminal", "procedure")


def _sample_uniform_progress_nodes(
    nodes: torch.Tensor,
    query_progress: torch.Tensor,
) -> torch.Tensor:
    """Differentiably sample ``[B,N+1,D]`` nodes defined at progress ``0..1``."""
    if nodes.ndim != 3 or query_progress.ndim != 2:
        raise ValueError("progress sampling requires [B,N+1,D] nodes and [B,Q] queries")
    if nodes.shape[0] != query_progress.shape[0] or nodes.shape[1] < 2:
        raise ValueError("progress sampling batch/segment shapes do not align")
    segments = nodes.shape[1] - 1
    scaled = query_progress.clamp(min=0.0, max=1.0) * segments
    left = torch.floor(scaled).to(torch.long).clamp(max=segments - 1)
    fraction = (scaled - left.to(scaled.dtype)).unsqueeze(-1)
    gather_index = left.unsqueeze(-1).expand(-1, -1, nodes.shape[-1])
    left_values = torch.gather(nodes, 1, gather_index)
    right_values = torch.gather(nodes, 1, gather_index + 1)
    return left_values + fraction * (right_values - left_values)


def state_prediction_loss_components(
    prediction: StatePrediction,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    flight_weights: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor] | None = None,
    dense_supervision: FixedDTControlSupervision | None = None,
    *,
    multipliers: ProcedureMultipliers | None = None,
) -> LossComponents:
    """Return the direct-state physical-position/time airport-macro objective."""
    del dense_supervision
    if prediction.states.shape != target_states.shape:
        raise ValueError("state prediction and target tensors must align")

    position_indices = list(POSITION_IDX)
    position_std = torch.as_tensor(
        normalizer.std[position_indices],
        dtype=prediction.states.dtype,
        device=prediction.states.device,
    )
    predicted_position = prediction.states[..., position_indices]
    target_position = target_states[..., position_indices]
    point_weights = state_weights[..., position_indices].sum(dim=-1)

    # The output endpoint is the last position carrying supervision, not necessarily the
    # final tensor row: fixed-horizon targets can contain a padded suffix. This is a second
    # task over the same physical target, not a runway-centre prior or a fitted trajectory.
    valid_position = point_weights > 0.0
    if not torch.all(valid_position.any(dim=1)):
        raise ValueError("every state target must contain a supervised position endpoint")
    if torch.any(valid_position[:, 1:] & ~valid_position[:, :-1]):
        raise ValueError("supervised state positions must form a contiguous prefix")
    last_index = valid_position.sum(dim=1) - 1
    gather_index = last_index[:, None, None].expand(-1, 1, len(position_indices))
    predicted_endpoint = torch.gather(predicted_position, 1, gather_index).squeeze(1)
    target_endpoint = torch.gather(target_position, 1, gather_index).squeeze(1)

    if config.horizon_mode == HORIZON_NORMALIZED:
        points = config.validation_common_grid_points
        progress = torch.arange(
            1,
            points + 1,
            dtype=prediction.states.dtype,
            device=prediction.states.device,
        ) / points
        truth_progress = progress.unsqueeze(0).expand(len(target_states), -1)
        prediction_progress = (
            truth_progress
            * target_final_time_s.unsqueeze(1)
            / prediction.final_time_s.unsqueeze(1).clamp(min=1e-6)
        ).clamp(min=0.0, max=1.0)
        anchor_position = normalized_anchor_state[:, None, position_indices]
        predicted_nodes = torch.cat((anchor_position, predicted_position), dim=1)
        target_nodes = torch.cat((anchor_position, target_position), dim=1)
        anchor_weight = point_weights[:, :1]
        weight_nodes = torch.cat((anchor_weight, point_weights), dim=1)
        predicted_position = _sample_uniform_progress_nodes(
            predicted_nodes, prediction_progress
        )
        target_position = _sample_uniform_progress_nodes(
            target_nodes, truth_progress
        )
        point_weights = _sample_uniform_progress_nodes(
            weight_nodes.unsqueeze(-1), truth_progress
        ).squeeze(-1)

    physical_delta = (predicted_position - target_position) * position_std
    squared_distance = physical_delta.square().sum(dim=-1)
    state_loss = (
        (squared_distance * point_weights).sum(dim=1)
        / point_weights.sum(dim=1).clamp(min=1e-12)
        / (config.position_loss_scale_m**2)
    )
    endpoint_delta = (predicted_endpoint - target_endpoint) * position_std
    endpoint_loss = (
        endpoint_delta.square().sum(dim=-1)
        / (config.position_loss_scale_m**2)
    )
    time_loss = (
        (prediction.final_time_s - target_final_time_s) / config.final_time_scale_s
    ).square()
    zero = state_loss.new_zeros(state_loss.shape)
    # On the row grid (not the resampled progress nodes): the gate is a per-row decision.
    procedure, procedure_diagnostics = procedure_loss(
        prediction.states,
        target_states,
        state_weights[..., position_indices].sum(dim=-1),
        config,
        normalizer,
        dynamics,
        multipliers,
    )

    def weighted_mean(values: torch.Tensor) -> torch.Tensor:
        return (values * flight_weights).mean()

    return LossComponents(
        state=weighted_mean(state_loss),
        final_time=config.final_time_loss_weight * weighted_mean(time_loss),
        kinematic=weighted_mean(zero),
        terminal=config.state_endpoint_loss_weight * weighted_mean(endpoint_loss),
        extras={"procedure": weighted_mean(procedure)},
        diagnostics=procedure_diagnostics,
    )
