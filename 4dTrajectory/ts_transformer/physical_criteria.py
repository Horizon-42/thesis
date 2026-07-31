"""Shared physical-distance criterion for oracle and learned-control training."""

from __future__ import annotations

import torch

from channels import POSITION_IDX
from dataset import Normalizer
from fixed_dt_supervision import FixedDTControlSupervision


PHYSICAL_CRITERIA_DISTANCE_SCALE_M = 100.0
PHYSICAL_CRITERIA_SMOOTH_MAX_TEMPERATURE = 0.1


def smooth_maximum(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    temperature: float = PHYSICAL_CRITERIA_SMOOTH_MAX_TEMPERATURE,
) -> torch.Tensor:
    """Differentiable maximum with the same units and shape as its inputs."""
    if temperature <= 0.0:
        raise ValueError("smooth-maximum temperature must be positive")
    return temperature * torch.logsumexp(
        torch.stack((first, second), dim=0) / temperature,
        dim=0,
    )


def fixed_dt_position_ade_m(
    physical_query_states: torch.Tensor,
    supervision: FixedDTControlSupervision,
    normalizer: Normalizer,
) -> torch.Tensor:
    """Per-flight 3-D position ADE on valid measured/fitted fixed-dt rows."""
    dtype = physical_query_states.dtype
    device = physical_query_states.device
    indices = list(POSITION_IDX)
    mean = torch.as_tensor(normalizer.mean[indices], dtype=dtype, device=device)
    scale = torch.as_tensor(normalizer.std[indices], dtype=dtype, device=device)
    targets = supervision.states[..., indices].to(dtype=dtype, device=device)
    weights = supervision.weights[..., indices].to(dtype=dtype, device=device)
    component_active = (
        (weights > 0.0) & supervision.valid.to(device=device).unsqueeze(-1)
    )
    physical_targets = targets * scale + mean
    delta = (physical_query_states[..., indices] - physical_targets) * component_active
    distance = torch.sqrt(delta.square().sum(dim=-1) + 1e-12)
    row_active = component_active.any(dim=-1)
    return (distance * row_active).sum(dim=1) / row_active.sum(dim=1).clamp(min=1)


def terminal_position_error_m(
    normalized_segment_end_states: torch.Tensor,
    normalized_terminal_targets: torch.Tensor,
    normalizer: Normalizer,
) -> torch.Tensor:
    """Per-flight 3-D distance between the final rollout state and runway target."""
    indices = list(POSITION_IDX)
    endpoint = normalized_segment_end_states[:, -1, indices]
    target = normalized_terminal_targets[:, indices].to(
        dtype=endpoint.dtype, device=endpoint.device
    )
    scale = torch.as_tensor(
        normalizer.std[indices], dtype=endpoint.dtype, device=endpoint.device
    )
    return torch.linalg.vector_norm((endpoint - target) * scale, dim=-1)


def physical_criteria_loss(
    ade_m: torch.Tensor,
    terminal_error_m: torch.Tensor,
) -> torch.Tensor:
    """Smooth worst of ADE/100 m and terminal-error/100 m, per flight."""
    scale = PHYSICAL_CRITERIA_DISTANCE_SCALE_M
    return smooth_maximum(ade_m / scale, terminal_error_m / scale)
