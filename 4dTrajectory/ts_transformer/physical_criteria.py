"""Physical-distance criteria shared by the control loss and the basis-fit study."""

from __future__ import annotations

import torch

from ts_transformer.channels import POSITION_IDX
from ts_transformer.dataset import Normalizer
from ts_transformer.fixed_dt_supervision import FixedDTControlSupervision


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
