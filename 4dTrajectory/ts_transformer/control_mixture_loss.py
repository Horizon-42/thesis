"""Best-of-K training objective for the isolated control-mixture strategy."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config import TSConfig
from control_mixture import ControlMixturePrediction
from dataset import Normalizer
from prediction_outputs import ControlPrediction


def _repeat_dynamics(
    dynamics: dict[str, torch.Tensor], expert_count: int
) -> dict[str, torch.Tensor]:
    return {
        name: value.repeat_interleave(expert_count, dim=0)
        for name, value in dynamics.items()
    }


def _candidate_diversity(
    prediction: ControlMixturePrediction,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
) -> torch.Tensor:
    """Bounded pairwise similarity penalty over normalized controls and time fractions."""
    lower = dynamics["control_lower"][:, None, None, :]
    upper = dynamics["control_upper"][:, None, None, :]
    unit_controls = (prediction.controls - lower) / (upper - lower)
    fractions = prediction.segment_durations / prediction.final_time_s.unsqueeze(-1)
    normalized_final_time = prediction.final_time_s / config.final_time_scale_s
    pair_similarities = []
    for left in range(prediction.expert_count):
        for right in range(left + 1, prediction.expert_count):
            control_distance = (
                unit_controls[:, left] - unit_controls[:, right]
            ).square().mean(dim=(1, 2))
            duration_distance = (
                fractions[:, left] - fractions[:, right]
            ).square().mean(dim=1)
            final_time_distance = (
                normalized_final_time[:, left] - normalized_final_time[:, right]
            ).square()
            # Similar candidates approach one; meaningfully different bounded profiles
            # approach zero. The exponential prevents an unbounded reward for extremes.
            pair_similarities.append(
                torch.exp(
                    -(control_distance + duration_distance + final_time_distance) / 0.01
                )
            )
    return torch.stack(pair_similarities, dim=1).mean(dim=1)


def mixture_prediction_loss_components(
    prediction: ControlMixturePrediction,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    flight_weights: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor],
):
    """Return best-of-K reconstruction plus deployable selector/diversity losses."""
    # Imported lazily so the prediction/model module never depends on the training loop.
    from train import LossComponents, control_prediction_loss_terms

    batch, experts, segments, controls = prediction.controls.shape
    flat_prediction = ControlPrediction(
        controls=prediction.controls.reshape(batch * experts, segments, controls),
        segment_durations=prediction.segment_durations.reshape(batch * experts, segments),
        final_time_s=prediction.final_time_s.reshape(batch * experts),
    )
    flat_terms = control_prediction_loss_terms(
        flat_prediction,
        normalized_anchor_state.repeat_interleave(experts, dim=0),
        target_states.repeat_interleave(experts, dim=0),
        state_weights.repeat_interleave(experts, dim=0),
        target_final_time_s.repeat_interleave(experts, dim=0),
        config,
        normalizer,
        _repeat_dynamics(dynamics, experts),
    )

    def shaped(values: torch.Tensor) -> torch.Tensor:
        return values.reshape(batch, experts)

    expert_total = shaped(flat_terms.total)
    winner = expert_total.detach().argmin(dim=1)
    rows = torch.arange(batch, device=winner.device)

    def selected(values: torch.Tensor) -> torch.Tensor:
        return shaped(values)[rows, winner]

    selector = config.control_mixture_selector_loss_weight * F.cross_entropy(
        prediction.selection_logits, winner, reduction="none"
    )
    diversity = (
        config.control_mixture_diversity_loss_weight
        * _candidate_diversity(prediction, dynamics, config)
    )

    def weighted_mean(values: torch.Tensor) -> torch.Tensor:
        return (values * flight_weights).mean()

    zero = weighted_mean(expert_total.new_zeros(batch))
    return LossComponents(
        state=weighted_mean(selected(flat_terms.state)),
        final_time=weighted_mean(selected(flat_terms.final_time)),
        kinematic=zero,
        terminal=weighted_mean(selected(flat_terms.terminal)),
        extras={
            "control_effort": weighted_mean(selected(flat_terms.effort)),
            "control_smoothness": weighted_mean(selected(flat_terms.smoothness)),
            "mixture_selector": weighted_mean(selector),
            "mixture_diversity": weighted_mean(diversity),
        },
    )
