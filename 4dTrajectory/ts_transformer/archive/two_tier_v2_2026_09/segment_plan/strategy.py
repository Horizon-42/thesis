"""The segment-plan path (two-tier v2 §4): the plan layer that reads the observed window
as coarse SEGMENT TOKENS and decodes the next M segments' waypoints in one shot.

What this strategy answers the spine with:

- the batch context is the truth's coarse plan at each drawn anchor (`SegmentPlanContext`
  → `labels.segment_labels`: the position at each segment's end in runway axes, whether
  the flight has arrived by it, the arrival fraction) and the flight's runway course — the
  one context key the MODEL reads (the features are rotated into runway axes by it);
- the model is `SegmentPlanModel` (the vendored iTransformer encoder over the segment
  features, `features.segment_features`, under one of two token axes); the loss
  `segment_plan_loss_components`;
- the deployable forecast is the decoded plan (`decode.decode_plan` → `plan_rows`): the
  waypoints every 30 s until the predicted arrival segment, then the threshold at the
  predicted arrival time — a coarse trajectory of M rows at most, velocities derived;
- the validation replay is the same plan padded to M rows (`decode.replay_rows`), on the
  same clock, so the epoch report's common-grid ADE reads the plan the head drew.

No dynamics, no guidance: the plan is what L1 (`plan_conditioning=waypoints`) is told.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.backbone.adapters import ITransformerEncoderStack
from ts_transformer.config import PLAN_WAYPOINT_SEGMENT_S, PREDICTION_SEGMENT_PLAN, SegmentPlanOutput, TSConfig
from ts_transformer.data.batch_contract import LossComponents, anchor_state, model_forward
from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.geometry.metrics import states_with_derived_velocity
from ts_transformer.inference.forecast import Forecast, history_batch
from ts_transformer.outputs.base import ForecastOptions, OutputStrategy, Replay, WindowContext
from ts_transformer.outputs.segment_plan.decode import DecodedPlan, decode_plan, plan_rows, replay_rows
from ts_transformer.outputs.segment_plan.features import SEGMENT_FEATURES
from ts_transformer.outputs.segment_plan.labels import (
    CONTEXT_ARRIVED,
    CONTEXT_RUNWAY_HEADING,
    CONTEXT_TARGET_CHART,
    CONTEXT_VALID,
    SEGMENT_TARGET_CONTRACT,
    runway_heading_rad,
    segment_labels,
)
from ts_transformer.outputs.segment_plan.model import (
    SEGMENT_PLAN_LOSS_COMPONENT_NAMES,
    SegmentPlanModel,
    SegmentPlanPrediction,
    encoder_token_length,
    prediction_rows,
    probe_segment_plan_context,
    segment_plan_loss_components,
)
from ts_transformer.outputs.segment_plan.readout import plan_reading, pool_airports, summarize_readings

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries, Normalizer, TrajectoryWindows
    from ts_transformer.data.fixed_dt_supervision import FixedDTControlSupervision
    from ts_transformer.training.objective import ProcedureMultipliers

#: The record's own entries (`record_fields`): the arrival segment the plan decoded
#: (0-based; null when none clears the threshold) and that segment's arrival probability.
ARRIVAL_SEGMENT_KEY = "segmentPlanArrivalSegment"
ARRIVAL_PROBABILITY_KEY = "segmentPlanArrivalProbability"


class SegmentPlanContext(WindowContext):
    """The truth's coarse plan at each window's anchor (`labels.segment_labels`). A
    fixed-anchor set (``cache_context_rows``) builds its rows once; a random-anchor set
    reads each row for the anchor it drew."""

    def __init__(self, windows: TrajectoryWindows):
        self.windows = windows
        self._rows: list[dict[str, np.ndarray]] | None = None
        segments = int(windows.config.segment_plan_segments)
        if windows.cache_context_rows:
            self._rows = [self._build_row(index) for index in range(len(windows.index))]
            reached = float(np.mean([row[CONTEXT_VALID].sum() for row in self._rows])) if self._rows else 0.0
            arriving = sum(1 for row in self._rows if row[CONTEXT_ARRIVED][-1] > 0.5)
            self.summary = (
                f"segment plan: {len(self._rows)} anchors, {reached:.1f} of {segments} segments reached "
                f"by the truth on average, {arriving} arrive within the plan"
            )
        else:
            self.summary = f"segment plan: {segments} segments, read at each drawn anchor"

    def row(self, i: int) -> dict[str, np.ndarray]:
        return self._rows[i] if self._rows is not None else self._build_row(i)

    def _build_row(self, i: int) -> dict[str, np.ndarray]:
        s_idx, anchor = self.windows.index[i]
        return segment_labels(self.windows.series[s_idx], int(anchor), self.windows.config)


def decode_batch(
    prediction: SegmentPlanPrediction, anchors: np.ndarray, headings: np.ndarray,
) -> list[DecodedPlan]:
    """Every flight's plan off one prediction batch: ``anchors`` ``[B, C]`` physical anchor
    states, ``headings`` ``[B]`` runway courses."""
    positions, arrived, fraction = prediction_rows(prediction)
    return [
        decode_plan(positions[i], arrived[i], fraction[i], anchors[i, list(POSITION_IDX)], float(headings[i]))
        for i in range(len(positions))
    ]


def _states(anchor: np.ndarray, positions: np.ndarray, durations: np.ndarray, channels: int) -> np.ndarray:
    rows = np.zeros((len(positions), channels), dtype=np.float64)
    rows[:, list(POSITION_IDX)] = positions
    return states_with_derived_velocity(anchor, rows, durations)


def decode_series(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    device: torch.device,
    *,
    batch_size: int | None = None,
) -> list[DecodedPlan]:
    """Every flight's plan at ``anchor``: forward passes of ``batch_size`` flights (the
    checkpoint's own by default), the model in eval mode, no gradients."""
    batch_size = batch_size or config.batch_size
    model.eval()
    plans: list[DecodedPlan] = []
    with torch.no_grad():
        for start in range(0, len(series), batch_size):
            chunk = series[start : start + batch_size]
            history = torch.from_numpy(history_batch(chunk, config, normalizer, anchor)).to(device)
            headings = np.array([runway_heading_rad(item) for item in chunk], dtype=np.float64)
            context = {
                CONTEXT_RUNWAY_HEADING: torch.as_tensor(headings, dtype=torch.float32, device=device),
                CONTEXT_TARGET_CHART: torch.as_tensor(
                    np.stack([item.target_chart for item in chunk]), dtype=torch.float32, device=device,
                ),
            }
            prediction = model_forward(model, history, context)
            anchors = np.stack([item.values[anchor] for item in chunk]).astype(np.float64)
            plans.extend(decode_batch(prediction, anchors, headings))
    return plans


def forecast_from_plan(item: FlightSeries, plan: DecodedPlan, anchor: int, config: TSConfig) -> Forecast:
    """One decoded plan laid as the rows a record carries (`decode.plan_rows`), velocities derived."""
    anchor_values = np.asarray(item.values[anchor], dtype=np.float64)
    offsets, positions = plan_rows(plan, item.target_chart)
    durations = np.diff(np.concatenate([[0.0], offsets]))
    final_time_s = float(offsets[-1])
    return Forecast(
        times=float(item.times[anchor]) + offsets,
        values=_states(anchor_values, positions, durations, len(config.channels)),
        normalized_progress=offsets / final_time_s,
        anchor=anchor,
        final_time_s=final_time_s,
        predicted_final_time_s=final_time_s,
        horizon_mode=config.horizon_mode,
        passes=1,
        # a plan that arrives ends AT the threshold; one that does not is cut by its M segments
        truncated_at_threshold=plan.arrives,
        horizon_capped=not plan.arrives,
        sample_durations_s=durations,
        segment_durations_s=durations,
        prediction_output=config.prediction_output,
        segment_plan_arrival_segment=plan.arrival_segment,
        segment_plan_arrival_probability=plan.arrival_probability,
    )


def forecast_segment_plan(
    model: nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    anchor: int,
    device: torch.device,
) -> list[Forecast]:
    """Every flight's plan at ``anchor``, decoded and laid as the rows a record carries."""
    return [
        forecast_from_plan(item, plan, anchor, config)
        for item, plan in zip(series, decode_series(model, series, config, normalizer, anchor, device), strict=True)
    ]


class SegmentPlanStrategy(OutputStrategy):
    name = PREDICTION_SEGMENT_PLAN
    view = SegmentPlanOutput

    def bind_windows(
        self, windows: TrajectoryWindows, *, training_input: Any | None = None
    ) -> WindowContext:
        return SegmentPlanContext(windows)

    def build_model(self, config: TSConfig, normalizer: Normalizer | None) -> nn.Module:
        return SegmentPlanModel(config, ITransformerEncoderStack(config, encoder_token_length(config)), normalizer)

    def validation_extras(
        self, model: nn.Module, val_sets: dict[str, TrajectoryWindows], device: torch.device, config: TSConfig,
    ) -> dict[str, Any]:
        # The head's own reading of the val split at the fixed anchor (`readout`): the plan
        # judged inside its span, the per-segment errors, the arrival confusion — beside the
        # objective, never selected on.
        blocks: dict[str, dict[str, Any]] = {}
        for airport, windows in val_sets.items():
            series = [windows.series[int(s_idx)] for s_idx, _anchor in windows.index]
            anchors = sorted({int(a) for _s, a in windows.index})
            if len(anchors) != 1:
                raise ValueError("the segment-plan validation readout reads ONE common anchor per split")
            plans = decode_series(model, series, config, windows.normalizer, anchors[0], device)
            span_s = float(config.segment_plan_segments) * PLAN_WAYPOINT_SEGMENT_S
            blocks[airport] = summarize_readings(
                [plan_reading(plan, item, anchors[0], span_s) for plan, item in zip(plans, series, strict=True)], config,
            )
        return {"segment_plan_validation": pool_airports(blocks)}

    def target_contract(self, config: TSConfig) -> str:
        return SEGMENT_TARGET_CONTRACT

    def loss_component_names(self, config: TSConfig) -> tuple[str, ...]:
        return SEGMENT_PLAN_LOSS_COMPONENT_NAMES

    def loss(
        self,
        prediction: Any,
        normalized_anchor_state: torch.Tensor,
        target_states: torch.Tensor,
        state_weights: torch.Tensor,
        target_final_time_s: torch.Tensor,
        flight_weights: torch.Tensor,
        config: TSConfig,
        normalizer: Normalizer,
        context: dict[str, torch.Tensor] | None,
        dense_supervision: FixedDTControlSupervision | None,
        *,
        multipliers: ProcedureMultipliers | None = None,
    ) -> LossComponents:
        del normalized_anchor_state, target_states, state_weights, target_final_time_s, normalizer
        del dense_supervision, multipliers
        if not isinstance(prediction, SegmentPlanPrediction):
            raise TypeError("the segment-plan loss requires a SegmentPlanPrediction")
        return segment_plan_loss_components(prediction, flight_weights, config, context)

    def probe_context(
        self, batch_size: int, device: torch.device, config: TSConfig
    ) -> dict[str, torch.Tensor] | None:
        return probe_segment_plan_context(batch_size, device, config)

    def checkpoint_metadata(self, config: TSConfig) -> dict[str, Any]:
        return {"segment_plan": {
            "contract": SEGMENT_TARGET_CONTRACT,
            "segments": int(config.segment_plan_segments),
            "attention": config.segment_plan_attention,
            "features": list(SEGMENT_FEATURES),
        }}

    def forecast(
        self,
        model: nn.Module,
        series: Sequence[FlightSeries],
        config: TSConfig,
        normalizer: Normalizer,
        anchor: int,
        device: torch.device,
        options: ForecastOptions,
    ) -> list[Forecast]:
        if options.project_final is not None:
            raise ValueError("the final-approach projection applies to state forecasts only")
        if options.cta_s is not None or options.cta_offset_s:
            raise ValueError("the segment-plan path takes no CTA: the arrival is its own prediction")
        if options.conformal is not None:
            raise ValueError("the segment-plan path publishes no duration quantiles: nothing to calibrate")
        # `truncate` (the fixed-time state postprocessor) has nothing to act on; a plan that
        # arrives already ends at the threshold, so `truncate_at_threshold` finds it there.
        return forecast_segment_plan(model, series, config, normalizer, anchor, device)

    def replay(
        self,
        output: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        final_time_s: torch.Tensor,
        context: dict[str, torch.Tensor] | None,
        dataset: TrajectoryWindows,
    ) -> Replay:
        if not isinstance(output, SegmentPlanPrediction):
            raise TypeError("segment-plan replay requires a SegmentPlanPrediction")
        if context is None or CONTEXT_RUNWAY_HEADING not in context or CONTEXT_TARGET_CHART not in context:
            raise ValueError("segment-plan replay needs the per-flight runway course and threshold in the context")
        config = dataset.config
        channels = len(config.channels)
        segments = int(config.segment_plan_segments)
        anchors = dataset.normalizer.decode(anchor_state(x, channels).detach().cpu().numpy().astype(np.float64))
        headings = context[CONTEXT_RUNWAY_HEADING].detach().cpu().numpy().astype(np.float64)
        predicted = np.zeros((len(anchors), segments, channels), dtype=np.float32)
        durations = np.zeros((len(anchors), segments), dtype=np.float64)
        times = np.zeros(len(anchors), dtype=np.float32)
        row_valid = np.zeros((len(anchors), segments), dtype=bool)
        thresholds = context[CONTEXT_TARGET_CHART].detach().cpu().numpy().astype(np.float64)
        for i, plan in enumerate(decode_batch(output, anchors, headings)):
            durations[i], positions, times[i], rows = replay_rows(plan, thresholds[i], segments)
            predicted[i] = _states(anchors[i], positions, durations[i], channels)
            row_valid[i, :rows] = True
        return Replay(
            predicted_physical=predicted, segment_durations_s=durations, predicted_time_s=times,
            metric_targets=y, metric_weights=mask, row_valid=row_valid,
        )

    def record_fields(self, forecast: Forecast) -> dict[str, Any]:
        return {
            ARRIVAL_SEGMENT_KEY: forecast.segment_plan_arrival_segment,
            ARRIVAL_PROBABILITY_KEY: forecast.segment_plan_arrival_probability,
        }


STRATEGY = SegmentPlanStrategy()


__all__ = [
    "ARRIVAL_PROBABILITY_KEY", "ARRIVAL_SEGMENT_KEY", "STRATEGY", "SegmentPlanContext", "SegmentPlanStrategy",
    "decode_batch", "decode_series", "forecast_from_plan", "forecast_segment_plan",
]
