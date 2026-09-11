"""The plan-and-guidance path (design v5): the network predicts the operating parameters
and the NEXT instruction at any anchor; the guidance layer flies them, one instruction at
a time, re-asking the network at every fix.

What this strategy answers the spine with:

- the batch context is the plan labels read at each drawn anchor (`PlanContext`) — the
  extractors run per sample at batch time (0.4 ms each on KRDU val), cached on a
  fixed-anchor set; the skeletons are read once per runway;
- the model is `PlanOutputModel` on the shared backbone; the loss `plan_loss_components`;
- the deployable forecast is the ROLLED flight in lockstep (`forecast.fly_lockstep`, v5.1):
  the head's order every 30 s from the aircraft's pose and window, the route in force
  tracked, the group stepped together, until the closing onto the final; cut at the
  threshold crossing (`fly_rolling_orders`, one leg per order, stays for `--rolling leg`);
- the validation replay is the DRAWN single-step flight (`forecast.draw_order`): the route
  through the predicted next fix at the predicted schedule on the normalized grid — the
  checkpoint-selection clock, milliseconds per flight.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.backbone.adapters import build_state_forecaster
from ts_transformer.config import PREDICTION_PLAN, PlanOutput, TSConfig
from ts_transformer.data.batch_contract import LossComponents, anchor_state
from ts_transformer.data.channels import IDX, states_from_channels
from ts_transformer.data.dataset import series_conditioning
from ts_transformer.data.target_conditioning import conditioned_history
from ts_transformer.inference.forecast import Forecast, cut_at_threshold_crossing
from ts_transformer.outputs.base import ForecastOptions, OutputStrategy, Replay, WindowContext
from ts_transformer.outputs.plan.extractors import extract_plan
from ts_transformer.outputs.plan.forecast import (
    KIND_ROLLED,
    LOCKSTEP_S,
    Anchor,
    FlightState,
    LegOrder,
    RolledFlight,
    draw_order,
    fly_lockstep,
    fly_rolling_orders,
    lockstep_states,
    order_to_leg,
    rolled_history,
)
from ts_transformer.outputs.plan.labels import (
    CONTEXT_SERIES,
    TARGETS,
    PlanOrder,
    order_from_prediction,
    targets_from_labels,
)
from ts_transformer.outputs.plan.model import (
    PLAN_LOSS_COMPONENT_NAMES,
    PlanOutputModel,
    PlanPrediction,
    plan_loss_components,
    prediction_rows,
    probe_plan_context,
)
from ts_transformer.outputs.plan.skeleton import RunwaySkeleton, runway_skeleton

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries, Normalizer, TrajectoryWindows
    from ts_transformer.data.fixed_dt_supervision import FixedDTControlSupervision
    from ts_transformer.training.objective import ProcedureMultipliers
    from ts_transformer.outputs.control.basis_fit import FittedTeacherTable

#: The target contract stamped into a checkpoint: the target vector's composition — its
#: names in order, digested, so a reordered or swapped target refuses a stale checkpoint.
PLAN_TARGET_CONTRACT = "plan-v1-" + hashlib.sha1(",".join(TARGETS).encode()).hexdigest()[:8]


class SkeletonCache:
    """One skeleton per (airport, runway), read on first use."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], RunwaySkeleton] = {}

    def for_series(self, series: FlightSeries) -> RunwaySkeleton:
        key = (series.airport, str(series.scenario.source.get("runway")))
        if key not in self._by_key:
            self._by_key[key] = runway_skeleton(series)
        return self._by_key[key]


class PlanContext(WindowContext):
    """The plan labels at each window's anchor, as the head's targets (`PlanTargets.context`).
    A fixed-anchor set (``cache_context_rows``) builds its rows once; a random-anchor set
    reads each row for the anchor it drew."""

    def __init__(self, windows: TrajectoryWindows):
        self.windows = windows
        self.skeletons = SkeletonCache()
        self._rows: list[dict[str, np.ndarray]] | None = None
        if windows.cache_context_rows:
            self._rows = [self._build_row(index) for index in range(len(windows.index))]
            with_fix = sum(1 for row in self._rows if float(row["plan_next_is_join"]) == 0.0)
            self.summary = (
                f"plan labels: {len(self._rows)} anchors, {with_fix} with a next fix, "
                f"{len(self._rows) - with_fix} with none ahead (the join next, or on the final)"
            )
        else:
            self.summary = "plan labels: read at each drawn anchor"

    def skeleton(self, series_index: int) -> RunwaySkeleton:
        return self.skeletons.for_series(self.windows.series[series_index])

    def row(self, i: int) -> dict[str, np.ndarray]:
        return self._rows[i] if self._rows is not None else self._build_row(i)

    def _build_row(self, i: int) -> dict[str, np.ndarray]:
        s_idx, anchor = self.windows.index[i]
        series = self.windows.series[s_idx]
        skeleton = self.skeleton(s_idx)
        labels = extract_plan(series, anchor, skeleton)
        return targets_from_labels(labels, series, anchor, skeleton).context(s_idx)


def rolled_prediction(
    model: nn.Module, series: FlightSeries, config: TSConfig, normalizer: Normalizer, anchor: int,
    device: torch.device, skeleton: RunwaySkeleton,
) -> RolledFlight:
    """One flight's rolled prediction as the legs it flew: the head's order at every
    anchor (read off the window of the observed track continued by the flown rows),
    one leg on the guidance per order, until the closing onto the final."""
    conditioning = series_conditioning(series, config, normalizer, anchor=anchor)
    model.eval()

    def order_at(current: Anchor, window: np.ndarray) -> PlanOrder:
        x = conditioned_history(normalizer.encode(window), conditioning).astype(np.float32)
        with torch.no_grad():
            prediction = model(torch.from_numpy(x[None]).to(device))
        values, probability = prediction_rows(prediction)
        return order_from_prediction(values[0], float(probability[0]), current.e, current.n, skeleton)

    return fly_rolling_orders(series, anchor, skeleton, config, order_at=order_at, device=device)


def rolled_predictions_lockstep(
    model: nn.Module, series: Sequence[FlightSeries], config: TSConfig, normalizer: Normalizer,
    anchors: Sequence[int], device: torch.device, skeletons: Sequence[RunwaySkeleton],
    *, step_s: float = LOCKSTEP_S,
) -> list[RolledFlight]:
    """A group's rolled predictions in lockstep (design v5.1): every `step_s` the head is
    asked again for every flight still flying — one forward pass on the windows of the
    observed tracks continued by the flown rows — and the group is stepped together."""
    model.eval()
    conditioning = {
        id(item): series_conditioning(item, config, normalizer, anchor=int(a)) for item, a in zip(series, anchors, strict=True)
    }
    # the first order gives the remaining path the first step is laid with
    def orders_for(states: Sequence[FlightState]) -> list[PlanOrder]:
        windows = np.stack([
            conditioned_history(normalizer.encode(rolled_history(s.series, s.anchor, s.chunks, config)), conditioning[id(s.series)])
            for s in states
        ]).astype(np.float32)
        with torch.no_grad():
            prediction = model(torch.from_numpy(windows).to(device))
        values, probability = prediction_rows(prediction)
        return [
            order_from_prediction(values[i], float(probability[i]), s.current.e, s.current.n, s.skeleton)
            for i, s in enumerate(states)
        ]

    states = lockstep_states(series, anchors, skeletons, [1.0] * len(series))
    first = orders_for(states)
    for state, order in zip(states, first, strict=True):
        state.current = replace(state.current, remaining_m=order.remaining_m)

    def policy(active: Sequence[FlightState]) -> list[LegOrder]:
        orders = first if all(s.steps == 0 for s in active) and len(active) == len(states) else orders_for(active)
        return [order_to_leg(order, s, s.steps) for order, s in zip(orders, active, strict=True)]

    return fly_lockstep(states, config, policy=policy, step_s=step_s, device=device)


def forecast_plan(
    model: nn.Module, series: FlightSeries, config: TSConfig, normalizer: Normalizer, anchor: int,
    device: torch.device, skeleton: RunwaySkeleton,
) -> Forecast:
    """One flight's rolled prediction as a forecast (lockstep, v5.1), cut at the threshold
    crossing; the record carries every order flown (`planOrders`) and the first order's
    arrival time as the prediction."""
    flight, = rolled_predictions_lockstep(model, [series], config, normalizer, [anchor], device, [skeleton])
    return _plan_forecast(flight, series)


def _plan_forecast(flight: RolledFlight, series: FlightSeries) -> Forecast:
    forecast = cut_at_threshold_crossing(flight.forecast, series)
    diagnostics = dict(forecast.command_hook_diagnostics or {})
    diagnostics["planLegs"] = float(len(flight.routes))
    diagnostics["planInstructionsFlown"] = float(flight.instructions_flown)
    diagnostics["planInstructionsSkipped"] = float(flight.instructions_skipped)
    diagnostics["planRouteTimeS"] = float(flight.route_time_s)
    diagnostics["planTurnsIncomplete"] = float(flight.turns_incomplete)
    diagnostics["planOrders"] = flight.orders
    return replace(forecast, command_hook_diagnostics=diagnostics, prediction_output=PREDICTION_PLAN)


class PlanStrategy(OutputStrategy):
    name = PREDICTION_PLAN
    view = PlanOutput

    def bind_windows(
        self, windows: TrajectoryWindows, *, fitted_teacher: FittedTeacherTable | None = None
    ) -> WindowContext:
        return PlanContext(windows)

    def build_model(self, config: TSConfig, normalizer: Normalizer | None) -> nn.Module:
        del normalizer  # the targets are regressed in physical units
        return PlanOutputModel(config, build_state_forecaster(config))

    def target_contract(self, config: TSConfig) -> str:
        return PLAN_TARGET_CONTRACT

    def loss_component_names(self, config: TSConfig) -> tuple[str, ...]:
        return PLAN_LOSS_COMPONENT_NAMES

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
        if not isinstance(prediction, PlanPrediction):
            raise TypeError("the plan loss requires a PlanPrediction")
        return plan_loss_components(prediction, flight_weights, config, context)

    def probe_context(
        self, batch_size: int, device: torch.device, config: TSConfig
    ) -> dict[str, torch.Tensor] | None:
        return probe_plan_context(batch_size, device, config)

    def checkpoint_metadata(self, config: TSConfig) -> dict[str, Any]:
        return {"plan": {"targets": list(TARGETS), "contract": PLAN_TARGET_CONTRACT}}

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
            raise ValueError("the plan path takes no CTA yet (design v5 §9 step 4)")
        if options.conformal is not None:
            raise ValueError("the plan path publishes no duration quantiles yet: nothing to calibrate")
        # `truncate` (the fixed-time state postprocessor) has nothing to act on here, and
        # `truncate_at_threshold` is what this path ALWAYS does (`forecast_plan` cuts every
        # rolled flight at its threshold crossing): both values of both flags are accepted
        skeletons = SkeletonCache()
        flights = rolled_predictions_lockstep(
            model, list(series), config, normalizer, [anchor] * len(series), device, [skeletons.for_series(item) for item in series],
        )
        return [_plan_forecast(flight, item) for flight, item in zip(flights, series, strict=True)]

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
        if not isinstance(output, PlanPrediction):
            raise TypeError("plan replay requires a PlanPrediction")
        if context is None or CONTEXT_SERIES not in context:
            raise ValueError("plan replay needs the per-anchor context (the flight each row is)")
        plan_context = dataset.context
        if not isinstance(plan_context, PlanContext):
            raise TypeError("plan replay needs a window set bound by the plan strategy")
        channels = len(dataset.config.channels)
        anchors = dataset.normalizer.decode(anchor_state(x, channels).detach().cpu().numpy().astype(np.float64))
        values, probabilities = prediction_rows(output)
        nodes = dataset.config.pred_len
        drawn = np.zeros((len(values), nodes, channels), dtype=np.float32)
        durations = np.zeros((len(values), nodes), dtype=np.float32)
        times = np.zeros(len(values), dtype=np.float32)
        for i, (row, probability, series_index) in enumerate(
            zip(values, probabilities, context[CONTEXT_SERIES].detach().cpu().numpy(), strict=True)
        ):
            series = dataset.series[int(series_index)]
            skeleton = plan_context.skeleton(int(series_index))
            state = states_from_channels(
                np.array([0.0]), anchors[i:i + 1], series.frame, mass_kg=float(series.scenario.initial.m),
            )[0][1]
            order = order_from_prediction(row, float(probability), float(anchors[i, IDX["e"]]), float(anchors[i, IDX["n"]]), skeleton)
            channels_drawn, node_durations, T = draw_order(
                order, float(anchors[i, IDX["e"]]), float(anchors[i, IDX["n"]]), float(anchors[i, IDX["u"]]),
                float(state.psi), float(state.V * np.cos(state.gamma)), skeleton, nodes,
            )
            drawn[i], durations[i], times[i] = channels_drawn, node_durations, T
        return Replay(
            predicted_physical=drawn, segment_durations_s=durations, predicted_time_s=times,
            metric_targets=y, metric_weights=mask,
        )

    def record_fields(self, forecast: Forecast) -> dict[str, Any]:
        diagnostics = forecast.command_hook_diagnostics or {}
        # the orders flown ride in `commandHookDiagnostics.planOrders`; here the counts
        return {
            "planRouteKind": KIND_ROLLED,
            "planLegs": diagnostics.get("planLegs"),
            "planInstructionsFlown": diagnostics.get("planInstructionsFlown"),
            "planCappedBy": diagnostics.get("planCappedBy") or None,
        }


STRATEGY = PlanStrategy()


__all__ = ["PLAN_TARGET_CONTRACT", "PlanContext", "PlanStrategy", "STRATEGY", "forecast_plan", "rolled_prediction", "rolled_predictions_lockstep"]
