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
  checkpoint-selection clock, milliseconds per flight;
- the training-time input is the ROLLED-WINDOW TABLE (design v5.2, `outputs.plan.rolled`):
  with `plan_rolled_windows_path` set, every window set's flights must be in it, a share
  of each epoch's per-flight draws is one of the flight's rolled windows
  (`PlanContext.override`), and the val split's rolled windows are scored every epoch as a
  readout beside the observed objective (`validation_extras`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Sequence

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
    closing_budget_s,
    KIND_ROLLED,
    LOCKSTEP_S,
    Anchor,
    FlightState,
    LegOrder,
    RolledFlight,
    draw_order,
    ORDER_HOLD_ASKS,
    fly_lockstep,
    fly_rolling_orders,
    lockstep_states,
    order_to_leg,
    rolled_history,
)
from ts_transformer.outputs.plan.labels import (
    CONTEXT_NEXT_IS_JOIN,
    CONTEXT_ROLLED,
    CONTEXT_SERIES,
    CONTEXT_TARGETS,
    CONTEXT_VALID,
    PLAN_TARGET_CONTRACT,
    TARGETS,
    PlanOrder,
    order_from_prediction,
    targets_from_labels,
    T_MIN_S,
)
from ts_transformer.outputs.plan.model import (
    PLAN_LOSS_COMPONENT_NAMES,
    PlanOutputModel,
    PlanPrediction,
    loss_group_carriers,
    fan_rows,
    plan_loss_components,
    prediction_rows,
    probe_plan_context,
)
from ts_transformer.outputs.plan.rolled import RolledDraw, RolledWindowTable, load_rolled_windows
from ts_transformer.outputs.plan.skeleton import RunwaySkeleton, SkeletonCache

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries, Normalizer, TrajectoryWindows
    from ts_transformer.data.fixed_dt_supervision import FixedDTControlSupervision
    from ts_transformer.training.objective import ProcedureMultipliers


class PlanContext(WindowContext):
    """The plan labels at each window's anchor, as the head's targets (`PlanTargets.context`).
    A fixed-anchor set (``cache_context_rows``) builds its rows once; a random-anchor set
    reads each row for the anchor it drew. With a rolled-window table (design v5.2) every
    flight of the set must be in it, and a training draw is replaced by one of the
    flight's rolled windows at `plan_rolled_share` (`override`)."""

    def __init__(self, windows: TrajectoryWindows, rolled: RolledWindowTable | None = None):
        self.windows = windows
        self.skeletons = SkeletonCache()
        self.rolled = rolled
        self.draw: RolledDraw | None = None
        self.encoded: np.ndarray | None = None    # the table's windows under this set's normalizer, once
        if rolled is not None:
            series = [windows.series[int(index)] for index in windows.eligible_series]
            rolled.require_cover(series, windows.config, what=f"this window set ({len(series)} flights)")
            self.draw = RolledDraw(rolled, windows.config.plan_rolled_share)
            self.encoded = windows.normalizer.encode(np.asarray(rolled.windows, dtype=np.float64)).astype(np.float32)
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
        if rolled is not None:
            self.summary += (
                f"; rolled windows {rolled.path.name}: {rolled.samples} samples over {rolled.flights} flights "
                f"({rolled.header['policy']} policy, every {rolled.header['lockstep_s']:g} s), drawn at share "
                f"{windows.config.plan_rolled_share:g}"
            )

    def skeleton(self, series_index: int) -> RunwaySkeleton:
        return self.skeletons.for_series(self.windows.series[series_index])

    def row(self, i: int) -> dict[str, np.ndarray]:
        return self._rows[i] if self._rows is not None else self._build_row(i)

    def override(self, i: int, epoch_seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
        if self.draw is None:
            return None
        s_idx = int(self.windows.index[i][0])
        k = self.draw.sample(self.windows.series[s_idx].dataset_id, epoch_seed)
        if k is None:
            return None
        return self.encoded[k], self.rolled.context_row(k, s_idx)

    def _build_row(self, i: int) -> dict[str, np.ndarray]:
        s_idx, anchor = self.windows.index[i]
        series = self.windows.series[s_idx]
        skeleton = self.skeleton(s_idx)
        labels = extract_plan(series, anchor, skeleton)
        return targets_from_labels(labels, series, anchor, skeleton).context(s_idx)


class PlanRolledShare:
    """The epoch's realised rolled share — the training accumulator
    (`OutputStrategy.training_diagnostics`): how many of the epoch's samples were rolled
    windows, read off every batch's `plan_rolled` column."""

    def __init__(self) -> None:
        self.samples = 0
        self.rolled = 0.0

    def record_prediction(self, prediction: Any, context: dict[str, torch.Tensor] | None) -> None:
        del prediction
        if context is None or CONTEXT_ROLLED not in context:
            raise ValueError("a plan batch carries the plan_rolled column")
        self.samples += int(context[CONTEXT_ROLLED].numel())
        self.rolled += float(context[CONTEXT_ROLLED].sum())

    def record_gradients_and_clip(self, model: nn.Module) -> None:
        del model  # the plan head clips nothing

    def summary(self) -> dict[str, float | int]:
        return {"samples": self.samples, "rolled": int(self.rolled), "share": self.rolled / max(self.samples, 1)}


def rolled_validation_readout(
    model: nn.Module, context: PlanContext, device: torch.device, config: TSConfig,
) -> dict[str, Any]:
    """The plan loss over one window set's rolled windows (every sample of every flight of
    the set, equal weight per sample): a READOUT of how the head reads its own flown
    windows, beside the observed objective the checkpoint is selected on."""
    windows, table = context.windows, context.rolled
    if table is None:
        raise ValueError("the window set carries no rolled-window table")
    rows, owners = [], []
    for s_idx in windows.eligible_series:
        flight_rows = table.samples_for(windows.series[int(s_idx)].dataset_id)
        rows.append(flight_rows)
        owners.append(np.full(len(flight_rows), int(s_idx), dtype=np.int64))
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    owners = np.concatenate(owners) if owners else np.zeros(0, dtype=np.int64)
    # each component is a mean over the samples that CARRY it (`plan_loss_components`), so
    # the batches are averaged by their carriers, never by their size
    numerator: dict[str, float] = {}
    carriers_total: dict[str, int] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), config.batch_size):
            batch, owner = rows[start:start + config.batch_size], owners[start:start + config.batch_size]
            x = np.stack([
                conditioned_history(context.encoded[k], windows.conditioning[int(s_idx)])
                for k, s_idx in zip(batch, owner, strict=True)
            ]).astype(np.float32)
            prediction = model(torch.from_numpy(x).to(device))
            batch_context = {
                CONTEXT_TARGETS: torch.from_numpy(table.targets[batch]).to(device),
                CONTEXT_VALID: torch.from_numpy(table.valid[batch]).to(device),
                CONTEXT_NEXT_IS_JOIN: torch.from_numpy(table.next_is_join[batch]).to(device),
                CONTEXT_SERIES: torch.from_numpy(owner).to(device),
                CONTEXT_ROLLED: torch.ones(len(batch), dtype=torch.float32, device=device),
            }
            components = plan_loss_components(
                prediction, torch.ones(len(batch), dtype=torch.float32, device=device), config, batch_context,
            )
            carriers = loss_group_carriers(table.valid[batch])
            for name, value in components.tensors().items():
                numerator[name] = numerator.get(name, 0.0) + float(value.detach()) * carriers[name]
                carriers_total[name] = carriers_total.get(name, 0) + carriers[name]
    per_component = {name: value / max(carriers_total[name], 1) for name, value in numerator.items()}
    return {
        "samples": int(len(rows)), "flights": int(len(windows.eligible_series)),
        "loss": float(sum(per_component.values())), "components": per_component,
        "carriers": carriers_total,
        "on_final_share": float(np.mean(table.on_final[rows])) if len(rows) else 0.0,
        "next_is_join_share": float(np.mean(table.next_is_join[rows])) if len(rows) else 0.0,
    }


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


@dataclass(frozen=True)
class Assignment:
    """What the scheduler assigns one flight (design §5; v5.4, §9 step 4): an arrival time,
    ABSOLUTE on the series clock, and/or a join distance-to-go; None leaves the head's own."""

    arrival_time_s: float | None = None
    d_join_m: float | None = None


NO_ASSIGNMENT = Assignment()


def assigned_order(order: PlanOrder, assignment: Assignment, state: FlightState) -> PlanOrder:
    """The head's order under the assignment: its arrival time replaced by the assigned
    REMAINING time from the flight's current clock (never under `T_MIN_S`), its join by the
    assigned one (never past the remaining path); every other parameter the head's."""
    changes: dict = {}
    clamped = list(order.clamped)
    if assignment.arrival_time_s is not None:
        remaining_s = float(assignment.arrival_time_s) - state.current.time_s
        changes["T_s"] = max(remaining_s, T_MIN_S)
        if remaining_s < T_MIN_S:
            clamped.append("assigned_T_s")
    if assignment.d_join_m is not None:
        d_join = min(max(float(assignment.d_join_m), 1.0), order.remaining_m)
        changes["d_join_m"] = d_join
        if d_join != float(assignment.d_join_m):
            clamped.append("assigned_d_join_m")
    return replace(order, clamped=tuple(clamped), **changes) if changes else order


def lockstep_model_policy(
    model: nn.Module, series: Sequence[FlightSeries], config: TSConfig, normalizer: Normalizer,
    anchors: Sequence[int], device: torch.device, skeletons: Sequence[RunwaySkeleton],
    *, first_component: int | None = None,
    first_order: Callable[[PlanOrder, FlightState], PlanOrder] | None = None,
    assignments: Sequence[Assignment] | None = None,
) -> tuple[list[FlightState], Callable[[Sequence[FlightState]], list[LegOrder]]]:
    """The head as a lockstep policy (design v5.1): the group's flight states, and the
    policy that asks the head for every active flight at once — one forward pass on the
    windows of the observed tracks continued by the flown rows — and returns each order
    as the leg order the step flies. The first order is read here, so the states start
    with the head's own remaining path. **A fan member (v5.3, §9 step 3(g)) is the fan ONE
    STEP DEEP**: ``first_component`` reads the FIRST order from that mixture component
    (`fan_rows`) and every later ask from the top-weight one as usual; ``first_order``
    transforms each flight's first order given its state (the control fan's displaced
    fix, laid in the flight's runway axes). ``assignments`` (v5.4, one per flight) replace
    the head's arrival time and/or join at EVERY ask (`assigned_order`) and hand the
    lockstep the assigned arrival to close the time on."""
    model.eval()
    conditioning = {
        id(item): series_conditioning(item, config, normalizer, anchor=int(a)) for item, a in zip(series, anchors, strict=True)
    }
    # the first order gives the remaining path the first step is laid with
    def orders_for(states: Sequence[FlightState], component: int | None = None) -> list[PlanOrder]:
        windows = np.stack([
            conditioned_history(normalizer.encode(rolled_history(s.series, s.anchor, s.chunks, config)), conditioning[id(s.series)])
            for s in states
        ]).astype(np.float32)
        with torch.no_grad():
            prediction = model(torch.from_numpy(windows).to(device))
        values, probability = prediction_rows(prediction)
        if component is not None:
            values = fan_rows(prediction)[0][:, component]
        return [
            order_from_prediction(values[i], float(probability[i]), s.current.e, s.current.n, s.skeleton)
            for i, s in enumerate(states)
        ]

    states = lockstep_states(series, anchors, skeletons, [1.0] * len(series))
    assigned = {
        id(state): assignment
        for state, assignment in zip(states, [NO_ASSIGNMENT] * len(states) if assignments is None else assignments, strict=True)
    }
    first = orders_for(states, component=first_component)
    if first_order is not None:
        first = [first_order(order, state) for order, state in zip(first, states, strict=True)]
    for state, order in zip(states, first, strict=True):
        state.current = replace(state.current, remaining_m=order.remaining_m)

    def policy(active: Sequence[FlightState]) -> list[LegOrder]:
        orders = first if all(s.steps == 0 for s in active) and len(active) == len(states) else orders_for(active)
        legs = []
        for order, s in zip(orders, active, strict=True):
            assignment = assigned[id(s)]
            under = assigned_order(order, assignment, s)
            leg = order_to_leg(under, s, s.steps)
            if assignment.arrival_time_s is not None:
                # the assigned time is a target the closure aims at, never the flight's
                # guillotine: the budget covers the later of the head's own time and the
                # assigned one, so a flight the closure cannot bring forward lands LATE and
                # reports it as dt (with X < 0), instead of being cut short of the final
                # (measured on the 48-flight smoke: vectored established 0.96 → 0.58 with
                # the assigned time as the budget)
                # the head's OWN arrival time stays the prediction the rolled flight reports
                # (`eta_predicted_s`); the assigned one rides in `assigned_arrival_s`
                leg = replace(
                    leg, assigned_arrival_s=assignment.arrival_time_s, arrival_time_s=float(order.T_s),
                    budget_s=closing_budget_s(max(float(under.T_s), float(order.T_s))),
                )
            legs.append(leg)
        return legs

    return states, policy


def rolled_predictions_lockstep(
    model: nn.Module, series: Sequence[FlightSeries], config: TSConfig, normalizer: Normalizer,
    anchors: Sequence[int], device: torch.device, skeletons: Sequence[RunwaySkeleton],
    *, step_s: float = LOCKSTEP_S, hold_asks: int = ORDER_HOLD_ASKS, hold_flips_only: bool = False,
    first_component: int | None = None,
    first_order: Callable[[PlanOrder, FlightState], PlanOrder] | None = None,
    assignments: Sequence[Assignment] | None = None,
) -> list[RolledFlight]:
    """A group's rolled predictions in lockstep (design v5.1): every `step_s` the head is
    asked again for every flight still flying and the group is stepped together; a
    material change of order (or a fix ↔ none flip only) is adopted only once given on
    `hold_asks` consecutive asks (v5.3, `forecast.held_order`); a fan member's first order
    from one mixture component, or transformed (`lockstep_model_policy`) — ONE step deep,
    which an order hold would silently deepen to `hold_asks` steps, so the two are refused
    together."""
    if (first_component is not None or first_order is not None) and hold_asks != 1:
        raise ValueError("a fan member is the fan one step deep; an order hold would hold its first order longer")
    states, policy = lockstep_model_policy(
        model, series, config, normalizer, anchors, device, skeletons,
        first_component=first_component, first_order=first_order, assignments=assignments,
    )
    return fly_lockstep(
        states, config, policy=policy, step_s=step_s, hold_asks=hold_asks, hold_flips_only=hold_flips_only, device=device,
    )


def forecast_plan(
    model: nn.Module, series: FlightSeries, config: TSConfig, normalizer: Normalizer, anchor: int,
    device: torch.device, skeleton: RunwaySkeleton,
) -> Forecast:
    """One flight's rolled prediction as a forecast (lockstep, v5.1), cut at the threshold
    crossing; the record carries every order flown (`planOrders`) and the first order's
    arrival time as the prediction."""
    flight, = rolled_predictions_lockstep(model, [series], config, normalizer, [anchor], device, [skeleton])
    return rolled_flight_forecast(flight, series)


def rolled_flight_forecast(flight: RolledFlight, series: FlightSeries) -> Forecast:
    """A rolled flight as the plan path's forecast: cut at the threshold crossing, stamped `plan`, with the
    plan diagnostics every published plan record carries (`planOrders`, the counts)."""
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
        self, windows: TrajectoryWindows, *, training_input: RolledWindowTable | None = None
    ) -> WindowContext:
        return PlanContext(windows, training_input)

    def training_input(self, config: TSConfig) -> RolledWindowTable | None:
        # The rolled-window table is a TRAINING input, opened exactly here (design v5.2);
        # every replay path builds its window sets without one.
        if config.plan_rolled_windows_path:
            return load_rolled_windows(config.plan_rolled_windows_path)
        return None

    def training_diagnostics(self, config: TSConfig) -> PlanRolledShare | None:
        return PlanRolledShare() if config.plan_rolled_share else None

    def epoch_record(
        self, config: TSConfig, epoch: int, diagnostic_totals: dict[str, float], diagnostics: Any | None,
    ) -> dict[str, Any]:
        del config, epoch, diagnostic_totals
        return {} if diagnostics is None else {"plan_rolled_training": diagnostics.summary()}

    def validation_extras(
        self, model: nn.Module, val_sets: dict[str, TrajectoryWindows], device: torch.device, config: TSConfig,
    ) -> dict[str, Any]:
        contexts = {
            airport: dataset.context for airport, dataset in val_sets.items()
            if isinstance(dataset.context, PlanContext) and dataset.context.rolled is not None
        }
        if not contexts:
            return {}
        by_airport = {airport: rolled_validation_readout(model, context, device, config) for airport, context in contexts.items()}
        names = list(next(iter(by_airport.values()))["components"])
        return {"plan_rolled_validation": {
            # equal airport weight, as the observed objective is read
            "loss": float(np.mean([block["loss"] for block in by_airport.values()])),
            "components": {name: float(np.mean([block["components"][name] for block in by_airport.values()])) for name in names},
            "samples": int(sum(block["samples"] for block in by_airport.values())),
            "flights": int(sum(block["flights"] for block in by_airport.values())),
            "by_airport": by_airport,
        }}

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
        return [rolled_flight_forecast(flight, item) for flight, item in zip(flights, series, strict=True)]

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


__all__ = [
    "PLAN_TARGET_CONTRACT", "PlanContext", "PlanRolledShare", "PlanStrategy", "STRATEGY", "forecast_plan",
    "lockstep_model_policy", "rolled_prediction", "rolled_predictions_lockstep", "rolled_validation_readout",
]
