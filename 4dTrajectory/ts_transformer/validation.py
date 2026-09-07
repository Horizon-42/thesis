"""How a fitted model is scored on a split, and which epoch is kept.

Three layers, all deterministic and all sharing ONE forward pass per split:

1. the replay (:class:`SplitPredictionReplay`, :func:`predict_split`) — a fixed ``L-1``
   anchor, ``model.eval()``, no gradients, sequential batches, so train and validation
   numbers are directly comparable and no metric view re-runs the model;
2. the per-epoch validation pass — the batch plan, the per-airport objective and the
   duration-bucket breakdown;
3. checkpoint selection — ``VALIDATION_SELECTIONS`` maps
   ``config.checkpoint_selection_metric`` to the function that turns that pass into one
   comparable number.

`train.fit_model` drives all three. Keeping them out of the loop is what lets
`evaluate-fit`, the cross-validation folds and the standalone replay runners score a
checkpoint without instantiating a training run.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from anchor_grid import (
    DEFAULT_GRID_MIN_FUTURE_S,
    VALIDATION_ANCHOR_GRID_KM,
    VALIDATION_ANCHOR_GRID_M,
    anchors_for_bin,
    remaining_path_profiles,
)
from batch_contract import anchor_state, model_forward, unpack_batch
from closure_output import ClosurePrediction, replay_batch as closure_replay_batch
from config import (
    CHECKPOINT_SELECTION_ANCHOR_GRID_ADE,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CHECKPOINT_SELECTION_COMMON_GRID_METRICS,
    CHECKPOINT_SELECTION_OBJECTIVE,
    PREDICTION_STATE,
    TSConfig,
    uses_control_dynamics,
)
from control.basis_fit import FittedTeacherTable
from control.constraints import build_command_hook
from control.dynamics import rollout as control_rollout
from dataset import (
    ExplicitAnchorTrajectoryWindows,
    FixedAnchorTrajectoryWindows,
    FlightSeries,
    Normalizer,
    TrajectoryWindows,
    iter_batches,
)
from fixed_anchor_validation import (
    CommonGridTruth,
    common_truth_at_anchors,
    fixed_anchor_common_truth,
    fixed_anchor_common_grid_ade_metrics,
    fixed_anchor_common_grid_metrics,
    fixed_anchor_common_grid_report_metrics,
)
from metrics import raw_kinematic_metrics, states_with_derived_velocity
from objective import (
    ProcedureMultipliers,
    align_control_targets_to_query_clock,
    loss_component_names,
    move_dynamics,
    move_fixed_dt_supervision,
    prediction_loss_components,
)
from prediction_outputs import ControlPrediction, StatePrediction
from time_grids import numpy_inference_time_grid
from training_performance import EpochProfiler


@dataclass(frozen=True)
class SplitPredictionReplay:
    """One immutable deployable prediction pass reused by every metric view."""

    predicted: np.ndarray
    truth: np.ndarray
    mask: np.ndarray
    predicted_time_s: np.ndarray
    truth_time_s: np.ndarray
    anchors: np.ndarray
    segment_durations_s: np.ndarray


def _prediction_batch_replay(
    output: StatePrediction | ControlPrediction,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    final_time_s: torch.Tensor,
    dynamics: dict[str, torch.Tensor] | None,
    dataset: TrajectoryWindows,
) -> SplitPredictionReplay:
    """Materialize deployable physical arrays from an already-computed model output."""
    metric_targets = y
    metric_weights = mask
    if uses_control_dynamics(dataset.config.prediction_output):
        deployable = output
        if dynamics is None:
            raise ValueError("control replay requires per-flight dynamics")
        points = dataset.config.validation_common_grid_points
        progress = torch.arange(
            1,
            points + 1,
            dtype=torch.float64,
            device=deployable.segment_durations.device,
        ) / points
        predicted_total_s = deployable.segment_durations.to(torch.float64).sum(dim=1)
        query_offsets_s = predicted_total_s.unsqueeze(1) * progress.unsqueeze(0)
        query_valid = torch.ones_like(query_offsets_s, dtype=torch.bool)
        rollout = control_rollout.rollout_control_dense(
            deployable.controls,
            deployable.segment_durations,
            dynamics,
            query_offsets_s,
            query_valid,
            dataset.config,
            command_hook=build_command_hook(dataset.config, dynamics),
        )
        predicted_physical = (
            rollout.query_channels.detach().cpu().numpy().astype(np.float32)
        )
        metric_targets, metric_weights = align_control_targets_to_query_clock(
            anchor_state(x, len(dataset.config.channels)),
            y,
            mask,
            query_offsets_s,
            final_time_s,
        )
        segment_durations_s = np.broadcast_to(
            (
                predicted_total_s.detach().cpu().numpy().astype(np.float64)
                / points
            )[:, None],
            (len(x), points),
        ).copy()
        predicted_time_s = deployable.final_time_s.detach().cpu().numpy()
    elif isinstance(output, ClosurePrediction):
        # Drawn, not rolled out: every decision reconstructed in numpy and sampled on the
        # target grid's fractions of its own duration (the context carries the course).
        if dynamics is None:
            raise ValueError("closure replay requires the per-flight label context")
        anchors_physical = dataset.normalizer.decode(
            anchor_state(x, len(dataset.config.channels))
            .detach().cpu().numpy().astype(np.float64)
        )
        predicted_physical, segment_durations_s, predicted_time_s = closure_replay_batch(
            output, anchors_physical, dynamics, dataset.config, dataset.config.pred_len
        )
    else:
        if not isinstance(output, StatePrediction):
            raise TypeError("state replay requires StatePrediction")
        out = output.states.detach().cpu().numpy()
        predicted_physical = dataset.normalizer.decode(
            out.astype(np.float64)
        ).astype(np.float32)
        segment_durations_s = numpy_inference_time_grid(
            output.final_time_s.detach().cpu().numpy(), dataset.config
        )[0]
        predicted_time_s = output.final_time_s.detach().cpu().numpy()

    # Decode in float64 (the normalizer stats' dtype), store float32: a pooled split is
    # tens of thousands of [N,C] windows and metre-scale metrics do not need float64 storage.
    truth = dataset.normalizer.decode(
        metric_targets.detach().cpu().numpy().astype(np.float64)
    ).astype(np.float32)
    anchors = dataset.normalizer.decode(
        anchor_state(x, len(dataset.config.channels))
        .detach().cpu().numpy().astype(np.float64)
    ).astype(np.float32)
    if dataset.config.prediction_output == PREDICTION_STATE:
        # The state output predicts positions + duration only; the control rollout and
        # the closure reconstruction both carry exact velocities.
        predicted_physical = states_with_derived_velocity(
            anchors,
            predicted_physical,
            segment_durations_s,
        ).astype(np.float32)
    raw_mask = metric_weights.detach().cpu().numpy()
    if raw_mask.ndim == 3:
        raw_mask = np.all(raw_mask > 0.0, axis=-1).astype(np.float32)
    return SplitPredictionReplay(
        predicted=predicted_physical,
        truth=truth,
        mask=raw_mask,
        predicted_time_s=predicted_time_s,
        truth_time_s=final_time_s.detach().cpu().numpy(),
        anchors=anchors,
        segment_durations_s=segment_durations_s,
    )


def _merge_prediction_replays(
    chunks: Sequence[tuple[np.ndarray, SplitPredictionReplay]],
    *,
    count: int,
) -> SplitPredictionReplay:
    """Restore dataset order after validation-only duration bucketing."""
    if not chunks:
        raise ValueError("prediction replay requires at least one batch")
    indices = np.concatenate([item[0] for item in chunks])
    order = np.argsort(indices, kind="stable")
    if not np.array_equal(indices[order], np.arange(count, dtype=np.int64)):
        raise ValueError("prediction replay indices must cover the dataset exactly once")

    def merged(name: str) -> np.ndarray:
        return np.concatenate(
            [getattr(item[1], name) for item in chunks], axis=0
        )[order]

    return SplitPredictionReplay(
        predicted=merged("predicted"),
        truth=merged("truth"),
        mask=merged("mask"),
        predicted_time_s=merged("predicted_time_s"),
        truth_time_s=merged("truth_time_s"),
        anchors=merged("anchors"),
        segment_durations_s=merged("segment_durations_s"),
    )


def predict_split(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    device: torch.device,
    batch_size: int,
) -> SplitPredictionReplay:
    """Return physical anchor/output arrays, masks, predicted time and true time."""
    model.eval()
    chunks: list[tuple[np.ndarray, SplitPredictionReplay]] = []
    cursor = 0
    with torch.no_grad():
        for raw_batch in iter_batches(dataset, batch_size, shuffle=False, seed=0):
            (
                x,
                y,
                mask,
                final_time_s,
                _flight_weights,
                dynamics,
                _dense_supervision,
            ) = unpack_batch(raw_batch)
            x_device = x.to(device)
            y_device = y.to(device)
            mask_device = mask.to(device)
            final_time_device = final_time_s.to(device)
            dynamics_device = move_dynamics(dynamics, device)
            output = model_forward(model, x_device, dynamics_device)
            batch_replay = _prediction_batch_replay(
                output,
                x_device,
                y_device,
                mask_device,
                final_time_device,
                dynamics_device,
                dataset,
            )
            batch_count = len(x)
            chunks.append(
                (np.arange(cursor, cursor + batch_count, dtype=np.int64), batch_replay)
            )
            cursor += batch_count
    return _merge_prediction_replays(
        chunks,
        count=len(dataset),
    )


def evaluate_split(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    *,
    replay: SplitPredictionReplay | None = None,
) -> dict[str, Any]:
    """Formal fixed-anchor metrics on the common true-physical-time grid."""
    replay = replay or predict_split(
        model, dataset, normalizer, device, config.batch_size
    )
    block = fixed_anchor_common_grid_report_metrics(
        dataset.series,
        config,
        replay.anchors,
        replay.predicted,
        replay.predicted_time_s,
        replay.segment_durations_s,
        points=config.validation_common_grid_points,
    )
    indices_by_airport: dict[str, list[int]] = {}
    for index, item in enumerate(dataset.series):
        indices_by_airport.setdefault(item.airport or "<unknown>", []).append(index)
    by_airport: dict[str, dict[str, Any]] = {}
    for airport, indices in sorted(indices_by_airport.items()):
        selected = np.asarray(indices, dtype=np.int64)
        airport_block = fixed_anchor_common_grid_report_metrics(
            [dataset.series[index] for index in indices],
            config,
            replay.anchors[selected],
            replay.predicted[selected],
            replay.predicted_time_s[selected],
            replay.segment_durations_s[selected],
            points=config.validation_common_grid_points,
        )
        by_airport[airport] = {
            key: airport_block[key]
            for key in (
                "flights",
                "ade_m",
                "fde_m",
                "arrival_endpoint_error_m",
                "horizontal_m",
                "along_track_m",
                "cross_track_m",
                "vertical_m",
                "final_time_s",
                "prediction_horizon_cap_rate",
                "invalid_flights",
            )
        }
    block["flight_micro_ade_m"] = block["ade_m"]
    block["flight_micro_fde_m"] = block["fde_m"]
    block["per_airport"] = by_airport
    block["airport_macro"] = {
        "ade_m": float(np.mean([item["ade_m"] for item in by_airport.values()])),
        "fde_m": float(np.mean([item["fde_m"] for item in by_airport.values()])),
        "arrival_endpoint_error_m": float(np.mean([
            item["arrival_endpoint_error_m"]["mean"]
            for item in by_airport.values()
        ])),
        "final_time_mae_s": float(np.mean([
            item["final_time_s"]["mae"] for item in by_airport.values()
        ])),
    }
    # Compatibility scalar names now point at the single formal airport-macro score.
    block["ade_m"] = block["airport_macro"]["ade_m"]
    block["fde_m"] = block["airport_macro"]["fde_m"]
    # Raw model nodes on their own predicted clock: no measured-track interpolation,
    # spline, filtering or CZML resampling. Durations are explicit [B,N] so this call site
    # remains valid when the output layer moves from uniform to nonuniform segments.
    active_segments = replay.segment_durations_s > 0.0
    block["raw_kinematics"] = raw_kinematic_metrics(
        replay.anchors,
        replay.predicted,
        replay.segment_durations_s,
        valid_segments=active_segments,
    )
    observed_nodes, observed_duration_s, _ = fixed_anchor_common_truth(
        dataset.series, config, replay.predicted.shape[1]
    )
    observed_segment_durations_s = np.broadcast_to(
        (observed_duration_s / replay.predicted.shape[1])[:, None],
        replay.segment_durations_s.shape,
    ).copy()
    block["raw_kinematics_observed_baseline"] = raw_kinematic_metrics(
        replay.anchors,
        observed_nodes,
        observed_segment_durations_s,
    )
    return block


def evaluate_fixed_anchor_common_grid(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    *,
    replay: SplitPredictionReplay | None = None,
) -> dict[str, Any]:
    """Deployable fixed-anchor metrics on one shared physical-time grid."""
    replay = replay or predict_split(
        model, dataset, normalizer, device, config.batch_size
    )
    block = fixed_anchor_common_grid_metrics(
        dataset.series,
        config,
        replay.anchors,
        replay.predicted,
        replay.predicted_time_s,
        replay.segment_durations_s,
        points=config.validation_common_grid_points,
        normalizer=normalizer,
    )
    return {
        key: value
        for key, value in block.items()
        if not isinstance(value, np.ndarray)
    }


def validation_datasets(
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    *,
    minimum_anchor_index: int | None = None,
    fitted_teacher: FittedTeacherTable | None = None,
) -> dict[str, TrajectoryWindows]:
    by_airport: dict[str, list[FlightSeries]] = {}
    for item in series:
        by_airport.setdefault(item.airport or "<unknown>", []).append(item)
    return {
        airport: FixedAnchorTrajectoryWindows(
            group,
            config,
            normalizer,
            minimum_anchor_index=minimum_anchor_index,
            fitted_teacher=fitted_teacher,
        )
        for airport, group in sorted(by_airport.items())
    }


VALIDATION_DURATION_BUCKETS_S = (180.0, 360.0, 600.0)


@dataclass(frozen=True)
class ValidationBatch:
    indices: np.ndarray
    raw_batch: tuple
    bucket: str
    query_points: int


@dataclass(frozen=True)
class ValidationBatchPlan:
    """Cached, validation-only batches grouped by fixed-anchor remaining duration."""

    dataset: TrajectoryWindows
    batches: tuple[ValidationBatch, ...]
    bucket_flights: dict[str, int]
    common_truth: CommonGridTruth | None

    @property
    def flights(self) -> int:
        return len(self.dataset)

    @property
    def query_points(self) -> int:
        return sum(batch.query_points for batch in self.batches)


def _duration_bucket_label(bucket: int) -> str:
    lower = 0.0 if bucket == 0 else VALIDATION_DURATION_BUCKETS_S[bucket - 1]
    if bucket < len(VALIDATION_DURATION_BUCKETS_S):
        upper = VALIDATION_DURATION_BUCKETS_S[bucket]
        return f"({lower:g},{upper:g}]s"
    return f"({lower:g},inf)s"


def build_validation_batch_plan(
    dataset: TrajectoryWindows,
    batch_size: int,
    *,
    duration_bucketed: bool = False,
    common_truth: CommonGridTruth | None = None,
) -> ValidationBatchPlan:
    """Build each fixed validation tensor once; optionally group no-grad rows by duration.

    ``common_truth`` is the truth cache the selection metric scores against. It is the
    ``L-1`` one by default; an anchor-grid set passes its own, because its truth is
    measured from ITS anchors and computing the L-1 one for it would be both wasted and
    wrong.
    """
    if not isinstance(dataset, FixedAnchorTrajectoryWindows):
        raise TypeError("validation batching requires a fixed-anchor dataset")
    durations = np.asarray([
        dataset.series[series_index].supervision_times[-1]
        - dataset.series[series_index].times[anchor]
        for series_index, anchor in dataset.index
    ], dtype=np.float64)
    bucket_ids = (
        np.searchsorted(
            np.asarray(VALIDATION_DURATION_BUCKETS_S, dtype=np.float64),
            durations,
            side="left",
        )
        if duration_bucketed
        else np.zeros(len(dataset), dtype=np.int64)
    )
    batches: list[ValidationBatch] = []
    bucket_flights: dict[str, int] = {}
    bucket_count = len(VALIDATION_DURATION_BUCKETS_S) + 1 if duration_bucketed else 1
    for bucket in range(bucket_count):
        indices = np.flatnonzero(bucket_ids == bucket).astype(np.int64, copy=False)
        if not len(indices):
            continue
        label = _duration_bucket_label(bucket) if duration_bucketed else "unbucketed"
        bucket_flights[label] = len(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            raw_batch = dataset.batch(batch_indices)
            dense = unpack_batch(raw_batch)[-1]
            query_points = (
                int(dense.valid.sum())
                if dense is not None
                else len(batch_indices) * dataset.config.pred_len
            )
            batches.append(
                ValidationBatch(
                    indices=batch_indices,
                    raw_batch=raw_batch,
                    bucket=label,
                    query_points=query_points,
                )
            )
    covered = (
        np.concatenate([batch.indices for batch in batches])
        if batches else np.array([], dtype=np.int64)
    )
    if not np.array_equal(np.sort(covered), np.arange(len(dataset), dtype=np.int64)):
        raise ValueError("validation duration buckets must cover every row exactly once")
    if common_truth is None and (
        dataset.config.checkpoint_selection_metric
        in CHECKPOINT_SELECTION_COMMON_GRID_METRICS
    ):
        common_truth = fixed_anchor_common_truth(
            dataset.series,
            dataset.config,
            dataset.config.validation_common_grid_points,
        )
    return ValidationBatchPlan(
        dataset=dataset,
        batches=tuple(batches),
        bucket_flights=bucket_flights,
        common_truth=common_truth,
    )


@dataclass(frozen=True)
class ValidationAirportEvaluation:
    components: dict[str, float]
    replay: SplitPredictionReplay
    profile: dict[str, Any]


def evaluate_validation_airport(
    model: nn.Module,
    plan: ValidationBatchPlan,
    device: torch.device,
    *,
    config: TSConfig,
    profiler: EpochProfiler | None = None,
    multipliers: ProcedureMultipliers | None = None,
) -> ValidationAirportEvaluation:
    """Evaluate both validation clocks from one model forward per cached batch.

    ``config`` is the config the OBJECTIVE is scored under — the epoch's, which differs
    from the dataset's own only in the annealed ``latent_beta`` (train.py). The plan's
    dataset keeps owning the batches, the normalizer and the replay.
    """
    dataset = plan.dataset
    names = loss_component_names(config)
    component_totals = {name: 0.0 for name in names}
    flight_weight_total = 0.0
    replay_chunks: list[tuple[np.ndarray, SplitPredictionReplay]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in plan.batches:
            section = profiler.section if profiler is not None else None
            data_context = section("val_data_s") if section else nullcontext()
            with data_context:
                (
                    x,
                    y,
                    mask,
                    final_time_s,
                    flight_weights,
                    dynamics,
                    dense_supervision,
                ) = unpack_batch(batch.raw_batch)
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                final_time_s = final_time_s.to(device)
                flight_weights = flight_weights.to(device)
                dynamics = move_dynamics(dynamics, device)
                dense_supervision = move_fixed_dt_supervision(
                    dense_supervision, device
                )
            objective_context = (
                section("val_objective_s") if section else nullcontext()
            )
            with objective_context:
                prediction = model_forward(model, x, dynamics, future=(y, final_time_s))
                components = prediction_loss_components(
                    prediction,
                    anchor_state(x, len(config.channels)),
                    y,
                    mask,
                    final_time_s,
                    flight_weights,
                    config,
                    dataset.normalizer,
                    dynamics,
                    dense_supervision,
                    multipliers=multipliers,
                )
            for name, value in components.tensors().items():
                component_totals[name] += float(value) * len(flight_weights)
            flight_weight_total += float(flight_weights.sum())
            selection_context = (
                section("val_checkpoint_selection_s") if section else nullcontext()
            )
            with selection_context:
                # The DEPLOYABLE replay: what the checkpoint predicts without the
                # truth's future. A latent model's objective forward above decoded a
                # posterior sample (it read the future); the replay the fixed-anchor
                # selection metrics score must be the prior top-1 decode. (The
                # objective-based selection metric would still read the posterior —
                # config refuses it for a latent run.)
                deployable = (
                    model_forward(model, x, dynamics)
                    if getattr(model, "consumes_future", False) else prediction
                )
                replay_chunks.append((
                    batch.indices,
                    _prediction_batch_replay(
                        deployable,
                        x,
                        y,
                        mask,
                        final_time_s,
                        dynamics,
                        dataset,
                    ),
                ))
    denominator = max(flight_weight_total, 1.0)
    replay = _merge_prediction_replays(replay_chunks, count=len(dataset))
    profile = {
        "flights": plan.flights,
        "batches": len(plan.batches),
        "query_points": plan.query_points,
        "wall_s": time.perf_counter() - started,
        "duration_bucket_flights": dict(plan.bucket_flights),
    }
    return ValidationAirportEvaluation(
        components={
            name: value / denominator for name, value in component_totals.items()
        },
        replay=replay,
        profile=profile,
    )


def replay_validation_plan(
    model: nn.Module,
    plan: ValidationBatchPlan,
    device: torch.device,
) -> SplitPredictionReplay:
    """The DEPLOYABLE replay of one cached plan: one forward per batch, no objective.

    `evaluate_validation_airport` scores the objective and replays in the same pass; an
    anchor-grid set is scored on the replay alone, so it does neither the loss nor the
    posterior forward. ``model_forward`` without ``future`` is the deployable decode for
    every output — a latent model's prior top-1, not its posterior sample.
    """
    model.eval()
    chunks: list[tuple[np.ndarray, SplitPredictionReplay]] = []
    with torch.no_grad():
        for batch in plan.batches:
            (
                x,
                y,
                mask,
                final_time_s,
                _flight_weights,
                dynamics,
                _dense_supervision,
            ) = unpack_batch(batch.raw_batch)
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            final_time_s = final_time_s.to(device)
            dynamics = move_dynamics(dynamics, device)
            chunks.append((
                batch.indices,
                _prediction_batch_replay(
                    model_forward(model, x, dynamics),
                    x,
                    y,
                    mask,
                    final_time_s,
                    dynamics,
                    plan.dataset,
                ),
            ))
    return _merge_prediction_replays(chunks, count=len(plan.dataset))


#: The anchor-grid metric's cached validation work: bin (metres of remaining path) ->
#: airport -> that airport's flights anchored at the bin. Built once per fit, replayed
#: every epoch, exactly like the ``L-1`` plans beside it.
AnchorGridPlans = dict[float, dict[str, ValidationBatchPlan]]


def build_anchor_grid_validation_plans(
    val_sets: dict[str, TrajectoryWindows],
    batch_size: int,
) -> AnchorGridPlans:
    """One cached validation plan per (bin, airport), at each flight's own bin anchor.

    The bins, the future floor and the per-flight anchor rule are `anchor_grid`'s — the
    same ones `run_ts_anytime_curve.py` draws the curve on. A flight with no admissible
    anchor at a bin is simply absent from that bin's plan (it has no reading there, and a
    reading taken elsewhere on its track would not be one). A bin no flight can reach is
    refused here rather than silently dropped from the mean: a five-set metric that
    quietly became a four-set one would not be comparable across arms.

    No ``fitted_teacher`` is passed on purpose: a fitted table is bound to the L−1 anchor
    it was fitted at, and these sets score a REPLAY, never an objective — nothing here
    reads the imitation supervision the table would have replaced.
    """
    plans: AnchorGridPlans = {}
    for target_m in VALIDATION_ANCHOR_GRID_M:
        by_airport: dict[str, ValidationBatchPlan] = {}
        for airport, dataset in val_sets.items():
            config = dataset.config
            series = dataset.series
            anchors = anchors_for_bin(
                series,
                remaining_path_profiles(series),
                target_m,
                seq_len=config.seq_len,
                min_future_s=DEFAULT_GRID_MIN_FUTURE_S,
            )
            if not anchors:
                raise ValueError(
                    f"no {airport} validation flight can be anchored at "
                    f"{target_m / 1000:g} km of remaining path with "
                    f"{DEFAULT_GRID_MIN_FUTURE_S:g} s of truth after it, so the "
                    f"{CHECKPOINT_SELECTION_ANCHOR_GRID_ADE} metric has no reading there; "
                    "this cohort cannot support the anchor grid"
                )
            subset = [series[index] for index in anchors]
            windows = ExplicitAnchorTrajectoryWindows(
                subset,
                config,
                dataset.normalizer,
                anchors={
                    series[index].dataset_id: anchor
                    for index, anchor in anchors.items()
                },
            )
            by_airport[airport] = build_validation_batch_plan(
                windows,
                batch_size,
                common_truth=common_truth_at_anchors(
                    subset,
                    config,
                    config.validation_common_grid_points,
                    list(anchors.values()),
                ),
            )
        plans[target_m] = by_airport
    return plans


def anchor_grid_coverage(plans: AnchorGridPlans) -> dict[str, dict[str, int]]:
    """Bin -> airport -> flights with a reading there. Every bounded coverage is stated."""
    return {
        f"{target_m:g}": {
            airport: len(plan.dataset) for airport, plan in by_airport.items()
        }
        for target_m, by_airport in plans.items()
    }


@dataclass(frozen=True)
class ValidationSelection:
    """One deterministic checkpoint-selection result over fixed-anchor validation."""

    metric: str
    value: float
    by_airport: dict[str, float]
    details_by_airport: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The anchor-grid metric's per-anchor-set record: which bins were scored, on how many
    #: flights, and what each scored. Empty for every other metric.
    anchor_grid: dict[str, Any] = field(default_factory=dict)


def _objective_validation_selection(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
    anchor_grid_plans: AnchorGridPlans | None = None,
) -> ValidationSelection:
    del model, val_sets, normalizer, config, device, precomputed_details_by_airport
    del anchor_grid_plans
    return ValidationSelection(
        metric=CHECKPOINT_SELECTION_OBJECTIVE,
        value=float(np.mean(list(val_by_airport.values()))),
        by_airport=dict(val_by_airport),
    )


def common_grid_validation_details(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
    replays_by_airport: dict[str, SplitPredictionReplay] | None = None,
    common_truth_by_airport: dict[str, CommonGridTruth] | None = None,
) -> dict[str, dict[str, Any]]:
    del val_by_airport
    if precomputed_details_by_airport is not None:
        if set(precomputed_details_by_airport) != set(val_sets):
            raise ValueError("precomputed validation details do not match airport sets")
        return precomputed_details_by_airport
    if replays_by_airport is not None and set(replays_by_airport) != set(val_sets):
        raise ValueError("validation replays do not match airport sets")
    if (
        common_truth_by_airport is not None
        and set(common_truth_by_airport) != set(val_sets)
    ):
        raise ValueError("validation common-truth caches do not match airport sets")
    details: dict[str, dict[str, Any]] = {}
    for airport, dataset in val_sets.items():
        replay = (
            replays_by_airport[airport]
            if replays_by_airport is not None
            else predict_split(
                model, dataset, normalizer, device, config.batch_size
            )
        )
        if (
            config.checkpoint_selection_metric
            in CHECKPOINT_SELECTION_COMMON_GRID_METRICS
        ):
            block = fixed_anchor_common_grid_ade_metrics(
                dataset.series,
                config,
                replay.anchors,
                replay.predicted,
                replay.predicted_time_s,
                replay.segment_durations_s,
                points=config.validation_common_grid_points,
                common_truth=(
                    common_truth_by_airport[airport]
                    if common_truth_by_airport is not None
                    else None
                ),
            )
            details[airport] = {
                "ade_m": block["ade_m"],
                "fde_m": block["fde_m"],
                "final_time_mae_s": block["final_time_mae_s"],
                "flights": block["flights"],
            }
            continue
        block = evaluate_fixed_anchor_common_grid(
            model,
            dataset,
            normalizer,
            config,
            device,
            replay=replay,
        )
        details[airport] = {
            "ade_m": block["ade_m"],
            "fde_m": block["fde_m"],
            "dense_state_loss": block["dense_state_loss"],
            "terminal_velocity_error_mps": block[
                "terminal_velocity_error_mps"
            ],
            "final_time_mae_s": block["final_time_mae_s"],
            "flights": block["flights"],
            "arc_length_geometry_loss": block["arc_length_geometry_loss"],
            "arc_length_geometry_unweighted_loss": block[
                "arc_length_geometry_unweighted_loss"
            ],
            "arc_length_distance_mean_m": block["arc_length_distance_mean_m"],
            "arc_length_path_length_ratio": block[
                "arc_length_path_length_ratio"
            ],
            "arc_length_path_length_log_error": block[
                "arc_length_path_length_log_error"
            ],
            "arc_length_horizontal_velocity_mae_mps": block[
                "arc_length_horizontal_velocity_mae_mps"
            ],
            "arc_length_horizontal_velocity_p95_mps": block[
                "arc_length_horizontal_velocity_p95_mps"
            ],
            "arc_length_horizontal_tangent_mean": block[
                "arc_length_horizontal_tangent_mean"
            ],
            "arc_length_horizontal_tangent_p95": block[
                "arc_length_horizontal_tangent_p95"
            ],
            "arc_length_horizontal_speed_mae_mps": block[
                "arc_length_horizontal_speed_mae_mps"
            ],
            "arc_length_horizontal_speed_p95_mps": block[
                "arc_length_horizontal_speed_p95_mps"
            ],
            "arc_length_vertical_velocity_mae_mps": block[
                "arc_length_vertical_velocity_mae_mps"
            ],
            "arc_length_vertical_velocity_p95_mps": block[
                "arc_length_vertical_velocity_p95_mps"
            ],
            "arc_length_horizontal_mean_m": block[
                "arc_length_horizontal_mean_m"
            ],
            "arc_length_horizontal_p95_m": block[
                "arc_length_horizontal_p95_m"
            ],
            "arc_length_vertical_mae_m": block["arc_length_vertical_mae_m"],
            "arc_length_vertical_p95_m": block["arc_length_vertical_p95_m"],
            "arc_length_terminal_position_m": block[
                "arc_length_terminal_position_m"
            ],
            "arc_length_terminal_velocity_error_mps": block[
                "arc_length_terminal_velocity_error_mps"
            ],
            "arc_length_terminal_position_runway_components_m": block[
                "arc_length_terminal_position_runway_components_m"
            ],
            "arc_length_terminal_velocity_runway_components_mps": block[
                "arc_length_terminal_velocity_runway_components_mps"
            ],
        }
    return details


def _common_grid_validation_selection(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
    anchor_grid_plans: AnchorGridPlans | None = None,
) -> ValidationSelection:
    del anchor_grid_plans
    details = common_grid_validation_details(
        model=model, val_sets=val_sets, normalizer=normalizer, config=config,
        device=device, val_by_airport=val_by_airport,
        precomputed_details_by_airport=precomputed_details_by_airport,
    )
    by_airport = {
        airport: float(block["ade_m"]) for airport, block in details.items()
    }
    return ValidationSelection(
        metric=CHECKPOINT_SELECTION_COMMON_GRID_ADE,
        value=float(np.mean(list(by_airport.values()))),
        by_airport=by_airport,
        details_by_airport=details,
    )


#: The key the ``L-1`` anchor set carries in the recorded grid block. It is the anchor the
#: whole package evaluates at, not a bin of the remaining-path grid, so it is named rather
#: than numbered.
ANCHOR_GRID_L1_KEY = "l-1"


def _anchor_set_block(
    ade_by_airport: dict[str, float], flights_by_airport: dict[str, int]
) -> dict[str, Any]:
    """One anchor set's published cell: its ADE, and the coverage it was read on.

    The set's ADE is the AIRPORT MACRO — the mean over airports of each airport's
    flight-mean ADE — which is what `fixed-anchor-common-grid-ade` already means, so the
    ``L-1`` cell of this block IS that metric's value and the two are comparable. On a
    single-airport run (every arm in this line) the macro and the flight mean coincide.
    """
    return {
        "flights": sum(flights_by_airport.values()),
        "ade_m": float(np.mean(list(ade_by_airport.values()))),
        "by_airport": {
            airport: {"flights": flights_by_airport[airport], "ade_m": ade}
            for airport, ade in ade_by_airport.items()
        },
    }


def _anchor_grid_validation_selection(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
    anchor_grid_plans: AnchorGridPlans | None = None,
) -> ValidationSelection:
    """The mean, over FIVE anchor sets, of that set's common-grid ADE.

    The five sets are ``L-1`` and `anchor_grid`'s four remaining-path bins
    (16 / 12 / 8 / 6 km), each flight anchored at its own closest admissible sample.

    Two facts decide what the number is, and both are deliberate:

    * **equal weight per ANCHOR SET.** The five sets contribute one fifth each, whatever
      their coverage. Pooling all (flight, anchor) pairs instead would weight each bin by
      how many flights happened to reach it, i.e. by the cohort's route mix — the far bins
      would fade out of the metric exactly on the arms whose flights are vectored.
    * **each set's ADE is the mean over the flights that HAVE that anchor**, per airport,
      then the airport macro (`_anchor_set_block`). A flight absent from a bin is absent
      from that bin's mean; it is never scored 0 and never carried over from another bin.

    The ADE itself is `fixed_anchor_validation`'s common-grid ADE at every set — the same
    evaluator, the same query grid, only the anchor (and hence the truth cache) moves. The
    ``L-1`` term is therefore exactly the value `fixed-anchor-common-grid-ade` selects on,
    and it is recorded every epoch beside the four bins so the two metrics stay readable
    against each other.

    Lower is better, like every other selection metric.
    """
    if anchor_grid_plans is None:
        raise ValueError(
            f"{CHECKPOINT_SELECTION_ANCHOR_GRID_ADE} needs its cached anchor-grid plans; "
            "build them once per fit with build_anchor_grid_validation_plans"
        )
    fixed = _common_grid_validation_selection(
        model=model, val_sets=val_sets, normalizer=normalizer, config=config,
        device=device, val_by_airport=val_by_airport,
        precomputed_details_by_airport=precomputed_details_by_airport,
    )
    sets: dict[str, dict[str, Any]] = {
        ANCHOR_GRID_L1_KEY: _anchor_set_block(
            fixed.by_airport,
            {airport: len(dataset) for airport, dataset in val_sets.items()},
        )
    }
    ade_by_airport_by_set: list[dict[str, float]] = [dict(fixed.by_airport)]
    for target_m, by_airport in anchor_grid_plans.items():
        ade_by_airport = {}
        for airport, plan in by_airport.items():
            replay = replay_validation_plan(model, plan, device)
            block = fixed_anchor_common_grid_ade_metrics(
                plan.dataset.series,
                config,
                replay.anchors,
                replay.predicted,
                replay.predicted_time_s,
                replay.segment_durations_s,
                points=config.validation_common_grid_points,
                common_truth=plan.common_truth,
            )
            ade_by_airport[airport] = float(block["ade_m"])
        ade_by_airport_by_set.append(ade_by_airport)
        sets[f"{target_m:g}"] = _anchor_set_block(
            ade_by_airport,
            {airport: len(plan.dataset) for airport, plan in by_airport.items()},
        )
    value = float(np.mean([block["ade_m"] for block in sets.values()]))
    return ValidationSelection(
        metric=CHECKPOINT_SELECTION_ANCHOR_GRID_ADE,
        value=value,
        # Per airport, the same mean over the same five sets — the decomposition of the
        # value, not a second definition of it.
        by_airport={
            airport: float(np.mean([
                ade_by_airport[airport] for ade_by_airport in ade_by_airport_by_set
            ]))
            for airport in val_sets
        },
        details_by_airport=fixed.details_by_airport,
        anchor_grid={
            "selection_metric": CHECKPOINT_SELECTION_ANCHOR_GRID_ADE,
            "grid_km": list(VALIDATION_ANCHOR_GRID_KM),
            "min_future_s": DEFAULT_GRID_MIN_FUTURE_S,
            "anchor_sets": sets,
            "mean_ade_m": value,
            # The other metric's number, named so a reader comparing arms does not have to
            # know this block's key convention. It IS anchor_sets["l-1"]["ade_m"].
            "fixed_anchor_common_grid_ade_m": sets[ANCHOR_GRID_L1_KEY]["ade_m"],
        },
    )


VALIDATION_SELECTIONS: dict[str, Callable[..., ValidationSelection]] = {
    CHECKPOINT_SELECTION_OBJECTIVE: _objective_validation_selection,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE: _common_grid_validation_selection,
    CHECKPOINT_SELECTION_ANCHOR_GRID_ADE: _anchor_grid_validation_selection,
}
