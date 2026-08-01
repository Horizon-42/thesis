#!/usr/bin/env python3
"""Benchmark equivalent legacy/shared validation execution on sealed development data.

Only checkpoint-listed validation identities are loaded.  The script never opens outer-test
tracks and never changes RK4, dtype, loss, model, batch size or checkpoint-selection math.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train as train_module
from dataset import (
    FixedAnchorTrajectoryWindows,
    Normalizer,
    arrival_data_provenance,
    build_series,
    flight_keys_by_split,
    load_flight_dicts,
    require_matching_data_provenance,
    split_by_flight,
)
from config import (
    AIRCRAFT_FILTER_OPENAP_DIRECT,
    CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
    CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
    PREDICTION_CONTROL,
    TSConfig,
)
from models import build_model, resolve_device
from train import load_checkpoint


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(call: Callable[[], Any], device: torch.device) -> tuple[Any, float, float]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    value = call()
    _sync(device)
    elapsed = time.perf_counter() - started
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda" else 0.0
    )
    return value, elapsed, peak_mb


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _scalar_quality_delta(
    legacy: dict[str, Any], optimized: dict[str, Any]
) -> dict[str, float]:
    pairs: list[tuple[float, float]] = []
    for airport in legacy:
        for family in ("objective", "common_grid"):
            left = legacy[airport][family]
            right = optimized[airport][family]
            for key in left:
                if key in right and isinstance(left[key], (int, float)):
                    pairs.append((float(left[key]), float(right[key])))
    absolute = [abs(right - left) for left, right in pairs]
    relative = [
        abs(right - left) / max(abs(left), 1e-12) for left, right in pairs
    ]
    return {
        "compared_scalars": len(pairs),
        "max_absolute": max(absolute, default=0.0),
        "max_relative": max(relative, default=0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", choices=("itransformer", "patchtst"), default="itransformer")
    parser.add_argument("--flights-per-airport", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--duration-bucketed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.flights_per_airport <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("flight limit/iterations must be positive and warmup non-negative")

    current_provenance = arrival_data_provenance(args.data)
    if args.checkpoint is not None:
        model, config, normalizer, payload = load_checkpoint(args.checkpoint)
        require_matching_data_provenance(payload, current_provenance, allow_subset=True)
        validation_keys = set(payload["split"]["val"])
        development_keys = validation_keys
    else:
        config = TSConfig(
            model=args.model,
            prediction_output=PREDICTION_CONTROL,
            aircraft_filter=AIRCRAFT_FILTER_OPENAP_DIRECT,
            n_segments=64,
            batch_size=512,
            split_seed=1337,
            seed=1337,
            device=args.device,
            control_dynamics_backend=(
                CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY
            ),
            control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
            control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
            control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
            checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        )
        split_keys = flight_keys_by_split(current_provenance, config)
        validation_keys = set(split_keys["val"])
        development_keys = set(split_keys["train"] + split_keys["val"])
    flights = load_flight_dicts(
        args.data,
        include_flight_keys=development_keys,
        verbose=False,
    )
    series, report = build_series(flights, config)
    if args.checkpoint is None:
        train_series, validation_series, leaked_test = split_by_flight(series, config)
        if leaked_test:
            raise ValueError("development-only load unexpectedly contained outer-test rows")
        normalizer = Normalizer.fit(
            train_series, balance_airports_and_flights=True
        )
        torch.manual_seed(config.seed)
        model = build_model(config)
        series = validation_series
        validation_keys = {item.dataset_id for item in validation_series}
    by_airport: dict[str, list] = {}
    for item in series:
        if item.dataset_id in validation_keys:
            by_airport.setdefault(item.airport, []).append(item)
    selected = {
        airport: sorted(group, key=lambda item: item.dataset_id)[
            : args.flights_per_airport
        ]
        for airport, group in sorted(by_airport.items())
    }
    missing = validation_keys - {item.dataset_id for group in by_airport.values() for item in group}
    if missing:
        raise ValueError(f"failed to rebuild {len(missing)} checkpoint validation flights")

    datasets = {
        airport: FixedAnchorTrajectoryWindows(group, config, normalizer)
        for airport, group in selected.items()
    }
    plans = {
        airport: train_module.build_validation_batch_plan(
            dataset,
            config.batch_size,
            duration_bucketed=args.duration_bucketed,
        )
        for airport, dataset in datasets.items()
    }
    device = resolve_device(args.device)
    model = model.to(device).eval()

    def legacy() -> dict[str, Any]:
        result = {}
        for airport, dataset in datasets.items():
            objective = train_module._dataset_loss_components(
                model, dataset, device, config.batch_size
            )
            common = train_module.evaluate_fixed_anchor_common_grid(
                model, dataset, normalizer, config, device
            )
            result[airport] = {"objective": objective, "common_grid": common}
        return result

    def optimized() -> dict[str, Any]:
        result = {}
        for airport, plan in plans.items():
            evaluation = train_module._evaluate_validation_airport(
                model,
                plan,
                device,
                None,
                include_deployable_replay=True,
            )
            common = train_module.evaluate_fixed_anchor_common_grid(
                model,
                plan.dataset,
                normalizer,
                config,
                device,
                replay=evaluation.replay,
            )
            result[airport] = {
                "objective": evaluation.components,
                "common_grid": common,
            }
        return result

    for index in range(args.warmup):
        (legacy if index % 2 == 0 else optimized)()
        (optimized if index % 2 == 0 else legacy)()

    legacy_seconds: list[float] = []
    optimized_seconds: list[float] = []
    legacy_peak: list[float] = []
    optimized_peak: list[float] = []
    legacy_result = optimized_result = None
    for index in range(args.iterations):
        order = ((legacy, legacy_seconds, legacy_peak),
                 (optimized, optimized_seconds, optimized_peak))
        if index % 2:
            order = tuple(reversed(order))
        for call, seconds, peaks in order:
            value, elapsed, peak = _timed(call, device)
            seconds.append(elapsed)
            peaks.append(peak)
            if call is legacy:
                legacy_result = value
            else:
                optimized_result = value
    assert legacy_result is not None and optimized_result is not None
    legacy_median = statistics.median(legacy_seconds)
    optimized_median = statistics.median(optimized_seconds)
    document = {
        "schema_version": "ts-validation-execution-benchmark-v1",
        "isolation": {
            "loaded_split": "validation only",
            "outer_test_tracks_loaded": False,
            "checkpoint": (
                str(args.checkpoint.resolve()) if args.checkpoint is not None else None
            ),
            "fresh_model_seed": config.seed if args.checkpoint is None else None,
        },
        "contract": {
            "rk4_step_s": config.control_rollout_integrator_dt_s,
            "batch_size": config.batch_size,
            "model": config.model,
            "prediction_output": config.prediction_output,
            "state_objective": config.control_state_objective,
        },
        "data": {
            "airports": {key: len(value) for key, value in selected.items()},
            "built_development_flights": report.built,
        },
        "protocol": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "alternating_order": True,
            "duration_bucketed": args.duration_bucketed,
        },
        "legacy_two_forward": {
            "median_s": legacy_median,
            "p90_s": _percentile(legacy_seconds, 90),
            "peak_cuda_memory_mb": max(legacy_peak),
        },
        "cached_shared_forward": {
            "median_s": optimized_median,
            "p90_s": _percentile(optimized_seconds, 90),
            "peak_cuda_memory_mb": max(optimized_peak),
        },
        "speedup": legacy_median / optimized_median,
        "quality_delta": _scalar_quality_delta(legacy_result, optimized_result),
    }
    text = json.dumps(document, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
