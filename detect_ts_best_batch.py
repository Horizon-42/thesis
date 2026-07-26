#!/usr/bin/env python
"""Benchmark the fastest TS training batch using outer-train trajectory data only.

This is a performance probe, not model fitting: every candidate rebuilds the same model,
runs warm-up plus timed FP32 forward/backward/Adam steps, and discards its weights. Split
membership is decided from manifest roster identities before source tracks are opened, so
validation/test trajectory files are never read by this process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_ts_pipeline as pipeline  # noqa: E402
from config import (  # noqa: E402
    COORDINATE_FRAMES,
    DEFAULT_AIRCRAFT_TYPE,
    HORIZON_MODES,
    MODELS,
    SAMPLING_AIRPORT_FLIGHT_BALANCED,
    TSConfig,
    config_for_mode,
)
from cross_validation import TUNED_FIELDS  # noqa: E402
from dataset import (  # noqa: E402
    Normalizer,
    TrajectoryWindows,
    build_series,
    dataset_flight_key,
    iter_batches,
    split_name_for_dataset_id,
)
from models import build_model, parameter_count, resolve_device  # noqa: E402
from train import masked_mse, usable_series  # noqa: E402
from trajectory_data_process.harvest.arrivals import (  # noqa: E402
    load_arrival_flights,
    resolve_arrival_manifest,
)

RESULT_SCHEMA = "ts-batch-throughput-benchmark-v1-outer-train-only"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity_digest(identities: Sequence[str]) -> str:
    return _sha256_bytes("\n".join(sorted(identities)).encode())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _parse_airports(raw: str) -> tuple[str, ...]:
    airports = tuple(sorted({token.strip().upper() for token in raw.split(",") if token.strip()}))
    if not airports:
        raise argparse.ArgumentTypeError("--airports requires at least one ICAO code")
    return airports


def _power_of_two(value: int, flag: str) -> int:
    if value <= 0 or value & (value - 1):
        raise argparse.ArgumentTypeError(f"{flag} must be a positive power of two")
    return value


def candidate_batch_sizes(minimum: int, maximum: int) -> list[int]:
    if minimum <= 0 or minimum & (minimum - 1):
        raise ValueError("minimum batch must be a positive power of two")
    if maximum < minimum or maximum & (maximum - 1):
        raise ValueError("maximum batch must be a power of two >= minimum")
    values: list[int] = []
    candidate = minimum
    while candidate <= maximum:
        values.append(candidate)
        candidate *= 2
    return values


def _load_tuned_overrides(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path)
    try:
        loaded = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read --config-overrides {source}: {exc}") from exc
    allowed_config = {field.name for field in fields(TSConfig)}
    if not isinstance(loaded, dict) or any(key not in allowed_config for key in loaded):
        raise ValueError("--config-overrides must be a JSON object containing TSConfig fields")
    unsupported = sorted(set(loaded) - set(TUNED_FIELDS))
    if unsupported:
        raise ValueError(
            "--config-overrides is for CV best_config.json; unsupported field(s): "
            + ", ".join(unsupported)
        )
    return loaded


def load_outer_train_flights(
    manifests: Sequence[Path],
    config: TSConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open only source tracks assigned to outer-train by roster identity."""
    flights: list[dict[str, Any]] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    train_dataset_ids: list[str] = []
    manifest_digests: dict[str, str] = {}

    for raw_path in manifests:
        manifest_path = resolve_arrival_manifest(raw_path)
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError(f"{manifest_path} is not an arrival manifest object")
        airport = str(manifest.get("airport") or "").strip().upper()
        records = manifest.get("records")
        if not airport or not isinstance(records, list):
            raise ValueError(f"{manifest_path} lacks airport or records roster")
        if airport in manifest_digests:
            raise ValueError(f"duplicate arrival manifest for airport {airport}")
        manifest_digests[airport] = _sha256_bytes(manifest_bytes)

        selected_keys: set[str] = set()
        selected_dataset_ids: set[str] = set()
        for index, row in enumerate(records):
            if not isinstance(row, dict) or not isinstance(row.get("flight_key"), str):
                raise ValueError(f"{manifest_path}: record {index} lacks flight_key")
            key = row["flight_key"]
            dataset_id = f"{airport}:{key}"
            split = split_name_for_dataset_id(dataset_id, config)
            split_counts[split] += 1
            if split == "train":
                if key in selected_keys:
                    raise ValueError(f"{manifest_path} lists duplicate flight_key {key!r}")
                selected_keys.add(key)
                selected_dataset_ids.add(dataset_id)

        selected_flights = load_arrival_flights(
            manifest_path,
            include_flight_keys=selected_keys,
        )
        loaded_ids = {
            dataset_flight_key(flight, index)
            for index, flight in enumerate(selected_flights)
        }
        if loaded_ids != selected_dataset_ids:
            raise ValueError(
                f"{manifest_path}: loaded outer-train identities differ from locked roster"
            )
        flights.extend(selected_flights)
        train_dataset_ids.extend(sorted(selected_dataset_ids))
        print(
            f"  {airport}: roster train {len(selected_keys)} / val+test "
            f"{len(records) - len(selected_keys)}; opened train source tracks only",
            flush=True,
        )

    audit = {
        "selection_population": "outer_train_only",
        "split_counts_from_manifest_rosters": split_counts,
        "loaded_source_tracks": {
            "train": len(flights),
            "validation": 0,
            "test": 0,
        },
        "outer_train_roster_sha256": _identity_digest(train_dataset_ids),
        "arrival_manifests": dict(sorted(manifest_digests.items())),
    }
    return flights, audit


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def benchmark_candidate(
    dataset: TrajectoryWindows,
    config: TSConfig,
    device: torch.device,
    *,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    repeats: int,
) -> dict[str, Any]:
    """Measure the current end-to-end in-RAM DataLoader + FP32 training path."""
    model = optimizer = loader = iterator = None
    x = y = weights = predicted = loss = None
    started = time.perf_counter()
    try:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        model = build_model(config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        total_steps = warmup_steps + measure_steps * repeats
        loader = iter_batches(
            dataset,
            batch_size,
            shuffle=False,
            seed=config.seed,
            balanced=True,
            num_samples=batch_size * total_steps,
        )
        iterator = iter(loader)
        torch.cuda.reset_peak_memory_stats(device)

        def step() -> None:
            nonlocal x, y, weights, predicted, loss
            x, y, weights = next(iterator)
            x = x.to(device)
            y = y.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            predicted = model(x)
            loss = masked_mse(predicted, y, weights)
            loss.backward()
            optimizer.step()

        for _ in range(warmup_steps):
            step()
        torch.cuda.synchronize(device)

        repeat_samples_per_second: list[float] = []
        repeat_step_ms: list[float] = []
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            timed_start = time.perf_counter()
            for _step in range(measure_steps):
                step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - timed_start
            repeat_samples_per_second.append(batch_size * measure_steps / elapsed)
            repeat_step_ms.append(elapsed * 1000.0 / measure_steps)

        return {
            "batch_size": batch_size,
            "status": "ok",
            "median_samples_per_second": statistics.median(repeat_samples_per_second),
            "median_step_ms": statistics.median(repeat_step_ms),
            "repeat_samples_per_second": repeat_samples_per_second,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            "wall_seconds": time.perf_counter() - started,
        }
    except RuntimeError as exc:
        if not _is_cuda_oom(exc):
            raise
        return {
            "batch_size": batch_size,
            "status": "oom",
            "error": str(exc).splitlines()[0],
            "wall_seconds": time.perf_counter() - started,
        }
    finally:
        del loss, predicted, weights, y, x, iterator, loader, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def select_best_batch(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "ok"]
    if not successful:
        raise RuntimeError("no candidate batch completed the benchmark")
    return max(
        successful,
        key=lambda result: (result["median_samples_per_second"], -result["batch_size"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airports", type=_parse_airports, default=None,
                        help="comma-separated airport roster; default: all discovered K-airports")
    parser.add_argument("--model", choices=MODELS, default="itransformer")
    parser.add_argument("--mode", choices=HORIZON_MODES, default="full")
    parser.add_argument("--coordinate-frame", choices=COORDINATE_FRAMES, default="enu")
    parser.add_argument("--config-overrides", default=None,
                        help="optional CV best_config.json for the exact selected architecture")
    parser.add_argument("--aircraft-type", default=DEFAULT_AIRCRAFT_TYPE)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-batch", type=lambda value: _power_of_two(int(value), "--min-batch"),
                        default=32)
    parser.add_argument("--max-batch", type=lambda value: _power_of_two(int(value), "--max-batch"),
                        default=32768)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.max_batch < args.min_batch:
        parser.error("--max-batch must be >= --min-batch")
    for name in ("warmup_steps", "measure_steps", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        tuned = _load_tuned_overrides(args.config_overrides)
    except ValueError as exc:
        parser.error(str(exc))

    config = config_for_mode(
        args.mode,
        **tuned,
        model=args.model,
        coordinate_frame=args.coordinate_frame,
        aircraft_type=args.aircraft_type,
        seed=args.seed,
        device=args.device,
        sampling_strategy=SAMPLING_AIRPORT_FLIGHT_BALANCED,
        eval_anchor_policy="first",
    )
    device = resolve_device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this benchmark requires an available CUDA GPU")

    airports = args.airports or tuple(pipeline.discover_k_airports())
    if not airports:
        parser.error(f"no K-airport arrivals manifests found under {pipeline.HARVEST_ROOT}")
    manifests = [pipeline.arrival_manifest_path(airport) for airport in airports]
    missing = [path for path in manifests if not path.is_file()]
    if missing:
        parser.error(f"missing arrival manifest {missing[0]}")

    output_path = (args.output or (
        pipeline.OPT_OUTPUTS_ROOT / "POOLED" / "batch_benchmarks" /
        f"{args.model}_{args.mode}_{args.coordinate_frame.replace('-', '_')}.json"
    )).resolve()
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(
        f"GPU: {properties.name}, total {total_bytes / 1024**3:.2f} GiB, "
        f"free {free_bytes / 1024**3:.2f} GiB",
        flush=True,
    )
    print(
        f"model={config.model}, mode={config.horizon_mode}, frame={config.coordinate_frame}, "
        f"L={config.seq_len}, H={config.pred_len}, parameters={parameter_count(build_model(config)):,}",
        flush=True,
    )
    print("loading outer-train source tracks only (validation/test trajectory files stay closed)",
          flush=True)
    flights, data_audit = load_outer_train_flights(manifests, config)
    series, report = build_series(flights, config, aircraft_type=config.aircraft_type)
    print(report.format(), flush=True)
    series = usable_series(series, config, verbose=True)
    if not series:
        parser.error("outer-train produced no usable trajectory windows")
    if any(split_name_for_dataset_id(item.dataset_id, config) != "train" for item in series):
        raise RuntimeError("non-training flight reached the batch benchmark")

    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    dataset = TrajectoryWindows(series, config, normalizer)
    if not len(dataset):
        parser.error("outer-train produced an empty benchmark dataset")
    data_audit["usable_outer_train_flights"] = len(series)
    data_audit["usable_outer_train_sha256"] = _identity_digest(
        [item.dataset_id for item in series]
    )
    data_audit["trajectory_windows"] = len(dataset)

    results: list[dict[str, Any]] = []
    candidates = candidate_batch_sizes(args.min_batch, args.max_batch)
    for index, batch_size in enumerate(candidates, 1):
        print(f"\n[{index}/{len(candidates)}] benchmarking batch_size={batch_size}", flush=True)
        result = benchmark_candidate(
            dataset,
            config,
            device,
            batch_size=batch_size,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            repeats=args.repeats,
        )
        results.append(result)
        if result["status"] == "oom":
            print(f"  OOM: {result['error']}", flush=True)
            break
        print(
            f"  {result['median_samples_per_second']:.1f} samples/s, "
            f"{result['median_step_ms']:.2f} ms/step, "
            f"peak reserved {result['peak_reserved_mib']:.0f} MiB",
            flush=True,
        )

    best = select_best_batch(results)
    largest_tested = results[-1]
    search_boundary_reached = (
        largest_tested["status"] == "ok"
        and largest_tested["batch_size"] == args.max_batch
    )
    payload = {
        "schema_version": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "maximum median end-to-end outer-train samples per second",
        "best_parameter": {"batch_size": best["batch_size"]},
        "best_median_samples_per_second": best["median_samples_per_second"],
        "search_boundary_reached_without_oom": search_boundary_reached,
        "gpu": {
            "name": properties.name,
            "total_memory_mib": total_bytes / 1024**2,
            "free_memory_at_start_mib": free_bytes / 1024**2,
        },
        "config": config.to_dict(),
        "benchmark": {
            "precision": "float32",
            "training_step": "DataLoader + host-to-device + forward + backward + Adam",
            "warmup_steps": args.warmup_steps,
            "measure_steps_per_repeat": args.measure_steps,
            "repeats": args.repeats,
            "candidates": results,
        },
        "data_leakage_guard": data_audit,
    }
    _write_json_atomic(output_path, payload)

    print(f"\n✓ benchmark result: {output_path}", flush=True)
    if search_boundary_reached:
        print(
            f"WARNING: max candidate {args.max_batch} succeeded; result is best among tested "
            "values. Raise --max-batch to prove the upper boundary.",
            flush=True,
        )
    print(json.dumps({
        "best_batch_size": best["batch_size"],
        "median_samples_per_second": best["median_samples_per_second"],
    }, indent=2), flush=True)
    print(f"BEST_BATCH_SIZE={best['batch_size']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
