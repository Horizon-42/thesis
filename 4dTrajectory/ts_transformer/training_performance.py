"""Low-overhead, device-aware timing for the development training loop.

CUDA events preserve asynchronous execution between sections.  One synchronization at the
end of an epoch resolves all events and peak-memory counters without inserting a barrier
between model forward, rollout/loss and backward.  CPU-only tests use ``perf_counter`` with
the same public contract.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class _CudaInterval:
    name: str
    start: torch.cuda.Event
    end: torch.cuda.Event


class EpochProfiler:
    """Accumulate non-overlapping stage durations without changing model operations."""

    def __init__(self, device: torch.device):
        self.device = device
        self._cuda = device.type == "cuda"
        self._cuda_intervals: list[_CudaInterval] = []
        self._cpu_seconds: dict[str, float] = defaultdict(float)
        self.started = time.perf_counter()
        if self._cuda:
            torch.cuda.reset_peak_memory_stats(device)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Measure device work in ``name`` without synchronizing at the boundary."""
        if self._cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._cuda_intervals.append(_CudaInterval(name, start, end))
            return

        started = time.perf_counter()
        try:
            yield
        finally:
            self._cpu_seconds[name] += time.perf_counter() - started

    def add_cpu_seconds(self, name: str, seconds: float) -> None:
        self._cpu_seconds[name] += float(seconds)

    def finish(self, *, optimizer_updates: int) -> dict[str, float]:
        """Synchronize once and return stable JSON-serializable epoch diagnostics."""
        if self._cuda:
            torch.cuda.synchronize(self.device)
            for interval in self._cuda_intervals:
                self._cpu_seconds[interval.name] += (
                    interval.start.elapsed_time(interval.end) / 1000.0
                )
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
        else:
            peak_memory_mb = 0.0
        total_s = time.perf_counter() - self.started
        train_s = sum(
            self._cpu_seconds.get(name, 0.0)
            for name in (
                "train_data_s",
                "train_forward_s",
                "train_rollout_loss_s",
                "train_backward_step_s",
            )
        )
        result = dict(sorted(self._cpu_seconds.items()))
        result.update(
            {
                "epoch_total_s": float(total_s),
                "peak_cuda_memory_mb": float(peak_memory_mb),
                "optimizer_updates_per_s": (
                    float(optimizer_updates / train_s) if train_s > 0.0 else 0.0
                ),
            }
        )
        return result
