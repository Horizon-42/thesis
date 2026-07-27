"""Resolve an efficient batch size against the actual model and CUDA device.

The auto path probes complete FP32 training steps (forward, backward, Adam update) on
synthetic tensors with the run's real ``L/N/C`` and architecture.  That is more reliable
than naming GPU models in a table: free memory, model width, layer count and output grid all
matter. The largest successful power of two is used directly.
"""

from __future__ import annotations

import gc

import torch

from config import TSConfig
from models import build_model

_CANDIDATES = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _probe_training_step(config: TSConfig, batch_size: int, device: torch.device) -> None:
    """Run one isolated optimizer step or raise CUDA OOM."""
    model = optimizer = x = prediction = loss = None
    try:
        torch.manual_seed(config.seed)
        model = build_model(config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        x = torch.zeros(
            (batch_size, config.seq_len, len(config.channels)),
            dtype=torch.float32,
            device=device,
        )
        prediction = model(x)
        loss = prediction.states.square().mean() + config.final_time_loss_weight * (
            prediction.final_time_s / config.final_time_scale_s
        ).square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
    finally:
        del loss, prediction, x, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def resolve_batch_size(
    config: TSConfig,
    device: torch.device,
    *,
    auto: bool,
    verbose: bool = True,
) -> int:
    """Return explicit config size, or probe a safe power-of-two CUDA batch."""
    if not auto:
        return config.batch_size
    if device.type != "cuda" or not torch.cuda.is_available():
        if verbose:
            print(f"  batch      auto -> {config.batch_size} (CPU/default; no CUDA probe)")
        return config.batch_size

    successful: list[int] = []
    for candidate in _CANDIDATES:
        try:
            _probe_training_step(config, candidate, device)
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            torch.cuda.empty_cache()
            break
        successful.append(candidate)

    if not successful:
        raise RuntimeError(
            "automatic batch-size probe could not fit batch_size=8; reduce model width/layers"
        )
    largest = successful[-1]
    selected = largest
    props = torch.cuda.get_device_properties(device)
    if verbose:
        memory_gib = props.total_memory / 1024**3
        cap = "+" if largest == _CANDIDATES[-1] else ""
        print(
            f"  batch      auto -> {selected} on {props.name} ({memory_gib:.1f} GiB; "
            f"largest successful probe {largest}{cap})"
        )
    return selected
