"""Resolve an efficient batch size against the actual model and CUDA device.

The auto path probes complete FP32 training steps (forward, backward, Adam update) on
synthetic tensors with the run's real ``L/N/C`` and architecture.  That is more reliable
than naming GPU models in a table: free memory, model width, layer count and output grid all
matter. The largest successful power of two is used directly.
"""

from __future__ import annotations

import gc

import numpy as np
import torch

from config import TSConfig
from models import build_model

_CANDIDATES = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _probe_training_step(config: TSConfig, batch_size: int, device: torch.device) -> None:
    """Run the real state/time/physics loss once or raise CUDA OOM."""
    # Local imports avoid a module cycle: train imports resolve_batch_size, while the probe
    # must share train's loss implementation so its retained CUDA graph cannot drift.
    from dataset import Normalizer
    from train import prediction_loss

    model = optimizer = x = target = state_weights = None
    target_final_time_s = flight_weights = prediction = loss = normalizer = None
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
        target = torch.zeros(
            (batch_size, config.pred_len, len(config.channels)),
            dtype=torch.float32,
            device=device,
        )
        state_weights = torch.ones_like(target)
        target_final_time_s = torch.full(
            (batch_size,),
            config.final_time_scale_s,
            dtype=torch.float32,
            device=device,
        )
        flight_weights = torch.ones(batch_size, dtype=torch.float32, device=device)
        normalizer = Normalizer(
            mean=np.zeros(len(config.channels), dtype=np.float64),
            std=np.ones(len(config.channels), dtype=np.float64),
        )
        optimizer.zero_grad()
        prediction = model(x)
        loss = prediction_loss(
            prediction,
            x[:, -1],
            target,
            state_weights,
            target_final_time_s,
            flight_weights,
            config,
            normalizer,
        )
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
    finally:
        del loss, prediction, flight_weights, target_final_time_s, state_weights, target, x
        del normalizer, optimizer, model
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
