"""Resolve an efficient batch size against the actual model and CUDA device.

The auto path probes complete training steps (forward, backward, configured gradient
diagnostics/clipping, Adam update) on synthetic tensors with the run's real ``L/N/C`` and
architecture. Control probes also use heterogeneous non-uniform duration partitions because
their batched graph depth is governed by per-segment maxima. That is more reliable than
naming GPU models in a table: free memory, model width, layer count and output grid all
matter.
"""

from __future__ import annotations

import gc

import numpy as np
import torch

from ts_transformer.data.batch_contract import anchor_state, model_forward
from ts_transformer.config import TSConfig
from ts_transformer.outputs import strategy
from ts_transformer.data.dataset import Normalizer
from ts_transformer.backbone.adapters import build_model
from ts_transformer.training.objective import prediction_loss

_CANDIDATES = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _probe_training_step(config: TSConfig, batch_size: int, device: torch.device) -> None:
    """Run the real state/time/physics loss once or raise CUDA OOM.

    The probe runs the SHARED objective, not a copy of it, so the CUDA graph it retains is
    the one the epoch will build. That import used to be deferred into this function to
    break a cycle through ``train`` (which imports ``resolve_batch_size``); the objective
    lives in its own module now, so the cycle — and the deferral — are gone.
    """
    model = optimizer = x = target = state_weights = control_diagnostics = None
    target_final_time_s = flight_weights = prediction = loss = normalizer = None
    output_strategy = strategy(config)
    try:
        torch.manual_seed(config.seed)
        model = build_model(config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        x = torch.zeros(
            (batch_size, config.seq_len, len(config.input_channels)),
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
        context = output_strategy.probe_context(batch_size, device, config)
        dense_supervision = output_strategy.probe_dense_supervision(batch_size, device, config)
        optimizer.zero_grad()
        # the future as the training loop hands it over (`train.fit_model`): a model that consumes it (the latent
        # control model's posterior encoder and its KL) builds the graph it trains, every other model ignores it
        prediction = output_strategy.probe_prediction(
            model_forward(model, x, context, future=(target, target_final_time_s))
        )
        control_diagnostics = output_strategy.training_diagnostics(config)
        if control_diagnostics is not None:
            control_diagnostics.record_prediction(prediction, context)
        loss = prediction_loss(
            prediction,
            anchor_state(x, len(config.channels)),
            target,
            state_weights,
            target_final_time_s,
            flight_weights,
            config,
            normalizer,
            context,
            dense_supervision,
        )
        loss.backward()
        if control_diagnostics is not None:
            control_diagnostics.record_gradients_and_clip(model)
        optimizer.step()
        finite_model = all(torch.isfinite(parameter).all() for parameter in model.parameters())
        finite_optimizer = all(
            not isinstance(value, torch.Tensor) or torch.isfinite(value).all()
            for state in optimizer.state.values()
            for value in state.values()
        )
        if not finite_model or not finite_optimizer:
            raise RuntimeError(
                "automatic batch-size probe produced non-finite model/Adam state"
            )
        torch.cuda.synchronize(device)
    finally:
        del loss, prediction, flight_weights, target_final_time_s, state_weights, target, x
        del control_diagnostics, normalizer, optimizer, model
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
            if not is_cuda_oom(exc):
                raise
            torch.cuda.empty_cache()
            break
        successful.append(candidate)

    if not successful:
        raise RuntimeError(
            "automatic batch-size probe could not fit batch_size=8; reduce model width/layers"
        )
    largest = successful[-1]
    # Control rollout depth can grow further as learned duration logits sharpen. Retain one
    # tested power-of-two as a memory margin after the heterogeneous probe; the state path
    # keeps its historical largest-successful behavior.
    selected = (
        successful[-2]
        if strategy(config).keeps_batch_margin and len(successful) > 1
        else largest
    )
    props = torch.cuda.get_device_properties(device)
    if verbose:
        memory_gib = props.total_memory / 1024**3
        cap = "+" if largest == _CANDIDATES[-1] else ""
        print(
            f"  batch      auto -> {selected} on {props.name} ({memory_gib:.1f} GiB; "
            f"largest successful probe {largest}{cap}"
            f"{' with one-step control safety margin' if selected != largest else ''})"
        )
    return selected
