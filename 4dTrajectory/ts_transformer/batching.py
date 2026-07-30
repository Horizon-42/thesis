"""Resolve an efficient batch size against the actual model and CUDA device.

The auto path probes complete FP32 training steps (forward, backward, Adam update) on
synthetic tensors with the run's real ``L/N/C`` and architecture. Control probes also use
heterogeneous non-uniform duration partitions because their batched graph depth is governed
by per-segment maxima. That is more reliable than naming GPU models in a table: free memory,
model width, layer count and output grid all matter.
"""

from __future__ import annotations

import gc

import numpy as np
import torch

from config import TSConfig, uses_control_dynamics
from control_prediction_adapters import map_control_candidates
from models import build_model
from prediction_outputs import ControlPrediction

_CANDIDATES = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _heterogeneous_control_probe_prediction(
    prediction: ControlPrediction,
) -> ControlPrediction:
    """Replace identical probe partitions with deterministic heterogeneous schedules.

    Batched rollout graph length is governed by the maximum duration in each segment, not
    by one row's total duration. Zero histories otherwise make every probe row identical and
    systematically understate that graph. A phase-shifted circular profile exercises about
    three times the uniform graph depth while remaining smooth, positive, and connected to
    the model's duration output for backward-memory fidelity.
    """
    batch, segments = prediction.segment_durations.shape
    dtype, device = prediction.segment_durations.dtype, prediction.segment_durations.device
    learned_fractions = prediction.segment_durations / prediction.final_time_s.unsqueeze(1)
    segment_phase = (
        torch.arange(segments, dtype=dtype, device=device) * (2.0 * torch.pi / segments)
    ).unsqueeze(0)
    row_phase = (
        torch.arange(batch, dtype=dtype, device=device) * (2.0 * torch.pi / batch)
    ).unsqueeze(1)
    probe_profile = torch.softmax(2.0 * torch.cos(segment_phase - row_phase), dim=-1)
    fractions = learned_fractions * probe_profile
    fractions = fractions / fractions.sum(dim=-1, keepdim=True)
    durations = fractions * prediction.final_time_s.unsqueeze(1)
    return ControlPrediction(
        controls=prediction.controls,
        segment_durations=durations,
        final_time_s=prediction.final_time_s,
    )


def _probe_training_step(config: TSConfig, batch_size: int, device: torch.device) -> None:
    """Run the real state/time/physics loss once or raise CUDA OOM."""
    # Local imports avoid a module cycle: train imports resolve_batch_size, while the probe
    # must share train's loss implementation so its retained CUDA graph cannot drift.
    from dataset import Normalizer
    from train import model_forward, prediction_loss

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
        dynamics = None
        if uses_control_dynamics(config.prediction_output):
            dynamics = {
                "condition": torch.tensor(
                    [[0.66, 0.24, 0.2452, 0.9, 0.2, 0.4, 0.9, 0.5]],
                    dtype=torch.float32,
                    device=device,
                ).expand(batch_size, -1),
                "initial_state": torch.tensor(
                    [[35.9, -78.8, 1000.0, 80.0, 2.0, -0.05, 66_000.0]],
                    dtype=torch.float32,
                    device=device,
                ).expand(batch_size, -1),
                "aero_params": torch.tensor(
                    [[122.6, 2.7, 0.02, 0.04, 0.9, 0.1]],
                    dtype=torch.float32,
                    device=device,
                ).expand(batch_size, -1),
                "control_lower": torch.tensor(
                    [[0.0, -0.7853982, 0.5]], dtype=torch.float32, device=device
                ).expand(batch_size, -1),
                "control_upper": torch.tensor(
                    [[240_000.0, 0.7853982, 2.0]], dtype=torch.float32, device=device
                ).expand(batch_size, -1),
                "frame_params": torch.tensor(
                    [[35.9, -78.8, 100.0, 0.0]], dtype=torch.float32, device=device
                ).expand(batch_size, -1),
            }
        optimizer.zero_grad()
        prediction = model_forward(model, x, dynamics)
        if uses_control_dynamics(config.prediction_output):
            prediction = map_control_candidates(
                prediction, _heterogeneous_control_probe_prediction
            )
        loss = prediction_loss(
            prediction,
            x[:, -1],
            target,
            state_weights,
            target_final_time_s,
            flight_weights,
            config,
            normalizer,
            dynamics,
        )
        loss.backward()
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
    # Control rollout depth can grow further as learned duration logits sharpen. Retain one
    # tested power-of-two as a memory margin after the heterogeneous probe; the state path
    # keeps its historical largest-successful behavior.
    selected = (
        successful[-2]
        if uses_control_dynamics(config.prediction_output) and len(successful) > 1
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
