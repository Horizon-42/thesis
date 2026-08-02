"""Progressively refine a train-only cached teacher from coarse to fine controls."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from config import (
    CONTROL_DURATION_FACTORIZED,
    CONTROL_VALUE_ABSOLUTE,
    PREDICTION_CONTROL,
    TSConfig,
)
from dataset import FixedAnchorTrajectoryWindows, FlightSeries, Normalizer
from models import build_model
from oracle_teacher.evaluation import move_dynamics
from oracle_teacher.imitation import control_imitation_loss
from train import model_forward


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coarsen_schedule(
    controls: torch.Tensor,
    durations_s: torch.Tensor,
    target_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Duration-weight adjacent controls while preserving total physical time."""
    source_segments = controls.shape[1]
    if source_segments % target_segments != 0:
        raise ValueError("teacher segment count must be divisible by target segments")
    group = source_segments // target_segments
    grouped_duration = durations_s.view(len(controls), target_segments, group)
    grouped_controls = controls.view(
        len(controls), target_segments, group, controls.shape[-1]
    )
    coarse_duration = grouped_duration.sum(dim=2)
    coarse_controls = (
        grouped_controls * grouped_duration.unsqueeze(-1)
    ).sum(dim=2) / coarse_duration.unsqueeze(-1)
    return coarse_controls, coarse_duration


def _copy_shared_parameters(source: nn.Module, target: nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name].copy_(value)
    target.load_state_dict(target_state)


def refine_control_model(source: nn.Module, target: nn.Module) -> None:
    """Copy a coarse model and duplicate each control interval exactly."""
    _copy_shared_parameters(source, target)
    coarse = source.control_head
    fine = target.control_head
    if fine.n_segments % coarse.n_segments != 0:
        raise ValueError("fine control head must be an integer refinement")
    ratio = fine.n_segments // coarse.n_segments
    with torch.no_grad():
        coarse_control_weight = coarse.control_projection.weight.view(
            coarse.n_segments, 3, -1
        )
        coarse_control_bias = coarse.control_projection.bias.view(
            coarse.n_segments, 3
        )
        fine.control_projection.weight.copy_(
            coarse_control_weight.repeat_interleave(ratio, dim=0).reshape_as(
                fine.control_projection.weight
            )
        )
        fine.control_projection.bias.copy_(
            coarse_control_bias.repeat_interleave(ratio, dim=0).reshape_as(
                fine.control_projection.bias
            )
        )
        fine.duration_projection.weight.copy_(
            coarse.duration_projection.weight.repeat_interleave(ratio, dim=0)
        )
        fine.duration_projection.bias.copy_(
            coarse.duration_projection.bias.repeat_interleave(ratio, dim=0)
        )


@dataclass(frozen=True)
class ProgressiveSchedulePretrainer:
    """Imitate N=16 -> 32 -> 64 with an exact head refinement between stages."""

    schedule_path: Path
    stages: tuple[int, ...] = (16, 32, 64)
    steps_per_stage: tuple[int, ...] = (300, 300, 400)
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 20.0
    log_every: int = 100

    def __call__(
        self,
        model: nn.Module,
        train_series: Sequence[FlightSeries],
        normalizer: Normalizer,
        config: TSConfig,
        device: torch.device,
    ) -> dict[str, object]:
        if (
            config.prediction_output != PREDICTION_CONTROL
            or config.control_duration_parameterization != CONTROL_DURATION_FACTORIZED
            or config.control_value_parameterization != CONTROL_VALUE_ABSOLUTE
        ):
            raise ValueError("progressive teacher requires factorized absolute control")
        if len(self.stages) != len(self.steps_per_stage):
            raise ValueError("one optimization-step count is required per stage")
        if self.stages[-1] != int(config.n_segments):
            raise ValueError("last progressive stage must match formal n_segments")
        if any(fine % coarse for coarse, fine in zip(self.stages, self.stages[1:])):
            raise ValueError("each progressive stage must refine its predecessor")

        with np.load(self.schedule_path, allow_pickle=False) as source:
            dataset_ids = [str(value) for value in source["dataset_ids"].tolist()]
            full_controls = torch.as_tensor(
                np.asarray(source["controls"], dtype=np.float32), device=device
            )
            full_durations = torch.as_tensor(
                np.asarray(source["segment_durations_s"], dtype=np.float32),
                device=device,
            )
        indexed = {item.dataset_id: item for item in train_series}
        missing = [key for key in dataset_ids if key not in indexed]
        if missing:
            raise ValueError(
                f"teacher schedule contains non-train/missing flight {missing[0]!r}"
            )
        cohort = [indexed[key] for key in dataset_ids]

        stage_model: nn.Module | None = None
        stage_history: list[dict[str, object]] = []
        for stage_segments, stage_steps in zip(self.stages, self.steps_per_stage):
            stage_config = replace(config, n_segments=stage_segments)
            dataset = FixedAnchorTrajectoryWindows(cohort, stage_config, normalizer)
            x, _target, _weights, final_time, _flight_weights, dynamics, _supervision = (
                dataset.batch(np.arange(len(dataset)))
            )
            x = x.to(device)
            final_time = final_time.to(device)
            dynamics = move_dynamics(dynamics, device)
            target_controls, target_durations = coarsen_schedule(
                full_controls, full_durations, stage_segments
            )

            next_model = build_model(stage_config).to(device)
            if stage_model is None:
                _copy_shared_parameters(model, next_model)
            else:
                refine_control_model(stage_model, next_model)
            stage_model = next_model
            optimizer = torch.optim.Adam(
                stage_model.parameters(), lr=self.learning_rate
            )
            rows: list[dict[str, float | int]] = []
            print(
                f"  progressive teacher N={stage_segments}: "
                f"{stage_steps} imitation steps"
            )
            for step in range(1, stage_steps + 1):
                stage_model.train()
                optimizer.zero_grad()
                prediction = model_forward(stage_model, x, dynamics)
                loss = control_imitation_loss(
                    prediction,
                    target_controls,
                    target_durations,
                    final_time,
                    dynamics["control_lower"],
                    dynamics["control_upper"],
                    final_time_scale_s=config.final_time_scale_s,
                )
                loss.total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    stage_model.parameters(), self.gradient_clip_norm
                )
                optimizer.step()
                if step == 1 or step % self.log_every == 0 or step == stage_steps:
                    row = {
                        "step": step,
                        "loss": float(loss.total.detach()),
                        "control": float(loss.control.detach()),
                        "duration_fraction": float(
                            loss.duration_fraction.detach()
                        ),
                        "final_time": float(loss.final_time.detach()),
                        "gradient_norm": float(gradient_norm.detach()),
                    }
                    rows.append(row)
                    print(
                        f"             step {step:4d}: loss={row['loss']:.7f} "
                        f"control={row['control']:.7f} time={row['final_time']:.7f}"
                    )
            stage_history.append(
                {
                    "n_segments": stage_segments,
                    "steps": stage_steps,
                    "history": rows,
                }
            )

        assert stage_model is not None
        model.load_state_dict(stage_model.state_dict())
        model.eval()
        return {
            "schema_version": "ts-progressive-oracle-teacher-pretraining-v1",
            "scope": "outer-train only",
            "schedule_path": str(self.schedule_path.resolve()),
            "schedule_sha256": _sha256(self.schedule_path),
            "dataset_ids": dataset_ids,
            "stages": stage_history,
            "total_steps": sum(self.steps_per_stage),
            "learning_rate": self.learning_rate,
            "gradient_clip_norm": self.gradient_clip_norm,
            "refinement": "duplicate controls and split softmax duration exactly",
            "loss": (
                "unit-box control MSE + N-scaled duration-fraction MSE + time/600 MSE"
            ),
        }
