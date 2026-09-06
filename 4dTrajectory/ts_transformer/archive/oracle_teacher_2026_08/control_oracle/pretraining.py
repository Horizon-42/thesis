"""Apply cached train-only teacher schedules before ordinary rollout training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_RECIPE_CUSTOM,
    CONTROL_RECIPE_NAMES,
    CONTROL_RECIPE_SIMPLE_V1,
    TSConfig,
)
from dataset import FixedAnchorTrajectoryWindows, FlightSeries, Normalizer
from control.oracle.evaluation import move_dynamics
from control.oracle.imitation import control_imitation_loss
from batch_contract import model_forward, unpack_batch


SIMPLE_V1_TEACHER_SCHEDULE_SHA256 = (
    "60e40e00e90b12880a76b68e1eaf28bdce74110ffc60c7d017d73ef54b973adc"
)
SIMPLE_V1_TEACHER_SCHEDULES = 32
SIMPLE_V1_TEACHER_SEGMENTS = 64
SIMPLE_V1_TEACHER_STEPS = 1000
SIMPLE_V1_TEACHER_LEARNING_RATE = 1e-4
SIMPLE_V1_TEACHER_GRADIENT_CLIP_NORM = 20.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_schedule(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"teacher schedule does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as source:
            dataset_ids = [str(value) for value in source["dataset_ids"].tolist()]
            controls = np.asarray(source["controls"], dtype=np.float32)
            durations = np.asarray(source["segment_durations_s"], dtype=np.float32)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid teacher schedule {path}: {exc}") from exc
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("teacher schedule dataset IDs must be unique")
    return dataset_ids, controls, durations


def validate_teacher_durations(
    target_durations: torch.Tensor,
    final_time: torch.Tensor,
    config: TSConfig,
) -> None:
    """Enforce the formal duration contract before any imitation optimizer step."""

    if not torch.allclose(
        target_durations.sum(dim=1), final_time, rtol=1e-5, atol=1e-3
    ):
        raise ValueError("teacher schedules do not match formal train final times")
    if config.control_duration_parameterization == CONTROL_DURATION_UNIFORM:
        expected_durations = final_time.unsqueeze(1) / int(config.n_segments)
        if not torch.allclose(
            target_durations,
            expected_durations.expand_as(target_durations),
            rtol=1e-5,
            atol=1e-3,
        ):
            raise ValueError(
                "uniform-duration control recipe requires uniform teacher durations"
            )


@dataclass(frozen=True)
class CachedSchedulePretrainer:
    """Imitate physical schedules using the formal outer-train normalizer."""

    schedule_path: Path
    steps: int = 1000
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 20.0
    log_every: int = 100
    recipe_name: str = CONTROL_RECIPE_CUSTOM

    def __post_init__(self) -> None:
        path = Path(self.schedule_path)
        object.__setattr__(self, "schedule_path", path)
        if self.recipe_name not in CONTROL_RECIPE_NAMES:
            raise ValueError(f"unknown teacher recipe {self.recipe_name!r}")
        if self.steps <= 0:
            raise ValueError("teacher pretraining steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("teacher pretraining learning rate must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("teacher pretraining gradient clip norm must be positive")
        dataset_ids, controls, durations = _load_schedule(path)
        if self.recipe_name != CONTROL_RECIPE_SIMPLE_V1:
            return
        if self.steps != SIMPLE_V1_TEACHER_STEPS:
            raise ValueError(
                f"simple-v1 teacher initialization requires "
                f"{SIMPLE_V1_TEACHER_STEPS} steps"
            )
        if not math.isclose(
            self.learning_rate,
            SIMPLE_V1_TEACHER_LEARNING_RATE,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "simple-v1 teacher initialization requires learning rate "
                f"{SIMPLE_V1_TEACHER_LEARNING_RATE:g}"
            )
        if not math.isclose(
            self.gradient_clip_norm,
            SIMPLE_V1_TEACHER_GRADIENT_CLIP_NORM,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "simple-v1 teacher initialization requires gradient clip norm "
                f"{SIMPLE_V1_TEACHER_GRADIENT_CLIP_NORM:g}"
            )
        schedule_sha256 = _sha256(path)
        if schedule_sha256 != SIMPLE_V1_TEACHER_SCHEDULE_SHA256:
            raise ValueError(
                "simple-v1 teacher initialization requires frozen schedule SHA-256 "
                f"{SIMPLE_V1_TEACHER_SCHEDULE_SHA256}; got {schedule_sha256}"
            )
        if len(dataset_ids) != SIMPLE_V1_TEACHER_SCHEDULES:
            raise ValueError(
                "simple-v1 teacher initialization requires exactly "
                f"{SIMPLE_V1_TEACHER_SCHEDULES} schedules"
            )
        expected_controls = (
            SIMPLE_V1_TEACHER_SCHEDULES,
            SIMPLE_V1_TEACHER_SEGMENTS,
            3,
        )
        expected_durations = (
            SIMPLE_V1_TEACHER_SCHEDULES,
            SIMPLE_V1_TEACHER_SEGMENTS,
        )
        if controls.shape != expected_controls:
            raise ValueError(
                f"simple-v1 teacher controls must have shape {expected_controls}"
            )
        if durations.shape != expected_durations:
            raise ValueError(
                f"simple-v1 teacher durations must have shape {expected_durations}"
            )

    def __call__(
        self,
        model: nn.Module,
        train_series: Sequence[FlightSeries],
        normalizer: Normalizer,
        config: TSConfig,
        device: torch.device,
    ) -> dict[str, object]:
        if config.control_recipe_name != self.recipe_name:
            raise ValueError(
                "teacher pretrainer recipe does not match the formal training config"
            )
        dataset_ids, controls_np, durations_np = _load_schedule(self.schedule_path)
        indexed = {item.dataset_id: item for item in train_series}
        missing = [key for key in dataset_ids if key not in indexed]
        if missing:
            raise ValueError(
                f"teacher schedule contains {len(missing)} non-train/missing flight(s); "
                f"first {missing[0]!r}"
            )
        cohort = [indexed[key] for key in dataset_ids]
        dataset = FixedAnchorTrajectoryWindows(cohort, config, normalizer)
        if len(dataset) != len(dataset_ids):
            raise ValueError("teacher pretraining cohort did not rebuild one fixed window each")
        x, _target, _weights, final_time, _flight_weights, dynamics, _supervision = (
            unpack_batch(dataset.batch(np.arange(len(dataset))))
        )
        x = x.to(device)
        final_time = final_time.to(device)
        dynamics = move_dynamics(dynamics, device)
        target_controls = torch.as_tensor(controls_np, device=device)
        target_durations = torch.as_tensor(durations_np, device=device)
        if target_controls.shape != (len(dataset), int(config.n_segments), 3):
            raise ValueError("teacher control schedule shape does not match formal config")
        if target_durations.shape != (len(dataset), int(config.n_segments)):
            raise ValueError("teacher duration schedule shape does not match formal config")
        validate_teacher_durations(target_durations, final_time, config)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        history: list[dict[str, float | int]] = []
        print(
            f"  pretrain   imitate {len(dataset)} outer-train oracle schedules for "
            f"{self.steps} steps"
        )
        for step in range(1, self.steps + 1):
            model.train()
            optimizer.zero_grad()
            prediction = model_forward(model, x, dynamics)
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
                model.parameters(), self.gradient_clip_norm
            )
            optimizer.step()
            if step == 1 or step % self.log_every == 0 or step == self.steps:
                row = {
                    "step": step,
                    "loss": float(loss.total.detach()),
                    "control": float(loss.control.detach()),
                    "duration_fraction": float(loss.duration_fraction.detach()),
                    "final_time": float(loss.final_time.detach()),
                    "gradient_norm": float(gradient_norm.detach()),
                }
                history.append(row)
                print(
                    f"             step {step:4d}: loss={row['loss']:.7f} "
                    f"control={row['control']:.7f} time={row['final_time']:.7f}"
                )
        model.eval()
        return {
            "schema_version": "ts-cached-oracle-teacher-pretraining-v1",
            "scope": "outer-train only",
            "recipe_name": self.recipe_name,
            "schedule_path": str(self.schedule_path.resolve()),
            "schedule_sha256": _sha256(self.schedule_path),
            "dataset_ids": dataset_ids,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "gradient_clip_norm": self.gradient_clip_norm,
            "loss": (
                "unit-box control MSE + N-scaled duration-fraction MSE + time/600 MSE"
            ),
            "history": history,
        }
