"""Apply cached train-only teacher schedules before ordinary rollout training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from config import TSConfig
from dataset import FixedAnchorTrajectoryWindows, FlightSeries, Normalizer
from oracle_teacher.evaluation import move_dynamics
from oracle_teacher.imitation import control_imitation_loss
from train import model_forward


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CachedSchedulePretrainer:
    """Imitate physical schedules using the formal outer-train normalizer."""

    schedule_path: Path
    steps: int = 1000
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
        with np.load(self.schedule_path, allow_pickle=False) as source:
            dataset_ids = [str(value) for value in source["dataset_ids"].tolist()]
            controls_np = np.asarray(source["controls"], dtype=np.float32)
            durations_np = np.asarray(source["segment_durations_s"], dtype=np.float32)
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
            dataset.batch(np.arange(len(dataset)))
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
        if not torch.allclose(
            target_durations.sum(dim=1), final_time, rtol=1e-5, atol=1e-3
        ):
            raise ValueError("teacher schedules do not match formal train final times")

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
