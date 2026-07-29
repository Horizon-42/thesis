from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
for path in (REPO_ROOT, TS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_ts_control_capacity_ceiling as ceiling  # noqa: E402


def test_balanced_key_sample_is_deterministic_and_airport_balanced() -> None:
    keys = [f"KAAA:f{index}" for index in range(5)] + [
        f"KBBB:f{index}" for index in range(5)
    ]

    selected = ceiling.balanced_key_sample(
        keys, per_airport=2, seed=17, split="val"
    )

    assert selected == ceiling.balanced_key_sample(
        list(reversed(keys)), per_airport=2, seed=17, split="val"
    )
    assert sum(key.startswith("KAAA:") for key in selected) == 2
    assert sum(key.startswith("KBBB:") for key in selected) == 2


def test_oracle_prediction_preserves_bounds_and_true_total_time() -> None:
    logits = torch.zeros(2, 3, 3)
    duration_logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    true_total = torch.tensor([12.0, 6.0])
    lower = torch.tensor([[0.0, -1.0, 0.5], [10.0, -0.5, 0.75]])
    upper = torch.tensor([[100.0, 1.0, 2.0], [50.0, 0.5, 1.75]])

    prediction = ceiling.oracle_prediction(
        logits, duration_logits, true_total, lower, upper
    )

    assert torch.all(prediction.controls >= lower.unsqueeze(1))
    assert torch.all(prediction.controls <= upper.unsqueeze(1))
    torch.testing.assert_close(prediction.final_time_s, true_total)
    assert torch.all(prediction.segment_durations > 0.0)


def test_uniform_oracle_uses_equal_positive_segments() -> None:
    logits = torch.zeros(1, 4, 3)
    total = torch.tensor([8.0])
    lower = torch.zeros(1, 3)
    upper = torch.ones(1, 3)

    prediction = ceiling.oracle_prediction(logits, None, total, lower, upper)

    torch.testing.assert_close(
        prediction.segment_durations, torch.full((1, 4), 2.0)
    )


def test_balanced_key_sample_rejects_unqualified_identity() -> None:
    with pytest.raises(ValueError, match="airport-qualified"):
        ceiling.balanced_key_sample(["flight"], per_airport=1, seed=1, split="train")
