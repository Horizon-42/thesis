from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
for path in (REPO_ROOT, TS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_ts_control_capacity_ceiling as ceiling  # noqa: E402
from config import TSConfig  # noqa: E402


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


def test_nonfinite_gradient_diagnostics_reports_affected_rows() -> None:
    first = torch.nn.Parameter(torch.zeros((3, 2)))
    second = torch.nn.Parameter(torch.zeros((3, 2)))
    first.grad = torch.tensor([
        [0.0, 1.0],
        [float("nan"), 0.0],
        [0.0, float("inf")],
    ])
    second.grad = torch.ones_like(second)

    assert ceiling.nonfinite_gradient_diagnostics([first, second]) == [{
        "parameter_index": 0,
        "nonfinite_values": 2,
        "affected_batch_rows": [1, 2],
    }]


def test_float64_gradient_clipping_handles_finite_float32_overflow() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.full_like(parameter, 1e30)

    norm = ceiling.clip_grad_norm_float64_([parameter], max_norm=10.0)

    assert norm == pytest.approx(2e30)
    assert torch.isfinite(parameter.grad).all()
    assert torch.linalg.vector_norm(parameter.grad) == pytest.approx(10.0)


def test_balanced_key_sample_rejects_unqualified_identity() -> None:
    with pytest.raises(ValueError, match="airport-qualified"):
        ceiling.balanced_key_sample(["flight"], per_airport=1, seed=1, split="train")


def test_capacity_runner_accepts_historical_six_field_control_batch() -> None:
    raw = (
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, 3),
        torch.ones(1),
        torch.ones(1),
        {"control_lower": torch.zeros(1, 3)},
    )

    prepared = ceiling.prepare_capacity_batch(raw, torch.device("cpu"))

    assert len(prepared) == 7
    assert prepared[-1] is None
    assert prepared[-2]["control_lower"].device.type == "cpu"


def test_oracle_optimizer_carries_dense_supervision_into_objective(monkeypatch) -> None:
    captured = []

    def fake_components(
        prediction,
        _anchor,
        _target,
        _mask,
        _final_time,
        _flight_weights,
        _config,
        _normalizer,
        _dynamics,
        dense_supervision,
    ):
        captured.append(dense_supervision)
        state = prediction.controls.square().mean()
        return SimpleNamespace(total=state, state=state, terminal=state.new_zeros(()))

    monkeypatch.setattr(ceiling, "prediction_loss_components", fake_components)
    dense = object()
    lower = torch.tensor([[0.0, -1.0, 0.5]])
    upper = torch.tensor([[100.0, 1.0, 2.0]])
    initial = ceiling.oracle_prediction(
        torch.zeros(1, 2, 3), None, torch.tensor([4.0]), lower, upper
    )
    dynamics = {"control_lower": lower, "control_upper": upper}

    ceiling.optimize_oracle(
        "uniform_partition",
        initial,
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, 3),
        torch.tensor([4.0]),
        torch.ones(1),
        dynamics,
        TSConfig(),
        SimpleNamespace(),
        dense,
        steps=1,
        learning_rate=1e-3,
        max_grad_norm=10.0,
    )

    assert captured == [dense]
