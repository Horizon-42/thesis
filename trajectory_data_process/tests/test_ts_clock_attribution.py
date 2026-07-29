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

import run_ts_clock_attribution as attribution  # noqa: E402


def test_duration_variants_separate_total_time_and_partition() -> None:
    predicted = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
    truth = torch.tensor([8.0, 2.0])

    variants = attribution.duration_variants(predicted, truth)

    assert tuple(variants) == attribution.VARIANT_LABELS
    torch.testing.assert_close(
        variants["predicted_clock"], torch.tensor([[1.0, 3.0], [2.0, 2.0]])
    )
    torch.testing.assert_close(
        variants["true_total_timewarp_only"], torch.tensor([[2.0, 6.0], [1.0, 1.0]])
    )
    torch.testing.assert_close(
        variants["true_total_learned_partition_rerollout"],
        torch.tensor([[2.0, 6.0], [1.0, 1.0]]),
    )
    torch.testing.assert_close(
        variants["predicted_total_uniform_partition_rerollout"],
        torch.tensor([[2.0, 2.0], [2.0, 2.0]]),
    )
    torch.testing.assert_close(
        variants["true_total_uniform_partition_rerollout"],
        torch.tensor([[4.0, 4.0], [1.0, 1.0]]),
    )


@pytest.mark.parametrize(
    ("predicted", "truth"),
    [
        (torch.tensor([[0.0, 1.0]]), torch.tensor([2.0])),
        (torch.tensor([[1.0, 1.0]]), torch.tensor([0.0])),
    ],
)
def test_duration_variants_reject_nonpositive_clocks(
    predicted: torch.Tensor, truth: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        attribution.duration_variants(predicted, truth)
