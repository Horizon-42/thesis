"""Contracts for the network-free direct-control oracle."""

from __future__ import annotations

from pathlib import Path
import copy
import json
import sys

import pytest
import numpy as np
import torch

_TS_DIR = Path(__file__).resolve().parents[1]
if str(_TS_DIR) not in sys.path:
    sys.path.insert(0, str(_TS_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from control_oracle import (  # noqa: E402
    ORACLE_DURATION_LEARNED,
    ORACLE_DURATION_UNIFORM,
    ORACLE_OBJECTIVE_ALL_STATE,
    ORACLE_OBJECTIVE_PHYSICAL_ADE,
    ORACLE_OBJECTIVE_PHYSICAL_CRITERIA,
    ORACLE_OBJECTIVE_POSITION_ONLY,
    DirectControlOracle,
    OracleObjective,
    fit_control_oracle,
    oracle_state_loss,
    smooth_maximum,
)
from control_oracle_initialization import (  # noqa: E402
    inverse_dynamics_controls,
    refine_piecewise_constant_schedule,
)
from control_oracle_curriculum import (  # noqa: E402
    build_horizon_curriculum,
    build_horizon_stage_view,
    stage_terminal_target,
    truncate_prediction,
    truncate_supervision,
)
from dataset import Normalizer  # noqa: E402
from fixed_dt_control_loss import FixedDTStateLossResult  # noqa: E402
from fixed_dt_supervision import FixedDTControlSupervision  # noqa: E402
from prediction_outputs import ControlPrediction  # noqa: E402
import run_ts_control_oracle as oracle_runner  # noqa: E402
from train_only_diagnostics import rank_outer_train_candidates  # noqa: E402


def _oracle(duration_mode: str, *, segments: int = 4) -> DirectControlOracle:
    return DirectControlOracle(
        n_segments=segments,
        control_lower=torch.tensor([0.0, -1.0, 0.5], dtype=torch.float64),
        control_upper=torch.tensor([100.0, 1.0, 2.0], dtype=torch.float64),
        total_duration_s=20.0,
        duration_mode=duration_mode,
    )


@pytest.mark.parametrize(
    "duration_mode", [ORACLE_DURATION_UNIFORM, ORACLE_DURATION_LEARNED]
)
def test_direct_control_oracle_maps_parameters_to_physical_contract(duration_mode):
    oracle = _oracle(duration_mode)
    prediction = oracle()

    assert prediction.controls.shape == (1, 4, 3)
    assert prediction.segment_durations.shape == (1, 4)
    assert prediction.final_time_s.tolist() == pytest.approx([20.0])
    assert prediction.segment_durations.sum().item() == pytest.approx(20.0)
    assert torch.all(prediction.controls >= oracle.control_lower)
    assert torch.all(prediction.controls <= oracle.control_upper)
    assert prediction.controls[0, 0].tolist() == pytest.approx([20.0, 0.0, 1.0])


def test_uniform_oracle_has_no_trainable_duration_parameters():
    oracle = _oracle(ORACLE_DURATION_UNIFORM)

    assert oracle.duration_logits is None
    assert dict(oracle.named_parameters()).keys() == {"control_logits"}
    assert torch.allclose(
        oracle().segment_durations,
        torch.full((1, 4), 5.0, dtype=torch.float64),
    )


def test_learned_oracle_duration_partition_is_differentiable():
    oracle = _oracle(ORACLE_DURATION_LEARNED)
    prediction = oracle()
    loss = prediction.controls[0, 0, 0] + prediction.segment_durations[0, 0]
    loss.backward()

    assert oracle.control_logits.grad is not None
    assert oracle.duration_logits is not None
    assert oracle.duration_logits.grad is not None
    assert torch.count_nonzero(oracle.duration_logits.grad).item() == 4


def test_direct_control_oracle_accepts_a_physical_warm_start():
    controls = torch.tensor(
        [[10.0, -0.2, 0.8], [30.0, 0.1, 1.2]], dtype=torch.float64
    )
    oracle = DirectControlOracle(
        n_segments=2,
        control_lower=torch.tensor([0.0, -1.0, 0.5], dtype=torch.float64),
        control_upper=torch.tensor([100.0, 1.0, 2.0], dtype=torch.float64),
        total_duration_s=20.0,
        duration_mode=ORACLE_DURATION_UNIFORM,
        initial_controls=controls,
    )

    assert torch.allclose(oracle().controls[0], controls)


def test_learned_oracle_accepts_a_nonuniform_duration_warm_start():
    durations = torch.tensor([2.0, 3.0, 5.0, 10.0], dtype=torch.float64)
    oracle = DirectControlOracle(
        n_segments=4,
        control_lower=torch.tensor([0.0, -1.0, 0.5], dtype=torch.float64),
        control_upper=torch.tensor([100.0, 1.0, 2.0], dtype=torch.float64),
        total_duration_s=20.0,
        duration_mode=ORACLE_DURATION_LEARNED,
        initial_segment_durations_s=durations,
    )

    assert torch.allclose(oracle().segment_durations[0], durations)


def test_inverse_dynamics_initializer_recovers_a_level_coordinated_turn():
    times = np.arange(0.0, 12.0, 2.0)
    speed = 80.0
    heading_rate = 0.01
    mass = 60_000.0
    states = np.column_stack(
        (
            np.zeros_like(times),
            np.zeros_like(times),
            np.full_like(times, 1_000.0),
            np.full_like(times, speed),
            heading_rate * times,
            np.zeros_like(times),
            np.full_like(times, mass),
        )
    )
    result = inverse_dynamics_controls(
        states,
        times,
        aero_params=np.array([122.6, 1.5, 0.02, 0.04, 0.8, 0.2]),
        control_lower=np.array([0.0, -np.pi / 4.0, 0.5]),
        control_upper=np.array([300_000.0, np.pi / 4.0, 2.0]),
        n_segments=4,
        total_duration_s=10.0,
    )
    lateral = heading_rate * speed / 9.81
    expected_load = np.hypot(lateral, 1.0)
    expected_bank = np.arctan2(lateral, 1.0)

    assert result.controls[:, 1] == pytest.approx([expected_bank] * 4)
    assert result.controls[:, 2] == pytest.approx([expected_load] * 4)
    assert np.all(result.controls[:, 0] > 0.0)
    assert result.clipped_fraction.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_schedule_refinement_preserves_piecewise_controls_and_total_duration():
    controls = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    durations = np.array([3.0, 5.0])

    refined_controls, refined_durations = refine_piecewise_constant_schedule(
        controls,
        durations,
        target_segments=4,
    )

    assert refined_controls.tolist() == [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0],
    ]
    assert refined_durations.tolist() == pytest.approx([1.5, 1.5, 2.5, 2.5])
    with pytest.raises(ValueError, match="integer source multiple"):
        refine_piecewise_constant_schedule(controls, durations, target_segments=5)


def test_adam_fitter_restores_the_best_direct_parameters():
    oracle = _oracle(ORACLE_DURATION_UNIFORM, segments=2)
    initial = oracle.control_logits.detach().square().mean().item()

    def objective() -> OracleObjective:
        state = oracle.control_logits.square().mean()
        zero = state.new_zeros(())
        return OracleObjective(total=state, state=state, terminal=zero)

    result = fit_control_oracle(
        oracle,
        objective,
        steps=80,
        control_learning_rate=0.1,
        duration_learning_rate=0.1,
        record_every=20,
    )
    restored = objective().total.item()

    assert result.best_step > 0
    assert restored == pytest.approx(result.best_objective["total"])
    assert restored < initial * 1e-2
    assert [row.step for row in result.history] == [0, 20, 40, 60, 80]


def test_adam_fitter_can_freeze_duration_logits():
    oracle = _oracle(ORACLE_DURATION_LEARNED, segments=2)
    initial_duration = oracle.duration_logits.detach().clone()

    def objective() -> OracleObjective:
        state = oracle.control_logits.square().mean()
        duration_term = (oracle.duration_logits - 2.0).square().mean()
        zero = state.new_zeros(())
        return OracleObjective(
            total=state + duration_term,
            state=state + duration_term,
            terminal=zero,
        )

    fit_control_oracle(
        oracle,
        objective,
        steps=20,
        control_learning_rate=0.1,
        duration_learning_rate=0.1,
        optimize_duration=False,
        record_every=10,
    )

    assert torch.equal(oracle.duration_logits, initial_duration)


def test_curriculum_crops_supervision_prediction_and_stage_target():
    supervision = FixedDTControlSupervision(
        query_offsets_s=torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float64),
        states=torch.arange(18, dtype=torch.float32).reshape(1, 3, 6),
        weights=torch.ones((1, 3, 6), dtype=torch.float32),
        valid=torch.tensor([[True, True, True]]),
    )
    cropped_supervision = truncate_supervision(supervision, 4.0)
    prediction = ControlPrediction(
        controls=torch.arange(9, dtype=torch.float64).reshape(1, 3, 3),
        segment_durations=torch.tensor([[3.0, 3.0, 4.0]], dtype=torch.float64),
        final_time_s=torch.tensor([10.0], dtype=torch.float64),
    )
    cropped_prediction = truncate_prediction(
        prediction, 5.0, detach_durations=True
    )

    assert cropped_supervision.valid.tolist() == [[True, True]]
    assert cropped_supervision.query_offsets_s.tolist() == [[2.0, 4.0]]
    assert torch.equal(stage_terminal_target(supervision, 4.0), supervision.states[:, 1])
    assert cropped_prediction.controls.shape == (1, 2, 3)
    assert torch.allclose(
        cropped_prediction.segment_durations,
        torch.tensor([[3.0, 2.0]], dtype=torch.float64),
    )
    assert cropped_prediction.final_time_s.tolist() == pytest.approx([5.0])
    assert not cropped_prediction.segment_durations.requires_grad


def test_horizon_curriculum_requires_fixed_dt_stages_and_full_terminal_stage():
    stages = build_horizon_curriculum(
        ["60", "120", "full"],
        [10, 20, 30],
        total_duration_s=386.5,
        supervision_dt_s=2.0,
    )

    assert [stage.horizon_s for stage in stages] == pytest.approx(
        [60.0, 120.0, 386.5]
    )
    assert [stage.steps for stage in stages] == [10, 20, 30]
    assert [stage.optimize_duration for stage in stages] == [False, False, True]
    frozen = build_horizon_curriculum(
        ["60", "full"],
        [10, 20],
        total_duration_s=386.5,
        supervision_dt_s=2.0,
        optimize_full_duration=False,
    )
    assert [stage.optimize_duration for stage in frozen] == [False, False]

    with pytest.raises(ValueError, match="fixed-dt"):
        build_horizon_curriculum(
            ["61", "full"],
            [10, 20],
            total_duration_s=386.5,
            supervision_dt_s=2.0,
        )
    with pytest.raises(ValueError, match="end with a full"):
        build_horizon_curriculum(
            ["60", "120"],
            [10, 20],
            total_duration_s=386.5,
            supervision_dt_s=2.0,
        )


def test_full_horizon_stage_view_preserves_original_prediction_and_targets():
    supervision = FixedDTControlSupervision(
        query_offsets_s=torch.tensor([[2.0, 4.0]], dtype=torch.float64),
        states=torch.zeros((1, 2, 6)),
        weights=torch.ones((1, 2, 6)),
        valid=torch.tensor([[True, True]]),
    )
    prediction = ControlPrediction(
        controls=torch.ones((1, 2, 3), dtype=torch.float64),
        segment_durations=torch.tensor([[2.0, 2.0]], dtype=torch.float64),
        final_time_s=torch.tensor([4.0], dtype=torch.float64),
    )
    target = torch.ones((1, 6))
    stage = build_horizon_curriculum(
        ["full"],
        [10],
        total_duration_s=4.0,
        supervision_dt_s=2.0,
    )[0]

    view = build_horizon_stage_view(prediction, supervision, target, stage)

    assert view.prediction is prediction
    assert view.supervision is supervision
    assert view.terminal_target is target


def test_oracle_position_objective_excludes_velocity_channels():
    rollout = FixedDTStateLossResult(
        per_flight_loss=torch.tensor([123.0], dtype=torch.float64),
        normalized_segment_end_states=torch.zeros((1, 1, 6), dtype=torch.float64),
        physical_query_states=torch.tensor(
            [[[10.0, 20.0, 30.0, 999.0, 999.0, 999.0]]],
            dtype=torch.float64,
        ),
    )
    supervision = FixedDTControlSupervision(
        query_offsets_s=torch.tensor([[2.0]], dtype=torch.float64),
        states=torch.zeros((1, 1, 6), dtype=torch.float32),
        weights=torch.ones((1, 1, 6), dtype=torch.float32),
        valid=torch.tensor([[True]]),
    )
    normalizer = Normalizer(
        mean=np.zeros(6, dtype=np.float64),
        std=np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]),
    )

    assert oracle_state_loss(
        rollout, supervision, normalizer, ORACLE_OBJECTIVE_ALL_STATE
    ).item() == pytest.approx(123.0)
    assert oracle_state_loss(
        rollout, supervision, normalizer, ORACLE_OBJECTIVE_POSITION_ONLY
    ).item() == pytest.approx((1.0 + 4.0 + 9.0) / 3.0)
    assert oracle_state_loss(
        rollout, supervision, normalizer, ORACLE_OBJECTIVE_PHYSICAL_ADE
    ).item() == pytest.approx(np.sqrt(10.0**2 + 20.0**2 + 30.0**2) / 1_000.0)
    assert oracle_state_loss(
        rollout, supervision, normalizer, ORACLE_OBJECTIVE_PHYSICAL_CRITERIA
    ).item() == pytest.approx(np.sqrt(10.0**2 + 20.0**2 + 30.0**2) / 1_000.0)


def test_physical_criteria_smooth_max_tracks_the_worse_normalized_metric():
    ade_ratio = torch.tensor(2.4, dtype=torch.float64)
    terminal_ratio = torch.tensor(2.7, dtype=torch.float64)

    value = smooth_maximum(ade_ratio, terminal_ratio, temperature=0.1)

    assert value.item() > terminal_ratio.item()
    assert value.item() < terminal_ratio.item() + 0.01


def test_oracle_runner_optimizer_seed_does_not_change_locked_outer_split():
    first = oracle_runner.build_oracle_config(
        n_segments=4,
        optimizer_seed=11,
        split_seed=1337,
        device="cpu",
    )
    second = oracle_runner.build_oracle_config(
        n_segments=4,
        optimizer_seed=99,
        split_seed=1337,
        device="cpu",
    )

    assert first.seed == 11
    assert second.seed == 99
    assert first.resolved_split_seed == second.resolved_split_seed == 1337
    keys = ["KSJC:a", "KSJC:b", "KSJC:c"]
    assert rank_outer_train_candidates(
        keys,
        ranking_namespace="oracle",
        split_seed=first.resolved_split_seed,
    ) == rank_outer_train_candidates(
        keys,
        ranking_namespace="oracle",
        split_seed=second.resolved_split_seed,
    )


def test_oracle_experiment_identity_covers_supported_run_dimensions():
    recipe = {
        "n_segments": 64,
        "duration_mode": "learned",
        "objective_mode": "all-state",
        "initialization": "neutral",
        "optimizer_seed": 1337,
        "split_seed": 1337,
        "steps": 100,
    }
    baseline = oracle_runner.oracle_experiment_identity(
        "KSJC:flight-a", recipe
    )
    variants = []
    for field, value in (
        ("duration_mode", "uniform"),
        ("objective_mode", "position-only"),
        ("initialization", "inverse-dynamics"),
        ("optimizer_seed", 2027),
    ):
        changed = copy.deepcopy(recipe)
        changed[field] = value
        variants.append(
            oracle_runner.oracle_experiment_identity("KSJC:flight-a", changed)
        )
    variants.append(
        oracle_runner.oracle_experiment_identity("KSJC:flight-b", recipe)
    )

    assert len({baseline, *variants}) == 1 + len(variants)


def test_oracle_report_writer_rejects_an_existing_directory(tmp_path):
    result = {
        "flight": {"dataset_id": "KSJC:flight-a"},
        "optimizer": {"n_segments": 2, "duration_mode": "uniform"},
        "config": {"dt_s": 2.0},
        "verdict": {"status": "diagnostic"},
        "best": {
            "restart": 0,
            "best_step": 1,
            "controls": [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            "segment_durations_s": [2.0, 2.0],
            "metrics": {
                "fixed_dt": {"ade_m": 10.0},
                "terminal": {"distance_3d_m": 20.0},
            },
        },
    }
    output = tmp_path / "oracle"
    oracle_runner._write_report(output, result)
    original = (output / "oracle_result.json").read_bytes()
    changed = copy.deepcopy(result)
    changed["best"]["controls"][0][0] = 999.0

    with pytest.raises(FileExistsError):
        oracle_runner._write_report(output, changed)

    assert (output / "oracle_result.json").read_bytes() == original
    assert json.loads(original)["best"]["controls"][0][0] == 1.0
