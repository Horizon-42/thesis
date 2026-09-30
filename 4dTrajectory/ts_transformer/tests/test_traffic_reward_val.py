"""M4's validation readout (`experiments/traffic_reward_val`): the run is read by content, and on its own select split
each model's sentences are checked against the run's own readouts."""

from __future__ import annotations

import pytest


def _row(key, kind="real", sample=0, outcome="landed", reward=1.0, steps=40, starts=False):
    return {"dataset_id": key, "kind": kind, "sample": sample, "outcome": outcome, "reward": reward, "steps_said": steps,
            "starts_in_a_loss": starts}


def test_the_run_is_read_by_content(monkeypatch):
    from ts_transformer.experiments import traffic_reward_val as runner

    monkeypatch.setattr(runner, "edge_source_sha256", lambda: "e" * 64)
    config = {"prior": {"checkpoint_sha256": "p" * 64}, "executor": {"sha256": "x" * 64}, "edge_source_sha256": "e" * 64,
              "instructions": "/somewhere/.claude/worktrees/gone/4dTrajectory/outputs/POOLED/instruction_language/v5"}
    here = runner.REPO_ROOT / "4dTrajectory/outputs/POOLED/instruction_language/v5"
    assert runner.run_differences(config, "p" * 64, "x" * 64, here) == []
    other = runner.REPO_ROOT / "4dTrajectory/outputs/POOLED/instruction_language/v4"
    assert runner.run_differences(config, "q" * 64, "x" * 64, other) == ["the start's checkpoint",
                                                                         "the instruction artefact"]
    monkeypatch.setattr(runner, "edge_source_sha256", lambda: "f" * 64)
    assert runner.run_differences(config, "p" * 64, "x" * 64, here) == ["the edge code"]


def test_the_sentences_are_checked_against_the_run_s_and_the_reward_reads_those_it_counts():
    from ts_transformer.experiments.traffic_reward_val import rewards, said_alike

    run = [_row("a"), _row("a", sample=1, outcome="lost_separation", reward=0.0), _row("b", starts=True, reward=0.0)]
    ours = [_row("a"), _row("a", sample=1), _row("b", starts=True, reward=0.0)]
    assert said_alike(ours, run) == {"sentences": 3, "alike": 2}
    assert rewards(run) == {("a", "real", 0): 1.0, ("a", "real", 1): 0.0}
    with pytest.raises(SystemExit, match="not the run's"):
        said_alike(ours[:2], run)
