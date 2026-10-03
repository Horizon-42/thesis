"""The go-around word's probability in an M4-in-windows run (`experiments/traffic_window_probe_readout`, R45): a round
spoken again is held to the stored sentences as R37 writes them, word for word; the probability read at the probe's step
is the one the tuner's cross-entropy learns; the readout's groups and counts."""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _spoken():
    """Two sentences as `traffic_window_reward.write_sentences` reads them: rows and records with their words."""
    from ts_transformer.experiments.traffic_window_generation import WindowSentences

    grids = [np.arange(18, dtype=np.int64).reshape(3, 6), np.ones((4, 6), dtype=np.int64)]
    rows = [{"window": w, "dataset_id": f"KXXX:f{w}", "sample": s, "outcome": "landed", "runway": 0, "reward": r,
             "probed": p, "given": False, "forced": f, "said_steps": n}
            for w, s, r, p, f, n in ((0, 1, 1.0, False, None, 3), (1, 5, 0.25, True, 2, 3))]
    return WindowSentences(rows, [SimpleNamespace(grid=g) for g in grids], [{}, {}]), np.array([0.5, -0.5]), \
        np.array([0.0, 0.75])


def test_a_round_spoken_again_reads_as_its_stored_sentences_and_any_difference_is_named(tmp_path):
    from ts_transformer.experiments import traffic_window_probe_readout as probe
    from ts_transformer.experiments.traffic_window_reward import write_sentences

    spoken, advantages, gains = _spoken()
    write_sentences(tmp_path / "sentences.npz", spoken, advantages, gains)
    stored = probe.stored_rows(np.load(tmp_path / "sentences.npz"))
    again = probe.spoken_rows(spoken, advantages, gains)
    probe.require_reproduced(1, stored, again)                  # the same: passes
    assert stored[1]["forced"] == 2 and stored[0]["forced"] == -1 and stored[1]["said"].shape == (3, 6)
    for change, named in (({"reward": 0.5}, "['reward']"), ({"forced": 1}, "['forced']"),
                          ({"probe_gain": 0.0}, "['probe_gain']")):
        moved = [dict(again[0]), {**again[1], **change}]
        with pytest.raises(ValueError, match=rf"round 1, sentence 1 \(KXXX:f1, sample 5\): {re.escape(named)} differ"):
            probe.require_reproduced(1, stored, moved)
    words = [dict(again[0]), {**again[1], "said": again[1]["said"].copy()}]
    words[1]["said"][2, 3] += 1
    with pytest.raises(ValueError, match=r"\['said'\] differ"):
        probe.require_reproduced(1, stored, words)
    with pytest.raises(ValueError, match="2 sentences stored, 1 spoken again"):
        probe.require_reproduced(1, stored, again[:1])


def test_the_probability_read_is_the_one_the_probe_s_cross_entropy_learns():
    from ts_transformer.experiments.traffic_window_tuner import forced_go_around_log_p
    from ts_transformer.instructions.words import APPROACH_GO_AROUND
    from ts_transformer.prior.scene import N_LOOK

    torch.manual_seed(0)
    approach = torch.randn(3, 1, N_LOOK + 6, 5)
    sentences, forced = np.array([0, 2]), np.array([1, 4])
    got = forced_go_around_log_p(approach, sentences, forced)
    expected = [torch.log_softmax(approach[i, 0, N_LOOK + f], dim=-1)[APPROACH_GO_AROUND + 1]
                for i, f in zip(sentences, forced)]
    assert torch.allclose(got, torch.stack(expected))


def test_the_readout_groups_learned_and_other_sentences_and_counts_the_rises_against_round_0():
    from ts_transformer.experiments.traffic_window_probe_readout import readout, summarise

    def row(round_number, learned, *log_p):
        return {"round": round_number, "learned": learned,
                "log_p": dict(zip(("round_00", "round_01", "round_02"), log_p))}

    rows = [row(1, True, -6.0, -5.0, -4.0), row(1, False, -6.0, -7.0, -6.0), row(2, True, -3.0, -3.0, -2.0)]
    got = readout(rows, ["round_00", "round_01", "round_02"])
    first = got["round_01_sentences"]
    assert (first["sentences"], first["learned"]) == (2, 1)
    assert first["round_01"]["learned"]["rose_against_round_0"] == 1
    assert first["round_01"]["not_learned"]["rose_against_round_0"] == 0
    assert first["round_02"]["all"]["rose_against_round_0"] == 1
    assert first["round_02"]["learned"]["median"] == pytest.approx(math.exp(-4.0))
    assert got["round_02_sentences"]["round_00"]["all"]["rose_against_round_0"] == 0
    assert summarise([]) == {"n": 0}


def test_a_path_the_run_recorded_from_its_checkout_is_found_in_this_one():
    from ts_transformer.experiments.traffic_window_probe_readout import run_path
    from ts_transformer.repo_layout import REPO_ROOT

    recorded = "/elsewhere/.claude/worktrees/run/4dTrajectory/outputs/POOLED/prior/start/round_07"
    assert run_path(recorded) == REPO_ROOT / "4dTrajectory/outputs/POOLED/prior/start/round_07"

