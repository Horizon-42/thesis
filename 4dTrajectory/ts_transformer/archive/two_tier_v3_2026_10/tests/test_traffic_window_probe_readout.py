"""The go-around word's probability in an M4-in-windows run (`experiments/traffic_window_probe_readout`, R45): a round
spoken again is held to the stored sentences as R37 writes them, word for word; the probability read at the probe's step
is the one the tuner's cross-entropy learns; the readout's groups and counts."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _spoken():
    """Two sentences as `traffic_window_reward.write_sentences` reads them: rows and records with their words."""
    from ts_transformer.experiments.traffic_window_generation import WindowSentences

    grids = [np.arange(18, dtype=np.int64).reshape(3, 6), np.ones((4, 6), dtype=np.int64)]
    rows = [{"window": w, "dataset_id": f"KXXX:f{w}", "sample": s, "outcome": "landed", "runway": 0, "reward": r,
             "probed": p, "given": False, "forced": f, "said_steps": n, "counted": n, "kind": k,
             "starts_in_a_loss": False}
            for w, s, r, p, f, n, k in ((0, 1, 1.0, False, None, 3, "real"), (1, 5, 0.25, True, 2, 3, "A"))]
    return WindowSentences(rows, [SimpleNamespace(grid=g) for g in grids], [{}, {}]), np.array([0.5, -0.5]), \
        np.array([0.0, 0.75])


def _stored(tmp_path):
    """`_spoken` written as R37 writes a round (``sentences.npz`` by its own writer; ``sentences.json``'s aircraft
    fields the readout reads), read back by the readout."""
    from ts_transformer.experiments import traffic_window_probe_readout as probe
    from ts_transformer.experiments.traffic_window_reward import write_sentences

    spoken, advantages, gains = _spoken()
    write_sentences(tmp_path / "sentences.npz", spoken, advantages, gains)
    aircraft = [{name: r[name] for name in probe.FROM_JSON} for r in spoken.rows]
    npz = np.load(tmp_path / "sentences.npz")
    return npz, probe.stored_rows(npz, aircraft), probe.spoken_rows(spoken, advantages, gains)


def test_a_round_spoken_again_reads_as_its_stored_sentences(tmp_path):
    from ts_transformer.experiments import traffic_window_probe_readout as probe

    npz, stored, again = _stored(tmp_path)
    # every array R37 keeps is compared, or is the words
    assert set(npz.files) == set(probe.COMPARED) | {"step_offsets", "said"}
    probe.require_reproduced(1, stored, again)                  # the same: passes
    assert stored[1]["forced"] == 2 and stored[0]["forced"] == -1 and stored[1]["said"].shape == (3, 6)
    assert (stored[1]["counted"], stored[1]["kind"]) == (3, "A")
    with pytest.raises(ValueError, match="sentences.json holds 1 aircraft sentences, sentences.npz 2"):
        probe.stored_rows(npz, [{name: 0 for name in probe.FROM_JSON}])


@pytest.mark.parametrize("name", ["window", "dataset_id", "sample", "outcome", "runway", "reward", "probed", "given",
                                  "forced", "advantage", "probe_gain", "counted", "kind", "starts_in_a_loss"])
def test_any_field_spoken_otherwise_is_named(tmp_path, name):
    from ts_transformer.experiments import traffic_window_probe_readout as probe

    _, stored, again = _stored(tmp_path)
    assert name in (*probe.COMPARED, *probe.FROM_JSON)
    value = again[1][name]
    moved = [dict(again[0]), {**again[1], name: (not value) if isinstance(value, bool)
                              else value + 1 if isinstance(value, (int, float)) else f"{value}x"}]
    with pytest.raises(ValueError, match=rf"round 1, sentence 1 \(KXXX:f1, sample 5\): \['{name}'\] differ"):
        probe.require_reproduced(1, stored, moved)


def test_a_word_said_otherwise_or_a_sentence_missing_is_named(tmp_path):
    from ts_transformer.experiments import traffic_window_probe_readout as probe

    _, stored, again = _stored(tmp_path)
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


def test_each_probed_sentence_is_read_under_its_key_whatever_order_the_parts_score_in(tmp_path, monkeypatch):
    """`WindowRewardTuner.forced_log_probs` on four window samples probed at different steps (one not probed): a value
    under each probed sample's key, each the probability its sample alone gives the go-around there; an empty split is
    refused by name."""
    import copy
    import dataclasses

    from ts_transformer.experiments.traffic_window_tuner import WindowRewardTuner, WindowSplit
    from ts_transformer.instructions.words import APPROACH, APPROACH_GO_AROUND, Words
    from ts_transformer.prior.scene import N_LOOK
    from ts_transformer.prior.train import RewardConfig
    from ts_transformer.tests.test_prior_speaker import _model
    from ts_transformer.tests.test_traffic_window_tuner import CPU, _one_commanded_round

    model, spec, split, _, _, _ = _one_commanded_round(tmp_path, monkeypatch)
    steps = [3, -1, 4, 2]
    rebuilt, masks = [], []
    for s, step in enumerate(steps):
        flight, allows = split.flights[s][0], {c: m.copy() for c, m in split.allowed[s][0].items()}
        if step >= 0:
            asked = flight.asked.copy()
            asked[N_LOOK + step, APPROACH] = False
            flight = dataclasses.replace(flight, asked=asked)
            allows[APPROACH][step] = np.packbits(np.ones(model.config.classes[APPROACH], dtype=bool),
                                                 bitorder="little")
        rebuilt.append([flight])
        masks.append([allows])
    probed = WindowSplit(split.table, split.windows, rebuilt, split.records, split.trained, split.advantages, masks,
                         [[step] for step in steps], [np.ones(1)] * 4)
    tuner = WindowRewardTuner(copy.deepcopy(model), _model(Words(spec), slots=2), RewardConfig(), CPU, seed=0,
                              traffic_learning_rate=3e-4, step_s=spec.step_s)
    got = tuner.forced_log_probs(probed)
    assert set(got) == {(0, 0), (2, 0), (3, 0)}
    for s in (0, 2, 3):
        _, logits, _, _, _, _, _ = tuner._window_scored(probed, [s], None)
        alone = torch.log_softmax(logits[APPROACH][0, 0, N_LOOK + steps[s]], dim=-1)[APPROACH_GO_AROUND + 1]
        assert got[(s, 0)] == pytest.approx(float(alone), abs=1e-5)
    empty = WindowSplit(split.table, [], [], [], [], [], [], [], [])
    with pytest.raises(ValueError, match="no window sample to read the probes' go-arounds on"):
        tuner.forced_log_probs(empty)


def test_a_path_the_run_recorded_from_its_checkout_is_found_in_this_one():
    from ts_transformer.experiments.traffic_window_probe_readout import run_path
    from ts_transformer.repo_layout import REPO_ROOT

    recorded = "/elsewhere/.claude/worktrees/run/4dTrajectory/outputs/POOLED/prior/start/round_07"
    assert run_path(recorded) == REPO_ROOT / "4dTrajectory/outputs/POOLED/prior/start/round_07"

