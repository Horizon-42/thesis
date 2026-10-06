"""D149 (multi-aircraft control §6.1): stage C's code, generalised where one commanded aircraft becomes several, keeps
stage C's behaviour — on fixed synthetic windows of one commanded aircraft (the fixtures of `test_post_window_loop` and
`test_post_branches`), the same words, states, rewards, samples and branch groups, bit for bit (the CPU, one thread), as
stage C's code before the change: `BEFORE_GENERALISATION`, the digests that code wrote (dev-two-tier-v4 c7a0b6b0)."""

from __future__ import annotations

import hashlib
from dataclasses import astuple, replace

import numpy as np
import torch

from ts_transformer.experiments.prior_speaking_loop import flight_numbers
from ts_transformer.post.branches import samples
from ts_transformer.post.scene import moved_start_window
from ts_transformer.tests.test_post_branches import _ahead, _round
from ts_transformer.tests.test_post_window_loop import CPU, _window_loop, _with_module, setup  # noqa: F401


class _Digest:
    """A sha256 of output data (outline D139 item 5), fed arrays and values in order."""

    def __init__(self) -> None:
        self.hash = hashlib.sha256()

    def add(self, *values) -> None:
        for value in values:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                self.hash.update(str(value.dtype).encode() + str(value.shape).encode())
                self.hash.update(np.ascontiguousarray(value).tobytes())
            else:
                self.hash.update(repr(value).encode())

    def rows(self, rows) -> None:
        for name in ("time_s", "own", "candidates", "runway_in_force", "go_around", "heading_in_force",
                     "words_in_force", "since", "targets"):
            self.add(getattr(rows, name))

    def permitted(self, permitted) -> None:
        self.add(*permitted.masks, permitted.time_s, permitted.own, permitted.temperature)

    def sentence(self, rows, permitted, tokens) -> None:
        self.rows(rows)
        self.permitted(permitted)
        self.add(len(tokens), *tokens)

    def result(self, result) -> None:
        self.add(result.index, result.kind, result.outcome, None if result.loss is None else astuple(result.loss),
                 result.loss_step, result.other, result.reward, result.go_arounds, result.words, result.states,
                 result.speed_mask_rows, result.faulty_steps, result.loss_reads_fault)

    def samples(self, batch) -> None:
        self.add(*batch.rows, *batch.permitted.masks, batch.permitted.time_s, batch.permitted.own,
                 batch.traffic.tokens, batch.traffic.present, batch.advantage, batch.counted)


def stage_c_scenario(s) -> dict[str, str]:
    """Stage C on its synthetic windows (one commanded aircraft each): the traffic module's state as a checkpoint holds
    it (names, shapes, first values: a stage C checkpoint loads into it), the window loop of a real window, of window A
    (its own flight inserted ahead: lost at its first row) and of window B (its start moved), each with its samples; a
    branch round of window A and of the real window; the samples of window A's group with rewards that differ. Their
    digests, by part."""
    model = _with_module(s["base"])
    (window,) = s["windows"]
    moved = moved_start_window(window, np.random.default_rng(1), turn_deg=15.0, height_m=300.0, speed_scale=0.1)
    digest = _Digest()                    # the module's state as a stage C checkpoint holds it: names, shapes, values
    for name, value in _with_module(s["base"], weights=False).state_dict().items():
        digest.add(name, tuple(value.shape), value)
    out = {"module": digest.hash.hexdigest()}
    for name, chosen in (("real", window), ("A", _ahead(window)), ("B", moved)):
        digest = _Digest()
        loop = _window_loop(s, model, [chosen])
        for result in loop.run([flight_numbers(7, 0, 0)]):
            digest.result(result)
        for rows, permitted, tokens in loop.samples("train"):
            digest.sentence(rows, permitted, tokens)
        digest.add(loop.end_step(0))
        out[f"window {name}"] = digest.hash.hexdigest()
    for name, chosen in (("A", _ahead(window)), ("real", window)):
        digest = _Digest()
        round_ = _round(s, model, [chosen])
        for result in round_.first:
            digest.result(result)
        digest.add(round_.spoken_again, round_.differed, len(round_.groups))
        for group in round_.groups:
            digest.add(group.window, group.branch, group.informative)
            for sentence in group.sentences:
                digest.sentence(sentence.rows, sentence.permitted, sentence.tokens)
                digest.add(sentence.reward)
        out[f"round {name}"] = digest.hash.hexdigest()
        if name == "A":
            (group,) = round_.groups
            rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0),
                                                     *group.continuations[1:]))
            digest = _Digest()
            digest.samples(samples([rewarded], CPU))
            out["samples A"] = digest.hash.hexdigest()
    return out


#: `stage_c_scenario`'s digests, written by stage C's code before its generalisation (dev-two-tier-v4 c7a0b6b0, the CPU,
#: one thread).
BEFORE_GENERALISATION = {
    "module": "8ef2cf46f1b72b4b62813330bf8f53bb59819a99862dc087c036e978031b4122",
    "window real": "62e8e4e86314c20be46d9a70988b9cefbfb9a7cc9939bc6a7994c73a2a3b4026",
    "window A": "7c76b34704ab763318ce39232634b1315b276cc0c9cc94104f968c87bedd0444",
    "window B": "afe5951db940afbdd830714c5485771f9046e14c5d22bf852e4efd37deaba271",
    "round A": "1e7dadf27004b47b1582c4c4ea801a35cb775db75ce9224c1c8b25134ac77ed5",
    "samples A": "96c0207aaba55f9089327068ca64239f1db9df538865040ed679c45e32b08ab9",
    "round real": "cf8eacd236f3e55403ef464f884ab231a60e4166de8f6dabac17ed200f518172",
}


def test_stage_c_on_its_windows_is_the_code_before_its_generalisation_bit_for_bit(setup):
    """D149: the window loop, its samples, the branch rounds and the samples of a group, on windows of one commanded
    aircraft, are those of stage C's code before the change, bit for bit (their digests)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert stage_c_scenario(setup) == BEFORE_GENERALISATION
    finally:
        torch.set_num_threads(threads)
