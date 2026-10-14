"""D149 (multi-aircraft control §6.1): stage C's code, generalised where one commanded aircraft becomes several, keeps
stage C's behaviour — on fixed synthetic windows of one commanded aircraft (the fixtures of `test_post_window_loop` and
`test_post_branches`), the same words, states, rewards, samples and branch groups, bit for bit (the CPU, one thread), as
stage C's code before the change: `BEFORE_GENERALISATION`, the digests that code wrote (dev-two-tier-v4 c7a0b6b0)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from ts_transformer.experiments.prior_speaking_loop import flight_numbers
from ts_transformer.experiments.window_behaviour import Digest as _Digest
from ts_transformer.post.branches import samples
from ts_transformer.post.scene import moved_start_window
from ts_transformer.tests.test_post_branches import _ahead, _round
from ts_transformer.tests.test_post_window_loop import CPU, _window_loop, _with_module, setup  # noqa: F401


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


def stage_c_batch_scenario(s, monkeypatch) -> str:
    """Stage C's window loop on a batch of two windows (copies of the real window, `WindowLoop.copy`), the first lost
    at its first row judged (the judge's pairs stubbed: a loss that aircraft 0 answers for, once) while the second
    flies on: the ends, the samples and the speaker's records of both rows — the rows that the speaker says of a window
    after its end included. Its digest."""
    from ts_transformer.inference.separation import Loss
    from ts_transformer.post import traffic

    calls = []
    real = traffic.losses

    def judged(*args):
        calls.append(1)
        return [Loss(0, 0, "in_trail", "none", 1.0, 0.5, 0.0, (0,), True)] if len(calls) == 1 else real(*args)

    monkeypatch.setattr(traffic, "losses", judged)
    loop = _window_loop(s, _with_module(s["base"]), s["windows"])
    while loop.speaking.observing:
        loop.observe()
    batch = loop.copy([0, 0])
    digest = _Digest()
    for result in batch.finish([np.random.default_rng(11), np.random.default_rng(12)]):
        digest.result(result)
    for rows, permitted, tokens in batch.samples("train"):
        digest.sentence(rows, permitted, tokens)
    speaker = batch.speaking.speaker
    digest.add(*(np.stack(rows) for rows in speaker.forbidden.values()), np.stack(speaker.drawn_probability),
               np.stack(speaker.go_around_probability))
    return digest.hash.hexdigest()


def stage_c_campaign_scenario(s, tmp_path, monkeypatch) -> dict[str, str]:
    """Stage C's campaign (post-training §9 item 11): (a) two rounds of its own draw, nothing stubbed; (b) two rounds whose
    speaking writes informative groups, so that the pass updates the model. Each: the digest of its round records (the
    commit, the time and the bytes of the groups written left out), its checkpoints' states and identities."""
    import json
    from dataclasses import asdict

    from ts_transformer.experiments import post_train
    from ts_transformer.post.scene import REAL
    from ts_transformer.tests.test_post_train import _context, _settings

    def digest_of(out):
        digest = _Digest()
        for r in range(2):
            record = json.loads((out / f"round_{r}" / "round.json").read_text())
            # the commit and the time, and the bytes of the groups written (information: a branch group of MC0 holds two
            # fields more, `Sentence.until` and `Group.varied`, so its pickle is a few bytes longer), are left out
            record.pop("git"), record.pop("finished_utc"), record.pop("groups_bytes")
            record.pop("speak_device")              # the workers' device, recorded after it (information, `--speak-device`)
            record.pop("cpu_workers")               # the CPU workers, recorded after it (information, D191)
            # the pass's records added after it (information: the gradients' norms, D168; each pass's means, D169)
            for key in ("grad_norm_mean", "grad_norm_max", "grad_clipped_share", "passes"):
                record["pass"].pop(key, None)
            digest.add(json.dumps(record, sort_keys=True))
            state = torch.load(out / f"round_{r}" / "checkpoint.pt", weights_only=False)
            digest.add(json.dumps(state["identity"], sort_keys=True))
            for name, value in state["model"].items():
                digest.add(name, value)
            digest.add(repr(state["optimizer"]["param_groups"]))
            for key, held in sorted(state["optimizer"]["state"].items()):
                for name, value in sorted(held.items()):
                    digest.add(key, name, torch.as_tensor(value))
        return digest.hash.hexdigest()

    out = {}
    settings = _settings(rounds=2, per_kind={**dict.fromkeys(post_train.KINDS, 0), REAL: 1}, continuations=2)
    drawn = tmp_path / "drawn"
    post_train.open_campaign(drawn, {"settings": asdict(settings)}, {"head": "x", "dirty": False}, {})
    post_train.run_campaign(drawn, settings, _context(s))
    out["campaign drawn"] = digest_of(drawn)
    settings = _settings(rounds=2, update_groups=1)
    model, _ = post_train.start_model(_context(s), settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))

    def speak(model, context, windows, settings, round_, directory, speakers, *, stage, launch):
        torch.save([rewarded, rewarded], directory / "groups_0.pt")
        return {"windows": len(windows)}

    monkeypatch.setattr(post_train, "speak_round", speak)
    monkeypatch.setattr(post_train, "draw_round", lambda context, per_kind, rng: ([_ahead(s["windows"][0])], {}))
    updated = tmp_path / "updated"
    post_train.open_campaign(updated, {"settings": asdict(settings)}, {"head": "x", "dirty": False}, {})
    post_train.run_campaign(updated, settings, _context(s))
    assert json.loads((updated / "round_1" / "round.json").read_text())["pass"]["updates"] == 2
    out["campaign updated"] = digest_of(updated)
    return out


#: `stage_c_campaign_scenario`'s digests, written by stage C's campaign before the stage skeleton (c7a0b6b0's
#: `run_campaign`, the CPU, one thread).
#: Written again on 2026-10-13 (two-tier2-eval-landing): the executor of `ts-executor-spec-v11` (223f1e30) moved every
#: flown part (the values before pass at 6c3cd8f6), and the closed-loop sentences of vocabulary D197–D199 (after
#: 04378395, by 0ac795e5) moved the parts that train. The judge of D196 (04378395) and the merge of dev-two-tier moved
#: none. A part that neither flies nor trains keeps c7a0b6b0's value. Since then the digests pin this code itself.
#: Written again on 2026-10-14 (two-tier2-glidepath): the closed-loop sentences of vocabulary D203 moved the part that
#: trains (the fixture's angle corrections now aim at the published glidepath); the drawn part did not move.
#: Written again on 2026-10-14 (two-tier2-glidepath, prior D204): the offset right of the centreline goes into the
#: prior on the glidepath's 100 m scale, so the synthetic prior says other words; both parts moved.
CAMPAIGN_BEFORE_GENERALISATION = {
    "campaign drawn": "52d88a8a08e049fcb3e06eb08a2f1a82b113e5ddaa0dd95708f100d1211c4408",
    "campaign updated": "30cf0c83b8f9d11985195494a055b1dc85b220fb6c0e471b3efa7d65221b7343",
}


def test_stage_cs_campaign_is_the_campaign_before_the_skeleton_bit_for_bit(setup, tmp_path, monkeypatch):
    """D149, post-training §9 item 11: stage C's campaign through the round's skeleton (`Stage`, `STAGE_C`) writes the
    records, checkpoints and identities that its campaign wrote before, bit for bit (a draw not stubbed, a pass that
    updates)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert stage_c_campaign_scenario(setup, tmp_path, monkeypatch) == CAMPAIGN_BEFORE_GENERALISATION
    finally:
        torch.set_num_threads(threads)


#: `stage_c_scenario`'s digests, written by stage C's code before its generalisation (dev-two-tier-v4 c7a0b6b0, the CPU,
#: one thread). "round real" written again by d5cc5d6a with each flight's own permitted record alone (S6, multi-aircraft
#: control D193 (3), requests_from_d item 99): its sentences' records are no longer padded to the most rows of those
#: asked; the other digests stay.
#: Written again on 2026-10-13 (two-tier2-eval-landing): the executor of `ts-executor-spec-v11` (223f1e30) moved every
#: flown part (the values before pass at 6c3cd8f6), and the closed-loop sentences of vocabulary D197–D199 (after
#: 04378395, by 0ac795e5) moved the parts that train. The judge of D196 (04378395) and the merge of dev-two-tier moved
#: none. A part that neither flies nor trains keeps c7a0b6b0's value. Since then the digests pin this code itself.
#: Written again on 2026-10-14 (two-tier2-glidepath, prior D204): the offset right of the centreline goes into the
#: prior on the glidepath's 100 m scale, so the synthetic prior says other words; every part it says moved; the module did not.
BEFORE_GENERALISATION = {
    "module": "8ef2cf46f1b72b4b62813330bf8f53bb59819a99862dc087c036e978031b4122",
    "window real": "51a76e943fc78b37ed8494e555a4ccbe3eb6824acd10171bc4263c4328031871",
    "window A": "db8c881aec3303f5b42282df00157e64af9db8968782f97fb32e8c980654e15f",
    "window B": "e011f2ebb5fba68810008cb554f04bd0e765a8f89950dc88f2cead2d050e8a74",
    "round A": "4220191b3a03d85378742084a2a9eb4cbe10f7078650fb8dee7438b21f0298ef",
    "samples A": "10686a47c3d94fcedf19b8fab84471f9cddb869c362fdf79c4126b24e89488d6",
    "round real": "02f49b07a745d969acf83caae4f4b8f50b95cd2b5b14508e13e615a2eac5b1fa",
}


#: `stage_c_batch_scenario`'s digest, written by the same code, and again as "round real" (S6: its two rows' records each
#: their own).
#: Written again on 2026-10-13 (two-tier2-eval-landing): the executor of `ts-executor-spec-v11` (223f1e30) moved every
#: flown part (the values before pass at 6c3cd8f6), and the closed-loop sentences of vocabulary D197–D199 (after
#: 04378395, by 0ac795e5) moved the parts that train. The judge of D196 (04378395) and the merge of dev-two-tier moved
#: none. A part that neither flies nor trains keeps c7a0b6b0's value. Since then the digests pin this code itself.
#: Written again on 2026-10-14 (two-tier2-glidepath, prior D204): the offset right of the centreline goes into the
#: prior on the glidepath's 100 m scale, so the synthetic prior says other words; every part it says moved.
BATCH_BEFORE_GENERALISATION = "3727c7d07861c6c72a368b7e734d0db70169e12ffeb5bf61756590c02650199e"


def test_a_stage_c_batch_whose_first_window_ended_is_the_code_before_its_generalisation_bit_for_bit(setup, monkeypatch):
    """D149: a window that ended in a batch still flown is said on as before (its rows after its end, which are not its
    sentence's, included), and the window flown on is unchanged — bit for bit (the digest)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert stage_c_batch_scenario(setup, monkeypatch) == BATCH_BEFORE_GENERALISATION
    finally:
        torch.set_num_threads(threads)


def test_stage_c_on_its_windows_is_the_code_before_its_generalisation_bit_for_bit(setup):
    """D149: the window loop, its samples, the branch rounds and the samples of a group, on windows of one commanded
    aircraft, are those of stage C's code before the change, bit for bit (their digests)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert stage_c_scenario(setup) == BEFORE_GENERALISATION
    finally:
        torch.set_num_threads(threads)
