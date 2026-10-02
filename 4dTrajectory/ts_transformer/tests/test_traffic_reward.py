"""Multi-aircraft M4's runner (`experiments/traffic_reward`, design §6.6 step 6): its sentences, cut at their judged ends
and read back as training rows in their scenes, give back the distribution each word was sampled from — the others off
the steps, a moved leader, the landing context; a scene starting in a loss is not trained on; the ordering and the traffic
attention's readouts; the landing gap reads the landing before on the runway; a round refuses to run short of augmented
scenes; a traffic prior it writes is read back by `load_prior`; the guards exclude a round by name and the choice takes the
earliest within the tie. On the scene-data fixture (a tmp artefact)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from ts_transformer.instructions.words import Words
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.test_autopilot import _params, _physics
from ts_transformer.tests.test_prior_speaker import _repeated
from ts_transformer.tests.test_traffic_speaking import LIMIT_S, _batch
from ts_transformer.tests.test_traffic_tuner import _airport, _split, _traffic_model

CPU = torch.device("cpu")


def _patch_physics(monkeypatch, airport, signals, keys, samples):
    """The runner's executor inputs for the flights at ``keys``: the test aircraft's, looked up by the flights each loop
    batch holds (`_batch` puts a flight's key where its series would be)."""
    from ts_transformer.experiments import traffic_free_generation, traffic_reward

    physics = {}
    for key in keys:
        flight = signals[key]
        start = replace(flight, **{name: getattr(flight, name)[N_LOOK:] for name in
                                   ("time_s", "e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps",
                                    "vertical_rate_mps")})
        physics[key] = _physics(start, airport.flights.geometry)

    def stacked(order, k):
        tables = [physics[key][k] for key in order]
        if isinstance(tables[0], torch.Tensor):
            return torch.cat(tables)
        return type(tables[0])(**{f.name: torch.cat([getattr(t, f.name) for t in tables])
                                  for f in dataclasses.fields(tables[0])})

    monkeypatch.setattr(traffic_free_generation, "flight_inputs", lambda series, device, anchor: stacked(series, 0))
    monkeypatch.setattr(traffic_free_generation, "_physics", lambda batch, device: tuple(
        stacked([s.dataset_id for s in batch.signals], k) for k in (1, 2, 3)))
    for module in (traffic_free_generation, traffic_reward):
        monkeypatch.setattr(module, "limits_s", lambda batch, params, step_s, augmented: [LIMIT_S] * len(batch.readings))


@pytest.mark.parametrize("moved", [False, True])
def test_a_round_s_sentences_train_on_what_their_words_were_sampled_from(tmp_path, monkeypatch, moved):
    """f1 follows f0 by 120 s on its approach (in its scene, not in a loss; f0's rows 0.7 s off f1's steps), f2 an hour
    later alone; ``moved``: f0 moved 37 s closer (augmentation D; its rows still 0.7 s off the steps) and the prior reads
    the landing context — the scene's."""
    from ts_transformer.experiments.traffic_augment import moved as moved_flight
    from ts_transformer.experiments.traffic_free_generation import speaking_batches
    from ts_transformer.experiments.traffic_reward import Round, round_summary, sentence_split, speak
    from ts_transformer.experiments.traffic_speaking import scene_of
    from ts_transformer.experiments.traffic_tuner import scene_layout, scene_logits
    from ts_transformer.prior.train import to_batch

    airport, signals, spec, *context = _airport(tmp_path, monkeypatch, [2.0, 120.0, 3_600.0], [0, 1, 2], context=moved)
    words, samples, keys = Words(spec), 2, ["KXXX:f1", "KXXX:f2"]
    _patch_physics(monkeypatch, airport, signals, keys, samples)
    batch = _batch(airport, signals, spec, keys)
    scenes = [scene_of(airport, key, LIMIT_S, spec.step_s) for key in keys]
    assert scenes[0].others == ("KXXX:f0",) and scenes[1].others == ()
    geometry = airport.flights.geometry
    landings, variant = None, "no-context"
    if moved:
        leader = scenes[0]
        scenes[0] = replace(leader, moved=(moved_flight(leader.rows("KXXX:f0"), leader.track("KXXX:f0"), 38.0, "KXXX:f0",
                                                        spec.step_s),))
        (landings,), variant = context, "full"
    round_ = Round(batch, scenes, [None, None], ["D" if moved else "real", "A"],
                   [np.ones(len(geometry.candidates), dtype=bool)] * 2)
    model = _traffic_model(words, variant=variant)
    recorded, original = [], model.logits

    def spy(h, tokens, valid, chosen, first):
        out = original(h, tokens, valid, chosen, first)
        recorded.append([logit[:, 0, 0].detach().clone() for logit in out])
        return out

    model.logits = spy
    try:
        spoken = speak(model, round_, samples, words, _params(), landings, ProcedureMasks.none(),
                       seed=4, budget=10 ** 9, source="train")
    finally:
        del model.logits
    assert [r["dataset_id"] for r in spoken.rows] == [k for k in keys for _ in range(samples)]
    assert [r["kind"] for r in spoken.rows] == [round_.kinds[0]] * 2 + ["A", "A"]
    assert all(r["reward"] == float(r["visual"]["outcome"] == "landed") for r in spoken.rows)
    assert not any(r["starts_in_a_loss"] for r in spoken.rows)
    for s, row in enumerate(spoken.rows):
        assert len(spoken.said[s]) == row["judged_steps"] and len(spoken.positions[s]) == N_LOOK + row["judged_steps"]
        assert all(len(codes) == row["judged_steps"] for codes in spoken.allowed[s].values())
    described = round_summary(round_, spoken, samples)
    assert described["scenes"] == 2 and described["kinds"] == {round_.kinds[0]: 1, "A": 1}
    keep = np.arange(len(spoken.rows))
    table = replace(_split([], words, geometry), variant=variant)
    sentences = sentence_split(round_, spoken, keep, samples, table, landings, ("KXXX",), spec.step_s)
    single = to_batch(sentences.split, list(keep), CPU)
    with torch.no_grad():
        logits = scene_logits(model, scene_layout(sentences, list(keep), single, spec.step_s))
    # the loop spoke in its batches' order (by scene size: f2 first), the round reads flight-major in its own
    (chunk,) = speaking_batches(scenes, [LIMIT_S] * 2, [len(r.words) for r in batch.readings], samples, 10 ** 9,
                                spec.step_s)
    assert chunk == [1, 0]
    for s, row in enumerate(spoken.rows):
        spoke = chunk.index(s // samples) * samples + s % samples
        for step in range(row["judged_steps"]):
            for c in range(6):
                want, got = recorded[6 * step + c][c][spoke], logits[c][s, 0, N_LOOK + step]
                finite = torch.isfinite(want)
                assert torch.equal(finite, torch.isfinite(got))
                assert torch.allclose(got[finite], want[finite], atol=1e-4), (s, step, c)


def test_the_landing_gap_reads_the_landing_before_on_the_runway(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_reward import landing_gap_s
    from ts_transformer.experiments.traffic_speaking import scene_of

    airport, signals, spec = _airport(tmp_path, monkeypatch, [0.0, 120.0, 3_600.0], [0, 1, 2])
    scene = scene_of(airport, "KXXX:f1", LIMIT_S, spec.step_s)
    own = scene.track("KXXX:f1").presence
    assert landing_gap_s(scene, own.runway, own.landing_s) == pytest.approx(120.0, abs=1.0)
    assert landing_gap_s(scene, own.runway, scene.track("KXXX:f0").presence.landing_s) is None
    alone = scene_of(airport, "KXXX:f2", LIMIT_S, spec.step_s)
    assert landing_gap_s(alone, own.runway, alone.track("KXXX:f2").presence.landing_s) is None


def test_a_round_refuses_to_run_short_of_augmented_scenes(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_reward import first_per_airport

    airport, signals, spec = _airport(tmp_path, monkeypatch, [0.0, 120.0, 3_600.0], [0, 1, 2])
    batch = _batch(airport, signals, spec, ["KXXX:f0", "KXXX:f1", "KXXX:f2"])
    assert first_per_airport(batch, 2, ["KXXX"]) == [0, 1]
    with pytest.raises(ValueError, match="KXXX"):
        first_per_airport(batch, 4, ["KXXX"])
    with pytest.raises(ValueError, match="KYYY"):
        first_per_airport(batch, 1, ["KXXX", "KYYY"])


def _row(number, *, reward, landed=0.9, runway=0.95, lost=0.1, time_ratio=1.0, gap_ratio=1.0, heading=3.0):
    def side(real):
        out = {"free_generation": {"all": {"outcomes": {"landed": landed}, "landed_on_observed_runway": runway,
                                           "words_after_first_per_flight": {"approach": 1.0, "heading": heading,
                                                                            "altitude": 1.0, "angle": 1.0,
                                                                            "speed": 1.0}}},
               "separation": {"lost_separation": lost}, "reward": reward}
        if real:
            out["ordering"] = {"time_ratio": time_ratio, "gap_ratio": gap_ratio}
        return out

    return {"round": number, "real": side(True), "augmented": side(False)}


def _rewards(ones, flips_down=0, n=100):
    """An augmented select readout's rewards: sentences 0 … ones − 1 rewarded, and of those the first ``flips_down`` not
    (keyed as `select_rewards` keys them)."""
    return {("KXXX:f", "A", k): float(flips_down <= k < ones) for k in range(n)}


def test_the_guards_exclude_a_round_by_name_and_the_choice_reads_paired_standard_errors():
    from ts_transformer.experiments.traffic_reward import guarded_choice

    labelled = {side: {"approach": 1.0, "heading": 3.0, "altitude": 1.0, "angle": 1.0, "speed": 1.0}
                for side in ("real", "augmented")}
    history = [_row(0, reward=0.50), _row(1, reward=0.60), _row(2, reward=0.61), _row(3, reward=0.90, lost=0.12),
               _row(4, reward=0.90, landed=0.85), _row(5, reward=0.90, gap_ratio=1.3), _row(6, reward=0.90, heading=4.0),
               _row(7, reward=0.90, time_ratio=None)]
    rewards = [_rewards(50), _rewards(60), _rewards(61)] + [_rewards(90)] * 5
    kept, excluded = guarded_choice(history, labelled, rewards)
    # round 1 beats round 0 by 0.10 against 2 × √10 / 100 = 0.063; round 2 (the highest) beats round 1 by only
    # 0.01 against 2 × √1 / 100: the earlier one
    assert kept == 1
    assert excluded == {3: ["lost separation"], 4: ["landed"], 5: ["landing gap"],
                        6: ["real heading words", "augmented heading words"], 7: ["time to land"]}
    # clearly above the earlier candidate: the later one
    assert guarded_choice(history[:3], labelled, [_rewards(50), _rewards(60), _rewards(80)])[0] == 2
    # +0.05 over round 0 but 55 sentences flipped (30 up, 25 down): 2 × √55 / 100 = 0.148 — noise, round 0 kept
    noisy = {k: v for k, v in _rewards(50).items()}
    noisy.update({("KXXX:f", "A", k): 0.0 for k in range(25)})
    noisy.update({("KXXX:f", "A", k): 1.0 for k in range(50, 80)})
    assert guarded_choice(history[:2], labelled, [_rewards(50), noisy]) == (0, {})
    assert guarded_choice(history[:2], labelled, [_rewards(50), _rewards(50)]) == (0, {})
    assert guarded_choice([_row(0, reward=0.5), _row(1, reward=0.4)], labelled, [_rewards(50), _rewards(40)]) == (0, {})


def test_paired_differences_read_the_sentences_both_rounds_count(tmp_path):
    from ts_transformer.experiments.traffic_reward import paired_difference, select_rewards

    first, then = _rewards(4, n=6), _rewards(6, flips_down=1, n=6)
    del then[("KXXX:f", "A", 5)]                                # not counted in the later round: not paired
    difference, error = paired_difference(first, then)
    # pairs 0 … 4: 1→0, 1→1, 1→1, 1→1, 0→1
    assert difference == 0.0 and error == pytest.approx(np.sqrt(2) / 5)
    for number, rows in enumerate(([{"dataset_id": "a", "kind": "A", "sample": 0, "reward": 1.0,
                                      "starts_in_a_loss": False},
                                     {"dataset_id": "a", "kind": "A", "sample": 1, "reward": 0.0,
                                      "starts_in_a_loss": True}],
                                    [{"dataset_id": "a", "kind": "A", "sample": 0, "reward": 0.0,
                                      "starts_in_a_loss": False}])):
        (tmp_path / f"round_{number:02d}").mkdir()
        (tmp_path / f"round_{number:02d}" / "readout.json").write_text(
            json.dumps({"augmented": {"flights": rows}, "real": {"flights": []}}))
    assert select_rewards(tmp_path, 1) == [{("a", "A", 0): 1.0}, {("a", "A", 0): 0.0}]
    with pytest.raises(ValueError, match="nothing to pair"):
        paired_difference({("a", "A", 0): 1.0}, {("a", "A", 1): 1.0})


def test_the_highest_candidate_is_the_highest_on_the_paired_sentences():
    """Round 3's own reward (over what it counts: 100 sentences more, all rewarded) is the highest, but on the sentences
    it pairs with round 0 round 2 is: ranked on its own reward round 3 would not beat round 1 (+0.03 against 2 × √3 /
    100), keeping 1; ranked on the pairs, as the candidates were found, round 2 beats round 1 and is kept."""
    from ts_transformer.experiments.traffic_reward import guarded_choice

    labelled = {side: {"approach": 1.0, "heading": 3.0, "altitude": 1.0, "angle": 1.0, "speed": 1.0}
                for side in ("real", "augmented")}
    history = [_row(0, reward=0.50), _row(1, reward=0.60), _row(2, reward=0.80), _row(3, reward=0.815)]
    third = _rewards(63)
    third.update({("KXXX:f", "A", k): 1.0 for k in range(100, 200)})     # counted only in round 3
    assert guarded_choice(history, labelled, [_rewards(50), _rewards(60), _rewards(80), third]) == (2, {})


def test_a_scene_starting_in_a_loss_is_not_trained_on_and_the_readouts_read_the_rest(tmp_path, monkeypatch):
    """f1 follows f0 by 30 s (in a loss it answers for at its first predicted step), f3 an hour later alone."""
    from ts_transformer.experiments.traffic_reward import (
        Round, ordering, ordering_failures, round_summary, side_readout, speak, trained_scenes,
    )
    from ts_transformer.experiments.traffic_speaking import scene_of

    airport, signals, spec = _airport(tmp_path, monkeypatch)
    words, samples, keys = Words(spec), 2, ["KXXX:f1", "KXXX:f3"]
    _patch_physics(monkeypatch, airport, signals, keys, samples)
    batch = _batch(airport, signals, spec, keys)
    scenes = [scene_of(airport, key, LIMIT_S, spec.step_s) for key in keys]
    geometry = airport.flights.geometry
    round_ = Round(batch, scenes, [None, None], ["real", "real"], [np.ones(len(geometry.candidates), dtype=bool)] * 2)
    spoken = speak(_traffic_model(words), round_, samples, words, _params(), None, ProcedureMasks.none(),
                   seed=4, budget=10 ** 9, source="scene")
    assert [r["starts_in_a_loss"] for r in spoken.rows] == [True, True, False, False]
    for r in spoken.rows[2:]:
        r["reward"] = float(r is spoken.rows[2])                   # f3's two sentences differ
    advantages, starts_lost, keep = trained_scenes(spoken, samples)
    assert list(starts_lost) == [True, False] and list(keep) == [2, 3]
    assert np.allclose(advantages[2:], [0.5, -0.5])
    described = round_summary(round_, spoken, samples)
    assert described["scenes_starting_in_a_loss"] == 1 and described["scenes_with_contrast"] == 1
    # the ordering reads the landed sentences against their own flights' records: f1 never lands (a loss ends it at
    # its first step), f3 has no landing before it in its scene — no gap to read, which fails the gap guard
    order = ordering(spoken.rows, round_, samples, spec.step_s)
    landed = [r for r in spoken.rows if r["visual"]["outcome"] == "landed"]
    assert all(r["dataset_id"] == "KXXX:f3" for r in landed)
    assert order["gap_s"] is None and order["recorded_gap_s"] is None and order["landed_with_a_gap"] == 0
    if landed:
        assert order["recorded_time_to_land_s"] == landed[0]["observed_remaining_s"]
    assert "landing gap" in ordering_failures({"real": {"ordering": order}})
    # the select readout leaves out the flights starting in a loss, as the separation summary does
    readout = side_readout(spoken.rows, round_, samples, spec.step_s, real=True)
    assert readout["reward"] == pytest.approx(0.5) and readout["separation"]["left_out_starting_in_a_loss"] == 2


def test_the_traffic_readout_is_zero_at_the_start_and_reads_the_layer_once_it_moves(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_reward import traffic_readout
    from ts_transformer.experiments.traffic_scene_data import build_split

    _airport(tmp_path, monkeypatch)
    from ts_transformer.instructions.artefact import load_spec
    spec = load_spec(tmp_path / "artefact")
    built = [b for b in build_split(tmp_path / "artefact", "train", spec, ("KXXX",), None, 2_048)[0] if b.sample.asks]
    words = Words(spec)
    at_zero = traffic_readout(_traffic_model(words, reads=False), built, 1, CPU)
    moved = traffic_readout(_traffic_model(words, reads=True), built, 1, CPU)
    assert at_zero["aircraft_steps_with_another"] > 0 and at_zero["traffic_output_over_residual"] == [0.0, 0.0]
    assert all(r > 0.0 for r in moved["traffic_output_over_residual"])
    assert moved["teacher_forced"]["nll_per_step"] != at_zero["teacher_forced"]["nll_per_step"]


def test_a_traffic_prior_the_runner_writes_is_read_back_by_load_prior(tmp_path, monkeypatch):
    import json

    from ts_transformer.experiments import prior_train
    from ts_transformer.experiments.prior_train import TRAFFIC_CHECKPOINT_SCHEMA, load_prior
    from ts_transformer.experiments.traffic_reward import write_traffic_prior
    from ts_transformer.inference.scene_edges import EDGE_FEATURES
    from ts_transformer.io_utils import file_sha256
    from ts_transformer.prior.model import with_traffic
    from ts_transformer.tests.support import fly_legs, instruction_flight
    from ts_transformer.tests.test_autopilot import DOWNWIND_BASE_FINAL
    from ts_transformer.tests.test_instruction_training_export import _artefact
    from ts_transformer.tests.test_training_overlays import _prior_dir

    flights = [instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0), dataset_id="KXXX:own",
                                  split="val")]
    _artefact(tmp_path / "artefact", flights)
    roster = tmp_path / "tracks.json"
    roster.write_text(json.dumps({"records": []}), encoding="utf-8")
    monkeypatch.setattr(prior_train, "tracks_manifest_path", lambda code: roster)
    _prior_dir(tmp_path / "prior", tmp_path / "artefact", roster)
    start = load_prior(tmp_path / "prior", tmp_path / "artefact")
    torch.manual_seed(0)
    model = with_traffic(start.model, EDGE_FEATURES)
    for layer in model.layers:
        torch.nn.init.normal_(layer.traffic.out.weight)
    (tmp_path / "traffic").mkdir()
    write_traffic_prior(tmp_path / "traffic", model, tmp_path / "prior", start, start.payload["spec_sha256"],
                        git={"head": "test", "dirty": False}, smoke=True, fine_tuning={"round": 1})
    loaded = load_prior(tmp_path / "traffic", tmp_path / "artefact")
    assert loaded.payload["schema"] == loaded.config["schema"] == TRAFFIC_CHECKPOINT_SCHEMA
    assert loaded.payload["start"] == loaded.config["start"] == {
        "directory": str(tmp_path / "prior"), "checkpoint_sha256": file_sha256(tmp_path / "prior" / "checkpoint.pt")}
    assert loaded.model.traffic_features == EDGE_FEATURES and loaded.config["fine_tuning"] == {"round": 1}
    assert all(torch.equal(value, loaded.model.state_dict()[name]) for name, value in model.state_dict().items())
    assert loaded.procedure_masks.names == start.procedure_masks.names


def test_the_ordering_guards_read_the_record_and_an_unreadable_ratio_fails():
    from ts_transformer.experiments.traffic_reward import ordering_failures

    def row(time_ratio, gap_ratio):
        return {"real": {"ordering": {"time_ratio": time_ratio, "gap_ratio": gap_ratio}}}

    assert ordering_failures(row(1.0, 1.19)) == []
    assert ordering_failures(row(1.21, 1.0)) == ["time to land"]
    assert ordering_failures(row(None, 1.3)) == ["time to land", "landing gap"]


def test_a_resumed_run_continues_only_from_finished_rounds_of_the_same_configuration(tmp_path):
    from ts_transformer.experiments.traffic_reward import completed_rounds, round_seed, run_differences

    out = tmp_path / "run"
    for k in range(3):
        (out / f"round_{k:02d}").mkdir(parents=True)
        (out / f"round_{k:02d}" / "readout.json").write_text("{}")
    assert completed_rounds(out) == 2
    (out / "round_03").mkdir()                                     # cut short: no readout
    with pytest.raises(SystemExit, match=r"round\(s\) \[3\] did not finish"):
        completed_rounds(out)
    (out / "round_03").rename(out / "round_03.aborted-20260928T2000Z")
    assert completed_rounds(out) == 2
    (out / "round_01" / "readout.json").unlink()
    with pytest.raises(SystemExit, match="did not finish"):
        completed_rounds(out)
    with pytest.raises(SystemExit, match="finished no round"):
        completed_rounds(tmp_path / "empty")
    (out / "round_05").mkdir()
    (out / "round_05" / "readout.json").write_text("{}")
    (out / "round_01" / "readout.json").write_text("{}")
    with pytest.raises(SystemExit, match="not 0"):
        completed_rounds(out)
    stored = {"written_utc": "a", "rounds": 1, "resumed": [], "seed": 1337, "git": {"head": "x", "dirty": False},
              "select": {"per_airport": 200}}
    record = {**stored, "written_utc": "b", "rounds": 3, "git": {"head": "y", "dirty": False}}
    assert run_differences(stored, record) == ["git"]
    assert run_differences(stored, {**stored, "rounds": 8, "extra": 1}) == ["extra"]
    assert run_differences(stored, {**stored, "select": {"per_airport": 100}}) == ["select"]
    # each round's streams are its own
    seeds = {round_seed(1337, r, s) for r in (1, 2) for s in (1, 2)}
    assert len(seeds) == 4 and round_seed(1337, 1, 1) == round_seed(1337, 1, 1)


def test_the_sentences_do_not_depend_on_how_many_processes_speak_them(tmp_path, monkeypatch):
    """`Speakers` (design §6.6 step 6 item 13): one and two forked processes say what this process says, bit for bit —
    every loop batch from its own stream — and a process that fails ends the call with its traceback. The reference is
    spoken on one thread as the speaking processes are (a CPU sum over several threads rounds otherwise: the words are
    the same, a probability read beside them differs in its eighth digit)."""
    from ts_transformer.experiments.traffic_reward import Round, Speakers, Speaking, speak
    from ts_transformer.experiments.traffic_speaking import scene_of

    airport, signals, spec = _airport(tmp_path, monkeypatch, [2.0, 120.0, 3_600.0], [0, 1, 2])
    words, samples, keys = Words(spec), 2, ["KXXX:f1", "KXXX:f2"]
    _patch_physics(monkeypatch, airport, signals, keys, samples)
    batch = _batch(airport, signals, spec, keys)
    scenes = [scene_of(airport, key, LIMIT_S, spec.step_s) for key in keys]
    geometry = airport.flights.geometry
    round_ = Round(batch, scenes, [None, None], ["real", "A"], [np.ones(len(geometry.candidates), dtype=bool)] * 2)
    model = _traffic_model(words)
    budget = 1                                        # a flight a batch: two batches to deal
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        here = speak(model, round_, samples, words, _params(), None, ProcedureMasks.none(), seed=7, budget=budget,
                     source="train")
    finally:
        torch.set_num_threads(threads)
    speaking = Speaking(words, _params(), None, ProcedureMasks.none(), budget)
    for count in (1, 2):
        speakers = Speakers(count, lambda n: round_, {}, model, "cpu", speaking)
        try:
            there, peaks = speakers.speak("train", 1, round_, model, samples, seed=7, source="train")
        finally:
            speakers.close()
        assert peaks == [0.0] * count                  # on the CPU: no GPU peak
        assert [r["dataset_id"] for r in there.rows] == [r["dataset_id"] for r in here.rows]
        different = [(k, a[k], b[k]) for a, b in zip(there.rows, here.rows) for k in a if json.dumps(a[k], default=str)
                     != json.dumps(b[k], default=str)]
        assert not different, different[:3]
        assert all(np.array_equal(a, b) for a, b in zip(there.said, here.said))
        assert all(np.array_equal(a, b) for a, b in zip(there.positions, here.positions))
        assert all(a.keys() == b.keys() and all(np.array_equal(a[c], b[c]) for c in a)
                   for a, b in zip(there.allowed, here.allowed))
    # another seed says other words somewhere (the stream is read)
    other = speak(model, round_, samples, words, _params(), None, ProcedureMasks.none(), seed=8, budget=budget,
                  source="train")
    assert any(not np.array_equal(a, b) for a, b in zip(other.said, here.said))
    # a process that fails says why; one that rebuilt another round, or is gone, ends the call
    failing = Speakers(1, lambda n: (_ for _ in ()).throw(RuntimeError("no round")), {}, model, "cpu", speaking)
    try:
        with pytest.raises(SystemExit, match="no round"):
            failing.speak("train", 1, round_, model, samples, seed=7, source="train")
    finally:
        failing.close()
    other = Round(replace(batch, signals=batch.signals[::-1]), scenes[::-1], [None, None], ["A", "real"],
                  round_.directions)
    wrong = Speakers(1, lambda n: other, {}, model, "cpu", speaking)
    try:
        with pytest.raises(SystemExit, match="rebuilt another round"):
            wrong.speak("train", 1, round_, model, samples, seed=7, source="train")
    finally:
        wrong.close()
    gone = Speakers(2, lambda n: round_, {}, model, "cpu", speaking)
    gone.processes[1].kill()
    gone.processes[1].join(timeout=10)
    try:
        with pytest.raises(SystemExit, match="speaking process 1 is gone"):
            gone.speak("train", 1, round_, model, samples, seed=7, source="train")
    finally:
        gone.close()
