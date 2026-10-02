"""The round protocol of the multi-aircraft post-training (`experiments/traffic_rounds`, design §6.6 step 6; lifted out of
M4's first runner in step 9's 9.4.1): the guards exclude a round by name and the choice reads paired standard errors,
taking the earliest within the tie; paired differences read the sentences both rounds count; the ordering guards read the
record; a resumed run continues only from finished rounds of the same configuration; the traffic attention's readout;
a traffic prior written by a runner is read back by `load_prior`. The speaking processes are tested with window rounds
(`test_traffic_window_reward.py`)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.instructions.words import Words
from ts_transformer.prior.model import with_traffic
from ts_transformer.tests.test_prior_speaker import _model

CPU = torch.device("cpu")


def _airport(tmp_path, monkeypatch):
    """The scene-data fixture: f0, f2 (background) and f1 entering 0, 10 and 30 s apart on one approach, f3 an hour
    later, alone."""
    from ts_transformer.experiments import traffic_scene_data
    from ts_transformer.experiments.traffic_speaking import scene_airports
    from ts_transformer.tests.test_traffic_scene_data import _artefact

    directory, manifest, spec, _ = _artefact(tmp_path, [0.0, 30.0, 10.0, 3_600.0], [0, 1, 3])
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    airports, _ = scene_airports(directory, "train", spec, ("KXXX",), None, 2_048)
    return airports["KXXX"], spec


def _traffic_model(words, seed=1, reads=True):
    """A traffic prior; ``reads``: its traffic attention's output layer moved off zero, so the others change its words."""
    torch.manual_seed(seed)
    model = with_traffic(_model(words), EDGE_FEATURES).eval()
    if reads:
        for layer in model.layers:
            torch.nn.init.normal_(layer.traffic.out.weight, std=0.2)
    return model


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
    from ts_transformer.experiments.traffic_rounds import guarded_choice

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


def test_paired_differences_read_the_sentences_both_rounds_count():
    from ts_transformer.experiments.traffic_rounds import paired_difference

    first, then = _rewards(4, n=6), _rewards(6, flips_down=1, n=6)
    del then[("KXXX:f", "A", 5)]                                # not counted in the later round: not paired
    difference, error = paired_difference(first, then)
    # pairs 0 … 4: 1→0, 1→1, 1→1, 1→1, 0→1
    assert difference == 0.0 and error == pytest.approx(np.sqrt(2) / 5)
    with pytest.raises(ValueError, match="nothing to pair"):
        paired_difference({("a", "A", 0): 1.0}, {("a", "A", 1): 1.0})


def test_the_highest_candidate_is_the_highest_on_the_paired_sentences():
    """Round 3's own reward (over what it counts: 100 sentences more, all rewarded) is the highest, but on the sentences
    it pairs with round 0 round 2 is: ranked on its own reward round 3 would not beat round 1 (+0.03 against 2 × √3 /
    100), keeping 1; ranked on the pairs, as the candidates were found, round 2 beats round 1 and is kept."""
    from ts_transformer.experiments.traffic_rounds import guarded_choice

    labelled = {side: {"approach": 1.0, "heading": 3.0, "altitude": 1.0, "angle": 1.0, "speed": 1.0}
                for side in ("real", "augmented")}
    history = [_row(0, reward=0.50), _row(1, reward=0.60), _row(2, reward=0.80), _row(3, reward=0.815)]
    third = _rewards(63)
    third.update({("KXXX:f", "A", k): 1.0 for k in range(100, 200)})     # counted only in round 3
    assert guarded_choice(history, labelled, [_rewards(50), _rewards(60), _rewards(80), third]) == (2, {})


def test_the_traffic_readout_is_zero_at_the_start_and_reads_the_layer_once_it_moves(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_rounds import traffic_readout
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
    from ts_transformer.experiments.traffic_rounds import write_traffic_prior
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
                        git={"head": "test", "dirty": False}, smoke=True, fine_tuning={"round": 1},
                        writer="ts_transformer.experiments.traffic_window_reward")
    loaded = load_prior(tmp_path / "traffic", tmp_path / "artefact")
    assert loaded.payload["schema"] == loaded.config["schema"] == TRAFFIC_CHECKPOINT_SCHEMA
    assert loaded.payload["start"] == loaded.config["start"] == {
        "directory": str(tmp_path / "prior"), "checkpoint_sha256": file_sha256(tmp_path / "prior" / "checkpoint.pt")}
    assert loaded.model.traffic_features == EDGE_FEATURES and loaded.config["fine_tuning"] == {"round": 1}
    assert all(torch.equal(value, loaded.model.state_dict()[name]) for name, value in model.state_dict().items())
    assert loaded.procedure_masks.names == start.procedure_masks.names
    assert json.loads((tmp_path / "traffic" / "procedure_masks.json").read_text())["written_by"] == \
        "ts_transformer.experiments.traffic_window_reward"


def test_the_ordering_guards_read_the_record_and_an_unreadable_ratio_fails():
    from ts_transformer.experiments.traffic_rounds import ordering_failures

    def row(time_ratio, gap_ratio):
        return {"real": {"ordering": {"time_ratio": time_ratio, "gap_ratio": gap_ratio}}}

    assert ordering_failures(row(1.0, 1.19)) == []
    assert ordering_failures(row(1.21, 1.0)) == ["time to land"]
    assert ordering_failures(row(None, 1.3)) == ["time to land", "landing gap"]


def test_a_resumed_run_continues_only_from_finished_rounds_of_the_same_configuration(tmp_path):
    from ts_transformer.experiments.traffic_rounds import completed_rounds, round_seed, run_differences

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
