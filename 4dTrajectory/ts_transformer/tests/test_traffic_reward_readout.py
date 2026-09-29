"""The M4 run's readout (`experiments/traffic_reward_readout`): the scene classes of the training signal, how the losses
spread over flights, and the whole runner on a tmp run directory (never a live root)."""

import json

import numpy as np
import pytest

from ts_transformer.experiments.traffic_free_generation import SCHEMA as FREE_GENERATION_SCHEMA
from ts_transformer.experiments.traffic_reward import TRAFFIC_REWARD_SCHEMA
from ts_transformer.experiments.traffic_reward_readout import (
    CLASSES,
    concentration,
    main,
    scene_class,
    training_signal,
)
from ts_transformer.instructions.words import APPROACH_GO_AROUND, COLUMNS

LOST, LANDED, TIMEOUT = "lost_separation", "landed", "timeout"


def test_scene_class_covers_every_case():
    reward = np.array
    assert scene_class(np.array([LANDED, LANDED]), reward([1.0, 1.0])) == CLASSES[0]
    assert scene_class(np.array([LOST, LOST]), reward([0.0, 0.0])) == CLASSES[1]
    assert scene_class(np.array([LOST, TIMEOUT]), reward([0.0, 0.0])) == CLASSES[2]
    assert scene_class(np.array([TIMEOUT, TIMEOUT]), reward([0.0, 0.0])) == CLASSES[2]
    assert scene_class(np.array([LANDED, LOST]), reward([1.0, 0.0])) == CLASSES[3]
    assert scene_class(np.array([LANDED, TIMEOUT]), reward([1.0, 0.0])) == CLASSES[4]


def _sentences(kinds, outcomes, steps=3):
    """A ``sentences.npz``'s arrays: one scene per kind, its sentences' outcomes (reward 1 = landed), advantages as the
    run computes them (reward less the scene's mean)."""
    outcome = np.array([o for scene in outcomes for o in scene])
    reward = (outcome == LANDED).astype(float)
    by_scene = reward.reshape(len(kinds), -1)
    said = np.zeros((len(outcome) * steps, len(COLUMNS)), dtype=np.int16)
    said[1, COLUMNS.index("approach")] = APPROACH_GO_AROUND
    return {"kind": np.array(kinds), "outcome": outcome, "reward": reward,
            "advantage": (by_scene - by_scene.mean(axis=1, keepdims=True)).reshape(-1),
            "step_offsets": np.arange(0, len(outcome) * steps + 1, steps), "said": said}


def test_training_signal_splits_the_advantage_by_class():
    signal = training_signal(_sentences(["real", "A", "A"], [[LANDED, LOST], [LANDED, TIMEOUT], [LOST, LOST]]))
    assert signal["samples"] == 2
    assert signal["scenes"]["real"] == {"scenes": 1, **{c: float(c == CLASSES[3]) for c in CLASSES}}
    assert signal["scenes"]["A"][CLASSES[4]] == 0.5 and signal["scenes"]["A"][CLASSES[1]] == 0.5
    assert signal["scenes"]["all"]["scenes"] == 3
    # two differing scenes, |advantage| 1.0 each: half from the one with a lost sentence
    assert signal["advantage_share"][CLASSES[3]] == pytest.approx(0.5)
    assert signal["advantage_share"][CLASSES[4]] == pytest.approx(0.5)
    assert signal["advantage_share"][CLASSES[1]] == 0.0
    assert signal["go_around_steps"] == 1 and signal["steps"] == 18


def test_training_signal_refuses_uneven_scenes():
    arrays = _sentences(["real", "A"], [[LANDED, LOST], [LANDED, LOST]])
    arrays["outcome"] = arrays["outcome"][:3]
    with pytest.raises(ValueError, match="3 sentences over 2 scenes"):
        training_signal(arrays)


def test_concentration_groups_flights_by_how_often_they_lose():
    tries = {"a": [False] * 6, "b": [True, False] * 3, "c": [True] * 5 + [False], "d": [True] * 6}
    references = {"a": {"recorded": False, "labelled": False, "vectored": False},
                  "b": {"recorded": False, "labelled": True, "vectored": True},
                  "c": {"recorded": True, "labelled": True, "vectored": True},
                  "d": {"recorded": False, "labelled": True, "vectored": False}}
    spread = concentration(tries, references)
    assert (spread["tries"], spread["losses"]) == (6, 3 + 5 + 6)
    assert spread["never"]["flights"] == 1 and spread["never"]["labelled"] == 0.0
    assert spread["sometimes"]["flights"] == 1 and spread["sometimes"]["loss_share"] == pytest.approx(3 / 14)
    # 5 of 6 is "almost always" (ALMOST_ALWAYS = 5/6)
    assert spread["almost_always"]["flights"] == 2
    assert spread["almost_always"]["recorded"] == 0.5 and spread["almost_always"]["labelled"] == 1.0
    assert spread["histogram"] == {0: 1, 3: 1, 5: 1, 6: 1}
    assert concentration(tries, {k: {"vectored": v["vectored"]} for k, v in references.items()})["never"] == {
        "flights": 1, "flight_share": 0.25, "loss_share": 0.0, "vectored": 0.0}


def test_concentration_refuses_unequal_tries():
    with pytest.raises(ValueError, match="tried equally"):
        concentration({"a": [False, False], "b": [True]}, {"a": {}, "b": {}})


def _flight(dataset_id, kind, sample, lost, stratum="vectored", starts=False, go_arounds=0):
    return {"dataset_id": dataset_id, "kind": kind, "sample": sample, "stratum": stratum, "starts_in_a_loss": starts,
            "go_arounds": go_arounds, "visual": {"outcome": LOST if lost else LANDED}}


def _summary(number, lost):
    side = {"reward": 1.0 - lost, "separation": {"lost_separation": lost, "lost_separation_ifr": lost + 0.01},
            "free_generation": {"all": {"outcomes": {"landed": 0.9}, "landed_on_observed_runway": 0.8}}}
    row = {"round": number, "real": side, "augmented": side,
           "traffic": {"traffic_output_over_residual": [0.0], "teacher_forced": {"nll_per_step": 0.28}}}
    if number:
        row.update({"sentences": {"all": {"sentences": 4}}, "train_pass": {"kl_mean": 0.01, "kl_trace": [0.01]}})
    return row


def _run(tmp_path, rounds=2):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"schema": TRAFFIC_REWARD_SCHEMA}))
    (run / "history.json").write_text(json.dumps({"rounds": [_summary(k, 0.5) for k in range(rounds)]}))
    (run / "choice.json").write_text(json.dumps({"kept": 0}))
    for number in range(rounds):
        directory = run / f"round_{number:02d}"
        directory.mkdir()
        real = [_flight("f1", "real", s, lost=True) for s in range(2)] + [
            _flight("f2", "real", s, lost=False, stratum="straight-in", go_arounds=s) for s in range(2)] + [
            _flight("f3", "real", s, lost=True, starts=True) for s in range(2)]
        augmented = [_flight("f1", "A", s, lost=s == 0) for s in range(2)]
        (directory / "readout.json").write_text(json.dumps({"real": {"flights": real},
                                                            "augmented": {"flights": augmented}}))
        if number:
            np.savez(directory / "sentences.npz", **_sentences(["real", "A"], [[LANDED, LOST], [LOST, LOST]]))
    reference = tmp_path / "m3"
    reference.mkdir()
    (reference / "free_generation.json").write_text(json.dumps({
        "schema": FREE_GENERATION_SCHEMA, "augment_seed": None, "split": "select", "flights_file": "flights.jsonl"}))
    rows = [{**_flight(f, "real", 0, lost=f == "f1"), "source": source}
            for f in ("f1", "f2", "f3") for source in ("recorded", "labelled", "scene")]
    (reference / "flights.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run, reference


def test_runner_reads_every_round(tmp_path):
    run, reference = _run(tmp_path)
    out = tmp_path / "readout"
    assert main(["--run", str(run), "--reference", str(reference), "--out", str(out)]) == 0
    readout = json.loads((out / "traffic_reward_readout.json").read_text())
    assert [r["round"] for r in readout["rounds"]] == [0, 1] and "training" not in readout["rounds"][0]
    first = readout["rounds"][0]["select"]
    # f3 starts in a loss: left out of the per-flight shares, as the run's own separation readout leaves it out
    assert first["real"]["by_stratum"] == {"straight-in": {"flights": 2, "lost_separation": 0.0},
                                           "vectored": {"flights": 2, "lost_separation": 1.0}}
    assert first["augmented"]["by_kind"] == {"A": {"flights": 2, "lost_separation": 0.5}}
    assert first["real"]["go_around_sentences"] == 1
    assert "kl_trace" not in readout["rounds"][1]["training"]["pass"]
    assert readout["rounds"][1]["training"]["signal"]["scenes"]["all"]["scenes"] == 2
    real = readout["spread"]["real"]
    assert (real["flights"], real["tries"]) == (2, 4)
    assert real["almost_always"]["flights"] == 1 and real["almost_always"]["recorded"] == 1.0
    assert real["never"]["flights"] == 1 and real["never"]["vectored"] == 0.0 and real["never"]["labelled"] == 0.0
    assert readout["spread"]["augmented"]["sometimes"] == {"flights": 1, "flight_share": 1.0, "loss_share": 1.0,
                                                          "vectored": 1.0}
    with pytest.raises(SystemExit):
        main(["--run", str(run), "--reference", str(reference), "--out", str(out)])


def test_runner_refuses_an_unfinished_round_and_an_augmented_reference(tmp_path):
    run, reference = _run(tmp_path)
    (run / "round_02").mkdir()
    with pytest.raises(SystemExit, match="did not finish"):
        main(["--run", str(run), "--reference", str(reference), "--out", str(tmp_path / "a")])
    (run / "round_02").rmdir()
    header = json.loads((reference / "free_generation.json").read_text())
    (reference / "free_generation.json").write_text(json.dumps({**header, "augment_seed": 7919}))
    with pytest.raises(SystemExit, match="real select scenes"):
        main(["--run", str(run), "--reference", str(reference), "--out", str(tmp_path / "b")])


def test_runner_refuses_a_real_flight_without_references(tmp_path):
    run, reference = _run(tmp_path)
    rows = [json.loads(line) for line in (reference / "flights.jsonl").read_text().splitlines()]
    (reference / "flights.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["dataset_id"] != "f2"))
    with pytest.raises(SystemExit, match="no record / labelled"):
        main(["--run", str(run), "--reference", str(reference), "--out", str(tmp_path / "c")])
