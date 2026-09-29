"""The M4 run's readout (`experiments/traffic_reward_readout`): the scene classes of the training signal, the select
flights first round against last, and the whole runner on a tmp run directory (never a live root)."""

import json

import numpy as np
import pytest

from ts_transformer.experiments.traffic_free_generation import SCHEMA as FREE_GENERATION_SCHEMA
from ts_transformer.experiments.traffic_free_generation import SOURCES
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
from ts_transformer.experiments.traffic_reward import TRAFFIC_REWARD_SCHEMA
from ts_transformer.experiments.traffic_reward_readout import (
    CLASSES,
    artefact_name,
    groups_by_samples_lost,
    main,
    paired_change,
    scene_class,
    spread,
    training_signal,
)
from ts_transformer.instructions.readout import STRATA
from ts_transformer.instructions.words import APPROACH_GO_AROUND, COLUMNS
from ts_transformer.prior.landing_reward import LANDED

LOST = LOST_SEPARATION
OTHER = "timeout"  # any failing outcome other than a loss of separation
STRAIGHT_IN, VECTORED = STRATA
RECORDED, LABELLED = SOURCES[3], SOURCES[2]


def test_scene_class_covers_every_case():
    reward = np.array
    assert scene_class(np.array([LANDED, LANDED]), reward([1.0, 1.0])) == CLASSES[0]
    assert scene_class(np.array([LOST, LOST]), reward([0.0, 0.0])) == CLASSES[1]
    assert scene_class(np.array([LOST, OTHER]), reward([0.0, 0.0])) == CLASSES[2]
    assert scene_class(np.array([OTHER, OTHER]), reward([0.0, 0.0])) == CLASSES[2]
    assert scene_class(np.array([LANDED, LOST]), reward([1.0, 0.0])) == CLASSES[3]
    assert scene_class(np.array([LANDED, OTHER]), reward([1.0, 0.0])) == CLASSES[4]


def _sentences(kinds, outcomes, steps=3):
    """A ``sentences.npz``'s arrays: one scene per kind, its sentences' outcomes (reward 1 = landed), advantages as the
    run computes them (reward less the scene's mean)."""
    outcome = np.array([o for scene in outcomes for o in scene])
    reward = (outcome == LANDED).astype(float)
    by_scene = reward.reshape(len(kinds), -1)
    said = np.zeros((len(outcome) * steps, len(COLUMNS)), dtype=np.int16)
    said[1, COLUMNS.index("approach")] = APPROACH_GO_AROUND
    lengths = np.where(outcome == LOST, steps - 1, steps)
    return {"kind": np.array(kinds), "outcome": outcome, "reward": reward,
            "advantage": (by_scene - by_scene.mean(axis=1, keepdims=True)).reshape(-1),
            "step_offsets": np.concatenate(([0], np.cumsum(lengths))), "said": said}


def test_training_signal_splits_the_advantage_by_class():
    signal = training_signal(_sentences(["real", "A", "A"], [[LANDED, LOST], [LANDED, OTHER], [LOST, LOST]]), 2)
    assert signal["scenes"]["real"] == {"scenes": 1, **{c: float(c == CLASSES[3]) for c in CLASSES}}
    assert signal["scenes"]["A"][CLASSES[4]] == 0.5 and signal["scenes"]["A"][CLASSES[1]] == 0.5
    assert signal["scenes"]["all"]["scenes"] == 3 and signal["differing_scenes"] == 2
    # two differing scenes, |advantage| 1.0 each: half from the one with a lost sentence
    assert signal["advantage_share"][CLASSES[3]] == pytest.approx(0.5)
    assert signal["advantage_share"][CLASSES[4]] == pytest.approx(0.5)
    assert signal["advantage_share"][CLASSES[1]] == 0.0
    assert signal["steps_per_sentence"] == {"lost_separation": 2.0, "other": 3.0}
    assert signal["go_around_steps"] == 1


def test_training_signal_refuses_another_number_of_samples():
    with pytest.raises(ValueError, match="not 3 for each of 2 scenes"):
        training_signal(_sentences(["real", "A"], [[LANDED, LOST], [LANDED, LOST]]), 3)


def test_groups_by_samples_lost():
    lost = {"a": [False, False], "b": [True, False], "c": [True, True], "d": [False, False]}
    facts = {"a": {"labelled": False}, "b": {"labelled": True}, "c": {"labelled": True}, "d": {"labelled": True}}
    groups = groups_by_samples_lost(lost, facts)
    assert groups["0/2"] == {"flights": 2, "flight_share": 0.5, "loss_share": 0.0, "labelled": 0.5}
    assert groups["1/2"]["loss_share"] == pytest.approx(1 / 3) and groups["2/2"]["loss_share"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="samples"):
        groups_by_samples_lost({"a": [True], "b": [True, False]}, {"a": {}, "b": {}})


def test_paired_change_counts_the_same_draws():
    change = paired_change({"a": [True, False], "b": [True, True]}, {"a": [False, False], "b": [True, False]})
    assert (change["fixed"], change["newly_lost"], change["sentences"]) == (2, 0, 4)
    assert change["net"] == -0.5 and change["standard_error"] == pytest.approx(np.sqrt(2) / 4)


def test_spread_leaves_out_a_flight_starting_in_a_loss_in_any_sample():
    rounds = [{"a": [True, False], "b": [False, True], "c": [False, False]},
              {"a": [True, False], "b": [False, False], "c": [False, False]}]
    out = spread(rounds, {"c"}, {"c"}, {key: {} for key in "abc"})
    assert (out["left_out_starting_in_a_loss"], out["left_out_reference_starting_in_a_loss"], out["flights"]) == (1, 0, 2)
    assert out["same_in_every_round"] == 0.75
    assert out["paired_change"]["fixed"] == 1 and out["last_round"]["1/2"]["flights"] == 1


def _flight(dataset_id, kind, sample, lost, stratum=VECTORED, starts=False, go_arounds=0):
    return {"dataset_id": dataset_id, "kind": kind, "sample": sample, "stratum": stratum, "starts_in_a_loss": starts,
            "go_arounds": go_arounds, "visual": {"outcome": LOST if lost else LANDED}}


def _side(reward, lost, observed_runway):
    return {"reward": reward, "separation": {"lost_separation": lost, "lost_separation_ifr": lost + 0.01},
            "free_generation": {"all": {"outcomes": {"landed": reward + 0.05},
                                        "landed_on_observed_runway": observed_runway}}}


def _summary(number):
    row = {"round": number, "real": _side(0.8 + number / 100, 0.1, 0.8),
           "augmented": _side(0.6, 0.3 - number / 100, 0.7),
           "traffic": {"traffic_output_over_residual": [number / 100],
                       "teacher_forced": {"nll_per_step": 0.28 + number / 1000}}}
    if number:
        row.update({"sentences": {"all": {"sentences": 4}, "scenes_with_contrast": 1},
                    "train_pass": {"kl_mean": 0.01, "kl_trace": [0.01], "clipped_trace": [0.0]}})
    return row


EXECUTOR = {"directory": "executor", "sha256": "e" * 64}
INSTRUCTIONS = "4dTrajectory/outputs/POOLED/instruction_language/v5"
LAST = 2


def _real(number):
    """Round ``number``'s real select rows: f1 loses both samples every round; f2 (straight-in) loses sample 1 in round
    0 and sample 0 in round 1, none in the last; f3 loses sample 1, which starts in a loss (with a go-around) in the
    last round only; f4 never loses (its labelled reading starts in a loss)."""
    return ([_flight("f1", "real", s, lost=True) for s in range(2)]
            + [_flight("f2", "real", s, lost=(number, s) in ((0, 1), (1, 0)), stratum=STRAIGHT_IN) for s in range(2)]
            + [_flight("f3", "real", s, lost=s == 1, starts=s == 1 and number == LAST, go_arounds=s) for s in range(2)]
            + [_flight("f4", "real", s, lost=False) for s in range(2)])


def _run(tmp_path, last=LAST):
    """A finished run of rounds 0 … ``last`` and its M3 reference (docstring of `_real`; augmented g1 kind A loses
    sample 0)."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({
        "schema": TRAFFIC_REWARD_SCHEMA, "smoke": False, "samples": 2, "rounds": last, "executor": EXECUTOR,
        "instructions": f"/elsewhere/.claude/worktrees/run/{INSTRUCTIONS}",
        "select": {"recorded": {"flights": 3, "lost_separation": 1 / 3}}}))
    (run / "history.json").write_text(json.dumps({"rounds": [_summary(k) for k in range(last + 1)]}))
    (run / "choice.json").write_text(json.dumps({"round": 0, "augmented_reward": [0.6] * (last + 1)}))
    for number in range(last + 1):
        directory = run / f"round_{number:02d}"
        directory.mkdir()
        augmented = [_flight("g1", "A", s, lost=s == 0) for s in range(2)]
        (directory / "readout.json").write_text(json.dumps({"real": {"flights": _real(number)},
                                                            "augmented": {"flights": augmented}}))
        if number:
            np.savez(directory / "sentences.npz", **_sentences(["real", "A"], [[LANDED, LOST], [LOST, LOST]]))
    reference = tmp_path / "m3"
    reference.mkdir()
    (reference / "free_generation.json").write_text(json.dumps({
        "schema": FREE_GENERATION_SCHEMA, "augment_seed": None, "split": "select", "flights_file": "flights.jsonl",
        "executor": EXECUTOR, "instructions": INSTRUCTIONS}))
    # f1: its record loses, its labelled words do not; f2 the other way round; f3's record starts in a loss; f4's
    # labelled reading starts in one; f9 is not a select flight of the run (its record must not enter the check)
    outcome = {("f1", RECORDED): True, ("f1", LABELLED): False, ("f2", RECORDED): False, ("f2", LABELLED): True,
               ("f3", RECORDED): True, ("f3", LABELLED): False, ("f4", RECORDED): False, ("f4", LABELLED): True,
               ("f9", RECORDED): True, ("f9", LABELLED): True}
    starting = {("f3", RECORDED), ("f4", LABELLED)}
    rows = [{**_flight(f, "real", 0, lost=lost, starts=(f, source) in starting), "source": source}
            for (f, source), lost in outcome.items()] + [{**_flight("f1", "real", 0, lost=False), "source": "scene"}]
    (reference / "flights.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run, reference


def _main(run, reference, out):
    return main(["--run", str(run), "--reference", str(reference), "--out", str(out)])


def test_runner_reads_every_round(tmp_path):
    run, reference = _run(tmp_path)
    out = tmp_path / "readout"
    assert _main(run, reference, out) == 0
    readout = json.loads((out / "traffic_reward_readout.json").read_text())
    assert readout["last_round"] == LAST and readout["choice"]["round"] == 0
    assert [r["round"] for r in readout["rounds"]] == [0, 1, 2] and "training" not in readout["rounds"][0]
    first, second = (r["select"] for r in readout["rounds"][:2])
    assert first["real"]["reward"] == 0.8 and second["real"]["reward"] == 0.81
    assert second["augmented"]["lost_separation"] == 0.29 and second["augmented"]["lost_separation_ifr"] == 0.3
    assert first["real"]["landed"] == pytest.approx(0.85) and first["real"]["landed_on_observed_runway"] == 0.8
    assert first["augmented"]["landed_on_observed_runway"] == 0.7
    assert second["traffic_output_over_residual"] == [0.01] and second["teacher_forced_nll"] == pytest.approx(0.281)
    assert first["real"]["by_stratum"] == {STRAIGHT_IN: {"sentences": 2, "lost_separation": 0.5},
                                           VECTORED: {"sentences": 6, "lost_separation": 0.5}}
    assert first["augmented"]["by_kind"] == {"A": {"sentences": 2, "lost_separation": 0.5}}
    assert "by_stratum" not in first["augmented"]
    # the last round: f3's sample 1 starts in a loss — left out of the shares, its go-around not counted
    last = readout["rounds"][LAST]["select"]["real"]
    assert last["by_stratum"][VECTORED] == {"sentences": 5, "lost_separation": 0.4}
    assert first["real"]["go_around_sentences"] == 1 and last["go_around_sentences"] == 0
    training = readout["rounds"][1]["training"]
    assert training["pass"] == {"kl_mean": 0.01} and training["sentences"]["scenes_with_contrast"] == 1
    assert training["signal"]["differing_scenes"] == 1
    real = readout["spread"]["real"]
    # f3 starts in a loss in the last round only, f4's labelled reading starts in one: both left out
    assert (real["left_out_starting_in_a_loss"], real["left_out_reference_starting_in_a_loss"]) == (1, 1)
    assert real["flights"] == 2
    # f2 differs in the middle round: only f1's two samples are the same in every round
    assert real["same_in_every_round"] == 0.5
    assert (real["paired_change"]["fixed"], real["paired_change"]["newly_lost"]) == (1, 0)
    # round 0: f1 lost both (its record lost, its labelled words did not), f2 one (the other way round)
    assert real["first_round"]["2/2"] == {"flights": 1, "flight_share": 0.5, "loss_share": 2 / 3, "vectored": 1.0,
                                          RECORDED: 1.0, LABELLED: 0.0}
    assert real["first_round"]["1/2"] == {"flights": 1, "flight_share": 0.5, "loss_share": 1 / 3, "vectored": 0.0,
                                          RECORDED: 0.0, LABELLED: 1.0}
    # the last round, not round 1 (where f2 lost one)
    assert real["last_round"]["0/2"]["flights"] == 1 and real["last_round"]["1/2"]["flights"] == 0
    assert readout["spread"]["augmented"]["first_round"]["1/2"] == {"flights": 1, "flight_share": 1.0,
                                                                    "loss_share": 1.0}
    with pytest.raises(SystemExit):
        _main(run, reference, out)


def test_artefact_name_is_the_place_under_outputs():
    assert artefact_name(f"/a/.claude/worktrees/b/{INSTRUCTIONS}") == artefact_name(INSTRUCTIONS) == \
        "POOLED/instruction_language/v5"
    with pytest.raises(SystemExit, match="not under"):
        artefact_name("/tmp/instructions")


def test_runner_refuses_an_unfinished_or_smoke_run(tmp_path):
    run, reference = _run(tmp_path)
    (run / f"round_{LAST + 1:02d}").mkdir()
    with pytest.raises(SystemExit, match="did not finish"):
        _main(run, reference, tmp_path / "a")
    (run / f"round_{LAST + 1:02d}").rmdir()
    config = json.loads((run / "config.json").read_text())
    (run / "config.json").write_text(json.dumps({**config, "rounds": LAST + 1}))
    with pytest.raises(SystemExit, match="not finished"):
        _main(run, reference, tmp_path / "b")
    (run / "config.json").write_text(json.dumps(config))
    (run / "choice.json").write_text(json.dumps({"round": 0, "augmented_reward": [0.6] * LAST}))
    with pytest.raises(SystemExit, match="cover rounds"):
        _main(run, reference, tmp_path / "c")
    (run / "config.json").write_text(json.dumps({**config, "smoke": True}))
    with pytest.raises(SystemExit, match="formal"):
        _main(run, reference, tmp_path / "d")


def test_runner_takes_the_samples_from_the_config(tmp_path):
    run, reference = _run(tmp_path)
    config = json.loads((run / "config.json").read_text())
    (run / "config.json").write_text(json.dumps({**config, "samples": 4}))
    with pytest.raises(ValueError, match="not 4 for each"):
        _main(run, reference, tmp_path / "out")


@pytest.mark.parametrize("change, message", [
    (lambda h, rows: ({**h, "augment_seed": 7919}, rows), "real select scenes"),
    (lambda h, rows: ({**h, "executor": {**EXECUTOR, "sha256": "f" * 64}}, rows), "another executor"),
    (lambda h, rows: ({**h, "instructions": INSTRUCTIONS.replace("v5", "v4")}, rows), "instruction artefact"),
    (lambda h, rows: (h, [r for r in rows if r["dataset_id"] != "f2"]), "no record / labelled"),
    (lambda h, rows: (h, rows + rows[:1]), "twice"),
    (lambda h, rows: (h, [{**r, "visual": {"outcome": LANDED}} if r["source"] == RECORDED else r for r in rows]),
     "recorded reading"),
])
def test_runner_refuses_a_reference_not_of_this_run(tmp_path, change, message):
    run, reference = _run(tmp_path)
    header = json.loads((reference / "free_generation.json").read_text())
    rows = [json.loads(line) for line in (reference / "flights.jsonl").read_text().splitlines()]
    header, rows = change(header, rows)
    (reference / "free_generation.json").write_text(json.dumps(header))
    (reference / "flights.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match=message):
        _main(run, reference, tmp_path / "out")
