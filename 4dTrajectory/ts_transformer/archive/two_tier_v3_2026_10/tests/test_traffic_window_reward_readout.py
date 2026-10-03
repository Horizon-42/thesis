"""Reading an M4-in-windows run (`experiments/traffic_window_reward_readout`, R39) from its files — a run written by the
runner's own pieces on the scene-data fixture: each round's select readout (`side_readout`, `traffic_readout`), a
training round's sentences (`round_summary`) and a real window pass moving the model, the history rows
(`history_row`) and the round choice (`write_choice`). The fixture's aircraft never land within their time limit, so
every round's select sentences would read alike: each round's stored sentences are given seeded outcomes of their own
(`_varied`) — the rounds then differ, as a run's do, and a wrong pair, round or kind shows. (After the side's numbers
are made: a landed sentence there would need a landing time the fixture has not. The readout copies those numbers and
pairs the sentences; the test checks each against its own source.)"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import torch

from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.tests.test_traffic_window_reward import _round, _speaking
from ts_transformer.tests.test_traffic_window_tuner import _reading_model

CPU = torch.device("cpu")


def _kinds(rows):
    """The augmented side's sentences: window 0's of kind A, window 1's of kind C."""
    for row in rows:
        row["kind"] = "A" if row["window"] == 0 else "C"


def _varied(rows, number, side):
    """Round ``number``'s stored select sentences on ``side`` given seeded outcomes: landed (reward 1), or lost
    separation — ended by a commanded or a replayed aircraft — or timed out (reward 0); the first sentence starting in a
    loss in round 1 only."""
    from ts_transformer.experiments.traffic_loop import LOST_SEPARATION

    rng = np.random.default_rng([7, number, side == "augmented"])
    for i, row in enumerate(rows):
        draw = rng.random()
        outcome = "landed" if draw < 0.4 else LOST_SEPARATION if draw < 0.8 else "timeout"
        row.update(outcome=outcome, reward=float(outcome == "landed"), starts_in_a_loss=number == 1 and i == 0,
                   ended_with=None if outcome != LOST_SEPARATION else ("commanded", "replayed")[int(rng.random() < 0.5)])


def _write_run(tmp_path, monkeypatch, rounds=2, events=False):
    """A window M4 run of ``rounds`` training rounds at ``tmp_path / "run"``, as the runner writes one (``events``: with
    a select hard-event side — here the select windows themselves, every aircraft read)."""
    from ts_transformer.experiments.traffic_rounds import traffic_readout
    from ts_transformer.experiments.traffic_scene_data import build_split
    from ts_transformer.experiments.traffic_window_reward import (
        SCHEMA, event_readout, history_row, round_summary, side_readout, window_split, write_choice,
    )
    from ts_transformer.experiments.traffic_window_tuner import WindowRewardTuner, window_advantages
    from ts_transformer.io_utils import write_json_atomic
    from ts_transformer.prior import data as prior_data
    from ts_transformer.prior.data import Split, column_classes
    from ts_transformer.prior.train import RewardConfig
    from ts_transformer.tests.test_prior_speaker import _model

    round_, airport, spec = _round(tmp_path, monkeypatch)
    speaking = _speaking(spec, airport)
    model = _reading_model(spec)
    tuner = WindowRewardTuner(model, _model(Words(spec), slots=2), RewardConfig(), CPU, seed=0,
                              traffic_learning_rate=3e-4, step_s=spec.step_s)
    table = Split([], ("KXXX",), prior_data.candidate_table({"KXXX": airport.flights.geometry}, ("KXXX",), 2),
                  (("09",),), ((90.0,),), column_classes(Words(spec), 2), "no-context")
    data, _ = build_split(tmp_path / "artefact", "train", spec, ("KXXX",), None, 2_048)
    built = [b for b in data if b.sample.asks]
    out = tmp_path / "run"
    out.mkdir()
    write_json_atomic(out / "config.json", {"schema": SCHEMA, "rounds": rounds})

    def spoken(source, seed):
        plan = speaking.plan(round_, 2)
        return speaking.assemble(round_, 2, plan, {n: speaking.speak(model, round_, chunk, n, 2, seed=seed,
                                                                      source=source) for n, chunk in enumerate(plan)})

    history = []
    for number in range(rounds + 1):
        more = {}
        if number:
            sentences = spoken("train", 10 + number)
            for row in sentences.rows:     # a contrast for every aircraft the runner can train (f1 starts in a loss)
                row["reward"] = float(row["sample"] == 0 and not row["starts_in_a_loss"])
            advantages, gains, trained = window_advantages(sentences.rows, 2)
            split, _ = window_split(round_, sentences, advantages, gains, trained, table, None, spec.step_s)
            start = {"real": tuner.window_distance(split), "augmented": None}
            tuner.restart(np.random.default_rng([0, number]))
            more = {"train_pass": {**tuner.window_pass(split, data, slots=2), "distance_at_start": start},
                    "sentences": round_summary(round_, sentences, trained, gains)}
        model.eval()
        readout = {}
        for side in ("real", "augmented"):
            select = spoken("scene", 5)
            if side == "augmented":
                _kinds(select.rows)
            readout[side] = side_readout(select, round_, spec.step_s, real=side == "real")
            _varied(readout[side]["aircraft"], number, side)
        if events:
            select = spoken("scene", 5)
            _varied(select.rows, number, "events")
            for row in select.rows:
                row["landed_here"] = row["outcome"] == "landed"
            readout["events"] = event_readout(select)
        readout["traffic"] = traffic_readout(model, built, 2, CPU)
        history.append(history_row(number, readout, **more))
        (out / f"round_{number:02d}").mkdir()
        write_json_atomic(out / "history.json", {"rounds": history})
        write_json_atomic(out / f"round_{number:02d}" / "readout.json", readout)
    for row in history:                                  # (the fixture's aircraft land after their time limit)
        row["real"]["ordering"] = {"time_ratio": 1.0, "gap_ratio": 1.0}
        row["real"]["free_generation"]["all"]["landed_on_observed_runway"] = 1.0
    write_choice(out, history, {side: {name: 1.0 for name in COLUMNS} for side in ("real", "augmented")},
                 tmp_path / "start", lambda line: None)
    return out


def test_a_run_is_read_round_by_round_from_its_files_paired_as_the_round_choice_pairs(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
    from ts_transformer.experiments.traffic_window_reward_readout import read_run

    out = _write_run(tmp_path, monkeypatch)
    read = read_run(out)
    assert read["finished_rounds"] == 2 and read["not_finished"] == [] and [r["round"] for r in read["rounds"]] == [0, 1, 2]
    history = json.loads((out / "history.json").read_text())["rounds"]
    choice = json.loads((out / "choice.json").read_text())
    assert any(x["difference"] != 0.0 for x in choice["against_round_0"])      # the rounds differ
    readouts = [json.loads((out / f"round_{n:02d}" / "readout.json").read_text()) for n in range(3)]

    def by_hand(side, n, value, kind=None):
        """Round n against round 0, paired by (window, aircraft, sample) over the sentences neither starts in a loss."""
        pick = [{(r["window"], r["dataset_id"], r["sample"]): value(r) for r in readouts[m][side]["aircraft"]
                 if not r["starts_in_a_loss"] and kind in (None, r["kind"])} for m in (0, n)]
        keys = sorted(pick[0].keys() & pick[1].keys())
        a, b = np.array([pick[0][k] for k in keys]), np.array([pick[1][k] for k in keys])
        return float(np.mean(b - a)), float(np.sqrt(np.sum(a != b)) / len(keys)), len(keys)

    def lost(r):
        return float(r["outcome"] == LOST_SEPARATION)

    for n, row in enumerate(read["rounds"]):
        for side in ("real", "augmented"):
            part = row["select"][side]
            assert part["reward"] == readouts[n][side]["reward"] == history[n][side]["reward"]
            assert part["lost_separation"] == readouts[n][side]["separation"]["lost_separation"]
            for name, value in (("reward", lambda r: r["reward"]), ("lost_separation", lost)):
                change = part["against_round_0"][name]
                assert (change["difference"], change["standard_error"], change["sentences"]) == pytest.approx(
                    by_hand(side, n, value))
            counted = [r for r in readouts[n][side]["aircraft"] if not r["starts_in_a_loss"]]
            losing = [r for r in counted if r["outcome"] == LOST_SEPARATION]
            assert part["losses_with"]["sentences"] == len(losing)
            if losing:
                assert part["losses_with"]["commanded"] == sum(r["ended_with"] == "commanded" for r in losing) / len(losing)
        # the augmented reward's pairs are the round choice's; per kind by hand
        change = row["select"]["augmented"]["against_round_0"]["reward"]
        assert (change["difference"], change["standard_error"]) == pytest.approx(
            (choice["against_round_0"][n]["difference"], choice["against_round_0"][n]["standard_error"]))
        kinds = row["select"]["augmented"]["against_round_0_by_kind"]
        assert kinds.keys() == {"A", "C"}
        for kind in ("A", "C"):
            assert (kinds[kind]["reward"]["difference"], kinds[kind]["reward"]["standard_error"],
                    kinds[kind]["reward"]["sentences"]) == pytest.approx(by_hand("augmented", n, lambda r: r["reward"], kind))
        assert row["select"]["traffic"]["traffic_output_over_residual"] == readouts[n]["traffic"]["traffic_output_over_residual"]
        if n == 0:
            assert row["training"] is None and change["difference"] == 0.0 and change["standard_error"] == 0.0
        else:
            assert row["training"]["pass"] == {k: history[n]["train_pass"][k] for k in row["training"]["pass"]}
            assert row["training"]["sentences"]["trained_on"] == history[n]["sentences"]["trained_on"] == 4
    # round 1's first sentence starts in a loss: out of its pairs, in round 1 only
    assert read["rounds"][1]["select"]["real"]["against_round_0"]["reward"]["sentences"] == 5
    assert read["rounds"][2]["select"]["real"]["against_round_0"]["reward"]["sentences"] == 6
    # each training round read from its own history row (the passes differ), the model moved by them
    assert read["rounds"][1]["training"]["pass"]["kl_mean"] != read["rounds"][2]["training"]["pass"]["kl_mean"]
    assert (read["rounds"][2]["select"]["traffic"]["traffic_output_over_residual"]
            != read["rounds"][0]["select"]["traffic"]["traffic_output_over_residual"])
    assert read["choice"]["covers_rounds"] == 2 and read["choice"]["round"] == choice["round"]

    # a round running (its directory, no readout yet) is named and not read
    running = tmp_path / "running"
    shutil.copytree(out, running)
    (running / "round_03").mkdir()
    assert read_run(running)["not_finished"] == ["round_03"] and read_run(running)["finished_rounds"] == 2
    # a gap in the finished rounds, or a finished round without its history row, is refused
    gap = tmp_path / "gap"
    shutil.copytree(out, gap)
    (gap / "round_01" / "readout.json").unlink()
    with pytest.raises(SystemExit, match="not 0 … k"):
        read_run(gap)
    short = tmp_path / "short"
    shutil.copytree(out, short)
    (short / "history.json").write_text(json.dumps({"rounds": history[:2]}))
    with pytest.raises(SystemExit, match="no row for the finished round"):
        read_run(short)


def test_a_run_s_select_hard_events_are_read_paired_against_round_0(tmp_path, monkeypatch, capsys):
    """Multi-aircraft design §6.6 step 8 item 11: a run with ``--select-events`` — its hard-event side read per round and
    paired against round 0 as the other sides are, and printed."""
    from ts_transformer.experiments.traffic_window_reward_readout import main, read_run

    out = _write_run(tmp_path, monkeypatch, rounds=1, events=True)
    read = read_run(out)
    readouts = [json.loads((out / f"round_{n:02d}" / "readout.json").read_text()) for n in range(2)]
    for n, row in enumerate(read["rounds"]):
        part = row["select"]["events"]
        assert part["reward"] == readouts[n]["events"]["reward"] and part["sentences"] == len(readouts[n]["events"]["aircraft"])
        rows_0, rows = readouts[0]["events"]["aircraft"], readouts[n]["events"]["aircraft"]
        keys = [(r["window"], r["dataset_id"], r["sample"]) for r in rows if not r["starts_in_a_loss"]]
        first = {(r["window"], r["dataset_id"], r["sample"]): r["reward"] for r in rows_0 if not r["starts_in_a_loss"]}
        now = {(r["window"], r["dataset_id"], r["sample"]): r["reward"] for r in rows if not r["starts_in_a_loss"]}
        both = sorted(first.keys() & now.keys())
        assert part["against_round_0"]["reward"]["sentences"] == len(both) and keys
        assert part["against_round_0"]["reward"]["difference"] == pytest.approx(
            float(np.mean([now[k] - first[k] for k in both])))
    assert main(["--run", str(out)]) == 0
    assert "select hard events" in capsys.readouterr().out


def test_the_readout_prints_and_writes_into_a_new_directory_only(tmp_path, monkeypatch, capsys):
    from ts_transformer.experiments.traffic_window_reward_readout import SCHEMA, main

    out = _write_run(tmp_path, monkeypatch, rounds=1)
    assert main(["--run", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "rounds 0 … 1 finished (the last invocation asked for rounds up to 1)" in printed and "choice over rounds 0 … 1" in printed and "BEHIND" not in printed
    assert "select hard events" not in printed                      # a run without them prints none
    # a choice behind the rounds finished (a resumed run between invocations) is said to be; none yet, said so
    choice = json.loads((out / "choice.json").read_text())
    (out / "choice.json").write_text(json.dumps({**choice, "against_round_0": choice["against_round_0"][:1]}))
    main(["--run", str(out)])
    assert "BEHIND the 1 rounds finished" in capsys.readouterr().out
    (out / "choice.json").unlink()
    main(["--run", str(out)])
    assert "no choice yet" in capsys.readouterr().out
    target = tmp_path / "readout"
    assert main(["--run", str(out), "--out", str(target)]) == 0
    written = json.loads((target / "traffic_window_reward_readout.json").read_text())
    assert written["schema"] == SCHEMA and written["finished_rounds"] == 1
    with pytest.raises(SystemExit):
        main(["--run", str(out), "--out", str(target)])            # never overwritten
