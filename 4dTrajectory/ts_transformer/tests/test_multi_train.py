"""Stage D, MC4: the campaign of stage D (multi-aircraft control §3–§5, §7; `experiments/multi_train.py`) — its draw and
its select windows on the synthetic stream of MC1's tests, and one round end to end from a round of stage C's campaign on
the synthetic artefact of stage C's campaign tests."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.experiments import multi_train, post_train
from ts_transformer.experiments.multi_train import (
    MULTI_CAMPAIGN_SCHEMA, MULTI_CHECKPOINT_SCHEMA, MULTI_CLAIM_READER, MultiSettings, draw_windows,
    kind_of, selection_windows, stage_d, window_record,
)
from ts_transformer.multi.separation import PAIRS
from ts_transformer.multi.tokens import PART_FEATURES, TOKENS_SCHEMA
from ts_transformer.multi.windows import COMPRESSED, REAL_KIND, left_out
from ts_transformer.tests.support import INSTRUCTION_STEP_S
from ts_transformer.tests.test_multi_windows import built  # noqa: F401
from ts_transformer.tests.test_post_branches import _ahead
from ts_transformer.tests.test_post_train import _context, _settings, short_round
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401


def _multi_settings(**changed):
    values = dict(rounds=1, per_kind={REAL_KIND: 1, COMPRESSED: 0}, span_s=0.0, c_min=0.6, batch_windows=2,
                  continuations=2, seed=2024, prior_lr=1e-4, traffic_lr=1e-3, weight_decay=0.0, update_groups=1,
                  data_sentences=1, select_per_airport=1,
                  start={"campaign": "c", "round": 0, "checkpoint_sha256": "0" * 64})
    return MultiSettings(**{**values, **changed})


def _stream_context(built):
    """What the draw and the select windows read of a context, on the synthetic stream (its train split read as the
    select days too)."""
    windows, separation, fin = built
    code = windows[0].scene.geometry.code
    split = {"windows": windows}
    return SimpleNamespace(splits={"train": split, "select": split}, separations={code: separation},
                           finals={code: fin}, words=SimpleNamespace(spec=SimpleNamespace(step_s=INSTRUCTION_STEP_S)))


def test_the_settings_refuse_what_is_not_a_campaign_of_stage_d():
    for wrong in ({"per_kind": {REAL_KIND: 1}}, {"per_kind": {REAL_KIND: 0, COMPRESSED: 0}}, {"span_s": -1.0},
                  {"c_min": 1.0}, {"rounds": 0}, {"start": {"campaign": "c"}}):
        with pytest.raises(ValueError):
            _multi_settings(**wrong)


def test_a_rounds_draw_of_stage_d_windows(built):
    """D146: anchors in the round's permutation, each one's window of span L; a compressed window moves its later
    aircraft (a window without one has no compressed form, counted); a window left out is counted, never drawn; a
    shortfall is recorded; the same seed and round draw the same windows."""
    context = _stream_context(built)
    settings = _multi_settings(per_kind={REAL_KIND: 4, COMPRESSED: 4}, span_s=300.0)
    windows, drawn = draw_windows(context, settings, np.random.default_rng([1337, 0]))
    again, _ = draw_windows(context, settings, np.random.default_rng([1337, 0]))
    assert [window_record(w) for w in windows] == [window_record(w) for w in again]
    real = [w for w in windows if kind_of(w) == REAL_KIND]
    moved = [w for w in windows if kind_of(w) == COMPRESSED]
    assert drawn["drawn"][REAL_KIND] == 4 and len(real) >= 4 - drawn["drawn"][COMPRESSED]
    assert {w.commanded.key for w in windows[:4]} == {"KXXX:a", "KXXX:b", "KXXX:c", "KXXX:far"}
    assert drawn["no_later_aircraft"][COMPRESSED] >= 2                     # c and far: none after them within 300 s
    assert drawn["shortfall"] == {COMPRESSED: 4 - drawn["drawn"][COMPRESSED]}
    _, separation, fin = built
    for w in windows:
        assert not left_out(w, separation, fin, INSTRUCTION_STEP_S)
    for w in moved:
        assert w.joined and all(item.shift_s <= 0.0 for item in w.joined) and any(item.shift_s for item in w.joined)
    record = window_record(windows[0])
    assert set(record) == {"anchor", "row0_s", "commanded", "shifts_s", "kind"}
    assert record["commanded"][0] == record["anchor"] and len(record["shifts_s"]) == len(record["commanded"])


def test_the_select_windows_are_drawn_once_per_airport_and_never_left_out(built):
    """§5 item 2: at most ``select_per_airport`` real windows of span L of each airport, drawn with the seed, in the
    order of their anchors, none left out; the same every time."""
    context = _stream_context(built)
    settings = _multi_settings(span_s=300.0, select_per_airport=2)
    first, second = selection_windows(context, settings), selection_windows(context, settings)
    assert [window_record(w) for w in first] == [window_record(w) for w in second] and len(first) == 2
    assert [w.row0_s for w in first] == sorted(w.row0_s for w in first)
    assert all(kind_of(w) == REAL_KIND for w in first)
    assert len(selection_windows(context, _multi_settings(span_s=300.0, select_per_airport=10))) == 4


def _stage_c(s, tmp_path, context, *, smoke=False):
    """A stage C campaign of one round (seed 1337, `short_round`'s window) and the start of round 0 (`start_of`)."""
    source = tmp_path / ("stage_c_smoke" if smoke else "stage_c")
    settings = _settings(rounds=1, continuations=2)
    post_train.open_campaign(source, {"settings": asdict(settings), "smoke": smoke}, {"head": "x", "dirty": False}, {})
    post_train.run_campaign(source, settings, context)
    return post_train.start_of(source, 0, formal=False)


def test_a_round_of_stage_d_from_a_round_of_stage_c_end_to_end(setup, tmp_path, monkeypatch):
    """MC4, D164: one round of stage D (its draw replaced by window A, lost at its first row, as stage C's round test
    does) from round 0 of a stage C campaign: its start is that round through `round_start`, with stage D's token part at
    zero (the module's other weights the round's), its record names its windows by anchor, commanded flights and shifts,
    its readout reads the real window of the select days by kind and pair, and its checkpoint's identity is stage D's
    (its start the round's identity); a resume of the campaign (its rounds raised) continues it from its checkpoint."""
    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    start = _stage_c(s, tmp_path, context)
    window = _ahead(s["windows"][0])
    monkeypatch.setattr(multi_train, "draw_windows",
                        lambda context, settings, rng: ([window], {"drawn": {REAL_KIND: 1}}))
    stage = stage_d()
    settings = _multi_settings(rounds=2, start=start)
    model, _ = stage.start(context, settings)
    round_c, identity_c = post_train.round_start(context, start, seed=2024)
    parts = {name: value for name, value in model.state_dict().items()}
    assert {n for n in parts if ".part." in n} and all(not parts[n].any() for n in parts if ".part." in n)
    assert all(torch.equal(parts[n], v) for n, v in round_c.state_dict().items())        # the round's weights
    shaped, _ = stage.start_model(context, settings)
    assert shaped.state_dict().keys() == model.state_dict().keys()                     # what a resume loads into
    out = tmp_path / "stage_d"
    options = dict(schema=MULTI_CAMPAIGN_SCHEMA, reader=MULTI_CLAIM_READER)

    def inputs(rounds):
        paths = {key: str(tmp_path / "inputs" / key) for key in post_train.INPUT_PATHS}
        return {**paths, "settings": asdict(_multi_settings(rounds=rounds, start=start)), "smoke": True}

    post_train.open_campaign(out, inputs(1), {"head": "x", "dirty": False}, {}, **options)
    post_train.run_campaign(out, _multi_settings(rounds=1, start=start), context, stage=stage)
    record = json.loads((out / "round_0" / "round.json").read_text())
    assert record["windows"] == [window_record(window)]
    assert record["speaking"]["windows"] == 1 and record["speaking"]["outcomes"] == {"lost_separation": 1}
    readout = record["selection_readout"][s["geometry"].code]
    assert readout["all"]["windows"] == 1 and readout["all"]["aircraft"] == 1
    assert set(readout["all"]["loss_steps"]) == set(PAIRS)
    assert readout["reward_mean"] == readout["all"]["reward_mean"] == readout[REAL_KIND]["reward_mean"]
    assert {"delays_s", "spacing_s", "record_spacing_s", "order"} <= set(readout["all"])        # O18's readouts
    identity = torch.load(out / "round_0" / "checkpoint.pt", weights_only=False)["identity"]
    assert identity["schema"] == MULTI_CHECKPOINT_SCHEMA and identity["start"] == identity_c
    assert identity["token_part"] == {"schema": TOKENS_SCHEMA, "features": list(PART_FEATURES)}
    assert identity["rounds"] == [record["windows"]] and identity["span_s"] == 0.0
    assert json.loads((out / "campaign.json").read_text())["schema"] == MULTI_CAMPAIGN_SCHEMA
    post_train.open_campaign(out, inputs(2), {"head": "y", "dirty": False}, {},
                             **options)                                   # raised to two rounds: the resume goes on
    post_train.run_campaign(out, _multi_settings(rounds=2, start=start), context, stage=stage_d())
    assert post_train.done_rounds(out) == 2
    resumed = torch.load(out / "round_1" / "checkpoint.pt", weights_only=False)["identity"]
    assert resumed["start"] == identity_c and len(resumed["rounds"]) == 2      # a resumed stage reads the start again
    with pytest.raises(SystemExit, match="a ts-post-train-v1 campaign, not"):
        post_train.open_campaign(tmp_path / "stage_c", {"settings": {}}, {"head": "x", "dirty": False}, {}, **options)


def test_stage_ds_start_is_refused_by_name_through_the_one_function(setup, tmp_path, monkeypatch):
    """D164 (post-training §9 item 12): stage D's start refuses by name other bytes than the recorded ones, a checkpoint
    of another base or of another round, a smoke source under a formal campaign, and the source's seed."""
    import shutil

    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    start = _stage_c(s, tmp_path, context)
    stage = stage_d()
    settings = _multi_settings(start=start)
    with pytest.raises(ValueError, match="not the checkpoint the campaign started from"):
        stage.start(context, _multi_settings(start={**start, "checkpoint_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="another base, masks, traffic shape or round"):
        stage.start(replace(context, base_identity={"base": "another"}), settings)
    with pytest.raises(ValueError, match=r"takes another seed.*\(D162\)"):
        stage.start(context, _multi_settings(start=start, seed=1337))
    smoke = _stage_c(s, tmp_path, context, smoke=True)
    with pytest.raises(ValueError, match="is a smoke campaign: a formal campaign does not start from it"):
        stage.start(replace(context, formal=True), _multi_settings(start=smoke))
    later = tmp_path / "later"                                     # round 0's place holding round 1's checkpoint
    post_train.open_campaign(later, {"settings": asdict(_settings(rounds=2, continuations=2)), "smoke": False},
                             {"head": "x", "dirty": False}, {})
    post_train.run_campaign(later, _settings(rounds=2, continuations=2), context)
    shutil.copy(later / "round_1" / "checkpoint.pt", later / "round_0" / "checkpoint.pt")
    with pytest.raises(ValueError, match="another base, masks, traffic shape or round"):
        stage.start(context, _multi_settings(start=post_train.start_of(later, 0, formal=False)))


def test_stage_ds_round_with_speaking_workers_is_the_round_of_one_process(setup, tmp_path, monkeypatch):
    """C13 for stage D: its round spoken and read by two workers running its stage gives the speaking record and the
    readout of one process (the workers fork with stage D's parts)."""
    from ts_transformer.experiments.post_train import Speakers

    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    start = _stage_c(s, tmp_path, context)
    window = _ahead(s["windows"][0])
    monkeypatch.setattr(multi_train, "draw_windows",
                        lambda context, settings, rng: ([window], {"drawn": {REAL_KIND: 1}}))
    stage = stage_d()
    settings = _multi_settings(start=start)
    options = dict(schema=MULTI_CAMPAIGN_SCHEMA, reader=MULTI_CLAIM_READER)
    records = []
    for name, workers in (("here", 0), ("there", 2)):
        out = tmp_path / name
        post_train.open_campaign(out, {"settings": asdict(settings)}, {"head": "x", "dirty": False}, {}, **options)
        speakers = Speakers(context, settings, workers, torch.device("cpu"), stage=stage) if workers else None
        try:
            post_train.run_campaign(out, settings, context, speakers, stage=stage)
        finally:
            if speakers is not None:
                speakers.close()
        record = json.loads((out / "round_0" / "round.json").read_text())
        records.append((record["speaking"], record["selection_readout"]))
    assert records[0] == records[1]


def test_the_readouts_losses_by_pair_are_the_loops_on_a_window_of_several_aircraft(setup):
    """§5 item 4 (the review of MC4, S2-4): on the window of three commanded aircraft (each lost against another), the
    readout's judgement again on the states flown (`window_losses_of`) finds, at the step of each loss the loop charged,
    that same pair; its counts hold commanded–commanded losses."""
    from ts_transformer.multi.census import flown_positions
    from ts_transformer.multi.separation import judged_step
    from ts_transformer.post.runways import airport_separation
    from ts_transformer.tests.test_post_window_loop import _with_module
    from ts_transformer.tests.test_post_window_multi import JOINS, _multi, _numbers

    s = setup
    multi, loop_of = _multi(s)
    loop = loop_of(_with_module(s["base"]))
    results = loop.run(_numbers(len(JOINS)))
    code = s["geometry"].code
    context = SimpleNamespace(words=s["words"], separations={code: airport_separation(s["geometry"])},
                              finals=s["finals"])
    losses, steps = multi_train.window_losses_of(multi, results, loop.speaking.start, context)
    assert steps > 0 and losses["commanded_commanded"] >= 1
    positions = flown_positions(multi, [(r.states, r.words, loop.speaking.start, None) for r in results], s["words"])
    for record, r in zip(multi.commanded_all, results):
        assert r.loss is not None
        aircraft, _, found = judged_step(multi, positions, r.loss_step, context.separations[code], s["finals"][code],
                                         s["words"].spec.step_s)
        pairs = {frozenset((aircraft.keys[x.i], aircraft.keys[x.j])) for x in found}
        assert frozenset((record.key, r.other)) in pairs, (record.key, r.loss_step)


def test_a_landed_aircraft_is_judged_once_over_its_threshold_on_the_states_flown(built):
    """Requests item 7 in the readout (the review of MC4, S2-1): `flown_positions` judges a landed aircraft at the
    first step at or after its last row, over its threshold at its last state on its landed runway, and never after;
    at the steps before, at its states; an aircraft not landed ends at its last row."""
    from ts_transformer.multi.census import flown_positions

    windows, _, _ = built
    window = windows[0]
    words = SimpleNamespace(spec=SimpleNamespace(step_s=INSTRUCTION_STEP_S))
    row0 = window.commanded.first_step_s - 16.0                         # OBSERVATION_S: the record's row 0
    for rows, over in ((7, 6), (8, 8)):
        states = np.arange(rows * 6, dtype=float).reshape(rows, 6)
        landed = flown_positions(window, [(states, np.zeros((0, 5), dtype=np.int64), 0, 1)], words)
        assert landed.end_s == row0 + over * INSTRUCTION_STEP_S
        at = landed.at(0, row0 + over * INSTRUCTION_STEP_S)
        assert at[1] == tuple(states[-1, :3]) and at[2] == tuple(states[-2, :3]) and at[4] == 1 and at[6] is True
        assert landed.at(0, row0 + (over + 2) * INSTRUCTION_STEP_S) is None
        before = landed.at(0, row0 + 2 * INSTRUCTION_STEP_S)
        assert before[1] == tuple(states[2, :3]) and before[6] is False
        plain = flown_positions(window, [(states, np.zeros((0, 5), dtype=np.int64), 0, None)], words)
        assert plain.end_s == row0 + (rows - 1) * INSTRUCTION_STEP_S
        assert plain.at(0, row0 + 8 * INSTRUCTION_STEP_S) is None and plain.at(0, row0 + 6 * INSTRUCTION_STEP_S)[6] is False


def test_the_runner_refuses_a_smoke_source_under_a_formal_campaign_before_anything_opens(tmp_path, capsys):
    """D164 (the review of MC4, S2-4): `multi_train --start-campaign --start-round` refuses by name, before anything is
    opened, a smoke source under a formal campaign and a round not done."""
    source = tmp_path / "source"
    (source / "round_0").mkdir(parents=True)
    (source / "round_0" / "checkpoint.pt").write_bytes(b"weights")
    (source / "campaign.json").write_text(json.dumps({"schema": post_train.CAMPAIGN_SCHEMA, "inputs": {"smoke": True}}))
    argv = ["--prior", str(tmp_path / "p"), "--instructions", str(tmp_path / "i"), "--executor", str(tmp_path / "e"),
            "--windows", str(tmp_path / "w"), "--out", str(tmp_path / "campaign"), "--rounds", "1", "--span-s", "0",
            "--c-min", "0.6", "--batch-windows", "1", "--seed", "2024", "--prior-lr", "1e-4", "--traffic-lr", "1e-3",
            "--weight-decay", "0", "--update-groups", "1", "--data-sentences", "1", "--select-per-airport", "1",
            "--windows-real", "1", "--windows-compressed", "0", "--start-campaign", str(source)]
    with pytest.raises(SystemExit):
        multi_train.main(argv + ["--start-round", "0"])
    assert "is a smoke campaign: a formal campaign does not start from it" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        multi_train.main(argv + ["--start-round", "1", "--smoke"])
    assert "holds the checkpoints of rounds 0–0, not 1" in capsys.readouterr().err


def test_a_landed_aircraft_counts_in_the_readout_only_where_the_loop_judges_it(built):
    """The review of MC4, round 2: `window_losses_of` judges a landed aircraft over its threshold only in a window of
    several commanded aircraft (the loop's rule, `_landed`), where it counts as a recorded one (`classify`'s ``over``:
    it answers for nothing); a window of one commanded aircraft ignores its crossing."""
    from ts_transformer.multi.separation import classify
    from ts_transformer.multi.windows import Anchors

    windows, separation, fin = built
    code = windows[0].scene.geometry.code
    context = SimpleNamespace(words=SimpleNamespace(spec=SimpleNamespace(step_s=INSTRUCTION_STEP_S)),
                              separations={code: separation}, finals={code: fin})

    def ends(window, crossing):
        rows = 8                                                  # its last row odd: the step over it one row later
        states = np.zeros((rows, 6))
        states[:, 0] = np.arange(rows) * 150.0 - 30_000.0
        return [SimpleNamespace(states=states, words=np.zeros((0, 5), dtype=np.int64), crossing=crossing)
                for _ in window.commanded_all]

    landed = {"runway_index": 0}
    alone = windows[3]
    assert multi_train.window_losses_of(alone, ends(alone, landed), 0, context) == \
        multi_train.window_losses_of(alone, ends(alone, None), 0, context)
    several = Anchors(windows).window_of(windows[0], 300.0)
    with_crossing = multi_train.window_losses_of(several, ends(several, landed), 0, context)
    without = multi_train.window_losses_of(several, ends(several, None), 0, context)
    assert with_crossing[1] > without[1]                                     # the step over the threshold judged
    assert classify(0, 2, (2,), 2, frozenset({0})) is None                   # landed and recorded: nobody's
    assert classify(0, 1, (1,), 2, frozenset({0})) == "commanded_answers"    # its follower answers, as behind a recorded one
    assert classify(0, 1, (1,), 2) == "commanded_commanded"


def test_the_runner_records_stage_ds_campaign_and_opens_its_context_formal_or_smoke(tmp_path, monkeypatch):
    """§7 row 4 (the review of MC4, S2-4 b): `multi_train.main` opens its context formal unless ``--smoke`` (D132,
    D164), checks the start before its campaign record, records `MULTI_CAMPAIGN_SCHEMA` with stage D's claim reader,
    and runs stage D's stage."""
    from dataclasses import dataclass

    @dataclass
    class FakeContext:                                             # what main replaces: its device and its base
        device: torch.device
        base: torch.nn.Module

    calls = []
    start = {"campaign": "c", "round": 0, "checkpoint_sha256": "0" * 64}
    fake_stage = SimpleNamespace(start=lambda context, settings: calls.append(("start", settings.start)))
    monkeypatch.setattr(multi_train, "start_of", lambda source, round_, formal: (calls.append(("start_of", formal)),
                                                                                    start)[1])
    monkeypatch.setattr(multi_train, "git_state", lambda: {"head": "x", "dirty": False})
    monkeypatch.setattr(multi_train, "require_conforming_closed_loop", lambda i, e: (None, {"checks": {}}, None))
    monkeypatch.setattr(multi_train, "checked_edges", lambda reference: None)
    monkeypatch.setattr(multi_train, "open_context", lambda *a, formal, **k: (calls.append(("context", formal)),
                                                                              FakeContext(torch.device("cpu"), torch.nn.Linear(1, 1)))[1])
    monkeypatch.setattr(multi_train, "stage_d", lambda: fake_stage)
    monkeypatch.setattr(multi_train, "open_campaign", lambda out, inputs, git, checks, **k: calls.append(("campaign", k)))
    monkeypatch.setattr(multi_train, "done_rounds", lambda out: 0)
    monkeypatch.setattr(multi_train, "run_campaign", lambda out, settings, context, speakers, stage:
                        calls.append(("run", stage is fake_stage)))
    argv = ["--prior", str(tmp_path / "p"), "--instructions", str(tmp_path / "i"), "--executor", str(tmp_path / "e"),
            "--windows", str(tmp_path / "w"), "--out", str(tmp_path / "campaign"), "--rounds", "1", "--span-s", "0",
            "--c-min", "0.6", "--batch-windows", "1", "--seed", "2024", "--prior-lr", "1e-4", "--traffic-lr", "1e-3",
            "--weight-decay", "0", "--update-groups", "1", "--data-sentences", "1", "--select-per-airport", "1",
            "--windows-real", "1", "--windows-compressed", "0", "--start-campaign", str(tmp_path / "c"),
            "--start-round", "0", "--device", "cpu"]
    for extra, formal in (([], True), (["--smoke"], False)):
        calls.clear()
        assert multi_train.main(argv + extra) == 0
        assert calls == [("start_of", formal), ("context", formal), ("start", start),
                         ("campaign", {"schema": MULTI_CAMPAIGN_SCHEMA, "reader": MULTI_CLAIM_READER}), ("run", True)]


def test_the_time_the_aircraft_take(built):
    """§5 item 4, O18 (`multi.timing`; the review of MC4's timing): each landing's delay against its record; the spacing
    at the threshold of successive landings on one runway that hold a commanded aircraft, the recorded aircraft landing
    between them counted, in the loop and on the records; the pairs landed in the recorded order; a silent aircraft's
    crossing is no landing; the readout of a window holds them."""
    from ts_transformer.multi import timing
    from ts_transformer.multi.windows import Anchors

    windows, _, _ = built
    window = Anchors(windows).window_of(windows[0], 130.0)                # a and b commanded, c recorded
    records = window.commanded_all
    c = next(f for f in window.scene.flights if f.key == "KXXX:c")
    cycle_s = 1.0

    def at(record, delay_s):
        return {"runway_index": 0, "at_row": (record.landing_s + delay_s - record.first_step_s) / cycle_s}

    # a lands 150 s late (after b, before c): the order of the pair reversed; b 10 s early
    crossings = [at(records[0], 150.0), at(records[1], -10.0)]
    items = timing.landed(records, crossings, cycle_s)
    assert [x.key for x in items] == ["KXXX:a", "KXXX:b"]
    assert np.allclose(timing.delays(items), [150.0, -10.0]) and timing.order(items) == (0, 1)
    recorded = timing.recorded_landings(window, max(records[0].landing_s + 150.0, records[1].landing_s))
    # c (in the air while the window flies) is its recorded aircraft; far (2 h later, landing far after) is not
    assert recorded == [(c.runway_index, c.landing_s, False)] and c.landing_s - records[1].landing_s == 122.0
    # in the loop: b (−10 s), a (+150 s), c; on the records: a, b, c 120 s and 122 s apart
    assert np.allclose(timing.loop_spacings(items, recorded),
                       [records[0].landing_s + 150.0 - (records[1].landing_s - 10.0), c.landing_s - records[0].landing_s - 150.0])
    assert np.allclose(timing.record_spacings(records, recorded), [120.0, 122.0])
    assert timing.spacings([(0, 10.0, True), (1, 5.0, False), (0, 40.0, False), (1, 15.0, False), (0, 70.0, False)]) \
        == [30.0]                                                             # gaps that hold a commanded aircraft only
    assert timing.quantiles([], (50,)) == {"n": 0} and timing.quantiles([1.0, 3.0], (50,))["p50"] == 2.0
    ends = [SimpleNamespace(crossing=crossings[0], outcome="landed"),
            SimpleNamespace(crossing=crossings[1], outcome="lost_separation")]          # silent b crossed: no landing
    out = multi_train.window_timing(window, ends, cycle_s)
    assert np.allclose(out["delays_s"], [150.0]) and out["order"] == [0, 0]
    # every spacing within the window's own time (none to "far", 2 h later, or to another day)
    assert out["spacing_s"] and max(out["spacing_s"] + out["record_spacing_s"]) < 400.0
    far = windows[3]                                   # alone, 2 h after c: no neighbour from outside its own time
    assert multi_train.window_timing(far, [SimpleNamespace(crossing=None, outcome="timeout")], cycle_s)[
        "record_spacing_s"] == []


def test_the_loops_crossing_time_is_the_timing_readouts(setup, monkeypatch):
    """The review of MC4's timing (S3): the time of a loop landing that `multi.timing.landed` reads is the one the window
    loop adds to the other aircraft's landings (`WindowLoop._landed`, D147)."""
    from ts_transformer.experiments.post_window_loop import LANDED
    from ts_transformer.multi import timing
    from ts_transformer.tests.test_post_window_loop import _with_module
    from ts_transformer.tests.test_post_window_multi import _multi

    s = setup
    _, loop_of = _multi(s)
    loop = loop_of(_with_module(s["base"]))
    crossing = {"at_row": 37.5, "runway_index": 0, "cross_m": 0.0, "height_m": 15.0}
    monkeypatch.setattr(loop.speaking.loop, "outcome", lambda b: SimpleNamespace(outcome=LANDED, crossing=crossing))
    loop._landed(np.array([True, False, False]))
    (added,) = [x for x in loop.landings[1].landings if x.flight_key == loop.keys[0]]
    (item,) = timing.landed(loop.records[:1], [crossing], loop.speaking.loop.params.cycle_s)
    assert item.loop_s == added.time_s


def test_a_round_of_stage_d_is_refused_unless_its_identity_is_the_campaigns(setup, tmp_path, monkeypatch):
    """§7 row 3 (the review of MC4's timing, S2-3): `round_model` refuses a round whose identity is not the campaign's
    — another seed, span, c_min, base — and a round not of that count; the campaign's own round opens."""
    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    start = _stage_c(s, tmp_path, context)
    window = _ahead(s["windows"][0])
    monkeypatch.setattr(multi_train, "draw_windows", lambda context, settings, rng: ([window], {"drawn": {REAL_KIND: 1}}))
    settings = _multi_settings(start=start)
    out = tmp_path / "stage_d"
    post_train.open_campaign(out, {"settings": asdict(settings)}, {"head": "x", "dirty": False}, {},
                             schema=MULTI_CAMPAIGN_SCHEMA, reader=MULTI_CLAIM_READER)
    post_train.run_campaign(out, settings, context, stage=stage_d())
    assert not multi_train.round_model(context, settings, out, 0).training
    for wrong in (replace(settings, seed=7), replace(settings, span_s=60.0), replace(settings, c_min=0.8)):
        with pytest.raises(ValueError, match="another start, base, masks, token part, settings or round"):
            multi_train.round_model(context, wrong, out, 0)
    with pytest.raises(ValueError, match="another start, base"):
        multi_train.round_model(replace(context, base_identity={"base": "another"}), settings, out, 0)
    (out / "round_1").mkdir()
    (out / "round_1" / "checkpoint.pt").write_bytes((out / "round_0" / "checkpoint.pt").read_bytes())
    with pytest.raises(ValueError, match="or round than this campaign's"):
        multi_train.round_model(context, settings, out, 1)
