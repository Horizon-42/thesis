"""Stage C, C24 (post-training D175): the diagnostic readout — a campaign of one round on A26's one-flight synthetic
artefact (the fixture of `test_post_window_loop`), whose train split stands in for the select days, and the five
fields of a loss on synthetic windows of a two-runway airport. Every write root under tmp."""

from __future__ import annotations

import copy
import json
import os
import stat
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.experiments import post_diagnose as diagnose
from ts_transformer.experiments import post_train
from ts_transformer.experiments.post_ceiling import CEILING_SCHEMA
from ts_transformer.experiments.post_train import KINDS, done_rounds, open_campaign, read_batch, run_campaign, window_record
from ts_transformer.experiments.post_window_loop import LOST_SEPARATION
from ts_transformer.instructions.words import COLUMNS, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED
from ts_transformer.post.landings import roster_key
from ts_transformer.post.reward import LANDED
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.landings import window_landings
from ts_transformer.post.scene import INSERTED, INSERTED_SUFFIX, REAL, MovedScene, Recorded, Scene, Window
from ts_transformer.post.traffic_attention import traffic_modules
from ts_transformer.prior.procedure import Final
from ts_transformer.tests.support import parallel_airport
from ts_transformer.tests.test_post_branches import _ahead, _round
from ts_transformer.tests.test_post_train import _context, _settings
from ts_transformer.tests.test_post_window_loop import DELTA as LOOP_DELTA
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401
from flight_scenarios.fas_geometry import fas_course_geometry

INTENT = {"G2": "off below on by < 2 points with the interval holding 0: the traffic hardly used"}


@pytest.fixture
def campaign(setup, tmp_path, monkeypatch):  # noqa: F811
    """A campaign of one round (as `test_post_ceiling`'s) and the runner's world: the checks stubbed, the context the
    fixture's, the intent written."""
    s = setup
    context = _context(s)
    settings = _settings(rounds=1, per_kind={**dict.fromkeys(KINDS, 0), REAL: 1})
    model, _ = post_train.start_model(context, settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))
    with monkeypatch.context() as patch:
        patch.setattr(post_train, "speak_round", lambda model, context, windows, settings, round_, directory, speakers, *, stage: (
            torch.save([rewarded], directory / "groups_0.pt"), {"windows": 1})[1])
        out = tmp_path / "campaign"
        inputs = {"prior": str(tmp_path / "prior"), "instructions": str(s["directory"]),
                  "executor": str(tmp_path / "executor"), "windows": str(tmp_path / "census"),
                  "procedure_root": str(tmp_path / "cifp"), "settings": asdict(settings), "smoke": False}
        open_campaign(out, inputs, {"head": "x", "dirty": False}, {})
        run_campaign(out, settings, context)
    assert done_rounds(out) == 1
    opened = []
    monkeypatch.setattr(diagnose, "git_state", lambda: {"head": "x", "dirty": False})
    monkeypatch.setattr(diagnose, "require_conforming_closed_loop", lambda *a: (None, {"checks": {"stub": True}}, None))
    monkeypatch.setattr(diagnose, "checked_edges", lambda path: None)
    monkeypatch.setattr(diagnose, "open_context",
                        lambda *a, **k: (opened.append((k["splits"], k["formal"], k["data"])), context)[1])
    intent = tmp_path / "intent.json"
    intent.write_text(json.dumps(INTENT))
    return SimpleNamespace(out=out, opened=opened, context=context, settings=settings, intent=intent)


def _argv(c, out, *more):
    return ["--campaign", str(c.out), "--round", "0", "--intent", str(c.intent), "--out", str(out), "--device", "cpu",
            *more]


def _ceiling(c, tmp_path, *, models=("start", "0"), windows=None, campaign=None):
    """A D161 readout's files of the fixture's campaign (two draws a model): the start lands its window in draw 0."""
    root = tmp_path / "ceiling"
    root.mkdir()
    selected = post_train.selection_windows(c.context, c.settings)
    (root / "config.json").write_text(json.dumps({
        "schema": CEILING_SCHEMA, "campaign": str(campaign or c.out), "models": list(models), "draws": 2,
        "windows": windows if windows is not None else [window_record(w) for w in selected]}))
    for name in models:
        (root / f"model_{name}.json").write_text(json.dumps({
            "schema": CEILING_SCHEMA, "model": name, "outcomes": [[LANDED, LOST_SEPARATION]] * len(selected),
            "rewards": [[1.0, 0.0]] * len(selected)}))
    return root


def test_a_round_is_read_with_traffic_on_then_off_and_the_outputs_are_sealed(campaign, tmp_path):
    """The read with traffic on is the round's readout (checked); then off; a line per window and read; the summary,
    the failures and the intent written; the directory sealed. Only the select days are opened."""
    c = campaign
    out = tmp_path / "diagnose"
    assert diagnose.main(_argv(c, out)) == 0
    assert c.opened == [(("select",), True, False)]
    assert json.loads((out / "intent.json").read_text()) == INTENT
    lines = [json.loads(x) for x in (out / "per_window.jsonl").read_text().splitlines()]
    assert [x["read"] for x in lines] == [diagnose.ON, diagnose.OFF]
    assert {x["place"] for x in lines} == {0} and all(x["kind"] == REAL for x in lines)
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["reads"]) == {diagnose.ON, diagnose.OFF} and set(summary["differences"]) == {"on-off"}
    difference = summary["differences"]["on-off"]["all"]["landed"]
    assert len(difference["interval"]) == 2 and difference["interval"][0] <= difference["difference"] <= difference["interval"][1]
    failures = json.loads((out / "failures.json").read_text())
    assert set(failures) == {"schema", diagnose.ON, diagnose.OFF}
    config = json.loads((out / "config.json").read_text())
    assert config["schema"] == diagnose.DIAGNOSE_SCHEMA and not config["start_from_ceiling"]
    sums = (out / "SHA256SUMS").read_text().splitlines()
    assert {line.split("  ")[1] for line in sums} == {"intent.json", "config.json", "per_window.jsonl", "summary.json",
                                                      "failures.json", "log.jsonl"}
    assert all(not os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) for p in (out, *out.iterdir()))


def test_a_read_with_traffic_on_that_is_not_the_round_s_readout_stops_the_read(campaign, tmp_path):
    c = campaign
    held = json.loads((c.out / "round_0" / "round.json").read_text())
    code = next(iter(held["selection_readout"]))
    held["selection_readout"][code]["reward_sum"] += 1.0
    (c.out / "round_0" / "round.json").write_text(json.dumps(held))
    with pytest.raises(SystemExit, match="not the round's selection readout"):
        diagnose.main(_argv(c, tmp_path / "diagnose"))
    assert not (tmp_path / "diagnose" / "summary.json").exists()


def test_the_start_comes_from_the_ceiling_s_draw_0_and_other_windows_are_refused(campaign, tmp_path):
    """With a D161 readout of the campaign: the start's reading is its draw 0 (paired with each read), and the round's
    draws give field 5. A readout of other windows, or of another campaign, is refused by name before anything is
    written."""
    c = campaign
    out = tmp_path / "diagnose"
    assert diagnose.main(_argv(c, out, "--ceiling", str(_ceiling(c, tmp_path)))) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["reads"]["start"]["all"]["landed"] == 1.0
    assert set(summary["differences"]) == {"on-off", "off-start", "on-start"}
    config = json.loads((out / "config.json").read_text())
    assert config["start_from_ceiling"] and config["lost_every_draw_from_ceiling"]
    for name, more in (("ceiling_other", {"windows": [{"flight": "elsewhere", "row0_s": 0.0}]}),
                       ("ceiling_campaign", {"campaign": tmp_path / "another"})):
        root = tmp_path / name
        root.mkdir()
        made = _ceiling(c, root, **more)
        with pytest.raises(SystemExit, match="other windows|not campaign"):
            diagnose.main(_argv(c, root / "diagnose", "--ceiling", str(made)))
        assert not (root / "diagnose").exists()


@pytest.fixture
def refused(capsys):
    """A call of the runner refused by its parser with ``message``."""
    def check(argv, message):
        with pytest.raises(SystemExit):
            diagnose.main(argv)
        assert message in capsys.readouterr().err
    return check


def test_the_options_are_refused_by_name_before_anything_is_read(campaign, tmp_path, refused):
    c = campaign
    (tmp_path / "list.json").write_text("[1]")
    refused([*_argv(c, tmp_path / "diagnose"), "--round", "1"], "holds the checkpoints of rounds 0–0")
    refused(["--campaign", str(c.out), "--round", "0", "--intent", str(tmp_path / "list.json"), "--out",
             str(tmp_path / "diagnose"), "--device", "cpu"], "the decision rules are a JSON object")
    (tmp_path / "taken").mkdir()
    refused(_argv(c, tmp_path / "taken"), "writes into a new directory")
    assert c.opened == [] and not (tmp_path / "diagnose").exists()


def test_a_ceiling_s_start_is_refused_for_a_campaign_from_a_round_or_a_ceiling_without_it(campaign, tmp_path):
    """A campaign from another campaign's round starts with that round's traffic: no ceiling's start stands for it; a
    ceiling given for the start must have read it."""
    c = campaign
    with pytest.raises(SystemExit, match="did not read the campaign's start"):
        diagnose.main(_argv(c, tmp_path / "d0", "--ceiling", str(_ceiling(c, tmp_path, models=("0",)))))
    record = json.loads((c.out / "campaign.json").read_text())
    record["inputs"]["settings"]["start"] = {"campaign": "elsewhere", "round": 8, "checkpoint_sha256": "x"}
    (c.out / "campaign.json").write_text(json.dumps(record))
    root = tmp_path / "second"
    root.mkdir()
    with pytest.raises(SystemExit, match="starts from a round of another campaign"):
        diagnose.main(_argv(c, tmp_path / "d1", "--ceiling", str(_ceiling(c, root))))
    assert not (tmp_path / "d0").exists() and not (tmp_path / "d1").exists()


def test_traffic_off_zeroes_every_module_s_output_layer_and_nothing_else(campaign):
    c = campaign
    model = post_train.round_model(c.context, c.settings, c.out, 0)
    for module in traffic_modules(model):                     # weights of their own, as a trained round's
        torch.nn.init.normal_(module.out.weight)
        torch.nn.init.normal_(module.out.bias)
    before = copy.deepcopy(model.state_dict())
    diagnose.traffic_off(model)
    after = model.state_dict()
    zeroed = {prefix + f"out.{part}" for prefix in _module_prefixes(model) for part in ("weight", "bias")}
    assert zeroed and all(not after[name].any() for name in zeroed)
    assert all(torch.equal(before[name], after[name]) for name in after if name not in zeroed)


def _module_prefixes(model):
    names = {id(m): name for name, m in model.named_modules()}
    return [names[id(m)] + "." for m in traffic_modules(model)]


def _inserted(window, shift_s):
    """The window with the flight itself inserted ``shift_s`` ahead (window A, as `test_post_branches._ahead`)."""
    own = window.scene.flights[0]
    key = own.key + INSERTED_SUFFIX
    return replace(window, kind=INSERTED, moved=((key, -shift_s),),
                   scene=MovedScene(window.scene, added=(own.shifted(-shift_s, LOOP_DELTA, key=key),)))


def _read(c, model, window):
    (result,) = read_batch(model, c.context, [window], [0], c.settings, "select")
    return result


def test_a_model_whose_traffic_output_is_zero_reads_the_same_on_and_off(campaign):
    """A window with another aircraft 120 s ahead all along: a campaign's start (its traffic modules' output zero) says
    and flies the same with traffic off; the control — the same model with traffic weights of its own — does not."""
    c = campaign
    window = _inserted(c.context.splits["select"]["windows"][0], 120.0)
    start = post_train.round_model(c.context, c.settings, c.out, None)
    trained = copy.deepcopy(start)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for module in traffic_modules(trained):
            for p in module.parameters():
                p.copy_(torch.randn(p.shape, generator=generator) * 0.3)
    on, control = _read(c, start, window), _read(c, trained, window)
    off, control_off = _read(c, diagnose.traffic_off(start), window), _read(c, diagnose.traffic_off(trained), window)
    assert np.array_equal(on.words, off.words) and np.array_equal(on.states, off.states)
    assert (on.outcome, on.reward) == (off.outcome, off.reward)
    assert np.array_equal(control_off.words, off.words)                 # off, the module's other weights do nothing
    assert not (np.array_equal(control.words, control_off.words) and np.array_equal(control.states, control_off.states))


def test_the_fields_of_a_loss_read_in_the_loop(campaign):
    """The flight inserted 8 s ahead (`_ahead`): lost at the first row flown. Its fields from the loop's result: the
    state at the loss is the last flown, Δ after the first predicted step; the fixture's flight flies away from its
    threshold (west of runway 09, heading west), so the aircraft 8 s further along its track is behind it on the
    approach clock: a follower on its runway; the straight line loses separation there too."""
    c = campaign
    window = _ahead(c.context.splits["select"]["windows"][0])
    result = _read(c, post_train.round_model(c.context, c.settings, c.out, None), window)
    assert result.outcome == LOST_SEPARATION and result.other == window.moved[0][0]
    code = window.scene.geometry.code
    separation = c.context.separations[code]
    fields = diagnose.loss_fields(window, result, separation, c.context.finals[code],
                                  window_landings(window, c.context.rosters[code]), STEP, True)
    runway = diagnose.runway_at_loss(result)
    candidates = window.scene.geometry.candidates
    assert np.diff(window.commanded.e_m[:3]).max() < 0.0 and candidates[runway].course_deg == 90.0
    assert fields["other_class"] == diagnose.FOLLOWER and fields["runway"] == candidates[runway].ident
    at = result.states[-1]
    assert fields["distance_km"] == pytest.approx(np.hypot(at[0] - candidates[runway].threshold_e_m,
                                                           at[1] - candidates[runway].threshold_n_m) / 1000.0)
    assert fields["after_first_step_s"] == LOOP_DELTA and fields["time_bin"] == "below 60"
    assert fields["start_conflict"] is True and fields["lost_every_draw"] is True


# ---- the five fields, on synthetic windows of a two-runway airport ("09", candidate 0, and "09L" 891 m north of it)
GEOMETRY = parallel_airport()
SEPARATION = airport_separation(GEOMETRY)
FINALS = tuple(Final(GEOMETRY, k, 9_000.0, fas_course_geometry(c.length_m)) for k, c in enumerate(GEOMETRY.candidates))
DELTA, STEP = 4.0, 2.0
ROWS = 600


def _record(key, e0, speed, runway=0, n=0.0, height=1000.0):
    """A flight eastward along ``n`` from ``e0`` at ``speed`` m/s and ``height`` m, rows 2 s apart from UTC 0."""
    e = e0 + speed * STEP * np.arange(ROWS)
    return Recorded(key=f"KXXX:{key}", airport="KXXX", runway_index=runway, category=None, start_s=0.0, step_s=STEP,
                    first_step_s=16.0, landing_s=ROWS * STEP, e_m=e, n_m=np.full(ROWS, n),
                    height_m=np.full(ROWS, height), go_around=np.zeros(ROWS, dtype=bool))


def _window(*others):
    own = _record("own", -40_000.0, 100.0)
    return Window(REAL, own, 0, Scene(GEOMETRY, "select", (own, *others), DELTA))


def _landings(*records):
    return SimpleNamespace(landings=tuple(SimpleNamespace(flight_key=roster_key(r.key, "KXXX")) for r in records))


def test_field_1_names_the_other_aircraft_by_its_landing_runway_and_place_on_the_approach_clock():
    ahead, behind, beside = _record("ahead", -20_000.0, 100.0), _record("behind", -60_000.0, 100.0), \
        _record("beside", -20_000.0, 100.0, runway=1, n=891.0)
    window = _window(ahead, behind, beside)
    at, time_s = np.array([-40_000.0, 0.0, 1000.0]), 0.0
    every = _landings(ahead, behind, beside)
    for other, expected in ((ahead, diagnose.LEADER), (behind, diagnose.FOLLOWER), (beside, diagnose.OTHER_RUNWAY)):
        assert diagnose.other_class(window, other.key, 0, at, time_s, SEPARATION, every) == expected
    assert diagnose.other_class(window, ahead.key, 0, at, time_s, SEPARATION, _landings(behind)) == diagnose.NO_LANDING
    # the commanded aircraft's runway decides: said 09L, the flight beside it lands on its runway
    assert diagnose.other_class(window, beside.key, 1, at + [0.0, 891.0, 0.0], time_s, SEPARATION, every) == diagnose.LEADER


def test_fields_2_and_3_put_a_value_on_an_edge_in_the_upper_bin():
    def distance(km):
        return diagnose.bin_of(km, diagnose.DISTANCE_EDGES_KM, diagnose.DISTANCE_BINS)

    def time(s):
        return diagnose.bin_of(s, diagnose.TIME_EDGES_S, diagnose.TIME_BINS)

    assert [distance(x) for x in (0.0, 9.99, 10.0, 19.99, 20.0, 39.99, 40.0, 80.0)] == [
        "0-10", "0-10", "10-20", "10-20", "20-40", "20-40", "over 40", "over 40"]
    assert [time(x) for x in (4.0, 56.0, 60.0, 116.0, 120.0, 296.0, 300.0, 600.0)] == [
        "below 60", "below 60", "60-120", "60-120", "120-300", "120-300", "over 300", "over 300"]


def test_field_4_judges_the_straight_line_only_up_to_the_loss_or_120_s():
    """The commanded aircraft from 40 km out at 100 m/s behind another at 50 m/s on the same final: 8 km apart they
    lose separation within 120 s, 20 km apart only after it; a loss at an earlier step stops the judging there."""
    at, velocity = np.array([-40_000.0, 0.0, 1000.0]), np.array([100.0, 0.0, 0.0])
    steps = int(diagnose.STRAIGHT_LINE_S // DELTA)

    def lost(gap, steps_):
        other = _record("other", -40_000.0 + gap - 50.0 * 16.0, 50.0)       # ``gap`` ahead at the first predicted step
        return diagnose.straight_line_loss(_window(other), other.key, 0, at, velocity, steps_, SEPARATION, FINALS, STEP)

    assert lost(8_000.0, steps) and not lost(20_000.0, steps)
    assert lost(20_000.0, 4 * steps)                                    # the cap alone keeps the 20 km loss out
    assert not lost(8_000.0, 5)


def test_the_runway_at_the_loss_is_the_last_candidate_said():
    def result(runway_words):
        words = np.full((len(runway_words), len(COLUMNS)), UNCHANGED)
        words[:, RUNWAY] = runway_words
        return SimpleNamespace(words=words)

    assert diagnose.runway_at_loss(result([1, UNCHANGED, RUNWAY_GO_AROUND, UNCHANGED])) == 1
    assert diagnose.runway_at_loss(result([1, 0, UNCHANGED])) == 0
    with pytest.raises(ValueError, match="before any runway was said"):
        diagnose.runway_at_loss(SimpleNamespace(index=3, **vars(result([UNCHANGED]))))


def test_the_paired_difference_and_the_failures_counts():
    codes = ["KA", "KA", "KB", "KB"]
    on = ([LANDED, LANDED, LANDED, LOST_SEPARATION], [1.0, 1.0, 1.0, 0.0])
    off = ([LANDED, LOST_SEPARATION, LOST_SEPARATION, LOST_SEPARATION], [1.0, 0.0, 0.0, 0.0])
    paired = diagnose.paired(codes, on, off, np.random.default_rng(0))
    assert paired["all"]["landed"]["difference"] == 0.5 and paired["all"]["lost_separation"]["difference"] == -0.5
    assert paired["airports"]["KA"]["reward_mean"]["difference"] == 0.5
    lines = [{"airport": "KA", "outcome": LOST_SEPARATION, "fields": {"other_class": diagnose.LEADER,
                                                                      "start_conflict": True}},
             {"airport": "KB", "outcome": LOST_SEPARATION, "fields": {"other_class": diagnose.FOLLOWER,
                                                                      "start_conflict": False}},
             {"airport": "KB", "outcome": LANDED}]
    counted = diagnose.failures(lines)
    assert counted["all"]["lost"] == 2 and counted["all"]["other_class"][diagnose.LEADER] == {"count": 1, "share": 0.5}
    assert counted["all"]["start_conflict"]["true"]["count"] == 1 and "lost_every_draw" not in counted["all"]
    assert counted["airports"]["KB"]["other_class"][diagnose.FOLLOWER]["share"] == 1.0
