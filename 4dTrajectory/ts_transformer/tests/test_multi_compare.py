"""Stage D: a campaign's start and rounds side by side (`multi_compare`, the user 2026-10-10) — a stage D campaign of one
round from round 0 of a stage C campaign, on A26's one-flight synthetic artefact (the fixture of
`test_post_window_loop`), whose train split stands in for the select days. Every write root under tmp."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace

import pytest

from ts_transformer.experiments import multi_compare as compare
from ts_transformer.experiments import multi_train, post_train
from ts_transformer.experiments.multi_train import MULTI_CAMPAIGN_SCHEMA, MULTI_CLAIM_READER, MultiSettings, stage_d
from ts_transformer.multi.windows import REAL_KIND
from ts_transformer.tests.test_multi_train import _multi_settings
from ts_transformer.tests.test_post_branches import _ahead
from ts_transformer.tests.test_post_train import _context, _settings, short_round
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401

GIT = {"head": "x", "dirty": False}


@pytest.fixture
def campaign(setup, tmp_path, monkeypatch):  # noqa: F811
    """A stage D campaign of one round (its draw replaced by window A) from round 0 of a stage C campaign, and the
    runner's world: the checks stubbed, the context the fixture's, each opening recorded (``opened``)."""
    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    source = tmp_path / "stage_c"
    post_train.open_campaign(source, {"settings": asdict(_settings(rounds=1, continuations=2)), "smoke": False}, GIT, {})
    post_train.run_campaign(source, _settings(rounds=1, continuations=2), context)
    start = post_train.start_of(source, 0, formal=False)
    window = replace(_ahead(s["windows"][0]), span_s=1.0)                  # of the campaign's span
    with monkeypatch.context() as patch:
        patch.setattr(multi_train, "draw_windows", lambda context, settings, rng: ([window], {"drawn": {REAL_KIND: 1}}))
        settings = _multi_settings(start=start, spans_s=[1.0, 2.0])
        out = tmp_path / "multi"
        inputs = {"prior": str(tmp_path / "prior"), "instructions": str(s["directory"]),
                  "executor": str(tmp_path / "executor"), "windows": str(tmp_path / "census"),
                  "procedure_root": str(tmp_path / "cifp"), "settings": asdict(settings), "smoke": False}
        post_train.open_campaign(out, inputs, GIT, {}, schema=MULTI_CAMPAIGN_SCHEMA, reader=MULTI_CLAIM_READER,
                                 settings_type=MultiSettings)
        post_train.run_campaign(out, settings, context, stage=stage_d())
    assert post_train.done_rounds(out) == 1
    opened = []
    monkeypatch.setattr(compare, "git_state", lambda: GIT)
    monkeypatch.setattr(compare, "require_conforming_closed_loop", lambda *a: (None, {"checks": {"stub": True}}, None))
    monkeypatch.setattr(compare, "checked_edges", lambda path: None)
    monkeypatch.setattr(compare, "open_context", lambda *a, **k: (opened.append((k["splits"], k["data"], k["formal"])),
                                                                  context)[1])
    return out, context, settings, opened


def _readout(cells, whole):
    """A readout of one airport (`multi_train.selection_readout`'s shape): its cells by span and kind, and its all."""
    def cell(aircraft, landed, lost, reward):
        return {"aircraft": aircraft, "outcomes": {"landed": landed, "lost_separation": lost}, "reward_sum": reward}

    spans = {}
    for span, kind, numbers in cells:
        spans.setdefault(span, {})[kind] = cell(*numbers)
    return {"KAAA": {**spans, "all": {"all": cell(*whole)}, "reward_mean": whole[3] / whole[0]}}


def test_each_airport_is_its_cells_summed_and_checked_against_its_all():
    """The cells of span × kind summed (each span's ``all`` and the logged reward mean left out) and checked against the
    airport's ``all`` cell; a cell that is not refused by name."""
    readout = _readout([("1", "real", (3, 2, 1, 1.9)), ("2", "real", (2, 1, 0, 1.0)), ("2", "all", (2, 1, 0, 1.0))],
                       (5, 3, 1, 2.9))
    assert compare.counted(readout, "r") == {"KAAA": {"aircraft": 5, "landed": 3, "lost_separation": 1,
                                                      "reward_sum": pytest.approx(2.9)}}
    for wrong in ((6, 3, 1, 2.9), (5, 4, 1, 2.9), (5, 3, 0, 2.9), (5, 3, 1, 3.0)):
        with pytest.raises(ValueError, match="r: KAAA's cells of span and kind are not its all cell"):
            compare.counted(_readout([("1", "real", (3, 2, 1, 1.9)), ("2", "real", (2, 1, 0, 1.0))], wrong), "r")


def test_a_rounds_pass_gives_its_pull_to_the_base_and_its_updates():
    """The pass's means (the pull to the base ``kl``, ``data``, ``surrogate``) beside its updates; a pass of no update
    has none (`post_train` records none), the table's two cells a dash."""
    passed = {"updates": 3, "kl": 0.05, "data": 1.1, "surrogate": -0.9, "loss": 0.1, "passes": []}
    assert compare.pass_numbers(passed) == {"updates": 3, "kl": 0.05, "data": 1.1, "surrogate": -0.9}
    assert compare.pass_numbers({"updates": 0, "passes": []}) == {"updates": 0}
    row = {"model": "round 0", "airports": {"KAAA": {"aircraft": 4, "landed": 3, "lost_separation": 1,
                                                     "reward_sum": 2.9}},
           "all": {"aircraft": 4, "landed": 3, "lost_separation": 1, "reward_sum": 2.9}}
    printed = compare.table({"rows": [{**row, "pass": compare.pass_numbers(passed)},
                                      {**row, "model": "round 1", "pass": {"updates": 0}}]})
    assert printed.splitlines()[1:] == ["round 0 | 75.0% | 3/4 = 75.0% | 0.7250 | 25.0% | 0.0500 | 1.1000",
                                        "round 1 | 75.0% | 3/4 = 75.0% | 0.7250 | 25.0% | – | –"]


def test_the_start_read_is_the_campaigns_readout_of_its_start_model(campaign, tmp_path):
    """``--read-start``: the select days opened formal without data, the last done round's model read again as its
    readout (the behaviour check), then the start model (`Stage.start`) read on the select windows by the campaign's
    own readout, written with the campaign's inputs; a second read to the same directory refused."""
    out, context, settings, opened = campaign
    read = tmp_path / "start"
    assert compare.main(["--campaign", str(out), "--read-start", str(read), "--device", "cpu"]) == 0
    assert opened == [(("select",), False, True)]
    held = json.loads((read / compare.START_READ).read_text())
    stage = stage_d()
    model, _ = stage.start(context, settings)
    expected = multi_train.selection_readout(model, context, stage.selection(context, settings, "select"), settings,
                                             None, stage=stage)
    assert held["schema"] == compare.START_READ_SCHEMA and held["readout"] == json.loads(json.dumps(expected))
    assert held["inputs"] == json.loads((out / "campaign.json").read_text())["inputs"]
    assert held["checks"] == {"stub": True} and held["reread_round"] == 0 and held["smoke"] is False
    assert len(held["windows"]) == 2 and held["device"] == "cpu"
    with pytest.raises(SystemExit):
        compare.main(["--campaign", str(out), "--read-start", str(read), "--device", "cpu"])


def test_a_start_read_stops_unless_this_code_reads_a_round_as_its_readout(campaign, tmp_path):
    """The behaviour check: a round whose recorded readout this code does not give again stops the start read before
    the start is read, nothing written; a campaign with no done round is refused before anything opens."""
    out, _, _, opened = campaign
    path = out / "round_0" / "round.json"
    held = json.loads(path.read_text())
    (airport,) = held["selection_readout"].values()
    airport["all"]["all"]["reward_sum"] += 1.0
    path.write_text(json.dumps(held))
    with pytest.raises(SystemExit, match="round 0: this code's readout of its model on the cpu is not the round's"):
        compare.main(["--campaign", str(out), "--read-start", str(tmp_path / "start"), "--device", "cpu"])
    assert not (tmp_path / "start").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "campaign.json").write_text((out / "campaign.json").read_text())
    opened.clear()
    with pytest.raises(SystemExit):
        compare.main(["--campaign", str(empty), "--read-start", str(tmp_path / "start"), "--device", "cpu"])
    with pytest.raises(SystemExit):
        compare.main(["--campaign", str(empty), "--out", str(tmp_path / "e.json")])          # nothing to compare
    assert opened == [] and not (tmp_path / "e.json").exists()


def test_the_comparison_names_each_number_by_its_file_and_the_rounds_pass(campaign, tmp_path, capsys):
    """``--out``: the start's row from its read (no pass) and each round's from its ``round.json`` (its readout and its
    pass's pull to the base), each the cells summed; the table printed."""
    out, _, _, _ = campaign
    read = tmp_path / "start"
    compare.main(["--campaign", str(out), "--read-start", str(read), "--device", "cpu"])
    capsys.readouterr()
    assert compare.main(["--campaign", str(out), "--start", str(read), "--out", str(tmp_path / "c.json")]) == 0
    written = json.loads((tmp_path / "c.json").read_text())
    start_row, round_row = written["rows"]
    held = json.loads((out / "round_0" / "round.json").read_text())
    assert start_row["model"] == "start" and start_row["pass"] is None
    assert start_row["airports"] == compare.counted(json.loads((read / compare.START_READ).read_text())["readout"], "s")
    assert round_row["model"] == "round 0" and round_row["source"] == str(out / "round_0" / "round.json")
    assert round_row["airports"] == compare.counted(held["selection_readout"], "r")
    assert round_row["pass"] == compare.pass_numbers(held["pass"]) == {"updates": 0}     # the fixture's round: none
    (airport,) = round_row["airports"].values()
    assert round_row["all"] == {**airport, "reward_sum": pytest.approx(airport["reward_sum"])}
    printed = capsys.readouterr().out
    assert printed.startswith("model | ") and "\nstart | " in printed and "\nround 0 | " in printed
    assert compare.main(["--campaign", str(out), "--out", str(tmp_path / "rounds.json")]) == 0   # no start: rounds only
    assert [r["model"] for r in json.loads((tmp_path / "rounds.json").read_text())["rows"]] == ["round 0"]


def test_a_start_read_of_other_inputs_or_windows_and_the_options_are_refused_by_name(campaign, tmp_path, capsys):
    """A start read of other settings is refused, one apart only in the rule of the CPU's batches compared (D191: a
    resume may change it); a row of other aircraft than the first's is refused; the options alone or together refused."""
    out, _, _, _ = campaign
    read = tmp_path / "start"
    compare.main(["--campaign", str(out), "--read-start", str(read), "--device", "cpu"])
    held = json.loads((read / compare.START_READ).read_text())
    for name, changed, refused in (("seed", {"seed": 7}, "a start read of other inputs or settings"),
                                   ("device", {}, "read on the cuda, round 0's readout on cpu"),
                                   ("rule", {"cpu_batches": 1, "cpu_period": 2}, None)):
        other = tmp_path / name
        other.mkdir()
        moved = copy.deepcopy(held)
        moved["inputs"]["settings"].update(changed)
        if name == "device":
            moved["device"] = "cuda"                                         # the rounds' readouts on the CPU here
        (other / compare.START_READ).write_text(json.dumps(moved))
        argv = ["--campaign", str(out), "--start", str(other), "--out", str(tmp_path / f"{name}.json")]
        if refused is not None:
            with pytest.raises(SystemExit):
                compare.main(argv)
            assert refused in capsys.readouterr().err
        else:
            assert compare.main(argv) == 0
    fewer = copy.deepcopy(held)
    (airport,) = fewer["readout"].values()
    first = next(span for span in airport if span not in ("all", "reward_mean"))
    for span in (first, "all"):                                          # one aircraft more, its cells consistent
        for cell in airport[span].values():
            cell["aircraft"] += 1
    (tmp_path / "fewer").mkdir()
    (tmp_path / "fewer" / compare.START_READ).write_text(json.dumps(fewer))
    with pytest.raises(SystemExit):
        compare.main(["--campaign", str(out), "--start", str(tmp_path / "fewer"), "--out", str(tmp_path / "f.json")])
    assert "other aircraft than" in capsys.readouterr().err and not (tmp_path / "f.json").exists()
    for argv in ([], ["--read-start", str(tmp_path / "a"), "--out", str(tmp_path / "a.json")],
                 ["--read-start", str(tmp_path / "a"), "--start", str(read)],
                 ["--out", str(tmp_path / "b.json"), "--device", "cpu"],      # the start read's options
                 ["--out", str(tmp_path / "rule.json")]):                 # written above: a new file only
        with pytest.raises(SystemExit):
            compare.main(["--campaign", str(out), *argv])
    assert not (tmp_path / "a").exists()
