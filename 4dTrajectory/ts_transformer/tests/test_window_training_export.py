"""The Training module's multi-aircraft windows (`experiments/window_training_export`, the Training module §2.9): which
windows are chosen, the set's windows as recorded, and each sample's sentences, losses and landings as the window loop
flew them. On the scene-data fixture (a tmp artefact) the window runner's tests fly."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ts_transformer.instructions.words import Words
from ts_transformer.tests.support import signal_attitudes
from ts_transformer.tests.test_traffic_window import (
    LIMIT_S, STEP_S, _as_drawn, _patch_runner_physics, _scene_airport, _traffic_model,
)


def test_windows_are_shared_over_the_sizes_as_evenly_as_they_divide_the_larger_sizes_first():
    from ts_transformer.experiments.window_training_export import WINDOWS, shares

    assert WINDOWS == 20 and shares(20) == {"1": 6, "2": 7, "3+": 7}
    assert shares(3) == {"1": 1, "2": 1, "3+": 1} and shares(4) == {"1": 1, "2": 1, "3+": 2}


def _openings():
    """Two airports' drawn windows: at KAAA ten of one aircraft, three of two and five of three or more."""
    sizes = [1] * 10 + [2] * 3 + [3, 4, 3, 5, 3]
    out = [("KAAA", 600.0 * k, tuple(f"KAAA:f{k}_{m}" for m in range(size))) for k, size in enumerate(sizes)]
    return out + [("KBBB", 600.0 * k, (f"KBBB:f{k}",)) for k in range(4)]


def test_the_choice_takes_each_size_s_share_fills_a_short_one_from_the_rest_and_orders_by_opening():
    from ts_transformer.experiments.traffic_window_generation import size_of
    from ts_transformer.experiments.window_training_export import choose_windows

    openings = _openings()
    chosen, counts = choose_windows(openings, ["KAAA"], 9, seed=7)
    sizes = [size_of(len(openings[w][2])) for w in chosen]
    # shares 3 / 3 / 3, but only three windows of two: all of them, and nothing to fill
    assert sorted(sizes) == ["1"] * 3 + ["2"] * 3 + ["3+"] * 3
    assert counts["KAAA"] == {"drawn": 18, "drawnBySize": {"1": 10, "2": 3, "3+": 5}, "shares": {"1": 3, "2": 3, "3+": 3},
                              "taken": {"1": 3, "2": 3, "3+": 3}, "filled": 0}
    assert chosen == sorted(chosen, key=lambda w: openings[w][1])
    # twelve: shares 4 / 4 / 4, one window of two short — filled from the others' rest
    chosen, counts = choose_windows(openings, ["KAAA"], 12, seed=7)
    assert len(set(chosen)) == 12 and counts["KAAA"]["filled"] == 1 and counts["KAAA"]["taken"]["2"] == 3


def test_an_airport_s_choice_does_not_depend_on_the_airports_named_and_too_few_windows_are_refused():
    from ts_transformer.experiments.window_training_export import choose_windows

    openings = _openings()
    together, _ = choose_windows(openings, ["KBBB", "KAAA"], 4, seed=7)
    apart, _ = choose_windows(openings, ["KAAA"], 4, seed=7)
    assert len(together) == 8 and [w for w in together if openings[w][0] == "KAAA"] == apart
    with pytest.raises(ValueError, match="KBBB: 4 windows drawn, 5 wanted"):
        choose_windows(openings, ["KBBB"], 5, seed=7)


def _flown(tmp_path, monkeypatch, *, samples=2, seed=4):
    """The window runner's fixture flown: f0 and f1 commanded together (both lose separation at their first predicted
    step — f0 to the background f2, f1 to f0 — and fly on silent), f3 alone an hour later."""
    import torch

    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments.instruction_training_export import Globe
    from ts_transformer.experiments.traffic_window import window_of
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params
    from ts_transformer.tests.test_traffic_speaking import _batch

    airport, signals, spec = _scene_airport(tmp_path, monkeypatch)
    geometry = airport.flights.geometry
    commanded = [("KXXX:f0", "KXXX:f1"), ("KXXX:f3",)]
    keys = [k for keys in commanded for k in keys]
    batch = _batch(airport, signals, spec, keys)
    windows = [window_of(airport, 0.0, c, [LIMIT_S] * len(c), STEP_S) for c in commanded]
    drawn = _as_drawn(windows, [range(0, 2), range(2, 3)], batch, [LIMIT_S] * 3)
    _patch_runner_physics(monkeypatch, signals, geometry, keys)
    flown = runner.fly_windows(_traffic_model(spec), drawn, [0, 1], "scene", Words(spec), _params(), None, samples,
                               generator=torch.Generator().manual_seed(seed), temperature=1.0,
                               procedure_masks=ProcedureMasks.none())
    offsets = {candidate.ident: 12.5 for candidate in geometry.candidates}
    return flown, drawn, {"KXXX": Globe(geometry, offsets)}, spec, airport, signals


def test_every_window_sample_writes_each_commanded_aircraft_s_sentence_its_losses_and_its_landings(tmp_path, monkeypatch):
    from ts_transformer.autopilot.judge import CROSSINGS
    from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
    from ts_transformer.experiments.window_training_export import batch_payloads
    from ts_transformer.instructions.training_files import serialise
    from ts_transformer.prior.scene import N_LOOK

    flown, drawn, globes, spec, _, _ = _flown(tmp_path, monkeypatch)
    try:
        payloads = batch_payloads(flown, drawn, 2, globes, Words(spec))
    finally:
        flown.loop.close()
    serialise({"windows": payloads})                        # finite, writable
    # a window's samples in order, each window's commanded aircraft in its order
    assert [(w, p["sample"]) for w, p in payloads] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for w, payload in payloads:
        assert [a["datasetId"] for a in payload["aircraft"]] == list(drawn.windows[w].commanded)
        ended = {item["datasetId"] for item in payload["visual"]["ended"]}
        for sentence in payload["aircraft"]:
            track, events = sentence["track"], sentence["events"]
            # the first predicted step says every column; the words on the flight's own steps
            assert sorted(e["column"] for e in events if e["row"] == N_LOOK) == list(range(6))
            assert all(N_LOOK <= e["row"] < sentence["rows"] for e in events)
            # its states from its first predicted step, a step apart, to its own end; the words end no later
            assert track["tS"][0] == pytest.approx(N_LOOK * STEP_S)
            assert np.allclose(np.diff(track["tS"]), STEP_S)
            assert track["tS"][-1] == sentence["ownEndS"] and sentence["endS"] <= sentence["ownEndS"]
            assert np.allclose(np.subtract(track["altitudeHaeM"], track["altitudeM"]), 12.5, atol=0.011)
            # drawn in the executor's attitude at each of its states
            assert all(len(values) == len(track["tS"]) for values in track["attitude"].values())
            # ended by the judge ⇔ lost separation, and then the end names whom with
            assert (sentence["outcome"] == LOST_SEPARATION) == (sentence["end"] is not None) \
                == (sentence["datasetId"] in ended)
            assert (sentence["crossing"] is not None) == (sentence["own"] in CROSSINGS)
        landed = [a["datasetId"] for a in payload["aircraft"] if a["own"] == "landed"]
        assert sorted(item["datasetId"] for item in payload["landings"]) == sorted(landed)
        assert [item["atS"] for item in payload["landings"]] == sorted(item["atS"] for item in payload["landings"])
    # f0 and f1 end at their first predicted step (the loop's own test), say nothing more and fly on to the time limit
    f0, f1 = payloads[0][1]["aircraft"]
    assert (f0["outcome"], f1["outcome"], f0["end"]["with"], f1["end"]["with"]) == \
        (LOST_SEPARATION, LOST_SEPARATION, "KXXX:f2", "KXXX:f0")
    assert f0["endS"] == pytest.approx(N_LOOK * STEP_S) and f0["ownEndS"] == pytest.approx(N_LOOK * STEP_S + LIMIT_S)
    assert all(e["row"] == N_LOOK for e in f0["events"]) and f0["own"] == "timeout"
    assert any(e["pair"] == ["KXXX:f0", "KXXX:f2"] for e in payloads[0][1]["visual"]["episodes"])


def test_the_same_seed_writes_the_same_samples(tmp_path, monkeypatch):
    from ts_transformer.experiments.window_training_export import batch_payloads

    def once(name, seed):
        (tmp_path / name).mkdir()
        flown, drawn, globes, spec, _, _ = _flown(tmp_path / name, monkeypatch, seed=seed)
        try:
            return batch_payloads(flown, drawn, 2, globes, Words(spec))
        finally:
            flown.loop.close()

    assert json.dumps(once("a", 4)) == json.dumps(once("b", 4))


def test_a_set_window_lists_who_is_in_it_and_what_the_record_made_of_it(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_window_generation import fixed_paths
    from ts_transformer.experiments.window_training_export import window_payload
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params

    flown, drawn, globes, spec, _, signals = _flown(tmp_path, monkeypatch)
    flown.loop.close()
    _, _, recorded = fixed_paths(drawn, [0, 1], "recorded", Words(spec), _params(), ProcedureMasks.none())
    window = drawn.windows[0]
    by_id = signals                                                     # the fixture's signals, by dataset id
    payload = window_payload(window, [LIMIT_S, LIMIT_S], recorded[0], globes["KXXX"], STEP_S, by_id,
                             signal_attitudes(None, list(signals.values())))
    assert [c["datasetId"] for c in payload["commanded"]] == ["KXXX:f0", "KXXX:f1"]
    # each commanded aircraft's row 0 on the scene's steps, from the window's opening (f1 entered 30 s after f0)
    assert payload["commanded"][1]["rowZeroS"] - payload["commanded"][0]["rowZeroS"] == pytest.approx(30.0, abs=1.0)
    others = {o["datasetId"]: o for o in payload["others"]}
    assert others["KXXX:f2"]["role"] == "background" and others["KXXX:f2"]["callsign"] == "f2"
    track = others["KXXX:f2"]["track"]
    assert len(track["tS"]) == len(track["lon"]) == len(track["altitudeHaeM"]) > 1
    # drawn in its observed attitude, on its recorded rows: the first rows of its signals
    assert all(values is None or len(values) == len(track["tS"]) for values in track["attitude"].values())
    assert track["attitude"]["headingDeg"] == pytest.approx(
        np.mod(by_id["KXXX:f2"].track_deg[: len(track["tS"])], 360.0), abs=0.006)
    assert payload["opensUtc"].endswith("Z")
    # the record's losses and landings are the recorded paths' judge's
    assert payload["recorded"]["visual"]["ended"] == [
        {"datasetId": key, "atS": round(end["t_s"] - window.opens_s, 3), "kind": end["kind"], "relation": end["relation"],
         "with": end["with"]} for key, end in recorded[0].run.ended.items()]
    landed = sorted((path.landing_s, path.key) for path, own in zip(recorded[0].paths, recorded[0].owns) if own == "landed")
    assert [item["datasetId"] for item in payload["recorded"]["landings"]] == [key for _, key in landed]


def test_a_set_an_earlier_export_wrote_is_accepted_only_as_the_same_set():
    from ts_transformer.experiments.window_training_export import require_same_set

    built = {"schema": "s", "writtenUtc": "now", "producedBy": {"git": "b"}, "windows": [1, 2]}
    require_same_set({**built, "writtenUtc": "then", "producedBy": {"git": "a"}}, built, "set.json")
    with pytest.raises(ValueError, match=r"differs in \['windows'\]"):
        require_same_set({**built, "windows": [1]}, built, "set.json")


def test_the_readout_is_copied_only_when_it_is_this_prior_s_over_these_windows(tmp_path):
    from ts_transformer.experiments.traffic_window_generation import SCHEMA, WINDOWS_PER_AIRPORT
    from ts_transformer.experiments.window_training_export import readout_block

    prior = tmp_path / "4dTrajectory/outputs/POOLED/prior/run/round_05"
    cell = {"aircraft": 10, "outcomes": {"landed": 0.8, "lost_separation": 0.2}, "lost_separation": 0.2,
            "lost_separation_ifr": 0.3}
    readout = {"schema": SCHEMA, "written_utc": "2026-09-30T00:00:00Z", "split": "select", "commanded": "every",
               "windows_per_airport": WINDOWS_PER_AIRPORT, "seed": 1337, "samples": 4, "temperature": 1.0,
               "augment_seed": None, "instructions": "x/4dTrajectory/outputs/POOLED/instruction_language/v5",
               "prior": {"directory": str(prior), "checkpoint_sha256": "c"}, "executor": {"sha256": "e"},
               "readout": {"pooled": {"scene": cell, "recorded": {**cell, "outcomes": {"lost_separation": 1.0}}},
                           "airports": {"KXXX": {"scene": cell, "recorded": cell}}}}
    kwargs = dict(prior_dir=prior, checkpoint_sha256="c", executor_sha256="e",
                  instructions=tmp_path / "4dTrajectory/outputs/POOLED/instruction_language/v5", samples=4,
                  temperature=1.0, seed=1337, airport="KXXX")
    got = readout_block(readout, tmp_path / "4dTrajectory/outputs/POOLED/traffic/r", **kwargs)
    assert got["scene"]["here"] == {"aircraft": 10, "landed": 0.8, "lostSeparation": 0.2, "lostSeparationIfr": 0.3}
    assert got["recorded"]["all"]["landed"] == 0.0 and got["directory"] == "4dTrajectory/outputs/POOLED/traffic/r"
    with pytest.raises(ValueError, match="samples 4, expected 2"):
        readout_block(readout, tmp_path / "r", **{**kwargs, "samples": 2})
    with pytest.raises(ValueError, match="commanded 'one', expected 'every'"):     # one commanded aircraft a window
        readout_block({**readout, "commanded": "one"}, tmp_path / "r", **kwargs)
    with pytest.raises(ValueError, match="not ts-traffic-window-generation"):
        readout_block({**readout, "schema": "ts-traffic-window-generation-v1"}, tmp_path / "r", **kwargs)


def _ts_constant(file: str, name: str) -> str:
    import re

    from ts_transformer.repo_layout import REPO_ROOT

    source = (REPO_ROOT / "aeroviz-4d" / "src" / "data" / file).read_text(encoding="utf-8")
    match = re.search(rf"export const {name}\b[^=]*=\s*(?P<value>[^;]+);", source)
    assert match is not None, f"{name} not found in {file}"
    return match.group("value").strip()


def test_the_frontend_reader_mirrors_the_window_files_names():
    from ts_transformer.experiments import window_training_export as export
    from ts_transformer.instructions import training_files as files

    assert json.loads(_ts_constant("trainingTraffic.ts", "TRAINING_TRAFFIC_SCHEMA")) == files.TRAFFIC_SCHEMA
    assert json.loads(_ts_constant("trainingTraffic.ts", "TRAINING_WINDOW_GENERATION_SCHEMA")) == export.SCHEMA
    assert json.loads(_ts_constant("trainingSample.ts", "TRAINING_TRAFFIC_SET_KIND")) == files.KIND_TRAFFIC


def test_the_landings_are_the_aircraft_whose_own_end_is_a_landing(tmp_path, monkeypatch):
    """The loop keeps a landing time for an aircraft the glidepath lower edge stopped first; the overlay's landings are the
    aircraft whose own end is a landing, as the record's are (the reader holds them to exactly those)."""
    from ts_transformer.experiments.window_training_export import batch_payloads

    flown, drawn, globes, spec, _, _ = _flown(tmp_path, monkeypatch, samples=1)
    try:
        stopped = flown.results[2]                          # f3, alone in its window
        stopped.own, stopped.outcome, stopped.landing_s = "below_glidepath", "below_glidepath", stopped.first_s + 40.0
        payloads = batch_payloads(flown, drawn, 1, globes, Words(spec))
        assert payloads[1][1]["landings"] == []
    finally:
        flown.loop.close()


def _set_files(root, code="KXXX", set_id="windows", file_text=None):
    """A training directory holding a window set ``set_id`` listed in its index (``file_text``: its file; None: none)."""
    from ts_transformer.instructions.training_files import INDEX_SCHEMA, KIND_TRAFFIC, TRAFFIC_FILE, TRAFFIC_SCHEMA
    from ts_transformer.instructions.spec import READING_RULE

    training = root / code / "training"
    (training / set_id).mkdir(parents=True)
    entry = {"id": set_id, "kind": KIND_TRAFFIC, "readingRule": READING_RULE, "file": f"{set_id}/{TRAFFIC_FILE}"}
    (training / "index.json").write_text(json.dumps({"schema": INDEX_SCHEMA, "airport": code, "sets": [entry]}))
    payload = {"schema": TRAFFIC_SCHEMA, "setId": set_id, "airport": code, "writtenUtc": "then", "producedBy": {},
               "cohort": {"split": "select"}, "vocabulary": {"readingRule": READING_RULE}, "windows": [1]}
    if file_text is not False:
        (training / set_id / TRAFFIC_FILE).write_text(json.dumps(payload))
    return training, entry, payload


def test_what_is_on_disk_is_refused_before_any_work_and_a_set_an_earlier_export_wrote_is_kept(tmp_path):
    from ts_transformer.experiments.window_training_export import on_disk, write_export

    # nothing yet: the set and the overlay are new, and both manifests are written
    found = on_disk(tmp_path, ["KXXX"], "windows", "windows_base")
    assert found["KXXX"].index == [] and found["KXXX"].overlays == []
    overlay_entry = {"id": "windows_base", "kind": "window-generation", "base": "windows", "title": "t",
                     "file": "windows_base/window_generation.json", "flights": 1, "source": {}}
    from ts_transformer.instructions.training_files import TRAFFIC_SCHEMA
    from ts_transformer.instructions.spec import READING_RULE
    payload = {"schema": TRAFFIC_SCHEMA, "setId": "windows", "airport": "KXXX", "writtenUtc": "now", "producedBy": {},
               "cohort": {"split": "select"}, "vocabulary": {"readingRule": READING_RULE}, "windows": [1]}
    entry = {"id": "windows", "kind": "traffic-windows", "title": "w", "file": "windows/traffic.json", "readingRule": READING_RULE}
    written = write_export(tmp_path, found, {"KXXX": (payload, entry)}, {"KXXX": ("{}", overlay_entry)})
    assert [path.name for path in written] == ["traffic.json", "window_generation.json"]
    index = json.loads((tmp_path / "KXXX/training/index.json").read_text())
    assert [item["id"] for item in index["sets"]] == ["windows"]
    # a second export names the set: kept, and its overlay added beside the first
    again = on_disk(tmp_path, ["KXXX"], "windows", "windows_traffic")
    assert again["KXXX"].index is None and [item["id"] for item in again["KXXX"].overlays] == ["windows_base"]
    second = {**overlay_entry, "id": "windows_traffic", "file": "windows_traffic/window_generation.json"}
    written = write_export(tmp_path, again, {"KXXX": ({**payload, "writtenUtc": "later"}, entry)}, {"KXXX": ("{}", second)})
    assert [path.name for path in written] == ["window_generation.json"]
    # ... but not a set that differs from the one it would write
    third = {**overlay_entry, "id": "windows_other", "file": "windows_other/window_generation.json"}
    with pytest.raises(ValueError, match=r"differs in \['windows'\]"):
        write_export(tmp_path, on_disk(tmp_path, ["KXXX"], "windows", "windows_other"),
                     {"KXXX": ({**payload, "windows": [2]}, entry)}, {"KXXX": ("{}", third)})
    # an overlay already there, and a set directory a stopped export left without its file
    with pytest.raises(ValueError, match="windows_base exists; an overlay is never overwritten"):
        on_disk(tmp_path, ["KXXX"], "windows", "windows_base")
    (tmp_path / "KXXX/training/half").mkdir()
    with pytest.raises(ValueError, match="exists without its traffic.json"):
        on_disk(tmp_path, ["KXXX"], "half", "windows_new")


def test_a_set_listed_as_another_kind_is_refused_as_a_window_set(tmp_path):
    from ts_transformer.experiments.window_training_export import on_disk

    training, entry, _ = _set_files(tmp_path)
    index = json.loads((training / "index.json").read_text())
    index["sets"][0]["kind"] = "vocabulary-readback"
    (training / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="is a vocabulary-readback set of .*, not a traffic-windows set"):
        on_disk(tmp_path, ["KXXX"], "windows", "windows_base")
