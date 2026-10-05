"""The Training export of stage A (`experiments/training_export.py`, `instructions/training_files.py`; vocabulary §12.1
A23, outline §6): who is drawn, the words as events, the envelopes the views draw, and the set and its index beside the
old ones — never over an existing set, never into the instruction-v3 view's index."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ts_transformer.experiments import training_export as export
from ts_transformer.instructions import training_files as files
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import UNCHANGED


def _rows(airport, strata):
    """Formal replay rows: ``strata`` a list of (stratum, count) for ``airport``, ids F0, F1, …"""
    out, k = {}, 0
    for stratum, count in strata:
        for _ in range(count):
            out[f"F{k}"] = {"airport": airport, "stratum": stratum, "kind": "without go-around"}
            k += 1
    return out


def test_each_split_draws_per_stratum_flights_flown_at_every_row_interval_seeded():
    two = _rows("KAAA", [("straight-in", 6), ("vectored", 5)])
    rows = {2.0: two, 4.0: dict(two), 8.0: {k: v for k, v in two.items() if k != "F0"}}   # F0 not flown at 8 s
    chosen = export.choose(rows, "KAAA", 3, 1337)
    assert len(chosen) == 6 and "F0" not in chosen
    assert [rows[2.0][d]["stratum"] for d in chosen] == ["straight-in"] * 3 + ["vectored"] * 3
    assert export.choose(rows, "KAAA", 3, 1337) == chosen != export.choose(rows, "KAAA", 3, 7)
    with pytest.raises(ValueError, match="5 straight-in flights flown at every row interval, 6 asked"):
        export.choose(rows, "KAAA", 6, 1337)


def test_the_words_of_a_sentence_are_its_events_decoded_with_the_corrections_marked():
    """Each word as the views show it, decoded by `Words` (the views decode nothing): the runway's candidate or
    go-around, a heading relative to the course of the runway in force and its track, a level above E and in MSL or "no
    level-off", an angle's nominal, a speed or "unspecified"."""
    from ts_transformer.instructions.words import Words
    from ts_transformer.tests.support import instruction_airport, instruction_spec

    words, geometry = Words(instruction_spec()), instruction_airport()
    grid = np.full((3, 5), UNCHANGED)
    grid[0] = [0, 2, 10, 0, words.speed_unspecified]
    grid[2, 1], grid[2, 2], grid[2, 0] = 3, words.altitude_no_level_off, -2
    correction = np.zeros((3, 5), bool)
    correction[2, 1] = True
    said = export.events(grid, None, geometry, words)
    assert [(e["row"], e["column"], e["value"], e["correction"]) for e in said] == [
        (0, 0, 0, False), (0, 1, 2, False), (0, 2, 10, False), (0, 3, 0, False), (0, 4, words.speed_unspecified, False),
        (2, 0, -2, False), (2, 1, 3, False), (2, 2, words.altitude_no_level_off, False)]
    course = geometry.candidates[0].course_deg
    step = words.spec.heading_step_deg
    assert said[0]["says"] == {"runway": geometry.candidates[0].ident, "runwayIndex": 0}
    assert said[1]["says"] == {"relativeDeg": 2 * step, "trackDeg": round((course + 2 * step) % 360.0, 3)}
    level = words.altitude_level_m(10)
    assert said[2]["says"] == {"noLevelOff": False, "levelM": level, "mslM": round(level + geometry.elevation_m, 1)}
    assert said[3]["says"] == {"angleDeg": 0.0, "climb": False, "level": True}
    assert said[4]["says"] == {"speedMps": None}
    # go-around changes no runway: the heading word after it is still relative to candidate 0's course
    assert said[5]["says"] == {"goAround": True}
    assert said[6]["says"]["trackDeg"] == round((course + 3 * step) % 360.0, 3)
    assert said[7]["says"] == {"noLevelOff": True, "levelM": None, "mslM": None}
    assert export.events(grid, correction, geometry, words)[6]["correction"] is True


def test_a_heading_band_is_the_judges_rows_and_verdicts():
    """The rows and the verdicts are `envelope.heading_words_inside`'s (a word judged from a lead after it to a lead
    after the next one, never past the end), drawn row by row."""
    from ts_transformer.instructions import envelope

    track = np.array([90.0] * 6 + [100.0, 120.0, 130.0, 130.0])
    words = [(0, 90.0), (5, 130.0)]
    bands = files.heading_bands(track, words, 2, len(track), 4.5)
    judged = envelope.heading_words_inside(track, words, 2, len(track), 4.5)
    assert [(b["row"], len(b["inside"]), sum(b["inside"])) for b in bands] == [
        (j["row"], j["rows"], j["inside"]) for j in judged]
    assert (bands[0]["firstRow"], bands[0]["stopRow"], bands[1]["firstRow"], bands[1]["stopRow"]) == (2, 7, 7, 10)
    assert bands[1]["inside"] == [0, 1, 1]
    # a word whose lead runs past the end has an empty band past the track: not judged, nothing drawn
    (late,) = files.heading_bands(track, [(9, 130.0)], 2, len(track), 4.5)
    assert (late["firstRow"], late["stopRow"], late["inside"]) == (11, 11, [])


def test_a_tube_is_written_in_msl_beside_its_level_above_the_airport():
    tube = files.tube_payload(4, 9, 600.0, np.array([550.0, 560.0]), np.array([640.0, 650.0]), 60.0,
                              {"rows": 5, "inside": 4, "contained": False})
    assert tube == {"row": 4, "endRow": 9, "levelM": 600.0, "lowMslM": [610.0, 620.0], "highMslM": [700.0, 710.0],
                    "rows": 5, "inside": 4, "contained": False}
    assert files.nullable([1.04, float("nan")], 1) == [1.0, None]


def _entry(set_id):
    return {"id": set_id, "kind": files.SET_KIND, "readingRule": READING_RULE, "file": f"{set_id}/{files.SAMPLE_FILE}"}


def _sample(set_id, airport):
    return {"schema": files.SAMPLE_SCHEMA, "setId": set_id, "airport": airport, "vocabulary": {"readingRule": READING_RULE}}


def test_a_set_goes_beside_the_old_index_and_is_never_overwritten(tmp_path):
    """Outline §6 item 3: the set and its entry go into `index_v4.json`; the instruction-v3 view's `index.json` is never
    read or written; a set already listed, or an index another run changed, is refused."""
    training = tmp_path / "KAAA" / "training"
    training.mkdir(parents=True)
    old = training / "index.json"
    old.write_text('{"schema": "aeroviz-training-index-v1", "sets": []}', encoding="utf-8")
    before = old.read_bytes()
    existing = files.read_index(training, "KAAA", "set_a")
    assert existing == []
    files.write_set(training, "KAAA", _entry("set_a"), files.serialise(_sample("set_a", "KAAA")), existing)
    assert old.read_bytes() == before
    index = json.loads((training / files.INDEX_FILE).read_text(encoding="utf-8"))
    assert index["schema"] == files.INDEX_SCHEMA and [s["id"] for s in index["sets"]] == ["set_a"]
    entry, sample = files.listed_set(training, "KAAA", "set_a")
    assert entry["id"] == sample["setId"] == "set_a"
    with pytest.raises(ValueError, match="already lists set set_a"):
        files.read_index(training, "KAAA", "set_a")
    stale = files.read_index(training, "KAAA", "set_b")
    files.write_set(training, "KAAA", _entry("set_c"), files.serialise(_sample("set_c", "KAAA")), stale)
    with pytest.raises(ValueError, match="changed since this run read it"):
        files.write_set(training, "KAAA", _entry("set_b"), files.serialise(_sample("set_b", "KAAA")), stale)
    with pytest.raises(files.NotListed):
        files.listed_set(training, "KAAA", "set_b")
    with pytest.raises(ValueError, match="not KBBB's"):
        files.read_index(training, "KBBB", "set_d")
    with pytest.raises(ValueError, match="refusing NaN|Out of range float"):
        files.serialise({"x": float("nan")})


def test_a_listed_set_of_another_format_is_refused_by_name(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    files.write_set(training, "KAAA", _entry("set_a"), files.serialise({**_sample("set_a", "KAAA"),
                                                                        "schema": "aeroviz-training-sample-v8"}), [])
    with pytest.raises(ValueError, match="aeroviz-training-sample-v8 file, not a closed-loop-readback set"):
        files.listed_set(training, "KAAA", "set_a")


def test_the_datum_is_the_artefacts_own_manifests(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runway_targets": {"09": {"hae_minus_msl_m": -33.0}}}), encoding="utf-8")
    recorded = hashlib.sha256(manifest.read_bytes()).hexdigest()
    sources = [{"airport": "KAAA", "arrival_manifest_sha256": recorded}]
    assert files.runway_hae_minus_msl_m(sources, "KAAA", manifest) == {"09": -33.0}
    manifest.write_text(json.dumps({"runway_targets": {"09": {"hae_minus_msl_m": -32.0}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not the arrival manifest"):
        files.runway_hae_minus_msl_m(sources, "KAAA", manifest)


# ---- the frontend's fixtures, written by the export code (outline §6 item 2)
#: Where the frontend's Vitest readers find the stage-A fixtures; `AEROVIZ_WRITE_FIXTURES=1` writes them again.
FIXTURES = Path(__file__).resolve().parents[3] / "aeroviz-4d" / "src" / "data" / "__tests__" / "fixtures" / "stage_a"
FIXTURE_SET = "fixture_set"


def stage_a_fixture() -> tuple[dict, dict]:
    """A set of one synthetic flight (`support.closed_loop_flight`: a downwind, base and final that ends unstable at the
    DA) and its index, written by the export's own functions: the observed track, the open-loop sentence and the
    closed-loop sentence at Δ = 2, 4, 8 s, each flown again and judged. The observed attitude is the no-airframe one
    (heading and path angle; a synthetic flight has no data-plane series)."""
    import torch

    from ts_transformer.autopilot import replay
    from ts_transformer.tests.support import closed_loop_flight

    flights = {interval: closed_loop_flight(interval) for interval in export.ROW_INTERVALS_S}
    one = flights[2.0]
    assert one.sentence.first_row == 0   # the observed rows from the flight's first: the open-loop sentence's own
    signals, geometry, words = one.signals, one.geometry, one.words
    attitude = {"headingDeg": signals.track_deg,
                "pathAngleDeg": np.degrees(np.arctan2(signals.vertical_rate_mps, signals.ground_speed_mps)),
                "bankRightDeg": None, "attackDeg": None}
    flight = export.flight_head(signals, f"{signals.dataset_id}_fixture", "train", "vectored", "without go-around",
                                "own", one.reading, geometry, attitude, words)
    for interval, item in flights.items():
        flown, (verdict,) = replay.fly_batch(item.batch, item.params, words, device=torch.device("cpu"))
        replayed = export.replay_payload(flown, 0, verdict, item.batch, item.sentence, {"outcome": verdict.outcome},
                                         item.inputs.aero_params[0].numpy(), words.spec, words)
        flight["closedLoop"][f"{interval:g}"] = export.closed_loop_payload(item.sentence, replayed, interval, geometry,
                                                                              words)
    source = {"instructions": "fixture/instruction_language", "executor": "fixture/executor", "specSha256": "fixture",
              "executorSpecSha256": "fixture", "git": {"head": "fixture", "dirty": False}}
    cohort = {"splits": {"train": 1, "select": 0}, "perStratum": 1, "strata": list(export.STRATA), "seed": 1337,
              "drawnFrom": "a synthetic flight"}
    sample = export.sample_of(FIXTURE_SET, geometry, {c.ident: -33.0 for c in geometry.candidates}, source, cohort,
                              words, one.params.cycle_s, [flight])
    index = {"schema": files.INDEX_SCHEMA, "writtenUtc": "fixture", "airport": geometry.code,
             "sets": [export.index_entry(FIXTURE_SET, sample)]}
    return index, sample


def test_the_frontend_fixtures_are_what_the_export_writes():
    """The fixture the frontend's readers are tested on is the export's own output today; a change of the export moves
    it, and this test says so until it is written again (``AEROVIZ_WRITE_FIXTURES=1``)."""
    index, sample = stage_a_fixture()
    texts = {files.INDEX_FILE: json.dumps(index, indent=1) + "\n",
             f"{FIXTURE_SET}/{files.SAMPLE_FILE}": json.dumps(sample, separators=(",", ":"), allow_nan=False) + "\n"}
    if os.environ.get("AEROVIZ_WRITE_FIXTURES") == "1":
        for name, text in texts.items():
            (FIXTURES / name).parent.mkdir(parents=True, exist_ok=True)
            (FIXTURES / name).write_text(text, encoding="utf-8")
    for name, text in texts.items():
        assert (FIXTURES / name).read_text(encoding="utf-8") == text, (
            f"{FIXTURES / name} is not what the export writes now: AEROVIZ_WRITE_FIXTURES=1 writes it again")
    flight = sample["flights"][0]
    assert sorted(flight["closedLoop"]) == ["2", "4", "8"]
    replayed = flight["closedLoop"]["2"]["replay"]
    assert replayed["outcome"] == "unstable_at_minimums" and replayed["crossing"]["decision"]["passed"] is False
    assert any(event["correction"] for event in flight["closedLoop"]["2"]["events"])


def test_split_flights_gives_the_row_intervals_asked_and_reads_the_stratum_off_the_first(monkeypatch, tmp_path):
    """`split_flights` (A36, stage B's export calls it with its own Δ): only the Δ given are set up and flown, in their
    order, and the flight's stratum and kind are its rows' at the first Δ given; ``formal_rows`` reads the Δ asked."""
    from types import SimpleNamespace

    import torch

    from ts_transformer.autopilot import replay
    from ts_transformer.experiments import training_flights
    from ts_transformer.tests.support import closed_loop_flight

    flown = {interval: closed_loop_flight(interval) for interval in (4.0, 8.0)}
    one = flown[4.0]
    signals, geometry, words = one.signals, one.geometry, one.words
    outcomes = {interval: replay.fly_batch(item.batch, item.params, words, device=torch.device("cpu"))[1][0].outcome
                for interval, item in flown.items()}
    series = SimpleNamespace(scenario=SimpleNamespace(source={"flight_key": f"{signals.dataset_id}_key"}))
    flights = SimpleNamespace(drawn=SimpleNamespace(signals=[signals], series=[series], groups=["own"],
                                                    geometries={geometry.code: geometry}),
                              readings=[one.reading])
    asked = []
    monkeypatch.setattr(training_flights, "open_flights", lambda *args: flights)
    monkeypatch.setattr(training_flights, "stored_closed_loop", lambda *args: {})
    monkeypatch.setattr(training_flights, "closed_loop_batch", lambda drawn, stored, interval, words: (
        asked.append(interval) or (flown[interval].batch, [flown[interval].sentence])))
    monkeypatch.setattr(export, "observed_attitude", lambda series: {
        "headingDeg": signals.track_deg, "bankRightDeg": None, "attackDeg": None,
        "pathAngleDeg": np.degrees(np.arctan2(signals.vertical_rate_mps, signals.ground_speed_mps))})
    rows = {4.0: {signals.dataset_id: {"stratum": "straight-in", "kind": "with go-around", "outcome": outcomes[4.0]}},
            8.0: {signals.dataset_id: {"stratum": "vectored", "kind": "without go-around", "outcome": outcomes[8.0]}}}
    (flight,), got = export.split_flights(Path("artefact"), "train", [signals.dataset_id], rows, one.params, words,
                                          device=torch.device("cpu"))
    assert asked == [4.0, 8.0] and list(flight["closedLoop"]) == ["4", "8"] and got is geometry
    assert [flight["closedLoop"][k]["replay"]["outcome"] for k in ("4", "8")] == [outcomes[4.0], outcomes[8.0]]
    assert (flight["stratum"], flight["kind"]) == ("straight-in", "with go-around")
    (flight,), _ = export.split_flights(Path("artefact"), "train", [signals.dataset_id], {8.0: rows[8.0]}, one.params,
                                        words, device=torch.device("cpu"))
    assert asked[2:] == [8.0] and list(flight["closedLoop"]) == ["8"] and flight["stratum"] == "vectored"
    with pytest.raises(ValueError, match="at least one row interval"):
        export.split_flights(Path("artefact"), "train", [signals.dataset_id], {}, one.params, words,
                             device=torch.device("cpu"))

    replays = tmp_path / "replay-closed-train-4s"
    replays.mkdir()
    (replays / "replay.json").write_text(json.dumps({"closed_loop": True, "row_interval_s": 4.0, "split": "train",
                                                     "flights": [{"dataset_id": "F0"}]}), encoding="utf-8")
    assert export.formal_rows(tmp_path, "train", (4.0,)) == {4.0: {"F0": {"dataset_id": "F0"}}}
    with pytest.raises(FileNotFoundError):                       # the default reads every Δ of the ablation
        export.formal_rows(tmp_path, "train")


def test_a_flight_flown_again_is_refused_unless_it_gives_its_stored_states_and_its_formal_outcome():
    import dataclasses

    import torch

    from ts_transformer.autopilot import replay
    from ts_transformer.tests.support import closed_loop_flight

    one = closed_loop_flight(4.0)
    flown, (verdict,) = replay.fly_batch(one.batch, one.params, one.words, device=torch.device("cpu"))
    aero = one.inputs.aero_params[0].numpy()
    states = one.sentence.states.copy()
    states[-1, 2] += 1e-3                                       # a stored height a millimetre off
    moved = dataclasses.replace(one.sentence, states=states)
    with pytest.raises(ValueError, match="from its closed-loop states"):
        export.replay_payload(flown, 0, verdict, one.batch, moved, {"outcome": verdict.outcome}, aero, one.words.spec,
                              one.words)
    states[-1, 2] = float("nan")
    with pytest.raises(ValueError, match="from its closed-loop states"):
        export.replay_payload(flown, 0, verdict, one.batch, dataclasses.replace(one.sentence, states=states),
                              {"outcome": verdict.outcome}, aero, one.words.spec, one.words)
    with pytest.raises(ValueError, match="the formal replay to landed"):
        export.replay_payload(flown, 0, verdict, one.batch, one.sentence, {"outcome": "landed"}, aero, one.words.spec,
                              one.words)
    other = "timeout" if verdict.outcome != "timeout" else "landed"                 # D74: the stored outcome
    with pytest.raises(ValueError, match=f"stored {other}"):
        export.replay_payload(flown, 0, verdict, one.batch, dataclasses.replace(one.sentence, outcome=other),
                              {"outcome": verdict.outcome}, aero, one.words.spec, one.words)


def test_a_set_whose_directory_exists_is_refused_before_any_airport_is_written(tmp_path):
    training = tmp_path / "training"
    (training / "set_a").mkdir(parents=True)                    # a leftover directory, not listed
    with pytest.raises(ValueError, match="exists; an export is never overwritten"):
        files.require_writable(training, "KAAA", _entry("set_a"), [])
    assert not (training / files.INDEX_FILE).exists()


def test_a_kept_flight_owns_its_arrays():
    """The split's signals and closed-loop sentences are views into one array of every flight; a kept view keeps it
    whole (0.1–0.7 GB a split), so the set's flights are copied."""
    from ts_transformer.experiments import training_flights
    from ts_transformer.tests.support import closed_loop_flight

    one = closed_loop_flight(2.0)
    signals = training_flights.owned_signals(one.signals)
    sentence = training_flights.owned_sentence(one.sentence)
    assert not np.shares_memory(signals.e_m, one.signals.e_m) and np.array_equal(signals.e_m, one.signals.e_m)
    assert not np.shares_memory(sentence.states, one.sentence.states)
    assert np.array_equal(sentence.grid, one.sentence.grid) and sentence.start == one.sentence.start


def test_the_set_s_batch_holds_no_view_into_the_loaded_closed_loop_file(monkeypatch):
    """`closed_loop_batch` gives the batch and the sentences arrays of their own — a words grid kept as a view would keep
    the split's whole file (72 MB of words at train, Δ = 2 s, an airport) — and refuses a flight without a sentence."""
    import dataclasses
    from types import SimpleNamespace

    from ts_transformer.autopilot import replay
    from ts_transformer.experiments import training_flights
    from ts_transformer.tests import test_closed_loop
    from ts_transformer.tests.support import closed_loop_flight

    one = closed_loop_flight(2.0)
    before, _, _ = test_closed_loop._batch(2.0)                  # the batch `replay.batch_of` would build
    before.drawn = {"refused_on_interval": {}}
    monkeypatch.setattr(replay, "batch_of", lambda *args: before)
    whole = np.concatenate([one.sentence.grid, one.sentence.grid])   # the file's words: this sentence is a view of it
    stored = {0: dataclasses.replace(one.sentence, grid=whole[: len(one.sentence.grid)])}
    flights = SimpleNamespace(drawn=None, readings=before.readings)
    batch, (sentence,) = training_flights.closed_loop_batch(flights, stored, 2.0, one.words)
    assert np.array_equal(batch.sentences[0].grid, one.sentence.grid) and np.array_equal(sentence.grid, one.sentence.grid)
    assert not np.shares_memory(batch.sentences[0].grid, whole) and not np.shares_memory(sentence.grid, whole)
    with pytest.raises(ValueError, match="1 of the set's flights have no closed-loop sentence at 2 s"):
        training_flights.closed_loop_batch(flights, {}, 2.0, one.words)


def test_a_flight_is_drawn_to_its_outcome_s_state_but_a_dynamics_failure_s_state_before_it():
    """One rule for the export's flown track and the live answer's (the frontend's `lastStateCycle` mirrors it)."""
    from ts_transformer.autopilot.judge import OUTCOMES
    from ts_transformer.experiments.training_flights import last_state_cycle

    assert [last_state_cycle(outcome, 300) for outcome in OUTCOMES] == [
        299 if outcome == "dynamics_failure" else 300 for outcome in OUTCOMES]
