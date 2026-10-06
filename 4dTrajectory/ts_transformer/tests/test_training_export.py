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
from ts_transformer.instructions.words import RUNWAY, RUNWAY_GO_AROUND, UNCHANGED


def _strata(airport, strata):
    """The sentence file's strata by signal index — ``strata`` a list of (stratum, count) for ``airport`` — and the
    split's flight records, ids F0, F1, … (and G0, another airport's)."""
    by_index, records = {}, []
    for stratum, count in strata:
        for _ in range(count):
            by_index[len(records)] = stratum
            records.append({"dataset_id": f"F{len(records)}", "airport": airport})
    by_index[len(records)] = "vectored"
    records.append({"dataset_id": "G0", "airport": "KBBB"})
    return by_index, records


def test_each_split_draws_per_stratum_flights_with_a_sentence_at_every_row_interval_seeded():
    """D86: the flights with a closed-loop sentence at every Δ, by the sentence file's stratum (D70), in a seeded
    permutation — no formal replay row."""
    strata, records = _strata("KAAA", [("straight-in", 6), ("vectored", 5)])
    every = set(strata)
    indices = [every, every, every - {0}]                                  # F0 has no sentence at 8 s
    chosen = export.choose(indices, strata, records, "KAAA", 3, 1337)
    assert len(chosen) == 6 and "F0" not in chosen and "G0" not in chosen
    by_id = {r["dataset_id"]: strata[k] for k, r in enumerate(records)}
    assert [by_id[d] for d in chosen] == ["straight-in"] * 3 + ["vectored"] * 3
    assert export.choose(indices, strata, records, "KAAA", 3, 1337) == chosen \
        != export.choose(indices, strata, records, "KAAA", 3, 7)
    with pytest.raises(ValueError, match="5 straight-in flights with a closed-loop sentence at every row interval, 6"):
        export.choose(indices, strata, records, "KAAA", 6, 1337)


def test_the_airports_draw_reads_each_flights_stratum_by_its_signal_index_in_any_row_order(monkeypatch):
    """A37: `build_airport` takes each flight's stratum from the sentence file by signal index — a file whose rows are
    not in signal order (here reversed) draws the same flights, each of its own stratum."""
    from types import SimpleNamespace

    strata, records = _strata("KAAA", [("straight-in", 6), ("vectored", 5)])
    order = sorted(strata)
    for rows in (order, order[::-1]):
        drawn = {}
        monkeypatch.setattr(export, "load_sentences", lambda directory, split, spec, fields: {
            "signal_index": np.array(rows), "stratum": np.array([strata[i] for i in rows])})
        monkeypatch.setattr(export, "closed_loop_indices", lambda directory, split, interval, spec: set(strata))
        monkeypatch.setattr(export, "signals_flights", lambda directory, split: records)
        monkeypatch.setattr(export, "split_flights", lambda directory, split, chosen, intervals, params, words, *, device:
                            (drawn.setdefault(split, list(chosen)) and [], "geometry"))
        export.build_airport("KAAA", None, None, SimpleNamespace(spec=None), per_stratum=3, seed=1337, device=None)
        if rows is order:
            first = dict(drawn)
        assert drawn == first
    by_id = {r["dataset_id"]: strata[k] for k, r in enumerate(records)}
    assert [by_id[d] for d in first["train"]] == ["straight-in"] * 3 + ["vectored"] * 3


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
    return {"schema": files.SAMPLE_SCHEMA, "setId": set_id, "airport": airport, "readingRule": READING_RULE}


def test_a_set_goes_beside_the_old_index_and_is_never_overwritten(tmp_path):
    """Outline §6 item 3: the set and its entry go into `index_v5.json`; the instruction-v3 view's `index.json` is never
    read or written; a set already listed, or an index another run changed, is refused."""
    training = tmp_path / "KAAA" / "training"
    training.mkdir(parents=True)
    old = training / "index.json"
    old.write_text('{"schema": "aeroviz-training-index-v1", "sets": []}', encoding="utf-8")
    before = old.read_bytes()
    existing = files.FILES.read_index(training, "KAAA", "set_a")
    assert existing == []
    files.FILES.write_set(training, "KAAA", _entry("set_a"), files.serialise(_sample("set_a", "KAAA")), existing)
    assert old.read_bytes() == before
    index = json.loads((training / files.INDEX_FILE).read_text(encoding="utf-8"))
    assert index["schema"] == files.INDEX_SCHEMA and [s["id"] for s in index["sets"]] == ["set_a"]
    entry, sample = files.FILES.listed_set(training, "KAAA", "set_a")
    assert entry["id"] == sample["setId"] == "set_a"
    with pytest.raises(ValueError, match="already lists set set_a"):
        files.FILES.read_index(training, "KAAA", "set_a")
    stale = files.FILES.read_index(training, "KAAA", "set_b")
    files.FILES.write_set(training, "KAAA", _entry("set_c"), files.serialise(_sample("set_c", "KAAA")), stale)
    with pytest.raises(ValueError, match="changed since this run read it"):
        files.FILES.write_set(training, "KAAA", _entry("set_b"), files.serialise(_sample("set_b", "KAAA")), stale)
    with pytest.raises(files.NotListed):
        files.FILES.listed_set(training, "KAAA", "set_b")
    with pytest.raises(ValueError, match="not KBBB's"):
        files.FILES.read_index(training, "KBBB", "set_d")
    with pytest.raises(ValueError, match="refusing NaN|Out of range float"):
        files.serialise({"x": float("nan")})


def test_a_listed_set_of_another_format_is_refused_by_name(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    files.FILES.write_set(training, "KAAA", _entry("set_a"),
                          files.serialise({**_sample("set_a", "KAAA"), "schema": "aeroviz-training-sample-v8"}), [])
    with pytest.raises(ValueError, match="aeroviz-training-sample-v8 file, not a closed-loop-readback set"):
        files.FILES.listed_set(training, "KAAA", "set_a")
    # the reading is the sample's own (top level, every stage): another one is refused, whatever its vocabulary says
    other = tmp_path / "other"
    other.mkdir()
    files.FILES.write_set(other, "KAAA", _entry("set_a"),
                          files.serialise({**_sample("set_a", "KAAA"), "readingRule": "instruction-v5"}), [])
    with pytest.raises(ValueError, match="holds set set_a at KAAA"):
        files.FILES.listed_set(other, "KAAA", "set_a")


def test_each_candidates_height_offset_comes_from_the_published_runway_data(tmp_path, monkeypatch):
    """The export's HAE − MSL offset of each candidate comes from the published runway data its candidates were read
    from (A37, D78: a candidate needs no arrival), refused by name when those files changed (by sha256) or when the data
    gives a candidate no offset."""
    from types import SimpleNamespace

    from ts_transformer.tests.support import instruction_airport

    config, cifp = tmp_path / "runway_thresholds.json", tmp_path / "FAACIFP18"
    config.write_text("{}", encoding="utf-8")
    cifp.write_text("cifp", encoding="utf-8")
    monkeypatch.setattr(export, "DEFAULT_CONFIG", config)
    monkeypatch.setattr(export, "DEFAULT_CIFP", cifp)
    published = [SimpleNamespace(ident="09", hae_minus_msl_m=-33.0), SimpleNamespace(ident="27", hae_minus_msl_m=-34.0)]
    monkeypatch.setattr(export, "load_airport", lambda code, config_file, cifp_file: SimpleNamespace(runways=published))
    recorded = {"config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "cifp_sha256": hashlib.sha256(cifp.read_bytes()).hexdigest()}
    geometry = instruction_airport()                                      # one candidate, 09
    assert export.candidate_hae_minus_msl_m(recorded, geometry) == {"09": -33.0}
    with pytest.raises(ValueError, match="is not the cifp the artefact's candidates were read from"):
        export.candidate_hae_minus_msl_m({**recorded, "cifp_sha256": "0" * 64}, geometry)
    published.pop(0)
    with pytest.raises(ValueError, match=r"KXXX: the published runway data gives no height offset for candidate\(s\) \['09'\]"):
        export.candidate_hae_minus_msl_m(recorded, geometry)


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
    assert one.sentence.rows.first_row == 0   # the observed rows from the flight's first: the open-loop sentence's own
    signals, geometry, words = one.signals, one.geometry, one.words
    attitude = {"headingDeg": signals.track_deg,
                "pathAngleDeg": np.degrees(np.arctan2(signals.vertical_rate_mps, signals.ground_speed_mps)),
                "bankRightDeg": None, "attackDeg": None}
    flight = export.flight_head(signals, f"{signals.dataset_id}_fixture", "train", "vectored", "without go-around",
                                "own", one.reading, geometry, attitude, words)
    for interval, item in flights.items():
        flown, (verdict,) = replay.fly_batch(item.batch, item.params, words, device=torch.device("cpu"))
        replayed = export.replay_payload(flown, 0, verdict, item.batch, item.sentence,
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


def test_split_flights_gives_the_row_intervals_asked_of_any_split_from_the_stored_sentences(monkeypatch):
    """`split_flights` (A36; D86): only the Δ given are set up and flown, in their order, one at a time, for any split —
    val included, no formal replay row read — the flight's stratum its stored sentence's at the first Δ given (the
    sentence file's, D70) and its kind whether that closed-loop sentence says a go-around, each sentence checked against
    its stored outcome."""
    from dataclasses import replace
    from types import SimpleNamespace

    import torch

    from ts_transformer.experiments import training_flights
    from ts_transformer.tests.support import closed_loop_flight

    flown = {interval: closed_loop_flight(interval) for interval in (4.0, 8.0)}
    one = flown[4.0]
    signals, geometry, words = one.signals, one.geometry, one.words
    said = one.sentence.rows.grid.copy()
    said[-1, RUNWAY] = RUNWAY_GO_AROUND                                     # the 4 s sentence says a go-around
    stored = {4.0: {0: replace(one.sentence, rows=replace(one.sentence.rows, grid=said),
                               withheld=replace(one.sentence.withheld, stratum="straight-in"))},
              8.0: {0: replace(flown[8.0].sentence, withheld=replace(flown[8.0].sentence.withheld, stratum="vectored",
                                                                     go_around_rows=np.array([5])))}}
    series = SimpleNamespace(scenario=SimpleNamespace(source={"flight_key": f"{signals.dataset_id}_key"}))
    flights = SimpleNamespace(drawn=SimpleNamespace(indices=[0], signals=[signals], series=[series], groups=["own"],
                                                    geometries={geometry.code: geometry}),
                              readings=[one.reading])
    asked = []
    monkeypatch.setattr(training_flights, "open_flights", lambda *args: flights)
    monkeypatch.setattr(training_flights, "stored_closed_loop", lambda instructions, split, interval, words: (
        stored[interval]))
    monkeypatch.setattr(export, "closed_loop_indices", lambda instructions, split, interval, spec: set(stored[interval]))
    monkeypatch.setattr(training_flights, "closed_loop_batch", lambda instructions, split, drawn, by_index, interval,
                        params, words: asked.append((split, interval)) or (flown[interval].batch, [by_index[0]]))
    monkeypatch.setattr(export, "observed_attitude", lambda series: {
        "headingDeg": signals.track_deg, "bankRightDeg": None, "attackDeg": None,
        "pathAngleDeg": np.degrees(np.arctan2(signals.vertical_rate_mps, signals.ground_speed_mps))})
    cpu = torch.device("cpu")
    (flight,), got = export.split_flights(Path("artefact"), "train", [signals.dataset_id], (4.0, 8.0), one.params, words,
                                          device=cpu)
    assert asked == [("train", 4.0), ("train", 8.0)] and list(flight["closedLoop"]) == ["4", "8"] and got is geometry
    assert [flight["closedLoop"][k]["replay"]["outcome"] for k in ("4", "8")] == [
        stored[4.0][0].withheld.outcome, stored[8.0][0].withheld.outcome]
    assert (flight["stratum"], flight["kind"]) == ("straight-in", "with go-around")
    (flight,), _ = export.split_flights(Path("artefact"), "val", [signals.dataset_id], (8.0,), one.params, words,
                                        device=cpu)                                             # D86: val too
    assert asked[2:] == [("val", 8.0)] and list(flight["closedLoop"]) == ["8"]
    assert (flight["split"], flight["stratum"], flight["kind"]) == ("val", "vectored", "without go-around")
    with pytest.raises(ValueError, match="at least one row interval"):
        export.split_flights(Path("artefact"), "train", [signals.dataset_id], (), one.params, words, device=cpu)
    del stored[8.0][0]                                                  # no sentence at a Δ asked: refused by name
    with pytest.raises(ValueError, match="have no closed-loop sentence at every row interval of \\[4.0, 8.0\\]"):
        export.split_flights(Path("artefact"), "train", [signals.dataset_id], (4.0, 8.0), one.params, words, device=cpu)


def test_a_flight_flown_again_is_refused_unless_it_gives_its_stored_states_and_its_stored_outcome():
    """D86: a closed-loop sentence flown again must give its stored states on every 2 s row and its stored outcome
    (D74) — no formal replay row is read."""
    import dataclasses

    import torch

    from ts_transformer.autopilot import replay
    from ts_transformer.tests.support import closed_loop_flight

    one = closed_loop_flight(4.0)
    flown, (verdict,) = replay.fly_batch(one.batch, one.params, one.words, device=torch.device("cpu"))
    aero = one.inputs.aero_params[0].numpy()
    sentence = one.sentence
    export.replay_payload(flown, 0, verdict, one.batch, sentence, aero, one.words.spec, one.words)
    states = sentence.rows.states.copy()
    states[-1, 2] += 1e-3                                       # a stored height a millimetre off
    for off in (states, np.where(np.arange(len(states))[:, None] == len(states) - 1, np.nan, states)):
        moved = dataclasses.replace(sentence, rows=dataclasses.replace(sentence.rows, states=off))
        with pytest.raises(ValueError, match="from its closed-loop states"):
            export.replay_payload(flown, 0, verdict, one.batch, moved, aero, one.words.spec, one.words)
    other = "timeout" if verdict.outcome != "timeout" else "landed"                 # D74: the stored outcome
    judged = dataclasses.replace(sentence, withheld=dataclasses.replace(sentence.withheld, outcome=other))
    with pytest.raises(ValueError, match=f"flown again to {verdict.outcome}, stored {other}"):
        export.replay_payload(flown, 0, verdict, one.batch, judged, aero, one.words.spec, one.words)


def test_an_envelope_past_the_flown_track_is_refused(monkeypatch):
    """Outline §6.2 item 8 (D135): `flown_sentence` refuses envelopes that end past the flight's flown track — the
    judge's envelopes replaced by ones whose altitude tube ends one row past the track's rows; one ending at the track's
    end is kept."""
    import torch

    from ts_transformer.autopilot import replay
    from ts_transformer.tests.support import closed_loop_flight

    one = closed_loop_flight(4.0)
    flown, (verdict,) = replay.fly_batch(one.batch, one.params, one.words, device=torch.device("cpu"))
    aero = one.inputs.aero_params[0].numpy()
    block = export.replay_payload(flown, 0, verdict, one.batch, one.sentence, aero, one.words.spec, one.words)
    rows = block["track"]["rows"]
    real = export.envelopes

    def past(*args, **kwargs):
        judged = real(*args, **kwargs)
        return {**judged, "altitude": [{**tube, "endRow": rows + 1} for tube in judged["altitude"]] or
                [{"endRow": rows + 1}]}

    monkeypatch.setattr(export, "envelopes", past)
    with pytest.raises(ValueError, match=f"an envelope ends at row {rows + 1}, past the flown track's {rows} rows"):
        export.replay_payload(flown, 0, verdict, one.batch, one.sentence, aero, one.words.spec, one.words)
    rows -= 1                                                    # an end at the track's own end (exclusive): kept
    assert export.replay_payload(flown, 0, verdict, one.batch, one.sentence, aero, one.words.spec, one.words)


def test_a_set_whose_directory_exists_is_refused_before_any_airport_is_written(tmp_path):
    training = tmp_path / "training"
    (training / "set_a").mkdir(parents=True)                    # a leftover directory, not listed
    with pytest.raises(ValueError, match="exists; an export is never overwritten"):
        files.FILES.require_writable(training, "KAAA", _entry("set_a"), [])
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
    assert not np.shares_memory(sentence.rows.states, one.sentence.rows.states)
    assert not np.shares_memory(sentence.withheld.lateral_m, one.sentence.withheld.lateral_m)
    assert np.array_equal(sentence.rows.grid, one.sentence.rows.grid) and sentence.rows.start == one.sentence.rows.start
    assert sentence.withheld.outcome == one.sentence.withheld.outcome


def test_the_set_s_batch_holds_no_view_into_the_loaded_closed_loop_file(monkeypatch):
    """`closed_loop_batch` gives the batch and the sentences arrays of their own — a words grid kept as a view would keep
    the split's whole file (72 MB of words at train, Δ = 2 s, an airport) — refuses a flight without a sentence, and runs
    the start's refusals on the set's sentences (`start.require_startable`, D67)."""
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
    grid = one.sentence.rows.grid
    whole = np.concatenate([grid, grid])                          # the file's words: this sentence is a view of it
    stored = {0: dataclasses.replace(one.sentence, rows=dataclasses.replace(one.sentence.rows, grid=whole[: len(grid)]))}
    flights = SimpleNamespace(drawn=SimpleNamespace(indices=[7], signals=before.observed, geometries={}),
                              readings=before.readings)
    before.indices = [7]
    stored = {7: stored[0]}
    started = []
    monkeypatch.setattr(training_flights, "require_startable", lambda instructions, split, interval_s, sentences, signals,
                        geometries, params, words: started.append((instructions, split, interval_s, sorted(sentences),
                                                                   signals, params)))
    batch, (sentence,) = training_flights.closed_loop_batch(Path("artefact"), "train", flights, stored, 2.0, one.params,
                                                            one.words)
    assert started == [(Path("artefact"), "train", 2.0, [7], {7: before.observed[0]}, one.params)]   # by signal index
    assert np.array_equal(batch.sentences[0].grid, grid) and np.array_equal(sentence.rows.grid, grid)
    assert not np.shares_memory(batch.sentences[0].grid, whole) and not np.shares_memory(sentence.rows.grid, whole)
    with pytest.raises(ValueError, match="1 of the set's flights have no closed-loop sentence at 2 s"):
        training_flights.closed_loop_batch(Path("artefact"), "train", flights, {}, 2.0, one.params, one.words)


def test_a_flight_is_drawn_to_its_outcome_s_state_but_a_dynamics_failure_s_state_before_it():
    """One rule for the export's flown track and the live answer's (the frontend's `lastStateCycle` mirrors it)."""
    from ts_transformer.autopilot.judge import OUTCOMES
    from ts_transformer.experiments.training_flights import last_state_cycle

    assert [last_state_cycle(outcome, 300) for outcome in OUTCOMES] == [
        299 if outcome == "dynamics_failure" else 300 for outcome in OUTCOMES]


#: The blocks of the flown sentences in the three stages' fixtures before D135 (outline §6.2 item 8), by the sha256 of
#: their canonical JSON (`_digest`), the envelopes apart: stage A's as written then (its track rounded), B's and C's
#: (their tracks unrounded, D127), and stage A's envelopes. Written from the fixtures of 65314499 before they were
#: written again.
EARLIER_BLOCKS = {
    "A": {"2": "d3087166338e03edccfbdc2a93055f0330a0e55b396d2865f00c19578a18294c",
          "4": "9635a26b7071eccc0d275f92f0f0fd575fa3e60aaec77441b64d1d3b8469d454",
          "8": "6027c3b98b1189e7183a4c0f90423a1fdff9482c40655e788a2d4864dcc377ac"},
    "A envelopes": {"2": "d84f98d6382bc916c477b5872c7e532c469056fadcbc3758ca25f203fabf6011",
                    "4": "51a58e8dbf25305aadbbf4c0a38f37276c10f682c424baad0a7c2a7d62b5aca8",
                    "8": "f84eb5558825b1fb33c4119f6aba096e1de2ca257b34a0cfc40af92ca6d0f777"},
    "B": ["55455e7bdbafab588ca2481bc04a57e5f401e76a76e8bbb23f080fb58b101a9e",
          "607a9c3b2b5006efa28b981ee5d9b30a5b6a83527c1cc8cea54547b7a05bb618"],
    "C": ["ed2dffb65df1eb4b2cba44fda92087bb9f171b8992e92821003ad5f503c4a2a4",
          "527b569e4c8b6e8dbca057bc4a40dfc1ad078009172fbb1ede07fea518d7760e",
          "f400c0a43aca8a05fc1727d838142e82660195dbf26eee435919b55981962e7e"],
}
#: The digits stage A's track was written to before D135 (`files.rounded`).
EARLIER_TRACK_DIGITS = {"eM": 1, "nM": 1, "latDeg": 7, "lonDeg": 7, "heightMslM": 1, "trackDeg": 2, "groundSpeedMps": 2,
                        "verticalRateMps": 2}
_BLOCK = ("outcome", "endCycle", "crossing", "track", "attitude")


def _digest(block) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(block, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def test_the_flown_block_of_every_stage_is_its_earlier_one_with_the_envelopes():
    """Outline §6.2 item 8 (D135): `flown_sentence` gives stage A's earlier block within its earlier rounding (its
    envelopes unchanged), and B's and C's earlier blocks exactly, apart from the envelopes it adds — read from the
    frontend's fixtures, which the export writes (each stage's fixture test)."""
    fixtures = FIXTURES.parent
    a = json.loads((fixtures / "stage_a" / FIXTURE_SET / files.SAMPLE_FILE).read_text(encoding="utf-8"))
    for interval, sentence in a["flights"][0]["closedLoop"].items():
        replayed = sentence["replay"]
        block = {key: replayed[key] for key in (*_BLOCK, "flewTheSentence", "notReached")}
        block["track"] = {key: [round(v, EARLIER_TRACK_DIGITS[key]) for v in value] if key in EARLIER_TRACK_DIGITS
                          else value for key, value in block["track"].items()}
        assert _digest(block) == EARLIER_BLOCKS["A"][interval], interval
        assert _digest(replayed["envelopes"]) == EARLIER_BLOCKS["A envelopes"][interval], interval
    b = json.loads((fixtures / "stage_b" / "fixture_set" / "sample.json").read_text(encoding="utf-8"))
    c = json.loads((fixtures / "stage_c" / "fixture-windows" / "sample.json").read_text(encoding="utf-8"))
    for stage, sentences in (("B", [s for f in b["flights"] for s in f["prior"]]),
                             ("C", [r for w in c["windows"] for r in w["rounds"]])):
        assert [_digest({key: s[key] for key in (*_BLOCK, "timedOut", "goArounds")}) for s in sentences] \
            == EARLIER_BLOCKS[stage], stage
        assert all(s["envelopes"] is not None and s["envelopes"]["heading"] for s in sentences), stage


def test_a_path_recorded_in_another_checkout_is_read_through_the_live_data_link(tmp_path):
    """Outline §5 rule 1: each checkout's linked data trees are links to the live ones; a path recorded under one of them
    — in the main checkout or in any worktree, which lie under the main one, even one deleted since — is the same path
    under this checkout's tree; a relative path is the repository's; others as recorded."""
    main = tmp_path / "thesis"
    live = main / "4dTrajectory" / "outputs"
    (live / "POOLED").mkdir(parents=True)
    (main / "aeroviz-4d" / "public" / "data" / "airports").mkdir(parents=True)
    worktree = main / ".claude" / "worktrees" / "v4"
    for tree in export.LINKED_TREES:
        (worktree / tree).parent.mkdir(parents=True, exist_ok=True)
        (worktree / tree).symlink_to(main / tree)
    other = main / ".claude" / "worktrees" / "v4-post"            # recorded there, deleted since: never resolved
    for here in (worktree, main):
        assert export.this_checkout(live / "POOLED" / "x", here) == here / "4dTrajectory/outputs/POOLED/x"
        assert export.this_checkout(other / "4dTrajectory/outputs/POOLED/x", here) == here / "4dTrajectory/outputs/POOLED/x"
        assert export.this_checkout(other / "aeroviz-4d/public/data/airports", here) == here / "aeroviz-4d/public/data/airports"
        assert export.this_checkout("4dTrajectory/outputs/POOLED/x", here) == here / "4dTrajectory/outputs/POOLED/x"
        assert export.this_checkout(main / "aeroviz-4d" / "src", here) == main / "aeroviz-4d" / "src"   # not a linked tree
        assert export.this_checkout(tmp_path / "elsewhere", here) == tmp_path / "elsewhere"
        assert export.this_checkout(f"{live}/../../x", here) == main / "x"                  # normalised, not rewritten
