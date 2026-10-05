"""The labelling runner and the labeller's conformance (vocabulary §7.2 #2, D21; `instructions.conformance`): the reference
written with the sentences, read again by the code in the process before every use (D73)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ts_transformer.instructions import conformance
from ts_transformer.instructions.artefact import load_sentences, write_candidates, write_signals, write_spec
from ts_transformer.tests.support import (
    fixture_days, fly_legs, instruction_airport, instruction_flight, instruction_spec as spec, landing_on,
)

LEGS = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
        (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
#: A straight-in flight: the final alone, on runway 09's course, a 3° descent ending 400 m before the threshold.
STRAIGHT_IN = [(200, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
CLEAN = {"head": "h", "dirty": False}


def _artefact(directory: Path, monkeypatch, *, straight_in: bool = False) -> Path:
    """A tmp artefact labelled by the runner: three train flights (one refused) — and with ``straight_in`` a straight-in
    flight between them — one select, one val."""
    from ts_transformer.experiments import instruction_labels

    good = fly_legs(LEGS, 270.0, 1110.0, -400.0, 0.0)
    slow = (*good[:4], np.full(len(good[0]), 10.0))                     # impossible ground speed
    direct = [instruction_flight(*fly_legs(STRAIGHT_IN, 90.0, 1300.0, -400.0, 0.0), dataset_id="KXXX:s")]
    flights = {"train": [instruction_flight(*good, dataset_id="KXXX:a"), instruction_flight(*slow, dataset_id="KXXX:b"),
                         *(direct if straight_in else []), instruction_flight(*good, dataset_id="KXXX:c")],
               "select": [instruction_flight(*good, dataset_id="KXXX:e", split="select")],
               "val": [instruction_flight(*good, dataset_id="KXXX:d", split="val")]}
    directory.mkdir()
    write_signals(directory, flights, {"note": "test"}, fixture_days())
    write_candidates(directory, {"KXXX": instruction_airport()})
    write_spec(directory, spec(), {"n": 1}, {"git": CLEAN})
    monkeypatch.setattr(instruction_labels, "CHUNK", 2)
    monkeypatch.setattr(instruction_labels, "git_state", lambda: CLEAN)
    assert instruction_labels.main(["--dir", str(directory), "--workers", "1"]) == 0
    return directory


def test_the_labels_runner_maps_each_sentence_to_its_signals_row_and_writes_the_reference(tmp_path, monkeypatch):
    directory = _artefact(tmp_path / "a", monkeypatch)
    train = load_sentences(directory, "train", spec())
    assert train["signal_index"].tolist() == [0, 2]
    assert "labeller_code_sha256" not in train                                  # D73: no digest of code
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    assert [r["dataset_id"] for r in labels["train"]["labelled"]] == ["KXXX:a", "KXXX:c"]
    assert [(r["dataset_id"], r["reason"]) for r in labels["train"]["refused"]] == [("KXXX:b", "impossible ground speed")]
    assert load_sentences(directory, "val", spec())["signal_index"].tolist() == [0]
    assert [r["dataset_id"] for r in labels["select"]["labelled"]] == ["KXXX:e"]
    reference = json.loads((directory / "conformance" / "reference.json").read_text(encoding="utf-8"))
    assert reference["schema"] == conformance.REFERENCE_SCHEMA and reference["spec_sha256"] == spec().sha256
    assert [(o["dataset_id"], o["status"]) for o in reference["outcomes"]] == [
        ("KXXX:a", "labelled"), ("KXXX:b", "refused"), ("KXXX:c", "labelled")]
    # nothing is written twice
    from ts_transformer.experiments import instruction_labels
    with pytest.raises(SystemExit):
        instruction_labels.main(["--dir", str(directory), "--workers", "1"])


def test_the_check_runs_where_the_labeller_is_used_and_refuses_a_difference_by_name(tmp_path, monkeypatch):
    """D73: `require_conforming_labeller` reads the reference again every time — the labeller in the process reads it as
    it was read: it passes with nothing run beforehand and writes nothing; a rule that moves the words is refused by
    name."""
    from ts_transformer.instructions.labeller import lateral

    directory = _artefact(tmp_path / "a", monkeypatch)
    before = sorted(p.name for p in (directory / conformance.DIRECTORY).iterdir())
    checked = conformance.require_conforming_labeller(directory)
    assert checked.passed and checked.flights == 3
    assert sorted(p.name for p in (directory / conformance.DIRECTORY).iterdir()) == before == [
        "reference.json", "reference.npz"]
    monkeypatch.setattr(lateral, "_snap", lambda value, step: np.floor(value / step) * step)
    with pytest.raises(ValueError, match="the labeller reads 2 of a's 3 reference flights otherwise: KXXX:a"):
        conformance.require_conforming_labeller(directory)

def test_the_conformance_fails_when_a_labeller_rule_changes(tmp_path, monkeypatch):
    from ts_transformer.instructions.labeller import lateral, read

    directory = _artefact(tmp_path / "a", monkeypatch)
    # the heading grid read toward the lower cell: the words move
    monkeypatch.setattr(lateral, "_snap", lambda value, step: np.floor(value / step) * step)
    checked = conformance.check(directory)
    assert set(checked.mismatches) == {"KXXX:a", "KXXX:c"}
    assert all("the words differ (2 s rows)" in problems for problems in checked.mismatches.values())
    monkeypatch.undo()
    # a refusal that moves is found too: the ground-speed refusal no longer refuses
    original = read.admit

    def lenient(signals, geometry, one):
        return original(signals, geometry, spec(ground_speed_floor_mps=5.0))
    monkeypatch.setattr(read, "admit", lenient)
    checked = conformance.check(directory)
    assert checked.mismatches["KXXX:b"] == [
        "reason: 'ground speed outside the speed words', the reference 'impossible ground speed'"]


def test_the_conformance_reads_the_interval_grids_and_finds_a_change_of_their_rule(tmp_path, monkeypatch):
    """D49: the reference holds each labelled sentence on its UTC Δ grid at 4 and 8 s beside its 2 s rows, and the
    check finds a change of the Δ grid's own rule — a tie read at the earlier row (D45) — which leaves the 2 s words as
    they are."""
    from ts_transformer.instructions.labeller import interval

    directory = _artefact(tmp_path / "a", monkeypatch)
    reference = json.loads((directory / "conformance" / "reference.json").read_text(encoding="utf-8"))
    assert reference["intervals"] == ["2", "4", "8"]
    assert all(o["interval_refusals"] == {} for o in reference["outcomes"] if o["status"] == "labelled")
    with np.load(directory / "conformance" / "reference.npz") as data:
        assert {"words_2s", "words_4s", "words_8s"} <= set(data.files)
        assert len(data["word_offsets_4s"]) == 3 and data["word_offsets_4s"][-1] > 0
    monkeypatch.setattr(interval, "last_heard_row", lambda row, every: int(row + every / 2.0))
    checked = conformance.check(directory)
    assert set(checked.mismatches) == {"KXXX:a", "KXXX:c"}
    problems = {p for found in checked.mismatches.values() for p in found}
    assert "the words differ (2 s rows)" not in problems and "the words differ (4 s rows)" in problems


def test_a_delta_grid_that_refuses_a_sentence_is_held_as_its_reason_and_no_rows(tmp_path, monkeypatch):
    """D49: a Δ grid that refuses a labelled sentence is in the reference as its reason (no rows); the check reads the
    same, and finds a grid that no longer refuses."""
    from ts_transformer.instructions.labeller import interval

    original = interval.on_utc_grid

    def refuse_8(grid, entry, interval_s, *rest):
        if interval_s == 8.0:
            raise conformance.Refused("too short", "no row on the 8 s grid")
        return original(grid, entry, interval_s, *rest)
    monkeypatch.setattr(conformance, "on_utc_grid", refuse_8)
    directory = _artefact(tmp_path / "a", monkeypatch)
    reference = json.loads((directory / "conformance" / "reference.json").read_text(encoding="utf-8"))
    assert all(o["interval_refusals"] == {"8": "too short"} for o in reference["outcomes"] if o["status"] == "labelled")
    with np.load(directory / "conformance" / "reference.npz") as data:
        assert data["word_offsets_8s"].tolist() == [0, 0, 0] and data["word_offsets_4s"][-1] > 0
    assert conformance.check(directory).passed
    monkeypatch.setattr(conformance, "on_utc_grid", original)
    checked = conformance.check(directory)
    assert set(checked.mismatches) == {"KXXX:a", "KXXX:c"}
    assert all(any(p.startswith("interval_refusals") for p in found) and "the words differ (8 s rows)" in found
               for found in checked.mismatches.values())


def test_a_reference_of_another_schema_or_spec_is_refused(tmp_path, monkeypatch):
    directory = _artefact(tmp_path / "a", monkeypatch)
    path = directory / "conformance" / "reference.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**payload, "schema": "ts-instruction-conformance-reference-v0"}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"is not a {conformance.REFERENCE_SCHEMA} file"):
        conformance.check(directory)
    path.write_text(json.dumps({**payload, "spec_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="read with spec 000000000000"):
        conformance.check(directory)


def test_the_check_refuses_a_reference_flight_of_another_split_s_day(tmp_path, monkeypatch):
    """C32 (A32): the check reads the reference's flights only if each lands on a train day of the artefact's split."""
    directory = _artefact(tmp_path / "a", monkeypatch)
    assert conformance.check(directory).passed
    path = directory / conformance.DIRECTORY / "reference.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["flights"][0]["landing_time_utc"] = landing_on("select")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lands on a select day, not a train day"):
        conformance.check(directory)


def test_the_draw_takes_labelled_and_refused_flights_of_every_airport_and_the_go_arounds_besides(monkeypatch):
    """§7.2 #2: the first labelled and refused flights of each airport, then up to `GO_AROUND_PER_AIRPORT` more labelled
    flights with a go-around of each airport not drawn already — as many as there are."""
    monkeypatch.setattr(conformance, "PER_AIRPORT", 2)
    monkeypatch.setattr(conformance, "REFUSED_PER_AIRPORT", 1)
    monkeypatch.setattr(conformance, "GO_AROUND_PER_AIRPORT", 2)
    flights = [type("F", (), {"airport": airport})() for airport in ["KAAA"] * 10 + ["KBBB"] * 4]
    statuses = ["labelled", "refused"] * 7
    go_arounds = [status == "labelled" for status in statuses]          # every labelled flight has a go-around
    chosen = conformance.draw(flights, statuses, go_arounds)
    assert chosen == sorted(chosen) and chosen == conformance.draw(flights, statuses, go_arounds)
    picked = [(flights[i].airport, statuses[i]) for i in chosen]
    # KAAA: 2 labelled + 1 refused + 2 more with a go-around; KBBB has only 2 labelled, both drawn already: none more
    assert sorted(picked) == [("KAAA", "labelled")] * 4 + [("KAAA", "refused")] + \
        [("KBBB", "labelled")] * 2 + [("KBBB", "refused")]
    without = conformance.draw(flights, statuses, [False] * len(flights))     # no go-around: the usual draw alone
    assert len(without) == 6 and set(without) < set(chosen)



def test_the_sentence_file_stores_each_flight_s_stratum_from_the_one_definition(tmp_path, monkeypatch):
    """D70: the stored stratum of each labelled flight is `readout.stratum` of its reading, by its name in `STRATA`."""
    from ts_transformer.instructions.artefact import load_candidates, load_signals
    from ts_transformer.instructions.labeller.read import read_flight
    from ts_transformer.instructions.readout import STRATA, stratum

    directory = _artefact(tmp_path / "a", monkeypatch, straight_in=True)
    for split in ("train", "select", "val"):
        stored = load_sentences(directory, split, spec())
        signals = load_signals(directory, split)
        geometry = load_candidates(directory)["KXXX"]
        expected = [stratum(read_flight(signals[int(i)], geometry, spec())) for i in stored["signal_index"]]
        assert stored["stratum"].tolist() == expected and set(expected) <= set(STRATA)
    # in signal order: two 90° turns, the final alone, two 90° turns
    assert load_sentences(directory, "train", spec())["stratum"].tolist() == ["vectored", "straight-in", "vectored"]


def test_a_flight_is_vectored_from_90_degrees_of_turns_before_the_capture_row():
    from types import SimpleNamespace

    from ts_transformer.instructions.readout import stratum

    assert stratum(SimpleNamespace(checks={"turning_deg": 90.0})) == "vectored"
    assert stratum(SimpleNamespace(checks={"turning_deg": 89.0})) == "straight-in"


def test_a_changed_stratum_fails_the_conformance_by_name(tmp_path, monkeypatch):
    """D70: the labeller conformance compares the stratum as it compares the words — a rule that moves it fails there."""
    from ts_transformer.instructions import readout

    directory = _artefact(tmp_path / "a", monkeypatch, straight_in=True)
    assert conformance.check(directory).passed
    monkeypatch.setattr(readout, "VECTORED_TURN_DEG", 1000.0)       # every flight straight-in: the vectored ones move
    checked = conformance.check(directory)
    assert checked.mismatches == {flight: ["stratum: 'straight-in', the reference 'vectored'"]
                                  for flight in ("KXXX:a", "KXXX:c")}


def test_a_sentence_file_of_the_former_format_is_refused_by_name(tmp_path, monkeypatch):
    """D70: the sentence file before the stratum (`ts-instruction-sentences-v4`) is refused by its name; so is a stratum
    that is no name of `STRATA`."""
    from ts_transformer.instructions.artefact import SENTENCES_SCHEMA

    directory = _artefact(tmp_path / "a", monkeypatch)
    with np.load(directory / "sentences_train.npz") as data:
        arrays = {name: data[name] for name in data.files}
    for name, changes in (("old", {"schema": np.array("ts-instruction-sentences-v4")}),
                          ("odd", {"stratum": np.array(["vectored", "circling"])})):
        (tmp_path / name).mkdir()
        np.savez_compressed(tmp_path / name / "sentences_train.npz", **{**arrays, **changes})
    with pytest.raises(ValueError, match=f"is not a {SENTENCES_SCHEMA} file"):
        load_sentences(tmp_path / "old", "train", spec())
    with pytest.raises(ValueError, match="names strata \\['circling'\\]"):
        load_sentences(tmp_path / "odd", "train", spec())
