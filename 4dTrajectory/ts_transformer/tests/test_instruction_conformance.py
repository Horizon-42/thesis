"""The labelling runner and the labeller's conformance (design §9.2 #2, D21; `instructions.conformance`): the reference
written with the sentences, read again by today's code, the passed record that later runners ask for."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ts_transformer.instructions import conformance
from ts_transformer.instructions.artefact import load_sentences, write_candidates, write_signals, write_spec
from ts_transformer.tests.support import (
    fixture_days, fly_legs, instruction_airport, instruction_flight, instruction_spec as spec,
)

LEGS = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
        (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
CLEAN = {"head": "h", "dirty": False}


def _artefact(directory: Path, monkeypatch) -> Path:
    """A tmp artefact labelled by the runner: three train flights (one refused), one select, one val."""
    from ts_transformer.experiments import instruction_labels

    good = fly_legs(LEGS, 270.0, 1110.0, -400.0, 0.0)
    slow = (*good[:4], np.full(len(good[0]), 10.0))                     # impossible ground speed
    flights = {"train": [instruction_flight(*good, dataset_id="KXXX:a"), instruction_flight(*slow, dataset_id="KXXX:b"),
                         instruction_flight(*good, dataset_id="KXXX:c")],
               "select": [instruction_flight(*good, dataset_id="KXXX:e", split="select")],
               "val": [instruction_flight(*good, dataset_id="KXXX:d", split="val")]}
    directory.mkdir()
    write_signals(directory, flights, {"note": "test"}, fixture_days())
    write_candidates(directory, {"KXXX": instruction_airport()})
    write_spec(directory, spec(), {"n": 1}, {"labeller_code_sha256": conformance.labeller_code_sha256(), "git": CLEAN})
    monkeypatch.setattr(instruction_labels, "CHUNK", 2)
    monkeypatch.setattr(instruction_labels, "git_state", lambda: CLEAN)
    assert instruction_labels.main(["--dir", str(directory), "--workers", "1"]) == 0
    return directory


def test_the_labels_runner_maps_each_sentence_to_its_signals_row_and_writes_the_reference(tmp_path, monkeypatch):
    directory = _artefact(tmp_path / "a", monkeypatch)
    train = load_sentences(directory, "train", spec())
    assert train["signal_index"].tolist() == [0, 2]
    assert str(train["labeller_code_sha256"]) == conformance.labeller_code_sha256()
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


def test_the_conformance_passes_on_the_same_code_and_writes_the_record_runners_ask_for(tmp_path, monkeypatch):
    directory = _artefact(tmp_path / "a", monkeypatch)
    with pytest.raises(ValueError, match="has not been checked"):
        conformance.require_conforming_labeller(directory)
    checked = conformance.check(directory, git=CLEAN)
    assert checked.passed and checked.flights == 3
    with pytest.raises(RuntimeError, match="clean checkout"):
        conformance.write_passed(directory, conformance.check(directory, git={"head": "h", "dirty": True}),
                                 git={"head": "h", "dirty": True})
    path = conformance.write_passed(directory, checked, git=CLEAN)
    assert path.name == f"passed-{conformance.labeller_code_sha256()[:12]}.json"
    conformance.require_conforming_labeller(directory)
    # another labeller code: its own record is missing
    monkeypatch.setattr(conformance, "labeller_code_sha256", lambda: "f" * 64)
    with pytest.raises(ValueError, match="has not been checked"):
        conformance.require_conforming_labeller(directory)


def test_the_conformance_fails_when_a_labeller_rule_changes(tmp_path, monkeypatch):
    from ts_transformer.instructions.labeller import lateral, read

    directory = _artefact(tmp_path / "a", monkeypatch)
    # the heading grid read toward the lower cell: the words move
    monkeypatch.setattr(lateral, "_snap", lambda value, step: np.floor(value / step) * step)
    checked = conformance.check(directory, git=CLEAN)
    assert set(checked.mismatches) == {"KXXX:a", "KXXX:c"}
    assert all("the words differ (2 s rows)" in problems for problems in checked.mismatches.values())
    with pytest.raises(ValueError, match="read otherwise"):
        conformance.write_passed(directory, checked, git=CLEAN)
    monkeypatch.undo()
    # a refusal that moves is found too: the ground-speed refusal no longer refuses
    original = read.admit

    def lenient(signals, geometry, one):
        return original(signals, geometry, spec(ground_speed_floor_mps=5.0))
    monkeypatch.setattr(read, "admit", lenient)
    checked = conformance.check(directory, git=CLEAN)
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
    checked = conformance.check(directory, git=CLEAN)
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
    assert conformance.check(directory, git=CLEAN).passed
    monkeypatch.setattr(conformance, "on_utc_grid", original)
    checked = conformance.check(directory, git=CLEAN)
    assert set(checked.mismatches) == {"KXXX:a", "KXXX:c"}
    assert all(any(p.startswith("interval_refusals") for p in found) and "the words differ (8 s rows)" in found
               for found in checked.mismatches.values())


def test_a_reference_of_another_schema_or_spec_is_refused(tmp_path, monkeypatch):
    directory = _artefact(tmp_path / "a", monkeypatch)
    path = directory / "conformance" / "reference.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**payload, "schema": "ts-instruction-conformance-reference-v0"}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"is not a {conformance.REFERENCE_SCHEMA} file"):
        conformance.check(directory, git=CLEAN)
    path.write_text(json.dumps({**payload, "spec_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="read with spec 000000000000"):
        conformance.check(directory, git=CLEAN)


def test_the_draw_takes_labelled_and_refused_flights_of_every_airport(monkeypatch):
    monkeypatch.setattr(conformance, "PER_AIRPORT", 2)
    monkeypatch.setattr(conformance, "REFUSED_PER_AIRPORT", 1)
    flights = [type("F", (), {"airport": airport})() for airport in ["KAAA"] * 6 + ["KBBB"] * 4]
    statuses = ["labelled", "refused"] * 5
    chosen = conformance.draw(flights, statuses)
    assert chosen == sorted(chosen) and chosen == conformance.draw(flights, statuses)
    picked = [(flights[i].airport, statuses[i]) for i in chosen]
    assert sorted(picked) == [("KAAA", "labelled"), ("KAAA", "labelled"), ("KAAA", "refused"),
                              ("KBBB", "labelled"), ("KBBB", "labelled"), ("KBBB", "refused")]


def test_the_code_name_covers_the_reading_code_only():
    covered = {label for label, _ in conformance.labeller_code_files()}
    assert {"spec.py", "words.py", "grammar.py", "envelope.py", "labeller/read.py", "labeller/go_around.py",
            "labeller/interval.py", "final_approach.crossing"} <= covered
    assert not covered & {"artefact.py", "readout.py", "figures.py", "measure.py", "conformance.py", "__init__.py"}
