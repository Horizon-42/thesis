"""The executor checked by what it flies (`experiments/executor_conformance.py`, executor design §12.3): a flight flown
again matches its reference, the bounds are where they are said to be, every discrete difference is named, a changed law
is found, and the reference and the passed record are bound to their spec, flights and every way of flying."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from geokit import METRES_PER_DEG_LAT
from ts_transformer.autopilot.frame import ALT, LAT
from ts_transformer.experiments import executor_conformance as conformance
from ts_transformer.tests.support import fly_legs, instruction_flight
from ts_transformer.tests.test_autopilot import DOWNWIND_BASE_FINAL, _fly_sentence, _params


def _flown(params=None):
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    flown, verdict, _ = _fly_sentence(signals, params=params)
    return conformance.flight_results(flown, [verdict])[0]


def _compare(reference, flown):
    difference = conformance.Difference()
    conformance.compare(reference, flown, difference, "KXXX:f1")
    return difference


def _with(result, **arrays):
    return replace(result, **arrays)


def test_a_flight_flown_again_by_the_same_code_matches_its_reference_through_the_files(tmp_path):
    first = _flown()
    conformance.save_results(tmp_path / "reference.npz", [first])
    verdicts = json.loads(json.dumps([first.verdict]))                    # as the reference file holds them
    stored = conformance.load_results(tmp_path / "reference.npz", verdicts)[0]
    difference = _compare(stored, _flown())
    assert difference.passed and difference.flights == 1
    assert (difference.horizontal_m, difference.vertical_m, difference.other) == (0.0, 0.0, 0.0)
    assert stored.done == first.done and len(stored.states) == first.done + 2
    with pytest.raises(FileExistsError):                                  # written once
        conformance.save_results(tmp_path / "reference.npz", [first])


def test_states_and_floats_pass_inside_the_bounds_and_fail_outside():
    reference = _flown()
    k = reference.done // 2

    def moved(column, by):
        states = reference.states.copy()
        states[k, column] += by
        return _with(reference, states=states)

    metre_in_lat = 1.0 / METRES_PER_DEG_LAT
    assert _compare(reference, moved(LAT, 0.5 * conformance.STATE_BOUND_M * metre_in_lat)).passed
    outside = _compare(reference, moved(LAT, 2.0 * conformance.STATE_BOUND_M * metre_in_lat))
    assert not outside.passed and "horizontally" in outside.mismatches["KXXX:f1"][0]
    assert outside.horizontal_m == pytest.approx(2.0 * conformance.STATE_BOUND_M, rel=1e-3)
    assert _compare(reference, moved(ALT, 0.5 * conformance.STATE_BOUND_M)).passed
    assert not _compare(reference, moved(ALT, 2.0 * conformance.STATE_BOUND_M)).passed
    commands = reference.commands.copy()
    commands[k, 0] += 0.5 * conformance.ROUNDOFF
    assert _compare(reference, _with(reference, commands=commands)).passed
    commands[k, 0] += 2.0 * conformance.ROUNDOFF
    far = _compare(reference, _with(reference, commands=commands))
    assert ["a float" in p for p in far.mismatches["KXXX:f1"]] == [True]


def test_every_discrete_difference_is_named():
    reference = _flown()
    k = reference.done // 2
    modes = {**reference.modes, "captured": reference.modes["captured"].copy()}
    modes["captured"][k] = ~modes["captured"][k]
    assert _compare(reference, _with(reference, modes=modes)).mismatches["KXXX:f1"] == [
        f"mode captured differs at cycle {k} (1 cycles)"]
    shorter = _with(reference, done=reference.done - 1, states=reference.states[:-1])
    assert any("done at cycle" in p for p in _compare(reference, shorter).mismatches["KXXX:f1"])
    verdict = json.loads(json.dumps(reference.verdict))
    verdict["end_row"] += 1
    assert any(p.startswith("verdict .end_row") for p in
               _compare(reference, _with(reference, verdict=verdict)).mismatches["KXXX:f1"])


def test_a_verdict_s_floats_carry_the_roundoff_and_its_other_entries_none():
    found: list[str] = []
    conformance._verdict_differences({"x": 1.0, "nan": float("nan"), "n": 3, "w": [True]},
                                     {"x": 1.0 + 0.5 * conformance.ROUNDOFF, "nan": float("nan"), "n": 3, "w": [True]},
                                     "", found)
    assert found == []
    conformance._verdict_differences({"x": 1.0, "n": 3, "w": [True], "only": 1},
                                     {"x": 1.0 + 2 * conformance.ROUNDOFF, "n": 4, "w": [False, True]}, "", found)
    assert found == [f"verdict .n: 3, 4", "verdict .only: only in one", f"verdict .w: 1 entries, 2",
                     "verdict .w[0]: True, False", f"verdict .x: 1.0, {1.0 + 2 * conformance.ROUNDOFF!r}"]


def test_a_nan_counts_where_both_have_one_and_fails_where_one_has_a_number():
    reference = _flown()
    states = reference.states.copy()
    states[-1, :] = np.nan
    both = _with(reference, states=states)
    assert _compare(both, both).passed
    assert any("NaN" in p for p in _compare(reference, both).mismatches["KXXX:f1"])


def test_a_changed_law_is_found():
    """A behaviour change the source hash would have caught is caught by what it flies."""
    reference = _flown()
    changed = _compare(reference, _flown(_params(heading_time_constant_s=6.0)))
    assert not changed.passed and changed.horizontal_m > 1.0


def _spec_dir(tmp_path, monkeypatch, results, *, keys=("KXXX:f1",)):
    record = {"sha256": "s" * 64, "vocabulary_spec_sha256": "v" * 64}
    monkeypatch.setattr(conformance.replay, "open_executor", lambda executor, instructions: (_params(), record, None))
    monkeypatch.setattr(conformance, "git_state", lambda: {"head": "h", "dirty": False})
    monkeypatch.setattr(conformance, "executor_source_sha256", lambda: "c" * 64)
    monkeypatch.setattr(conformance, "MODES", {"batch": lambda batch, params, words: results})
    batch = SimpleNamespace(signals=[SimpleNamespace(dataset_id=k) for k in keys], drawn={"split": "train"})
    executor = tmp_path / "executor"
    executor.mkdir()
    return executor, record, batch


def test_the_reference_is_written_once_from_a_clean_checkout_and_checked_against_its_own_spec_and_flights(
        tmp_path, monkeypatch):
    results = [_flown()]
    executor, record, batch = _spec_dir(tmp_path, monkeypatch, results)
    instructions = tmp_path / "instructions"
    directory = conformance.write_reference(executor, instructions, batch=batch)
    payload = json.loads((directory / "reference.json").read_text())
    assert payload["schema"] == conformance.REFERENCE_SCHEMA and payload["flights"] == ["KXXX:f1"]
    assert payload["flown_by"] == {"executor_source_sha256": "c" * 64, "mode": "batch"}
    differences = conformance.check(executor, instructions, batch=batch)
    assert differences["batch"].passed
    passed = conformance.write_passed(executor, differences)
    assert passed.name == f"passed-{'c' * 12}.json"
    record_ = json.loads(passed.read_text())
    assert record_["reference_sha256"] == conformance.reference_sha256(directory)
    assert record_["modes"]["batch"]["mismatched_flights"] == 0
    with pytest.raises(FileExistsError):
        conformance.write_reference(executor, instructions, batch=batch)
    other = SimpleNamespace(signals=[SimpleNamespace(dataset_id="KXXX:f2")], drawn={})
    with pytest.raises(ValueError, match="no longer gives its flights"):
        conformance.check(executor, instructions, batch=other)
    record["sha256"] = "t" * 64
    with pytest.raises(ValueError, match="the reference was flown for"):
        conformance.check(executor, instructions, batch=batch)


def test_a_dirty_checkout_writes_no_reference(tmp_path, monkeypatch):
    executor, _, batch = _spec_dir(tmp_path, monkeypatch, [_flown()])
    monkeypatch.setattr(conformance, "git_state", lambda: {"head": "h", "dirty": True})
    with pytest.raises(RuntimeError, match="clean checkout"):
        conformance.write_reference(executor, tmp_path / "instructions", batch=batch)
    assert not (executor / conformance.DIRECTORY).exists()


def test_a_passed_record_needs_every_way_of_flying_within_the_bounds(tmp_path, monkeypatch):
    executor, _, _ = _spec_dir(tmp_path, monkeypatch, [])
    (executor / conformance.DIRECTORY).mkdir()
    failing = conformance.Difference(flights=1, mismatches={"KXXX:f1": ["done at cycle 3, the reference at 4"]})
    with pytest.raises(ValueError, match="every way of flying"):
        conformance.write_passed(executor, {"batch": failing})
    with pytest.raises(ValueError, match="every way of flying"):
        conformance.write_passed(executor, {})


def test_the_verdict_json_is_the_verdict_s_fields():
    reference = _flown()
    assert set(reference.verdict) == {f.name for f in dataclasses.fields(conformance.Verdict)}
    assert json.loads(json.dumps(reference.verdict)) == reference.verdict
