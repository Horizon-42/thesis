"""The executor checked by what it flies (`experiments/executor_conformance.py`, executor design §12.3): a flight flown
again matches its reference, the bounds are where they are said to be, every discrete difference is named, a changed law
is found, and the reference and the passed record are bound to their spec, flights and every way of flying."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from geokit import METRES_PER_DEG_LAT
from ts_transformer.autopilot.frame import ALT, LAT, LON, PSI
from ts_transformer.autopilot import conformance
from ts_transformer.instructions.words import Words
from ts_transformer.tests.support import fly_legs, instruction_flight, instruction_spec
from ts_transformer.tests.test_autopilot import DOWNWIND_BASE_FINAL, _fly_sentence, _params


def _flown(params=None):
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    flown, verdict, _ = _fly_sentence(signals, params=params)
    return conformance.flight_results(flown, [verdict])[0]


def _compare(reference, flown):
    difference = conformance.Difference(expected=1)
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
    modes = {**reference.modes, "level_captured": reference.modes["level_captured"].copy()}
    modes["level_captured"][k] = ~modes["level_captured"][k]
    assert _compare(reference, _with(reference, modes=modes)).mismatches["KXXX:f1"] == [
        f"mode level_captured differs at cycle {k} (1 cycles)"]
    runway = reference.runway.copy()
    runway[k:] = 1
    assert _compare(reference, _with(reference, runway=runway)).mismatches["KXXX:f1"] == [
        f"the runway in force differs at cycle {k} ({reference.done + 1 - k} cycles)"]
    shorter = _with(reference, done=reference.done - 1, states=reference.states[:-1],
                    **{f: getattr(reference, f)[:-1] for f in ("commands", "wanted", "sentence_s", "runway")},
                    limits={k: v[:-1] for k, v in reference.limits.items()},
                    modes={k: v[:-1] for k, v in reference.modes.items()})
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


def test_a_nan_counts_where_both_have_one_and_fails_where_one_has_another_value():
    reference = _flown()
    states = reference.states.copy()
    states[-1, :] = np.nan
    both = _with(reference, states=states)
    assert _compare(both, both).passed
    assert any("NaN" in p for p in _compare(reference, both).mismatches["KXXX:f1"])
    infinite = states.copy()
    infinite[-1, ALT] = np.inf
    assert not _compare(both, _with(reference, states=infinite)).passed          # NaN against inf
    nan_lon = reference.states.copy()
    nan_lon[-1, LON] = np.nan
    moved = nan_lon.copy()
    moved[-1, LAT] += 1.0                                                      # a latitude apart beside a NaN longitude
    assert not _compare(_with(reference, states=nan_lon), _with(reference, states=moved)).passed


def test_a_heading_is_compared_round_the_circle():
    reference = _flown()
    states = reference.states.copy()
    states[1, PSI] = math.pi - 1e-12
    across = states.copy()
    across[1, PSI] = -math.pi + 1e-12
    assert _compare(_with(reference, states=states), _with(reference, states=across)).passed


def test_a_way_of_flying_that_drops_flights_or_cuts_them_short_does_not_pass():
    reference = _flown()
    assert not conformance.Difference(expected=2, flights=1).passed
    with pytest.raises(ValueError, match="not cut at its done cycle"):
        _compare(reference, _with(reference, states=reference.states[:-1]))


def test_a_changed_law_is_found():
    """A behaviour change the source hash would have caught is caught by what it flies."""
    reference = _flown()
    changed = _compare(reference, _flown(_params(bank_rate_deg_s=3.0)))
    assert not changed.passed and changed.horizontal_m > 1.0


def _spec_dir(tmp_path, monkeypatch, results, *, keys=("KXXX:f1",)):
    record = {"sha256": "s" * 64, "vocabulary_spec_sha256": "v" * 64, "source": {"executor_source_sha256": "c" * 64}}
    monkeypatch.setattr(conformance.replay, "open_spec",
                        lambda executor, instructions: (_params(), record, Words(instruction_spec())))
    monkeypatch.setattr(conformance, "git_state", lambda: {"head": "h", "dirty": False})
    monkeypatch.setattr(conformance, "executor_source_sha256", lambda: "c" * 64)
    monkeypatch.setattr(conformance, "MODES", {"batch": lambda batch, params, words: results})
    monkeypatch.setattr(conformance, "input_digests", lambda batch, params, words: [f"in-{s.dataset_id}"
                                                                                     for s in batch.signals])
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
    checked = conformance.check(executor, instructions, batch=batch)
    assert checked.differences["batch"].passed and checked.executor_source_sha256 == "c" * 64
    passed = conformance.write_passed(executor, checked)
    assert passed.name == f"passed-{'c' * 12}.json"
    record_ = json.loads(passed.read_text())
    assert record_["reference_sha256"] == conformance.reference_sha256(directory)
    assert record_["modes"]["batch"]["mismatched_flights"] == 0
    with pytest.raises(FileExistsError):
        conformance.write_reference(executor, instructions, batch=batch)
    other = SimpleNamespace(signals=[SimpleNamespace(dataset_id="KXXX:f2")], drawn={})
    with pytest.raises(ValueError, match="no longer gives its flights"):
        conformance.check(executor, instructions, batch=other)
    monkeypatch.setattr(conformance, "input_digests", lambda batch, params, words: ["moved"])
    with pytest.raises(ValueError, match="the inputs of 1 reference flights moved"):
        conformance.check(executor, instructions, batch=batch)
    monkeypatch.setattr(conformance, "STATE_BOUND_M", 1e-3)
    with pytest.raises(ValueError, match="made under the bounds"):
        conformance.check(executor, instructions, batch=batch)
    record["sha256"] = "t" * 64
    with pytest.raises(ValueError, match="the reference was flown for"):
        conformance.check(executor, instructions, batch=batch)
    monkeypatch.setattr(conformance, "MODES", {"batch": lambda batch, params, words: []})
    record["sha256"] = "s" * 64
    monkeypatch.setattr(conformance, "STATE_BOUND_M", 1e-6)
    monkeypatch.setattr(conformance, "input_digests", lambda batch, params, words: ["in-KXXX:f1"])
    with pytest.raises(ValueError, match="flew 0 of the reference's 1 flights"):
        conformance.check(executor, instructions, batch=batch)


def test_a_reference_is_written_only_from_a_clean_checkout_by_the_code_that_measured_the_spec(tmp_path, monkeypatch):
    executor, record, batch = _spec_dir(tmp_path, monkeypatch, [_flown()])
    monkeypatch.setattr(conformance, "git_state", lambda: {"head": "h", "dirty": True})
    with pytest.raises(RuntimeError, match="clean checkout"):
        conformance.write_reference(executor, tmp_path / "instructions", batch=batch)
    monkeypatch.setattr(conformance, "git_state", lambda: {"head": "h", "dirty": False})
    record["source"]["executor_source_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="measured by other executor code"):
        conformance.write_reference(executor, tmp_path / "instructions", batch=batch)
    assert not (executor / conformance.DIRECTORY).exists()


def test_a_passed_record_needs_every_way_of_flying_within_the_bounds(tmp_path, monkeypatch):
    executor, _, _ = _spec_dir(tmp_path, monkeypatch, [])
    (executor / conformance.DIRECTORY).mkdir()
    failing = conformance.Difference(expected=1, flights=1,
                                     mismatches={"KXXX:f1": ["done at cycle 3, the reference at 4"]})
    with pytest.raises(ValueError, match="every way of flying"):
        conformance.write_passed(executor, conformance.Checked({"batch": failing}, "c" * 64, {"head": "h", "dirty": False}))
    with pytest.raises(ValueError, match="every way of flying"):
        conformance.write_passed(executor, conformance.Checked({}, "c" * 64, {"head": "h", "dirty": False}))
    monkeypatch.setattr(conformance, "git_state", lambda: {"head": "h", "dirty": True})
    with pytest.raises(RuntimeError, match="clean checkout"):
        conformance.write_passed(executor, conformance.Checked({"batch": conformance.Difference(expected=1, flights=1)},
                                                               "c" * 64, {"head": "h", "dirty": False}))
    monkeypatch.setattr(conformance, "git_state", lambda: {"head": "h", "dirty": False})
    passing = {"batch": conformance.Difference(expected=1, flights=1)}
    # the code or the commit moved while the check flew: the record would name code that was not flown
    for code, git in (("d" * 64, {"head": "h", "dirty": False}), ("c" * 64, {"head": "g", "dirty": False})):
        with pytest.raises(RuntimeError, match="the code the check flew"):
            conformance.write_passed(executor, conformance.Checked(passing, code, git))


def test_the_verdict_json_is_the_verdict_s_fields():
    reference = _flown()
    assert set(reference.verdict) == {f.name for f in dataclasses.fields(conformance.Verdict)}
