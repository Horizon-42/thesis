"""Two window readouts of the same windows paired aircraft by aircraft (`experiments/traffic_window_pair.py`): the
differences and their errors on rows whose answers are known, every refusal of readouts that did not read the same
windows the same way, the groups, and the file."""

from __future__ import annotations

import json
import math

import pytest

from ts_transformer.experiments import traffic_window_pair as pair
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
from ts_transformer.experiments.traffic_window_generation import SCHEMA as READOUT_SCHEMA


#: flight identities ending in their landing stamps (`flight_scenarios.identity.flight_key`): window 0 on one operating
#: day, window 1 on the next
A, B, C = "KXXX:A1_09_aaa_20260601T120000Z", "KXXX:B1_09_bbb_20260601T121000Z", "KXXX:C1_09_ccc_20260602T120000Z"


def _header(**changes):
    header = {"schema": READOUT_SCHEMA, "git": {"head": "h", "dirty": False}, "split": "val", "drawn": {"seed": 1337}, "windows_per_airport": 2, "samples": 2,
              "temperature": 1.0, "seed": 1337, "augment_seed": None, "augmenting": None,
              "executor": {"sha256": "e"}, "instructions": "i", "scenes": {"n": 1}, "history_s": 1200.0,
              "readings": {"ends": "visual"}, "aircraft_steps": 100000, "batches": 1, "model_sources": ["scene", "alone"],
              "probes": {"samples": 0, "margin": 1.5},
              "prior": {"directory": "p"},
              "aircraft_file": "aircraft.jsonl"}
    return {**header, **changes}


def _row(window, key, sample, source, *, lost=False, landed=True, airport="KXXX", commanded=1, start_lost=False,
         augmented=None, role=None):
    outcome = LOST_SEPARATION if lost else ("landed" if landed else "timeout")
    return {"window": window, "dataset_id": key, "sample": sample, "source": source, "airport": airport,
            "commanded": commanded, "starts_in_a_loss": start_lost, "outcome": outcome,
            "ifr_outcome": outcome, "reward": float(outcome == "landed"), "runway": 0, "observed_runway": 0,
            "augmented": augmented, "role": role, "batch": 0}


def _write(directory, rows, **header):
    directory.mkdir()
    (directory / "window_generation.json").write_text(json.dumps(_header(**header)))
    (directory / "aircraft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return directory


def _fixed(window, key):
    return [_row(window, key, None, "labelled"), _row(window, key, None, "recorded")]


def _readouts(tmp_path, second_lost):
    """Windows 0 (two aircraft) and 1 (one), two samples each; the first readout loses nothing, the second loses the
    sentences in ``second_lost``."""
    keys = [(0, A, 2), (0, B, 2), (1, C, 1)]
    first, second = [], []
    for window, key, commanded in keys:
        for sample in (0, 1):
            for source in pair.MODEL_SOURCES:
                first.append(_row(window, key, sample, source, commanded=commanded))
                second.append(_row(window, key, sample, source, commanded=commanded,
                                   lost=(window, key, sample, source) in second_lost))
        first += _fixed(window, key)
        second += _fixed(window, key)
    return _write(tmp_path / "first", first), _write(tmp_path / "second", second, prior={"directory": "q"})


def test_the_difference_is_second_less_first_and_its_error_clustered_by_airport_and_day(tmp_path):
    first, second = _readouts(tmp_path, {(0, A, 0, "scene"), (0, A, 1, "scene"), (1, C, 0, "scene")})
    result = pair.pair(first, second)
    scene = result["report"]["scene"]["pooled"]["all"]
    assert (scene["aircraft"], scene["sentences"], scene["clusters"]) == (3, 6, 2)
    # per sentence: window 0 (day 1) −1, −1, 0, 0; window 1 (day 2) −1, 0 → mean −0.5 = 0.5 − 1.0
    reward = scene["reward"]
    assert reward["difference"] == pytest.approx(reward["second"] - reward["first"]) == pytest.approx(-0.5)
    spread = (-2.0 - (-0.5) * 4) ** 2 + (-1.0 - (-0.5) * 2) ** 2
    assert reward["standard_error"] == pytest.approx(math.sqrt(2 / 1 * spread) / 6)
    assert reward["first"] == 1.0 and reward["second"] == pytest.approx(0.5)
    assert reward["per_sentence"]["difference"] == pytest.approx(-0.5)
    assert reward["per_sentence"]["standard_error"] == pytest.approx(math.sqrt(3) / 6)
    lost = scene["lost_separation"]
    assert (lost["difference"], lost["avoided"], lost["added"]) == (pytest.approx(0.5), 0, 3)
    assert result["report"]["alone"]["pooled"]["all"]["reward"]["difference"] == 0.0
    sizes = result["report"]["scene"]["window_sizes"]
    assert set(sizes) == {"1", "2"} and sizes["1"]["aircraft"] == 1 and sizes["2"]["aircraft"] == 2


def test_readouts_that_did_not_read_the_same_windows_the_same_way_are_refused(tmp_path):
    first, second = _readouts(tmp_path, set())
    for name, value in (("samples", 4), ("seed", 7), ("augment_seed", 7919), ("executor", {"sha256": "f"}),
                        ("git", {"head": "other", "dirty": False}), ("model_sources", ["scene"]),
                        ("probes", {"samples": 2, "margin": 1.5})):
        header = json.loads((second / "window_generation.json").read_text())
        (second / "window_generation.json").write_text(json.dumps({**header, name: value}))
        with pytest.raises(ValueError, match=f"\\['{name}'\\] differ"):
            pair.pair(first, second)
        (second / "window_generation.json").write_text(json.dumps(header))
    rows = (second / "aircraft.jsonl").read_text().splitlines()
    (second / "aircraft.jsonl").write_text("\n".join(rows[1:]) + "\n")
    with pytest.raises(ValueError, match="other rows: 1 only in the first"):
        pair.pair(first, second)
    edited = [json.loads(line) for line in rows]
    edited[0]["observed_runway"] = 1
    (second / "aircraft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in edited))
    with pytest.raises(ValueError, match="differ in what no model decides"):
        pair.pair(first, second)
    edited = [json.loads(line) for line in rows]
    labelled = next(i for i, r in enumerate(edited) if r["source"] == "labelled")
    edited[labelled]["outcome"] = LOST_SEPARATION
    (second / "aircraft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in edited))
    with pytest.raises(ValueError, match="rows no model reads differ"):
        pair.pair(first, second)


def test_a_sentence_starting_in_a_loss_in_either_readout_is_not_counted_and_is_reported(tmp_path):
    """The model flies the aircraft ahead of a later one, so a later one can start in a loss under one model only."""
    first, second = _readouts(tmp_path, set())
    for directory, starting in ((first, {(C, 0)}), (second, {(C, 0), (B, 1)})):
        rows = [json.loads(line) for line in (directory / "aircraft.jsonl").read_text().splitlines()]
        for r in rows:
            r["starts_in_a_loss"] = (r["dataset_id"], r["sample"]) in starting
        (directory / "aircraft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    scene = pair.pair(first, second)["report"]["scene"]["pooled"]["all"]
    assert (scene["aircraft"], scene["sentences"]) == (3, 4)
    assert scene["starting_in_a_loss"] == {"first": 1, "second": 2, "both": 1}


def test_augmented_windows_are_split_by_kind_and_part(tmp_path):
    kinds = {0: {"kind": "C"}, 1: {"kind": "A"}}
    rows = []
    for window, key, role in ((0, A, "shifted"), (1, B, "inserted"), (1, C, None)):
        for sample in (0, 1):
            rows.append(_row(window, key, sample, "scene", augmented=kinds[window], role=role))
    first = _write(tmp_path / "first", rows, augment_seed=7919, augmenting={"kinds": {"C": 1, "A": 1}})
    second = _write(tmp_path / "second", rows, augment_seed=7919, augmenting={"kinds": {"C": 1, "A": 1}})
    report = pair.pair(first, second)["report"]["scene"]
    assert set(report["kinds"]) == {"C", "A"}
    assert {k: v["aircraft"] for k, v in report["kinds"].items()} == {"C": 1, "A": 2}
    assert {k: v["aircraft"] for k, v in report["roles"].items()} == {"shifted": 1, "inserted": 1, "as drawn": 1}


def test_the_pair_is_written_into_a_new_directory(tmp_path, capsys):
    first, second = _readouts(tmp_path, {(0, A, 0, "scene")})
    out = tmp_path / "pair"
    assert pair.main(["--first", str(first), "--second", str(second), "--out", str(out)]) == 0
    written = json.loads((out / "traffic_window_pair.json").read_text())
    assert written["schema"] == pair.SCHEMA and written["second"]["prior"] == {"directory": "q"}
    assert "reward 1.0000 → 0.8333" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pair.main(["--first", str(first), "--second", str(second), "--out", str(out)])


def test_a_window_s_operating_day_is_its_earliest_landing_s():
    assert pair.operating_day_of(A) == "2026-06-01" and pair.operating_day_of("KXXX:Z_09_z_20260602T080000Z") == "2026-06-01"
    with pytest.raises(ValueError, match="landing stamp"):
        pair.operating_day_of("KXXX:no_stamp")


def test_a_window_s_day_is_its_own_aircraft_s_never_an_inserted_one():
    """Augmentation A inserts another day's flight (keyed `INSERTED` after its own key): the window's cluster is the
    day of its own earliest-landing aircraft, even when the inserted one landed earlier."""
    from ts_transformer.experiments.traffic_speaking import INSERTED

    inserted = "KXXX:Z1_09_zzz_20260531T080000Z" + INSERTED
    pairs = {(0, inserted, 0, "scene"): (_row(0, inserted, 0, "scene"), None),
             (0, B, 0, "scene"): (_row(0, B, 0, "scene"), None),
             (1, C, 0, "scene"): (_row(1, C, 0, "scene"), None)}
    assert pair.cluster_of(pairs) == {0: ("KXXX", pair.operating_day_of(B)), 1: ("KXXX", pair.operating_day_of(C))}
