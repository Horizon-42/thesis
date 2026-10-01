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


def _header(**changes):
    header = {"schema": READOUT_SCHEMA, "split": "val", "drawn": {"seed": 1337}, "windows_per_airport": 2, "samples": 2,
              "temperature": 1.0, "seed": 1337, "augment_seed": None, "augmenting": None,
              "executor": {"sha256": "e"}, "instructions": "i", "scenes": {"n": 1}, "history_s": 1200.0,
              "readings": {"ends": "visual"}, "aircraft_steps": 100000, "batches": 1, "prior": {"directory": "p"},
              "aircraft_file": "aircraft.jsonl"}
    return {**header, **changes}


def _row(window, key, sample, source, *, lost=False, landed=True, airport="KXXX", commanded=1, start_lost=False,
         augmented=None, role=None):
    outcome = LOST_SEPARATION if lost else ("landed" if landed else "timeout")
    return {"window": window, "dataset_id": key, "sample": sample, "source": source, "airport": airport,
            "commanded": commanded, "starts_in_a_loss": start_lost, "outcome": outcome,
            "ifr_outcome": outcome, "reward": float(outcome == "landed"), "runway": 0, "observed_runway": 0,
            "augmented": augmented, "role": role}


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
    keys = [(0, "KXXX:a", 2), (0, "KXXX:b", 2), (1, "KXXX:c", 1)]
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


def test_the_difference_is_per_aircraft_and_its_error_clustered_by_window(tmp_path):
    first, second = _readouts(tmp_path, {(0, "KXXX:a", 0, "scene"), (0, "KXXX:a", 1, "scene"),
                                         (1, "KXXX:c", 0, "scene")})
    result = pair.pair(first, second)
    scene = result["report"]["scene"]["pooled"]["all"]
    assert (scene["aircraft"], scene["sentences"]) == (3, 6)
    # per aircraft: a −1 (both samples), b 0, c −0.5 → mean −0.5; windows: 0 sums −1 (2 aircraft), 1 sums −0.5 (1)
    reward = scene["reward"]
    assert reward["difference"] == pytest.approx(-0.5)
    spread = (-1.0 - (-0.5) * 2) ** 2 + (-0.5 - (-0.5) * 1) ** 2
    assert reward["standard_error"] == pytest.approx(math.sqrt(2 / 1 * spread) / 3)
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
    for name, value in (("samples", 4), ("seed", 7), ("augment_seed", 7919), ("executor", {"sha256": "f"})):
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
    edited[0]["starts_in_a_loss"] = True
    (second / "aircraft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in edited))
    with pytest.raises(ValueError, match="start in a loss in one readout"):
        pair.pair(first, second)
    edited = [json.loads(line) for line in rows]
    labelled = next(i for i, r in enumerate(edited) if r["source"] == "labelled")
    edited[labelled]["outcome"] = LOST_SEPARATION
    (second / "aircraft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in edited))
    with pytest.raises(ValueError, match="rows no model reads differ"):
        pair.pair(first, second)


def test_a_sentence_starting_in_a_loss_is_not_counted(tmp_path):
    first, second = _readouts(tmp_path, set())
    for directory in (first, second):
        rows = [json.loads(line) for line in (directory / "aircraft.jsonl").read_text().splitlines()]
        for r in rows:
            r["starts_in_a_loss"] = r["dataset_id"] == "KXXX:c"
        (directory / "aircraft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    scene = pair.pair(first, second)["report"]["scene"]["pooled"]["all"]
    assert (scene["aircraft"], scene["sentences"]) == (2, 4)


def test_augmented_windows_are_split_by_kind_and_part(tmp_path):
    kinds = {0: {"kind": "C"}, 1: {"kind": "A"}}
    rows = []
    for window, key, role in ((0, "KXXX:a", "shifted"), (1, "KXXX:b", "inserted"), (1, "KXXX:c", None)):
        for sample in (0, 1):
            rows.append(_row(window, key, sample, "scene", augmented=kinds[window], role=role))
    first = _write(tmp_path / "first", rows, augment_seed=7919, augmenting={"kinds": {"C": 1, "A": 1}})
    second = _write(tmp_path / "second", rows, augment_seed=7919, augmenting={"kinds": {"C": 1, "A": 1}})
    report = pair.pair(first, second)["report"]["scene"]
    assert set(report["kinds"]) == {"C", "A"}
    assert {k: v["aircraft"] for k, v in report["kinds"].items()} == {"C": 1, "A": 2}
    assert {k: v["aircraft"] for k, v in report["roles"].items()} == {"shifted": 1, "inserted": 1, "as drawn": 1}


def test_the_pair_is_written_into_a_new_directory(tmp_path, capsys):
    first, second = _readouts(tmp_path, {(0, "KXXX:a", 0, "scene")})
    out = tmp_path / "pair"
    assert pair.main(["--first", str(first), "--second", str(second), "--out", str(out)]) == 0
    written = json.loads((out / "traffic_window_pair.json").read_text())
    assert written["schema"] == pair.SCHEMA and written["second"]["prior"] == {"directory": "q"}
    assert "reward 1.0000 → 0.8333" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pair.main(["--first", str(first), "--second", str(second), "--out", str(out)])
