"""Two models' window readouts of the same windows compared aircraft by aircraft (`experiments/traffic_window_compare.py`):
the differences and their errors on rows whose answers are known; the three rules (multi-aircraft design §6.6 step
9.9.3) — configurations equal but for the model, the code version's evidence (the same clean code version, or a
conformance record either way), the same rows; the groups, and the file."""

from __future__ import annotations

import json
import math

import pytest

from ts_transformer.experiments import traffic_window_compare as compare
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION


#: flight identities ending in their landing stamps (`flight_scenarios.identity.flight_key`): window 0 on one operating
#: day, window 1 on the next
A, B, C = "KXXX:A1_09_aaa_20260601T120000Z", "KXXX:B1_09_bbb_20260601T121000Z", "KXXX:C1_09_ccc_20260602T120000Z"
CODE = {"commit": "c" * 40, "dirty": False, "python": "3.12.8", "torch": "2.7.1", "cuda": "12.8", "gpu": "RTX 4060",
        "device": "cuda", "constants": {"history_s": 1200.0, "readings": {"ends": "visual", "beside": "ifr"}}}


def _config(**changes):
    from ts_transformer.experiments.traffic_window_generation import PROGRAM, load_config

    return load_config({"program": PROGRAM, "prior": "4dTrajectory/outputs/POOLED/prior/p", "prior_checkpoint_sha256": "a",
                        "executor": "4dTrajectory/outputs/POOLED/executor/e", "executor_sha256": "e",
                        "instructions": "4dTrajectory/outputs/POOLED/instruction_language/i", "split": "val",
                        "windows_per_airport": 2, "samples": 2, **changes}).as_json()


def _row(window, key, sample, source, *, lost=False, landed=True, airport="KXXX", commanded=1, start_lost=False,
         augmented=None, role=None):
    outcome = LOST_SEPARATION if lost else ("landed" if landed else "timeout")
    return {"window": window, "dataset_id": key, "sample": sample, "source": source, "airport": airport,
            "commanded": commanded, "starts_in_a_loss": start_lost, "outcome": outcome,
            "ifr_outcome": outcome, "reward": float(outcome == "landed"), "runway": 0, "observed_runway": 0,
            "augmented": augmented, "role": role, "batch": 0}


def _write(directory, rows, code=None, **config):
    """A readout directory of the new format: its configuration (`_config` with ``config``), code version (`CODE`, or
    ``code``) and rows."""
    from ts_transformer.experiments import traffic_window_generation as runner

    directory.mkdir()
    (directory / runner.CONFIG_FILE).write_text(json.dumps(_config(**config)))
    (directory / runner.CODE_FILE).write_text(json.dumps(CODE if code is None else code))
    (directory / runner.AIRCRAFT_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
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
            for source in compare.MODEL_SOURCES:
                first.append(_row(window, key, sample, source, commanded=commanded))
                second.append(_row(window, key, sample, source, commanded=commanded,
                                   lost=(window, key, sample, source) in second_lost))
        first += _fixed(window, key)
        second += _fixed(window, key)
    return _write(tmp_path / "first", first), _write(tmp_path / "second", second,
                                                     prior="4dTrajectory/outputs/POOLED/prior/q",
                                                     prior_checkpoint_sha256="b")


def test_the_difference_is_second_less_first_and_its_error_clustered_by_airport_and_day(tmp_path):
    first, second = _readouts(tmp_path, {(0, A, 0, "scene"), (0, A, 1, "scene"), (1, C, 0, "scene")})
    result = compare.compare_readouts(first, second)
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


def test_readouts_whose_configurations_differ_in_more_than_the_model_are_refused(tmp_path):
    from ts_transformer.experiments import traffic_window_generation as runner

    first, second = _readouts(tmp_path, set())
    written = json.loads((second / runner.CONFIG_FILE).read_text())
    for name, value in (("samples", 4), ("seed", 7), ("augment_seed", 7919), ("executor_sha256", "f"),
                        ("model_sources", ["scene"]), ("probe_samples", 2), ("commanded", "one"),
                        ("temperature", 0.5), ("aircraft_steps", 50_000), ("split", "select")):
        (second / runner.CONFIG_FILE).write_text(json.dumps({**written, name: value}))
        with pytest.raises(ValueError, match=f"configurations differ in \\['{name}'\\], not only in the model"):
            compare.compare_readouts(first, second)
    (second / runner.CONFIG_FILE).write_text(json.dumps(written))
    # an old readout's single header is refused by name
    (first / runner.CONFIG_FILE).unlink()
    with pytest.raises(ValueError, match="holds no config.json"):
        compare.compare_readouts(first, second)


def test_the_rows_must_be_the_same_windows_read_the_same_way(tmp_path):
    from ts_transformer.experiments import traffic_window_generation as runner

    first, second = _readouts(tmp_path, set())
    rows = (second / runner.AIRCRAFT_FILE).read_text().splitlines()
    (second / runner.AIRCRAFT_FILE).write_text("\n".join(rows[1:]) + "\n")
    with pytest.raises(ValueError, match="other rows: 1 only in the first"):
        compare.compare_readouts(first, second)
    edited = [json.loads(line) for line in rows]
    edited[0]["observed_runway"] = 1
    (second / runner.AIRCRAFT_FILE).write_text("".join(json.dumps(r) + "\n" for r in edited))
    with pytest.raises(ValueError, match="differ in what no model decides"):
        compare.compare_readouts(first, second)
    edited = [json.loads(line) for line in rows]
    labelled = next(i for i, r in enumerate(edited) if r["source"] == "labelled")
    edited[labelled]["outcome"] = LOST_SEPARATION
    (second / runner.AIRCRAFT_FILE).write_text("".join(json.dumps(r) + "\n" for r in edited))
    with pytest.raises(ValueError, match="rows no model reads differ"):
        compare.compare_readouts(first, second)


def _record(directory, code, **changes):
    """A conformance record of readout ``directory`` as it is now (`traffic_window_conformance.record_payload`),
    passed under ``code``, written where the check writes it."""
    from ts_transformer.experiments.traffic_window_conformance import record_path, record_payload

    path = record_path(directory, code["commit"])
    path.parent.mkdir(exist_ok=True)
    checked = {"batches": 40, "checked_batches": list(range(24)), "rows_compared": 1000, "rows": 2000}
    path.write_text(json.dumps({**record_payload(directory, code, checked, 1.0), **changes}))
    return path


def test_the_code_versions_are_the_same_clean_one_or_a_conformance_record_shows_one_reads_the_same_under_the_other(
        tmp_path):
    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.repo_layout import repo_relative

    first, second = _readouts(tmp_path, set())
    assert compare.compare_readouts(first, second)["code_evidence"] == {"kind": "same code version",
                                                                         "commit": CODE["commit"]}
    # the same code version, dirty: no evidence
    dirty = {**CODE, "dirty": True}
    for directory in (first, second):
        (directory / runner.CODE_FILE).write_text(json.dumps(dirty))
    with pytest.raises(ValueError, match=r"differ in \[\] \(dirty: \['first', 'second'\]\)"):
        compare.compare_readouts(first, second)
    # another commit and another torch for the second: refused, naming what is missing ...
    later = {**CODE, "commit": "d" * 40, "torch": "2.8.0"}
    (first / runner.CODE_FILE).write_text(json.dumps(CODE))
    (second / runner.CODE_FILE).write_text(json.dumps(later))
    with pytest.raises(ValueError, match=r"differ in \['commit', 'torch'\] and no conformance record.*"
                                         r"at the second's code \(dddddddddddd\)"):
        compare.compare_readouts(first, second)
    # ... a record under another code version (another GPU), or naming another readout, is no evidence ...
    for code, changes in (({**later, "gpu": "another"}, {}),
                          (later, {"readout": "4dTrajectory/outputs/POOLED/traffic/another"})):
        wrong = _record(first, code, **changes)
        with pytest.raises(ValueError, match="no conformance record"):
            compare.compare_readouts(first, second)
        wrong.unlink()
    # ... a record of the first passed under the second's code version is
    record = _record(first, later)
    got = compare.compare_readouts(first, second)["code_evidence"]
    assert got == {"kind": "conformance record", "record": repo_relative(record), "readout": repo_relative(first),
                   "batches": 40, "checked_batches": 24, "rows_compared": 1000}
    # the other way round: the second checked under the first's code version
    record.unlink()
    reverse = _record(second, CODE)
    assert compare.compare_readouts(first, second)["code_evidence"]["record"] == repo_relative(reverse)
    # a code.json with other keys is refused
    (first / runner.CODE_FILE).write_text(json.dumps({**CODE, "host": "x"}))
    with pytest.raises(ValueError, match="has keys"):
        compare.compare_readouts(first, second)


def test_a_sentence_starting_in_a_loss_in_either_readout_is_not_counted_and_is_reported(tmp_path):
    """The model flies the aircraft ahead of a later one, so a later one can start in a loss under one model only."""
    first, second = _readouts(tmp_path, set())
    for directory, starting in ((first, {(C, 0)}), (second, {(C, 0), (B, 1)})):
        rows = [json.loads(line) for line in (directory / "aircraft.jsonl").read_text().splitlines()]
        for r in rows:
            r["starts_in_a_loss"] = (r["dataset_id"], r["sample"]) in starting
        (directory / "aircraft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    scene = compare.compare_readouts(first, second)["report"]["scene"]["pooled"]["all"]
    assert (scene["aircraft"], scene["sentences"]) == (3, 4)
    assert scene["starting_in_a_loss"] == {"first": 1, "second": 2, "both": 1}


def test_augmented_windows_are_split_by_kind_and_part(tmp_path):
    kinds = {0: {"kind": "C"}, 1: {"kind": "A"}}
    rows = []
    for window, key, role in ((0, A, "shifted"), (1, B, "inserted"), (1, C, None)):
        for sample in (0, 1):
            rows.append(_row(window, key, sample, "scene", augmented=kinds[window], role=role))
    first = _write(tmp_path / "first", rows, augment_seed=7919)
    second = _write(tmp_path / "second", rows, augment_seed=7919, prior="4dTrajectory/outputs/POOLED/prior/q")
    report = compare.compare_readouts(first, second)["report"]["scene"]
    assert set(report["kinds"]) == {"C", "A"}
    assert {k: v["aircraft"] for k, v in report["kinds"].items()} == {"C": 1, "A": 2}
    assert {k: v["aircraft"] for k, v in report["roles"].items()} == {"shifted": 1, "inserted": 1, "as drawn": 1}


def test_the_comparison_is_written_into_a_new_directory(tmp_path, capsys):
    first, second = _readouts(tmp_path, {(0, A, 0, "scene")})
    out = tmp_path / "compare"
    assert compare.main(["--first", str(first), "--second", str(second), "--out", str(out)]) == 0
    written = json.loads((out / compare.OUT_FILE).read_text())
    assert compare.OUT_FILE == "traffic_window_compare.json"
    assert written["schema"] == compare.SCHEMA == "ts-traffic-window-compare-v1"
    assert written["second"] == {"directory": written["second"]["directory"],
                                 "prior": "4dTrajectory/outputs/POOLED/prior/q", "prior_checkpoint_sha256": "b"}
    assert written["code_evidence"]["kind"] == "same code version"
    assert "reward 1.0000 → 0.8333" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        compare.main(["--first", str(first), "--second", str(second), "--out", str(out)])


def test_a_window_s_operating_day_is_its_earliest_landing_s():
    assert compare.operating_day_of(A) == "2026-06-01" and compare.operating_day_of("KXXX:Z_09_z_20260602T080000Z") == "2026-06-01"
    with pytest.raises(ValueError, match="landing stamp"):
        compare.operating_day_of("KXXX:no_stamp")


def test_a_window_s_day_is_its_own_aircraft_s_never_an_inserted_one():
    """Augmentation A inserts another day's flight (keyed `INSERTED` after its own key): the window's cluster is the
    day of its own earliest-landing aircraft, even when the inserted one landed earlier."""
    from ts_transformer.experiments.traffic_speaking import INSERTED

    inserted = "KXXX:Z1_09_zzz_20260531T080000Z" + INSERTED
    pairs = {(0, inserted, 0, "scene"): (_row(0, inserted, 0, "scene"), None),
             (0, B, 0, "scene"): (_row(0, B, 0, "scene"), None),
             (1, C, 0, "scene"): (_row(1, C, 0, "scene"), None)}
    assert compare.cluster_of(pairs) == {0: ("KXXX", compare.operating_day_of(B)), 1: ("KXXX", compare.operating_day_of(C))}
