"""The Training overlays — another model's output drawn over an exported set's own flights: the executor's replay
(`experiments/executor_training_export.py`), the prior's predictions (`experiments/prior_training_export.py`) and the
overlay manifest they share (`experiments/instruction_training_export.py`).

Each word's executor verdict is rebuilt from the judge on synthetic flights and must give back the judge's own count;
the prior's per-step predictions come from a small untrained network; the manifest's refusals are exercised on files in
``tmp_path``. The runners' ``main()`` run end to end on the formal artefacts when they are on this machine, every write
into ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import replace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot.replay import word_results
from ts_transformer.experiments import executor_training_export as executor_export
from ts_transformer.experiments import instruction_training_export as export
from ts_transformer.experiments import prior_training_export as prior_export
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import APPROACH, APPROACH_CLEARED, COLUMNS, HEADING, RUNWAY, UNCHANGED, Words
from ts_transformer.prior import data as prior_data
from ts_transformer.prior.data import Flight, Split, column_classes, flight_steps
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.repo_layout import REPO_ROOT
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec as spec
from ts_transformer.tests.test_autopilot import DOWNWIND_BASE_FINAL, _fly_sentence, _orbit, _params

TRAINING_OVERLAYS_TS = REPO_ROOT / "aeroviz-4d" / "src" / "data" / "trainingOverlays.ts"


def _verdicts(signals, **fly):
    flown, verdict, reading = _fly_sentence(signals, **fly)
    words, judged, not_judged = executor_export.word_verdicts(flown, 0, verdict, reading, signals, instruction_airport(),
                                                              spec(), Words(spec()))
    return verdict, reading, words, judged, not_judged


# ---- the executor: a verdict per word
def test_every_word_gets_one_verdict_and_the_judged_ones_give_back_the_judges_count():
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    verdict, reading, words, judged, not_judged = _verdicts(signals)
    assert (judged, not_judged) == word_results(verdict)
    assert [(w["row"], w["column"], w["value"]) for w in words] == [(i.row, i.column, i.value) for i in reading.instructions]
    assert {w["status"] for w in words} <= set(executor_export.STATUSES)
    # a flight flown inside every envelope: every checked word inside, the rest say why they have no check
    for word in words:
        if word["status"] == "no check":
            assert word["reason"] and not word["checks"]
        else:
            assert word["status"] == "inside" and word["checks"] and all(check["ok"] for check in word["checks"])
    assert [w["status"] for w in words if w["column"] == RUNWAY] == ["no check"]
    # on the time clock with no delay every word is told at its own step
    assert all(w["flownRow"] == w["row"] for w in words)
    assert sum(w["status"] == "inside" for w in words) <= len(judged)


def test_a_turn_the_capture_takes_over_is_not_judged_and_counted_as_the_judge_counts_it():
    legs = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (6, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    verdict, _, words, judged, not_judged = _verdicts(instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0)))
    assert (judged, not_judged) == word_results(verdict) and not_judged == 1
    (base,) = [w for w in words if w["column"] == HEADING and w["row"] > 0]
    assert base["status"] == "not judged" and "capture took this turn over" in base["reason"]


def test_a_turn_flown_too_slowly_is_outside_on_every_part_with_the_check_that_failed():
    verdict, reading, words, judged, _ = _verdicts(_orbit(), params=_params(bank_cap_deg=5.0))
    assert judged == word_results(verdict)[0]
    parts = [w for w, i in zip(words, reading.instructions) if i.kind == "turn-split"]
    assert len(parts) == 3 and all(w["status"] == "outside" for w in parts)
    failed = {check["name"] for check in parts[0]["checks"] if not check["ok"]}
    assert failed == {"turn rate and bank"}
    assert parts[0]["checks"] == parts[1]["checks"] == parts[2]["checks"]


def test_the_heading_word_left_to_intercept_the_final_on_its_own_is_the_one_failed():
    """Cleared at once 3 km north of the final on the course it flies: bent by the heading tolerance it cannot reach the
    line before the threshold, so the executor intercepts at 30° on its own — the judge fails the heading word in force."""
    legs = [(100, 0.0, 100.0, 0.0), (15, 4.0, 100.0, 0.0), (45, 0.0, 100.0, 0.0), (15, -4.0, 90.0, 0.0),
            (100, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    signals = instruction_flight(*fly_legs(legs, 90.0, 1600.0, -400.0, 0.0))
    _, _, reading = _fly_sentence(signals)
    grid = reading.words.copy()
    grid[0, APPROACH] = APPROACH_CLEARED
    flown, verdict, reading = _fly_sentence(signals, grid)
    assert verdict.words["intercepting_off_word_cycles"] > 0
    words, judged, not_judged = executor_export.word_verdicts(flown, 0, verdict, reading, signals, instruction_airport(),
                                                              spec(), Words(spec()))
    assert (judged, not_judged) == word_results(verdict) and judged[0] == ("heading", False)
    (failed,) = [w for w in words if any(check["name"].startswith("held until the capture") for check in w["checks"])]
    assert (failed["row"], failed["column"], failed["status"]) == (0, HEADING, "outside")


def test_a_flown_track_the_gate_refuses_leaves_every_checked_word_not_judged_and_says_why():
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    flown, verdict, reading = _fly_sentence(signals)
    refused = replace(verdict, words=None, refused="too short")
    words, judged, not_judged = executor_export.word_verdicts(flown, 0, refused, reading, signals, instruction_airport(),
                                                              spec(), Words(spec()))
    assert judged is None and not_judged == 0
    for word, instruction in zip(words, reading.instructions):
        expected = "not judged" if executor_export.checkable(instruction, Words(spec())) else "no check"
        assert word["status"] == expected and word["reason"]
    assert any("too short" in w["reason"] for w in words)


# ---- the prior: predictions per step
def _flights(words):
    grid = np.full((6, 6), UNCHANGED, dtype=np.int16)
    grid[0] = [0, 0, words.heading_index(270.0), words.altitude_index(1200.0), 0, words.speed_index(100.0)]
    grid[3, HEADING] = words.heading_index(180.0)
    grid[5, 5] = words.speed_unspecified
    signals = instruction_flight(*fly_legs([(10, 0.0, 90.0, 0.0)], 270.0, 900.0, -5000.0, 300.0))
    return [Flight(f"F{n}", 0, *flight_steps(signals, grid, instruction_airport())) for n in range(2)]


def test_the_prior_ranks_words_given_one_is_said_and_keeps_the_flights_likelihood():
    words = Words(spec())
    classes = column_classes(words, 2)
    candidates = torch.zeros(1, 2, len(prior_data.CANDIDATE_FEATURES))
    candidates[0, 0, -1] = 1.0                                           # one real candidate, one empty slot
    torch.manual_seed(0)
    model = Prior(PriorConfig(classes=classes, airports=("KXXX",), candidate_slots=2, d_model=32, layers=2, heads=4,
                              feedforward=64, dropout=0.0), candidates).eval()
    flights = _flights(words)
    out = prior_export.predictions(model, Split(flights, ("KXXX",), candidates.numpy(), classes))
    assert [item["rows"] for item in out] == [6, 6]
    item = out[0]
    assert len(item["columns"]) == len(COLUMNS)
    # the runway head ranks only the airport's one candidate, never the masked slot
    assert item["columns"][RUNWAY]["k"] == 1 and set(item["columns"][RUNWAY]["words"]) == {0}
    for column, values in enumerate(item["columns"]):
        k = values["k"]
        assert len(values["words"]) == len(values["wordsP"]) == 6 * k
        assert len(values["changeP"]) == len(values["truthP"]) == 6
        assert all(0 <= value < classes[column] - 1 for value in values["words"])
        ranked = np.asarray(values["wordsP"]).reshape(6, k)
        assert (np.diff(ranked, axis=1) <= 1e-9).all()                 # most likely first
    # the flight's likelihood is its truth's, per step; the columns add up to it
    truth = np.array([item["columns"][c]["truthP"] for c in range(6)])
    assert -np.log(truth).sum() / 6 == pytest.approx(item["nllPerStep"], rel=1e-2)
    assert sum(item["columnNllPerStep"]) == pytest.approx(item["nllPerStep"], abs=1e-5)


# ---- the manifest and the set an overlay is drawn over
def _base_files(training, *, kind=export.KIND_READBACK, rule=READING_RULE, schema=export.SAMPLE_SCHEMA):
    one = spec()
    sample = {"schema": schema, "setId": "set_a", "airport": "KXXX", "writtenUtc": "2026-09-24T00:00:00+00:00",
              "vocabulary": {"specSha256": one.sha256}, "cohort": {"split": export.SPLIT},
              "flights": [{"datasetId": "KXXX:F_09_abc_20260101T000000Z", "flightKey": "F_09_abc_20260101T000000Z",
                           "rows": 3, "words": {"events": [{"row": 0, "column": 0, "value": 0},
                                                           {"row": 2, "column": 2, "value": 18}]}}]}
    index = {"schema": export.INDEX_SCHEMA, "airport": "KXXX",
             "sets": [{"id": "set_a", "kind": kind, "readingRule": rule, "vocabularySha256": one.sha256,
                       "file": "set_a/sample.json"}]}
    (training / "set_a").mkdir(parents=True)
    (training / "set_a" / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
    (training / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return sample


def test_an_overlay_is_drawn_only_over_a_set_this_reader_reads(tmp_path):
    training = tmp_path / "KXXX" / "training"
    _base_files(training)
    base = export.open_base_set(training, "KXXX", "set_a", spec())
    raw = (training / "set_a" / "sample.json").read_bytes()
    assert base.block["sampleSha256"] == hashlib.sha256(raw).hexdigest()
    assert base.block["sampleWrittenUtc"] == "2026-09-24T00:00:00+00:00"
    with pytest.raises(SystemExit, match="lists no set other"):
        export.open_base_set(training, "KXXX", "other", spec())
    for change, message in ((dict(kind="prior-generated"), "prior-generated set"), (dict(rule="instruction-v1"), "instruction-v1"),
                            (dict(schema="aeroviz-training-sample-v4"), "re-export the set first")):
        shutil.rmtree(training)
        _base_files(training, **change)
        with pytest.raises(SystemExit, match=message):
            export.open_base_set(training, "KXXX", "set_a", spec())


def test_the_set_s_flights_must_carry_the_artefact_s_own_sentences(tmp_path):
    training = tmp_path / "KXXX" / "training"
    _base_files(training)
    base = export.open_base_set(training, "KXXX", "set_a", spec())
    signals = [instruction_flight(*fly_legs([(3, 0.0, 90.0, 0.0)], 270.0, 900.0, -5000.0, 300.0),
                                  dataset_id="KXXX:F_09_abc_20260101T000000Z")]
    grid = np.full((3, 6), UNCHANGED, dtype=np.int64)
    grid[0, 0], grid[2, 2] = 0, 18
    sentences = {"signal_index": np.array([0]), "offsets": np.array([0, 3]), "words": grid,
                 **{name: np.array([0]) for name in ("runway_index", "capture_row", "join_row", "unspecified_row")}}
    assert [k for _, k in export.base_flights(base, signals, sentences)] == [0]
    grid[2, 2] = 19
    with pytest.raises(SystemExit, match="is not the one set set_a shows"):
        export.base_flights(base, signals, sentences)
    with pytest.raises(SystemExit, match="has no labelled sentence"):
        export.base_flights(base, [replace(signals[0], dataset_id="KXXX:another")], sentences)


def test_an_overlay_is_added_beside_its_set_and_never_overwritten(tmp_path):
    training = tmp_path / "KXXX" / "training"
    _base_files(training)
    base = export.open_base_set(training, "KXXX", "set_a", spec())
    assert export.read_overlays(training, "KXXX", "ov_1") == []
    entry = export.overlay_entry("ov_1", export.KIND_EXECUTOR, base, "a title", "executor.json", 1, {"runner": "test"})
    with pytest.raises(ValueError):                                     # a payload that cannot be written stops the build
        export.serialise_overlay({"values": [float("nan")]})
    out = export.write_overlay(training, "KXXX", "ov_1", entry, export.serialise_overlay({"schema": "x", "values": [1, 2, 3]}),
                               [])
    assert out == training / "ov_1" / "executor.json"
    assert "\n" not in out.read_text(encoding="utf-8")                    # compact: its arrays are long
    manifest = json.loads((training / export.OVERLAYS_FILE).read_text(encoding="utf-8"))
    assert manifest["schema"] == export.OVERLAYS_SCHEMA and [item["id"] for item in manifest["overlays"]] == ["ov_1"]
    assert manifest["overlays"][0]["baseSampleSha256"] == base.sha256 and manifest["overlays"][0]["base"] == "set_a"
    with pytest.raises(SystemExit, match="already lists overlay ov_1"):
        export.read_overlays(training, "KXXX", "ov_1")
    with pytest.raises(FileExistsError):
        export.write_overlay(training, "KXXX", "ov_2", entry, "{}", export.read_overlays(training, "KXXX", "ov_2"))
    # another export wrote the manifest since this run read it: adding to the old list would drop its entry
    second = export.overlay_entry("ov_2", export.KIND_PRIOR, base, "t", "prior.json", 1, {})
    with pytest.raises(SystemExit, match="changed since this run read it"):
        export.write_overlay(training, "KXXX", "ov_2", second, "{}", [])
    assert not (training / "ov_2").exists()
    with pytest.raises(SystemExit, match="is KXXX's overlays, not KYYY's"):
        export.read_overlays(training, "KYYY", "ov_2")
    with pytest.raises(ValueError, match="unknown overlay kind"):
        export.overlay_entry("ov_3", "free-generation", base, "t", "f.json", 1, {})


# ---- the frontend's mirrors
def _ts_constant(name: str) -> str:
    source = TRAINING_OVERLAYS_TS.read_text(encoding="utf-8")
    match = re.search(rf"export const {name}\b[^=]*=\s*(?P<value>[^;]+);", source)
    assert match is not None, f"{name} not found in {TRAINING_OVERLAYS_TS}"
    return match.group("value")


def test_the_frontend_reader_mirrors_the_exporters_names():
    assert json.loads(_ts_constant("TRAINING_OVERLAYS_SCHEMA")) == export.OVERLAYS_SCHEMA
    assert tuple(re.findall(r'"([^"]+)"', _ts_constant("TRAINING_OVERLAY_KINDS"))) == export.OVERLAY_KINDS
    assert json.loads(_ts_constant("TRAINING_EXECUTOR_SCHEMA")) == executor_export.SCHEMA
    assert tuple(re.findall(r'"([^"]+)"', _ts_constant("TRAINING_EXECUTOR_STATUSES"))) == executor_export.STATUSES
    assert json.loads(_ts_constant("TRAINING_PRIOR_SCHEMA")) == prior_export.SCHEMA


# ---- the runners, end to end on the formal artefacts
POOLED = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED"
INSTRUCTIONS = POOLED / "instruction_language" / "v2_20260924"
EXECUTOR = POOLED / "executor" / "v2_20260924"
PRIOR = POOLED / "prior" / "v1_20260924"
LIVE_SET = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports" / "KSMF" / "training"
FORMAL = pytest.mark.skipif(
    not all(path.exists() for path in (INSTRUCTIONS, EXECUTOR / "replay-val" / "replay.json", PRIOR / "checkpoint.pt",
                                       LIVE_SET / "instruction_v2" / "sample.json")),
    reason="the formal instruction, executor and prior artefacts and the published KSMF set are not on this machine")


def _three_flight_set(tmp_path):
    """The published KSMF `instruction_v2` set cut to its first three flights, in ``tmp_path`` — read, never written."""
    training = tmp_path / "airports" / "KSMF" / "training"
    (training / "instruction_v2").mkdir(parents=True)
    sample = json.loads((LIVE_SET / "instruction_v2" / "sample.json").read_text(encoding="utf-8"))
    sample["flights"] = sample["flights"][:3]
    (training / "instruction_v2" / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
    index = json.loads((LIVE_SET / "index.json").read_text(encoding="utf-8"))
    index["sets"] = [item for item in index["sets"] if item["id"] == "instruction_v2"]
    (training / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return tmp_path / "airports", training


@FORMAL
def test_the_executor_export_reflies_the_set_and_matches_the_formal_replay(tmp_path, monkeypatch):
    # `conftest.py` puts this tree's `geokit/src` on the path, and `autopilot.spec.executor_source_files` then hashes
    # geokit with the executor — which the formal spec's hash (measured in a worktree, geokit outside it) does not
    # cover, so the spec is refused here though no executor file differs (docs/code-health-followups.md, 2026-09-24).
    # The export's own check stands in for it: every re-flown flight must reproduce its formal replay row.
    from ts_transformer.autopilot import replay
    monkeypatch.setattr(replay, "require_current_executor", lambda record: None)
    root, training = _three_flight_set(tmp_path)
    args = ["--executor", str(EXECUTOR), "--replay", str(EXECUTOR / "replay-val"), "--instructions", str(INSTRUCTIONS),
            "--airports-root", str(root), "--set", "instruction_v2", "--airport", "KSMF", "--overlay-id", "ex_test"]
    assert executor_export.main(args) == 0
    payload = json.loads((training / "ex_test" / "executor.json").read_text(encoding="utf-8"))
    assert payload["schema"] == executor_export.SCHEMA and len(payload["flights"]) == 3
    assert payload["gate"]["own dynamics"]["all"]["all"]["flights"] == 7773
    sample = json.loads((training / "instruction_v2" / "sample.json").read_text(encoding="utf-8"))
    for flight, item in zip(payload["flights"], sample["flights"]):
        if flight["flown"]:
            assert [(w["row"], w["column"], w["value"]) for w in flight["words"]] == \
                [(e["row"], e["column"], e["value"]) for e in item["words"]["events"]]
            assert flight["track"]["tS"][0] == 0 and math.isfinite(flight["track"]["altitudeHaeM"][-1])
    with pytest.raises(SystemExit):   # never overwritten
        executor_export.main(args)


@FORMAL
def test_the_prior_export_predicts_the_set_under_the_checkpoint_it_names(tmp_path):
    root, training = _three_flight_set(tmp_path)
    args = ["--prior", str(PRIOR), "--instructions", str(INSTRUCTIONS), "--airports-root", str(root),
            "--set", "instruction_v2", "--airport", "KSMF", "--overlay-id", "pr_test"]
    assert prior_export.main(args) == 0
    payload = json.loads((training / "pr_test" / "prior.json").read_text(encoding="utf-8"))
    assert payload["schema"] == prior_export.SCHEMA and payload["columns"] == list(COLUMNS)
    assert payload["readout"]["model"]["nllPerStep"] == pytest.approx(0.17782562442333233)
    sample = json.loads((training / "instruction_v2" / "sample.json").read_text(encoding="utf-8"))
    assert [f["rows"] for f in payload["flights"]] == [f["rows"] for f in sample["flights"]]
    manifest = json.loads((training / export.OVERLAYS_FILE).read_text(encoding="utf-8"))
    assert [item["id"] for item in manifest["overlays"]] == ["pr_test"]
