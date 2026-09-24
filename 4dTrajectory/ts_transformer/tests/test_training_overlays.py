"""The Training overlays — another model's output drawn over an exported set's own flights: the executor's replay
(`experiments/executor_training_export.py`), the prior's predictions (`experiments/prior_training_export.py`) and the
overlay manifest they share (`experiments/instruction_training_export.py`).

Each word's executor verdict is rebuilt from the judge on synthetic flights and must give back the judge's own count;
the prior's per-step predictions come from a small untrained network; the manifest's refusals are exercised on files in
``tmp_path``; the prior's runner runs end to end on a synthetic artefact and checkpoint, every write into ``tmp_path``.
The tests that ran both runners on the formal `v2_20260924` artefacts (the publication of 2026-09-24, `dba44622` on
`dev-publish-executor-prior`) are gone: this code no longer opens that generation (signals v2, the labeller after
9fb1b137), and a test that can never run again binds nothing.
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
from ts_transformer.instructions.words import (
    APPROACH, APPROACH_CLEARED, COLUMNS, HEADING, RUNWAY, UNCHANGED, Words, wrap180,
)
from ts_transformer.prior import data as prior_data
from ts_transformer.prior.data import Flight, Split, column_classes, flight_steps
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.repo_layout import REPO_ROOT
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec as spec
from ts_transformer.tests.test_autopilot import DOWNWIND_BASE_FINAL, _fly_sentence, _params

TRAINING_OVERLAYS_TS = REPO_ROOT / "aeroviz-4d" / "src" / "data" / "trainingOverlays.ts"

#: downwind west, then one continuous left turn onto the final east (course 090): a heading word said every step of
#: the turn, the last two too close to the clearance for the lead to leave them a row to judge
TURN_ONTO_FINAL = [(60, 0.0, 100.0, 0.0), (30, -6.0, 90.0, 0.0), (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]


def _word_verdicts(flown, verdict, reading, signals):
    """`word_verdicts` of the one flight flown, on the chart branch of its own smoothed track at row 0."""
    shift = executor_export.chart_shift_deg(flown, 0, instruction_airport(), float(signals.track_deg[0]))
    return executor_export.word_verdicts(flown, 0, verdict, reading, signals, instruction_airport(), spec(),
                                         Words(spec()), shift)


def _verdicts(signals, **fly):
    flown, verdict, reading = _fly_sentence(signals, **fly)
    words, judged, not_judged, track = _word_verdicts(flown, verdict, reading, signals)
    return verdict, reading, words, judged, not_judged, track


# ---- the executor: a verdict per word
def test_every_word_gets_one_verdict_and_the_judged_ones_give_back_the_judges_count():
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    verdict, reading, words, judged, not_judged, _ = _verdicts(signals)
    assert (judged, not_judged) == word_results(verdict)
    assert [(w["row"], w["column"], w["value"]) for w in words] == [(i.row, i.column, i.value) for i in reading.instructions]
    assert {w["status"] for w in words} <= set(executor_export.STATUSES)
    for word in words:
        if word["status"] == "no check":
            assert word["reason"] and not word["checks"]
        elif word["status"] in ("inside", "outside"):
            assert word["checks"] and (word["status"] == "inside") == all(check["ok"] for check in word["checks"])
    assert [w["status"] for w in words if w["column"] == RUNWAY] == ["no check"]
    # on the time clock with no delay every word is told at its own step
    assert all(w["flownRow"] == w["row"] for w in words)
    # only a heading word carries a band
    assert all((w["heading"] is not None) == (w["column"] == HEADING) for w in words)


def test_a_heading_word_is_judged_on_its_band_a_lead_after_it_was_told_row_by_row_as_the_judge_counts():
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    verdict, _, words, _, _, track = _verdicts(signals, params=_params(bank_cap_deg=5.0))   # too slow to keep up
    one = spec()
    lead, tolerance = one.rows_exact(one.heading_lead_s), one.heading_tolerance_deg
    heading = [w for w in words if w["column"] == HEADING]
    assert len(heading) == len(verdict.words["heading"])
    for word, result in zip(heading, verdict.words["heading"]):
        band = word["heading"]
        assert (band["firstRow"], band["stopRow"]) == (word["flownRow"] + lead, word["flownRow"] + lead + result["rows"])
        assert (len(band["inside"]), sum(band["inside"])) == (result["rows"], result["inside"])
        rows = slice(band["firstRow"], band["stopRow"])
        target = band["targetOnTrackDeg"]
        assert band["inside"] == [int(abs(float(wrap180(value - target))) <= tolerance) for value in track[rows]]
        assert band["bandDeg"] == pytest.approx([target - tolerance, target + tolerance])
        if result["rows"]:
            assert word["status"] == ("inside" if result["inside"] == result["rows"] else "outside")
            assert (word["checks"][0]["inside"], word["checks"][0]["rows"]) == (result["inside"], result["rows"])
    # the slow executor falls behind its words: rows outside, and the word outside with them
    failed = [w for w in heading if w["status"] == "outside"]
    assert failed and all(0 in w["heading"]["inside"] for w in failed)
    # the track the rows are read on is the flown one as the gate read it, on the chart's branch (row 0 unwrapped)
    assert len(track) >= max(w["heading"]["stopRow"] for w in heading) and abs(track[0] - 270.0) < 180.0


def test_a_heading_word_the_lead_carries_to_the_clearance_is_not_judged_and_counted_as_the_judge_counts_it():
    verdict, _, words, judged, not_judged, _ = _verdicts(instruction_flight(*fly_legs(TURN_ONTO_FINAL, 270.0, 1110.0,
                                                                                      -400.0, 0.0)))
    assert (judged, not_judged) == word_results(verdict) and not_judged == 2
    unjudged = [w for w in words if w["column"] == HEADING and w["status"] == "not judged"]
    assert len(unjudged) == 2
    for word in unjudged:
        assert "begin at or past the clearance" in word["reason"] and not word["checks"]
        assert word["heading"]["stopRow"] == word["heading"]["firstRow"] and word["heading"]["inside"] == []


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
    words, judged, not_judged, _ = _word_verdicts(flown, verdict, reading, signals)
    assert (judged, not_judged) == word_results(verdict) and judged[0] == ("heading", False)
    (failed,) = [w for w in words if any(check["name"].startswith("held until the capture") for check in w["checks"])]
    assert (failed["row"], failed["column"], failed["status"]) == (0, HEADING, "outside")


def test_a_flown_track_the_gate_refuses_leaves_every_checked_word_not_judged_and_says_why():
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    flown, verdict, reading = _fly_sentence(signals)
    refused = replace(verdict, words=None, refused="too short")
    words, judged, not_judged, track = _word_verdicts(flown, refused, reading, signals)
    assert judged is None and not_judged == 0 and track is None
    assert all(w["heading"] is None for w in words)
    for word, instruction in zip(words, reading.instructions):
        expected = "not judged" if executor_export.checkable(instruction, Words(spec())) else "no check"
        assert word["status"] == expected and word["reason"]
    assert any("too short" in w["reason"] for w in words)


def test_a_dynamics_failure_keeps_its_verdicts_and_draws_no_band():
    """The judge of a dynamics failure read the failed state, which the exported track leaves out: the words keep the
    judge's statuses and checks, with no band and no judged track to draw them on."""
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    flown, verdict, reading = _fly_sentence(signals)
    judged_words, *_ = _word_verdicts(flown, verdict, reading, signals)
    failed = replace(verdict, outcome="dynamics_failure")
    words, judged, not_judged, track = _word_verdicts(flown, failed, reading, signals)
    assert track is None and all(w["heading"] is None for w in words)
    assert (judged, not_judged) == word_results(failed)
    assert [(w["status"], w["checks"]) for w in words] == [(w["status"], w["checks"]) for w in judged_words]


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


# ---- the prior's runner end to end, on a synthetic artefact and an untrained checkpoint (every write in tmp_path)
def _prior_dir(directory, artefact):
    """A prior directory as `prior_train` writes one, holding a small untrained network of ``artefact``'s spec."""
    from ts_transformer.experiments.prior_train import PRIOR_CHECKPOINT_SCHEMA
    from ts_transformer.instructions.artefact import load_candidates, load_spec, spec_labeller_source

    one = load_spec(artefact)
    geometries = load_candidates(artefact)
    airports = tuple(sorted(geometries))
    slots = max(len(g.candidates) for g in geometries.values())
    config = PriorConfig(classes=column_classes(Words(one), slots), airports=airports, candidate_slots=slots, d_model=32,
                         layers=2, heads=4, feedforward=64, dropout=0.0)
    torch.manual_seed(0)
    model = Prior(config, torch.as_tensor(prior_data.candidate_table(geometries, airports, slots)))
    directory.mkdir()
    torch.save({"schema": PRIOR_CHECKPOINT_SCHEMA, "model_config": config.to_dict(), "train_config": {},
                "state": model.state_dict(), "spec_sha256": one.sha256}, directory / "checkpoint.pt")
    (directory / "config.json").write_text(json.dumps({
        "schema": PRIOR_CHECKPOINT_SCHEMA, "written_utc": "2026-09-24T00:00:00+00:00", "parameters": 1, "limit": None,
        "smoke": False, "git": {"head": "test", "dirty": False}, "train": {},
        "instructions": {"labeller_source_sha256": spec_labeller_source(artefact)}}))
    per_column = {name: {"nll_per_step": 0.1, "change_steps": 1, "mean_change_probability_where_changed": 0.5,
                         "top1_given_change": 0.5, "top5_given_change": 1.0, "false_change_share_where_kept": 0.0}
                  for name in COLUMNS}
    scores = {**{name: 0.2 for name in COLUMNS}, "all": 1.2}
    (directory / "readout.json").write_text(json.dumps({
        "split": "val", "best_epoch": 1, "baselines": {"repeat": scores, "previous word": scores},
        "model": {"steps": 10, "nll_per_step": 0.6, "perplexity_per_step": 1.8, "per_column": per_column}}))


def test_the_prior_export_writes_a_set_s_predictions_beside_it_and_refuses_a_second(tmp_path):
    from ts_transformer.tests.test_instruction_training_export import SET_ID, _artefact, _run, _straight, _vectored

    flights = [_vectored("KXXX:V1_09_abc123_20260101T000000Z"), _straight("KXXX:S1_09_abc124_20260101T000100Z")]
    _artefact(tmp_path / "artefact", flights)
    assert _run(tmp_path) == 0                                            # the set, as the Training export writes it
    _prior_dir(tmp_path / "prior", tmp_path / "artefact")
    args = ["--prior", str(tmp_path / "prior"), "--instructions", str(tmp_path / "artefact"),
            "--airports-root", str(tmp_path / "airports"), "--set", SET_ID, "--airport", "KXXX"]
    assert prior_export.main(args) == 0
    training = tmp_path / "airports" / "KXXX" / "training"
    payload = json.loads((training / "prior_prior" / "prior.json").read_text(encoding="utf-8"))
    sample = json.loads((training / SET_ID / "sample.json").read_text(encoding="utf-8"))
    assert payload["schema"] == prior_export.SCHEMA and payload["base"]["setId"] == SET_ID
    assert payload["base"]["sampleWrittenUtc"] == sample["writtenUtc"]
    assert [f["flightKey"] for f in payload["flights"]] == [f["flightKey"] for f in sample["flights"]]
    assert [f["rows"] for f in payload["flights"]] == [f["rows"] for f in sample["flights"]]
    assert all(math.isfinite(f["nllPerStep"]) for f in payload["flights"])
    assert payload["readout"]["baselines"]["previousWord"]["all"] == 1.2
    manifest = json.loads((training / export.OVERLAYS_FILE).read_text(encoding="utf-8"))
    assert [(o["id"], o["kind"], o["base"]) for o in manifest["overlays"]] == [("prior_prior", export.KIND_PRIOR, SET_ID)]
    with pytest.raises(SystemExit):   # never overwritten
        prior_export.main(args)
