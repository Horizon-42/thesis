"""The vocabulary's acceptance test: its own sentences, flown, must reach the runway.

The runner is `experiments/instruction_replay`. What is pinned here is that it flies what the
ARTEFACT says rather than re-reading the tracks, that it refuses an artefact whose sentences were
read under another spec, and that its table states the model the tracks were flown under.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ts_transformer.experiments import instruction_replay as runner
from ts_transformer.manoeuvre import instruction_kinematics as kinematics
from ts_transformer.manoeuvre import instructions as ins


def _reading(dataset_id: str = "KRDU:F1") -> ins.Reading:
    v = ins.Vocabulary()
    words = np.zeros((2, len(ins.INSTRUCTION_KINDS)), dtype=np.int64)
    words[:, ins.INSTRUCTION_KINDS.index("vertical")] = v.vertical_modes_deg.index(0.0)
    words[:, ins.INSTRUCTION_KINDS.index("speed")] = 4
    words[1, ins.INSTRUCTION_KINDS.index("heading")] = 6
    words[1, ins.INSTRUCTION_KINDS.index("terminal")] = ins.TERMINAL_LANDED
    return ins.Reading(
        dataset_id=dataset_id, flight_id=dataset_id.split(":", 1)[1],
        instructions=(ins.Instruction("heading", 0, 0.0, 0.0, 0.0),
                      ins.Instruction("runway", 0, ins.NO_TARGET, 0.0, 0.0)),
        event_times_s=np.array([0.0, 60.0]), words=words, runway="KRDU:05L",
        established_from_start=True, duration_s=120.0, duration_clamped=1,
        absorbed=(ins.Absorbed("heading", 10.0, 20.0, 0, 1.5, ins.ABSORBED_SAME_WORD),),
        vertical_fit_rms_m=12.5, vertical_pieces_merged=2,
    )


def test_a_reading_round_trips_through_its_own_dict():
    """The gate flies the FILE, so the file has to be able to become a `Reading` again — and
    every field has to survive, not just the ones the flying uses."""
    original = _reading()
    back = ins.Reading.from_dict(original.to_dict())
    assert back.dataset_id == original.dataset_id and back.flight_id == original.flight_id
    assert np.array_equal(back.event_times_s, original.event_times_s)
    assert np.array_equal(back.words, original.words)
    assert back.runway == original.runway and back.duration_s == original.duration_s
    assert back.duration_clamped == original.duration_clamped
    assert back.established_from_start == original.established_from_start
    assert back.instructions == original.instructions and back.absorbed == original.absorbed
    assert back.vertical_fit_rms_m == original.vertical_fit_rms_m
    assert back.vertical_pieces_merged == original.vertical_pieces_merged
    # and it is the same object a flight is flown from
    assert np.array_equal(back.words_at(np.array([0.0, 90.0])), original.words_at(np.array([0.0, 90.0])))


def test_the_table_states_the_landing_rate_by_stratum_and_the_model_it_flew_under():
    rows = [
        {"landed": True, "end_reason": kinematics.END_CROSSED, "gap_p95_m": 900.0, "gap_mean_m": 400.0,
         "events": 8, "vertical_fit_rms_m": 10.0, "stratum": "straight-in"},
        {"landed": False, "end_reason": kinematics.END_TIME_CAP, "gap_p95_m": 5000.0, "gap_mean_m": 2000.0,
         "events": 14, "vertical_fit_rms_m": 40.0, "stratum": "vectored"},
    ]
    table = runner.render(rows, "abc123def456", "train")
    assert "1 / 2 = 50.0%" in table
    assert "straight-in 1 / 1 = 100.0%" in table and "vectored    0 / 1 = 0.0%" in table
    # the assumptions travel with the number: a landing rate without the model that flew it is
    # not a measurement of the vocabulary
    for field in ("headingWordZeroTracksTheCentreline", "bankDeg", "accelMaxMps2", "stopRule"):
        assert field in table, field
    assert kinematics.METHOD in table


def test_it_refuses_sentences_read_under_another_spec(tmp_path):
    """The sentences and the spec beside them are one artefact. If they disagree, the gate would
    be flying one vocabulary's words under another's centres — every number would be wrong and
    nothing would say so."""
    v = ins.Vocabulary()
    ins.write_vocabulary(tmp_path, v, runway_vocabulary=ins.RunwayVocabulary.from_idents(["KRDU:05L"]),
                         cohort_identity={}, counts={}, source={})
    (tmp_path / "sentences_train.json").write_text(json.dumps({
        "schema": "x", "split": "train", "vocabulary_sha256": "0" * 64,
        "runway_idents": ["KRDU:05L"], "token_step_s": v.token_step_s, "flights": [],
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="the spec beside them"):
        runner.main(["--vocabulary", str(tmp_path / ins.VOCABULARY_FILE), "--airports", "KRDU",
                     "--cohort", str(tmp_path / "cohort.json"), "--out", str(tmp_path / "out")])


def test_exactly_one_cohort_door(tmp_path):
    tail = ["--vocabulary", str(tmp_path / "v.json"), "--cohort", str(tmp_path / "c.json"), "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        runner.main(tail)
    with pytest.raises(SystemExit):
        runner.main(["--executor", str(tmp_path / "none.pt"), "--airports", "KRDU", *tail])
