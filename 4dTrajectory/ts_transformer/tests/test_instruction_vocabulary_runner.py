"""The labeller runner (`experiments/instruction_vocabulary.py`): the summary's statistics
(absorbed by reason, clamps, unclamped residuals, an empty split refused), the rendered table,
the hand-check page, and the command line's refusals — on synthetic flights, writing into tmp."""

from __future__ import annotations


import numpy as np
import pytest

from ts_transformer.config import CONTROL_RECIPE_SIMPLE_V3, PREDICTION_CONTROL, TSConfig, recipe_settings
from ts_transformer.data.dataset import build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.experiments import instruction_vocabulary as runner
from ts_transformer.manoeuvre import instructions as ins
from ts_transformer.tests.support import AIRPORT, RUNWAY

#: The cohort the runner would build the runway word's classes from (D62).
RUNWAYS = ins.RunwayVocabulary.from_idents([RUNWAY, "23R"])


def _config() -> TSConfig:
    settings = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False)
    settings.update(dict(prediction_output=PREDICTION_CONTROL, control_horizon_s=20.0, n_segments=2, control_imitation_loss_weight=0.0,
                         final_time_loss_weight=0.0, state_endpoint_loss_weight=0.0, seq_len=8, d_model=16, n_heads=4, d_ff=32,
                         e_layers=1, dropout=0.0, device="cpu", epochs=1, patience=1))
    return TSConfig(**settings)


@pytest.fixture(scope="module")
def readings():
    vocabulary = ins.Vocabulary()
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=4, seed=5), _config(), airport=AIRPORT)
    return vocabulary, series, [ins.read_instructions(item, vocabulary, RUNWAYS) for item in series]


def test_the_summary_counts_what_the_hand_check_and_the_bin_decision_read(readings):
    vocabulary, _series, items = readings
    summary = runner.summarise(items, vocabulary)
    assert summary["flights"] == 4 and summary["events"] == sum(len(r.event_times_s) for r in items)
    # D52: every event IS a change, so what the summary reports is how many and how far apart
    assert summary["gap_s_p50"] <= summary["gap_s_p95"] and summary["duration_clamped"] >= 0
    assert set(summary["absorbed"]) == set(ins.MANDATORY_KINDS)
    assert all(set(v) == {ins.ABSORBED_SAME_WORD, ins.ABSORBED_SMALL_CHANGE, ins.ABSORBED_SHORT_TAIL} for v in summary["absorbed"].values())
    assert sum(sum(v.values()) for v in summary["absorbed"].values()) == sum(len(r.absorbed) for r in items)
    assert set(summary["clamped"]) == {"altitude", "speed"}
    # the runway is a kind of its own (D62): one word per flight, one class used by this cohort
    assert set(summary["words_used"]) == set(ins.INSTRUCTION_KINDS) and summary["words_used"]["runway"] == 1
    assert summary["instructions_per_flight_p50"]["runway"] == 1.0
    for kind, half_bin in (("heading_deg", vocabulary.heading_bin_deg / 2), ("altitude_m", vocabulary.altitude_bin_m / 2),
                           ("speed_mps", vocabulary.speed_bin_mps / 2)):
        assert 0.0 <= summary["target_to_bin_centre_p50"][kind] <= summary["target_to_bin_centre_p95"][kind] <= half_bin + 1e-9
    # a clamped word never enters the residuals: an 11 000 ft start reads as the top word, 1000 ft off its centre
    clamped = ins.Instruction("altitude", vocabulary.altitude_words - 1, 11_000 * ins.FT, 0.0, 0.0, clamped=True)
    with_clamp = ins.Reading("d", "f", (clamped,), np.array([0.0]),
                             np.array([[0, vocabulary.altitude_words - 1, 0, 0, 0, ins.TERMINAL_LANDED]]),
                             RUNWAY, True, 0.0)
    summary = runner.summarise([*items, with_clamp], vocabulary)
    assert summary["clamped"]["altitude"] == 1 and summary["target_to_bin_centre_p95"]["altitude_m"] <= vocabulary.altitude_bin_m / 2 + 1e-9
    with pytest.raises(ValueError, match="no flights"):
        runner.summarise([], vocabulary)
    table = runner.render({"train": runner.summarise(items, vocabulary)}, vocabulary, RUNWAYS)
    assert "train: 4 flights" in table and "absorbed manoeuvres (same word / small change / short tail)" in table
    assert "event sequence (D52)" in table and "gap p50" in table   # D52 / D71: events and their gaps, not grid positions
    # the runway count printed is the COHORT's, not the spec's (the spec has no runway word)
    assert f"runways {', '.join(RUNWAYS.idents)}" in table and f"runway 1/{len(RUNWAYS)}" in table


def test_the_hand_check_page_is_written(readings, tmp_path):
    vocabulary, series, items = readings
    path = tmp_path / "page.png"
    runner.hand_check_figure(series[0], items[0], vocabulary, path)
    assert path.is_file() and path.stat().st_size > 10_000


def test_the_command_line_refuses_what_cannot_be_set(tmp_path):
    base = ["--executor", str(tmp_path / "none.pt"), "--cohort", str(tmp_path / "none.json"), "--out", str(tmp_path / "out")]
    for override in ("reading_rule=plateau-v9", "established_cross_track_m=1000", "not_a_field=1", "heading_bin_deg="):
        with pytest.raises(SystemExit):
            runner.main([*base, "--set", override])
    with pytest.raises(SystemExit):
        runner.main([*base, "--hand-check", "3"])
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit):
        runner.main(base)                                                            # an existing --out is refused first
    assert not (tmp_path / "out" / "summary.json").exists()
