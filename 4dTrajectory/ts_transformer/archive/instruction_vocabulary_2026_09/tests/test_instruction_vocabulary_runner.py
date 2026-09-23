"""The labeller runner (`experiments/instruction_vocabulary.py`): the summary's statistics
(absorbed by reason, clamps, unclamped residuals, an empty split refused), the rendered table,
the hand-check page, and the command line's refusals — on synthetic flights, writing into tmp."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ts_transformer.config import CONTROL_RECIPE_SIMPLE_V3, PREDICTION_CONTROL, TSConfig, recipe_settings
from ts_transformer.data.dataset import build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.experiments import instruction_vocabulary as runner
from ts_transformer.manoeuvre import instructions as ins
from ts_transformer.tests.support import AIRPORT, OTHER_RUNWAY_WORD, RUNWAY, RUNWAY_WORD

#: The cohort the runner would build the runway word's classes from (D62).
RUNWAYS = ins.RunwayVocabulary.from_idents([RUNWAY_WORD, OTHER_RUNWAY_WORD])


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
    # `summarise` builds this dict by comprehension over the reasons, so its KEY SET is a
    # property of the comprehension and asserting on it proves nothing (it used to). What can
    # fail is which reasons actually FIRE: the vertical kind's segments tile the track, so it
    # never has a plateau to call a small change or a short tail, and a fold onto the word beside
    # it is COUNTED (`vertical_pieces_merged`) rather than recorded as a manoeuvre — so the only
    # vertical rows are the fit's own short segments.
    fired = {kind: {reason for reason, count in counts.items() if count} for kind, counts in summary["absorbed"].items()}
    assert fired["vertical"] <= {ins.ABSORBED_SHORT_SEGMENT}, fired["vertical"]
    assert not any(ins.ABSORBED_SHORT_SEGMENT in fired[k] for k in ("heading", "speed"))
    assert summary["vertical_pieces_merged"] >= 0 and summary["vertical_fit_rms_m_p50"] >= 0.0
    assert sum(sum(v.values()) for v in summary["absorbed"].values()) == sum(len(r.absorbed) for r in items)
    # the clamp counter is the whole claim that a value outside the ends is COUNTED, never
    # silent — so it is pinned, not merely present (the old assertion on it was dropped when the
    # altitude word went)
    assert set(summary["outside"]) == {"vertical", "speed"}
    assert summary["outside"] == {kind: sum(1 for r in items for i in r.instructions if i.kind == kind and i.clamped)
                                  for kind in ("vertical", "speed")}
    # the runway is a kind of its own (D62): one word per flight, one class used by this cohort
    assert set(summary["words_used"]) == set(ins.INSTRUCTION_KINDS) and summary["words_used"]["runway"] == 1
    assert summary["instructions_per_flight_p50"]["runway"] == 1.0
    # Only the heading is a uniform bin, so only it has a "half a bin" to be inside. The vertical
    # and the speed are fitted centres: what their readout states is the distance to the nearest
    # centre, which is what the summary carries, and the largest gap between neighbours bounds it.
    widest = {"vertical_deg": max(b - a for a, b in zip(vocabulary.vertical_modes_deg, vocabulary.vertical_modes_deg[1:])),
              "speed_mps": max(b - a for a, b in zip(vocabulary.speed_centres_mps, vocabulary.speed_centres_mps[1:]))}
    for kind, half_bin in (("heading_deg", vocabulary.heading_bin_deg / 2),
                           ("vertical_deg", widest["vertical_deg"] / 2),
                           ("speed_mps", widest["speed_mps"] / 2)):
        assert 0.0 <= summary["target_to_bin_centre_p50"][kind] <= summary["target_to_bin_centre_p95"][kind] <= half_bin + 1e-9
    # a clamped word never enters the residuals: an 11 000 ft start reads as the top word, 1000 ft off its centre
    clamped = ins.Instruction("vertical", vocabulary.vertical_words - 1, 9.0, 0.0, 0.0, clamped=True)
    with_clamp = ins.Reading("d", "f", (clamped,), np.array([0.0]),
                             np.array([[0, vocabulary.vertical_words - 1, 0, 0, 0, ins.TERMINAL_LANDED]]),
                             RUNWAY, True, 0.0)
    summary = runner.summarise([*items, with_clamp], vocabulary)
    assert summary["target_to_bin_centre_p95"]["vertical_deg"] <= widest["vertical_deg"] / 2 + 1e-9
    assert summary["outside"]["vertical"] == 1                # and the clamp itself is counted
    with pytest.raises(ValueError, match="no flights"):
        runner.summarise([], vocabulary)
    table = runner.render({"train": runner.summarise(items, vocabulary)}, vocabulary, RUNWAYS)
    assert "train: 4 flights" in table and "absorbed manoeuvres (same word / small change / short tail / short segment)" in table
    assert "vertical fit against the track" in table          # the segment ceiling's cost is stated
    assert "event sequence (D52)" in table and "gap p50" in table   # D52 / D71: events and their gaps, not grid positions
    # the runway count printed is the COHORT's, not the spec's (the spec has no runway word)
    assert f"runways {', '.join(RUNWAYS.idents)}" in table and f"runway 1/{len(RUNWAYS)}" in table


def test_the_hand_check_page_is_written(readings, tmp_path):
    vocabulary, series, items = readings
    path = tmp_path / "page.png"
    runner.hand_check_figure(series[0], items[0], vocabulary, path)
    assert path.is_file() and path.stat().st_size > 10_000


def test_the_manifest_door_writes_a_whole_artefact(tmp_path, monkeypatch):
    """The WRITE path of the `--airports` door, end to end.

    Reading the cohort and writing the artefact are different halves, and only the second one
    knows about the executor. On 2026-09-21 the write half still asked for a variable that only
    the executor half binds, so a five-airport run read 26,382 flights and then raised — seven
    minutes of work thrown away by a line no test touched. This builds a tiny cohort through the
    manifests door and writes the artefact, so that half is covered whichever door is used.
    """
    from types import SimpleNamespace
    from ts_transformer.experiments import support

    series = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=4, seed=11), _config(), airport=AIRPORT)[0]
    keys = [item.dataset_id for item in series]
    cohort = SimpleNamespace(train_flight_ids=keys[:3], val_flight_ids=keys[3:], name="tiny",
                             artifact_sha256="x" * 64, selection={}, splits={})
    monkeypatch.setattr(runner, "load_development_cohort", lambda _path: cohort)
    monkeypatch.setattr(runner, "development_cohort_audit", lambda _path, _cohort: {"name": "tiny"})
    monkeypatch.setattr(support, "arrival_manifest_path", lambda code: tmp_path / f"{code}.json")
    (tmp_path / "KRDU.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(support, "load_flight_dicts", lambda *a, **k: [])
    monkeypatch.setattr(support, "build_series", lambda *a, **k: (series, SimpleNamespace(format=lambda: "4 built")))

    out = tmp_path / "artefact"
    assert runner.main(["--airports", "KRDU", "--cohort", str(tmp_path / "cohort.json"),
                        "--out", str(out), "--hand-check", "0"]) == 0
    vocabulary, runways, payload = ins.load_vocabulary(out / ins.VOCABULARY_FILE)
    assert payload["source"]["read_from"] == "arrival manifests"          # the door names itself
    assert payload["source"]["airports"] == ["KRDU"] and payload["source"]["config"]["seq_len"]
    assert runways.idents == (f"{AIRPORT}:{RUNWAY}",)
    for split, count in (("train", 3), ("val", 1)):
        sentences = json.loads((out / f"sentences_{split}.json").read_text(encoding="utf-8"))
        assert len(sentences["flights"]) == count
        assert sentences["vocabulary_sha256"] == vocabulary.sha256
        # every sentence must read back as the Reading it was written from — this is what the
        # replay gate flies
        assert all(ins.Reading.from_dict(f).flight_id == f["flight_id"] for f in sentences["flights"])
    assert (out / "summary.json").exists() and (out / "summary.txt").exists()


def test_exactly_one_cohort_door(tmp_path):
    """A pooled cohort has no executor and a per-airport one has no reason to re-read manifests:
    the runner takes ONE door and says which in its provenance. Neither and both are refused at
    the command line, before anything is read."""
    tail = ["--cohort", str(tmp_path / "none.json"), "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        runner.main(tail)                                                            # neither door
    with pytest.raises(SystemExit):
        runner.main(["--executor", str(tmp_path / "none.pt"), "--airports", "KRDU", *tail])


def test_the_manifest_door_names_the_flights_the_series_builder_skipped(tmp_path, monkeypatch, capsys):
    """The door's exclusion policy, which both runners now share: a cohort flight the data layer
    cannot build is NAMED in the provenance and dropped from its split, and more than a percent of
    them is a different cohort, not an exclusion. The manifests themselves are stubbed — what is
    under test is the policy, not `build_series`."""
    from types import SimpleNamespace
    from ts_transformer.experiments import support

    keys = [f"KRDU:f{i}" for i in range(200)]
    cohort = SimpleNamespace(train_flight_ids=keys[:160], val_flight_ids=keys[160:])
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(support, "arrival_manifest_path", lambda code: manifest)
    monkeypatch.setattr(support, "load_flight_dicts", lambda *a, **k: [])

    def build(_flights, _config, **_kwargs):
        built = [SimpleNamespace(dataset_id=key) for key in skipped_from(keys)]
        return built, SimpleNamespace(format=lambda: f"{len(built)} built")

    skip = 1
    skipped_from = lambda ids: [k for k in ids if k != f"KRDU:f{skip}"]
    monkeypatch.setattr(support, "build_series", build)
    series, splits, provenance = support.cohort_from_manifests(["krdu"], cohort, TSConfig())
    assert len(series) == 199 and provenance["excluded_short_track"] == ["KRDU:f1"]
    assert provenance["airports"] == ["KRDU"] and len(provenance["manifest_sha256"][0]) == 64
    assert len(splits["train"]) == 159 and len(splits["val"]) == 40
    assert "0.50%" in capsys.readouterr().out

    skipped_from = lambda ids: ids[:190]                                             # 5 %: a different cohort
    with pytest.raises(SystemExit, match="different cohort"):
        support.cohort_from_manifests(["KRDU"], cohort, TSConfig())


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
