"""The tokenizer readout (`manoeuvre/readout.py`, gate T): the pairing is flight by flight with
the twin of the SAME seed, the gain sign reads "positive = the code helps", the usage numbers
are what T(iii) judges, the K rule takes the smallest passing K near the best, and the runner's
arm discovery separates trained from pending."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ts_transformer.config import TSConfig, recipe_settings, CONTROL_RECIPE_SIMPLE_V3
from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED
from ts_transformer.backbone.adapters import build_model
from ts_transformer.data.dataset import Normalizer, build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.experiments.manoeuvre_readout import discover_arms
from ts_transformer.inference.export import write_batch
from ts_transformer.manoeuvre import readout as ro
from ts_transformer.tests.support import AIRPORT, RUNWAY


def _rows(ade: dict[str, float], *, code: int | None = None, tortuosity: float = 1.0) -> dict[str, dict]:
    return {
        key: {"flight_id": key, "anchor": 59, "ade_m": value, "end_error_m": None if value is None else value * 2.0,
              "truth_shorter_than_horizon": value is None, "code": code if code is None else (code + i) % 4,
              "code_source": None if code is None else "truth", "route_tortuosity": tortuosity,
              "established_at_anchor": False, "remaining_path_m": 20_000.0}
        for i, (key, value) in enumerate(ade.items())
    }


def _config(**overrides) -> TSConfig:
    settings = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False)
    settings.update(dict(
        prediction_output="control", control_horizon_s=20.0, n_segments=2, control_imitation_loss_weight=0.0,
        final_time_loss_weight=0.0, state_endpoint_loss_weight=0.0, seq_len=8, d_model=16, n_heads=4, d_ff=32,
        e_layers=1, device="cpu", epochs=1, patience=1,
    ))
    settings.update(overrides)
    return TSConfig(**settings)


def test_paired_gain_is_flight_by_flight_and_positive_when_the_code_helps():
    arm = _rows({"a": 100.0, "b": 200.0, "c": 300.0, "d": None, "e": 50.0}, code=0)
    twin = _rows({"a": 150.0, "b": 190.0, "c": 400.0, "d": 100.0, "f": 10.0})
    gain = ro.paired_gain(arm, twin)
    assert gain[STRATUM_ALL]["n"] == 3                       # d has no reading, e/f are unpaired
    assert gain[STRATUM_ALL]["gain_p50_m"] == pytest.approx(50.0)   # deltas +50, -10, +100 → p50 +50
    assert gain[STRATUM_ALL]["arm_better_share"] == pytest.approx(2 / 3)
    assert gain[STRATUM_STRAIGHT_IN]["n"] == 3 and gain[STRATUM_VECTORED]["n"] == 0
    with pytest.raises(ValueError, match="share no flight"):
        ro.paired_gain(_rows({"x": 1.0}), _rows({"y": 1.0}))


def test_code_usage_reads_the_histogram_and_judges_t_iii():
    rows = _rows({f"f{i}": 1.0 for i in range(40)}, code=0)      # codes 0..3 evenly, K = 16
    usage = ro.code_usage(rows, 16)
    assert usage["used"] == 4 and usage["unused"] == 12 and usage["max_share"] == pytest.approx(0.25)
    assert usage["entropy_bits"] == pytest.approx(2.0)
    assert not usage["gate_t_iii"]                               # 12/16 unused
    even = {key: {**row, "code": i % 16} for i, (key, row) in enumerate(rows.items())}
    assert ro.code_usage(even, 16)["gate_t_iii"]
    with pytest.raises(ValueError, match="no-token"):
        ro.code_usage(_rows({"a": 1.0}), 16)


def test_gate_t_pairs_each_seed_with_its_own_twin_and_picks_the_smallest_k_near_the_best():
    def arm(key, kind, k, seed, ade_shift, code_count=None):
        rows = _rows({f"f{i}": 200.0 + ade_shift + (i % 7) for i in range(60)}, code=0 if kind != "no-token" else None)
        if kind != "no-token":
            rows = {key_: {**row, "code": i % code_count} for i, (key_, row) in enumerate(rows.items())}
        usage = None if kind == "no-token" else ro.code_usage(rows, code_count)
        return ro.ArmReading(key=key, kind=kind, k=k, seed=seed, rows=rows, usage=usage, summary=ro.stratum_summary(rows))

    arms = [
        arm("nt_s1", "no-token", None, 1, 0.0), arm("nt_s2", "no-token", None, 2, 0.0),
        arm("K16_s1", "learned", 16, 1, -40.0, 16), arm("K16_s2", "learned", 16, 2, -35.0, 16),     # gain 40 / 35: passes
        arm("K64_s1", "learned", 64, 1, -45.0, 64), arm("K64_s2", "learned", 64, 2, -44.0, 64),     # best, within 10 m of K16
        arm("K256_s1", "learned", 256, 1, -80.0, 256), arm("K256_s2", "learned", 256, 2, -10.0, 256),  # seed 2 fails (i)
        arm("cv_s1", "command-vocabulary", 63, 1, -31.0, 63),                                      # one seed only: pending
    ]
    verdict = ro.gate_t(arms)
    assert verdict["twins"] == {1: "nt_s1", 2: "nt_s2"}
    assert verdict["verdicts"]["K16"]["passes"] and verdict["verdicts"]["K64"]["passes"]
    assert verdict["verdicts"]["K256"]["complete"] and not verdict["verdicts"]["K256"]["gate_t_i"]
    assert not verdict["verdicts"]["command-vocabulary"]["complete"]
    assert verdict["selected_k"] == 16                            # 37.5 vs 44.5: within 10 m, the smaller wins
    assert verdict["arms"]["K16"][1]["twin"] == "nt_s1" and verdict["arms"]["K16"][2]["twin"] == "nt_s2"
    # a token arm whose seed has no twin is pending, never paired with the other seed's twin
    partial = ro.gate_t([arms[0], arms[2], arms[3]])
    assert partial["arms"]["K16"][2]["gain"] == "pending" and not partial["verdicts"]["K16"]["complete"]


def test_arm_reading_names_the_kind_and_k_from_the_config():
    rows = _rows({"a": 1.0, "b": 2.0}, code=0)
    learned = ro.arm_reading("S20_K32_s1337", _config(plan_conditioning="manoeuvre-code", manoeuvre_fsq_levels=(8, 4), seed=1337), rows)
    assert (learned.kind, learned.k, learned.seed) == ("learned", 32, 1337) and learned.usage["count"] == 32
    command = ro.arm_reading("S20_cv_s2024", _config(plan_conditioning="manoeuvre-code", manoeuvre_tokenizer="command-vocabulary", seed=2024), rows)
    assert (command.kind, command.k) == ("command-vocabulary", 63)
    twin = ro.arm_reading("S20_nt_s1337", _config(seed=1337), _rows({"a": 1.0}))
    assert (twin.kind, twin.k, twin.usage) == ("no-token", None, None)


def test_render_prints_every_arm_and_the_verdicts():
    rows = _rows({f"f{i}": 100.0 + i for i in range(8)}, code=0)
    arm = ro.arm_reading("S20_K16_s1337", _config(plan_conditioning="manoeuvre-code", manoeuvre_fsq_levels=(4, 4), seed=1337), rows)
    twin = ro.arm_reading("S20_nt_s1337", _config(seed=1337), _rows({f"f{i}": 150.0 + i for i in range(8)}))
    verdict = ro.gate_t([arm, twin])
    payload = {
        "campaign": "c", "segment_s": 20.0, "anchor": 59, "split": "val", "flights": 8, "truth_shorter_than_horizon": 0,
        "arms": {
            arm.key: {"seed": 1337, "summary": arm.summary, "usage": arm.usage, "gain": verdict["arms"]["K16"][1337]["gain"]},
            twin.key: {"seed": 1337, "summary": twin.summary, "usage": None, "gain": None},
        },
        "gate_t": verdict,
    }
    text = ro.render(payload)
    assert "S20_K16_s1337" in text and "S20_nt_s1337" in text and "+50" in text and "pending" in text
    json.dumps(payload)   # the artefact is JSON


def test_discover_arms_separates_trained_from_pending(tmp_path):
    for key, done in (("S60_K32_s1337", True), ("S60_nt_s1337", False), ("S30_K32_s1337", True), ("smoke_S60_cv_s1337", True)):
        (tmp_path / key).mkdir()
        if done:
            (tmp_path / key / "history.json").write_text("{}")
    (tmp_path / "S60_K32_s1337_pred_val").mkdir()
    trained, pending = discover_arms(tmp_path, 60.0)
    assert [p.name for p in trained] == ["S60_K32_s1337", "smoke_S60_cv_s1337"] and pending == ["S60_nt_s1337"]


def test_fixed_anchor_readings_read_every_flight_and_can_write_the_same_forecasts_as_records(tmp_path):
    import torch
    torch.manual_seed(0)
    config = _config(plan_conditioning="manoeuvre-code", manoeuvre_fsq_levels=(4, 4), seed=1337)
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=5, seed=3), config, airport=AIRPORT)
    model = build_model(config).eval()
    normalizer = Normalizer.fit(series)
    pairs = []
    rows = ro.fixed_anchor_readings(model, config, normalizer, series, torch.device("cpu"), batch_size=2, records=pairs)
    assert len(rows) == 5 and all(row["ade_m"] is not None and row["code"] is not None for row in rows.values())
    assert all(row["code_source"] == "truth" and row["anchor"] == 7 for row in rows.values())
    reading = ro.arm_reading("S20_K16_s1337", config, rows)
    assert reading.usage["count"] == 16 and reading.summary[ro.STRATUM_ALL]["n"] == 5
    # the records are the same forecasts, in cohort order, in the shape the publisher takes
    assert [index for index, _r, _m in pairs] == list(range(5))
    assert all(record.source["manoeuvreCode"] == rows[item.dataset_id]["code"] for (_i, record, _m), item in zip(pairs, series))
    write_batch([r for _, r, _ in pairs], output_dir=tmp_path / "rec", config_dict=config.to_dict(),
                flight_metrics=[m for _, _, m in pairs], split="val",
                extra_summary={ro.RECORDS_BLOCK: ro.records_block(reading, config, campaign="c", flights=5, records=5, limit=None)})
    summary = json.loads((tmp_path / "rec" / "summary.json").read_text())
    assert summary[ro.RECORDS_BLOCK]["protocol"] == "C" and summary[ro.RECORDS_BLOCK]["horizon_s"] == 20.0
    assert len(summary["results"]) == 5


def test_the_code_atlas_flies_every_code_from_each_state_in_its_own_frame(tmp_path):
    import torch
    from ts_transformer.manoeuvre import tokenizer as tok
    torch.manual_seed(0)
    codebook = tok.write_codebook(tmp_path / "cb", tok.tokenizer_for("learned", levels=(4, 4), segment_s=20.0, dt_s=2.0),
                                  segment_s=20.0, dt_s=2.0, data_identity={"eligible_set_sha256": {"KRDU": "b" * 64}}, source={})
    config = _config(plan_conditioning="manoeuvre-code", manoeuvre_fsq_levels=(4, 4), manoeuvre_codebook=str(codebook.path), seed=1337)
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=2, seed=3), config, airport=AIRPORT)
    model = build_model(config).eval()
    with torch.no_grad():
        model.control_head.control_projection.weight.normal_(std=0.05)
    atlas = ro.code_atlas(model, config, Normalizer.fit(series), codebook, series, torch.device("cpu"))
    assert len(atlas) == 2
    for entry in atlas:
        assert len(entry["codes"]) == 16 and [c["code"] for c in entry["codes"]] == list(range(16))
        assert 0 <= entry["truth_code"] < 16 and len(entry["truth_path"]) == 11
        ends = np.array([c["end"] for c in entry["codes"]])
        assert ends.shape == (16, 3) and np.isfinite(ends).all() and (ends[:, 0] > 0).all()   # every code flies ahead
        assert len(entry["codes"][0]["path"]) >= 2
        # the codes differ in where they end: the executor is conditioned on z
        assert np.linalg.norm(ends - ends.mean(axis=0), axis=1).max() > 0.0
