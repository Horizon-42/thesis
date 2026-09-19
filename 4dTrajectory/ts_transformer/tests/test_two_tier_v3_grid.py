"""Two-tier plan v3, stage A (`docs/2026-09-18_two_tier_plan_v3.zh.md` §5–§6): the (L, Δ) grid
declaration pins the numbered decisions (a silently changed lookback is what the 09-18 campaign
was built on), every cell names its own development cohort, `plan_cohort --arms` resolves one
config per cell, and `cohort_selection` keeps exactly the flights long enough for the cell —
on both sides of the split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ts_transformer.config import (
    CONTROL_RECIPE_SIMPLE_V3, PLAN_CONDITIONING_OFF, PREDICTION_CONTROL, RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM,
    TSConfig, default_anchor, recipe_settings,
)
from ts_transformer.data.dataset import build_series, series_from_row
from ts_transformer.data.splits import split_by_flight
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.experiments import plan_cohort
from ts_transformer.experiments.support import TS_DIR, arm_config, declaration_base
from ts_transformer.tests.support import AIRPORT, RUNWAY

GRID = TS_DIR / "docs" / "experiments" / "two_tier_v3_grid_arms.json"
LOOKBACKS_S = (30, 60, 90, 120)      # D4
SEGMENTS_S = (20, 30, 60, 90, 120)   # D4
SEEDS = (1337, 2024)                 # D9


def _declaration() -> dict:
    return json.loads(GRID.read_text(encoding="utf-8"))


def test_the_grid_declaration_pins_the_stage_a_decisions():
    declaration = _declaration()
    base = declaration_base(declaration)
    arms = declaration["arms"]
    assert declaration["predict"] is False and "development_cohort" not in declaration
    assert len(arms) == len(LOOKBACKS_S) * len(SEGMENTS_S) * len(SEEDS) == 40
    # file order = queue order (§8): Δ ascending, then L ascending, the two seeds together
    expected_keys = [f"L{L}_D{D}_s{seed}" for D in SEGMENTS_S for L in LOOKBACKS_S for seed in SEEDS]
    assert [arm["key"] for arm in arms] == expected_keys
    cohorts: dict[str, set[str]] = {}
    for arm in arms:
        config, settings = arm_config(base, arm["overrides"])
        L, D, seed = (int(part[1:]) for part in arm["key"].split("_"))
        assert config.seq_len * config.dt_s == L and config.control_horizon_s == D and config.seed == seed
        assert default_anchor(config) == config.seq_len - 1 and config.anchor_floor_index == 0        # D2 / A-dev3
        assert config.n_segments == D // 10                                                          # D6
        assert config.random_train_anchor_min_future_s == D                                          # one floor rule
        assert config.random_train_anchor and config.random_train_anchor_l1_share == 0.0             # D7 / A-dev5
        assert config.random_train_anchor_sampling == RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM       # D7
        assert config.plan_conditioning == PLAN_CONDITIONING_OFF                                     # D10
        assert config.control_thrust_parameterization == "specific-force+path-angle"                 # D5
        assert config.control_dynamics_model == "first-order-lag"                                    # D5
        assert config.epochs == config.patience == 180                                               # D8: no early stop
        assert config.learning_rate == pytest.approx(3e-5) and config.batch_size == 512              # D8
        assert config.checkpoint_selection_metric == "fixed-anchor-common-grid-ade"                  # D8
        assert config.split_seed == 1337                                                             # D1
        assert config.prediction_output == PREDICTION_CONTROL and config.control_recipe_name == "custom"
        assert "seed" in settings and "split_seed" in settings      # the runner adds neither flag
        cohorts.setdefault(arm["development_cohort"], set()).add(arm["key"])
    # one cohort per cell, shared by the cell's two seeds and by nothing else
    assert len(cohorts) == 20
    for path, keys in cohorts.items():
        assert len(keys) == 2 and {key.rsplit("_s", 1)[0] for key in keys} == {Path(path).parent.name}
        assert "{airport}" in path and "two_tier_v3_grid_20260918/cohorts/" in path


def test_declared_cohorts_resolves_one_config_per_cell_and_refuses_an_unpinned_split(tmp_path):
    cohorts = plan_cohort.declared_cohorts(_declaration(), "KRDU")
    assert len(cohorts) == 20 and all("KRDU" in str(path) for path in cohorts)
    assert {(config.seq_len, config.control_horizon_s) for config in cohorts.values()} == {
        (L // 2, float(D)) for L in LOOKBACKS_S for D in SEGMENTS_S
    }
    declaration = _declaration()
    del declaration["base"]["split_seed"]
    with pytest.raises(ValueError, match="split_seed"):
        plan_cohort.declared_cohorts(declaration, "KRDU")
    declaration = _declaration()
    declaration["arms"][1]["overrides"]["n_segments"] = 3      # the cell's second seed differs beyond the seed
    with pytest.raises(ValueError, match="differ beyond the seed"):
        plan_cohort.declared_cohorts(declaration, "KRDU")
    declaration = _declaration()
    del declaration["arms"][0]["development_cohort"]
    with pytest.raises(ValueError, match="development_cohort"):
        plan_cohort.declared_cohorts(declaration, "KRDU")
    # one recipe on the grid's axes: a cell differing on any other field is refused by name
    declaration = _declaration()
    declaration["arms"][2]["overrides"]["epochs"] = 90
    with pytest.raises(ValueError, match="differ on epochs"):
        plan_cohort.declared_cohorts(declaration, "KRDU")
    declaration = _declaration()
    declaration["arms"][2]["development_cohort"] = declaration["arms"][2]["development_cohort"].replace("cohorts/L60_D20", "elsewhere/L30_D20")
    declaration["arms"][3]["development_cohort"] = declaration["arms"][2]["development_cohort"]
    with pytest.raises(ValueError, match="share a directory name"):
        plan_cohort.declared_cohorts(declaration, "KRDU")
    # every cell's config is the declaration's: a config flag beside --arms is refused before any load
    with pytest.raises(SystemExit):
        plan_cohort.main(["--data", str(tmp_path / "none"), "--output-dir", str(tmp_path / "out"), "--airport", "KRDU",
                          "--arms", str(GRID), "--name", "x", "--seq-len", "15"])
    # D3's bins are the anchor grid's, and the queue reads exactly them
    from ts_transformer.data.anchor_strata import DEFAULT_ANCHOR_GRID_KM
    from ts_transformer.experiments.two_tier_grid_queue import DEFAULT_READINGS
    assert {12, 8, 6} <= set(DEFAULT_ANCHOR_GRID_KM) and DEFAULT_READINGS == ("L-1", "12km", "8km", "6km")


def _cell_config(seq_len: int, horizon_s: float) -> TSConfig:
    settings = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False)
    settings.update(dict(
        prediction_output=PREDICTION_CONTROL, control_horizon_s=horizon_s, n_segments=2, control_imitation_loss_weight=0.0,
        final_time_loss_weight=0.0, state_endpoint_loss_weight=0.0, seq_len=seq_len, d_model=16, n_heads=4, d_ff=32,
        e_layers=1, device="cpu", epochs=1, patience=1, random_train_anchor=True,
        random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, random_train_anchor_min_future_s=horizon_s,
        split_seed=1337,
    ))
    return TSConfig(**settings)


def test_cohort_selection_keeps_only_the_flights_long_enough_for_the_cell():
    """D1: a cell's usable set is the flights with L of lookback and Δ of truth after it, on BOTH
    sides of the split; a shorter flight is `not_usable`, never silently absent."""
    short = _cell_config(seq_len=4, horizon_s=20.0)
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3), short, airport=AIRPORT)
    # two flights first seen late enough that only the shorter lookback leaves a window:
    # 15 rows hold anchor 3 + 10 rows of truth, not anchor 7 + 10
    series[0] = series_from_row(series[0], series[0].n_samples - 15)
    series[5] = series_from_row(series[5], series[5].n_samples - 15)
    train, val, _test = split_by_flight(series, short)
    outer = {"train": [s.dataset_id for s in train], "val": [s.dataset_id for s in val]}
    cut = {series[0].dataset_id, series[5].dataset_id}

    keep, val_keys, selection = plan_cohort.cohort_selection(series, outer, short)
    assert set(keep) | set(val_keys) == set(outer["train"]) | set(outer["val"]) and selection["not_usable"] == []
    assert selection["lookback_s"] == 8.0 and selection["control_horizon_s"] == 20.0

    longer = _cell_config(seq_len=8, horizon_s=20.0)
    keep, val_keys, selection = plan_cohort.cohort_selection(series, outer, longer)
    assert set(selection["not_usable"]) == cut and not cut & (set(keep) | set(val_keys))
    assert (set(keep) | set(val_keys)) == (set(outer["train"]) | set(outer["val"])) - cut
    assert "16 s of lookback and 20 s of truth after it" in selection["rule"]


def test_a_cell_cohort_names_only_flights_the_cell_s_own_build_keeps():
    """Review 2026-09-18 (1): the window rule admits a flight with exactly L observed rows when
    its supervision runs past them, but `build_series` needs L + 1 rows — a cohort naming such
    a flight would make the arm's own build refuse the run after the full load. The cohort
    applies the build's gate (`minimum_build_samples`), so every cohort key is buildable."""
    from dataclasses import replace

    from ts_transformer.data.dataset import minimum_build_samples
    from ts_transformer.training.train import usable_series

    short, longer = _cell_config(seq_len=4, horizon_s=20.0), _cell_config(seq_len=8, horizon_s=20.0)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    series, _report = build_series(flights, short, airport=AIRPORT)
    cut = series_from_row(series[1], series[1].n_samples - longer.seq_len)        # exactly L rows observed …
    tail_t = float(cut.supervision_times[-1]) + longer.dt_s * np.arange(1, 12)     # … and 22 s of truth past them
    cut = replace(cut, supervision_times=np.concatenate((cut.supervision_times, tail_t)),
                  supervision_values=np.concatenate((cut.supervision_values, np.repeat(cut.supervision_values[-1:], 11, axis=0))),
                  supervision_weights=np.concatenate((cut.supervision_weights, np.repeat(cut.supervision_weights[-1:], 11, axis=0))))
    series[1] = cut
    assert cut.n_samples == longer.seq_len < minimum_build_samples(longer)
    assert cut.dataset_id in {item.dataset_id for item in usable_series(series, longer, verbose=False)}     # the window rule keeps it
    train, val, _test = split_by_flight(series, short)
    outer = {"train": [s.dataset_id for s in train], "val": [s.dataset_id for s in val]}
    keep, val_keys, selection = plan_cohort.cohort_selection(series, outer, longer)
    assert cut.dataset_id in selection["not_usable"] and cut.dataset_id not in set(keep) | set(val_keys)
    buildable = {item.dataset_id for item in build_series(flights, longer, airport=AIRPORT)[0]}
    assert set(keep) | set(val_keys) <= buildable


def test_series_from_row_keeps_the_clock_and_cuts_the_truth_alike():
    config = _cell_config(seq_len=4, horizon_s=20.0)
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=1, seed=3), config, airport=AIRPORT)
    whole = series[0]
    later = series_from_row(whole, 10)
    assert later.n_samples == whole.n_samples - 10 and later.times[0] == whole.times[10]
    np.testing.assert_array_equal(later.values, whole.values[10:])
    assert later.supervision_times[0] == pytest.approx(whole.times[10]) and later.supervision_times[-1] == whole.supervision_times[-1]
    assert later.n_supervision_samples == whole.n_supervision_samples - 10
    assert later.flight_id == whole.flight_id and later.scenario is whole.scenario
    assert series_from_row(whole, 0) is not whole and series_from_row(whole, 0).n_samples == whole.n_samples
    with pytest.raises(ValueError, match="first_row"):
        series_from_row(whole, whole.n_samples)


def test_the_grid_gate_runner_groups_arms_by_cell_and_reports_pending_readings(tmp_path, capsys):
    from ts_transformer.experiments import executor_grid_gate as gate_runner
    from ts_transformer.manoeuvre.gates import cell_name

    cells = gate_runner.grid_cells(_declaration())
    assert len(cells) == 20 and all(sorted(by_seed) == list(SEEDS) for by_seed in cells.values())
    assert cells[(30.0, 20.0)] == {1337: "L30_D20_s1337", 2024: "L30_D20_s2024"}
    assert cell_name(30.0, 20.0) == "L30_D20"
    # a campaign with one cell's two readings written: the table has them, the verdict waits
    campaign = tmp_path / "campaign"
    payload = {"protocol": "none", "segment_s": 20.0, "flights": 1400, "strata": {
        "all": {"n": 1400, "established_share": 0.7, "fully_flyable_share": 0.99, "fde_p50_m": 500.0},
        "vectored (tortuosity >= 1.05, not established)": {"n": 600, "established_share": 0.3, "ade_mean_m": 2000.0},
        "straight-in (tortuosity < 1.05)": {"n": 800, "established_share": 1.0}}}
    for arm in ("L30_D20_s1337", "L30_D20_s2024"):
        path = gate_runner.reading_path(campaign, arm, "L-1")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    assert gate_runner.main(["--campaign", str(campaign), "--arms", str(GRID), "--out", str(tmp_path / "gate")]) == 0
    result = json.loads((tmp_path / "gate" / "grid_gate.json").read_text(encoding="utf-8"))
    assert result["verdict"] is None and len(result["pending"]) == 38 and result["pending"][0] == "L60_D20_s1337/L-1"
    assert result["table"]["L30_D20"]["1337"]["established_all"] == 0.7 and result["table"]["L60_D20"]["1337"] is None
    assert "verdict: pending (38 readings missing" in capsys.readouterr().out
    with pytest.raises(SystemExit):        # never overwritten
        gate_runner.main(["--campaign", str(campaign), "--arms", str(GRID), "--out", str(tmp_path / "gate")])


def test_the_queue_plans_one_cell_at_a_time_and_skips_written_readings(tmp_path, capsys):
    from ts_transformer.experiments import two_tier_grid_queue as queue

    cells = queue.cells_of(_declaration())
    assert list(cells)[:3] == ["L30_D20", "L60_D20", "L90_D20"] and cells["L30_D20"] == ["L30_D20_s1337", "L30_D20_s2024"]
    campaign = tmp_path / "campaign"
    steps = queue.cell_steps("L30_D20", cells["L30_D20"], declaration=GRID, campaign=campaign, airport="KRDU",
                             readings=["L-1", "12km"], device="cpu")
    labels = [label for label, _command, _artefact in steps]
    assert labels == ["L30_D20: train L30_D20_s1337, L30_D20_s2024", "L30_D20_s1337: lockstep none, L-1", "L30_D20_s1337: lockstep none, 12km",
                      "L30_D20_s2024: lockstep none, L-1", "L30_D20_s2024: lockstep none, 12km", "L30_D20: grid gate"]
    train = steps[0][1]
    assert train[train.index("--only") + 1:] == ["L30_D20_s1337", "L30_D20_s2024"] and steps[0][2] is None
    twelve = steps[2][1]
    assert twelve[twelve.index("--anchor-remaining-km") + 1] == "12" and "--write-records" not in twelve
    assert steps[1][2] == campaign / "lockstep" / "L30_D20_s1337" / "L-1" / "manoeuvre_lockstep.json"
    assert "--anchor-remaining-km" not in steps[1][1] and steps[-1][2] == campaign / "gate" / "after_L30_D20" / "grid_gate.json"
    assert queue.reading_flags("row59") == ["--first-prediction-row", "59"]
    for bad in ("12", "rowx", "row"):
        with pytest.raises(ValueError, match="reading"):
            queue.reading_flags(bad)
    # a dry run lists the steps and marks a written reading done; a cell the declaration lacks is refused
    steps[1][2].parent.mkdir(parents=True)
    steps[1][2].write_text("{}", encoding="utf-8")
    assert queue.main(["--arms", str(GRID), "--campaign", str(campaign), "--airport", "KRDU", "--cells", "L30_D20",
                       "--readings", "L-1", "12km", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert printed.count("CELL ") == 1 and "done L30_D20_s1337: lockstep none, L-1" in printed and "todo L30_D20_s2024: lockstep none, L-1" in printed
    assert not (campaign / queue.PID_FILE).exists()
    with pytest.raises(SystemExit):
        queue.main(["--arms", str(GRID), "--campaign", str(campaign), "--airport", "KRDU", "--cells", "L7_D7", "--dry-run"])
