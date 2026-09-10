"""The A0 anchor grid: what a bin means, and what must NOT move when the anchor does.

Four things are silently wrong if they drift, and each is measured here rather than argued:

* the bin coordinate is the SAME remaining path the NEAR / FAR strata are cut on — one
  arithmetic, not two that agree today;
* a bin's anchor is the closest sample and nothing else, and a flight that cannot be
  anchored there has no reading at that bin rather than a reading taken elsewhere;
* the stratum label is taken ONCE at L−1, so a flight that rolls out on the centreline at
  4 km is still a vectored flight there (otherwise the curve is a survivor curve);
* the forecast really is re-anchored — its history ends at the new anchor and its error is
  scored against the truth AFTER it, not against the whole track.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ts_transformer.inference.forecast as forecast_module
import ts_transformer.outputs.state.forecast as state_forecast_module
import ts_transformer.experiments.anytime_curve as runner
from ts_transformer.data.approach_difficulty import (
    STRATUM_ALL,
    STRATUM_ESTABLISHED,
    STRATUM_STRAIGHT_IN,
    STRATUM_VECTORED,
    approach_difficulty,
    remaining_path_profile_m,
)
from ts_transformer.config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CTA_CONDITIONING_GIVEN,
    PREDICTION_CONTROL,
    TSConfig,
)
from ts_transformer.data.data_provenance import provenance_manifest_digests
from ts_transformer.data.dataset import (
    FlightSeries,
    build_series,
    dataset_flight_key,
    truth_duration_s,
)
from ts_transformer.inference.export import observed_series_metrics
from ts_transformer.inference.forecast import Forecast, history_at_anchor, forecast_approaches
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.training.train import load_checkpoint, train
from ts_transformer.experiments.support import forecast_geometry
from ts_transformer.tests.support import fake_data_provenance

AIRPORT, RUNWAY = "KRDU", "05L"


def _config(**overrides) -> TSConfig:
    """A tiny CPU control recipe — the curve is output-agnostic, but the control path is
    the one every arm in §2.2 is on."""
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_rollout_integrator_dt_s=0.5,
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=1, patience=1, batch_size=8, dropout=0.0,
        val_fraction=0.25, test_fraction=0.25,
    )
    settings.update(overrides)
    return TSConfig(**settings)


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory):
    """One tiny control checkpoint on synthetic arrivals, plus the flights behind it."""
    torch.manual_seed(0)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    config = _config()
    series, _report = build_series(flights, config, airport=AIRPORT)
    out = tmp_path_factory.mktemp("anytime_run")
    train(series, config, output_dir=out, data_provenance=fake_data_provenance(), verbose=False)
    return flights, out / "checkpoint.pt"


@pytest.fixture(scope="module")
def built_series(trained_checkpoint):
    flights, _checkpoint = trained_checkpoint
    series, _report = build_series(flights, _config(), airport=AIRPORT)
    return series


def _patch_data_plane(monkeypatch, flights, tmp_path):
    """The runner's three data-plane seams, pointed at the synthetic flights.

    The fingerprint seam is `checkpoint_data_provenance` — the package helper that owns the
    roster rule — so a runner that went back to the plain manifest hash would not be patched
    here and would fail against the real rosters, which is the bug this seam once hid.
    """
    manifest = tmp_path / "manifest.json"
    indexed = {dataset_flight_key(flight, index): flight
               for index, flight in enumerate(flights)}
    monkeypatch.setattr(runner.pipeline, "arrival_manifest_path", lambda _airport: manifest)
    monkeypatch.setattr(
        runner, "checkpoint_data_provenance", lambda _payload, _manifests: fake_data_provenance()
    )
    monkeypatch.setattr(
        runner, "load_flight_dicts",
        lambda _paths, include_flight_keys, verbose=True: [
            flight for key, flight in indexed.items() if key in include_flight_keys
        ],
    )


# ── the bin coordinate is the covariate's own arithmetic ────────────────────

def _bare_series(horizontal: list[tuple[float, float]], *, dt_s: float = 2.0) -> FlightSeries:
    """A flight that is nothing but a horizontal path — enough for the profile and the
    anchor rules, which read no target and no frame."""
    values = np.zeros((len(horizontal), 6), dtype=np.float64)
    values[:, :2] = np.array(horizontal, dtype=np.float64)
    return FlightSeries(
        flight_id="BARE1_05L_abc123_20260101T000000Z",
        scenario=SimpleNamespace(target=None),
        frame=None,
        times=np.arange(len(horizontal), dtype=np.float64) * dt_s,
        values=values,
    )


def test_the_profile_is_the_arc_length_still_to_fly() -> None:
    series = _bare_series([(-3_000.0, 0.0), (-2_000.0, 0.0), (-1_000.0, 0.0), (0.0, 0.0)])
    assert remaining_path_profile_m(series) == pytest.approx([3_000.0, 2_000.0, 1_000.0, 0.0])


def test_the_profile_is_the_covariate_at_every_anchor(built_series) -> None:
    """One definition of remaining path: the bin coordinate IS `remaining_path_m`."""
    for series in built_series[:3]:
        profile = remaining_path_profile_m(series)
        assert len(profile) == series.n_samples
        for anchor in (0, 7, series.n_samples // 2, series.n_samples - 2):
            assert profile[anchor] == pytest.approx(
                approach_difficulty(series, anchor).remaining_path_m, rel=1e-12
            )


# ── the bin's anchor ────────────────────────────────────────────────────────

def test_the_bin_takes_the_closest_sample() -> None:
    series = _bare_series([(-3_000.0, 0.0), (-2_000.0, 0.0), (-1_000.0, 0.0), (0.0, 0.0)])
    profile = remaining_path_profile_m(series)
    for target, expected in ((3_000.0, 0), (1_800.0, 1), (1_200.0, 2)):
        assert runner.bin_anchor(
            series, profile, target, seq_len=1, min_future_s=0.0
        ) == expected


def test_a_bin_whose_closest_sample_has_no_full_lookback_is_empty() -> None:
    series = _bare_series([(-3_000.0, 0.0), (-2_000.0, 0.0), (-1_000.0, 0.0), (0.0, 0.0)])
    profile = remaining_path_profile_m(series)
    # 3 km is sample 0; a lookback of 3 samples needs the anchor at index 2 or later.
    assert runner.bin_anchor(series, profile, 3_000.0, seq_len=3, min_future_s=0.0) is None
    assert runner.bin_anchor(series, profile, 1_000.0, seq_len=3, min_future_s=0.0) == 2


def test_a_bin_with_too_little_truth_after_it_is_empty() -> None:
    """The last sample is always closest to 0 km and always has nothing left to predict."""
    series = _bare_series([(-3_000.0, 0.0), (-2_000.0, 0.0), (-1_000.0, 0.0), (0.0, 0.0)])
    profile = remaining_path_profile_m(series)
    assert truth_duration_s(series, 3) == 0.0
    assert runner.bin_anchor(series, profile, 0.0, seq_len=1, min_future_s=1.0) is None
    # ...and the rule is the truth AFTER the anchor, not the samples after it: sample 2 has
    # one 2 s step left, which passes a 2 s floor and fails a 3 s one.
    assert runner.bin_anchor(series, profile, 1_000.0, seq_len=1, min_future_s=2.0) == 2
    assert runner.bin_anchor(series, profile, 1_000.0, seq_len=1, min_future_s=3.0) is None


# ── the data identity the split is rebuilt under ────────────────────────────

def test_the_fingerprint_goes_through_the_package_helper(monkeypatch, tmp_path,
                                                         trained_checkpoint) -> None:
    """The roster rule has ONE owner: `data_provenance.checkpoint_data_provenance`.

    The pre-split lateral-pass roster is part of the data identity — fingerprinting the v5
    cohort without it lists every arrival candidate where the checkpoint carries only the
    eligible ones (14 435 vs 14 378 at KRDU) and the run dies claiming the manifest changed.
    That rule belongs to the helper, not to this runner: what is asserted here is that
    `load_arm` asks it, with the checkpoint's own payload and its own manifests.
    """
    _flights, checkpoint = trained_checkpoint
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(runner.pipeline, "arrival_manifest_path", lambda _airport: manifest)
    seen: dict = {}

    def spy(payload, manifests):
        seen["payload"], seen["manifests"] = payload, list(manifests)
        return fake_data_provenance()

    monkeypatch.setattr(runner, "checkpoint_data_provenance", spy)
    grid = runner.Grid(split="val", bins_m=(10_000.0,), min_future_s=10.0,
                       batch_size=None, limit=0)
    arm = runner.load_arm("tiny", checkpoint, grid, torch.device("cpu"))

    assert seen["manifests"] == [manifest]
    assert seen["payload"]["data_provenance"] == fake_data_provenance()
    # ...and the digests published are the checkpoint's own, never a fresh hash that would
    # have to repeat the roster rule to stay comparable.
    assert provenance_manifest_digests(arm.payload["data_provenance"]) == {AIRPORT: "a" * 64}


# ── the re-anchored forecast ────────────────────────────────────────────────

def _arm(checkpoint, label: str = "tiny") -> runner.Arm:
    model, config, normalizer, payload = load_checkpoint(checkpoint)
    return runner.Arm(
        label=label, path=checkpoint, model=model, config=config, normalizer=normalizer,
        payload=payload, airports=(AIRPORT,), manifests=[checkpoint.parent / "manifest.json"],
    )


def test_the_forecast_at_a_late_anchor_reads_the_history_ending_there(
    monkeypatch, trained_checkpoint, built_series
) -> None:
    """`measure_bin` re-anchors the MODEL, not just the scoring: the lookback it is shown
    ends at the bin's anchor, and the forecast says which anchor it came from."""
    _flights, checkpoint = trained_checkpoint
    arm = _arm(checkpoint)
    series = built_series[:2]
    profiles = [remaining_path_profile_m(item) for item in series]
    keys = [item.dataset_id for item in series]

    seen: list[tuple[int, np.ndarray]] = []
    original = forecast_module.history_at_anchor

    def spy(item, config, normalizer, anchor):
        history = original(item, config, normalizer, anchor)
        seen.append((anchor, history))
        return history

    monkeypatch.setattr(forecast_module, "history_at_anchor", spy)
    monkeypatch.setattr(state_forecast_module, "history_at_anchor", spy)
    rows, records = runner.measure_bin(
        arm.model, series, profiles, keys, 4_000.0, config=arm.config,
        normalizer=arm.normalizer, device=torch.device("cpu"), batch_size=8,
        min_future_s=10.0, split="val",
    )

    assert set(rows) == set(keys)
    assert records == []                    # no --write-records: nothing is assembled
    for item, profile, key in zip(series, profiles, keys):
        anchor = runner.bin_anchor(
            item, profile, 4_000.0, seq_len=arm.config.seq_len, min_future_s=10.0
        )
        assert anchor > arm.config.seq_len - 1           # a genuinely later anchor
        assert rows[key]["anchor_index"] == anchor
        assert rows[key]["remaining_path_m"] == pytest.approx(profile[anchor])
        history = next(row for row in seen if row[0] == anchor)[1]
        # Independently: this recipe carries no conditioning column, so the history the
        # model is shown IS the encoded lookback ENDING at the anchor. (Comparing against
        # `history_at_anchor` again would only prove the spy delegates.)
        expected = arm.normalizer.encode(item.values)[anchor - arm.config.seq_len + 1 : anchor + 1]
        assert history.shape == expected.shape
        assert history == pytest.approx(expected)
        assert history == pytest.approx(
            history_at_anchor(item, arm.config, arm.normalizer, anchor)
        )
        forecast = forecast_approaches(
            arm.model, [item], arm.config, arm.normalizer, anchor=anchor,
            device=torch.device("cpu"),
        )[0]
        assert forecast.anchor == anchor
        assert forecast.times[0] > float(item.times[anchor])


def _truth_forecast(series: FlightSeries, anchor: int, config: TSConfig) -> Forecast:
    """The truth after the anchor, dressed as a forecast: the zero-error control."""
    anchor_time = float(series.times[anchor])
    future = series.supervision_times > anchor_time
    offsets = np.asarray(series.supervision_times)[future] - anchor_time
    values = np.asarray(series.supervision_values)[future]
    durations = np.diff(np.concatenate(([0.0], offsets)))
    return Forecast(
        times=anchor_time + offsets, values=values,
        normalized_progress=offsets / float(offsets[-1]), anchor=anchor,
        final_time_s=float(offsets[-1]), predicted_final_time_s=float(offsets[-1]),
        horizon_mode=config.horizon_mode, passes=1, truncated_at_threshold=False,
        horizon_capped=False, sample_durations_s=durations, segment_durations_s=durations,
    )


def test_the_metrics_are_scored_after_the_anchor_at_every_bin(built_series) -> None:
    """`observed_series_metrics` is anchor-aware: a forecast that copies the truth from the
    bin's anchor onward scores zero there — at L−1 and at every later bin alike."""
    config = _config()
    for series in built_series[:3]:
        profile = remaining_path_profile_m(series)
        for target in (20_000.0, 10_000.0, 4_000.0):
            anchor = runner.bin_anchor(
                series, profile, target, seq_len=config.seq_len, min_future_s=10.0
            )
            metrics = observed_series_metrics(series, _truth_forecast(series, anchor, config))
            assert metrics["ade_m"] == pytest.approx(0.0, abs=1e-6)
            assert metrics["fde_m"] == pytest.approx(0.0, abs=1e-6)
            assert metrics["final_time_error_s"] == pytest.approx(0.0, abs=1e-9)
            # ...and the difficulty it carries is the one at THAT anchor, which is why the
            # runner takes its strata from L−1 separately.
            assert metrics["difficulty"]["remaining_path_m"] == pytest.approx(profile[anchor])


# ── the invariant: the stratum label is taken once, at L−1 ──────────────────

def test_the_strata_are_fixed_at_the_l_minus_one_anchor(
    trained_checkpoint, built_series
) -> None:
    """A flight vectored at L−1 and established at 4 km stays in the vectored stratum.

    The cohort is one of each: SYN05L000 is tortuous and off the centreline at L−1 and
    rolled out on final by 4 km; SYN05L002 is a straight-in from the start. Relabelling per
    bin would put BOTH in `established at anchor` at 4 km and empty the vectored stratum —
    which is exactly how a survivor curve looks like an improving one.
    """
    _flights, checkpoint = trained_checkpoint
    by_id = {item.flight_id.split("_")[0]: item for item in built_series}
    vectored, straight = by_id["SYN05L000"], by_id["SYN05L002"]
    arm = _arm(checkpoint)
    anchor_l1 = arm.config.seq_len - 1

    at_l1 = {item.flight_id: approach_difficulty(item, anchor_l1) for item in (vectored, straight)}
    assert not at_l1[vectored.flight_id].established_at_anchor
    assert at_l1[vectored.flight_id].route_tortuosity > 1.05          # vectored at L−1
    assert at_l1[straight.flight_id].route_tortuosity < 1.05          # straight-in at L−1

    late = {
        item.flight_id: runner.bin_anchor(
            item, remaining_path_profile_m(item), 4_000.0,
            seq_len=arm.config.seq_len, min_future_s=10.0,
        )
        for item in (vectored, straight)
    }
    for item in (vectored, straight):                                 # both established there
        assert approach_difficulty(item, late[item.flight_id]).established_at_anchor

    grid = runner.Grid(split="val", bins_m=(4_000.0,), min_future_s=10.0,
                       batch_size=8, limit=0)
    result = runner.measure_checkpoint(arm, [vectored, straight], grid, torch.device("cpu"))

    assert result["stratum_n_at_l1"] == {
        STRATUM_ALL: 2, STRATUM_STRAIGHT_IN: 1, STRATUM_VECTORED: 1,
        STRATUM_ESTABLISHED: 0, "remaining path < 13 km": 0, "remaining path >= 13 km": 2,
    }
    cells = result["bins"]["4000.0"]["strata"]
    assert cells[STRATUM_ALL]["n"] == 2
    assert cells[STRATUM_VECTORED]["n"] == 1        # NOT 0: the label did not follow the anchor
    assert cells[STRATUM_STRAIGHT_IN]["n"] == 1
    assert cells[STRATUM_ESTABLISHED]["n"] == 0     # NOT 2
    assert cells[STRATUM_VECTORED]["coverage"] == pytest.approx(1.0)
    assert cells[STRATUM_ESTABLISHED]["stratum_n_at_l1"] == 0


# ── refusals, and the artifact ──────────────────────────────────────────────

def test_the_sealed_test_split_is_refused(tmp_path, trained_checkpoint, capsys) -> None:
    _flights, checkpoint = trained_checkpoint
    with pytest.raises(SystemExit):
        runner.main(["--checkpoint", f"a={checkpoint}", "--out", str(tmp_path / "sealed"),
                     "--split", "test", "--device", "cpu"])
    assert "sealed" in capsys.readouterr().err
    assert not (tmp_path / "sealed").exists()


def test_a_cta_conditioned_checkpoint_is_refused(monkeypatch, tmp_path) -> None:
    """`cta_conditioning=given` IS the truth duration: its curve would improve for free."""
    torch.manual_seed(0)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3)
    config = _config(cta_conditioning=CTA_CONDITIONING_GIVEN)
    series, _report = build_series(flights, config, airport=AIRPORT)
    run = tmp_path / "cta_run"
    train(series, config, output_dir=run, data_provenance=fake_data_provenance(), verbose=False)
    _patch_data_plane(monkeypatch, flights, tmp_path)

    out = tmp_path / "never"
    with pytest.raises(SystemExit, match="reads the future"):
        runner.main(["--checkpoint", f"cta={run / 'checkpoint.pt'}", "--out", str(out),
                     "--device", "cpu"])
    assert not out.exists()        # the immutable dir is claimed after every refusal


def test_an_intent_conditioned_checkpoint_is_refused(monkeypatch, tmp_path,
                                                     trained_checkpoint) -> None:
    """`intent_conditioning=truth-…` reads the future, and reads it AGAIN at every anchor.

    Its lead and remaining-time channels are measured at the window's anchor, so a
    re-anchored replay hands the model a fresh oracle answer at every bin: the curve would
    measure how quickly the oracle converges, not how quickly intent is exposed. Three such
    checkpoints exist under `scene_phase0_20260905`, which is why this is a refusal and not
    a note.
    """
    _flights, checkpoint = trained_checkpoint
    model, config, normalizer, payload = load_checkpoint(checkpoint)
    monkeypatch.setattr(
        runner, "load_checkpoint",
        lambda _path: (model, replace(config, intent_conditioning="truth-join"),
                       normalizer, payload),
    )
    out = tmp_path / "never"
    with pytest.raises(SystemExit, match="reads the FUTURE"):
        runner.main(["--checkpoint", f"intent={checkpoint}", "--out", str(out),
                     "--device", "cpu"])
    assert not out.exists()


@pytest.mark.parametrize("argv, message", [
    (["--checkpoint", "nolabel"], "LABEL=PATH"),
    (["--checkpoint", "a=x", "--checkpoint", "a=y"], "used twice"),
    (["--checkpoint", "a=x", "--bins-km", "8,8"], "repeats a bin"),
    (["--checkpoint", "a=x", "--bins-km", "0"], "positive"),
])
def test_the_command_line_is_refused_before_anything_is_created(
    tmp_path, capsys, argv, message
) -> None:
    with pytest.raises(SystemExit):
        runner.main([*argv, "--out", str(tmp_path / "never"), "--device", "cpu"])
    assert message in capsys.readouterr().err
    assert not (tmp_path / "never").exists()


def test_the_curve_runs_end_to_end_and_states_its_coverage(
    monkeypatch, tmp_path, trained_checkpoint, capsys
) -> None:
    flights, checkpoint = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    out = tmp_path / "anytime"
    assert runner.main([
        "--checkpoint", f"tiny={checkpoint}", "--out", str(out), "--device", "cpu",
        "--bins-km", "20,10,4", "--min-future-s", "10", "--batch-size", "4",
    ]) == 0

    payload = json.loads((out / "anytime_curve.json").read_text())
    assert payload["schema"] == runner.RESULT_SCHEMA
    assert payload["grid"] == {
        "split": "val", "bins_m": [20_000.0, 10_000.0, 4_000.0],
        "min_future_s": 10.0, "limit": 0, "write_records": False,
    }
    assert "closest" in payload["anchor_definition"]
    assert payload["strata_anchor"].startswith("L-1")
    assert "supervision rows" in payload["geometry_truth"]

    arm = payload["checkpoints"]["tiny"]
    assert arm["arm"] == runner.ARM_FIXED and arm["random_train_anchor"] is False
    assert arm["command_hook"] is None
    assert arm["checkpoint"] == str(checkpoint) and len(arm["checkpoint_sha256"]) == 64
    assert arm["airports"] == [AIRPORT] and arm["prediction_output"] == "control"
    assert arm["anchor_l1"] == 7 and arm["flights"] > 0
    assert arm["flights"] == arm["split_flights"]          # no --limit here
    assert arm["stratum_n_at_l1"][STRATUM_ALL] == arm["flights"]
    assert set(arm["bins"]) == {"20000.0", "10000.0", "4000.0"}
    cell_keys = {"n", "stratum_n_at_l1", "coverage", "partial"} | {
        metric.key for metric in runner.CELL_METRICS
    }
    for value, block in arm["bins"].items():
        assert set(block) == {"strata", "flights"}
        assert set(block["strata"]) == set(arm["stratum_n_at_l1"])
        for stratum, cell in block["strata"].items():
            assert set(cell) == cell_keys
            assert cell["stratum_n_at_l1"] == arm["stratum_n_at_l1"][stratum]
            assert cell["partial"] == (cell["coverage"] < runner.PARTIAL_COVERAGE)
            assert (cell["ade_mean_m"] is None) == (cell["n"] == 0)
        # The per-flight rows ARE the `all` stratum's population at that bin, which is what
        # makes a paired reading across bins possible at all.
        assert len(block["flights"]) == block["strata"][STRATUM_ALL]["n"]
        for row in block["flights"].values():
            assert set(row) == {
                "anchor_index", "remaining_path_m", "ade_m", "fde_m", "final_time_error_s",
                "predicted_final_time_s", "chamfer_m", "frechet_m",
            }
            assert row["anchor_index"] >= arm["anchor_l1"]
    # Every flight of this synthetic split reaches every bin (they all fly 25 km in).
    for value in arm["bins"]:
        assert arm["bins"][value]["strata"][STRATUM_ALL]["n"] == arm["flights"]

    verdicts = arm["verdicts"]
    assert verdicts["stratum"] == STRATUM_VECTORED and verdicts["statistic"] == "ade_p50_m"
    assert set(verdicts) == {
        "stratum", "statistic", "bins_read_m", "bins_skipped_partial_m", "monotone",
        "reachable", "freeze",
    }
    # This cohort has vectored flights in every bin, so the verdict is READ (never "unread")
    # and every adjacent pair is paired over flights that reached both.
    assert verdicts["monotone"]["status"] in {"pass", "fail"}
    assert len(verdicts["monotone"]["steps"]) == len(verdicts["bins_read_m"]) - 1
    for step in verdicts["monotone"]["steps"]:
        assert step["paired_n"] > 0
        assert step["delta_p50_m"] == pytest.approx(
            step["ade_p50_to_m"] - step["ade_p50_from_m"]
        )
    assert verdicts["freeze"]["duration_head_floor_s"] == runner.DURATION_HEAD_FLOOR_S

    text = (out / "anytime_curve.txt").read_text()
    assert "A0 anytime curve" in text
    assert f"{runner.ARM_FIXED}: the checkpoint was trained at L-1" in text
    assert f"[{runner.ARM_RANDOM}]" not in text            # one arm, and it is the fixed one
    assert "This run holds BOTH arms" not in text
    for metric in runner.CELL_METRICS:                     # every published column prints
        assert metric.header in text
    for stratum in arm["stratum_n_at_l1"]:
        assert stratum in text
    assert "1. paired vectored ADE p50" in text
    assert verdicts["monotone"]["status"].upper() in text
    assert "3. s_freeze" in text and "duration head cannot predict below ~125 s" in text
    assert arm["record_dirs"] == {}                        # no --write-records was asked for
    assert "records (" not in text
    # The bin's flight count reaches the terminal too, not only the artifact.
    assert "4.0 km:" in capsys.readouterr().out


# ── --write-records: the forecasts the curve scored, publishable as they are ─

def test_write_records_publishes_the_forecasts_the_curve_itself_scored(
    monkeypatch, tmp_path, trained_checkpoint, built_series
) -> None:
    """The record set and the curve come out of ONE forward pass.

    Three things make a published anytime prediction mean what it says, and each is checked
    against the curve rather than against itself: the record is anchored at the BIN's anchor
    (`source.anchorTimeS` is that sample's time — the shared clock §六 7 puts a re-anchored
    forecast on), its states begin AT that sample, and the summary's per-flight error is the
    very number the curve's cell was computed from. A second replay would satisfy the first
    two and quietly break the third.
    """
    flights, checkpoint = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    out = tmp_path / "with_records"
    assert runner.main([
        "--checkpoint", f"tiny={checkpoint}", "--out", str(out), "--device", "cpu",
        "--bins-km", "10", "--min-future-s", "10", "--write-records",
    ]) == 0

    payload = json.loads((out / "anytime_curve.json").read_text())
    assert payload["grid"]["write_records"] is True
    arm = payload["checkpoints"]["tiny"]
    assert arm["record_dirs"] == {"10000.0": f"{runner.RECORDS_DIR}/tiny/10km"}
    directory = out / runner.RECORDS_DIR / "tiny" / "10km"
    summary = json.loads((directory / "summary.json").read_text())

    # The summary says WHOSE numbers these are (§六 2: a bin is a subset, never the split).
    block = summary["anytime"]
    assert block["schema"] == runner.RECORDS_SCHEMA
    assert block["campaign"] == out.name and block["label"] == "tiny"
    assert block["arm"] == runner.ARM_FIXED
    assert block["bin_m"] == 10_000.0 and block["bin_label"] == "10km"
    assert block["anchor_rule"] == runner.ANCHOR_DEFINITION
    assert block["min_future_s"] == 10.0 and block["split"] == "val" and block["limit"] == 0
    cell = arm["bins"]["10000.0"]["strata"][STRATUM_ALL]
    assert block["records"] == len(summary["results"]) == cell["n"]
    assert block["measured_flights"] == arm["flights"]
    assert block["split_flights"] == arm["split_flights"]
    assert block["coverage"] == pytest.approx(cell["coverage"])

    by_key = {series.dataset_id: series for series in built_series}
    rows = arm["bins"]["10000.0"]["flights"]
    assert summary["results"]                              # the bin is populated at all
    for result in summary["results"]:
        stem = result["states_file"][: -len("_states.json")]
        row = rows[f"{AIRPORT}:{stem}"]
        states = json.loads((directory / result["states_file"]).read_text())
        anchor = row["anchor_index"]
        # 1. the record is anchored at the BIN's anchor, not at L−1...
        assert anchor > arm["anchor_l1"]
        assert states["source"]["anchorIndex"] == anchor
        assert states["source"]["anchorTimeS"] == pytest.approx(
            float(by_key[f"{AIRPORT}:{stem}"].times[anchor])
        )
        # 2. ...and its first predicted row IS that sample: t=0 there, at the observed
        # position, which is what lets a viewer place it on the observed flight's clock.
        assert states["predicted_states"][0]["t"] == 0.0
        assert states["observed_states"][anchor]["t"] == 0.0
        for axis in ("lat", "lon"):
            assert states["predicted_states"][0][axis] == pytest.approx(
                states["observed_states"][anchor][axis]
            )
        # 3. one forward pass: the published error IS the curve's.
        assert result["ade_m"] == row["ade_m"]
        assert result["fde_m"] == row["fde_m"]
        assert result["final_time_error_s"] == row["final_time_error_s"]
        assert result["predicted_final_time_s"] == row["predicted_final_time_s"]

    assert "records (the SAME forecasts" in (out / "anytime_curve.txt").read_text()


def test_a_limited_record_run_states_the_limit_it_was_taken_under(
    monkeypatch, tmp_path, trained_checkpoint
) -> None:
    """A subset's numbers must never read as the split's: the block carries both counts."""
    flights, checkpoint = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    out = tmp_path / "limited_records"
    assert runner.main([
        "--checkpoint", f"tiny={checkpoint}", "--out", str(out), "--device", "cpu",
        "--bins-km", "10", "--min-future-s", "10", "--limit", "2", "--write-records",
    ]) == 0

    block = json.loads(
        (out / runner.RECORDS_DIR / "tiny" / "10km" / "summary.json").read_text()
    )["anytime"]
    assert block["limit"] == 2
    assert block["measured_flights"] == 2
    assert block["split_flights"] > 2
    assert block["records"] <= 2


def test_a_failed_record_run_stages_nothing_under_the_published_name(
    monkeypatch, tmp_path, trained_checkpoint
) -> None:
    """The records are built INSIDE the staging directory, so a crash after some bins have
    been written still leaves no half-published record set a viewer could load."""
    flights, checkpoint = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    real = runner.write_bin_records

    def explode_on_the_second(*args, **kwargs):
        if getattr(explode_on_the_second, "seen", False):
            raise RuntimeError("the writer died")
        explode_on_the_second.seen = True
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "write_bin_records", explode_on_the_second)
    out = tmp_path / "never_records"
    with pytest.raises(RuntimeError, match="the writer died"):
        runner.main(["--checkpoint", f"tiny={checkpoint}", "--out", str(out),
                     "--device", "cpu", "--bins-km", "10,8", "--min-future-s", "10",
                     "--write-records"])
    assert not out.exists()
    assert not list(tmp_path.glob("never_records.partial-*"))


def test_the_geometry_is_reported_beside_the_time_aligned_error(built_series) -> None:
    """MEDIUM 3 / the package rule: a cell carries chamfer and Fréchet as well as ADE.

    The zero-error control pins the seam — a forecast that IS the truth after the anchor has
    zero chamfer and zero Fréchet, so the two paths handed to `geometric_metrics` are the
    same pair `observed_series_metrics` scores.
    """
    config = _config()
    for series in built_series[:2]:
        profile = remaining_path_profile_m(series)
        anchor = runner.bin_anchor(
            series, profile, 8_000.0, seq_len=config.seq_len, min_future_s=10.0
        )
        geometry = forecast_geometry(series, _truth_forecast(series, anchor, config))
        assert geometry["chamfer_m"] == pytest.approx(0.0, abs=1e-6)
        assert geometry["frechet_m"] == pytest.approx(0.0, abs=1e-6)


def test_an_arm_is_named_from_its_own_training_anchor_policy(
    monkeypatch, tmp_path, trained_checkpoint
) -> None:
    """HIGH 2: `A0-fixed` / `A0-random` is a per-CHECKPOINT fact, and a mixed run says so.

    The two arms are not comparable as runs — their difference IS the out-of-distribution
    cost §六 3 asks for — so the label rides in each checkpoint's own block and the banner
    covers the arms actually present. The random arm here is the fixed checkpoint re-read
    with `random_train_anchor=True`: this test is about the LABEL, and training a random-
    anchor model is currently broken upstream anyway (`RandomAnchorTrajectoryWindows`
    takes no `fitted_teacher`).
    """
    flights, checkpoint = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    twin = tmp_path / "twin.pt"
    twin.write_bytes(checkpoint.read_bytes())
    plain = load_checkpoint(checkpoint)

    def load(path):
        model, config, normalizer, payload = plain
        if Path(path) == twin:
            config = replace(config, random_train_anchor=True,
                             random_train_anchor_min_future_s=20.0)
        return model, config, normalizer, payload

    monkeypatch.setattr(runner, "load_checkpoint", load)

    out = tmp_path / "mixed"
    assert runner.main([
        "--checkpoint", f"fixed={checkpoint}", "--checkpoint", f"random={twin}",
        "--out", str(out), "--device", "cpu", "--bins-km", "10", "--min-future-s", "10",
    ]) == 0

    payload = json.loads((out / "anytime_curve.json").read_text())
    assert payload["checkpoints"]["fixed"]["arm"] == runner.ARM_FIXED
    assert payload["checkpoints"]["fixed"]["random_train_anchor"] is False
    assert payload["checkpoints"]["random"]["arm"] == runner.ARM_RANDOM
    assert payload["checkpoints"]["random"]["random_train_anchor"] is True
    assert "arm" not in payload                       # never a run-level claim

    text = (out / "anytime_curve.txt").read_text()
    assert f"fixed [{runner.ARM_FIXED}]" in text and f"random [{runner.ARM_RANDOM}]" in text
    assert f"{runner.ARM_FIXED}: the checkpoint was trained at L-1" in text
    assert f"{runner.ARM_RANDOM}: the checkpoint was trained with random anchors" in text
    assert "This run holds BOTH arms" in text


def test_a_limited_run_says_so_and_counts_its_own_denominators(
    monkeypatch, tmp_path, trained_checkpoint
) -> None:
    """MEDIUM 7: `--limit` is a smoke test, and the coverage is of what was BUILT."""
    flights, checkpoint = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    out = tmp_path / "smoke"
    assert runner.main([
        "--checkpoint", f"tiny={checkpoint}", "--out", str(out), "--device", "cpu",
        "--bins-km", "10", "--min-future-s", "10", "--limit", "2",
    ]) == 0

    payload = json.loads((out / "anytime_curve.json").read_text())
    arm = payload["checkpoints"]["tiny"]
    assert payload["grid"]["limit"] == 2
    assert arm["flights"] == 2 and arm["split_flights"] >= 2
    assert arm["stratum_n_at_l1"][STRATUM_ALL] == 2
    assert arm["bins"]["10000.0"]["strata"][STRATUM_ALL]["coverage"] == pytest.approx(1.0)
    assert "SMOKE TEST" in (out / "anytime_curve.txt").read_text()


def test_the_command_hook_is_passed_through_like_predict(
    monkeypatch, tmp_path, trained_checkpoint
) -> None:
    """LOW 21: the adopted deployment form is `predict --command-hook barrier`, and a curve
    of the deployed predictor has to be able to fly the same way."""
    _flights, checkpoint = trained_checkpoint
    grid = runner.Grid(split="val", bins_m=(10_000.0,), min_future_s=10.0,
                       batch_size=None, limit=0)
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(runner.pipeline, "arrival_manifest_path", lambda _airport: manifest)
    monkeypatch.setattr(
        runner, "checkpoint_data_provenance", lambda _payload, _manifests: fake_data_provenance()
    )

    plain = runner.load_arm("plain", checkpoint, grid, torch.device("cpu"))
    assert plain.config.control_command_hook == "off"

    # This tiny recipe is point-mass, and the hook lives on the lag dynamics: TSConfig
    # refuses it with the reason rather than flying a hook that reads no actuator state.
    with pytest.raises(ValueError, match="first-order-lag"):
        runner.load_arm("hooked", checkpoint, grid, torch.device("cpu"),
                        command_hook="barrier", hook_saturation="soft")


def test_the_hook_flags_come_as_a_pair_and_only_on_a_control_checkpoint(
    tmp_path, capsys, trained_checkpoint
) -> None:
    _flights, checkpoint = trained_checkpoint
    with pytest.raises(SystemExit):
        runner.main(["--checkpoint", f"a={checkpoint}", "--out", str(tmp_path / "never"),
                     "--device", "cpu", "--command-hook", "barrier"])
    assert "given together" in capsys.readouterr().err
    assert not (tmp_path / "never").exists()


def test_a_failed_measurement_leaves_no_artifact_under_the_name_a_reader_cites(
    monkeypatch, tmp_path, trained_checkpoint
) -> None:
    """LOW 15: the directory appears whole. A crash mid-measurement leaves a `.partial-*`
    directory beside it, never a half-written curve under `--out`."""
    flights, checkpoint = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)

    def explode(*args, **kwargs):
        raise RuntimeError("the rollout died")

    monkeypatch.setattr(runner, "measure_bin", explode)
    out = tmp_path / "never"
    with pytest.raises(RuntimeError, match="the rollout died"):
        runner.main(["--checkpoint", f"tiny={checkpoint}", "--out", str(out),
                     "--device", "cpu", "--bins-km", "10", "--min-future-s", "10"])
    assert not out.exists()
    assert not list(tmp_path.glob("never.partial-*"))     # nothing was staged either


def test_the_output_directory_is_immutable(monkeypatch, tmp_path, trained_checkpoint) -> None:
    flights, checkpoint = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    out = tmp_path / "once"
    argv = ["--checkpoint", f"tiny={checkpoint}", "--out", str(out), "--device", "cpu",
            "--bins-km", "10", "--min-future-s", "10"]
    assert runner.main(argv) == 0
    with pytest.raises(FileExistsError):
        runner.main(argv)
    assert not list(tmp_path.glob("once.partial-*"))      # and the refusal staged nothing
