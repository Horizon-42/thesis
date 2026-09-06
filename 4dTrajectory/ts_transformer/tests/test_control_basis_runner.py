"""The L0 runner's joins: three key spaces meet, and a stratum must follow its flight.

The cohort's readout key (`id_runway_icao24_landing`), the compact
`flight_scenarios.identity.flight_key` the dataset builds under, and the ORDER
`build_series` returns are three different things. The strata masks are aligned to the
cohort's order; the scored rows arrive in the series' order. Zipping the masks against the
wrong one is size-matched and silent, and it turns the gated stratum into a random subset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from approach_difficulty import STRATUM_ALL, STRATUM_VECTORED, strata_masks
from control.basis_fit import inverse_dynamics_seed
from flight_scenarios.identity import summary_row_key
from run_ts_control_basis_oracle import summarise


def _reference() -> dict[str, dict]:
    """Three flights whose readout keys sort into a different order than they are listed."""
    rows = [
        {"id": "ZZZ1", "runway": "05L", "icao24": "aaa001", "landing_time_utc": "2026-01-01T00:00:00Z",
         "route_tortuosity": 2.4, "established_at_anchor": False, "remaining_path_m": 30_000.0},
        {"id": "AAA2", "runway": "05L", "icao24": "aaa002", "landing_time_utc": "2026-01-01T00:10:00Z",
         "route_tortuosity": 1.01, "established_at_anchor": True, "remaining_path_m": 9_000.0},
        {"id": "MMM3", "runway": "23R", "icao24": "aaa003", "landing_time_utc": "2026-01-01T00:20:00Z",
         "route_tortuosity": 1.9, "established_at_anchor": False, "remaining_path_m": 25_000.0},
    ]
    return {summary_row_key(row): row for row in rows}


def _row(key: str, ade: float) -> dict:
    return {
        "flight_key": key, "ade_m": ade, "fde_m": ade, "fixed_dt_ade_m": ade,
        "seed_fixed_dt_ade_m": ade * 2, "chamfer_m": ade, "frechet_m": ade,
        "saturated_fraction": 0.0, "tail_gain": 0.0,
    }


def test_a_stratum_follows_its_flight_whatever_order_the_rows_arrive_in():
    reference = _reference()
    cohort_keys = sorted(reference)                     # what the masks are built against
    masks = strata_masks(reference, cohort_keys)
    vectored_keys = {key for key, member in zip(cohort_keys, masks[STRATUM_VECTORED]) if member}
    assert len(vectored_keys) == 2                      # the two tortuous, unestablished flights

    # The rows come back in the series' order, which is neither the cohort's nor stable.
    ades = {key: 100.0 * (index + 1) for index, key in enumerate(cohort_keys)}
    for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
        rows = [_row(cohort_keys[index], ades[cohort_keys[index]]) for index in order]
        strata = summarise(rows, masks, cohort_keys)
        assert strata[STRATUM_ALL]["n"] == 3
        assert strata[STRATUM_VECTORED]["n"] == 2
        assert strata[STRATUM_VECTORED]["ade_mean_m"] == pytest.approx(
            float(np.mean([ades[key] for key in vectored_keys]))
        )


def test_summarise_refuses_a_row_outside_the_cohort_the_masks_cover():
    reference = _reference()
    cohort_keys = sorted(reference)
    masks = strata_masks(reference, cohort_keys)
    with pytest.raises(RuntimeError, match="outside the cohort"):
        summarise([_row("not_a_cohort_key", 100.0)], masks, cohort_keys)


def test_summary_row_key_maps_a_null_field_to_the_empty_string():
    """A present-but-null field must read as absent, not as the string "None"."""
    assert summary_row_key({"id": "AAA", "runway": None, "icao24": "x", "landing_time_utc": "t"}) == (
        "AAA__x_t"
    )


@dataclass
class _Series:
    flight_id: str
    values: np.ndarray
    times: np.ndarray
    frame: object


def test_the_seed_refuses_a_batch_that_does_not_cover_the_same_flights():
    dynamics = {
        "control_lower": torch.zeros(2, 3),
        "control_upper": torch.ones(2, 3),
        "initial_state": torch.zeros(2, 7),
        "aero_params": torch.zeros(2, 4),
        "max_thrust_n": torch.ones(2),
    }
    series = [_Series("only-one", np.zeros((2, 6)), np.zeros(2), None)]
    with pytest.raises(ValueError, match="same flights"):
        inverse_dynamics_seed(
            series, 0, dynamics, config=None, n_segments=2, final_time_s=np.array([10.0])
        )


# ── the teacher table (--checkpoint mode, design §六 L5.a) ───────────────────

import json                                                        # noqa: E402

import run_ts_control_basis_oracle as runner                       # noqa: E402
from forecast import _control_prediction_batch                     # noqa: E402
from config import (                                               # noqa: E402
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
)
from control.basis_fit import (                                    # noqa: E402
    DURATION_UNIFORM,
    FITTED_TEACHER_SCHEMA,
    BasisSchedule,
    load_fitted_teacher,
)
from data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA  # noqa: E402
from dataset import (  # noqa: E402
    FixedAnchorTrajectoryWindows,
    Normalizer,
    build_series,
    dataset_flight_key,
    truth_duration_s,
)
from io_utils import file_sha256                                   # noqa: E402
from models import build_model                                     # noqa: E402
from synthetic import synthetic_arrivals                           # noqa: E402
from train import load_checkpoint, train                           # noqa: E402

AIRPORT, RUNWAY = "KRDU", "05L"


def _teacher_config(**overrides) -> TSConfig:
    """A tiny CPU control recipe; the teacher mode inherits everything from it."""
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


def _provenance() -> dict:
    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory):
    """One tiny control checkpoint on synthetic arrivals, plus the flights behind it."""
    torch.manual_seed(0)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    config = _teacher_config()
    series, _report = build_series(flights, config, airport=AIRPORT)
    out = tmp_path_factory.mktemp("teacher_run")
    train(series, config, output_dir=out, data_provenance=_provenance(), verbose=False)
    checkpoint = out / "checkpoint.pt"
    _model, _config, _normalizer, payload = load_checkpoint(checkpoint)
    assert payload["split"]["train"] and payload["split"]["val"]
    return flights, checkpoint, payload


def _patch_data_plane(monkeypatch, flights, tmp_path):
    """The runner's three data-plane seams, pointed at the synthetic flights."""
    manifest = tmp_path / "manifest.json"
    indexed = {dataset_flight_key(flight, index): flight
               for index, flight in enumerate(flights)}
    monkeypatch.setattr(runner.pipeline, "arrival_manifest_path", lambda _airport: manifest)
    monkeypatch.setattr(runner, "arrival_data_provenance", lambda _paths: _provenance())
    monkeypatch.setattr(
        runner, "load_flight_dicts",
        lambda _paths, include_flight_keys, verbose=True: [
            flight for key, flight in indexed.items() if key in include_flight_keys
        ],
    )


def test_the_teacher_table_carries_its_stamps_and_the_whole_cohort(
    monkeypatch, tmp_path, trained_checkpoint
):
    """End to end: the checkpoint's own splits, fitted at its own width, keyed by flight."""
    flights, checkpoint, payload = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    out = tmp_path / "l5_teacher"
    assert runner.main([
        "--checkpoint", str(checkpoint), "--out", str(out), "--steps", "2",
        "--batch-size", "8", "--device", "cpu",
    ]) == 0

    table = json.loads((out / "basis_fit.json").read_text())
    assert table["schema"] == FITTED_TEACHER_SCHEMA
    assert table["airports"] == [AIRPORT]
    assert table["n_segments"] == 4 and table["duration_mode"] == DURATION_UNIFORM
    assert table["anchor_index"] == 7                      # seq_len - 1
    assert table["checkpoint_sha256"] == file_sha256(checkpoint)
    assert len(table["config_sha256"]) == 64
    assert table["init"] == runner.INIT_NETWORK and table["splits"] == ["train", "val"]
    assert table["optimizer"] == {
        "steps": 2, "batch_size": 8,
        "control_learning_rate": 0.08, "duration_learning_rate": 0.08,
        "gradient_clip_norm": 20.0, "learning_rate_floor": 0.05,
        "seed": 0, "device": "cpu",
    }
    assert table["wall_time_s"] > 0.0

    wanted = payload["split"]["train"] + payload["split"]["val"]
    assert table["coverage"] == {
        "train": len(payload["split"]["train"]), "val": len(payload["split"]["val"])
    }
    assert set(table["flights"]) == {key.split(":", 1)[1] for key in wanted}
    for key in payload["split"]["val"]:
        assert table["flights"][key.split(":", 1)[1]]["split"] == "val"
    for entry in table["flights"].values():
        assert np.asarray(entry["controls"]).shape == (4, 3)
        assert entry["total_duration_s"] > 0.0
        # The fit starts AT the seed (step 0 is evaluated before any update), so it can
        # never come out worse than the schedule it was seeded from.
        assert entry["fit_ade_m"] <= entry["seed_ade_m"] + 1e-9
        assert 0 <= entry["best_step"] <= 2
    assert set(table["quantiles"]) == {"train", "val"}


def test_the_written_table_is_the_one_the_dataset_loader_accepts(
    monkeypatch, tmp_path, trained_checkpoint
):
    """The fitter's schema constant and the loader's are one constant, and the stamps
    the loader checks are the stamps the fitter writes — including the 1e-6 s duration."""
    flights, checkpoint, payload = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    out = tmp_path / "l5_teacher"
    runner.main(["--checkpoint", str(checkpoint), "--out", str(out), "--steps", "1",
                 "--device", "cpu"])

    table = load_fitted_teacher(out / "basis_fit.json")
    config = runner.teacher_config(load_checkpoint(checkpoint)[1], "cpu")
    series, _report = build_series(flights, config, airport=AIRPORT)
    anchor = config.seq_len - 1
    wanted = set(payload["split"]["train"] + payload["split"]["val"])
    covered = [
        (item.flight_id, truth_duration_s(item, anchor))
        for item in series if item.dataset_id in wanted
    ]
    table.require_cover(
        covered, airports={AIRPORT}, anchor_indices={anchor}, n_segments=config.n_segments
    )
    assert table.provenance["flights"] == len(covered)
    assert table.provenance["sha256"] == file_sha256(out / "basis_fit.json")


def test_the_network_seed_is_the_checkpoints_own_forward(tmp_path):
    """`--init network` takes the CONTROLS of the deterministic forward and nothing else:
    the durations stay the truth's, spread uniformly, as in the width study."""
    torch.manual_seed(0)
    config = _teacher_config(latent_dim=3)          # a latent model decodes its prior top-1
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=3, seed=3)
    series, _report = build_series(flights, config, airport=AIRPORT)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    device = torch.device("cpu")
    history, _final_time, dynamics, _supervision = runner.prepare_batch(
        windows, np.arange(len(series)), device
    )

    # The independent reference is the pass `predict` itself makes (per flight, through
    # `forecast`), not the batched call `network_seed` uses: the two must agree exactly, or
    # the teacher is not seeded from the schedule this checkpoint would fly.
    seed = runner.network_seed(model, history, dynamics, device)
    expected = _control_prediction_batch(
        model, history.numpy(), dynamics, device
    ).controls
    assert np.array_equal(seed, expected.numpy().astype(np.float64))

    anchor = config.seq_len - 1
    durations = np.array([truth_duration_s(item, anchor) for item in series])
    schedule = BasisSchedule(
        torch.tensor(seed, dtype=torch.float64),
        dynamics["control_lower"].to(torch.float64),
        dynamics["control_upper"].to(torch.float64),
        torch.tensor(durations, dtype=torch.float64),
        DURATION_UNIFORM,
    )
    prediction = schedule()
    assert torch.allclose(prediction.controls, torch.tensor(seed), atol=1e-9)
    assert torch.allclose(
        prediction.segment_durations,
        torch.tensor(durations / config.n_segments, dtype=torch.float64).unsqueeze(1)
        .expand(-1, config.n_segments),
        atol=1e-9,
    )


def test_the_inverse_dynamics_init_still_fits_a_table(
    monkeypatch, tmp_path, trained_checkpoint
):
    """The width study's own seed remains selectable, and it is a DIFFERENT starting point."""
    flights, checkpoint, _payload = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    network_out, inverse_out = tmp_path / "network", tmp_path / "inverse"
    runner.main(["--checkpoint", str(checkpoint), "--out", str(network_out), "--steps", "1",
                 "--device", "cpu"])
    runner.main(["--checkpoint", str(checkpoint), "--out", str(inverse_out), "--steps", "1",
                 "--device", "cpu", "--init", runner.INIT_INVERSE_DYNAMICS])

    network = json.loads((network_out / "basis_fit.json").read_text())
    inverse = json.loads((inverse_out / "basis_fit.json").read_text())
    assert inverse["init"] == runner.INIT_INVERSE_DYNAMICS
    assert set(network["flights"]) == set(inverse["flights"])
    key = next(iter(network["flights"]))
    assert network["flights"][key]["seed_ade_m"] != inverse["flights"][key]["seed_ade_m"]


def test_the_teacher_refuses_the_sealed_test_split(monkeypatch, tmp_path, trained_checkpoint):
    flights, checkpoint, _payload = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    with pytest.raises(SystemExit):
        runner.main(["--checkpoint", str(checkpoint), "--out", str(tmp_path / "sealed"),
                     "--device", "cpu", "--splits", "train,test"])
    assert not (tmp_path / "sealed").exists()   # the immutable dir is claimed after the refusals


def test_the_teacher_output_directory_is_immutable(monkeypatch, tmp_path, trained_checkpoint):
    flights, checkpoint, _payload = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    out = tmp_path / "once"
    runner.main(["--checkpoint", str(checkpoint), "--out", str(out), "--steps", "1",
                 "--device", "cpu"])
    with pytest.raises(FileExistsError):
        runner.main(["--checkpoint", str(checkpoint), "--out", str(out), "--steps", "1",
                 "--device", "cpu"])


def test_a_state_checkpoint_has_no_control_schedule_to_fit(monkeypatch, tmp_path):
    torch.manual_seed(0)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    config = TSConfig(prediction_output=PREDICTION_STATE, seq_len=8, n_segments=4,
                      d_model=16, n_heads=4, d_ff=32, e_layers=1, device="cpu",
                      epochs=1, patience=1, batch_size=8, dropout=0.0,
                      val_fraction=0.25, test_fraction=0.25)
    series, _report = build_series(flights, config, airport=AIRPORT)
    out = tmp_path / "state_run"
    train(series, config, output_dir=out, data_provenance=_provenance(), verbose=False)
    _patch_data_plane(monkeypatch, flights, tmp_path)
    with pytest.raises(SystemExit):
        runner.main(["--checkpoint", str(out / "checkpoint.pt"), "--out", str(tmp_path / "no"),
                     "--device", "cpu"])
    assert not (tmp_path / "no").exists()


@pytest.mark.parametrize("argv, message", [
    ([], "exactly one"),
    (["--reference", "ref", "--checkpoint", "ckpt"], "exactly one"),
    (["--checkpoint", "ckpt", "--segments", "8"], "--segments belongs"),
    (["--checkpoint", "ckpt", "--limit", "10"], "--limit belongs"),
    (["--reference", "ref", "--splits", "train"], "--splits belongs"),
    (["--reference", "ref", "--init", "network"], "--init belongs"),
])
def test_each_mode_refuses_the_other_mode_s_options(tmp_path, capsys, argv, message):
    with pytest.raises(SystemExit):
        runner.main([*argv, "--out", str(tmp_path / "never")])
    assert message in capsys.readouterr().err
    assert not (tmp_path / "never").exists()      # nothing is created before the refusal


def test_the_table_does_not_depend_on_the_order_the_cohort_arrived_in(
    monkeypatch, tmp_path, trained_checkpoint
):
    """The fit batches by DURATION, not by split order, so the input order is not observable.

    The padding of a batch's dense supervision is set by its longest flight, so a batch
    drawn in split order integrates far more flight-seconds than it needs. Sorting also
    makes the result independent of the order the checkpoint's splits happened to list.
    """
    flights, checkpoint, payload = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    forward, reversed_out = tmp_path / "forward", tmp_path / "reversed"
    runner.main(["--checkpoint", str(checkpoint), "--out", str(forward), "--steps", "2",
                 "--batch-size", "4", "--device", "cpu"])

    reversed_payload = dict(payload)
    reversed_payload["split"] = {
        name: list(reversed(keys)) for name, keys in payload["split"].items()
    }
    monkeypatch.setattr(
        runner, "load_checkpoint",
        lambda path: (*load_checkpoint(path)[:3], reversed_payload),
    )
    runner.main(["--checkpoint", str(checkpoint), "--out", str(reversed_out), "--steps", "2",
                 "--batch-size", "4", "--device", "cpu"])

    first = json.loads((forward / "basis_fit.json").read_text())["flights"]
    second = json.loads((reversed_out / "basis_fit.json").read_text())["flights"]
    assert first == second


def test_the_width_study_still_runs_end_to_end(monkeypatch, tmp_path, trained_checkpoint):
    """The ADE(N) mode shares the batch mechanics with the teacher and must keep working."""
    flights, _checkpoint, payload = trained_checkpoint
    _patch_data_plane(monkeypatch, flights, tmp_path)
    reference = tmp_path / "reference"
    reference.mkdir()
    rows = [{
        "id": flight["id"], "runway": flight["runway"], "icao24": flight["icao24"],
        "landing_time_utc": flight["landing_time_utc"],
        "ade_m": 100.0 + index, "route_tortuosity": 1.05 + 0.5 * (index % 2),
        "established_at_anchor": bool(index % 2), "remaining_path_m": 20_000.0,
    } for index, flight in enumerate(flights[:4])]
    (reference / "summary.json").write_text(json.dumps(
        {"config": payload["config"], "split": "val", "results": rows}
    ))

    out = tmp_path / "l0"
    assert runner.main([
        "--reference", str(reference), "--out", str(out), "--airport", AIRPORT,
        "--segments", "4", "--duration-modes", "uniform,free",
        "--steps", "2", "--batch-size", "4", "--device", "cpu",
    ]) == 0
    result = json.loads((out / "oracle_basis.json").read_text())
    assert result["schema"] == runner.RESULT_SCHEMA
    assert set(result["arms"]) == {"N=4 uniform", "N=4 free"}
    assert result["verdict"]["status"] in {"pass", "fail"}
    assert result["coverage"]["measured_flights"] == 4
    schedules = json.loads((out / "basis_fit.json").read_text())
    assert schedules["schema"] == runner.RESULT_SCHEMA        # NOT the teacher schema
    assert len(schedules["arms"]["N=4 uniform"]) == 4
    text = (out / "oracle_basis.txt").read_text()
    assert "gate:" in text and "N=4 uniform" in text
