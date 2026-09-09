"""The latent sample fan read by the B line's chamfer-to-nearest-leaf protocol.

The geometry is planted, so every number the readout prints is known in closed form: each
path is a straight north-bound segment of the same length, offset from the truth by a fixed
east distance, so its chamfer to the truth IS that offset. What is under test is the join,
the strata, the nearest-leaf reduction, the random control and the endpoint spread — not
the chamfer, which `geometric_metrics` owns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ts_transformer.approach_difficulty import (
    STRATUM_ALL,
    STRATUM_ESTABLISHED,
    STRATUM_FAR,
    STRATUM_NEAR,
    STRATUM_STRAIGHT_IN,
    STRATUM_VECTORED,
)
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon
from run_ts_latent_fan_readout import TOP1_REFERENCE_KEY, main, readout

TARGET = {"lat": 35.0, "lon": -78.0, "alt": 100.0,
          "V": 75.0, "psi": 0.0, "gamma": 0.0, "m": 66_300.0}
PATH_LENGTH_M = 1_000.0
TRUE_FINAL_TIME_S = 100.0

# Two flights, planted into different strata, each with its own set of east offsets:
# the top-1 decode, the six prior modes, and the six N(0, I) control modes. Flight A is
# the case the fan wins (a mode at 100 m against a top-1 at 1000 m, control far away);
# flight B is the case it loses (every mode further than the top-1, one control nearer).
FLIGHTS = {
    "AAA111": {
        "runway": "05L", "icao24": "a00001", "landing_time_utc": "2026-05-01T00:00:00Z",
        "route_tortuosity": 1.0, "established_at_anchor": True, "remaining_path_m": 5_000.0,
        "top1": 1_000.0,
        "modes": [100.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0, 6_000.0],
        "random": [5_000.0] * 6,
    },
    "BBB222": {
        "runway": "23R", "icao24": "a00002", "landing_time_utc": "2026-05-02T00:00:00Z",
        "route_tortuosity": 2.0, "established_at_anchor": False, "remaining_path_m": 20_000.0,
        "top1": 300.0,
        "modes": [700.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0, 6_000.0],
        "random": [200.0, 5_000.0, 5_000.0, 5_000.0, 5_000.0, 5_000.0],
    },
}


def _lat_lon(east_m: float, north_m: float) -> tuple[float, float]:
    """The chart's own inverse: `geometric_metrics.chart_en` is linear about the target."""
    return (TARGET["lat"] + north_m / METRES_PER_DEG_LAT,
            TARGET["lon"] + east_m / metres_per_deg_lon(TARGET["lat"]))


def _segment(east_m: float, *, points: int = 11) -> list[dict]:
    """A north-bound path from ``PATH_LENGTH_M`` to the threshold latitude, held ``east_m``
    east of the truth, arriving at ``TRUE_FINAL_TIME_S``."""
    rows = []
    for step in range(points):
        fraction = step / (points - 1)
        lat, lon = _lat_lon(east_m, PATH_LENGTH_M * (1.0 - fraction))
        rows.append({"t": TRUE_FINAL_TIME_S * fraction, "lat": lat, "lon": lon,
                     "alt": TARGET["alt"] + 300.0 * (1.0 - fraction),
                     "V": 75.0, "psi": 0.0, "gamma": 0.0, "m": 66_300.0})
    return rows


def _write_leaf(directory: Path, offsets: dict[str, float]) -> None:
    """One prediction directory: a summary rostering both flights, and their records."""
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for flight_id, spec in FLIGHTS.items():
        stem = f"{flight_id}_{spec['runway']}"
        (directory / f"{stem}_states.json").write_text(json.dumps({
            "observed_states": _segment(0.0),          # the truth, identical in every leaf
            "predicted_states": _segment(offsets[flight_id]),
            "final_time_s": TRUE_FINAL_TIME_S,
            "source": {"id": flight_id},
        }))
        (directory / f"{stem}_eval.json").write_text(json.dumps({
            "target_state": TARGET, "initial_state": TARGET,
            "final_time_s": TRUE_FINAL_TIME_S, "states": [], "controls": [],
            "source": {"id": flight_id},
        }))
        results.append({
            "id": flight_id, "runway": spec["runway"], "icao24": spec["icao24"],
            "landing_time_utc": spec["landing_time_utc"],
            "states_file": f"{stem}_states.json", "eval_file": f"{stem}_eval.json",
            "true_final_time_s": TRUE_FINAL_TIME_S,
            # The planted ADE is the same east offset, so the ADE cross-check columns are
            # readable in closed form beside the chamfer ones.
            "ade_m": offsets[flight_id], "fde_m": offsets[flight_id],
            "route_tortuosity": spec["route_tortuosity"],
            "established_at_anchor": spec["established_at_anchor"],
            "remaining_path_m": spec["remaining_path_m"],
        })
    (directory / "summary.json").write_text(json.dumps({"split": "val", "results": results}))


def _arm(tmp_path: Path, *, random_leaves: bool = True) -> Path:
    arm = tmp_path / "arm"
    _write_leaf(arm, {key: spec["top1"] for key, spec in FLIGHTS.items()})
    for index in range(6):
        _write_leaf(arm / "modes" / f"mode{index:02d}",
                    {key: spec["modes"][index] for key, spec in FLIGHTS.items()})
        if random_leaves:
            _write_leaf(arm / "random" / f"mode{index:02d}",
                        {key: spec["random"][index] for key, spec in FLIGHTS.items()})
    return arm


def test_the_planted_fan_reads_out_exactly(tmp_path: Path):
    payload = readout(_arm(tmp_path), geometry_truth="closed")
    assert payload["flights"] == 2
    assert (payload["modes"], payload["random_modes"]) == (6, 6)

    # Each path's chamfer to the truth IS its east offset (parallel segments of equal
    # length), so every column below is the planted geometry read back.
    straight = payload["strata"][STRATUM_STRAIGHT_IN]
    assert straight["n"] == 1
    assert straight["prior_fan"][TOP1_REFERENCE_KEY] == pytest.approx(1_000.0, abs=1.0)
    assert straight["prior_fan"]["chamfer_nearest_p50_m"] == pytest.approx(100.0, abs=1.0)
    assert straight["prior_fan"]["nearest_better_share"] == 1.0
    # The control's nearest leaf is 5 km out: further than the top-1 it would replace.
    assert straight["random_fan"]["chamfer_nearest_p50_m"] == pytest.approx(5_000.0, abs=5.0)
    assert straight["random_fan"]["nearest_better_share"] == 0.0
    assert straight["top1_ade_mean_m"] == 1_000.0
    assert straight["min_ade_mean_m"] == 100.0
    # The spread is the largest gap between two of the six endpoints: 6000 - 100.
    assert straight["endpoint_spread_p50_m"] == pytest.approx(5_900.0, abs=5.0)

    # ...and the flight the fan loses on: every mode further out than the top-1, while one
    # random draw lands nearer. A fan readout that could not report this would be useless.
    vectored = payload["strata"][STRATUM_VECTORED]
    assert vectored["prior_fan"][TOP1_REFERENCE_KEY] == pytest.approx(300.0, abs=1.0)
    assert vectored["prior_fan"]["chamfer_nearest_p50_m"] == pytest.approx(700.0, abs=1.0)
    assert vectored["prior_fan"]["nearest_better_share"] == 0.0
    assert vectored["random_fan"]["chamfer_nearest_p50_m"] == pytest.approx(200.0, abs=1.0)
    assert vectored["random_fan"]["nearest_better_share"] == 1.0
    assert vectored["min_ade_mean_m"] == 300.0          # top-1 wins: minADE includes it

    # The pooled stratum is the median over both flights, and the strata are the shared ones.
    pooled = payload["strata"][STRATUM_ALL]
    assert pooled["n"] == 2
    assert pooled["prior_fan"][TOP1_REFERENCE_KEY] == pytest.approx(650.0, abs=1.0)
    assert pooled["prior_fan"]["chamfer_nearest_p50_m"] == pytest.approx(400.0, abs=1.0)
    assert pooled["prior_fan"]["nearest_better_share"] == 0.5
    assert pooled["random_fan"]["chamfer_nearest_p50_m"] == pytest.approx(2_600.0, abs=5.0)
    assert pooled["random_fan"]["nearest_better_share"] == 0.5
    assert pooled["top1_ade_mean_m"] == 650.0
    assert pooled["min_ade_mean_m"] == 200.0           # (100 + 300) / 2
    assert payload["strata"][STRATUM_ESTABLISHED]["n"] == 1
    assert payload["strata"][STRATUM_NEAR]["n"] == 1
    assert payload["strata"][STRATUM_FAR]["n"] == 1


def test_the_verdict_is_the_comparison_against_the_random_control(tmp_path: Path, capsys):
    assert main(["--arm", str(_arm(tmp_path)), "--json", str(tmp_path / "fan.json")]) == 0
    text = capsys.readouterr().out
    # Pooled: nearer than the control (400 vs 2600 m) but winning on no more flights
    # (0.5 vs 0.5) — a tie is not a win, and the sentence says which half held.
    assert "beats the random control on the chamfer only" in text
    assert "not a coverage guarantee" in text
    payload = json.loads((tmp_path / "fan.json").read_text())
    assert payload["schema"] == "ts-latent-fan-readout-v1"
    assert payload["split"] == "val"
    # A readout is an immutable artifact.
    with pytest.raises(FileExistsError):
        main(["--arm", str(_arm(tmp_path)), "--json", str(tmp_path / "fan.json")])


def test_a_fan_without_a_control_says_so_instead_of_inventing_one(tmp_path: Path, capsys):
    arm = _arm(tmp_path, random_leaves=False)
    payload = readout(arm, geometry_truth="closed")
    assert payload["random_modes"] == 0
    assert payload["strata"][STRATUM_ALL]["random_fan"] is None
    assert main(["--arm", str(arm)]) == 0
    assert "no random/ leaves" in capsys.readouterr().out


def test_a_directory_without_modes_is_refused(tmp_path: Path):
    """The top-1 records alone are not a fan; the readout refuses rather than reporting
    a one-leaf 'distribution'."""
    plain = tmp_path / "plain"
    _write_leaf(plain, {key: spec["top1"] for key, spec in FLIGHTS.items()})
    with pytest.raises(SystemExit, match="no modes/ leaves"):
        readout(plain, geometry_truth="closed")


def test_the_endpoint_spread_is_the_widest_pair():
    """The three pairs are 5, 10 and 9.85 m apart: the spread is the widest of them, not
    the distance from the first leaf to the furthest."""
    from run_ts_latent_fan_readout import endpoint_spread_m

    endpoints = np.array([[[0.0, 0.0]], [[3.0, 4.0]], [[-6.0, 8.0]]])   # [K=3, F=1, 2]
    assert endpoint_spread_m(endpoints)[0] == pytest.approx(10.0)
    assert endpoint_spread_m(endpoints[:1])[0] == 0.0
