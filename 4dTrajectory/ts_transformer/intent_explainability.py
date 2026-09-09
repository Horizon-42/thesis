"""How much of the join decision the observable context explains — the population and the
cross-validated fit behind the scene design's Phase 0 `context`/`timing` readings and the
latent-intent design's L4 acceptance (§六 L4).

One row per arrival of the airport's manifest, from the RAW harvest track: the truth join
distance ``d_join`` (the first row of the truth final gate), whether that gate had already
opened at the anchor, the raw-track duration and path length after the anchor, the ego's
anchor state, the COARSE causal traffic context Phase 0 used (counts and the time since
the last landing, from the rosters alone), and the TRUTH lead ETA (an upper-bound
instrument — it reads the lead's landing time). The row also carries the ego's identity
and anchor position/time so a richer context (the scene data plane) can be attached to
the same population and scored under the same protocol.

The fit is a 5-fold gradient boosting with a fixed seed; R² and the median absolute error
are reported against a constant-median baseline. Numbers to reproduce (KRDU, joins after
the anchor, n ≈ 6.6 k): d_join R² ego 0.34 → +coarse context 0.38 → +truth lead ETA 0.47;
remaining-duration median error from the ego 35.8 s → 22.8 s with the truth d_join.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import ts_transformer.final_approach_geometry as fag
import ts_transformer.geometric_metrics as gm
import ts_transformer.intent_conditioning as ic
from ts_transformer.config import DEFAULT_DT_S, DEFAULT_SEQ_LEN
from flight_scenarios.identity import flight_key
from geokit import compass_bearing_to_math_enu_rad
from trajectory_data_process.harvest.arrivals import load_arrival_flights

#: The model's anchor: the end of the lookback window, seconds after the arrival slice's entry.
ANCHOR_S = (DEFAULT_SEQ_LEN - 1) * DEFAULT_DT_S
CONTEXT_NAMES = ("since_last_landing_s", "airborne_same_runway", "airborne_other_runway",
                 "landings_last_30min", "hour_utc", "weekday", "runway")


def utc_s(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def arrivals_manifest(harvest_root: Path, airport: str) -> Path:
    return harvest_root / airport / "arrivals" / "manifest.json"


def _axes(xy: np.ndarray, psi: float) -> tuple[torch.Tensor, torch.Tensor]:
    return fag.runway_axes(
        torch.tensor(xy[:, 0])[None], torch.tensor(xy[:, 1])[None], torch.tensor([psi])
    )


def population(harvest_root: Path, airport: str, *, verbose: bool = True) -> list[dict[str, Any]]:
    """Per rostered arrival: the join truth, the ego anchor state, the coarse context, the
    truth lead ETA, and the identity + anchor position the scene query needs."""
    manifest_path = arrivals_manifest(harvest_root, airport)
    manifest = json.loads(manifest_path.read_text())
    tracks = json.loads((manifest_path.parent / manifest["source_manifest"]).resolve().read_text())
    flights = load_arrival_flights(manifest_path)
    targets = manifest["runway_targets"]
    landings = sorted(
        (utc_s(r["landing_time_utc"]), r["runway"]) for r in tracks["records"] if r["outcome"] == "assigned"
    )
    land_t = np.array([x[0] for x in landings])
    land_rw = np.array([x[1] for x in landings])
    entries = [(utc_s(r["entry_time_utc"]), utc_s(r["landing_time_utc"]), r["runway"]) for r in manifest["records"]]
    ent_e = np.array([x[0] for x in entries])
    ent_l = np.array([x[1] for x in entries])
    ent_rw = np.array([x[2] for x in entries])
    runways = sorted({r["runway"] for r in manifest["records"]})
    clip = ic.LEAD_ETA_CLIP_S
    rows: list[dict[str, Any]] = []
    dropped = 0
    for index, flight in enumerate(flights):
        runway = flight["runway"]
        target = targets[runway]
        psi = compass_bearing_to_math_enu_rad(math.radians(target["course_deg"]))
        wp = np.array(flight["waypoints"])
        e, n = gm.chart_en(wp[:, 2], wp[:, 1], target["lat"], target["lon"])
        d, xt = _axes(np.stack([e, n], 1), psi)
        gate = fag.truth_final_gate(d, xt, torch.ones_like(d, dtype=torch.bool))[0].numpy()
        opened = np.flatnonzero(gate)
        ia = int(np.searchsorted(wp[:, 0], ANCHOR_S))
        if not len(opened) or ia < 1 or ia >= len(wp) - 2:
            dropped += 1
            continue
        d, xt = d[0].numpy(), xt[0].numpy()
        t0 = utc_s(flight["entry_time_utc"]) + ANCHOR_S
        tl = utc_s(flight["landing_time_utc"])
        de, dn = e[ia + 1] - e[ia - 1], n[ia + 1] - n[ia - 1]
        speed = math.hypot(de, dn) / (wp[ia + 1, 0] - wp[ia - 1, 0])
        heading = math.atan2(dn, de)
        same = land_rw == runway
        earlier = land_t[same][land_t[same] < t0]
        since_last = (t0 - earlier[-1]) if len(earlier) else clip
        before_own = land_t[same][land_t[same] < tl]
        lead_eta_truth = (before_own[-1] - t0) if len(before_own) else -clip
        when = datetime.fromtimestamp(t0, tz=timezone.utc)
        rows.append({
            "flight_key": flight_key(flight, index),
            "runway": runway,
            "t0_utc_s": t0,
            "anchor_lat": float(wp[ia, 1]),
            "anchor_lon": float(wp[ia, 2]),
            # Raw harvest altitude is HAE: the height above the threshold uses its HAE elevation.
            "anchor_alt_hae_m": float(wp[ia, 3]),
            "anchor_ground_speed_mps": speed,
            "d_join": float(d[opened[0]]),
            "join_before_anchor": bool(opened[0] <= ia),
            "raw_duration_s": float(wp[-1, 0] - wp[ia, 0]),
            "raw_track_path_m": float(np.sum(np.hypot(np.diff(e[ia:]), np.diff(n[ia:])))),
            "ego": [float(d[ia]), float(xt[ia]), math.cos(heading - psi), math.sin(heading - psi), speed,
                    float(wp[ia, 3]) - target["elevation_hae_m"]],
            "context": [min(since_last, clip),
                        int(((ent_e <= t0) & (ent_l > t0) & (ent_rw == runway)).sum()) - 1,
                        int(((ent_e <= t0) & (ent_l > t0) & (ent_rw != runway)).sum()),
                        int(((land_t > t0 - 1800.0) & (land_t <= t0)).sum()),
                        when.hour + when.minute / 60.0, when.weekday(), runways.index(runway)],
            "lead": [float(np.clip(lead_eta_truth, -clip, clip))],
        })
    if verbose:
        print(f"{airport}: {len(rows)} arrivals with an open gate and a full anchor window; "
              f"{dropped} dropped (never established, or too short)")
    return rows


def boosting(n_columns: int, categorical: list[int]):
    from sklearn.ensemble import HistGradientBoostingRegressor

    mask = np.zeros(n_columns, dtype=bool)
    mask[categorical] = True
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=0, categorical_features=mask
    )


def cv_r2(X: np.ndarray, y: np.ndarray, categorical: list[int]) -> tuple[float, float]:
    """5-fold out-of-fold R² and median |error| of the boosting on (X, y)."""
    from sklearn.model_selection import KFold, cross_val_predict

    pred = cross_val_predict(
        boosting(X.shape[1], categorical), X, y, cv=KFold(5, shuffle=True, random_state=0)
    )
    return 1.0 - np.mean((pred - y) ** 2) / np.var(y), float(np.median(np.abs(pred - y)))
