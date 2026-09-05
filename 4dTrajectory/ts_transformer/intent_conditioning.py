"""Phase 0 of the scene / join-anchor design: the intent, read from the FUTURE, as input.

The km-level error of the control model sits on vectored flights, and the design doc
(``docs/2026-09-07_scene_join_anchor_design.zh.md``) argues it is the ATC join decision —
how far down the downwind the aircraft is turned onto the final — that the model cannot
see. Before building a scene encoder to infer that decision, this module measures what
knowing it is WORTH: the truth join point and the lead aircraft's true landing time are
appended to the observed history as input-only constant channels, through the same
covariate-token path ``target_conditioning`` uses, and one simple-v3 arm is trained with
them. The gap to the baseline on the vectored stratum is the upper bound any inferred
intent can reach.

    e_join, n_join, u_join   the chart position of the first observed row from which the
                             track stays inside the final-approach cone
                             (``final_approach_geometry.truth_final_gate``, evaluated on ALL
                             observed rows, lookback included — the readouts evaluate the
                             same gate on the post-anchor rows only, so for a flight already
                             established at the anchor this point lies up to a lookback
                             earlier than the readout's ``gate_start_d_m``), standardised
                             with the POSITION channels' statistics like the target
                             conditioning; a track that never establishes carries the target
                             itself (joined at the threshold)
    lead_eta                 the previous same-runway landing's time relative to the anchor,
                             ``(t_lead_landing − t_anchor) / LEAD_ETA_SCALE_S``, clipped to
                             ±``LEAD_ETA_CLIP_S`` (negative: the runway has been clear that
                             long; the negative clip when the roster has no earlier landing
                             on that runway)

These are TRUTH values — both lie in the future of the anchor on a vectored flight, and
the lead's landing time is what the scene encoder would have to estimate from its current
state. A checkpoint trained with them is a development measurement, never a deployable
predictor; the run name carries ``intent=truth-…`` so no table can quote it as one. The
leak red line of the design (neighbour features only from ``t ≤ t₀``) is what Phase 1
enforces; here it is deliberately crossed, once, to size the prize.

Not to be confused with ``control/oracle/*``: that is the inverse-dynamics TEACHER
supplying control TARGETS; this module supplies INPUT covariates.

The lead is looked up in the tracks roster (every ``assigned`` landing on the runway,
including arrivals the model-ready roster excludes — a landing the arrival filter dropped
still occupied the runway), by the roster and never by globbing.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import torch

from channels import POSITION_IDX
# The mode and channel names live in config (beside ``input_channels``) because
# final_approach_geometry imports config: defining them here would close a cycle.
from config import INTENT_CONDITIONING_TRUTH_JOIN_LEAD, intent_channel_names
from final_approach_geometry import runway_axes, truth_final_gate

if TYPE_CHECKING:  # a data-plane value type; avoids a dataset <-> intent import cycle
    from dataset import FlightSeries

# One input scale for the lead ETA: the horizon's own order of magnitude (the full
# horizon is 600 s), so a lead landing one horizon away reads as ±1. The clip is where
# the roster stops carrying information about the queue — at KRDU/KSJC 95 % of leads
# land within ±1800 s of the anchor, and a runway clear for half an hour is simply clear.
LEAD_ETA_SCALE_S = 600.0
LEAD_ETA_CLIP_S = 1800.0

# Mirror of the harvest's outcome literal (trajectory_data_process/harvest/arrivals.py
# selects the roster with ``row["outcome"] != "assigned"``; it names no constant).
TRACK_OUTCOME_ASSIGNED = "assigned"


@dataclass(frozen=True)
class LeadLanding:
    """The previous landing on the flight's runway, as the tracks roster records it.

    ``landing_time_utc`` is ``None`` when the roster has no earlier landing on that runway
    (the first of the harvest). Distinct from the roster never having been consulted —
    ``FlightSeries.lead_landing is None``, a series built outside
    ``dataset.load_flight_dicts`` — which the lead channel refuses rather than reading as
    a clear runway.
    """

    landing_time_utc: str | None


# ── The join point ───────────────────────────────────────────────────────────

def truth_join_index(
    positions: np.ndarray, target_chart: np.ndarray, runway_heading_rad: float
) -> int | None:
    """Index of the first row from which the track stays on the final.

    ``positions[N, 3]`` are chart coordinates; the gate is evaluated on ``(d, xt)`` about
    the target along the runway course — chart east/north against the WORLD course, which
    is final_approach_geometry's assumption (the ``enu`` and ``airport-enu`` charts; a
    runway-aligned chart is already rotated, and ``TSConfig`` refuses that pairing).
    ``None`` when no row opens the gate (the track never establishes before the last
    300 m).
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be [N, 3], got shape {positions.shape}")
    relative = torch.as_tensor(positions - np.asarray(target_chart), dtype=torch.float64)
    psi = torch.tensor([float(runway_heading_rad)], dtype=torch.float64)
    d, xt = runway_axes(relative[None, :, 0], relative[None, :, 1], psi)
    gate = truth_final_gate(d, xt, torch.ones_like(d, dtype=torch.bool))[0]
    opened = np.flatnonzero(gate.numpy())
    return int(opened[0]) if len(opened) else None


def truth_join_point(series: "FlightSeries") -> np.ndarray:
    """Chart ``(e, n, u)`` where the observed track (all rows, lookback included) joins the
    final; the target when it never does (no final to speak of: joined at the threshold)."""
    positions = np.asarray(series.values[:, list(POSITION_IDX)], dtype=np.float64)
    index = truth_join_index(
        positions, series.target_chart, float(series.scenario.target.psi)
    )
    if index is None:
        return np.asarray(series.target_chart, dtype=np.float64)
    return positions[index]


# ── The lead aircraft ────────────────────────────────────────────────────────

def parse_utc(text: str) -> float:
    """POSIX seconds of a harvest ``...Z`` timestamp (the harvest CLI's own idiom)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def lead_landings(
    manifest: dict[str, Any], *, manifest_path: str | Path, flight_keys: Iterable[str]
) -> dict[str, LeadLanding]:
    """``flight_key -> LeadLanding`` for the requested arrivals of one parsed manifest.

    Read from the manifest's source tracks roster: every ``assigned`` track's landing, on
    every runway, whether or not it is a model-ready arrival. The lead is the latest
    landing strictly before the flight's own on the same runway.
    """
    # Mirror of load_arrival_flights' resolution of the tracks roster (relative to the
    # arrival manifest's directory); the loader does not expose the path it resolved.
    source_path = (Path(manifest_path).parent / manifest["source_manifest"]).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    # Per runway, landings sorted once as (seconds, text) pairs; the bisect runs on the
    # seconds and the same index reads the text.
    landings: dict[str, list[tuple[float, str]]] = {}
    for row in source["records"]:
        if row["outcome"] != TRACK_OUTCOME_ASSIGNED:
            continue
        landings.setdefault(row["runway"], []).append(
            (parse_utc(row["landing_time_utc"]), row["landing_time_utc"])
        )
    for rows in landings.values():
        rows.sort()
    seconds_by_runway = {
        runway: [seconds for seconds, _text in rows] for runway, rows in landings.items()
    }
    wanted = set(flight_keys)
    leads: dict[str, LeadLanding] = {}
    for row in manifest["records"]:
        key = row["flight_key"]
        if key not in wanted:
            continue
        runway = row["runway"]
        before = bisect_left(
            seconds_by_runway.get(runway, []), parse_utc(row["landing_time_utc"])
        )
        leads[key] = LeadLanding(landings[runway][before - 1][1] if before else None)
    return leads


def lead_eta_s(series: "FlightSeries", *, anchor_time_s: float) -> float:
    """``t_lead_landing − t_anchor`` in seconds, clipped to ±``LEAD_ETA_CLIP_S``.

    ``anchor_time_s`` is the anchor's time on the series' own clock (``series.times``,
    zero at the first observed sample), whose absolute time is the arrival record's
    ``entry_time_utc`` (measured on the KRDU and KSJC manifests: within 2 s of the first
    sample, p50 ≈ 0.9 s and always early — nothing against ``LEAD_ETA_SCALE_S``). No lead
    in the roster reads as the negative clip — the runway has been clear for as long as
    the channel can express.
    """
    if series.lead_landing is None:
        raise ValueError(
            f"flight {series.flight_id}: no scene context — the lead channel needs series "
            "loaded through dataset.load_flight_dicts, which consults the tracks roster"
        )
    if series.lead_landing.landing_time_utc is None:
        return -LEAD_ETA_CLIP_S
    entry = series.scenario.source.get("entry_time_utc")
    if not entry:
        raise ValueError(
            f"flight {series.flight_id}: lead ETA needs the arrival record's entry_time_utc"
        )
    anchor = parse_utc(entry) + anchor_time_s
    eta = parse_utc(series.lead_landing.landing_time_utc) - anchor
    return float(np.clip(eta, -LEAD_ETA_CLIP_S, LEAD_ETA_CLIP_S))


# ── The conditioning row ─────────────────────────────────────────────────────

def intent_vector(
    series: "FlightSeries",
    intent_conditioning: str,
    *,
    anchor_time_s: float,
    position_mean: np.ndarray,
    position_std: np.ndarray,
) -> np.ndarray | None:
    """One flight's constant intent row in the model's normalized input space, or
    ``None`` when the mode is off. Column order is :func:`config.intent_channel_names`."""
    names = intent_channel_names(intent_conditioning)
    if not names:
        return None
    join = (truth_join_point(series) - np.asarray(position_mean)) / np.asarray(position_std)
    parts = [join]
    if intent_conditioning == INTENT_CONDITIONING_TRUTH_JOIN_LEAD:
        parts.append([lead_eta_s(series, anchor_time_s=anchor_time_s) / LEAD_ETA_SCALE_S])
    row = np.concatenate(parts).astype(np.float32)
    if len(row) != len(names):
        raise RuntimeError(
            f"intent row has {len(row)} columns for channels {names}: the row builder "
            "and the channel contract disagree"
        )
    return row
