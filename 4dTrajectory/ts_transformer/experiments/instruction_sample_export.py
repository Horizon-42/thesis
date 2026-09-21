"""Export a few flights' sentences from a BOX vocabulary artefact for the frontend's Training view
(design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`).

    python run_ts.py instruction_sample_export --vocabulary <…/vocabulary_box_v3_five_airports> \\
        --out aeroviz-4d/public/data/airports/KRDU/training/<set> [--prior <…/prior_s1337>] \\
        [--flights 40] [--split val] [--set-id box_v3] [--title "…"]

**A WORD IS AN INTERVAL** (`box-v2-wedge`, 2026-09-21): a sentence is a chain of bounding boxes
and the criterion is CONTAINMENT — every row of the track has to lie inside the boxes in force at
its moment. There is no flown track in this file and there cannot be one yet: flying a box
sentence needs a height-tracking executor (the design's replay gate, not built). So what this
exports is the track, the boxes, and the verdict — which is exactly what a box vocabulary is
checked by.

**It re-reads nothing.** The sentences come from the artefact's own ``sentences_<split>.json``;
only the TRACK is rebuilt, straight from the airport's arrival manifest. Re-running the labeller
here would let the view drift from the artefact it claims to show.

**THE PRODUCER OF THE ARTEFACT IS NOT IN THIS REPOSITORY** (2026-09-21). The box labeller was run
outside the tree, so nothing here can import its constants; this module reads the artefact's own
``spec`` block and derives the boxes from it. That makes the derivation below a MIRROR with no
original to check against, and the check that stands in for one is the containment verdict: the
boxes are rebuilt here from the spec and every row is measured against them. Over 75 flights at
five airports the reconstruction contains 13,922 of 13,922 rows, which is what says the reading is
the labeller's. Any drift shows up as rows outside a box that the artefact accepted.

The three signals the boxes judge are SMOOTHED, and the smoothing is the artefact's own
(``course_smoothing_s`` 6 s, ``smoothing_s`` 10 s). The raw rows are exported beside them so the
smoothing is visible rather than hidden — a chart that showed only the smoothed line would be
showing a signal nobody flew.

Written under ``--out`` (refused if it exists), plus an entry in the parent's ``index.json``:

    <out>/sample.json    the vocabulary's spec, the reading's own rule, and per flight the
                         sentence (events × 6 words + the hold), the observed track IN THE RUNWAY
                         FRAME, the three signals the boxes judge, the ENVELOPE those boxes make
                         and the containment verdict
    ../index.json        the manifest the frontend lists; this set added or replaced in place

Altitudes leave here as HAE: a record is MSL, Cesium reads the ellipsoid, and the conversion
belongs at this boundary (`hae_offset_m`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from flight_scenarios.datum import geoid_undulation_m

from ts_transformer.config import TSConfig, default_anchor
from ts_transformer.data.dataset import build_series, load_flight_dicts
from ts_transformer.repo_layout import arrival_manifest_path
from ts_transformer.data.approach_difficulty import (
    STRAIGHT_TORTUOSITY, STRATUM_SHORT, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, approach_difficulty,
)
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre.instructions import (
    VOCABULARY_FILE, course_frame, min_rows, smooth, wrap_deg,
)
from ts_transformer.manoeuvre.segments import MINIMUM_GROUND_SPEED_MPS


#: MIRROR of `src/data/trainingSample.ts` (`TRAINING_INDEX_SCHEMA` / `TRAINING_SAMPLE_SCHEMA`).
#: The reader refuses anything else by name, so these move together or not at all.
INDEX_SCHEMA = "aeroviz-training-index-v1"
#: v2 because the file is a different object: no flown track, an envelope instead.
SAMPLE_SCHEMA = "aeroviz-training-sample-v2"
#: The set kinds the frontend knows (`TRAINING_SET_KINDS` there).
KIND_READBACK = "vocabulary-readback"
#: A set that also carries what the PRIOR said at every event. The kind is what the reader keys
#: the prior block's presence on: a `vocabulary-readback` set must not have one and a
#: `prior-generated` set must, so neither is read leniently.
KIND_PRIOR = "prior-generated"

INDEX_FILE = "index.json"
SAMPLE_FILE = "sample.json"

#: The artefact this module reads, refused by name. Its producer is outside the tree.
BOX_SCHEMA = "ts-box-vocabulary-v1"
READING_RULE = "box-v2-wedge"
#: MIRROR of the artefact's `spec.kinds`, checked against it on load. The `words` columns are
#: POSITIONAL, so this order is load-bearing — and the second kind is an ALTITUDE (a target
#: height), not the retired vertical angle.
KINDS = ("heading", "altitude", "speed", "runway", "duration", "terminal")
#: The three kinds that are intervals. Runway is the frame, duration is the hold, terminal is a
#: label; none of the three bounds a signal, so none of them has a box.
BOX_KINDS = ("heading", "altitude", "speed")

#: How close to a box edge still counts as inside, in each kind's own unit. It exists because the
#: columns are written at display precision and the edge tables at full precision: without it a
#: row sitting exactly on an edge (measured: one row in 13,922) reads as a violation of 1e-14 m.
#: MIRROR of `TRAINING_INSIDE_EPSILON`; the verdict is computed on both sides and refused if the
#: two disagree, so the same number has to be used here and there.
INSIDE_EPSILON = 1e-3

#: How many candidates are rebuilt per flight wanted. The stratum is a property of the TRACK
#: (`approach_difficulty` at the executor's anchor), so the draw has to rebuild before it can
#: stratify; this is the margin that makes both strata fill on a cohort whose mix is unknown.
POOL_FACTOR = 5


# ── the artefact ─────────────────────────────────────────────────────────────

def load_box_vocabulary(path: Path) -> dict[str, Any]:
    """The box vocabulary, refused by schema, by reading rule and by its own table lengths.

    The lengths are the check that matters: `heading_deg` tiles the 65 heading words, so it holds
    66 edges, while `altitude_m` is a ladder of 61 TARGETS and holds exactly 61. Reading one as
    the other would silently shift every word by half a box.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != BOX_SCHEMA:
        raise SystemExit(f"{path} has schema {payload.get('schema')!r}, not {BOX_SCHEMA!r}")
    spec = payload["spec"]
    if spec["reading_rule"] != READING_RULE:
        raise SystemExit(
            f"{path} was read under {spec['reading_rule']!r}, and this exporter is written for "
            f"{READING_RULE!r} — the rule decides what a word MEANS"
        )
    if tuple(spec["kinds"]) != KINDS:
        raise SystemExit(f"{path} names kinds {tuple(spec['kinds'])}, not {KINDS} — the columns are positional")
    words = payload["words"]
    edges = payload["edges"]
    for kind, table, wanted in (("heading", "heading_deg", words["heading"] + 1),
                                ("speed", "speed_mps", words["speed"] + 1),
                                ("altitude", "altitude_m", words["altitude"])):
        if len(edges[table]) != wanted:
            raise SystemExit(
                f"{path}: edges.{table} holds {len(edges[table])} values, but {words[kind]} "
                f"{kind} words need {wanted} — an edge table tiles its words, a target ladder does not"
            )
    return payload


def runway_sha256(idents: list[str]) -> str:
    """A digest of the runway CLASSES, carried and checked wherever the spec's sha is.

    The classes sit outside the spec so one vocabulary serves five airports — which also means two
    artefacts with the SAME spec sha can disagree about what word 1 means.
    """
    import hashlib
    return hashlib.sha256(json.dumps(list(idents), sort_keys=True).encode()).hexdigest()


def sentences_by_flight(artefact: Path, split: str, sha256: str) -> dict[str, dict[str, Any]]:
    """The artefact's sentences for one split, keyed by ``flight_id``; refused if they were read
    under a different vocabulary than the spec beside them."""
    path = artefact / f"sentences_{split}.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: --split {split} is not in this artefact")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["vocabulary_sha256"] != sha256:
        raise SystemExit(
            f"{path} was read under vocabulary {payload['vocabulary_sha256'][:12]}…, "
            f"but the artefact's spec is {sha256[:12]}… — these are not one vocabulary"
        )
    return {item["flight_id"]: item for item in payload["flights"]}


# ── the draw ─────────────────────────────────────────────────────────────────

def drawn_flights(artefact: dict[str, Any], per_stratum: int, seed: int) -> list[str]:
    """The candidate pool, in draw order: a seeded permutation of the split's flights.

    The stratum is NOT decided here, because it cannot be: it is a property of the rebuilt track.
    This returns the order to rebuild in, and `stratify` takes the first `per_stratum` of each.
    """
    payload_ids = sorted(artefact)      # the sentence map: sorted for determinism
    if len(payload_ids) < per_stratum * 2:
        raise SystemExit(f"the split holds {len(payload_ids)} flights, fewer than the {per_stratum * 2} asked for")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(payload_ids))
    return [payload_ids[int(i)] for i in order[:min(len(payload_ids), per_stratum * 2 * POOL_FACTOR)]]


def stratify(candidates: list[str], series: list[Any], config: TSConfig,
             per_stratum: int) -> list[tuple[str, str, Any]]:
    """``[(flight_id, stratum, series)]``: the first ``per_stratum`` of each stratum, in the draw
    order above.

    A stratum that cannot be filled from the pool is REFUSED. Quietly returning 31 straight-in and
    9 vectored would publish a set whose stratum column says something the draw does not.
    """
    anchor = default_anchor(config)
    taken: dict[str, list[tuple[str, str, Any]]] = {STRATUM_STRAIGHT_IN: [], STRATUM_VECTORED: []}
    for flight_id, item in zip(candidates, series, strict=True):
        difficulty = approach_difficulty(item, anchor)
        if difficulty.route_tortuosity < STRAIGHT_TORTUOSITY:
            stratum = STRATUM_STRAIGHT_IN
        elif not difficulty.established_at_anchor:
            stratum = STRATUM_VECTORED
        else:
            continue
        if len(taken[stratum]) < per_stratum:
            taken[stratum].append((flight_id, STRATUM_SHORT[stratum], item))
    short = {STRATUM_SHORT[name]: len(rows) for name, rows in taken.items() if len(rows) < per_stratum}
    if short:
        raise SystemExit(
            f"the draw filled {short} of the {per_stratum} per stratum asked for, out of "
            f"{len(candidates)} rebuilt candidates — raise --flights' pool or lower --flights"
        )
    return [row for name in sorted(taken) for row in taken[name]]


def series_by_dataset_id(airport: str, dataset_ids: list[str], config: TSConfig):
    """The drawn flights' tracks, rebuilt STRAIGHT FROM the airport's arrival manifest — the same
    door the vocabulary runner reads through, for a handful of named flights instead of a cohort."""
    manifest = arrival_manifest_path(airport)
    built, report = build_series(
        load_flight_dicts([manifest], include_flight_keys=set(dataset_ids), verbose=False),
        config, aircraft_type=config.aircraft_type,
    )
    print(f"  {report.format()}", flush=True)
    by_id = {item.dataset_id: item for item in built}
    missing = [key for key in dataset_ids if key not in by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(dataset_ids)} drawn flights could not be rebuilt from "
            f"{manifest} (first {missing[0]!r}) — a readout over a silent subset is a different draw"
        )
    return [by_id[key] for key in dataset_ids], manifest


# ── the track, and the signals the boxes judge ───────────────────────────────

def _round(values, digits: int) -> list[float]:
    """A column at display precision. `INSIDE_EPSILON` is what keeps this rounding from turning a
    row that sits on an edge into a violation."""
    return [round(float(value), digits) for value in np.asarray(values, dtype=np.float64)]


def cumulative_path_m(times: np.ndarray, ground_speed_mps: np.ndarray) -> np.ndarray:
    """Cumulative horizontal distance along the track — the axis the ALTITUDE wedge is measured
    on (§2.4: ``r`` is remaining path length, NOT the projection on the course; the projection
    grows on a downwind leg and would read the remaining distance as negative).

    Each step is the row's own speed times the gap before it, floored at
    `MINIMUM_GROUND_SPEED_MPS`. The speed handed in is the SMOOTHED one, and which of the two it
    is was measured rather than assumed: over 75 flights the smoothed axis reproduces the
    artefact's segments to 5e-6 m, the raw one to 3e-4 m. Neither is a visible difference, but
    only one of them is what the labeller did, and the wedge is the one place a reconstruction
    can drift without looking wrong.
    """
    step = np.maximum(np.asarray(ground_speed_mps)[1:], MINIMUM_GROUND_SPEED_MPS) * np.diff(times)
    return np.concatenate([[0.0], np.cumsum(step)])


def read_signals(frame: dict[str, np.ndarray], spec: dict[str, Any]) -> dict[str, Any]:
    """The three signals the boxes are checked against, and the path axis the wedge rides on.

    WHICH SIGNAL, EXACTLY, is the whole game — a box is 2° wide at the course, so reading the raw
    course instead of the smoothed one puts 7 % of the rows outside a box the artefact accepted.
    Measured over 75 flights at five airports, these four lines put 13,922 of 13,922 rows inside:

      * the course is the WRAPPED relative ground track, smoothed over `course_smoothing_s` and
        wrapped again — smoothed wrapped, not smoothed unwrapped. The difference is only visible
        near ±180°, where a moving average across the cut averages +179° and −179° to 0°; that is
        the labeller's own arithmetic and this reproduces it rather than improving on it, because
        a view that disagreed with the artefact would be showing a different reading.
      * speed and height are smoothed over `smoothing_s`.
      * the path axis is integrated from the SMOOTHED ground speed.
    """
    times = frame["t"]
    dt = float(np.median(np.diff(times)))
    course_rows = min_rows(spec["course_smoothing_s"], dt)
    signal_rows = min_rows(spec["smoothing_s"], dt)
    return {
        "dt_s": dt,
        "course_window_rows": course_rows,
        "signal_window_rows": signal_rows,
        "course_deg": wrap_deg(smooth(frame["relative_course_deg"], course_rows)),
        "speed_mps": smooth(frame["ground_speed_mps"], signal_rows),
        "height_m": smooth(frame["height_m"], signal_rows),
        "path_m": cumulative_path_m(times, smooth(frame["ground_speed_mps"], signal_rows)),
    }


def lonlat_from_frame(series, to_go_m, cross_m) -> tuple[np.ndarray, np.ndarray]:
    """Course-frame horizontal coordinates back to (lon, lat).

    `course_frame_rows`' own algebra read backwards: with ``to_go = -(e·cosψ + n·sinψ)`` and
    ``cross = e·sinψ - n·cosψ``, the offsets from the threshold are ``e = -to_go·cosψ +
    cross·sinψ`` and ``n = -to_go·sinψ - cross·cosψ``.
    """
    psi = float(series.scenario.target.psi)
    cosine, sine = math.cos(psi), math.sin(psi)
    to_go = np.asarray(to_go_m, dtype=np.float64)
    cross = np.asarray(cross_m, dtype=np.float64)
    east = -to_go * cosine + cross * sine
    north = -to_go * sine - cross * cosine
    lons, lats = [], []
    for east_m, north_m in zip(east, north):
        first, second = series.frame.from_world_horizontal(float(east_m), float(north_m))
        lat, lon = series.frame.latlon_from_horizontal(
            series.target_chart[0] + first, series.target_chart[1] + second)
        lons.append(lon)
        lats.append(lat)
    return np.asarray(lons), np.asarray(lats)


def hae_offset_m(series, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """What to ADD to a height above the threshold to get HAE, per row.

    THE VERTICAL DATUM IS CONVERTED HERE, on the way out, exactly as the CZML exporter does it: a
    record's altitude is MSL (observed ADS-B is ellipsoidal and is converted once at the
    `flight_scenarios` seam), while Cesium reads `cartographicDegrees` as metres above the WGS84
    ELLIPSOID. Since h = H + N and N is NEGATIVE here (−33.5 m at KRDU), a line handed the MSL
    number renders |N| too HIGH — above its own terrain.

    It is an OFFSET rather than a converted column because the envelope has four more height
    columns over the same ground track; recomputing the geodesy per column would be the same
    inverse four times, and the five could drift.

    The chart's vertical axis is measured from the FRAME's anchor, not from sea level
    (`channels.py`: u = altitude − frame.alt0), so absolute MSL is the anchor's elevation plus the
    threshold's height above it plus the height above the threshold. Leaving the anchor out put
    touchdown at HAE −4 m.
    """
    msl_offset = float(series.frame.alt0) + float(series.target_chart[2])
    return msl_offset + np.asarray(geoid_undulation_m(lats, lons), dtype=np.float64)


def observed_track(frame: dict[str, np.ndarray], signals: dict[str, Any], series) -> dict[str, Any]:
    """The flight in the FINAL APPROACH COURSE's frame — the frame the words were read in — with
    the geodetic columns the 3D layer needs and the three signals the boxes judge."""
    lons, lats = lonlat_from_frame(series, frame["to_go_m"], frame["cross_m"])
    offset = hae_offset_m(series, lats, lons)
    return {
        "lon": [round(float(v), 7) for v in lons],
        "lat": [round(float(v), 7) for v in lats],
        "altHaeM": _round(offset + frame["height_m"], 2),
        "haeOffsetM": _round(offset, 2),
        "tS": _round(frame["t"] - frame["t"][0], 1),
        "toGoM": _round(frame["to_go_m"], 1),
        "crossM": _round(frame["cross_m"], 1),
        "heightM": _round(frame["height_m"], 2),
        "pathM": _round(signals["path_m"], 1),
        "relCourseDeg": _round(frame["relative_course_deg"], 4),
        "groundSpeedMps": _round(frame["ground_speed_mps"], 4),
        "established": [int(value) for value in frame["established"]],
        # the three the boxes are measured against, at the same precision as the boxes
        "readCourseDeg": _round(signals["course_deg"], 4),
        "readSpeedMps": _round(signals["speed_mps"], 4),
        "readHeightM": _round(signals["height_m"], 4),
    }


# ── the boxes ────────────────────────────────────────────────────────────────

def word_runs(column: list[int]) -> list[tuple[int, int]]:
    """``[(first, last)]`` event indices holding the same word, last INCLUSIVE. The altitude
    segment is a run: the reader cut the profile into segments and every event inside one repeats
    its word, so the segment the wedge is anchored to is recovered by run-length, not by re-reading."""
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(column) + 1):
        if index == len(column) or column[index] != column[start]:
            runs.append((start, index - 1))
            start = index
    return runs


def altitude_envelope(event_times_s: np.ndarray, words: np.ndarray, t_s: np.ndarray,
                      path_m: np.ndarray, spec: dict[str, Any], targets: np.ndarray,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """The wedge, per row: ``T − r·tan(γ_up) − f(T) ≤ h ≤ T + r·tan(γ_down) + f(T)`` with
    ``f(T) = redundancy · (T + h0)`` and ``r`` the REMAINING PATH to the segment's end (§2.4).

    Three details are load-bearing, and each one was measured against the artefact:

      * the segment's end is the NEXT ALTITUDE EVENT'S INSTANT, not the last row before it. Taking
        the last row instead leaves three rows in 2,670 outside a box the artefact accepted.
      * ``r`` is path length, not the projection on the course.
      * the last segment ends at the track's last row: there is no event after it, and the words
        in force run to the threshold.

    The wedge is ASYMMETRIC by design (γ_down 1.5° above, γ_up 1.0° below): it is the set the
    target is backward-reachable from, and low is the dangerous side.
    """
    column = words[:, KINDS.index("altitude")]
    frac, h0 = spec["redundancy_fraction"], spec["altitude_h0_m"]
    tan_down = math.tan(math.radians(spec["altitude_down_deg"]))
    tan_up = math.tan(math.radians(spec["altitude_up_deg"]))
    low = np.full(len(t_s), np.nan)
    high = np.full(len(t_s), np.nan)
    runs = word_runs([int(v) for v in column])
    for position, (first, last) in enumerate(runs):
        opens_s = float(event_times_s[first])
        final = position == len(runs) - 1
        closes_s = float(t_s[-1]) if final else float(event_times_s[last + 1])
        rows = np.where((t_s >= opens_s - 1e-9) & ((t_s <= closes_s + 1e-9) if final else (t_s < closes_s)))[0]
        if not len(rows):
            continue
        target = float(targets[int(column[first])])
        floor = frac * (target + h0)
        remaining = np.maximum(float(np.interp(closes_s, t_s, path_m)) - path_m[rows], 0.0)
        low[rows] = target - remaining * tan_up - floor
        high[rows] = target + remaining * tan_down + floor
    if np.isnan(low).any():
        missed = int(np.isnan(low).sum())
        raise SystemExit(
            f"{missed} of {len(t_s)} rows fall in no altitude segment — the events are supposed to "
            f"tile the track, so a row with no box in force means the sentence and the track disagree"
        )
    return low, high


#: How finely the sector's arc is drawn: one point per degree of its own opening, between these
#: two. A 2° box at the course needs 3 points and an 18° one at the reciprocal needs 16 — a
#: constant would either facet the wide ones or spend sixteen points on a sliver.
SECTOR_ARC_MIN = 3
SECTOR_ARC_MAX = 16


def reachable_sector(heading_lo_deg: float, heading_hi_deg: float, speed_hi: float,
                     hold_s: float) -> tuple[list[float], list[float]]:
    """``(to_go_offsets, cross_offsets)``: where ONE word lets the aircraft be, relative to where
    it was when that word opened — **a circular sector, not a box**.

    The words bound the STATE: the heading word holds the ground track's angle inside
    ``[ψ_lo, ψ_hi]`` and the speed word holds its rate inside ``[v_lo, v_hi]``, at every instant.
    The positions that follow over a hold of T seconds are

        { ∫₀^τ v(s)·(cos ψ(s), sin ψ(s)) ds : τ ∈ [0,T], v(s) ∈ [v_lo, v_hi], ψ(s) ∈ [ψ_lo, ψ_hi] }

    and that set is the **pie slice** of radius ``T · v_hi`` spanning the heading box, apex at the
    aircraft: the integral of a set-valued map over [0, τ] is ``τ · conv(V)``, and sweeping τ from
    0 to T scales that hull all the way down to the apex.

    **The speed word's LOWER edge does not bound it at all.** It says where the aircraft is at the
    END of the hold (no nearer than ``T · v_lo``), but at every earlier instant it is nearer
    still, so the swept region runs back to the apex. Only ``v_hi`` sets the radius.

    THE BOUNDING BOX OF THIS IS NOT IT, and drawing one was wrong (2026-09-21, caught by the
    user): a rectangle puts flyable-looking volume where the words allow none — the corners beside
    the apex, which the aircraft cannot reach without having turned outside its heading box. At a
    2° box the rectangle is about twice the slice's area, at a 9° one worse, and it hides the one
    thing the shape exists to show: that the region FANS OUT FROM THE AIRCRAFT.

    The frame's signs are `course_frame_rows`' own: ``to_go`` counts DOWN toward the threshold and
    ``cross`` is positive to the right, so a displacement at relative course ψ over a distance d
    is ``(-d·cos ψ, -d·sin ψ)``.

    IT IS A DERIVED SET, NOT THE WORD, and it carries no aircraft in it: nothing here limits how
    fast the heading may swing inside its box, because the word does not. A turn rate belongs to
    an executor, and there is no executor in this file.
    """
    radius = max(hold_s, 0.0) * speed_hi
    low = math.radians(heading_lo_deg)
    high = math.radians(heading_hi_deg)
    arc = min(max(int(math.ceil(heading_hi_deg - heading_lo_deg)), SECTOR_ARC_MIN), SECTOR_ARC_MAX)
    to_go = [0.0]
    cross = [0.0]
    for index in range(arc):
        angle = low + (high - low) * index / (arc - 1)
        to_go.append(-radius * math.cos(angle))
        cross.append(-radius * math.sin(angle))
    return to_go, cross


def event_boxes(sentence: dict[str, Any], observed: dict[str, np.ndarray], low_m: np.ndarray,
                high_m: np.ndarray, vocabulary: dict[str, Any], series) -> list[dict[str, Any]]:
    """One box per event: the three intervals, the wedge over the event's own span, and the ground
    SECTOR the words allow the aircraft to be in while that word stands.

    A word is a box in STATE space — an interval of heading, one of speed, one of altitude — and
    that is what the vocabulary means by a bounding box. What it makes in POSITION space is not a
    box: it is a pie slice fanning out from the aircraft (`reachable_sector`). The two are named
    and drawn differently for that reason.

    The sector is built in the course frame and comes out as (lon, lat) as well, because the
    frame's transform lives on this side of the wire — the frontend has lon/lat columns and no way
    to place a point at an arbitrary (to_go, cross).
    """
    edges = vocabulary["edges"]
    heading_edges = np.asarray(edges["heading_deg"], dtype=np.float64)
    speed_edges = np.asarray(edges["speed_mps"], dtype=np.float64)
    targets = np.asarray(edges["altitude_m"], dtype=np.float64)
    columns = {kind: KINDS.index(kind) for kind in KINDS}

    times = observed["t_s"]
    words = np.asarray(sentence["words"], dtype=np.int64)
    event_times = np.asarray(sentence["event_times_s"], dtype=np.float64)
    holds = np.asarray(sentence["hold_s"], dtype=np.float64)

    corners_to_go: list[float] = []
    corners_cross: list[float] = []
    spans: list[int] = []                  # how many points each sector drew
    rows_of: list[np.ndarray] = []
    boxes: list[dict[str, Any]] = []
    for index in range(len(event_times)):
        opens_s = float(event_times[index])
        closes_s = opens_s + float(holds[index])
        rows = np.where((times >= opens_s - 1e-9) & (times <= closes_s + 1e-9))[0]
        if not len(rows):
            rows = np.asarray([int(np.searchsorted(times, opens_s, side="right")) - 1])
        rows_of.append(rows)
        heading_word = int(words[index, columns["heading"]])
        speed_word = int(words[index, columns["speed"]])
        heading_lo, heading_hi = heading_edges[heading_word], heading_edges[heading_word + 1]
        speed_lo, speed_hi = speed_edges[speed_word], speed_edges[speed_word + 1]
        offsets_to_go, offsets_cross = reachable_sector(
            heading_lo, heading_hi, speed_hi, float(holds[index]))
        anchor_to_go = float(observed["to_go_m"][rows[0]])
        anchor_cross = float(observed["cross_m"][rows[0]])
        sector_to_go = [anchor_to_go + value for value in offsets_to_go]
        sector_cross = [anchor_cross + value for value in offsets_cross]
        spans.append(len(sector_to_go))
        corners_to_go.extend(sector_to_go)
        corners_cross.extend(sector_cross)
        boxes.append({
            "eventS": round(opens_s, 1),
            "holdS": round(float(holds[index]), 1),
            # the same sector in the COURSE FRAME, which is the plan view's own axes. The FIRST
            # point is the apex — the aircraft's own position when this word opened.
            "toGoM": [round(float(v), 1) for v in sector_to_go],
            "crossM": [round(float(v), 1) for v in sector_cross],
            "headingLoDeg": round(float(heading_lo), 4),
            "headingHiDeg": round(float(heading_hi), 4),
            "speedLoMps": round(float(speed_lo), 4),
            "speedHiMps": round(float(speed_hi), 4),
            "altitudeTargetM": round(float(targets[int(words[index, columns["altitude"]])]), 2),
            # The wedge at the instant the box OPENS, which is the widest it gets while this word
            # stands (`r` only shrinks along a segment). So the prism's height is an OUTER BOUND
            # over the hold rather than the wedge at any one moment inside it.
            "altLoM": round(float(low_m[rows].min()), 4),
            "altHiM": round(float(high_m[rows].max()), 4),
        })

    lons, lats = lonlat_from_frame(series, corners_to_go, corners_cross)
    offset = hae_offset_m(series, lats, lons)
    cursor = 0
    for index, box in enumerate(boxes):
        window = slice(cursor, cursor + spans[index])
        cursor += spans[index]
        # the box's own corners sit metres from the track, so the geoid offset at the corners is
        # the offset at the rows; taking it here rather than at the rows keeps one conversion.
        base = float(offset[window].mean())
        box["lon"] = [round(float(v), 7) for v in lons[window]]
        box["lat"] = [round(float(v), 7) for v in lats[window]]
        box["altHaeLoM"] = round(base + box["altLoM"], 2)
        box["altHaeHiM"] = round(base + box["altHiM"], 2)
    return boxes


def containment(read: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict[str, int]:
    """How many of the rows this box holds, and how many it does not. `INSIDE_EPSILON` is the
    same number the reader uses, because the two verdicts are compared."""
    outside = int(((read < low - INSIDE_EPSILON) | (read > high + INSIDE_EPSILON)).sum())
    return {"rows": int(len(read)), "outside": outside}


def envelope_block(sentence: dict[str, Any], observed_columns: dict[str, np.ndarray],
                   signals: dict[str, Any], vocabulary: dict[str, Any], series) -> dict[str, Any]:
    """The boxes one sentence makes over one track, and the verdict on every row.

    `inside` is the CRITERION of this vocabulary (§1: a sentence holds if the track is inside its
    boxes), and it is computed here on the rounded columns the file carries so that the reader
    recomputing it gets the same answer to the row.
    """
    spec = vocabulary["spec"]
    targets = np.asarray(vocabulary["edges"]["altitude_m"], dtype=np.float64)
    heading_edges = np.asarray(vocabulary["edges"]["heading_deg"], dtype=np.float64)
    speed_edges = np.asarray(vocabulary["edges"]["speed_mps"], dtype=np.float64)
    times = observed_columns["t_s"]
    words = np.asarray(sentence["words"], dtype=np.int64)
    event_times = np.asarray(sentence["event_times_s"], dtype=np.float64)

    low, high = altitude_envelope(event_times, words, times, signals["path_m"], spec, targets)
    low = np.round(low, 4)
    high = np.round(high, 4)
    boxes = event_boxes(sentence, observed_columns, low, high, vocabulary, series)

    # the event in force at each row: the last one that has opened. The events tile the track, so
    # every row has exactly one.
    in_force = np.clip(np.searchsorted(event_times, times, side="right") - 1, 0, len(event_times) - 1)
    heading_word = words[in_force, KINDS.index("heading")]
    speed_word = words[in_force, KINDS.index("speed")]
    verdict = {
        "heading": containment(np.round(signals["course_deg"], 4),
                               heading_edges[heading_word], heading_edges[heading_word + 1]),
        "altitude": containment(np.round(signals["height_m"], 4), low, high),
        "speed": containment(np.round(signals["speed_mps"], 4),
                             speed_edges[speed_word], speed_edges[speed_word + 1]),
    }
    offset = observed_columns["hae_offset_m"]
    return {
        "altLoM": [round(float(v), 4) for v in low],
        "altHiM": [round(float(v), 4) for v in high],
        "altHaeLoM": _round(offset + low, 2),
        "altHaeHiM": _round(offset + high, 2),
        "events": boxes,
        "inside": verdict,
    }


# ── the prior ────────────────────────────────────────────────────────────────

PRIOR_FILE = "prior.pt"
#: MIRROR of `instruction_prior.PRIOR_SCHEMA`.
PRIOR_SCHEMA = "ts-instruction-prior-v1"
#: What the prior was asked, in the file, because it is NOT what a reader assumes. At every event
#: the model saw the TRUTH's words and the TRUTH's state up to that point and said what the next
#: event would be. It did not generate the sentence.
PRIOR_METHOD = "teacher-forced-next-word"
TERMINAL_LANDED = 1


def load_prior(directory: Path, vocabulary_sha256: str):
    """The trained prior, refused unless it was trained on THIS vocabulary. The sha is the point:
    the heads are one per kind and sized by the class counts, so a prior from another vocabulary
    would still load, still run, and still emit word indices — into another vocabulary's classes."""
    import torch
    from ts_transformer.manoeuvre.context import TypeVocabulary
    from ts_transformer.manoeuvre.instruction_prior import InstructionPrior, PriorConfig

    path = directory / PRIOR_FILE
    if not path.exists():
        raise SystemExit(f"{path} is missing: --prior takes the directory a prior run wrote")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != PRIOR_SCHEMA:
        raise SystemExit(f"{path} has schema {payload.get('schema')!r}, not {PRIOR_SCHEMA!r}")
    if payload["vocabulary_sha256"] != vocabulary_sha256:
        raise SystemExit(
            f"{path} was trained on vocabulary {payload['vocabulary_sha256'][:12]}…, the artefact is "
            f"{vocabulary_sha256[:12]}… — the heads are sized by THAT vocabulary's classes"
        )
    config = PriorConfig.from_dict(payload["prior_config"])
    model = InstructionPrior(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, TypeVocabulary.from_dict(payload["types"]), payload


def prior_sentences(model, config, types, sentences: list[dict[str, Any]], series: list[Any]) -> list[dict[str, Any]]:
    """What the prior says at every event of each drawn flight, and how sure it was.

    Position k's logits are the model's answer for event k + 1, so the said sentence is the truth's
    OPENING EVENT followed by the model's answers at positions 0 … E-2; `givenEvents` says so.

    The event TIMES and the HOLDS stay the truth's. The duration word is predicted like every
    other kind and is carried, but it does not place the events: each prediction was conditioned
    on the truth's state at the truth's instant, so re-timing the sentence by the model's own gaps
    would put its words at moments its conditioning never saw.
    """
    import torch
    from ts_transformer.manoeuvre.instruction_prior import collate
    from ts_transformer.manoeuvre.instruction_sequences import flight_sequence
    from ts_transformer.manoeuvre.instructions import Reading

    sequences = []
    for sentence, item in zip(sentences, series, strict=True):
        sequences.append(flight_sequence(item, Reading(
            dataset_id=sentence["dataset_id"], flight_id=sentence["flight_id"], instructions=(),
            event_times_s=np.asarray(sentence["event_times_s"], dtype=np.float64),
            words=np.asarray(sentence["words"], dtype=np.int64), runway=sentence["runway"],
            established_from_start=False,
            duration_s=float(sentence["event_times_s"][-1] + sentence["hold_s"][-1]),
        )))
    with torch.no_grad():
        output = model(collate(sequences, config, types))

    said: list[dict[str, Any]] = []
    terminal_column = KINDS.index("terminal")
    for index, sequence in enumerate(sequences):
        length = sequence.length
        words = [[int(v) for v in sequence.words[0]]]
        confidence = [[1.0] * len(KINDS)]       # the opening event is given, not said
        for position in range(length - 1):
            row, sure = [], []
            for kind in KINDS:
                probabilities = torch.softmax(output.logits[kind][index, position], dim=-1)
                word = int(probabilities.argmax())
                row.append(word)
                sure.append(round(float(probabilities[word]), 4))
            words.append(row)
            confidence.append(sure)
        landed = next((float(sequence.positions_s[k]) for k in range(1, length)
                       if words[k][terminal_column] == TERMINAL_LANDED), None)
        said.append({"words": words, "confidence": confidence, "givenEvents": 1, "landedAtS": landed})
    return said


# ── one flight ───────────────────────────────────────────────────────────────

def flight_payload(sentence: dict[str, Any], series, stratum: str, vocabulary: dict[str, Any],
                   said: dict[str, Any] | None = None) -> dict[str, Any]:
    """One flight as the frontend reads it: the sentence copied UNCHANGED from the artefact, the
    track rebuilt from the manifest, and the envelope those two make together."""
    frame = course_frame(series)
    signals = read_signals(frame, vocabulary["spec"])
    times = frame["t"] - frame["t"][0]
    duration_s = float(times[-1])

    # The sentence's events TILE the track (§2.5: the hold is written on the row it describes), so
    # the last event plus its hold is the flight's length. This is the seam between a sentence
    # copied from the artefact and a track rebuilt here: nothing else says they are one flight.
    spoken_s = float(sentence["event_times_s"][-1] + sentence["hold_s"][-1])
    if abs(spoken_s - duration_s) > 2.0 * float(vocabulary["spec"]["duration_bin_s"]):
        raise SystemExit(
            f"{sentence['flight_id']}: the artefact's sentence covers {spoken_s:g} s but the rebuilt "
            f"track spans {duration_s:g} s — these are not the same flight"
        )

    observed = observed_track(frame, signals, series)
    columns = {
        "t_s": times, "to_go_m": frame["to_go_m"], "cross_m": frame["cross_m"],
        "hae_offset_m": np.asarray(observed["haeOffsetM"], dtype=np.float64),
    }
    own = {"event_times_s": sentence["event_times_s"], "hold_s": sentence["hold_s"],
           "words": sentence["words"]}
    payload = {
        "flightKey": sentence["flight_id"],
        "callsign": sentence["flight_id"].split("_", 1)[0],
        "runway": sentence["runway"],
        "stratum": stratum,
        "durationS": round(duration_s, 1),
        "dtS": round(signals["dt_s"], 3),
        "courseWindowRows": signals["course_window_rows"],
        "signalWindowRows": signals["signal_window_rows"],
        "sentence": {
            "eventTimesS": [float(v) for v in sentence["event_times_s"]],
            "holdS": [float(v) for v in sentence["hold_s"]],
            "words": [[int(v) for v in row] for row in sentence["words"]],
        },
        "observed": observed,
        "envelope": envelope_block(own, columns, signals, vocabulary, series),
    }
    if said is not None:
        model = {"event_times_s": sentence["event_times_s"], "hold_s": sentence["hold_s"],
                 "words": said["words"]}
        payload["prior"] = {
            **{key: said[key] for key in ("words", "confidence", "givenEvents", "landedAtS")},
            "envelope": envelope_block(model, columns, signals, vocabulary, series),
        }
    return payload


def update_index(directory: Path, airport: str, entry: dict[str, Any]) -> Path:
    """Add or replace this set in the parent's manifest, keeping every other set. A manifest of
    another schema is refused rather than rewritten — it is not ours to reinterpret."""
    path = directory / INDEX_FILE
    sets: list[dict[str, Any]] = []
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != INDEX_SCHEMA:
            raise SystemExit(f"{path} has schema {payload.get('schema')!r}, not {INDEX_SCHEMA!r}")
        if payload.get("airport") != airport:
            raise SystemExit(f"{path} is {payload.get('airport')!r}'s manifest, not {airport}'s")
        sets = [item for item in payload["sets"] if item.get("id") != entry["id"]]
    sets.append(entry)
    sets.sort(key=lambda item: item["id"])
    write_json_atomic(path, {"schema": INDEX_SCHEMA, "writtenUtc": utc_now(), "airport": airport, "sets": sets})
    return path


def reading_block() -> dict[str, Any]:
    """How a track became the signals the boxes judge — every line of `read_signals`, in the file.

    It is stated rather than assumed because the choice of signal decides the verdict: the same
    track against the same boxes is 100 % inside on these four lines and 93 % inside on the raw
    ones, and a reader who could not see which was used would be reading a number with no meaning.
    """
    return {
        "rule": READING_RULE,
        "courseSignal": "wrap(moving average of the wrapped relative ground track over courseSmoothingS)",
        "speedSignal": "moving average of the ground speed over smoothingS",
        "heightSignal": "moving average of the height above the threshold over smoothingS",
        "pathSignal": "the smoothed ground speed integrated, floored at MINIMUM_GROUND_SPEED_MPS",
        "remainingPathTo": "the next altitude event's instant; the track's last row for the final segment",
        "windowRows": "round(seconds / median dt) + 1, centred, edges padded with the edge value",
        "insideEpsilon": INSIDE_EPSILON,
        "producedBy": "ts_transformer.experiments.instruction_sample_export (the artefact's own labeller is NOT in this repository)",
        "constantsFrom": [
            "the artefact's spec block (redundancy, the wedge's angles, the ladder, the smoothing)",
            "ts_transformer/manoeuvre/instructions.py (course_frame, smooth, min_rows, wrap_deg)",
            "ts_transformer/manoeuvre/segments.py (MINIMUM_GROUND_SPEED_MPS)",
        ],
    }


def vocabulary_block(payload: dict[str, Any]) -> dict[str, Any]:
    spec = payload["spec"]
    edges = payload["edges"]
    return {
        "sha256": payload["sha256"],
        "runwaySha256": runway_sha256(payload["runway_idents"]),
        "readingRule": spec["reading_rule"],
        "redundancyFraction": spec["redundancy_fraction"],
        "headingEdgesDeg": list(edges["heading_deg"]),
        "headingFloorDeg": spec["heading_floor_deg"],
        "speedEdgesMps": list(edges["speed_mps"]),
        "altitudeTargetsM": list(edges["altitude_m"]),
        "altitudeH0M": spec["altitude_h0_m"],
        "altitudeDownDeg": spec["altitude_down_deg"],
        "altitudeUpDeg": spec["altitude_up_deg"],
        "altitudeForm": spec["altitude_form"],
        "altitudeReading": spec["altitude_reading"],
        "durationBinS": spec["duration_bin_s"],
        "durationMaxS": spec["duration_max_s"],
        "courseSmoothingS": spec["course_smoothing_s"],
        "smoothingS": spec["smoothing_s"],
        "runwayIdents": list(payload["runway_idents"]),
        "words": {kind: int(payload["words"][kind]) for kind in KINDS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--vocabulary", type=Path, required=True,
                        help="the artefact DIRECTORY (or its instruction_vocabulary.json)")
    parser.add_argument("--prior", type=Path, default=None,
                        help="a prior run's directory: adds what the model says at every event, and its boxes")
    parser.add_argument("--out", type=Path, required=True, help="…/airports/<ICAO>/training/<set id>")
    parser.add_argument("--flights", type=int, default=40, help="drawn half per stratum (default 40)")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--seed", type=int, default=1337, help="the draw's seed; it is written into the manifest")
    parser.add_argument("--set-id", default=None, help="defaults to the output directory's name")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    if args.flights % 2:
        parser.error(f"--flights draws half per stratum: {args.flights} is odd")
    artefact = args.vocabulary if args.vocabulary.is_absolute() else REPO_ROOT / args.vocabulary
    if artefact.name == VOCABULARY_FILE:
        artefact = artefact.parent
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a sample export is never overwritten")
    training = out.parent                          # …/airports/<ICAO>/training/<set> → training
    airport = training.parent.name                 # → <ICAO>
    set_id = args.set_id or out.name
    started = time.perf_counter()

    vocabulary = load_box_vocabulary(artefact / VOCABULARY_FILE)
    sentences = sentences_by_flight(artefact, args.split, vocabulary["sha256"])
    # A pooled artefact holds five airports' sentences; ONE TRAINING SET IS ONE AIRPORT'S, and the
    # airport is the one the --out path names. The runway word carries the prefix, so this is the
    # artefact's own answer to "whose flight is this", not a guess from the flight id.
    sentences = {key: row for key, row in sentences.items() if row["runway"].split(":")[0] == airport}
    if not sentences:
        raise SystemExit(f"no flight in sentences_{args.split}.json lands at {airport}")
    per_stratum = args.flights // 2
    candidates = drawn_flights(sentences, per_stratum, args.seed)

    config = TSConfig()
    print(f"  {len(candidates)} candidates drawn (seed {args.seed}); rebuilding their tracks to stratify", flush=True)
    rebuilt, manifest = series_by_dataset_id(
        airport, [sentences[flight_id]["dataset_id"] for flight_id in candidates], config)
    draw = stratify(candidates, rebuilt, config, per_stratum)
    print(f"  {len(draw)} flights drawn ({per_stratum} per stratum)", flush=True)

    said_by_flight: list[dict[str, Any] | None] = [None] * len(draw)
    prior_block = None
    if args.prior:
        prior_dir = args.prior if args.prior.is_absolute() else REPO_ROOT / args.prior
        model, prior_config, types, prior_payload = load_prior(prior_dir, vocabulary["sha256"])
        said_by_flight = prior_sentences(
            model, prior_config, types,
            [sentences[flight_id] for flight_id, _, _ in draw], [item for _, _, item in draw])
        print(f"  the prior said {sum(len(s['words']) for s in said_by_flight)} events over {len(draw)} flights", flush=True)
        trained_on = set(prior_payload["split"]["train"])
        drawn_ids = {sentences[flight_id]["dataset_id"] for flight_id, _, _ in draw}
        prior_block = {
            "sha256": file_sha256(prior_dir / PRIOR_FILE),
            "method": PRIOR_METHOD,
            "seed": prior_payload["settings"]["seed"],
            "bestEpoch": prior_payload["best_epoch"],
            "trainedOnTheseFlights": len(drawn_ids & trained_on),
            "readout": json.loads((prior_dir / "readings.json").read_text(encoding="utf-8"))["readings"],
        }

    flights = [flight_payload(sentences[flight_id], item, stratum, vocabulary, said)
               for (flight_id, stratum, item), said in zip(draw, said_by_flight, strict=True)]
    out.mkdir(parents=True)
    write_json_atomic(out / SAMPLE_FILE, {
        "schema": SAMPLE_SCHEMA, "setId": set_id, "airport": airport, "writtenUtc": utc_now(),
        "kinds": list(KINDS),
        "reading": reading_block(),
        **({"prior": prior_block} if prior_block is not None else {}),
        "vocabulary": vocabulary_block(vocabulary),
        "flights": flights,
    })

    outside = {kind: sum(flight["envelope"]["inside"][kind]["outside"] for flight in flights)
               for kind in BOX_KINDS}
    rows = sum(flight["envelope"]["inside"]["heading"]["rows"] for flight in flights)
    print(f"  containment over {rows} rows: " +
          ", ".join(f"{kind} {rows - outside[kind]}/{rows}" for kind in BOX_KINDS), flush=True)

    entry = {
        "id": set_id, "kind": KIND_PRIOR if prior_block is not None else KIND_READBACK,
        "title": args.title or f"Box vocabulary · {READING_RULE} · {args.split}",
        "file": f"{out.name}/{SAMPLE_FILE}",
        "vocabularySha256": vocabulary["sha256"],
        "runwaySha256": runway_sha256(vocabulary["runway_idents"]),
        "readingRule": READING_RULE, "flights": len(flights),
        "cohort": {"split": args.split, "perStratum": per_stratum, "seed": args.seed,
                   "drawnFrom": (f"a seeded permutation of the {args.split} split at {airport}, "
                                 f"stratified by approach_difficulty at the executor's anchor "
                                 f"(pool {len(candidates)})")},
        **({"prior": {"sha256": prior_block["sha256"], "seed": prior_block["seed"],
                      "method": prior_block["method"]}} if prior_block is not None else {}),
        "source": {"artefact": str(artefact), "manifest": str(manifest),
                   "manifestSha256": file_sha256(manifest)},
    }
    index = update_index(training, airport, entry)
    print(f"  wrote {out / SAMPLE_FILE} and {index} in {time.perf_counter() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
