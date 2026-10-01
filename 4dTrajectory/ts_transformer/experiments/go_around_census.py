"""Multi-aircraft step 8 (design §6.6 step 8 item 7): how long a real go-around takes to land again, on the TRAINING days.

The stored track of a landing is the aircraft's last contiguous stretch within 30 km of the field, ending on the landing's
ground run (harvest TD7), so a go-around flown back inside 30 km is in it together with the landing that followed. Every
`assigned` track of each airport's live tracks roster whose landing falls on a training day is read as the harvest's
derived view (`store.read_track_view`: altitude outliers repaired, harvest TD10 — a needle reads as a climb); a track of
any other day is never opened.

STRAY SAMPLES are set aside first. A stored track carries the odd run of another aircraft's samples — a position
kilometres off, with that aircraft's altitude (KRDU RPA5593's sample 570: 15 km away and 400 m up, between two samples on
the final) — which the altitude repair cannot see, since a run of three is the median of its own five-sample window. A
sample farther than `STRAY_DISTANCE_M` from the median position of the `STRAY_WINDOW` samples either side of it (east and
north taken apart, in a runway end's frame) is one; it is counted, and the rest of the track — distances included — is read
without it. Its limits (review 2026-10-01): a run of more than `STRAY_WINDOW` strays is its own median, a stray among the
first or last `STRAY_WINDOW` samples is padded with itself, and the window counts samples, not seconds — on 262 training-day
tracks it flagged 5 samples, every one a jump of 5–40 km, none on a turn or across a reception gap.

On the remaining samples before the landing sample (`landing_sample_index`):

- a LOW PASS is a run of samples (at most `MAX_SKIPPED_SAMPLES` between two of them) on one runway end's extended
  centreline — every runway end of the airport (`Airport.frames("hae")`, the harvest's datum), since a go-around may come
  back to another runway: at most `MAX_CROSS_M` off it, between `ALONG_RANGE_M` of its threshold along the landing
  direction and at most ``--max-height-m`` above it — that moves at least `MIN_ALONG_PROGRESS_M` along the landing
  direction between its first and last sample (flying the final, not crossing it). Passes of two runway ends whose sample
  ranges overlap (close parallels) are one pass, kept on the end with the smaller median |cross-track|. Its lowest sample
  is the GO-AROUND POINT;
- a pass is a GO-AROUND when the aircraft HELD a level at least `MIN_DESCENT_M` above that point for `HOLD_S` (over at
  least `MIN_HOLD_SAMPLES` samples) before it — it came down to it: a takeoff from the field starts on the runway — and
  held one at least `MIN_CLIMB_M` above it after it, before the next pass begins (or the landing sample): a dip and a
  shallow level-off on one final are not two go-arounds. Held, not reached, and held over several samples: a stray the
  window missed, or two across a reception gap, are not a climb.

Two kinds of what passes these rules are set aside, counted, and left out of the times:

- a point ON THE RUNWAY — past the threshold and at most `ON_RUNWAY_HEIGHT_M` above it, under every published threshold
  crossing height (15.3–18.1 m, harvest TD9): a touch-and-go (training circuits), or a landing balked in the flare;
- every go-around of a flight whose stored landing is NOT ITS LAST — after its landing sample it holds a level at least
  `MIN_CLIMB_M` higher — its time to "landing" is not to a landing. Such landings are counted whether or not a go-around
  precedes them, with their flight keys, in two kinds: LANDED LATER, the track ending inside a low pass after that climb
  (back on a final: at some airports reception ends on short final, not on the runway) — a low go-around the harvest took
  for the landing (it takes the best-aligned crossing under 100 m, not the last one, `harvest/threshold_event.py`), so
  the flight's arrival slice and sentence end at a go-around — with the time from the stored landing to the track's end
  (about that go-around's: the track ends on the ground run or on short final); and LEFT, every other end (a touch-and-go
  and away, training circuits).

Each go-around gives the time from its point to the landing sample, the time the aborted approach still needed at its
point (the distance left to the threshold at the reported ground speed there), their difference (what the go-around
cost), how far from the field the aircraft flew before landing (how close the loop came to the 30 km edge), and where the
flight stands on the two-tier line: in the arrival roster or not, whether its arrival slice (`first_sample_index`)
contains the go-around point, and in the instruction artefact's train split — labelled (and whether the sentence's rows
contain the go-around point), refused, or absent. A go-around whose loop left 30 km, or whose aircraft did not land here,
is not in any stored landing track: the tail of the times is short by those, and the report says so.

The time limit a real start flies to today (`prior_free_generation.limits_s`: the sentence's rows from the first predicted
step × the step × the executor spec's ``timeout_factor``) leaves a flight its SLACK — the limit less its observed time
from the first predicted step to the landing (`observed_remaining_s`) — and a go-around fits in it only when its cost does.
The slack of every labelled train sentence is reported per airport and pooled, with the share of sentences whose slack
covers the go-arounds' cost p50 / p95 (the airport's own and the pooled).

Writes ``go_arounds.json`` (every go-around, per-airport and pooled summaries) into a NEW directory.

    python run_ts.py go_around_census --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v11_20260927 --out 4dTrajectory/outputs/POOLED/traffic/go_arounds_<date>
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from geokit import haversine_km

from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
from final_approach import TrackPoint
from final_approach.frame import RunwayFrame
from trajectory_data_process.harvest.airports import load_airport
from trajectory_data_process.harvest.store import HarvestPaths, read_manifest, read_track_view
from ts_transformer.autopilot.spec import load_spec as load_executor_spec
from ts_transformer.data.day_split import DaySplit, landing_day, parse_utc
from ts_transformer.instructions.artefact import (
    arrival_manifest_sha256s, load_day_split, load_sentences, load_spec, signals_flights,
)
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.repo_layout import HARVEST_ROOT, REPO_ROOT, arrival_manifest_path, git_state, repo_relative

SCHEMA = "ts-go-around-census-v1"
SPLIT = "train"
#: A low pass (design §6.6 step 8 item 7; my reading, not a regulation): on a runway end's extended centreline
MAX_CROSS_M = 500.0
#: along the landing direction from the threshold: 10 km before it (a 3° path is ~540 m up there) to 3 km past it
ALONG_RANGE_M = (-10_000.0, 3_000.0)
#: above the threshold (the default ceiling; ``--max-height-m`` reads another)
MAX_HEIGHT_M = 600.0
#: moving along the landing direction between its first and last sample
MIN_ALONG_PROGRESS_M = 1_000.0
#: two low samples with at most this many others between them are one pass
MAX_SKIPPED_SAMPLES = 3
#: came down to the go-around point from a level at least this much higher, and climbed to one at least this much
#: above it after it, each level held for `HOLD_S` over at least `MIN_HOLD_SAMPLES` samples
MIN_DESCENT_M = 150.0
MIN_CLIMB_M = 150.0
HOLD_S = 20.0
MIN_HOLD_SAMPLES = 5
#: a sample this far from the median position of the `STRAY_WINDOW` samples either side of it is another aircraft's
STRAY_WINDOW = 5
STRAY_DISTANCE_M = 1_000.0
#: a go-around point past the threshold and at most this high is on the runway (under every published TCH, harvest TD9)
ON_RUNWAY_HEIGHT_M = 15.0
#: the go-around point's height above its threshold, reported in these bands (up to the run's ceiling)
HEIGHT_BAND_EDGES_M = (300.0, 600.0)
#: what a row of the census is: a go-around (timed), or set aside (counted)
GO_AROUND, ON_RUNWAY, LANDING_NOT_LAST = "go-around", "on the runway", "landing not the last"
#: what follows a stored landing
THE_LAST, LANDED_LATER, LEFT = "the last", "landed later", "left"


@dataclass(frozen=True)
class Pass:
    """A low pass: samples ``first`` … ``last`` on runway end ``runway``'s final, lowest at ``lowest``."""

    runway: str
    first: int
    last: int
    lowest: int
    median_abs_cross_m: float


@dataclass(frozen=True)
class GoAround:
    runway: str
    index: int              # the go-around point: the pass's lowest sample
    height_m: float         # above the runway end's threshold
    along_m: float          # along its landing direction from the threshold (negative before it)
    cross_m: float
    descent_m: float        # the highest level held before the point, above it
    climb_m: float          # the highest level held after it (before the next pass or the landing), above it


def strays(points: Sequence[TrackPoint], frame: RunwayFrame) -> np.ndarray:
    """Which samples are another aircraft's: farther than `STRAY_DISTANCE_M` from the median position of the
    `STRAY_WINDOW` samples either side (edges padded with the end sample), positions in ``frame``."""
    projected = np.array([(p.along_m, p.cross_m) for p in frame.project_all(points)]).reshape(-1, 2)
    padded = np.pad(projected, ((STRAY_WINDOW, STRAY_WINDOW), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * STRAY_WINDOW + 1, axis=0)
    centre = np.median(windows, axis=-1)
    return np.hypot(*(projected - centre).T) > STRAY_DISTANCE_M


def low_passes(points: Sequence[TrackPoint], frames: Sequence[RunwayFrame], max_height_m: float) -> list[Pass]:
    """Every low pass over ``points`` (at most ``max_height_m`` above the threshold), in time order; passes of two
    runway ends whose sample ranges overlap are one pass."""
    found: list[Pass] = []
    for frame in frames:
        projected = frame.project_all(points)
        low = [i for i, p in enumerate(projected)
               if abs(p.cross_m) <= MAX_CROSS_M and ALONG_RANGE_M[0] <= p.along_m <= ALONG_RANGE_M[1]
               and p.height_m <= max_height_m]
        runs: list[list[int]] = []
        for i in low:
            if runs and i - runs[-1][-1] - 1 <= MAX_SKIPPED_SAMPLES:
                runs[-1].append(i)
            else:
                runs.append([i])
        for run in runs:
            if projected[run[-1]].along_m - projected[run[0]].along_m < MIN_ALONG_PROGRESS_M:
                continue
            found.append(Pass(frame.ident, run[0], run[-1], min(run, key=lambda i: projected[i].height_m),
                              statistics.median(abs(projected[i].cross_m) for i in run)))
    found.sort(key=lambda p: p.first)
    clusters: list[list[Pass]] = []
    for candidate in found:
        if clusters and candidate.first <= max(p.last for p in clusters[-1]):
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    return [min(cluster, key=lambda p: p.median_abs_cross_m) for cluster in clusters]


def held_level(times: Sequence[float], altitudes: Sequence[float], first: int, end: int) -> float:
    """The highest altitude samples ``first`` … ``end - 1`` stay at or above for `HOLD_S`: over every stretch of them
    spanning at least `HOLD_S` and holding at least `MIN_HOLD_SAMPLES` samples, its lowest sample, at its highest
    (``-inf`` when no stretch qualifies)."""
    best = float("-inf")
    last = first
    for start in range(first, end):
        last = max(last, start + MIN_HOLD_SAMPLES - 1)
        while last < end and times[last] - times[start] < HOLD_S:
            last += 1
        if last >= end:
            break
        best = max(best, min(altitudes[start:last + 1]))
    return best


def go_arounds(points: Sequence[TrackPoint], times: Sequence[float], frames: Sequence[RunwayFrame], landing_index: int,
               max_height_m: float) -> list[GoAround]:
    """The go-arounds among the samples before ``landing_index`` (see the module docstring); ``times`` are the
    samples' seconds. The samples are taken as they are: strays are set aside by the caller (`read_flight`)."""
    before = points[:landing_index]
    passes = low_passes(before, frames, max_height_m)
    by_ident = {frame.ident: frame for frame in frames}
    altitude = [p.alt_m for p in before]
    found = []
    for n, low in enumerate(passes):
        point = low.lowest
        end = passes[n + 1].first if n + 1 < len(passes) else landing_index
        descent = held_level(times, altitude, 0, point) - altitude[point]
        climb = held_level(times, altitude, point + 1, end) - altitude[point]
        if descent < MIN_DESCENT_M or climb < MIN_CLIMB_M:
            continue
        where = by_ident[low.runway].project(before[point])
        found.append(GoAround(low.runway, point, where.height_m, where.along_m, where.cross_m, descent, climb))
    return found


@dataclass(frozen=True)
class FlightReading:
    go_arounds: list[GoAround]      # indices into the stored samples
    strays: int
    #: `THE_LAST`, or — a level `MIN_CLIMB_M` above the landing sample held after it — `LANDED_LATER` / `LEFT`
    after_landing: str
    #: the samples read (strays left out), as indices into the stored samples
    kept: tuple[int, ...]

    @property
    def landing_not_last(self) -> bool:
        return self.after_landing != THE_LAST


def read_flight(points: Sequence[TrackPoint], times: Sequence[float], frames: Sequence[RunwayFrame], landing_index: int,
                max_height_m: float) -> FlightReading:
    """One stored track: its strays set aside, its go-arounds (indices back in the stored samples) and whether its
    landing is followed by a held climb."""
    stray = strays(points, frames[0])
    kept = [i for i in range(len(points)) if not stray[i]]
    before = sum(1 for i in kept if i < landing_index)
    found = go_arounds([points[i] for i in kept], [times[i] for i in kept], frames, before, max_height_m)
    after = [i for i in kept if i > landing_index]
    climbed = held_level([times[i] for i in after], [points[i].alt_m for i in after], 0, len(after))
    if climbed - points[landing_index].alt_m < MIN_CLIMB_M:
        after_landing = THE_LAST
    else:
        passes = low_passes([points[i] for i in after], frames, max_height_m)
        after_landing = LANDED_LATER if passes and passes[-1].last == len(after) - 1 else LEFT
    return FlightReading([GoAround(**{**asdict(event), "index": kept[event.index]}) for event in found],
                         int(stray.sum()), after_landing, tuple(kept))


@dataclass(frozen=True)
class AirportInputs:
    """What one airport's census reads, all of it resolved before a worker starts."""

    code: str
    harvest_root: Path
    frames: tuple[RunwayFrame, ...]
    reference: tuple[float, float]              # (lat, lon): distances from the field
    train: tuple[dict[str, Any], ...]           # the tracks roster's `assigned` rows landing on a training day
    #: flight_key → the arrival roster's `first_sample_index`
    slice_start: Mapping[str, int]
    #: flight_key → ("labelled", entry UTC, sentence rows) | ("refused", entry UTC, None)
    artefact: Mapping[str, tuple[str, str, int | None]]
    step_s: float
    max_height_m: float


def census_airport(inputs: AirportInputs) -> dict[str, Any]:
    """Every go-around of one airport's training-day landings — timed or set aside, one row each — and the counts of
    landings read, stray samples and landings that were not the last."""
    paths = HarvestPaths(root=inputs.harvest_root, code=inputs.code)
    rows = []
    stray_samples = 0
    not_last: dict[str, list[dict[str, Any]]] = {LANDED_LATER: [], LEFT: []}
    for record in inputs.train:
        track = read_track_view(paths, record["file"])
        samples = track["samples"]
        landing = int(track["landing_sample_index"])
        reading = read_flight([TrackPoint(lat=s[2], lon=s[1], alt_m=s[3]) for s in samples], [s[0] for s in samples],
                              inputs.frames, landing, inputs.max_height_m)
        stray_samples += reading.strays
        key = record["flight_key"]
        status, entry, words = inputs.artefact.get(key, ("absent", None, None))
        if reading.landing_not_last:
            not_last[reading.after_landing].append({
                "flight_key": key, "artefact": status, "go_arounds_before": len(reading.go_arounds),
                "landing_to_track_end_s": float(samples[reading.kept[-1]][0]) - float(samples[landing][0])})
        start = parse_utc(track["start_time_utc"])
        for number, event in enumerate(reading.go_arounds):
            point_s = float(samples[event.index][0])
            speed = float(track["reported_ground_speeds_m_s"][event.index])
            to_landing = float(samples[landing][0]) - point_s
            still_needed = max(0.0, -event.along_m) / speed
            at = start + timedelta(seconds=point_s)
            sentence_row = None if entry is None else (at - parse_utc(entry)).total_seconds() / inputs.step_s
            on_runway = event.along_m >= 0.0 and event.height_m <= ON_RUNWAY_HEIGHT_M
            rows.append({
                "airport": inputs.code, "flight_key": key, "landing_runway": record["runway"],
                "kind": LANDING_NOT_LAST if reading.landing_not_last else ON_RUNWAY if on_runway else GO_AROUND,
                "number": number, "of": len(reading.go_arounds), **asdict(event),
                "time_utc": at.isoformat().replace("+00:00", "Z"),
                "ground_speed_mps": speed, "to_landing_s": to_landing, "still_needed_s": still_needed,
                "cost_s": to_landing - still_needed,
                "farthest_km": max(haversine_km(samples[i][2], samples[i][1], *inputs.reference)
                                   for i in reading.kept if event.index <= i <= landing),
                "same_runway": event.runway == record["runway"],
                "arrival_roster": key in inputs.slice_start,
                "in_arrival_slice": key in inputs.slice_start and inputs.slice_start[key] <= event.index,
                "artefact": status,
                "in_sentence": words is not None and 0.0 <= sentence_row < words,
            })
    return {"airport": inputs.code, "landings": len(inputs.train), "stray_samples": stray_samples,
            "landings_not_the_last": not_last, "go_arounds": rows}


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0}
    return {"n": int(len(array)), "min": float(array.min()), "p25": float(np.percentile(array, 25)),
            "p50": float(np.percentile(array, 50)), "p75": float(np.percentile(array, 75)),
            "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def height_band(height_m: float, max_height_m: float) -> str:
    edges = (0.0, *(edge for edge in HEIGHT_BAND_EDGES_M if edge < max_height_m), max_height_m)
    for low, high in zip(edges, edges[1:]):
        if height_m <= high:
            return f"<={high:.0f} m" if low == 0.0 else f"{low:.0f}-{high:.0f} m"
    raise ValueError(f"a go-around point {height_m:.0f} m up is not a low pass")


def standing(row: Mapping[str, Any]) -> str:
    return (f"{row['artefact']}, {'inside' if row['in_sentence'] else 'outside'} the sentence, "
            f"{'inside' if row['in_arrival_slice'] else 'outside'} the arrival slice")


def summarise(rows: Sequence[dict[str, Any]], landings: int, max_height_m: float) -> dict[str, Any]:
    """The go-arounds of ``rows`` (kind `GO_AROUND`) timed, all and by group, with the rate per 1,000 landings; the rows
    set aside counted by kind and standing."""
    timed = [r for r in rows if r["kind"] == GO_AROUND]

    def times(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {"go_arounds": len(group), "to_landing_s": distribution([r["to_landing_s"] for r in group]),
                "cost_s": distribution([r["cost_s"] for r in group]),
                "farthest_km": distribution([r["farthest_km"] for r in group])}

    groups = {
        "height": {band: [r for r in timed if height_band(r["height_m"], max_height_m) == band]
                   for band in sorted({height_band(r["height_m"], max_height_m) for r in timed})},
        "runway": {"same": [r for r in timed if r["same_runway"]], "other": [r for r in timed if not r["same_runway"]]},
        "artefact": {status: [r for r in timed if r["artefact"] == status] for status in ("labelled", "refused", "absent")},
        "in_sentence": {"yes": [r for r in timed if r["in_sentence"]], "no": [r for r in timed if not r["in_sentence"]]},
        "in_arrival_slice": {"yes": [r for r in timed if r["in_arrival_slice"]],
                             "no": [r for r in timed if not r["in_arrival_slice"]]},
    }
    set_aside = Counter(f"{r['kind']}: {r['artefact']}" for r in rows if r["kind"] != GO_AROUND)
    per_flight = Counter(r["flight_key"] for r in timed)
    return {"landings": landings, "flights_with_a_go_around": len(per_flight),
            "per_1000_landings": 1000.0 * len(timed) / landings,
            "flights_with_several": sum(1 for n in per_flight.values() if n > 1),
            "standing": dict(sorted(Counter(standing(r) for r in timed).items())),
            "set_aside": dict(sorted(set_aside.items())),
            **times(timed),
            "by": {axis: {name: times(group) for name, group in parts.items()} for axis, parts in groups.items()}}


def not_the_last(flights: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Per kind (`LANDED_LATER`, `LEFT`): how many, by artefact standing, the stored landing to the track's end, and
    every flight."""
    return {kind: {"flights": len(group), "by_artefact": dict(sorted(Counter(f["artefact"] for f in group).items())),
                   "landing_to_track_end_s": distribution([f["landing_to_track_end_s"] for f in group]),
                   "list": list(group)}
            for kind, group in flights.items()}


def observed_remaining_s(words: int, step_s: float) -> float:
    """A sentence of ``words`` rows: from its first predicted step to the landing (its last row is at the landing) —
    `prior_free_generation.observed_remaining_s`, pinned to it by a test (that runner imports torch)."""
    return (words - 1 - N_LOOK) * step_s


def limit_s(words: int, step_s: float, timeout_factor: float) -> float:
    """A real start's time limit from its first predicted step — `prior_free_generation.limits_s`, pinned by a test."""
    return (words - N_LOOK) * step_s * timeout_factor


def slack(words: Sequence[int], step_s: float, timeout_factor: float,
          costs: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """The slack of sentences of ``words`` rows, and the share of them whose slack covers each named set of go-around
    costs' p50 and p95."""
    spare = np.array([limit_s(n, step_s, timeout_factor) - observed_remaining_s(n, step_s) for n in words])
    covers = {}
    for name, values in costs.items():
        if len(values):
            p50, p95 = np.percentile(values, [50, 95])
            covers[name] = {"cost_p50_s": float(p50), "covers_p50": float(np.mean(spare >= p50)),
                            "cost_p95_s": float(p95), "covers_p95": float(np.mean(spare >= p95))}
    return {"sentences": len(spare), "slack_s": distribution(spare), "covers": covers}


def artefact_index(directory: Path, airports: Sequence[str]) -> dict[str, dict[str, tuple[str, str, int | None]]]:
    """Per airport, flight_key → its standing in the artefact's train split (labelled with its sentence rows, or
    refused), from the signals' records and the sentences' row offsets — no signal arrays are read."""
    spec = load_spec(directory)
    meta = signals_flights(directory, SPLIT)
    sentences = load_sentences(directory, SPLIT, spec)
    offsets = sentences["offsets"]
    words = {int(signal): int(offsets[i + 1] - offsets[i]) for i, signal in enumerate(sentences["signal_index"])}
    index: dict[str, dict[str, tuple[str, str, int | None]]] = {code: {} for code in airports}
    for i, flight in enumerate(meta):
        if flight["airport"] not in index:
            continue
        key = flight["dataset_id"].split(":", 1)[1]
        index[flight["airport"]][key] = (("labelled", flight["entry_time_utc"], words[i]) if i in words
                                         else ("refused", flight["entry_time_utc"], None))
    return index


def training_landings(roster: Mapping[str, Any], days: DaySplit) -> tuple[dict[str, Any], ...]:
    """The roster's `assigned` rows whose landing falls on a training day — decided from the roster alone, so a track of
    any other day is never opened; a landing on a day the split does not list is refused (`DaySplit.split_of`)."""
    return tuple(r for r in roster["records"]
                 if r["outcome"] == "assigned" and days.split_of(landing_day(r["landing_time_utc"])) == SPLIT)


def airport_inputs(code: str, harvest_root: Path, days: DaySplit, artefact: Mapping[str, tuple[str, str, int | None]],
                   step_s: float, max_height_m: float) -> tuple[AirportInputs, dict[str, Any]]:
    paths = HarvestPaths(root=harvest_root, code=code)
    roster = read_manifest(paths)
    train = training_landings(roster, days)
    arrivals = json.loads(arrival_manifest_path(code, harvest_root).read_text(encoding="utf-8"))
    airport = load_airport(code, config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP)
    inputs = AirportInputs(code, harvest_root, tuple(airport.frames("hae")), (airport.lat, airport.lon), train,
                           {r["flight_key"]: int(r["first_sample_index"]) for r in arrivals["records"]}, artefact, step_s,
                           max_height_m)
    read = {"tracks_manifest_sha256": file_sha256(paths.tracks / "manifest.json"),
            "arrival_manifest_sha256": file_sha256(arrival_manifest_path(code, harvest_root)),
            "assigned": sum(r["outcome"] == "assigned" for r in roster["records"]), "train_days_assigned": len(train),
            "runway_ends": [frame.ident for frame in inputs.frames]}
    return inputs, read


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True,
                        help="the instruction artefact whose train split the go-arounds are placed in")
    parser.add_argument("--executor", type=Path, required=True,
                        help="the executor spec whose timeout factor sets a real start's time limit (the slack readout)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--harvest-root", type=Path, default=HARVEST_ROOT, help="the harvest the tracks are read from")
    parser.add_argument("--airports", nargs="+", help="only these airports (a smoke test; the report says so)")
    parser.add_argument("--workers", type=int, default=5, help="processes, one airport each")
    parser.add_argument("--max-height-m", type=float, default=MAX_HEIGHT_M,
                        help=f"a low pass's ceiling above the threshold (default {MAX_HEIGHT_M:.0f} m, design §6.6 step 8 "
                             "item 7; another value is a sensitivity reading and the report records it)")
    args = parser.parse_args(argv)
    directory = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a census is never overwritten")
    started = time.perf_counter()
    recorded = arrival_manifest_sha256s(directory)
    airports = sorted(args.airports or recorded)
    days = load_day_split(directory)
    step_s = load_spec(directory).step_s
    params, executor_record = load_executor_spec(executor)
    placed = artefact_index(directory, airports)
    inputs, read = [], {}
    for code in airports:
        airport, read[code] = airport_inputs(code, args.harvest_root, days, placed[code], step_s, args.max_height_m)
        if read[code]["arrival_manifest_sha256"] != recorded[code]:
            raise ValueError(f"{code}'s arrival manifest is not the one the artefact's signals were read from "
                             f"(sha256 {recorded[code][:12]} recorded)")
        inputs.append(airport)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(census_airport, inputs))
    rows = [row for result in results for row in result["go_arounds"]]
    pooled_costs = [r["cost_s"] for r in rows if r["kind"] == GO_AROUND]
    report = {}
    for result in results:
        code = result["airport"]
        report[code] = {**summarise(result["go_arounds"], result["landings"], args.max_height_m),
                        "stray_samples": result["stray_samples"],
                        "landings_not_the_last": not_the_last(result["landings_not_the_last"])}
        report[code]["time_limit"] = slack(
            [n for status, _, n in placed[code].values() if status == "labelled"], step_s, params.timeout_factor,
            {"airport": [r["cost_s"] for r in result["go_arounds"] if r["kind"] == GO_AROUND], "pooled": pooled_costs})
        times = report[code]["to_landing_s"]
        spread = f", to landing p50 / p95 {times['p50']:.0f} / {times['p95']:.0f} s" if times["n"] else ""
        print(f"{code}: {result['landings']} landings, {report[code]['go_arounds']} go-arounds "
              f"({report[code]['per_1000_landings']:.2f} per 1,000){spread}; set aside {report[code]['set_aside']}",
              flush=True)
    pooled = summarise(rows, sum(r["landings"] for r in results), args.max_height_m)
    pooled["stray_samples"] = sum(r["stray_samples"] for r in results)
    pooled["landings_not_the_last"] = not_the_last(
        {kind: [f for r in results for f in r["landings_not_the_last"][kind]] for kind in (LANDED_LATER, LEFT)})
    pooled["time_limit"] = slack([n for code in airports for status, _, n in placed[code].values() if status == "labelled"],
                                 step_s, params.timeout_factor, {"pooled": pooled_costs})
    payload = {
        "schema": SCHEMA, "written_utc": utc_now(), "split": SPLIT, "only_airports": args.airports,
        "harvest_root": repo_relative(args.harvest_root.resolve()), "instructions": repo_relative(directory),
        "executor": repo_relative(executor), "executor_spec_sha256": executor_record["sha256"],
        "timeout_factor": params.timeout_factor,
        "criteria": {"max_cross_m": MAX_CROSS_M, "along_range_m": list(ALONG_RANGE_M),
                     "max_height_m": args.max_height_m, "design_max_height_m": MAX_HEIGHT_M,
                     "min_along_progress_m": MIN_ALONG_PROGRESS_M, "max_skipped_samples": MAX_SKIPPED_SAMPLES,
                     "min_descent_m": MIN_DESCENT_M, "min_climb_m": MIN_CLIMB_M, "hold_s": HOLD_S,
                     "min_hold_samples": MIN_HOLD_SAMPLES, "stray_window": STRAY_WINDOW,
                     "stray_distance_m": STRAY_DISTANCE_M, "on_runway_height_m": ON_RUNWAY_HEIGHT_M,
                     "height_band_edges_m": list(HEIGHT_BAND_EDGES_M)},
        "unseen": ["a go-around whose loop left 30 km: the stored track keeps only the last stretch inside it, so the first "
                   "approach and the go-around are cropped",
                   "a go-around whose aircraft did not land here",
                   "a low go-around the harvest took for the landing (the best-aligned crossing under 100 m, not the last): "
                   "its flight is counted in landings_not_the_last as 'landed later', untimed beyond an upper bound, with "
                   "any go-around before it set aside"],
        "read": read, "git": git_state(), "seconds": time.perf_counter() - started,
        "pooled": pooled, "airports": report, "go_arounds": rows,
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "go_arounds.json", payload)
    print(f"{pooled['go_arounds']} go-arounds in {pooled['landings']} landings; wrote {out / 'go_arounds.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
