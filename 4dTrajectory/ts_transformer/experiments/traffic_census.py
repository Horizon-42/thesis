"""Multi-aircraft M0 (design §6.4 step 4): the observed traffic of the TRAINING days, judged as the closed loop will be.

From one instruction artefact split by operating day and each airport's arrivals manifest (its runway targets, for the
separation rules), every arrival of the training days — with a sentence or without (background) — is put in its
airport's scene from its first row to its last in the scene (`prior.scene.presence`). At every scene step (§2.1: even
UTC seconds) the aircraft present are judged by `inference.separation`:

- each aircraft's position is interpolated at the step from its rows; its runway is the one it landed on (what it was
  cleared for — the observed census is a measurement, not a model input); its position on the approach clock is its
  runway threshold's `Separation.along_nm` less its distance before that threshold along the course; it is established
  on the final from its capture row on (the artefact's for a flight with a sentence, the labeller's own `capture_row`
  for a background one; a background flight that never stays in the corridor is never established); its CWT category
  comes from its type (`runway_schedule.wake_category`; a flight without a type is counted and judged on radar alone);
- a loss that lasts over consecutive steps is one EPISODE of that pair;
- at every landing (the leader over its threshold at its landing time) the established aircraft next behind it is
  judged against TBL 5-5-2, and the gap, the distance the rules require and their difference are kept for every such
  pair (design §2.5 / §9 item 6: how often the required distance is just the radar minimum);
- on every final, consecutive established aircraft give a closing speed (the one behind's rate along the approach
  clock less the one ahead's, from the rows);
- order swaps (§4.1): two arrivals whose times in the scene overlap and that land on the same runway, or on two
  same-direction runways, where the one that entered later landed first;
- the samples each segment is cut into (§2.3, `prior.scene.samples`): how many flights each holds (``A_max``), and the
  windows of the window setting (§2.2: 20 minutes every 10 minutes): how many flights with a sentence enter each, and
  how many aircraft are already in the scene when it opens.

A type the tables do not list stops the run before any judging and writes the list (``types_not_listed.json``): add each
to `docs/literature/arrival_separation/cwt_supplement.csv` with its source. Only training days are read; a day boundary
(09Z) cuts no scene short of anything but its neighbours from the next day. Writes ``census.json`` into a NEW directory.

    python run_ts.py traffic_census --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --out 4dTrajectory/outputs/POOLED/traffic/census_<date>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from geokit import NM_M

from ts_transformer.inference.runway_schedule import SAME, faa_separation, read_cwt_tables, wake_category
from ts_transformer.inference.separation import Traffic, losses, next_behind, wake_at_threshold
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals, load_spec
from ts_transformer.instructions.labeller.lateral import capture_row
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.scene import Presence, SceneIndex, presence, samples, scene_steps
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, git_state

SPLIT = "train"
#: `faa_separation` turns distances into times at an approach speed; nothing here reads a time, so any speed does.
APPROACH_SPEED_MPS = 70.0
#: The window setting (design §2.2): a window this long opens every `WINDOW_EVERY_S`.
WINDOW_S, WINDOW_EVERY_S = 1_200.0, 600.0


@dataclass(frozen=True)
class Track:
    """One arrival as the census judges it: its rows in the scene, in epoch seconds."""

    presence: Presence
    e_m: np.ndarray
    n_m: np.ndarray
    height_m: np.ndarray
    along_m: np.ndarray            # its position on the approach clock at each row
    along_rate_mps: np.ndarray     # d along / dt at each row
    captured_s: float              # the time of its capture row; inf when it is never established
    category: str | None

    @property
    def key(self) -> str:
        return self.presence.dataset_id


def _quantiles(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0}
    return {"n": int(len(array)), "min": float(array.min()), "p1": float(np.percentile(array, 1)),
            "p5": float(np.percentile(array, 5)), "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def track(flight: FlightSignals, sentence_rows: int | None, capture: int | None, geometry: AirportGeometry,
          along_threshold_m: float, spec: VocabularySpec) -> Track:
    """``capture``: the artefact's capture row of a flight with a sentence; None for a background flight, whose capture
    the labeller's own rule finds (or not)."""
    seen = presence(flight, sentence_rows, geometry)
    rows = len(seen.times_s)
    candidate = geometry.candidates[geometry.candidate_index(flight.runway)]
    relative = relative_to_runway(flight.e_m[:rows], flight.n_m[:rows], flight.track_deg[:rows],
                                  flight.altitude_m[:rows], candidate)
    if capture is None:
        try:
            capture = capture_row(relative, spec)
        except Refused:             # never stays in the corridor to the end: never established
            capture = None
    along = along_threshold_m - relative.before_threshold_m
    category = None if flight.typecode is None else wake_category(flight.typecode)
    return Track(seen, flight.e_m[:rows], flight.n_m[:rows], flight.altitude_m[:rows], along,
                 np.gradient(along, seen.times_s) if rows > 1 else np.zeros(rows),
                 float(seen.times_s[capture]) if capture is not None else np.inf, category)


def _traffic(tracks: Sequence[Track], t_s: float) -> Traffic:
    def at(values: np.ndarray, item: Track) -> float:
        return float(np.interp(t_s, item.presence.times_s, values))

    return Traffic(np.array([at(t.e_m, t) for t in tracks]), np.array([at(t.n_m, t) for t in tracks]),
                   np.array([at(t.height_m, t) for t in tracks]), tuple(t.presence.runway for t in tracks),
                   np.array([at(t.along_m, t) for t in tracks]), np.array([t_s >= t.captured_s for t in tracks]),
                   tuple(t.category for t in tracks))


def census_airport(tracks: list[Track], separation, step_s: float) -> dict[str, Any]:
    by_key = {t.key: t for t in tracks}
    index = SceneIndex([t.presence for t in tracks])
    segments = index.segments()
    episodes: list[dict[str, Any]] = []
    open_episodes: dict[tuple[str, str, str], dict[str, Any]] = {}
    coexisting: set[tuple[str, str]] = set()
    closing: list[float] = []
    judged_steps = scene_seconds = 0.0
    for segment in segments:
        members = [by_key[p.dataset_id] for p in segment]
        start, end = min(p.start_s for p in segment), max(p.end_s for p in segment)
        scene_seconds += end - start
        for t_s in scene_steps(start, end, step_s):
            here = [m for m in members if m.presence.start_s <= t_s <= m.presence.end_s]
            if len(here) < 2:
                continue
            judged_steps += 1
            traffic = _traffic(here, float(t_s))
            for a in range(len(here)):
                for b in range(a + 1, len(here)):
                    coexisting.add(tuple(sorted((here[a].key, here[b].key))))
            now = set()
            for loss in losses(traffic, separation):
                pair = tuple(sorted((here[loss.i].key, here[loss.j].key)))
                key = (*pair, loss.kind)
                now.add(key)
                ratio = loss.distance_m / loss.required_m
                episode = open_episodes.get(key)
                if episode is None or episode["last_s"] != t_s - step_s:
                    episode = {"pair": list(pair), "kind": loss.kind, "relation": loss.relation, "first_s": float(t_s),
                               "steps": 0, "min_ratio": ratio, "wake_known": loss.wake_known,
                               "responsible": sorted(here[k].key for k in loss.responsible),
                               "responsible_speaking": [here[k].presence.speaking for k in loss.responsible]}
                    open_episodes[key] = episode
                    episodes.append(episode)
                episode["steps"] += 1
                episode["last_s"] = float(t_s)
                episode["min_ratio"] = min(episode["min_ratio"], ratio)
            # closing speeds on each final: consecutive established aircraft on one runway (or a pair as one)
            established = [k for k in range(len(here)) if traffic.established[k]]
            for k in established:
                follower = next_behind(traffic, k, separation)
                if follower is not None:
                    closing.append(float(np.interp(t_s, here[follower].presence.times_s, here[follower].along_rate_mps)
                                         - np.interp(t_s, here[k].presence.times_s, here[k].along_rate_mps)))

    # at every landing: the established aircraft next behind, against TBL 5-5-2
    threshold: dict[str, Any] = {"judged": 0, "losses": 0, "untyped": 0, "gap_minus_required_m": [],
                                 "required_nm": Counter()}
    for leader in tracks:
        t_s = leader.presence.landing_s
        near = [by_key[p.dataset_id] for p in index.overlapping(t_s, t_s) if p.dataset_id != leader.key]
        if not near:
            continue
        traffic = _traffic(near, t_s)
        # the leader over its threshold: on the approach clock at its threshold, established (its position is its last
        # row's; the threshold check reads only the approach clock)
        along_threshold = separation.along_nm[leader.presence.runway] * NM_M
        scene = Traffic(np.append(traffic.e_m, leader.e_m[-1]), np.append(traffic.n_m, leader.n_m[-1]),
                        np.append(traffic.height_m, leader.height_m[-1]), (*traffic.runway, leader.presence.runway),
                        np.append(traffic.along_m, along_threshold), np.append(traffic.established, True),
                        (*traffic.category, leader.category))
        lead = len(near)
        follower = next_behind(scene, lead, separation)
        if follower is None:
            continue
        if leader.category is None or near[follower].category is None:
            threshold["untyped"] += 1
            continue
        threshold["judged"] += 1
        required_nm = separation.distance_nm(leader.presence.runway, leader.category, near[follower].presence.runway,
                                             near[follower].category)
        threshold["required_nm"][f"{required_nm:g}"] += 1
        gap = along_threshold - float(scene.along_m[follower])
        threshold["gap_minus_required_m"].append(gap - required_nm * NM_M)
        threshold["losses"] += wake_at_threshold(scene, lead, separation) is not None

    # order swaps (§4.1)
    swaps = {"same_runway": [0, 0], "same_direction_other_runway": [0, 0]}
    involved: set[str] = set()
    for p in index.flights:
        for q in index.overlapping(p.start_s, p.end_s):
            if q.dataset_id <= p.dataset_id:
                continue
            relation = separation.relation(p.runway, q.runway)
            group = "same_runway" if relation == SAME else (
                "same_direction_other_runway" if frozenset((p.runway, q.runway)) in separation.relations else None)
            if group is None:
                continue
            swaps[group][1] += 1
            if (p.start_s - q.start_s) * (p.landing_s - q.landing_s) < 0:
                swaps[group][0] += 1
                involved.update((p.dataset_id, q.dataset_id))

    # samples and windows
    sample_flights, loss_minutes, entering, airborne = [], [], [], []
    for segment in segments:
        for sample in samples(segment, step_s):
            sample_flights.append(len(sample.flights))
            loss_minutes.append((sample.loss_end_s - sample.loss_start_s) / 60.0)
        start, end = min(p.start_s for p in segment), max(p.end_s for p in segment)
        opens = start
        while opens < end:
            entering.append(sum(1 for p in segment if p.speaking and opens <= p.start_s < opens + WINDOW_S))
            airborne.append(sum(1 for p in segment if p.start_s < opens <= p.end_s))
            opens += WINDOW_EVERY_S

    hours = scene_seconds / 3600.0
    kinds = Counter(e["kind"] for e in episodes)
    relations = Counter(e["relation"] for e in episodes)
    return {
        "flights": {"speaking": sum(t.presence.speaking for t in tracks),
                    "background": sum(not t.presence.speaking for t in tracks),
                    "untyped": sum(t.category is None for t in tracks),
                    "background_never_established": sum(not t.presence.speaking and not np.isfinite(t.captured_s)
                                                        for t in tracks)},
        "scene_hours": hours, "steps_with_two_or_more": int(judged_steps),
        "pairs_in_the_scene_together": len(coexisting),
        "losses": {"episodes": len(episodes), "per_scene_hour": len(episodes) / hours if hours else None,
                   "pairs_share": len({tuple(e["pair"]) for e in episodes}) / len(coexisting) if coexisting else None,
                   "by_kind": dict(kinds), "by_relation": dict(relations),
                   "untyped_in_trail": sum(not e["wake_known"] for e in episodes),
                   "responsible_has_a_sentence": sum(any(e["responsible_speaking"]) for e in episodes),
                   "steps": _quantiles([e["steps"] for e in episodes]),
                   "min_distance_over_required": _quantiles([e["min_ratio"] for e in episodes])},
        "at_threshold": {"judged": threshold["judged"], "losses": threshold["losses"], "untyped": threshold["untyped"],
                         "gap_minus_required_m": _quantiles(threshold["gap_minus_required_m"]),
                         "required_nm": dict(sorted(threshold["required_nm"].items())),
                         "required_is_the_radar_minimum_share": (
                             threshold["required_nm"][f"{separation.same_nm:g}"] / threshold["judged"]
                             if threshold["judged"] else None)},
        "closing_speed_on_a_final_mps": _quantiles(closing),
        "order_swaps": {group: {"swapped": s, "pairs": n, "share": s / n if n else None} for group, (s, n) in swaps.items()},
        "flights_in_a_swap_share": len(involved) / len(tracks),
        "samples": {"count": len(sample_flights), "flights": _quantiles(sample_flights),
                    "A_max": max(sample_flights), "loss_minutes": _quantiles(loss_minutes)},
        "windows": {"count": len(entering), "entering_with_a_sentence": _quantiles(entering),
                    "airborne_at_opening": _quantiles(airborne)},
        "episodes": episodes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True, help="an instruction artefact split by operating day")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--airports", nargs="+", help="only these airports (a smoke test; the census says so)")
    args = parser.parse_args(argv)
    directory = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a census is never overwritten")
    started = time.perf_counter()
    spec = load_spec(directory)
    candidates = load_candidates(directory)
    flights = load_signals(directory, SPLIT)
    sentences = load_sentences(directory, SPLIT, spec)
    offsets = sentences["offsets"]
    spoken = {int(signal): (int(offsets[i + 1] - offsets[i]), int(sentences["capture_row"][i]))
              for i, signal in enumerate(sentences["signal_index"])}

    wanted = [(i, f) for i, f in enumerate(flights) if not args.airports or f.airport in args.airports]
    missing = sorted({f.typecode for _, f in wanted if f.typecode is not None} - set(read_cwt_tables()))
    if missing:
        out.mkdir(parents=True)
        write_json_atomic(out / "types_not_listed.json", {"types": missing, "counts": dict(Counter(
            f.typecode for _, f in wanted if f.typecode in set(missing)))})
        raise SystemExit(f"{len(missing)} types are in neither CWT table: {missing} (see {out / 'types_not_listed.json'})")

    by_airport: dict[str, list[tuple[int, FlightSignals]]] = defaultdict(list)
    for i, flight in wanted:
        by_airport[flight.airport].append((i, flight))
    report: dict[str, Any] = {}
    for airport in sorted(by_airport):
        targets = json.loads(arrival_manifest_path(airport).read_text(encoding="utf-8"))["runway_targets"]
        separation = faa_separation(targets, speed_mps=APPROACH_SPEED_MPS)
        tracks = []
        for i, flight in by_airport[airport]:
            rows, capture = spoken.get(i, (None, None))
            tracks.append(track(flight, rows, capture, candidates[airport],
                                separation.along_nm[flight.runway] * NM_M, spec))
        report[airport] = census_airport(tracks, separation, spec.step_s)
        print(f"{airport}: {len(tracks)} flights, {report[airport]['losses']['episodes']} loss episodes, "
              f"A_max {report[airport]['samples']['A_max']}, {time.perf_counter() - started:.0f}s", flush=True)

    out.mkdir(parents=True)
    write_json_atomic(out / "census.json", {
        "schema": "ts-traffic-census-v1", "written_utc": utc_now(), "split": SPLIT, "only_airports": args.airports,
        "instructions": str(directory.relative_to(REPO_ROOT)), "spec_sha256": spec.sha256,
        "git": git_state(), "seconds": time.perf_counter() - started, "airports": report})
    print(f"wrote {out / 'census.json'}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
