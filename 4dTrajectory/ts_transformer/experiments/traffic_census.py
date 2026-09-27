"""Multi-aircraft M0 (design §6.4 step 4): the observed traffic of the TRAINING days, judged as the closed loop will be.

From one instruction artefact split by operating day and each airport's arrivals manifest (its runway targets, for the
separation rules; checked to be the manifest the artefact's signals were read from), every arrival of the training days
— with a sentence or without (background) — is in its airport's scene on the steps its rows hang on (§2.1: even UTC
seconds, `prior.scene.hung_span`), and at every step with two or more aircraft they are judged by `inference.separation`
under both readings — `VISUAL`, which the closed loop's checks and the reward use, and `IFR`, reported beside it:

- an aircraft's position is interpolated at the step from its rows (at most half a step past its first or last row, held
  there); its runway is the one it landed on (the observed census is a measurement, not a model input); its position on
  the approach clock is its threshold's `Separation.along_nm` less its distance before that threshold along the course;
  it is established on the final from its capture row on — the artefact's for a flight with a sentence, the labeller's
  own rule (`capture_row` on `admit`'s smoothed track) for a background flight `admit` takes, the same rule on the raw
  track for one it refuses (each group counted); a flight that never stays in the corridor is never established; its CWT
  category comes from its type (`runway_schedule.wake_category`; a flight without a type is counted);
- a pair's loss over consecutive steps is one EPISODE, whatever the kind at each step (the kinds seen are listed); the
  pairs with a loss are counted beside the episodes;
- at every landing (the leader over its threshold at its roster landing time) the established aircraft next behind it
  gives a landing interval on the approach clock, the distance the rules require (radar and TBL 5-5-2, design §2.5 / §9
  item 6), whether it was below it, and whether TBL 5-5-2 alone was broken (`wake_at_threshold`);
- on every final, consecutive established aircraft that both have a sentence give a closing speed: the rates along the
  landing direction, ground speed × cos(track − course), of the one behind less the one ahead (pairs with a background
  flight are left out and counted: its ground-speed channel is what the labeller may have refused it for);
- order swaps (§4.1): two arrivals in the scene together that land on the same runway, or on two same-direction runways,
  where the one that entered later landed first;
- segment lengths; the samples each segment is cut into (§2.3, `prior.scene.samples`) and the windows of the window
  setting (§2.2: 20 minutes every 10 minutes), each counted as the model's aircraft axis counts them (every flight with a
  step in it) — ``A_max`` is the most of either — and each window's flights with a sentence entering it, and aircraft
  already in the scene when it opens.

The flights the executor cannot fly (contract C31) are counted by step 5, which rebuilds every flight for the executor.
A type the tables do not list stops the run before any judging: it writes ``types_not_listed.json`` into ``--out`` (so a
rerun, after each type is added to `docs/literature/arrival_separation/cwt_supplement.csv` with its source, needs a new
directory). Only training days are read. Writes ``census.json`` into a NEW directory.

    python run_ts.py traffic_census --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --out 4dTrajectory/outputs/POOLED/traffic/census_<date>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from geokit import NM_M

from ts_transformer.inference.runway_schedule import SAME, Separation, faa_separation, read_cwt_tables, wake_category
from ts_transformer.inference.separation import READINGS, VISUAL, Traffic, losses, next_behind, wake_at_threshold
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import (
    arrival_manifest_sha256s, load_candidates, load_sentences, load_signals, load_spec,
)
from ts_transformer.instructions.labeller.lateral import capture_row
from ts_transformer.instructions.labeller.read import admit
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.scene import Presence, SceneIndex, hung_span, presence, samples, scene_steps
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, git_state, repo_relative

SPLIT = "train"
#: `faa_separation` turns distances into times at an approach speed; nothing here reads a time, so any speed does.
APPROACH_SPEED_MPS = 70.0
#: The window setting (design §2.2): a window this long opens every `WINDOW_EVERY_S`.
WINDOW_S, WINDOW_EVERY_S = 1_200.0, 600.0


@dataclass(frozen=True)
class Track:
    """One arrival as the census judges it: its rows in the scene (epoch seconds) and the steps they hang on."""

    presence: Presence
    first_step_s: float
    last_step_s: float
    e_m: np.ndarray
    n_m: np.ndarray
    height_m: np.ndarray
    along_m: np.ndarray            # its position on the approach clock at each row
    along_rate_mps: np.ndarray     # ground speed × cos(track − course) at each row
    captured_s: float              # the time of its capture row; inf when it is never established
    capture_from: str              # "artefact", "admitted" or "raw" (`track`)
    category: str | None

    @property
    def key(self) -> str:
        return self.presence.dataset_id

    def on_step(self, t_s: float) -> bool:
        return self.first_step_s <= t_s <= self.last_step_s


def _quantiles(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0}
    return {"n": int(len(array)), "min": float(array.min()), "p1": float(np.percentile(array, 1)),
            "p5": float(np.percentile(array, 5)), "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def track(flight: FlightSignals, sentence_rows: int | None, capture: int | None, geometry: AirportGeometry,
          along_threshold_m: float, spec: VocabularySpec, step_s: float) -> Track:
    """``capture``: the artefact's capture row of a flight with a sentence; None for a background flight, whose capture
    the labeller's own rule finds (or not) on `admit`'s smoothed track, or on the raw track where `admit` refuses it."""
    seen = presence(flight, sentence_rows, geometry)
    rows = len(seen.times_s)
    candidate = geometry.candidates[geometry.candidate_index(flight.runway)]
    raw = relative_to_runway(flight.e_m[:rows], flight.n_m[:rows], flight.track_deg[:rows], flight.altitude_m[:rows],
                             candidate)
    source = "artefact"
    if capture is None:
        try:
            relative, source = admit(flight, geometry, spec).relative, "admitted"    # its rows: a prefix of these
        except Refused:
            relative, source = raw, "raw"
        try:
            capture = capture_row(relative, spec)
        except Refused:             # never stays in the corridor to the end: never established
            capture = None
    rate = flight.ground_speed_mps[:rows] * np.cos(np.radians(flight.track_deg[:rows] - candidate.course_deg))
    first, last = hung_span(seen, step_s)
    return Track(seen, first, last, flight.e_m[:rows], flight.n_m[:rows], flight.altitude_m[:rows],
                 along_threshold_m - raw.before_threshold_m, rate,
                 float(seen.times_s[capture]) if capture is not None else math.inf, source,
                 None if flight.typecode is None else wake_category(flight.typecode))


def _traffic(tracks: Sequence[Track], t_s: float) -> Traffic:
    def at(values: np.ndarray, item: Track) -> float:
        return float(np.interp(t_s, item.presence.times_s, values))

    return Traffic(np.array([at(t.e_m, t) for t in tracks]), np.array([at(t.n_m, t) for t in tracks]),
                   np.array([at(t.height_m, t) for t in tracks]), tuple(t.presence.runway for t in tracks),
                   np.array([at(t.along_m, t) for t in tracks]), np.array([t_s >= t.captured_s for t in tracks]),
                   tuple(t.category for t in tracks))


def _record(episodes: list[dict[str, Any]], open_episodes: dict[tuple[str, str], dict[str, Any]], here: list[Track],
            found, t_s: float, step_s: float) -> None:
    """Add this step's losses to the episodes: a pair's loss over consecutive steps is one episode."""
    for loss in found:
        pair = tuple(sorted((here[loss.i].key, here[loss.j].key)))
        episode = open_episodes.get(pair)
        if episode is None or episode["last_s"] != t_s - step_s:
            episode = {"pair": list(pair), "relation": loss.relation, "first_s": float(t_s), "steps": 0,
                       "kinds": [], "min_ratio": math.inf, "wake_known": True, "responsible": []}
            open_episodes[pair] = episode
            episodes.append(episode)
        episode["steps"] += 1
        episode["last_s"] = float(t_s)
        episode["min_ratio"] = min(episode["min_ratio"], loss.distance_m / loss.required_m)
        episode["wake_known"] = episode["wake_known"] and loss.wake_known
        if loss.kind not in episode["kinds"]:
            episode["kinds"].append(loss.kind)
        for k in loss.responsible:
            answer = {"key": here[k].key, "speaking": here[k].presence.speaking}
            if answer not in episode["responsible"]:
                episode["responsible"].append(answer)


def census_airport(tracks: list[Track], separation: Separation, step_s: float) -> dict[str, Any]:
    by_key = {t.key: t for t in tracks}
    index = SceneIndex([t.presence for t in tracks])
    segments = index.segments(step_s)
    episodes: dict[str, list[dict[str, Any]]] = {reading: [] for reading in READINGS}
    open_episodes: dict[str, dict[tuple[str, str], dict[str, Any]]] = {reading: {} for reading in READINGS}
    together: set[tuple[str, str]] = set()
    closing: list[float] = []
    closing_left_out = judged_steps = 0
    scene_seconds = 0.0
    for segment in segments:
        members = [by_key[p.dataset_id] for p in segment]
        start, end = min(m.first_step_s for m in members), max(m.last_step_s for m in members)
        scene_seconds += end - start
        for t_s in scene_steps(start, end, step_s):
            here = [m for m in members if m.on_step(t_s)]
            if len(here) < 2:
                continue
            judged_steps += 1
            traffic = _traffic(here, float(t_s))
            together.update(tuple(sorted((here[a].key, here[b].key)))
                            for a in range(len(here)) for b in range(a + 1, len(here)))
            for reading in READINGS:
                _record(episodes[reading], open_episodes[reading], here, losses(traffic, separation, reading),
                        float(t_s), step_s)
            # closing speeds between consecutive established aircraft on one runway
            for k in range(len(here)):
                if not traffic.established[k]:
                    continue
                follower = next_behind(traffic, k, separation, VISUAL)
                if follower is None:
                    continue
                if not (here[k].presence.speaking and here[follower].presence.speaking):
                    closing_left_out += 1
                    continue
                closing.append(float(np.interp(t_s, here[follower].presence.times_s, here[follower].along_rate_mps)
                                     - np.interp(t_s, here[k].presence.times_s, here[k].along_rate_mps)))

    # at every landing: the established aircraft next behind, its landing interval on the approach clock
    landings = {reading: {"judged": 0, "untyped": 0, "below_required": 0, "wake_losses": 0, "interval_m": [],
                          "interval_minus_required_m": [], "required_nm": Counter()} for reading in READINGS}
    for leader in tracks:
        t_s = leader.presence.landing_s
        near = [by_key[p.dataset_id] for p in index.overlapping(t_s, t_s) if p.dataset_id != leader.key]
        if not near:
            continue
        traffic = _traffic(near, t_s)
        # the leader over its threshold: on the approach clock at its threshold, established (its position is its last
        # row's; the threshold's rule reads only the approach clock)
        along_threshold = separation.along_nm[leader.presence.runway] * NM_M
        scene = Traffic(np.append(traffic.e_m, leader.e_m[-1]), np.append(traffic.n_m, leader.n_m[-1]),
                        np.append(traffic.height_m, leader.height_m[-1]), (*traffic.runway, leader.presence.runway),
                        np.append(traffic.along_m, along_threshold), np.append(traffic.established, True),
                        (*traffic.category, leader.category))
        lead = len(near)
        for reading, landing in landings.items():
            follower = next_behind(scene, lead, separation, reading)
            if follower is None:
                continue
            if leader.category is None or near[follower].category is None:
                landing["untyped"] += 1
                continue
            landing["judged"] += 1
            required_nm = separation.distance_nm(leader.presence.runway, leader.category,
                                                 near[follower].presence.runway, near[follower].category)
            landing["required_nm"][f"{required_nm:g}"] += 1
            interval = along_threshold - float(scene.along_m[follower])
            landing["interval_m"].append(interval)
            landing["interval_minus_required_m"].append(interval - required_nm * NM_M)
            landing["below_required"] += interval < required_nm * NM_M
            landing["wake_losses"] += wake_at_threshold(scene, lead, separation, reading) is not None

    # order swaps (§4.1)
    swaps = {"same_runway": [0, 0], "same_direction_other_runway": [0, 0]}
    involved: set[str] = set()
    for p in index.flights:
        for q in index.overlapping(p.start_s, p.end_s):
            if q.dataset_id <= p.dataset_id:
                continue
            if separation.relation(p.runway, q.runway) == SAME:
                group = "same_runway"
            elif frozenset((p.runway, q.runway)) in separation.relations:
                group = "same_direction_other_runway"
            else:
                continue
            swaps[group][1] += 1
            if (p.start_s - q.start_s) * (p.landing_s - q.landing_s) < 0:
                swaps[group][0] += 1
                involved.update((p.dataset_id, q.dataset_id))

    # segments, samples and windows: the aircraft axis counts every flight with a step inside
    segment_minutes, sample_flights, loss_minutes = [], [], []
    window_flights, entering, airborne = [], [], []
    for segment in segments:
        members = [by_key[p.dataset_id] for p in segment]
        start, end = min(m.first_step_s for m in members), max(m.last_step_s for m in members)
        segment_minutes.append((end - start) / 60.0)
        for sample in samples(segment, step_s):
            sample_flights.append(len(sample.flights))
            loss_minutes.append((sample.loss_end_s - sample.loss_start_s) / 60.0)
        opens = start
        while opens <= end:
            closes = opens + WINDOW_S
            window_flights.append(sum(1 for m in members if m.last_step_s >= opens and m.first_step_s < closes))
            entering.append(sum(1 for m in members if m.presence.speaking and opens <= m.first_step_s < closes))
            airborne.append(sum(1 for m in members if m.first_step_s < opens <= m.last_step_s))
            opens += WINDOW_EVERY_S

    hours = scene_seconds / 3600.0

    def judged(reading: str) -> dict[str, Any]:
        found, landing = episodes[reading], landings[reading]
        pairs = {tuple(e["pair"]) for e in found}
        return {
            "losses": {"episodes": len(found), "episodes_per_scene_hour": len(found) / hours,
                       "pairs_with_a_loss": len(pairs), "pairs_with_a_loss_per_scene_hour": len(pairs) / hours,
                       "pairs_with_a_loss_share": len(pairs) / len(together) if together else None,
                       "by_kind": dict(Counter(kind for e in found for kind in e["kinds"])),
                       "by_relation": dict(Counter(e["relation"] for e in found)),
                       "untyped_in_trail": sum(not e["wake_known"] for e in found),
                       "a_flight_with_a_sentence_answers": sum(any(r["speaking"] for r in e["responsible"])
                                                               for e in found),
                       "steps": _quantiles([e["steps"] for e in found]),
                       "min_distance_over_required": _quantiles([e["min_ratio"] for e in found])},
            "at_landing": {"judged": landing["judged"], "untyped": landing["untyped"],
                           "below_required": landing["below_required"], "wake_losses": landing["wake_losses"],
                           "interval_m": _quantiles(landing["interval_m"]),
                           "interval_minus_required_m": _quantiles(landing["interval_minus_required_m"]),
                           "required_nm": dict(sorted(landing["required_nm"].items())),
                           "required_is_the_radar_minimum_share": (
                               landing["required_nm"][f"{separation.same_nm:g}"] / landing["judged"]
                               if landing["judged"] else None)},
            "episodes": found,
        }

    return {
        "flights": {"speaking": sum(t.presence.speaking for t in tracks),
                    "background": sum(not t.presence.speaking for t in tracks),
                    "untyped": sum(t.category is None for t in tracks),
                    "background_capture_from": dict(Counter(t.capture_from for t in tracks if not t.presence.speaking)),
                    "background_never_established": sum(not t.presence.speaking and math.isinf(t.captured_s)
                                                        for t in tracks)},
        "scene_hours": hours, "steps_with_two_or_more": judged_steps,
        "segment_minutes": _quantiles(segment_minutes),
        "pairs_in_the_scene_together": len(together),
        "check_reading": VISUAL,
        "readings": {reading: judged(reading) for reading in READINGS},
        "closing_speed_on_a_final_mps": {**_quantiles(closing), "pairs_with_a_background_flight_left_out": closing_left_out},
        "order_swaps": {group: {"swapped": s, "pairs": n, "share": s / n if n else None}
                        for group, (s, n) in swaps.items()},
        "flights_in_a_swap_share": len(involved) / len(tracks),
        "samples": {"count": len(sample_flights), "flights": _quantiles(sample_flights),
                    "loss_minutes": _quantiles(loss_minutes)},
        "windows": {"count": len(window_flights), "flights": _quantiles(window_flights),
                    "entering_with_a_sentence": _quantiles(entering), "airborne_at_opening": _quantiles(airborne)},
        "A_max": max(max(sample_flights), max(window_flights)),
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
        raise SystemExit(f"{len(missing)} types are in neither CWT table: {missing} (listed in "
                         f"{out / 'types_not_listed.json'}; add each to the supplement with its source and rerun "
                         "into a new directory)")

    recorded = arrival_manifest_sha256s(directory)
    by_airport: dict[str, list[tuple[int, FlightSignals]]] = defaultdict(list)
    for i, flight in wanted:
        by_airport[flight.airport].append((i, flight))
    report: dict[str, Any] = {}
    manifests: dict[str, str] = {}
    for airport in sorted(by_airport):
        manifest = arrival_manifest_path(airport)
        if file_sha256(manifest) != recorded[airport]:
            raise ValueError(f"{manifest} is not the manifest the artefact's signals were read from "
                             f"(sha256 {recorded[airport][:12]} recorded)")
        manifests[airport] = recorded[airport]
        separation = faa_separation(json.loads(manifest.read_text(encoding="utf-8"))["runway_targets"],
                                    speed_mps=APPROACH_SPEED_MPS)
        tracks = [track(flight, *spoken.get(i, (None, None)), candidates[airport],
                        separation.along_nm[flight.runway] * NM_M, spec, spec.step_s)
                  for i, flight in by_airport[airport]]
        report[airport] = census_airport(tracks, separation, spec.step_s)
        found = {reading: report[airport]["readings"][reading]["losses"]["pairs_with_a_loss"] for reading in READINGS}
        print(f"{airport}: {len(tracks)} flights, pairs with a loss {found}, A_max {report[airport]['A_max']}, "
              f"{time.perf_counter() - started:.0f}s", flush=True)

    payload = {"schema": "ts-traffic-census-v1", "written_utc": utc_now(), "split": SPLIT,
               "only_airports": args.airports, "instructions": repo_relative(directory), "spec_sha256": spec.sha256,
               "arrival_manifest_sha256": manifests, "git": git_state(), "seconds": time.perf_counter() - started,
               "airports": report}
    out.mkdir(parents=True)
    write_json_atomic(out / "census.json", payload)
    print(f"wrote {out / 'census.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
