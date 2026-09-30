"""The scene's closed loop (multi-aircraft design §2.2, §3.4 layer 1): the aircraft of one airport on the scene's steps,
judged pairwise every step, and an aircraft that answers for a loss of separation ended. M0 step 5 runs it with every
flight flying its labelled words (`traffic_labelled`); M3 and M4 will run it with the prior's.

Two kinds of aircraft (design §2.2):

- **controlled** (`Controlled`) — flown by the loop: by an executor (on the labelled words, or on the prior's), or, as
  the control an executor is compared with, along its own recorded rows (`recorded`). It lives on the scene's steps
  (design §2.5: the executor's aircraft are on the steps): from the step its first row hangs on (`prior.scene.hang`),
  one state a step, so its whole flight is moved by at most half a step from its recorded time. It is in the scene to
  its own end (``outcome``: the executor judge's, `autopilot.judge`, or the glidepath lower edge) unless a loss of
  separation it answers for ends it first (`LOST_SEPARATION`);
- **replayed** (`traffic_census.Track`) — follows its record at its recorded times, interpolated at the step: the
  background arrivals, and flights with a sentence the executor cannot fly (contract C31). Never ended: a loss it answers
  for is counted, nothing more.

The aircraft are chained into segments by their spans, as the census chains them. At each step ``t`` of a segment, in
this order:

1. every landing in ``(t − step, t]`` — a controlled aircraft's threshold crossing (its outcome ``landed``), at the
   interpolated crossing; a replayed one's roster landing time — is checked against the established aircraft next
   behind it on the approach clock (`separation.wake_at_threshold`, the same under both readings) with the others at
   that instant: a controlled one interpolated between its states at ``t − step`` and ``t`` (its angle, side, runway
   and capture those of ``t − step``), a replayed one from its rows. A controlled follower under the minimum is ended at
   the landing;
2. the aircraft in the scene at ``t`` are judged pairwise under the loop's reading (`separation.losses`); a pair's loss
   over consecutive steps is one episode (as in the census);
3. every controlled aircraft that answers for a loss at ``t`` is ended at ``t`` — two when both are still being vectored
   (§3.4); one whose own end is ``t`` leaves after it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from geokit import NM_M

from ts_transformer.experiments.traffic_census import Track, traffic_at
from ts_transformer.inference.runway_schedule import Separation
from ts_transformer.inference.separation import AT_THRESHOLD, Traffic, losses, wake_at_threshold
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import wrap180
from ts_transformer.prior.scene import Presence, hang, scene_steps

#: The outcome of a controlled aircraft ended for a loss of separation it answers for (design §3.4 layer 1).
LOST_SEPARATION = "lost_separation"


@dataclass(frozen=True)
class Controlled:
    """A controlled aircraft: its state at each of its steps (``times_s``, one step apart from the step its first row
    hangs on), its own end (``outcome``) after the last, and its threshold crossing on the loop's clock (None: it did
    not land)."""

    presence: Presence                  # the recorded flight: its key and the runway it landed on. Its times are the
                                        # RECORDED ones, never the loop's: those are `times_s`
    times_s: np.ndarray                 # [S] epoch seconds
    e_m: np.ndarray                     # [S] airport frame
    n_m: np.ndarray
    height_m: np.ndarray
    runway: tuple[str, ...]             # [S] the runway in force
    along_m: np.ndarray                 # [S] position on the approach clock
    track_minus_course_deg: np.ndarray  # [S] −180–180°
    right_of_course_m: np.ndarray       # [S]
    along_speed_mps: np.ndarray         # [S] its speed along its course: ground speed × cos(track − course)
    established: np.ndarray             # [S] bool
    category: str | None
    outcome: str
    landing_s: float | None

    def __post_init__(self) -> None:
        count = len(self.times_s)
        if count == 0 or not all(len(values) == count for values in (
                self.e_m, self.n_m, self.height_m, self.runway, self.along_m, self.track_minus_course_deg,
                self.right_of_course_m, self.along_speed_mps, self.established)):
            raise ValueError(f"{self.key}: a controlled aircraft holds one value per step, and at least one step")

    @property
    def key(self) -> str:
        return self.presence.dataset_id

    @property
    def first_step_s(self) -> float:
        return float(self.times_s[0])

    @property
    def last_step_s(self) -> float:
        return float(self.times_s[-1])

    def on_step(self, t_s: float) -> bool:
        return self.first_step_s <= t_s <= self.last_step_s


def _on_runway(presence: Presence, first_step_s: float, step_s: float, e_m: np.ndarray, n_m: np.ndarray,
               height_m: np.ndarray, track_deg: np.ndarray, ground_speed_mps: np.ndarray, established: np.ndarray,
               runway: str, geometry: AirportGeometry, separation: Separation, category: str | None, outcome: str,
               landing_s: float | None) -> Controlled:
    candidate = geometry.candidates[geometry.candidate_index(runway)]
    relative = relative_to_runway(e_m, n_m, track_deg, height_m, candidate)
    angle = np.asarray(relative.track_minus_course_deg, dtype=np.float64)
    return Controlled(presence, first_step_s + step_s * np.arange(len(e_m)), np.asarray(e_m, dtype=np.float64),
                      np.asarray(n_m, dtype=np.float64), np.asarray(height_m, dtype=np.float64), (runway,) * len(e_m),
                      separation.along_nm[runway] * NM_M - relative.before_threshold_m, angle,
                      relative.right_of_course_m,
                      np.asarray(ground_speed_mps, dtype=np.float64) * np.cos(np.radians(angle)),
                      np.asarray(established, dtype=bool), category, outcome, landing_s)


def flown(presence: Presence, step_s: float, e_m: np.ndarray, n_m: np.ndarray, height_m: np.ndarray,
          track_deg: np.ndarray, ground_speed_mps: np.ndarray, captured: np.ndarray, runway: str,
          geometry: AirportGeometry, separation: Separation, category: str | None, outcome: str,
          landing_from_first_s: float | None) -> Controlled:
    """An aircraft flown from the step its first row hangs on, on one runway throughout (the labelled words say it once):
    its states at its steps (``track_deg`` compass), the executor's capture at each (the established flag, design §3.2),
    and its threshold crossing counted from its first step (None: it did not land)."""
    first = float(hang(presence.start_s, step_s))
    return _on_runway(presence, first, step_s, e_m, n_m, height_m, track_deg, ground_speed_mps, captured, runway,
                      geometry, separation, category, outcome,
                      None if landing_from_first_s is None else first + landing_from_first_s)


def recorded(presence: Presence, flight: FlightSignals, capture_row: int, landing_from_first_row_s: float,
             geometry: AirportGeometry, separation: Separation, category: str | None, step_s: float) -> Controlled:
    """The control: a flight with a sentence along its own recorded rows, on the loop's steps like a flown one — row r at
    the step its first row hangs on plus r steps, established from the artefact's capture row, landed at its threshold
    crossing read off its own rows (``landing_from_first_row_s``, `autopilot.replay.observed_landing_s`: the last row
    carried to the threshold at its ground speed) — on the same clock as a flown aircraft's crossing, where the roster's
    landing time can fall seconds before the last row."""
    rows = len(presence.times_s)
    first = float(hang(presence.start_s, step_s))
    return _on_runway(presence, first, step_s, flight.e_m[:rows], flight.n_m[:rows], flight.altitude_m[:rows],
                      flight.track_deg[:rows], flight.ground_speed_mps[:rows], np.arange(rows) >= capture_row,
                      flight.runway, geometry, separation, category, "landed", first + landing_from_first_row_s)


def join(*parts: Traffic) -> Traffic:
    """Several groups of aircraft at one instant as one, in the order given."""
    return Traffic(np.concatenate([p.e_m for p in parts]), np.concatenate([p.n_m for p in parts]),
                   np.concatenate([p.height_m for p in parts]), tuple(r for p in parts for r in p.runway),
                   np.concatenate([p.along_m for p in parts]),
                   np.concatenate([p.track_minus_course_deg for p in parts]),
                   np.concatenate([p.right_of_course_m for p in parts]),
                   np.concatenate([np.asarray(p.established, dtype=bool) for p in parts]),
                   tuple(c for p in parts for c in p.category))


def at_steps(aircraft: Sequence[Controlled], t_s: float, step_s: float) -> Traffic:
    """Controlled aircraft at one of their steps."""
    rows = [(a, int(round((t_s - a.first_step_s) / step_s))) for a in aircraft]
    return Traffic(np.array([a.e_m[k] for a, k in rows]), np.array([a.n_m[k] for a, k in rows]),
                   np.array([a.height_m[k] for a, k in rows]), tuple(a.runway[k] for a, k in rows),
                   np.array([a.along_m[k] for a, k in rows]), np.array([a.track_minus_course_deg[k] for a, k in rows]),
                   np.array([a.right_of_course_m[k] for a, k in rows]),
                   np.array([bool(a.established[k]) for a, k in rows], dtype=bool),
                   tuple(a.category for a, _ in rows))


def between(aircraft: Sequence[Controlled], t_s: float, step_s: float) -> Traffic:
    """Controlled aircraft at an instant inside their spans: position and approach clock interpolated between the steps
    either side, everything else the earlier step's (module docstring, 1)."""
    e, n, h, along, runway, angle, right, established = [], [], [], [], [], [], [], []
    for a in aircraft:
        position = (t_s - a.first_step_s) / step_s
        k = min(int(math.floor(position)), len(a.times_s) - 1)
        later = min(k + 1, len(a.times_s) - 1)
        w = position - k
        for out, values in ((e, a.e_m), (n, a.n_m), (h, a.height_m), (along, a.along_m)):
            out.append(float(values[k] + w * (values[later] - values[k])))
        runway.append(a.runway[k])
        angle.append(float(a.track_minus_course_deg[k]))
        right.append(float(a.right_of_course_m[k]))
        established.append(bool(a.established[k]))
    return Traffic(np.array(e), np.array(n), np.array(h), tuple(runway), np.array(along), np.array(angle),
                   np.array(right), np.array(established, dtype=bool), tuple(a.category for a in aircraft))


def over_threshold(leader: Controlled | Track, separation: Separation) -> Traffic:
    """A landing aircraft over its threshold: on the approach clock at its threshold, established, its position, angle
    and side its last step's (or row's) — the threshold's rule reads only the approach clock."""
    if isinstance(leader, Controlled):
        runway, e, n, h = leader.runway[-1], leader.e_m[-1], leader.n_m[-1], leader.height_m[-1]
        angle, right = leader.track_minus_course_deg[-1], leader.right_of_course_m[-1]
    else:
        runway, e, n, h = leader.presence.runway, leader.e_m[-1], leader.n_m[-1], leader.height_m[-1]
        angle = wrap180(float(leader.track_minus_course_deg[-1]))
        right = leader.right_of_course_m[-1]
    return Traffic(np.array([e]), np.array([n]), np.array([h]), (runway,),
                   np.array([separation.along_nm[runway] * NM_M]), np.array([float(angle)]), np.array([right]),
                   np.array([True]), (leader.category,))


@dataclass
class Run:
    """One pass of the loop over an airport's aircraft under one reading: who was ended and why (by key), every episode,
    every wake loss at a landing, and the scene's time (the segments' spans) and steps judged (two or more aircraft)."""

    reading: str
    ended: dict[str, dict[str, Any]] = field(default_factory=dict)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    at_threshold: list[dict[str, Any]] = field(default_factory=list)
    landings_checked: int = 0
    scene_seconds: float = 0.0
    steps_judged: int = 0


def segments(aircraft: Sequence[Controlled | Track]) -> list[list[Controlled | Track]]:
    """The aircraft chained by their spans on the steps: one whose first step is no later than the segment's last joins
    it (`prior.scene.SceneIndex.segments`' rule)."""
    out: list[list[Controlled | Track]] = []
    end = -math.inf
    for item in sorted(aircraft, key=lambda a: (a.first_step_s, a.key)):
        if item.first_step_s > end:
            out.append([])
        out[-1].append(item)
        end = max(end, item.last_step_s)
    return out


class Judging:
    """One segment judged a step at a time into ``out`` (module docstring, 1–3): at each step ``t``, `landing` for every
    landing in ``(t − step, t]``, then `step`. `Loop` walks whole flights with it; a closed loop whose aircraft react to
    one another steps it as it flies (`traffic_window`: an aircraft ended there flies on, and stops speaking).

    Three kinds of aircraft are handed to it: **controlled** ones still judged (a loss or wake shortfall they answer for
    ends them), **replayed** ones (`Track`, at their records) and **passive** ones — controlled aircraft already ended that
    are still in the scene (multi-aircraft design §6.6 step 7 item 3, §9 item 29): on their states like a controlled
    one, judged like a replayed one (a loss they answer for is recorded, nothing more)."""

    def __init__(self, separation: Separation, reading: str, step_s: float, out: Run) -> None:
        self.separation, self.reading, self.step_s, self.out = separation, reading, step_s, out
        self.open_episodes: dict[tuple[str, str], dict[str, Any]] = {}

    def step(self, t_s: float, controlled: Sequence[Controlled], replayed: Sequence[Track],
             passive: Sequence[Controlled] = ()) -> dict[str, dict[str, Any]]:
        """The aircraft on step ``t_s`` judged pairwise (fewer than two: nothing): each controlled one answering for a loss
        there is ended (recorded in ``out.ended`` and returned)."""
        if len(controlled) + len(passive) + len(replayed) < 2:
            return {}
        out, step = self.out, self.step_s
        out.steps_judged += 1
        here: list[Controlled | Track] = [*controlled, *passive, *replayed]
        found = losses(join(at_steps([*controlled, *passive], t_s, step), traffic_at(replayed, t_s)), self.separation,
                       self.reading)
        ended_now: dict[str, dict[str, Any]] = {}
        judged = len(controlled)
        for loss in found:
            pair = tuple(sorted((here[loss.i].key, here[loss.j].key)))
            episode = self.open_episodes.get(pair)
            if episode is None or episode["last_s"] != t_s - step:
                episode = {"pair": list(pair), "relation": loss.relation, "first_s": t_s, "steps": 0, "kinds": [],
                           "min_ratio": math.inf, "closest_m": None, "required_m": None, "wake_known": True,
                           "responsible": [], "ended": []}
                self.open_episodes[pair] = episode
                out.episodes.append(episode)
            episode["steps"] += 1
            episode["last_s"] = t_s
            if loss.distance_m / loss.required_m < episode["min_ratio"]:
                episode["min_ratio"] = loss.distance_m / loss.required_m
                episode["closest_m"], episode["required_m"] = loss.distance_m, loss.required_m
            episode["wake_known"] = episode["wake_known"] and loss.wake_known
            if loss.kind not in episode["kinds"]:
                episode["kinds"].append(loss.kind)
            for k in loss.responsible:
                answer = {"key": here[k].key, "controlled": k < judged}
                if answer not in episode["responsible"]:
                    episode["responsible"].append(answer)
                if k < judged and here[k].key not in ended_now:
                    other = loss.j if k == loss.i else loss.i
                    ended_now[here[k].key] = {"t_s": t_s, "kind": loss.kind, "relation": loss.relation,
                                              "with": here[other].key, "with_controlled": other < judged}
                    episode["ended"].append(here[k].key)
        out.ended.update(ended_now)
        return ended_now

    def landing(self, t_s: float, leader: Controlled | Track, controlled: Sequence[Controlled],
                replayed: Sequence[Track], passive: Sequence[Controlled] = (), *, leader_passive: bool = False) -> None:
        """``leader`` landing at ``t_s``, checked against the established aircraft next behind it among the others there
        (none the leader): a controlled follower under the minimum is ended at the landing. ``leader_passive``: the
        leader is a passive aircraft (recorded as not controlled, as a passive partner is in `step`)."""
        out = self.out
        out.landings_checked += 1
        scene = join(between([*controlled, *passive], t_s, self.step_s), traffic_at(replayed, t_s),
                     over_threshold(leader, self.separation))
        loss = wake_at_threshold(scene, len(controlled) + len(passive) + len(replayed), self.separation)
        if loss is None:
            return
        others: list[Controlled | Track] = [*controlled, *passive, *replayed]
        follower = others[loss.j]
        judged = loss.j < len(controlled)
        leader_judged = isinstance(leader, Controlled) and not leader_passive
        out.at_threshold.append({"t_s": t_s, "leader": leader.key, "leader_controlled": leader_judged,
                                 "follower": follower.key, "follower_controlled": judged, "gap_m": loss.distance_m,
                                 "required_m": loss.required_m, "relation": loss.relation})
        if judged:
            out.ended[follower.key] = {"t_s": t_s, "kind": AT_THRESHOLD, "relation": loss.relation,
                                       "with": leader.key, "with_controlled": leader_judged}


class Loop:
    """The loop's judge over one airport (module docstring): `run` walks every segment of ``controlled`` and
    ``replayed`` (`Judging`), an aircraft ended there leaving the scene."""

    def __init__(self, separation: Separation, reading: str, step_s: float) -> None:
        self.separation, self.reading, self.step_s = separation, reading, step_s

    def run(self, controlled: Sequence[Controlled], replayed: Sequence[Track]) -> Run:
        keys = [a.key for a in (*controlled, *replayed)]
        if len(set(keys)) != len(keys):
            raise ValueError("an aircraft is in the loop twice")
        out = Run(self.reading)
        for segment in segments([*controlled, *replayed]):
            self._segment(segment, out)
        return out

    def _segment(self, segment: list[Controlled | Track], out: Run) -> None:
        step = self.step_s
        flown_here = [a for a in segment if isinstance(a, Controlled)]
        replayed_here = [a for a in segment if not isinstance(a, Controlled)]
        start, end = min(a.first_step_s for a in segment), max(a.last_step_s for a in segment)
        out.scene_seconds += end - start
        landings = sorted([(a.landing_s, a.key, a) for a in flown_here if a.landing_s is not None]
                          + [(a.presence.landing_s, a.key, a) for a in replayed_here], key=lambda x: (x[0], x[1]))
        # a landing after the segment's last step (its last row is before the crossing) is still checked
        last = max(end, math.ceil(landings[-1][0] / step) * step) if landings else end
        judging = Judging(self.separation, self.reading, step, out)
        pending = 0
        for t_s in scene_steps(start, last, step):
            t_s = float(t_s)
            while pending < len(landings) and landings[pending][0] <= t_s:
                at, _, leader = landings[pending]
                if leader.key not in out.ended:
                    judging.landing(at, leader,
                                    [a for a in flown_here
                                     if a.key != leader.key and a.on_step(at) and a.key not in out.ended],
                                    [a for a in replayed_here
                                     if a.key != leader.key and a.presence.times_s[0] <= at <= a.presence.times_s[-1]])
                pending += 1
            judging.step(t_s, [a for a in flown_here if a.on_step(t_s) and a.key not in out.ended],
                         [a for a in replayed_here if a.on_step(t_s)])
