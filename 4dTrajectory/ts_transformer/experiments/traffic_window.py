"""The prior commanding every aircraft of a time window (multi-aircraft design §2.2 "窗口内由模型指挥", §6.6 step 7) —
shared by M3's and M4's second pass, not a runner.

**Windows** (step 7 item 1): an airport's flights are chained into segments on the steps their rows hang on
(`prior.scene.SceneIndex.segments`, design §2.3); from each segment's first step a window opens every `WINDOW_EVERY_S`
up to its last step, `WINDOW_S` long. The prior commands the flights with a sentence whose first row hangs on a step
inside the window and that fly on their own dynamics (the replay gate's group, `autopilot.replay`); everything else in
the air — before, after, the background, a flight the executor cannot fly — is replayed along its record. Windows
overlap, so a flight is commanded in two of them, once in each. A window's scene runs from its first commanded aircraft's
first row to the last one's time limit.

What the loop reads is placed by the one-aircraft code (`traffic_speaking`, in the edge features' source hash): each
commanded aircraft's view of its window is a one-speaking `traffic_speaking.Scene` (`Window.scene`) — none of the files
in `traffic_scene_data.EDGE_SOURCES` is changed, so every traffic prior trained so far still loads.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ts_transformer.autopilot import replay
from ts_transformer.experiments.traffic_census import Track
from ts_transformer.experiments.traffic_scene_data import FlightRows
from ts_transformer.experiments.traffic_speaking import Scene, SceneAirport
from ts_transformer.instructions.artefact import load_sentences, load_signals
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import Words
from ts_transformer.prior.scene import N_LOOK, SceneIndex, hang, hung_span

#: A window's length and how often one opens (design §8: 20 minutes, every 10).
WINDOW_S = 1_200.0
WINDOW_EVERY_S = 600.0
#: Windows whose flights are rebuilt at a time while drawing (a rebuild opens the flights' tracks).
DRAW_CHUNK = 64


@dataclass(frozen=True)
class Window:
    """One window of an airport's scene: the step it opens on, the flights the prior commands (by the step their first
    row hangs on, then key), the others replayed (in the air at a step from the first commanded aircraft's first row to
    the end of the last one's time limit) and — an augmented window's — flights moved in time or inserted (their rows
    and tracks, by their keys among the commanded or the others)."""

    airport: SceneAirport
    opens_s: float
    commanded: tuple[str, ...]
    others: tuple[str, ...]
    moved: tuple[tuple[FlightRows, Track], ...] = ()

    def rows(self, key: str) -> FlightRows:
        """A flight of the window as the samples place it (a moved one's own)."""
        return next((rows for rows, _ in self.moved if rows.presence.dataset_id == key), None) \
            or self.airport.flights.flights[key]

    def track(self, key: str) -> Track:
        """A flight of the window as the judge replays it (a moved one's own)."""
        return next((track for _, track in self.moved if track.key == key), None) or self.airport.tracks[key]

    def first_step_s(self, key: str, step_s: float) -> float:
        """The step a commanded aircraft's first row hangs on."""
        return float(hang(self.rows(key).presence.start_s, step_s))

    def scene(self, key: str, step_s: float) -> Scene:
        """Commanded aircraft ``key``'s view of the window as a one-speaking scene: it speaking, every other aircraft of
        the window among the others — the other commanded ones first, in the window's order, then the replayed."""
        if self.moved:
            raise NotImplementedError("an augmented window's scene view (design §6.6 step 7.5)")
        others = tuple(k for k in self.commanded if k != key) + self.others
        return Scene(self.airport, key, self.first_step_s(key, step_s), others)


def window_tiles(airport: SceneAirport, step_s: float) -> list[tuple[float, tuple[str, ...]]]:
    """Every window of ``airport``'s segments (module docstring) that holds a flight with a sentence: its opening step and
    those flights, by the step their first row hangs on (then key) — which of them fly is the draw's to find."""
    out = []
    for segment in SceneIndex(f.presence for f in airport.flights.flights.values()).segments(step_s):
        spans = [hung_span(p, step_s) for p in segment]
        first, last = min(s for s, _ in spans), max(e for _, e in spans)
        entering = sorted((s, p.dataset_id) for p, (s, _) in zip(segment, spans) if p.speaking)
        opens = first
        while opens <= last:
            keys = tuple(k for s, k in entering if opens <= s < opens + WINDOW_S)
            if keys:
                out.append((opens, keys))
            opens += WINDOW_EVERY_S
    return out


def window_of(airport: SceneAirport, opens_s: float, commanded: Sequence[str], limits_s: Sequence[float],
              step_s: float) -> Window:
    """The window opening at ``opens_s`` with ``commanded`` (each flying ``limits_s`` from its first predicted step):
    everything else in the air from the first one's first row to the end of the last one's time limit replayed."""
    firsts = [float(hang(airport.flights.flights[k].presence.start_s, step_s)) for k in commanded]
    start = min(firsts)
    end = max(first + N_LOOK * step_s + limit for first, limit in zip(firsts, limits_s))
    taken = set(commanded)
    others = tuple(k for k, t in airport.tracks.items()
                   if k not in taken and t.first_step_s <= end and t.last_step_s >= start)
    return Window(airport, opens_s, tuple(commanded), others)


@dataclass(frozen=True)
class WindowDraw:
    """Windows drawn from a split (`draw_windows`): each window's opening and commanded flights, the commanded flights
    as one replay batch (window after window, each window's in its order) and what the draw read."""

    openings: list[tuple[str, float, tuple[str, ...]]]     # (airport, opening step, commanded) per window
    batch: replay.Batch
    counts: dict[str, Any]

    def members(self) -> list[range]:
        """Each window's commanded flights' places in `batch`."""
        out, at = [], 0
        for _, _, commanded in self.openings:
            out.append(range(at, at + len(commanded)))
            at += len(commanded)
        return out


def draw_windows(directory: Path, split: str, spec: VocabularySpec, words: Words, airports: Mapping[str, SceneAirport],
                 *, per_airport: int, seed: int, step_s: float) -> WindowDraw:
    """``per_airport`` windows of each airport of ``airports`` (in order), drawn from its tiles (`window_tiles`) in a
    permutation of one generator seeded ``seed``: in each, the flights that fly on their own dynamics are commanded (each
    re-read and checked against its stored sentence, as `replay.draw` does); a tile none of whose flights flies is passed
    over and counted; refused when an airport runs short."""
    sentences = load_sentences(directory, split, spec)
    stored = {int(index): k for k, index in enumerate(sentences["signal_index"])}
    index = {s.dataset_id: i for i, s in enumerate(load_signals(directory, split))}
    rng = np.random.default_rng(seed)
    openings: list[tuple[str, float, tuple[str, ...]]] = []
    parts: list[replay.Batch] = []
    counts: dict[str, Any] = {"split": split, "seed": seed, "per_airport": per_airport, "airports": {}}
    for code, airport in airports.items():
        tiles = window_tiles(airport, step_s)
        order = rng.permutation(len(tiles))
        taken, read, passed = 0, 0, 0
        # of the windows taken: their flights with a sentence, and those replayed as the executor cannot fly them
        entering_unflown: Counter = Counter(entering=0, not_flown=0)
        for start in range(0, len(order), DRAW_CHUNK):
            if taken == per_airport:
                break
            chunk = [tiles[int(i)] for i in order[start: start + DRAW_CHUNK]]
            candidates = sorted({index[k] for _, keys in chunk for k in keys})
            drawn = replay.draw_flights(directory, split, candidates, per_airport=0, seed=0)
            flies = {s.dataset_id: n for n, s in enumerate(drawn.signals)}
            readings: dict[str, Any] = {}
            keep: list[int] = []
            for opens, keys in chunk:
                if taken == per_airport:
                    break
                read += 1
                commanded = tuple(k for k in keys if k in flies)
                if not commanded:
                    passed += 1
                    continue
                entering_unflown.update({"entering": len(keys), "not_flown": len(keys) - len(commanded)})
                for k in commanded:
                    if k not in readings:
                        readings[k] = _reading(drawn.signals[flies[k]], index[k], drawn.geometries, sentences, stored,
                                               spec, words)
                    keep.append(flies[k])
                openings.append((code, opens, commanded))
                taken += 1
            if keep:
                parts.append(replay.batch_of(drawn, keep, [readings[drawn.signals[n].dataset_id] for n in keep]))
        if taken < per_airport:
            raise ValueError(f"{code}: {taken} windows with a flight that flies, {per_airport} wanted")
        counts["airports"][code] = {"tiles": len(tiles), "read": read, "passed_over": passed, "windows": taken,
                                    "commanded": sum(len(c) for a, _, c in openings if a == code),
                                    **dict(entering_unflown)}
    batch = replay.Batch(**{f.name: [item for part in parts for item in getattr(part, f.name)]
                            for f in dataclasses.fields(replay.Batch) if f.name != "drawn"}, drawn=counts)
    return WindowDraw(openings, batch, counts)


def _reading(signals: Any, i: int, geometries: Mapping[str, Any], sentences: Mapping[str, np.ndarray],
             stored: Mapping[int, int], spec: VocabularySpec, words: Words) -> Any:
    """A commanded flight re-read and checked against its stored sentence (MIRROR of `replay.draw`'s check: the replay
    module is in the executor's source hash)."""
    reading = read_flight(signals, geometries[signals.airport], spec, words)
    k = stored[i]
    grid = sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]]
    if not np.array_equal(reading.words, grid) or reading.runway_index != int(sentences["runway_index"][k]):
        raise ValueError(f"{signals.dataset_id}: the re-read sentence differs from the stored one")
    return reading
