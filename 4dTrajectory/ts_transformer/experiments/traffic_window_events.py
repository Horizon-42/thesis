"""Multi-aircraft step 8's hard events as windows to fly again (multi-aircraft design §6.6 step 8 item 11, the user's
2026-10-02 decision).

A **hard event** is a loss of separation of a window rewind run (R43, `traffic_window_rewind`) that its answered aircraft
— the one the judge ended — could not undo speaking again from its first predicted step: none of its `start` branches
rescued. Only an answered aircraft's (an episode that ended both aircraft has no one answerable).

A **scene** is its window flown again as that branch flew it: every other commanded aircraft given the words it said in
the original pass as far as it spoke there (`traffic_window.Given`, past them the prior speaking for one still flying),
the answered aircraft spoken by the model from its first predicted step. Without the others' words the same conflict
seldom happens again; with them the hard part stays.

`event_pool` rebuilds a rewind run's draw (its split, seed and windows an airport; its augmentation, if any) from the
instruction artefact — the windows themselves are not stored — refuses a run whose draw does not come back as it was
(its counts, its augmentation's record, each hard event's window: airport, commanded aircraft and augmentation) or that
holds no hard event, and keeps its hard events' windows and scenes. `pick` draws a round's scenes, at most so many an
airport.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ts_transformer.experiments.traffic_window import Given, draw_windows
from ts_transformer.experiments.traffic_window_generation import (
    Drawn, augmented_windows, drawn_subset, drawn_windows,
)
from ts_transformer.experiments.traffic_window_rewind import (
    ANSWERED, SCHEMA as REWIND_SCHEMA, START, read_original_words,
)
from ts_transformer.repo_layout import REPO_ROOT, repo_relative


@dataclasses.dataclass(frozen=True)
class EventScene:
    """A hard event (module docstring) as a window to fly again: its event number in the rewind run, its window (an index
    into the pool's ``drawn``: the run's hard windows alone), its airport, the answered aircraft and every commanded
    aircraft's given line in the window's order (None: the answered one, spoken)."""

    event: int
    window: int
    airport: str
    answered: str
    given: tuple[Given | None, ...]


@dataclasses.dataclass(frozen=True)
class EventPool:
    """A rewind run's hard events (`event_pool`): the windows holding them out of the run's draw rebuilt (``drawn``),
    the scenes, and what was read — the run, its events and the hard ones per airport."""

    drawn: Drawn
    scenes: list[EventScene]
    record: dict[str, Any]


def hard(record: Mapping[str, Any], branches: int) -> str | None:
    """The answered aircraft of an event record (`events.jsonl`) when it is hard (module docstring), else None — an
    event with no answered aircraft, or whose answered one's `start` branches are not all there, is not."""
    answered = [key for key, role in record["speakers"] if role == ANSWERED]
    if not answered:
        return None
    start = [b for b in record["branches"] if b["speaker"] == answered[0] and b["offset"] == START]
    return answered[0] if len(start) == branches and not any(b["rescued"] for b in start) else None


def event_pool(run: Path, split: str, *, instructions: Path, spec: Any, words: Any, airports: Mapping[str, Any],
               params: Any, executor_sha256: str, most: Mapping[str, int], max_rows: int, windows_alt: Any) -> EventPool:
    """A rewind run's hard events as scenes (module docstring); ``airports``: the split's scene airports
    (`traffic_speaking.with_tracks`), the rest what the run's draw and augmentation read. Refused: another schema, split,
    instruction artefact or executor spec, a run whose answered aircraft did not speak from ``start``, a draw that does
    not come back as the run had it, and a run with no hard event."""
    header = json.loads((run / "window_rewind.json").read_text(encoding="utf-8"))
    def refuse(why: str) -> ValueError:
        return ValueError(f"{run}: {why} — not a rewind run hard events can be read from")
    if header["schema"] != REWIND_SCHEMA:
        raise refuse(f"schema {header['schema']}, this code reads {REWIND_SCHEMA}")
    if header["split"] != split:
        raise refuse(f"the {header['split']} days, {split} wanted")
    if (REPO_ROOT / header["instructions"]).resolve() != instructions.resolve():
        raise refuse(f"instructions {header['instructions']}, not {repo_relative(instructions)}")
    if header["executor"]["sha256"] != executor_sha256:
        raise refuse("another executor spec")
    if ANSWERED not in header["roles"]:
        raise refuse("its answered aircraft did not speak again")
    draw = draw_windows(instructions, split, spec, words, airports, per_airport=header["windows_per_airport"],
                        seed=header["seed"], step_s=spec.step_s)
    drawn = drawn_windows(draw, airports, params, spec.step_s)
    augmenting = None
    if header["augment_seed"] is not None:
        drawn, augmenting = augmented_windows(drawn, params, most, max_rows, header["augment_seed"], windows_alt, spec)
    if as_json(draw.counts) != header["drawn"] or as_json(augmenting) != header["augmenting"]:
        raise refuse("the draw or its augmentation does not come back as the run had it")
    words_of = read_original_words(run / header["files"]["original_words"])
    scenes, windows, events, by_airport = [], [], 0, Counter()
    with (run / header["files"]["events"]).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            events += 1
            answered = hard(record, header["branches"])
            if answered is None:
                continue
            w = record["window"]
            lines, window = words_of[w], drawn.windows[w]
            if ([key for key, _, _ in lines] != list(window.commanded) or record["commanded"] != len(window.commanded)
                    or record["airport"] != window.airport.flights.code
                    or record["augmented"] != as_json(drawn.augmented[w])):
                raise refuse(f"window {w} came back otherwise")
            given = tuple(None if key == answered else Given(said, spoken) for key, spoken, said in lines)
            if w not in windows:
                windows.append(w)
            scenes.append(EventScene(record["event"], windows.index(w), record["airport"], answered, given))
            by_airport[record["airport"]] += 1
    if not scenes:
        raise refuse(f"none of its {events} events is hard")
    return EventPool(drawn_subset(drawn, windows), scenes,
                     {"run": repo_relative(run), "split": split, "events": events,
                      "hard": dict(sorted(by_airport.items())), "augment_seed": header["augment_seed"]})


def as_json(value: Any) -> Any:
    """``value`` as a JSON file gives it back (tuples as lists, numbers as JSON's)."""
    return json.loads(json.dumps(value))


def pick(pools: Sequence[EventPool], per_airport: int, rng: np.random.Generator
         ) -> list[tuple[int, int]]:
    """A round's scenes: of each airport (in name order), ``per_airport`` drawn without replacement from every pool's
    hard events together (all of them where it has fewer), as ``(pool, scene)`` places — an airport's in the pools'
    order."""
    places = [(p, s) for p, pool in enumerate(pools) for s in range(len(pool.scenes))]
    by_airport: dict[str, list[tuple[int, int]]] = {}
    for p, s in places:
        by_airport.setdefault(pools[p].scenes[s].airport, []).append((p, s))
    out = []
    for airport in sorted(by_airport):
        here = by_airport[airport]
        chosen = rng.choice(len(here), size=min(per_airport, len(here)), replace=False)
        out += [here[i] for i in sorted(chosen)]
    return out
