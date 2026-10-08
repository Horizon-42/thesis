"""Window lists (post-training D176 (1), §8 C25): a list of select windows chosen by some rule, for stages C and D — the
format ``ts-window-list-v1``, written and read here.

A list names windows and moves none out of its split: its stage (`STAGES`); its split, the select days only (a val
window is read once, outline D85); the selection windows it indexes (the stage's ``selection_windows``, by their
select seed and windows an airport, `SELECTION_FIELDS`; stage D's spans too); one sentence saying what chose them;
the readouts it read, each by path and sha256; and for each window its place among those selection windows, its
airport, its identity (`IDENTITY_FIELDS`: stage C, the commanded flight, ``row0_s`` and kind; stage D, the anchor
flight, ``row0_s`` and span) and information fields that no reader acts on (``info``). A reader finds each window at its
place among a campaign's selection windows and checks it by its identity (the Training export, D176 (3)).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ts_transformer.io_utils import utc_now, write_json_atomic

#: The format of a window list.
WINDOW_LIST_SCHEMA = "ts-window-list-v1"
#: The stages a list is of: stage C (one commanded aircraft a window) and stage D (multi-aircraft control).
STAGES = ("C", "D")
#: The only split a list names (D176: a val window is read once, outline D85).
SPLIT = "select"
#: A window's identity in each stage (D176 (1)).
IDENTITY_FIELDS = {"C": ("flight", "row0_s", "kind"), "D": ("flight", "row0_s", "span_s")}
#: The selection windows a list indexes, in each stage.
SELECTION_FIELDS = {"C": ("select_seed", "per_airport"), "D": ("select_seed", "per_airport", "spans_s")}


@dataclass(frozen=True)
class ListedWindow:
    """One window of a list: its place among the selection windows, its airport, its identity and information fields
    that no reader acts on."""

    place: int
    airport: str
    identity: Mapping[str, Any]
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowList:
    """A window list (module docstring): checked on construction."""

    stage: str
    split: str
    selection: Mapping[str, Any]
    chose: str
    readouts: tuple[Mapping[str, str], ...]
    windows: tuple[ListedWindow, ...]

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"a window list is of stage {' or '.join(STAGES)}, not {self.stage!r}")
        if self.split != SPLIT:
            raise ValueError(f"a window list names {SPLIT} windows only, not {self.split} (outline D85: a val window is "
                             f"read once)")
        if set(self.selection) != set(SELECTION_FIELDS[self.stage]):
            raise ValueError(f"stage {self.stage}'s selection is {SELECTION_FIELDS[self.stage]}, got "
                             f"{sorted(self.selection)}")
        if not self.chose:
            raise ValueError("a window list says what chose its windows")
        if any(set(r) != {"path", "sha256"} for r in self.readouts):
            raise ValueError("each readout a window list read is named by its path and sha256")
        for w in self.windows:
            if not isinstance(w.place, int) or w.place < 0:
                raise ValueError(f"a listed window has a place among the selection windows, got {w.place!r}")
            if set(w.identity) != set(IDENTITY_FIELDS[self.stage]):
                raise ValueError(f"window at place {w.place}: stage {self.stage}'s identity is "
                                 f"{IDENTITY_FIELDS[self.stage]}, got {sorted(w.identity)}")
        places = [w.place for w in self.windows]
        if len(set(places)) != len(places) or places != sorted(places):
            raise ValueError("a window list names each place once, in order")


def write_window_list(path: Path, listed: WindowList) -> dict[str, Any]:
    """``listed`` written to ``path`` (a new file), with the time, the count and the windows by airport."""
    if Path(path).exists():
        raise FileExistsError(f"{path} exists: a window list is written once")
    payload = {"schema": WINDOW_LIST_SCHEMA, "written_utc": utc_now(), "stage": listed.stage, "split": listed.split,
               "selection": dict(listed.selection), "chose": listed.chose,
               "readouts": [dict(r) for r in listed.readouts], "count": len(listed.windows),
               "by_airport": dict(sorted(Counter(w.airport for w in listed.windows).items())),
               "windows": [{"place": w.place, "airport": w.airport, "identity": dict(w.identity), "info": dict(w.info)}
                           for w in listed.windows]}
    write_json_atomic(path, payload)
    return payload


def read_window_list(path: Path) -> WindowList:
    """The window list at ``path``, refused by name for another schema, a split other than select, or a window without
    its place or identity."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["schema"] != WINDOW_LIST_SCHEMA:
        raise ValueError(f"{path} is a {payload['schema']} file, not a {WINDOW_LIST_SCHEMA} window list")
    if payload["split"] != SPLIT:
        raise ValueError(f"{path} names {payload['split']} windows: a window list names {SPLIT} windows only")
    windows = []
    for k, w in enumerate(payload["windows"]):
        if "place" not in w or "identity" not in w:
            raise ValueError(f"{path}: window {k} has no place or no identity")
        windows.append(ListedWindow(w["place"], w["airport"], w["identity"], w["info"]))
    return WindowList(payload["stage"], payload["split"], payload["selection"], payload["chose"],
                      tuple(payload["readouts"]), tuple(windows))
