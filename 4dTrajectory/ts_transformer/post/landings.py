"""The landings of a window's scene (post-training §3 "Landing context", D105): what the commanded aircraft's inputs and
the reward's present landing direction count.

A window's landings are its airport's roster landings (`prior.landings.LandingIndex`, prior §7 item 2: the tracks
roster less the sealed test days) with the window's changes:

- a real window, and window B (only its commanded aircraft's start is moved): the roster's landings, unchanged — so a
  window without other aircraft reads what free generation reads (§2 item 1);
- window A: the inserted aircraft's landing added, at its source flight's roster landing shifted by the window's shift,
  on the same runway, under the inserted aircraft's own key (`scene.INSERTED_SUFFIX`);
- window D: the moved aircraft's roster landing moved by its shift.

A landing that a shift puts on a sealed test day is left out and counted with the index's sealed landings (C32); one
that a shift puts on a day outside the day split is refused by the index itself (`LandingIndex`, prior D105). The
commanded aircraft's own landing stays in the index: the function of a loop's row leaves it out by its key (D31, D63).

**A window of several commanded aircraft** (multi-aircraft control D147 item 1, D149; post-training §9 item 2): each
commanded aircraft's landings (`commanded_landings`) are the window's, less every other commanded aircraft's own (its
recorded landing is that aircraft's future), with its own at its record's time in the window (moved by its shift in a
compressed window), which the function of a loop's row leaves out by its key. A stage C window: the window's landings.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from ts_transformer.data.day_split import landing_day
from ts_transformer.post.scene import INSERTED, INSERTED_SUFFIX, LEADER_MOVED, MOVED_START, REAL, Window
from ts_transformer.prior.inputs import own_flight_key
from ts_transformer.prior.landings import Landing, LandingIndex


def roster_key(dataset_id: str, airport: str) -> str:
    """The tracks roster's key of a flight of the signals (`prior.inputs.own_flight_key`)."""
    return own_flight_key({"dataset_id": dataset_id, "airport": airport})


def commanded_landings(window: Window, roster: LandingIndex) -> list[LandingIndex]:
    """Each commanded aircraft's landings in ``window`` (module docstring), the anchor first."""
    base = window_landings(window, roster)
    if not window.joined:
        return [base]
    airport = window.scene.geometry.code
    landings = {landing.flight_key: landing for landing in base.landings}
    own = [roster_key(record.key, airport) for record in window.commanded_all]
    shifts = [0.0, *(item.shift_s for item in window.joined)]
    out = []
    for key, shift in zip(own, shifts):
        kept = {k: landing for k, landing in landings.items() if k not in own}
        kept[key] = replace(landings[key], time_s=landings[key].time_s + shift)
        out.append(LandingIndex(base.runways, tuple(sorted(kept.values(), key=lambda item: (item.time_s, item.flight_key))),
                                base.sealed, base.days))
    return out


def window_landings(window: Window, roster: LandingIndex) -> LandingIndex:
    """``window``'s landings (module docstring) from its airport's ``roster`` landings (with their day split)."""
    if window.kind in (REAL, MOVED_START):
        return roster
    if len(window.moved) != 1:
        raise ValueError(f"a window {window.kind} moves one flight, not {len(window.moved)}")
    (key, shift), = window.moved
    airport = window.scene.geometry.code
    landings = {landing.flight_key: landing for landing in roster.landings}
    if window.kind == INSERTED:
        source = landings[roster_key(key.removesuffix(INSERTED_SUFFIX), airport)]
        changed = Landing(source.time_s + shift, source.runway, roster_key(key, airport))
    elif window.kind == LEADER_MOVED:
        moved = landings[roster_key(key, airport)]
        changed = replace(moved, time_s=moved.time_s + shift)
        del landings[moved.flight_key]
    else:
        raise ValueError(f"no landings for a window {window.kind}")
    sealed = roster.sealed
    if landing_day(datetime.fromtimestamp(changed.time_s, tz=timezone.utc).isoformat()) in roster.days.days["test"]:
        sealed += 1
    else:
        landings[changed.flight_key] = changed
    kept = sorted(landings.values(), key=lambda landing: (landing.time_s, landing.flight_key))
    return LandingIndex(roster.runways, tuple(kept), sealed, roster.days)
