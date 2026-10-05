"""The census of the recorded aircraft with a faulty observed track (post-training §8 C1, the last item; vocabulary D111):
how much of a window's scene reads a point that no aircraft flies. No criterion is applied: the user decides from the
counts whether such steps or windows need a rule.

The marks are vocabulary D111's (`instructions.faults`: a jump, a held position, a reversal), each at the row of the
stored track it names. **A recorded aircraft reads a faulty point at a step** when its row there is a faulty point, or
its 2 s motion reads one (the row before it is one: the motion is the displacement from that row, `prior.inputs.motion`).

For each real window, over its steps from the commanded aircraft's row 0 to the end of its record (the census reads
records; a loop flies longer or shorter):

- whether a marked recorded aircraft is in the air during them;
- the steps at which a recorded aircraft reads a faulty point, and the tokens of the traffic attention that do (one for
  each such aircraft and step);
- with every aircraft on its record (the commanded aircraft too: its recorded runway after its first predicted step and
  its labelled G, D99), the first loss of separation that the commanded aircraft answers for from its first predicted
  step on (the separation judge, reading VISUAL; the event that would end the window, D93), and whether one aircraft of
  the pair reads a faulty point at the event's step or in the 2 Δ before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from geokit import FT_M, NM_M

from ts_transformer.inference.runway_schedule import (
    CWT_DIRECTLY_BEHIND_NM, CWT_ON_APPROACH_NM, FAA_PARALLEL_REGIMES, FAA_RADAR_NM, Separation,
)
from ts_transformer.instructions.faults import Fault
from ts_transformer.post.scene import REAL, AircraftAt, Recorded, Window
from ts_transformer.post.traffic import commanded_loss, joined, traffic
from ts_transformer.prior.procedure import Final

#: The steps before a loss in which a faulty point counts (C1: the event's step and the 2 Δ before it).
STEPS_BEFORE_EVENT = 2
#: No minimum of the judge is larger (the radar minimum, the wake minima of TBL 5-5-1 and 5-5-2), measured horizontally
#: or, at the threshold, along the approach clock — between two aircraft at most a pair separated as one apart across
#: their courses (parallels under 2,500 ft, `FAA_PARALLEL_REGIMES`): a step with no other aircraft this near has no
#: loss, and the judge is not asked there.
FARTHEST_MINIMUM_M = float(np.hypot(max(FAA_RADAR_NM, *CWT_DIRECTLY_BEHIND_NM.values(), *CWT_ON_APPROACH_NM.values())
                                    * NM_M, FAA_PARALLEL_REGIMES[0].below_ft * FT_M))


@dataclass(frozen=True)
class WindowFaults:
    """One window's counts (module docstring)."""

    marked_in_air: bool
    faulty_steps: int
    faulty_tokens: int
    loss: bool
    loss_reads_fault: bool


def reads_fault(flight: Recorded, time_s: np.ndarray, faults: frozenset[int]) -> np.ndarray:
    """``[T]`` bool: whether ``flight`` reads a faulty point at each time (one of its rows; module docstring)."""
    rows = np.round((np.asarray(time_s, dtype=np.float64) - flight.start_s) / flight.step_s).astype(np.int64)
    marked = np.array(sorted(faults), dtype=np.int64)
    return np.isin(rows, marked) | np.isin(rows - 1, marked)


def fault_rows(faults: Sequence[Fault]) -> frozenset[int]:
    return frozenset(fault.row for fault in faults)


def window_faults(window: Window, faults: Mapping[str, frozenset[int]], separation: Separation,
                  finals: Sequence[Final], step_s: float) -> WindowFaults:
    """``window``'s counts (module docstring); ``faults`` the fault rows of each marked flight of its scene, by key."""
    if window.kind != REAL:
        raise ValueError("the census counts real windows")
    scene, interval = window.scene, window.scene.interval_s
    own = scene.flights[scene.index[window.commanded.key]]            # its record, its labelled G (D99)
    times = window.row0_s + np.arange(int((own.end_s - window.row0_s) // interval) + 1) * interval
    end, start = times[-1], times[0]
    reading = np.zeros((len(times),), dtype=np.int64)
    marked_in_air = False
    for key, rows in faults.items():
        flight = scene.flights[scene.index[key]]
        if key == own.key or flight.end_s < start or flight.start_s > end:
            continue
        marked_in_air = True
        present = (times >= flight.start_s) & (times <= flight.end_s)
        reading[present] += reads_fault(flight, times[present], rows)
    loss, loss_reads = _first_loss(window, own, times, faults, separation, finals, step_s)
    return WindowFaults(marked_in_air, int((reading > 0).sum()), int(reading.sum()), loss, loss_reads)


def _first_loss(window: Window, own: Recorded, times: np.ndarray, faults: Mapping[str, frozenset[int]],
                separation: Separation, finals: Sequence[Final], step_s: float) -> tuple[bool, bool]:
    scene, interval = window.scene, window.scene.interval_s
    for t, time_s in enumerate(times):
        if time_s <= window.first_step_s:
            continue
        others = scene.others_at(time_s, own.key)
        mine = AircraftAt.of([own.at_step(time_s, interval)])
        if not len(others) or np.hypot(*(others.at[:, :2] - mine.at[0, :2]).T).min() > FARTHEST_MINIMUM_M:
            continue
        aircraft = joined(mine, others)
        loss = commanded_loss(traffic(aircraft, scene.geometry, separation, finals, step_s), aircraft.last_step,
                              separation)
        if loss is None:
            continue
        before = times[max(t - STEPS_BEFORE_EVENT, 0): t + 1]
        pair = [own] + [scene.flights[scene.index[aircraft.keys[k]]] for k in (loss.i, loss.j) if k != 0]
        reads = any(flight.key in faults and reads_fault(flight, before[(before >= flight.start_s)
                                                                          & (before <= flight.end_s)],
                                                         faults[flight.key]).any() for flight in pair)
        return True, reads
    return False, False


def fault_census(windows: Sequence[Window], faults: Mapping[str, Mapping[str, frozenset[int]]],
                 separations: Mapping[str, Separation], finals: Mapping[str, Sequence[Final]],
                 step_s: float) -> dict[str, dict]:
    """For each airport of ``windows`` (real windows of one split; ``faults`` each airport's fault rows by flight key):
    the windows, those with a marked recorded aircraft in the air, the steps and tokens that read a faulty point, the
    losses on the records and those of them with a faulty point at or 2 Δ before the event (module docstring)."""
    out: dict[str, dict] = {}
    for window in windows:
        code = window.scene.geometry.code
        counted = out.setdefault(code, {"windows": 0, "windows_with_marked_in_air": 0, "steps": 0,
                                        "steps_reading_fault": 0, "tokens_reading_fault": 0,
                                        "losses_on_records": 0, "losses_reading_fault": 0})
        found = window_faults(window, faults[code], separations[code], finals[code], step_s)
        scene = window.scene
        own = scene.flights[scene.index[window.commanded.key]]
        counted["windows"] += 1
        counted["steps"] += int((own.end_s - window.row0_s) // scene.interval_s) + 1
        counted["windows_with_marked_in_air"] += found.marked_in_air
        counted["steps_reading_fault"] += found.faulty_steps
        counted["tokens_reading_fault"] += found.faulty_tokens
        counted["losses_on_records"] += found.loss
        counted["losses_reading_fault"] += found.loss_reads_fault
    return out
