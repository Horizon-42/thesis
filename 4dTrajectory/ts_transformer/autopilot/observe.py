"""The observation operator O (executor design §9 method B, §12): a flown track read the way the data plane
reads an ADS-B track.

What the executor flies is its own state every cycle; what the labeller reads of an observed flight is
positions reported about once a second, turned into velocities by a 15 s centred least-squares fit and
resampled on the 2 s grid. O applies that same chain to the flown track, with the data plane's own
functions: the positions become waypoints, `flight_scenarios.start_state.state_samples_from_track` fits
the velocities, `data.channels.channels_from_states` and `resample_uniform` put them on the grid in the
flight's own frame, and `instructions.signals.signals_from_series` reads the signals as it read the
observed flight's. The result is what the labeller would have read, had the executor's track been the
one on the radar.

MIRROR of the data plane's velocity source `REFERENCE_VELOCITY_TRACK_FIT` (the series the artefact was read
from are built with it; `observe` refuses another). One difference stays, by construction: an observed track
runs on past the threshold, so its last rows' fits are centred; the flown track ends at the crossing, so its
last ~7 s are fitted one-sided.

Method B's delays (`delays_from_leads`) are measured on the words that START a manoeuvre: a heading word
that begins a turn (a later part of a split turn, or an inserted intercept, would measure the executor's
turn rate against the observed one, not the reading's lead), every altitude, angle and speed word.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from flight_scenarios.start_state import state_samples_from_track
from ts_transformer.autopilot.frame import ALT, LAT, LON
from ts_transformer.autopilot.sentence import DELAY_GROUPS, Delays
from ts_transformer.config import TSConfig
from ts_transformer.data.channels import channels_from_states, resample_uniform
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.data.reference_velocity import REFERENCE_VELOCITY_TRACK_FIT
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.labeller.read import Reading, read_flight
from ts_transformer.instructions.signals import FlightSignals, signals_from_series
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import HEADING, Words

#: The columns method B measures a delay on (a manoeuvre's onset); the runway pointer and the clearance
#: follow the heading's delay.
MEASURED_COLUMNS = tuple(column for columns in DELAY_GROUPS.values() for column in columns)


def observe(states: np.ndarray, cycle_s: float, series: FlightSeries, geometry: AirportGeometry,
            step_s: float) -> FlightSignals:
    """``states`` ``[T, 7]`` geodetic, one row per cycle boundary from the flight's row 0 → the signals
    the labeller would read of it. ``series`` is the flight's own (its frame, mass and runway)."""
    if TSConfig().reference_velocity_source != REFERENCE_VELOCITY_TRACK_FIT:
        raise ValueError("the observation operator mirrors the track-fit velocity source; the data plane now "
                         f"builds with {TSConfig().reference_velocity_source!r}")
    waypoints = [(k * cycle_s, float(row[LON]), float(row[LAT]), float(row[ALT])) for k, row in enumerate(states)]
    samples = state_samples_from_track(waypoints, mass_kg=float(series.scenario.initial.m))
    times, values = channels_from_states(samples, series.frame)
    grid, resampled = resample_uniform(times, values, step_s)
    return signals_from_series(replace(series, times=grid, values=resampled), geometry)


def word_leads(received: list[tuple[int, int, float]], reread: list[tuple[int, int, float]],
               window_s: float) -> list[tuple[int, float]]:
    """How much earlier the labeller places each word than the executor received it: for every word the
    executor received, ``(column, value, time)``, the re-read word of the same column and value nearest
    in time within ``window_s``; ``(column, received − re-read)``. A word with no match in the window is
    left out (the re-read sentence need not say it again: a turn the executor did not need to fly)."""
    leads = []
    for column, value, time in received:
        matches = [t for c, v, t in reread if c == column and v == value and abs(t - time) <= window_s]
        if matches:
            nearest = min(matches, key=lambda t: abs(t - time))
            leads.append((column, time - nearest))
    return leads


def starts(word) -> bool:
    """A word that starts a manoeuvre (module docstring): every word but a heading word that continues or
    intercepts a turn."""
    return word.column != HEADING or (word.kind in ("turn", "turn-split") and word.info["part"] == 1)


def flight_leads(states: np.ndarray, cycle_s: float, series: FlightSeries, geometry: AirportGeometry, reading: Reading,
                 spec: VocabularySpec, words: Words, window_s: float) -> tuple[list[tuple[int, float]], Counter]:
    """Method B on one flight flown with no delay: ``states`` up to where its flight ended. The words the
    executor received — every word of `MEASURED_COLUMNS` after step 0 (step 0 describes what the aircraft
    is already doing) at a time the flight reached — against the labeller's re-reading of the flown track
    through `observe`. Returns the ``(column, lead)`` pairs and how many words of each column were received;
    raises `Refused` when the labeller will not read the flown track."""
    flown_s = (len(states) - 1) * cycle_s
    received = [(i.column, i.value, i.row * spec.step_s) for i in reading.instructions
                if i.column in MEASURED_COLUMNS and i.row > 0 and i.row * spec.step_s < flown_s and starts(i)]
    again = read_flight(observe(states, cycle_s, series, geometry, spec.step_s), geometry, spec, words)
    reread = [(i.column, i.value, i.row * spec.step_s) for i in again.instructions if i.row > 0]
    return word_leads(received, reread, window_s), Counter(column for column, _, _ in received)


def delays_from_leads(leads: dict[int, list[float]]) -> Delays:
    """Each delay group's median lead (`DELAY_GROUPS`), floored at 0: a negative lead is the labeller reading a
    manoeuvre's start late, and a word cannot act before it is said. A group with no matched word is refused."""
    values = {}
    for name, columns in DELAY_GROUPS.items():
        pooled = [lead for column in columns for lead in leads[column]]
        if not pooled:
            raise ValueError(f"method B matched no word of {name} (columns {columns}); draw a larger sample")
        values[name] = max(0.0, float(np.median(pooled)))
    return Delays(**values)
