"""One flight's sentence drawn over its track, for checking the labeller by eye: the plan view
with the heading words and the capture, the altitude against distance flown with the altitude and
angle words and their tubes, the ground speed against time with the speed words."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.labeller.read import Reading, admit
from ts_transformer.instructions.labeller.vertical import tube_bounds
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import (
    ANGLE, APPROACH, HEADING, SPEED, UNCHANGED, Words,
)


def draw_flight(signals: FlightSignals, reading: Reading, geometry: AirportGeometry, spec: VocabularySpec,
                path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    words = Words(spec)
    flight = admit(signals, geometry, spec)          # the rows the sentence covers
    signals, smoothed = flight.signals, flight.smoothed
    candidate = geometry.candidates[reading.runway_index]
    grid = reading.words
    fig, (plan, vertical, speed) = plt.subplots(1, 3, figsize=(18, 6))

    # plan view
    e, n = signals.e_m / 1000, signals.n_m / 1000
    plan.plot(e, n, color="0.6", lw=1)
    course = np.radians(candidate.course_deg)
    back = np.array([candidate.threshold_e_m - 30000 * np.sin(course), candidate.threshold_e_m]) / 1000
    back_n = np.array([candidate.threshold_n_m - 30000 * np.cos(course), candidate.threshold_n_m]) / 1000
    plan.plot(back, back_n, "k--", lw=0.8, label=f"{candidate.ident} centreline")
    for row in np.nonzero(grid[:, HEADING] != UNCHANGED)[0]:
        plan.plot(e[row], n[row], "o", color="tab:blue")
        plan.annotate(f"{words.heading_deg(int(grid[row, HEADING])):03.0f}", (e[row], n[row]), fontsize=8,
                      xytext=(4, 4), textcoords="offset points", color="tab:blue")
    for row in np.nonzero(grid[:, APPROACH] != UNCHANGED)[0]:
        label = ["not cleared", "cleared", "go-around"][int(grid[row, APPROACH])]
        plan.annotate(label, (e[row], n[row]), fontsize=8, xytext=(4, -10), textcoords="offset points", color="tab:green")
    plan.plot(e[reading.capture_row], n[reading.capture_row], "s", color="tab:red", label="capture")
    plan.plot(e[0], n[0], "^", color="k", label="entry")
    plan.set_aspect("equal")
    plan.set_xlabel("E (km, airport frame)")
    plan.set_ylabel("N (km)")
    plan.legend(fontsize=8)
    plan.set_title(f"{signals.dataset_id}  runway {candidate.ident}")

    # altitude against distance flown, with each altitude word's tube
    s = smoothed.distance_m
    vertical.plot(s / 1000, signals.altitude_m, color="0.75", lw=1, label="altitude (raw)")
    vertical.plot(s / 1000, smoothed.altitude_m, color="k", lw=1, label="altitude (smoothed)")
    angle_rows = list(np.nonzero(grid[:, ANGLE] != UNCHANGED)[0])
    for word, stop, low, high in tube_bounds(reading.instructions, s, smoothed.altitude_m, spec, words):
        vertical.fill_between(s[word.row: stop] / 1000, low, high, color="tab:orange", alpha=0.25, lw=0)
        target = words.altitude_m(word.value)
        label = "land" if target is None else f"{target:.0f} m"
        vertical.annotate(label, (s[word.row] / 1000, smoothed.altitude_m[word.row]), fontsize=8, color="tab:orange",
                          xytext=(2, 6), textcoords="offset points")
    for row in angle_rows:
        vertical.annotate(f"{words.angle_deg(int(grid[row, ANGLE])):.1f}°", (s[row] / 1000, smoothed.altitude_m[row]),
                          fontsize=7, color="tab:purple", xytext=(2, -12), textcoords="offset points")
    vertical.axhline(candidate.elevation_m, color="k", lw=0.5, ls=":")
    vertical.set_xlabel("distance flown (km)")
    vertical.set_ylabel("altitude MSL (m)")
    vertical.legend(fontsize=8)

    # speed against time
    t = signals.time_s
    speed.plot(t, signals.ground_speed_mps, color="0.75", lw=1, label="ground speed (raw)")
    speed.plot(t, smoothed.ground_speed_mps, color="k", lw=1, label="ground speed (smoothed)")
    speed_rows = list(np.nonzero(grid[:, SPEED] != UNCHANGED)[0]) + [len(grid)]
    for start, stop in zip(speed_rows, speed_rows[1:]):
        target = words.speed_mps(int(grid[start, SPEED]))
        if target is None:
            speed.axvspan(t[start], t[stop - 1], color="tab:gray", alpha=0.15, lw=0)
            speed.annotate("unspecified", (t[start], smoothed.ground_speed_mps[start]), fontsize=8, xytext=(2, 6),
                           textcoords="offset points")
            continue
        speed.fill_between(t[start:stop], target - spec.speed_tolerance_mps, target + spec.speed_tolerance_mps,
                           color="tab:cyan", alpha=0.25, lw=0)
        speed.annotate(f"{target:.0f}", (t[start], target), fontsize=8, xytext=(2, 6), textcoords="offset points")
    speed.axvline(t[reading.join_row], color="tab:green", lw=0.8, ls="--", label="clearance")
    speed.set_xlabel("time (s)")
    speed.set_ylabel("ground speed (m/s)")
    speed.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
