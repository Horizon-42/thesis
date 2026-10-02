"""Multi-aircraft step 8: the reward of a sentence that says a go-around (multi-aircraft design §6.6 step 8 item 9, the
user's decisions of 2026-10-02).

A sentence without a go-around keeps M4's reward: 1 for landing in the airport's landing direction not ended, else 0.
A sentence that says one (its first go-around word, at own step ``k_G``) is scored on what the go-around did:

- lost separation anywhere (the judge ended it): 0;
- landed (as above): ``0.48 + 0.14·(S + H + Q)``; timed out with no loss (the user: a safe go-around beats a loss):
  ``0.14·(S + H + Q)``; any other end (another runway's direction, below the glidepath's lower edge, a dynamics failure,
  a crossing without a capture): 0. The most is 0.9 — a clean landing's 1 stays the best, and the landing outweighs the
  other three together.
- **S, separation** (with every other aircraft in the scene — replayed, passive, commanded): the aircraft's tightest
  margin at each judged step (`inference.separation.margins`: the larger of horizontal over its minimum and, where it
  counts, vertical over its minimum; a pair the reading leaves alone does not count), from `SEPARATION_AFTER_S` after
  the go-around to the executor's capture again (none: the sentence's last judged step); a capture before that reads
  the one step at `SEPARATION_AFTER_S`. ``S = min(1, max(0, m_min − 1))``: full at twice the minimum.
- **H, out of the approach altitude**: the approach altitude is the glidepath's height at the FAF of the runway in force
  at the go-around (`RunwayProcedure.entry_m`, MSL). ``T_need = (entry − h_G) / (V_G · GO_AROUND_CLIMB_GRADIENT)`` from
  the go-around step's height and ground speed; reached within `CLIMB_FULL_SHARE` · T_need: 1, falling linearly to 0 at
  `CLIMB_ZERO_SHARE` · T_need; at or above it at the go-around: 1; never: 0.
- **Q, back on the approach** — only when S and H are both above 0: from the go-around to the executor's capture
  again, 1 within `RETURN_FULL_S`, falling linearly to 0 at `RETURN_ZERO_S` (R40 v2's p50 and p95 of the recorded
  go-arounds to the capture turn that followed, `outputs/POOLED/traffic/go_arounds_v2_20261002/`); no capture: 0.

A go-around gives its sentence `GO_AROUND_EXTRA_S` more time (`traffic_window.WindowLoop`): the p95 of what a recorded
go-around cost, 893 s, rounded up — less only where the model's positions end first (2 of 61,840 labelled sentences are
that long), and its row says how much it got.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
from trajectory_data_process.harvest.airports import load_airport
from ts_transformer.autopilot.vertical import GO_AROUND_CLIMB_GRADIENT
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.words import RUNWAY, UNCHANGED
from ts_transformer.prior.procedure import airport_procedures

#: a go-around's sentence flies this much longer (module docstring)
GO_AROUND_EXTRA_S = 900.0
#: the weights: the landing, and each of S, H and Q
LANDED_WEIGHT = 0.48
PART_WEIGHT = 0.14
#: S: read from this long after the go-around; full at this margin
SEPARATION_AFTER_S = 30.0
SEPARATION_FULL = 2.0
#: H: full within this share of T_need, nothing from that share on
CLIMB_FULL_SHARE = 1.25
CLIMB_ZERO_SHARE = 2.0
#: Q: full within, nothing from (R40 v2: 420.9 s and 597.6 s)
RETURN_FULL_S = 421.0
RETURN_ZERO_S = 598.0
#: the probes (multi-aircraft design §6.6 step 8 item 10, my provisional choice): a probe says a go-around for a cleared
#: aircraft established on its final at the first step its tightest margin went under this; a probe's go-around that did
#: better than its aircraft's unprobed samples is learned by a cross-entropy of this weight times how much better (its
#: probe gain, `traffic_window_tuner.window_advantages`; its own word: the probe, not the model, said it)
PROBE_MARGIN = 1.5
IMITATION_WEIGHT = 1.0
#: the ends a go-around's sentence is scored on: landed (with its direction), a time limit (safe, not landed)
LANDED, TIMEOUT = "landed", "timeout"


@dataclass(frozen=True)
class AfterGoAround:
    """What a sentence's first go-around did (module docstring): its own step, S, H, Q and L, and what they were read
    from — the tightest margin (None: no step read, or no other aircraft judged with it there), the climb's time against
    T_need, the time back on the approach — and the time the go-around gave it against the time it was to give."""

    step: int
    separation: float
    climb: float
    back: float
    landed: float
    reward: float
    margin_min: float | None
    need_s: float | None
    climbed_s: float | None
    back_s: float | None
    extra_s: float
    extra_wanted_s: float

    def fields(self) -> dict[str, float | int | None]:
        return asdict(self)


#: each airport's approach altitudes, by candidate (`approach_altitude_m`), read once a process
_APPROACH_M: dict[str, tuple[float, ...]] = {}


def approach_altitude_m(geometry: AirportGeometry, pointer: int) -> float:
    """The approach altitude of candidate ``pointer`` of ``geometry``'s airport: the glidepath's height at its FAF
    (`RunwayProcedure.entry_m`, MSL) from the published procedure — the harvest's runway data and CIFP, as the
    procedure's altitudes read them (`prior.procedure.published_procedures`)."""
    if geometry.code not in _APPROACH_M:
        runways = load_airport(geometry.code, config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP).runways
        _APPROACH_M[geometry.code] = tuple(final.entry_m for final in airport_procedures(geometry, runways))
    return _APPROACH_M[geometry.code][pointer]


def runway_at(said: np.ndarray, step: int) -> int:
    """The runway pointer in force at own step ``step``: the last runway word said at or before it."""
    pointed = said[: step + 1, RUNWAY]
    return int(pointed[pointed != UNCHANGED][-1])


def linear(value: float, full: float, zero: float) -> float:
    """1 up to ``full``, falling linearly to 0 at ``zero``."""
    return float(min(1.0, max(0.0, (zero - value) / (zero - full))))


def after_go_around(said: np.ndarray, step: int, outcome: str, landed_here: bool, margins: np.ndarray,
                    heights_m: Sequence[float], ground_speeds_mps: Sequence[float], captured: Sequence[bool],
                    judged_to: int, entry_m: float, step_s: float, extra_s: float,
                    extra_wanted_s: float) -> AfterGoAround:
    """A go-around's sentence scored (module docstring): ``said`` its words, ``step`` its first go-around's own step;
    ``outcome`` its outcome in the loop (`traffic_loop.LOST_SEPARATION` for the judge's, else its own end) and
    ``landed_here`` whether its landing counts (landed in the airport's landing direction); per own step from its first
    predicted one: ``margins`` (inf where judged alone or not at all), the executor's height, ground speed and capture;
    ``judged_to`` its last judged own step; ``entry_m`` the approach altitude of the runway in force at the go-around;
    ``extra_s`` the time the go-around gave it of the ``extra_wanted_s`` the loop gives (less where the model's
    positions end first: recorded)."""
    if outcome not in (LANDED, TIMEOUT) or (outcome == LANDED and not landed_here):
        return AfterGoAround(step, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, None, extra_s, extra_wanted_s)
    later = [k for k in range(step + 1, min(len(captured), judged_to + 1)) if captured[k]]
    recaptured = later[0] if later else None
    first = step + int(round(SEPARATION_AFTER_S / step_s))
    last = min(judged_to, max(first, recaptured if recaptured is not None else judged_to))
    window = margins[first: last + 1]
    tightest = float(window.min()) if len(window) else math.nan          # inf: judged alone throughout
    separation = float(min(1.0, max(0.0, tightest - 1.0))) if len(window) else 0.0
    margin_min = tightest if math.isfinite(tightest) else None
    height, speed = float(heights_m[step]), float(ground_speeds_mps[step])
    need_s = max(0.0, (entry_m - height) / (speed * GO_AROUND_CLIMB_GRADIENT))
    reached = [k for k in range(step, min(len(heights_m), judged_to + 1)) if heights_m[k] >= entry_m]
    climbed_s = (reached[0] - step) * step_s if reached else None
    climb = (0.0 if climbed_s is None else 1.0 if need_s == 0.0
             else linear(climbed_s, CLIMB_FULL_SHARE * need_s, CLIMB_ZERO_SHARE * need_s))
    back_s = (recaptured - step) * step_s if recaptured is not None else None
    back = (linear(back_s, RETURN_FULL_S, RETURN_ZERO_S) if back_s is not None and separation > 0.0 and climb > 0.0
            else 0.0)
    landed = 1.0 if outcome == LANDED else 0.0
    reward = LANDED_WEIGHT * landed + PART_WEIGHT * (separation + climb + back)
    return AfterGoAround(step, separation, climb, back, landed, reward, margin_min, need_s, climbed_s, back_s, extra_s,
                         extra_wanted_s)
