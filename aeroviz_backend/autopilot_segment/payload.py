"""The answer to ``POST /autopilot/segment``: a word's segment of a closed-loop sentence, flown every control cycle from
the word on, how it ended, and how far it lies from the artefact's stored flown states.

Units are SI. Time is from the sentence's first predicted step (the executor's cycle 0, the export's ``flownFromRow``);
positions are in the airport frame and on the globe, heights MSL and — the height Cesium draws in — plus the flight's
runway's HAE − MSL offset, the one its observed track is drawn with (the set's ``haeMinusMslM``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ts_transformer.autopilot.frame import ALT, LAT, LON
from ts_transformer.autopilot.judge import flown_track
from ts_transformer.experiments.training_attitude import attitude_payload, executor_attitude
from ts_transformer.experiments.training_flights import crossing_payload
from ts_transformer.instructions.training_files import rounded

from aeroviz_backend.autopilot_segment.fly import FlownSegment, last_state

#: MIRROR of `aeroviz-4d/src/data/trainingAutopilot.ts` (`TRAINING_AUTOPILOT_SCHEMA`); the reader refuses anything
#: else by name. A name changes with the payload's shape, on both sides, in one change. v9 (A23, 2026-10-04): stage A —
#: a word of the closed-loop sentence at Δ, flown from its first predicted step; no model sentence, no procedure's masks.
SCHEMA = "aeroviz-autopilot-segment-v9"
#: The end of a segment its stop ended (the executor got there); otherwise the judge's outcome (`judge.OUTCOMES`).
#: MIRROR of `trainingAutopilot.ts` (`TRAINING_AUTOPILOT_SEGMENT_END`).
SEGMENT_END = "segment_end"


def track_payload(result: FlownSegment, geometry: Any, hae_minus_msl_m: float, aero_params: np.ndarray
                  ) -> dict[str, Any]:
    """The flown segment every control cycle from the word's cycle to its last state, with the commands each cycle flew
    (one fewer than the states: the last state's is null) and the attitude it is drawn in."""
    flown = result.flown
    first, last = result.segment.start_cycle, last_state(result)
    states = flown.states[0, : last + 1].cpu().numpy()
    track = flown_track(states, geometry)
    commands = flown.commands[0, :last].cpu().numpy()
    shown = slice(first, None)
    cycles = np.arange(first, last + 1)
    return {
        "cycle": [int(c) for c in cycles], "tS": rounded(cycles * flown.cycle_s, 3),
        "eM": rounded(track["e"][shown], 1), "nM": rounded(track["n"][shown], 1),
        "latDeg": rounded(states[shown, LAT], 7), "lonDeg": rounded(states[shown, LON], 7),
        "altitudeMslM": rounded(states[shown, ALT], 2), "altitudeHaeM": rounded(states[shown, ALT] + hae_minus_msl_m, 2),
        "trackDeg": rounded(np.mod(track["track"][shown], 360.0), 3),
        "groundSpeedMps": rounded(track["ground_speed"][shown], 3),
        "verticalRateMps": rounded(track["vertical_rate"][shown], 3),
        # the dynamics' bank turns left when positive
        "thrustFraction": rounded(commands[shown, 0], 4), "bankRightDeg": rounded(-np.degrees(commands[shown, 1]), 3),
        "loadFactor": rounded(commands[shown, 2], 4),
        "attitude": attitude_payload(executor_attitude(flown, 0, cycles, aero_params)),
    }


def segment_payload(result: FlownSegment, geometry: Any, hae_minus_msl_m: float, aero_params: np.ndarray,
                    apart: dict[str, Any]) -> dict[str, Any]:
    segment, verdict = result.segment, result.verdict
    return {
        "segment": {"column": segment.column, "row": segment.row, "word": segment.word,
                    "correction": segment.correction, "startCycle": segment.start_cycle,
                    "stopCycle": segment.stop_cycle,
                    "end": SEGMENT_END if verdict is None else verdict.outcome,
                    # the cycle the flight ended: the stop's last state, or the judge's outcome row (as the export's
                    # replay writes it; a dynamics failure's track ends one state before it)
                    "endCycle": last_state(result) if verdict is None else verdict.end_row},
        "track": track_payload(result, geometry, hae_minus_msl_m, aero_params),
        # flown to its outcome: the judge's crossing and decision-altitude check, as the export writes them
        "crossing": None if verdict is None else crossing_payload(verdict, result.flown, 0, geometry),
        # the live flight against the artefact's stored flown states on the 2 s rows both have
        "stored": {"rows": apart["rows"], "horizontalM": round(apart["horizontalM"], 9),
                   "verticalM": round(apart["verticalM"], 9)},
    }
