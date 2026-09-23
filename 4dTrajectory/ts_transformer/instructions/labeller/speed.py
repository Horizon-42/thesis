"""Speed words and the "unspecified" speed (vocabulary design §2.6, §3.4).

The smoothed ground speed is fitted against time by straight pieces. A piece at least the
minimum hold long, flatter than the flat-acceleration bound and inside one speed target's
band is a HOLD; every other piece is a TRANSITION. Consecutive transitions in one direction
form a run; a run's word is the next hold's target (the turning speed when the direction
reverses without a hold), issued where the run begins. After the approach clearance the speed
is the pilot's own ("unspecified", 7110.65BB 5-7-1 d), unless a hold of at least
`unspecified_plateau_s` still ends `unspecified_distance_m` or more before the threshold
(ATC may assign a speed until 5 NM, 5-7-1 b.4): then the words run on until that hold ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ts_transformer.instructions import envelope
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.piecewise import fit_pieces
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import SPEED, Words

HOLD = "hold"
TRANSITION = "transition"


@dataclass
class SpeedPiece:
    start: int
    stop: int  # one past the last row
    kind: str
    target_index: int = -1     # HOLD
    accelerating: bool = False  # TRANSITION

    @property
    def rows(self) -> int:
        return self.stop - self.start


@dataclass
class SpeedReading:
    instructions: list[Instruction]
    pieces: list[SpeedPiece]
    unspecified_row: int


def speed_pieces(time: np.ndarray, speed: np.ndarray, spec: VocabularySpec, words: Words) -> list[SpeedPiece]:
    minimum = spec.rows(spec.speed_min_hold_s)
    pieces: list[SpeedPiece] = []
    for piece in fit_pieces(time, speed, spec.speed_fit_tolerance_mps):
        rows = speed[piece.start: piece.stop]
        index = int(round((float(np.median(rows)) - spec.speed_min_mps) / spec.speed_step_mps))
        hold = (piece.rows >= minimum and abs(piece.slope) <= spec.speed_flat_accel_mps2
                and 0 <= index < words.n_speed_levels
                and bool(envelope.speed_band(rows, words.speed_mps(index), spec.speed_tolerance_mps).all()))
        if hold:
            previous = pieces[-1] if pieces else None
            if (previous is not None and previous.kind == HOLD
                    and envelope.speed_band(rows, words.speed_mps(previous.target_index), spec.speed_tolerance_mps).all()):
                previous.stop = piece.stop
            else:
                pieces.append(SpeedPiece(piece.start, piece.stop, HOLD, target_index=index))
        else:
            pieces.append(SpeedPiece(piece.start, piece.stop, TRANSITION, accelerating=piece.slope > 0.0))
    return pieces


def _groups(pieces: list[SpeedPiece]) -> list[list[SpeedPiece]]:
    groups: list[list[SpeedPiece]] = []
    for piece in pieces:
        previous = groups[-1][-1] if groups else None
        if (piece.kind == TRANSITION and previous is not None and previous.kind == TRANSITION
                and previous.accelerating == piece.accelerating):
            groups[-1].append(piece)
        else:
            groups.append([piece])
    return groups


def read_speed(time: np.ndarray, speed: np.ndarray, before_threshold_m: np.ndarray, join_row: int,
               spec: VocabularySpec, words: Words) -> SpeedReading:
    pieces = speed_pieces(time, speed, spec, words)
    groups = _groups(pieces)
    minimum_plateau = spec.rows(spec.unspecified_plateau_s)
    kept = [g[0] for g in groups
            if g[0].kind == HOLD and g[0].stop - 1 >= join_row and g[0].rows >= minimum_plateau
            and before_threshold_m[g[0].stop - 1] >= spec.unspecified_distance_m]
    unspecified_row = kept[-1].stop if kept else join_row
    # a change of speed already under way at that row, begun less than a minimum hold before
    # it, is the pilot's own speed too: "unspecified" from where it began
    for group in groups:
        first = group[0]
        if (first.kind == TRANSITION and first.start < unspecified_row <= group[-1].stop - 1
                and unspecified_row - first.start < spec.rows(spec.speed_min_hold_s)):
            unspecified_row = first.start
            break

    def target_word(value_mps: float) -> int:
        try:
            return words.speed_index(value_mps)
        except ValueError as error:
            raise Refused("speed target out of range", str(error)) from None

    instructions: list[Instruction] = []
    for position, group in enumerate(groups):
        first = group[0]
        if first.start >= unspecified_row:
            break
        kind = "initial" if first.start == 0 else "target"
        if first.kind == HOLD:
            if position == 0 or groups[position - 1][0].kind == HOLD:
                instructions.append(Instruction(SPEED, first.target_index, first.start, kind if position == 0 else "step",
                                                {"target_mps": words.speed_mps(first.target_index)}))
            continue
        following = groups[position + 1] if position + 1 < len(groups) else None
        if following is not None and following[0].start >= unspecified_row:
            following = None                 # what comes after is the pilot's own speed
        if following is not None and following[0].kind == HOLD:
            target_index = following[0].target_index
        elif following is not None:          # the direction reverses without a hold
            target_index = target_word(float(speed[group[-1].stop - 1]))
        else:                                # the run carries on into the unspecified speed
            target_index = target_word(float(speed[unspecified_row - 1]))
        instructions.append(Instruction(SPEED, target_index, first.start, kind,
                                        {"target_mps": words.speed_mps(target_index)}))
    instructions.append(Instruction(SPEED, words.speed_unspecified, unspecified_row,
                                    "initial" if unspecified_row == 0 else "unspecified"))
    return SpeedReading(instructions=instructions, pieces=pieces, unspecified_row=unspecified_row)


def span_checks(instructions: list[Instruction], speed: np.ndarray, spec: VocabularySpec, words: Words) -> list[dict[str, Any]]:
    """Each speed word's span: a monotone transition to the target, then the band. Run on the
    assembled sentence, so a word the assembly dropped is not judged."""
    ordered = sorted((item for item in instructions if item.column == SPEED), key=lambda item: item.row)
    ends = [item.row for item in ordered[1:]] + [len(speed)]
    results = []
    for word, end in zip(ordered, ends):
        target = words.speed_mps(word.value)
        if target is None:
            continue
        span = speed[word.row: end]
        inside = envelope.speed_band(span, target, spec.speed_tolerance_mps)
        arrival = int(np.argmax(inside)) if inside.any() else len(span)
        transition_ok = envelope.speed_transition_ok(span[: arrival + 1], target, spec.speed_tolerance_mps)
        accel = np.abs(np.diff(span[: arrival + 1])) / spec.step_s if arrival > 0 else np.zeros(0)
        after = inside[arrival:]
        results.append({
            "row": word.row, "rows": end - word.row, "arrival_rows": arrival,
            "cut_before_arrival": arrival >= len(span),
            "transition_ok": transition_ok,
            "accel_ok": bool(len(accel) == 0 or accel.max() <= spec.speed_accel_max_mps2),
            "band_rows": int(len(after)), "band_inside": int(np.count_nonzero(after)),
            # a span the next word cuts before the target is reached is judged on its transition alone
            "contained": bool(transition_ok and after.all()),
        })
    return results
