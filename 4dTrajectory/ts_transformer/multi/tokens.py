"""Stage D's token part (multi-aircraft control D152, §3.2 item 2): what a commanded aircraft's token of another aircraft
of its window carries beside stage C's edge features (post-training §9 item 7, `post.traffic_attention.add_token_part`:
its own projection, which starts at zero, under the name `TOKENS_SCHEMA`).

**The features** (`PART_FEATURES`, in order), for another commanded aircraft at the tick of the row:

- ``commanded`` — 1;
- ``silent`` — 1 when it answered for a loss of separation (D144);
- ``in_force`` — 1 once its first predicted step has been said: from the row after it (D148: a word that one aircraft
  is told is read by the others one row later); every word feature below is 0 before it;
- ``heading_sin``, ``heading_cos`` — its heading word's track less the course of its runway in force, as the prior's own
  input of the heading in force reads it (prior §7 item 2, `Heard.inputs`);
- ``altitude``, ``no_level_off`` — its altitude word: the level's height above the airport elevation over
  `HEIGHT_SCALE_M`; 0 and the flag 1 for "no level-off";
- ``angle`` — its angle word's nominal angle (descending positive) over `ANGLE_SCALE_DEG`;
- ``speed``, ``speed_unspecified`` — its speed word: the target over `SPEED_SCALE_MPS`; 0 and the flag 1 for
  "unspecified".

A recorded aircraft's part is 0 (its flags 0, no words): its labelled words use later rows and are never an input
(post-training D99). The scales are fixed constants in SI units (prior D41), the edge features' own where they have one.

`TokenPart` is the window loop's token part (post-training §9 item 5, its ``token_part``): it reads the loop's words in
force and silent aircraft once a tick.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ts_transformer.instructions.words import Words

#: The name of this token part (D152; a checkpoint of stage D holds it, §7 item 3): a changed feature is a new name.
TOKENS_SCHEMA = "multi-commanded-tokens-v1"
PART_FEATURES = ("commanded", "silent", "in_force", "heading_sin", "heading_cos", "altitude", "no_level_off", "angle",
                 "speed", "speed_unspecified")
#: MIRROR of `post.edges.HEIGHT_SCALE_M` and `SPEED_SCALE_MPS` (not names of post-training §9, which stage D imports
#: only); `tests/test_multi_control.py` pins them.
HEIGHT_SCALE_M = 1_000.0
SPEED_SCALE_MPS = 100.0
#: An angle word's scale: the nominal glidepath's 3° reads 1.
ANGLE_SCALE_DEG = 3.0


def commanded_part(silent: bool, heading: np.ndarray, levels: np.ndarray, words: Words) -> np.ndarray:
    """``[len(PART_FEATURES)]`` float32: the part of another commanded aircraft (module docstring), from its words in
    force as the prior's inputs read them (`Heard.inputs`: the heading's sine and cosine, the altitude, angle and speed
    words, −1 before anything is said)."""
    out = np.zeros(len(PART_FEATURES), dtype=np.float64)
    index = {name: k for k, name in enumerate(PART_FEATURES)}
    out[index["commanded"]], out[index["silent"]] = 1.0, float(silent)
    altitude, angle, speed = (int(v) for v in levels)
    if altitude < 0:                                    # nothing said yet: no word in force
        return out.astype(np.float32)
    out[index["in_force"]] = 1.0
    out[index["heading_sin"]], out[index["heading_cos"]] = heading
    level = words.altitude_level_m(altitude)
    if level is None:
        out[index["no_level_off"]] = 1.0
    else:
        out[index["altitude"]] = level / HEIGHT_SCALE_M
    out[index["angle"]] = words.angle_deg(angle) / ANGLE_SCALE_DEG
    target = words.speed_mps(speed)
    if target is None:
        out[index["speed_unspecified"]] = 1.0
    else:
        out[index["speed"]] = target / SPEED_SCALE_MPS
    return out.astype(np.float32)


class TokenPart:
    """The window loop's token part (module docstring): ``(loop, row, others)`` → ``[N, len(PART_FEATURES)]``, for each
    other aircraft of the row (its row of the batch, None for a recorded aircraft) its part. Each row's part as another
    aircraft is computed once a tick of a loop, from the speaker's words in force (`Speaker.heard`) and the loop's
    silent rows at the start of the row."""

    width = len(PART_FEATURES)

    def __init__(self, words: Words) -> None:
        self.words = words
        #: the loop (kept, so that its id is not taken again), its tick and each of its rows' part
        self._table: tuple[Any, int, np.ndarray] | None = None

    def __call__(self, loop: Any, row: int, others: Sequence[int | None]) -> np.ndarray:
        table = self.table(loop)
        out = np.zeros((len(others), self.width), dtype=np.float32)
        for n, c in enumerate(others):
            if c is not None:
                out[n] = table[c]
        return out

    def table(self, loop: Any) -> np.ndarray:
        """``[B, width]`` each row's part as another aircraft at the loop's present tick (module docstring)."""
        t = int(loop.speaking.t)
        if self._table is None or self._table[0] is not loop or self._table[1] != t:
            parts = []
            for b, heard in enumerate(loop.speaking.speaker.heard):
                # at the time of its last word (the heading and the levels do not depend on it): its times since a
                # word, not read here, are then not negative
                _, _, heading, levels, _ = heard.inputs(float(heard.said_s.max()))
                parts.append(commanded_part(bool(loop.silent[b]), heading, levels, self.words))
            self._table = (loop, t, np.stack(parts) if parts else np.zeros((0, self.width), dtype=np.float32))
        return self._table[2]
