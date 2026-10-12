#!/usr/bin/env python
"""Q4 (the user, 2026-10-12): two end-of-track rules for the angle words, on `lead_eval_1000`'s 999 train flights at
leads 4/4/4 (D9). Compared, flight by flight, with `lead_eval_1000_combo`'s arm combo_a4_s4 (the same flights, the
reading without a tail rule).

THE FINAL STRAIGHT-IN ROWS are chosen as the evaluation chooses its fit's samples (`final_approach.fit`'s
`_final_inbound_run` + `_straight_final_suffix`, on the labeller's smoothed track in the landed runway's frame): the last
inbound run, straight on the centreline. (Distance before the threshold alone is also small on a downwind leg abeam
the runway — the first version of this script used it and is superseded.)

  A  no new angle word is read from a descending piece that starts on the final straight-in rows within 300 m of the
     threshold (the evaluation window's inner edge, `final_approach.fit.DEFAULT_WINDOW_M`): the piece joins the piece
     before it, whose word stays in force.
  B  the last piece of the final descent takes the path angle of the line fitted over the final straight-in rows
     5000–300 m before the threshold (the evaluation window and sample choice; OLS of the labeller's smoothed height on
     the distance before the threshold); where there is no such fit, the piece is kept as read.

Prototype only: patches `vertical.vertical_pieces` and `read.read_heights` in this process; the repository is not
edited. The rule's own counts (pieces merged / last angles replaced, the class changes) go to <out>/rule_counts.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lead_eval_1000 as run   # noqa: E402  (puts the worktree on sys.path)

import numpy as np   # noqa: E402

from final_approach import fit as fit_module                                                # noqa: E402
from final_approach.fit import DEFAULT_MIN_SAMPLES, DEFAULT_MIN_SPAN_M, DEFAULT_WINDOW_M   # noqa: E402
from final_approach.frame import Projected                                                 # noqa: E402
from ts_transformer.instructions.labeller import read as read_module                       # noqa: E402
from ts_transformer.instructions.labeller import vertical                                  # noqa: E402

OUTER_M, INNER_M = -DEFAULT_WINDOW_M[0], -DEFAULT_WINDOW_M[1]     # 5000, 300 m before the threshold
STATE = {"rule": None, "to_go": None, "right": None}
COUNTS = Counter()
original_pieces = vertical.vertical_pieces
original_heights = read_module.read_heights


def angle_class(words, angle_deg):
    try:
        return words.angle_index(angle_deg)
    except ValueError:
        return None


def straight_in(to_go, right, height, window):
    """The final straight-in rows inside ``window`` (along-track, negative before the threshold), as the evaluation
    chooses them; empty when there are none."""
    projected = [Projected(-float(t), float(c), float(h)) for t, c, h in zip(to_go, right, height)]
    run = fit_module._final_inbound_run(projected, window)
    suffix = fit_module._straight_final_suffix(run, min_samples=DEFAULT_MIN_SAMPLES, min_span_m=DEFAULT_MIN_SPAN_M)
    return np.array([index for index, _ in suffix], dtype=np.int64)


def rule_a(pieces, altitude, to_go, spec, words):
    final = set(straight_in(to_go, STATE["right"], altitude, (-math.inf, 0.0)).tolist())
    out = []
    for piece in pieces:
        previous = out[-1] if out else None
        if (piece.kind == vertical.MOVE and piece.descending(spec) and piece.start in final and to_go[piece.start] < INNER_M
                and previous is not None and previous.kind == vertical.MOVE and previous.descending(spec)):
            COUNTS["A_pieces_merged"] += 1
            COUNTS["A_class_dropped"] += angle_class(words, piece.angle_deg) != angle_class(words, previous.angle_deg)
            previous.stop = piece.stop
            continue
        out.append(piece)
    return out


def rule_b(pieces, altitude, to_go, spec, words):
    if not pieces or pieces[-1].kind != vertical.MOVE or not pieces[-1].descending(spec):
        COUNTS["B_no_final_descent"] += 1
        return pieces
    inside = straight_in(to_go, STATE["right"], altitude, DEFAULT_WINDOW_M)
    if not len(inside):
        COUNTS["B_no_fit"] += 1
        return pieces
    slope = float(np.polyfit(to_go[inside], altitude[inside], 1)[0])      # height per metre before the threshold
    fitted = math.degrees(math.atan(slope))
    last = pieces[-1]
    before, after = angle_class(words, last.angle_deg), angle_class(words, fitted)
    if after is None:
        COUNTS["B_fit_out_of_range"] += 1
        return pieces
    COUNTS["B_replaced"] += 1
    COUNTS[f"B_class_{before}_to_{after}"] += 1
    last.angle_deg = fitted
    return pieces


def patched_pieces(distance, altitude, spec, words):
    pieces = original_pieces(distance, altitude, spec, words)
    to_go = STATE["to_go"]
    assert to_go is not None and len(to_go) == len(altitude)
    return (rule_a(pieces, altitude, to_go, spec, words) if STATE["rule"] == "A"
            else rule_b(pieces, altitude, to_go, spec, words))


def patched_heights(flight, height_m, spec, words):
    STATE["to_go"], STATE["right"] = flight.relative.before_threshold_m, flight.relative.right_of_course_m
    try:
        return original_heights(flight, height_m, spec, words)
    finally:
        STATE["to_go"] = STATE["right"] = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=("A", "B"), required=True)
    args, rest = ap.parse_known_args()
    STATE["rule"] = args.rule
    vertical.vertical_pieces = patched_pieces
    read_module.read_heights = patched_heights
    run.arms = lambda: [(f"tail{args.rule}_a4_s4", {"heading": 4.0, "angle": 4.0, "speed": 4.0}, None)]
    sys.argv = [sys.argv[0], *rest]
    code = run.main()
    out = Path(rest[rest.index("--out") + 1])
    # the counts cover every reading of the run (one per flight; the closed loop reads no flight again)
    (out / "rule_counts.json").write_text(json.dumps(dict(COUNTS), indent=1))
    print(dict(COUNTS))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
