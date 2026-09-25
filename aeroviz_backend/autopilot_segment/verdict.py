"""The selected word's verdict, read off the executor's judge's verdict of the segment (`autopilot.judge.judge`).

The word is the one at the segment's step 0 in its column; the judge judged the segment's sentence, whose words are
each drawn from where the executor was told them — the selected word's from the observed state it was said at. A word
inside or outside carries its checks and they decide it (the reader's rule: inside exactly when every one passed); any
other status says why instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ts_transformer.autopilot.judge import Verdict
from ts_transformer.instructions.words import ALTITUDE, ANGLE, APPROACH, HEADING, SPEED, Words

from aeroviz_backend.autopilot_segment.segment import Segment

#: MIRROR of `aeroviz-4d/src/data/trainingAutopilot.ts` (`TRAINING_AUTOPILOT_STATUSES`): the selected word's verdict
#: (pinned by `test_autopilot_segment.MirrorTest`).
STATUSES = ("inside", "outside", "not judged", "no check")
INSIDE, OUTSIDE, NOT_JUDGED, NO_CHECK = STATUSES


@dataclass(frozen=True)
class HeadingFacts:
    """What the judge's verdict does not say of a selected heading word, read off the flown segment: the cycles the
    executor left it to intercept the final on its own while it was in force (before the next heading word was heard),
    and how many flown rows its judge read. (The clock never skips it — `replay.skipped_by_clock`: it is told alone, at
    the first flown step.)"""
    off_word_cycles: int
    judged_rows: int


def _check(name: str, ok: bool, inside: int | None = None, rows: int | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "inside": inside, "rows": rows}


def _settled(checks: list[dict[str, Any]], reason: str | None = None) -> dict[str, Any]:
    """A word judged by its checks: inside exactly when every one passed."""
    return {"status": INSIDE if all(check["ok"] for check in checks) else OUTSIDE, "checks": checks, "reason": reason}


def _not_judged(reason: str) -> dict[str, Any]:
    return {"status": NOT_JUDGED, "checks": [], "reason": reason}


def _no_check(reason: str) -> dict[str, Any]:
    return {"status": NO_CHECK, "checks": [], "reason": reason}


def selected_heading(judged: dict[str, Any]) -> dict[str, int]:
    """The judge's result for the selected heading word: the one told at the segment's step 0 (the next heading word,
    told at its ``end_row``, is judged on whatever rows the segment flies past its lead — never the selected word's)."""
    (result,) = [item for item in judged["heading"] if item["row"] == 0]
    return result


def _heading_verdict(judged: dict[str, Any], spec: Any, facts: HeadingFacts) -> dict[str, Any]:
    """A heading word: its band's rows, and — whatever its rows — whether the executor left it to intercept the final on
    its own (the judge fails that word, as the replay's export does)."""
    result = selected_heading(judged)
    checks = []
    if result["rows"]:
        checks.append(_check(f"track within ±{spec.heading_tolerance_deg:g}° of the word, "
                             f"{spec.heading_lead_s:g} s after it was told, to the next heading word's",
                             result["inside"] == result["rows"], result["inside"], result["rows"]))
    if facts.off_word_cycles:
        checks.append(_check(f"held until the capture — left for {facts.off_word_cycles} cycles to intercept the final "
                             "on its own", False))
    if checks:
        return _settled(checks)
    lead = spec.heading_lead_s
    if spec.rows_exact(lead) >= facts.judged_rows:
        return _not_judged(f"no row to judge: its rows, {lead:g} s after it was told, begin past the end of the flown "
                           "track its judge read")
    return _not_judged(f"no row to judge: its rows, {lead:g} s after it was told, begin at or past the clearance the "
                       "executor was told or its capture (the capture turn is judged in its place)")


def word_verdict(verdict: Verdict, segment: Segment, spec: Any, words: Words, heading: HeadingFacts | None) -> dict[str, Any]:
    """The selected word's verdict — the word at the segment's step 0 in its column — read off the judge's verdict of
    the segment. ``heading``: the flown facts a heading word's verdict needs (`HeadingFacts`); None for any other word,
    and when the gate refused the flown segment."""
    column = segment.column
    word = segment.instructions[column]
    if verdict.words is None:
        return _not_judged(f"the flown segment did not pass the labeller's gate ({verdict.refused})")
    judged = verdict.words
    if column == HEADING:
        if heading is None:
            raise ValueError("a heading word's verdict needs its flown facts")
        return _heading_verdict(judged, spec, heading)
    if column == APPROACH:
        if word.kind != "clear":
            return _no_check("not cleared, or a go-around: no envelope of its own (a clearance is judged by its capture "
                             "turn and corridor)")
        corridor, capture = judged["corridor"], judged["capture_turn"]
        checks = ([_check("captured the final", False)] if capture is None else
                  [_check("capture turn monotone", capture["progress_ok"]),
                   _check("capture turn rate and bank", capture["rate_ok"])])
        checks += [_check("corridor entered", corridor["entered"]),
                   _check("corridor held", corridor["inside"] == corridor["rows"], corridor["inside"], corridor["rows"])]
        return _settled(checks)
    if column == ALTITUDE:
        (tube,) = [item for item in judged["vertical"] if item["row"] == 0]
        return _settled([_check("in its tube", tube["contained"], tube["inside"], tube["rows"])])
    if column == ANGLE:
        # an angle word re-anchors the tube of every altitude word it is in force over (the one in force at its step,
        # told with it at the segment's step 0, and any said before the segment's end): judged in each
        def named(target_m: float | None) -> str:
            return "descend to land" if target_m is None else f"{target_m:.0f} m"

        checks = [_check(f"in the tube of the altitude word {named(tube['target_m'])}", tube["contained"], tube["inside"],
                         tube["rows"]) for tube in judged["vertical"]]
        return _settled(checks, "judged in the altitude tubes it anchors")
    if column == SPEED:
        if words.speed_mps(word.value) is None:
            return _no_check("the pilot's own speed: no band to hold")
        (span,) = [item for item in judged["speed"] if item["row"] == 0]
        checks = [_check("transition monotone toward the target", span["transition_ok"]),
                  _check("band held" if not span["cut_before_arrival"] else "band not reached before the segment's end",
                         span["band_inside"] == span["band_rows"], span["band_inside"], span["band_rows"])]
        # the judge's verdict is the transition and the band (`contained`); its acceleration check is not part of it
        result = _settled(checks)
        if (result["status"] == INSIDE) != bool(span["contained"]):
            raise ValueError(f"the speed word's checks say {result['status']}, the judge says contained={span['contained']}")
        return result
    return _no_check("the runway pointer: judged by the landing, the flight's outcome")
