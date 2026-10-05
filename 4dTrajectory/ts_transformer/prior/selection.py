"""The selection of the base's sentences (prior design §8 item 1; D75; milestone B8): a named rule over the artefact's
closed-loop sentences, applied when they are read; the artefact is never changed.

- ``all`` keeps every sentence;
- ``landed`` keeps a sentence whose stored outcome (vocabulary §6 item 3, D74: the judge's outcome of the flight the
  closed-loop reading flew on the sentence's own words) is a landing (`LANDING`) and whose flight stage A does not mark
  as having a faulty observed track (D111: `instructions.faults`; whatever its outcome — its observed rows, which the
  prior reads, hold the fault).

A sentence left out is left out for one reason (`left_out`): its flight's faulty track (``fault``) before its outcome
(``outcome``). The mark, like the outcome, keeps or leaves a sentence and never reaches an input.

A run applies its rule to train and select (and the teacher-forced loss of the validation readout); free generation
starts from every flight. The identity of a run's data holds the rule and, for each split, airport, stratum and
outcome, the sentences kept and left out for each reason (`selection_record`), so a run under another rule, or over
other outcomes or marks, is refused by the identity's comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

RULES = ("all", "landed")
#: Why a rule leaves a sentence out, in the order it is asked: a faulty observed track (D111), then the outcome (D75).
FAULT, OUTCOME = "fault", "outcome"
REASONS = (FAULT, OUTCOME)
#: The judge's name of a landing (vocabulary §6 item 6). MIRROR of the first of `autopilot.judge.OUTCOMES`, which
#: `prior/` cannot import (it reads no `autopilot/`); `tests/test_prior_inputs.py` pins it there.
LANDING = "landed"


@dataclass(frozen=True)
class Stored:
    """What the selection reads of one closed-loop sentence: its split, its flight's airport, its stratum (D70), its
    stored outcome (D74), by name, and whether stage A marks its flight's observed track as faulty (D111)."""

    split: str
    airport: str
    stratum: str
    outcome: str
    faulty: bool


def require_rule(rule: str) -> str:
    if rule not in RULES:
        raise ValueError(f"selection rule {rule!r} is not one of {RULES}")
    return rule


def left_out(rule: str, outcome: str, faulty: bool) -> str | None:
    """Why ``rule`` leaves out a sentence of stored ``outcome`` whose flight is marked ``faulty`` (`REASONS`); None when
    it keeps it."""
    if require_rule(rule) == "all":
        return None
    if faulty:
        return FAULT
    return None if outcome == LANDING else OUTCOME


def kept(rule: str, outcome: str, faulty: bool) -> bool:
    """Whether ``rule`` keeps a sentence of stored ``outcome`` whose flight is marked ``faulty``."""
    return left_out(rule, outcome, faulty) is None


#: Where a readout puts a sentence: inside the selection, or outside it for each reason (`side`).
SIDES = ("inside", *(f"outside_{reason}" for reason in REASONS))


def side(rule: str, outcome: str, faulty: bool) -> str:
    """The `SIDES` of a sentence under ``rule`` (the readouts give each apart, D75, D111)."""
    reason = left_out(rule, outcome, faulty)
    return "inside" if reason is None else f"outside_{reason}"


#: A cell of the record: the sentences kept, and left out for each reason.
CELL = ("kept", *(f"left_out_{reason}" for reason in REASONS))


def selection_record(rule: str, stored: Iterable[Stored]) -> dict[str, Any]:
    """The rule and, for each split, airport, stratum and outcome, the sentences kept and left out for each reason (§8
    item 1; D111: a faulty track apart from the outcome)."""
    counts: dict[str, Any] = {}
    for item in stored:
        cell = (counts.setdefault(item.split, {}).setdefault(item.airport, {}).setdefault(item.stratum, {})
                .setdefault(item.outcome, dict.fromkeys(CELL, 0)))
        reason = left_out(rule, item.outcome, item.faulty)
        cell["kept" if reason is None else f"left_out_{reason}"] += 1
    return {"rule": require_rule(rule), "counts": counts}


def selection_totals(record: dict[str, Any]) -> dict[str, dict[str, int]]:
    """A `selection_record`'s sentences kept and left out for each reason, for each split, summed over the airports,
    strata and outcomes (what the readouts print)."""
    return {split: {side: sum(cell[side] for airport in airports.values() for stratum in airport.values()
                              for cell in stratum.values()) for side in CELL}
            for split, airports in record["counts"].items()}
