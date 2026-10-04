"""The selection of the base's sentences (prior design §8 item 1; D75; milestone B8): a named rule over the artefact's
closed-loop sentences, applied when they are read; the artefact is never changed.

- ``all`` keeps every sentence;
- ``landed`` keeps a sentence whose stored outcome (vocabulary §6 item 3, D74: the judge's outcome of the flight the
  closed-loop reading flew on the sentence's own words) is a landing (`LANDING`).

A run applies its rule to train and select (and the teacher-forced loss of the validation readout); free generation
starts from every flight. The identity of a run's data holds the rule and, for each split, airport, stratum and
outcome, the sentences kept and left out (`selection_record`), so a run under another rule, or over other outcomes, is
refused by the identity's comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

RULES = ("all", "landed")
#: The judge's name of a landing (vocabulary §6 item 6). MIRROR of the first of `autopilot.judge.OUTCOMES`, which
#: `prior/` cannot import (it reads no `autopilot/`); `tests/test_prior_inputs.py` pins it there.
LANDING = "landed"


@dataclass(frozen=True)
class Stored:
    """What the selection reads of one closed-loop sentence: its split, its flight's airport, its stratum (D70) and its
    stored outcome (D74), by name."""

    split: str
    airport: str
    stratum: str
    outcome: str


def require_rule(rule: str) -> str:
    if rule not in RULES:
        raise ValueError(f"selection rule {rule!r} is not one of {RULES}")
    return rule


def kept(rule: str, outcome: str) -> bool:
    """Whether ``rule`` keeps a sentence of stored ``outcome``."""
    return require_rule(rule) == "all" or outcome == LANDING


def selection_record(rule: str, stored: Iterable[Stored]) -> dict[str, Any]:
    """The rule and, for each split, airport, stratum and outcome, the sentences kept and left out (§8 item 1)."""
    counts: dict[str, Any] = {}
    for item in stored:
        cell = (counts.setdefault(item.split, {}).setdefault(item.airport, {}).setdefault(item.stratum, {})
                .setdefault(item.outcome, {"kept": 0, "left_out": 0}))
        cell["kept" if kept(rule, item.outcome) else "left_out"] += 1
    return {"rule": require_rule(rule), "counts": counts}


def selection_totals(record: dict[str, Any]) -> dict[str, dict[str, int]]:
    """A `selection_record`'s sentences kept and left out for each split, summed over the airports, strata and outcomes
    (what the readouts print)."""
    return {split: {side: sum(cell[side] for airport in airports.values() for stratum in airport.values()
                              for cell in stratum.values()) for side in ("kept", "left_out")}
            for split, airports in record["counts"].items()}
