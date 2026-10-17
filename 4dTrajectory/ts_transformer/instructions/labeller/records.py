"""What the column readings hand to the sentence assembly: instructions, and the one way a
flight is refused."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


class Refused(Exception):
    """A flight the labeller will not read. ``reason`` is a fixed category (it is counted);
    ``detail`` says what was found."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class Instruction:
    """One word issued at one row. ``kind`` says why it was issued (it is counted in the
    readout); ``info`` carries the column's own diagnostics (targets, envelope results)."""

    column: int
    value: int
    row: int
    kind: str
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "value": self.value, "row": self.row, "kind": self.kind, **self.info}


def said_earlier(instructions: list[Instruction], lead: int) -> list[Instruction]:
    """A column's words said ``lead`` rows before the row they were read at (a lead: the word says the value that much
    later, executor design §17, D7, D8): one shift for every word, so the words keep their order; a word that would fall
    before row 0 is said at row 0 (as "initial"), and of several words of one column at one row, the last read stands."""
    kept: dict[tuple[int, int], Instruction] = {}
    for item in instructions:
        row = max(item.row - lead, 0)
        kept[(item.column, row)] = replace(item, row=row, kind="initial" if row == 0 else item.kind)
    return list(kept.values())
