"""What the column readings hand to the sentence assembly: instructions, and the one way a
flight is refused."""

from __future__ import annotations

from dataclasses import dataclass, field
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
