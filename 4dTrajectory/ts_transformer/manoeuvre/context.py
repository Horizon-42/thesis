"""The prior's context tokens (plan §2.5): the aircraft type and the runway.

* `TypeVocabulary`: the dynamics type codes of the fitted cohort (`scenario.aircraft.code`, the
  openap airframe the executor's condition vector is built from), index 0 reserved for a type
  the vocabulary has not seen. Serialised into the prior's artefact.
* `runway_token`: P2–P3 run runway-KNOWN, and the sequence's states are already in that
  runway's threshold chart, so the token is the course alone as (cos, sin) — the one thing the
  chart does not carry. P4 replaces it with one token per candidate runway end (entry position
  and course in the airport frame) and the procedure fixes (`outputs/guidance/skeleton.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

UNKNOWN_TYPECODE = "<unk>"
RUNWAY_TOKEN_WIDTH = 2


@dataclass(frozen=True)
class TypeVocabulary:
    """Sorted type codes; ``index(code)`` is 0 for an unseen code."""

    codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.codes)) != len(self.codes) or UNKNOWN_TYPECODE in self.codes:
            raise ValueError("type codes are unique and never the unknown marker")

    @classmethod
    def from_typecodes(cls, typecodes: Iterable[str]) -> TypeVocabulary:
        return cls(tuple(sorted(set(str(code) for code in typecodes))))

    @property
    def size(self) -> int:
        return len(self.codes) + 1

    def index(self, typecode: str) -> int:
        try:
            return self.codes.index(typecode) + 1
        except ValueError:
            return 0

    def to_dict(self) -> dict[str, Any]:
        return {"unknown": UNKNOWN_TYPECODE, "codes": list(self.codes)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TypeVocabulary:
        return cls(tuple(data["codes"]))


def runway_token(course_rad: float) -> np.ndarray:
    """``(cos, sin)`` of the landed runway's course (math-ENU, the target's ψ)."""
    return np.array([math.cos(course_rad), math.sin(course_rad)], dtype=np.float32)


__all__ = ["RUNWAY_TOKEN_WIDTH", "UNKNOWN_TYPECODE", "TypeVocabulary", "runway_token"]
