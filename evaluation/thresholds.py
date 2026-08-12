"""Assessment context and standards-backed threshold resolution.

The evaluator does not own runway or procedure data.  A producer supplies an
explicit :class:`AssessmentContext`; this module only turns that context into
the LPV or RNP APCH LNAV/VNAV limits documented in
``FINAL_APPROACH_VERDICT_STANDARD.md``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

Benchmark = Literal["lpv", "rnp_apch_lnav_vnav_baro"]
ComponentResult = Literal["pass", "fail", "indeterminate", "not_applicable"]
Verdict = Literal["pass", "fail", "indeterminate"]

LNAV_FINAL_XTK_M = 0.15 * 1852.0
BARO_VNAV_VERTICAL_M = 22.0
NORMAL_95_MULTIPLIER = 1.96


def _positive(name: str, value: float | None, *, required: bool = True) -> float | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return numeric


def _finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numeric


@dataclass(frozen=True)
class AssessmentContext:
    """Runway/procedure facts required to resolve one terminal verdict.

    No trajectory fit or verdict belongs here.  ``lpv_vertical_fsd_m`` is
    deliberately optional: until the licensed RTCA scaling model is validated,
    the U.S. pipeline leaves it ``None`` and LPV vertical remains indeterminate.
    """

    benchmark: Benchmark
    airport: str
    runway: str
    runway_course_deg: float
    runway_width_m: float
    runway_source: str
    runway_source_cycle: str
    procedure_source: str
    procedure_source_cycle: str
    lpv_lateral_fsd_m: float | None = None
    lpv_vertical_fsd_m: float | None = None
    baro_vnav_approved: bool = False

    def __post_init__(self) -> None:
        if self.benchmark not in ("lpv", "rnp_apch_lnav_vnav_baro"):
            raise ValueError(f"unsupported benchmark {self.benchmark!r}")
        for name in (
            "airport", "runway", "runway_source", "runway_source_cycle",
            "procedure_source", "procedure_source_cycle",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        _finite("runway_course_deg", self.runway_course_deg)
        _positive("runway_width_m", self.runway_width_m)
        _positive("lpv_lateral_fsd_m", self.lpv_lateral_fsd_m, required=False)
        _positive("lpv_vertical_fsd_m", self.lpv_vertical_fsd_m, required=False)
        if not isinstance(self.baro_vnav_approved, bool):
            raise ValueError("baro_vnav_approved must be boolean")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssessmentContext":
        return cls(
            benchmark=data["benchmark"],
            airport=str(data["airport"]),
            runway=str(data["runway"]),
            runway_course_deg=float(data["runway_course_deg"]),
            runway_width_m=float(data["runway_width_m"]),
            runway_source=str(data["runway_source"]),
            runway_source_cycle=str(data["runway_source_cycle"]),
            procedure_source=str(data["procedure_source"]),
            procedure_source_cycle=str(data["procedure_source_cycle"]),
            lpv_lateral_fsd_m=(
                None if data.get("lpv_lateral_fsd_m") is None
                else float(data["lpv_lateral_fsd_m"])
            ),
            lpv_vertical_fsd_m=(
                None if data.get("lpv_vertical_fsd_m") is None
                else float(data["lpv_vertical_fsd_m"])
            ),
            baro_vnav_approved=data.get("baro_vnav_approved", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def limits(self) -> "ResolvedLimits":
        runway_half = self.runway_width_m / 2.0
        if self.benchmark == "lpv":
            guidance = (
                None if self.lpv_lateral_fsd_m is None
                else self.lpv_lateral_fsd_m / 2.0
            )
            lateral = None if guidance is None else min(guidance, runway_half)
            vertical = (
                None if self.lpv_vertical_fsd_m is None
                else self.lpv_vertical_fsd_m / 2.0
            )
            return ResolvedLimits(
                benchmark=self.benchmark,
                guidance_lateral_m=guidance,
                runway_lateral_m=runway_half,
                effective_lateral_m=lateral,
                vertical_lower_m=None if vertical is None else -vertical,
                vertical_upper_m=vertical,
                vertical_reason=(
                    "LPV vertical FSD unavailable: RTCA DO-229 scaling model not validated"
                    if vertical is None else None
                ),
            )

        vertical_available = self.baro_vnav_approved
        return ResolvedLimits(
            benchmark=self.benchmark,
            guidance_lateral_m=LNAV_FINAL_XTK_M,
            runway_lateral_m=runway_half,
            effective_lateral_m=min(LNAV_FINAL_XTK_M, runway_half),
            vertical_lower_m=-BARO_VNAV_VERTICAL_M if vertical_available else None,
            vertical_upper_m=BARO_VNAV_VERTICAL_M if vertical_available else None,
            vertical_reason=(
                None
                if vertical_available
                else "LNAV/VNAV benchmark requires explicit approved Baro-VNAV context"
            ),
        )


@dataclass(frozen=True)
class ResolvedLimits:
    """Numeric component limits after resolving one assessment context."""

    benchmark: Benchmark
    guidance_lateral_m: float | None
    runway_lateral_m: float
    effective_lateral_m: float | None
    vertical_lower_m: float | None
    vertical_upper_m: float | None
    vertical_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
