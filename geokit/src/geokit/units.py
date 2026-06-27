"""Speed unit conversions.

One place for every speed conversion in the project. Internally everything works in
metres/second; these convert to and from the units the data and aircraft specs use
(knots for ground speed, feet/minute for vertical rate). km/h and mph are included for
completeness. The length/angle factors live in :mod:`geokit.constants`.
"""

from __future__ import annotations

from .constants import FT_MIN_MS, KMH_MS, KT_MS, MPH_MS

__all__ = [
    "kt_to_ms",
    "ms_to_kt",
    "ft_min_to_ms",
    "ms_to_ft_min",
    "kmh_to_ms",
    "ms_to_kmh",
    "mph_to_ms",
    "ms_to_mph",
]


# ── Knots (nautical miles per hour) ──────────────────────────────────────────

def kt_to_ms(knots: float) -> float:
    """Knots -> metres/second."""
    return knots * KT_MS


def ms_to_kt(metres_per_second: float) -> float:
    """Metres/second -> knots."""
    return metres_per_second / KT_MS


# ── Feet per minute (vertical rate) ──────────────────────────────────────────

def ft_min_to_ms(feet_per_minute: float) -> float:
    """Feet/minute -> metres/second."""
    return feet_per_minute * FT_MIN_MS


def ms_to_ft_min(metres_per_second: float) -> float:
    """Metres/second -> feet/minute."""
    return metres_per_second / FT_MIN_MS


# ── Kilometres per hour ──────────────────────────────────────────────────────

def kmh_to_ms(kmh: float) -> float:
    """Kilometres/hour -> metres/second."""
    return kmh * KMH_MS


def ms_to_kmh(metres_per_second: float) -> float:
    """Metres/second -> kilometres/hour."""
    return metres_per_second / KMH_MS


# ── Miles per hour ───────────────────────────────────────────────────────────

def mph_to_ms(mph: float) -> float:
    """Miles/hour -> metres/second."""
    return mph * MPH_MS


def ms_to_mph(metres_per_second: float) -> float:
    """Metres/second -> miles/hour."""
    return metres_per_second / MPH_MS
