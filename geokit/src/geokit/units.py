"""Speed and length unit conversions.

One place for every unit conversion in the project, so nothing re-wraps the factors in
its own helper. Speed conversions work to/from metres/second (knots for ground speed,
feet/minute for vertical rate; km/h and mph for completeness); length conversions
to/from metres (nautical miles, feet). The raw factors live in :mod:`geokit.constants`.
"""

from __future__ import annotations

from .constants import FT_M, FT_MIN_MS, KMH_MS, KT_MS, MPH_MS, NM_M

__all__ = [
    # speed
    "kt_to_ms",
    "ms_to_kt",
    "ft_min_to_ms",
    "ms_to_ft_min",
    "kmh_to_ms",
    "ms_to_kmh",
    "mph_to_ms",
    "ms_to_mph",
    # length
    "nm_to_m",
    "m_to_nm",
    "ft_to_m",
    "m_to_ft",
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


# ── Length ───────────────────────────────────────────────────────────────────

def nm_to_m(nautical_miles: float) -> float:
    """Nautical miles -> metres."""
    return nautical_miles * NM_M


def m_to_nm(metres: float) -> float:
    """Metres -> nautical miles."""
    return metres / NM_M


def ft_to_m(feet: float) -> float:
    """Feet -> metres."""
    return feet * FT_M


def m_to_ft(metres: float) -> float:
    """Metres -> feet."""
    return metres / FT_M
