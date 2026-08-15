"""Shared discriminators for the policy-free runway-threshold event.

The producer and evaluator intentionally live in different packages.  These
constants are the small neutral seam between them, preventing method/schema
strings from drifting without making evaluation depend on harvest I/O code.
"""

from __future__ import annotations


EVENT_SCHEMA_VERSION = "runway-threshold-event-v1"
DIRECT_EVENT_METHOD = "direct_linear_bracket"
CENSORED_EVENT_METHOD = "censored_robust_line"
NO_EVENT_METHOD = "none"

ESTIMATED_OBSERVABILITY_BY_METHOD = {
    DIRECT_EVENT_METHOD: "within_observed_support",
    CENSORED_EVENT_METHOD: "right_censored",
}
UNAVAILABLE_OBSERVABILITIES = frozenset({"invalid_support", "unavailable"})

