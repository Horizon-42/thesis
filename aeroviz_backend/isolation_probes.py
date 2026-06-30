"""isolation_probes.py
=====================
Tiny worker functions used to exercise :class:`IsolatedRunner` in tests.

They live in the installed ``aeroviz_backend`` package (not the un-packaged
``tests/`` directory) so the ``spawn`` start method can import them by their
stable dotted name inside the worker process.
"""

from __future__ import annotations

import os
from typing import Any


def echo(payload: Any) -> dict[str, Any]:
    """Return the payload, to verify a value round-trips through a worker."""
    return {"echoed": payload}


def worker_pid(_payload: Any) -> int:
    """Return the worker's PID — a stable PID across calls proves the worker is
    resident (reused); a changing PID proves an ephemeral worker per call."""
    return os.getpid()


def abort_worker(_payload: Any) -> Any:
    """Kill the worker hard (no cleanup) to simulate a native solver abort.

    Exit code 134 = 128 + SIGABRT, the same death casadi's ``abort()`` produces.
    """
    os._exit(134)


def raise_value_error(_payload: Any) -> Any:
    """Raise a normal Python error — it must propagate as ``ValueError`` (a lost
    request), not be mistaken for a worker crash."""
    raise ValueError("bad input")
