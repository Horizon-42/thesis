"""isolated_backend.py
=====================
Run the casadi-heavy backend calls (the trajectory optimizer and the dynamics
comparison) in a **worker subprocess**, so a *native* crash in casadi / IPOPT
cannot take down the HTTP server.

Why this exists
---------------
casadi's solver stack aborts the **whole process** on certain inputs: building
the IPOPT NLP computes a symbolic Jacobian by automatic differentiation, and a
heap fault there raises ``SIGABRT`` (``Abort trap: 6``) from C++ ``abort()``.
That is a process-level signal, **not** a Python exception — the ``try/except``
in ``http_server`` cannot catch it, so the entire backend dies. The launcher
then tears the frontend down with it, and the whole "service" appears to shut
itself down.

Containment + self-healing
---------------------------
``IsolatedRunner`` submits the call to a worker process (``ProcessPoolExecutor(
max_workers=1)``). If the worker aborts, ``Future.result()`` raises
``BrokenProcessPool``; the runner drops the dead worker and re-raises a clean
:class:`SolverCrashError` (which ``http_server`` turns into a 500). The next
request spawns a fresh worker — the server self-heals and keeps running.

Worker lifecycle (resident while the tab is open)
-------------------------------------------------
A persistent worker keeps casadi's compiled-NLP cache warm (fast repeated
solves) but holds hundreds of MB of casadi + numpy resident — dead weight on a
memory-constrained machine once the user leaves the optimizer. So the worker is
tied to the frontend tab's lifecycle:

* ``open_session()``  — the Optimize/Compare tab opened: ref-count up, spawn the
  resident worker if needed (warm).
* ``close_session()`` — the tab closed: ref-count down; at zero, shut the worker
  down and reclaim ALL its memory.
* a call with no session open falls back to an **ephemeral** worker (spawned for
  that one call, torn down after), so the endpoint still works.

Leak backstop
-------------
``close_session`` can be lost (browser crash, a dropped unmount). An idle
watchdog reclaims the resident worker if it goes unused past
``AEROVIZ_WORKER_IDLE_TIMEOUT_S`` (default 600 s) regardless of the ref-count, so
a stranded worker can never linger indefinitely. The watchdog thread stops
itself once the worker is gone (no idle thread left resident either).

Set ``AEROVIZ_ISOLATE_SOLVER=0`` to run everything in-process (useful when you
*want* the native traceback while reproducing a crash); the session calls then
become no-ops.
"""

from __future__ import annotations

import concurrent.futures as _futures
import multiprocessing as _mp
import os
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable


def _isolation_enabled() -> bool:
    return os.environ.get("AEROVIZ_ISOLATE_SOLVER", "1") != "0"


def _default_idle_timeout_s() -> float:
    return float(os.environ.get("AEROVIZ_WORKER_IDLE_TIMEOUT_S", "600"))


class SolverCrashError(RuntimeError):
    """A solver worker died on a native abort (e.g. casadi SIGABRT). The request
    is lost, but the server stays up and the next request gets a fresh worker."""


class IsolatedRunner:
    """Owns a casadi worker subprocess and its tab-driven lifecycle.

    A single worker (``max_workers=1``) is deliberate: solves are CPU-bound, so
    concurrency would not help throughput, and serial use keeps casadi's global
    state uncontended. Solves are serialised so at most one worker (one casadi
    footprint) ever runs at a time.
    """

    def __init__(self, name: str, idle_timeout_s: float | None = None) -> None:
        self._name = name
        self._context = _mp.get_context("spawn")
        self._idle_timeout_s = (
            _default_idle_timeout_s() if idle_timeout_s is None else idle_timeout_s
        )
        # Check often enough to reclaim promptly, but never busy-spin.
        self._watchdog_interval_s = max(0.05, min(self._idle_timeout_s / 4.0, 30.0))

        self._state_lock = threading.RLock()  # guards the fields below
        self._solve_lock = threading.Lock()  # serialises solves -> one footprint
        self._executor: _futures.ProcessPoolExecutor | None = None
        self._refcount = 0
        self._last_used = 0.0  # time.monotonic() of the last open/run
        self._watchdog: threading.Thread | None = None
        self._watchdog_stop: threading.Event | None = None

    # ── tab lifecycle ─────────────────────────────────────────────────────────

    def open_session(self) -> int:
        """A tab opened: ref-count up and warm the resident worker. Returns the
        new open-session count."""
        if not _isolation_enabled():
            return 0
        with self._state_lock:
            self._refcount += 1
            self._last_used = time.monotonic()
            if self._executor is None:
                self._executor = self._new_executor()
            self._ensure_watchdog_locked()
            return self._refcount

    def close_session(self) -> int:
        """A tab closed: ref-count down; at zero, shut the worker down and reclaim
        its memory. Returns the remaining open-session count."""
        if not _isolation_enabled():
            return 0
        with self._state_lock:
            if self._refcount > 0:
                self._refcount -= 1
            if self._refcount == 0:
                self._discard_executor_locked()
            return self._refcount

    # ── the actual work ───────────────────────────────────────────────────────

    def run(self, func: Callable[[Any], Any], payload: Any) -> Any:
        if not _isolation_enabled():
            return func(payload)
        with self._solve_lock:
            with self._state_lock:
                self._last_used = time.monotonic()
                if self._refcount > 0:
                    # A tab is open: (re)use a resident worker. Recreate it if a
                    # prior crash discarded it, so the session self-heals.
                    if self._executor is None:
                        self._executor = self._new_executor()
                        self._ensure_watchdog_locked()
                    executor, ephemeral = self._executor, False
                else:
                    # No tab open — serve this one call from a throwaway worker.
                    executor, ephemeral = self._new_executor(), True
            try:
                return executor.submit(func, payload).result()
            except BrokenProcessPool as exc:
                # The worker aborted (native crash). If it was the resident worker,
                # drop it so the next call respawns a fresh one (self-heal).
                with self._state_lock:
                    if executor is self._executor:
                        self._discard_executor_locked()
                raise SolverCrashError(
                    f"the {self._name} worker crashed (native solver abort); the "
                    "request was aborted but the backend is still running — retry it"
                ) from exc
            finally:
                with self._state_lock:
                    self._last_used = time.monotonic()
                if ephemeral:
                    executor.shutdown(wait=True, cancel_futures=True)

    # ── internals ─────────────────────────────────────────────────────────────

    def _new_executor(self) -> _futures.ProcessPoolExecutor:
        return _futures.ProcessPoolExecutor(max_workers=1, mp_context=self._context)

    def _discard_executor_locked(self) -> None:
        # Non-blocking shutdown: an idle worker exits promptly; a worker still
        # finishing a solve exits when that solve returns. Either way memory is
        # reclaimed without blocking the caller (e.g. a tab-close request).
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._stop_watchdog_locked()

    def _ensure_watchdog_locked(self) -> None:
        if self._watchdog is None:
            self._watchdog_stop = threading.Event()
            self._watchdog = threading.Thread(
                target=self._watch,
                args=(self._watchdog_stop,),
                name=f"{self._name}-idle-watchdog",
                daemon=True,
            )
            self._watchdog.start()

    def _stop_watchdog_locked(self) -> None:
        if self._watchdog_stop is not None:
            self._watchdog_stop.set()
        self._watchdog = None
        self._watchdog_stop = None

    def _watch(self, stop: threading.Event) -> None:
        while not stop.wait(self._watchdog_interval_s):
            with self._state_lock:
                if stop.is_set():
                    return
                if self._executor is None:
                    self._stop_watchdog_locked()
                    return
                idle_for = time.monotonic() - self._last_used
                if idle_for > self._idle_timeout_s:
                    # Leak backstop: a session was opened but never closed. Force
                    # the ref-count to zero and reclaim — the watchdog stops itself
                    # inside _discard_executor_locked.
                    self._refcount = 0
                    self._discard_executor_locked()
                    return


# ── Worker entry points ───────────────────────────────────────────────────────
# Module-level (so the 'spawn' start method can pickle them by reference) and each
# reuses one backend per worker process, keeping casadi's compiled-NLP cache warm
# across requests in that worker.

_WORKER_OPTIMIZER: Any = None
_WORKER_COMPARISON: Any = None


def _worker_optimize(payload: Any) -> Any:
    global _WORKER_OPTIMIZER
    if _WORKER_OPTIMIZER is None:
        from aeroviz_backend.optimization_backend import OptimizationBackend

        _WORKER_OPTIMIZER = OptimizationBackend()
    return _WORKER_OPTIMIZER.optimize(payload)


def _worker_compare(payload: Any) -> Any:
    global _WORKER_COMPARISON
    if _WORKER_COMPARISON is None:
        from aeroviz_backend.dynamics_comparison_backend import DynamicsComparisonBackend

        _WORKER_COMPARISON = DynamicsComparisonBackend()
    return _WORKER_COMPARISON.run(payload)


# ── Backend decorators ────────────────────────────────────────────────────────
# Same call interface as the wrapped backends, so the app delegates to them
# transparently. Only the casadi-heavy methods are isolated; the comparison's
# history endpoints are pure numpy / file IO and stay in-process (sharing the same
# HISTORY_DIR), so they pay no subprocess cost.


class IsolatedOptimizationBackend:
    def __init__(self) -> None:
        self._runner = IsolatedRunner("optimizer")

    def optimize(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._runner.run(_worker_optimize, payload)

    def open_session(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"ok": True, "sessions": self._runner.open_session()}

    def close_session(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"ok": True, "sessions": self._runner.close_session()}


class IsolatedDynamicsComparisonBackend:
    def __init__(self) -> None:
        from aeroviz_backend.dynamics_comparison_backend import DynamicsComparisonBackend

        self._runner = IsolatedRunner("dynamics-comparison")
        self._inprocess = DynamicsComparisonBackend()

    def run(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._runner.run(_worker_compare, payload)

    def open_session(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"ok": True, "sessions": self._runner.open_session()}

    def close_session(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"ok": True, "sessions": self._runner.close_session()}

    def history_count(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._inprocess.history_count(payload)

    def average(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._inprocess.average(payload)

    def clear(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._inprocess.clear(payload)
