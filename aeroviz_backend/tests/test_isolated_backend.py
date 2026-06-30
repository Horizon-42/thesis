import multiprocessing
import os
import threading
import time
import unittest
from unittest import mock

from aeroviz_backend import isolation_probes
from aeroviz_backend.isolated_backend import IsolatedRunner, SolverCrashError


def _wait_until(predicate, timeout=10.0, interval=0.05):
    """Poll predicate() until true or timeout; returns the final value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


class TestIsolatedRunnerCalls(unittest.TestCase):
    """The core isolation contract: results round-trip, errors are distinguished
    from crashes, and a native abort is contained + self-healed."""

    def test_runs_in_subprocess_and_returns_result(self):
        runner = IsolatedRunner("probe")
        self.assertEqual(runner.run(isolation_probes.echo, 7), {"echoed": 7})

    def test_value_error_propagates_not_treated_as_crash(self):
        runner = IsolatedRunner("probe")
        with self.assertRaises(ValueError):
            runner.run(isolation_probes.raise_value_error, None)

    def test_worker_abort_raises_clean_error_and_self_heals(self):
        runner = IsolatedRunner("probe")
        with self.assertRaises(SolverCrashError):
            runner.run(isolation_probes.abort_worker, None)
        # The dead worker was dropped; the next call spawns a fresh one.
        self.assertEqual(runner.run(isolation_probes.echo, "ok"), {"echoed": "ok"})

    def test_disabled_isolation_runs_in_process(self):
        runner = IsolatedRunner("probe")
        sentinel = object()
        with mock.patch.dict(os.environ, {"AEROVIZ_ISOLATE_SOLVER": "0"}):
            # An un-picklable closure returning a local object proves the call ran
            # in-process (it could not have crossed a spawn boundary).
            result = runner.run(lambda payload: (sentinel, payload), 5)
        self.assertEqual(result, (sentinel, 5))


class TestEphemeralLifecycle(unittest.TestCase):
    """With no session open, every call uses a throwaway worker that is fully
    reclaimed afterwards."""

    def test_ephemeral_calls_use_a_fresh_worker_each_time(self):
        runner = IsolatedRunner("probe")
        pid1 = runner.run(isolation_probes.worker_pid, None)
        pid2 = runner.run(isolation_probes.worker_pid, None)
        self.assertNotEqual(pid1, pid2)

    def test_no_worker_process_or_thread_lingers_after_call(self):
        runner = IsolatedRunner("probe")
        children_before = len(multiprocessing.active_children())
        threads_before = threading.active_count()

        for _ in range(3):
            self.assertEqual(runner.run(isolation_probes.echo, 1), {"echoed": 1})

        self.assertLessEqual(len(multiprocessing.active_children()), children_before)
        self.assertLessEqual(threading.active_count(), threads_before)


class TestResidentSessionLifecycle(unittest.TestCase):
    """A session keeps ONE worker resident (warm) while a tab is open, and
    reclaims it when the tab closes — without leaking a process or thread."""

    def test_open_keeps_one_resident_worker_reused_across_solves(self):
        runner = IsolatedRunner("probe")
        runner.open_session()
        try:
            pid1 = runner.run(isolation_probes.worker_pid, None)
            pid2 = runner.run(isolation_probes.worker_pid, None)
            self.assertEqual(pid1, pid2)  # same worker reused -> resident/warm
        finally:
            runner.close_session()

    def test_close_reclaims_the_worker_and_stops_the_watchdog(self):
        runner = IsolatedRunner("probe")
        children_before = len(multiprocessing.active_children())
        runner.open_session()
        runner.run(isolation_probes.echo, 1)  # spawns the resident worker
        self.assertTrue(
            _wait_until(lambda: len(multiprocessing.active_children()) > children_before),
            "resident worker should be alive while the session is open",
        )
        remaining = runner.close_session()
        self.assertEqual(remaining, 0)
        # Worker reclaimed (async shutdown) and the idle watchdog thread stopped.
        self.assertTrue(
            _wait_until(
                lambda: len(multiprocessing.active_children()) <= children_before
            ),
            "worker must be reclaimed after close",
        )
        self.assertTrue(
            _wait_until(
                lambda: not any(
                    t.name == "probe-idle-watchdog" for t in threading.enumerate()
                )
            ),
            "idle watchdog thread must stop after close",
        )

    def test_refcount_keeps_worker_until_the_last_close(self):
        runner = IsolatedRunner("probe")
        self.assertEqual(runner.open_session(), 1)
        self.assertEqual(runner.open_session(), 2)
        pid1 = runner.run(isolation_probes.worker_pid, None)
        # One close: still one tab open -> same resident worker.
        self.assertEqual(runner.close_session(), 1)
        pid2 = runner.run(isolation_probes.worker_pid, None)
        self.assertEqual(pid1, pid2)
        # Last close: worker reclaimed -> a later call spawns a new one.
        self.assertEqual(runner.close_session(), 0)
        runner.open_session()
        try:
            pid3 = runner.run(isolation_probes.worker_pid, None)
            self.assertNotEqual(pid1, pid3)
        finally:
            runner.close_session()

    def test_resident_worker_abort_self_heals_within_the_session(self):
        runner = IsolatedRunner("probe")
        runner.open_session()
        try:
            with self.assertRaises(SolverCrashError):
                runner.run(isolation_probes.abort_worker, None)
            # Session still open: the next call recreates a resident worker.
            self.assertEqual(runner.run(isolation_probes.echo, "ok"), {"echoed": "ok"})
        finally:
            runner.close_session()

    def test_idle_watchdog_reclaims_a_session_that_was_never_closed(self):
        # Leak backstop: a tab opened but close() was lost (browser crash). The
        # watchdog must reclaim the worker on its own.
        runner = IsolatedRunner("probe", idle_timeout_s=0.3)
        children_before = len(multiprocessing.active_children())
        runner.open_session()
        runner.run(isolation_probes.echo, 1)  # resident worker alive
        self.assertTrue(
            _wait_until(lambda: len(multiprocessing.active_children()) > children_before)
        )
        # Do NOT close. The watchdog must reclaim it after the idle timeout.
        self.assertTrue(
            _wait_until(
                lambda: len(multiprocessing.active_children()) <= children_before,
                timeout=5.0,
            ),
            "idle watchdog must reclaim an unclosed worker",
        )
        # And it reset the ref-count, so a fresh open starts a new session at 1.
        self.assertEqual(runner.open_session(), 1)
        runner.close_session()

    def test_disabled_isolation_makes_sessions_a_noop(self):
        runner = IsolatedRunner("probe")
        with mock.patch.dict(os.environ, {"AEROVIZ_ISOLATE_SOLVER": "0"}):
            self.assertEqual(runner.open_session(), 0)
            self.assertEqual(runner.close_session(), 0)
            self.assertEqual(len(multiprocessing.active_children()), 0)


if __name__ == "__main__":
    unittest.main()
