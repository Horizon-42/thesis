"""casadi_lock.py
===============
One process-wide lock serialising every casadi entry point that runs **in the
server process**.

Why
---
casadi's symbolic construction — the ``SXElem`` expression graph and its global
node / memory pools — is **not thread-safe**. A crash report showed three
``ThreadingHTTPServer`` request threads building casadi graphs at the same
instant (``SXElem`` construction/destruction during IPOPT NLP setup), racing the
allocator's free list into a heap corruption (``free_tiny_botch`` → ``abort()``,
which kills the whole process). casadi work is CPU-bound, so serialising it costs
no real throughput.

Scope
-----
The optimizer and dynamics-comparison endpoints run in isolated worker
subprocesses by default (see ``isolated_backend``), so they no longer share
casadi globals with anything in the server process. This lock guards what is
LEFT in the server process — the simulation endpoints, which build a
``CasadiSimulator`` on reset and evaluate it on every step — and it also guards
the optimizer / comparison **in-process fallback** used when isolation is turned
off (``AEROVIZ_ISOLATE_SOLVER=0``). Inside a worker subprocess this lock is just
the worker's own (uncontended) lock, so it is harmless there too.

It is an ``RLock`` so a casadi-guarded method may call another without
self-deadlocking.
"""

from __future__ import annotations

import threading

CASADI_LOCK = threading.RLock()
