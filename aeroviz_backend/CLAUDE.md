# aeroviz_backend — Python HTTP backend

Simulation / optimization / dynamics-comparison endpoints, plus the observed-track service the
frontend reads. Solver internals and defaults live in `4dTrajectory/CLAUDE.md`.

- **The backend does NOT hot-reload** — restart `./start_aeroviz_fullstack.sh` after backend
  changes. The launcher is a supervisor: restarts a dead child individually (crash-loop
  protection: >5 deaths within 8 s ⇒ give up); Ctrl-C/SIGTERM kills both subtrees.
- **casadi is not thread-safe**, so casadi-heavy endpoints run in an isolated worker subprocess
  (`isolated_backend.py`) and in-process entry points serialize on `casadi_lock.CASADI_LOCK`.
  `AEROVIZ_ISOLATE_SOLVER=0` disables isolation (to get a native traceback). Full rationale:
  `4dTrajectory/CLAUDE.md`.
- **Worker sessions**: `AEROVIZ_WORKER_IDLE_TIMEOUT_S` (default 600) idle watchdog reclaims a
  stranded resident solver worker.
- **Probe bug**: `build_optimized_trajectory_playback` needs a REAL optimizer name —
  `simulation_mode_for_optimizer` on an unknown name selects the alpha-control mode and misreads
  casadi load-factor controls (fake 8–11 km "drift").
- Playback drift guard: `playbackDriftM` on every optimize response; stderr WARNING above
  `PLAYBACK_DRIFT_WARN_M = 50`.

## Observed tracks have TWO windows — the comparison overlay must use the model one

A stored track's `samples[i][0]` is relative to FIRST RECEPTION (`store.track_record`; absolute
time in `start_time_utc`), but every modeling artifact lives on the ARRIVAL slice, rebased by
`load_arrival_flights` (`t0 = waypoints[0][0]`) so `t = 0` is the 25 km terminal-ring entry — and
`t0` is discarded there. Measured over 300 random KRDU arrivals it is a median **45.1 s**
(p95 123.1, max 526.3). Draw a full-track reference beside a group built on the arrival origin
and the group renders that far early: median **5055 m** apart at group start over the 471 KRDU
05L prediction groups (p95 47.1 km). Undetectable downstream — both start at `t = 0`, both name
the right flight, the schema is satisfied — and it reads as model error, not a publication bug.

Hence `aeroviz_backend.observed_trajectories` takes `window` ∈ `full` (default; Observe/Baseline;
rostered by `tracks/manifest.json`) | `arrival` (the comparison reference; rostered by
`arrivals/manifest.json`), and the arrival window is built by **`load_arrival_flights` itself** —
the same loader the scenario/optimizer/training paths use, so there is no second slicer to drift
from. The slice is READ-TIME: `tracks/` is never edited and no artifact is written (same rule as
the altitude-outlier repair). `observed-trajectories-v2` echoes `trackWindow` and the frontend
refuses anything but `arrival` for the reference — a v1 backend would ignore the argument and
silently serve full tracks. `anchorTimeS` is still added on top for predictions (that shift is ③
of three origins; this one is ②).

Corollary: the pre-entry segment is *not* missing from the comparison view by accident — it was
never model input, supervision target, or evaluated, and drawing it as white "truth" beside a
forecast invites reading it as something the model failed to produce.
