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
- **`POST /autopilot/segment`** (the `autopilot_segment/` package, the Training view's live executor): flies one word's segment
  of a Training flight (to where the word's envelope ends: a heading word's a lead past the next heading word — the answer's
  `segment.nextWordHeardS`, from the judge's `words_said`, says where the executor heard it: the views draw the rest as a tail) with
  the SINGLE-FLIGHT executor (`single.py`, 2026-09-27: the executor's cycle for one flight in plain floats, ~9 ms for 200 cycles
  where the torch `Executor` on a batch of one took ~0.55 s; the same result, not bitwise — torch's own atan2 / hypot differ
  between a batch and one flight — checked by `test_single_executor.py` and the fleet check `check_single`; PINNED to the code it
  mirrors, `MIRRORED_SOURCE_SHA256`: a code change in `autopilot/` or the dynamics it reaches makes the backend refuse to fly
  until it is ported and the pin updated), driven as `executor.fly` drives it and stopped at the segment's stop (`fly_until`);
  never a replay record or overlay; the answer carries per-part wall-clock `timing`. The spec is the ONE under
  `4dTrajectory/outputs/POOLED/executor/` that `replay.open_executor` accepts for the set's artefact, or it is refused
  naming each (looked up again when a spec is added, moved or rewritten). A bad request is 400 (`errors.RequestRefused`), a set
  or flight not listed 404 (`errors.NotListed`), superseded by a later request from the same page 409 (`errors.Superseded`: every
  request names its page, `clientId`, and its number there, `seq`; the page's lower-numbered ones still waiting are not
  flown, one flying stops before its next cycle, one arriving late is refused — the page's numbers decide, not arrival), a flight the data cannot fly (no aircraft dynamics) 422 (`errors.NotFlyable`; `errors` is stdlib-only so the server
  maps them without torch), anything else 500 with its reason. WARMED UP AT START (2026-09-28; `http_server.warm_autopilot`
  runs `AutopilotSegmentBackend.warm_up` in a thread): the backend is built (torch + ts_transformer, ~470 MB) and every
  read-back set of the current reading rule it can fly is opened — flights, spec, the procedure's masks — ~10 s for the five
  airports (the first 4 s, then ~1 s each: the val split's files are read once for all of them and let go after), so a first
  click answers in milliseconds (it took 2–4 s a set, the first model word 2 s more); a set the code cannot fly (another spec)
  is logged as skipped; a request during the warm-up waits at most for the set being opened (the same lock). One flight at a
  time; a Training set's flights are rebuilt TOGETHER (`open_flights`: one `rebuild_series`, procedure files and arrival
  manifests read once; each airport's published vertical paths kept) and kept for the process; each flight answers for itself (one a check refuses is refused alone,
  `rebuilt_each`). A request with a `sentence` (a model's sample) flies
  that sentence as `prior_free_generation` flew it — from the observed state at its first step, the time clock, the
  generation's time limit (`fly.model_time_limit_s` MIRRORS `limits_s`), the sentence's last runway — and so re-flies the exported
  sample to round-off (the single-flight executor against the torch one that flew it). Full text:
  `aeroviz-4d/docs/35-viewer-reference.md` AV26.

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
