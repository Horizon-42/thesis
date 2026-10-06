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
- **Playback**: every optimizer emits load-factor controls, so `build_optimized_trajectory_playback` flies them in
  the "casadi" simulator mode (the alpha-control optimizers are archived, 2026-10-05).
- Playback drift guard: `playbackDriftM` on every optimize response; stderr WARNING above
  `PLAYBACK_DRIFT_WARN_M = 50`.
- **`POST /autopilot/segment`** (the `autopilot_segment/` package, the Training view's live executor; rewritten for stage A
  of the two-tier vocabulary, A23, 2026-10-04): flies one word of a Training flight's CLOSED-LOOP sentence at the row
  interval Δ the view shows (`{clientId, seq, airport, setId, flightKey, rowIntervalS, column, row}`, ``row`` a Δ row) with
  the SINGLE-FLIGHT executor (`ts_transformer/autopilot/single.py`), set up as the Training export and the formal replay set
  it up (`ts_transformer/experiments/training_flights.py`: from the first predicted step, its time limit and reserve) — to
  the cycle the next word of its column is heard (a heading word a lead later), or, the column's last word, to its outcome,
  judged. The answer (`aeroviz-autopilot-segment-v9`) is the flight every cycle from the word on, the crossing with its
  decision-altitude check when flown to the outcome, and its distance from the artefact's stored flown states (`stored`).
  Sets are read from `<airport>/training/index_v4.json` (never `index.json`); the set's sample names its artefact and
  executor spec, opened with `replay.open_executor` (only for executor code that flies the spec's reference tracks within
  1e-6 m). 400 a bad request, 404 not listed, 409 superseded by a later request of the same page (`clientId` + rising
  `seq`), 500 anything else with its reason. Warmed up at start (`http_server.warm_autopilot`). `--training-airports-root`
  points the server at a test stack's own airports tree. `python -m aeroviz_backend.autopilot_segment.check_live` flies
  every word of a published set against its sample (outline §6 item 6).

## Traffic jobs — the Optimize task's multi-aircraft mode (`traffic_jobs.py`, T11)

`GET /traffic/arrivals?airport=&date=YYYY-MM-DD` lists the arrivals that LAND on that UTC day from the arrivals roster alone
(`flightKey, callsign, runway, type, entryUtc, landingUtc`; `type` by the identity resolver of `icao24`, null when untyped; no
track is opened). `POST /traffic/jobs` with `{mode: "m1", airport, flightKey}` or `{mode: "m2", airport, blockStartUtc, blockS}`
(`blockS` an integer, 900, 1800 or 3600) answers `{jobId}`; **409 while a job runs** (one per backend), **400** for a flight key the
roster lacks, a bad block, an empty block, any other field, or a field of the wrong type. `GET /traffic/jobs/<id>` is `{state:
running | done | failed | cancelled, progress: {done, total, current}, error}` and, once done, `summary` (the job's
`traffic.readout`); `total` is null until the job has chosen its aircraft, `current` is the aircraft last FINISHED. `POST
/traffic/jobs/<id>/cancel` stops the job's process group (SIGTERM, SIGKILL after 5 s) and marks it cancelled; a job with a
`state.json` (`done` included) is left as it is, a `state.json` that appears while it stops is never overwritten.
`GET /traffic/jobs/<id>/files/<name>` serves a file **only if the job's `comparison/comparison_index.json` lists it** (the index,
the CZML of its groups, the evaluation report it names; any other name — or a listed file a prune has removed — is 404), as the
file's own bytes. A job is a subprocess of this interpreter running `4dTrajectory/optimization/traffic_job.py` under `nice -n 10`,
one solver thread (`scenario_batch.SOLVER_THREAD_ENV` = 1), in a session of its own (`start_new_session`), so a cancel reaches its
children too and stopping the backend (`finally` and SIGTERM, `stop_traffic_jobs_then_terminate`) cancels it.
**The job directory records the job's process (`process.json`: the group = the leader's pid, the leader's start time from
`/proc/<pid>/stat` field 22, the boot id): a job that outlives its backend still counts as running — for a start (409), its status,
pruning — whichever backend started it, but a process group number is REUSED once its leader is gone, so a group counts as alive, and
is signalled, ONLY while its leader exists (not a zombie) with the recorded start time and boot id (checked before each signal);
otherwise the job is `failed` ("process is gone") and nothing is signalled. A job this backend started is held by its unreaped
`Popen`: when its main process exits, the rest of its group is killed while the zombie still holds the pid (`waitid` WNOWAIT), the
process is reaped and dropped from `_children` (polled at every status and start), and a job with no `state.json` is written
`failed` with the exit code at once — so its group number is never consulted again. Linux only (`/proc`); a start elsewhere is refused.** Jobs live under `--traffic-jobs-root` (default `~/.cache/aeroviz/traffic_jobs/<port>`,
so two backends never share a directory; never `public/data`), one directory per job (`spec.json`, `process.json`, `job.log`,
`progress.json`, `records/`, `comparison/`, `state.json` last — layout in `traffic_job_files.py`); the newest 5 are kept, pruned
BEFORE the new job starts and never a job whose group is alive. The job's Traffic carries the runway targets of the whole airport,
as the batch's. The harvest root is `observed_trajectories.DEFAULT_HARVEST_ROOT`.

## Observed tracks have TWO windows — the comparison overlay must use the model one

A stored track's `samples[i][0]` is relative to FIRST RECEPTION (`store.track_record`; absolute
time in `start_time_utc`), but every modeling artifact lives on the ARRIVAL slice, rebased by
`load_arrival_flights` (`t0 = waypoints[0][0]`) so `t = 0` is the 25 km terminal-ring entry — and
`t0` is discarded there. Measured over 300 random KRDU arrivals it is a median **45.1 s**
(p95 123.1, max 526.3). Draw a full-track reference beside a group built on the arrival origin
and the group renders that far early: median **5055 m** apart at group start over the 471 KRDU
05L prediction groups (p95 47.1 km). Undetectable downstream — both start at `t = 0`, both name
the right flight, the schema is satisfied — and it reads as model error, not a publication bug.

Hence `aeroviz_backend.observed_trajectories` takes `window` ∈ `full` (default; Evaluation/Ground Truth;
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
