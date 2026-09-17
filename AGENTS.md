# AGENTS.md

Guidance for Codex in this repository.

**The map is not here.** Layout, commands, architecture, data formats, contracts and gotchas live
in the `CLAUDE.md` files — the root one and one per subsystem — and in the reference docs they
index. They are maintained with the code; this file was a fork of an early root `CLAUDE.md` and had
been stale for a month (rewritten 2026-09-18). Read, in this order:

- **Root `CLAUDE.md`** — project overview, repository layout, build & dev commands, data flow, key
  data formats, environment, coding conventions, cross-cutting invariants, open-item hazards.
- **The subsystem's own `CLAUDE.md`** — it loads with the tree you are changing:
  `aeroviz-4d/`, `aeroviz_backend/`, `trajectory_data_process/`, `final_approach/`,
  `flight_scenarios/`, `evaluation/`, `4dTrajectory/`, `4dTrajectory/ts_transformer/`, `geokit/`.
  Each is an INDEX: one line per contract/gotcha/default ending in an ID (`C7`, `TD16`, `EV12`, …).
- **The reference doc behind an ID** — `4dTrajectory/ts_transformer/docs/reference/*.md`,
  `4dTrajectory/docs/optimizer_reference.md`, `evaluation/docs/EVALUATION_REFERENCE.md`,
  `trajectory_data_process/docs/06-harvest-reference.md`, `aeroviz-4d/docs/35-viewer-reference.md`,
  `flight_scenarios/docs/population_reference.md`, `docs/environment.md`.
  Find one with `grep -n '^### C7 ·' <reference dir>/*.md`.
- **Status and history** — `docs/open-items.md` (what is blocked or measured-but-unfixed),
  `docs/code-health-followups.md` (deferred findings), `docs/CHANGELOG.md` (dated log; append new
  entries at the top).

When a change produces a durable fact, add it to the subsystem's reference doc under a new ID and
ONE line to that `CLAUDE.md` index — not to this file.

## Python environment

- Use the conda `aeroviz` environment for every Python script and test; `conda run -n aeroviz …`
  for non-interactive commands. Never the system Python, Homebrew Python, or a bare `python`.
- Prefer `conda activate` / `scripts/activate_aeroviz_env.sh` over
  `envs/aeroviz/bin/python`: the env's `activate.d` hook sets `LD_LIBRARY_PATH`, and bypassing it
  brings back the torch/matplotlib CXXABI clash (`docs/environment.md` E5).

## Data safety

- Downloaded tracks are NOT regenerable. `tracks/` is never edited; derived repairs and slices are
  read-time (root `CLAUDE.md` → Cross-Cutting Invariants).
- Know which rebuild you are running before you run one: `--observed-only` rebuilds `approach/`
  alone, while `--evaluate-only` re-rosters `arrivals/` and DELETES
  `lateral_pass_eligibility.json`, which changes every ts dataset split
  (`trajectory_data_process/CLAUDE.md` → "Which rebuild").
- Never re-run an evaluation/harvest or overwrite published artifacts without the owner asking for
  that specific action.

## Change Scope

- Each turn must only make the changes explicitly requested by the user.
- Do not modify additional files, modules, APIs, tests, or docs merely to make broader test suites pass or to synchronize adjacent code.
- If a requested change exposes unrelated failures or stale interfaces, report them clearly instead of fixing them without explicit permission (`docs/code-health-followups.md` is where such findings go).
- Refactoring may reorganize an implementation, but it must preserve every existing user-facing
  feature and experiment mode unless the user explicitly requests that feature's removal.
- Never infer permission to delete, replace, or retire functionality from a request to add a new
  design, simplify an interface, drop backward compatibility, or clean up legacy code. If the new
  design conflicts with an existing feature, stop and ask the user which behavior to keep.
- Do not introduce a new architecture, mode, or public contract as an inferred substitute for an
  ambiguous request. Clarify the intended design before implementing or running experiments.

## Compatibility and Risk Decisions

- Backward compatibility is opt-in, not the default. For regenerable artifacts and
  derived data, retire obsolete formats and require regeneration instead of adding
  dual-read paths, fallbacks, migration branches, or speculative defensive code.
- Add compatibility or migration protection only when dropping it could damage or
  discard non-regenerable data, such as downloaded/raw source data. Keep that protection
  no broader than necessary.
- If a change presents a meaningful compatibility, data-loss, architecture, or migration
  risk, stop and ask the user to choose. Do not make that product/design decision
  autonomously.

## ML Experiment Isolation

- Treat outer-test as a one-time final release, never as a routine pipeline output or a source
  of debugging, model, loss, epoch, feature, or hyperparameter decisions.
- Use train metrics to diagnose fitting and validation/CV metrics for every development choice.
  Do not run or inspect test predictions until the user explicitly declares the experiment
  frozen and requests the final test release.
- TS development runs use `--split development` (train + validation). A final test must use
  `--split test --release-test`; preserve the checkpoint-adjacent `test_release.json` audit
  ledger and never delete, reset, or bypass it.
- If any development decision occurs after test results were exposed, label that partition a
  development test. A new later-time or otherwise untouched dataset is required for a new blind
  final result; changing the split seed does not restore blindness.

## Domain Context

This is a thesis research project, serving both thesis visualization/validation and a reusable
research component library. Key aviation concepts:

- **TMA** (Terminal Maneuvering Area) — controlled airspace around airports
- **OCS** (Obstacle Clearance Surface) — PANS-OPS geometry ensuring terrain clearance on approach
- **4D Trajectory** — aircraft position (lon, lat, alt) + time; the "4th dimension" is the scheduled arrival time
- **CTA** (Controlled Time of Arrival) — ATC-assigned time slot at a fix point
