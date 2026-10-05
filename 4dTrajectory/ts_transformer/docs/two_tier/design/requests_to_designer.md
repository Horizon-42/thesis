# Requests to the designer

What Claude, as stage A's implementer, asks of the designer: what another stage must do or know because of stage A's
work, and anything else the designer must handle. It holds only the current state: it is rewritten in full each time,
never appended to, and an item leaves when it is done. The decisions themselves are in the design documents (cited by
D number); this note says what follows from them and where stage A's code is.

State of 2026-10-05.

## 1 Stage B to do: B11, the selection leaves out flights with a faulty observed track (D111)

The user (2026-10-05): a flight whose stored observed track has a faulty point is marked when the artefact is read,
and the base does not learn from it; nothing is rebuilt. Prior D111 and milestone B11 hold stage B's part.

- **What stage A gives** (vocabulary A40, `instructions/faults.py`, torch-free):
  - `track_faults(signals) -> tuple[Fault, ...]`: the faults of one flight's stored signals, in row order.
    `Fault(kind, row)`, where `kind` is `JUMP`, `HELD` or `REVERSAL`.
  - `faulty_flights(artefact_dir, split) -> dict[int, tuple[Fault, ...]]`: the marked flights of a split. The key is
    the flight's place in `load_signals(artefact_dir, split)`, the same key a closed-loop sentence has
    (`closed_loop_sentences`, `signal_index`).
  - The rule's constants, named once: `JUMP_RATIO = 3`, `AROUND_STEPS = 5`, `REVERSAL_DEG = 120`.
- **What stage B changes** (`prior/selection.py` and its callers):
  - `landed` also leaves out a marked flight, whatever its outcome; `all` keeps every sentence.
  - The selection record counts the sentences left out for a faulty track apart from those left out by their outcome,
    for each split, airport and stratum. That changes a run's data identity, so the checkpoint's schema changes.
  - Free generation and the post-training's starts do not change (the user: the post-training may start from flights
    that do not land).
- **Size on A34's artefact** (Δ = 4 s, closed-loop sentences): train 498 of 40,530 marked, 469 of them landed;
  select 177 of 6,199, 157 landed. Most are held positions. With `landed`, train keeps 39,154 instead of 39,623.
- **When:** after A40 is on the stage A line that stage B merges (it is in review now, branch `dev-two-tier-v4-a40`),
  and before B5's formal campaign.

## 2 Stage B to follow when stage A's branches are merged (the user merges)

None of these is on `dev-two-tier-v4` yet (it is at `58fd8a2b`). They wait for the user's merge:

| Branch | Commits | Contents |
|---|---|---|
| `dev-two-tier-v4-a38` | `e7e01461`, `b41b24a5`, `89729d66` (A37), `b6ee4958`, `98c5e010` (A38) | A37 and A38 |
| `dev-two-tier-v4-a39` | A37's three, the merge `0126f7ff` of dev-two-tier `9e44a687`, `658e4717` (A39) | A37 and A39 |
| `dev-two-tier-v4-a40` | A38's branch + A40 (in review) | A40 |

What stage B's code must follow after the merge:

- **A37 (D90 narrowed, D85):**
  - `Loop.timed_out()` is gone. Read why a flight ended from the judge: `loop.outcome(f).outcome == judge.TIMEOUT`.
  - `Loop.executor` stays public. A caller reads `done_cycle`, `flown()` and `inputs.aero_params` from it.
  - `training_files.runway_hae_minus_msl_m` is gone. Use `training_export.candidate_hae_minus_msl_m(runway_ends_from,
    geometry)`, which takes the published runway data.
  - `training_files.SPLITS` is `artefact.READ_SPLITS` (train, select).
  - Stage B's log (`readouts/2026-10-05_stage_b_implementation_log.md`) already lists these callers.
- **A38 (D97): the interface for the post-training** (`autopilot/start.py`, `autopilot/executor.py`,
  `autopilot/sentence.py`):
  - `Loop.copy(flights)` gives copies of chosen flights, repeats permitted, by index; a bool mask is refused.
  - `Executor.take` and `Spoken.take` are what it uses.
  - `Loop.halt(mask)`: a halted flight is held where it is, hears no words and never becomes done.
  - `start_moved(…, moves)` returns `(loop, order, moved observed rows)`. Each flight gets a `Move(turn_deg, height_m,
    speed_scale)`, or `NO_MOVE` for none.
    - The flight is checked against its stored signals before it is moved.
    - The time limit stays the flight's own.
    - The speed change stretches positions and heights about the first predicted step.
  - `start` keeps its signature and return.
  - The bound the user set: a flight alone and in a batch, or a copy and its original, give the same words, done and
    outcome; their states agree within `STATE_BOUND_M`.
  - Claude's reading, for stage C: a turn is made in the airport frame, so the start rule's ground scale (D87) moves
    the start speed slightly (about ×0.9998 for 10°, about 0.4 % at 90°).
- **A39 (outline D109): the Training view's splits come from the caller.** This is for B10's set from the claimed
  validation readout.
  - Backend: `AutopilotSegmentBackend(splits=…, airports_root=…)`; `stage_a_service()` is stage A's train and select.
    For the validation readout's set, build the service with `("train", "select", "val")`.
  - Frontend: `parseTrainingSample(raw, splits)` and `fetchTrainingSample(…, splits)`. `TRAINING_SET_SPLITS` is every
    split a set may hold, and `TRAINING_SPLITS` is stage A's.

## 3 Stage B to know

- **A34's artefact `instruction_language/v12_20261005` / `executor/v17_20261005` stays.**
  - A37's code was checked on it by sample.
  - Its readout was written again with val as counts only.
  - Its three conformance checks pass with A38's code.
- **Closed-loop landed share** (A34, Δ = 4 s): train 97.76 %, select 97.08 %. With D111 the selection shrinks by the
  counts in §1.
- **Free generation starting from a flight with a faulty start** (start ground speed 7–58 m/s) fails whatever is said.
  The user leaves this as it is; such a failure is not the model's.
