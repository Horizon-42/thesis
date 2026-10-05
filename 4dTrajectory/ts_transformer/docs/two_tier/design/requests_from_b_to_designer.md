# Requests from stage B to the designer

What Claude, as stage B's implementer, asks of the designer: the readings that wait for the user's decision, what stage
A must change for stage B, and the design text that stage B's work now needs. It holds only the current state: it is
rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits are in
the implementation log (`readouts/2026-10-05_stage_b_implementation_log.md` §1).

State of 2026-10-05. Branch `dev-two-tier-v4-prior` at `0a8ad371` (`dev-two-tier` at `931f4e5d` merged): B0–B4, B6,
B8, B9, B10 and B11 done and reviewed; stage A's line (A37–A41) followed; full suite passed. Stopped before B5.

## 1 Readings for the user to decide

None. The eight readings of the last round are decided (D118).

## 2 For stage A

None open. The two changes ordered in `notes/stage_a.md` (the comment on stage B's hook in `backend.py`;
`runway_ends_from` in the synthetic artefact of `tests/support.py`) are not yet on `dev-two-tier-v4`. When they are,
stage B merges them and deletes its stand-in (`tests/test_prior_training_export.py` `with_runway_ends`).

## 3 For the design text

None. §7's column "Code" was checked against the code at `0a8ad371`: every name it gives exists, with the signatures it
gives (`kept(rule, outcome, faulty)`, `SpeakingLoop.copy(flights)`, `said`, `states`, `generated(flights)`,
`LandingIndex` with its day split, the speaker's state read only). §8 item 1 holds the signals files.

## 4 The plan

- **B5** waits for the user's order.
- **B6's publication** of the folds and the base comes after B5.
- **B7** adds `docs/reference/runners.md` entries for the new runner `prior_behaviour` and for the changes to
  `prior_campaign`, `prior_free_generation`, `prior_validation` and `prior_training_export`.
