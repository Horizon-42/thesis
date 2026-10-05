# Requests from stage B to the designer

What Claude, as stage B's implementer, asks of the designer: the readings that wait for the user's decision, what stage
A must change for stage B, and the design text that stage B's work now needs. It holds only the current state: it is
rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits are in
the implementation log (`readouts/2026-10-05_stage_b_implementation_log.md` §1).

State of 2026-10-05. Branch `dev-two-tier-v4-prior` at `5818fbbf` (`dev-two-tier-v4` at `06b8fde1` with A42 and
`dev-two-tier` at `65c77ecd` merged): B0–B4, B6, B8, B9, B10 and B11 done and reviewed; stage A's line (A37–A42)
followed; the stand-in for A42 deleted. Stopped before B5.

## 1 Readings for the user to decide

None.

## 2 For stage A

None.

## 3 For the design text

None. §7's column "Code" matches the code.

## 4 The plan

- **B5** waits for the user's order.
- **B6's publication** of the folds and the base comes after B5.
- **B7** adds `docs/reference/runners.md` entries for the new runner `prior_behaviour` and for the changes to
  `prior_campaign`, `prior_free_generation`, `prior_validation` and `prior_training_export`.
