# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

State of 2026-10-05. Branch `dev-two-tier-v4-post` at `2c60de9c`: C0–C7 and C9 built and reviewed; D112–D117 followed.

## 1 Questions for the user

1. **Window B's ranges** (C9; log §15, P28): a turn about the airport reference within ±15°, a height within ±300 m, a
   speed scale within 1 ± 0.1, each uniform. With them, D113 leaves out 107 of 40,530 train windows B (0.26 %).
2. **The readings of C6, C9 and D114** (log §15: P26, P27, P29, P30). Each is built as written. None changes a result
   the user has seen.

## 2 For the design of C10 (log §15, P31)

- **The rounds feed the loss branch point by branch point.** A round's groups are large: the reviewer's estimate is
  several MB a group, tens of GB for a round held at once.
- **Window B's D113 check happens at the draw.** It needs the moved start: `post_window_loop.moved_commanded`, whose
  rows equal what `start_moved` gives back.
- **A batch commands each flight once.** So a real window and its A, B or D go to different batches; `start_loop` maps
  each window's move.

## 3 The plan

C8 and C10 wait for B5's base and Claude's check of stage B. C11 and C12 come after C10.
