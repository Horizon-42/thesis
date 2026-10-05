# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

State of 2026-10-05. Branch `dev-two-tier-v4-post` at `69531aa3`: C0–C7 and C9 built and reviewed, the code of C10 and C8 too; D112–D117 followed.

## 1 Questions for the user

1. **Window B's ranges** (C9; log §15, P28): a turn about the airport reference within ±15°, a height within ±300 m, a
   speed scale within 1 ± 0.1, each uniform. With them, D113 leaves out 107 of 40,530 train windows B (0.26 %).
2. **The readings of C6, C9 and D114** (log §15: P26, P27, P29, P30). Each is built as written. None changes a result
   the user has seen.

## 2 For the design of C10 (log §15, P31; log §16)

- **The windows of a round share their flights** (the user, 2026-10-05, P32): A, D and B are built from the round's real
  windows, as `post_train` does; the design text (§2 item 4, D100) does not say it yet.

- **The rounds feed the loss branch point by branch point.** A round's groups are large: the reviewer's estimate is
  several MB a group, tens of GB for a round held at once.
- **Window B's D113 check happens at the draw.** It needs the moved start: `post_window_loop.moved_commanded`, whose
  rows equal what `start_moved` gives back.
- **A batch commands each flight once.** So a real window and its A, B or D go to different batches; `start_loop` maps
  each window's move.

## 3 The plan

C11's code next (the user's order of 2026-10-05). The formal C8 (`post_profile`) and C10 (`post_train`) wait for B5's base and Claude's check of stage B, C10 also for the user's criteria; C12 after C10.
