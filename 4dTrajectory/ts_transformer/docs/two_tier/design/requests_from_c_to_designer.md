# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

**State: 2026-10-08, evening (local).**
- C25 (D176) is built and reviewed: `c7e41626` on `dev-two-tier-v4-post`; it waits for the user's merge, then
  fronter's F5 exports the two sets by the list `outputs/POOLED/post/windows_lost_separation_select_20261008`.

**Resolved:** item 1 of the last version (the loss windows as a Training set: D176, C25, frontend F5).

No open request.
