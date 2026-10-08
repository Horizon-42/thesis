# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

**State: 2026-10-08, evening (local).**
- C24 (D175, the diagnostic readout) is built and reviewed: `5880f26e` on `dev-two-tier-v4-post`; `dev-two-tier` can
  fast-forward to it. The reads wait for the user's merge (notes/stage_c.md, part two).
- Claude's readings of D175 where its text left a choice (runway in force, `no_landing`, field 4's rows, field 5 in both
  reads, the bins' edges) are in the log §30, the C24 entry, for the designer to confirm or correct.

**Resolved:** item 1 of the last version (the diagnostic runner: D175, C24).

No open request.
