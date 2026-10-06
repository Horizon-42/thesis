# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

State of 2026-10-06 (branch `dev-two-tier-v4-post` at `483b81d7`).

| # | Request | For | Log |
|---|---|---|---|
| 1 | P45, a reading for the user: a window set's flown tracks are written unrounded (prior D127 followed for windows), so a set's file is larger (1.3× on the smoke set); a formal set's size is read at its export | The user | §21 |
| 2 | `requests_from_b_to_designer.md` §3 item 5 ("what stage C needs after B12 and B13") is followed on stage C's branch and can leave: `SpeakingLoop` with the start's observed rows (`post_window_loop.py:118`), the val read's lock, claim with its options and spent mark (`post_validation`, D132), the unrounded track of D127 (sample v2) | The designer | §18–§21, §23 |

The inputs the formal C10 waits for are in the design: O13 (Claude proposes from C8's report) and the user's criteria
(D7). C8 runs now (§0.3).
