# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

State of 2026-10-06, 18:40 (C10 `post_train_20261006` resumed at round 4 with five workers; stage C on `dev-two-tier`
at `af36ebe8`). Items of the earlier note resolved: O13 (D137), the speed proposals (D138), the workers' memory (O15).

| # | Request | For | Log |
|---|---|---|---|
| 1 | **An update of the pass in pieces** (`832555a5`, the user's decision of 2026-10-06 after C10's round 4 ran out of the GPU's memory twice, an exception to "C runs C10 without changing code"): `post.loss.update_step` computes an update's surrogate and pull piece by piece (a branch group each), each divided by the counted rows of the whole update, its backward taken at once; the data term in one piece (its dropout); one optimizer step. The loss is unchanged (§2 item 5, D115: the gradient equal to the whole update's to float rounding); its memory is a piece's (round 4's largest update: 6.3–6.5 GB whole, 1.58 GB in pieces). `update_loss` stays as the reference. Rounds 0–3 ran the whole update. The design text of §2 item 5 (and C13, built on this code by stage B's implementer) is the designer's | The designer | §26 |
| 2 | P45, a reading for the user: a window set's flown tracks are written unrounded (prior D127 followed for windows), so a set's file is larger (1.3× on the smoke set); a formal set's size is read at its export | The user | §21 |
| 3 | `requests_from_b_to_designer.md` §3 item 5 ("what stage C needs after B12 and B13") is followed on stage C's branch and can leave: `SpeakingLoop` with the start's observed rows (`post_window_loop.py:118`), the val read's lock, claim with its options and spent mark (`post_validation`, D132), the unrounded track of D127 (sample v2) | The designer | §18–§21, §23 |
