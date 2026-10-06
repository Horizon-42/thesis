# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

State of 2026-10-06 (stage C merged onto `dev-two-tier` at `7288de0e`).

| # | Request | For | Log |
|---|---|---|---|
| 1 | O13: Claude's proposal of the campaign's settings from C8's profiles (update groups 4, data sentences 64, batch windows 64, K 8, 1,000 windows of each kind, 10 rounds of about 2.5 h, rates 1e-5 / 1e-4, weight decay 0.01, 200 select windows an airport, traffic 64 / 4) | The user | §25 |
| 2 | P45, a reading for the user: a window set's flown tracks are written unrounded (prior D127 followed for windows), so a set's file is larger (1.3× on the smoke set); a formal set's size is read at its export | The user | §21 |
| 3 | `requests_from_b_to_designer.md` §3 item 5 ("what stage C needs after B12 and B13") is followed on stage C's branch and can leave: `SpeakingLoop` with the start's observed rows (`post_window_loop.py:118`), the val read's lock, claim with its options and spent mark (`post_validation`, D132), the unrounded track of D127 (sample v2) | The designer | §18–§21, §23 |

The formal C10 waits for O13 (item 1) and the user's criteria (D7); then its intent in `intents.json` and the launch
from a clean main tree.
