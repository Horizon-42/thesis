# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

**State: 2026-10-07, 16:10 (local).**
- C19 (`f3ce543f`) and C18 (`364c2f4c`) are built and reviewed on `dev-two-tier-v4-post`, which fast-forwards
  `dev-two-tier`; the user merges (log §30).
- The campaign with K = 16 (`post_branch16_20261007`) runs from its run worktree; C20 waits for it.

**Resolved:** items 1–7 of the last version (D161, D162, D165, D167, D168, D169, §9 items 10 and 11).

| # | Request | For | Log |
|---|---|---|---|
| 1 | **A reading of D167 (mine):** "more workers than the measure allows" read as more workers than the measure was checked for and admitted (its `speak_workers`); a launch with as many or fewer reads it. The measure is taken per pair of devices (the pass's, the workers'): a launch on other devices measures. D167 could say so in a line | The designer | §30 |
| 2 | **`open_campaign`'s `settings_type` (C19).** A resume compares the recorded settings as the stage's settings class reads them, so a setting added with a default (D168, D169) matches an old record. Stage D's runner passes `MultiSettings` (changed in `multi_train.py` and its two tests in this commit). Post-training §9 item 11 could name it among the stage's parts of a campaign (beside `schema` and `reader`) | The designer, stage D | §30 |
| 3 | **Stage D's merge of C18.** `dev-multi-control` changes `workers_fit` the same way (`held_now`), and its pass line drops the guard for workers on the CPU (`--speak-device`, `ac455d2e`); the merge has to keep the guard. Its `profiled_fit` already subtracts `passed["now"]` from the GPU, which C18's `workers_fit` now does for every caller with nothing held | Stage D | §30 |
