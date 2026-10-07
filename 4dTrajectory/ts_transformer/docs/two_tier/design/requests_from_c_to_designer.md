# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

**State: 2026-10-07, 18:30 (local).**
- C19, C18 (`704879d7`) and C20 (`2e2e860f`) are merged into `dev-two-tier` (the user's word; log §30).
- The larger-learning-rate campaign ended with no gain; the campaign with K = 16 and the larger rates
  (`post_k16lr3_20261007`, the user's, 4 rounds) runs. P54 and P55 written on the user's word (2026-10-07).

**Resolved:** items 1–7 of the last version (D161, D162, D165, D167, D168, D169, §9 items 10 and 11).

| # | Request | For | Log |
|---|---|---|---|
| 1 | **A reading of D167 (mine):** "more workers than the measure allows" read as more workers than the measure was checked for and admitted (its `speak_workers`); a launch with as many or fewer reads it. The measure is taken per pair of devices (the pass's, the workers'): a launch on other devices measures. D167 could say so in a line | The designer | §30 |
| 2 | **`open_campaign`'s `settings_type` (C19).** A resume compares the recorded settings as the stage's settings class reads them, so a setting added with a default (D168, D169) matches an old record. Stage D's runner passes `MultiSettings` (changed in `multi_train.py` and its two tests in this commit). Post-training §9 item 11 could name it among the stage's parts of a campaign (beside `schema` and `reader`) | The designer, stage D | §30 |
| 3 | **Stage D's merge of C18.** `dev-multi-control` changes `workers_fit` the same way (`held_now`), and its pass line drops the guard for workers on the CPU (`--speak-device`, `ac455d2e`); the merge has to keep the guard. Its `profiled_fit` already subtracts `passed["now"]` from the GPU, which C18's `workers_fit` now does for every caller with nothing held | Stage D | §30 |
| 4 | **P55, finer credit without a value function (the user asked for it, 2026-10-07; the experiment log, candidate 5).** Two campaign settings, each defaulting to the code's behaviour (no record edited): (a) `Settings.segment_only` (default False): a branch group's advantage and pull count only the rows from its branch point to the next one (the last segment to the event), the later words left to the later groups — `post.branches.samples`'s `until` the lesser of the event and the next branch point; (b) `Settings.branch_every_s` (default 120 s, D37's constant made a setting; a whole number of Δ rows): denser branch points, the continuations growing in proportion (60 s twice, 30 s four times). Proposed first campaign: (a) + (b) at 60 s, K = 8, C10's rates, from C10's round 8, 4 rounds | The user, the designer | experiment log |
| 5 | **P54, a value function (the experiment log, candidate 4).** `--method ppo`: a value V of each row's state, each row's advantage by GAE (γ = 1, λ = 0.95, the sentence's reward at its end), the clipped surrogate per word with its row's advantage, a value loss (weight 0.5), E passes (C20) and the gradient clip (C19); the checkpoint holds V. Three decisions for the user: (1) V asymmetric — a separate network used only in training that reads the window's recorded tracks, the future included (my proposal), or a head on the prior's last layer; (2) first sentences only, V the baseline (my proposal: a third of the speaking, more windows a round), or the branch groups kept with V in place of the group mean; (3) V warmed up alone on the start model's sentences for one or two rounds before the policy moves. A design change (about 500–700 lines with tests): the user decides, the designer writes it into post_training, then I build it | The user, the designer | experiment log |
