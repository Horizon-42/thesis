# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

**State: 2026-10-07, 07:10.**
- C17's code is `6afa3c0b` (reviewed).
- The run waits for stage D's `multi_profile` to end (log §29).

**Resolved:** P49 (D165, the user), P50 (not wanted).

| # | Request | For | Log |
|---|---|---|---|
| 1 | **D165's form, as the user chose it (2026-10-07), replaces "its own runner with its own campaign schema name".** It is the same runner, `post_train --method landed`, with a required `Settings.method` (`branch` / `landed`; a resume compares it) and the same campaign schema (`ts-post-train-v1`). The checkpoint identity is unchanged: it does not name the method; the campaign's settings hold it. C10's `campaign.json` gains `"method": "branch"` in the same step as the merge. `post_validation`, the Training export, `post_ceiling` and `model_speed` read a landed campaign as they read C10. D165 and §8 C17 need the designer's text | The designer | §29 |
| 2 | **`Settings.select_seed` (the user, 2026-10-07).** The selection readout's windows and numbers come from a seed of their own, so a campaign started from a round (D162: another seed) reads its source's select windows; so does its val read (`selection_windows` with "val"). C10's record gains `"select_seed": 1337` (its seed) with `method`, and C10 is unchanged. D161, D162 and D165 need a line | The designer | §29 |
| 3 | **Two readings of mine (told to the user):** 16 kept sentences an update (about 250 updates a round, against C10's 244–391); a landing the reward scores 0 (D105) is not kept, since only rewarded sentences are learned | The designer | §29 |
| 4 | **Stage D and C10's record.** After the edit, any code without C17 that reads C10's settings (`settings_of`) stops by name (`unexpected keyword 'method'`). That includes stage D's runners on `dev-multi-control` (start, worker rounds, `multi_profile`). Stage D has to merge `dev-two-tier` before its next run that starts from C10. The edit waits for the running `multi_profile` | The designer, stage D | §29 |
