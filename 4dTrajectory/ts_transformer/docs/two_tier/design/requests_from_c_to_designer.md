# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

**State: 2026-10-07, 12:00.**
- C17 runs: `post_landed_20261007`, rounds 0–1 done.
- It was killed in round 2's pass when the previous session's terminal closed, and resumed at 11:52 in a systemd unit
  of its own (log §29).

**Resolved:** P49 (D165, the user), P50 (not wanted).

| # | Request | For | Log |
|---|---|---|---|
| 1 | **D165's form, as the user chose it (2026-10-07), replaces "its own runner with its own campaign schema name".** It is the same runner, `post_train --method landed`, with a required `Settings.method` (`branch` / `landed`; a resume compares it) and the same campaign schema (`ts-post-train-v1`). The checkpoint identity is unchanged: it does not name the method; the campaign's settings hold it. C10's `campaign.json` gains `"method": "branch"` in the same step as the merge. `post_validation`, the Training export, `post_ceiling` and `model_speed` read a landed campaign as they read C10. D165 and §8 C17 need the designer's text | The designer | §29 |
| 2 | **`Settings.select_seed` (the user, 2026-10-07).** The selection readout's windows and numbers come from a seed of their own, so a campaign started from a round (D162: another seed) reads its source's select windows; so does its val read (`selection_windows` with "val"). C10's record gains `"select_seed": 1337` (its seed) with `method`, and C10 is unchanged. D161, D162 and D165 need a line | The designer | §29 |
| 3 | **Two readings of mine (told to the user):** 16 kept sentences an update (about 250 updates a round, against C10's 244–391); a landing the reward scores 0 (D105) is not kept, since only rewarded sentences are learned | The designer | §29 |
| 4 | **Stage D and C10's record.** After the edit, any code without C17 that reads C10's settings (`settings_of`) stops by name (`unexpected keyword 'method'`). That includes stage D's runners on `dev-multi-control` (start, worker rounds, `multi_profile`). Stage D has to merge `dev-two-tier` before its next run that starts from C10. The edit waits for the running `multi_profile` | The designer, stage D | §29 |
| 5 | **P51, a resume reuses the first launch's memory measure (the user, 2026-10-07: written now, built after C17).** Today `post_train` runs O15 at every launch, before its first round: one worker speaks a batch at the formal size (for C17, 64 windows × 8 draws, about 3 min). A resume repeats it on the same settings and host. Proposal: the first launch writes its measure into `campaign.json` (one worker's host and GPU peaks, what a worker holds in a round, the pass's peak). A resume reads them, checks only the host's and the GPU's free memory against N workers (seconds), and refuses by name where they do not fit. A new measure runs only when the record holds none, or when `--speak-workers` asks for more workers than the measure allows; the record names the measure it read. As stage D's "measure once, reuse the profile" (the user, 2026-10-07) | The designer | §29 |
