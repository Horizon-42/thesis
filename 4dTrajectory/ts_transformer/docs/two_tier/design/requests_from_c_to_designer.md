# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

**State: 2026-10-07, 05:50.**
- C15 is done: `outputs/POOLED/post/ceiling_20261007` (log §28).
- C16 is built and merged into `dev-two-tier` (`67cd9bc8`).

**The evidence behind both proposals** (C15, the 1,000 select windows, 32 draws each):
- With one draw, the models land 79.5 % (the start), 85.1 % (round 6) and 85.7 % (round 8).
- With 32 draws, at least one landing comes in 99.3 / 99.3 / 99.4 % of the windows.
- Only 4 windows are never landed by any of the three models.
- The three curves meet at large n. The 14 rounds raised the one-draw share and did not widen the set of windows a
  model can land. The selection readout has been flat since round 6.
- My reading: the policy can say a landing sentence in nearly every window but gives it too little probability, and
  the clipped surrogate of D94 stopped raising it. The two proposals below use the landings that sampling finds.

| # | Request | For | Log |
|---|---|---|---|
| 1 | **P49, training on the landed sentences** (closed-loop supervised training, "expert iteration"; the user asked for the proposal on 2026-10-07). See *P49 in full* below the table. | The user, the designer | §28 |
| 2 | **P50, choosing at speaking time** (best of k, flown by the executor; the user asked for the proposal on 2026-10-07). See *P50 in full* below the table. | The user, the designer | §28 |

## P49 in full: training on the landed sentences

**A campaign of rounds.** It is a new campaign, with D162's start (round 6 or round 8 of `post_train_20261006`, or the
base) and a seed other than C10's. Each round r:
1. Draw the windows of the train days as C10 does (`draw_round`: real, A, D and B, from the seed and r).
2. Speak each window N times from the round's model. This is the first pass with no branch, the draws' numbers as in
   C15, read by the workers.
3. Keep, for each window, the landed sentence with the highest reward (ties: the lowest draw). A window with no landing
   in N gives nothing.
4. One pass over the kept sentences, with the loss below.
5. The selection readout (the same 1,000 select windows and numbers as C10) and the checkpoint.

**The loss.** The commanded aircraft's words of each kept sentence, teacher-forced under its traffic as the surrogate
reads them (`post/loss.py` `Samples`), negative log-likelihood. The data term (D36) and the pull toward the base (D29)
are kept as in C10. No clipped surrogate and no advantage.

**What C10's code already gives.** The windows, the workers, the draws (C15), `Samples`, the data term, the pull,
`one_pass`, the selection readout, resume, the start (C16) and the validation readout (D132: one read for this
campaign).

**New code.** About 120 lines: the kept-sentence step and the loss. About 100 lines of tests: a kept sentence is the
best landed one; the loss of a sentence is its words' log-likelihood; a resume is the campaign run through.

**Cost.** With N = 8, a round flies 4,000 × 8 = 32,000 windows. On the CPU with 16 workers (C15: about 50 s for 1,000
windows), that is about 27 min of speaking, plus the pass and the readout: about 35 min a round.

**Decisions for the user:**
- the start: round 6, round 8 or the base;
- N: 8 is proposed; C15 shows 97.5–98.2 % of select windows with a landing within 8;
- what is kept: one best landed sentence per window (proposed), or every landed sentence;
- the learning rates, weight decay and loss weights: C10's are proposed (1e-5 / 1e-4, 0.01, pull 0.04, data term 1);
- the rounds: 6 are proposed, then D7 over them.

**Open for the designer:** whether a window where all N land is kept. Keeping it teaches what the model already says;
leaving it out trains only on the windows it sometimes fails, which shrinks the data. I propose keeping it, as the
plain form.

## P50 in full: choosing at speaking time

**What it is.** A way to speak, beside the plain one. For each window, k sentences are spoken from the start of the
window (the first pass, k draws as in C15) and each is flown by the executor and judged. The one with the highest
reward is chosen (ties: the lowest draw). No training is needed.

**Its select-day reading is in C15's files already** (the best reward of the first k draws, and the share landed):

| Model | Landed, k = 1 / 2 / 4 / 8 | Mean reward, k = 1 / 2 / 4 / 8 |
|---|---|---|
| Start | 79.5 / 88.1 / 94.3 / 97.5 % | 0.787 / 0.868 / 0.929 / 0.965 |
| Round 6 | 85.1 / 92.3 / 95.7 / 97.9 % | 0.842 / 0.913 / 0.947 / 0.970 |
| Round 8 | 85.7 / 92.4 / 96.5 / 98.2 % | 0.849 / 0.912 / 0.953 / 0.974 |

**What it reads (a caveat to decide on).** The choice judges each sentence by flying it among the other aircraft's
recorded tracks: their future, which a speaker in service does not know. So its numbers are those of a search with a
perfect model of the traffic's future. My reading is that it is a search over the model's sentences, not the model's
own skill (as the procedure-computed answers were kept apart, a diagnostic or a baseline). Reported, it stands beside
the plain readout as "round r, best of k", never in place of it.

**Code.** A runner, or a mode of `post_validation` and of the Training export, that takes k and chooses. About 60 lines
and its tests. The select-day numbers above need no new run.

**Decisions for the user:**
- k: 4 or 8 is proposed;
- whether the val read reports it: proposed, one val read (D132) that reports both the plain readout and best of k,
  since a second read of val is not allowed;
- whether the choice works on whole sentences (proposed, the plain form) or segment by segment (several futures at
  each branch point, the best chain kept: closer to the user's picture of 2026-09-20, but a tree search, k times dearer
  at each point; a later step);
- whether the traffic's recorded future may be used in it, or a traffic prediction must take its place before it is
  more than a diagnostic.
