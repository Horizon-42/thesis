# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

**State: 2026-10-07, 06:00.**
- C15 is done: `outputs/POOLED/post/ceiling_20261007` (log §28).
- C16 is built and merged into `dev-two-tier` (`67cd9bc8`).

**The evidence behind both proposals** (C15, the 1,000 select windows, 32 draws each):
- With one draw, the models land 79.5 % (the start), 85.1 % (round 6) and 85.7 % (round 8).
- With 32 draws, at least one landing comes in 99.3 / 99.3 / 99.4 % of the windows.
- Only 4 windows are never landed by any of the three models.
- The three curves meet at large n. The 14 rounds raised the one-draw share and did not widen the set of windows a
  model can land. The selection readout has been flat since round 6.
- My reading: the policy can say a landing sentence in nearly every window but gives it too little probability, and
  the clipped surrogate of D94 stopped raising it. The proposal below trains on the landings that sampling finds.

**Resolved:** P50 (choosing the best of k sentences at speaking time, judged by the executor) — not wanted (the user,
2026-10-07: "我们只要训练出的能力"): only what the model is trained to say counts, no choice at speaking time.

| # | Request | For | Log |
|---|---|---|---|
| 1 | **P49, training on the landed sentences** (closed-loop supervised training, "expert iteration"; the user asked for the proposal on 2026-10-07). See *P49 in full* below the table. | The user, the designer | §28 |

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
