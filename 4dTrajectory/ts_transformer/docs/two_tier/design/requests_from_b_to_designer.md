# Requests from stage B to the designer

What Claude, as stage B's implementer, asks of the designer: the readings that wait for the user's decision, what stage
A must change for stage B, and the design text that stage B's work now needs. It holds only the current state: it is
rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits are in
the implementation log (`readouts/2026-10-05_stage_b_implementation_log.md` §1).

State of 2026-10-06. B5 runs from the main tree (`dev-two-tier` `deaf9690`, campaign `prior_base_20261005`). Branch
`dev-two-tier-v4-prior` at `0e33385c`: A43 merged and followed (`d559e970`, `1777b3ec`), B12 done and reviewed twice
(`39d02ef8`); not merged into `dev-two-tier` before B5 ends.

## 1 Readings for the user to decide

1. **B5's val claims after B12 is merged.** B5 runs the code before B12: its two claims of the base's val read
   (`base/run/val_read_prior_validation.json`, `val_read_prior_free_generation.json`) will hold no `spent_utc`. After
   the merge they stay closed while `base/validation/readout.json` and `base/free_generation/readout.json` are there;
   moving or deleting either output would open that val read again. Proposal: once B5's base readouts are written,
   stamp the two claim records with `spent_utc` (a write into B5's run directory: the user's go), or have
   `claim_validation_read` add it when it finds the readout written.
2. **A B5 val readout that stops before it writes** (D119: Claude reports, the user decides on the claim file) — if it
   happens, settle it before B12 is merged: with B12 the code would let it run again to the same output.
3. Claude's readings in B12 (proposals): the spent mark `spent_utc` in the claim record; the executor's step run
   inside the speaker's `accept` (a refused row leaves speaker and loop as they were).

## 2 For stage A

- `aeroviz_backend/autopilot_segment/fly.py` `refuse_past_bound`'s docstring says stage B's hook uses it; the hook does
  not (§3 item 1). Correct it once §3 item 1 is decided.

## 3 For the design text

1. **A43 for the hook: the bound against the prior's sentence.** The order: `apart_from_exported` calls
   `fly.refuse_past_bound(…, "the readout's flown states")`. It compares the live flight with the export's track,
   written to 0.1 m (`prior_training_export.py`), not with the readout's states: the 1e-6 m bound refused every answer
   (0.045 m on the synthetic set). Not applied. Options: (a) the export writes the prior sentence's track unrounded (a
   new sample format, `aeroviz-training-prior-sample-v3`; "a value a later step computes from is not display-rounded"),
   then the hook calls `refuse_past_bound`; (b) `refuse_past_bound` takes the bound (stage A's code) and the hook
   gives `STATE_BOUND_M` plus the rounding; (c) the hook reads the readout's unrounded states. Claude recommends (a):
   no set of stage B is published yet.
2. **The outcome half of A43's check for the prior's sentence** (`apart_from_stored` refuses an outcome other than the
   stored one; `apart_from_exported` has no such check). Exact (no rounding): the live outcome and end cycle against
   the sentence's `outcome` and `endCycle`. Not ordered; can be added with item 1.
3. **§7 items 3 and 7, the names and signatures B12 changed:** `Speaker.speak(row, at, numbers, caller, extra, accept)`
   (refuses a row marked first otherwise than "nothing in force yet"); `Speaker.observe(rows, positions, extra)`
   (refuses positions not one for each row and aircraft); `SpeakingLoop(model, loop, order, sentences, observed,
   flights, geometries, landings, finals, words, *, interval_s, variant, device, temperature)` (``observed``: the
   start's, `start.start_moved`); `prior_free_generation.speak_and_fly(…, sentences, observed, flights, …)`;
   `prior.loop.LoopRows(…, start: int, …)`; `prior.checkpoint.spend_validation_claim`, `written_claim`,
   `CLAIM_SPENT_BY`. Stage C's `post_window_loop.py` calls `SpeakingLoop` with the old arguments: it follows after the
   merge.

## 4 The plan

- **B5** runs (`prior_base_20261005`); B12 is merged into `dev-two-tier` after it ends, then the full suite.
- **B6's publication** of the folds and the base after B5 (with §3 item 1 decided).
- **B7** adds `docs/reference/runners.md` entries for the new runner `prior_behaviour` and the changes to
  `prior_campaign`, `prior_free_generation`, `prior_validation`, `prior_select` and `prior_training_export`.
