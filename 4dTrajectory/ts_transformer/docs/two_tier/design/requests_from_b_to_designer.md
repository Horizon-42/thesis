# Requests from stage B to the designer

What Claude, as stage B's implementer, asks of the designer: the readings that wait for the user's decision, what stage
A must change for stage B, and the design text that stage B's work now needs. It holds only the current state: it is
rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits are in
the implementation log (`readouts/2026-10-05_stage_b_implementation_log.md` §1).

State of 2026-10-06. B5 runs from the main tree on B12's code (`prior_base_20261006`, started 2026-10-05 22:40Z).
Branch `dev-two-tier-v4-prior` at `142272f0`: B13 done and reviewed twice; not merged into `dev-two-tier` before B5
ends (its behaviour check gives another answer).

## 1 Readings for the user to decide

1. **The claim's options beyond D128's list** (proposal, built in B13): both readers also record the device, and free
   generation its temperature and bound of go-arounds. Why: the words are drawn by comparing uniform numbers with
   probabilities computed on the device (and at the temperature), so a rerun on another device need not be the same
   read, which is D128's premise. Accept, or drop them.
2. **An error found after the claim locks the val read for good.** Free generation's check "an airport with no val
   sentence" can only run after the claim (it reads val, D85). With D128 a rerun with the same options fails the same
   way, and one with corrected airports is refused for its other options. Unlikely for the base (every airport, the
   default). Options: accept; or let the user clear the claim file by hand when it happens (Claude reports it).

## 2 For stage A

None.

## 3 For the design text

1. **§7, the names and signatures B13 changed:** `prior.checkpoint.lock_val_read(prior_dir, reader)` (new: the read's
   exclusive lock, taken before a reader looks at its output, kept to its end); `claim_validation_read(prior_dir,
   reader, out, options)` (now takes the reader's options, no longer locks or returns anything);
   `settle_written_claim`, `holds_written_claim` (new); `holds_claim` (deleted); `prior_campaign.settle_val_steps`,
   `VAL_READERS` (new); the sample format `aeroviz-training-prior-sample-v3` (D127); the backend's
   `apart_from_exported(result, said, batch, j, step_s)` (``said``: the prior sentence, no longer its track).
2. B13's "Not ordered" item (tying the observed rows to the `Loop` the start returns) is left as the design says.

## 4 The plan

- **B5** runs (`prior_base_20261006`); after it ends, B13 is merged into `dev-two-tier`, then the full suite.
- **B6's publication** of the folds and the base after B5 (on B13's sample format v3).
- **B7** adds `docs/reference/runners.md` entries for the new runner `prior_behaviour` and the changes to
  `prior_campaign`, `prior_free_generation`, `prior_validation`, `prior_select` and `prior_training_export`.
